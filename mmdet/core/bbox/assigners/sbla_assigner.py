import math

import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class SBLAAssigner(BaseAssigner):
    """Scale-Balanced Label Assignment for anchor-based RPN training.

    ``balanced`` mode assigns an equal top-IoU quota to every GT. ``full``
    mode additionally adjusts that quota with object scale, detached RPN
    objectness, and an epoch-dependent global budget.
    """

    requires_pred_scores = True

    def __init__(self,
                 mode='full',
                 topk=9,
                 gamma=0.5,
                 alpha=0.5,
                 beta=0.6,
                 kappa=-0.06,
                 phi=1.4,
                 base_sample_num=256,
                 min_pos_per_gt=1,
                 schedule_start_epoch=0,
                 schedule_end_epoch=12,
                 candidate_mode='center',
                 scores_are_logits=True,
                 ignore_iof_thr=-1,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        if mode not in ('balanced', 'full'):
            raise ValueError(
                f'mode must be "balanced" or "full", but got {mode!r}')
        if topk <= 0:
            raise ValueError('topk must be positive')
        if not 0 <= gamma <= 1:
            raise ValueError('gamma must be in [0, 1]')
        if base_sample_num <= 0:
            raise ValueError('base_sample_num must be positive')
        if min_pos_per_gt < 0:
            raise ValueError('min_pos_per_gt must be non-negative')
        if candidate_mode not in ('center', 'all'):
            raise ValueError(
                'candidate_mode must be "center" or "all", '
                f'but got {candidate_mode!r}')

        self.mode = mode
        self.topk = topk
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.phi = phi
        self.base_sample_num = base_sample_num
        self.min_pos_per_gt = min_pos_per_gt
        self.schedule_start_epoch = schedule_start_epoch
        self.schedule_end_epoch = schedule_end_epoch
        self.candidate_mode = candidate_mode
        self.scores_are_logits = scores_are_logits
        self.ignore_iof_thr = ignore_iof_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.epoch = schedule_start_epoch

    def set_epoch(self, epoch):
        """Update training progress used by the full SBLA schedule."""
        self.epoch = int(epoch)

    @property
    def positive_budget(self):
        """Global positive budget for one image at the current epoch."""
        if self.mode == 'balanced':
            return int(self.base_sample_num)
        start = self.schedule_start_epoch
        end = self.schedule_end_epoch
        epoch = min(max(self.epoch, start), end)
        duration = max(end - start, 1)
        sigma = self.alpha + self.beta * (end - epoch) / duration
        return int(math.ceil(self.base_sample_num * sigma))

    def _candidate_mask(self, bboxes, gt_bboxes):
        if self.candidate_mode == 'all':
            return torch.ones(
                (bboxes.size(0), gt_bboxes.size(0)),
                dtype=torch.bool,
                device=bboxes.device)
        centers = (bboxes[:, :2] + bboxes[:, 2:]) * 0.5
        return (
            (centers[:, None, 0] >= gt_bboxes[None, :, 0])
            & (centers[:, None, 0] <= gt_bboxes[None, :, 2])
            & (centers[:, None, 1] >= gt_bboxes[None, :, 1])
            & (centers[:, None, 1] <= gt_bboxes[None, :, 3]))

    def _foreground_probability(self, pred_scores, candidate_inds,
                                gt_label):
        if pred_scores.dim() == 1:
            logits = pred_scores[candidate_inds]
        elif pred_scores.size(1) == 1 or gt_label is None:
            logits = pred_scores[candidate_inds, 0]
        else:
            logits = pred_scores[candidate_inds, int(gt_label)]
        if self.scores_are_logits:
            return logits.sigmoid().clamp_min(1e-8)
        return logits.clamp(min=1e-8, max=1.0)

    @torch.no_grad()
    def assign(self,
               bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               pred_scores=None):
        """Assign anchors, using detached predictions only for label choice."""
        num_bboxes = bboxes.size(0)
        num_gts = gt_bboxes.size(0)
        assigned_gt_inds = bboxes.new_zeros(
            (num_bboxes, ), dtype=torch.long)
        max_overlaps = bboxes.new_zeros((num_bboxes, ))

        if num_gts == 0 or num_bboxes == 0:
            labels = None
            if gt_labels is not None:
                labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            result = AssignResult(
                num_gts, assigned_gt_inds, max_overlaps, labels=labels)
            result.set_extra_property(
                'num_pos_per_gt',
                assigned_gt_inds.new_zeros((num_gts, )))
            result.set_extra_property(
                'certainty', bboxes.new_zeros((num_gts, )))
            result.set_extra_property('positive_budget', self.positive_budget)
            return result

        if self.mode == 'full' and pred_scores is None:
            raise ValueError('full SBLA requires pred_scores')
        if pred_scores is not None and pred_scores.size(0) != num_bboxes:
            raise ValueError(
                'pred_scores and bboxes must have the same first dimension')

        overlaps = self.iou_calculator(gt_bboxes, bboxes)
        max_overlaps = overlaps.max(dim=0).values

        ignore_mask = torch.zeros(
            num_bboxes, dtype=torch.bool, device=bboxes.device)
        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0):
            ignore_overlaps = self.iou_calculator(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_mask = (
                ignore_overlaps.max(dim=1).values > self.ignore_iof_thr)
            assigned_gt_inds[ignore_mask] = -1

        candidates = self._candidate_mask(bboxes, gt_bboxes)
        candidates[ignore_mask, :] = False
        available = torch.nonzero(~ignore_mask, as_tuple=False).flatten()
        average_quota = int(math.ceil(self.positive_budget / num_gts))
        certainties = bboxes.new_ones((num_gts, ))
        requested = assigned_gt_inds.new_zeros((num_gts, ))
        best_claim_iou = bboxes.new_full((num_bboxes, ), -1)
        best_claim_quality = bboxes.new_full((num_bboxes, ), -1)

        for gt_idx in range(num_gts):
            candidate_inds = torch.nonzero(
                candidates[:, gt_idx], as_tuple=False).flatten()
            if candidate_inds.numel() == 0 and available.numel() > 0:
                fallback = overlaps[gt_idx, available].argmax()
                candidate_inds = available[fallback].view(1)
            if candidate_inds.numel() == 0:
                continue

            candidate_ious = overlaps[gt_idx, candidate_inds].clamp_min(1e-8)
            if self.mode == 'full':
                label = None if gt_labels is None else gt_labels[gt_idx]
                probability = self._foreground_probability(
                    pred_scores, candidate_inds, label)
                quality = (
                    probability.pow(self.gamma)
                    * candidate_ious.pow(1.0 - self.gamma))
                certainty = quality.topk(
                    min(self.topk, candidate_inds.numel())).values.mean()
                certainties[gt_idx] = certainty
                wh = (gt_bboxes[gt_idx, 2:] -
                      gt_bboxes[gt_idx, :2]).clamp_min(1.0)
                scale = torch.sqrt(wh[0] * wh[1]).clamp_min(1.0)
                scale_term = torch.log10(scale)
                factor = (
                    self.kappa * (certainty * scale_term).square() + self.phi)
                quota = int(math.ceil(average_quota * factor.item()))
            else:
                quality = candidate_ious
                quota = average_quota

            quota = min(
                max(self.min_pos_per_gt, quota), candidate_inds.numel())
            requested[gt_idx] = quota
            local_inds = candidate_ious.topk(quota).indices
            selected = candidate_inds[local_inds]
            selected_ious = overlaps[gt_idx, selected]
            selected_quality = quality[local_inds]

            better = selected_ious > best_claim_iou[selected]
            tied = selected_ious == best_claim_iou[selected]
            better = better | (
                tied & (selected_quality > best_claim_quality[selected]))
            winners = selected[better]
            best_claim_iou[winners] = selected_ious[better]
            best_claim_quality[winners] = selected_quality[better]
            assigned_gt_inds[winners] = gt_idx + 1

        assigned_labels = None
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            positive = assigned_gt_inds > 0
            assigned_labels[positive] = gt_labels[
                assigned_gt_inds[positive] - 1]

        result = AssignResult(
            num_gts,
            assigned_gt_inds,
            max_overlaps,
            labels=assigned_labels)
        actual_per_gt = torch.stack([
            (assigned_gt_inds == gt_idx + 1).sum()
            for gt_idx in range(num_gts)
        ])
        result.set_extra_property('num_pos_per_gt', actual_per_gt)
        result.set_extra_property('requested_pos_per_gt', requested)
        result.set_extra_property('certainty', certainties)
        result.set_extra_property('positive_budget', self.positive_budget)
        return result
