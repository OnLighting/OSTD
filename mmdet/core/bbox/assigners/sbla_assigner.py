import math

import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class SBLAAssigner(BaseAssigner):
    """Scale-Balanced Label Assignment.

    ``balanced`` mode gives every GT the same IoU-top-k quota. ``full`` mode
    scales that quota with object scale, detached RPN objectness, and an
    epoch-decaying global budget.
    """

    requires_pred_scores = True
    _dense_overlap_limit = 8_000_000

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
        assert mode in ('balanced', 'full'), mode
        assert topk > 0 and 0 <= gamma <= 1 and base_sample_num > 0
        assert min_pos_per_gt >= 0
        assert candidate_mode in ('center', 'all'), candidate_mode

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
        self.epoch = int(epoch)

    @property
    def positive_budget(self):
        if self.mode == 'balanced':
            return self.base_sample_num
        start, end = self.schedule_start_epoch, self.schedule_end_epoch
        epoch = min(max(self.epoch, start), end)
        sigma = self.alpha + self.beta * (end - epoch) / max(end - start, 1)
        return int(math.ceil(self.base_sample_num * sigma))

    def _candidate_mask(self, bboxes, gt_bboxes):
        if self.candidate_mode == 'all':
            return torch.ones(
                (bboxes.size(0), gt_bboxes.size(0)),
                dtype=torch.bool, device=bboxes.device)
        c = (bboxes[:, :2] + bboxes[:, 2:]) * 0.5
        return (
            (c[:, None, 0] >= gt_bboxes[None, :, 0])
            & (c[:, None, 0] <= gt_bboxes[None, :, 2])
            & (c[:, None, 1] >= gt_bboxes[None, :, 1])
            & (c[:, None, 1] <= gt_bboxes[None, :, 3]))

    def _ignore_mask(self, bboxes, gt_bboxes_ignore):
        n = bboxes.size(0)
        if self.ignore_iof_thr <= 0 or gt_bboxes_ignore is None \
                or gt_bboxes_ignore.numel() == 0:
            return torch.zeros(n, dtype=torch.bool, device=bboxes.device)
        iof = self.iou_calculator(bboxes, gt_bboxes_ignore, mode='iof')
        return iof.max(dim=1).values > self.ignore_iof_thr

    def _foreground_prob(self, scores, idx, label):
        logit = scores[idx, 0] if scores.dim() > 1 else scores[idx]
        if scores.dim() > 1 and scores.size(1) > 1 and label is not None:
            logit = scores[idx, int(label)]
        p = logit.sigmoid() if self.scores_are_logits else logit.clamp(max=1.0)
        return p.clamp_min(1e-8)

    @staticmethod
    def _claim(sel, sel_ious, sel_q, gt_idx_plus1,
               best_iou, best_quality, assigned_gt_inds):
        win = (sel_ious > best_iou[sel]) | (
            (sel_ious == best_iou[sel]) & (sel_q > best_quality[sel]))
        winners = sel[win]
        best_iou[winners] = sel_ious[win]
        best_quality[winners] = sel_q[win]
        assigned_gt_inds[winners] = gt_idx_plus1

    @torch.no_grad()
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None,
               gt_labels=None, pred_scores=None):
        n_anchor, n_gt = bboxes.size(0), gt_bboxes.size(0)
        assigned_gt_inds = bboxes.new_zeros((n_anchor,), dtype=torch.long)
        certainties = bboxes.new_ones((n_gt,))
        requested = assigned_gt_inds.new_zeros((n_gt,))

        if n_gt == 0 or n_anchor == 0:
            return self._pack(n_gt, assigned_gt_inds,
                              bboxes.new_zeros((n_anchor,)),
                              certainties, requested, gt_labels)

        if self.mode == 'full' and pred_scores is None:
            raise ValueError('full SBLA requires pred_scores')
        assert pred_scores is None or pred_scores.size(0) == n_anchor

        ignore = self._ignore_mask(bboxes, gt_bboxes_ignore)
        has_ignore = bool(ignore.any())
        # A dense GT x anchor matrix can exceed GPU memory for large images.
        # The per-GT path uses the same IoUs and assignment rules, but keeps
        # only one GT's overlaps and candidate mask in memory at a time.
        stream_overlaps = n_gt * n_anchor > self._dense_overlap_limit
        if stream_overlaps:
            max_overlaps = bboxes.new_full((n_anchor,), -1.0)
        else:
            overlaps = self.iou_calculator(gt_bboxes, bboxes)
            if has_ignore:
                overlaps[:, ignore] = -1
            candidates = (self._candidate_mask(bboxes, gt_bboxes)
                          & ~ignore.unsqueeze(1))
        available = torch.nonzero(~ignore, as_tuple=False).flatten()
        avg_quota = math.ceil(self.positive_budget / n_gt)

        best_iou = bboxes.new_full((n_anchor,), -1.0)
        best_quality = bboxes.new_full((n_anchor,), -1.0)

        for g in range(n_gt):
            if stream_overlaps:
                gt_ious = self.iou_calculator(
                    gt_bboxes[g:g + 1], bboxes).reshape(-1)
                if has_ignore:
                    gt_ious[ignore] = -1
                torch.maximum(max_overlaps, gt_ious, out=max_overlaps)
                candidate_mask = (self._candidate_mask(
                    bboxes, gt_bboxes[g:g + 1])[:, 0] & ~ignore)
            else:
                gt_ious = overlaps[g]
                candidate_mask = candidates[:, g]
            cand = torch.nonzero(candidate_mask, as_tuple=False).flatten()
            if cand.numel() == 0 and available.numel() > 0:
                cand = available[gt_ious[available].argmax().view(1)]
            if cand.numel() == 0:
                continue

            ious = gt_ious[cand].clamp_min(1e-8)
            if self.mode == 'full':
                label = None if gt_labels is None else gt_labels[g]
                p = self._foreground_prob(pred_scores, cand, label)
                q = p.pow(self.gamma) * ious.pow(1.0 - self.gamma)
                certainties[g] = q.topk(
                    min(self.topk, q.numel())).values.mean()
                wh = (gt_bboxes[g, 2:] - gt_bboxes[g, :2]).clamp_min(1.0)
                scale_term = torch.log10((wh[0] * wh[1]).clamp_min(1.0).sqrt())
                factor = (self.kappa * (certainties[g] * scale_term).square()
                          + self.phi)
                if not torch.isfinite(factor):
                    raise FloatingPointError(
                        'SBLA nonfinite factor: '
                        f'gt={g}/{n_gt}, anchors={n_anchor}, '
                        f'gt_box={gt_bboxes[g].tolist()}, '
                        f'finite_boxes={bool(torch.isfinite(bboxes).all())}, '
                        f'finite_scores={bool(torch.isfinite(pred_scores).all())}, '
                        f'finite_ious={bool(torch.isfinite(ious).all())}, '
                        f'finite_p={bool(torch.isfinite(p).all())}, '
                        f'finite_q={bool(torch.isfinite(q).all())}, '
                        f'certainty={certainties[g].item()}, '
                        f'scale_term={scale_term.item()}')
                quota = math.ceil(avg_quota * factor.item())
            else:
                q = ious
                quota = avg_quota

            quota = min(max(self.min_pos_per_gt, quota), cand.numel())
            requested[g] = quota
            topk_indices = ious.topk(quota).indices
            sel = cand[topk_indices]
            self._claim(sel, gt_ious[sel], q[topk_indices],
                        g + 1, best_iou, best_quality, assigned_gt_inds)

        if not stream_overlaps:
            max_overlaps = overlaps.max(dim=0).values
        return self._pack(n_gt, assigned_gt_inds, max_overlaps,
                          certainties, requested, gt_labels)

    def _pack(self, n_gt, gt_inds, max_overlaps, certainties, requested,
              gt_labels):
        labels = None
        if gt_labels is not None:
            labels = gt_inds.new_full((gt_inds.size(0),), -1)
            pos = gt_inds > 0
            labels[pos] = gt_labels[gt_inds[pos] - 1]
        result = AssignResult(n_gt, gt_inds, max_overlaps, labels=labels)
        result.set_extra_property(
            'num_pos_per_gt',
            (torch.stack([(gt_inds == g + 1).sum() for g in range(n_gt)])
             if n_gt else gt_inds.new_zeros((0,))))
        result.set_extra_property('requested_pos_per_gt', requested)
        result.set_extra_property('certainty', certainties)
        result.set_extra_property('positive_budget', self.positive_budget)
        return result
