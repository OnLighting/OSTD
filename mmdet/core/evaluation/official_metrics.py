"""Shared official Recall/FDR metric module.

This module is the single source of truth for the competition's
official metrics. It is consumed by:

* ``AircraftDataset.evaluate`` (training-time validation hooks).
* ``tools/eval_val_to_json`` (ordinary validation inference).
* ``tools/infer_big_image`` (10k×10k mosaic inference).
* ``tools/eval_recall_fdr`` (post-hoc reporting CLI).
* ``tools/search_recall_fdr_thresholds`` (post-training threshold search).

It exposes the candidate score floor, per-class IoU thresholds, the
superclass (ship/aircraft/vehicle) grouping, score-ranked matching
events, explicit-threshold filtering/evaluation, the training-time
per-superclass threshold search, the official/merged aggregation, and
the comparator used by checkpoint hooks to choose the best model.

Decision thresholds are **never** built in: every consumer must supply
them explicitly (from the training-time search or the frozen checkpoint-
bound artifact). The only module-level score constant is
``CANDIDATE_SCORE_FLOOR``, which keeps model-side candidates alive
before a decision threshold exists; it is not a decision threshold.
"""

import math
from collections import defaultdict
from heapq import heappop, heappush

import numpy as np


# --- Constants ----------------------------------------------------------

CLASS_NAMES = (
    'HM', 'LQS', 'QHS', 'MS',
    'A1_SU-35', 'A2_C-130', 'A3_C-17', 'A4_C-5', 'A5_F-16', 'A6_TU-160',
    'A7_E-3', 'A8_B-52', 'A9_P-3C', 'A10_B-1B', 'A11_E-8', 'A12_TU-22',
    'A13_F-15', 'A14_KC-135', 'A15_F-22', 'A16_FA-18', 'A17_TU-95',
    'A18_KC-10', 'A19_SU-34', 'A20_SU-24', 'FSC',
)

# Candidate retention floor applied on the model side so that threshold
# search never loses candidates before a decision threshold exists. This
# is NOT a decision threshold; final filtering always uses explicit
# thresholds supplied by the caller.
CANDIDATE_SCORE_FLOOR = 0.0

# Per-class IoU thresholds: 0.50 for ship/aircraft, 0.35 for FSC.
CLASS_IOU_THRESHOLDS = tuple(0.35 if i == 24 else 0.5 for i in range(25))

SUPERCLASS_INDICES = {
    'ship': (0, 1, 2, 3),
    'aircraft': tuple(range(4, 24)),
    'vehicle': (24,),
}

SUPERCLASS_NAMES = ('ship', 'aircraft', 'vehicle')

_SUPERCLASS_OF_INDEX = {
    index: name
    for name, indices in SUPERCLASS_INDICES.items()
    for index in indices
}


def _validate_constants():
    """Assert that constants cover the expected category id range."""
    expected = set(range(25))
    if len(CLASS_NAMES) != 25:
        raise ValueError(
            f'CLASS_NAMES must cover 25 ids, got {len(CLASS_NAMES)}')
    if len(CLASS_IOU_THRESHOLDS) != 25:
        raise ValueError(
            'CLASS_IOU_THRESHOLDS must cover 25 ids')
    seen = set()
    for ids in SUPERCLASS_INDICES.values():
        seen.update(ids)
    if seen != expected:
        raise ValueError(
            f'SUPERCLASS_INDICES must cover 0..24, missing '
            f'{sorted(expected - seen)}, extra {sorted(seen - expected)}')
    if set(_SUPERCLASS_OF_INDEX) != expected:
        raise ValueError('each category id must map to exactly one superclass')


_validate_constants()


# --- Matching -----------------------------------------------------------


def _iou_xywh(box, boxes):
    """Vector IoU between one xywh box and an array of xywh boxes."""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1, y1, w1, h1 = box
    x2 = x1 + w1
    y2 = y1 + h1
    area1 = w1 * h1
    xs1 = boxes[:, 0]
    ys1 = boxes[:, 1]
    ws = boxes[:, 2]
    hs = boxes[:, 3]
    xs2 = xs1 + ws
    ys2 = ys1 + hs
    areas2 = ws * hs
    inter_x1 = np.maximum(x1, xs1)
    inter_y1 = np.maximum(y1, ys1)
    inter_x2 = np.minimum(x2, xs2)
    inter_y2 = np.minimum(y2, ys2)
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h
    union = area1 + areas2 - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match_class_events(pred_boxes, pred_scores, gt_boxes, tau):
    """Score-ranked matching events for a single (image, class) pair.

    Predictions are processed in score-descending order. Each prediction
    is greedily matched to the highest-IoU **unmatched** GT above ``tau``.
    Already-matched GTs are excluded from the candidate set so a
    prediction whose highest-IoU GT is taken still has a chance to match
    a different GT that crosses the threshold.

    Because matching walks predictions in score-descending order,
    retaining the events with ``score >= t`` (a prefix of the ranked
    list) reproduces exactly the matches that filtering-then-matching
    would produce — this is what lets thresholds be applied after
    matching instead of before.

    Args:
        pred_boxes (np.ndarray): (P, 4) xywh prediction boxes.
        pred_scores (np.ndarray): (P,) prediction scores.
        gt_boxes (np.ndarray): (G, 4) xywh GT boxes.
        tau (float): IoU threshold for this class.

    Returns:
        list[tuple[float, int]]: one ``(score, is_tp)`` per prediction,
        sorted by descending score; ``is_tp`` is 1 for a TP and 0 for an
        eventual FP at any threshold at or below ``score``.
    """
    pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
    matched = np.zeros(len(gt_boxes), dtype=bool)
    events = []
    # Stable sort keeps tie order deterministic across runs.
    for pred_index in np.argsort(-pred_scores, kind='stable'):
        is_tp = 0
        unmatched = np.flatnonzero(~matched)
        if unmatched.size:
            overlaps = _iou_xywh(pred_boxes[pred_index], gt_boxes[unmatched])
            best_local = int(overlaps.argmax())
            if overlaps[best_local] >= tau:
                matched[unmatched[best_local]] = True
                is_tp = 1
        events.append((float(pred_scores[pred_index]), is_tp))
    return events


def _match_class(pred_boxes, pred_scores, gt_boxes, tau):
    """One-to-one matching counts for a single (image, class) pair.

    Thin counting wrapper over :func:`match_class_events`.
    """
    events = match_class_events(pred_boxes, pred_scores, gt_boxes, tau)
    tp = sum(flag for _, flag in events)
    fp = len(events) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


# --- Public API ---------------------------------------------------------


def match_class(pred_boxes, pred_scores, gt_boxes, tau):
    """Public wrapper for one-to-one matching for a single (image, class) pair.

    Args:
        pred_boxes (np.ndarray): (P, 4) xywh prediction boxes.
        pred_scores (np.ndarray): (P,) prediction scores.
        gt_boxes (np.ndarray): (G, 4) xywh GT boxes.
        tau (float): IoU threshold for this class.

    Returns:
        tuple[int, int, int]: (tp, fp, fn) counts for this (image, class).
    """
    return _match_class(pred_boxes, pred_scores, gt_boxes, tau)


def normalize_score_thresholds(score_thresholds):
    """Normalize caller-supplied decision thresholds to a 25-value tuple.

    Args:
        score_thresholds (float | sequence[float]): a scalar broadcast to
            every class, or exactly ``len(CLASS_NAMES)`` finite
            non-negative values.

    Returns:
        tuple[float, ...]: 25 finite non-negative thresholds.

    Raises:
        ValueError: on a wrong count or a non-finite/negative value.
    """
    if np.isscalar(score_thresholds):
        values = [float(score_thresholds)] * len(CLASS_NAMES)
    else:
        values = [float(value) for value in score_thresholds]
    if len(values) != len(CLASS_NAMES):
        raise ValueError(
            f'expected {len(CLASS_NAMES)} score thresholds, '
            f'got {len(values)}')
    for value in values:
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                'score thresholds must be finite non-negative numbers')
    return tuple(values)


def filter_mmdet_results(results, score_thresholds):
    """Filter each class's predictions by explicit score thresholds.

    Args:
        results (list[np.ndarray]): one ``(N, 5)`` array per class (x1, y1,
            x2, y2, score); empty arrays are fine.
        score_thresholds (float | sequence[float]): scalar or 25 finite
            non-negative thresholds, one per class.

    Returns:
        list[np.ndarray]: same length as ``results``, each row preserved as
            ``(x1, y1, x2, y2, score)`` and rows below the per-class
            threshold removed. Empty classes become ``np.zeros((0, 5))``.
    """
    thresholds = normalize_score_thresholds(score_thresholds)
    if len(results) != len(CLASS_NAMES):
        raise ValueError(
            f'expected {len(CLASS_NAMES)} classes, got {len(results)}')
    out = []
    for i, boxes in enumerate(results):
        if len(boxes) == 0:
            out.append(np.zeros((0, 5), dtype=np.float32))
            continue
        keep = boxes[:, 4] >= thresholds[i]
        if not np.any(keep):
            out.append(np.zeros((0, 5), dtype=np.float32))
            continue
        out.append(boxes[keep].astype(np.float32, copy=False))
    return out


def _convert_xyxy_to_xywh(boxes):
    """Convert (x1, y1, x2, y2) → (x, y, w, h). Accepts empty arrays."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = boxes[:, :4].astype(np.float32, copy=True)
    out[:, 2] = out[:, 2] - out[:, 0]
    out[:, 3] = out[:, 3] - out[:, 1]
    return out


def build_mmdet_score_events(results, gt_infos):
    """Build per-class score-ranked matching events for an mmdet result set.

    Args:
        results (list[list[np.ndarray]]): one entry per image; each entry is
            a list of 25 ``(N, 5)`` arrays (xyxy + score) indexed by class.
        gt_infos (list[list[dict]]): one list per image; each ann dict has
            ``bbox`` (xywh) and ``category_id``. Image order must match
            ``results``.

    Returns:
        tuple[dict[int, list[tuple[float, int]]], dict[int, int]]: per-class
        events sorted by descending score, and the per-class GT totals.
    """
    if len(results) != len(gt_infos):
        raise ValueError(
            f'results has {len(results)} entries but gt_infos has '
            f'{len(gt_infos)}; must match image count')
    events = {cls_idx: [] for cls_idx in range(len(CLASS_NAMES))}
    total_gt = {cls_idx: 0 for cls_idx in range(len(CLASS_NAMES))}
    for img_results, img_gts in zip(results, gt_infos):
        if not isinstance(img_results, list):
            raise ValueError(
                'each per-image entry must be a list of 25 class arrays')
        gt_by_class = defaultdict(list)
        for ann in img_gts:
            cat = int(ann['category_id'])
            if cat < 0 or cat >= len(CLASS_NAMES):
                raise ValueError(
                    f'GT category_id {cat} outside valid range 0..24')
            gt_by_class[cat].append(ann['bbox'])
        for cls_idx in range(len(CLASS_NAMES)):
            gt_boxes = gt_by_class.get(cls_idx, [])
            gt_arr = (np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
                      if gt_boxes else np.zeros((0, 4), dtype=np.float32))
            total_gt[cls_idx] += len(gt_arr)
            pred = img_results[cls_idx]
            if len(pred) == 0:
                continue
            pred_boxes = _convert_xyxy_to_xywh(pred)
            pred_scores = pred[:, 4].astype(np.float64)
            events[cls_idx].extend(match_class_events(
                pred_boxes, pred_scores, gt_arr,
                CLASS_IOU_THRESHOLDS[cls_idx]))
    for cls_idx in events:
        events[cls_idx].sort(key=lambda item: item[0], reverse=True)
    return events, total_gt


def _per_class_counts_at_threshold(events, total_gt, threshold):
    """Count (tp, fp, fn) for one class at ``threshold`` from its events."""
    tp = 0
    fp = 0
    for score, flag in events:
        if score >= threshold:
            if flag:
                tp += 1
            else:
                fp += 1
    return tp, fp, total_gt - tp


def evaluate_score_events(events, total_gt, score_thresholds):
    """Compute official metrics from matching events at explicit thresholds.

    Args:
        events (dict[int, list[tuple[float, int]]]): per-class events as
            produced by :func:`build_mmdet_score_events`. Only events with
            ``score >= threshold`` count; event order is irrelevant.
        total_gt (dict[int, int]): per-class GT totals.
        score_thresholds (float | sequence[float]): scalar or 25 finite
            non-negative decision thresholds.

    Returns:
        dict with keys ``per_class`` (list of dicts, one per category id
        0..24), ``by_super``, ``official``, and ``merged`` — the same
        shape :func:`evaluate_mmdet_results` returns.
    """
    thresholds = normalize_score_thresholds(score_thresholds)
    per_class_tp = [0] * len(CLASS_NAMES)
    per_class_fp = [0] * len(CLASS_NAMES)
    per_class_fn = [0] * len(CLASS_NAMES)
    for cls_idx in range(len(CLASS_NAMES)):
        tp, fp, fn = _per_class_counts_at_threshold(
            events.get(cls_idx, []), int(total_gt.get(cls_idx, 0)),
            thresholds[cls_idx])
        per_class_tp[cls_idx] = tp
        per_class_fp[cls_idx] = fp
        per_class_fn[cls_idx] = fn

    return _aggregate_counts(per_class_tp, per_class_fp, per_class_fn)


def _aggregate_counts(per_class_tp, per_class_fp, per_class_fn):
    """Build the standard metrics payload from per-class TP/FP/FN counts."""
    per_class_rows = []
    for cls_idx, name in enumerate(CLASS_NAMES):
        tp = per_class_tp[cls_idx]
        fp = per_class_fp[cls_idx]
        fn = per_class_fn[cls_idx]
        recall = tp / max(tp + fn, 1)
        fdr = fp / max(fp + tp, 1)
        prec = (tp / max(tp + fp, 1)) if (tp + fp) > 0 else None
        per_class_rows.append({
            'category_id': cls_idx,
            'name': name,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'recall': recall,
            'fdr': fdr,
            'prec': prec,
            'ap_iou_thr': CLASS_IOU_THRESHOLDS[cls_idx],
        })

    by_super, official = aggregate_official_per_class(per_class_rows)

    overall_tp = sum(per_class_tp)
    overall_fp = sum(per_class_fp)
    overall_fn = sum(per_class_fn)
    merged_recall = overall_tp / max(overall_tp + overall_fn, 1)
    merged_fdr = overall_fp / max(overall_fp + overall_tp, 1)
    merged = {
        'tp': overall_tp,
        'fp': overall_fp,
        'fn': overall_fn,
        'recall': merged_recall,
        'fdr': merged_fdr,
    }

    return {
        'per_class': per_class_rows,
        'by_super': by_super,
        'official': official,
        'merged': merged,
    }


def aggregate_official_per_class(per_class_rows):
    """Aggregate 25 per-class rows using the official three-group rule.

    A superclass with no GT makes the overall official metric unavailable;
    it must not be silently averaged as a zero-valued group.
    """
    rows_by_id = {int(row['category_id']): row for row in per_class_rows}
    expected_ids = set(range(len(CLASS_NAMES)))
    if set(rows_by_id) != expected_ids or len(rows_by_id) != len(per_class_rows):
        raise ValueError('per_class_rows must contain category_id 0..24 once')

    by_super = {}
    official_recalls = []
    official_fdrs = []
    unavailable_superclasses = []
    for super_name in SUPERCLASS_NAMES:
        ids = SUPERCLASS_INDICES[super_name]
        rows = [rows_by_id[i] for i in ids]
        gt_total = sum(int(row['tp']) + int(row['fn']) for row in rows)
        if gt_total == 0:
            by_super[super_name] = {'recall': None, 'fdr': None}
            unavailable_superclasses.append(super_name)
            continue
        recall = sum(float(row['recall']) for row in rows) / len(rows)
        fdr = sum(float(row['fdr']) for row in rows) / len(rows)
        by_super[super_name] = {'recall': recall, 'fdr': fdr}
        official_recalls.append(recall)
        official_fdrs.append(fdr)

    if unavailable_superclasses or not official_recalls:
        official = {
            'recall': float('nan'),
            'fdr': float('nan'),
            'available': False,
            'unavailable_superclasses': unavailable_superclasses,
        }
    else:
        official = {
            'recall': sum(official_recalls) / len(official_recalls),
            'fdr': sum(official_fdrs) / len(official_fdrs),
            'available': True,
            'unavailable_superclasses': [],
        }
    return by_super, official


def evaluate_mmdet_results(results, gt_infos, score_thresholds):
    """Compute per-class, superclass, official, and merged metrics.

    Args:
        results (list[list[np.ndarray]]): one entry per image; each entry is
            a list of 25 ``(N, 5)`` arrays (xyxy + score) indexed by class.
        gt_infos (list[list[dict]]): one list per image; each ann dict has
            ``bbox`` (xywh) and ``category_id``. Image order must match
            ``results``.
        score_thresholds (float | sequence[float]): scalar or 25 finite
            non-negative decision thresholds. There is no module-level
            default; callers pass searched/frozen values.

    Returns:
        dict with keys ``per_class`` (list of dicts, one per category id 0..24),
        ``by_super`` (dict ship/aircraft/vehicle → recall/fdr; missing GTs
        produce ``None`` values),
        ``official`` (dict with ``recall``, ``fdr``, ``available`` boolean,
        and ``unavailable_superclasses`` list — the recall/fdr are ``NaN``
        when any of ship/aircraft/vehicle has no GT, so callers like the
        checkpoint hook can refuse to use the run), and ``merged``
        (counts-based recall/fdr over every category).
    """
    thresholds = normalize_score_thresholds(score_thresholds)
    events, total_gt = build_mmdet_score_events(results, gt_infos)
    return evaluate_score_events(events, total_gt, thresholds)


def search_superclass_thresholds(events, total_gt, max_fdr=0.19):
    """Search one exact operating threshold per superclass for training.

    For each superclass, candidate thresholds are every distinct event
    score in that superclass plus an empty-prediction threshold above its
    maximum. One shared threshold is evaluated across every child class;
    among points whose superclass mean FDR is at most ``max_fdr`` the
    literal comparison key ``(recall, -fdr, threshold)`` picks the
    highest Recall, then the lowest FDR, then the highest threshold.

    A superclass with zero total GT has no defined operating point: its
    threshold is ``None``, the official aggregate is reported
    unavailable, and ``score_thresholds`` is ``None`` so callers cannot
    silently fall back to a global value.

    Args:
        events (dict[int, list[tuple[float, int]]]): per-class events,
            sorted by descending score (see
            :func:`build_mmdet_score_events`).
        total_gt (dict[int, int]): per-class GT totals.
        max_fdr (float): maximum allowed superclass mean FDR.

    Returns:
        dict with ``thresholds_by_super`` (ship/aircraft/vehicle →
        threshold or ``None``), ``score_thresholds`` (25-value tuple, or
        ``None`` when any superclass is unavailable), and ``metrics``
        (the standard evaluation payload at the selected thresholds).
    """
    if not math.isfinite(max_fdr) or max_fdr < 0:
        raise ValueError('max_fdr must be a finite non-negative number')
    thresholds_by_super = {}
    for super_name in SUPERCLASS_NAMES:
        ids = SUPERCLASS_INDICES[super_name]
        group_gt = sum(int(total_gt.get(i, 0)) for i in ids)
        if group_gt == 0:
            thresholds_by_super[super_name] = None
            continue
        # Sort defensively; producers already emit descending order.
        sorted_events = {
            cls_idx: sorted(events.get(cls_idx, []),
                            key=lambda item: item[0], reverse=True)
            for cls_idx in ids
        }
        # Sweep all child-class events once in global score order. The old
        # implementation rescanned every prediction in every child class for
        # every distinct candidate score (quadratic on dense validation
        # output), leaving the GPU idle for tens of minutes after COCO eval.
        heap = []
        for cls_idx in ids:
            class_events = sorted_events[cls_idx]
            if class_events:
                score, flag = class_events[0]
                heappush(heap, (-score, cls_idx, 0, flag))

        if heap:
            max_score = -heap[0][0]
            empty_threshold = max_score + max(
                abs(max_score) * 1e-12, 1e-12)
        else:
            empty_threshold = 1.0
        best_key = (0.0, -0.0, empty_threshold)
        best_threshold = empty_threshold
        per_class_tp = {cls_idx: 0 for cls_idx in ids}
        per_class_fp = {cls_idx: 0 for cls_idx in ids}
        recall_sum = 0.0
        fdr_sum = 0.0

        while heap:
            threshold = -heap[0][0]
            old_contributions = {}
            while heap and -heap[0][0] == threshold:
                _, cls_idx, event_index, flag = heappop(heap)
                if cls_idx not in old_contributions:
                    class_gt = int(total_gt.get(cls_idx, 0))
                    tp = per_class_tp[cls_idx]
                    fp = per_class_fp[cls_idx]
                    old_contributions[cls_idx] = (
                        tp / max(class_gt, 1),
                        fp / max(tp + fp, 1),
                    )
                if flag:
                    per_class_tp[cls_idx] += 1
                else:
                    per_class_fp[cls_idx] += 1
                next_index = event_index + 1
                class_events = sorted_events[cls_idx]
                if next_index < len(class_events):
                    score, next_flag = class_events[next_index]
                    heappush(
                        heap, (-score, cls_idx, next_index, next_flag))

            for cls_idx, (old_recall, old_fdr) in \
                    old_contributions.items():
                class_gt = int(total_gt.get(cls_idx, 0))
                tp = per_class_tp[cls_idx]
                fp = per_class_fp[cls_idx]
                recall_sum += tp / max(class_gt, 1) - old_recall
                fdr_sum += fp / max(tp + fp, 1) - old_fdr
            super_recall = recall_sum / len(ids)
            super_fdr = fdr_sum / len(ids)
            if super_fdr <= max_fdr:
                key = (super_recall, -super_fdr, threshold)
                if key > best_key:
                    best_key = key
                    best_threshold = threshold
        if best_threshold is None:
            # The empty-prediction point always has FDR 0, so a group with
            # GT can only land here if the caller passed a negative budget.
            raise RuntimeError(
                f'superclass {super_name!r} has GT but no feasible '
                f'operating point under max_fdr={max_fdr}')
        thresholds_by_super[super_name] = float(best_threshold)

    unavailable = [name for name, value in thresholds_by_super.items()
                   if value is None]
    if unavailable:
        # Diagnostic evaluation only: available groups run at their searched
        # threshold, unavailable ones at the candidate floor. The official
        # aggregate stays unavailable (NaN) so hooks refuse to save.
        expanded_eval = tuple(
            thresholds_by_super[_SUPERCLASS_OF_INDEX[cls_idx]]
            if thresholds_by_super[_SUPERCLASS_OF_INDEX[cls_idx]] is not None
            else CANDIDATE_SCORE_FLOOR
            for cls_idx in range(len(CLASS_NAMES)))
        score_thresholds = None
    else:
        expanded_eval = tuple(
            thresholds_by_super[_SUPERCLASS_OF_INDEX[cls_idx]]
            for cls_idx in range(len(CLASS_NAMES)))
        score_thresholds = expanded_eval
    metrics = evaluate_score_events(events, total_gt, expanded_eval)
    return {
        'thresholds_by_super': thresholds_by_super,
        'score_thresholds': score_thresholds,
        'metrics': metrics,
    }


def compare_official_candidates(candidate, best, recall_target=0.85,
                                fdr_limit=0.20, recall_tolerance=0.005):
    """Decide whether ``candidate`` should replace ``best``.

    Implements the six-rule ordering described in the design doc:

    1. Both-pass beats one-pass beats none-pass.
    2. Among non-passing models, only Recall matters; higher Recall wins.
    3. Once a passing best exists, only another passing candidate can win.
    4. Between two passing candidates whose Recall differs by more than
       ``recall_tolerance``, higher Recall wins.
    5. Within tolerance, lower FDR wins.
    6. Exact-equal or both within tolerance + equal FDR → keep the earlier
       one (return False).

    Args:
        candidate (dict): ``{'recall': float, 'fdr': float}``.
        best (dict | None): same shape, or ``None`` when no best exists yet.
        recall_target (float): minimum Recall to count as "passing".
        fdr_limit (float): maximum FDR to count as "passing".
        recall_tolerance (float): tied-recall tolerance for FDR tiebreak.

    Returns:
        bool: True iff candidate strictly beats best.
    """
    if best is None:
        return True
    cand_pass = (
        candidate['recall'] >= recall_target
        and candidate['fdr'] <= fdr_limit)
    best_pass = (
        best['recall'] >= recall_target
        and best['fdr'] <= fdr_limit)
    if cand_pass != best_pass:
        return cand_pass
    if not cand_pass:
        return candidate['recall'] > best['recall']
    delta = candidate['recall'] - best['recall']
    if abs(delta) > recall_tolerance:
        return delta > 0
    # Within tolerance: tiebreak by lower FDR; equal FDR keeps best.
    return candidate['fdr'] < best['fdr'] - 1e-12


__all__ = [
    'CLASS_NAMES',
    'CANDIDATE_SCORE_FLOOR',
    'CLASS_IOU_THRESHOLDS',
    'SUPERCLASS_INDICES',
    'aggregate_official_per_class',
    'filter_mmdet_results',
    'evaluate_mmdet_results',
    'evaluate_score_events',
    'build_mmdet_score_events',
    'match_class_events',
    'normalize_score_thresholds',
    'search_superclass_thresholds',
    'compare_official_candidates',
    'match_class',
]
