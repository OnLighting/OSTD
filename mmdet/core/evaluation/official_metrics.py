"""Shared official Recall/FDR metric module.

This module is the single source of truth for the competition's
official metrics. It is consumed by:

* ``AircraftDataset.evaluate`` (training-time validation hooks).
* ``tools/eval_val_to_json`` (ordinary validation inference).
* ``tools/infer_big_image`` (10k×10k mosaic inference).
* ``tools/eval_recall_fdr`` (post-hoc reporting CLI).

It exposes fixed per-class score thresholds, per-class IoU thresholds,
the superclass (ship/aircraft/vehicle) grouping, the matching algorithm,
the official/merged aggregation, and the comparator used by checkpoint
hooks to choose the best model.
"""

from collections import defaultdict

import numpy as np


# --- Constants ----------------------------------------------------------

CLASS_NAMES = (
    'HM', 'LQS', 'QHS', 'MS',
    'A1_SU-35', 'A2_C-130', 'A3_C-17', 'A4_C-5', 'A5_F-16', 'A6_TU-160',
    'A7_E-3', 'A8_B-52', 'A9_P-3C', 'A10_B-1B', 'A11_E-8', 'A12_TU-22',
    'A13_F-15', 'A14_KC-135', 'A15_F-22', 'A16_FA-18', 'A17_TU-95',
    'A18_KC-10', 'A19_SU-34', 'A20_SU-24', 'FSC',
)

# Score thresholds — hard-coded from ret/threshold_search_fdr_0.19_selected.csv.
# The CSV exists for audit only; runtime imports these literals so the
# pipeline is self-contained.
CLASS_SCORE_THRESHOLDS = (
    .348537356, .00188686815, .00998581946, .0174246859,
    .854384005, .554479778, .469661117, .424196422, .451967984,
    .927510798, .927627504, .33063519, .791748285, .987444103,
    .186482817, .69058013, .586889446, .936379254, .362753332,
    .145738885, .877446949, .038396392, .141620845, .0739553422,
    .0403826572,
)

# Per-class IoU thresholds: 0.50 for ship/aircraft, 0.35 for FSC.
CLASS_IOU_THRESHOLDS = tuple(0.35 if i == 24 else 0.5 for i in range(25))

SUPERCLASS_INDICES = {
    'ship': (0, 1, 2, 3),
    'aircraft': tuple(range(4, 24)),
    'vehicle': (24,),
}


def _validate_constants():
    """Assert that constants cover the expected category id range."""
    expected = set(range(25))
    if len(CLASS_NAMES) != 25:
        raise ValueError(
            f'CLASS_NAMES must cover 25 ids, got {len(CLASS_NAMES)}')
    if len(CLASS_SCORE_THRESHOLDS) != 25:
        raise ValueError(
            'CLASS_SCORE_THRESHOLDS must cover 25 ids')
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


def _match_class(pred_boxes, pred_scores, gt_boxes, tau):
    """One-to-one matching for a single (image, class) pair.

    Predictions are processed in score-descending order. Each prediction
    is greedily matched to the highest-IoU unmatched GT above ``tau``.
    Unmatched predictions → FP, unmatched GT → FN.
    """
    n_gt = len(gt_boxes)
    n_p = len(pred_boxes)
    if n_p == 0:
        tp = 0
        fp = 0
        fn = n_gt
        return tp, fp, fn
    gt_matched = np.zeros(n_gt, dtype=bool)
    pr_matched = np.zeros(n_p, dtype=bool)
    order = np.argsort(-pred_scores)
    for pi in order:
        b = pred_boxes[pi]
        if n_gt == 0:
            break
        ious = _iou_xywh(b, gt_boxes)
        best_gi = int(ious.argmax())
        best_iou = float(ious[best_gi])
        if best_iou >= tau and not gt_matched[best_gi]:
            gt_matched[best_gi] = True
            pr_matched[pi] = True
    tp = int(pr_matched.sum())
    fp = int((~pr_matched).sum())
    fn = int((~gt_matched).sum())
    return tp, fp, fn


# --- Public API ---------------------------------------------------------


def filter_mmdet_results(results):
    """Filter each class's predictions by its fixed score threshold.

    Args:
        results (list[np.ndarray]): one ``(N, 5)`` array per class (x1, y1,
            x2, y2, score); empty arrays are fine.

    Returns:
        list[np.ndarray]: same length as ``results``, each row preserved as
            ``(x1, y1, x2, y2, score)`` and rows below the per-class
            threshold removed. Empty classes become ``np.zeros((0, 5))``.
    """
    if len(results) != len(CLASS_NAMES):
        raise ValueError(
            f'expected {len(CLASS_NAMES)} classes, got {len(results)}')
    out = []
    for i, boxes in enumerate(results):
        if len(boxes) == 0:
            out.append(np.zeros((0, 5), dtype=np.float32))
            continue
        keep = boxes[:, 4] >= CLASS_SCORE_THRESHOLDS[i]
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


def evaluate_mmdet_results(results, gt_infos):
    """Compute per-class, superclass, official, and merged metrics.

    Args:
        results (list[list[np.ndarray]]): one entry per image; each entry is
            a list of 25 ``(N, 5)`` arrays (xyxy + score) indexed by class.
            Filtering by per-class thresholds is applied internally.
        gt_infos (list[list[dict]]): one list per image; each ann dict has
            ``bbox`` (xywh) and ``category_id``. Image order must match
            ``results``.

    Returns:
        dict with keys ``per_class`` (list of dicts, one per category id 0..24),
        ``by_super`` (dict ship/aircraft/vehicle → recall/fdr),
        ``official`` (dict recall/fdr averaged across the superclasses that
        have at least one GT), and ``merged`` (counts-based recall/fdr over
        every category).
    """
    if len(results) != len(gt_infos):
        raise ValueError(
            f'results has {len(results)} entries but gt_infos has '
            f'{len(gt_infos)}; must match image count')
    per_class_tp = [0] * len(CLASS_NAMES)
    per_class_fp = [0] * len(CLASS_NAMES)
    per_class_fn = [0] * len(CLASS_NAMES)
    for img_results, img_gts in zip(results, gt_infos):
        # img_results may be a list of 25 arrays (mmdet style) or a single
        # ndarray when only one image is processed.
        if isinstance(img_results, list):
            per_class = img_results
        else:
            raise ValueError(
                'each per-image entry must be a list of 25 class arrays')
        filtered = filter_mmdet_results(per_class)
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
            pred = filtered[cls_idx]
            if len(pred) == 0:
                pred_boxes = np.zeros((0, 4), dtype=np.float32)
                pred_scores = np.zeros((0,), dtype=np.float32)
            else:
                pred_boxes = _convert_xyxy_to_xywh(pred)
                pred_scores = pred[:, 4].astype(np.float32)
            tau = CLASS_IOU_THRESHOLDS[cls_idx]
            tp, fp, fn = _match_class(
                pred_boxes, pred_scores, gt_arr, tau)
            per_class_tp[cls_idx] += tp
            per_class_fp[cls_idx] += fp
            per_class_fn[cls_idx] += fn

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

    by_super = {}
    official_recalls = []
    official_fdrs = []
    for super_name, ids in SUPERCLASS_INDICES.items():
        rs = [per_class_rows[i]['recall'] for i in ids]
        fs = [per_class_rows[i]['fdr'] for i in ids]
        gt_total = sum(per_class_rows[i]['tp'] + per_class_rows[i]['fn']
                       for i in ids)
        if gt_total == 0:
            by_super[super_name] = {'recall': None, 'fdr': None}
            continue
        r = sum(rs) / len(rs)
        f = sum(fs) / len(fs)
        by_super[super_name] = {'recall': r, 'fdr': f}
        official_recalls.append(r)
        official_fdrs.append(f)
    if official_recalls:
        official = {
            'recall': sum(official_recalls) / len(official_recalls),
            'fdr': sum(official_fdrs) / len(official_fdrs),
        }
    else:
        official = {'recall': 0.0, 'fdr': 0.0}

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
    'CLASS_SCORE_THRESHOLDS',
    'CLASS_IOU_THRESHOLDS',
    'SUPERCLASS_INDICES',
    'filter_mmdet_results',
    'evaluate_mmdet_results',
    'compare_official_candidates',
]
