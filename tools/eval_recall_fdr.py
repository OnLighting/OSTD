"""Competition-grade evaluation — NOT COCO mAP.

Implements the §2 protocol from data/DATASET_OVERVIEW.md using the
shared ``mmdet.core.evaluation.official_metrics`` module so the CLI,
training EvalHook, and big-image inference all agree on:

  - Score-desc one-to-one matching.
  - Per-prediction IoU threshold τ (FSC=24 → 0.35, others → 0.5).
  - Duplicate matches (one GT, multiple preds) count the top-1 as TP, the rest
    as FP.
  - Unmatched preds → FP. Unmatched GT → FN.
  - 25 fixed per-class score thresholds from ``CLASS_SCORE_THRESHOLDS``.

Inputs:
  --pred  json produced by tools/infer_big_image.py (COCO-style with `score`).
  --gt    json in the same COCO format
          (e.g. data/annotations/instances_val.json).

Per-image TP/FP/FN aggregates to per-class and overall Recall/FDR. Each
per-class row also reports Precision and 101-point interpolated AP under the
same class-specific IoU threshold used by TP/FP matching. Pass --out-prefix
to additionally write <prefix>.json and <prefix>.csv.
"""

import argparse
import csv
import json
import math
import os
import os.path as osp
from collections import defaultdict

import numpy as np

from mmdet.core.evaluation import CLASS_IOU_THRESHOLDS, CLASS_NAMES


def _iou_xywh(box, boxes):
    """Vector IoU between one box and an array of boxes (COCO xywh)."""
    if len(boxes) == 0:
        return np.zeros((0,), np.float32)
    x1 = box[0]
    y1 = box[1]
    w1 = box[2]
    h1 = box[3]
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


def group_by_image_and_class(records, has_score):
    """records: list of COCO ann dicts. Returns:
        image_id -> class_id -> {boxes (N,4), scores (N,), ids (N,)}
    """
    grouped = defaultdict(
        lambda: defaultdict(lambda: {
            'boxes': [], 'scores': []
        }))
    if has_score:
        for r in records:
            grouped[r['image_id']][r['category_id']]['boxes'].append(r['bbox'])
            grouped[r['image_id']][r['category_id']]['scores'].append(
                r['score'])
    else:
        for r in records:
            grouped[r['image_id']][r['category_id']]['boxes'].append(r['bbox'])
    # Convert to numpy.
    out = {}
    for img_id, by_cls in grouped.items():
        out[img_id] = {}
        for cls_id, d in by_cls.items():
            boxes = np.asarray(d['boxes'], dtype=np.float32).reshape(-1, 4)
            entry = {'boxes': boxes}
            if has_score:
                entry['scores'] = np.asarray(d['scores'], dtype=np.float32)
            out[img_id][cls_id] = entry
    return out


def aggregate_confusion_counts(pred_by_img, gt_by_img, image_ids):
    """Aggregate per-class TP, FP, and FN across all official images.

    Implemented locally (rather than going through
    ``evaluate_mmdet_results``) so this CLI can also report AP, which
    the shared module deliberately omits. The matching logic mirrors
    ``official_metrics._match_class`` to stay consistent.
    """
    total_tp = defaultdict(int)
    total_fp = defaultdict(int)
    total_fn = defaultdict(int)
    for img_id in image_ids:
        preds = pred_by_img.get(img_id, {})
        gts = gt_by_img.get(img_id, {})
        classes = set(preds.keys()) | set(gts.keys())
        for c in classes:
            gt_entry = gts.get(
                c, {'boxes': np.zeros((0, 4), dtype=np.float32)})
            p_entry = preds.get(
                c, {
                    'boxes': np.zeros((0, 4), dtype=np.float32),
                    'scores': np.zeros((0,), dtype=np.float32),
                })
            gt_boxes = gt_entry['boxes']
            n_gt = len(gt_boxes)
            n_p = len(p_entry['boxes'])
            gt_matched = np.zeros(n_gt, dtype=bool)
            pr_matched = np.zeros(n_p, dtype=bool)
            if n_p == 0:
                total_fn[c] += n_gt
                continue
            order = np.argsort(-p_entry['scores'])
            tau = CLASS_IOU_THRESHOLDS[int(c)]
            for pi in order:
                b = p_entry['boxes'][pi]
                best_iou, best_gi = 0.0, -1
                for gi in range(n_gt):
                    if gt_matched[gi]:
                        continue
                    iou = _iou_xywh(b, gt_boxes[gi:gi + 1])[0]
                    if iou > best_iou:
                        best_iou, best_gi = iou, gi
                if best_gi >= 0 and best_iou >= tau:
                    gt_matched[best_gi] = True
                    pr_matched[pi] = True
            total_tp[c] += int(pr_matched.sum())
            total_fp[c] += int((~pr_matched).sum())
            total_fn[c] += int((~gt_matched).sum())
    return total_tp, total_fp, total_fn


def average_precision_for_class(pred_by_img, gt_by_img, class_id, image_ids):
    """Compute 101-point AP with the same class-specific IoU matching rule."""
    total_gt = 0
    matched_by_img = {}
    predictions = []
    for img_id in image_ids:
        gt_entry = gt_by_img.get(img_id, {}).get(
            class_id, {'boxes': np.zeros((0, 4), dtype=np.float32)})
        num_gt = len(gt_entry['boxes'])
        total_gt += num_gt
        matched_by_img[img_id] = np.zeros(num_gt, dtype=bool)

        pred_entry = pred_by_img.get(img_id, {}).get(class_id)
        if pred_entry is None:
            continue
        for box, score in zip(pred_entry['boxes'], pred_entry['scores']):
            predictions.append((float(score), img_id, box))

    if total_gt == 0:
        return float('nan')
    if not predictions:
        return 0.0

    predictions.sort(key=lambda item: item[0], reverse=True)
    tp_flags = np.zeros(len(predictions), dtype=np.float64)
    fp_flags = np.ones(len(predictions), dtype=np.float64)
    tau = CLASS_IOU_THRESHOLDS[int(class_id)]

    for pred_idx, (_, img_id, box) in enumerate(predictions):
        gt_entry = gt_by_img.get(img_id, {}).get(
            class_id, {'boxes': np.zeros((0, 4), dtype=np.float32)})
        gt_boxes = gt_entry['boxes']
        unmatched = np.flatnonzero(~matched_by_img[img_id])
        if unmatched.size == 0:
            continue
        overlaps = _iou_xywh(box, gt_boxes[unmatched])
        best_local = int(overlaps.argmax())
        if overlaps[best_local] >= tau:
            matched_by_img[img_id][unmatched[best_local]] = True
            tp_flags[pred_idx] = 1.0
            fp_flags[pred_idx] = 0.0

    cumulative_tp = np.cumsum(tp_flags)
    cumulative_fp = np.cumsum(fp_flags)
    recalls = cumulative_tp / total_gt
    precisions = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp, 1e-12)
    recall_points = np.linspace(0.0, 1.0, 101)
    interpolated = [
        precisions[recalls >= recall].max()
        if np.any(recalls >= recall) else 0.0
        for recall in recall_points
    ]
    return float(np.mean(interpolated))


def super_of(name):
    """Map a class name to its official三大类 group."""
    if name in {'HM', 'LQS', 'QHS', 'MS'}:
        return 'ship'
    if name == 'FSC':
        return 'vehicle'
    if isinstance(name, str) and name.startswith('A'):
        return 'aircraft'
    return None


def build_metrics_payload(overall, per_class_rows):
    """Build the JSON-serializable evaluation result."""
    per_class_json = []
    for row in per_class_rows:
        cid, name, tp, fp, fn, rec, fdr_v, prec_v, ap, ap_tau = row
        per_class_json.append({
            'category_id': int(cid),
            'name': name,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'recall': float(rec),
            'fdr': float(fdr_v),
            'prec': None if math.isnan(float(prec_v)) else float(prec_v),
            'ap': None if math.isnan(float(ap)) else float(ap),
            'ap_iou_thr': float(ap_tau),
        })
    return {'overall': overall, 'per_class': per_class_json}


def build_per_class_csv(per_class_rows):
    """Build CSV rows, including per-class AP and its IoU threshold."""
    rows = [[
        'category_id', 'name', 'tp', 'fp', 'fn', 'recall', 'fdr', 'prec',
        'ap', 'ap_iou_thr'
    ]]
    for row in per_class_rows:
        cid, name, tp, fp, fn, rec, fdr_v, prec_v, ap, ap_tau = row
        prec_cell = '' if math.isnan(float(prec_v)) else f'{prec_v:.4f}'
        ap_cell = '' if math.isnan(float(ap)) else f'{ap:.4f}'
        rows.append([
            cid, name, tp, fp, fn, f'{rec:.4f}', f'{fdr_v:.4f}',
            prec_cell, ap_cell, f'{ap_tau:.2f}'
        ])
    return rows


def _emit_files(prefix, overall, per_class_rows, class_names):
    """Write overall + per-class summary to <prefix>.json and per-class table
    to <prefix>.csv. Creates parent dir if missing. Overwrites existing files.

    Args:
        prefix (str): path prefix; '.json' and '.csv' are appended.
        overall (dict): Aggregate counts, rates, and class-mean AP.
        per_class_rows (list[tuple]): one tuple per class with shape
            (category_id, name, tp, fp, fn, recall, fdr, prec, ap, ap_tau).
        class_names (list[str]): not used by JSON, kept for symmetry / future.
    """
    parent = osp.dirname(prefix)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # --- JSON ---
    with open(prefix + '.json', 'w', encoding='utf-8') as f:
        json.dump(
            build_metrics_payload(overall, per_class_rows),
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --- CSV ---
    with open(prefix + '.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerows(build_per_class_csv(per_class_rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True)
    parser.add_argument('--gt', required=True)
    parser.add_argument('--classes', type=int, default=25)
    parser.add_argument(
        '--names',
        help='Optional comma-separated class names for per-class print.')
    parser.add_argument(
        '--out-prefix',
        default=None,
        help='Write <prefix>.json and <prefix>.csv with per-class metrics.')
    args = parser.parse_args()

    with open(args.pred, 'r', encoding='utf-8') as f:
        pred = json.load(f)
    with open(args.gt, 'r', encoding='utf-8') as f:
        gt = json.load(f)

    has_score = (
        bool(pred['annotations']) and 'score' in pred['annotations'][0])
    pred_by_img = group_by_image_and_class(
        pred['annotations'], has_score=has_score)
    gt_by_img = group_by_image_and_class(gt['annotations'], has_score=False)
    image_ids = [image['id'] for image in gt['images']]

    # Ensure every GT image has an entry (zero predictions).
    for img_id in image_ids:
        pred_by_img.setdefault(img_id, {})

    total_tp, total_fp, total_fn = aggregate_confusion_counts(
        pred_by_img, gt_by_img, image_ids)

    if args.names:
        names = args.names.split(',')
    else:
        names = list(CLASS_NAMES)

    overall_tp = sum(total_tp.values())
    overall_fp = sum(total_fp.values())
    overall_fn = sum(total_fn.values())
    recall = overall_tp / max(overall_tp + overall_fn, 1)
    fdr = overall_fp / max(overall_fp + overall_tp, 1)

    print(f'Overall  TP={overall_tp}  FP={overall_fp}  FN={overall_fn}  '
          f'Recall={recall:.4f}  FDR={fdr:.4f}')
    rows = []
    valid_aps = []
    for c in range(args.classes):
        tp = total_tp[c]
        fp = total_fp[c]
        fn = total_fn[c]
        r = tp / max(tp + fn, 1)
        f = fp / max(fp + tp, 1)
        p = tp / max(tp + fp, 1) if (tp + fp) > 0 else float('nan')
        ap = average_precision_for_class(
            pred_by_img, gt_by_img, c, image_ids)
        ap_tau = CLASS_IOU_THRESHOLDS[c]
        if not math.isnan(ap):
            valid_aps.append(ap)
        rows.append((
            c, names[c] if c < len(names) else '?', tp, fp, fn, r, f, p,
            ap, ap_tau))
    mean_ap = sum(valid_aps) / len(valid_aps) if valid_aps else float('nan')
    print(f'mAP@class-specific-IoU={mean_ap:.4f}')
    print()
    print(f'{"cls":>3s}  {"name":<14s}  {"TP":>5s}  {"FP":>5s}  {"FN":>5s}  '
          f'{"Recall":>7s}  {"FDR":>7s}  {"Prec":>7s}  {"AP":>7s}  '
          f'{"IoU":>4s}')
    for r in rows:
        c, name, tp, fp, fn, rv, fv, pv, ap, ap_tau = r
        prec_s = '   nan' if tp + fp == 0 else f'{pv:.4f}'
        ap_s = '   nan' if math.isnan(ap) else f'{ap:.4f}'
        print(f'{c:>3d}  {name[:14]:<14s}  {tp:>5d}  {fp:>5d}  {fn:>5d}  '
              f'{rv:>7.4f}  {fv:>7.4f}  {prec_s:>7s}  {ap_s:>7s}  '
              f'{ap_tau:>4.2f}')

    # P0-A: 三大类官方补充口径聚合
    super_recalls = defaultdict(list)
    super_fdrs = defaultdict(list)
    for row in rows:
        cid, name, *_ = row
        sn = super_of(name)
        if sn is None:
            continue
        super_recalls[sn].append(row[5])  # recall
        super_fdrs[sn].append(row[6])     # fdr
    super_avg = {}
    for s in ('ship', 'aircraft', 'vehicle'):
        rs = super_recalls.get(s, [])
        fs = super_fdrs.get(s, [])
        if not rs:
            super_avg[s] = (None, None)
            continue
        super_avg[s] = (sum(rs) / len(rs), sum(fs) / len(fs))
    valid_recalls = [v[0] for v in super_avg.values() if v[0] is not None]
    valid_fdrs = [v[1] for v in super_avg.values() if v[1] is not None]
    official_recall = (sum(valid_recalls) / len(valid_recalls)) if valid_recalls else float('nan')
    official_fdr = (sum(valid_fdrs) / len(valid_fdrs)) if valid_fdrs else float('nan')

    print()
    print('=== 官方补充口径（三大类均值再平均） ===')
    for s in ('ship', 'aircraft', 'vehicle'):
        r, f = super_avg[s]
        if r is None:
            print(f'{s:<8s}  (empty)')
        else:
            print(f'{s:<8s}  R={r:.4f}  FDR={f:.4f}')
    print('-' * 41)
    print(f'official R={official_recall:.4f}  FDR={official_fdr:.4f}')

    if args.out_prefix:
        overall = {
            'tp': int(overall_tp),
            'fp': int(overall_fp),
            'fn': int(overall_fn),
            'recall': float(recall),
            'fdr': float(fdr),
            'prec': (
                overall_tp / max(overall_tp + overall_fp, 1)
                if (overall_tp + overall_fp) > 0 else None),
            'map': None if math.isnan(mean_ap) else float(mean_ap),
        }
        # P0-A: official三大类字段
        overall['official'] = {
            'recall': float(official_recall),
            'fdr': float(official_fdr),
            'by_super': {
                s: {
                    'recall': None if super_avg[s][0] is None else float(super_avg[s][0]),
                    'fdr':    None if super_avg[s][1] is None else float(super_avg[s][1]),
                } for s in ('ship', 'aircraft', 'vehicle')
            },
        }
        _emit_files(args.out_prefix, overall, rows, names)


if __name__ == '__main__':
    main()
