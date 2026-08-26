"""Automatically search class-wise score thresholds for Recall/FDR.

The matching protocol is identical to ``tools/eval_recall_fdr.py``:

* predictions are matched score-descending and one-to-one per image/class;
* FSC (class 24) uses IoU 0.35 and all other classes use IoU 0.50;
* official metrics average subtype metrics within ship/aircraft/vehicle, then
  average the three super classes.

The script first builds an exact threshold curve for every class. It then
uses a conservative, discretised dynamic programme to maximise official
recall while keeping official FDR below ``--max-official-fdr``. Thresholds
must be selected on validation data, never on the final test set.

Example:

    python tools/search_recall_fdr_thresholds.py \
        --pred work_dirs/arfc_only_v2/dense/val_preds_dense.json \
        --gt data/annotations/instances_val.json \
        --checkpoint work_dirs/arfc_only_v2/best_official_recall_fdr.pth \
        --names HM,LQS,QHS,MS,A1_SU-35,A2_C-130,A3_C-17,A4_C-5,A5_F-16,A6_TU-160,A7_E-3,A8_B-52,A9_P-3C,A10_B-1B,A11_E-8,A12_TU-22,A13_F-15,A14_KC-135,A15_F-22,A16_FA-18,A17_TU-95,A18_KC-10,A19_SU-34,A20_SU-24,FSC \
        --max-official-fdr 0.19 \
        --target-official-recall 0.85 \
        --out-prefix work_dirs/arfc_only_v2/dense/threshold_search

Outputs:

* ``<prefix>.json``: selected thresholds and complete metrics;
* ``<prefix>_selected.csv``: per-class selected operating points;
* ``<prefix>_global_curve.csv``: common/global threshold sweep;
* ``<prefix>_class_curves.csv``: all exact per-class operating points;
* ``<prefix>_filtered_preds.json``: predictions filtered by selected values.
"""

import argparse
import csv
import json
import math
import os
import os.path as osp
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from mmdet.core.evaluation import (CLASS_IOU_THRESHOLDS, CLASS_NAMES,
                                   SUPERCLASS_INDICES, match_class_events,
                                   write_threshold_artifact)


DEFAULT_GRID = (
    0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15,
    0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 0.90,
)
SUPER_CLASSES = tuple(SUPERCLASS_INDICES)
SUPERCLASS_OF_ID = {
    class_id: super_name
    for super_name, class_ids in SUPERCLASS_INDICES.items()
    for class_id in class_ids
}


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    tp: int
    fp: int
    fn: int
    recall: float
    fdr: float
    precision: float


def parse_grid(value):
    values = sorted({float(item) for item in value.split(',')})
    if not values:
        raise argparse.ArgumentTypeError('threshold grid cannot be empty')
    if any(not math.isfinite(item) or item < 0 for item in values):
        raise argparse.ArgumentTypeError(
            'thresholds must be finite non-negative numbers')
    return values


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_inputs(pred, gt, class_count):
    for label, payload in (('prediction', pred), ('ground truth', gt)):
        if ('images' not in payload or 'annotations' not in payload
                or 'categories' not in payload):
            raise ValueError(
                f'{label} JSON must contain images, annotations, '
                'and categories')

    category_maps = {}
    for label, payload in (('prediction', pred), ('ground truth', gt)):
        categories = payload['categories']
        if not isinstance(categories, list) or len(categories) != class_count:
            raise ValueError(
                f'{label} categories must contain exactly {class_count} rows')
        category_map = {}
        for category in categories:
            category_id = category.get('id')
            name = category.get('name')
            if (not isinstance(category_id, int)
                    or not 0 <= category_id < class_count
                    or category_id in category_map
                    or not isinstance(name, str)):
                raise ValueError(f'{label} categories are malformed')
            category_map[category_id] = name
        if set(category_map) != set(range(class_count)):
            raise ValueError(
                f'{label} categories must cover ids 0..{class_count - 1}')
        category_maps[label] = category_map
    if category_maps['prediction'] != category_maps['ground truth']:
        raise ValueError('prediction/GT categories differ')

    image_maps = {}
    for label, payload in (('prediction', pred), ('ground truth', gt)):
        image_map = {}
        for image in payload['images']:
            image_id = image.get('id')
            if image_id in image_map:
                raise ValueError(
                    f'{label} contains duplicate image id={image_id}')
            image_map[image_id] = image.get('file_name')
        image_maps[label] = image_map
    pred_images = image_maps['prediction']
    gt_images = image_maps['ground truth']
    if set(pred_images) != set(gt_images):
        missing = sorted(set(gt_images) - set(pred_images))
        extra = sorted(set(pred_images) - set(gt_images))
        raise ValueError(
            'prediction/GT image-id sets differ: '
            f'missing predictions for {len(missing)} IDs, '
            f'extra prediction IDs={len(extra)}')

    mismatches = [
        (image_id, pred_images[image_id], gt_images[image_id])
        for image_id in gt_images
        if pred_images[image_id] is not None
        and gt_images[image_id] is not None
        and pred_images[image_id] != gt_images[image_id]
    ]
    if mismatches:
        image_id, pred_name, gt_name = mismatches[0]
        raise ValueError(
            f'prediction/GT file mismatch for image_id={image_id}: '
            f'pred={pred_name!r}, gt={gt_name!r}; check the dataset split')

    gt_ids = set(gt_images)
    for ann in pred['annotations']:
        if ann.get('image_id') not in gt_ids:
            raise ValueError(
                f'prediction annotation references unknown image_id='
                f'{ann.get("image_id")}')
        class_id = ann.get('category_id')
        if not isinstance(class_id, int) or not 0 <= class_id < class_count:
            raise ValueError(f'invalid prediction category_id={class_id}')
        if 'score' not in ann or not math.isfinite(float(ann['score'])):
            raise ValueError('every prediction must have a finite score')
        if len(ann.get('bbox', [])) != 4:
            raise ValueError('every prediction bbox must contain four values')

    for ann in gt['annotations']:
        if ann.get('image_id') not in gt_ids:
            raise ValueError(
                f'GT annotation references unknown image_id='
                f'{ann.get("image_id")}')
        class_id = ann.get('category_id')
        if not isinstance(class_id, int) or not 0 <= class_id < class_count:
            raise ValueError(f'invalid GT category_id={class_id}')
        if len(ann.get('bbox', [])) != 4:
            raise ValueError('every GT bbox must contain four values')


def resolve_names(args, pred, class_count):
    if args.names:
        names = [name.strip() for name in args.names.split(',')]
    else:
        by_id = {
            int(category['id']): str(category['name'])
            for category in pred.get('categories', [])
        }
        names = [by_id.get(class_id, str(class_id))
                 for class_id in range(class_count)]
    if len(names) != class_count:
        raise ValueError(
            f'expected {class_count} class names, received {len(names)}')
    return names


def build_ranked_events(pred, gt, class_count):
    """Return score/TP events per class under official greedy matching."""
    gt_by_key = defaultdict(list)
    pred_by_key = defaultdict(list)
    total_gt = [0] * class_count

    for ann in gt['annotations']:
        class_id = int(ann['category_id'])
        gt_by_key[(ann['image_id'], class_id)].append(ann['bbox'])
        total_gt[class_id] += 1
    for ann in pred['annotations']:
        class_id = int(ann['category_id'])
        pred_by_key[(ann['image_id'], class_id)].append(
            (ann['bbox'], float(ann['score'])))

    events = {class_id: [] for class_id in range(class_count)}
    for key, predictions in pred_by_key.items():
        _, class_id = key
        boxes = np.asarray([item[0] for item in predictions],
                           dtype=np.float32).reshape(-1, 4)
        scores = np.asarray([item[1] for item in predictions],
                            dtype=np.float64)
        gt_boxes = np.asarray(gt_by_key.get(key, []),
                              dtype=np.float32).reshape(-1, 4)
        events[class_id].extend(match_class_events(
            boxes, scores, gt_boxes, CLASS_IOU_THRESHOLDS[class_id]))

    for class_id in events:
        events[class_id].sort(key=lambda item: item[0], reverse=True)
    return events, total_gt


def make_point(threshold, tp, fp, total_gt):
    fn = total_gt - tp
    recall = tp / max(total_gt, 1)
    fdr = fp / max(tp + fp, 1)
    precision = tp / max(tp + fp, 1)
    return OperatingPoint(
        threshold=float(threshold),
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        recall=float(recall),
        fdr=float(fdr),
        precision=float(precision),
    )


def build_exact_curve(events, total_gt):
    """Build one point after every tied-score group, plus no predictions."""
    if events:
        empty_threshold = events[0][0] + max(
            abs(events[0][0]) * 1e-12, 1e-12)
    else:
        empty_threshold = 1.0
    curve = [make_point(empty_threshold, 0, 0, total_gt)]
    tp = 0
    fp = 0
    index = 0
    while index < len(events):
        score = events[index][0]
        while index < len(events) and events[index][0] == score:
            if events[index][1]:
                tp += 1
            else:
                fp += 1
            index += 1
        curve.append(make_point(score, tp, fp, total_gt))
    return curve


def point_at_threshold(events, total_gt, threshold):
    tp = 0
    fp = 0
    for score, is_tp in events:
        if score < threshold:
            break
        if is_tp:
            tp += 1
        else:
            fp += 1
    return make_point(threshold, tp, fp, total_gt)


def pareto_points(curve):
    """Drop points dominated by another point with <=FDR and >=Recall."""
    ordered = sorted(
        curve,
        key=lambda point: (point.fdr, -point.recall, point.fp,
                           -point.threshold),
    )
    result = []
    best_recall = -1.0
    for point in ordered:
        if point.recall > best_recall + 1e-12:
            result.append(point)
            best_recall = point.recall
    return result


def official_weights(names):
    members = defaultdict(list)
    for class_id, _ in enumerate(names):
        members[SUPERCLASS_OF_ID[class_id]].append(class_id)
    missing = [name for name in SUPER_CLASSES if not members[name]]
    if missing:
        raise ValueError(f'missing official super classes: {missing}')
    return {
        class_id: 1.0 / (len(SUPER_CLASSES) * len(class_ids))
        for _, class_ids in members.items()
        for class_id in class_ids
    }


def search_class_thresholds(curves, weights, max_fdr, budget_bins):
    """Conservative multiple-choice knapsack over class operating points."""
    candidates = {
        class_id: pareto_points(curve)
        for class_id, curve in curves.items()
    }
    # cost_bin -> (official_recall contribution, total FP, selections)
    states = {0: (0.0, 0, tuple())}
    for class_id in sorted(candidates):
        new_states = {}
        for old_cost, (old_recall, old_fp, selections) in states.items():
            for point in candidates[class_id]:
                raw_cost = weights[class_id] * point.fdr
                point_cost = int(math.ceil(
                    raw_cost * budget_bins / max_fdr - 1e-12))
                new_cost = old_cost + point_cost
                if new_cost > budget_bins:
                    continue
                new_recall = old_recall + weights[class_id] * point.recall
                new_fp = old_fp + point.fp
                current = new_states.get(new_cost)
                if (current is None
                        or new_recall > current[0] + 1e-12
                        or (abs(new_recall - current[0]) <= 1e-12
                            and new_fp < current[1])):
                    new_states[new_cost] = (
                        new_recall, new_fp, selections + ((class_id, point), ))

        # A higher-cost state with no recall improvement cannot help later.
        pruned = {}
        best_recall = -1.0
        for cost in sorted(new_states):
            state = new_states[cost]
            if state[0] > best_recall + 1e-12:
                pruned[cost] = state
                best_recall = state[0]
        states = pruned
        if not states:
            raise RuntimeError(
                f'no feasible state remained after class {class_id}')

    best_cost, best_state = max(
        states.items(), key=lambda item: (item[1][0], -item[0], -item[1][1]))
    del best_cost
    return {class_id: point for class_id, point in best_state[2]}


def aggregate_metrics(points, names):
    by_super_rows = defaultdict(list)
    for class_id, point in points.items():
        by_super_rows[SUPERCLASS_OF_ID[class_id]].append(point)

    by_super = {}
    for super_name in SUPER_CLASSES:
        rows = by_super_rows[super_name]
        by_super[super_name] = {
            'recall': sum(row.recall for row in rows) / len(rows),
            'fdr': sum(row.fdr for row in rows) / len(rows),
        }
    official_recall = sum(
        by_super[name]['recall'] for name in SUPER_CLASSES) / len(SUPER_CLASSES)
    official_fdr = sum(
        by_super[name]['fdr'] for name in SUPER_CLASSES) / len(SUPER_CLASSES)

    tp = sum(point.tp for point in points.values())
    fp = sum(point.fp for point in points.values())
    fn = sum(point.fn for point in points.values())
    return {
        'merged': {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'recall': tp / max(tp + fn, 1),
            'fdr': fp / max(tp + fp, 1),
        },
        'official': {
            'recall': official_recall,
            'fdr': official_fdr,
            'by_super': by_super,
        },
    }


def write_csv(path, header, rows):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_outputs(args, pred, names, events, total_gt, curves, selected,
                  metrics):
    parent = osp.dirname(args.out_prefix)
    if parent:
        os.makedirs(parent, exist_ok=True)

    constraints = {
        'max_official_fdr': args.max_official_fdr,
        'target_official_recall': args.target_official_recall,
        'budget_bins': args.budget_bins,
    }
    write_threshold_artifact(
        args.out_prefix + '.json',
        [selected[class_id].threshold for class_id in range(len(names))],
        args.checkpoint,
        args.pred,
        args.gt,
        constraints,
        metrics,
    )

    write_csv(
        args.out_prefix + '_selected.csv',
        ['category_id', 'name', 'threshold', 'tp', 'fp', 'fn',
         'recall', 'fdr', 'precision'],
        [
            [class_id, names[class_id], f'{point.threshold:.9g}', point.tp,
             point.fp, point.fn, f'{point.recall:.6f}', f'{point.fdr:.6f}',
             f'{point.precision:.6f}']
            for class_id, point in sorted(selected.items())
        ],
    )

    class_curve_rows = []
    for class_id, curve in sorted(curves.items()):
        for point in curve:
            class_curve_rows.append([
                class_id, names[class_id], f'{point.threshold:.9g}', point.tp,
                point.fp, point.fn, f'{point.recall:.6f}',
                f'{point.fdr:.6f}', f'{point.precision:.6f}',
            ])
    write_csv(
        args.out_prefix + '_class_curves.csv',
        ['category_id', 'name', 'threshold', 'tp', 'fp', 'fn',
         'recall', 'fdr', 'precision'],
        class_curve_rows,
    )

    global_rows = []
    for threshold in args.grid:
        points = {
            class_id: point_at_threshold(
                events[class_id], total_gt[class_id], threshold)
            for class_id in curves
        }
        row_metrics = aggregate_metrics(points, names)
        official = row_metrics['official']
        merged = row_metrics['merged']
        global_rows.append([
            f'{threshold:.9g}',
            f'{official["recall"]:.6f}', f'{official["fdr"]:.6f}',
            f'{official["by_super"]["ship"]["recall"]:.6f}',
            f'{official["by_super"]["ship"]["fdr"]:.6f}',
            f'{official["by_super"]["aircraft"]["recall"]:.6f}',
            f'{official["by_super"]["aircraft"]["fdr"]:.6f}',
            f'{official["by_super"]["vehicle"]["recall"]:.6f}',
            f'{official["by_super"]["vehicle"]["fdr"]:.6f}',
            merged['tp'], merged['fp'], merged['fn'],
            f'{merged["recall"]:.6f}', f'{merged["fdr"]:.6f}',
        ])
    write_csv(
        args.out_prefix + '_global_curve.csv',
        ['threshold', 'official_recall', 'official_fdr',
         'ship_recall', 'ship_fdr', 'aircraft_recall', 'aircraft_fdr',
         'vehicle_recall', 'vehicle_fdr', 'merged_tp', 'merged_fp',
         'merged_fn', 'merged_recall', 'merged_fdr'],
        global_rows,
    )

    thresholds = {
        class_id: point.threshold for class_id, point in selected.items()
    }
    filtered = dict(pred)
    filtered['annotations'] = [
        ann for ann in pred['annotations']
        if float(ann['score']) >= thresholds[int(ann['category_id'])]
    ]
    with open(args.out_prefix + '_filtered_preds.json', 'w',
              encoding='utf-8') as f:
        json.dump(filtered, f, ensure_ascii=False)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Search class-wise thresholds under official FDR limit.')
    parser.add_argument('--pred', required=True,
                        help='Dense COCO-style prediction JSON with scores.')
    parser.add_argument('--gt', required=True,
                        help='COCO ground-truth JSON for the same split.')
    parser.add_argument('--classes', type=int, default=25)
    parser.add_argument('--names',
                        help='Comma-separated class names. Defaults to pred categories.')
    parser.add_argument('--checkpoint', required=True,
                        help='Best checkpoint to bind into the artifact.')
    parser.add_argument('--max-official-fdr', type=float, default=0.19)
    parser.add_argument('--target-official-recall', type=float, default=0.85)
    parser.add_argument('--budget-bins', type=int, default=10000,
                        help='FDR budget resolution; larger is more exact.')
    parser.add_argument('--grid', type=parse_grid,
                        default=list(DEFAULT_GRID),
                        help='Comma-separated common thresholds for global curve.')
    parser.add_argument('--out-prefix', required=True)
    return parser


def main():
    args = build_parser().parse_args()
    if not 0 < args.max_official_fdr < 1:
        raise ValueError('--max-official-fdr must be between 0 and 1')
    if not 0 <= args.target_official_recall <= 1:
        raise ValueError('--target-official-recall must be between 0 and 1')
    if args.classes <= 0:
        raise ValueError('--classes must be positive')
    if args.budget_bins <= 0:
        raise ValueError('--budget-bins must be positive')

    pred = load_json(args.pred)
    gt = load_json(args.gt)
    validate_inputs(pred, gt, args.classes)
    names = resolve_names(args, pred, args.classes)
    if args.classes != len(CLASS_NAMES) or tuple(names) != CLASS_NAMES:
        raise ValueError(
            'threshold search requires the official 25-class names/order')
    category_names = tuple(
        category['name']
        for category in sorted(pred['categories'], key=lambda row: row['id']))
    if category_names != CLASS_NAMES:
        raise ValueError(
            'prediction/GT categories must use the official 25-class names')
    events, total_gt = build_ranked_events(pred, gt, args.classes)
    missing_superclasses = [
        super_name for super_name, class_ids in SUPERCLASS_INDICES.items()
        if sum(total_gt[class_id] for class_id in class_ids) == 0
    ]
    if missing_superclasses:
        raise ValueError(
            'threshold search requires GT in every superclass; missing '
            f'{missing_superclasses}')
    curves = {
        class_id: build_exact_curve(events[class_id], total_gt[class_id])
        for class_id in range(args.classes)
    }
    weights = official_weights(names)
    selected = search_class_thresholds(
        curves, weights, args.max_official_fdr, args.budget_bins)
    if set(selected) != set(range(args.classes)):
        raise RuntimeError('threshold search did not select every class')
    metrics = aggregate_metrics(selected, names)
    if metrics['official']['fdr'] > args.max_official_fdr + 1e-12:
        raise RuntimeError(
            'internal error: selected official FDR exceeds its budget')
    write_outputs(args, pred, names, events, total_gt, curves, selected,
                  metrics)

    official = metrics['official']
    merged = metrics['merged']
    passed = (official['recall'] >= args.target_official_recall
              and official['fdr'] <= args.max_official_fdr)
    print('Selected class-wise thresholds')
    print(f'official Recall={official["recall"]:.4f}  '
          f'FDR={official["fdr"]:.4f}  PASS={passed}')
    for super_name in SUPER_CLASSES:
        values = official['by_super'][super_name]
        print(f'{super_name:<8s} Recall={values["recall"]:.4f}  '
              f'FDR={values["fdr"]:.4f}')
    print(f'merged   Recall={merged["recall"]:.4f}  '
          f'FDR={merged["fdr"]:.4f}')
    print(f'Wrote outputs with prefix: {args.out_prefix}')


if __name__ == '__main__':
    main()
