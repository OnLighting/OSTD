"""Batch-run inference on the val split and emit a single COCO-style JSON.

Output is consumed by tools/eval_recall_fdr.py. Each detected box keeps its
score so the Recall/FDR protocol's score-desc one-to-one matching works.
This avoids the pkl-format gap in tools/test.py — for the competition metric
(score-ranked, class-conditional IoU) we need per-box scores, which pkl
doesn't carry cleanly.

By default the script exports dense predictions at the shared candidate
floor for post-training threshold search. Passing ``--thresholds`` loads a
checkpoint-bound frozen artifact and applies its 25 class thresholds.

Usage:
    python tools/eval_val_to_json.py \
        --config configs/bafnet/aircraft_bafnet_1x.py \
        --checkpoint work_dirs/aircraft_bafnet_1x/latest.pth \
        --img-dir data/images/val \
        --gt data/annotations/instances_val.json \
        --out work_dirs/val_preds.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from mmcv import Config
from mmdet.apis import init_detector, inference_detector

from mmdet.core.evaluation import (CANDIDATE_SCORE_FLOOR, CLASS_NAMES,
                                   filter_mmdet_results,
                                   load_threshold_artifact)

from sbla_config import apply_model_ablation_config


def detections_to_coco_annotations(result, image_id, next_ann_id,
                                   score_thresholds=None):
    """Convert a per-image mmdet result (25-class list) into COCO anns.

    ``score_thresholds=None`` preserves every model-retained candidate.
    Otherwise the supplied 25 class thresholds are applied explicitly.

    Args:
        result (list[np.ndarray]): 25-class mmdet output (xyxy+score).
        image_id (int): COCO image id used for image_id alignment with GT.
        next_ann_id (int): next free annotation id; returned unchanged when
            no boxes survive.

    Returns:
        tuple[list[dict], int]: COCO-style annotations and the next free
        annotation id.
    """
    filtered = (result if score_thresholds is None
                else filter_mmdet_results(result, score_thresholds))
    anns = []
    ann_id = next_ann_id
    for cls_idx, boxes in enumerate(filtered):
        if len(boxes) == 0:
            continue
        for row in boxes:
            x1, y1, x2, y2, s = row.tolist()
            anns.append({
                'id': ann_id,
                'image_id': int(image_id),
                'category_id': int(cls_idx),
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'score': float(s),
                'area': float((x2 - x1) * (y2 - y1)),
            })
            ann_id += 1
    return anns, ann_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--gt', required=True,
                        help='COCO gt json (for image id alignment).')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--thresholds', default=None,
        help='Optional frozen threshold artifact; omit for dense export.')
    args = parser.parse_args()

    score_thresholds = (
        load_threshold_artifact(args.thresholds, args.checkpoint)
        if args.thresholds else None)
    cfg = Config.fromfile(args.config)
    cfg.model.test_cfg.rcnn.score_thr = CANDIDATE_SCORE_FLOOR
    apply_model_ablation_config(cfg)
    model = init_detector(cfg, args.checkpoint, device=args.device)

    with open(args.gt, 'r', encoding='utf-8') as f:
        gt = json.load(f)
    # file_name -> image_id (1:1 from gt so eval_recall_fdr can join)
    name_to_id = {im['file_name']: im['id'] for im in gt['images']}
    if len(name_to_id) != len(gt['images']):
        raise ValueError('validation GT contains duplicate file_name values')

    img_dir = Path(args.img_dir)
    discovered = {
        p.name: p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
    }
    missing = sorted(set(name_to_id) - set(discovered))
    unexpected = sorted(set(discovered) - set(name_to_id))
    if missing or unexpected:
        raise ValueError(
            'GT/image directory mismatch: missing={}, unexpected={}'.format(
                missing, unexpected))
    img_files = [discovered[image['file_name']] for image in gt['images']]
    print(f'Found {len(img_files)} val images.')

    ann = []
    ann_id = 1
    n_empty = 0
    t0 = time.perf_counter()
    for i, img_path in enumerate(img_files, start=1):
        result = inference_detector(model, str(img_path))
        # result is list[np.ndarray (N,5)] indexed by class.
        if not name_to_id.__contains__(img_path.name):
            raise KeyError(
                f'Image file {img_path.name} not found in GT file_name set; '
                'check that --gt points at the same split as --img-dir.')
        added, ann_id = detections_to_coco_annotations(
            result, name_to_id[img_path.name], ann_id,
            score_thresholds=score_thresholds)
        ann.extend(added)
        if not added:
            n_empty += 1
        if i % 50 == 0 or i == len(img_files):
            elapsed = time.perf_counter() - t0
            print(f'  {i}/{len(img_files)}  boxes={len(ann)}  '
                  f'elapsed={elapsed:.1f}s  empty={n_empty}')

    out = {
        'images': gt['images'],
        'annotations': ann,
        'categories': [{'id': i, 'name': n} for i, n in enumerate(CLASS_NAMES)],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)
    mode = 'dense candidate floor' if score_thresholds is None \
        else f'frozen thresholds from {args.thresholds}'
    print(f'Wrote {out_path}: {len(ann)} boxes over {len(img_files)} images '
          f'({n_empty} empty images; mode={mode})')


if __name__ == '__main__':
    main()
