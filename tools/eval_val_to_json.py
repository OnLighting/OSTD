"""Batch-run inference on the val split and emit a single COCO-style JSON.

Output is consumed by tools/eval_recall_fdr.py. Each detected box keeps its
score so the Recall/FDR protocol's score-desc one-to-one matching works.

This avoids the pkl-format gap in tools/test.py — for the competition metric
(score-ranked, class-conditional IoU) we need per-box scores, which pkl
doesn't carry cleanly.

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

from sbla_config import apply_model_ablation_config


CLASS_NAMES = [
    'HM', 'LQS', 'QHS', 'MS',
    'A1_SU-35', 'A2_C-130', 'A3_C-17', 'A4_C-5', 'A5_F-16', 'A6_TU-160',
    'A7_E-3', 'A8_B-52', 'A9_P-3C', 'A10_B-1B', 'A11_E-8', 'A12_TU-22',
    'A13_F-15', 'A14_KC-135', 'A15_F-22', 'A16_FA-18', 'A17_TU-95',
    'A18_KC-10', 'A19_SU-34', 'A20_SU-24', 'FSC',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--gt', required=True,
                        help='COCO gt json (for image id alignment).')
    parser.add_argument('--out', required=True)
    parser.add_argument('--score', type=float, default=0.05)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.model.test_cfg.rcnn.score_thr = args.score
    apply_model_ablation_config(cfg)
    model = init_detector(cfg, args.checkpoint, device=args.device)

    with open(args.gt, 'r', encoding='utf-8') as f:
        gt = json.load(f)
    # file_name -> image_id (1:1 from gt so eval_recall_fdr can join)
    name_to_id = {im['file_name']: im['id'] for im in gt['images']}

    img_dir = Path(args.img_dir)
    img_files = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
    )
    print(f'Found {len(img_files)} val images.')

    ann = []
    ann_id = 1
    n_empty = 0
    t0 = time.perf_counter()
    for i, img_path in enumerate(img_files, start=1):
        result = inference_detector(model, str(img_path))
        # result is list[np.ndarray (N,5)] indexed by class.
        added = 0
        for cls_idx, bboxes in enumerate(result):
            if len(bboxes) == 0:
                continue
            keep = bboxes[:, 4] >= args.score
            if not np.any(keep):
                continue
            b = bboxes[keep]
            for row in b:
                x1, y1, x2, y2, s = row.tolist()
                ann.append({
                    'id': ann_id,
                    'image_id': name_to_id[img_path.name],
                    'category_id': int(cls_idx),
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'score': float(s),
                    'area': float((x2 - x1) * (y2 - y1)),
                })
                ann_id += 1
                added += 1
        if added == 0:
            n_empty += 1
        if i % 50 == 0 or i == len(img_files):
            elapsed = time.perf_counter() - t0
            print(f'  {i}/{len(img_files)}  boxes={ann_id - 1}  '
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
    print(f'Wrote {out_path}: {len(ann)} boxes over {len(img_files)} images '
          f'({n_empty} images with no prediction above score={args.score})')


if __name__ == '__main__':
    main()
