"""Sliding-window inference for very large images (≤ 10000 × 10000 px).

Designed for the competition test pipeline: a single image, single 3090, ≤ 20 s
budget (data reading excluded). The script tiles the input with `tile` × `tile`
patches, `overlap` overlap, runs CascadeRCNN_BAF on each patch, projects boxes
back to full-image coordinates, and merges cross-patch duplicates via
class-aware NMS.

Usage:
    python tools/infer_big_image.py \
        --config configs/bafnet/aircraft_bafnet_1x.py \
        --checkpoint work_dirs/aircraft/latest.pth \
        --img path/to/big.jpg \
        --out big.pred.json \
        --tile 800 --overlap 0.25 --iou 0.5 --score 0.05

Output JSON is COCO-style (one entry per image):
    {"images": [{...}], "annotations": [{...}], "categories": [...]}
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from mmcv import Config
from mmcv.ops import nms
from mmdet.apis import init_detector


CLASS_NAMES = [
    'HM', 'LQS', 'QHS', 'MS',
    'A1_SU-35', 'A2_C-130', 'A3_C-17', 'A4_C-5', 'A5_F-16', 'A6_TU-160',
    'A7_E-3', 'A8_B-52', 'A9_P-3C', 'A10_B-1B', 'A11_E-8', 'A12_TU-22',
    'A13_F-15', 'A14_KC-135', 'A15_F-22', 'A16_FA-18', 'A17_TU-95',
    'A18_KC-10', 'A19_SU-34', 'A20_SU-24', 'FSC',
]


def read_image_fast(path):
    """cv2.imdecode from bytes — fastest decoder on Windows for the 20s budget."""
    with open(path, 'rb') as f:
        buf = f.read()
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def tile_coords(size, tile, stride):
    """Yield (start, end) slices that cover the image, padding at the right/bottom."""
    coords = []
    for start in range(0, max(size - tile, 1) + 1, stride):
        end = min(start + tile, size)
        coords.append((start, end))
        if end == size:
            break
    if not coords or coords[-1][1] < size:
        coords.append((max(size - tile, 0), size))
    return coords


def run_patch(model, patch, device, img_scale):
    """Run a single patch through mmdet; resize respecting img_scale aspect.

    `img_scale` is the (W, H) used by the model pipeline (matches the config
    test pipeline). The patch is resized to fit within img_scale keeping
    aspect ratio; coord projection happens via patch meta in caller.
    """
    from mmdet.apis import inference_detector
    # inference_detector handles resize through the model's data pipeline.
    return inference_detector(model, patch)


def project_boxes(boxes, dx, dy):
    """Translate box coordinates by (dx, dy) — patch origin in full image."""
    if len(boxes) == 0:
        return boxes
    out = boxes.copy()
    out[:, [0, 2]] += dx
    out[:, [1, 3]] += dy
    return out


def class_aware_nms(all_boxes, all_scores, all_cls, iou_thr, per_class=True):
    """NMS — optionally per-class (recommended for duplicate-merge)."""
    if len(all_boxes) == 0:
        return np.zeros(0, dtype=int)
    if not per_class:
        keep = nms(torch.from_numpy(np.concatenate([all_boxes, all_scores[:, None]], 1).astype(np.float32)),
                   iou_thr)
        return keep.numpy().astype(int)
    keep_all = []
    for c in np.unique(all_cls):
        mask = all_cls == c
        b = all_boxes[mask]
        s = all_scores[mask]
        if len(b) == 0:
            continue
        stacked = torch.from_numpy(np.concatenate([b, s[:, None]], 1).astype(np.float32))
        kept = nms(stacked, iou_thr).numpy().astype(int)
        # Map back to indices in all_boxes.
        idx_local = np.where(mask)[0][kept]
        keep_all.append(idx_local)
    if keep_all:
        return np.concatenate(keep_all)
    return np.zeros(0, dtype=int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--img', required=True)
    parser.add_argument('--out', required=True,
                        help='Output JSON path. If ends with .pkl use pickle.')
    parser.add_argument('--tile', type=int, default=800)
    parser.add_argument('--overlap', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--score', type=float, default=0.05)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-class-aware-nms', action='store_true',
                        help='Disable per-class NMS merge.')
    parser.add_argument('--max-det', type=int, default=300)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    model = init_detector(cfg, args.checkpoint, device=args.device)
    # Upper bound on test time: enforce tile ≤ img_scale but auto-shrink if huge.
    test_scale = cfg.data.test.pipeline[1].img_scale
    tile = min(args.tile, test_scale[0])

    img = read_image_fast(args.img)
    if img is None:
        raise SystemExit(f'Failed to read {args.img}')
    H, W = img.shape[:2]
    print(f'image: {W} x {H}  tile={tile}  overlap={args.overlap}')

    stride = int(tile * (1 - args.overlap))
    stride = max(stride, 1)
    xs = tile_coords(W, tile, stride)
    ys = tile_coords(H, tile, stride)

    boxes_all, scores_all, cls_all = [], [], []

    read_t0 = time.perf_counter()
    # Inference timing EXCLUDES the read_image_fast call.
    infer_t0 = time.perf_counter()
    patch_count = 0
    for (y0, y1) in ys:
        for (x0, x1) in xs:
            patch = img[y0:y1, x0:x1]
            # Pad to tile if last row/col (rare).
            if patch.shape[0] != tile or patch.shape[1] != tile:
                pad_b = tile - patch.shape[0]
                pad_r = tile - patch.shape[1]
                if pad_b > 0 or pad_r > 0:
                    patch = cv2.copyMakeBorder(
                        patch, 0, pad_b, 0, pad_r,
                        cv2.BORDER_CONSTANT, value=(114, 114, 114))
            results = run_patch(model, patch, args.device, test_scale)
            for cls_idx, bboxes in enumerate(results):
                if len(bboxes) == 0:
                    continue
                keep = bboxes[:, 4] >= args.score
                if not np.any(keep):
                    continue
                b = bboxes[keep, :4]
                s = bboxes[keep, 4]
                bb = project_boxes(b, x0, y0)
                boxes_all.append(bb)
                scores_all.append(s)
                cls_all.append(np.full(len(b), cls_idx, dtype=np.int32))
            patch_count += 1
    infer_t = time.perf_counter() - infer_t0
    read_t = time.perf_counter() - read_t0 - infer_t

    if boxes_all:
        boxes_all = np.concatenate(boxes_all, axis=0).astype(np.float32)
        scores_all = np.concatenate(scores_all, axis=0).astype(np.float32)
        cls_all = np.concatenate(cls_all, axis=0).astype(np.int32)
    else:
        boxes_all = np.zeros((0, 4), np.float32)
        scores_all = np.zeros((0,), np.float32)
        cls_all = np.zeros((0,), np.int32)

    keep = class_aware_nms(
        boxes_all, scores_all, cls_all,
        iou_thr=args.iou,
        per_class=not args.no_class_aware_nms,
    )
    # Sort by score desc (helpful for downstream eval).
    if len(keep):
        order = np.argsort(-scores_all[keep])
        keep = keep[order]
        # Cap per image.
        keep = keep[:args.max_det]
    boxes_all = boxes_all[keep]
    scores_all = scores_all[keep]
    cls_all = cls_all[keep]

    print(f'patches={patch_count}  read={read_t:.3f}s  infer={infer_t:.3f}s  '
          f'kept_boxes={len(keep)}')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ann = []
    for i, (b, s, c) in enumerate(zip(boxes_all, scores_all, cls_all), start=1):
        x1, y1, x2, y2 = b.tolist()
        ann.append({
            'id': i,
            'image_id': 1,
            'category_id': int(c),
            'bbox': [x1, y1, x2 - x1, y2 - y1],
            'score': float(s),
            'area': float((x2 - x1) * (y2 - y1)),
        })
    res = {
        'images': [{
            'id': 1,
            'file_name': os.path.basename(args.img),
            'width': W,
            'height': H,
        }],
        'annotations': ann,
        'categories': [{'id': i, 'name': n} for i, n in enumerate(CLASS_NAMES)],
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False)
    print(f'wrote {out_path}  ({len(ann)} boxes)')


if __name__ == '__main__':
    main()
