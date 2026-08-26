"""Sliding-window inference for very large images (≤ 10000 × 10000 px).

Designed for the competition test pipeline: a single image (or a batch
of mosaic canvases), single 3090, ≤ 20 s budget per image (data
reading excluded). The script tiles the input with ``tile`` × ``tile``
patches, ``overlap`` overlap, runs CascadeRCNN_BAF on each patch,
projects boxes back to full-image coordinates, merges cross-patch
duplicates via class-aware NMS, and applies the fixed per-class score
thresholds from :mod:`mmdet.core.evaluation.official_metrics`.

Two modes are supported:

* Single image (legacy ``--img``/``--out``) — writes one COCO-style
  JSON per image and prints per-image timing.
* Directory mode (``--img-dir``/``--gt``/``--out``/``--timing-out``) —
  reads mosaic GT to map ``file_name`` to image id, processes every
  mosaic image, combines predictions into a single COCO JSON, and
  writes a timing JSON containing ``per_image_seconds``,
  ``mean_inference_seconds``, and ``max_inference_seconds``.

Per-image timing strictly excludes disk reads and result writes: it
spans CUDA synchronization, sliding-window inference, box projection,
class-aware NMS, fixed-threshold filtering, and final annotation
construction. The script uses :func:`summarize_timings` to report the
maximum inference time, which is what the official pipeline uses for
timing-comparability checks.

Usage:
    # Single image
    python tools/infer_big_image.py \
        --config configs/bafnet/aircraft_bafnet_1x.py \
        --checkpoint work_dirs/aircraft/latest.pth \
        --img path/to/big.jpg \
        --out big.pred.json \
        --tile 800 --overlap 0.25 --iou 0.5

    # Batch (mosaic directory)
    python tools/infer_big_image.py \
        --config configs/bafnet/aircraft_bafnet_1x.py \
        --checkpoint work_dirs/aircraft/best_official_recall_fdr.pth \
        --img-dir work_dirs/big_val/images \
        --gt work_dirs/big_val/instances_big_val.json \
        --out work_dirs/big_val/predictions.json \
        --timing-out work_dirs/big_val/timing.json \
        --tile 800 --overlap 0.25 --iou 0.5
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

from mmdet.core.evaluation import (CLASS_NAMES, CLASS_SCORE_THRESHOLDS,
                                   filter_mmdet_results)


# Minimum model-side score threshold. The per-class fixed thresholds
# vary down to ~0.002 (LQS); the model stage must keep that low to
# avoid losing LQS candidates before the per-class filter applies.
_MIN_SCORE_FLOOR = 0.0


def _synchronize_cuda(device):
    """Synchronize only the CUDA device used for this inference run."""
    selected = torch.device(device)
    if selected.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(device=selected)


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


def _run_patch(model, patch):
    """Run a single patch through mmdet; coord projection happens in caller."""
    from mmdet.apis import inference_detector
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
        keep = nms(torch.from_numpy(np.concatenate(
            [all_boxes, all_scores[:, None]], 1).astype(np.float32)),
            iou_thr)
        return keep.numpy().astype(int)
    keep_all = []
    for c in np.unique(all_cls):
        mask = all_cls == c
        b = all_boxes[mask]
        s = all_scores[mask]
        if len(b) == 0:
            continue
        stacked = torch.from_numpy(np.concatenate(
            [b, s[:, None]], 1).astype(np.float32))
        kept = nms(stacked, iou_thr).numpy().astype(int)
        idx_local = np.where(mask)[0][kept]
        keep_all.append(idx_local)
    if keep_all:
        return np.concatenate(keep_all)
    return np.zeros(0, dtype=int)


def apply_class_thresholds(boxes, scores, classes, nms_keep):
    """Apply per-class fixed score thresholds from official_metrics.

    Args:
        boxes (np.ndarray): (N, 4) xyxy boxes after NMS.
        scores (np.ndarray): (N,) scores after NMS.
        classes (np.ndarray): (N,) class ids after NMS.
        nms_keep (np.ndarray): indices into the original arrays.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: filtered boxes,
        scores, classes (xyxy, score, class).
    """
    if len(nms_keep) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    keep_boxes = boxes[nms_keep]
    keep_scores = scores[nms_keep]
    keep_classes = classes[nms_keep]
    mask = np.array(
        [keep_scores[i] >= CLASS_SCORE_THRESHOLDS[int(keep_classes[i])]
         for i in range(len(keep_scores))], dtype=bool)
    return (
        keep_boxes[mask].astype(np.float32, copy=False),
        keep_scores[mask].astype(np.float32, copy=False),
        keep_classes[mask].astype(np.int32, copy=False),
    )


def infer_big_image(model, image, tile=800, overlap=0.25, iou_thr=0.5,
                    max_det=3000, no_class_aware_nms=False, device='cuda:0'):
    """Run one image through the sliding-window pipeline and time it.

    The timer starts after the image is already in memory and stops
    after the final annotation list has been constructed.

    Args:
        model: an initialized mmdet detector.
        image (np.ndarray): HxWxC image (already decoded).
        tile (int): patch side length.
        overlap (float): fractional overlap between adjacent patches.
        iou_thr (float): NMS IoU threshold.
        max_det (int): cap on kept boxes after sorting.
        no_class_aware_nms (bool): disable per-class NMS.
        device (str): device string, used only for ``torch.cuda.synchronize``.

    Returns:
        tuple[list[dict], float, int]: COCO-style annotations (xywh),
        elapsed inference seconds, and number of patches processed.
    """
    H, W = image.shape[:2]
    stride = max(int(tile * (1 - overlap)), 1)
    xs = tile_coords(W, tile, stride)
    ys = tile_coords(H, tile, stride)

    boxes_all, scores_all, cls_all = [], [], []

    _synchronize_cuda(device)
    infer_t0 = time.perf_counter()
    patch_count = 0
    for (y0, y1) in ys:
        for (x0, x1) in xs:
            patch = image[y0:y1, x0:x1]
            # Pad to tile if last row/col (rare).
            if patch.shape[0] != tile or patch.shape[1] != tile:
                pad_b = tile - patch.shape[0]
                pad_r = tile - patch.shape[1]
                if pad_b > 0 or pad_r > 0:
                    patch = cv2.copyMakeBorder(
                        patch, 0, pad_b, 0, pad_r,
                        cv2.BORDER_CONSTANT, value=(114, 114, 114))
            results = _run_patch(model, patch)
            for cls_idx, bboxes in enumerate(results):
                if len(bboxes) == 0:
                    continue
                b = bboxes[:, :4]
                s = bboxes[:, 4]
                bb = project_boxes(b, x0, y0)
                boxes_all.append(bb)
                scores_all.append(s)
                cls_all.append(np.full(len(b), cls_idx, dtype=np.int32))
            patch_count += 1

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
        iou_thr=iou_thr, per_class=not no_class_aware_nms)

    f_boxes, f_scores, f_classes = apply_class_thresholds(
        boxes_all, scores_all, cls_all, keep)

    if len(f_boxes):
        order = np.argsort(-f_scores)
        f_boxes = f_boxes[order][:max_det]
        f_scores = f_scores[order][:max_det]
        f_classes = f_classes[order][:max_det]

    annotations = []
    for i, (b, s, c) in enumerate(zip(f_boxes, f_scores, f_classes), start=1):
        x1, y1, x2, y2 = b.tolist()
        annotations.append({
            'id': i,
            'category_id': int(c),
            'bbox': [x1, y1, x2 - x1, y2 - y1],
            'score': float(s),
            'area': float((x2 - x1) * (y2 - y1)),
        })
    _synchronize_cuda(device)
    elapsed = time.perf_counter() - infer_t0
    return annotations, elapsed, patch_count


def summarize_timings(per_image_seconds):
    """Build the timing JSON payload from a {file_name: seconds} map."""
    if not per_image_seconds:
        return {
            'per_image_seconds': {},
            'mean_inference_seconds': 0.0,
            'max_inference_seconds': 0.0,
        }
    values = list(per_image_seconds.values())
    return {
        'per_image_seconds': dict(per_image_seconds),
        'mean_inference_seconds': float(sum(values) / len(values)),
        'max_inference_seconds': float(max(values)),
    }


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--img', default=None,
                        help='Single image path. Use --img-dir for batch mode.')
    parser.add_argument('--out', default=None,
                        help='Single-image prediction JSON path.')
    parser.add_argument('--img-dir', default=None,
                        help='Directory mode: directory containing input images.')
    parser.add_argument('--gt', default=None,
                        help='Directory mode: COCO GT for file_name → image_id mapping.')
    parser.add_argument('--timing-out', default=None,
                        help='Directory mode: path for timing JSON.')
    parser.add_argument('--tile', type=int, default=800)
    parser.add_argument('--overlap', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-class-aware-nms', action='store_true',
                        help='Disable per-class NMS merge.')
    parser.add_argument('--max-det', type=int, default=3000)
    return parser


def _init_model(args):
    cfg = Config.fromfile(args.config)
    # Lower the model-side threshold to the minimum of all class thresholds
    # so low-threshold classes survive the model stage and reach the
    # per-class filter.
    min_thr = max(_MIN_SCORE_FLOOR, float(min(CLASS_SCORE_THRESHOLDS)) - 1e-6)
    cfg.model.test_cfg.rcnn.score_thr = min_thr
    return init_detector(cfg, args.checkpoint, device=args.device), cfg


def _run_single(args, model, cfg):
    img = read_image_fast(args.img)
    if img is None:
        raise SystemExit(f'Failed to read {args.img}')
    H, W = img.shape[:2]
    test_scale = cfg.data.test.pipeline[1].img_scale
    tile = min(args.tile, test_scale[0])

    # Read timing boundary: timer starts AFTER read_image_fast.
    annotations, elapsed, patch_count = infer_big_image(
        model, img,
        tile=tile, overlap=args.overlap, iou_thr=args.iou,
        max_det=args.max_det,
        no_class_aware_nms=args.no_class_aware_nms,
        device=args.device)
    print(f'image: {W} x {H}  tile={tile}  overlap={args.overlap}  '
          f'patches={patch_count}  infer={elapsed:.3f}s  '
          f'kept_boxes={len(annotations)}')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res = {
        'images': [{
            'id': 1,
            'file_name': os.path.basename(args.img),
            'width': W,
            'height': H,
        }],
        'annotations': annotations,
        'categories': [{'id': i, 'name': n}
                       for i, n in enumerate(CLASS_NAMES)],
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False)
    print(f'wrote {out_path}  ({len(annotations)} boxes)')


def _run_batch(args, model, cfg):
    gt_path = Path(args.gt)
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)
    name_to_id = {im['file_name']: im['id'] for im in gt['images']}
    if len(name_to_id) != len(gt['images']):
        raise ValueError('mosaic GT contains duplicate file_name values')

    img_dir = Path(args.img_dir)
    discovered = {
        p.name: p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in
        {'.jpg', '.jpeg', '.png', '.bmp'}
    }
    missing = sorted(set(name_to_id) - set(discovered))
    unexpected = sorted(set(discovered) - set(name_to_id))
    if missing or unexpected:
        raise ValueError(
            'GT/image directory mismatch: missing={}, unexpected={}'.format(
                missing, unexpected))
    img_files = [discovered[im['file_name']] for im in gt['images']]
    print(f'Found {len(img_files)} mosaic images.')

    test_scale = cfg.data.test.pipeline[1].img_scale
    tile = min(args.tile, test_scale[0])

    per_image_seconds = {}
    ann_all = []
    for idx, img_path in enumerate(img_files, start=1):
        img = read_image_fast(str(img_path))
        if img is None:
            print(f'  WARN: failed to read {img_path}, skipping',
                  file=__import__('sys').stderr)
            continue
        H, W = img.shape[:2]
        annotations, elapsed, patch_count = infer_big_image(
            model, img,
            tile=tile, overlap=args.overlap, iou_thr=args.iou,
            max_det=args.max_det,
            no_class_aware_nms=args.no_class_aware_nms,
            device=args.device)
        per_image_seconds[img_path.name] = float(elapsed)
        # Align image id from the mosaic GT (file_name → image_id).
        image_id = name_to_id.get(img_path.name)
        if image_id is None:
            raise KeyError(
                f'mosaic {img_path.name} not in GT file_name set; '
                'check that --gt points at the matching big_val GT')
        next_annotation_id = len(ann_all) + 1
        for offset, ann in enumerate(annotations):
            ann['id'] = next_annotation_id + offset
            ann['image_id'] = int(image_id)
        ann_all.extend(annotations)
        print(f'  {idx}/{len(img_files)}  {img_path.name}  '
              f'{W}x{H}  patches={patch_count}  infer={elapsed:.3f}s  '
              f'kept={len(annotations)}')

    timing_payload = summarize_timings(per_image_seconds)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res = {
        'images': gt['images'],
        'annotations': ann_all,
        'categories': [{'id': i, 'name': n}
                       for i, n in enumerate(CLASS_NAMES)],
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False)
    if args.timing_out:
        t_path = Path(args.timing_out)
        t_path.parent.mkdir(parents=True, exist_ok=True)
        with open(t_path, 'w', encoding='utf-8') as f:
            json.dump(timing_payload, f, ensure_ascii=False, indent=2)
    print(f'wrote {out_path}  ({len(ann_all)} boxes)')
    if args.timing_out:
        print(f'wrote {args.timing_out}  max_inference_seconds='
              f'{timing_payload["max_inference_seconds"]:.3f}')


def main():
    parser = _build_parser()
    args = parser.parse_args()
    model, cfg = _init_model(args)
    if args.img_dir is not None:
        if args.out is None or args.gt is None:
            raise SystemExit('--img-dir requires --out and --gt')
        _run_batch(args, model, cfg)
    else:
        if args.img is None or args.out is None:
            raise SystemExit('--img and --out are required in single-image mode')
        _run_single(args, model, cfg)


if __name__ == '__main__':
    main()
