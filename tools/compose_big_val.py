"""Compose 10000×10000 mosaics from validation images.

The competition tests inference on simulated 10000×10000 inputs that are
built by tiling real validation images without resizing them. This script
selects a deterministic subset, lays them out on fixed-size canvases,
remaps GT annotations onto the canvas coordinates, and emits the
mosaic images plus a COCO-style GT file and a source map.

The packing strategy is row-based ("shelf"): walk canvases in canvas
order; within each canvas fill horizontal shelves top-to-bottom; on each
shelf place images left-to-right until no more fit, then start the next
shelf. If an image is taller than the remaining vertical space we move
to the next canvas. Images are never cropped or rescaled.

Usage:
    python tools/compose_big_val.py \
        --gt data/annotations/instances_val.json \
        --img-dir data/images/val \
        --out-dir work_dirs/big_val \
        --num-canvases 2 \
        --seed 0
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


CANVAS_SIZE = 10000
DEFAULT_BACKGROUND = (114, 114, 114)
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def shift_bbox(bbox, dx, dy):
    """Translate an xywh bbox by (dx, dy)."""
    return [bbox[0] + dx, bbox[1] + dy, bbox[2], bbox[3]]


def pack_images(sizes, canvas_size=CANVAS_SIZE, seed=0):
    """Pack ``sizes`` (list of (w, h)) onto ``canvas_size`` square canvases.

    Each source image is placed whole (no cropping, no resizing). Returns
    a list of canvases; each canvas is a list of (source_index, x, y)
    placements that all fit inside the canvas.
    """
    if not sizes:
        return []
    for idx, (w, h) in enumerate(sizes):
        if w > canvas_size or h > canvas_size:
            raise ValueError(
                f'source {idx} size {(w, h)} exceeds canvas {canvas_size}')

    rng = random.Random(seed)
    order = list(range(len(sizes)))
    rng.shuffle(order)

    canvases = []
    while order:
        canvas = []
        cursor_x = 0
        cursor_y = 0
        shelf_height = 0
        remaining = []
        for source_idx in order:
            w, h = sizes[source_idx]
            if cursor_y + h > canvas_size:
                # No vertical room left on this canvas; defer to next.
                remaining.append(source_idx)
                continue
            if cursor_x + w > canvas_size:
                # Wrap to a new shelf.
                cursor_x = 0
                cursor_y += shelf_height
                shelf_height = 0
                if cursor_y + h > canvas_size:
                    remaining.append(source_idx)
                    continue
            canvas.append((source_idx, cursor_x, cursor_y))
            cursor_x += w
            shelf_height = max(shelf_height, h)
        if not canvas:
            # Nothing fit at all — bail out rather than loop forever.
            raise ValueError(
                'failed to place any image on the current canvas; check sizes')
        canvases.append(canvas)
        order = remaining
    return canvases


def _gather_sources(gt_images, img_dir):
    """Resolve GT image entries to on-disk paths and record sizes.

    Raises:
        FileNotFoundError: when an image file is missing.
        ValueError: when duplicate file names exist or an entry is empty.
    """
    by_name = {}
    for entry in gt_images:
        fn = entry['file_name']
        path = Path(img_dir) / Path(fn).name
        if not path.is_file():
            raise FileNotFoundError(f'missing source image {path}')
        if fn in by_name:
            raise ValueError(f'duplicate GT file_name: {fn}')
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f'failed to decode source image {path}')
        h, w = img.shape[:2]
        by_name[fn] = {
            'path': path,
            'width': int(w),
            'height': int(h),
            'image_id': int(entry['id']),
        }
    return by_name


def compose_canvases(sources, sizes, placements, canvas_size=CANVAS_SIZE,
                     background=DEFAULT_BACKGROUND):
    """Render mosaics and return (canvas_arrays, per_canvas_placements).

    ``placements`` is a list of canvases from :func:`pack_images`;
    ``sources`` is the lookup dict from :func:`_gather_sources`.
    """
    rendered = []
    per_canvas = []
    for canvas_placements in placements:
        canvas = np.full(
            (canvas_size, canvas_size, 3), background, dtype=np.uint8)
        per_entry = []
        for source_idx, x, y in canvas_placements:
            w, h = sizes[source_idx]
            info = sources[source_idx]
            tile = cv2.imread(str(info['path']), cv2.IMREAD_COLOR)
            canvas[y:y + h, x:x + w] = tile
            per_entry.append({
                'source_index': source_idx,
                'source_file_name': info['file_name'],
                'source_image_id': info['image_id'],
                'x': int(x),
                'y': int(y),
            })
        rendered.append(canvas)
        per_canvas.append(per_entry)
    return rendered, per_canvas


def remap_annotations(gt, placements, per_canvas, canvas_size=CANVAS_SIZE):
    """Translate GT boxes to canvas coordinates.

    Returns (new_images, new_annotations). Image ids are regenerated.
    Categories are preserved from the input GT.
    """
    # Map source_image_id → (canvas_index, x_offset, y_offset).
    source_to_offset = {}
    for canvas_idx, entries in enumerate(per_canvas):
        for entry in entries:
            source_to_offset[entry['source_image_id']] = (
                canvas_idx, entry['x'], entry['y'])

    new_images = []
    new_anns = []
    image_id_map = {}
    next_image_id = 1
    next_ann_id = 1
    for canvas_idx, entries in enumerate(per_canvas):
        image_id_map[canvas_idx] = next_image_id
        new_images.append({
            'id': next_image_id,
            'file_name': f'mosaic_{canvas_idx:04d}.jpg',
            'width': canvas_size,
            'height': canvas_size,
        })
        next_image_id += 1
    for ann in gt.get('annotations', []):
        offset = source_to_offset.get(int(ann['image_id']))
        if offset is None:
            raise ValueError(
                f'annotation image_id {ann["image_id"]} not present in mosaics')
        canvas_idx, dx, dy = offset
        new_bbox = shift_bbox(list(ann['bbox']), dx, dy)
        x, y, bw, bh = new_bbox
        if (x < 0 or y < 0
                or x + bw > canvas_size
                or y + bh > canvas_size):
            raise ValueError(
                f'translated bbox {new_bbox} for canvas {canvas_idx} '
                'falls outside the canvas; refusing to write mosaic')
        new_anns.append({
            'id': next_ann_id,
            'image_id': image_id_map[canvas_idx],
            'category_id': int(ann['category_id']),
            'bbox': new_bbox,
            'area': float(bw * bh),
            'iscrowd': int(ann.get('iscrowd', 0)),
        })
        next_ann_id += 1
    return new_images, new_anns


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True,
                        help='Path to COCO-style GT json (instances_val.json).')
    parser.add_argument('--img-dir', required=True,
                        help='Directory containing the source images.')
    parser.add_argument('--out-dir', required=True,
                        help='Where to write mosaics/, instances_big_val.json, '
                        'and source_map.json.')
    parser.add_argument('--num-canvases', type=int, default=2,
                        help='Maximum number of canvases to produce.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--canvas-size', type=int, default=CANVAS_SIZE)
    parser.add_argument('--overwrite', action='store_true',
                        help='Wipe existing outputs before writing.')
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    images_dir = out_dir / 'images'
    gt_out = out_dir / 'instances_big_val.json'
    map_out = out_dir / 'source_map.json'

    if out_dir.exists():
        if not args.overwrite and (
                images_dir.exists() or gt_out.exists() or map_out.exists()):
            print(
                f'ERROR: output directory {out_dir} already contains mosaic '
                'artifacts; pass --overwrite to recreate.',
                file=sys.stderr)
            return 2
        if args.overwrite:
            if images_dir.exists():
                shutil.rmtree(images_dir)
            for p in (gt_out, map_out):
                if p.exists():
                    p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    with open(args.gt, 'r', encoding='utf-8') as f:
        gt = json.load(f)

    sources = _gather_sources(gt['images'], Path(args.img_dir))
    if not sources:
        print('ERROR: no source images collected from GT', file=sys.stderr)
        return 1

    # Stable order: by source_image_id so the seed-shuffle is reproducible
    # independent of GT file ordering.
    ordered = sorted(sources.values(), key=lambda v: v['image_id'])
    sizes = [(v['width'], v['height']) for v in ordered]
    index_to_source = list(ordered)

    placements = pack_images(sizes, canvas_size=args.canvas_size,
                             seed=args.seed)
    placements = placements[:args.num_canvases]
    if not placements:
        print('ERROR: no placements fit on any canvas', file=sys.stderr)
        return 1

    rendered, per_canvas = compose_canvases(
        index_to_source, sizes, placements,
        canvas_size=args.canvas_size, background=DEFAULT_BACKGROUND)
    for idx, canvas in enumerate(rendered):
        cv2.imwrite(str(images_dir / f'mosaic_{idx:04d}.jpg'), canvas,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    new_images, new_anns = remap_annotations(
        gt, placements, per_canvas, canvas_size=args.canvas_size)

    new_gt = {
        'images': new_images,
        'annotations': new_anns,
        'categories': gt.get('categories', []),
    }
    with open(gt_out, 'w', encoding='utf-8') as f:
        json.dump(new_gt, f, ensure_ascii=False, indent=2)

    source_map = []
    for canvas_idx, entries in enumerate(per_canvas):
        for entry in entries:
            entry = dict(entry)
            entry['mosaic_index'] = canvas_idx
            entry['mosaic_image_id'] = new_images[canvas_idx]['id']
            entry['mosaic_file_name'] = new_images[canvas_idx]['file_name']
            source_map.append(entry)
    with open(map_out, 'w', encoding='utf-8') as f:
        json.dump(source_map, f, ensure_ascii=False, indent=2)

    print(
        f'Wrote {len(rendered)} mosaics ({len(new_anns)} translated GT boxes)'
        f' to {out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
