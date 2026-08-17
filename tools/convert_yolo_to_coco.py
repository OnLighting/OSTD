"""Convert YOLO-format aircraft labels to COCO instances_*.json files.

Mirrors the structure mmdet 2.x CocoDataset expects:
    {
      "images": [{"id", "file_name", "width", "height"}, ...],
      "annotations": [{"id", "image_id", "category_id", "bbox" (xywh),
                       "area", "iscrowd", "segmentation" []}, ...],
      "categories": [{"id": 0..nc-1, "name": str}, ...]
    }

Each category id matches the class id in data/dataset.yaml (0..24). Skips
images with empty or missing label files.

Usage:
    python tools/convert_yolo_to_coco.py \
        --root data --split train --out data/annotations/instances_train.json
"""

import argparse
import json
import os
from pathlib import Path


CLASS_NAMES = [
    'HM', 'LQS', 'QHS', 'MS',
    'A1_SU-35', 'A2_C-130', 'A3_C-17', 'A4_C-5', 'A5_F-16', 'A6_TU-160',
    'A7_E-3', 'A8_B-52', 'A9_P-3C', 'A10_B-1B', 'A11_E-8', 'A12_TU-22',
    'A13_F-15', 'A14_KC-135', 'A15_F-22', 'A16_FA-18', 'A17_TU-95',
    'A18_KC-10', 'A19_SU-34', 'A20_SU-24', 'FSC',
]
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def _resolve_size(img_path: Path):
    """Read width/height without pulling in PIL if possible."""
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            return im.width, im.height
    except Exception:
        # Fallback: scan JPEG/PNG header manually; otherwise set to 0/0 and
        # let mmdet's pipeline raise a clearer error downstream.
        with open(img_path, 'rb') as f:
            head = f.read(64)
        if head[:8] == b'\x89PNG\r\n\x1a\n':
            w = int.from_bytes(head[16:20], 'big')
            h = int.from_bytes(head[20:24], 'big')
            return w, h
        # JPEG SOF marker
        try:
            i = 2
            while i < len(head) - 1:
                if head[i] != 0xFF:
                    break
                marker = head[i + 1]
                seg = head[i + 2] * 256 + head[i + 3]
                if marker in (0xC0, 0xC1, 0xC2):
                    h = head[i + 5] * 256 + head[i + 6]
                    w = head[i + 7] * 256 + head[i + 8]
                    return w, h
                i += 2 + seg
        except Exception:
            pass
        return 0, 0


def convert(root: Path, split: str):
    img_dir = root / 'images' / split
    lbl_dir = root / 'labels' / split

    images = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )

    coco_images = []
    coco_annotations = []
    ann_id = 1
    skipped_empty = 0
    for img_id, img_path in enumerate(images, start=1):
        stem = img_path.stem
        lbl_path = lbl_dir / f'{stem}.txt'
        w, h = _resolve_size(img_path)
        coco_images.append({
            'id': img_id,
            'file_name': img_path.name,
            'width': w,
            'height': h,
        })
        if not lbl_path.exists():
            skipped_empty += 1
            continue
        lines = [ln for ln in lbl_path.read_text(encoding='utf-8').splitlines() if ln.strip()]
        if not lines:
            skipped_empty += 1
            continue
        if w <= 0 or h <= 0:
            # Re-read once with PIL if header parsing failed.
            w, h = _resolve_size(img_path)
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, bw, bh = (float(parts[i]) for i in (1, 2, 3, 4))
            # YOLO normalised xywh -> COCO absolute xywh
            x = (xc - bw / 2.0) * w
            y = (yc - bh / 2.0) * h
            bw_px = bw * w
            bh_px = bh * h
            coco_annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': cls,
                'bbox': [x, y, bw_px, bh_px],
                'area': bw_px * bh_px,
                'iscrowd': 0,
                'segmentation': [],
            })
            ann_id += 1

    categories = [{'id': i, 'name': n} for i, n in enumerate(CLASS_NAMES)]
    return {
        'images': coco_images,
        'annotations': coco_annotations,
        'categories': categories,
    }, skipped_empty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data', type=str)
    parser.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    parser.add_argument('--out', required=True, type=str)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    coco, skipped = convert(root, args.split)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    print(f'Wrote {out_path}')
    print(f'  images={len(coco["images"])}  annotations={len(coco["annotations"])}  '
          f'skipped_empty={skipped}')


if __name__ == '__main__':
    main()
