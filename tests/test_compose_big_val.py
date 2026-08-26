"""Tests for the deterministic 10000×10000 mosaic composer."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

from tools.compose_big_val import (CANVAS_SIZE, DEFAULT_BACKGROUND, pack_images,
                                   shift_bbox)


# (source_index, w, h)
SIZES = [
    (3000, 2000),
    (4000, 2000),
    (2500, 3000),
    (1000, 1000),
]


def test_shift_bbox_preserves_size_and_moves_origin():
    assert shift_bbox([10, 20, 30, 40], 100, 200) == [110, 220, 30, 40]
    assert shift_bbox([0, 0, 5, 5], -3, 4) == [-3, 4, 5, 5]


def test_every_placement_stays_inside_canvas():
    placements = pack_images(SIZES, canvas_size=CANVAS_SIZE)
    assert placements, 'expected at least one canvas'
    for canvas in placements:
        for source_index, x, y in canvas:
            w, h = SIZES[source_index]
            assert 0 <= x and x + w <= CANVAS_SIZE
            assert 0 <= y and y + h <= CANVAS_SIZE


def test_no_split_source_images():
    """Each source image appears whole (one placement per source index)."""
    placements = pack_images(SIZES, canvas_size=CANVAS_SIZE)
    seen_indices = []
    for canvas in placements:
        for source_index, _, _ in canvas:
            seen_indices.append(source_index)
    assert sorted(seen_indices) == sorted(range(len(SIZES)))


def test_deterministic_seed_produces_identical_packing():
    a = pack_images(SIZES, canvas_size=CANVAS_SIZE, seed=42)
    b = pack_images(SIZES, canvas_size=CANVAS_SIZE, seed=42)
    assert a == b


def test_no_oversize_sources_accepted():
    """Sources bigger than the canvas should raise immediately."""
    with pytest.raises(ValueError):
        pack_images([(CANVAS_SIZE + 1, 1)], canvas_size=CANVAS_SIZE)


class ComposeCliTest(unittest.TestCase):

    def _make_fake_dataset(self, root):
        img_dir = root / 'images'
        img_dir.mkdir(parents=True)
        # Three small images of distinct sizes.
        self.size_specs = [(800, 600), (1200, 400), (500, 1000)]
        self.file_names = []
        for i, (w, h) in enumerate(self.size_specs):
            arr = np.full((h, w, 3), 64, dtype=np.uint8)
            import cv2
            cv2.imwrite(str(img_dir / f'src{i:02d}.jpg'), arr)
            self.file_names.append(f'src{i:02d}.jpg')
        # Build a tiny COCO GT.
        self.gt_path = root / 'gt.json'
        images = []
        anns = []
        ann_id = 1
        for i, fn in enumerate(self.file_names):
            w, h = self.size_specs[i]
            images.append({'id': i + 1, 'file_name': fn,
                           'width': w, 'height': h})
            # Two boxes per image, kept inside the canvas.
            for j in range(2):
                x1 = (j + 1) * 10
                y1 = (j + 1) * 10
                x2 = x1 + 50
                y2 = y1 + 50
                anns.append({
                    'id': ann_id,
                    'image_id': i + 1,
                    'category_id': 4 + j,
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'area': (x2 - x1) * (y2 - y1),
                    'iscrowd': 0,
                })
                ann_id += 1
        with open(self.gt_path, 'w', encoding='utf-8') as f:
            json.dump({
                'images': images,
                'annotations': anns,
                'categories': [
                    {'id': 0, 'name': 'A1'},
                    {'id': 1, 'name': 'A2'},
                    {'id': 2, 'name': 'A3'},
                    {'id': 3, 'name': 'A4'},
                    {'id': 4, 'name': 'A5'},
                ],
            }, f)

    def test_cli_creates_image_gt_and_source_map(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        src_dir = tmp / 'src'
        out_dir = tmp / 'mosaics'
        self._make_fake_dataset(src_dir)
        gt_path = src_dir / 'gt.json'

        # Run the CLI.
        from tools import compose_big_val as mod

        old_argv = __import__('sys').argv
        __import__('sys').argv = [
            'compose_big_val.py',
            '--gt', str(gt_path),
            '--img-dir', str(src_dir / 'images'),
            '--out-dir', str(out_dir),
            '--num-canvases', '1',
            '--seed', '7',
        ]
        try:
            rc = mod.main()
        finally:
            __import__('sys').argv = old_argv
        self.assertEqual(rc, 0)

        # Output files exist.
        self.assertTrue((out_dir / 'images').exists())
        self.assertTrue((out_dir / 'instances_big_val.json').exists())
        self.assertTrue((out_dir / 'source_map.json').exists())

        # Canvas image size.
        canvas_files = sorted(p for p in (out_dir / 'images').iterdir()
                              if p.suffix == '.jpg')
        self.assertEqual(len(canvas_files), 1)
        import cv2
        canvas = cv2.imread(str(canvas_files[0]))
        h, w = canvas.shape[:2]
        self.assertEqual(w, CANVAS_SIZE)
        self.assertEqual(h, CANVAS_SIZE)

        # All source images appear in source_map exactly once.
        with open(out_dir / 'source_map.json') as f:
            source_map = json.load(f)
        used = [entry['source_index'] for entry in source_map]
        self.assertEqual(sorted(used), [0, 1, 2])

        # GT json has all 6 original boxes translated.
        with open(out_dir / 'instances_big_val.json') as f:
            new_gt = json.load(f)
        self.assertEqual(len(new_gt['annotations']), 6)
        # Boxes must lie inside the canvas.
        for ann in new_gt['annotations']:
            x, y, bw, bh = ann['bbox']
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + bw, CANVAS_SIZE)
            self.assertLessEqual(y + bh, CANVAS_SIZE)

    def test_overwrite_required_when_outputs_exist(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        src_dir = tmp / 'src'
        out_dir = tmp / 'mosaics'
        self._make_fake_dataset(src_dir)
        out_dir.mkdir(parents=True)
        (out_dir / 'images').mkdir()
        # Drop a fake prior output.
        (out_dir / 'images' / 'old.jpg').write_bytes(b'x')

        from tools import compose_big_val as mod
        old_argv = __import__('sys').argv
        __import__('sys').argv = [
            'compose_big_val.py',
            '--gt', str(src_dir / 'gt.json'),
            '--img-dir', str(src_dir / 'images'),
            '--out-dir', str(out_dir),
            '--num-canvases', '1',
            '--seed', '7',
        ]
        try:
            rc = mod.main()
        finally:
            __import__('sys').argv = old_argv
        self.assertEqual(rc, 2)
        # Old file remains; we did not silently overwrite.
        self.assertTrue((out_dir / 'images' / 'old.jpg').exists())
