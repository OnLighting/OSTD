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


def test_partial_canvas_drops_unplaced_annotations(tmp_path):
    """When ``num_canvases`` is smaller than what fits all sources, only the
    annotations that landed on a chosen canvas are translated. Inputs that
    didn't fit are silently ignored (not raised as a ValueError)."""
    import json

    from tools import compose_big_val as mod

    src_dir = tmp_path / 'src'
    img_dir = src_dir / 'images'
    img_dir.mkdir(parents=True)

    # Three images: two fit comfortably; the third is too wide to share a
    # canvas with them, so with num_canvases=1 only the first two land.
    import cv2
    sizes = [(5000, 1000), (5000, 1000), (9000, 9000)]
    file_names = []
    for i, (w, h) in enumerate(sizes):
        arr = np.full((h, w, 3), 64, dtype=np.uint8)
        cv2.imwrite(str(img_dir / f'src{i:02d}.jpg'), arr)
        file_names.append(f'src{i:02d}.jpg')
    gt_path = src_dir / 'gt.json'
    images = [{'id': i + 1, 'file_name': fn, 'width': w, 'height': h}
              for i, ((w, h), fn) in enumerate(zip(sizes, file_names))]
    annotations = []
    for i in range(3):
        annotations.append({
            'id': i + 1,
            'image_id': i + 1,
            'category_id': 4,
            'bbox': [10, 10, 50, 50],
            'area': 2500,
            'iscrowd': 0,
        })
    with open(gt_path, 'w', encoding='utf-8') as f:
        json.dump({'images': images, 'annotations': annotations,
                   'categories': [{'id': 4, 'name': 'A1'}]}, f)

    out_dir = tmp_path / 'out'
    old_argv = __import__('sys').argv
    __import__('sys').argv = [
        'compose_big_val.py',
        '--gt', str(gt_path),
        '--img-dir', str(img_dir),
        '--out-dir', str(out_dir),
        '--num-canvases', '1',
        '--seed', '0',
    ]
    try:
        rc = mod.main()
    finally:
        __import__('sys').argv = old_argv
    assert rc == 0
    with open(out_dir / 'instances_big_val.json') as f:
        new_gt = json.load(f)
    # Only the sources that landed on canvas 0 contribute annotations. The
    # 9000x9000 image is too large to coexist on the same canvas as the two
    # 5000x1000 strips, so its annotation must be dropped silently.
    placed_image_ids = {
        entry['source_image_id']
        for entry in json.loads((out_dir / 'source_map.json').read_text())
    }
    assert placed_image_ids <= {1, 2, 3}
    expected_ann_count = len(placed_image_ids)
    assert len(new_gt['annotations']) == expected_ann_count
    for ann in new_gt['annotations']:
        # All translated boxes must lie inside the 10000x10000 canvas.
        x, y, bw, bh = ann['bbox']
        assert 0 <= x and x + bw <= CANVAS_SIZE
        assert 0 <= y and y + bh <= CANVAS_SIZE


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
