"""Tests for the big-image inference helpers and timing summary."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mmdet.core.evaluation import write_threshold_artifact
import tools.infer_big_image as big_image
from tools.infer_big_image import (apply_class_thresholds,
                                   summarize_timings,
                                   tile_coords)


BOXES = np.array([[0.0, 0.0, 10.0, 10.0],
                  [50.0, 50.0, 80.0, 80.0]], dtype=np.float32)
SCORES = np.array([0.49, 0.50], dtype=np.float32)
CLASSES = np.array([0, 1], dtype=np.int32)


def test_timing_summary_reports_maximum():
    summary = summarize_timings({'a.jpg': 1.2, 'b.jpg': 2.5, 'c.jpg': 0.7})
    assert summary['max_inference_seconds'] == 2.5
    assert math.isclose(summary['mean_inference_seconds'],
                        (1.2 + 2.5 + 0.7) / 3, rel_tol=1e-9)
    assert summary['per_image_seconds'] == {'a.jpg': 1.2, 'b.jpg': 2.5,
                                            'c.jpg': 0.7}


def test_timing_summary_empty_returns_zero():
    summary = summarize_timings({})
    assert summary['max_inference_seconds'] == 0.0
    assert summary['mean_inference_seconds'] == 0.0


def test_final_filter_uses_loaded_class_thresholds():
    boxes, scores, classes = apply_class_thresholds(
        BOXES, SCORES, CLASSES,
        nms_keep=np.array([0, 1], dtype=np.int32),
        score_thresholds=[0.50] * 25)
    # class 0 score below threshold → dropped; class 1 survives.
    assert classes.tolist() == [1]
    assert boxes.shape[0] == 1
    np.testing.assert_allclose(boxes[0], [50.0, 50.0, 80.0, 80.0])


def test_tile_coords_covers_full_extent():
    coords = tile_coords(1000, 400, 200)
    starts = [c[0] for c in coords]
    ends = [c[1] for c in coords]
    # Coverage must be contiguous and end at 1000.
    assert coords[0][0] == 0
    assert coords[-1][1] == 1000
    # No gaps.
    for prev_end, start in zip(ends[:-1], starts[1:]):
        assert start <= prev_end


def test_tile_coords_handles_short_axis():
    coords = tile_coords(300, 800, 200)
    # When size < tile, we still emit at least one tile.
    assert len(coords) >= 1
    assert coords[-1][1] == 300


def test_inference_synchronizes_the_requested_cuda_device(monkeypatch):
    """Timing must wait for the GPU selected by ``--device``."""
    synchronized = []
    monkeypatch.setattr(big_image.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(
        big_image.torch.cuda,
        'synchronize',
        lambda device=None: synchronized.append(device))
    monkeypatch.setattr(
        big_image,
        '_run_patch',
        lambda model, patch: [np.zeros((0, 5), dtype=np.float32)
                              for _ in range(25)])

    big_image.infer_big_image(
        model=None,
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        tile=2,
        overlap=0.0,
        score_thresholds=[0.0] * 25,
        device='cuda:1')

    assert synchronized == [torch.device('cuda:1'), torch.device('cuda:1')]


def _batch_args(tmp_path, img_dir, gt_path):
    return SimpleNamespace(
        gt=str(gt_path),
        img_dir=str(img_dir),
        out=str(tmp_path / 'pred.json'),
        timing_out=str(tmp_path / 'timing.json'),
        tile=2,
        overlap=0.0,
        iou=0.5,
        max_det=3000,
        no_class_aware_nms=False,
        device='cpu')


def test_batch_rejects_missing_gt_image(tmp_path):
    img_dir = tmp_path / 'images'
    img_dir.mkdir()
    (img_dir / 'one.jpg').write_bytes(b'placeholder')
    gt_path = tmp_path / 'gt.json'
    gt_path.write_text(json.dumps({
        'images': [
            {'id': 1, 'file_name': 'one.jpg'},
            {'id': 2, 'file_name': 'two.jpg'},
        ],
        'annotations': [],
    }), encoding='utf-8')
    cfg = SimpleNamespace(data=SimpleNamespace(
        test=SimpleNamespace(pipeline=[None, SimpleNamespace(img_scale=(2, 2))])))

    with pytest.raises(ValueError, match='GT/image directory mismatch'):
        big_image._run_batch(
            _batch_args(tmp_path, img_dir, gt_path), model=None, cfg=cfg,
            score_thresholds=[0.0] * 25)


def test_batch_assigns_unique_annotation_ids(tmp_path, monkeypatch):
    img_dir = tmp_path / 'images'
    img_dir.mkdir()
    for name in ('one.jpg', 'two.jpg'):
        (img_dir / name).write_bytes(b'placeholder')
    gt_path = tmp_path / 'gt.json'
    gt_path.write_text(json.dumps({
        'images': [
            {'id': 1, 'file_name': 'one.jpg'},
            {'id': 2, 'file_name': 'two.jpg'},
        ],
        'annotations': [],
    }), encoding='utf-8')
    cfg = SimpleNamespace(data=SimpleNamespace(
        test=SimpleNamespace(pipeline=[None, SimpleNamespace(img_scale=(2, 2))])))
    monkeypatch.setattr(
        big_image, 'read_image_fast',
        lambda path: np.zeros((2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr(
        big_image, 'infer_big_image',
        lambda *args, **kwargs: ([{
            'id': 1,
            'category_id': 0,
            'bbox': [0, 0, 1, 1],
            'score': 0.9,
            'area': 1.0,
        }], 0.1, 1))

    args = _batch_args(tmp_path, img_dir, gt_path)
    big_image._run_batch(
        args, model=None, cfg=cfg, score_thresholds=[0.0] * 25)

    payload = json.loads(Path(args.out).read_text(encoding='utf-8'))
    assert [ann['id'] for ann in payload['annotations']] == [1, 2]
    assert [ann['image_id'] for ann in payload['annotations']] == [1, 2]


def test_runtime_thresholds_reject_different_checkpoint(tmp_path):
    checkpoint_a = tmp_path / 'a.pth'
    checkpoint_b = tmp_path / 'b.pth'
    checkpoint_a.write_bytes(b'a')
    checkpoint_b.write_bytes(b'b')
    artifact = tmp_path / 'thresholds.json'
    write_threshold_artifact(
        artifact, [0.1] * 25, checkpoint_a, 'pred.json', 'gt.json',
        {
            'max_official_fdr': 0.19,
            'target_official_recall': 0.85,
        },
        {'official': {'recall': 0.85, 'fdr': 0.19}})
    args = SimpleNamespace(
        thresholds=str(artifact), checkpoint=str(checkpoint_b))

    with pytest.raises(ValueError, match='SHA-256'):
        big_image._load_runtime_thresholds(args)
