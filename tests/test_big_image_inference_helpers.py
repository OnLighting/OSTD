"""Tests for the big-image inference helpers and timing summary."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mmdet.core.evaluation import CLASS_SCORE_THRESHOLDS
from tools.infer_big_image import (apply_class_thresholds,
                                   summarize_timings,
                                   tile_coords)


BOXES = np.array([[0.0, 0.0, 10.0, 10.0],
                  [50.0, 50.0, 80.0, 80.0]], dtype=np.float32)
SCORES = np.array([CLASS_SCORE_THRESHOLDS[0] - 1e-6,
                   CLASS_SCORE_THRESHOLDS[1]], dtype=np.float32)
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


def test_final_filter_uses_class_thresholds():
    boxes, scores, classes = apply_class_thresholds(
        BOXES, SCORES, CLASSES,
        nms_keep=np.array([0, 1], dtype=np.int32))
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
