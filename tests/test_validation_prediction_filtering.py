"""Tests for the fixed-class-threshold validation prediction export."""

import numpy as np

from mmdet.core.evaluation import CLASS_SCORE_THRESHOLDS
from tools.eval_val_to_json import detections_to_coco_annotations


def _empty_result():
    return [np.zeros((0, 5), dtype=np.float32) for _ in range(25)]


def test_each_class_uses_its_own_threshold():
    result = _empty_result()
    # Box below class-0 threshold → dropped.
    result[0] = np.array(
        [[0, 0, 2, 2, CLASS_SCORE_THRESHOLDS[0] - 1e-6]])
    # Box at class-1 threshold → kept.
    result[1] = np.array([[0, 0, 2, 2, CLASS_SCORE_THRESHOLDS[1]]])
    anns, next_id = detections_to_coco_annotations(result, image_id=7,
                                                   next_ann_id=1)
    assert [ann['category_id'] for ann in anns] == [1]
    assert next_id == 2
    assert anns[0]['image_id'] == 7


def test_empty_detections_returns_zero_id():
    anns, next_id = detections_to_coco_annotations(
        _empty_result(), image_id=99, next_ann_id=42)
    assert anns == []
    assert next_id == 42


def test_bbox_and_area_converted_from_xyxy_to_xywh():
    result = _empty_result()
    result[5] = np.array([[10, 20, 30, 60, 0.9]])
    anns, _ = detections_to_coco_annotations(result, image_id=1, next_ann_id=1)
    assert len(anns) == 1
    ann = anns[0]
    assert ann['bbox'] == [10, 20, 20, 40]
    assert ann['area'] == 20 * 40
    assert ann['category_id'] == 5
