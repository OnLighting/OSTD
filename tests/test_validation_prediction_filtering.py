"""Tests for dense and frozen-threshold validation prediction export."""

import numpy as np

from tools.eval_val_to_json import detections_to_coco_annotations


def _empty_result():
    return [np.zeros((0, 5), dtype=np.float32) for _ in range(25)]


def test_dense_export_keeps_low_score_candidate():
    result = _empty_result()
    result[0] = np.array([[0, 0, 2, 2, 0.0001]], dtype=np.float32)
    anns, next_id = detections_to_coco_annotations(result, image_id=7,
                                                   next_ann_id=1,
                                                   score_thresholds=None)
    assert [ann['category_id'] for ann in anns] == [0]
    assert anns[0]['score'] == np.float32(0.0001).item()
    assert next_id == 2
    assert anns[0]['image_id'] == 7


def test_explicit_threshold_export_filters_by_class():
    result = _empty_result()
    result[0] = np.array([[0, 0, 2, 2, 0.39]], dtype=np.float32)
    result[1] = np.array([[0, 0, 2, 2, 0.40]], dtype=np.float32)
    anns, _ = detections_to_coco_annotations(
        result, image_id=7, next_ann_id=1,
        score_thresholds=[0.40] * 25)
    assert [ann['category_id'] for ann in anns] == [1]


def test_empty_detections_returns_zero_id():
    anns, next_id = detections_to_coco_annotations(
        _empty_result(), image_id=99, next_ann_id=42,
        score_thresholds=None)
    assert anns == []
    assert next_id == 42


def test_bbox_and_area_converted_from_xyxy_to_xywh():
    result = _empty_result()
    result[5] = np.array([[10, 20, 30, 60, 0.9]])
    anns, _ = detections_to_coco_annotations(
        result, image_id=1, next_ann_id=1, score_thresholds=None)
    assert len(anns) == 1
    ann = anns[0]
    assert ann['bbox'] == [10, 20, 20, 40]
    assert ann['area'] == 20 * 40
    assert ann['category_id'] == 5
