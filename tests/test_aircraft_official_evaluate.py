"""Training-time AircraftDataset official-threshold integration tests."""

from collections import OrderedDict

import numpy as np

from mmdet.datasets.aircraft import AircraftDataset
from mmdet.datasets.coco import CocoDataset


def _results_with_all_superclasses():
    per_class = [np.zeros((0, 5), dtype=np.float32) for _ in range(25)]
    per_class[0] = np.array([[0, 0, 10, 10, 0.91]], dtype=np.float32)
    per_class[4] = np.array([[20, 0, 30, 10, 0.61]], dtype=np.float32)
    per_class[24] = np.array([[40, 0, 50, 10, 0.21]], dtype=np.float32)
    return [per_class]


def test_dataset_evaluate_searches_and_emits_superclass_thresholds(
        monkeypatch):
    dataset = object.__new__(AircraftDataset)
    dataset.cat2label = {category_id: category_id for category_id in range(25)}
    monkeypatch.setattr(
        CocoDataset, 'evaluate',
        lambda self, results, metric='bbox', **kwargs: OrderedDict())
    monkeypatch.setattr(AircraftDataset, '__len__', lambda self: 1)
    monkeypatch.setattr(
        AircraftDataset, 'get_ann_info',
        lambda self, index: {
            'bboxes': np.array([
                [0, 0, 10, 10],
                [20, 0, 30, 10],
                [40, 0, 50, 10],
            ], dtype=np.float32),
            'labels': np.array([0, 4, 24], dtype=np.int64),
        })

    metrics = dataset.evaluate(
        _results_with_all_superclasses(), metric=['official'])

    assert metrics['training_score_thresholds'] == {
        'ship': 0.91,
        'aircraft': 0.61,
        'vehicle': 0.21,
    }
    assert metrics['official_threshold_ship'] == 0.91
    assert metrics['official_threshold_aircraft'] == 0.61
    assert metrics['official_threshold_vehicle'] == 0.21
    assert metrics['official_fdr'] <= 0.19
