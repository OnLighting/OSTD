"""Tests for the shared official Recall/FDR metric module.

Covers constants, threshold filtering, one-to-one matching, superclass
aggregation, merged counts, IoU per class, and the official model
comparison used by checkpoint hooks.
"""

import numpy as np

from mmdet.core.evaluation.official_metrics import (
    CLASS_NAMES,
    CLASS_SCORE_THRESHOLDS,
    CLASS_IOU_THRESHOLDS,
    SUPERCLASS_INDICES,
    compare_official_candidates,
    evaluate_mmdet_results,
    filter_mmdet_results,
)


EXPECTED_THRESHOLDS = (
    .348537356, .00188686815, .00998581946, .0174246859,
    .854384005, .554479778, .469661117, .424196422, .451967984,
    .927510798, .927627504, .33063519, .791748285, .987444103,
    .186482817, .69058013, .586889446, .936379254, .362753332,
    .145738885, .877446949, .038396392, .141620845, .0739553422,
    .0403826572)


def _empty_result():
    """Return a 25-class prediction list with one empty array per class."""
    return [np.zeros((0, 5), dtype=np.float32) for _ in range(25)]


def _gt(bboxes, category_ids, image_ids=None):
    """Build the gt_infos list expected by evaluate_mmdet_results.

    gt_infos has one entry per image; each entry is a list of annotation
    dicts in the COCO style with ``bbox`` (xywh) and ``category_id``.
    """
    if image_ids is None:
        image_ids = list(range(len(bboxes)))
    out = []
    for boxes, cats, img_id in zip(bboxes, category_ids, image_ids):
        anns = []
        for box, cat in zip(boxes, cats):
            anns.append({
                'bbox': [float(box[0]), float(box[1]),
                         float(box[2]), float(box[3])],
                'category_id': int(cat),
                'image_id': int(img_id),
            })
        out.append(anns)
    return out


def test_hardcoded_thresholds_match_selected_csv_values():
    assert CLASS_SCORE_THRESHOLDS == EXPECTED_THRESHOLDS


def test_class_names_cover_categories_0_through_24():
    assert len(CLASS_NAMES) == 25
    assert CLASS_SCORE_THRESHOLDS is not None and len(CLASS_SCORE_THRESHOLDS) == 25
    assert CLASS_IOU_THRESHOLDS is not None and len(CLASS_IOU_THRESHOLDS) == 25
    assert set(SUPERCLASS_INDICES.keys()) == {'ship', 'aircraft', 'vehicle'}
    seen = set()
    for ids in SUPERCLASS_INDICES.values():
        seen.update(ids)
    assert seen == set(range(25))


def test_vehicle_class_uses_lower_iou():
    """FSC (category 24) is the only vehicle class with IoU 0.35."""
    assert CLASS_IOU_THRESHOLDS[24] == 0.35
    for idx in range(24):
        assert CLASS_IOU_THRESHOLDS[idx] == 0.5


def test_filter_keeps_per_class_thresholds():
    result = _empty_result()
    result[0] = np.array([[0, 0, 2, 2, CLASS_SCORE_THRESHOLDS[0] - 1e-6]])
    result[1] = np.array([[0, 0, 2, 2, CLASS_SCORE_THRESHOLDS[1]]])
    filtered = filter_mmdet_results(result)
    assert len(filtered[0]) == 0
    assert len(filtered[1]) == 1


def test_duplicate_prediction_is_false_positive():
    """Two predictions covering the same GT: top score → TP, rest → FP."""
    result = _empty_result()
    result[0] = np.array([[0, 0, 10, 10, .9], [0, 0, 10, 10, .8]])
    metrics = evaluate_mmdet_results([result], _gt([[[0, 0, 10, 10]]], [[0]]))
    assert metrics['per_class'][0]['tp'] == 1
    assert metrics['per_class'][0]['fp'] == 1
    assert metrics['per_class'][0]['fn'] == 0


def test_score_below_threshold_is_false_positive_filtered():
    result = _empty_result()
    result[0] = np.array([[0, 0, 10, 10, CLASS_SCORE_THRESHOLDS[0] - 1e-6]])
    metrics = evaluate_mmdet_results([result], _gt([[[0, 0, 10, 10]]], [[0]]))
    assert metrics['per_class'][0]['tp'] == 0
    assert metrics['per_class'][0]['fp'] == 0
    assert metrics['per_class'][0]['fn'] == 1


def test_no_predictions_no_gt_is_unavailable():
    """Empty val pool → no GT in any superclass → official unavailable."""
    import math
    result = _empty_result()
    metrics = evaluate_mmdet_results([result], _gt([[]], [[]]))
    assert metrics['official']['available'] is False
    assert math.isnan(metrics['official']['recall'])
    assert math.isnan(metrics['official']['fdr'])


def test_superclass_average_excludes_missing_class():
    """When a superclass has zero GT, that superclass is excluded from the
    official mean rather than counted as 0/0. Per the design doc, ANY
    missing-superclass marks the official aggregate as unavailable
    (``NaN``) so checkpoint hooks can refuse to save such a run."""
    result = _empty_result()
    result[4] = np.array([[0, 0, 10, 10, .99]])  # aircraft SU-35
    # No aircraft GT, only ship GT.
    metrics = evaluate_mmdet_results(
        [result],
        _gt([[[0, 0, 10, 10]]], [[0]]))  # single ship GT
    assert metrics['by_super']['aircraft']['recall'] is None
    assert metrics['official']['available'] is False
    assert 'aircraft' in metrics['official']['unavailable_superclasses']
    import math
    assert math.isnan(metrics['official']['recall'])
    assert math.isnan(metrics['official']['fdr'])


def test_matcher_skips_already_matched_gt():
    """A prediction falls back to its best still-unmatched GT."""
    import math

    result = _empty_result()
    # Both predictions are exact matches for GT-A. After the higher-scoring
    # prediction takes A, the second must ignore A and match overlapping GT-B
    # at IoU 70/130 = 0.538 (> ship threshold 0.5).
    result[0] = np.array([
        [0, 0, 10, 10, .9],
        [0, 0, 10, 10, .8],
    ])
    metrics = evaluate_mmdet_results(
        [result],
        _gt([[[0, 0, 10, 10], [3, 0, 10, 10]]], [[0, 0]]))
    assert metrics['per_class'][0]['tp'] == 2
    assert metrics['per_class'][0]['fp'] == 0
    assert metrics['per_class'][0]['fn'] == 0
    # This fixture contains ship GT only, so the matcher result is valid but
    # the three-superclass official aggregate must remain unavailable.
    assert math.isnan(metrics['official']['recall'])
    assert metrics['official']['available'] is False


def test_official_unavailable_when_vehicle_missing():
    """No FSC GT → official aggregate is unavailable (NaN)."""
    result = _empty_result()
    result[0] = np.array([[0, 0, 10, 10, .9]])
    metrics = evaluate_mmdet_results(
        [result],
        _gt([[[0, 0, 10, 10]]], [[0]]))  # only ship GT
    assert metrics['official']['available'] is False
    assert 'vehicle' in metrics['official']['unavailable_superclasses']


def test_vehicle_iou_35_matches_lower_threshold():
    """FSC IoU 0.35: a box with IoU 0.40 matches, IoU 0.30 does not."""
    # GT box: 100x100 at origin → area 10000.
    # Pred box: 80x80 at (10, 10) → area 6400, intersection 80*80=6400,
    # union 10000+6400-6400=10000, IoU=0.64. We test instead the boundary
    # at 0.35 by using non-symmetric GT.
    # GT box: (0,0,10,10) → 10x10.
    # Pred box: (3,3,9,9) → 6x6 = 36, intersection (3,3,9,9)→ (3,3,9,9)=6*6=36,
    # union = 100+36-36=100, IoU=0.36 (above 0.35).
    result = _empty_result()
    result[24] = np.array([[3, 3, 6, 6, .99]])
    metrics = evaluate_mmdet_results(
        [result], _gt([[[0, 0, 10, 10]]], [[24]]))
    assert metrics['per_class'][24]['tp'] == 1


def test_ship_iou_50_rejects_iou_just_below():
    """Ship class requires IoU 0.50; a 0.40 IoU pred → FP."""
    # GT (0,0,10,10)=100, pred (2,2,9,9)=7*7=49, intersection
    # (2,2,9,9)→49, union=100+49-49=100, IoU=0.49 (below 0.50).
    result = _empty_result()
    result[0] = np.array([[2, 2, 7, 7, .99]])
    metrics = evaluate_mmdet_results(
        [result], _gt([[[0, 0, 10, 10]]], [[0]]))
    assert metrics['per_class'][0]['tp'] == 0
    assert metrics['per_class'][0]['fp'] == 1
    assert metrics['per_class'][0]['fn'] == 1


def test_merged_counts_aggregate_all_classes():
    """Merged counts sum across categories regardless of superclass."""
    result = _empty_result()
    # 2 ship TPs, 1 aircraft TP, 1 ship FP, 1 aircraft FN.
    result[0] = np.array([[0, 0, 5, 5, .99], [0, 10, 5, 5, .99],
                          [50, 50, 5, 5, .99]])
    result[4] = np.array([[0, 20, 5, 5, .99]])
    metrics = evaluate_mmdet_results(
        [result],
        _gt([[[0, 0, 5, 5], [0, 10, 5, 5], [0, 20, 5, 5]]], [[0, 0, 4]]))
    assert metrics['merged']['tp'] == 3
    assert metrics['merged']['fp'] == 1
    assert metrics['merged']['fn'] == 0


def test_compare_returns_true_when_best_is_none():
    assert compare_official_candidates(
        {'recall': .5, 'fdr': .3}, None) is True


def test_first_double_pass_replaces_fdr_failure():
    assert compare_official_candidates(
        {'recall': .86, 'fdr': .19}, {'recall': .90, 'fdr': .21}) is True


def test_second_double_pass_replaces_first_double_pass():
    assert compare_official_candidates(
        {'recall': .90, 'fdr': .15}, {'recall': .86, 'fdr': .19}) is True


def test_non_passing_cannot_replace_passing_best():
    assert compare_official_candidates(
        {'recall': .90, 'fdr': .25}, {'recall': .86, 'fdr': .19}) is False


def test_non_passing_best_prefers_higher_recall():
    assert compare_official_candidates(
        {'recall': .80, 'fdr': .25}, {'recall': .70, 'fdr': .10}) is True
    assert compare_official_candidates(
        {'recall': .70, 'fdr': .10}, {'recall': .80, 'fdr': .25}) is False


def test_tied_recall_within_tolerance_prefers_lower_fdr():
    assert compare_official_candidates(
        {'recall': .884, 'fdr': .16}, {'recall': .880, 'fdr': .18}) is True
    assert compare_official_candidates(
        {'recall': .880, 'fdr': .18}, {'recall': .884, 'fdr': .16}) is False


def test_over_tolerance_prefers_recall():
    assert compare_official_candidates(
        {'recall': .886, 'fdr': .19}, {'recall': .880, 'fdr': .10}) is True
    assert compare_official_candidates(
        {'recall': .880, 'fdr': .10}, {'recall': .886, 'fdr': .19}) is False


def test_exact_equal_does_not_replace():
    """Equal recall/FDR (or within tolerance) does not replace the best."""
    assert compare_official_candidates(
        {'recall': .86, 'fdr': .19}, {'recall': .86, 'fdr': .19}) is False
