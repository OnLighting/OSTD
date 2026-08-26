"""Tests for the exact per-superclass training-time threshold search.

The production mutation caught by this file is using one global threshold
or a hand-written grid instead of distinct exact score breakpoints per
superclass.
"""

import pytest

from mmdet.core.evaluation import search_superclass_thresholds


def _events_with_all_superclasses():
    """Literal events with one ship, one aircraft, and one vehicle class."""
    events = {i: [] for i in range(25)}
    total_gt = {i: 0 for i in range(25)}
    events[0] = [(0.91, 1), (0.90, 0)]
    total_gt[0] = 1
    events[4] = [(0.61, 1), (0.60, 0)]
    total_gt[4] = 1
    events[24] = [(0.21, 1), (0.20, 0)]
    total_gt[24] = 1
    return events, total_gt


def test_search_uses_one_exact_threshold_per_superclass():
    events, total_gt = _events_with_all_superclasses()
    result = search_superclass_thresholds(events, total_gt, max_fdr=0.19)
    assert result['thresholds_by_super'] == {
        'ship': 0.91,
        'aircraft': 0.61,
        'vehicle': 0.21,
    }
    assert result['score_thresholds'][:4] == (0.91,) * 4
    assert result['score_thresholds'][4:24] == (0.61,) * 20
    assert result['score_thresholds'][24] == 0.21


def test_search_marks_missing_superclass_gt_unavailable():
    events, total_gt = _events_with_all_superclasses()
    total_gt[24] = 0
    result = search_superclass_thresholds(events, total_gt, max_fdr=0.19)
    assert result['thresholds_by_super']['vehicle'] is None
    assert result['metrics']['official']['available'] is False
    assert 'vehicle' in result['metrics']['official']['unavailable_superclasses']
    # A group with no GT cannot be expanded to a usable 25-value tuple; the
    # caller must observe unavailability instead of receiving a fallback.
    assert result['score_thresholds'] is None


def test_search_independent_across_superclasses():
    """Changing vehicle events must not move ship/aircraft thresholds."""
    events, total_gt = _events_with_all_superclasses()
    first = search_superclass_thresholds(events, total_gt, max_fdr=0.19)
    events[24] = [(0.51, 1), (0.50, 0)]
    second = search_superclass_thresholds(events, total_gt, max_fdr=0.19)
    assert (first['thresholds_by_super']['ship']
            == second['thresholds_by_super']['ship'])
    assert (first['thresholds_by_super']['aircraft']
            == second['thresholds_by_super']['aircraft'])
    assert first['thresholds_by_super']['vehicle'] == 0.21
    assert second['thresholds_by_super']['vehicle'] == 0.51


def test_search_fdr_boundary_is_inclusive():
    """A point whose superclass mean FDR equals the limit stays feasible.

    Ship class 0: TP@0.95, TP@0.90, FP@0.85, TP@0.80 with 3 GTs; classes
    1..3 empty. At threshold 0.80: class-0 recall 1.0, FDR 1/4; mean over
    the 4 ship classes = recall 0.25, FDR 1/16 = 0.0625 (exact binary
    fraction, so the equality check is float-safe).
    """
    events, total_gt = _events_with_all_superclasses()
    events[0] = [(0.95, 1), (0.90, 1), (0.85, 0), (0.80, 1)]
    total_gt[0] = 3

    inclusive = search_superclass_thresholds(events, total_gt, max_fdr=0.0625)
    assert inclusive['thresholds_by_super']['ship'] == 0.80
    assert inclusive['metrics']['by_super']['ship']['fdr'] == \
        pytest.approx(0.0625)

    below = search_superclass_thresholds(events, total_gt, max_fdr=0.06)
    # 0.80 (FDR 0.0625) and 0.85 (FDR 1/12 ≈ 0.0833) become infeasible; the
    # best feasible recall point is 0.90 with zero FDR.
    assert below['thresholds_by_super']['ship'] == 0.90


def test_search_prefers_higher_recall():
    """Among feasible points, the highest superclass Recall wins.

    Ship class 0: TP@0.95, TP@0.91, FP@0.50, FP@0.49 with 2 GTs. Thresholds
    0.91..0.95 keep both TPs (mean recall 0.25); lower ones trade FDR for
    nothing, so the recall-first rule must pick a both-TP point, and the
    lower-FDR tiebreak then selects 0.91.
    """
    events, total_gt = _events_with_all_superclasses()
    events[0] = [(0.95, 1), (0.91, 1), (0.50, 0), (0.49, 0)]
    total_gt[0] = 2
    result = search_superclass_thresholds(events, total_gt, max_fdr=0.19)
    assert result['thresholds_by_super']['ship'] == 0.91
    assert result['metrics']['by_super']['ship']['recall'] == \
        pytest.approx(0.25)


def test_search_recall_ties_break_to_lower_fdr():
    """Equal superclass Recall → prefer the point with lower FDR.

    Ship class 0: TP@0.95, FP@0.60; class 1: TP@0.95, FP@0.70. Every
    threshold in (0.70, 0.95] keeps only the TPs (mean recall 0.5, FDR 0);
    0.70 admits class-1's FP (FDR 0.125); 0.60 admits both (FDR 0.25).
    All are feasible at 0.19, so the FDR tiebreak must select 0.95.
    """
    events, total_gt = _events_with_all_superclasses()
    events[0] = [(0.95, 1), (0.60, 0)]
    events[1] = [(0.95, 1), (0.70, 0)]
    total_gt[0] = 1
    total_gt[1] = 1
    result = search_superclass_thresholds(events, total_gt, max_fdr=0.19)
    assert result['thresholds_by_super']['ship'] == 0.95
    assert result['metrics']['by_super']['ship']['recall'] == \
        pytest.approx(0.5)
    assert result['metrics']['by_super']['ship']['fdr'] == \
        pytest.approx(0.0)


def test_search_full_ties_break_to_higher_threshold():
    """Equal Recall AND equal FDR → prefer the higher threshold.

    A genuine (recall, FDR) tie between exact breakpoints needs a child
    class without GT whose FDR saturates at 1.0 once any FP survives:
    adding more FPs to it leaves the superclass aggregate unchanged.
    Ship class 0: TP@0.85 with 1 GT; class 1 (no GT): FP@0.95, FP@0.80.
    Thresholds 0.85 and 0.80 both give mean recall 0.5 and mean FDR 0.5,
    so the determinism rule must keep the higher one, 0.85.
    """
    events, total_gt = _events_with_all_superclasses()
    events[0] = [(0.85, 1)]
    events[1] = [(0.95, 0), (0.80, 0)]
    total_gt[0] = 1
    total_gt[1] = 0
    result = search_superclass_thresholds(events, total_gt, max_fdr=0.5)
    assert result['thresholds_by_super']['ship'] == 0.85
    assert result['metrics']['by_super']['ship']['recall'] == \
        pytest.approx(0.5)
    assert result['metrics']['by_super']['ship']['fdr'] == \
        pytest.approx(0.5)


def test_search_returns_metrics_matching_evaluation_shape():
    """The bundled metrics dict mirrors evaluate_score_events' sections."""
    events, total_gt = _events_with_all_superclasses()
    result = search_superclass_thresholds(events, total_gt, max_fdr=0.19)
    metrics = result['metrics']
    assert set(metrics.keys()) >= {'per_class', 'by_super', 'official',
                                   'merged'}
    assert metrics['official']['available'] is True
    # Ship mean recall 0.25, aircraft 0.05, vehicle 1.0.
    assert metrics['official']['recall'] == \
        pytest.approx((0.25 + 0.05 + 1.0) / 3)
    assert metrics['by_super']['ship']['recall'] == pytest.approx(0.25)
