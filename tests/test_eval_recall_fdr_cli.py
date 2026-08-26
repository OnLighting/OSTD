"""Integration-level contracts for the standalone Recall/FDR CLI."""

import json
import sys

from tools import eval_recall_fdr


def test_cli_marks_official_metric_unavailable_when_a_superclass_has_no_gt(
        tmp_path, monkeypatch):
    gt_path = tmp_path / 'gt.json'
    pred_path = tmp_path / 'pred.json'
    out_prefix = tmp_path / 'metrics'
    gt_path.write_text(json.dumps({
        'images': [{'id': 1, 'file_name': 'one.jpg'}],
        'annotations': [{
            'id': 1,
            'image_id': 1,
            'category_id': 0,
            'bbox': [0, 0, 10, 10],
        }],
        'categories': [],
    }), encoding='utf-8')
    pred_path.write_text(json.dumps({
        'images': [{'id': 1, 'file_name': 'one.jpg'}],
        'annotations': [],
        'categories': [],
    }), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', [
        'eval_recall_fdr.py',
        '--pred', str(pred_path),
        '--gt', str(gt_path),
        '--out-prefix', str(out_prefix),
    ])

    eval_recall_fdr.main()

    payload = json.loads(
        (tmp_path / 'metrics.json').read_text(encoding='utf-8'))
    official = payload['overall']['official']
    assert official['available'] is False
    assert set(official['unavailable_superclasses']) == {'aircraft', 'vehicle'}
