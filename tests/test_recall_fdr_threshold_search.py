"""Integration contracts for the post-training class-threshold search."""

import json
import sys

from mmdet.core.evaluation import CLASS_NAMES
from mmdet.core.evaluation.threshold_artifact import sha256_file
from tools import search_recall_fdr_thresholds


def _write_search_inputs(tmp_path):
    images = [{'id': 1, 'file_name': 'one.jpg'}]
    gt_annotations = []
    pred_annotations = []
    for category_id in range(25):
        x = category_id * 20
        gt_annotations.append({
            'id': category_id + 1,
            'image_id': 1,
            'category_id': category_id,
            'bbox': [x, 0, 10, 10],
        })
        pred_annotations.append({
            'id': category_id + 1,
            'image_id': 1,
            'category_id': category_id,
            'bbox': [x, 0, 10, 10],
            'score': 0.9,
        })
    categories = [
        {'id': category_id, 'name': name}
        for category_id, name in enumerate(CLASS_NAMES)
    ]
    gt_path = tmp_path / 'gt.json'
    pred_path = tmp_path / 'pred.json'
    gt_path.write_text(json.dumps({
        'images': images,
        'annotations': gt_annotations,
        'categories': categories,
    }), encoding='utf-8')
    pred_path.write_text(json.dumps({
        'images': images,
        'annotations': pred_annotations,
        'categories': categories,
    }), encoding='utf-8')
    return pred_path, gt_path


def test_search_cli_writes_checkpoint_bound_versioned_artifact(
        tmp_path, monkeypatch):
    pred_path, gt_path = _write_search_inputs(tmp_path)
    checkpoint = tmp_path / 'best.pth'
    checkpoint.write_bytes(b'stage-two-best')
    prefix = tmp_path / 'final_thresholds'
    monkeypatch.setattr(sys, 'argv', [
        'search_recall_fdr_thresholds.py',
        '--pred', str(pred_path),
        '--gt', str(gt_path),
        '--checkpoint', str(checkpoint),
        '--max-official-fdr', '0.19',
        '--out-prefix', str(prefix),
    ])

    search_recall_fdr_thresholds.main()

    payload = json.loads(
        (tmp_path / 'final_thresholds.json').read_text(encoding='utf-8'))
    assert payload['schema_version'] == 1
    assert payload['checkpoint']['sha256'] == sha256_file(checkpoint)
    assert [row['category_id'] for row in payload['classes']] == list(range(25))
    assert [row['name'] for row in payload['classes']] == list(CLASS_NAMES)
    assert payload['constraints']['max_official_fdr'] == 0.19
    assert (tmp_path / 'final_thresholds_filtered_preds.json').is_file()
