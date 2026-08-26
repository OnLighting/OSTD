"""Validation tests for frozen score-threshold artifacts."""

import json

import pytest

from mmdet.core.evaluation.threshold_artifact import (
    load_threshold_artifact, write_threshold_artifact)


def _write_valid_artifact(tmp_path):
    checkpoint = tmp_path / 'best.pth'
    checkpoint.write_bytes(b'checkpoint-a')
    artifact = tmp_path / 'thresholds.json'
    write_threshold_artifact(
        artifact,
        [0.1] * 25,
        checkpoint,
        'dense.json',
        'val.json',
        {
            'max_official_fdr': 0.19,
            'target_official_recall': 0.85,
        },
        {'official': {'recall': 0.86, 'fdr': 0.18}},
    )
    return checkpoint, artifact


def test_artifact_round_trip_and_checkpoint_binding(tmp_path):
    checkpoint, artifact = _write_valid_artifact(tmp_path)
    assert load_threshold_artifact(artifact, checkpoint) == (0.1,) * 25


def test_artifact_rejects_different_checkpoint(tmp_path):
    _, artifact = _write_valid_artifact(tmp_path)
    checkpoint_b = tmp_path / 'other.pth'
    checkpoint_b.write_bytes(b'checkpoint-b')
    with pytest.raises(ValueError, match='SHA-256'):
        load_threshold_artifact(artifact, checkpoint_b)


@pytest.mark.parametrize('mutation, message', [
    (lambda payload: payload.update(schema_version=2), 'schema_version'),
    (lambda payload: payload['classes'].pop(), '25'),
    (lambda payload: payload['classes'].__setitem__(
        1, dict(payload['classes'][0])), 'category_id'),
    (lambda payload: payload['classes'][0].update(name='wrong'), 'name'),
    (lambda payload: payload['classes'][0].update(threshold=-0.1),
     'non-negative'),
    (lambda payload: payload['checkpoint'].pop('sha256'), 'sha256'),
    (lambda payload: payload.pop('source'), 'source'),
    (lambda payload: payload['source'].pop('gt_path'), 'gt_path'),
    (lambda payload: payload.pop('constraints'), 'constraints'),
    (lambda payload: payload['constraints'].pop('max_official_fdr'),
     'max_official_fdr'),
    (lambda payload: payload['constraints'].pop('target_official_recall'),
     'target_official_recall'),
    (lambda payload: payload.pop('metrics'), 'metrics'),
    (lambda payload: payload['metrics'].pop('official'), 'official'),
])
def test_artifact_rejects_malformed_payload(tmp_path, mutation, message):
    checkpoint, artifact = _write_valid_artifact(tmp_path)
    payload = json.loads(artifact.read_text(encoding='utf-8'))
    mutation(payload)
    artifact.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match=message):
        load_threshold_artifact(artifact, checkpoint)
