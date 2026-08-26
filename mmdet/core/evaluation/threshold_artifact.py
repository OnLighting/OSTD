"""Versioned, checkpoint-bound score-threshold artifact I/O."""

import hashlib
import json
import math
import os
from pathlib import Path

from .official_metrics import CLASS_NAMES, normalize_score_thresholds


SCHEMA_VERSION = 1


def sha256_file(path):
    """Return the lowercase SHA-256 hex digest for ``path``."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError('threshold artifact must contain a JSON object')
    if payload.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(
            f'unsupported schema_version={payload.get("schema_version")!r}')
    checkpoint = payload.get('checkpoint')
    if not isinstance(checkpoint, dict):
        raise ValueError('threshold artifact checkpoint must be an object')
    checkpoint_path = checkpoint.get('path')
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError('checkpoint path must be a non-empty string')
    checkpoint_hash = checkpoint.get('sha256')
    if (not isinstance(checkpoint_hash, str)
            or len(checkpoint_hash) != 64
            or any(ch not in '0123456789abcdef' for ch in checkpoint_hash)):
        raise ValueError(
            'checkpoint sha256 must be a lowercase SHA-256 hex digest')
    source = payload.get('source')
    if not isinstance(source, dict):
        raise ValueError('threshold artifact source must be an object')
    for key in ('prediction_path', 'gt_path'):
        if not isinstance(source.get(key), str) or not source[key]:
            raise ValueError(f'source {key} must be a non-empty string')
    constraints = payload.get('constraints')
    if not isinstance(constraints, dict):
        raise ValueError('threshold artifact constraints must be an object')
    for key in ('max_official_fdr', 'target_official_recall'):
        try:
            value = float(constraints[key])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f'constraints {key} must be numeric')
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'constraints {key} must be between 0 and 1')
    metrics = payload.get('metrics')
    if not isinstance(metrics, dict):
        raise ValueError('threshold artifact metrics must be an object')
    official = metrics.get('official')
    if not isinstance(official, dict):
        raise ValueError(
            'threshold artifact metrics official must be an object')
    for key in ('recall', 'fdr'):
        try:
            value = float(official[key])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f'metrics official {key} must be numeric')
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(
                f'metrics official {key} must be between 0 and 1')
    classes = payload.get('classes')
    if not isinstance(classes, list) or len(classes) != len(CLASS_NAMES):
        raise ValueError('threshold artifact must contain exactly 25 classes')
    thresholds = []
    for category_id, row in enumerate(classes):
        if not isinstance(row, dict) or row.get('category_id') != category_id:
            raise ValueError(
                f'class row {category_id} has invalid category_id')
        if row.get('name') != CLASS_NAMES[category_id]:
            raise ValueError(
                f'class row {category_id} has invalid name')
        try:
            threshold = float(row['threshold'])
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f'class row {category_id} threshold must be numeric')
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError('thresholds must be finite non-negative values')
        thresholds.append(threshold)
    normalize_score_thresholds(thresholds)
    return tuple(thresholds)


def write_threshold_artifact(path, thresholds, checkpoint_path,
                             prediction_path, gt_path, constraints, metrics):
    """Atomically write a schema-v1 threshold artifact and return it."""
    values = normalize_score_thresholds(thresholds)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise ValueError(f'checkpoint not found: {checkpoint_path}')
    payload = {
        'schema_version': SCHEMA_VERSION,
        'checkpoint': {
            'path': str(checkpoint_path),
            'sha256': sha256_file(checkpoint_path),
        },
        'source': {
            'prediction_path': str(prediction_path),
            'gt_path': str(gt_path),
        },
        'constraints': dict(constraints),
        'metrics': metrics,
        'classes': [
            {
                'category_id': category_id,
                'name': CLASS_NAMES[category_id],
                'threshold': float(threshold),
            }
            for category_id, threshold in enumerate(values)
        ],
    }
    _validated_payload(payload)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2,
                  allow_nan=False)
    os.replace(str(tmp_path), str(out_path))
    return payload


def load_threshold_artifact(path, checkpoint_path=None):
    """Load and validate thresholds, optionally binding them to a checkpoint."""
    artifact_path = Path(path)
    with artifact_path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    thresholds = _validated_payload(payload)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise ValueError(f'checkpoint not found: {checkpoint_path}')
        actual_hash = sha256_file(checkpoint_path)
        expected_hash = payload['checkpoint']['sha256']
        if actual_hash != expected_hash:
            raise ValueError(
                'checkpoint SHA-256 does not match threshold artifact: '
                f'expected {expected_hash}, got {actual_hash}')
    return thresholds


__all__ = [
    'SCHEMA_VERSION',
    'sha256_file',
    'write_threshold_artifact',
    'load_threshold_artifact',
]
