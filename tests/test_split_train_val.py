"""Tests for the 8:2 stratified train/val splitter.

The official pipeline trains on a single 80% pool and selects best on
the 20% validation pool — no test split. The splitter must still keep
rare-class protection and minimum-training-box protection.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import pytest

from tools.split_val import (CLASS_NAMES, MIN_TRAIN_PER_CLASS,
                             RARE_THRESHOLD, collect_train_files,
                             image_classes, stratified_train_val,
                             stratified_three_way)


# (stem, [classes in image]) for a tiny fixture pool
STEMS = [f's{i:02d}' for i in range(20)]
# 10 ship-class images (cls 0), 10 aircraft images (cls 4), evenly spread.
LABELS_PATHS = []


def _make_label(stem, classes):
    """Return a fake label path object with .stem and .exists working."""
    from types import SimpleNamespace
    return SimpleNamespace(stem=stem, exists=lambda: True)


def _fake_label_content(classes):
    """Serialize classes into the lines image_classes reads."""
    return '\n'.join(f'{c} 0.5 0.5 0.2 0.2' for c in classes) + '\n'


# Use a more controlled fixture approach: pass synthetic labels directly.
def _split_with_labels(per_image_classes, seed=0):
    """Run stratified_train_val with constructed (stem, labels) tuples.

    ``per_image_classes`` is a list of lists of class ids per image.
    """
    from unittest.mock import patch

    stems = [f's{i:02d}' for i in range(len(per_image_classes))]
    label_paths = [_make_label(s, c) for s, c in zip(stems, per_image_classes)]

    # Patch image_classes indirectly via path objects.
    class _FakeLabelPath:
        def __init__(self, stem, classes):
            self.stem = stem
            self.classes = classes
            self._exists = True

        def exists(self):
            return self._exists

    label_paths = [_FakeLabelPath(s, c)
                   for s, c in zip(stems, per_image_classes)]

    label_index = {lp.stem: lp for lp in label_paths}

    def fake_image_classes(path):
        return set(path.classes)

    # Patch the function inside split_val to use ours.
    with patch('tools.split_val.image_classes', side_effect=fake_image_classes):
        return stratified_train_val(
            stems, label_paths, (0.8, 0.2), seed), label_index


def test_train_val_are_disjoint_and_complete():
    per_image = [[0]] * 10 + [[4]] * 10
    (train, val), _ = _split_with_labels(per_image)
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(
        f's{i:02d}' for i in range(len(per_image)))


def test_rare_class_protection_keeps_all_in_train():
    """An entire class with < RARE_THRESHOLD images stays in train."""
    per_image = [[0]] * 4 + [[4]] * 50  # ship class has 4 → rare (<5)
    (train, val), _ = _split_with_labels(per_image)
    train_set = set(train)
    ship_stems = [s for s, c in zip(STEMS[:len(per_image)], per_image)
                  if 0 in c]
    assert set(ship_stems) <= train_set


def test_minimum_train_box_protection():
    """If val takes most of a class, post-check returns one to train."""
    # Build a pool where the only image containing cls=4 is in val by ratio.
    # 10 ship + 2 aircraft: val 20% → 2 images → both could be aircraft.
    per_image = [[0]] * 10 + [[4]] * 2
    (train, val), label_index = _split_with_labels(per_image)
    # Aircraft should not be entirely val (MIN_TRAIN_PER_CLASS protection).
    train_classes = set()
    for s in train:
        train_classes.update(label_index[s].classes)
    assert 4 in train_classes


def test_ratios_must_be_two_values():
    with pytest.raises(ValueError):
        stratified_train_val(['a'], [], (0.6, 0.2, 0.2), 0)


def test_stratified_three_way_still_works():
    """Backwards compatibility: 3-way split still works if called directly."""
    from unittest.mock import patch

    per_image = [[0]] * 10 + [[4]] * 10
    stems = [f's{i:02d}' for i in range(20)]

    class _FakeLabelPath:
        def __init__(self, stem, classes):
            self.stem = stem
            self.classes = classes

    label_paths = [_FakeLabelPath(s, c)
                   for s, c in zip(stems, per_image)]

    def fake_image_classes(path):
        return set(path.classes)

    with patch('tools.split_val.image_classes', side_effect=fake_image_classes):
        train, val, test = stratified_three_way(
            stems, label_paths, (0.6, 0.2, 0.2), 0)
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert set(train) | set(val) | set(test) == set(stems)


class CliSplitTest(unittest.TestCase):

    def test_cli_does_not_create_test_dir(self):
        """Running the CLI with --ratios 0.8 0.2 must skip images/test."""
        from tools import split_val as split_module

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)

        img_dir = tmp / 'images' / 'train'
        lbl_dir = tmp / 'labels' / 'train'
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        # Make 20 tiny images + matching labels (cls 0 across all).
        for i in range(20):
            (img_dir / f's{i:02d}.jpg').write_bytes(b'x')
            (lbl_dir / f's{i:02d}.txt').write_text(
                '0 0.5 0.5 0.2 0.2\n', encoding='utf-8')

        # Invoke main() with explicit argv.
        old_argv = __import__('sys').argv
        __import__('sys').argv = [
            'split_val.py', '--root', str(tmp),
            '--ratios', '0.8', '0.2', '--seed', '0', '--overwrite']
        try:
            split_module.main()
        finally:
            __import__('sys').argv = old_argv

        assert (tmp / 'images' / 'val').exists()
        assert (tmp / 'labels' / 'val').exists()
        assert not (tmp / 'images' / 'test').exists()
        assert not (tmp / 'labels' / 'test').exists()
