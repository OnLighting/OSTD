"""Tests for the official Recall/FDR best saver and early-stopping hooks.

These tests use a lightweight fake runner that mimics the parts of the
mmcv runner API the hooks touch (``work_dir``, ``epoch``, ``logger``,
``log_buffer.output``).
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np

from mmdet.core.evaluation.eval_hooks import (OfficialBestSaverHook,
                                              OfficialEarlyStoppingHook)
from mmdet.core.evaluation.official_metrics import compare_official_candidates


class _FakeLogBuffer:
    def __init__(self):
        self.output = {}


class _FakeRunner:
    def __init__(self, work_dir, epoch, metrics):
        self.work_dir = work_dir
        self.epoch = epoch
        self.log_buffer = _FakeLogBuffer()
        self.log_buffer.output.update(metrics)
        self.logger = MagicMock()


def test_first_double_pass_replaces_fdr_failure():
    assert compare_official_candidates(
        {'recall': .86, 'fdr': .19}, {'recall': .90, 'fdr': .21})


def test_non_passing_cannot_replace_passing_best():
    assert not compare_official_candidates(
        {'recall': .90, 'fdr': .25}, {'recall': .86, 'fdr': .19})


def test_tied_recall_prefers_lower_fdr():
    assert compare_official_candidates(
        {'recall': .884, 'fdr': .16}, {'recall': .880, 'fdr': .18})


def test_over_tolerance_prefers_recall():
    assert compare_official_candidates(
        {'recall': .886, 'fdr': .19}, {'recall': .880, 'fdr': .10})


class OfficialSaverHookTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        # Stage an epoch checkpoint the hook can copy.
        self.ckpt = os.path.join(self.tmp, 'epoch_1.pth')
        with open(self.ckpt, 'wb') as f:
            f.write(b'dummy-ckpt-1')

    def _make_runner(self, epoch, recall, fdr):
        return _FakeRunner(
            self.tmp, epoch=epoch,
            metrics={'official_recall': recall, 'official_fdr': fdr})

    def test_no_double_pass_first_epoch_is_best(self):
        hook = OfficialBestSaverHook()
        runner = self._make_runner(epoch=0, recall=.80, fdr=.15)
        hook.after_train_epoch(runner)
        best_ckpt = os.path.join(self.tmp, 'best_official_recall_fdr.pth')
        meta_path = os.path.join(self.tmp, 'best_official_recall_fdr.json')
        self.assertTrue(os.path.exists(best_ckpt))
        self.assertTrue(os.path.exists(meta_path))
        with open(meta_path) as f:
            meta = json.load(f)
        self.assertEqual(meta['epoch'], 1)
        self.assertAlmostEqual(meta['official_recall'], .80, places=6)
        self.assertAlmostEqual(meta['official_fdr'], .15, places=6)
        self.assertFalse(meta['passed'])

    def test_double_pass_replaces_non_double_pass(self):
        hook = OfficialBestSaverHook()
        hook.after_train_epoch(self._make_runner(epoch=0, recall=.80, fdr=.15))
        # Write a new checkpoint the hook will copy.
        with open(os.path.join(self.tmp, 'epoch_2.pth'), 'wb') as f:
            f.write(b'dummy-ckpt-2')
        hook.after_train_epoch(self._make_runner(epoch=1, recall=.86, fdr=.19))
        meta_path = os.path.join(self.tmp, 'best_official_recall_fdr.json')
        with open(meta_path) as f:
            meta = json.load(f)
        self.assertEqual(meta['epoch'], 2)
        self.assertTrue(meta['passed'])
        self.assertAlmostEqual(meta['official_recall'], .86, places=6)

    def test_weak_non_passing_does_not_replace_passing(self):
        hook = OfficialBestSaverHook()
        hook.after_train_epoch(self._make_runner(epoch=0, recall=.86, fdr=.19))
        with open(os.path.join(self.tmp, 'epoch_2.pth'), 'wb') as f:
            f.write(b'dummy-ckpt-2')
        hook.after_train_epoch(self._make_runner(epoch=1, recall=.90, fdr=.25))
        meta_path = os.path.join(self.tmp, 'best_official_recall_fdr.json')
        with open(meta_path) as f:
            meta = json.load(f)
        self.assertEqual(meta['epoch'], 1)
        self.assertTrue(meta['passed'])

    def test_higher_recall_over_tolerance_replaces_passing(self):
        hook = OfficialBestSaverHook()
        hook.after_train_epoch(self._make_runner(epoch=0, recall=.86, fdr=.19))
        with open(os.path.join(self.tmp, 'epoch_2.pth'), 'wb') as f:
            f.write(b'dummy-ckpt-2')
        hook.after_train_epoch(self._make_runner(epoch=1, recall=.90, fdr=.15))
        meta_path = os.path.join(self.tmp, 'best_official_recall_fdr.json')
        with open(meta_path) as f:
            meta = json.load(f)
        self.assertEqual(meta['epoch'], 2)


class OfficialEarlyStoppingHookTest(unittest.TestCase):

    def test_patience_counter_only_resets_on_acceptance(self):
        """A rejected candidate should NOT reset the patience counter."""
        from mmdet.core.evaluation.eval_hooks import EarlyStopping
        hook = OfficialEarlyStoppingHook(patience=2)
        # First epoch is the initial best (no comparator decision yet).
        runner = _FakeRunner('/tmp', epoch=0,
                             metrics={'official_recall': .80, 'official_fdr': .15})
        hook.after_train_epoch(runner)
        # Candidate is worse in Recall but still passes — best should hold.
        # Use candidate (.85,.21) which fails FDR (above 0.20).
        runner.log_buffer.output.update({'official_recall': .85,
                                         'official_fdr': .21})
        hook.after_train_epoch(runner)
        runner.log_buffer.output.update({'official_recall': .85,
                                         'official_fdr': .21})
        hook.after_train_epoch(runner)
        # After 2 non-improvements we expect EarlyStopping to be raised.
        with self.assertRaises(EarlyStopping):
            hook.after_train_epoch(runner)


if __name__ == '__main__':
    unittest.main()
