import json
import math
import os
import os.path as osp
import shutil

import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from mmcv.runner import HOOKS, Hook
from torch.nn.modules.batchnorm import _BatchNorm

from .official_metrics import compare_official_candidates


class EarlyStopping(Exception):
    """由 EarlyStoppingHook 触发、用于中断训练的专用异常。

    mmcv 1.4.0 的 EpochBasedRunner 主循环不检查 ``runner.should_stop``
    （只在更晚版本才纳入 while 条件），导致早停信号被忽略、训练继续跑到
    max_epochs。改用异常穿透 runner 的 epoch 循环，回到 tools/train.py 的
    main() 中被捕获后优雅退出——这是 1.4.0 上唯一可靠的早停中断方式。

    Attributes:
        epoch (int): 触发早停时的 0-indexed epoch（runner.epoch）。
        monitor (str): 监控的指标名。
        best_score (float): 停止时的历史最佳分数。
    """

    def __init__(self, epoch, monitor, best_score):
        super().__init__(
            f'EarlyStopping at epoch {epoch + 1}: {monitor} best={best_score:.4f}')
        self.epoch = epoch
        self.monitor = monitor
        self.best_score = best_score


class EvalHook(BaseEvalHook):

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmdet.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)


class DistEvalHook(BaseDistEvalHook):

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmdet.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    """轻量早停 hook（mmcv 1.4.0 无内置 EarlyStoppingHook 的兜底实现）。

    每个 epoch 验证后从 runner.log_buffer.output 读取 monitor 指标，
    连续 patience 个 epoch 无明显提升（< min_delta）则停止训练。

    Args:
        monitor (str): 监控的指标名，须与验证输出 metrics key 一致，
            默认 'bbox_mAP'。
        patience (int): 容忍连续多少个 epoch 无提升，默认 8。
        min_delta (float): 视为"提升"的最小幅度，默认 0.0。
    """

    def __init__(self, monitor='bbox_mAP', patience=8, min_delta=0.0):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -1.0
        self.wait_count = 0
        self.stopped_epoch = 0

    def after_train_epoch(self, runner):
        """训练 epoch 后检查验证指标，实现早停逻辑。"""
        metrics = runner.log_buffer.output
        if self.monitor not in metrics:
            return
        current = float(metrics[self.monitor])
        if current > self.best_score + self.min_delta:
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            # 注意：判断条件是 current > best + min_delta，而非 current > best。
            # 即使 current 略高于 best（如 0.6650 vs 0.6620），只要提升幅度
            # 不足 min_delta（此处 0.005），仍计为"无有效提升"。日志据此措辞，
            # 避免出现"未超过历史最佳"但数值其实更高的矛盾字样。
            runner.logger.info(
                f'EarlyStoppingHook: {self.monitor}={current:.4f} 提升幅度不足 '
                f'{self.min_delta:.4f}（当前最佳 {self.best_score:.4f}，'
                f'等待 {self.wait_count}/{self.patience}）')
            if self.wait_count >= self.patience:
                self.stopped_epoch = runner.epoch
                runner.should_stop = True
                runner.logger.info(
                    f'EarlyStoppingHook: 连续 {self.patience} 个 epoch 无提升，'
                    f'于 epoch {runner.epoch + 1} 提前停止训练')
                # mmcv 1.4.0 的 runner 不响应 should_stop，直接抛异常中断。
                # 分布式下只在 rank 0 抛，避免每个进程各抛一次造成混乱；
                # rank 0 退出后 DDP 训练自然终止。
                rank = dist.get_rank() if dist.is_available() and \
                    dist.is_initialized() else 0
                if rank == 0:
                    raise EarlyStopping(runner.epoch, self.monitor,
                                        self.best_score)


@HOOKS.register_module()
class BestSaverHook(Hook):
    """best 模型保存 hook（绕开 mmcv 1.4.0 save_best 的 bug）。

    每个验证 epoch 后从 runner.log_buffer.output 读取 monitor 指标，若创
    历史新高则把当前 latest checkpoint 复制为 work_dir/best_<monitor>.pth，
    始终只保留一份 best 文件。

    Args:
        monitor (str): 监控指标名，默认 'bbox_mAP'。
        save_prefix (str): best 文件名前缀，默认 'best_'。
    """

    def __init__(self, monitor='bbox_mAP', save_prefix='best_'):
        super().__init__()
        self.monitor = monitor
        self.save_prefix = save_prefix
        self.best_score = -1.0
        self.best_path = None

    def after_train_epoch(self, runner):
        metrics = runner.log_buffer.output
        if self.monitor not in metrics:
            runner.logger.warning(
                f'BestSaverHook: monitor 指标 "{self.monitor}" 未在验证输出'
                f'中找到（可用 keys: {list(metrics.keys())}），跳过')
            return
        current = float(metrics[self.monitor])
        if current <= self.best_score:
            return
        # 找当前 epoch 的 latest checkpoint（CheckpointHook 在本 hook 之前
        # 已保存 epoch_N.pth）
        ckpt = osp.join(runner.work_dir, f'epoch_{runner.epoch + 1}.pth')
        if not osp.exists(ckpt):
            runner.logger.warning(
                f'BestSaverHook: latest checkpoint {ckpt} 不存在，跳过复制')
            return
        out = osp.join(runner.work_dir, f'{self.save_prefix}{self.monitor}.pth')
        shutil.copy(ckpt, out)
        self.best_score = current
        self.best_path = out
        runner.logger.info(
            f'BestSaverHook: {self.monitor}={current:.4f} 创新高，'
            f'已保存 best 模型 -> {out}')


def _maybe_import_dist():
    """Helper for hooks that may run inside a DDP context."""
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            return _dist.get_rank()
    except Exception:
        pass
    return 0


@HOOKS.register_module()
class OfficialBestSaverHook(Hook):
    """按官方 Recall/FDR 选择 best checkpoint。

    从 ``runner.log_buffer.output`` 读取 ``official_recall`` 和
    ``official_fdr``（由 ``AircraftDataset.evaluate`` 在请求 ``official``
    metric 时写入），依据 ``compare_official_candidates`` 的六规则比较。
    通过则复制当前 epoch checkpoint 为 ``best_official_recall_fdr.pth``，
    并以原子方式写 ``best_official_recall_fdr.json`` 元数据。

    Args:
        recall_target (float): Recall 通过门槛，默认 0.85。
        fdr_limit (float): FDR 通过上限，默认 0.20。
        recall_tolerance (float): 双达标后的 Recall tiebreak 容差，默认
            0.005。
        ckpt_filename (str): 当前 epoch checkpoint 文件名模板（与
            CheckpointHook 一致），默认 ``epoch_{epoch}.pth``。
        save_filename (str): best checkpoint 文件名，
            默认 ``best_official_recall_fdr.pth``。
        meta_filename (str): best 元数据文件名，
            默认 ``best_official_recall_fdr.json``。
    """

    def __init__(self,
                 recall_target=0.85,
                 fdr_limit=0.20,
                 recall_tolerance=0.005,
                 ckpt_filename='epoch_{epoch}.pth',
                 save_filename='best_official_recall_fdr.pth',
                 meta_filename='best_official_recall_fdr.json'):
        super().__init__()
        self.recall_target = recall_target
        self.fdr_limit = fdr_limit
        self.recall_tolerance = recall_tolerance
        self.ckpt_filename = ckpt_filename
        self.save_filename = save_filename
        self.meta_filename = meta_filename
        # In-memory mirror of the on-disk best; avoids re-reading JSON each
        # epoch.
        self.best = None
        self.best_path = None

    def _candidate_payload(self, runner):
        metrics = runner.log_buffer.output
        if 'official_recall' not in metrics or 'official_fdr' not in metrics:
            return None
        recall = float(metrics['official_recall'])
        fdr = float(metrics['official_fdr'])
        # Any NaN here means the official aggregate is undefined — typically
        # because one of ship / aircraft / vehicle had no GT in the val pool.
        # Refuse to save so a missing-superclass epoch never silently beats a
        # passing one.
        if math.isnan(recall) or math.isnan(fdr):
            return None
        return {'recall': recall, 'fdr': fdr}

    def after_train_epoch(self, runner):
        candidate = self._candidate_payload(runner)
        if candidate is None:
            if ('official_recall' in runner.log_buffer.output
                    and math.isnan(float(runner.log_buffer.output['official_recall']))):
                runner.logger.warning(
                    'OfficialBestSaverHook: 官方指标因某一大类缺少 GT 而不可计算'
                    '，跳过本 epoch 的 best 保存')
            else:
                runner.logger.warning(
                    'OfficialBestSaverHook: official_recall/official_fdr 未在验证'
                    '输出中找到，跳过 best 保存')
            return
        if not compare_official_candidates(
                candidate, self.best,
                recall_target=self.recall_target,
                fdr_limit=self.fdr_limit,
                recall_tolerance=self.recall_tolerance):
            return
        # 找当前 epoch 的 latest checkpoint（CheckpointHook 在本 hook 之前
        # 已保存 epoch_{epoch+1}.pth）。
        ckpt = osp.join(
            runner.work_dir,
            self.ckpt_filename.format(epoch=runner.epoch + 1))
        if not osp.exists(ckpt):
            runner.logger.warning(
                f'OfficialBestSaverHook: latest checkpoint {ckpt} 不存在，'
                '跳过复制')
            return
        out_ckpt = osp.join(runner.work_dir, self.save_filename)
        out_meta = osp.join(runner.work_dir, self.meta_filename)
        shutil.copy(ckpt, out_ckpt)
        passed = (
            candidate['recall'] >= self.recall_target
            and candidate['fdr'] <= self.fdr_limit)
        meta = {
            'epoch': runner.epoch + 1,
            'official_recall': candidate['recall'],
            'official_fdr': candidate['fdr'],
            'passed': bool(passed),
            'recall_target': self.recall_target,
            'fdr_limit': self.fdr_limit,
            'recall_tolerance': self.recall_tolerance,
        }
        tmp_meta = out_meta + '.tmp'
        with open(tmp_meta, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        os.replace(tmp_meta, out_meta)
        self.best = candidate
        self.best_path = out_ckpt
        runner.logger.info(
            f'OfficialBestSaverHook: official_recall='
            f'{candidate["recall"]:.4f} official_fdr='
            f'{candidate["fdr"]:.4f} passed={passed}，'
            f'已保存 best -> {out_ckpt}')


@HOOKS.register_module()
class OfficialEarlyStoppingHook(Hook):
    """按官方 Recall/FDR 比较策略实现的早停 hook。

    与 ``EarlyStoppingHook`` 不同：当候选指标未被
    ``compare_official_candidates`` 接受时耐心计数才会增长，被接受时立刻
    重置。``patience`` 个连续未接受的 epoch 后抛出 ``EarlyStopping``，
    由 ``tools/train.py`` 捕获并优雅退出。

    Args:
        patience (int): 容忍多少个连续未接受 epoch，默认 16。
        recall_target (float): Recall 通过门槛，默认 0.85。
        fdr_limit (float): FDR 通过上限，默认 0.20。
        recall_tolerance (float): 双达标后 Recall tiebreak 容差，默认
            0.005。
    """

    def __init__(self,
                 patience=16,
                 recall_target=0.85,
                 fdr_limit=0.20,
                 recall_tolerance=0.005):
        super().__init__()
        self.patience = patience
        self.recall_target = recall_target
        self.fdr_limit = fdr_limit
        self.recall_tolerance = recall_tolerance
        self.best = None
        self.wait_count = 0
        self.stopped_epoch = 0

    def after_train_epoch(self, runner):
        metrics = runner.log_buffer.output
        if ('official_recall' not in metrics
                or 'official_fdr' not in metrics):
            return
        recall = float(metrics['official_recall'])
        fdr = float(metrics['official_fdr'])
        # NaN means the official aggregate is undefined (e.g. missing GT in
        # one of the three superclasses). Such an epoch must not be treated
        # as a new best; the wait counter advances to keep early-stopping
        # behaviour well-defined on degenerate val splits.
        if math.isnan(recall) or math.isnan(fdr):
            self.wait_count += 1
            runner.logger.warning(
                'OfficialEarlyStoppingHook: 官方指标因某一大类缺少 GT 而不可计算，'
                f'本 epoch 不被接受，等待 {self.wait_count}/{self.patience}')
            if self.wait_count >= self.patience:
                self.stopped_epoch = runner.epoch
                runner.should_stop = True
                runner.logger.info(
                    f'OfficialEarlyStoppingHook: 连续 {self.patience} 个 epoch '
                    '官方指标不可计算，于 epoch '
                    f'{runner.epoch + 1} 提前停止训练')
                rank = _maybe_import_dist()
                if rank == 0:
                    raise EarlyStopping(
                        runner.epoch, 'official_recall_fdr',
                        self.best.get('recall', 0.0) if self.best else 0.0)
            return
        candidate = {'recall': recall, 'fdr': fdr}
        accepted = compare_official_candidates(
            candidate, self.best,
            recall_target=self.recall_target,
            fdr_limit=self.fdr_limit,
            recall_tolerance=self.recall_tolerance)
        if accepted:
            self.best = candidate
            self.wait_count = 0
            return
        # 首个 epoch 也会被算作一次"接受"（best 从 None 变为候选），
        # 因此 wait_count 只在真正被拒绝的 epoch 才增长。
        if self.best is None:
            self.best = candidate
            self.wait_count = 0
            return
        self.wait_count += 1
        runner.logger.info(
            f'OfficialEarlyStoppingHook: official_recall='
            f'{candidate["recall"]:.4f} official_fdr='
            f'{candidate["fdr"]:.4f} 未超越当前最佳，'
            f'等待 {self.wait_count}/{self.patience}')
        if self.wait_count >= self.patience:
            self.stopped_epoch = runner.epoch
            runner.should_stop = True
            runner.logger.info(
                f'OfficialEarlyStoppingHook: 连续 {self.patience} 个 epoch '
                f'官方指标无提升，于 epoch {runner.epoch + 1} 提前停止训练')
            rank = _maybe_import_dist()
            if rank == 0:
                raise EarlyStopping(runner.epoch,
                                    'official_recall_fdr',
                                    self.best.get('recall', 0.0)
                                    if self.best else 0.0)

