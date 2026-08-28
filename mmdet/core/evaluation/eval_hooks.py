import os.path as osp
import shutil

import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from mmcv.runner import HOOKS, Hook
from torch.nn.modules.batchnorm import _BatchNorm


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


