"""Custom CopyPaste fallback for mmdet versions without built-in CopyPaste.

以概率 prob 把另一张随机图的 patch (含 boxes) 粘贴到当前样本上。简化版
CopyPaste，仅做粘贴（不做擦除/合成损失）。
"""
import random

import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class RandCopyPaste:
    """Random CopyPaste augmentation.

    Args:
        prob (float): probability of applying the transform.
        cache_size (int): number of additional images to cache for source
            sampling. Default 64 (cheap).
    """

    def __init__(self, prob=0.5, cache_size=64):
        self.prob = float(prob)
        self.cache_size = int(cache_size)
        # 采样源池在第一次 forward 时懒填充；通过 dataset 注入。
        self._pool = []

    def _populate_pool(self, results):
        """由第一个 sample 触发：根据 results['dataset'] 填一个池子。
        mmdet 自定义数据集会在 sample 字典中携带 img_info 与 ann_info，
        我们从 dataset.data_infos 取 N 张候选。
        """
        ds = results.get('dataset')
        if ds is None or self.cache_size <= 0:
            return
        infos = getattr(ds, 'data_infos', None)
        if infos is None or len(infos) == 0:
            return
        n = min(self.cache_size, len(infos))
        idxs = np.random.choice(len(infos), size=n, replace=False)
        self._pool = list(idxs)

    def __call__(self, results):
        if random.random() > self.prob:
            return results
        if not self._pool:
            self._populate_pool(results)
            if not self._pool:
                return results
        src_idx = random.choice(self._pool)
        # 获取源样本
        ds = results['dataset']
        src_info = ds.data_infos[src_idx]
        src_anns = ds.get_ann_info(src_idx)
        # 读源图像（mmcv.imread 会缓存到 file_client）
        src_img = mmcv.imread(src_info['filename'])
        src_bboxes = np.asarray(src_anns['bboxes'], dtype=np.float32)
        if src_bboxes.size == 0:
            return results
        src_masks = src_anns.get('masks', None)  # 可选
        # 简单做法：把 src 图直接 resize 到 results['img_shape'] 后整图叠加
        # （alpha=0.5）；boxes 全部追加；labels 取 src 的第一个类（保守）。
        # 简化版不做 mask 级别 paste，足够给目标域引入风格/分布漂移信号。
        h, w = results['img'].shape[:2]
        if src_img.shape[:2] != (h, w):
            src_img = mmcv.imresize(src_img, (w, h))
        # 叠合：alpha = 0.5 混合
        results['img'] = (0.5 * results['img'] + 0.5 * src_img).astype(np.uint8)
        # boxes 追加并 clip 到当前图边界
        src_bboxes[:, 0::2] = np.clip(src_bboxes[:, 0::2], 0, w)
        src_bboxes[:, 1::2] = np.clip(src_bboxes[:, 1::2], 0, h)
        if len(results['gt_bboxes']) == 0:
            results['gt_bboxes'] = src_bboxes
            results['gt_labels'] = np.asarray(
                src_anns['labels'], dtype=np.int64)[:src_bboxes.shape[0]]
        else:
            results['gt_bboxes'] = np.concatenate(
                [results['gt_bboxes'], src_bboxes], axis=0)
            results['gt_labels'] = np.concatenate([
                results['gt_labels'],
                np.asarray(src_anns['labels'], dtype=np.int64)[:src_bboxes.shape[0]]
            ], axis=0)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'