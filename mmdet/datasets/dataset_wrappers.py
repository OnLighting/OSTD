import bisect
import math
import numbers
import random
from collections import defaultdict

import numpy as np
from mmcv.utils import print_log
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
        separate_eval (bool): Whether to evaluate the results
            separately if it is used as validation dataset.
            Defaults to True.
    """

    def __init__(self, datasets, separate_eval=True):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        self.separate_eval = separate_eval
        if not separate_eval:
            if any([isinstance(ds, CocoDataset) for ds in datasets]):
                raise NotImplementedError(
                    'Evaluating concatenated CocoDataset as a whole is not'
                    ' supported! Please set "separate_eval=True"')
            elif len(set([type(ds) for ds in datasets])) != 1:
                raise NotImplementedError(
                    'All the datasets should have same types')

        if hasattr(datasets[0], 'flag'):
            flags = []
            for i in range(0, len(datasets)):
                flags.append(datasets[i].flag)
            self.flag = np.concatenate(flags)

    def get_cat_ids(self, idx):
        """Get category ids of concatenated dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length')
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_cat_ids(sample_idx)

    def evaluate(self, results, logger=None, **kwargs):
        """Evaluate the results.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str: float]: AP results of the total dataset or each separate
            dataset if `self.separate_eval=True`.
        """
        assert len(results) == self.cumulative_sizes[-1], \
            ('Dataset and results have different sizes: '
             f'{self.cumulative_sizes[-1]} v.s. {len(results)}')

        # Check whether all the datasets support evaluation
        for dataset in self.datasets:
            assert hasattr(dataset, 'evaluate'), \
                    f'{type(dataset)} does not implement evaluate function'

        if self.separate_eval:
            dataset_idx = -1
            total_eval_results = dict()
            for size, dataset in zip(self.cumulative_sizes, self.datasets):
                start_idx = 0 if dataset_idx == -1 else \
                    self.cumulative_sizes[dataset_idx]
                end_idx = self.cumulative_sizes[dataset_idx + 1]

                results_per_dataset = results[start_idx:end_idx]
                print_log(
                    f'\nEvaluateing {dataset.ann_file} with '
                    f'{len(results_per_dataset)} images now',
                    logger=logger)

                eval_results_per_dataset = dataset.evaluate(
                    results_per_dataset, logger=logger, **kwargs)
                dataset_idx += 1
                for k, v in eval_results_per_dataset.items():
                    total_eval_results.update({f'{dataset_idx}_{k}': v})

            return total_eval_results
        elif any([isinstance(ds, CocoDataset) for ds in self.datasets]):
            raise NotImplementedError(
                'Evaluating concatenated CocoDataset as a whole is not'
                ' supported! Please set "separate_eval=True"')
        elif len(set([type(ds) for ds in self.datasets])) != 1:
            raise NotImplementedError(
                'All the datasets should have same types')
        else:
            original_data_infos = self.datasets[0].data_infos
            self.datasets[0].data_infos = sum(
                [dataset.data_infos for dataset in self.datasets], [])
            eval_results = self.datasets[0].evaluate(
                results, logger=logger, **kwargs)
            self.datasets[0].data_infos = original_data_infos
            return eval_results


@DATASETS.register_module()
class RepeatDataset:
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        if hasattr(self.dataset, 'flag'):
            self.flag = np.tile(self.dataset.flag, times)

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def get_cat_ids(self, idx):
        """Get category ids of repeat dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.dataset.get_cat_ids(idx % self._ori_len)

    def __len__(self):
        """Length after repetition."""
        return self.times * self._ori_len


# Modified from https://github.com/facebookresearch/detectron2/blob/41d475b75a230221e21d9cac5d69655e3415e3a4/detectron2/data/samplers/distributed_sampler.py#L57 # noqa
@DATASETS.register_module()
class ClassBalancedDataset:
    """A wrapper of repeated dataset with repeat factor.

    Suitable for training on class imbalanced datasets like LVIS. Following
    the sampling strategy in the `paper <https://arxiv.org/abs/1908.03195>`_,
    in each epoch, an image may appear multiple times based on its
    "repeat factor".
    The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1]
    is defined by the fraction of images in the training set (without repeats)
    in which category c appears.
    The dataset needs to instantiate :func:`self.get_cat_ids` to support
    ClassBalancedDataset.

    The repeat factor is computed as followed.

    1. For each category c, compute the fraction # of images
       that contain it: :math:`f(c)`
    2. For each category c, compute the category-level repeat factor:
       :math:`r(c) = max(1, sqrt(t/f(c)))`
    3. For each image I, compute the image-level repeat factor:
       :math:`r(I) = max_{c in I} r(c)`

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be repeated.
        oversample_thr (float): frequency threshold below which data is
            repeated. For categories with ``f_c >= oversample_thr``, there is
            no oversampling. For categories with ``f_c < oversample_thr``, the
            degree of oversampling following the square-root inverse frequency
            heuristic above.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes will not be oversampled. Otherwise, they will be categorized
            as the pure background class and involved into the oversampling.
            Default: True.
    """

    def __init__(self, dataset, oversample_thr, filter_empty_gt=True):
        self.dataset = dataset
        self.oversample_thr = oversample_thr
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = dataset.CLASSES

        repeat_factors = self._get_repeat_factors(dataset, oversample_thr)
        repeat_indices = []
        for dataset_idx, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_idx] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

        flags = []
        if hasattr(self.dataset, 'flag'):
            for flag, repeat_factor in zip(self.dataset.flag, repeat_factors):
                flags.extend([flag] * int(math.ceil(repeat_factor)))
            assert len(flags) == len(repeat_indices)
        self.flag = np.asarray(flags, dtype=np.uint8)

    def _get_repeat_factors(self, dataset, repeat_thr):
        """Get repeat factor for each images in the dataset.

        Args:
            dataset (:obj:`CustomDataset`): The dataset
            repeat_thr (float): The threshold of frequency. If an image
                contains the categories whose frequency below the threshold,
                it would be repeated.

        Returns:
            list[float]: The repeat factors for each images in the dataset.
        """

        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        category_freq = defaultdict(int)
        num_images = len(dataset)
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) == 0 and not self.filter_empty_gt:
                cat_ids = set([len(self.CLASSES)])
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        category_repeat = {
            cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        repeat_factors = []
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) == 0 and not self.filter_empty_gt:
                cat_ids = set([len(self.CLASSES)])
            repeat_factor = 1
            if len(cat_ids) > 0:
                repeat_factor = max(
                    {category_repeat[cat_id]
                     for cat_id in cat_ids})
            repeat_factors.append(repeat_factor)

        return repeat_factors

    def __getitem__(self, idx):
        ori_index = self.repeat_indices[idx]
        return self.dataset[ori_index]

    def __len__(self):
        """Length after repetition."""
        return len(self.repeat_indices)


# P1-A: 在 ClassBalancedDataset 之外，对 target_class_id 的图像按文件名前缀
# (domain_prefixes) 做额外的源均衡采样。
@DATASETS.register_module()
class DomainBalancedDataset:
    """在 ClassBalancedDataset 之外，对 target_class_id 的图像按文件名前缀
    (domain_prefixes) 做额外的源均衡采样。

    Args:
        dataset: 已构造好的内层 dataset（典型为 ClassBalancedDataset）。
        target_class_id (int): 需要跨域均衡的目标类别（默认 MS=3）。
        domain_prefixes (tuple[str]): 文件名匹配前缀，按位置与 domain_extras
            对应。
        domain_extras (tuple[int]): 额外 repeat 倍数；1 表示不额外采样。
        filter_empty_gt (bool): 是否过滤空 GT（默认 True）。
    """

    def __init__(self,
                 dataset,
                 target_class_id=3,
                 domain_prefixes=('01-PAN', '02-PAN', 'OTHER'),
                 domain_extras=(1, 2, 2),
                 filter_empty_gt=True):
        assert len(domain_prefixes) == len(domain_extras), (
            f'len(domain_prefixes) ({len(domain_prefixes)}) must equal '
            f'len(domain_extras) ({len(domain_extras)})')
        self.dataset = dataset
        self.target_class_id = target_class_id
        self.domain_prefixes = tuple(domain_prefixes)
        self.domain_extras = tuple(domain_extras)
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = dataset.CLASSES

        # 拷贝内层 repeat_indices
        if hasattr(dataset, 'repeat_indices'):
            base_indices = list(dataset.repeat_indices)
        else:
            base_indices = list(range(len(dataset)))

        # 解析内层 dataset 上的 data_infos 与 get_cat_ids（ClassBalancedDataset
        # 直接持有 .dataset = 内层 dataset；AircraftDataset 等是叶子节点）。
        inner = dataset
        if hasattr(inner, 'dataset') and hasattr(inner.dataset, 'data_infos'):
            data_infos = getattr(inner.dataset, 'data_infos', None)
        else:
            data_infos = getattr(inner, 'data_infos', None)
        cat_ids_fn = getattr(inner, 'get_cat_ids', None)
        if cat_ids_fn is None and hasattr(inner, 'dataset'):
            cat_ids_fn = getattr(inner.dataset, 'get_cat_ids', None)

        expanded = []
        for source_idx in base_indices:
            factor = 1
            if cat_ids_fn is not None and data_infos is not None:
                try:
                    cat_ids = set(cat_ids_fn(source_idx))
                except Exception:
                    cat_ids = set()
                if self.target_class_id in cat_ids:
                    fn = data_infos[source_idx].get('file_name', '')
                    for d_idx, prefix in enumerate(self.domain_prefixes):
                        if fn.startswith(prefix):
                            factor = self.domain_extras[d_idx]
                            break
            expanded.extend([source_idx] * factor)
        self.repeat_indices = expanded

        # flag 处理：简化版用全零（与 brief 一致）；aspect-ratio grouped sampler
        # 退化为单一组，对单 GPU 训练无影响。
        if hasattr(dataset, 'flag') and len(getattr(dataset, 'flag', [])) == len(
                base_indices):
            base_flag = np.asarray(dataset.flag)
            self.flag = np.concatenate(
                [base_flag[i:i + 1]
                 for i, src in enumerate(base_indices)
                 for _ in range(expanded.count(src) // max(1, base_indices.count(src)))])
        else:
            self.flag = np.zeros(len(self.repeat_indices), dtype=np.uint8)

    def __len__(self):
        return len(self.repeat_indices)

    def __getitem__(self, idx):
        """取一个 sample。注入 self.dataset 进 results['dataset']，
        供下游 transform（如 RandCopyPaste）访问 data_infos / get_ann_info。
        """
        results = self.dataset[self.repeat_indices[idx]]
        if isinstance(results, dict) and 'dataset' not in results:
            results['dataset'] = self.dataset
        return results

    def get_cat_ids(self, idx):
        return self.dataset.get_cat_ids(self.repeat_indices[idx])


# 复制自 model_v4 分支的 commit 3f11e6b (feat: add deterministic source-balanced
# dataset wrapper)，用于在主分支上以固定的源采样权重混合多个数据源 (典型场景：
# 25 类主训集 + ShipRS mapped COCO，强化 HM/LQS/QHS(+MS) 4 个船舰类)。
@DATASETS.register_module()
class SourceBalancedDataset:
    """A deterministic fixed-ratio wrapper for multiple data sources.

    The wrapper constructs its complete per-epoch index schedule once.  It
    deliberately does not reshuffle that schedule: MMDetection's group
    samplers own epoch shuffling, while keeping this layer static makes the
    requested source ratio reproducible and auditable.

    Args:
        datasets (list[:obj:`Dataset`]): Source datasets with identical
            ``CLASSES`` metadata.
        source_weights (tuple[float]): Positive relative sampling weights for
            every source.
        epoch_length (int, optional): Number of samples in the static epoch.
            If omitted, use the smallest length that covers every source at
            least once at its normalized sampling weight.
        seed (int): Seed used only for deterministic per-source offsets.
    """

    def __init__(self,
                 datasets,
                 source_weights=(0.6, 0.4),
                 epoch_length=None,
                 seed=20260817):
        if not isinstance(datasets, (list, tuple)) or not datasets:
            raise ValueError('datasets must be a non-empty list or tuple')
        self.datasets = list(datasets)

        if not isinstance(source_weights, (list, tuple)) or \
                len(source_weights) != len(self.datasets):
            raise ValueError('source_weights must match the number of datasets')
        if any(not isinstance(weight, numbers.Real) or
               isinstance(weight, bool) or not math.isfinite(float(weight)) or
               float(weight) <= 0 for weight in source_weights):
            raise ValueError('source_weights must be finite positive numbers')
        total_weight = float(sum(source_weights))
        if not math.isfinite(total_weight) or total_weight <= 0:
            raise ValueError('source_weights must have a finite positive sum')
        self.source_weights = tuple(
            float(weight) / total_weight for weight in source_weights)

        if not isinstance(seed, numbers.Integral) or isinstance(seed, bool):
            raise ValueError('seed must be an integer')
        self.seed = int(seed)

        source_lengths = []
        for dataset in self.datasets:
            source_length = len(dataset)
            if source_length <= 0:
                raise ValueError('source datasets must be non-empty')
            source_lengths.append(source_length)
        self._source_lengths = tuple(source_lengths)

        if not all(hasattr(dataset, 'CLASSES') for dataset in self.datasets):
            raise ValueError('every source dataset must define CLASSES')
        self.CLASSES = self.datasets[0].CLASSES
        if any(dataset.CLASSES != self.CLASSES
               for dataset in self.datasets[1:]):
            raise ValueError('source datasets must have compatible CLASSES')

        if epoch_length is None:
            epoch_length = max(
                int(math.ceil(length / weight))
                for length, weight in zip(self._source_lengths,
                                          self.source_weights))
        elif not isinstance(epoch_length, numbers.Integral) or \
                isinstance(epoch_length, bool) or epoch_length <= 0:
            raise ValueError('epoch_length must be a positive integer')
        self.epoch_length = int(epoch_length)

        self.source_counts = self._allocate_source_counts()
        self._schedule = self._build_schedule()
        self.flag = self._build_flag()

    def _allocate_source_counts(self):
        """Allocate exact source counts with stable largest remainders."""
        raw_counts = [weight * self.epoch_length
                      for weight in self.source_weights]
        source_counts = [int(math.floor(count)) for count in raw_counts]
        remaining = self.epoch_length - sum(source_counts)
        order = sorted(
            range(len(self.datasets)),
            key=lambda source_id: (-(raw_counts[source_id] -
                                     source_counts[source_id]), source_id))
        for source_id in order[:remaining]:
            source_counts[source_id] += 1
        return tuple(source_counts)

    def _build_schedule(self):
        """Interleave the allocated source counts and cycle local indices."""
        random_state = random.Random(self.seed)
        offsets = [random_state.randrange(length)
                   for length in self._source_lengths]
        selected_counts = [0] * len(self.datasets)
        schedule = []
        for position in range(self.epoch_length):
            # The source with the greatest outstanding ideal quota is chosen.
            # ``-source_id`` makes exact ties deterministic.
            source_id = max(
                range(len(self.datasets)),
                key=lambda index: (
                    self.source_counts[index] * (position + 1) -
                    selected_counts[index] * self.epoch_length,
                    -index))
            local_index = (offsets[source_id] + selected_counts[source_id]) % \
                self._source_lengths[source_id]
            schedule.append((source_id, local_index))
            selected_counts[source_id] += 1
        if tuple(selected_counts) != self.source_counts:
            raise RuntimeError('source schedule does not match source_counts')
        return tuple(schedule)

    def _build_flag(self):
        """Construct an aspect-ratio group flag for every scheduled sample."""
        flags = []
        source_flags = []
        for dataset, source_length in zip(self.datasets, self._source_lengths):
            if hasattr(dataset, 'flag'):
                source_flag = np.asarray(dataset.flag, dtype=np.uint8)
                if source_flag.ndim != 1 or len(source_flag) != source_length:
                    raise ValueError('dataset flag must match dataset length')
            else:
                source_flag = np.zeros(source_length, dtype=np.uint8)
            source_flags.append(source_flag)
        for source_id, local_index in self._schedule:
            flags.append(source_flags[source_id][local_index])
        return np.asarray(flags, dtype=np.uint8)

    def __getitem__(self, idx):
        source_id, local_index = self._schedule[idx]
        return self.datasets[source_id][local_index]

    def __len__(self):
        return self.epoch_length

    def get_cat_ids(self, idx):
        """Get categories from the source sample selected by ``idx``."""
        source_id, local_index = self._schedule[idx]
        return self.datasets[source_id].get_cat_ids(local_index)
