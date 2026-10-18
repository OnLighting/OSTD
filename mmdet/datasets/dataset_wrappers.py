import math

import numpy as np
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as ``torch.utils.data.dataset.ConcatDataset``,
    but add ``get_cat_ids`` for dataset wrapper.
    """

    def __init__(self, datasets, separate_eval=True):
        super(ConcatDataset, self).__init__(datasets)
        self.separate_eval = separate_eval
        self.CLASSES = datasets[0].CLASSES

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image indexed by ``idx``.
        """
        if self.separate_eval:
            return self.datasets[idx].get_cat_ids(0)
        else:
            cat_ids = []
            for dataset in self.datasets:
                cat_ids.extend(dataset.get_cat_ids(idx))
            return cat_ids


@DATASETS.register_module()
class RepeatDataset:
    """A wrapper of repeated dataset.

    The length of repeated dataset will be ``times`` times of original dataset.
    This is useful when the data loading time is long but the dataset is small.
    Repeating it reduces the data loading time at the cost of requiring more
    GPU memory to store the augmented data.

    Args:
        dataset (Dataset): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        super(RepeatDataset, self).__init__()
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        # `infinite_sampler` would call self.dataset.sampler, which raises
        # NotImplementedError. Use a separate infinite_sampler for RepeatDataset.
        self._sampler = None

    def __getitem__(self, idx):
        """Get item from base dataset."""
        return self.dataset[idx % len(self.dataset)]

    def __len__(self):
        """Length after repetition."""
        return len(self.dataset) * self.times


@DATASETS.register_module()
class ClassBalancedDataset:
    """A wrapper of repeated dataset with repeat factor.

    Suitable for training on class imbalanced datasets like LVIS. Following
    the repeat factor calculation in repeat factor paper, repeat factor for
    each sample is based on the rarest category containing the sample.

    Args:
        dataset (Dataset): The dataset to be repeated.
        oversample_thr (float): Frequency threshold below which a sample is
            repeated. For the category c, if ``f_c < over_sample_thr``,
            ``repeat_factor[c] = math.ceil(over_sample_thr / f_c)``; else,
            ``repeat_factor[c] = 1``. Default is 1e-3.
    """

    def __init__(self, dataset, oversample_thr=1e-3):
        super(ClassBalancedDataset, self).__init__()
        self.dataset = dataset
        self.oversample_thr = oversample_thr
        self.CLASSES = dataset.CLASSES

        repeat_factors = self._get_repeat_factors(dataset, oversample_thr)
        repeat_indices = []
        for dataset_idx, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_idx] * repeat_factor)
        self.repeat_indices = repeat_indices

        flags = []
        if hasattr(dataset, 'flag'):
            for dataset_idx, repeat_factor in enumerate(repeat_factors):
                flags.extend([dataset.flag[dataset_idx]] * repeat_factor)
            # flags and repeat_indices should have the same length
            assert len(flags) == len(repeat_indices)
        self.flag = np.asarray(flags, dtype=np.uint8)

    def _get_repeat_factors(self, dataset, oversample_thr):
        # 1. For each category c, compute the fraction # images that contain it:
        # f_c = (# images containing c) / (total # images)
        # 2. For each category c, compute the repeat factor:
        # r_c = max(1, math.ceil(over_sample_thr / f_c))
        # 3. For each sample x, compute the max repeat factor of any category
        # in x's labels: r_x = max(r_c over c in labels[x])
        # 4. Repeat each sample x r_x times.

        # 1. For each category c, compute the fraction # images that contain it
        category_freq = defaultdict(int)
        num_images = len(dataset)
        for idx in range(num_images):
            cat_ids = set(dataset.get_cat_ids(idx))
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        # 2. For each category c, compute r_c
        for cat_id, freq in category_freq.items():
            category_freq[cat_id] = max(1., math.ceil(oversample_thr / freq))
        # 3. For each sample x, compute r_x
        repeat_factors = []
        for idx in range(num_images):
            cat_ids = set(dataset.get_cat_ids(idx))
            repeat_factor = max({category_freq[cat_id] for cat_id in cat_ids})
            repeat_factors.append(int(repeat_factor))
        return repeat_factors

    def __getitem__(self, idx):
        """Get item from base dataset."""
        return self.dataset[self.repeat_indices[idx]]

    def __len__(self):
        """Length after repetition."""
        return len(self.repeat_indices)