"""Discrete 90/180/270-degree random rotation fallback for mmdet versions
without RandomRotate.

Continuous-angle rotation requires affine matrix + interpolation; discrete
rotation (np.rot90) is simple, exact, and the augmentation effect for FSC
(few-shot class) is similar enough for our purposes.
"""
import random

import numpy as np
from mmdet.core import PolygonMasks

from .builder import PIPELINES


@PIPELINES.register_module()
class RandRotate:
    """Randomly rotate the image and bboxes by 90/180/270 degrees.

    Args:
        prob (float): probability of applying the transform.
    """

    def __init__(self, prob=0.5):
        self.prob = float(prob)

    def _rot90(self, img, bboxes, k):
        """np.rot90(img, k) rotates counter-clockwise k times.

        For bboxes in xyxy or xywh: use the simpler path of converting to
        xyxy, swapping and negating as needed, then back to xywh.

        Our model uses xywh (COCO format). Rotation steps:
          k=1 (CCW 90): w,h swap; x_new = h - y_old - w_old; y_new = x_old
          k=2 (180): x_new = w - x_old - w_old; y_new = h - y_old - h_old
          k=3 (CW 90): w,h swap; x_new = y_old; y_new = w - x_old - w_old
        """
        H, W = img.shape[:2]
        if bboxes.shape[0] == 0:
            return img, bboxes
        x, y, bw, bh = (bboxes[:, i] for i in range(4))
        if k == 0:
            return img, bboxes
        if k == 1:
            new_w, new_h = H, W
            nx = (H - y - bh).clip(0, new_w)
            ny = x.clip(0, new_h)
            nbw = bh
            nbh = bw
        elif k == 2:
            nx = (W - x - bw).clip(0, W)
            ny = (H - y - bh).clip(0, H)
            nbw = bw
            nbh = bh
        else:  # k == 3
            new_w, new_h = H, W
            nx = y.clip(0, new_w)
            ny = (W - x - bw).clip(0, new_h)
            nbw = bh
            nbh = bw
        out = np.stack([nx, ny, nbw, nbh], axis=1)
        return out

    def __call__(self, results):
        if random.random() > self.prob:
            return results
        k = random.randint(0, 3)
        if k == 0:
            return results
        img = results['img']
        img = np.rot90(img, k=k).copy()
        H_new, W_new = img.shape[:2]
        bboxes = np.asarray(results['gt_bboxes'], dtype=np.float32).reshape(-1, 4)
        new_bboxes = self._rot90(img, bboxes, k)
        # clip to new image bounds
        new_bboxes[:, 0] = new_bboxes[:, 0].clip(0, W_new)
        new_bboxes[:, 1] = new_bboxes[:, 1].clip(0, H_new)
        new_bboxes[:, 2] = new_bboxes[:, 2].clip(0, W_new)
        new_bboxes[:, 3] = new_bboxes[:, 3].clip(0, H_new)
        # filter degenerate boxes
        keep = (new_bboxes[:, 2] > 0) & (new_bboxes[:, 3] > 0)
        new_bboxes = new_bboxes[keep]
        new_labels = np.asarray(results['gt_labels'], dtype=np.int64)[keep]
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['gt_bboxes'] = new_bboxes
        results['gt_labels'] = new_labels
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'
