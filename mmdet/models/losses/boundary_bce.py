"""Class-balanced binary cross-entropy for the Laplacian boundary target.

Derived from https://github.com/HRNet/HRNet-Semantic-Segmentation; see
THIRD_PARTY_NOTICES.md.
"""
import torch.nn.functional as F


def weighted_bce(bd_pre, target):
    """像素级类别均衡 BCE (论文 Eq. boundary_pixel_weight).

    Args:
        bd_pre (Tensor): (N, H, W) 边界 logits。
        target (Tensor): 与 bd_pre 同形的二值边界目标。

    Returns:
        Tensor: 标量损失。
    """
    n, c, h, w = bd_pre.size()
    log_p = bd_pre.permute(0, 2, 3, 1).contiguous().view(1, -1)
    # 边界目标是标签, 即使构造时用过可学习融合核, BCE 也不能对其反向传播。
    target_t = target.detach().reshape(1, -1)

    pos_num = target_t.sum()
    neg_num = (1 - target_t).sum()
    sum_num = pos_num + neg_num
    weight = (target_t * neg_num + (1 - target_t) * pos_num) / sum_num.clamp_min(1)

    return F.binary_cross_entropy_with_logits(
        log_p, target_t, weight, reduction='mean')
