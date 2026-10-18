"""Laplacian boundary supervision (training-only auxiliary loss).

Builds a detached binary boundary target from the grayscale input image with a
fixed 3x3 Laplacian kernel at three strides, then supervises the predicted
boundary logits with class-balanced binary cross-entropy.
"""
from .boundary_bce import weighted_bce
import torch
from torch import nn
from torch.nn import functional as F


class LaplacianBoundaryLoss(nn.Module):
    """三尺度拉普拉斯边界真值 + 类别均衡 BCE.

    目标由输入图像灰度版本卷积固定 3x3 拉普拉斯核 (中心 8, 周围 -1)
    得到, 步长 r in {1, 2, 4}, clamp 到非负; stride 1 在 H x W 上按阈值
    0.1 二值化, stride 2/4 的响应二值化后最近邻上采样到 H x W; 三尺度按
    0.6/0.3/0.1 融合并再次二值化, 得到单通道 $T_{bd}$。目标不含梯度,
    描述图像轮廓 (含背景边缘), 是结构训练信号而非目标分割标签。

    Args:
        threshold (float): 二值化阈值, 默认 0.1。
    """

    def __init__(self, threshold=0.1, *args, **kwargs):
        super(LaplacianBoundaryLoss, self).__init__()
        self.threshold = threshold
        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False)
        # 三尺度融合系数 (stride 1, 2, 4) = (0.6, 0.3, 0.1)。
        self.fuse_kernel = torch.nn.Parameter(torch.tensor(
            [[6. / 10], [3. / 10], [1. / 10]], dtype=torch.float32).reshape(
                1, 3, 1, 1).requires_grad_())

    def forward(self, boundary_logits, img_gray):
        """预测边界 logits 与灰度输入图 (B, 1, H, W) 的边界损失."""
        size = img_gray.shape[2:]
        kernel = self.laplacian_kernel.to(img_gray.device)

        def binarize(x):
            return torch.where(x > self.threshold, torch.ones_like(x),
                               torch.zeros_like(x))

        target_x1 = binarize(
            F.conv2d(img_gray, kernel, padding=1).clamp(min=0))
        target_x2 = binarize(
            F.interpolate(
                F.conv2d(img_gray, kernel, stride=2, padding=1).clamp(min=0),
                size, mode='nearest'))
        target_x4 = binarize(
            F.interpolate(
                F.conv2d(img_gray, kernel, stride=4, padding=1).clamp(min=0),
                size, mode='nearest'))

        pyramid = torch.stack((target_x1, target_x2, target_x4), dim=1)
        boundary_target = binarize(
            F.conv2d(pyramid.squeeze(2), self.fuse_kernel)).detach()

        if boundary_logits.shape[-1] != size[-1]:
            boundary_logits = F.interpolate(
                boundary_logits, size, mode='bilinear', align_corners=True)
        return weighted_bce(boundary_logits, boundary_target)
