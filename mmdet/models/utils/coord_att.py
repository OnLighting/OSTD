import torch
import torch.nn as nn


class CoordAtt(nn.Module):
    """Coordinate Attention (CA).

    Reference: Hou et al., "Coordinate Attention for Efficient Mobile
    Network Design", CVPR 2021.

    Args:
        in_c (int): 输入通道数。
        reduction (int): 通道压缩比，默认 32。
    """

    def __init__(self, in_c, reduction=32):
        super().__init__()
        mid = max(8, in_c // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(in_c, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, in_c, 1, bias=False)
        self.conv_w = nn.Conv2d(mid, in_c, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        a_h = self.pool_h(x)                          # [B, C, H, 1]
        a_w = self.pool_w(x).permute(0, 1, 3, 2)     # [B, C, W, 1]
        a = torch.cat([a_h, a_w], dim=2)             # [B, C, H+W, 1]
        a = self.act(self.bn1(self.conv1(a)))
        s_h, s_w = torch.split(a, [H, W], dim=2)
        s_w = s_w.permute(0, 1, 3, 2)
        return x * torch.sigmoid(self.conv_h(s_h)) \
                 * torch.sigmoid(self.conv_w(s_w))
