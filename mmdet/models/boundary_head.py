# ------------------------------------------------------------------------------
# The boundary head below is derived from PIDNet (see the original header
# retained in git history). Only the component used by the OSTD detector is
# kept; see THIRD_PARTY_NOTICES.md.
# ------------------------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1


class boundary_head(nn.Module):
    """共享边界头: BN+ReLU -> 3x3 (保持 C 通道) -> BN+ReLU -> 1x1 单通道."""

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(boundary_head, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(
            inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(
                out, size=[height, width], mode='bilinear', align_corners=True)

        return out
