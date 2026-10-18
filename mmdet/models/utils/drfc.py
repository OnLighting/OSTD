"""Dynamic Receptive Field Convolution (DRFC).

DRFC routes and combines responses from experts with different receptive
fields according to regional image content, and supplements the routed
response with a shared long-range strip-context branch plus a shortcut from
the input. See ``docs/MODULE_MAPPING.md`` for the paper-to-code mapping.
"""
import torch
import torch.nn as nn

from .drfc_parts import Expert, GridRouter, LCE


class DRFC(nn.Module):
    """Dynamic Receptive Field Convolution.

    Multi-Scale Expert Bank (MSEB) + grid-aware Top-k router (GTR) +
    Long-range strip Context Expert (LCE), fused by Expert-Context Residual
    Fusion (ECRF).

    Args:
        in_c (int): 输入通道数。
        out_c (int, optional): 输出通道数，默认等于 in_c。
        expert_channels (tuple): 每个专家的输出通道。
        expert_kernels (tuple): 每个专家的卷积核大小。
        top_k (int): 每个网格单元激活的专家数。
        lce_kernel (int): LCE 条状卷积核大小。
        lightweight (bool): 若 True，使用 3 专家版本 (核 5/7/9)，用于 P0。
        stride (int): 空间下采样步长。
    """

    def __init__(self,
                 in_c,
                 out_c=None,
                 expert_channels=(128, 96, 64, 48),
                 expert_kernels=(5, 7, 9, 11),
                 top_k=3,
                 lce_kernel=11,
                 lightweight=False,
                 stride=1):
        super().__init__()
        out_c = out_c if out_c is not None else in_c

        if lightweight:
            expert_channels = (96, 64, 48)
            expert_kernels = (5, 7, 9)
            lce_kernel = 11

        self.experts = nn.ModuleList([
            Expert(in_c, c, k, stride=stride)
            for c, k in zip(expert_channels, expert_kernels)
        ])
        self.router = GridRouter(len(self.experts), in_c=in_c, top_k=top_k)
        self.lce = LCE(in_c, out_c, kernel=lce_kernel, stride=stride)
        # 各专家 out_c 不同 (如 (128,96,64,48)), 无法直接 stack。
        # 为每个专家补一个 1x1 对齐卷积投影到 out_c, 使它们可以
        # stack 成 [B, E, out_c, H, W] 再 gather + 加权求和。
        self.align = nn.ModuleList([
            nn.Conv2d(c, out_c, 1, bias=False)
            for c in expert_channels
        ])
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
        self.shortcut = (nn.Identity() if in_c == out_c and stride == 1
                         else nn.Sequential(
                             nn.Conv2d(
                                 in_c, out_c, 1, stride=stride, bias=False),
                             nn.BatchNorm2d(out_c)))
        # 路由器最近一次的 logits, 供 RouterBalanceLoss 调用
        self._last_router_logits = None

    def forward(self, x):
        logits, idx, w = self.router(x)
        self._last_router_logits = logits

        # 先算完所有专家, 再对齐通道并 stack, 供路由器 gather。
        outs = [algn(exp(x)) for exp, algn in zip(self.experts, self.align)]
        feat_stack = torch.stack(outs, dim=1)            # [B, E, out_c, H, W]
        _, _, C, H, W = feat_stack.shape
        # 网格决策按最近邻扩展回特征分辨率。
        idx = idx.permute(0, 3, 1, 2).float()
        idx = nn.functional.interpolate(idx, size=(H, W), mode='nearest').long()
        w = w.permute(0, 3, 1, 2)
        w = nn.functional.interpolate(w, size=(H, W), mode='nearest')
        gather_idx = idx.unsqueeze(2).expand(-1, -1, C, -1, -1)
        sel = torch.gather(feat_stack, 1, gather_idx)     # [B, K, out_c, H, W]
        w_expand = w.unsqueeze(2)
        mseb_feat = (sel * w_expand).sum(dim=1)           # [B, out_c, H, W]
        lce_feat = self.lce(x)
        out = self.act(self.bn(mseb_feat + lce_feat) + self.shortcut(x))
        return out
