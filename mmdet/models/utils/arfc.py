import torch
import torch.nn as nn

from .arfc_parts import Expert, GridRouter, LCE


class ARFC(nn.Module):
    """Adaptive Receptive Field Convolution.

    由方案 `ARFC_Improvement_Scheme_Summary.md` 第 3.1 节定义：
    MFE (Top-k 加权的多尺度专家组) + LCE (1xK + Kx1 长距离共享专家)。

    Args:
        in_c (int): 输入通道数。
        out_c (int, optional): 输出通道数，默认等于 in_c。
        expert_channels (tuple): 每个专家的输出通道。
        expert_kernels (tuple): 每个专家的卷积核大小。
        top_k (int): 每次激活的专家数。
        lce_kernel (int): LCE 条状卷积核大小。
        lightweight (bool): 若 True，使用 3 专家版本 (核 5/7/9)。
    """

    def __init__(self,
                 in_c,
                 out_c=None,
                 expert_channels=(128, 96, 64, 48),
                 expert_kernels=(5, 7, 9, 11),
                 top_k=3,
                 lce_kernel=11,
                 lightweight=False):
        super().__init__()
        out_c = out_c if out_c is not None else in_c

        if lightweight:
            expert_channels = (96, 64, 48)
            expert_kernels = (5, 7, 9)
            lce_kernel = 11

        self.experts = nn.ModuleList([
            Expert(in_c, c, k)
            for c, k in zip(expert_channels, expert_kernels)
        ])
        self.router = GridRouter(len(self.experts), in_c=in_c, top_k=top_k)
        self.lce = LCE(in_c, out_c, kernel=lce_kernel)
        # NOTE: 各专家 out_c 不同 (如 (128,96,64,48)), 无法直接 stack。
        # 为每个专家补一个 1x1 投影到 out_c 的"对齐卷积",使它们可以
        # stack 成 [B, E, out_c, H, W] 再 gather + 加权求和。
        self.align = nn.ModuleList([
            nn.Conv2d(c, out_c, 1, bias=False)
            for c in expert_channels
        ])
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
        self.shortcut = (nn.Identity() if in_c == out_c
                         else nn.Sequential(
                             nn.Conv2d(in_c, out_c, 1, bias=False),
                             nn.BatchNorm2d(out_c)))
        # 路由器最近一次的 logits，供均衡损失调用
        self._last_router_logits = None

    def forward(self, x):
        logits, idx, w = self.router(x)
        self._last_router_logits = logits

        # 对齐到统一通道 out_c 后再 stack, 避免各专家通道数不同导致的报错。
        outs = [algn(exp(x)) for exp, algn in zip(self.experts, self.align)]
        feat_stack = torch.stack(outs, dim=1)            # [B, E, out_c, H, W]
        B = x.size(0)
        E_i = idx.size(1)
        gather_idx = idx.view(B, E_i, 1, 1, 1).expand(
            -1, -1, feat_stack.size(2), feat_stack.size(3), feat_stack.size(4))
        sel = torch.gather(feat_stack, 1, gather_idx)     # [B, K, out_c, H, W]
        w_expand = w.view(B, E_i, 1, 1, 1)
        mfe_feat = (sel * w_expand).sum(dim=1)            # [B, out_c, H, W]
        lce_feat = self.lce(x)
        out = self.act(self.bn(mfe_feat + lce_feat) + self.shortcut(x))
        return out
