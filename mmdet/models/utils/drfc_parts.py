import torch
import torch.nn as nn

from .coord_att import CoordAtt


class Expert(nn.Module):
    """MSEB 单个专家：深度可分离大核 conv + 1x1 投影 + 坐标注意力.

    Args:
        in_c (int): 输入通道数。
        out_c (int): 输出通道数（各专家通道数可以不同）。
        kernel_size (int): 深度卷积核大小，必须为奇数。
        stride (int): 空间下采样步长。
    """

    def __init__(self, in_c, out_c, kernel_size, stride=1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        pad = kernel_size // 2
        self.dw = nn.Conv2d(in_c, in_c, kernel_size, stride=stride,
                            padding=pad,
                            groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
        self.ca = CoordAtt(out_c)

    def forward(self, x):
        x = self.pw(self.dw(x))
        x = self.act(self.bn(x))
        return self.ca(x)


class LCE(nn.Module):
    """长距离共享专家：1xK + Kx1 条状深度卷积.

    Args:
        in_c (int): 输入通道数。
        out_c (int, optional): 输出通道数，默认等于 in_c。
        kernel (int): 条状核大小，默认 11。
        stride (int): 空间下采样步长。
    """

    def __init__(self, in_c, out_c=None, kernel=11, stride=1):
        super().__init__()
        out_c = out_c if out_c is not None else in_c
        self.dw_h = nn.Conv2d(in_c, in_c, (1, kernel),
                              stride=(1, stride),
                              padding=(0, kernel // 2),
                              groups=in_c, bias=False)
        self.dw_w = nn.Conv2d(in_c, in_c, (kernel, 1),
                              stride=(stride, 1),
                              padding=(kernel // 2, 0),
                              groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw_w(self.dw_h(x))
        return self.act(self.bn(self.pw(x)))


class GridRouter(nn.Module):
    """网格感知 Top-k 专家路由器 (GTR).

    Args:
        num_experts (int): 专家数量。
        top_k (int): 每个网格单元激活的专家数。
        grid_size (int): 每个空间维度的最大网格数。
    """

    def __init__(self, num_experts, in_c=256, top_k=3, grid_size=8):
        super().__init__()
        if not 0 < top_k <= num_experts:
            raise ValueError(
                f'top_k must be in [1, {num_experts}], got {top_k}')
        if grid_size < 1:
            raise ValueError(f'grid_size must be positive, got {grid_size}')
        self.K = top_k
        self.G = grid_size
        self.E = num_experts
        # Routing only needs relative channel responses. GroupNorm prevents
        # high-amplitude backbone features from saturating the router while
        # remaining stable for the small per-GPU batches used by detection.
        self.norm = nn.GroupNorm(1, in_c)
        self.score = nn.Conv2d(in_c, num_experts, 1, bias=False)
        nn.init.normal_(self.score.weight, mean=0.0, std=0.01)

    def forward(self, x):
        grid_h = min(self.G, x.shape[-2])
        grid_w = min(self.G, x.shape[-1])
        grid = nn.functional.adaptive_avg_pool2d(
            self.norm(x), (grid_h, grid_w))
        logits = self.score(grid).permute(0, 2, 3, 1)       # [B, Gh, Gw, E]
        topk_val, topk_idx = logits.topk(self.K, dim=-1)    # [B, Gh, Gw, K]
        # Always evaluate the routing softmax in fp32. This also makes DRFC
        # safe when mixed precision is enabled later.
        weights = torch.softmax(topk_val.float(), dim=-1).to(x.dtype)
        return logits, topk_idx, weights
