import torch
import torch.nn as nn

from .coord_att import CoordAtt


class Expert(nn.Module):
    """单 ARFC 专家：深度可分离大核 conv + 1x1 投影 + CA.

    Args:
        in_c (int): 输入通道数。
        out_c (int): 输出通道数（来自 GridRouter 各专家不同的通道分配）。
        kernel_size (int): 深度卷积核大小，必须为奇数。
    """

    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        pad = kernel_size // 2
        self.dw = nn.Conv2d(in_c, in_c, kernel_size, padding=pad,
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
    """

    def __init__(self, in_c, out_c=None, kernel=11):
        super().__init__()
        out_c = out_c if out_c is not None else in_c
        self.dw_h = nn.Conv2d(in_c, in_c, (1, kernel),
                              padding=(0, kernel // 2),
                              groups=in_c, bias=False)
        self.dw_w = nn.Conv2d(in_c, in_c, (kernel, 1),
                              padding=(kernel // 2, 0),
                              groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw_w(self.dw_h(x))
        return self.act(self.bn(self.pw(x)))


class GridRouter(nn.Module):
    """全图 Top-k 专家路由器（论文 grid 版本的工程简化）.

    Args:
        num_experts (int): 专家数量。
        top_k (int): 每次激活的专家数。
        grid_size (int): 保留接口以备后续切块；当前实现忽略。
    """

    def __init__(self, num_experts, in_c=256, top_k=3, grid_size=8):
        super().__init__()
        self.K = top_k
        self.G = grid_size
        self.E = num_experts
        # GAP collapses spatial dims but keeps channels -> Linear in_features
        # equals input channel count. 原先用 LazyLinear 是为了不显式传 in_c，
        # 但 LazyLinear 在首次 forward 前权重为未初始化 meta 参数，会导致
        # mmcv BaseModule._dump_init_info 读取 .shape 时崩溃。这里改为显式
        # 传入 in_c 的普通 Linear，行为等价且无 lazy 参数。
        self.score_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_c, num_experts, bias=False),
        )

    def forward(self, x):
        # x: [B, C, H, W]
        logits = self.score_mlp(x)                     # [B, E]
        topk_val, topk_idx = logits.topk(self.K, dim=-1)  # [B, K], [B, K]
        weights = torch.softmax(topk_val, dim=-1)          # [B, K]
        return logits, topk_idx, weights
