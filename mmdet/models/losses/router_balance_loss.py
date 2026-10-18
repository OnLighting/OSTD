import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class RouterBalanceLoss(nn.Module):
    """DRFC 路由正则: KL 均衡项 + 路由熵项.

    对第 r 个 DRFC 模块:
        L_r = rho * D_KL(u_r || pi_r) + nu * [log N_r - H(pi_r)]
    其中 pi_r 是 mini-batch 与路由网格上 softmax 概率的平均 (专家分布),
    u_r = 1/N_r 为均匀参考分布, H 为 Shannon 熵。两项共同鼓励平均路由
    分布接近均匀, 同时允许不同区域使用不同的专家组合。

    Args:
        num_experts (int): 专家数量。
        alpha (float): 论文中的 rho, KL 均衡项权重, 默认 0.1。
        beta (float): 论文中的 nu, 熵正则权重, 默认 0.01。
    """

    def __init__(self, num_experts, alpha=0.1, beta=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.alpha = alpha
        self.beta = beta

    def forward(self, gate_logits):
        if gate_logits.numel() == 0:
            return gate_logits.sum() * 0.0
        # Compute probability/log terms in fp32 and keep probabilities away
        # from zero. Extreme router logits otherwise create very large
        # derivatives through log(avg_probs), which can poison all branches.
        probs = F.softmax(gate_logits.float(), dim=-1)
        avg_probs = probs.reshape(-1, probs.shape[-1]).mean(dim=0)
        avg_probs = avg_probs.clamp_min(torch.finfo(avg_probs.dtype).eps)
        avg_probs = avg_probs / avg_probs.sum()
        # 均匀目标按 logits 的实际专家数生成, 使 P0 的轻量 DRFC (N=3)
        # 与骨干的标准 DRFC (N=4) 共用同一个损失实例。
        num_experts = avg_probs.shape[0]
        target = torch.full_like(avg_probs, 1.0 / num_experts)
        balance_loss = target * (torch.log(target) - torch.log(avg_probs))
        # 路由熵正则：max-entropy = log(E)。偏差越大损失越大。
        ent = -(avg_probs * torch.log(avg_probs)).sum()
        ent_reg = self.beta * (math.log(num_experts) - ent)
        return self.alpha * balance_loss.sum() + ent_reg
