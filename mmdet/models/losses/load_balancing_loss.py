import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class LoadBalancingLoss(nn.Module):
    """专家路由均衡损失 (Switch Transformer 风格) + 路由熵正则.

    L_balance = alpha * sum_i f_i * log(f_i / P_i)
    L_entropy = beta  * (log(E) - H(P_bar))
    return L_balance + L_entropy

    其中 f_i = 均匀目标频率 = 1/E，P_bar_i = 专家 i 在 batch 内的平均
    softmax 概率，H(P_bar) = -sum_i P_bar_i * log P_bar_i。
    熵正则项鼓励路由接近均匀分布（最大熵 = log(E)），防止坍缩到单一专家。

    Args:
        num_experts (int): 专家数量。
        alpha (float): Switch 均衡项权重，默认 0.1（原 0.01）。
        beta (float): 熵正则权重，默认 0.01。
    """

    def __init__(self, num_experts, alpha=0.1, beta=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.alpha = alpha
        self.beta = beta
        # 缓存均匀目标（旧版保留以便向后兼容，但默认走动态 target）
        self.register_buffer(
            '_target',
            torch.full((num_experts,), 1.0 / num_experts))

    def forward(self, gate_logits):
        if gate_logits.numel() == 0:
            return gate_logits.sum() * 0.0
        probs = F.softmax(gate_logits, dim=-1)            # [B, E]
        avg_probs = probs.mean(dim=0)                     # [E]
        # 按实际专家数动态生成均匀目标，避免 num_experts 与 ARFC 实际
        # 专家数 (lightweight=3, 标准=4) 不一致导致的形状不匹配。
        num_experts = avg_probs.shape[0]
        target = torch.full_like(avg_probs, 1.0 / num_experts)
        # 避免 log(0)
        balance_loss = target * (torch.log(target + 1e-9)
                                 - torch.log(avg_probs + 1e-9))
        # 路由熵正则：max-entropy = log(E)。偏差越大损失越大。
        ent = -(avg_probs * torch.log(avg_probs + 1e-9)).sum()
        ent_reg = self.beta * (math.log(num_experts) - ent)
        return self.alpha * balance_loss.sum() + ent_reg
