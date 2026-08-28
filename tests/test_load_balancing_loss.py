import torch
from mmdet.models.losses import LoadBalancingLoss


def test_load_balancing_uniform_logits_zero():
    """完全均匀分布的 logits 应产生接近 0 的损失."""
    loss_fn = LoadBalancingLoss(num_experts=4, alpha=1.0)
    # 完全均匀的 gate logits
    logits = torch.zeros(100, 4)
    loss = loss_fn(logits)
    assert loss.item() < 1e-5, f"uniform logits should give ~0 loss, got {loss.item()}"


def test_load_balancing_extreme_logits_positive():
    """极度不均衡 logits 应得 > 0 损失."""
    loss_fn = LoadBalancingLoss(num_experts=4, alpha=1.0)
    # 把所有概率集中在第一个专家
    logits = torch.cat([
        torch.full((10, 1), 100.0),
        torch.full((10, 1), -100.0),
        torch.full((10, 1), -100.0),
        torch.full((10, 1), -100.0),
    ], dim=1)
    loss = loss_fn(logits)
    assert loss.item() > 0.01, f"extreme imbalance should give positive loss, got {loss.item()}"


def test_load_balancing_alpha_scales():
    loss_fn_small = LoadBalancingLoss(num_experts=4, alpha=0.01)
    loss_fn_big = LoadBalancingLoss(num_experts=4, alpha=1.0)
    logits = torch.randn(50, 4) * 5
    ls = loss_fn_small(logits).item()
    lb = loss_fn_big(logits).item()
    assert lb > ls, f"alpha=1.0 ({lb}) should exceed alpha=0.01 ({ls})"
