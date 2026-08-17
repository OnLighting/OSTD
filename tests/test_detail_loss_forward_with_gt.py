import torch
from mmdet.models.losses import DetailAggregateLoss


def test_forward_with_gt_basic_shape():
    """正常输入应该返回一个标量 loss."""
    loss_fn = DetailAggregateLoss()
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    gt = (torch.randn(2, 1, 32, 32) > 0).float()
    loss = loss_fn.forward_with_gt(logits, gt)
    assert loss.dim() == 0, f"expected scalar, got shape {loss.shape}"
    assert loss.requires_grad, "loss must be differentiable"


def test_forward_with_gt_shape_mismatch():
    """logits 和 gt 空间尺寸不同应自动 resize."""
    loss_fn = DetailAggregateLoss()
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    gt = (torch.randn(2, 1, 32, 32) > 0).float()
    loss = loss_fn.forward_with_gt(logits, gt)
    assert loss.dim() == 0


def test_forward_with_gt_gradient_flows():
    """logits 必须有梯度."""
    loss_fn = DetailAggregateLoss()
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    gt = (torch.randn(2, 1, 32, 32) > 0).float()
    loss = loss_fn.forward_with_gt(logits, gt)
    loss.backward()
    assert logits.grad is not None