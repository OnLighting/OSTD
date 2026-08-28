import torch
from mmdet.models.utils import Expert, LCE
from mmdet.models.utils import GridRouter


def test_expert_output_shape():
    exp = Expert(in_c=64, out_c=48, kernel_size=5)
    x = torch.randn(2, 64, 32, 32)
    y = exp(x)
    assert y.shape == (2, 48, 32, 32), f"got {y.shape}"


def test_expert_kernel_sizes():
    """测试 4 个方案指定的核大小都能 forward."""
    for k in (5, 7, 9, 11):
        exp = Expert(in_c=64, out_c=48, kernel_size=k)
        x = torch.randn(1, 64, 16, 16)
        y = exp(x)
        assert y.shape == (1, 48, 16, 16)


def test_lce_shape_with_out_c():
    lce = LCE(in_c=64, out_c=128, kernel=11)
    x = torch.randn(2, 64, 32, 32)
    y = lce(x)
    assert y.shape == (2, 128, 32, 32)


def test_lce_shape_default_out_c():
    lce = LCE(in_c=64)  # out_c default = in_c
    x = torch.randn(2, 64, 32, 32)
    y = lce(x)
    assert y.shape == (2, 64, 32, 32)


def test_lce_non_degenerate():
    """1x11 + 11x1 条状卷积不应退化为恒等."""
    lce = LCE(in_c=64)
    x = torch.randn(1, 64, 16, 16)
    y = lce(x)
    assert not torch.allclose(x, y), "LCE should transform input"


def test_grid_router_logit_shape():
    r = GridRouter(num_experts=4, top_k=3)
    x = torch.randn(2, 64, 32, 32)
    logits, idx, w = r(x)
    assert logits.shape == (2, 4), f"logits: {logits.shape}"
    assert idx.shape == (2, 3), f"idx: {idx.shape}"
    assert w.shape == (2, 3), f"w: {w.shape}"


def test_grid_router_topk_indices_in_range():
    r = GridRouter(num_experts=4, top_k=3)
    x = torch.randn(2, 64, 16, 16)
    logits, idx, _ = r(x)
    assert (idx >= 0).all() and (idx < 4).all(), \
        "top-k indices must be within [0, num_experts)"


def test_grid_router_weights_sum_to_one():
    r = GridRouter(num_experts=4, top_k=3)
    x = torch.randn(2, 64, 16, 16)
    _, _, w = r(x)
    sums = w.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        f"softmax weights should sum to 1, got {sums}"
