import torch
from mmdet.models.utils import ARFC


def test_arfc_full_output_shape():
    net = ARFC(in_c=64, out_c=64, top_k=3)
    x = torch.randn(2, 64, 32, 32)
    y = net(x)
    assert y.shape == (2, 64, 32, 32), f"got {y.shape}"


def test_arfc_channel_mismatch_handled():
    """in_c != out_c 时短路 1x1 投影不应报错."""
    net = ARFC(in_c=64, out_c=128, top_k=3)
    x = torch.randn(2, 64, 16, 16)
    y = net(x)
    assert y.shape == (2, 128, 16, 16)


def test_arfc_lightweight_smaller():
    """lightweight 模式应明显小于 full 模式."""
    full = ARFC(in_c=256, out_c=256)
    lite = ARFC(in_c=256, out_c=256, lightweight=True)
    # 物化 LazyLinear 路由器参数(首次 forward 才分配),否则 numel() 报错
    _ = full(torch.randn(1, 256, 8, 8))
    _ = lite(torch.randn(1, 256, 8, 8))
    full_p = sum(p.numel() for p in full.parameters())
    lite_p = sum(p.numel() for p in lite.parameters())
    assert lite_p < 0.7 * full_p, \
        f"lightweight should be <70% of full, got {lite_p}/{full_p}"


def test_arfc_records_router_logits():
    net = ARFC(in_c=64, out_c=64, top_k=3)
    x = torch.randn(2, 64, 16, 16)
    _ = net(x)
    assert hasattr(net, '_last_router_logits'), \
        "ARFC must record _last_router_logits for balance loss"
    assert net._last_router_logits.shape == (2, 4)


def test_arfc_gradient_flows():
    """确认 ARFC 内可学习参数能收到梯度."""
    net = ARFC(in_c=64, out_c=64, top_k=3)
    x = torch.randn(2, 64, 16, 16, requires_grad=True)
    y = net(x)
    y.sum().backward()
    assert x.grad is not None and (x.grad != 0).any()
