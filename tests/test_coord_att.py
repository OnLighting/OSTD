import torch
from mmdet.models.utils import CoordAtt

def test_coord_att_output_shape():
    ca = CoordAtt(in_c=64, reduction=32)
    x = torch.randn(2, 64, 32, 32)
    y = ca(x)
    assert y.shape == x.shape, f"expect {x.shape}, got {y.shape}"

def test_coord_att_non_zero_output():
    ca = CoordAtt(in_c=64, reduction=32)
    x = torch.randn(2, 64, 32, 32)
    y = ca(x)
    assert (y != 0).any().item(), "CA output should not be all-zero"

def test_coord_att_eval_mode():
    ca = CoordAtt(in_c=64, reduction=32).eval()
    x = torch.randn(1, 64, 16, 16)
    with torch.no_grad():
        y = ca(x)
    assert y.shape == x.shape
