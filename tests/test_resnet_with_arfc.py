import torch
from mmdet.models import build_backbone


def test_resnet50_with_arfc_outputs():
    cfg = dict(
        type='DetectoRS_ResNet',
        depth=50, num_stages=4, out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True, style='pytorch',
        arfc_cfg=dict(type='ARFC', in_c=64, out_c=64, top_k=3))
    bb = build_backbone(cfg)
    bb.init_weights()  # 不要求 pretrained
    y = bb(torch.randn(1, 3, 128, 128))
    # output_img=True 时返回 [img, c2, c3, c4, c5]
    assert len(y) == 5, f"expect 5 outputs, got {len(y)}"
    shapes = [t.shape for t in y]
    assert shapes[0][1] == 3, "first must be image"


def test_resnet50_arfc_is_trainable_when_stem_frozen():
    """frozen_stages=1 时 conv1 冻结，但 ARFC 必须可学习."""
    cfg = dict(
        type='DetectoRS_ResNet',
        depth=50, num_stages=4, out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True, style='pytorch',
        arfc_cfg=dict(type='ARFC', in_c=64, out_c=64, top_k=3))
    bb = build_backbone(cfg)
    bb.init_weights()
    bb.train()
    # ARFC stem 的参数应当 requires_grad=True
    assert hasattr(bb, 'arfc_stem'), "backbone must have arfc_stem"
    n_trainable = sum(p.numel() for p in bb.arfc_stem.parameters()
                      if p.requires_grad)
    assert n_trainable > 0, "arfc_stem parameters should not all be frozen"