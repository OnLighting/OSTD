import torch
from mmdet.models import build_detector


def test_extract_feat_returns_p0_refined():
    cfg = dict(
        type='CascadeRCNN_BAF',
        backbone=dict(type='DetectoRS_ResNet', depth=50, num_stages=4,
                      out_indices=(0, 1, 2, 3), frozen_stages=1,
                      norm_cfg=dict(type='BN', requires_grad=True),
                      conv_cfg=dict(type='ConvAWS'),
                      sac=dict(type='SAC', use_deform=True),
                      stage_with_sac=(False, True, True, True),
                      output_img=True, style='pytorch',
                      arfc_cfg=dict(type='ARFC', in_c=64, out_c=64, top_k=3)),
        neck=dict(type='RFP',
                  in_channels=[256, 512, 1024, 2048],
                  out_channels=256, num_outs=5, rfp_steps=2,
                  aspp_out_channels=64, aspp_dilations=(1, 3, 6, 1),
                  rfp_backbone=dict(rfp_inplanes=256,
                                    type='DetectoRS_ResNet', depth=50,
                                    num_stages=4, out_indices=(0, 1, 2, 3),
                                    frozen_stages=1,
                                    norm_cfg=dict(type='BN', requires_grad=True),
                                    norm_eval=True,
                                    conv_cfg=dict(type='ConvAWS'),
                                    sac=dict(type='SAC', use_deform=True),
                                    stage_with_sac=(False, True, True, True),
                                    pretrained='torchvision://resnet50',
                                    style='pytorch')),
        rpn_head=None,  # 跳过 RPN 以便快速 forward
        roi_head=None,
        pretrained=None,
        train_cfg=None, test_cfg=None)
    det = build_detector(cfg)
    img = torch.randn(1, 3, 128, 128)
    x, p0_refined = det.extract_feat(img, None)
    assert len(x) == 5, f"expect 5 levels, got {len(x)}"
    assert p0_refined.shape[1] == 256, "p0_refined should be 256-channel"
    assert p0_refined.shape[-2:] == x[0].shape[-2:], \
        "p0_refined spatial shape must match x[0]"