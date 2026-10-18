# OSTD on SODA-A HBB (paper Table III: AP 35.86).
# 800x800 training patches from official train originals; evaluation runs on
# all official test patches without merging (paper Section IV-A).
_base_ = [
    '../_base_/datasets/sodaa_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# NOTE (paper discrepancy): the paper states S0=512 with a 384@2/3 RPN
# sampler for SODA-A; the archived run that produced the reported numbers
# used base_sample_num=256 with the stock 256@1/2 sampler, kept here.
sbla = dict(
    enabled=True,
    mode='full',
    topk=9,
    gamma=0.5,
    alpha=0.5,
    beta=0.6,
    kappa=-0.06,
    phi=1.4,
    base_sample_num=256,
    min_pos_per_gt=1,
    schedule_start_epoch=0,
    schedule_end_epoch=11,
    candidate_mode='center',
    scores_are_logits=True)

model = dict(
    type='OSTDDetector',
    drfc_compatible_init=True,
    p0_residual_scale=0.1,
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='DetectoRS_ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        conv_cfg=dict(type='ConvAWS'),
        norm_eval=True,
        drfc_norm_eval=False,
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True,
        style='pytorch',
        drfc_cfg=dict(
            type='DRFC', in_c=3, out_c=64, top_k=3, stride=2),
        drfc_layer3_cfg=dict(
            type='DRFC', in_c=512, out_c=1024, top_k=3, stride=2),
    ),
    neck=dict(
        type='RFP',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            drfc_norm_eval=False,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            drfc_cfg=dict(
                type='DRFC', in_c=3, out_c=64, top_k=3, stride=2),
            drfc_layer3_cfg=dict(
                type='DRFC', in_c=512, out_c=1024, top_k=3, stride=2),
            pretrained='torchvision://resnet50',
            style='pytorch')),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='RFGenerator',
            fpn_layer='p2',
            fraction=0.5,
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=9,                    # SODA-A 9 classes
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=9,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=9,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]))
test_cfg = dict(
    rpn=dict(
        nms_pre=3000,
        max_per_img=3000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=3000))

# Paper protocol: 1 image per GPU (one GPU per queue worker), lr 0.005,
# 12 epochs on 37,980 training patches. Evaluation on all 30,974 test
# patches without merging predictions back to original images.
data = dict(samples_per_gpu=1,
            train=dict(type='SODAADataset'),
            val=dict(type='SODAADataset'),
            test=dict(type='SODAADataset', samples_per_gpu=1))
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', warmup='linear', warmup_iters=250,
                 warmup_ratio=0.001, step=[8, 11])
auto_scale_lr = dict(enable=False, base_batch_size=2)
checkpoint_config = dict(interval=1, max_keep_ckpts=12)
custom_hooks = [dict(type='NumClassCheckHook'), dict(type='SBLAEpochHook')]
seed = 0
