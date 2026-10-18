_base_ = [
    '../_base_/datasets/dior_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# OSTD main experiment on DIOR (paper Table III, "Ours" row: AP 53.91).
# Training uses the merged train+val split (11,725 images); see
# docs/DATA.md for how instances_trainval.json is produced.
sbla = dict(
    enabled=True,
    mode='full',
    topk=9,                      # 论文: 学习确定性取最高 9 个质量分均值
    gamma=0.5,                   # 论文: 质量分数指数 gamma
    alpha=0.5,                   # S_0 = 256, alpha_E = 0.5
    beta=0.6,                    # beta_E = 0.6, 预算从 ~282 衰减到 128
    kappa=-0.06,                 # 论文 alpha_O = -0.06
    phi=1.4,                     # 论文 beta_O = 1.4
    base_sample_num=256,         # 参考预算 S_0 = 256
    min_pos_per_gt=1,            # 每目标最小正样本配额
    schedule_start_epoch=0,      # MMCV hook 收到 0-based epoch
    schedule_end_epoch=11,
    candidate_mode='center',
    scores_are_logits=True)

model = dict(
    type='OSTDDetector',
    drfc_compatible_init=True,   # 复用 ImageNet stem/Layer3 权重做 DRFC 快捷路径
    p0_residual_scale=0.1,       # 论文: P0 残差缩放 lambda = 0.1
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
        drfc_norm_eval=False,   # DRFC 内 BN 保持训练统计
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
                num_classes=20,                   # DIOR 20 classes
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
                num_classes=20,
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
                num_classes=20,
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

# Paper protocol: 3 GPUs x 2 images (total batch 6), lr 0.01, 12 epochs.
# The released reference checkpoint is epoch 12 of this run (see README).
# Training set is the exact train+val merge (11,725 images) described in
# docs/DATA.md; file_name entries carry the split subdirectory.
data = dict(
    samples_per_gpu=2,
    train=dict(
        type='DIORDataset',
        ann_file='./data/DIOR_coco/annotations/instances_trainval.json',
        img_prefix='./data/DIOR_coco/images/'),
    val=dict(type='DIORDataset'),
    test=dict(type='DIORDataset', samples_per_gpu=8))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step', warmup='linear', warmup_iters=250,
    warmup_ratio=0.001, step=[8, 11])
auto_scale_lr = dict(enable=False, base_batch_size=6)
checkpoint_config = dict(interval=1, max_keep_ckpts=12)
custom_hooks = [dict(type='NumClassCheckHook'),
                dict(type='SBLAEpochHook')]
seed = 0
gpu_ids = range(0, 3)
# Main-run reference: experiments/DIOR/main (archived, not redistributed).
default_result_checkpoint = None
