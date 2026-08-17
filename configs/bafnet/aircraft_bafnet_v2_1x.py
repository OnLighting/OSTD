# BAFNet v2 — improvement plan P0-B + P1-A + P1-B + P2-A applied.
#
# 继承 aircraft_bafnet_1x.py (full model, 100ep schedule) 的 _base_，
# 在此基础上叠加以下改动：
#   P0-B  test_cfg.rcnn: score_thr 0.05→0.30, max_per_img 3000→300
#   P1-A  train_pipeline + RandCopyPaste; data.train 改为
#         DomainBalancedDataset(target_class_id=3, extras=(1,2,2))
#   P1-B  train_pipeline: img_scale (1280,800)→(1280,1024) + multiscale,
#         RandRotate (自定义 fallback)
#   P2-A  bbox_head[i].loss_cls.class_weight[24] = 5.0 (FSC 5× 加权)
#
# 训练侧 P2-B (alpha + 熵正则) 在 mmdet/models/* 直接生效，无需 config 改动。
# 评测侧 P0-A (三大类口径) 在 tools/eval_recall_fdr.py 直接生效。

_base_ = ['./aircraft_bafnet_1x.py']


# === P0-B: 推理阈值收紧 ===
test_cfg = dict(
    rpn=dict(
        nms_pre=3000,
        max_per_img=3000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.30,                                # P0-B
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=300))                              # P0-B


# === P1-B + P2-A: 覆盖 train_pipeline / test_pipeline ===
# img_norm_cfg 与 dataset_detection.py 一致
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# 多尺度：multiscale_mode='value'（此 mmdet 不支持 multiscale_range）
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize',
         img_scale=[(1280, 640), (1280, 800), (1280, 1024)],  # P1-B 多尺度
         keep_ratio=True,
         multiscale_mode='value'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandRotate', prob=0.5),                  # P2-A (P1-B fallback)
    dict(type='RandCopyPaste', prob=0.5),               # P1-A
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(1280, 1024),                        # P1-B 测试高分辩率
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ])
]


# === P1-A: data 覆盖为 DomainBalancedDataset ===
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='DomainBalancedDataset',                   # P1-A
        target_class_id=3,                              # MS
        domain_prefixes=('01-PAN', '02-PAN', 'OTHER'),
        domain_extras=(1, 2, 2),                        # 提增 02-PAN / OTHER
        dataset=dict(
            type='ClassBalancedDataset',
            oversample_thr=1e-3,
            dataset=dict(
                type='AircraftDataset',
                ann_file='./data/annotations/instances_train.json',
                img_prefix='./data/images/train/',
                pipeline=train_pipeline))),
    val=dict(
        type='AircraftDataset',
        ann_file='./data/annotations/instances_val.json',
        img_prefix='./data/images/val/',
        pipeline=test_pipeline),
    test=dict(
        type='AircraftDataset',
        ann_file='./data/annotations/instances_test.json',
        img_prefix='./data/images/test/',
        pipeline=test_pipeline))


# === P2-A: bbox_head 三阶段 loss_cls 加 class_weight（仅 FSC 5×） ===
# 注：mmdet 配置继承中 list 整体替换，所以这里完整重写三个 head；
# 其他字段（type=Shared2FCBBoxHead, num_classes=25, bbox_coder, loss_bbox 等）
# 与原 aircraft_bafnet_1x.py 一致，仅 loss_cls 增加 class_weight。
class_weight = [1.0] * 24 + [5.0]   # FSC=24 加权 5×

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=25,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0,
                    class_weight=class_weight),       # P2-A
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=25,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0,
                    class_weight=class_weight),       # P2-A
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=25,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0,
                    class_weight=class_weight),       # P2-A
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
        ]))