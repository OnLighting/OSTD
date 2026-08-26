# dataset settings for the 25-class aircraft/vehicle dataset
# produced by tools/convert_yolo_to_coco.py.
dataset_type = 'AircraftDataset'
data_root = './data/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1280, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 800),
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
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        # ClassBalancedDataset over-samples images containing rare classes
        # (HM=17, LQS=30, FSC=402) so the epoch sees them ~6x more often than
        # images containing A16_FA-18 (2147). oversample_thr is the per-image
        # category-frequency threshold below which the image is repeated.
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train.json',
            img_prefix=data_root + 'images/train/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'images/val/',
        pipeline=test_pipeline),
    # `data.test` 仅作为 tools/test.py 依赖的接口别名使用；官方流水线并
    # 不创建独立的 test 拆分，best checkpoint 选择和最终指标均在 data.val
    # 上计算和报告。
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'images/val/',
        pipeline=test_pipeline))
# 训练期 EvalHook 同时计算 bbox (COCO mAP, 用于诊断) 和 official
# (Recall/FDR, 用于 best 选择与早停);详见 mmdet/core/evaluation/official_metrics.py。
evaluation = dict(interval=1, metric=['bbox', 'official'])
