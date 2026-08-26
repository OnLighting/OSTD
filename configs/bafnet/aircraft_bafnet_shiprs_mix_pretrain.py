# configs/bafnet/aircraft_bafnet_shiprs_mix_pretrain.py
#
# 主目标: 在第一阶段 (官方数据 80%) 训练得到的 best_official_recall_fdr.pth
# 之上, 用 ShipRSImageNet level_3 数据集强化 25 类中 HM/LQS/QHS(+ 可选 MS)
# 4 个船舰类, 同时严格控制其他 21 类的回退。
#
# 设计要点:
# 1) num_classes=25 不改 —— 与 best_official_recall_fdr.pth strict 兼容,
#    直接 load_from。run.sh 会通过 --cfg-options load_from=<stage1 best> 覆写。
# 2) SourceBalancedDataset 混合 25 类官方 train + ShipRS mapped level_3,
#    sampling = 0.70 : 0.30 (默认 70% 官方 / 30% ShipRS)。
# 3) 短 schedule + 小 LR (lr=0.001, max_epochs=24, step=[16, 22])。
# 4) 关闭 SBLA —— HieAssigner 已是 25 类默认, 不要把它的超参 schedule
#    带到船上。
# 5) loss_bounder / LoadBalancingLoss 在 ShipRS 数据上照常跑 (不依赖类数)。
# 6) 不启用 ClassBalancedDataset oversample_thr (混合比例 + 短 schedule 已
#    覆盖类别不均)。
# 7) shiprs_class_mask=True —— fc_cls 通道 mask, 严格隔离 25 类中
#    非目标 21 类的回退。fc_reg 全开 (4 维 class-agnostic)。
# 8) val/test 都指向官方 val (官方流水线无独立 test 拆分, test 仅作兼容
#    别名);EvalHook 同时输出 bbox (mAP 诊断) 和 official (官方 Recall/FDR
#    选 best)。ShipRS val 用于审计而不参与 best 选择。

_base_ = ['./aircraft_bafnet_1x.py']


# === 关 SBLA (不继承 base 的 rpn_assigner 调度) ===
sbla = dict(enabled=False)


# === 数据: 25 类训练集 + ShipRS mapped level_3 混合 ===
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

ship_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_seg=False),
    dict(type='Resize', img_scale=(1280, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore']),
]

ship_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
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
        _delete_=True,
        # 70% 官方 train / 30% ShipRS — run.sh 可通过
        # --cfg-options data.train.source_weights="(0.7,0.3)" 覆盖。
        type='SourceBalancedDataset',
        source_weights=(0.70, 0.30),
        seed=20260819,
        datasets=[
            dict(
                type='AircraftDataset',
                ann_file='data/annotations/instances_train.json',
                img_prefix='data/images/train/',
                pipeline=ship_train_pipeline),
            dict(
                # ShipRS mapped 出来的 COCO 仍然 25 类 (HM/LQS/QHS/(MS) + 全部 ignore)
                type='AircraftDataset',
                ann_file='data/external/shiprs_mapped_train.json',
                # ShipRSImageNet 原图位于 external_data/ShipRSImageNet/images/
                # (相对仓库根). ShipRS mapped COCO 的 file_name 含 ``images/`` 前缀
                # (如 ``images/100001606.bmp``),所以 img_prefix 不能含 ``images/``,
                # 否则会拼成 ``external_data/ShipRSImageNet/images/images/...`` 双前缀。
                img_prefix='external_data/ShipRSImageNet/',
                pipeline=ship_train_pipeline),
        ]),
    val=dict(
        _delete_=True,
        # 官方 val：best 选择在官方 val 上做，ShipRS val 只用于数据审计，
        # 不进入 EvalHook。
        type='AircraftDataset',
        ann_file='data/annotations/instances_val.json',
        img_prefix='data/images/val/',
        pipeline=ship_test_pipeline),
    test=dict(
        _delete_=True,
        # 官方流水线无独立 test 拆分；test 仅作 tools/test.py 的接口别名。
        type='AircraftDataset',
        ann_file='data/annotations/instances_val.json',
        img_prefix='data/images/val/',
        pipeline=ship_test_pipeline),
)


# === 关键: strict load_from ===
# run.sh 会用第一阶段 best_official_recall_fdr.pth 覆盖这里的默认值。
load_from = None
resume_from = None  # 重新训练, 不续 epoch / optimizer 状态


# === 短 schedule + 小 LR ===
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear', warmup_iters=500, warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)


# === EvalHook 每 1 epoch 同时计算 bbox (mAP 诊断) 和 official (Recall/FDR) ===
evaluation = dict(interval=1, metric=['bbox', 'official'])


# === ShipRS 通道 mask 开关 ===
# 由 tools/train.py 读取, 决定是否调用 install_class_mask。
# target=(0,1,2) 对应 HM/LQS/QHS。如需启用 MS, 改为 (0,1,2,3) 并
# 在 prepare_shiprs 时加 --enable-ms。
shiprs_class_mask = True
shiprs_class_mask_targets = (0, 1, 2)
