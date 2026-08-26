checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=1)

log_config = dict(
    interval=100,            # = 200 样本 / samples_per_gpu(2)
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [
    dict(type='NumClassCheckHook'),
    # 普通 assigner 下为空操作；SBLA 下用于更新随 epoch 衰减的正样本预算。
    dict(type='SBLAEpochHook'),
    # 官方 Recall/FDR 口径 best 选择（与评估 metric 中的 'official' 对应）。
    # 保留原 bbox_mAP BestSaverHook/EarlyStoppingHook 用于 mAP 诊断输出，
    # 但早停与 best 保存以官方指标为准。
    dict(type='OfficialBestSaverHook',
         recall_target=.85,
         fdr_limit=.20,
         recall_tolerance=.005),
    dict(type='OfficialEarlyStoppingHook',
         patience=16,
         recall_target=.85,
         fdr_limit=.20,
         recall_tolerance=.005),
    dict(type='BestSaverHook',         # 绕开 mmcv 1.4.0 save_best bug，自存 best
         monitor='bbox_mAP'),
    dict(type='EarlyStoppingHook',
         monitor='bbox_mAP',      # 与 evaluation metric='bbox' 输出对应
         patience=16,
         min_delta=0.001),
]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
