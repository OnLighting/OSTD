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
    # 同时也是唯一的早停依据；旧版基于 bbox_mAP 的 BestSaverHook / EarlyStoppingHook
    # 已删除，避免其与官方 best 选择口径冲突。COCO mAP 通过 evaluation 阶段
    # 仍然输出，仅作诊断，不影响 best 保存与早停。
    dict(type='OfficialBestSaverHook',
         recall_target=.85,
         fdr_limit=.20,
         recall_tolerance=.005),
    dict(type='OfficialEarlyStoppingHook',
         patience=16,
         recall_target=.85,
         fdr_limit=.20,
         recall_tolerance=.005),
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
