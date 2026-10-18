checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=100,
    hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(type='NumClassCheckHook'),
    # 训练基础设施，与数据集无关；项目内已实现 EarlyStoppingHook/
    # BestSaverHook，绕开 mmcv 1.4.0 的 save_best bug。
    dict(type='BestSaverHook', monitor='bbox_mAP_50'),
    dict(type='EarlyStoppingHook',
         monitor='bbox_mAP_50',
         patience=8,
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
auto_scale_lr = dict(enable=True, base_batch_size=16)
