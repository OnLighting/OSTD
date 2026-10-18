"""Low-resource smoke test, NOT a paper experiment.

Checks the full train/eval path on a tiny subset (two DIOR-class images
created by tools/make_smoke_subset.py) with one epoch. Use it to verify an
installation before launching the 12-epoch paper runs. Numbers from this
config are meaningless.
"""

_base_ = ['./ostd_dior_1x.py']

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        ann_file='./data/smoke/annotations/instances_train.json',
        img_prefix='./data/smoke/images/'),
    val=dict(
        ann_file='./data/smoke/annotations/instances_val.json',
        img_prefix='./data/smoke/images/'),
    test=dict(
        ann_file='./data/smoke/annotations/instances_val.json',
        img_prefix='./data/smoke/images/',
        samples_per_gpu=1))
optimizer = dict(lr=0.0025)
lr_config = dict(warmup_iters=10)
runner = dict(type='EpochBasedRunner', max_epochs=1)
sbla = dict(schedule_end_epoch=1)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
evaluation = dict(interval=1, metric='bbox')
gpu_ids = range(0, 1)
