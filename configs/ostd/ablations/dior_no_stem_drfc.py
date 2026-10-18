"""DIOR: remove only the Conv1/stem DRFC in both backbone passes."""

_base_ = ['../ostd_dior_1x.py']

model = dict(
    backbone=dict(drfc_cfg=None),
    neck=dict(rfp_backbone=dict(drfc_cfg=None)))

# Only epoch 12 is tested; retain its checkpoint during training.
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
