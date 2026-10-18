"""DIOR: remove only layer3 DRFC in both backbone passes."""

_base_ = ['../ostd_dior_1x.py']

model = dict(
    backbone=dict(drfc_layer3_cfg=None),
    neck=dict(rfp_backbone=dict(drfc_layer3_cfg=None)))

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
