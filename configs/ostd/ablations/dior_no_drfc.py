"""DIOR ablation: w/o DRFC (paper Table IV: AP 52.59 / APs 15.67).

SBLA retained; every DRFC module is bypassed (stem and Layer3 in both the
initial and recursive backbone, and the P0 refinement), so the router
balance loss is identically zero.
"""

_base_ = ['../ostd_dior_1x.py']

model = dict(
    drfc_compatible_init=False,
    p0_drfc_enabled=False,
    backbone=dict(drfc_cfg=None, drfc_layer3_cfg=None),
    neck=dict(rfp_backbone=dict(drfc_cfg=None, drfc_layer3_cfg=None)))

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
