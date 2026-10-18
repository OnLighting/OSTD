"""DIOR: remove only the DRFC DRFC refinement before FBDA."""

_base_ = ['../ostd_dior_1x.py']

model = dict(p0_drfc_enabled=False)

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
