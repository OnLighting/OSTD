"""DIOR ablation: w/o SBLA (paper Table IV: AP 51.39 / APs 13.67).

Full DRFC model; the RPN falls back to the stock MaxIoU assigner and
RandomSampler inherited from schedules/schedule_1x.py.
"""

_base_ = ['../ostd_dior_1x.py']

sbla = dict(enabled=False)

custom_hooks = [dict(type='NumClassCheckHook')]

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
