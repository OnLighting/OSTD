def apply_sbla_config(cfg):
    """Inject the optional RPN SBLA assigner after config overrides."""
    sbla = cfg.get('sbla', None)
    if sbla is None or not sbla.get('enabled', False):
        return
    assigner_cfg = dict(sbla)
    assigner_cfg.pop('enabled')
    assigner_cfg['type'] = 'SBLAAssigner'
    assigner_cfg.setdefault('ignore_iof_thr', -1)
    cfg['model']['train_cfg']['rpn']['assigner'] = assigner_cfg


def apply_model_ablation_config(cfg):
    """Inject top-level architecture switches into the model config."""
    if 'use_arfc' in cfg:
        cfg['model']['use_arfc'] = cfg['use_arfc']
