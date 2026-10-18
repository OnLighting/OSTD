def apply_sbla_config(cfg):
    """Inject the optional RPN SBLA assigner after config overrides."""
    sbla = cfg.get('sbla', None)
    if sbla is None or not sbla.get('enabled', False):
        return

    assigner_cfg = dict(sbla)
    assigner_cfg.pop('enabled')
    assigner_cfg['type'] = 'SBLAAssigner'
    assigner_cfg.setdefault('ignore_iof_thr', -1)

    # Older MMDetection configs keep train_cfg at the top level and inject it
    # into model later in tools/train.py. Support both layouts here.
    model_cfg = cfg.get('model', {})
    train_cfg = model_cfg.get('train_cfg', None)
    if train_cfg is None:
        train_cfg = cfg.get('train_cfg', None)
    if train_cfg is None or train_cfg.get('rpn', None) is None:
        raise KeyError('SBLA requires an RPN train_cfg')
    train_cfg['rpn']['assigner'] = assigner_cfg

    custom_hooks = cfg.setdefault('custom_hooks', [])
    if not any(hook.get('type') == 'SBLAEpochHook'
               for hook in custom_hooks):
        custom_hooks.append(dict(type='SBLAEpochHook'))
