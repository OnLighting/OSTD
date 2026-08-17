from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SBLAEpochHook(Hook):
    """Pass the epoch to an RPN assigner when it supports scheduling."""

    def before_train_epoch(self, runner):
        model = getattr(runner.model, 'module', runner.model)
        rpn_head = getattr(model, 'rpn_head', None)
        assigner = getattr(rpn_head, 'assigner', None)
        if assigner is not None and hasattr(assigner, 'set_epoch'):
            assigner.set_epoch(runner.epoch)
