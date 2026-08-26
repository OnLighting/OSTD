"""Backward hook that isolates gradient flow to specific fc_cls / fc_reg columns.

Designed for ship-class (HM / LQS / QHS / MS) fine-tuning on the existing
25-class BAFNet checkpoint.  Other 21 classes are protected at the head layer:

  * ``fc_cls`` output dim = num_classes + 1 (background).  Only the target
    columns and the background column receive gradient; all other columns
    are masked to zero.
  * ``fc_reg`` is ``reg_class_connected=True`` here, so its output dim is
    always 4 -- all four columns are kept open (regression is class-agnostic).

Backbone, neck, BAFNet boundary modules, ARFC router, and shared_fcs are NOT
frozen -- only the head column-wise update is restricted.  Use the standard
freeze-prefix trick in your config if you also want to stop backbone updates.

Layer-name match rule (defensive):
  Only ``roi_head.bbox_head.<digit>.fc_cls`` and the matching ``fc_reg`` are
  hooked.  RPN's ``rpn_cls / rpn_reg`` and ARFC's ``grid_router.score_mlp``
  are deliberately excluded by the exact-name match.
"""

import torch
import torch.nn as nn


def _is_cascade_head_fc(name):
    """Match only cascade bbox_head.{0,1,2}.fc_cls / fc_reg."""
    parts = name.split('.')
    if len(parts) < 4:
        return False
    return (parts[0] == 'roi_head'
            and parts[1] == 'bbox_head'
            and parts[-1] in ('fc_cls', 'fc_reg')
            and parts[2].isdigit())


def install_class_mask(model, trainable_target_classes=(0, 1, 2)):
    """Register backward hooks on cascade head fc_cls / fc_reg columns.

    Args:
        model: A built mmdet detector (already on device, not yet in train()
            mode).  The hooks fire on the next ``loss.backward()``.
        trainable_target_classes: Target class ids whose fc_cls columns (plus
            the background column at ``num_classes``) should receive gradient.

    Returns:
        list[torch.utils.hooks.RemovableHandle]: Hook handles for cleanup.
    """
    target_set = set(int(c) for c in trainable_target_classes)
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not _is_cascade_head_fc(name):
            continue

        out_features = module.out_features
        if name.endswith('.fc_cls'):
            num_classes = out_features - 1
            allowed_cols = sorted(target_set | {num_classes})  # + background
        else:  # fc_reg (4 cols, class-agnostic, keep all)
            allowed_cols = list(range(out_features))

        mask = torch.zeros(out_features, dtype=module.weight.dtype,
                           device=module.weight.device)
        for c in allowed_cols:
            mask[c] = 1.0

        def make_hook(mask_tensor):
            def hook(grad):
                # Mask may live on CPU when installed before .cuda(); grads are
                # always on the model's device during backward.
                m = mask_tensor if mask_tensor.device == grad.device \
                    else mask_tensor.to(grad.device)
                if grad.dim() == 2:
                    return grad * m.unsqueeze(1)
                return grad * m
            return hook

        hooks.append(module.weight.register_hook(make_hook(mask)))
        if module.bias is not None:
            hooks.append(module.bias.register_hook(make_hook(mask)))

    if not hooks:
        raise RuntimeError(
            'install_class_mask found no cascade bbox_head fc_cls / fc_reg '
            'layers; check the detector architecture or hook-name pattern.')

    return hooks