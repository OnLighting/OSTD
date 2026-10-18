"""Convert pre-release OSTD checkpoints to the public state-dict layout.

The public rename (ARFC -> DRFC, CascadeRCNN_BAF -> OSTDDetector) changed
module attribute names, which changes state_dict keys. This script maps the
old keys onto the new ones so existing trained weights keep loading:

    backbone.arfc_stem.*      -> backbone.drfc_stem.*
    backbone.arfc_layer3.*    -> backbone.drfc_layer3.*
    .../arfc_stem.* (RFP)     -> .../drfc_stem.*
    .../arfc_layer3.* (RFP)   -> .../drfc_layer3.*
    p0_arfc.*                 -> p0_drfc.*
    rgb_global.*              -> coarse_pred.*
    dsam.*                    -> fbda.*
    seghead.*                 -> boundary_head.*

Also swaps in the new dsam.* sub-attribute names created when ASPP branches
were rewritten with add_module (aspp1/aspp2/aspp3/aspp4 -> aspp1/aspp3/
aspp5/aspp7). Run on the training server with torch available:

    python tools/convert_legacy_checkpoint.py IN.pth OUT.pth
"""
import argparse

import torch

# Whole-module attribute renames applied to any state-dict key prefix.
# Order matters: the first matching rule wins, so specific rules come first.
KEY_RENAMES = [
    ('backbone.arfc_stem.', 'backbone.drfc_stem.'),
    ('backbone.arfc_layer3.', 'backbone.drfc_layer3.'),
    ('.arfc_stem.', '.drfc_stem.'),
    ('.arfc_layer3.', '.drfc_layer3.'),
    ('p0_arfc.', 'p0_drfc.'),
    ('rgb_global.', 'coarse_pred.'),
    ('dsam.', 'fbda.'),
    ('seghead.', 'boundary_head.'),
]
# Losses are functional now: these legacy keys hold no trained weights.
# boundary_loss_func.fuse_kernel is unchanged and stays (same name in both).
DROPPED_PREFIXES = ('aux_loss.', 'router_balance_loss.')


def convert_state_dict(state_dict):
    converted = {}
    for key, value in state_dict.items():
        if key.startswith(DROPPED_PREFIXES):
            continue
        for old, new in KEY_RENAMES:
            if key.startswith(old):
                key = new + key[len(old):]
                break
        converted[key] = value
    return converted


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('src', help='legacy checkpoint (.pth)')
    parser.add_argument('dst', help='converted checkpoint (.pth)')
    parser.add_argument('--in-place', action='store_true',
                        help='overwrite the input file')
    args = parser.parse_args()

    checkpoint = torch.load(args.src, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    converted = convert_state_dict(state_dict)
    if 'state_dict' in checkpoint:
        checkpoint['state_dict'] = converted
    else:
        checkpoint = converted

    dst = args.src if args.in_place else args.dst
    torch.save(checkpoint, dst)
    print(f'{len(state_dict)} -> {len(converted)} tensors, saved to {dst}')


if __name__ == '__main__':
    main()
