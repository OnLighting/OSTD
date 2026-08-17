# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## ⚠️ IMPORTANT: No local execution environment

**This machine has no working runtime for this project** — no usable Python/mmdet/mmcv/CUDA install, no GPU, and datasets are not staged at the paths the configs expect. Therefore:

- **Do NOT write or run any test code or programs.** No `pytest`, no `python tools/train.py` / `test.py`, no linters, no throwaway verification scripts. They will fail here and add nothing.
- Reason about correctness by **reading the code**. Verification is done by code review only, never by execution.
- The "Install / environment" and "Common commands" sections below are **reference only** — they describe how the project runs on the training server, not anything to execute on this machine.

## Project overview

BAFNet — a remote-sensing small-object detector on mmdetection 2.x. It layers a **Boundary-Aware Feature Fusion** module (`DSAM` + boundary heads + `DetailAggregateLoss`) on top of Cascade R-CNN with a DetectoRS_ResNet + RFP backbone, and adds an **ARFC** (Adaptive Receptive Field Convolution) expert-routing block plus a dual-pathway boundary-GT scheme and a load-balancing auxiliary loss.

Current target dataset: **25-class optical-satellite aircraft/vehicle detection** (`AircraftDataset`). Configs and class heads are set for `num_classes=25`.

Key source locations:

- Detector: `mmdet/models/detectors/baf.py` → `CascadeRCNN_BAF` (registered as `type='CascadeRCNN_BAF'`).
- ARFC block: `mmdet/models/utils/arfc.py` (`ARFC`); parts in `arfc_parts.py` (`Expert`, `LCE`, `GridRouter`); `CoordAtt` in `coord_att.py`.
- Boundary heads / `segmenthead`: `mmdet/models/model_utils.py`.
- Losses: `DetailAggregateLoss` in `mmdet/models/losses/detail_loss.py`; `LoadBalancingLoss` in `load_balancing_loss.py`.
- Dataset: `mmdet/datasets/aircraft.py` → `AircraftDataset` (subclass of `CocoDataset`).
- SBLA assigner: `mmdet/core/bbox/assigners/sbla_assigner.py`; hierarchical baseline in `hierarchical_assigner.py` (`HieAssigner`).
- Configs: `configs/bafnet/aircraft_bafnet_1x.py` (full, 100ep) and `configs/bafnet/aircraft_bafnet_baseline_1x.py` (ARFC off, 50ep).
- Entry points: `tools/train.py` (custom; imports `sbla_config`), `tools/test.py`. Pipeline driver: `run.sh`.

## Architecture notes

**Detector (`baf.py` — `CascadeRCNN_BAF`)**
- `extract_feat(img, img_metas)`: backbone → RFP neck → returns `x` (5-level FPN list) and `p0_refined`. On the deepest feature (`x[4]`) it runs `rgb_global` (a `Pred_Layer`) to get boundary prior `p4`; on the shallowest feature (`x[0]`) it runs the **P0 ARFC** (`self.p0_arfc`, lightweight 3-expert, top_k=3) when `use_arfc=True`, else identity. DSAM is applied to the ARFC-refined `p0_refined` using `p4` as the prior; the enhanced feature is fused back as `x[0] = ef + x[0] + p0_refined`.
- `forward_train`: builds a **dual-pathway boundary GT** — `lb = Grayscale(img)` (image domain) and `fused_gt = self.dual_gt(lb, p0_refined)` (image-side 3-scale Laplacian stacked with feature-side 3-scale Laplacian, 6ch). The image-side half (`fused_gt[:, :3]`) is reduced to 1ch via `.max(dim=1)` and binarized at `> 0.1`, then fed to `boundary_loss_func.forward_with_gt(x_b0, img_side_gt)`. `x_b0` comes from `seghead_dual` on `cat([x[0], p0_refined])`; `x_b3` from `seghead` on `x[4]` against the grayscale `lb` via the standard `DetailAggregateLoss` path. These sum into `loss_bounder`.
- An **auxiliary load-balancing loss** (`loss_aux`) walks `self.modules()` for every `ARFC` and sums `LoadBalancingLoss` over its cached `_last_router_logits` (the router's last forward logits). Zero when no ARFC has routed (e.g. `use_arfc=False`).
- `simple_test` / `show_result` follow mmdet two-stage conventions; `show_result` unwraps the `'ensemble'` key from the result dict.

**ARFC (`arfc.py`)** — MFE (Top-k weighted multi-scale experts) + LCE (1×K + K×1 long-range shared expert) + shortcut.
- `Expert`: depthwise-separable large-kernel conv + 1×1 projection + `CoordAtt`.
- `GridRouter`: GAP → `Linear(in_c → num_experts)` → top-k softmax. Emits `(logits, topk_idx, weights)`; `logits` cached on the parent ARFC as `_last_router_logits` for the balancing loss. (Note: it was previously `LazyLinear`; switched to explicit `in_c` because mmcv `BaseModule` crashed reading the shape of uninitialized lazy params.)
- `lightweight=True` → 3 experts (kernels 5/7/9, channels 96/64/48); standard → 4 experts (5/7/9/11). Per-expert output channels differ, so each has a 1×1 `align` conv to a common `out_c` before stacking + gather + weighted sum.
- LCE is a strip-conv (1×K then K×1) depthwise path summed with the MFE output.

**Backbone & neck**: DetectoRS_ResNet (depth 50, `ConvAWS` + `SAC` with `use_deform=True`), `out_indices=(0,1,2,3)`. `RFP` neck (`mmdet/models/necks/rfp.py`) runs an internal `rfp_backbone` (also DetectoRS_ResNet) for `rfp_steps=2`. Both expect `pretrained='torchvision://resnet50'`. RFP produces 5 outputs (`num_outs=5`); the detector indexes `x[0]` (shallow) and `x[4]` (deepest).

**Anchor generator**: `RFGenerator` with `fraction=0.5` (Recall-Filtering, `mmdet/core/anchor`), strides `[4,8,16,32,64]` across the 5 FPN levels.

**RPN assigner / SBLA ablation**: The default RPN assigner is `HieAssigner` (`assign_metric='kl'`, `topk=[3,1]`, `BboxDistanceMetric`). The config optionally enables **SBLA** (`sbla` dict with `enabled`, `mode` ∈ `{balanced, full}`, and schedule knobs). `tools/sbla_config.py::apply_sbla_config` rewrites `cfg.model.train_cfg.rpn.assigner` to `SBLAAssigner` when `enabled=True`; `SBLAEpochHook` (registered in `default_runtime.py`) updates its per-epoch positive-sample budget. In the shipped configs SBLA is **disabled** (`enabled=False`), so the assigner stays `HieAssigner`.

**`use_arfc` ablation**: top-level `use_arfc = False` in the baseline config is injected into `cfg.model` by `apply_model_ablation_config` (called from `tools/train.py` after `apply_sbla_config`); `CascadeRCNN_BAF.__init__` reads it. With it off, P0 ARFC is skipped (identity) and `loss_aux` is zero, isolating ARFC's marginal contribution; DSAM, dual-boundary GT, `seghead_dual`, and `LoadBalancingLoss` stay in place.

**CUDA coupling**: `DetailAggregateLoss.__init__` forces its Laplacian/fuse kernels to `torch.cuda.FloatTensor`, and `.forward` casts GT masks to CUDA float. This loss only runs on GPU — relevant if anyone tries CPU smoke tests (don't, per the rule above).

## Data layout

The active dataset lives in **`new_data/`** (YOLO-style), not `data/`. `new_data/NEW_DATA_REPORT.md` is the authoritative, detailed data report — **read it for any dataset work**; do not enumerate individual image/label files.

- `new_data/dataset.yaml` — class names (25) and split declarations.
- `new_data/images/train/` (4481 imgs), `new_data/labels/train/` (4481 YOLO `.txt`).
- `new_data/val/`, `new_data/test/` are declared but **empty/absent** — `tools/split_val.py` creates the real train/val/test split at run time.
- `new_data/background_only.txt` — 3 explicitly-retained negative samples (images whose labels are empty on purpose).
- Class imbalance is severe: max `A16_FA-18` (2148) vs min `HM` (17), ratio ~177×. The dataset pipeline wraps training in `ClassBalancedDataset(oversample_thr=1e-3)` to up-sample rare classes.

The mmdet dataset config (`configs/_base_/datasets/aircraft_detection.py`) expects **COCO-format** annotations at `./data/annotations/instances_{train,val,test}.json` with images at `./data/images/{train,val,test}/`. The conversion is done by `tools/convert_yolo_to_coco.py` (driven by `run.sh`). So the flow is: YOLO data in `new_data/` → `split_val.py` → `convert_yolo_to_coco.py` → COCO annotations under `data/` → training.

## Install / environment

> Reference only — **do not run any of this on this machine** (see top of file).

`requirements.txt` pulls in `requirements/{build,runtime,optional,tests}.txt`. `pycocotools` is platform-conditional (Linux vs Windows) in `runtime.txt`. Install pattern:

```
pip install -r requirements/build.txt
pip install -r requirements/runtime.txt
pip install -r requirements/optional.txt
pip install -r requirements/tests.txt
pip install -e .   # installs the `mmdet` package in editable mode
```

Use mmcv-full 1.3.3+ (see `requirements/mminstall.txt`). `tools/train.py` is a **custom** entry point (not stock mmdet): it imports `sbla_config` from the repo root, so the working directory / `PYTHONPATH` must include the repo root when invoked.

## Common commands

> Reference only — these run on the training server, **not here** (see top of file).

The canonical end-to-end pipeline is `run.sh` (run from repo root). It backs up `data/images/train` + `data/labels/train`, splits, builds COCO annotations, trains, tests, and reports metrics:

```
./run.sh
```

Manual single steps (matches what `run.sh` invokes):

```
# split + COCO conversion
python tools/split_val.py --root data --ratios 0.6 0.2 0.2 --seed 0 --overwrite
python tools/convert_yolo_to_coco.py --root data --split train --out data/annotations/instances_train.json
python tools/convert_yolo_to_coco.py --root data --split val   --out data/annotations/instances_val.json
python tools/convert_yolo_to_coco.py --root data --split test  --out data/annotations/instances_test.json

# train (note: work_dir is a positional arg here, unlike stock mmdet)
python tools/train.py configs/bafnet/aircraft_bafnet_1x.py work_dirs/<run>

# evaluate
python tools/test.py configs/bafnet/aircraft_bafnet_1x.py work_dirs/<run>/best_bbox_mAP.pth --eval bbox

# recall / FDR metrics (class names passed via --names)
python tools/eval_recall_fdr.py --pred <preds.json> --gt data/annotations/instances_test.json \
    --classes 25 --names HM,LQS,QHS,MS,A1_SU-35,...,FSC --out-prefix work_dirs/<run>/test_metrics
```

Distributed: `./tools/dist_train.sh <config> <NUM_GPUS>` (matches stock `dist_test.sh` style).

Linting / formatting (config in `setup.cfg`): `flake8 mmdet`, `yapf -r mmdet`, `isort -rc mmdet` (pinned to 4.3.21). Pre-commit: `pre-commit run --all-files`.

## Adapting to a new dataset

1. Read the dataset's report doc (e.g. `new_data/NEW_DATA_REPORT.md`) and `dataset.yaml` first — do not scan `images/` or `labels/` directly.
2. The mmdet pipeline consumes **COCO JSON**; YOLO data must be converted via `tools/convert_yolo_to_coco.py` (and split via `tools/split_val.py`). The COCO paths are set in `configs/_base_/datasets/aircraft_detection.py`.
3. Set `num_classes` in **all three** `bbox_head` stages of the config (currently 25), and update `CLASSES` in `mmdet/datasets/aircraft.py` (or register a new dataset class) to match.
4. The boundary-related heads (`rgb_global`, `DSAM`, `seghead`, `seghead_dual`) operate on **256-channel** features — if you change backbone/neck output channels, update these and the ARFC `in_c` together.
5. `samples_per_gpu=2` with `lr=0.005`; `auto_scale_lr` is off by default (`base_batch_size=16`). Re-tune LR if batch size changes.

## Style & conventions

- mmdet 2.x registry pattern: components registered via `@DETECTORS.register_module()` / `@HEADS.register_module()` / `@LOSSES.register_module()` / `@DATASETS.register_module()`, picked up by the respective `builder.py`.
- Init: `ConvBNReLU.init_weight` uses kaiming-normal; `Pred_Layer`, `ASPP`, ARFC sub-modules, and `segmenthead` rely on PyTorch defaults — be aware when adding layers.
- In-code comments and several config comments are written in Chinese; match the surrounding language when editing.
- `tests/` contains unit tests (`test_arfc*.py`, `test_baf_forward_train.py`, `test_load_balancing_loss.py`, etc.) — **do not execute them here**; they exist for the CI/training-server environment.
