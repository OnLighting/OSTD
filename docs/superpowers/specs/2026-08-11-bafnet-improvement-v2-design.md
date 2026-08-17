# BAFNet Improvement v2 — Design

> Implements the priority list in `docs/improvement_plan.md` (§3 / §5). Targets
> official三大类补充口径 R≥85% / FDR≤20%. Reference baseline
> `work_dirs/new_data_arfc_only_v1` (官方口径 R=75.80% / FDR=34.67%, FAIL).
>
> **No code execution on this machine** (per `CLAUDE.md`). All changes land
> as in-repo code only; training & evaluation occur on the training server.

---

## 1. Scope & deliverables

| # | File | Change | Retrain? |
|---|------|--------|----------|
| P0-A | `tools/eval_recall_fdr.py` | Add official 三大类 metric aggregation | No |
| P0-B | `configs/bafnet/aircraft_bafnet_{1x,baseline_1x}.py` | `score_thr 0.05→0.30, max_per_img 3000→300` | No |
| P1-A | `mmdet/datasets/dataset_wrappers.py` + `__init__.py` + `builder.py` | New `DomainBalancedDataset` wrapper | Yes |
| P1-A | `configs/bafnet/aircraft_bafnet_v2_1x.py` (new) | Add `CopyPaste` to `train_pipeline` | Yes |
| P1-B | `configs/bafnet/aircraft_bafnet_v2_1x.py` (new) | `img_scale (1280,800)→(1280,1024)` + multiscale | Yes |
| P1-B | `mmdet/models/detectors/baf.py` | Drop `>0.1` binarization on `img_side_gt`; pass continuous Laplacian / 8 to BCE | Yes |
| P2-A | `configs/bafnet/aircraft_bafnet_v2_1x.py` (new) | `loss_cls.class_weight[24]=5.0` ×3 stages + `RandomRotate`/`CopyPaste` | Yes |
| P2-B | `mmdet/models/losses/load_balancing_loss.py` + `mmdet/models/detectors/baf.py` | `alpha 0.01→0.1` (default + `baf.py:267` call site) + add router-entropy reg `β=0.01` | Yes |

A single new config `configs/bafnet/aircraft_bafnet_v2_1x.py` carries all P1/P2 training-time changes. P0-A/B land in shared config + tool files (no retrain).

---

## 2. P0-A — 三大类官方口径聚合 (`tools/eval_recall_fdr.py`)

### 2.1 Mapping

```python
def super_of(name):
    if name in {'HM', 'LQS', 'QHS', 'MS'}:    return 'ship'
    if name == 'FSC':                         return 'vehicle'
    if name.startswith('A'):                  return 'aircraft'
    return None
```

Category IDs (per `tools/convert_yolo_to_coco.py::CLASS_NAMES`):
- ship = {0,1,2,3}; aircraft = {4..23}; vehicle = {24}.

### 2.2 Aggregation

- Group per-class `(recall, fdr)` by super-class.
- Per super: `R̄_s = mean of recalls in group`; `F̄_s = mean of fdr in group`.
- `official_recall = mean(R̄_s for s in {'ship','aircraft','vehicle'})`
- `official_fdr    = mean(F̄_s for s in {'ship','aircraft','vehicle'})`

### 2.3 Output

- Always print a block (regardless of `--out-prefix`):

  ```
  === 官方补充口径（三大类均值再平均） ===
  ship     R=0.7522  FDR=0.4691
  aircraft R=0.9685  FDR=0.1756
  vehicle  R=0.5532  FDR=0.3953
  -----------------------------------------
  official R=0.7580  FDR=0.3467
  ```

- When `--out-prefix` set, extend the `overall` dict in the JSON:

  ```python
  overall['official'] = {
      'recall': float(official_recall),
      'fdr':    float(official_fdr),
      'by_super': {
          s: {'recall': float(r), 'fdr': float(f)}
          for s, (r, f) in super_avg.items()
      },
  }
  ```

### 2.4 Edge cases

- Empty super-class: skip from the official mean, note `'(empty)'` in print.
- `--classes N < 25`: mapping still applies to whatever names exist; super may be missing → handled by 2.4.

### 2.5 Validation on existing baseline

- Re-run on baseline `work_dirs/new_data_arfc_only_v1/test_preds.json` + `data/annotations/instances_test.json`. Expect `official R=0.7580, FDR=0.3467`, matching `improvement_plan.md §1`.

---

## 3. P0-B — 推理阈值收紧 (config files)

### 3.1 Change

In both `configs/bafnet/aircraft_bafnet_1x.py` and `configs/bafnet/aircraft_bafnet_baseline_1x.py`, the `test_cfg.rcnn` block:

```python
# Before
rcnn=dict(
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=3000)

# After
rcnn=dict(
    score_thr=0.30,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=300)
```

### 3.2 Rationale

Per `improvement_plan.md §2.3`: `score_thr=0.05` admits many low-score boxes; `max_per_img=3000` wildly exceeds the dataset's ~5 targets/image. Expected FDR **−5~8pp**, no retrain.

### 3.3 Validation

Post-filter `work_dirs/new_data_arfc_only_v1/test_preds.json` by `score ≥ 0.30` and re-run `eval_recall_fdr.py` (after P0-A in). Compare official R/FDR.

### 3.4 Out of scope

- Per-class thresholds (FSC=0.20, high-FP aircraft=0.35): deferred to S3 experiment on training server.
- Per-class NMS: deferred to S3.

---

## 4. P1-A — MS 跨域采样均衡 + CopyPaste

### 4.1 Source info preservation

- MS images in `new_data/labels/train/` are named `01-PAN_*`, `02-PAN_*`, `OTHER_*`. (Per `new_data/NEW_DATA_REPORT.md §6`.)
- `tools/convert_yolo_to_coco.py` already preserves `file_name` (L87).
- **No changes needed** in the converter. Domain is recoverable from `dataset.data_infos[idx]['file_name']` at load time.

### 4.2 New `DomainBalancedDataset`

Add to `mmdet/datasets/dataset_wrappers.py`:

```python
@DATASETS.register_module()
class DomainBalancedDataset:
    """Wraps a base dataset and adds extra per-image repeats for under-represented
    sources of a target class. Works on filename prefixes.
    """

    def __init__(self,
                 dataset,
                 target_class_id=3,
                 domain_prefixes=('01-PAN', '02-PAN', 'OTHER'),
                 domain_extras=(1, 2, 2)):
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.target_class_id = target_class_id
        self.domain_prefixes = domain_prefixes
        self.domain_extras = domain_extras
        # ... build repeat_indices like ClassBalancedDataset,
        # but multiply by domain_extras for matching images.
```

Behavior:
- Base repeat indices come from inner `dataset` (typically `ClassBalancedDataset`).
- For each image index that contains `target_class_id`, detect prefix in `dataset.data_infos[idx]['file_name']`; multiply its slot count by `domain_extras[domain_idx]`.
- Default `domain_extras=(1, 2, 2)` boosts 02-PAN (688 imgs) and OTHER (604) relative to 01-PAN (1714).

Wire-up:
- Add export in `mmdet/datasets/__init__.py` and import in `mmdet/datasets/builder.py` (alongside `ClassBalancedDataset`).
- Use in v2 config's `data.train`:

  ```python
  train=dict(
      type='DomainBalancedDataset',
      target_class_id=3,
      domain_prefixes=('01-PAN', '02-PAN', 'OTHER'),
      domain_extras=(1, 2, 2),
      dataset=dict(
          type='ClassBalancedDataset',
          oversample_thr=1e-3,
          dataset=dict(
              type=dataset_type,
              ann_file=data_root + 'annotations/instances_train.json',
              img_prefix=data_root + 'images/train/',
              pipeline=train_pipeline)))
  ```

### 4.3 CopyPaste augmentation

Add to `train_pipeline` in the v2 config (after `RandomFlip`):

```python
dict(type='CopyPaste', prob=0.5),
```

If `mmdet.datasets.pipelines.copypaste.CopyPaste` is unavailable in this mmdet version (mmdet < 2.20):
- Fallback: register a custom `RandCopyPaste` (in `mmdet/datasets/pipelines/`) that:
  - randomly picks another image from the same dataset (cached at init)
  - pastes its bboxes + cropped image patches onto current sample
  - simpler than full CopyPaste; always-available.
- Document fallback choice in a code comment.

---

## 5. P1-B — 小目标召回 & 边界 loss 修复

### 5.1 Resolution & multiscale

In v2 config (which overrides `aircraft_detection.py`):

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize',
         img_scale=(1280, 1024),  # was (1280, 800)
         keep_ratio=True,
         multiscale_mode='range',
         multiscale_range=(640, 1024)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomRotate', prob=0.5, angle_range=(-30, 30)),
    dict(type='CopyPaste', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(1280, 1024),  # was (1280, 800)
         flip=False,
         transforms=[...])
]
```

If `RandomRotate` is unavailable in this mmdet version: fall back to manual 90/180/270-degree rotation transform (no interpolation, simpler than continuous-angle). Document fallback.

### 5.2 Boundary loss fix in `baf.py`

Current code (L322-323):
```python
img_side_gt = fused_gt[:, :3, :, :].max(dim=1, keepdim=True)[0]
img_side_gt = (img_side_gt > 0.1).float()
```

Re-evaluation:
- The `>0.1` threshold matches `mmdet/models/losses/detail_loss.py`'s convention (L20, L31, L33, L40). It is **not** itself broken.
- The actual issue is **binarization**: a ~99% background GT drives BCE to a plateau — model can predict all-zero and hit near-perfect accuracy, killing gradients.
- Fix: pass continuous Laplacian directly, divided by 8 (kernel sum) so the maximum (fully on-edge pixel) = 1.0, and clamped to [0, 1]. BCEWithLogits handles soft targets natively.

```python
img_side_gt = fused_gt[:, :3, :, :].max(dim=1, keepdim=True)[0]
img_side_gt = (img_side_gt / 8.0).clamp(0.0, 1.0)
# binarization removed
```

### 5.3 p2 verification

I confirmed:
- `RPNHead.anchor_generator.strides=[4,8,16,32,64]` (config L83) — covers `stride=4` (p2).
- `bbox_roi_extractor.featmap_strides=[4,8,16,32]` (config L99) — covers p2.
- **No code change needed** for plan's §P1-B "确认 p2 参与".

---

## 6. P2-A — FSC 车辆专项

### 6.1 Class-weighted CE

In v2 config, for each of the three `bbox_head[i].loss_cls`:

```python
loss_cls=dict(
    type='CrossEntropyLoss',
    use_sigmoid=False,
    loss_weight=1.0,
    class_weight=[1.0]*24 + [5.0])   # FSC index24
```

Rationale (per plan §3 P2-A): A16_FA-18 has 2148 training instances; FSC has ~428 → ratio ≈ 5×. `reg_class_agnostic=True` for all stages, so we only need to weight `loss_cls`.

### 6.2 Strong augmentation

Already covered in §5.1 — `RandomRotate` + `CopyPaste` apply globally, including to FSC images. (Plan §P2-A explicitly mentions rotation+mosaic; if `Mosaic` is unavailable in this mmdet, we defer it and rely on `RandomRotate` + `CopyPaste`.)

---

## 7. P2-B — ARFC 路由修复

### 7.1 Verified: cache is fine

`mmdet/models/utils/arfc.py:62-63`:
```python
def forward(self, x):
    logits, idx, w = self.router(x)
    self._last_router_logits = logits   # unconditional
```

And `mmdet/models/detectors/baf.py:332-335` walks modules, finds `ARFC._last_router_logits`, calls `aux_loss`. **No fix needed for the cache.**

### 7.2 Why `loss_aux = 0`

- `alpha=0.01` is small.
- Early training: router logits are near-uniform → `avg_probs ≈ 1/E` → `target*log(target/avg_probs) ≈ 0` even when unbalanced.
- Late training: if router collapses to one expert, `avg_probs` is one-hot → `target*log(target/avg_probs) = (1/E) * log(1/E / 1) < 0` → still small in absolute value × `alpha=0.01`.

### 7.3 Modify `mmdet/models/losses/load_balancing_loss.py`

Replace the class with:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class LoadBalancingLoss(nn.Module):
    """专家路由均���损失 + 路由熵正则 (Switch Transformer 风格).

    L_balance = alpha * sum_i f_i * log(f_i / P_i)
    L_entropy = beta  * (log(E) - H(P))
    return L_balance + L_entropy

    where f_i = 1/E (uniform target), P_i = avg softmax prob of expert i,
    E = num_experts, H(P) = -sum_i P_i * log P_i.

    Args:
        num_experts (int): 专家数量。
        alpha (float): Switch 均衡项权重，默认 0.1（从 0.01 上调）。
        beta (float): 熵正则权重，默认 0.01。防止路由坍缩到单一专家。
    """

    def __init__(self, num_experts, alpha=0.1, beta=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.alpha = alpha
        self.beta = beta

    def forward(self, gate_logits):
        if gate_logits.numel() == 0:
            return gate_logits.sum() * 0.0
        probs = F.softmax(gate_logits, dim=-1)            # [B, E]
        avg_probs = probs.mean(dim=0)                     # [E]
        num_experts = avg_probs.shape[0]
        target = torch.full_like(avg_probs, 1.0 / num_experts)
        balance_loss = target * (torch.log(target + 1e-9)
                                 - torch.log(avg_probs + 1e-9))
        # 路由熵正则：max-entropy = log(E)。偏差越大损失越大。
        ent = -(avg_probs * torch.log(avg_probs + 1e-9)).sum()
        ent_reg = self.beta * (math.log(num_experts) - ent)
        return self.alpha * balance_loss.sum() + ent_reg
```

### 7.4 Wire-up — also bump `baf.py:267` call site

`mmdet/models/detectors/baf.py:267` hardcodes `LoadBalancingLoss(num_experts=4, alpha=0.01)`. Changing only the default in `LoadBalancingLoss.__init__` would not affect this caller. **Also edit `baf.py:267`** to:

```python
self.aux_loss = LoadBalancingLoss(num_experts=4, alpha=0.1, beta=0.01)
```

This is the only `LoadBalancingLoss` construction in the codebase (verified via grep). Tests in `tests/test_load_balancing_loss.py` construct their own instances with explicit args, so they remain compatible.

### 7.5 Wire-up verification

At code-review time, confirm that `baf.py`'s `aux_loss` aggregation still flows into the trainer. Per CLAUDE.md `baf.py` description: "An auxiliary load-balancing loss (`loss_aux`) ... sums `LoadBalancingLoss` over its cached `_last_router_logits` ... Zero when no ARFC has routed". The change in `LoadBalancingLoss.forward` does not break this contract.

---

## 8. New config: `configs/bafnet/aircraft_bafnet_v2_1x.py`

Inherits `aircraft_bafnet_1x.py` (full model), then applies:
- `train_pipeline` & `test_pipeline` overrides (resolution + multiscale + RandomRotate + CopyPaste — §5.1)
- `data.train` switch from `ClassBalancedDataset` to `DomainBalancedDataset` (§4.2)
- `rcnn` and `bbox_head[i].loss_cls` updates from §3.1 and §6.1

Schedule (lr_config, runner) inherited unchanged. `optimizer` inherited unchanged. The new config is byte-identical to `aircraft_bafnet_1x.py` except the listed overrides.

---

## 9. Validation strategy

**On this machine (no execution per CLAUDE.md):**
- Code review only: imports, registration of new dataset class, shape of `class_weight` (=25), `_last_router_logits` still unconditional in `arfc.py`.

**On the training server:**
- **S1 (no retrain)**: re-run `eval_recall_fdr.py` on existing `test_preds.json` → confirm P0-A official R=0.7580 / FDR=0.3467; then post-filter by score≥0.30 and re-run → confirm FDR drop 5–8pp.
- **S2 (retrain)**: train `aircraft_bafnet_v2_1x.py` (~12h on 1×3090), run test, evaluate per §6 of plan.
- **S3 (retrain + threshold tuning)**: further refine `score_thr` per-class based on S2 metrics.

---

## 10. Risk register

| Risk | Mitigation |
|---|---|
| HM/LQS metrics unstable (single-digit test) | Report "ship-mean excluding HM/LQS" alongside |
| Inference time > 20s on 1e4² images | Out of scope for code PR; S4 separate |
| ARFC entropy reg may hurt high-freq aircraft precision | β=0.01 conservative; ablation on training server |
| `CopyPaste` unavailable in this mmdet version | Fallback `RandCopyPaste` custom transform |
| `RandomRotate` unavailable in this mmdet version | Fall back to 90/180/270-discrete rotation |
| `Mosaic` unavailable in this mmdet version | Defer; rely on `RandomRotate`+`CopyPaste` |

---

## 11. Out of scope

- SBLA retraining (`sbla.enabled=True`) — separate experiment, plan §P2 doesn't include it.
- RFP `rfp_steps=2→1` for inference speedup — S4 concern.
- Per-class NMS — S3 concern.
- Inference time benchmark — S4.
- `loss_bounder` *value* re-tuning beyond the binarization fix — server-side hyperparameter sweep.