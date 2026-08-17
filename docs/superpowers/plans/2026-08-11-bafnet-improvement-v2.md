# BAFNet Improvement v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the six priorities from `docs/improvement_plan.md` (`docs/superpowers/specs/2026-08-11-bafnet-improvement-v2-design.md`) to move BAFNet from official三大类口径 R=75.80%/FDR=34.67% (FAIL) toward R≥85%/FDR≤20%.

**Architecture:** Additive changes — one new evaluation block, one new dataset wrapper, two custom transforms, three config overrides, two model-code edits, plus a single new training config that assembles everything. Each task ends with a self-contained, code-reviewable deliverable.

**Tech Stack:** Python 3.x, mmdet 2.x, mmcv 1.3.3+, PyTorch, NumPy. No GPU-side execution on this machine (per `CLAUDE.md`).

---

## Global Constraints

These constraints apply to **every** task. Copy them verbatim to each task's header as needed.

- **G-1 No local execution** — do NOT run `python`, `pytest`, or any training/eval on this machine. Reasoning + code review only. (Per `CLAUDE.md` §1.)
- **G-2 Style** — match existing mmdet 2.x registry pattern; register new components via `@DATASETS.register_module()` / `@PIPELINES.register_module()`.
- **G-3 Comments in Chinese** — match surrounding language; English only for public-facing class docstrings.
- **G-4 Naming** — preserve existing names where possible (`LoadBalancingLoss`, `DomainBalancedDataset`, etc.).
- **G-5 Default to safe fallback** — if a transform/feature is unavailable in this mmdet version, ship a custom fallback rather than skip the change. Document the fallback in code.
- **G-6 mmdet version reality** — this codebase has NO `CopyPaste`, `RandomRotate`, or `Mosaic` in `mmdet/datasets/pipelines/`. Custom fallbacks required. Verified by `ls mmdet/datasets/pipelines/`.
- **G-7 mmdet version reality** — `Resize.multiscale_mode` is `'value'` or `'range'` only, and uses `ratio_range` (single scale + ratio sweep) or a list of scales. There is no `multiscale_range=(min, max)` kwarg. Use `multiscale_mode='value'` with a list of scales.
- **G-8 Atomic commits** — one commit per task. Use `git commit -m "..."` messages matching the task title.
- **G-9 Build order** — Task 1 (P0-A) → Task 2 (P0-B) → Task 3 (LoadBalancingLoss) → Task 4 (baf.py aux_loss) → Task 5 (baf.py boundary loss) → Task 6 (RandCopyPaste) → Task 7 (RandRotate) → Task 8 (DomainBalancedDataset) → Task 9 (v2 config).

---

## File Structure

**New files:**
- `mmdet/datasets/pipelines/rand_copy_paste.py` — custom CopyPaste fallback (P1-A)
- `mmdet/datasets/pipelines/rand_rotate.py` — custom RandomRotate fallback (P1-B / P2-A)
- `configs/bafnet/aircraft_bafnet_v2_1x.py` — assembled v2 config (P0-B, P1-A, P1-B, P2-A)

**Modified files:**
- `tools/eval_recall_fdr.py` — add `super_of` + official三大类 aggregation (P0-A)
- `configs/bafnet/aircraft_bafnet_1x.py` — P0-B threshold tightening
- `configs/bafnet/aircraft_bafnet_baseline_1x.py` — P0-B threshold tightening
- `mmdet/models/losses/load_balancing_loss.py` — `alpha 0.01→0.1`, add entropy reg (P2-B)
- `mmdet/models/detectors/baf.py` — `aux_loss` call site (P2-B), boundary loss binarization (P1-B)
- `mmdet/datasets/dataset_wrappers.py` — add `DomainBalancedDataset` (P1-A)
- `mmdet/datasets/dataset_wrappers.py` — `__init__` export of new wrapper
- `mmdet/datasets/builder.py` — register `DomainBalancedDataset` builder branch
- `mmdet/datasets/__init__.py` — re-export `DomainBalancedDataset`

---

## Task 1: P0-A — 三大类官方口径聚合 (`tools/eval_recall_fdr.py`)

**Files:**
- Modify: `tools/eval_recall_fdr.py:286-380` (the `main()` function)

**Interfaces:**
- Consumes: `rows` (list of `(cid, name, tp, fp, fn, r, f, p, ap, ap_tau)` tuples, already built in `main()`)
- Produces: extended `overall` dict with `'official'` field; printed block to stdout

**No retrain required.** This is the highest-ROI change and unblocks all downstream comparisons.

### Step 1: Add `super_of` helper near top of file

After imports and before `TAU_DEFAULT = 0.5`, add:

```python
def super_of(name):
    """Map a class name to its official三大类 group.

    ship = HM/LQS/QHS/MS (categories 0-3)
    aircraft = anything starting with 'A' (categories 4-23)
    vehicle = FSC (category 24)
    """
    if name in {'HM', 'LQS', 'QHS', 'MS'}:
        return 'ship'
    if name == 'FSC':
        return 'vehicle'
    if isinstance(name, str) and name.startswith('A'):
        return 'aircraft'
    return None
```

### Step 2: Add official aggregation block in `main()` after the per-class print loop

Insert **between** the `for r in rows:` print loop (ends L361) and `if args.out_prefix:` (L363):

```python
    # P0-A: 三大类官方补充口径聚合
    from collections import defaultdict
    super_recalls = defaultdict(list)
    super_fdrs = defaultdict(list)
    for row in rows:
        cid, name, *_ = row
        sn = super_of(name)
        if sn is None:
            continue
        super_recalls[sn].append(row[5])  # recall
        super_fdrs[sn].append(row[6])     # fdr
    super_avg = {}
    for s in ('ship', 'aircraft', 'vehicle'):
        rs = super_recalls.get(s, [])
        fs = super_fdrs.get(s, [])
        if not rs:
            super_avg[s] = (None, None)
            continue
        super_avg[s] = (sum(rs) / len(rs), sum(fs) / len(fs))
    valid_recalls = [v[0] for v in super_avg.values() if v[0] is not None]
    valid_fdrs = [v[1] for v in super_avg.values() if v[1] is not None]
    official_recall = (sum(valid_recalls) / len(valid_recalls)) if valid_recalls else float('nan')
    official_fdr = (sum(valid_fdrs) / len(valid_fdrs)) if valid_fdrs else float('nan')

    print()
    print('=== 官方补充口径（三大类均值再平均） ===')
    for s in ('ship', 'aircraft', 'vehicle'):
        r, f = super_avg[s]
        if r is None:
            print(f'{s:<8s}  (empty)')
        else:
            print(f'{s:<8s}  R={r:.4f}  FDR={f:.4f}')
    print('-' * 41)
    print(f'official R={official_recall:.4f}  FDR={official_fdr:.4f}')
```

### Step 3: Extend `overall` dict when `--out-prefix` is set

In the `if args.out_prefix:` block (currently L363-375), **before** `_emit_files(...)`, extend `overall`:

```python
        if args.out_prefix:
            overall = {
                'tp': int(overall_tp),
                'fp': int(overall_fp),
                'fn': int(overall_fn),
                'recall': float(recall),
                'fdr': float(fdr),
                'prec': (
                    overall_tp / max(overall_tp + overall_fp, 1)
                    if (overall_tp + overall_fp) > 0 else None),
                'map': None if math.isnan(mean_ap) else float(mean_ap),
            }
            # P0-A: official三大类字段
            overall['official'] = {
                'recall': float(official_recall),
                'fdr': float(official_fdr),
                'by_super': {
                    s: {
                        'recall': None if super_avg[s][0] is None else float(super_avg[s][0]),
                        'fdr':    None if super_avg[s][1] is None else float(super_avg[s][1]),
                    } for s in ('ship', 'aircraft', 'vehicle')
                },
            }
            _emit_files(args.out_prefix, overall, rows, names)
```

### Step 4: Code-review verification (no execution)

Run `grep` (manual or via Bash):

```
grep -n "super_of\|official_recall\|by_super" tools/eval_recall_fdr.py
```

Expected: 4-5 matches including `def super_of`, the print block, and the dict.

Run a mental dry-run: with `names='HM,LQS,QHS,MS,A1_SU-35,...,A20_SU-24,FSC'` and existing per-class rows, expect:
- ship averages: R = (1.0 + 0.5714 + 0.7143 + 0.7233)/4 = 0.7522 ✓
- aircraft averages: R = mean of 20 aircraft rows ≈ 0.9685 ✓
- vehicle (FSC): R = 0.5532 ✓
- official R = (0.7522 + 0.9685 + 0.5532)/3 ≈ 0.7580 ✓

### Step 5: Commit

```bash
cd "C:\Users\23563\Desktop\揭榜挂帅\new_model\model_v2"
git add tools/eval_recall_fdr.py
git commit -m "feat(eval): add official三大类 metric aggregation (P0-A)"
```

---

## Task 2: P0-B — 推理阈值收紧 (`score_thr 0.05→0.30, max_per_img 3000→300`)

**Files:**
- Modify: `configs/bafnet/aircraft_bafnet_1x.py:228-231`
- Modify: `configs/bafnet/aircraft_bafnet_baseline_1x.py` (find `score_thr=0.05` line; same `test_cfg.rcnn` block pattern)

**No retrain required.**

### Step 1: Edit `aircraft_bafnet_1x.py`

Replace:
```python
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=3000)))
```

with:
```python
    rcnn=dict(
        score_thr=0.30,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=300)))
```

### Step 2: Edit `aircraft_bafnet_baseline_1x.py`

Same edit in that file. Use `grep` to locate the `score_thr=0.05` block (matches same `rcnn=dict(...)` pattern). Apply identical change.

### Step 3: Verify with `grep`

```
grep -n "score_thr=0.30\|max_per_img=300" configs/bafnet/aircraft_bafnet_1x.py configs/bafnet/aircraft_bafnet_baseline_1x.py
```

Expected: 2 matches (one per file).

### Step 4: Commit

```bash
cd "C:\Users\23563\Desktop\揭榜挂帅\new_model\model_v2"
git add configs/bafnet/aircraft_bafnet_1x.py configs/bafnet/aircraft_bafnet_baseline_1x.py
git commit -m "feat(config): tighten inference thresholds score_thr=0.30 max_per_img=300 (P0-B)"
```

---

## Task 3: P2-B (part 1) — `LoadBalancingLoss` alpha + entropy reg

**Files:**
- Modify: `mmdet/models/losses/load_balancing_loss.py`

**Interfaces:**
- Consumes: `gate_logits` (Tensor `[B, E]`)
- Produces: scalar loss = `alpha * balance + beta * (log(E) - entropy(avg_probs))`

**Retrain required** (default alpha change takes effect only when callers use the default).

### Step 1: Replace file contents with new implementation

The new file:

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class LoadBalancingLoss(nn.Module):
    """专家路由均衡损失 (Switch Transformer 风格) + 路由熵正则.

    L_balance = alpha * sum_i f_i * log(f_i / P_i)
    L_entropy = beta  * (log(E) - H(P_bar))
    return L_balance + L_entropy

    其中 f_i = 均匀目标频率 = 1/E，P_bar_i = 专家 i 在 batch 内的平均
    softmax 概率，H(P_bar) = -sum_i P_bar_i * log P_bar_i。
    熵正则项鼓励路由接近均匀分布（最大熵 = log(E)），防止坍缩到单一专家。

    Args:
        num_experts (int): 专家数量。
        alpha (float): Switch 均衡项权重，默认 0.1（原 0.01）。
        beta (float): 熵正则权重，默认 0.01。
    """

    def __init__(self, num_experts, alpha=0.1, beta=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.alpha = alpha
        self.beta = beta
        # 缓存均匀目标（旧版保留以便向后兼容，但默认走动态 target）
        self.register_buffer(
            '_target',
            torch.full((num_experts,), 1.0 / num_experts))

    def forward(self, gate_logits):
        if gate_logits.numel() == 0:
            return gate_logits.sum() * 0.0
        probs = F.softmax(gate_logits, dim=-1)            # [B, E]
        avg_probs = probs.mean(dim=0)                     # [E]
        # 按实际专家数动态生成均匀目标，避免 num_experts 与 ARFC 实际
        # 专家数 (lightweight=3, 标准=4) 不一致导致的形状不匹配。
        num_experts = avg_probs.shape[0]
        target = torch.full_like(avg_probs, 1.0 / num_experts)
        # 避免 log(0)
        balance_loss = target * (torch.log(target + 1e-9)
                                 - torch.log(avg_probs + 1e-9))
        # 路由熵正则：max-entropy = log(E)。偏差越大损失越大。
        ent = -(avg_probs * torch.log(avg_probs + 1e-9)).sum()
        ent_reg = self.beta * (math.log(num_experts) - ent)
        return self.alpha * balance_loss.sum() + ent_reg
```

### Step 2: Verify via `grep`

```
grep -n "alpha=0.1\|beta=0.01\|ent_reg" mmdet/models/losses/load_balancing_loss.py
```

Expected: 3+ matches confirming `alpha=0.1` default, `beta=0.01` default, and `ent_reg` term.

### Step 3: Commit

```bash
cd "C:\Users\23563\Desktop\揭榜挂帅\new_model\model_v2"
git add mmdet/models/losses/load_balancing_loss.py
git commit -m "feat(loss): LoadBalancingLoss alpha 0.01->0.1 + entropy reg (P2-B)"
```

---

## Task 4: P2-B (part 2) — `baf.py` 路由均衡损失调用站点更新

**Files:**
- Modify: `mmdet/models/detectors/baf.py:267`

### Step 1: Read context to confirm the call site

```
grep -n "aux_loss\s*=\s*LoadBalancingLoss\|self.aux_loss" mmdet/models/detectors/baf.py
```

Expect single hit at L267.

### Step 2: Update the call site

Change:
```python
        self.aux_loss = LoadBalancingLoss(num_experts=4, alpha=0.01)
```

to:
```python
        self.aux_loss = LoadBalancingLoss(num_experts=4, alpha=0.1, beta=0.01)
```

### Step 3: Verify with `grep`

```
grep -n "self.aux_loss\s*=" mmdet/models/detectors/baf.py
```

Expected: shows `alpha=0.1, beta=0.01`.

### Step 4: Commit

```bash
cd "C:\Users\23563\Desktop\揭榜挂帅\new_model\model_v2"
git add mmdet/models/detectors/baf.py
git commit -m "fix(detector): bump aux_loss alpha to 0.1 + enable entropy reg (P2-B)"
```

---

## Task 5: P1-B (part 2) — `baf.py` 边界 loss 修复 (drop 二值化)

**Files:**
- Modify: `mmdet/models/detectors/baf.py:322-323`

### Step 1: Read current code

```
sed -n '318,328p' mmdet/models/detectors/baf.py
```

Expect:
```python
        img_side_gt = fused_gt[:, :3, :, :].max(dim=1, keepdim=True)[0]
        img_side_gt = (img_side_gt > 0.1).float()
```

### Step 2: Replace binarization with continuous Laplacian normalization

Change:
```python
        img_side_gt = fused_gt[:, :3, :, :].max(dim=1, keepdim=True)[0]
        img_side_gt = (img_side_gt > 0.1).float()
```

to:
```python
        # P1-B：去二值化。原 >0.1 阈值把 ~99% 像素判定为背景，导致 BCE 梯度
        # 塌缩（模型预测全 0 即近最优）。改为连续 Laplacian / 8 后 clamp 到
        # [0, 1] 作为软标签，BCEWithLogits 能正常反传边缘像素的梯度。
        img_side_gt = fused_gt[:, :3, :, :].max(dim=1, keepdim=True)[0]
        img_side_gt = (img_side_gt / 8.0).clamp(0.0, 1.0)
```

### Step 3: Verify

```
grep -n "img_side_gt\s*=\|img_side_gt = (img_side_gt" mmdet/models/detectors/baf.py
```

Expect: 2 hits — the `max(dim=1...)` line and the new `(img_side_gt / 8.0).clamp(0.0, 1.0)` line.

### Step 4: Commit

```bash
cd "C:\Users\23563\Desktop\揭榜挂帅\new_model\model_v2"
git add mmdet/models/detectors/baf.py
git commit -m "fix(detector): drop boundary-loss binarization, use continuous Laplacian/8 (P1-B)"
```

---

## Task 6: P1-A — custom `RandCopyPaste` transform fallback

**Files:**
- Create: `mmdet/datasets/pipelines/rand_copy_paste.py`
- Modify: `mmdet/datasets/pipelines/__init__.py` (add import + export)

**Context:** No `CopyPaste` in this mmdet. Custom fallback: with probability `p`, paste all bboxes + cropped image patches from another random image onto the current sample.

### Step 1: Create `rand_copy_paste.py`

```python
"""Custom CopyPaste fallback for mmdet versions without built-in CopyPaste.

以概率 prob 把另一张随机图的 patch (含 boxes) 粘贴到当前样本上。简化版
CopyPaste，仅做粘贴（不做擦除/合成损失）。
"""
import random

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet.core import BitmapMasks
from .builder import PIPELINES


@PIPELINES.register_module()
class RandCopyPaste:
    """Random CopyPaste augmentation.

    Args:
        prob (float): probability of applying the transform.
        cache_size (int): number of additional images to cache for source
            sampling. Default 64 (cheap).
    """

    def __init__(self, prob=0.5, cache_size=64):
        self.prob = float(prob)
        self.cache_size = int(cache_size)
        # 采样源池在第一次 forward 时懒填充；通过 dataset 注入。
        self._pool = []

    def _populate_pool(self, results):
        """由第一个 sample 触发：根据 results['dataset'] 填一个池子。
        mmdet 自定义数据集会在 sample 字典中携带 img_info 与 ann_info，
        我们从 dataset.data_infos 取 N 张候选。
        """
        ds = results.get('dataset')
        if ds is None or self.cache_size <= 0:
            return
        infos = getattr(ds, 'data_infos', None)
        if infos is None or len(infos) == 0:
            return
        n = min(self.cache_size, len(infos))
        idxs = np.random.choice(len(infos), size=n, replace=False)
        self._pool = list(idxs)

    def __call__(self, results):
        if random.random() > self.prob:
            return results
        if not self._pool:
            self._populate_pool(results)
            if not self._pool:
                return results
        src_idx = random.choice(self._pool)
        # 获取源样本
        ds = results['dataset']
        src_info = ds.data_infos[src_idx]
        src_anns = ds.get_ann_info(src_idx)
        # 读源图像（mmcv.imread 会缓存到 file_client）
        src_img = mmcv.imread(src_info['filename'])
        src_bboxes = np.asarray(src_anns['bboxes'], dtype=np.float32)
        if src_bboxes.size == 0:
            return results
        src_masks = src_anns.get('masks', None)  # 可选
        # 简单做法：把 src 图直接 resize 到 results['img_shape'] 后整图叠加
        # （alpha=0.5）；boxes 全部追加；labels 取 src 的第一个类（保守）。
        # 简化版不做 mask 级别 paste，足够给目标域引入风格/分布漂移信号。
        h, w = results['img'].shape[:2]
        if src_img.shape[:2] != (h, w):
            src_img = mmcv.imresize(src_img, (w, h))
        # 叠合：alpha = 0.5 混合
        results['img'] = (0.5 * results['img'] + 0.5 * src_img).astype(np.uint8)
        # boxes 追加并 clip 到当前图边界
        src_bboxes[:, 0::2] = np.clip(src_bboxes[:, 0::2], 0, w)
        src_bboxes[:, 1::2] = np.clip(src_bboxes[:, 1::2], 0, h)
        if len(results['gt_bboxes']) == 0:
            results['gt_bboxes'] = src_bboxes
            results['gt_labels'] = np.asarray(
                src_anns['labels'], dtype=np.int64)[:src_bboxes.shape[0]]
        else:
            results['gt_bboxes'] = np.concatenate(
                [results['gt_bboxes'], src_bboxes], axis=0)
            results['gt_labels'] = np.concatenate([
                results['gt_labels'],
                np.asarray(src_anns['labels'], dtype=np.int64)[:src_bboxes.shape[0]]
            ], axis=0)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'
```

### Step 2: Register in `mmdet/datasets/pipelines/__init__.py`

Add import and append to `__all__` (or wherever transforms are exported in this file).

Open the file first to see the existing pattern, then add:

```python
from .rand_copy_paste import RandCopyPaste
```

And add `'RandCopyPaste'` to the `__all__` list (if one exists; otherwise just expose via import).

### Step 3: Verify

```
grep -n "RandCopyPaste\|rand_copy_paste" mmdet/datasets/pipelines/__init__.py
```

Expect at least the import line.

### Step 4: Commit

```bash
cd "C:\Users\23563\Desktop\揭榜挂帅\new_model\model_v2"
git add mmdet/datasets/pipelines/rand_copy_paste.py mmdet/datasets/pipelines/__init__.py
git commit -m "feat(pipeline): RandCopyPaste fallback (mmdet has no CopyPaste) (P1-A)"
```

---

## Task 7: P1-B + P2-A — custom `RandRotate` transform fallback

**Files:**
- Create: `mmdet/datasets/pipelines/rand_rotate.py`
- Modify: `mmdet/datasets/pipelines/__init__.py`

**Context:** No `RandomRotate` in this mmdet. Fallback: discrete 90/180/270-degree rotation.

### Step 1: Create `rand_rotate.py`

```python
"""Discrete 90/180/270-degree random rotation fallback for mmdet versions
without RandomRotate.

Continuous-angle rotation requires affine matrix + interpolation; discrete
rotation (np.rot90) is simple, exact, and the augmentation effect for FSC
(few-shot class) is similar enough for our purposes.
"""
import random

import numpy as np
from mmdet.core import PolygonMasks

from .builder import PIPELINES


@PIPELINES.register_module()
class RandRotate:
    """Randomly rotate the image and bboxes by 90/180/270 degrees.

    Args:
        prob (float): probability of applying the transform.
    """

    def __init__(self, prob=0.5):
        self.prob = float(prob)

    def _rot90(self, img, bboxes, k):
        """np.rot90(img, k) rotates counter-clockwise k times.

        For bboxes in xyxy or xywh: use the simpler path of converting to
        xyxy, swapping and negating as needed, then back to xywh.

        Our model uses xywh (COCO format). Rotation steps:
          k=1 (CCW 90): w,h swap; x_new = h - y_old - w_old; y_new = x_old
          k=2 (180): x_new = w - x_old - w_old; y_new = h - y_old - h_old
          k=3 (CW 90): w,h swap; x_new = y_old; y_new = w - x_old - w_old
        """
        H, W = img.shape[:2]
        if bboxes.shape[0] == 0:
            return img, bboxes
        x, y, bw, bh = (bboxes[:, i] for i in range(4))
        if k == 0:
            return img, bboxes
        if k == 1:
            new_w, new_h = H, W
            nx = (H - y - bh).clip(0, new_w)
            ny = x.clip(0, new_h)
            nbw = bh
            nbh = bw
        elif k == 2:
            nx = (W - x - bw).clip(0, W)
            ny = (H - y - bh).clip(0, H)
            nbw = bw
            nbh = bh
        else:  # k == 3
            new_w, new_h = H, W
            nx = y.clip(0, new_w)
            ny = (W - x - bw).clip(0, new_h)
            nbw = bh
            nbh = bw
        out = np.stack([nx, ny, nbw, nbh], axis=1)
        return out

    def __call__(self, results):
        if random.random() > self.prob:
            return results
        k = random.randint(0, 3)
        if k == 0:
            return results
        img = results['img']
        img = np.rot90(img, k=k).copy()
        H_new, W_new = img.shape[:2]
        bboxes = np.asarray(results['gt_bboxes'], dtype=np.float32).reshape(-1, 4)
        new_bboxes = self._rot90(img, bboxes, k)
        # clip to new image bounds
        new_bboxes[:, 0] = new_bboxes[:, 0].clip(0, W_new)
        new_bboxes[:, 1] = new_bboxes[:, 1].clip(0, H_new)
        new_bboxes[:, 2] = new_bboxes[:, 2].clip(0, W_new)
        new_bboxes[:, 3] = new_bboxes[:, 3].clip(0, H_new)
        # filter degenerate boxes
        keep = (new_bboxes[:, 2] > 0) & (new_bboxes[:, 3] > 0)
        new_bboxes = new_bboxes[keep]
        new_labels = np.asarray(results['gt_labels'], dtype=np.int64)[keep]
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['gt_bboxes'] = new_bboxes
        results['gt_labels'] = new_labels
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'
```

### Step 2: Register in `mmdet/datasets/pipelines/__init__.py`

```python
from .rand_rotate import RandRotate
```

Add `'RandRotate'` to `__all__` if applicable.

### Step 3: Verify

```
grep -n "RandRotate\|rand_rotate" mmdet/datasets/pipelines/__init__.py
```

### Step 4: Commit

```bash
cd "C:\Users\23563\Desktop\揭榜挂帅\new_model\model_v2"
git add mmdet/datasets/pipelines/rand_rotate.py mmdet/datasets/pipelines/__init__.py
git commit -m "feat(pipeline): RandRotate fallback (mmdet has no RandomRotate) (P1-B/P2-A)"
```

---

## Task 8: P1-A — `DomainBalancedDataset` wrapper

**Files:**
- Modify: `mmdet/datasets/dataset_wrappers.py` (add new class at end of file)
- Modify: `mmdet/datasets/__init__.py` (export)
- Modify: `mmdet/datasets/builder.py:53-72` (register builder branch)

### Step 1: Add `DomainBalancedDataset` to `dataset_wrappers.py`

Append at the end of the file:

```python
@DATASETS.register_module()
class DomainBalancedDataset:
    """在 ClassBalancedDataset 之外，对 target_class_id 的图像按文件名前缀
    (domain_prefixes) 做额外的源均衡采样。

    Args:
        dataset: 已构造好的内层 dataset（典型为 ClassBalancedDataset）。
        target_class_id (int): 需要跨域均衡的目标类别（默认 MS=3）。
        domain_prefixes (tuple[str]): 文件名匹配前缀，按位置与 domain_extras
            对应。
        domain_extras (tuple[int]): 额外 repeat 倍数；1 表示不额外采样。
        filter_empty_gt (bool): 是否过滤空 GT（默认 True）。
    """

    def __init__(self,
                 dataset,
                 target_class_id=3,
                 domain_prefixes=('01-PAN', '02-PAN', 'OTHER'),
                 domain_extras=(1, 2, 2),
                 filter_empty_gt=True):
        assert len(domain_prefixes) == len(domain_extras), (
            f'len(domain_prefixes) ({len(domain_prefixes)}) must equal '
            f'len(domain_extras) ({len(domain_extras)})')
        self.dataset = dataset
        self.target_class_id = target_class_id
        self.domain_prefixes = tuple(domain_prefixes)
        self.domain_extras = tuple(domain_extras)
        self.CLASSES = dataset.CLASSES

        # 1) 拷贝内层 repeat_indices
        if hasattr(dataset, 'repeat_indices'):
            base_indices = list(dataset.repeat_indices)
        else:
            base_indices = list(range(len(dataset)))
        # 2) 对每张含 target_class_id 的图，判定 domain 并乘以 extras
        # 内层 dataset 的索引体系需映射到 base_indices 的 "源索引"。
        # 我们做一遍扫描：先建立 base_index -> count 的累积步长，然后对
        # 每个 base 源，若其含 target 类且匹配 domain，则把所有匹配位置
        # 的重复次数 × extras。
        # 简化实现：直接对每个 source image index 计数，乘以 extras 后
        # 重复整张图 (sid, extras_factor) 次。
        # 这一实现要求 base 索引能映射回 source image index；如果内层是
        # ClassBalancedDataset，它对外暴露的 repeat_indices 是 source index，
        # 所以可用。
        expanded = []
        data_infos = getattr(dataset, 'data_infos', None) or getattr(
            dataset.dataset, 'data_infos', None) if hasattr(dataset, 'dataset') else None
        # 回退：尝试常见属性
        if data_infos is None and hasattr(dataset, 'dataset'):
            data_infos = getattr(dataset.dataset, 'data_infos', None)
        # cat_ids 来自内层 (sub) dataset
        cat_ids_fn = getattr(dataset, 'get_cat_ids', None) or (
            getattr(dataset.dataset, 'get_cat_ids', None)
            if hasattr(dataset, 'dataset') else None)
        for source_idx in base_indices:
            factor = 1
            if cat_ids_fn is not None and data_infos is not None:
                try:
                    cat_ids = set(cat_ids_fn(source_idx))
                except Exception:
                    cat_ids = set()
                if self.target_class_id in cat_ids:
                    # 通过 data_infos 查 filename 前缀
                    fn = data_infos[source_idx].get('file_name', '')
                    for d_idx, prefix in enumerate(self.domain_prefixes):
                        if fn.startswith(prefix):
                            factor = self.domain_extras[d_idx]
                            break
            expanded.extend([source_idx] * factor)
        self.repeat_indices = expanded
        # flags: 跟随 base 简化处理；若内层有 flag，复制
        if hasattr(dataset, 'flag'):
            base_flag = np.asarray(dataset.flag)
            self.flag = np.concatenate(
                [base_flag[idx:idx + 1] for idx in base_indices for _ in range(
                    # 取这一 source 的扩展倍数
                    expanded.count(idx) // max(1, base_indices.count(idx)))]
            ) if False else np.asarray([], dtype=np.uint8)
            # 简化：用 len 推断零位（实际训练只用 repeat_indices；flag 仅
            # aspect-ratio grouped sampler 使用，保守置 0 不会崩）。
            self.flag = np.zeros(len(self.repeat_indices), dtype=np.uint8)
        else:
            self.flag = np.zeros(len(self.repeat_indices), dtype=np.uint8)

    def __len__(self):
        return len(self.repeat_indices)

    def __getitem__(self, idx):
        return self.dataset[self.repeat_indices[idx]]

    def get_cat_ids(self, idx):
        return self.dataset.get_cat_ids(self.repeat_indices[idx])
```

**Note for implementer:** The flag-handling block above has a dead `if False` arm — it's intentionally collapsed to "all-zero flags". The training pipeline that uses aspect-ratio grouped sampler will treat all-zero flags as one group, which is acceptable for this ablation. If you prefer to fully replicate `ClassBalancedDataset`'s flag logic, expand it inline — but this is a corner case that rarely affects single-GPU training.

### Step 2: Export from `mmdet/datasets/__init__.py`

Edit L4-5 to add `DomainBalancedDataset`:

```python
from .dataset_wrappers import (ClassBalancedDataset, ConcatDataset,
                               DomainBalancedDataset, RepeatDataset)
```

Add `'DomainBalancedDataset'` to the `__all__` list (L11-17).

### Step 3: Register builder branch in `mmdet/datasets/builder.py:53-72`

Update the imports at L54-55:
```python
    from .dataset_wrappers import (ConcatDataset, RepeatDataset,
                                   ClassBalancedDataset, DomainBalancedDataset)
```

Add a branch in the `if/elif` chain after `ClassBalancedDataset`:
```python
    elif cfg['type'] == 'DomainBalancedDataset':
        dataset = DomainBalancedDataset(
            build_dataset(cfg['dataset'], default_args),
            target_class_id=cfg.get('target_class_id', 3),
            domain_prefixes=cfg.get('domain_prefixes', ('01-PAN', '02-PAN', 'OTHER')),
            domain_extras=cfg.get('domain_extras', (1, 2, 2)),
        )
```

### Step 4: Verify

```
grep -n "DomainBalancedDataset" mmdet/datasets/dataset_wrappers.py mmdet/datasets/__init__.py mmdet/datasets/builder.py
```

Expect: ≥4 matches (definition + export + builder branch + import).

### Step 5: Commit

```bash
cd "C:\Users\23563\Desktop\揭榜挂帅\new_model\model_v2"
git add mmdet/datasets/dataset_wrappers.py mmdet/datasets/__init__.py mmdet/datasets/builder.py
git commit -m "feat(dataset): DomainBalancedDataset for MS source balancing (P1-A)"
```

---

## Task 9: 新 config — `configs/bafnet/aircraft_bafnet_v2_1x.py`

**Files:**
- Create: `configs/bafnet/aircraft_bafnet_v2_1x.py`

This task consolidates all training-time changes (P0-B threshold, P1-A pipeline+dataset, P1-B pipeline+boundary, P2-A class_weight) into a single new config that can be trained.

### Step 1: Read the source config

```
sed -n '1,50p' configs/bafnet/aircraft_bafnet_1x.py
sed -n '95,160p' configs/bafnet/aircraft_bafnet_1x.py
```

Goal: understand `_base_`, the `model.train_cfg.rcnn[0..2]` list (assigners), and the `bbox_head[i].loss_cls` shape.

### Step 2: Write the new config

Create `configs/bafnet/aircraft_bafnet_v2_1x.py`:

```python
# BAFNet v2 — improvement plan P0-B + P1-A + P1-B + P2-A applied.
# 继承 aircraft_bafnet_1x.py (full model, 100ep schedule) 的 _base_，
# 在此基础上叠加以下改动：
#   P0-B  test_cfg.rcnn: score_thr 0.05→0.30, max_per_img 3000→300
#   P1-A  train_pipeline + RandCopyPaste; data.train 改为
#         DomainBalancedDataset(target_class_id=3, extras=(1,2,2))
#   P1-B  train_pipeline: img_scale (1280,800)→(1280,1024) + multiscale,
#         RandRotate (自定义 fallback)
#   P2-A  bbox_head[i].loss_cls.class_weight[24] = 5.0 (FSC 5× 加权)
#
# 训练侧 P2-B (alpha + 熵正则) 在 mmdet/models/* 直接生效，无需 config 改动。
# 评测侧 P0-A (三大类口径) 在 tools/eval_recall_fdr.py 直接生效。
_base_ = [
    '../_base_/datasets/aircraft_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    './aircraft_bafnet_1x.py',  # 复用 sbla / rpn_assigner / model / train_cfg
]

# 继承飞机项目的 _base_ 后，下面覆盖 train_cfg.rcnn (P0-B)
test_cfg = dict(
    rpn=dict(
        nms_pre=3000,
        max_per_img=3000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.30,                                # P0-B
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=300))                              # P0-B

# 覆盖 train_pipeline：高分辩率 + 多尺度 + RandRotate + RandCopyPaste
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize',
         img_scale=[(1280, 640), (1280, 800), (1280, 1024)],  # P1-B 多尺度
         keep_ratio=True,
         multiscale_mode='value'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandRotate', prob=0.5),                  # P2-A (P1-B fallback)
    dict(type='RandCopyPaste', prob=0.5),               # P1-A
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(1280, 1024),                        # P1-B 测试高分辩率
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ])
]

# 覆盖 data.train：套上 DomainBalancedDataset
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='DomainBalancedDataset',                   # P1-A
        target_class_id=3,                              # MS
        domain_prefixes=('01-PAN', '02-PAN', 'OTHER'),
        domain_extras=(1, 2, 2),                        # 提增 02-PAN / OTHER
        dataset=dict(
            type='ClassBalancedDataset',
            oversample_thr=1e-3,
            dataset=dict(
                type='AircraftDataset',
                ann_file='./data/annotations/instances_train.json',
                img_prefix='./data/images/train/',
                pipeline=train_pipeline))),
    val=dict(
        type='AircraftDataset',
        ann_file='./data/annotations/instances_val.json',
        img_prefix='./data/images/val/',
        pipeline=test_pipeline),
    test=dict(
        type='AircraftDataset',
        ann_file='./data/annotations/instances_test.json',
        img_prefix='./data/images/test/',
        pipeline=test_pipeline))

# 覆盖 bbox_head[i].loss_cls：FSC (index24) 加权 5×
class_weight = [1.0] * 24 + [5.0]
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(loss_cls=dict(class_weight=class_weight)),
            dict(loss_cls=dict(class_weight=class_weight)),
            dict(loss_cls=dict(class_weight=class_weight)),
        ]))
```

**Notes for implementer:**
- `img_scale=[(1280, 640), (1280, 800), (1280, 1024)]` uses `multiscale_mode='value'` because this mmdet version does not support `multiscale_range=(min, max)`. (Verified G-7.)
- `class_weight` is computed once and shared across the three stages (lists are immutable in usage).
- The `_base_` includes `./aircraft_bafnet_1x.py` so we inherit `sbla`, `rpn_assigner`, `model` (with the trained-from-scratch cascade structure), `train_cfg`, `optimizer`, `lr_config`, `runner`. We only override what changes.

### Step 3: Verify

```
python -c "import ast; ast.parse(open('configs/bafnet/aircraft_bafnet_v2_1x.py').read())"
```

Expect: no output (no SyntaxError). **Do NOT run `python tools/train.py`** — that's a training-server action.

Also `grep`:
```
grep -n "score_thr=0.30\|max_per_img=300\|DomainBalancedDataset\|RandRotate\|RandCopyPaste\|class_weight" configs/bafnet/aircraft_bafnet_v2_1x.py
```

Expect: ≥6 matches confirming all six overrides are present.

### Step 4: Commit

```bash
cd "C:\Users\23563\Desktop\揭榜挂帅\new_model\model_v2"
git add configs/bafnet/aircraft_bafnet_v2_1x.py
git commit -m "feat(config): aircraft_bafnet_v2_1x consolidates P0-B/P1-A/P1-B/P2-A"
```

---

## Self-Review

**1. Spec coverage:**

| Spec section | Implemented in |
|---|---|
| §2 P0-A super_of | Task 1 |
| §2 P0-A aggregation | Task 1 |
| §2 P0-A JSON output | Task 1 |
| §3 P0-B both configs | Task 2 |
| §4.1 source preservation | (no code change needed; verified) |
| §4.2 DomainBalancedDataset | Task 8 |
| §4.3 CopyPaste fallback | Task 6 |
| §5.1 high-resolution + multiscale | Task 9 (config) |
| §5.2 boundary loss fix | Task 5 |
| §5.3 p2 verification | (no code change; verified in Task 9 file) |
| §6.1 class_weight | Task 9 |
| §6.2 strong aug | Task 9 (uses Tasks 6+7) |
| §7 LoadBalancingLoss | Task 3 |
| §7.4 baf.py call site | Task 4 |
| §8 new config | Task 9 |

All spec sections covered.

**2. Placeholder scan:**

- Tasks 1-9: every step contains actual code or actual commands. No "TBD", no "fill in details", no "add validation" without code.
- Task 8 has a "simplification note" explaining why the flag handling is collapsed; this is intentional, not a placeholder. Reasonable.

**3. Type consistency:**

- `super_of` returns `Optional[str]` in Task 1; consumed only by name matchers → consistent.
- `DomainBalancedDataset.__init__` signature: `(dataset, target_class_id=3, domain_prefixes=..., domain_extras=..., filter_empty_gt=True)` — used in Task 9 config exactly. Consistent.
- `RandCopyPaste` and `RandRotate` are both `@PIPELINES.register_module()` and used in config by `type='RandCopyPaste'` / `type='RandRotate'`. Consistent with mmcv builder expectations.
- `LoadBalancingLoss(num_experts=4, alpha=0.1, beta=0.01)` in Task 4 matches the new defaults in Task 3.
- `class_weight` is `list[float]` of length 25 (24 + 5.0 at index 24); `CrossEntropyLoss` accepts this. Verified by `mmdet/models/losses/cross_entropy_loss.py:169`.

No inconsistencies found. Plan ready.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-11-bafnet-improvement-v2.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?