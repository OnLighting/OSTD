# Official Recall/FDR Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an 8:2 official-data pipeline that selects checkpoints by official Recall/FDR, performs 70:30 official/ShipRS mixed fine-tuning, and reports validation metrics plus maximum inference time on simulated 10000×10000 images.

**Architecture:** A pure shared metric module owns class names, fixed per-class thresholds, class-specific IoUs, matching, aggregation, and checkpoint comparison. Dataset evaluation, CLI evaluation, ordinary inference, and tiled big-image inference all call that module. One shell script orchestrates splitting, conversion, official training, mixed fine-tuning, validation, mosaic construction, timed inference, and reporting.

**Tech Stack:** Python 3, NumPy, OpenCV, PyTorch, mmdetection 2.x, mmcv runner hooks, COCO JSON, Bash.

**Spec:** `docs/superpowers/specs/2026-08-26-official-recall-training-pipeline-design.md`

## Global Constraints

- Official Recall target: `0.85`; official FDR limit: `0.20`; tied-Recall tolerance after both gates pass: `0.005`.
- Runtime thresholds are hard-coded from `ret/threshold_search_fdr_0.19_selected.csv`; no training or validation command invokes `search_recall_fdr_thresholds.py`.
- Ship and aircraft use IoU `0.50`; vehicle uses IoU `0.35`.
- Official data is split only into train/val at `0.8/0.2`; no independent test split is generated.
- Fine-tuning samples official train at `0.70` and ShipRS train at `0.30`; selection always uses official val.
- Big-image timing excludes disk reads and includes patch inference, projection, merge/NMS, class filtering, and final result construction.
- Per `CLAUDE.md`, do not execute Python, pytest, training, inference, linters, or shell programs locally. Add tests and provide remote commands; verify locally by reading and diff checks only.
- Preserve unrelated dirty-worktree changes; commits stage only files belonging to the current task.

---

### Task 1: Shared official metric and threshold module

**Files:**
- Create: `mmdet/core/evaluation/official_metrics.py`
- Modify: `mmdet/core/evaluation/__init__.py`
- Create: `tests/test_official_metrics.py`

**Interfaces:**
- Produces constants `CLASS_NAMES`, `CLASS_SCORE_THRESHOLDS`, `CLASS_IOU_THRESHOLDS`, and `SUPERCLASS_INDICES`, each covering category ids 0–24 exactly once.
- Produces `filter_mmdet_results(results) -> list[np.ndarray]`.
- Produces `evaluate_mmdet_results(results, gt_infos) -> dict` with `per_class`, `by_super`, `official`, and `merged` sections.
- Produces `compare_official_candidates(candidate, best, recall_target=.85, fdr_limit=.20, recall_tolerance=.005) -> bool`.

- [ ] **Step 1: Write failing tests for constants and matching**

Use this exact threshold fixture and add duplicate-box, score-boundary, empty-prediction, superclass-average, merged-count, and class-IoU cases:

```python
EXPECTED_THRESHOLDS = (
    .348537356, .00188686815, .00998581946, .0174246859,
    .854384005, .554479778, .469661117, .424196422, .451967984,
    .927510798, .927627504, .33063519, .791748285, .987444103,
    .186482817, .69058013, .586889446, .936379254, .362753332,
    .145738885, .877446949, .038396392, .141620845, .0739553422,
    .0403826572)

def test_hardcoded_thresholds_match_selected_csv_values():
    assert CLASS_SCORE_THRESHOLDS == EXPECTED_THRESHOLDS

def test_duplicate_prediction_is_false_positive():
    result = _empty_result()
    result[0] = np.array([[0, 0, 10, 10, .9], [0, 0, 10, 10, .8]])
    metrics = evaluate_mmdet_results([result], [_gt([[0, 0, 10, 10]], [0])])
    assert metrics['per_class'][0]['tp'] == 1
    assert metrics['per_class'][0]['fp'] == 1
```

- [ ] **Step 2: Record remote red-test command**

`pytest tests/test_official_metrics.py -v` must initially fail because the module does not exist.

- [ ] **Step 3: Implement filtering and one-to-one matching**

Validate 25 result arrays, filter each with its class threshold, sort predictions by score descending, greedily match each prediction to the highest-IoU unmatched GT above that class's IoU, and count remaining predictions/GT as FP/FN.

```python
def filter_mmdet_results(results):
    if len(results) != len(CLASS_NAMES):
        raise ValueError(f'expected 25 classes, got {len(results)}')
    return [boxes[boxes[:, 4] >= CLASS_SCORE_THRESHOLDS[i]]
            if len(boxes) else np.zeros((0, 5), np.float32)
            for i, boxes in enumerate(results)]
```

- [ ] **Step 4: Implement official aggregation and model comparison**

```python
def compare_official_candidates(candidate, best, recall_target=.85,
                                fdr_limit=.20, recall_tolerance=.005):
    if best is None:
        return True
    candidate_pass = candidate['recall'] >= recall_target and candidate['fdr'] <= fdr_limit
    best_pass = best['recall'] >= recall_target and best['fdr'] <= fdr_limit
    if candidate_pass != best_pass:
        return candidate_pass
    if not candidate_pass:
        return candidate['recall'] > best['recall']
    delta = candidate['recall'] - best['recall']
    return delta > 0 if abs(delta) > recall_tolerance else candidate['fdr'] < best['fdr']
```

Reject metric calculation when a superclass has no GT; do not substitute zero.

- [ ] **Step 5: Export APIs and statically verify**

Add imports/`__all__` entries, compare all 25 literals with the 0.19 CSV by category id, and inspect `git diff --check` for the three files.

- [ ] **Step 6: Commit**

```bash
git add mmdet/core/evaluation/official_metrics.py mmdet/core/evaluation/__init__.py tests/test_official_metrics.py
git commit -m "feat: add official recall fdr metrics"
```

---

### Task 2: Dataset evaluation and official checkpoint hooks

**Files:**
- Modify: `mmdet/datasets/aircraft.py`
- Modify: `mmdet/core/evaluation/eval_hooks.py`
- Modify: `mmdet/core/evaluation/__init__.py`
- Modify: `configs/_base_/default_runtime.py`
- Modify: `configs/bafnet/aircraft_bafnet_1x.py`
- Create: `tests/test_official_checkpoint_hooks.py`

**Interfaces:**
- Consumes Task 1 metric and comparator APIs.
- Produces `AircraftDataset.evaluate(self, results, metric=['bbox', 'official'], **kwargs)` keys for official, merged, and three superclass Recall/FDR values.
- Produces `OfficialBestSaverHook`, saving `best_official_recall_fdr.pth` and `best_official_recall_fdr.json`.
- Produces `OfficialEarlyStoppingHook`, using the same comparator and patience 16.

- [ ] **Step 1: Write failing comparator and persistence tests**

```python
def test_first_double_pass_replaces_fdr_failure():
    assert compare_official_candidates(
        {'recall': .86, 'fdr': .19}, {'recall': .90, 'fdr': .21})

def test_tied_recall_prefers_lower_fdr():
    assert compare_official_candidates(
        {'recall': .884, 'fdr': .16}, {'recall': .880, 'fdr': .18})

def test_over_tolerance_prefers_recall():
    assert compare_official_candidates(
        {'recall': .886, 'fdr': .19}, {'recall': .880, 'fdr': .10})
```

Use a fake runner/temp epoch checkpoint to assert checkpoint copy and JSON fields `epoch`, `official_recall`, `official_fdr`, and `passed`.

- [ ] **Step 2: Record remote red-test command**

`pytest tests/test_official_checkpoint_hooks.py -v` must fail because the hooks are absent.

- [ ] **Step 3: Add official evaluation to AircraftDataset**

Split requested metrics into standard metrics and `official`; call `super().evaluate()` only for standard metrics, call `evaluate_mmdet_results(results, [self.get_ann_info(i) for i in range(len(self))])`, flatten stable scalar keys, then merge both ordered dictionaries.

- [ ] **Step 4: Implement best saver and official early stop**

Read `official_recall`/`official_fdr` after validation. Copy `epoch_N.pth` only after comparator acceptance; atomically replace metadata JSON; update in-memory best after successful writes. Early stop resets patience only on comparator acceptance and raises the existing `EarlyStopping` exception after 16 non-improvements.

- [ ] **Step 5: Switch runtime config**

```python
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='SBLAEpochHook'),
    dict(type='OfficialBestSaverHook', recall_target=.85,
         fdr_limit=.20, recall_tolerance=.005),
    dict(type='OfficialEarlyStoppingHook', patience=16,
         recall_target=.85, fdr_limit=.20, recall_tolerance=.005),
]
evaluation = dict(interval=1, metric=['bbox', 'official'])
```

Keep ordinary epoch checkpoints; do not enable mmcv `save_best`.

- [ ] **Step 6: Static review and commit**

Inspect hook registration/order, imports, `git diff --check`, then commit only the listed files:

```bash
git commit -m "feat: select checkpoints by official metrics"
```

---

### Task 3: Stratified 8:2 train/validation split

**Files:**
- Modify: `tools/split_val.py`
- Modify: `configs/_base_/datasets/aircraft_detection.py`
- Modify: `CLAUDE.md`
- Create: `tests/test_split_train_val.py`

**Interfaces:**
- Produces `stratified_train_val(stems, labels, ratios, seed) -> (train, val)`.
- Produces CLI `--ratios 0.8 0.2` and no test output.
- Keeps `data.test` only as a compatibility alias to val paths.

- [ ] **Step 1: Write failing split tests**

```python
def test_train_val_are_disjoint_and_complete():
    train, val = stratified_train_val(STEMS, LABELS, (.8, .2), 0)
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(STEMS)

def test_rare_classes_remain_in_train():
    train, val = stratified_train_val(RARE_STEMS, RARE_LABELS, (.8, .2), 0)
    assert set(RARE_STEMS) <= set(train)
```

Add a tiny YOLO fixture test asserting `images/test` and `labels/test` are not created.

- [ ] **Step 2: Record remote red-test command**

`pytest tests/test_split_train_val.py -v` must fail because the two-way API is absent.

- [ ] **Step 3: Rewrite split logic**

Retain deterministic class ordering, rare-class protection, minimum training box protection, and move-back post-check. Remove `test_set`, `test_stems`, all test copy/wipe/statistics branches, and normalize exactly two ratios.

- [ ] **Step 4: Point config test alias at val and update notes**

Set both val/test config entries to `instances_val.json` and `images/val/`. Replace operational 6:2:2 and test-set examples in `CLAUDE.md` with 8:2 validation-only commands.

- [ ] **Step 5: Static review and commit**

Search the modified files for obsolete `instances_test`, `images/test`, `test_stems`, and `0.6 0.2 0.2`; inspect `git diff --check`; commit as `feat: split official data into train and validation`.

---

### Task 4: Fixed-threshold ordinary validation inference

**Files:**
- Modify: `tools/eval_val_to_json.py`
- Modify: `tools/eval_recall_fdr.py`
- Create: `tests/test_validation_prediction_filtering.py`

**Interfaces:**
- Produces `detections_to_coco_annotations(result, image_id, next_ann_id)` using Task 1 thresholds.
- Preserves CSV/JSON outputs with per-class AP/TP/FP/FN/Recall/FDR/Precision, superclass, official, and merged metrics.

- [ ] **Step 1: Write failing per-class filtering test**

```python
def test_each_class_uses_its_own_threshold():
    result = _empty_result()
    result[0] = np.array([[0, 0, 2, 2, CLASS_SCORE_THRESHOLDS[0] - 1e-6]])
    result[1] = np.array([[0, 0, 2, 2, CLASS_SCORE_THRESHOLDS[1]]])
    anns, _ = detections_to_coco_annotations(result, 7, 1)
    assert [ann['category_id'] for ann in anns] == [1]
```

- [ ] **Step 2: Record remote red-test command**

`pytest tests/test_validation_prediction_filtering.py -v` must fail because global `--score` is still used.

- [ ] **Step 3: Refactor prediction export**

Remove global `--score`. Set model `test_cfg.rcnn.score_thr` to `min(CLASS_SCORE_THRESHOLDS)` so low-threshold classes survive, then use `filter_mmdet_results()` before serialization. Validate a one-to-one GT file-name/image-id mapping.

- [ ] **Step 4: Refactor metric CLI to shared matching**

Keep AP and report formatting, but import class names, superclass map, IoUs, threshold filtering, and confusion-count evaluation from Task 1. Remove duplicated `CLASS_NAMES`, `super_of`, and matching rules.

- [ ] **Step 5: Static review and commit**

Search both tools for `search_recall_fdr_thresholds`, `--score`, duplicate constants, and duplicated IoU rules; inspect `git diff --check`; commit as `feat: apply fixed class thresholds in validation`.

---

### Task 5: Deterministic 10000×10000 validation mosaics

**Files:**
- Create: `tools/compose_big_val.py`
- Create: `tests/test_compose_big_val.py`

**Interfaces:**
- Produces `pack_images(sizes, canvas_size=10000)` with placements `(source_index, x, y)`.
- CLI consumes `--gt`, `--img-dir`, `--out-dir`, `--num-canvases`, `--seed`, `--overwrite`.
- Produces `images/*.jpg`, `instances_big_val.json`, and `source_map.json`.

- [ ] **Step 1: Write failing packing/remapping tests**

```python
def test_shift_bbox_preserves_size_and_moves_origin():
    assert shift_bbox([10, 20, 30, 40], 100, 200) == [110, 220, 30, 40]

def test_every_placement_stays_inside_canvas():
    for canvas in pack_images([(3000, 2000), (4000, 2000), (2500, 3000)]):
        for index, x, y in canvas:
            width, height = SIZES[index]
            assert x + width <= 10000 and y + height <= 10000
```

Add fixture-image tests for exact output dimensions, deterministic seed, complete source map, no split source images, and overwrite refusal.

- [ ] **Step 2: Record remote red-test command**

`pytest tests/test_compose_big_val.py -v` must fail because the tool is absent.

- [ ] **Step 3: Implement shelf packing and COCO remapping**

Seed-shuffle then deterministically row-pack complete images without resizing. Fill unused pixels with `(114,114,114)`, stop at requested canvas count, translate bbox origins, preserve categories, regenerate ids, and recompute area.

- [ ] **Step 4: Add strict validation**

Reject sources over 10000 pixels, missing/duplicate files, empty selections, translated boxes outside the canvas, and existing output unless `--overwrite` is explicit.

- [ ] **Step 5: Static review and commit**

Inspect output paths and literal default size, run `git diff --check`, and commit as `feat: compose validation images into 10k mosaics`.

---

### Task 6: Batch big-image inference and maximum-time reporting

**Files:**
- Modify: `tools/infer_big_image.py`
- Create: `tests/test_big_image_inference_helpers.py`

**Interfaces:**
- Produces `infer_big_image(model, image, tile, overlap, nms_iou, max_det)` returning annotations, elapsed seconds, and patch count.
- Batch CLI consumes `--img-dir`, `--gt`, `--out`, `--timing-out`.
- Timing JSON provides `per_image_seconds`, `mean_inference_seconds`, `max_inference_seconds`.

- [ ] **Step 1: Write failing helper tests**

```python
def test_timing_summary_reports_maximum():
    summary = summarize_timings({'a.jpg': 1.2, 'b.jpg': 2.5})
    assert summary['max_inference_seconds'] == 2.5

def test_final_filter_uses_class_thresholds():
    _, _, classes = apply_class_thresholds(
        BOXES, np.array([CLASS_SCORE_THRESHOLDS[0] - 1e-6,
                         CLASS_SCORE_THRESHOLDS[1]]), np.array([0, 1]))
    assert classes.tolist() == [1]
```

Also cover tile coverage, coordinate projection, NMS index mapping, and GT image-id propagation.

- [ ] **Step 2: Record remote red-test command**

`pytest tests/test_big_image_inference_helpers.py -v` must fail because batch/timing helpers are absent.

- [ ] **Step 3: Extract one-image inference with correct timer**

Read the image before timing. Synchronize CUDA, start timer, run patches, project, concatenate, NMS, apply fixed class thresholds, sort/cap, construct annotations, synchronize CUDA, then stop timer. Default `max_det` becomes 3000. Remove global `--score`.

```python
if torch.cuda.is_available():
    torch.cuda.synchronize()
infer_t0 = time.perf_counter()
# inference through final annotation construction
if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed = time.perf_counter() - infer_t0
```

- [ ] **Step 4: Add GT-aligned directory mode**

Map mosaic `file_name` to image id, process every GT image, combine predictions into one COCO JSON, and write per-image/mean/max timing JSON.

- [ ] **Step 5: Static review and commit**

Verify timer boundaries, shared thresholds, no global score, max field, and `git diff --check`; commit as `feat: report max inference time on 10k images`.

---

### Task 7: Two-stage configuration and complete shell pipeline

**Files:**
- Modify: `configs/bafnet/aircraft_bafnet_shiprs_mix_pretrain.py`
- Modify: `run.sh`
- Modify: `README.md`

**Interfaces:**
- Uses stage dirs `${WORK_ROOT}/official_stage` and `${WORK_ROOT}/shiprs_finetune_stage`.
- Environment overrides: `PROJECT_ROOT`, `DATA_ROOT`, `SHIPRS_ROOT`, `WORK_ROOT`, `OFFICIAL_CONFIG`, `FINETUNE_CONFIG`, `OFFICIAL_WEIGHT`, `SHIPRS_WEIGHT`, `BIG_IMAGE_COUNT`, `DEVICE`.

- [ ] **Step 1: Update fine-tune config**

Set `source_weights=(0.70, 0.30)`, official val for both val and compatibility-test entries, `evaluation = dict(interval=1, metric=['bbox', 'official'])`, and document shell override of `load_from` with stage-one best.

- [ ] **Step 2: Rewrite run.sh with explicit contracts**

Start with:

```bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PROJECT_ROOT:=$SCRIPT_DIR}"
: "${OFFICIAL_WEIGHT:=0.70}"
: "${SHIPRS_WEIGHT:=0.30}"
```

Add `require_dir`, `require_file`, and safe backup/restore functions scoped to validated `$DATA_ROOT`. Never recursively delete an unresolved path.

- [ ] **Step 3: Implement ordered pipeline commands**

The script must: restore official source; split `0.8 0.2`; convert train/val; prepare ShipRS mappings; train official stage; set `STAGE1_BEST="$OFFICIAL_WORK/best_official_recall_fdr.pth"`; fine-tune with `--cfg-options load_from="$STAGE1_BEST" data.train.source_weights="(0.70,0.30)"`; require stage-two best; run bbox val evaluation; export fixed-threshold val predictions; report official val metrics; compose big val; batch-infer big val; report big-val metrics and timing paths.

- [ ] **Step 4: Document remote usage and outputs**

```bash
export DATA_ROOT=/path/to/official/data
export SHIPRS_ROOT=/path/to/ShipRSImageNet
bash run.sh
```

Document fixed thresholds, 8:2, 70:30, best filenames, validation-only reporting, output files, and RTX 3090 timing comparability.

- [ ] **Step 5: Static review and commit**

Search production paths for `instances_test`, `images/test`, `0.6 0.2 0.2`, `best_bbox_mAP`, and threshold-search calls. Confirm new values/paths, manually inspect Bash quoting/conditionals, run `git diff --check`, and commit as `feat: add two-stage official validation pipeline`.

---

### Task 8: Final static integration review and remote handoff

**Files:** Review all files changed in Tasks 1–7; modify only in-scope files if review finds defects.

**Interfaces:** Produces an exact remote verification checklist and expected artifact list.

- [ ] **Step 1: Check every spec requirement**

Map code to official aggregation, fixed thresholds, three-tier best logic, 8:2 only, 70:30 mixed fine-tuning, official-val selection, 10k composition, complete timed post-processing, all metrics, and maximum time.

- [ ] **Step 2: Search obsolete executable behavior**

```bash
rg -n "best_bbox_mAP|instances_test|images/test|labels/test|0\.6 0\.2 0\.2|search_recall_fdr_thresholds|--score" run.sh tools/split_val.py tools/eval_val_to_json.py tools/infer_big_image.py configs/_base_/datasets/aircraft_detection.py configs/bafnet/aircraft_bafnet_shiprs_mix_pretrain.py
```

Fix matches that remain executable; historical comments must be clearly labeled.

- [ ] **Step 3: Inspect final diffs without running project code**

Use `git diff --check`, `git status --short`, and `git diff --stat`; review imports, signatures, config inheritance, shell quoting, output paths, and overlap with pre-existing user changes.

- [ ] **Step 4: Handoff remote commands**

```bash
pytest tests/test_official_metrics.py tests/test_official_checkpoint_hooks.py \
  tests/test_split_train_val.py tests/test_validation_prediction_filtering.py \
  tests/test_compose_big_val.py tests/test_big_image_inference_helpers.py -v
bash -n run.sh
bash run.sh
```

Expected artifacts: both stage best checkpoints/metadata, ordinary val bbox/official reports, composed images/GT/source map, big-image predictions/official reports, and timing JSON containing `max_inference_seconds`.

- [ ] **Step 5: Commit review fixes only when necessary**

Stage only corrected in-scope files and commit `fix: align official validation pipeline integration`; do not create an empty commit.
