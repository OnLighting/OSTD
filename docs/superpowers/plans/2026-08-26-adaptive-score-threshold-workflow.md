# Adaptive Score Threshold Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace historical hard-coded class scores with exact three-superclass threshold search during training and a checkpoint-bound 25-class threshold artifact generated after training.

**Architecture:** The shared evaluation package owns score-ranked matching, exact operating curves, explicit threshold filtering, and training-time superclass search. The post-training search CLI reuses those primitives to generate a versioned JSON artifact bound to the selected checkpoint SHA-256; validation export, big-image inference, and official-test inference consume that artifact instead of Python constants.

**Tech Stack:** Python 3, NumPy, MMDetection 2.x/MMCV, COCO-style JSON, Bash, pytest/unittest.

**Spec:** `docs/superpowers/specs/2026-08-26-official-recall-training-pipeline-design.md`

## Global Constraints

- Preserve the official 25-class order in `CLASS_NAMES`; ship IDs are `0..3`, aircraft IDs are `4..23`, and vehicle ID is `24`.
- IoU is `0.50` for IDs `0..23` and `0.35` for ID `24`; matching is score-descending, one-to-one, and excludes already matched GTs.
- Model-side candidate retention uses `CANDIDATE_SCORE_FLOOR = 0.0` and `max_per_img = 3000`; the floor is not a decision threshold.
- Training-time search uses one exact threshold per superclass and requires each superclass mean FDR to be at most `0.19`.
- Post-training search uses 25 thresholds and requires official FDR to be at most `0.19`.
- Best comparison remains Recall target `0.85`, FDR limit `0.20`, and Recall tolerance `0.005`.
- The official independent test set never supplies GT to threshold search; it only consumes the frozen artifact.
- Historical `ret/*.csv` files are audit material only and must not be imported or read by production paths.
- Follow repository policy: do not run Python, pytest, training, inference, or Bash locally. Record the listed red/green commands for execution on the remote training server; use static checks locally.
- Preserve unrelated working-tree changes, especially `mmdet/datasets/pipelines/rand_rotate.py` and existing untracked files. Stage only files named by each task.

---

### Task 1: Shared explicit-threshold metrics and exact superclass search

**Files:**
- Modify: `mmdet/core/evaluation/official_metrics.py`
- Modify: `mmdet/core/evaluation/__init__.py`
- Modify: `tests/test_official_metrics.py`
- Create: `tests/test_superclass_threshold_search.py`

**Interfaces:**
- Produces: `CANDIDATE_SCORE_FLOOR: float = 0.0`.
- Produces: `match_class_events(pred_boxes, pred_scores, gt_boxes, tau) -> list[tuple[float, int]]`, sorted by descending score with `int` TP flags.
- Produces: `build_mmdet_score_events(results, gt_infos) -> tuple[dict[int, list[tuple[float, int]]], dict[int, int]]`.
- Produces: `normalize_score_thresholds(score_thresholds) -> tuple[float, ...]`; accepts a scalar or exactly 25 finite non-negative values.
- Produces: `evaluate_score_events(events, total_gt, score_thresholds) -> dict` with `per_class`, `by_super`, `official`, and `merged`.
- Produces: `search_superclass_thresholds(events, total_gt, max_fdr=0.19) -> dict` with `thresholds_by_super`, expanded `score_thresholds`, and `metrics`.
- Changes: `filter_mmdet_results(results, score_thresholds)` requires explicit thresholds.
- Changes: `evaluate_mmdet_results(results, gt_infos, score_thresholds)` requires explicit thresholds and never reads a module-level decision threshold.

- [ ] **Step 1: Replace hard-coded-threshold tests with explicit-threshold tests**

Remove `EXPECTED_THRESHOLDS`, `test_hardcoded_thresholds_match_selected_csv_values`, and imports of `CLASS_SCORE_THRESHOLDS`. Add tests that prove filtering depends on the supplied values:

```python
def test_filter_uses_explicit_thresholds():
    result = _empty_result()
    result[0] = np.array([[0, 0, 2, 2, 0.39]], dtype=np.float32)
    result[1] = np.array([[0, 0, 2, 2, 0.40]], dtype=np.float32)
    filtered = filter_mmdet_results(result, [0.40] * 25)
    assert len(filtered[0]) == 0
    assert len(filtered[1]) == 1


def test_threshold_count_must_equal_class_count():
    with pytest.raises(ValueError, match='25'):
        filter_mmdet_results(_empty_result(), [0.1] * 24)
```

Update every existing `evaluate_mmdet_results(...)` call in this test file to pass `[0.0] * 25`, except the score-filter boundary test, which passes its own literal thresholds.

- [ ] **Step 2: Write failing exact superclass-search tests**

Create `tests/test_superclass_threshold_search.py` with literal events. The production mutation caught by this test is using one global threshold or a hand-written grid instead of distinct exact score breakpoints:

```python
from mmdet.core.evaluation import search_superclass_thresholds


def _events_with_all_superclasses():
    events = {i: [] for i in range(25)}
    total_gt = {i: 0 for i in range(25)}
    events[0] = [(0.91, 1), (0.90, 0)]
    total_gt[0] = 1
    events[4] = [(0.61, 1), (0.60, 0)]
    total_gt[4] = 1
    events[24] = [(0.21, 1), (0.20, 0)]
    total_gt[24] = 1
    return events, total_gt


def test_search_uses_one_exact_threshold_per_superclass():
    events, total_gt = _events_with_all_superclasses()
    result = search_superclass_thresholds(events, total_gt, max_fdr=0.19)
    assert result['thresholds_by_super'] == {
        'ship': 0.91,
        'aircraft': 0.61,
        'vehicle': 0.21,
    }
    assert result['score_thresholds'][:4] == (0.91,) * 4
    assert result['score_thresholds'][4:24] == (0.61,) * 20
    assert result['score_thresholds'][24] == 0.21


def test_search_marks_missing_superclass_gt_unavailable():
    events, total_gt = _events_with_all_superclasses()
    total_gt[24] = 0
    result = search_superclass_thresholds(events, total_gt, max_fdr=0.19)
    assert result['metrics']['official']['available'] is False
    assert 'vehicle' in result['metrics']['official']['unavailable_superclasses']
```

Add separate tests for the deterministic tie order: higher Recall, then lower FDR, then higher threshold.

- [ ] **Step 3: Record the remote red-test command**

Run remotely:

```bash
pytest tests/test_official_metrics.py tests/test_superclass_threshold_search.py -v
```

Expected: FAIL because explicit threshold parameters and `search_superclass_thresholds` do not exist.

- [ ] **Step 4: Implement score events and explicit threshold evaluation**

Replace `CLASS_SCORE_THRESHOLDS` with:

```python
CANDIDATE_SCORE_FLOOR = 0.0
```

Extract the existing greedy loop so it returns ranked TP flags:

```python
def match_class_events(pred_boxes, pred_scores, gt_boxes, tau):
    matched = np.zeros(len(gt_boxes), dtype=bool)
    events = []
    for pred_index in np.argsort(-pred_scores):
        is_tp = 0
        unmatched = np.flatnonzero(~matched)
        if unmatched.size:
            overlaps = _iou_xywh(pred_boxes[pred_index], gt_boxes[unmatched])
            best_local = int(overlaps.argmax())
            if overlaps[best_local] >= tau:
                matched[unmatched[best_local]] = True
                is_tp = 1
        events.append((float(pred_scores[pred_index]), is_tp))
    return events
```

Make `_match_class` derive TP/FP/FN from these events. Build per-class events across images before thresholding; because thresholding retains a score-ranked prefix, lower-score predictions cannot alter earlier matches.

- [ ] **Step 5: Implement exact superclass search**

For each superclass, build candidate thresholds from all distinct event scores in that superclass plus an empty-prediction threshold above its maximum. Evaluate one shared threshold across every child class. Choose with this literal comparison key among points satisfying `mean_class_fdr <= max_fdr`:

```python
key = (super_recall, -super_fdr, threshold)
```

If the superclass has zero total GT, return `None` for its threshold and let the official aggregate report `available=False`. Expand valid group thresholds to a 25-value tuple using `SUPERCLASS_INDICES`.

- [ ] **Step 6: Export the new API and remove the old constant**

Update both `__all__` lists. A repository search must return no production import of `CLASS_SCORE_THRESHOLDS` after later migration tasks; tests in later tasks will still fail until their callers are updated.

- [ ] **Step 7: Record the remote green-test command and commit**

Run remotely:

```bash
pytest tests/test_official_metrics.py tests/test_superclass_threshold_search.py -v
```

Expected: PASS.

Then commit only Task 1 files:

```bash
git add mmdet/core/evaluation/official_metrics.py \
  mmdet/core/evaluation/__init__.py \
  tests/test_official_metrics.py tests/test_superclass_threshold_search.py
git commit -m "feat: search superclass score thresholds during evaluation"
```

---

### Task 2: Training evaluation and checkpoint metadata use searched superclass thresholds

**Files:**
- Modify: `mmdet/datasets/aircraft.py`
- Modify: `mmdet/core/evaluation/eval_hooks.py`
- Create: `tests/test_aircraft_official_evaluate.py`
- Modify: `tests/test_official_checkpoint_hooks.py`

**Interfaces:**
- Consumes: `build_mmdet_score_events` and `search_superclass_thresholds` from Task 1.
- Produces scalar evaluation keys: `official_threshold_ship`, `official_threshold_aircraft`, and `official_threshold_vehicle`.
- Changes best metadata: `best_official_recall_fdr.json` includes `training_score_thresholds` with exactly the three superclass keys.

- [ ] **Step 1: Write a failing dataset-evaluation test**

Use the existing lightweight Aircraft dataset fixture and assert:

```python
metrics = dataset.evaluate(results, metric=['official'])
assert set(metrics['training_score_thresholds']) == {
    'ship', 'aircraft', 'vehicle'
}
assert metrics['official_threshold_ship'] == \
    metrics['training_score_thresholds']['ship']
assert metrics['official_fdr'] <= 0.19
```

The test fixture must contain at least one GT in every superclass; otherwise the intended result is unavailable.

- [ ] **Step 2: Write a failing checkpoint-metadata test**

Extend `_FakeRunner` metrics with literal threshold values and assert the saved JSON contains them:

```python
runner.log_buffer.output.update({
    'official_threshold_ship': 0.31,
    'official_threshold_aircraft': 0.42,
    'official_threshold_vehicle': 0.17,
})
hook.after_train_epoch(runner)
assert meta['training_score_thresholds'] == {
    'ship': 0.31,
    'aircraft': 0.42,
    'vehicle': 0.17,
}
```

- [ ] **Step 3: Record the remote red-test command**

```bash
pytest tests/test_aircraft_official_evaluate.py \
  tests/test_official_checkpoint_hooks.py -v
```

Expected: FAIL because training evaluation still applies historical class thresholds and metadata omits group thresholds.

- [ ] **Step 4: Integrate superclass search into `AircraftDataset.evaluate`**

Replace the fixed evaluation call with:

```python
events, total_gt = build_mmdet_score_events(flat_results, gt_infos)
searched = search_superclass_thresholds(events, total_gt, max_fdr=0.19)
metrics = searched['metrics']
thresholds_by_super = searched['thresholds_by_super']
```

Emit the existing official, superclass, merged, and per-class keys plus the three scalar threshold keys. Do not emit nested dict/list values into fields consumed as scalar runner metrics unless the existing logger already supports them.

- [ ] **Step 5: Persist threshold metadata in `OfficialBestSaverHook`**

When a candidate is accepted, require all three threshold keys to be finite. Write:

```python
'training_score_thresholds': {
    'ship': float(output['official_threshold_ship']),
    'aircraft': float(output['official_threshold_aircraft']),
    'vehicle': float(output['official_threshold_vehicle']),
}
```

If any threshold is absent/NaN, log a warning and refuse to save that epoch, matching the existing missing-official-metric behavior. The early-stopping comparator continues to use official Recall/FDR only.

- [ ] **Step 6: Record remote green tests and commit**

```bash
pytest tests/test_aircraft_official_evaluate.py \
  tests/test_official_checkpoint_hooks.py -v
```

Expected: PASS.

```bash
git add mmdet/datasets/aircraft.py mmdet/core/evaluation/eval_hooks.py \
  tests/test_aircraft_official_evaluate.py tests/test_official_checkpoint_hooks.py
git commit -m "feat: select training checkpoints with superclass thresholds"
```

---

### Task 3: Versioned checkpoint-bound threshold artifact and repaired search CLI

**Files:**
- Create: `mmdet/core/evaluation/threshold_artifact.py`
- Modify: `mmdet/core/evaluation/__init__.py`
- Modify: `tools/search_recall_fdr_thresholds.py`
- Create: `tests/test_threshold_artifact.py`
- Create: `tests/test_recall_fdr_threshold_search.py`

**Interfaces:**
- Produces: `sha256_file(path) -> str`.
- Produces: `write_threshold_artifact(path, thresholds, checkpoint_path, prediction_path, gt_path, constraints, metrics) -> dict`.
- Produces: `load_threshold_artifact(path, checkpoint_path=None) -> tuple[float, ...]`.
- Search CLI adds required `--checkpoint`; `<out-prefix>.json` follows schema version `1`.
- Search CLI continues to produce `_selected.csv`, `_global_curve.csv`, `_class_curves.csv`, and `_filtered_preds.json`.

- [ ] **Step 1: Write failing artifact validation tests**

```python
def test_artifact_round_trip_and_checkpoint_binding(tmp_path):
    checkpoint = tmp_path / 'best.pth'
    checkpoint.write_bytes(b'checkpoint-a')
    artifact = tmp_path / 'thresholds.json'
    write_threshold_artifact(
        artifact, [0.1] * 25, checkpoint, 'dense.json', 'val.json',
        {'max_official_fdr': 0.19}, {'official': {'recall': 0.86, 'fdr': 0.18}})
    assert load_threshold_artifact(artifact, checkpoint) == (0.1,) * 25


def test_artifact_rejects_different_checkpoint(tmp_path):
    # Write artifact for checkpoint A, then load against checkpoint B.
    with pytest.raises(ValueError, match='SHA-256'):
        load_threshold_artifact(artifact, checkpoint_b)
```

Add parameterized malformed-artifact cases for 24/26 classes, duplicate IDs, wrong names, negative/NaN thresholds, unknown schema version, and missing checkpoint hash.

- [ ] **Step 2: Write failing CLI integration tests**

Construct a tiny COCO prediction/GT pair containing all three superclasses. Invoke `main()` with monkeypatched `sys.argv`, including `--checkpoint`. Assert the JSON has this exact shape:

```python
assert payload['schema_version'] == 1
assert payload['checkpoint']['sha256'] == sha256_file(checkpoint)
assert [row['category_id'] for row in payload['classes']] == list(range(25))
assert len(payload['classes']) == 25
assert payload['constraints']['max_official_fdr'] == 0.19
```

Also add a regression test proving the CLI imports successfully. This catches the current broken imports of removed `TAU_DEFAULT`, `TAU_OVERRIDE`, `iou_xywh`, and `super_of` symbols from `tools/eval_recall_fdr.py`.

- [ ] **Step 3: Record remote red tests**

```bash
pytest tests/test_threshold_artifact.py \
  tests/test_recall_fdr_threshold_search.py -v
```

Expected: FAIL because the artifact module and `--checkpoint` contract do not exist; the current search CLI import is also broken.

- [ ] **Step 4: Implement the artifact module**

Write JSON atomically through a temporary sibling followed by `os.replace`. Use this versioned structure:

```python
{
    'schema_version': 1,
    'checkpoint': {
        'path': str(checkpoint_path),
        'sha256': sha256_file(checkpoint_path),
    },
    'source': {
        'prediction_path': str(prediction_path),
        'gt_path': str(gt_path),
    },
    'constraints': dict(constraints),
    'metrics': metrics,
    'classes': [
        {'category_id': i, 'name': CLASS_NAMES[i], 'threshold': float(value)}
        for i, value in enumerate(thresholds)
    ],
}
```

`load_threshold_artifact` validates every field before returning a tuple. Path equality is not required; checkpoint SHA-256 equality is required when `checkpoint_path` is supplied.

- [ ] **Step 5: Repair and centralize the search CLI**

Replace imports from `eval_recall_fdr` with shared constants and `match_class_events`. Keep the existing conservative 25-class dynamic programme, but obtain matching events through the shared matcher. Change the default `--max-official-fdr` from `0.20` to `0.19`, require `--checkpoint`, and call `write_threshold_artifact` for the main JSON.

The filtered prediction output must preserve every top-level field and retain only annotations satisfying the selected threshold for their category.

- [ ] **Step 6: Record remote green tests and commit**

```bash
pytest tests/test_threshold_artifact.py \
  tests/test_recall_fdr_threshold_search.py -v
```

Expected: PASS.

```bash
git add mmdet/core/evaluation/threshold_artifact.py \
  mmdet/core/evaluation/__init__.py tools/search_recall_fdr_thresholds.py \
  tests/test_threshold_artifact.py tests/test_recall_fdr_threshold_search.py
git commit -m "feat: freeze searched thresholds to checkpoint-bound artifact"
```

---

### Task 4: Dense validation export with optional frozen-threshold application

**Files:**
- Modify: `tools/eval_val_to_json.py`
- Modify: `tests/test_validation_prediction_filtering.py`

**Interfaces:**
- Changes: `detections_to_coco_annotations(result, image_id, next_ann_id, score_thresholds=None)`; `None` exports every model-retained candidate.
- CLI adds optional `--thresholds`; when supplied, it verifies the artifact against `--checkpoint` and filters with its 25 values.
- Model configuration always uses `CANDIDATE_SCORE_FLOOR` before inference.

- [ ] **Step 1: Write failing dense/filter mode tests**

```python
def test_dense_export_keeps_low_score_candidate():
    result = _empty_result()
    result[0] = np.array([[0, 0, 2, 2, 0.0001]], dtype=np.float32)
    anns, _ = detections_to_coco_annotations(
        result, image_id=7, next_ann_id=1, score_thresholds=None)
    assert [ann['score'] for ann in anns] == [0.0001]


def test_explicit_threshold_export_filters_by_class():
    result = _empty_result()
    result[0] = np.array([[0, 0, 2, 2, 0.39]], dtype=np.float32)
    result[1] = np.array([[0, 0, 2, 2, 0.40]], dtype=np.float32)
    anns, _ = detections_to_coco_annotations(
        result, image_id=7, next_ann_id=1, score_thresholds=[0.40] * 25)
    assert [ann['category_id'] for ann in anns] == [1]
```

- [ ] **Step 2: Record remote red test**

```bash
pytest tests/test_validation_prediction_filtering.py -v
```

Expected: FAIL because export always applies removed hard-coded thresholds.

- [ ] **Step 3: Implement dense and artifact-filtered modes**

Set `cfg.model.test_cfg.rcnn.score_thr = CANDIDATE_SCORE_FLOOR`. If `--thresholds` is absent, pass raw 25-class results into conversion. If present, call `load_threshold_artifact(args.thresholds, args.checkpoint)` once before the image loop and pass the returned tuple to conversion.

Validate the image directory and GT file-name sets for exact equality before inference, matching the big-image batch contract; do not silently omit a GT image.

- [ ] **Step 4: Record remote green test and commit**

```bash
pytest tests/test_validation_prediction_filtering.py -v
```

Expected: PASS.

```bash
git add tools/eval_val_to_json.py tests/test_validation_prediction_filtering.py
git commit -m "feat: export dense validation predictions for threshold search"
```

---

### Task 5: Big-image inference requires the frozen threshold artifact

**Files:**
- Modify: `tools/infer_big_image.py`
- Modify: `tests/test_big_image_inference_helpers.py`

**Interfaces:**
- Changes: `apply_class_thresholds(boxes, scores, classes, nms_keep, score_thresholds)`.
- Changes: `infer_big_image(..., score_thresholds, device='cuda:0')` requires the loaded values.
- CLI adds required `--thresholds`; startup verifies it against `--checkpoint` before model/image work.

- [ ] **Step 1: Replace constant-based tests with explicit values**

```python
def test_final_filter_uses_loaded_class_thresholds():
    thresholds = [0.5] * 25
    boxes, scores, classes = apply_class_thresholds(
        BOXES,
        np.array([0.49, 0.50], dtype=np.float32),
        CLASSES,
        nms_keep=np.array([0, 1], dtype=np.int32),
        score_thresholds=thresholds)
    assert classes.tolist() == [1]
```

Add a parser/initialization test that creates an artifact for checkpoint A and passes checkpoint B; assert failure occurs before `init_detector` or image reads.

- [ ] **Step 2: Record remote red test**

```bash
pytest tests/test_big_image_inference_helpers.py -v
```

Expected: FAIL because big inference imports `CLASS_SCORE_THRESHOLDS` and has no artifact argument.

- [ ] **Step 3: Implement explicit threshold flow**

Remove `CLASS_SCORE_THRESHOLDS` and `_MIN_SCORE_FLOOR`. Load the artifact once in `main`, configure the model with `CANDIDATE_SCORE_FLOOR`, and thread the tuple through `_run_single`, `_run_batch`, `infer_big_image`, and `apply_class_thresholds`.

Keep the existing timer boundary: synchronize the selected CUDA device, start after image decode, include patch inference, projection, NMS, frozen-threshold filtering, sorting/capping and annotation construction, synchronize the same device, then stop.

- [ ] **Step 4: Record remote green test and commit**

```bash
pytest tests/test_big_image_inference_helpers.py -v
```

Expected: PASS.

```bash
git add tools/infer_big_image.py tests/test_big_image_inference_helpers.py
git commit -m "feat: apply frozen thresholds in large-image inference"
```

---

### Task 6: End-to-end shell workflow, reporting, documentation, and migration cleanup

**Files:**
- Modify: `run.sh`
- Modify: `tools/eval_recall_fdr.py`
- Modify: `README.md`
- Modify: `tests/test_eval_recall_fdr_cli.py`
- Create: `tests/test_adaptive_threshold_pipeline_contract.py`

**Interfaces:**
- Adds environment variable: `MAX_THRESHOLD_FDR`, default `0.19`.
- Produces: `${FINETUNE_WORK}/final_thresholds.json` and companion CSV/curve/filtered-prediction files.
- Validation metrics consume `${FINETUNE_WORK}/final_thresholds_filtered_preds.json`.
- Big-image inference consumes `--thresholds ${FINETUNE_WORK}/final_thresholds.json`.

- [ ] **Step 1: Write failing CLI/reporting tests**

Update the standalone evaluator tests so they use already-filtered predictions and assert no hard-coded threshold import is required. Add a report field carrying optional threshold provenance when passed through the prediction JSON metadata, without changing TP/FP/FN semantics.

- [ ] **Step 2: Write an executable shell contract test**

Create a temporary fake project with stub `python` and `git` executables that record arguments, placeholder required files/directories, and generated checkpoint/output artifacts. Run `bash run.sh` with all path variables redirected into that temporary project. Assert the recorded command sequence contains:

```text
tools/eval_val_to_json.py ... --out <finetune>/val_preds_dense.json
tools/search_recall_fdr_thresholds.py ... --checkpoint <stage2-best> --max-official-fdr 0.19 --out-prefix <finetune>/final_thresholds
tools/eval_recall_fdr.py ... --pred <finetune>/final_thresholds_filtered_preds.json
tools/infer_big_image.py ... --thresholds <finetune>/final_thresholds.json
```

Assert no recorded command reads `ret/*.csv` and no command invokes threshold search after big-image or official-test inference.

- [ ] **Step 3: Record remote red tests**

```bash
pytest tests/test_eval_recall_fdr_cli.py \
  tests/test_adaptive_threshold_pipeline_contract.py -v
```

Expected: FAIL because `run.sh` still exports fixed-threshold predictions and never calls the search CLI.

- [ ] **Step 4: Update `run.sh` in strict order**

Add:

```bash
: "${MAX_THRESHOLD_FDR:=0.19}"
VAL_DENSE_PRED="$FINETUNE_WORK/val_preds_dense.json"
FINAL_THRESHOLD_PREFIX="$FINETUNE_WORK/final_thresholds"
FINAL_THRESHOLD_JSON="${FINAL_THRESHOLD_PREFIX}.json"
FINAL_FILTERED_PRED="${FINAL_THRESHOLD_PREFIX}_filtered_preds.json"
```

After requiring `STAGE2_BEST`, execute dense export, then:

```bash
python tools/search_recall_fdr_thresholds.py \
    --pred "$VAL_DENSE_PRED" \
    --gt "$OFFICIAL_VAL_ANN" \
    --checkpoint "$STAGE2_BEST" \
    --classes 25 --names "$NAMES" \
    --max-official-fdr "$MAX_THRESHOLD_FDR" \
    --target-official-recall 0.85 \
    --out-prefix "$FINAL_THRESHOLD_PREFIX"
require_file "$FINAL_THRESHOLD_JSON"
require_file "$FINAL_FILTERED_PRED"
```

Evaluate ordinary val from `FINAL_FILTERED_PRED`. Pass `--thresholds "$FINAL_THRESHOLD_JSON"` to big-image inference. Print every threshold artifact path in the final artifact summary.

- [ ] **Step 5: Update documentation and remove obsolete claims**

README must describe:

- three-superclass exact search during each training validation epoch;
- post-training 25-class search on the 20% validation split;
- checkpoint-bound frozen JSON applied unchanged to big images and official test;
- `MAX_THRESHOLD_FDR=0.19` override;
- historical CSV files are not runtime inputs.

Update `tools/eval_recall_fdr.py` documentation to state that it evaluates its input predictions exactly as supplied. It must not import or apply decision thresholds.

- [ ] **Step 6: Record remote green tests**

```bash
pytest tests/test_eval_recall_fdr_cli.py \
  tests/test_adaptive_threshold_pipeline_contract.py -v
```

Expected: PASS.

- [ ] **Step 7: Run final static migration checks locally**

```bash
rg -n "CLASS_SCORE_THRESHOLDS|threshold_search_fdr_0\.19_selected|ret/.*\.csv" \
  mmdet tools configs run.sh README.md tests
rg -n "search_recall_fdr_thresholds" run.sh
git diff --check
git status --short
```

Expected: the first search has no production matches; test/documentation matches are only explicit assertions that historical runtime coupling is absent. The second search has exactly the post-stage-2 validation calibration command. `git diff --check` reports no errors, and unrelated pre-existing changes remain unstaged.

- [ ] **Step 8: Record the full remote verification suite**

```bash
pytest tests/test_official_metrics.py \
  tests/test_superclass_threshold_search.py \
  tests/test_aircraft_official_evaluate.py \
  tests/test_official_checkpoint_hooks.py \
  tests/test_threshold_artifact.py \
  tests/test_recall_fdr_threshold_search.py \
  tests/test_validation_prediction_filtering.py \
  tests/test_big_image_inference_helpers.py \
  tests/test_eval_recall_fdr_cli.py \
  tests/test_adaptive_threshold_pipeline_contract.py -v
bash -n run.sh
```

Expected: all tests pass and Bash syntax exits `0` on the remote training server.

- [ ] **Step 9: Commit integration changes**

```bash
git add run.sh tools/eval_recall_fdr.py README.md \
  tests/test_eval_recall_fdr_cli.py \
  tests/test_adaptive_threshold_pipeline_contract.py
git commit -m "feat: calibrate and freeze thresholds after training"
```

---

### Task 7: Final design-conformance review and remote handoff

**Files:**
- Review: all files changed in Tasks 1-6
- Modify only if review identifies an in-scope defect.

**Interfaces:**
- Produces a clean, checkpoint-bound flow: dense predictions → validation search → frozen artifact → filtered validation/big-image/official-test inference.

- [ ] **Step 1: Map every design requirement to code and tests**

Create a review checklist covering candidate floor `0.0`, exact three-group training search, FDR `0.19`, existing best comparator, 25-class final search, schema version, SHA-256 binding, dense validation export, big-image loading, official-test non-search rule, and preservation of 8:2/two-stage behavior.

- [ ] **Step 2: Inspect data leakage boundaries**

Confirm only the 20% validation GT is supplied to search. Confirm no command accepts official-test GT and no script automatically searches when `--thresholds` is supplied.

- [ ] **Step 3: Inspect failure behavior**

Confirm missing superclass GT blocks best saving; malformed artifacts, checkpoint mismatch, image-set mismatch, and missing outputs fail before inference/reporting; no historical threshold fallback exists.

- [ ] **Step 4: Inspect Git isolation**

```bash
git status --short
git log --oneline -8
git diff --check HEAD~6..HEAD
```

Confirm `mmdet/datasets/pipelines/rand_rotate.py`, `new_run.sh`, `test.py`, and user CSV files were neither overwritten nor included in implementation commits.

- [ ] **Step 5: Provide remote execution handoff**

Document these ordered commands for the training server:

```bash
cd /root/autodl-tmp/model_v2
unset PROJECT_ROOT DATA_ROOT WORK_ROOT OFFICIAL_CONFIG FINETUNE_CONFIG
export PROJECT_ROOT=/root/autodl-tmp/model_v2
export DATA_ROOT=/root/autodl-tmp/model_v2/data
export MAX_THRESHOLD_FDR=0.19
pytest tests/test_official_metrics.py \
  tests/test_superclass_threshold_search.py \
  tests/test_aircraft_official_evaluate.py \
  tests/test_official_checkpoint_hooks.py \
  tests/test_threshold_artifact.py \
  tests/test_recall_fdr_threshold_search.py \
  tests/test_validation_prediction_filtering.py \
  tests/test_big_image_inference_helpers.py \
  tests/test_eval_recall_fdr_cli.py \
  tests/test_adaptive_threshold_pipeline_contract.py -v
bash -n run.sh
bash run.sh
```

The expected final threshold file is `work_dirs/shiprs_finetune_stage/final_thresholds.json`; its checkpoint SHA-256 must match `work_dirs/shiprs_finetune_stage/best_official_recall_fdr.pth`.
