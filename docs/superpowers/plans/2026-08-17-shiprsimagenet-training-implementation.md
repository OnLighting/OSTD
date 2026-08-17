# ShipRSImageNet Training Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible ShipRSImageNet-to-25-class data pipeline, leakage-safe grouped OOF evaluation, source-balanced mixed training, official-domain fine-tuning, and frozen class-wise threshold application.

**Architecture:** Convert ShipRSImageNet into a 25-category COCO manifest with conservative HM/LQS/QHS mappings and `iscrowd` ignore boxes, without copying source images. Rebuild the official dataset as one grouped 10% blind holdout plus three grouped OOF folds, train three identical fold models and one final model, then choose thresholds from merged OOF predictions and apply them unchanged to the blind/hidden test predictions.

**Tech Stack:** Python 3, MMDetection 2.x, MMCV, PyCOCO-style JSON, Pillow, NumPy, Bash, pytest.

## Global Constraints

- Do not run project programs or tests on the current Windows workspace; all command verification in this plan runs on the configured Linux training server.
- Never initialize or commit into `C:/Users/23563/.git`; the current workspace has no independent Git repository.
- Keep the final model output at exactly 25 classes in the order defined by `mmdet/datasets/aircraft.py`.
- Use horizontal boxes end-to-end. Official validation and evaluation use HBB labels and HBB IoU; do not use rotated IoU in this workflow.
- ShipRSImageNet may enter training only; official OOF folds and the 10% local blind holdout contain official data only.
- Split official samples by parent scene, never by individual crop.
- Do not initialize OOF or final runs from `work_dirs/arfc_only_v2`; historical split exposure would leak into the rebuilt validation protocol.
- Use the same generic initialization, data rules, epochs, and seed policy for all three OOF runs.
- Search thresholds only on merged OOF predictions; freeze and apply them unchanged to blind and hidden test predictions.
- Use ShipRSImageNet under its academic-use/Google Earth terms, record the archive SHA-256, and do not redistribute source imagery.
- Every commit command below is run only in an independent Git checkout on the training server. If that checkout has no `.git`, skip the commit command and save `git diff --no-index` or a file manifest instead.

---

## File Structure

- `tools/dataset_utils/shiprs_mapping.py`: canonical 25-class names, ShipRS label normalization, conservative mapping, and source-category decisions.
- `tools/prepare_shiprs.py`: discover/validate ShipRS COCO data, convert mapped objects and ignored objects, and write audit artifacts.
- `tools/build_grouped_oof_splits.py`: derive parent-scene keys and generate blind/fold COCO manifests from the full official YOLO pool.
- `tools/audit_dataset_leakage.py`: exact/difference-hash duplicate audit across official partitions and ShipRS.
- `mmdet/datasets/dataset_wrappers.py`: add a deterministic two-source ratio wrapper.
- `mmdet/datasets/builder.py`: construct the new wrapper from config.
- `configs/_base_/datasets/aircraft_shiprs_oof.py`: mixed train and official-only validation pipelines, including ignore boxes.
- `configs/bafnet/aircraft_bafnet_shiprs_fold{0,1,2}.py`: three OOF configs.
- `configs/bafnet/aircraft_bafnet_shiprs_final.py`: final mixed-stage config.
- `configs/bafnet/aircraft_bafnet_shiprs_finetune.py`: official-only low-LR continuation config.
- `tools/eval_val_to_json.py`: infer exactly the images listed by GT, including nested file names.
- `tools/merge_oof_predictions.py`: validate and merge three disjoint OOF GT/prediction pairs.
- `tools/apply_class_thresholds.py`: apply a frozen selected-threshold CSV to any dense prediction JSON.
- `tools/run_shiprs_oof.sh`: training-server orchestration with fail-fast artifact checks.
- `tests/test_shiprs_mapping.py`, `tests/test_prepare_shiprs.py`, `tests/test_grouped_oof_splits.py`, `tests/test_source_balanced_dataset.py`, `tests/test_oof_tools.py`: focused server-side tests.
- `docs/shiprsimagenet_reproduction.md`: data provenance, commands, outputs, and report-ready disclosure.

---

### Task 1: Canonical ShipRS Category Mapping

**Files:**
- Create: `tools/dataset_utils/__init__.py`
- Create: `tools/dataset_utils/shiprs_mapping.py`
- Create: `tests/test_shiprs_mapping.py`

**Interfaces:**
- Produces: `CLASS_NAMES: tuple[str, ...]` containing exactly the 25 literal names shown in Step 3.
- Produces: `normalize_shiprs_name(name: str) -> str`
- Produces: `map_shiprs_category(name: str, enable_ms: bool = False) -> MappingDecision`
- Produces: `MappingDecision(action: Literal['map','ignore','drop'], target_id: int | None, reason: str)`

- [ ] **Step 1: Write mapping tests**

```python
from tools.dataset_utils.shiprs_mapping import (
    CLASS_NAMES, map_shiprs_category, normalize_shiprs_name)


def test_class_order_is_competition_order():
    assert len(CLASS_NAMES) == 25
    assert CLASS_NAMES[:4] == ('HM', 'LQS', 'QHS', 'MS')
    assert CLASS_NAMES[-1] == 'FSC'


def test_conservative_ship_mappings():
    assert map_shiprs_category('Nimitz').target_id == 0
    assert map_shiprs_category('Wasp LL').target_id == 1
    assert map_shiprs_category('Arleigh Burke DD').target_id == 2
    assert map_shiprs_category('Other Warship').action == 'ignore'
    assert map_shiprs_category('Dock').action == 'drop'


def test_ms_is_disabled_by_default():
    assert map_shiprs_category('Container Ship').action == 'ignore'
    assert map_shiprs_category('Container Ship', enable_ms=True).target_id == 3


def test_normalization_is_case_and_separator_stable():
    assert normalize_shiprs_name('  arleigh_burke-dd ') == 'ARLEIGH BURKE DD'
```

- [ ] **Step 2: Run the focused test on the training server and verify failure**

Run:

```bash
pytest -q tests/test_shiprs_mapping.py
```

Expected: collection fails because `tools.dataset_utils.shiprs_mapping` does not exist.

- [ ] **Step 3: Implement the immutable mapping module**

Use a frozen dataclass and explicit normalized-name tables. Set `Dock` to `drop`; set every other non-mapped ship class to `ignore`. Include all names listed in the approved design document. Do not infer a target from substrings such as `warship` or `landing`.

```python
from dataclasses import dataclass
from typing import Literal, Optional


CLASS_NAMES = (
    'HM', 'LQS', 'QHS', 'MS',
    'A1_SU-35', 'A2_C-130', 'A3_C-17', 'A4_C-5', 'A5_F-16',
    'A6_TU-160', 'A7_E-3', 'A8_B-52', 'A9_P-3C', 'A10_B-1B',
    'A11_E-8', 'A12_TU-22', 'A13_F-15', 'A14_KC-135',
    'A15_F-22', 'A16_FA-18', 'A17_TU-95', 'A18_KC-10',
    'A19_SU-34', 'A20_SU-24', 'FSC')


@dataclass(frozen=True)
class MappingDecision:
    action: Literal['map', 'ignore', 'drop']
    target_id: Optional[int]
    reason: str
```

- [ ] **Step 4: Run tests**

Run: `pytest -q tests/test_shiprs_mapping.py`  
Expected: `4 passed`.

- [ ] **Step 5: Commit in the training-server checkout**

```bash
git add tools/dataset_utils tests/test_shiprs_mapping.py
git commit -m "feat: define conservative ShipRS category mapping"
```

---

### Task 2: ShipRS COCO Conversion and Ignore Semantics

**Files:**
- Create: `tools/prepare_shiprs.py`
- Create: `tests/test_prepare_shiprs.py`

**Interfaces:**
- Consumes: `map_shiprs_category()` and `CLASS_NAMES` from Task 1.
- Produces: `discover_coco_annotations(root: Path) -> list[Path]`
- Produces: `to_horizontal_bbox(annotation: dict) -> tuple[list[float], str]`, returning COCO `[x, y, width, height]` and source geometry `hbb`, `obb-envelope`, or `polygon-envelope`.
- Produces: `convert_shiprs(coco_paths: list[Path], shiprs_root: Path, enable_ms: bool) -> tuple[dict, list[dict]]`
- Produces CLI artifacts:
  - `data/external/shiprs_mapped_train.json`
  - `data/external/shiprs_mapping_audit.csv`
  - `data/external/shiprs_summary.json`

- [ ] **Step 1: Write a miniature COCO conversion test**

```python
def test_mapped_and_unknown_ships_are_preserved(tmp_path):
    source = make_shiprs_fixture(
        tmp_path,
        categories={10: 'Nimitz', 20: 'Other Warship', 30: 'Dock'},
        annotations=[
            {'id': 1, 'image_id': 7, 'category_id': 10, 'bbox': [1, 2, 20, 30]},
            {'id': 2, 'image_id': 7, 'category_id': 20, 'bbox': [40, 2, 20, 30]},
            {'id': 3, 'image_id': 7, 'category_id': 30, 'bbox': [70, 2, 20, 30]},
        ])
    output, audit = convert_shiprs([source], tmp_path, enable_ms=False)
    assert [a['category_id'] for a in output['annotations']] == [0, 0]
    assert output['annotations'][0]['iscrowd'] == 0
    assert output['annotations'][1]['iscrowd'] == 1
    assert output['annotations'][1]['source_category_name'] == 'Other Warship'
    assert {r['action'] for r in audit} == {'map', 'ignore', 'drop'}
```

Add tests for duplicate image names, invalid boxes, missing images, unknown category IDs, and all mapped categories producing zero instances.

Add an HBB-priority test:

```python
def test_native_hbb_wins_over_rotated_geometry():
    ann = {'bbox': [10, 20, 30, 40], 'robndbox': [25, 40, 30, 40, 0.7]}
    bbox, source = to_horizontal_bbox(ann)
    assert bbox == [10, 20, 30, 40]
    assert source == 'hbb'


def test_polygon_fallback_uses_axis_aligned_envelope():
    ann = {'segmentation': [[10, 20, 30, 10, 40, 35, 15, 45]]}
    bbox, source = to_horizontal_bbox(ann)
    assert bbox == [10, 10, 30, 35]
    assert source == 'polygon-envelope'
```

- [ ] **Step 2: Verify the tests fail on the training server**

Run: `pytest -q tests/test_prepare_shiprs.py`  
Expected: import failure for `tools.prepare_shiprs`.

- [ ] **Step 3: Implement annotation discovery and validation**

The CLI is exact:

```text
python tools/prepare_shiprs.py \
  --shiprs-root external_data/ShipRSImageNet \
  --out-json data/external/shiprs_mapped_train.json \
  --audit-csv data/external/shiprs_mapping_audit.csv \
  --summary-json data/external/shiprs_summary.json
```

Discovery accepts official COCO JSON files whose image files resolve beneath `external_data/ShipRSImageNet`. Select native HBB annotations before any SBB/OBB/polygon variant. It merges annotated training and validation subsets, excludes any annotation-free ShipRS test subset, and rewrites each `file_name` relative to the ShipRS root. It rejects duplicate `(resolved image path, source annotation id)` records.

- [ ] **Step 4: Implement mapped, ignored, and dropped annotations**

Mapped objects receive target category IDs 0, 1, or 2 and `iscrowd=0`. Ignored ship objects receive `category_id=0` and `iscrowd=1`; category 0 is a carrier only as a COCO compatibility sentinel, and `iscrowd` prevents it from becoming an HM target. Dropped `Dock` annotations are absent from output. Keep `source_category_id`, `source_category_name`, and `mapping_reason` as audit-only custom fields.

Write all 25 competition categories into the output JSON. Clamp boxes to image bounds only when the clipped box retains positive area; record each clamp in the audit CSV.

Write `geometry_source` for every annotation. Native HBB remains unchanged apart from bounds validation. OBB or polygon fallback converts to the axis-aligned envelope using coordinate minima/maxima. The output contains no angle field, and all later matching uses the existing horizontal `xywh` IoU implementation.

- [ ] **Step 5: Run tests and a server-side conversion dry run**

Run:

```bash
pytest -q tests/test_prepare_shiprs.py tests/test_shiprs_mapping.py
python tools/prepare_shiprs.py \
  --shiprs-root external_data/ShipRSImageNet \
  --out-json data/external/shiprs_mapped_train.json \
  --audit-csv data/external/shiprs_mapping_audit.csv \
  --summary-json data/external/shiprs_summary.json
```

Expected: tests pass; summary reports non-zero HM, LQS, and QHS mapped instances, non-zero ignored instances, zero missing images, and zero invalid output boxes.

The same summary reports counts for `hbb`, `obb-envelope`, and `polygon-envelope`; `hbb` must be the selected source whenever the package provides native HBB for that object.

- [ ] **Step 6: Commit**

```bash
git add tools/prepare_shiprs.py tests/test_prepare_shiprs.py
git commit -m "feat: convert ShipRS annotations into competition labels"
```

---

### Task 3: Parent-Scene Grouping and OOF Manifests

**Files:**
- Create: `tools/build_grouped_oof_splits.py`
- Create: `tests/test_grouped_oof_splits.py`

**Interfaces:**
- Produces: `derive_scene_key(file_name: str) -> tuple[str, str]` returning `(source, scene_key)`.
- Produces: `assign_grouped_partitions(group_stats: list[GroupStats], seed: int = 20260817) -> dict[str, str]` with labels `blind`, `fold0`, `fold1`, `fold2`.
- Produces:
  - `data/oof/instances_blind.json`
  - `data/oof/instances_fold0.json`
  - `data/oof/instances_fold1.json`
  - `data/oof/instances_fold2.json`
  - `data/oof/instances_train_for_fold0.json`
  - `data/oof/instances_train_for_fold1.json`
  - `data/oof/instances_train_for_fold2.json`
  - `data/oof/instances_development.json`
  - `data/oof/group_assignments.csv`
  - `data/oof/split_summary.json`

- [ ] **Step 1: Write grouping and invariance tests**

```python
def test_crop_siblings_share_scene_key():
    assert derive_scene_key('01-PAN-20250407-X-CCD6_6_crop4.jpg') == (
        '01-PAN', '01-PAN-20250407-X-CCD6_6')
    assert derive_scene_key('01-PAN-20250407-X-CCD6_6_crop9.jpg')[1] == (
        '01-PAN-20250407-X-CCD6_6')


def test_group_never_crosses_partitions():
    assignment = assign_grouped_partitions(make_group_stats(), seed=20260817)
    assert set(assignment.values()) == {'blind', 'fold0', 'fold1', 'fold2'}
    assert len(assignment) == len(set(assignment))


def test_output_images_are_disjoint_and_complete(tmp_path):
    outputs = build_manifests(make_official_fixture(tmp_path), tmp_path / 'oof')
    image_sets = [set(read_json(p)['images'][i]['file_name']
                      for i in range(len(read_json(p)['images'])))
                  for p in outputs]
    assert len(set.union(*image_sets)) == sum(map(len, image_sets))
    assert set.union(*image_sets) == set(make_official_names())
```

- [ ] **Step 2: Verify failure**

Run: `pytest -q tests/test_grouped_oof_splits.py`  
Expected: import failure for the new splitter.

- [ ] **Step 3: Implement scene-key parsing**

Strip only a terminal crop/tile suffix matching `(?:_crop|_tile|_patch)[-_]?\d+$`, after removing the extension. Preserve source prefixes. Treat MAR20 names as independent groups unless the filename contains an explicit common parent identifier. Emit `grouping_rule` in the audit CSV for every image.

- [ ] **Step 4: Implement deterministic weighted group assignment**

Build a 25-dimensional target-count vector for each scene group. Sort groups rarest-first using inverse class frequency, then greedily assign each whole group to the partition minimizing squared normalized deviation from target weights `(0.10, 0.30, 0.30, 0.30)` across image count, source count, and per-class target count. Use seed `20260817` only to break equal-cost ties.

Fail if any scene key crosses partitions, any image is missing/duplicated, or any class present in at least four distinct groups is absent from a partition.

- [ ] **Step 5: Run tests and generate manifests on the training server**

```bash
pytest -q tests/test_grouped_oof_splits.py
python tools/build_grouped_oof_splits.py \
  --root new_data \
  --out-dir data/oof \
  --seed 20260817
```

Expected: four disjoint manifests cover all official labeled images; `split_summary.json` reports approximately 10/30/30/30 percent by image count and lists per-class box counts.

- [ ] **Step 6: Manually review the generated split summary before training**

Reject the split if HM, LQS, or FSC is absent from a partition despite having enough distinct scene groups, or if a single source is absent from the blind set. Changing the seed requires regenerating and refreezing all manifests before any OOF training begins.

- [ ] **Step 7: Commit**

```bash
git add tools/build_grouped_oof_splits.py tests/test_grouped_oof_splits.py
git commit -m "feat: build scene-grouped blind and OOF splits"
```

---

### Task 4: Exact and Near-Duplicate Leakage Audit

**Files:**
- Create: `tools/audit_dataset_leakage.py`
- Extend: `tests/test_grouped_oof_splits.py`

**Interfaces:**
- Produces: `sha256_file(path: Path) -> str`
- Produces: `difference_hash(path: Path, hash_size: int = 8) -> int`
- Produces: `hamming_distance(left: int, right: int) -> int`
- Produces: `data/oof/leakage_audit.csv` and `data/external/shiprs_exclusions.txt`.

- [ ] **Step 1: Add duplicate-audit tests**

```python
def test_exact_and_reencoded_duplicates_are_reported(tmp_path):
    paths = make_duplicate_fixture(tmp_path)
    rows = audit_images(paths, near_distance=4)
    assert any(r.kind == 'exact' for r in rows)
    assert any(r.kind == 'near' for r in rows)


def test_official_cross_partition_duplicate_is_fatal(tmp_path):
    with pytest.raises(ValueError, match='official partition leakage'):
        audit_manifests(make_cross_partition_fixture(tmp_path), near_distance=4)
```

- [ ] **Step 2: Implement streaming hashes and constrained near matching**

Use SHA-256 for exact matches. Use Pillow grayscale 9×8 difference hash for near matches, but compare dHashes only within compatible width/height aspect buckets to avoid quadratic all-pairs work. Mark distance `<=4` as a review candidate; do not automatically delete near matches.

- [ ] **Step 3: Define enforcement rules**

- Exact or approved near duplicate across two official partitions: fatal error; rebuild partitions.
- Exact duplicate from ShipRS to any official partition: add ShipRS image to exclusions.
- Near duplicate from ShipRS to official OOF/blind: add to review CSV; exclusion becomes effective only after audit approval.
- Duplicate inside ShipRS: retain one deterministic canonical path and exclude later paths.

The review CSV has a required `decision` column whose only accepted values are `exclude` and `keep`. After visual review, rerun the audit with `--reviewed-near-csv data/oof/leakage_audit.csv`; reject blank or unknown decisions, and append every `exclude` ShipRS path to `shiprs_exclusions.txt`.

- [ ] **Step 4: Run tests and audit**

```bash
pytest -q tests/test_grouped_oof_splits.py
python tools/audit_dataset_leakage.py \
  --official-dir data/oof \
  --official-image-root new_data/images/train \
  --shiprs-json data/external/shiprs_mapped_train.json \
  --shiprs-root external_data/ShipRSImageNet \
  --out-csv data/oof/leakage_audit.csv \
  --shiprs-exclusions data/external/shiprs_exclusions.txt \
  --near-distance 4
```

Expected: no official cross-partition matches; all exact ShipRS overlaps listed in exclusions; near candidates have resolvable image paths for manual review.

- [ ] **Step 5: Re-run ShipRS preparation with exclusions**

```bash
python tools/prepare_shiprs.py \
  --shiprs-root external_data/ShipRSImageNet \
  --exclude-list data/external/shiprs_exclusions.txt \
  --out-json data/external/shiprs_mapped_train.json \
  --audit-csv data/external/shiprs_mapping_audit.csv \
  --summary-json data/external/shiprs_summary.json
```

Expected: summary records the exclusion count and no excluded image remains in the output manifest.

- [ ] **Step 6: Commit**

```bash
git add tools/audit_dataset_leakage.py tools/prepare_shiprs.py tests/test_grouped_oof_splits.py
git commit -m "feat: audit cross-source and cross-fold leakage"
```

---

### Task 5: Deterministic Source-Balanced Dataset Wrapper

**Files:**
- Modify: `mmdet/datasets/dataset_wrappers.py`
- Modify: `mmdet/datasets/builder.py`
- Create: `tests/test_source_balanced_dataset.py`

**Interfaces:**
- Produces registered dataset `SourceBalancedDataset`.
- Constructor: `SourceBalancedDataset(datasets, source_weights=(0.6, 0.4), epoch_length=None, seed=20260817)`.
- `__getitem__`, `__len__`, `get_cat_ids`, and `flag` remain compatible with MMDetection group samplers.

- [ ] **Step 1: Write wrapper tests with two fake datasets**

```python
def test_source_ratio_and_index_bounds():
    ds = SourceBalancedDataset(
        [FakeDataset(5, flag=0), FakeDataset(20, flag=1)],
        source_weights=(0.6, 0.4), epoch_length=100, seed=7)
    assert len(ds) == 100
    assert ds.source_counts == (60, 40)
    assert sum(item['source'] == 0 for item in ds) == 60
    assert sum(item['source'] == 1 for item in ds) == 40
    assert len(ds.flag) == 100


def test_builder_constructs_source_balanced_dataset():
    cfg = dict(type='SourceBalancedDataset', datasets=[fake_cfg(3), fake_cfg(7)],
               source_weights=(0.5, 0.5), epoch_length=20, seed=1)
    assert len(build_dataset(cfg)) == 20
```

- [ ] **Step 2: Verify failure**

Run: `pytest -q tests/test_source_balanced_dataset.py`  
Expected: `SourceBalancedDataset` is undefined.

- [ ] **Step 3: Implement a static deterministic index schedule**

Normalize weights to sum to one. Derive integer source counts with largest-remainder allocation so counts sum exactly to `epoch_length`. If `epoch_length` is absent, choose the smallest length for which every source contributes at least its full dataset length once: `max(ceil(len(ds) / weight))`. Cycle each source index deterministically with a seeded offset; rely on MMDetection's sampler for epoch shuffling.

Expose `source_counts` for logging and tests. Build `flag` and `get_cat_ids` through the scheduled `(source_id, local_index)` list.

- [ ] **Step 4: Add builder support**

Import the wrapper in `build_dataset()` and recursively build every config in `cfg['datasets']`. Pass `source_weights`, `epoch_length`, and `seed` exactly.

- [ ] **Step 5: Run tests**

Run: `pytest -q tests/test_source_balanced_dataset.py`  
Expected: all wrapper and builder tests pass.

- [ ] **Step 6: Commit**

```bash
git add mmdet/datasets/dataset_wrappers.py mmdet/datasets/builder.py tests/test_source_balanced_dataset.py
git commit -m "feat: add deterministic source-balanced dataset wrapper"
```

---

### Task 6: Mixed OOF and Final Training Configs

**Files:**
- Create: `configs/_base_/datasets/aircraft_shiprs_oof.py`
- Create: `configs/bafnet/aircraft_bafnet_shiprs_fold0.py`
- Create: `configs/bafnet/aircraft_bafnet_shiprs_fold1.py`
- Create: `configs/bafnet/aircraft_bafnet_shiprs_fold2.py`
- Create: `configs/bafnet/aircraft_bafnet_shiprs_final.py`
- Create: `configs/bafnet/aircraft_bafnet_shiprs_finetune.py`
- Create: `tests/test_shiprs_configs.py`

**Interfaces:**
- Consumes all manifests from Tasks 2–4 and `SourceBalancedDataset` from Task 5.
- Produces five loadable MMCV configs with a 25-class model.

- [ ] **Step 1: Write static config tests**

```python
@pytest.mark.parametrize('name', [
    'aircraft_bafnet_shiprs_fold0.py',
    'aircraft_bafnet_shiprs_fold1.py',
    'aircraft_bafnet_shiprs_fold2.py',
    'aircraft_bafnet_shiprs_final.py',
    'aircraft_bafnet_shiprs_finetune.py'])
def test_shiprs_config_contract(name):
    cfg = Config.fromfile(str(CONFIG_DIR / name))
    assert all(h.num_classes == 25 for h in cfg.model.roi_head.bbox_head)
    if cfg.data.train.type == 'SourceBalancedDataset':
        assert 'gt_bboxes_ignore' in cfg.data.train.datasets[1].pipeline[-1].keys
    assert cfg.model.train_cfg.rpn.assigner.ignore_iof_thr == 0.5
    assert all(s.assigner.ignore_iof_thr == 0.5
               for s in cfg.model.train_cfg.rcnn)
```

- [ ] **Step 2: Create the shared mixed dataset base**

Use `LoadAnnotations(with_bbox=True)`, and collect `img`, `gt_bboxes`, `gt_labels`, and `gt_bboxes_ignore`. Configure `SourceBalancedDataset` with official source weight `0.6`, ShipRS source weight `0.4`, seed `20260817`, official `ClassBalancedDataset(oversample_thr=1e-3)` as source 0, and plain `AircraftDataset` as source 1.

Set official image prefix to `new_data/images/train/`. Set ShipRS image prefix to `external_data/ShipRSImageNet/`, because Task 2 stores relative source paths in the external COCO manifest.

- [ ] **Step 3: Enable ignore behavior in the mixed model configs**

Override all assigners used in training:

```python
model = dict(train_cfg=dict(
    rpn=dict(assigner=dict(ignore_iof_thr=0.5)),
    rcnn=[
        dict(assigner=dict(ignore_iof_thr=0.5)),
        dict(assigner=dict(ignore_iof_thr=0.5)),
        dict(assigner=dict(ignore_iof_thr=0.5)),
    ]))
```

Preserve the remaining assigner fields through a complete copied dictionary or `_delete_=False` merge verified by `Config.fromfile`; do not replace assigners with incomplete dictionaries.

- [ ] **Step 4: Define the three fold configs**

For fold 0, train official folds 1+2 and validate fold 0; for fold 1, train 0+2 and validate 1; for fold 2, train 0+1 and validate 2. Create combined official training manifests during Task 3 named `instances_train_for_fold{0,1,2}.json`, so each config references one official COCO JSON rather than a nested official concat.

Set `model.test_cfg.rcnn.score_thr=0.001` for dense validation export. Do not set `load_from` or `resume_from`; each run starts from the config's generic pretrained backbone path.

- [ ] **Step 5: Define final mixed and official-only fine-tune configs**

`aircraft_bafnet_shiprs_final.py` trains on `instances_development.json` plus mapped ShipRS with the same 60/40 source ratio and the same generic initialization. `aircraft_bafnet_shiprs_finetune.py` trains only `instances_development.json`, sets `optimizer.lr=0.0005`, uses 20 epochs, and uses steps `[14, 18]`. Supply the final mixed checkpoint through `--cfg-options load_from=work_dirs/shiprs_final_mixed/best_bbox_mAP.pth`; never use positional `resume_from` for this stage.

- [ ] **Step 6: Run config tests and dataset construction smoke checks on the server**

```bash
pytest -q tests/test_shiprs_configs.py tests/test_source_balanced_dataset.py
python -c "from mmcv import Config; from mmdet.datasets import build_dataset; c=Config.fromfile('configs/bafnet/aircraft_bafnet_shiprs_fold0.py'); d=build_dataset(c.data.train); print(len(d), d.source_counts)"
```

Expected: configs load, model has 25 outputs, ignore IoF is 0.5 for RPN and all cascade stages, and source counts are approximately 60/40 exactly at the configured epoch length.

- [ ] **Step 7: Commit**

```bash
git add configs/_base_/datasets/aircraft_shiprs_oof.py configs/bafnet/aircraft_bafnet_shiprs_*.py tests/test_shiprs_configs.py
git commit -m "feat: configure ShipRS OOF and final training"
```

---

### Task 7: Fold-Safe Dense Inference, OOF Merge, and Frozen Threshold Application

**Files:**
- Modify: `tools/eval_val_to_json.py`
- Create: `tools/merge_oof_predictions.py`
- Create: `tools/apply_class_thresholds.py`
- Create: `tests/test_oof_tools.py`

**Interfaces:**
- `eval_val_to_json.py` resolves and infers only `gt['images']` in GT order.
- Produces: `merge_oof(gt_paths: list[Path], pred_paths: list[Path]) -> tuple[dict, dict]`.
- Produces: `load_thresholds(csv_path: Path) -> dict[int, float]`.
- Produces: `filter_predictions(pred: dict, thresholds: dict[int, float]) -> dict`.

- [ ] **Step 1: Write regression tests for GT-driven inference input resolution**

Extract a pure helper:

```python
def test_resolve_gt_image_paths_ignores_unlisted_files(tmp_path):
    (tmp_path / 'listed.jpg').write_bytes(b'x')
    (tmp_path / 'extra.jpg').write_bytes(b'x')
    gt = {'images': [{'id': 9, 'file_name': 'listed.jpg'}]}
    assert resolve_gt_image_paths(gt, tmp_path) == [(9, 'listed.jpg', tmp_path / 'listed.jpg')]
```

The current implementation enumerates the entire directory; this test must fail before modification.

- [ ] **Step 2: Modify dense inference to iterate GT records**

For every GT image, resolve `img_dir / file_name`, require it to exist, and preserve its exact `image_id` and `file_name`. Reject duplicate GT IDs or names. Do not enumerate unrelated files. This permits all OOF folds to share `new_data/images/train/` safely.

- [ ] **Step 3: Write OOF merge tests**

```python
def test_merge_rejects_overlapping_fold_images(tmp_path):
    with pytest.raises(ValueError, match='overlapping image_id'):
        merge_oof(two_gt_paths_with_same_id(tmp_path), two_pred_paths(tmp_path))


def test_merge_preserves_all_images_and_reassigns_annotation_ids(tmp_path):
    gt, pred = merge_oof(disjoint_gt_paths(tmp_path), disjoint_pred_paths(tmp_path))
    assert len(gt['images']) == 3
    assert len({a['id'] for a in pred['annotations']}) == len(pred['annotations'])
    assert {a['image_id'] for a in pred['annotations']} <= {i['id'] for i in gt['images']}
```

- [ ] **Step 4: Implement strict OOF merge validation**

Require three GT/prediction pairs, identical 25-category name/order, exact per-fold prediction/GT image ID and filename equality, disjoint image IDs and filenames across folds, and finite scores. Reassign annotation IDs sequentially while preserving image IDs.

Exact command:

```bash
python tools/merge_oof_predictions.py \
  --gt data/oof/instances_fold0.json data/oof/instances_fold1.json data/oof/instances_fold2.json \
  --pred work_dirs/shiprs_fold0/dense.json work_dirs/shiprs_fold1/dense.json work_dirs/shiprs_fold2/dense.json \
  --out-gt work_dirs/shiprs_oof/oof_gt.json \
  --out-pred work_dirs/shiprs_oof/oof_dense.json
```

- [ ] **Step 5: Write and implement threshold application tests**

```python
def test_thresholds_are_class_specific_and_inclusive(tmp_path):
    thresholds = {0: 0.2, 1: 0.7}
    pred = fixture_predictions([(0, 0.2), (0, 0.19), (1, 0.71), (1, 0.69)])
    kept = filter_predictions(pred, thresholds)
    assert [(a['category_id'], a['score']) for a in kept['annotations']] == [
        (0, 0.2), (1, 0.71)]
```

Reject missing/duplicate class rows, non-finite thresholds, wrong category names, and a CSV that does not cover all 25 classes.

Exact command:

```bash
python tools/apply_class_thresholds.py \
  --pred work_dirs/shiprs_final/blind_dense.json \
  --thresholds work_dirs/shiprs_oof/threshold_search_fdr_0.18_selected.csv \
  --out work_dirs/shiprs_final/blind_filtered.json
```

- [ ] **Step 6: Run tests**

Run: `pytest -q tests/test_oof_tools.py`  
Expected: all inference-resolution, merge, and threshold tests pass.

- [ ] **Step 7: Commit**

```bash
git add tools/eval_val_to_json.py tools/merge_oof_predictions.py tools/apply_class_thresholds.py tests/test_oof_tools.py
git commit -m "feat: add fold-safe OOF prediction workflow"
```

---

### Task 8: Reproducible Server Orchestration and Documentation

**Files:**
- Create: `tools/run_shiprs_oof.sh`
- Create: `docs/shiprsimagenet_reproduction.md`

**Interfaces:**
- Consumes all prior tools/configs.
- Produces three fold work directories, merged OOF predictions, selected thresholds, final mixed/fine-tuned checkpoints, and one blind evaluation report.

- [ ] **Step 1: Write a fail-fast orchestration script**

Start with `set -euo pipefail`. Require these paths before training:

```bash
external_data/ShipRSImageNet
data/external/shiprs_mapped_train.json
data/oof/instances_fold0.json
data/oof/instances_fold1.json
data/oof/instances_fold2.json
data/oof/instances_blind.json
data/oof/instances_development.json
```

Compute and save the ShipRS archive/directory manifest hash, config hashes, Git revision when available, and `pip freeze` into `work_dirs/shiprs_reproduction/`.

- [ ] **Step 2: Encode the three OOF training commands**

Use seed `20260817` and deterministic mode for all folds:

```bash
python tools/train.py configs/bafnet/aircraft_bafnet_shiprs_fold0.py work_dirs/shiprs_fold0 --seed 20260817 --deterministic
python tools/train.py configs/bafnet/aircraft_bafnet_shiprs_fold1.py work_dirs/shiprs_fold1 --seed 20260817 --deterministic
python tools/train.py configs/bafnet/aircraft_bafnet_shiprs_fold2.py work_dirs/shiprs_fold2 --seed 20260817 --deterministic
```

After each run, export dense predictions at score `0.001` using the fold's best checkpoint and matching GT manifest. The script must stop if a best checkpoint or prediction file is absent.

- [ ] **Step 3: Encode OOF merge, evaluation, and threshold search**

Merge using Task 7, then run:

```bash
python tools/eval_recall_fdr.py \
  --pred work_dirs/shiprs_oof/oof_dense.json \
  --gt work_dirs/shiprs_oof/oof_gt.json \
  --classes 25 \
  --names HM,LQS,QHS,MS,A1_SU-35,A2_C-130,A3_C-17,A4_C-5,A5_F-16,A6_TU-160,A7_E-3,A8_B-52,A9_P-3C,A10_B-1B,A11_E-8,A12_TU-22,A13_F-15,A14_KC-135,A15_F-22,A16_FA-18,A17_TU-95,A18_KC-10,A19_SU-34,A20_SU-24,FSC \
  --out-prefix work_dirs/shiprs_oof/oof_dense_metrics
python tools/search_recall_fdr_thresholds.py \
  --pred work_dirs/shiprs_oof/oof_dense.json \
  --gt work_dirs/shiprs_oof/oof_gt.json \
  --max-official-fdr 0.18 \
  --target-official-recall 0.85 \
  --out-prefix work_dirs/shiprs_oof/threshold_search_fdr_0.18
```

Require `PASS=True`. Also record 0.16, 0.17, and 0.19 frontier points, but choose the final operating point before blind inference using only OOF results.

- [ ] **Step 4: Encode final mixed training and official-domain fine-tuning**

```bash
python tools/train.py configs/bafnet/aircraft_bafnet_shiprs_final.py work_dirs/shiprs_final_mixed --seed 20260817 --deterministic
python tools/train.py configs/bafnet/aircraft_bafnet_shiprs_finetune.py work_dirs/shiprs_final_finetune \
  --seed 20260817 --deterministic \
  --cfg-options load_from=work_dirs/shiprs_final_mixed/best_bbox_mAP.pth
```

The final mixed run starts from generic initialization. The fine-tune run loads weights but resets optimizer/epoch state through `load_from`.

- [ ] **Step 5: Encode the one-time blind evaluation gate**

Export dense predictions for `instances_blind.json`, apply the frozen OOF threshold CSV, and run `eval_recall_fdr.py`. Before this step the script requires a manually created marker file `work_dirs/shiprs_oof/THRESHOLDS_FROZEN`; its content contains the SHA-256 of the selected threshold CSV. The script verifies the hash before blind inference.

Do not automatically loop back into training after blind metrics are printed.

- [ ] **Step 6: Write the reproduction document**

Document extraction target `external_data/ShipRSImageNet`, archive hash command, mapping table, excluded categories, ignore semantics, exact split seed, source ratio, four-run schedule, threshold hash, output directory tree, and the ShipRS academic-use/Google Earth disclosure. Include a table for mapped/ignored/excluded counts copied from `shiprs_summary.json` after conversion.

- [ ] **Step 7: Shell syntax check and full non-training preflight on the server**

```bash
bash -n tools/run_shiprs_oof.sh
pytest -q tests/test_shiprs_mapping.py tests/test_prepare_shiprs.py tests/test_grouped_oof_splits.py tests/test_source_balanced_dataset.py tests/test_shiprs_configs.py tests/test_oof_tools.py
python tools/run_shiprs_oof.sh --preflight-only
```

Expected: shell syntax succeeds, all tests pass, preflight validates manifests/configs/hashes without starting GPU training.

- [ ] **Step 8: Commit**

```bash
git add tools/run_shiprs_oof.sh docs/shiprsimagenet_reproduction.md
git commit -m "docs: add reproducible ShipRS OOF training workflow"
```

---

## Final Verification Checklist

- [ ] ShipRS summary contains non-zero mapped HM, LQS, and QHS instances.
- [ ] Every external training annotation is a valid COCO horizontal box; native HBB is preferred and all geometry fallbacks are counted.
- [ ] Every unknown ship retained for training is represented by `iscrowd=1`, reaches `gt_bboxes_ignore`, and all relevant assigners use `ignore_iof_thr=0.5`.
- [ ] Official image files are complete and disjoint across blind/fold0/fold1/fold2 at the parent-scene level.
- [ ] Exact duplicate audit has no official cross-partition match.
- [ ] Approved ShipRS overlaps are excluded and recorded.
- [ ] Three OOF models start from generic initialization, not historical official checkpoints.
- [ ] Every development image has exactly one OOF prediction record set.
- [ ] OOF threshold search passes Recall ≥85% and FDR ≤20%, with the selected threshold CSV hashed and frozen.
- [ ] Final mixed model is trained on 90% official development data plus mapped ShipRS, then fine-tuned on official development data only.
- [ ] Local blind set is evaluated exactly once after the threshold freeze marker exists.
- [ ] Hidden-test inference uses dense score `0.001` followed by the exact frozen class-wise threshold CSV.
- [ ] OOF, blind, and hidden-test matching all use horizontal-box IoU; no rotated-box evaluator is invoked.
- [ ] Reproduction report records source URLs, version, hashes, mapping, exclusions, split seed, configs, checkpoints, and threshold hash.
