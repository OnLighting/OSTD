# BAFNet — Official Recall/FDR Training Pipeline

This repository trains a 25-class optical-satellite aircraft/vehicle
detector (BAFNet on mmdetection 2.x) and reports metrics in the format
required by the official competition. The pipeline uses an **8:2
train/val split**, **two-stage training** (official → mixed official +
ShipRS fine-tune), adaptive validation-time score calibration, and
**max-inference-time** reporting on simulated 10 000 × 10 000 mosaics.

## Quick start

```bash
export DATA_ROOT=/path/to/official/data
export SHIPRS_ROOT=/path/to/ShipRSImageNet
bash run.sh
```

`run.sh` is the canonical end-to-end pipeline. It expects:

| Variable        | Default                              | Purpose                                      |
| --------------- | ------------------------------------ | -------------------------------------------- |
| `DATA_ROOT`     | `$PROJECT_ROOT/data`                 | YOLO images/labels and COCO outputs          |
| `SHIPRS_ROOT`   | `$PROJECT_ROOT/external_data/ShipRSImageNet` | ShipRS source images (for stage 2)   |
| `WORK_ROOT`     | `$PROJECT_ROOT/work_dirs`            | Where to write stage-1 / stage-2 runs        |
| `OFFICIAL_CONFIG`  | `configs/bafnet/aircraft_bafnet_1x.py` | Stage-1 config                           |
| `FINETUNE_CONFIG`  | `configs/bafnet/aircraft_bafnet_shiprs_mix_pretrain.py` | Stage-2 config |
| `OFFICIAL_WEIGHT` | `0.70`                              | Stage-2 official-train sampling weight       |
| `SHIPRS_WEIGHT`   | `0.30`                              | Stage-2 ShipRS sampling weight               |
| `BIG_IMAGE_COUNT` | `2`                                 | How many 10 000 × 10 000 mosaics to compose  |
| `DEVICE`        | `cuda:0`                             | Inference device                             |
| `MAX_THRESHOLD_FDR` | `0.19`                           | Final threshold-search official FDR budget   |

## Pipeline stages

### Stage 1 — official-only training

* Splits official data into 80% train / 20% val (`tools/split_val.py`).
  No independent test split is created.
* Converts YOLO labels to COCO JSON (`tools/convert_yolo_to_coco.py`).
* Trains `aircraft_bafnet_1x.py` from scratch.
* Chooses the best checkpoint on the 20% val pool using the official
  Recall/FDR protocol (`OfficialBestSaverHook`). Each validation epoch
  searches one exact score threshold for ship, aircraft, and vehicle.
* Best is saved to `work_dirs/official_stage/best_official_recall_fdr.pth`.

### Stage 2 — mixed official + ShipRS fine-tune

* `tools/prepare_shiprs.py` remaps ShipRS labels to the 25-class schema.
* Loads stage-1 best as `load_from`.
* Trains `aircraft_bafnet_shiprs_mix_pretrain.py` with
  `source_weights=(0.70, 0.30)`.
* Best is again selected on official val (ShipRS val is audit-only).

### Stage 3 — validation reporting

* **bbox**: standard COCO mAP on official val.
* **dense validation predictions** (`tools/eval_val_to_json.py`) at the
  model candidate floor (`0.0`).
* **25-class threshold search** (`tools/search_recall_fdr_thresholds.py`)
  under official FDR ≤ 0.19. The resulting JSON is bound to the stage-2
  checkpoint SHA-256.
* **official metrics** (`tools/eval_recall_fdr.py`) — per-class,
  superclass, official, and merged counts on the filtered predictions.
* **10 000 × 10 000 mosaics** (`tools/compose_big_val.py`) composed
  from val images without resizing or splitting source images.
* **Sliding-window batch inference** (`tools/infer_big_image.py`)
  with per-image timing.
* **max inference time** is reported in
  `work_dirs/shiprs_finetune_stage/big_val/timing.json`.

## Official metric definitions

* Per-class IoU: 0.50 for ship/aircraft, 0.35 for FSC (vehicle).
* Training-time validation searches one exact threshold per superclass,
  with each superclass mean FDR constrained to 0.19.
* After stage 2, validation predictions are searched once for 25 final
  thresholds. `final_thresholds.json` is then reused unchanged for ordinary
  validation, simulated 10k images, and the independent official test set.
* Historical `ret/*.csv` files are audit-only and are never runtime inputs.
* `official_recall = mean(ship_recall, aircraft_recall, vehicle_recall)`
* `official_fdr    = mean(ship_fdr,    aircraft_fdr,    vehicle_fdr)`
* `merged_*` aggregates TP/FP/FN across all 25 classes before computing
  the rate (kept for diagnostic comparison only — best uses official).
* Best checkpoint logic (`OfficialBestSaverHook`):
  1. Both-pass (`Recall ≥ 0.85`, `FDR ≤ 0.20`) beats either-fail.
  2. Once a passing best exists, only another passing candidate can win.
  3. Between passing candidates with `|ΔRecall| > 0.005`, higher Recall wins.
  4. Within tolerance, lower FDR wins.
  5. Ties (or floating-point equality) keep the earlier checkpoint.

## Conventions

* Hardware: designed for a single RTX 3090. Timing comparability across
  runs assumes the same GPU + driver stack.
* Best checkpoint filenames are **`best_official_recall_fdr.pth`** plus
  matching `best_official_recall_fdr.json`. COCO mAP remains a reported
  diagnostic metric but does not save or select a separate best model.
* Frozen thresholds are stored at
  `work_dirs/shiprs_finetune_stage/final_thresholds.json`. Loading them
  against a different checkpoint fails the SHA-256 identity check.
* The official independent test set never participates in threshold search;
  it only consumes the frozen stage-2 artifact.
* `data.test` in the training dataset config is an alias pointing at
  `data.val` (so `tools/test.py` keeps working). The competition's
  separately supplied official test set remains external to this split and
  must only consume the frozen thresholds; it is never used for calibration.

To run the separately supplied official test images, use their COCO image
manifest (its `annotations` may be empty) and the same stage-2 checkpoint and
frozen artifact:

```bash
python tools/eval_val_to_json.py \
  --config configs/bafnet/aircraft_bafnet_1x.py \
  --checkpoint work_dirs/shiprs_finetune_stage/best_official_recall_fdr.pth \
  --thresholds work_dirs/shiprs_finetune_stage/final_thresholds.json \
  --img-dir /path/to/official_test/images \
  --gt /path/to/official_test/image_manifest.json \
  --out work_dirs/shiprs_finetune_stage/official_test_predictions.json
```

## Where to read more

* Design spec: `docs/superpowers/specs/2026-08-26-official-recall-training-pipeline-design.md`
* Implementation plan: `docs/superpowers/plans/2026-08-26-adaptive-score-threshold-workflow.md`
* Shared metrics module: `mmdet/core/evaluation/official_metrics.py`
* Checkpoint hooks: `mmdet/core/evaluation/eval_hooks.py`
  (`OfficialBestSaverHook`, `OfficialEarlyStoppingHook`)
