#!/usr/bin/env bash
#
# Official Recall/FDR training pipeline (8:2 + 70/30 ShipRS mix).
#
# Two stages:
#   1. Train on official 80% train pool, choose best by official Recall/FDR
#      on the 20% val pool.
#   2. Mix ShipRS train (mapped to 25 classes) with official train at a
#      70/30 ratio; load stage-1 best checkpoint as the starting point and
#      choose best again on official val.
#
# After both stages, the script:
#   - Runs standard COCO bbox mAP eval on the stage-2 best via
#     tools/test.py (the official pipeline has no independent test split;
#     data.test is an alias for data.val).
#   - Exports dense val predictions, searches 25 final thresholds once,
#     and freezes them to the stage-2 best checkpoint.
#   - Reports official Recall/FDR metrics on val.
#   - Composes 10000x10000 mosaics from val.
#   - Runs batched sliding-window inference on every mosaic.
#   - Reports official metrics on mosaics and max inference time.
#
# All paths, ratios, and the inference device can be overridden via
# environment variables. The script enforces `set -euo pipefail` so missing
# inputs/prior artifacts abort immediately.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PROJECT_ROOT:=$SCRIPT_DIR}"
: "${DATA_ROOT:=$PROJECT_ROOT/data}"
: "${SHIPRS_ROOT:=$PROJECT_ROOT/external_data/ShipRSImageNet}"
: "${WORK_ROOT:=$PROJECT_ROOT/work_dirs}"
: "${OFFICIAL_CONFIG:=$PROJECT_ROOT/configs/bafnet/aircraft_bafnet_1x.py}"
: "${FINETUNE_CONFIG:=$PROJECT_ROOT/configs/bafnet/aircraft_bafnet_shiprs_mix_pretrain.py}"
: "${OFFICIAL_WORK:=$WORK_ROOT/official_stage}"
: "${FINETUNE_WORK:=$WORK_ROOT/shiprs_finetune_stage}"
: "${OFFICIAL_WEIGHT:=0.70}"
: "${SHIPRS_WEIGHT:=0.30}"
: "${BIG_IMAGE_COUNT:=2}"
: "${DEVICE:=cuda:0}"
: "${MAX_THRESHOLD_FDR:=0.19}"
: "${NAMES:=HM,LQS,QHS,MS,A1_SU-35,A2_C-130,A3_C-17,A4_C-5,A5_F-16,A6_TU-160,A7_E-3,A8_B-52,A9_P-3C,A10_B-1B,A11_E-8,A12_TU-22,A13_F-15,A14_KC-135,A15_F-22,A16_FA-18,A17_TU-95,A18_KC-10,A19_SU-34,A20_SU-24,FSC}"

# We always pass --cfg-options overrides so that DATA_ROOT and SHIPRS_ROOT
# are honored without editing the config files. The CONFIG_*.py defaults
# stay usable out of the box (./data, ./external_data/ShipRSImageNet).

# build a path-prefixed value usable as DictAction input:
#   $DATA_ROOT/annotations/instances_train.json — note that DictAction
#   does NOT support trailing-slash-aware semantics, so the values must
#   be plain strings without spaces.
cfg_str() { printf '%s' "$1"; }

OFFICIAL_TRAIN_ANN="$(cfg_str "$DATA_ROOT")/annotations/instances_train.json"
OFFICIAL_TRAIN_IMG="$(cfg_str "$DATA_ROOT")/images/train/"
OFFICIAL_VAL_ANN="$(cfg_str "$DATA_ROOT")/annotations/instances_val.json"
OFFICIAL_VAL_IMG="$(cfg_str "$DATA_ROOT")/images/val/"
SHIPRS_TRAIN_ANN="$(cfg_str "$DATA_ROOT")/external/shiprs_mapped_train.json"
SHIPRS_IMG_PREFIX="$(cfg_str "$SHIPRS_ROOT")/"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

log() { printf '[run.sh] %s\n' "$*"; }

require_dir() {
    local p="$1"
    if [ ! -d "$p" ]; then
        echo "ERROR: required directory not found: $p" >&2
        exit 1
    fi
}

require_file() {
    local p="$1"
    if [ ! -f "$p" ]; then
        echo "ERROR: required file not found: $p" >&2
        exit 1
    fi
}

# Path safety: every rm is constrained to a path under a known anchor. Never
# pass an unresolved path to rm -rf.
safe_rm() {
    local anchor="$1"
    local target="$2"
    case "$target" in
        "$anchor"/*)
            rm -rf "$target"
            ;;
        *)
            echo "ERROR: refusing to remove $target (outside $anchor)" >&2
            exit 1
            ;;
    esac
}

# ----------------------------------------------------------------------------
# Stage 0: dataset preparation
# ----------------------------------------------------------------------------

require_file "$OFFICIAL_CONFIG"
require_file "$FINETUNE_CONFIG"
require_dir "$DATA_ROOT/images/train"
require_dir "$DATA_ROOT/labels/train"

OFFICIAL_TRAIN_BACKUP="$DATA_ROOT/../data_backup"
mkdir -p "$DATA_ROOT/images" "$DATA_ROOT/labels"
if [ -d "$OFFICIAL_TRAIN_BACKUP/images/train" ] && \
        [ -d "$OFFICIAL_TRAIN_BACKUP/labels/train" ]; then
    log "Restoring official source pool from $OFFICIAL_TRAIN_BACKUP"
    # Mirror backup back to DATA_ROOT; val/ is owned by the split step.
    safe_rm "$DATA_ROOT" "$DATA_ROOT/images/train"
    safe_rm "$DATA_ROOT" "$DATA_ROOT/labels/train"
    cp -r "$OFFICIAL_TRAIN_BACKUP/images/train" "$DATA_ROOT/images/"
    cp -r "$OFFICIAL_TRAIN_BACKUP/labels/train" "$DATA_ROOT/labels/"
else
    log "Backing up official source pool to $OFFICIAL_TRAIN_BACKUP"
    mkdir -p "$OFFICIAL_TRAIN_BACKUP/images" "$OFFICIAL_TRAIN_BACKUP/labels"
    cp -r "$DATA_ROOT/images/train" "$OFFICIAL_TRAIN_BACKUP/images/"
    cp -r "$DATA_ROOT/labels/train" "$OFFICIAL_TRAIN_BACKUP/labels/"
fi

log "Splitting official data 80/20 (no test split)"
python tools/split_val.py --root "$DATA_ROOT" \
    --ratios 0.8 0.2 --seed 0 --overwrite

log "Converting YOLO splits to COCO"
python tools/convert_yolo_to_coco.py --root "$DATA_ROOT" --split train \
    --out "$DATA_ROOT/annotations/instances_train.json"
python tools/convert_yolo_to_coco.py --root "$DATA_ROOT" --split val \
    --out "$DATA_ROOT/annotations/instances_val.json"

# ----------------------------------------------------------------------------
# Stage 1: official-only training
# ----------------------------------------------------------------------------

log "Stage 1: official-only training (best by official Recall/FDR)"
mkdir -p "$OFFICIAL_WORK"
python tools/train.py "$OFFICIAL_CONFIG" "$OFFICIAL_WORK" \
    --cfg-options \
        data.train.dataset.ann_file="$OFFICIAL_TRAIN_ANN" \
        data.train.dataset.img_prefix="$OFFICIAL_TRAIN_IMG" \
        data.val.ann_file="$OFFICIAL_VAL_ANN" \
        data.val.img_prefix="$OFFICIAL_VAL_IMG" \
        data.test.ann_file="$OFFICIAL_VAL_ANN" \
        data.test.img_prefix="$OFFICIAL_VAL_IMG"

STAGE1_BEST="$OFFICIAL_WORK/best_official_recall_fdr.pth"
require_file "$STAGE1_BEST"
log "Stage 1 best: $STAGE1_BEST"

# ----------------------------------------------------------------------------
# Stage 2: mixed fine-tune (70% official / 30% ShipRS)
# ----------------------------------------------------------------------------

log "Preparing ShipRS mappings"
require_dir "$SHIPRS_ROOT"
python tools/prepare_shiprs.py --shiprs-root "$SHIPRS_ROOT" \
    --out-json "$DATA_ROOT/external/shiprs_mapped_train.json" \
    --audit-csv "$DATA_ROOT/external/shiprs_mapping_audit.csv" \
    --summary-json "$DATA_ROOT/external/shiprs_summary.json"

log "Stage 2: mixed fine-tune (load_from=$STAGE1_BEST, weights=$OFFICIAL_WEIGHT/$SHIPRS_WEIGHT)"
mkdir -p "$FINETUNE_WORK"
python tools/train.py "$FINETUNE_CONFIG" "$FINETUNE_WORK" \
    --cfg-options \
        load_from="$STAGE1_BEST" \
        data.train.source_weights="($OFFICIAL_WEIGHT,$SHIPRS_WEIGHT)" \
        data.train.datasets.0.ann_file="$OFFICIAL_TRAIN_ANN" \
        data.train.datasets.0.img_prefix="$OFFICIAL_TRAIN_IMG" \
        data.train.datasets.1.ann_file="$SHIPRS_TRAIN_ANN" \
        data.train.datasets.1.img_prefix="$SHIPRS_IMG_PREFIX" \
        data.val.ann_file="$OFFICIAL_VAL_ANN" \
        data.val.img_prefix="$OFFICIAL_VAL_IMG" \
        data.test.ann_file="$OFFICIAL_VAL_ANN" \
        data.test.img_prefix="$OFFICIAL_VAL_IMG"

STAGE2_BEST="$FINETUNE_WORK/best_official_recall_fdr.pth"
require_file "$STAGE2_BEST"
log "Stage 2 best: $STAGE2_BEST"

# ----------------------------------------------------------------------------
# Stage 3: validation reporting (ordinary + bbox + big-image)
# ----------------------------------------------------------------------------

VAL_DENSE_PRED="$FINETUNE_WORK/val_preds_dense.json"
FINAL_THRESHOLD_PREFIX="$FINETUNE_WORK/final_thresholds"
FINAL_THRESHOLD_JSON="${FINAL_THRESHOLD_PREFIX}.json"
FINAL_FILTERED_PRED="${FINAL_THRESHOLD_PREFIX}_filtered_preds.json"
VAL_METRICS_PREFIX="$FINETUNE_WORK/val_metrics"
BBOX_JSON="$FINETUNE_WORK/bbox_metrics.json"

log "Standard COCO bbox mAP eval on official val (data.test alias)"
mkdir -p "$FINETUNE_WORK/bbox_eval"
python tools/test.py "$OFFICIAL_CONFIG" "$STAGE2_BEST" \
    "$FINETUNE_WORK/bbox_eval" \
    --eval bbox \
    --cfg-options \
        data.test.ann_file="$OFFICIAL_VAL_ANN" \
        data.test.img_prefix="$OFFICIAL_VAL_IMG"
# tools/test.py writes <work_dir>/eval_<timestamp>.json — copy a stable alias.
latest_eval=$(ls -t "$FINETUNE_WORK"/bbox_eval/eval_*.json 2>/dev/null | head -n 1 || true)
if [ -n "${latest_eval:-}" ]; then
    cp "$latest_eval" "$BBOX_JSON"
fi

log "Exporting dense val predictions at the candidate score floor"
python tools/eval_val_to_json.py \
    --config "$OFFICIAL_CONFIG" \
    --checkpoint "$STAGE2_BEST" \
    --img-dir "$DATA_ROOT/images/val" \
    --gt "$DATA_ROOT/annotations/instances_val.json" \
    --out "$VAL_DENSE_PRED" \
    --device "$DEVICE"

log "Searching and freezing 25 class thresholds on official val"
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

log "Reporting official Recall/FDR metrics on threshold-filtered val"
python tools/eval_recall_fdr.py \
    --pred "$FINAL_FILTERED_PRED" \
    --gt "$OFFICIAL_VAL_ANN" \
    --classes 25 --names "$NAMES" \
    --out-prefix "$VAL_METRICS_PREFIX"

BIG_VAL_ROOT="$FINETUNE_WORK/big_val"
BIG_VAL_IMG_DIR="$BIG_VAL_ROOT/images"
BIG_VAL_GT="$BIG_VAL_ROOT/instances_big_val.json"
BIG_VAL_MAP="$BIG_VAL_ROOT/source_map.json"
BIG_VAL_PRED="$BIG_VAL_ROOT/predictions.json"
BIG_VAL_TIMING="$BIG_VAL_ROOT/timing.json"
BIG_VAL_METRICS_PREFIX="$BIG_VAL_ROOT/metrics"

log "Composing $BIG_IMAGE_COUNT mosaic(s) at 10000x10000"
python tools/compose_big_val.py \
    --gt "$DATA_ROOT/annotations/instances_val.json" \
    --img-dir "$DATA_ROOT/images/val" \
    --out-dir "$BIG_VAL_ROOT" \
    --num-canvases "$BIG_IMAGE_COUNT" \
    --seed 0 \
    --overwrite

log "Running sliding-window batch inference on mosaics"
python tools/infer_big_image.py \
    --config "$OFFICIAL_CONFIG" \
    --checkpoint "$STAGE2_BEST" \
    --thresholds "$FINAL_THRESHOLD_JSON" \
    --img-dir "$BIG_VAL_IMG_DIR" \
    --gt "$BIG_VAL_GT" \
    --out "$BIG_VAL_PRED" \
    --timing-out "$BIG_VAL_TIMING" \
    --device "$DEVICE"

log "Reporting official metrics on mosaics"
python tools/eval_recall_fdr.py \
    --pred "$BIG_VAL_PRED" \
    --gt "$BIG_VAL_GT" \
    --classes 25 --names "$NAMES" \
    --out-prefix "$BIG_VAL_METRICS_PREFIX"

log "Done."
log "Artifacts:"
log "  Stage 1 best:        $STAGE1_BEST (+ $OFFICIAL_WORK/best_official_recall_fdr.json)"
log "  Stage 2 best:        $STAGE2_BEST (+ $FINETUNE_WORK/best_official_recall_fdr.json)"
log "  COCO bbox (stage 2): $BBOX_JSON (latest eval JSON copied from $FINETUNE_WORK/bbox_eval/)"
log "  Dense val predictions:$VAL_DENSE_PRED"
log "  Frozen thresholds:   $FINAL_THRESHOLD_JSON"
log "  Threshold audit:     ${FINAL_THRESHOLD_PREFIX}_{selected,global_curve,class_curves}.csv"
log "  Filtered val preds:  $FINAL_FILTERED_PRED"
log "  Val official metrics:${VAL_METRICS_PREFIX}.{json,csv}"
log "  Mosaic GT:           $BIG_VAL_GT"
log "  Mosaic source map:   $BIG_VAL_MAP"
log "  Mosaic predictions:  $BIG_VAL_PRED"
log "  Mosaic timing:       $BIG_VAL_TIMING (max_inference_seconds)"
log "  Mosaic official:     ${BIG_VAL_METRICS_PREFIX}.{json,csv}"
