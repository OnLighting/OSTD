#!/usr/bin/env bash
# 一次性脚本：跑 ShipRS 微调完整流程 (train + val convert + test dry-run +
# 微调 + 最终 eval)。
# 在 ~/autodl-tmp/model_v4 下执行: bash new_run.sh
#
# 远程机环境约定 (与 main 分支改造一致):
#   - ShipRSImageNet 数据: external_data/ShipRSImageNet/{annotations,images, SOURCE_MANIFEST.txt}
#   - ShipRS mapped COCO 输出: data/external/shiprs_mapped_{train,val}.json
#   - 已有 25 类 best checkpoint: work_dirs/arfc_only_v1/best_bbox_mAP.pth
#   - 微调输出 work_dir: work_dirs/shiprs_mix_v1/

set -euo pipefail
export PYTHONPATH=.

cd ~/autodl-tmp/model_v4

# === 步骤 A1: 转换 train ===
python tools/prepare_shiprs.py \
    --shiprs-root external_data/ShipRSImageNet \
    --target-levels 3 \
    --out-json data/external/shiprs_mapped_train.json \
    --audit-csv data/external/shiprs_mapping_audit_train.csv \
    --summary-json data/external/shiprs_summary_train.json

# === 步骤 A2: 转换 val ===
python tools/prepare_shiprs.py \
    --shiprs-root external_data/ShipRSImageNet \
    --target-levels 3 \
    --out-json data/external/shiprs_mapped_val.json \
    --audit-csv data/external/shiprs_mapping_audit_val.csv \
    --summary-json data/external/shiprs_summary_val.json

# === 步骤 B: dry-run 验证 checkpoint 加载 ===
mkdir -p work_dirs/shiprs_mix_v1_dryrun
python tools/test.py \
    configs/bafnet/aircraft_bafnet_1x.py \
    work_dirs/arfc_only_v1/best_bbox_mAP.pth \
    work_dirs/shiprs_mix_v1_dryrun \
    --eval bbox \
    --out work_dirs/shiprs_mix_v1_dryrun/results.pkl

# === 步骤 C: 启动微调 (24 epoch, lr=0.001, step=[16, 22], strict load_from) ===
mkdir -p work_dirs/shiprs_mix_v1
python tools/train.py \
    configs/bafnet/aircraft_bafnet_shiprs_mix_pretrain.py \
    work_dirs/shiprs_mix_v1

# === 步骤 D: 最终评估 ===
python tools/test.py \
    configs/bafnet/aircraft_bafnet_shiprs_mix_pretrain.py \
    work_dirs/shiprs_mix_v1/best_bbox_mAP.pth \
    work_dirs/shiprs_mix_v1 \
    --eval bbox \
    --out work_dirs/shiprs_mix_v1/test_results.pkl