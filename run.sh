#!/usr/bin/env bash

set -e

echo "备份数据集"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo "Backing up data/images/train and data/labels/train..."

if [ ! -d "data/images/train" ]; then
    echo "ERROR: data/images/train not found, cannot backup." >&2
    exit 1
fi

if [ -d "data_backup/images/train" ] || [ -d "data_backup/labels/train" ]; then
    echo "WARNING: data_backup/ already exists. Will be overwritten."
    rm -rf data_backup
fi

mkdir -p data_backup/images data_backup/labels
cp -r data/images/train data_backup/images/
cp -r data/labels/train data_backup/labels/

echo "分割数据集"

python tools/split_val.py --root data --ratios 0.6 0.2 0.2 --seed 0 --overwrite

echo "生成coco标注"
python tools/convert_yolo_to_coco.py --root data --split train --out data/annotations/instances_train.json
python tools/convert_yolo_to_coco.py --root data --split val --out data/annotations/instances_val.json
python tools/convert_yolo_to_coco.py --root data --split test --out data/annotations/instances_test.json


echo "仅ARFC"
python tools/train.py configs/bafnet/aircraft_bafnet_1x.py work_dirs/new_data_arfc_only_v1
echo "测试"
python tools/test.py configs/bafnet/aircraft_bafnet_1x.py work_dirs/new_data_arfc_only_v1/best_bbox_mAP.pth work_dirs/new_data_arfc_only_v1/test_results --eval bbox --out work_dirs/new_data_arfc_only_v1/test_results/results.pkl
echo "生成指标"
NAMES="HM,LQS,QHS,MS,A1_SU-35,A2_C-130,A3_C-17,A4_C-5,A5_F-16,A6_TU-160,A7_E-3,A8_B-52,A9_P-3C,A10_B-1B,A11_E-8,A12_TU-22,A13_F-15,A14_KC-135,A15_F-22,A16_FA-18,A17_TU-95,A18_KC-10,A19_SU-34,A20_SU-24,FSC"
python tools/eval_val_to_json.py --config configs/bafnet/aircraft_bafnet_1x.py --checkpoint work_dirs/new_data_arfc_only_v1/best_bbox_mAP.pth --img-dir data/images/test --gt data/annotations/instances_test.json --out work_dirs/new_data_arfc_only_v1/test_preds.json

python tools/eval_recall_fdr.py --pred work_dirs/new_data_arfc_only_v1/test_preds.json --gt data/annotations/instances_test.json --classes 25 --names $NAMES --out-prefix work_dirs/new_data_arfc_only_v1/test_metrics