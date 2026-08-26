"""Executable contract test for the adaptive-threshold shell pipeline."""

import os
import shutil
import subprocess
from pathlib import Path


FAKE_PYTHON = r'''#!/usr/bin/env bash
set -euo pipefail
tool="$1"
shift
printf '%s' "$tool" >> "$CALL_LOG"
printf ' %s' "$@" >> "$CALL_LOG"
printf '\n' >> "$CALL_LOG"

value_after() {
    local wanted="$1"
    shift
    while [ "$#" -gt 0 ]; do
        if [ "$1" = "$wanted" ]; then
            printf '%s' "$2"
            return
        fi
        shift
    done
    return 1
}

case "$tool" in
    tools/split_val.py)
        root=$(value_after --root "$@")
        mkdir -p "$root/images/val" "$root/labels/val"
        : > "$root/images/val/val.jpg"
        : > "$root/labels/val/val.txt"
        ;;
    tools/convert_yolo_to_coco.py)
        out=$(value_after --out "$@")
        mkdir -p "$(dirname "$out")"
        printf '{"images":[],"annotations":[],"categories":[]}' > "$out"
        ;;
    tools/train.py)
        work_dir="$2"
        mkdir -p "$work_dir"
        : > "$work_dir/best_official_recall_fdr.pth"
        printf '{}' > "$work_dir/best_official_recall_fdr.json"
        ;;
    tools/prepare_shiprs.py)
        for option in --out-json --audit-csv --summary-json; do
            out=$(value_after "$option" "$@")
            mkdir -p "$(dirname "$out")"
            : > "$out"
        done
        ;;
    tools/test.py)
        work_dir="$3"
        mkdir -p "$work_dir"
        printf '{}' > "$work_dir/eval_fake.json"
        ;;
    tools/eval_val_to_json.py)
        out=$(value_after --out "$@")
        mkdir -p "$(dirname "$out")"
        printf '{"images":[],"annotations":[],"categories":[]}' > "$out"
        ;;
    tools/search_recall_fdr_thresholds.py)
        prefix=$(value_after --out-prefix "$@")
        mkdir -p "$(dirname "$prefix")"
        printf '{}' > "${prefix}.json"
        printf '{"images":[],"annotations":[],"categories":[]}' \
            > "${prefix}_filtered_preds.json"
        : > "${prefix}_selected.csv"
        : > "${prefix}_global_curve.csv"
        : > "${prefix}_class_curves.csv"
        ;;
    tools/eval_recall_fdr.py)
        prefix=$(value_after --out-prefix "$@")
        mkdir -p "$(dirname "$prefix")"
        printf '{}' > "${prefix}.json"
        : > "${prefix}.csv"
        ;;
    tools/compose_big_val.py)
        out_dir=$(value_after --out-dir "$@")
        mkdir -p "$out_dir/images"
        : > "$out_dir/images/big.jpg"
        printf '{"images":[],"annotations":[]}' \
            > "$out_dir/instances_big_val.json"
        printf '{}' > "$out_dir/source_map.json"
        ;;
    tools/infer_big_image.py)
        out=$(value_after --out "$@")
        timing=$(value_after --timing-out "$@")
        printf '{"images":[],"annotations":[]}' > "$out"
        printf '{"max_inference_seconds":0.1}' > "$timing"
        ;;
esac
'''


def test_run_sh_calibrates_once_then_reuses_frozen_thresholds(tmp_path):
    project = tmp_path / 'project'
    project.mkdir()
    shutil.copy(Path(__file__).parents[1] / 'run.sh', project / 'run.sh')
    for relative in (
        'configs/bafnet/aircraft_bafnet_1x.py',
        'configs/bafnet/aircraft_bafnet_shiprs_mix_pretrain.py',
    ):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fake config\n', encoding='utf-8')
    for relative in (
        'data/images/train/train.jpg',
        'data/labels/train/train.txt',
        'external_data/ShipRSImageNet/.keep',
    ):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('', encoding='utf-8')
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    fake_python = fake_bin / 'python'
    fake_python.write_text(FAKE_PYTHON, encoding='utf-8')
    fake_python.chmod(0o755)
    call_log = tmp_path / 'calls.log'
    env = dict(os.environ)
    env.update({
        'PATH': str(fake_bin) + os.pathsep + env['PATH'],
        'CALL_LOG': str(call_log),
        'PROJECT_ROOT': str(project),
        'DATA_ROOT': str(project / 'data'),
        'SHIPRS_ROOT': str(project / 'external_data/ShipRSImageNet'),
        'WORK_ROOT': str(project / 'work_dirs'),
        'MAX_THRESHOLD_FDR': '0.19',
    })

    subprocess.run(
        ['bash', str(project / 'run.sh')], cwd=project, env=env,
        check=True, capture_output=True, text=True)

    calls = call_log.read_text(encoding='utf-8')
    finetune = project / 'work_dirs/shiprs_finetune_stage'
    assert ('tools/eval_val_to_json.py' in calls
            and f'--out {finetune / "val_preds_dense.json"}' in calls)
    assert calls.count('tools/search_recall_fdr_thresholds.py') == 1
    assert f'--checkpoint {finetune / "best_official_recall_fdr.pth"}' in calls
    assert '--max-official-fdr 0.19' in calls
    assert f'--out-prefix {finetune / "final_thresholds"}' in calls
    assert (f'--pred {finetune / "final_thresholds_filtered_preds.json"}'
            in calls)
    assert f'--thresholds {finetune / "final_thresholds.json"}' in calls
    assert 'ret/' not in calls
