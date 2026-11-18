#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

data_root="${DATA_ROOT:-/home/zk/Gaussian/data}"
test_root="$data_root/PU1K/PU1K/test/input_2048"
gt="$test_root/gt_8192"
e0=model/E0_PUCRN_seed42
e3=model/E3_PUGCSNet_seed42
e2=model/E2_PUGCSNet_seed42
out=model/diagnosis_seed42
gpu="${GPU:-0}"

pair() {
    python diagnose.py paired --reference "$1/test/PU1K/input_2048" \
        --candidate "$2/test/PU1K/input_2048" --gt "$gt" \
        --pugan-meshes "$data_root/PUGAN/PU-GAN/test" --expected-counts 127 27 --out "$out/$3"
}

patches() {
    python diagnose.py patches --ckpt "$1/best.pth" --val-h5 "$data_root/prepared/val.h5" \
        --height-scale "$2" --device "cuda:$gpu" --out "$out/$3"
}

train_run() {
    local run=$1 stage=$2 seed=$3 epochs=$4
    shift 4
    python main.py --phase train --name "${run##*/}" --log_dir model --device cuda --gpu "$gpu" \
        --stage2 "$stage" --sampling deterministic --lambda_surf 0 --lambda_normal 0 \
        --h5_data "$data_root/prepared/train.h5" --val_h5 "$data_root/prepared/val.h5" \
        --num_point 256 --up_ratio 4 --batch_size 16 --max_epoch "$epochs" --lr_init 0.001 \
        --decay_iter 50000 --save_step 4 --random_seed "$seed" "$@"
}

test_run() {
    # Use the original test seed and the same patch/FPS protocol for every training seed.
    python main.py --phase test --ckpt "$1/best.pth" --device cuda --gpu "$gpu" \
        --test_data "$test_root/input_2048" --gt_path "$gt" --num_point 256 --num_shape_point 2048 \
        --up_ratio 4 --patch_num_ratio 3 --random_seed 42 --result_dir "$1/test/PU1K"
}

overnight_run() {
    local run=$1 stage=$2 seed=43 target
    echo "[$(date)] $run"
    target=$(python - "$run" "$deadline" <<'PY'
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

run, deadline = Path(sys.argv[1]), int(sys.argv[2])
current = max((int(p.stem.split('_')[1]) for p in run.glob('model_[0-9]*.pth')), default=0)
history = run.with_name(run.name.replace('seed43', 'seed42'))
for folder in (history, Path('model/E2_PUGCSNet_seed42'), run):
    log = folder / 'train.log'
    if not log.is_file():
        continue
    samples = []
    previous = None
    with log.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            match = re.match(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ Validation epoch (\d+):', line)
            if not match:
                continue
            stamp = datetime.fromisoformat(match[1])
            epoch = int(match[2])
            if previous and epoch - previous[0] == 4:
                seconds = (stamp - previous[1]).total_seconds()
                if seconds > 0:
                    samples.append(seconds / 4)
            previous = epoch, stamp
    if samples:
        sec_per_epoch = statistics.median(samples)
        break
else:
    raise SystemExit('Cannot estimate overnight runtime: no timed validation intervals in E0/E2/E3 train.log.')

# Budget in complete four-epoch validation blocks, with a 15% slowdown allowance
# and 30 minutes for startup, checkpoint I/O and whole-shape testing.
available = max(0, deadline - time.time() - 1800)
blocks = int(available / (4 * 1.15 * sec_per_epoch))
target = min(100, current + 4 * blocks)
print(f'{log}: {sec_per_epoch / 60:.1f} min/epoch, saved epoch {current}, budget through epoch {target}', file=sys.stderr)
print(target if target > current or current == 100 else 0)
PY
    )
    if (( target == 0 )) || [[ "$stage" == baseline && "$target" != 100 ]]; then
        echo "Stopping the queue here to stay within the overnight budget."
        return 0
    fi
    if [[ ! -s "$run/model_100.pth" ]]; then
        if [[ -s "$run/latest.pth" ]]; then
            train_run "$run" "$stage" "$seed" "$target" --resume "$run/latest.pth"
        elif [[ -e "$run/config.json" || -e "$run/metrics.jsonl" ]]; then
            echo "Found an earlier run without a saved checkpoint in $run. Preserve it in a separate directory before restarting this run." >&2
            return 1
        else
            train_run "$run" "$stage" "$seed" "$target"
        fi
    fi
    if [[ ! -s "$run/model_100.pth" ]]; then
        echo "Saved training progress at epoch $target. The 100-epoch test is pending."
        return 0
    fi
    if [[ ! -s "$run/test/PU1K/input_2048/metrics.json" || ! -s "$run/test/PU1K/input_2048/evaluation.csv" ]]; then
        if (( $(date +%s) >= deadline )); then
            echo "The overnight cutoff has passed; keeping the checkpoint for testing later."
            return 0
        fi
        test_run "$run"
    fi
}

case "${1:-}" in
    paired)
        pair "$e0" "$e3" E3_vs_E0
        ;;
    validation)
        patches "$e0" 1 E0_val
        for h in 1 0.5 0; do
            patches "$e3" "$h" "E3_val_h$h"
        done
        ;;
    fps)
        for tag in E0 E3; do
            if [[ "$tag" == E0 ]]; then run="$e0"; else run="$e3"; fi
            python diagnose.py fps --ckpt "$run/best.pth" --device "cuda:$gpu" \
                --inputs "$test_root/input_2048" --gt "$gt" --paired-csv "$out/E3_vs_E0/paired.csv" \
                --top 5 --out "$out/${tag}_fps"
        done
        ;;
    train|resume)
        extra=()
        if [[ "$1" == resume ]]; then
            extra=(--resume "$e2/latest.pth")
        elif [[ -e "$e2/config.json" || -e "$e2/metrics.jsonl" ]]; then
            echo "E2 already has a training run in $e2. To continue it, run: bash run_e2.sh resume" >&2
            exit 1
        fi
        train_run "$e2" tangent 42 100 "${extra[@]}"
        ;;
    test)
        patches "$e2" 1 E2_val
        test_run "$e2"
        pair "$e0" "$e2" E2_vs_E0
        pair "$e3" "$e2" E2_vs_E3
        ;;
    overnight|replicates)
        if [[ -n "${2:-}" ]]; then
            deadline=$(TZ=Asia/Shanghai date -d "$2" +%s)
        else
            deadline=$(TZ=Asia/Shanghai date -d 'today 09:00' +%s)
            if (( deadline <= $(date +%s) )); then
                deadline=$(TZ=Asia/Shanghai date -d 'tomorrow 09:00' +%s)
            fi
        fi
        out=model/diagnosis_seed43
        ref=model/E0_PUCRN_seed43
        quadratic=model/E3_PUGCSNet_seed43
        overnight_run "$quadratic" quadratic
        if [[ -s "$quadratic/test/PU1K/input_2048/metrics.json" ]]; then
            overnight_run "$ref" baseline
            if [[ -s "$ref/test/PU1K/input_2048/metrics.json" ]]; then
                pair "$ref" "$quadratic" E3_vs_E0
            fi
        fi
        ;;
    *)
        echo 'Usage: bash run_e2.sh {paired|validation|fps|train|resume|test|overnight [YYYY-MM-DD HH:MM]}' >&2
        exit 2
        ;;
esac
