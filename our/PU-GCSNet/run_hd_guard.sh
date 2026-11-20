#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

action="${1:-all}"
case "$action" in
    ablations)
        for experiment in tail local no_growth; do
            echo "[$(date)] Starting HD-Guard ablation: $experiment"
            VARIANT="$experiment" bash ./run_hd_guard.sh all
        done
        echo "[$(date)] HD-Guard ablations complete."
        exit 0
        ;;
    all|train|resume|test) ;;
    *) echo 'Usage: bash run_hd_guard.sh [all|train|resume|test|ablations]' >&2; exit 2 ;;
esac

data_root="${DATA_ROOT:-/home/zk/Gaussian/data}"
gpu="${GPU:-0}"
seed="${SEED:-42}"
epochs="${EPOCHS:-100}"
variant="${VARIANT:-full}"
case "$variant" in
    full)      name="E6_HDGuard_seed${seed}"; refinement=local; tail=0.05; growth=0.05 ;;
    tail)      name="E6_HDGuard_tail_seed${seed}"; refinement=legacy; tail=0.05; growth=0.05 ;;
    local)     name="E6_HDGuard_local_seed${seed}"; refinement=local; tail=0; growth=0 ;;
    no_growth) name="E6_HDGuard_no_growth_seed${seed}"; refinement=local; tail=0.05; growth=0 ;;
    *) echo 'VARIANT must be full, tail, local, or no_growth' >&2; exit 2 ;;
esac
run="model/$name"
test_root="$data_root/PU1K/PU1K/test/input_2048"
mkdir -p "$run"
# Prevent two invocations from writing the same experiment simultaneously.
exec 9>"$run/.run.lock"
flock -n 9 || { echo "Another process is using $run" >&2; exit 1; }

train_run() {
    local resume_args=()
    if [[ "$1" == resume ]]; then
        [[ -s "$run/latest.pth" ]] || { echo "Missing $run/latest.pth" >&2; return 1; }
        resume_args=(--resume "$run/latest.pth")
    elif [[ -e "$run/config.json" || -e "$run/metrics.jsonl" || -e "$run/latest.pth" ]]; then
        echo "Existing run in $run. Use resume, or select a new experiment name/seed." >&2
        return 1
    fi
    [[ -f "$data_root/prepared/train.h5" && -f "$data_root/prepared/val.h5" ]] || {
        echo "Missing prepared train.h5/val.h5 under $data_root" >&2; return 1;
    }
    python -u main.py --phase train --name "$name" --log_dir model --device cuda --gpu "$gpu" \
        --h5_data "$data_root/prepared/train.h5" --val_h5 "$data_root/prepared/val.h5" \
        --stage2 quadratic --sampling deterministic --refinement "$refinement" --refine_radius 0.5 \
        --lambda_surf 0.1 --surf_eta 0 --lambda_normal 0 \
        --lambda_tail "$tail" --lambda_growth "$growth" --tail_fraction 0.02 --tail_warmup_epochs 10 \
        --num_point 256 --batch_size 16 --max_epoch "$epochs" --lr_init 0.001 --decay_iter 50000 \
        --save_step 4 --random_seed "$seed" "${resume_args[@]}"
}

test_run() {
    local checkpoint=$1 destination=$2
    [[ -s "$run/$checkpoint" ]] || { echo "Missing $run/$checkpoint" >&2; return 1; }
    python -u main.py --phase test --ckpt "$run/$checkpoint" --device cuda --gpu "$gpu" \
        --test_data "$test_root/input_2048" --gt_path "$test_root/gt_8192" \
        --num_point 256 --num_shape_point 2048 --patch_num_ratio 3 --random_seed 42 \
        --save_stages --result_dir "$run/$destination/PU1K"
    local label ref protocol
    local references=(E0_PUCRN E4_PUGCSNet)
    if [[ "$variant" != full && "$destination" == test ]]; then
        references+=(E6_HDGuard)
    fi
    for label in "${references[@]}"; do
        ref="model/${label}_seed${seed}/test/PU1K/input_2048"
        if [[ ! -d "$ref" ]]; then
            echo "Skipping paired report: reference $ref is not available."
            continue
        fi
        for protocol in official shared; do
            local coordinate_args=()
            if [[ "$protocol" == shared ]]; then coordinate_args=(--shared-inputs "$test_root/input_2048"); fi
            python -u diagnose.py paired --reference "$ref" --candidate "$run/$destination/PU1K/input_2048" \
                --gt "$test_root/gt_8192" --pugan-meshes "$data_root/PUGAN/PU-GAN/test" \
                --expected-counts 127 27 "${coordinate_args[@]}" \
                --out "$run/diagnosis/${destination}_vs_${label}_${protocol}"
        done
    done
}

case "$action" in
    train) train_run train ;;
    resume) train_run resume ;;
    all)
        if [[ -s "$run/latest.pth" ]]; then train_run resume; else train_run train; fi
        test_run best.pth test
        test_run best_hd.pth test_hd
        echo "[$(date)] HD-Guard complete: $run (training and both checkpoint evaluations)."
        ;;
    test)
        test_run best.pth test
        test_run best_hd.pth test_hd
        echo "[$(date)] HD-Guard evaluation complete: $run"
        ;;
esac
