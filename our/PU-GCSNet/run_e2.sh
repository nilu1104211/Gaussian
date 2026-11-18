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
        python main.py --phase train --name "${e2##*/}" --log_dir model --device cuda --gpu "$gpu" \
            --stage2 tangent --sampling deterministic --lambda_surf 0 --lambda_normal 0 \
            --h5_data "$data_root/prepared/train.h5" --val_h5 "$data_root/prepared/val.h5" \
            --num_point 256 --up_ratio 4 --batch_size 16 --max_epoch 100 --lr_init 0.001 \
            --decay_iter 50000 --save_step 4 --random_seed 42 "${extra[@]}"
        ;;
    test)
        patches "$e2" 1 E2_val
        python main.py --phase test --ckpt "$e2/best.pth" --device cuda --gpu "$gpu" \
            --test_data "$test_root/input_2048" --gt_path "$gt" --num_point 256 --num_shape_point 2048 \
            --up_ratio 4 --patch_num_ratio 3 --random_seed 42 --result_dir "$e2/test/PU1K"
        pair "$e0" "$e2" E2_vs_E0
        pair "$e3" "$e2" E2_vs_E3
        ;;
    *)
        echo 'Usage: bash run_e2.sh {paired|validation|fps|train|resume|test}' >&2
        exit 2
        ;;
esac
