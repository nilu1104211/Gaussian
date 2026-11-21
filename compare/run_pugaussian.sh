#!/usr/bin/env bash
set -euo pipefail

compare_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$compare_dir/PU-Gaussian-main"
runtime_dir="$compare_dir/.pugaussian-runtime"
toolchain="$runtime_dir/toolchain"
packages="$runtime_dir/python"
test_data="${PU1K_TEST:-$(dirname -- "$compare_dir")/data/PU1K/PU1K/test/input_2048}"

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

mode="${1:---help}"
case "$mode" in
    setup|check|smoke|test) ;;
    -h|--help)
        printf 'Usage: bash run_pugaussian.sh {setup|check|smoke|test}\n'
        exit 0
        ;;
    *) fail "Unknown mode: $mode. Use setup, check, smoke, or test." ;;
esac

active_prefix="${CONDA_PREFIX:-}"
[[ "${active_prefix##*/}" == scdata-v3 ]] || fail 'Run conda activate scdata-v3 first.'
python="$active_prefix/bin/python"
[[ -x "$python" ]] || fail "Cannot run $python"
[[ "$test_data" == /* ]] || test_data="$PWD/$test_data"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
cd -- "$project_dir"

check_packages() {
    PYTHONPATH="$1" "$python" - "$1" <<'PY'
from pathlib import Path
import sys
import torch
import open3d
import pointops_cuda
import chamfer_3D

base = Path(sys.prefix).resolve()
packages = Path(sys.argv[1]).resolve()
assert Path(torch.__file__).resolve().is_relative_to(base), torch.__file__
assert torch.version.cuda == '12.1' and torch.cuda.is_available(), torch.__version__
for module in (open3d, pointops_cuda, chamfer_3D):
    assert Path(module.__file__).resolve().is_relative_to(packages), module.__file__
print(f'PyTorch {torch.__version__} at {torch.__file__}')
print(f'Open3D {open3d.__version__} and both CUDA extensions loaded from {packages}')
PY
}

if [[ "$mode" == setup ]]; then
    PYTHONPATH= "$python" -c 'import torch; assert torch.version.cuda == "12.1" and torch.cuda.is_available(), torch.__version__'
    if [[ -d "$packages" ]] && check_packages "$packages"; then
        exit 0
    fi

    mkdir -p -- "$runtime_dir"
    if [[ ! -x "$toolchain/bin/nvcc" || ! -x "$toolchain/bin/x86_64-conda-linux-gnu-g++" || ! -x "$toolchain/bin/ninja" ]]; then
        "${CONDA_EXE:-conda}" create --yes --no-default-packages --prefix "$toolchain" \
            -c nvidia/label/cuda-12.1.1 -c nvidia/label/cuda-12.1.0 -c conda-forge \
            'cuda-toolkit=12.1.1' 'gcc_linux-64=11' 'gxx_linux-64=11' ninja
    fi

    export CUDA_HOME="$toolchain" PATH="$toolchain/bin:$PATH"
    export CC="$toolchain/bin/x86_64-conda-linux-gnu-gcc"
    export CXX="$toolchain/bin/x86_64-conda-linux-gnu-g++"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}" MAX_JOBS="${MAX_JOBS:-2}"
    cuda_target="$toolchain/targets/x86_64-linux"
    if [[ -f "$cuda_target/include/cuda_runtime.h" ]]; then
        export CPATH="$cuda_target/include${CPATH:+:$CPATH}"
        export LIBRARY_PATH="$cuda_target/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    fi
    nvcc --version
    "$CXX" --version

    work="$(mktemp -d "$runtime_dir/install.XXXXXXXX")"
    trap 'rm -rf -- "$work"' EXIT
    mkdir -p -- "$work/pointops" "$work/Chamfer3D"
    cp -- pointops/setup.py "$work/pointops/"
    cp -R -- pointops/src "$work/pointops/"
    cp -- utils/Chamfer3D/{setup.py,chamfer_cuda.cpp,chamfer3D.cu} "$work/Chamfer3D/"

    PYTHONPATH= "$python" -m pip install --target "$work/python" \
        'numpy==1.26.4' 'open3d-cpu==0.18.0' scikit-learn matplotlib tensorboard
    PYTHONPATH="$work/python" "$python" -m pip install \
        --target "$work/python" --no-build-isolation --no-deps "$work/pointops" "$work/Chamfer3D"
    check_packages "$work/python"

    rm -rf -- "$work/pointops" "$work/Chamfer3D"
    ln -s -- "$work/python" "$work/ready"
    mv -Tf -- "$work/ready" "$packages"
    trap - EXIT
    exit 0
fi

[[ -d "$packages" ]] || fail 'Run setup first.'
packages="$(realpath -- "$packages")"
check_packages "$packages"
[[ "$mode" == check ]] && exit 0

input_dir="$test_data/input_2048"
gt_dir="$test_data/gt_8192"
checkpoint="$project_dir/pretrained_model/pu_gaussian_pu1k_Best.pth"
[[ -s "$checkpoint" ]] || fail "Missing checkpoint: $checkpoint"
shopt -s nullglob
inputs=("$input_dir"/*.xyz)
if [[ "$mode" == smoke ]]; then
    [[ "${#inputs[@]}" -gt 0 ]] || fail "No XYZ files in $input_dir."
    inputs=("${inputs[0]}")
else
    [[ "${#inputs[@]}" -eq 127 ]] || fail "Expected 127 XYZ files in $input_dir, found ${#inputs[@]}."
fi
for input in "${inputs[@]}"; do
    [[ -f "$gt_dir/${input##*/}" ]] || fail "Missing ground truth for ${input##*/} in $gt_dir"
done

if [[ "$mode" == smoke ]]; then
    run_dir="$(mktemp -d "$runtime_dir/smoke.XXXXXXXX")"
    mkdir -- "$run_dir/input" "$run_dir/gt"
    sample="${inputs[0]##*/}"
    ln -s -- "${inputs[0]}" "$run_dir/input/$sample"
    ln -s -- "$gt_dir/$sample" "$run_dir/gt/$sample"
    input_dir="$run_dir/input"
    gt_dir="$run_dir/gt"
else
    mkdir -p -- "$project_dir/results"
    run_dir="$(mktemp -d "$project_dir/results/PU1K_4x.XXXXXXXX")"
fi

printf 'Output directory: %s\n' "$run_dir"
PYTHONPATH="$packages" "$python" -u test.py --dataset pu1k \
    --test_input_path "$input_dir" --test_gt_path "$gt_dir" \
    --ckpt "$checkpoint" --save_dir "$run_dir" --r 4 --up_rate 4 \
    2>&1 | tee "$run_dir/run.log"

PYTHONPATH="$packages" "$python" - "$input_dir" "$run_dir/xyz" <<'PY'
from pathlib import Path
import sys
import numpy as np

inputs, outputs = map(Path, sys.argv[1:])
names = {path.name for path in inputs.glob('*.xyz')}
assert names == {path.name for path in outputs.glob('*.xyz')}, 'Output names differ from input names'
for name in sorted(names):
    points = np.loadtxt(outputs / name)
    assert points.shape == (8192, 3) and np.isfinite(points).all(), name
print(f'Validated {len(names)} predictions of 8192 points in {outputs}')
PY
