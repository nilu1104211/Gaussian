#!/usr/bin/env bash
set -euo pipefail
mesh_dir="${1:?Provide the OFF mesh directory}"
pred_dir="${2:?Provide the predicted XYZ directory}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
for pred in "$pred_dir"/*.xyz; do
    [[ "$pred" == *_point2mesh_distance.xyz ]] && continue
    name="$(basename "$pred" .xyz)"
    "$script_dir/evaluation" "$mesh_dir/$name.off" "$pred"
done
