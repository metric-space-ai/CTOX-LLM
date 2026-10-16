#!/usr/bin/env bash
set -euo pipefail
base=/run/media/deck/STEAMOS_DATA/ctox
root=${1:?new run root}; source=$(cd "$(dirname "$0")" && pwd)
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$base/venvs/vllm-q8/lib/python3.12/site-packages/nvidia/cu13/lib:$base/llama.cpp/build-cuda-shim3/bin"
exec 9<"$base/q8-1m-readiness-20260906-v1/audit.lock"
flock -n 9 || exit 75
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then exit 75; fi
"$base/venvs/vllm-q8/bin/python" "$source/head_diagnosis.py" --root /home/deck/ctox-chat-calibration-20261016-v6/kl-variant --reference "$base/qwen38-q8-teacher/Qwen3.8-27B-Q8_0.gguf" --legacy-training /home/deck/ctox-p3a-clip-refit-source-20260908-v3 --output "$root/result.json"
sha256sum "$source/head_diagnosis.py" "$source/diagnose_gpu1.sh" > "$root/source-sha256.txt"
