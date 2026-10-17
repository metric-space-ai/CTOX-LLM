#!/usr/bin/env bash
set -euo pipefail
base=/run/media/deck/STEAMOS_DATA/ctox
root=${1:?new admitted root}; source=$(cd "$(dirname "$0")" && pwd)
python=$base/venvs/vllm-q8/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$base/venvs/vllm-q8/lib/python3.12/site-packages/nvidia/cu13/lib:$base/llama.cpp/build-cuda-shim3/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec 9<"$base/q8-1m-readiness-20260906-v1/audit.lock"
flock -n 9 || { echo 'Qwen lease occupied'; exit 75; }
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then echo 'GPU occupied'; exit 75; fi
artifact=/home/deck/ctox-p1-gate-20260908/clip-bestscales-v1.ctoxq
module=$base/recovery-diagnostic-20260905-v1/kernel-fixed/output/q2q4_fused_matvec_sm86.cubin
printf '%s  %s\n' b756d08e6e5db2ffe528b00055c24d1f0ed312e47a97a402898caf58193ad8e3 "$artifact" 6bc9191fb023135ccbc196ec8f2ce83bad9458df20b5892ca15a8d4759be3070 "$module" | sha256sum -c -
"$python" "$source/check_contracts.py"
"$python" "$source/prepare_checkpoint.py" --previous /home/deck/ctox-p3a-clip-refit-20260908-v3 --artifact "$artifact" --output "$root/scales" --chats "$source/frozen/chats.jsonl"
"$python" "$source/checkpoint_evaluate.py" --previous /home/deck/ctox-block-calibration-20261017-v2/result --scales "$root/scales" --artifact "$artifact" --native "$source/chat-calibration-student" --module "$module" --output "$root/result"
sha256sum "$source"/*.py "$source"/*.rs "$source"/*.cpp "$source"/*.sh "$source"/*student "$source/block_bridge.so" > "$root/source-sha256.txt"
du -sh "$root/result"
