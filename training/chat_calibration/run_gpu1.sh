#!/usr/bin/env bash
set -euo pipefail
base=/run/media/deck/STEAMOS_DATA/ctox
root=${1:?new run root}; source=$(cd "$(dirname "$0")" && pwd)
python=$base/venvs/vllm-q8/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$base/venvs/vllm-q8/lib/python3.12/site-packages/nvidia/cu13/lib:$base/llama.cpp/build-cuda-shim3/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec 9<"$base/q8-1m-readiness-20260906-v1/audit.lock"
flock -n 9 || { echo 'Qwen runtime lease held'; exit 75; }
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then echo 'GPU occupied'; exit 75; fi
artifact=/home/deck/ctox-p1-gate-20260908/clip-bestscales-v1.ctoxq
module=$base/recovery-diagnostic-20260905-v1/kernel-fixed/output/q2q4_fused_matvec_sm86.cubin
printf '%s  %s\n' b756d08e6e5db2ffe528b00055c24d1f0ed312e47a97a402898caf58193ad8e3 "$artifact" 6bc9191fb023135ccbc196ec8f2ce83bad9458df20b5892ca15a8d4759be3070 "$module" | sha256sum -c -
"$python" "$source/check_contracts.py"
"$python" "$source/prepare.py" --corpus "$base/recovery-data/q8-scale-100k-balanced-20260905-v3/balanced-train-1m.jsonl" --teacher-root "$base/qwen38-q8-teacher" --output "$root/result" --train 24 --tune 8 --heldout 16
"$python" "$source/capture.py" --chats "$root/result/chats.jsonl" --output "$root/result" --teacher-root "$base/qwen38-q8-teacher" --library "$base/llama.cpp/build-cuda-shim3/bin/libllama.so" --bridge "$source/q8_bridge.so"
"$source/chat-calibration-student" "$artifact" "$module" "$root/result/student" < "$root/result/chats.jsonl"
"$python" "$source/fit.py" --root "$root/result" --artifact "$artifact" --legacy-training /home/deck/ctox-p3a-clip-refit-source-20260908-v3
"$source/chat-calibration-head" "$artifact" "$module" "$root/result/student.f32" "$root/result/head" "$root/result/head-native.jsonl"
"$python" "$source/verify.py" --root "$root/result"
sha256sum "$source"/*.py "$source"/*.rs "$source"/*.sh "$source/chat-calibration-student" "$source/chat-calibration-head" > "$root/source-sha256.txt"
du -sh "$root/result"
