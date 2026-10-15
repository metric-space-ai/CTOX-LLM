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
previous=/home/deck/ctox-chat-calibration-20261016-v5/result
artifact=/home/deck/ctox-p1-gate-20260908/clip-bestscales-v1.ctoxq
module=$base/recovery-diagnostic-20260905-v1/kernel-fixed/output/q2q4_fused_matvec_sm86.cubin
printf '%s  %s\n' b756d08e6e5db2ffe528b00055c24d1f0ed312e47a97a402898caf58193ad8e3 "$artifact" 6bc9191fb023135ccbc196ec8f2ce83bad9458df20b5892ca15a8d4759be3070 "$module" | sha256sum -c -
"$python" "$source/check_sampler_contracts.py"
"$python" "$source/prepare_tool_supplement.py" --chats "$previous/chats.jsonl" --tokenizer "$base/qwen38-q8-teacher/tokenizer.json" --output "$root/supplement"
"$python" "$source/capture.py" --chats "$root/supplement/chats.jsonl" --output "$root/supplement" --teacher-root "$base/qwen38-q8-teacher" --library "$base/llama.cpp/build-cuda-shim3/bin/libllama.so" --bridge "$source/q8_bridge.so"
"$source/chat-calibration-student" "$artifact" "$module" "$root/supplement/student" < "$root/supplement/chats.jsonl"
"$python" "$source/merge_captures.py" --base "$previous" --supplement "$root/supplement" --output "$root/kl-variant"
"$python" "$source/fit.py" --root "$root/kl-variant" --artifact "$artifact" --legacy-training /home/deck/ctox-p3a-clip-refit-source-20260908-v3
mkdir "$root/tv-variant"
for name in chats.jsonl teacher.jsonl teacher.complete.json teacher-targets.npy student.jsonl student.f32; do ln -s "$root/kl-variant/$name" "$root/tv-variant/$name"; done
"$python" "$source/fit.py" --root "$root/tv-variant" --artifact "$artifact" --legacy-training /home/deck/ctox-p3a-clip-refit-source-20260908-v3 --tv-weight 1
"$python" "$source/select_candidate.py" --candidates "$root/kl-variant" "$root/tv-variant" --output "$root/selected"
"$source/chat-calibration-head" "$artifact" "$module" "$root/selected/student.f32" "$root/selected/head" "$root/selected/head-native.jsonl"
"$python" "$source/verify.py" --root "$root/selected"
"$python" "$source/analyze.py" --root "$root/selected" --tokenizer "$base/qwen38-q8-teacher/tokenizer.json"
"$python" "$source/legacy_check.py" --reference /home/deck/ctox-sampling-tv-20261016-v1/result --output "$root/legacy" --student "$source/chat-calibration-student" --head "$source/chat-calibration-head" --artifact "$artifact" --cubin "$module" --scales "$root/selected/head" --tokenizer "$base/qwen38-q8-teacher/tokenizer.json"
sha256sum "$source"/*.py "$source"/*.rs "$source"/*.sh "$source/chat-calibration-student" "$source/chat-calibration-head" > "$root/source-sha256.txt"
