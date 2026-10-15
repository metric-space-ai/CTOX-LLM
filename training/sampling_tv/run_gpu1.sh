#!/usr/bin/env bash
set -euo pipefail
# Run only through the admitted gpu1 build lane after the prior owner's release.
base=/run/media/deck/STEAMOS_DATA/ctox
out=${1:?Pass a new output directory}
script_dir=$(cd "$(dirname "$0")" && pwd)
python=$base/venvs/vllm-q8/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$base/venvs/vllm-q8/lib/python3.12/site-packages/nvidia/cu13/lib:$base/llama.cpp/build-cuda-shim3/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec 9<"$base/q8-1m-readiness-20260906-v1/audit.lock"
flock -n 9 || { echo 'Qwen runtime lease held by another process'; exit 75; }
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo 'GPU has another compute process; refusing model load'; exit 75
fi
artifact=/home/deck/ctox-p1-gate-20260908/clip-bestscales-v1.ctoxq
module=$base/recovery-diagnostic-20260905-v1/kernel-fixed/output/q2q4_fused_matvec_sm86.cubin
printf '%s  %s\n' b756d08e6e5db2ffe528b00055c24d1f0ed312e47a97a402898caf58193ad8e3 "$artifact" 6bc9191fb023135ccbc196ec8f2ce83bad9458df20b5892ca15a8d4759be3070 "$module" | sha256sum -c -
"$python" "$script_dir/test_measure.py"
timeout 900 "$python" "$script_dir/measure.py" capture --teacher-root "$base/qwen38-q8-teacher" \
    --library "$base/llama.cpp/build-cuda-shim3/bin/libllama.so" --bridge "$script_dir/q8_bridge.so" \
    --cases "$script_dir/cases.json" --output "$out" --seed 20261016 --max-tokens 128
# Q8 exits and releases VRAM before the native candidate starts.
timeout 1500 "$script_dir/sampling-tv-student" "$artifact" "$module" < "$out/requests.jsonl" > "$out/student.jsonl"
"$python" "$script_dir/measure.py" compare --output "$out"
sha256sum "$artifact" "$module" "$script_dir/sampling-tv-student" "$script_dir/q8_bridge.so" "$script_dir/measure.py" "$script_dir/cases.json" > "$out/runtime-sha256.txt"
du -sh "$out"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
