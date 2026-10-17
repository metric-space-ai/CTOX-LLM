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
"$python" - "$root" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
previous=Path('/home/deck/ctox-block-calibration-20261017-v1/result');root=Path(sys.argv[1]);dest=root/'result';dest.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads((previous/'analysis.json').read_text())
for name in ['teacher.jsonl','student.jsonl','chats.jsonl']:
 if sha(previous/name)!=report['sources'][name]:raise ValueError('capture input changed')
for name in ['analysis.json','scale-overrides.npz','chats.jsonl','teacher.jsonl','student.jsonl','tune-chats.jsonl','tune-teacher.jsonl']:
 shutil.copyfile(previous/name,dest/name)
 if sha(previous/name)!=sha(dest/name):raise ValueError('resume copy changed')
(root/'resume.json').write_text(json.dumps(dict(source=str(previous),analysis_sha256=sha(previous/'analysis.json'),capture_repeated=False))+'\n')
PY
for strength in 0.25 0.5 1.0; do
    "$python" "$source/replay.py" --artifact "$artifact" --overrides "$root/result/scale-overrides.npz" --native "$source/chat-calibration-student" --module "$module" --chats "$root/result/tune-chats.jsonl" --output "$root/result/candidate-$strength" --strength "$strength"
    "$python" "$source/verify.py" --reference "$root/result/tune-teacher.jsonl" --baseline "$root/result/student.jsonl" --candidate "$root/result/candidate-$strength/student.jsonl" --output "$root/result/candidate-$strength/tv.json"
done
"$python" "$source/evaluate.py" --root "$root/result" --frozen "$source/frozen" --legacy "$source/legacy" --tokenizer "$base/qwen38-q8-teacher/tokenizer.json" --artifact "$artifact" --native "$source/chat-calibration-student" --module "$module"
sha256sum "$source"/*.py "$source"/*.rs "$source"/*.cpp "$source"/*.sh "$source"/*student "$source/block_bridge.so" > "$root/source-sha256.txt"
du -sh "$root/result"
