#!/usr/bin/env python3
"""Native half of admitted run, with explicit ownership and process-group deadline."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);source=Path(__file__).resolve().parent
runner=sys.argv[2] if len(sys.argv)>2 else "run_gpu1.sh"
if runner not in {"run_gpu1.sh","refine_gpu1.sh","diagnose_gpu1.sh"}:raise SystemExit("unknown bounded runner")
deadline=int(sys.argv[3]) if len(sys.argv)>3 else 4500
if not 1<=deadline<=4500:raise SystemExit("invalid bounded deadline")
admitted=json.loads((root/'admitted.json').read_text())
if admitted['owner']!='01a070b3-b5ab-7ec3-bbfb-c8c47df32e1a' or not 0<=time.time()-admitted['unix_time']<120:raise SystemExit('invalid admission')
os.kill(admitted['pid'],0)
(root/'native-start.json').write_text(json.dumps(dict(owner=admitted['owner'],pid=os.getpid(),unix_time=time.time(),deadline_seconds=deadline,source=str(source),runner=runner)))
(root/'tmp').mkdir();rc=1
try:
    with (root/'native.log').open('x') as log:
        child=subprocess.Popen(['bash',str(source/runner),str(root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,TMPDIR=str(root/'tmp'),PYTHONDONTWRITEBYTECODE='1'))
        (root/'process.json').write_text(json.dumps(dict(owner=admitted['owner'],pid=child.pid,process_group=child.pid,stop=f'normal completion or {deadline}s deadline')))
        try:rc=child.wait(timeout=deadline)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            rc=124
finally:
    p=root/'terminal.partial';p.write_text(json.dumps(dict(exit=rc,finished=time.time(),owner=admitted['owner'])));p.rename(root/'terminal.json')
sys.exit(rc)
