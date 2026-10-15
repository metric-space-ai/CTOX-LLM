#!/usr/bin/env python3
"""Bounded native half of the existing gpu1 container/native admission pattern."""
import json, os, signal, subprocess, sys, time
from pathlib import Path
root=Path(sys.argv[1]); source=Path(__file__).resolve().parent
admitted=json.loads((root/'admitted.json').read_text())
if admitted['owner']!='01a070b3-b5ab-7ec3-bbfb-c8c47df32e1a' or not 0<=time.time()-admitted['unix_time']<120:
    raise SystemExit('missing or expired lane admission')
os.kill(admitted['pid'],0)
with (root/'native-start.json').open('x') as f:
    json.dump(dict(pid=os.getpid(),owner=admitted['owner'],source=str(source),unix_time=time.time(),deadline_seconds=2500),f)
(root/'tmp').mkdir()
env=dict(os.environ,TMPDIR=str(root/'tmp'),PYTHONDONTWRITEBYTECODE='1')
rc=1
try:
    with (root/'native.log').open('x') as log:
        child=subprocess.Popen(['bash',str(source/'run_gpu1.sh'),str(root/'result')],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        (root/'process.json').write_text(json.dumps(dict(pid=child.pid,process_group=child.pid,stop='completion or 2500-second deadline')))
        try: rc=child.wait(timeout=2500)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
            rc=124
finally:
    tmp=root/'terminal.partial'
    tmp.write_text(json.dumps(dict(exit=rc,finished=time.time(),owner=admitted['owner'])))
    tmp.rename(root/'terminal.json')
sys.exit(rc)
