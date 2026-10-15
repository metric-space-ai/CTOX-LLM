#!/usr/bin/env python3
"""Bounded shared-lane reservation for a native GPU run outside its CUDA-less container."""
import json,os,sys,time
from pathlib import Path
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
owner='01a070b3-b5ab-7ec3-bbfb-c8c47df32e1a'
(root/'admitted.json').write_text(json.dumps(dict(owner=owner,pid=os.getpid(),unix_time=time.time(),deadline_seconds=4620)))
while time.monotonic()-start<4620:
    if (root/'terminal.json').exists():
        r=json.loads((root/'terminal.json').read_text());print(json.dumps(r),flush=True);sys.exit(r['exit'])
    if time.monotonic()-start>120 and not (root/'native-start.json').exists():raise SystemExit('native start missing')
    time.sleep(2)
raise SystemExit('native deadline exceeded')
