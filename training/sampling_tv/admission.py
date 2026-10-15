#!/usr/bin/env python3
"""Hold the gpu1 lane while its bounded native CUDA process runs on SteamOS.

The lane container cannot expose CUDA. Native execution begins only AFTER this
admission record exists; the native supervisor publishes the terminal receipt.
"""
import json, os, sys, time
from pathlib import Path
root=Path(sys.argv[1]); root.mkdir(parents=True,exist_ok=False)
start=time.monotonic()
(root/'admitted.json').write_text(json.dumps(dict(owner='01a070b3-b5ab-7ec3-bbfb-c8c47df32e1a',pid=os.getpid(),unix_time=time.time(),deadline_seconds=2700)))
while time.monotonic()-start < 2700:
    if (root/'terminal.json').exists():
        receipt=json.loads((root/'terminal.json').read_text()); print(json.dumps(receipt),flush=True); sys.exit(receipt['exit'])
    if time.monotonic()-start>120 and not (root/'native-start.json').exists():
        raise SystemExit('native start missing; releasing unused lane')
    time.sleep(2)
raise SystemExit('native runtime deadline exceeded')
