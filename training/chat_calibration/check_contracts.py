#!/usr/bin/env python3
"""Run the real mask contract and show a one-line mutation makes it fail."""
import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(root/'test_contract.py')],check=True)
p=root/'prepare.py';original=p.read_text();needle="active=ids[i+1:i+1+len(header)]==header"
assert original.count(needle)==1
try:
    p.write_text(original.replace(needle,'active=True'))
    r=subprocess.run([sys.executable,str(root/'test_contract.py'),'Contracts.test_only_assistant_targets_with_shift'],capture_output=True,text=True)
    if r.returncode==0 or 'FAIL:' not in r.stderr:raise RuntimeError('mask mutation survived')
    print('MUTATION PROOF: role check -> active=True; test_only_assistant_targets_with_shift fails',flush=True)
finally:p.write_text(original)
