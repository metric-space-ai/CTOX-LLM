import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
for script in root.glob('*.py'):
 compile(script.read_text(),str(script),'exec')
subprocess.run([sys.executable,str(root/'test_contract.py')],check=True)
p=root/'analyze.py';original=p.read_text();needle='np.sum(x*(target-residual),axis=0)'
assert original.count(needle)==1
try:
 p.write_text(original.replace(needle,'np.sum(x*target,axis=0)'))
 r=subprocess.run([sys.executable,str(root/'test_contract.py'),'Contracts.test_known_scale_recovery_and_trust_bound'],capture_output=True,text=True)
 assert r.returncode!=0 and 'FAIL:' in r.stderr,r.stderr
 print('MUTATION PROOF: omit residual subtraction -> known scale recovery fails',flush=True)
finally:p.write_text(original)
