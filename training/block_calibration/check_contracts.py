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
p=root/'prepare_checkpoint.py';original=p.read_text();needle='if weights(original)!=weights(current):'
assert original.count(needle)==1
try:
 p.write_text(original.replace(needle,'if False:'))
 r=subprocess.run([sys.executable,str(root/'test_contract.py'),'Contracts.test_checkpoint_rejects_different_quantized_weights'],capture_output=True,text=True)
 assert r.returncode!=0 and 'FAIL:' in r.stderr,r.stderr
 print('MUTATION PROOF: remove checkpoint weight binding -> incompatible-weight rejection fails',flush=True)
finally:p.write_text(original)
if sys.platform=='linux':
 p=root/'replay.py';original=p.read_text();needle='fcntl.fcntl(fd,add,mask)'
 assert original.count(needle)==1
 try:
  p.write_text(original.replace(needle,'return'))
  r=subprocess.run([sys.executable,str(root/'test_contract.py'),'Contracts.test_sealed_candidate_cannot_be_modified'],capture_output=True,text=True)
  assert r.returncode!=0 and 'FAIL:' in r.stderr,r.stderr
  print('MUTATION PROOF: omit memfd sealing -> kernel immutability contract fails',flush=True)
 finally:p.write_text(original)
subprocess.run([sys.executable,str(root/'test_resume.py')],check=True)
p=root/'resume_holdout.py';original=p.read_text();needle='if sha(path)!=contract[\'sha256\'][name]:'
assert original.count(needle)==1
try:
 p.write_text(original.replace(needle,'if False:'))
 r=subprocess.run([sys.executable,str(root/'test_resume.py'),'ResumeContracts.test_changed_runtime_or_sampler_cannot_resume'],capture_output=True,text=True)
 assert r.returncode!=0 and 'FAIL:' in r.stderr,r.stderr
 print('MUTATION PROOF: omit resume hash comparison -> changed-runtime rejection fails',flush=True)
finally:p.write_text(original)
p=root/'resume_holdout.py';original=p.read_text();needle='if seen[key]==expected[key]'
assert original.count(needle)==1
try:
 p.write_text(original.replace(needle,'if seen[key]'))
 r=subprocess.run([sys.executable,str(root/'test_resume.py'),'ResumeContracts.test_partial_chat_is_replayed_whole_and_merge_is_exact'],capture_output=True,text=True)
 assert r.returncode!=0 and 'FAIL:' in r.stderr,r.stderr
 print('MUTATION PROOF: accept partial chats as complete -> whole-chat continuation contract fails',flush=True)
finally:p.write_text(original)
