#!/usr/bin/env python3
"""Recover completed scale-only fits without rewriting an 8 GB model artifact."""
import argparse,hashlib,json,struct
from pathlib import Path
import numpy as np
from replay import patched_manifest

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def manifest(path):
    with path.open('rb') as f:
        h=f.read(64)
        if h[:8]!=b'CTOXQ2Q4':raise ValueError('artifact magic')
        n=struct.unpack_from('<Q',h,16)[0]
        if n>64*1024*1024:raise ValueError('unbounded manifest')
        return json.loads(f.read(n))

def compare_weights(original,current):
    def weights(d):return {r['name']:{k:r[k] for k in ['sha256','dtype','shape','length']} for r in d['tensors'] if not r['name'].endswith(('.s_in','.s_out'))}
    if weights(original)!=weights(current):raise ValueError('checkpoint quantized weights or non-scale tensors differ')

def run(a):
    from safetensors.numpy import load_file
    fit=json.loads((a.previous/'full-v1/fit-report.json').read_text())
    selection_path=a.previous/'selection.json';selection=json.loads(selection_path.read_text())
    if fit['status']!='complete' or fit['selection_sha256']!=sha(selection_path):raise ValueError('completed fit/selection binding differs')
    evaluation={r['id'] for r in map(json.loads,a.chats.read_text().splitlines())}
    if not selection['sample_ids'] or evaluation&set(selection['sample_ids']):raise ValueError('historical calibration overlaps the evaluation chats')
    binding=selection['bindings']['student'];source=Path(binding['path'])
    if sha(source)!=binding['sha256']:raise ValueError('fit initializer artifact changed')
    current=manifest(a.artifact);compare_weights(manifest(source),current)
    scales_path=a.previous/'full-v1/scales.safetensors'
    with scales_path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        if n>8*1024*1024:raise ValueError('unbounded scale header')
        metadata=json.loads(f.read(n))['__metadata__']
    if metadata.get('status')!='complete' or metadata.get('fixed_logical_qcodes')!='true' or metadata.get('student_artifact_sha256')!=binding['sha256'] or metadata.get('report_sha256')!=sha(a.previous/'full-v1/fit-report.json'):
        raise ValueError('scale metadata does not bind the completed fit and initializer')
    if metadata.get('q8_main_sha256')!='f5c702d8820d36fb55985bb238fc83ee3a313e920f4b752a437c3a6a9e14e4c8':raise ValueError('checkpoint used a different Q8 teacher')
    scales=load_file(str(scales_path))
    if any(v.dtype!=np.float16 for v in scales.values()):raise ValueError('checkpoint scales are not stored FP16')
    body={k:v for k,v in scales.items() if k.startswith('model.language_model.layers.') and 0<=int(k.split('.')[3])<64}
    if {int(k.split('.')[3]) for k in body}!=set(range(64)):raise ValueError('main block scale checkpoint incomplete')
    a.output.mkdir(parents=True,exist_ok=False)
    for name,values in [('body-only',body),('all-scales',scales)]:
        patched_manifest(current,values,scope='recovery')
        np.savez(a.output/(name+'.npz'),**values)
    report=dict(format='ctox.completed-block-fit-replay.v1',artifact_sha256=sha(a.artifact),initializer_sha256=binding['sha256'],
        fit_report_sha256=sha(a.previous/'full-v1/fit-report.json'),selection_sha256=sha(selection_path),scales_sha256=sha(scales_path),
        same_non_scale_payload_manifest=True,evaluation_chats_sha256=sha(a.chats),historical_identity_overlap=0,variants=['body-only','all-scales'],
        scope='Historical completed fit on short windows; new full-chat TV evaluation only, no new training or promotion',target_met=False)
    (a.output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['previous','artifact','output','chats']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
