#!/usr/bin/env python3
"""Localize residual drift and fit bounded existing FFN output scales.

Fits use train rows only. Tuning rejects local regressions, then native replay
must independently test downstream TV. No additive causal error attribution.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sampling_tv'))
from measure import tv,summary,sha

def relative(a,b):
    import numpy as np
    return float(np.mean(np.sum((a-b)**2,axis=-1)/np.maximum(np.sum(b*b,axis=-1),1e-20)))

def fit_gain(x,residual,target,regularization=.1):
    import numpy as np
    energy=np.sum(x*x,axis=0);ridge=regularization*np.mean(energy)+1e-12
    return np.clip((np.sum(x*(target-residual),axis=0)+ridge)/(energy+ridge),.9,1.1)

def run(a):
    import numpy as np
    sys.path.insert(0,str(a.legacy_training))
    from ctox_artifact import CtoxArtifact
    t=[json.loads(x) for x in (a.root/'teacher.jsonl').read_text().splitlines()]
    s=[json.loads(x) for x in (a.root/'student.jsonl').read_text().splitlines()]
    if len(t)!=len(s) or any((x['id'],x['position'],x['row'])!=(y['id'],y['position'],y['row']) for x,y in zip(t,s)):
        raise ValueError('state/target alignment')
    n=len(t);teacher=np.memmap(a.root/'teacher.f32',dtype='<f4',mode='r',shape=(n,65,5120))
    student=np.memmap(a.root/'student.f32',dtype='<f4',mode='r',shape=(n,65,5120))
    terms=np.memmap(a.root/'student-ffn.f32',dtype='<f4',mode='r',shape=(n,64,2,5120))
    ix={k:np.array([i for i,r in enumerate(t) if r['split']==k]) for k in ['train','tune']}
    if any(len(v)==0 for v in ix.values()) or any(r['split'] not in ix for r in t):raise ValueError('invalid calibration splits')
    boundaries=[];candidates=[]
    for layer in range(65):
        record={'boundary':layer,'meaning':'final_normalized' if layer==64 else 'raw_block_input',
            'relative_squared_error':{k:relative(np.array(student[v,layer],dtype='f8'),np.array(teacher[v,layer],dtype='f8')) for k,v in ix.items()}}
        boundaries.append(record)
    # Last block has a normalized target and is deliberately not fit by this linear rule.
    with CtoxArtifact(a.artifact,verify_tensors=False) as art:
        for layer in range(63):
            tr=ix['train'];tu=ix['tune'];x=np.array(terms[:,layer,1],dtype='f8');r=np.array(terms[:,layer,0],dtype='f8');y=np.array(teacher[:,layer+1],dtype='f8')
            identity=relative(r+x,np.array(student[:,layer+1],dtype='f8'))
            if identity>1e-10:raise ValueError(f'FFN boundary identity failed at {layer}: {identity}')
            gain=fit_gain(x[tr],r[tr],y[tr])
            name=f'model.language_model.layers.{layer}.mlp.down_proj.weight.s_out'
            art.verify_tensor(name);buf=art.tensor_bytes(name)
            try:old=np.frombuffer(buf,dtype='<f2').astype('f8')
            finally:buf.release()
            values=np.clip(old*gain,.25,4).astype('<f2');actual=values.astype('f8')/old
            before={k:relative(r[v]+x[v],y[v]) for k,v in ix.items()}
            after={k:relative(r[v]+x[v]*actual,y[v]) for k,v in ix.items()}
            if after['train']<before['train'] and after['tune']<before['tune']:
                candidates.append(dict(layer=layer,name=name,before=before,after=after,
                    train_gain=before['train']-after['train'],identity_error=identity,
                    old=old.tolist(),values=values.astype('f8').tolist()))
    chosen=sorted(candidates,key=lambda r:(-r['train_gain'],r['layer']))[:4]
    arrays={r['name']:np.asarray(r.pop('values'),dtype='<f2') for r in chosen}
    for r in chosen:r.pop('old')
    np.savez(a.root/'scale-overrides.npz',**arrays)
    result=dict(format='ctox.full-context-block-calibration.v1',boundaries=boundaries,chosen=chosen,
        baseline_tv={k:summary([tv(t[i]['distribution'],s[i]['distribution']) for i in v]) for k,v in ix.items()},
        policy='train-only ridge fit of FFN s_out; gains restricted to +/-10%; at most4 blocks ranked by train local gain, each must improve local tuning MSE; native downstream TV still required',
        artifact_sha256=sha(a.artifact),sources={p:sha(a.root/p) for p in ['teacher.f32','student.f32','student-ffn.f32','teacher.jsonl','student.jsonl','chats.jsonl']},
        target_met=False,native_candidate_verified=False,
        limitations=['small train/tune sensitivity experiment, not final heldout acceptance','local residual alignment is not downstream quality','runtime/state differences included','no MTP calibration'])
    (a.root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'baseline_tv':result['baseline_tv'],'chosen':chosen,'boundaries':boundaries}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','artifact','legacy-training']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
