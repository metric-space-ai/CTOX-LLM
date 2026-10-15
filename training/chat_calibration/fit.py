#!/usr/bin/env python3
"""Bounded head-scale calibration screen. Packed weights and all body scales stay fixed.

Only train rows receive gradients; only tune rows select a checkpoint. Held-out
rows are evaluated once after selection. Native replay must verify the result.
"""
import argparse,json,math,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sampling_tv'))
from measure import distribution,tv,summary,sha

def run(a):
    sys.path.insert(0,str(a.legacy_training))
    import numpy as np
    import torch
    from ctox_artifact import CtoxArtifact
    from diagnose_q8_recovery import a8
    torch.set_num_threads(2);torch.manual_seed(20261016)
    torch.backends.cuda.matmul.allow_tf32=False
    complete=json.loads((a.root/'teacher.complete.json').read_text())
    if sha(a.root/'teacher-targets.npy')!=complete['targets_sha256'] or sha(a.root/'chats.jsonl')!=complete['chats_sha256']:
        raise ValueError('teacher targets or complete chats changed after capture')
    teacher=[json.loads(x) for x in (a.root/'teacher.jsonl').read_text().splitlines()]
    native=[json.loads(x) for x in (a.root/'student.jsonl').read_text().splitlines()]
    if len(teacher)!=len(native) or any((t['id'],t['position'],t['row'])!=(s['id'],s['position'],s['row']) for t,s in zip(teacher,native)):
        raise ValueError('teacher/native rows misaligned')
    target=np.load(a.root/'teacher-targets.npy',mmap_mode='r');n,vocab=target.shape
    x=np.fromfile(a.root/'student.f32',dtype='<f4').reshape(n,5120)
    x=torch.from_numpy(x).cuda()
    ids={s:[i for i,r in enumerate(teacher) if r['split']==s] for s in ['train','tune','heldout']}
    if any(not v for v in ids.values()):raise ValueError('empty split')
    print(json.dumps({'stage':'decode_head','positions':{k:len(v) for k,v in ids.items()}}),flush=True)
    name='lm_head.weight'
    with CtoxArtifact(a.artifact,verify_tensors=False) as art:
        art.verify_tensor(name);art.verify_tensor(name+'.s_in');art.verify_tensor(name+'.s_out')
        rows,cols=art.tensors[name]['shape']
        if rows!=vocab:raise ValueError('vocabulary shape mismatch')
        w=torch.empty((rows,cols),device='cuda',dtype=torch.float32)
        for first in range(0,rows,512):w[first:first+512]=art.decode_matrix_rows(name,first,min(rows,first+512),torch,'cuda')
        initial_in=art.decode_float_tensor(name+'.s_in',torch,'cuda').float()
        initial_out=art.decode_float_tensor(name+'.s_out',torch,'cuda').float()
    p_in=torch.nn.Parameter(initial_in.log());p_out=torch.nn.Parameter(initial_out.log())
    base_in=p_in.detach().clone();base_out=p_out.detach().clone()
    def fp16_ste(t):return t+(t.half().float()-t).detach()
    def predict(indices):
        sin=fp16_ste(p_in.exp());sout=fp16_ste(p_out.exp())
        z=x[indices]*sin
        z=z+(a8(z,torch)-z).detach()
        return (z@w.T)*sout
    def metrics(indices,check_native=False):
        values=[];parity=[];kl=[]
        with torch.no_grad():
            for first in range(0,len(indices),8):
                ix=indices[first:first+8];logits=predict(ix)
                ref=torch.from_numpy(np.array(target[ix],dtype=np.float32)).cuda().log_softmax(-1)
                kl.extend((ref.exp()*(ref-logits.log_softmax(-1))).sum(-1).cpu().tolist())
                for i,l in zip(ix,logits.cpu().numpy()):
                    d=distribution(l);values.append(tv(teacher[i]['distribution'],d))
                    if check_native:parity.append(tv(native[i]['distribution'],d))
        return dict(tv=summary(values),kl=float(np.mean(kl)),native_parity=summary(parity))
    # Check numerical equivalence on training/tuning only, before optimization.
    before={s:metrics(ids[s],True) for s in ['train','tune']}
    if any(v['native_parity']['mean_percent']>.1 for v in before.values()):
        (a.root/'fit-rejected.json').write_text(json.dumps(before,indent=2))
        raise ValueError('head surrogate/native post-sampler parity exceeds 0.1 percentage point')
    print(json.dumps({'baseline':before}),flush=True)
    best=before['tune']['tv']['mean_percent'];chosen=0
    best_in=initial_in.half().cpu();best_out=initial_out.half().cpu()
    optimizer=torch.optim.Adam([p_in,p_out],lr=a.lr)
    history=[];started=time.monotonic();generator=np.random.default_rng(20261016)
    for step in range(1,a.steps+1):
        ix=generator.choice(ids['train'],size=min(8,len(ids['train'])),replace=False).tolist()
        ref=torch.from_numpy(np.array(target[ix],dtype=np.float32)).cuda().log_softmax(-1)
        logits=predict(ix);kl=(ref.exp()*(ref-logits.log_softmax(-1))).sum(-1).mean()
        penalty=a.regularization*((p_in-base_in).square().mean()+(p_out-base_out).square().mean())
        loss=kl+penalty
        if not torch.isfinite(loss):raise ValueError('nonfinite calibration loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p_in,p_out],1.0);optimizer.step()
        with torch.no_grad():p_in.clamp_(math.log(.25),math.log(4));p_out.clamp_(math.log(.25),math.log(4))
        if step%32==0 or step==a.steps:
            score=metrics(ids['tune']);history.append(dict(step=step,train_batch_kl=float(kl.detach()),tune=score))
            print(json.dumps(history[-1]),flush=True)
            if score['tv']['mean_percent']<best:
                best=score['tv']['mean_percent'];chosen=step
                best_in=p_in.detach().exp().half().cpu();best_out=p_out.detach().exp().half().cpu()
    with torch.no_grad():p_in.copy_(best_in.cuda().float().log());p_out.copy_(best_out.cuda().float().log())
    best_in.numpy().astype('<f2').tofile(a.root/'head.s_in.f16');best_out.numpy().astype('<f2').tofile(a.root/'head.s_out.f16')
    after={s:metrics(ids[s]) for s in ids}
    native_before={s:summary([tv(teacher[i]['distribution'],native[i]['distribution']) for i in ids[s]]) for s in ids}
    result=dict(format='ctox.complete-chat-head-scale-screen.v1',scope='lm_head s_in/s_out only; body and quantized codes frozen',
        before=before,native_baseline=native_before,after=after,selected_step=chosen,history=history,
        elapsed=time.monotonic()-started,steps=a.steps,lr=a.lr,regularization=a.regularization,
        student_sha256=sha(a.artifact),sources={p:sha(a.root/p) for p in ['chats.jsonl','teacher.jsonl','student.jsonl','student.f32','teacher-targets.npy']},
        native_candidate_verified=False,target_met=False,
        limitations=['selected assistant positions on teacher-forced complete source chats','bounded 2048-token cohort','head-only calibration; no general quantization recovery claim','MTP not evaluated'])
    (a.root/'fit-result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'selected_step':chosen,'native_baseline':native_before,'after':after}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','artifact','legacy-training']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=192);p.add_argument('--lr',type=float,default=.001);p.add_argument('--regularization',type=float,default=.1)
    run(p.parse_args())
