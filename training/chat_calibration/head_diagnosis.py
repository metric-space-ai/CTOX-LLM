#!/usr/bin/env python3
"""Can a Q8 output projection recover distributions from the frozen native body?"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sampling_tv'))
from measure import distribution,tv,summary,sha

def run(a):
    import numpy as np
    import torch
    sys.path.insert(0,str(a.legacy_training))
    from q8_gguf import GgufReader,resolve_reference_tensor
    from diagnose_q8_recovery import a8
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    complete=json.loads((a.root/'teacher.complete.json').read_text())
    if sha(a.reference)!=complete['teacher_sha256']:raise ValueError('Q8 reference changed')
    teacher=list(map(json.loads,(a.root/'teacher.jsonl').read_text().splitlines()))
    native=list(map(json.loads,(a.root/'student.jsonl').read_text().splitlines()))
    if len(teacher)!=len(native) or any((t['id'],t['position'],t['row'])!=(n['id'],n['position'],n['row']) for t,n in zip(teacher,native)):raise ValueError('capture rows differ')
    x=np.fromfile(a.root/'student.f32',dtype='<f4').reshape(len(teacher),5120)
    ids=[i for i,t in enumerate(teacher) if t['split'] in ['train','tune']]
    if not ids:raise ValueError('no calibration positions')
    inputs=torch.from_numpy(x[ids]).cuda();groups={}
    print(json.dumps({'stage':'decode_q8_head','positions':len(ids),'heldout_evaluated':False}),flush=True)
    with GgufReader(a.reference) as reader:
        name,offset=resolve_reference_tensor(reader,'lm_head.weight');rows,cols=reader.tensors[name].logical_shape
        if rows!=complete['vocab'] or cols!=5120 or offset!=0:raise ValueError('Q8 output geometry differs')
        weights=torch.empty((rows,cols),device='cuda',dtype=torch.float32)
        for first in range(0,rows,512):weights[first:first+512]=reader.decode_q8_rows(name,first,min(rows,first+512),torch,'cuda')
    with torch.inference_mode():
        for first in range(0,len(ids),8):
            chosen=ids[first:first+8];h=inputs[first:first+8]
            outputs={'q8_head_fp32_input':h@weights.T,'q8_head_a8_input':a8(h,torch)@weights.T}
            arrays={key:value.cpu().numpy() for key,value in outputs.items()}
            for j,i in enumerate(chosen):
                t=teacher[i];split=t['split']
                groups.setdefault(split+'/native_q2q4',[]).append(tv(t['distribution'],native[i]['distribution']))
                for key,values in arrays.items():groups.setdefault(split+'/'+key,[]).append(tv(t['distribution'],distribution(values[j])))
    result=dict(format='ctox.native-body-q8-head-diagnostic.v1',metrics={k:summary(v) for k,v in groups.items()},heldout_evaluated=False,
        scope='Frozen native normalized body states, Q8 head weights, no learned Q2/Q4 head corrections; FP32 matmul with either FP32 input or native-style A8 rounding',
        limitations=['Diagnostic only; not a native Q8 engine replay or shippable mixed-precision candidate','Not an additive attribution of body versus head quantization error; body/kernel/state differences remain'],
        teacher_sha256=complete['teacher_sha256'],helper_hashes={name:sha(a.legacy_training/name) for name in ['q8_gguf.py','diagnose_q8_recovery.py']},source_hashes={n:sha(a.root/n) for n in ['teacher.complete.json','teacher.jsonl','student.jsonl','student.f32']})
    with a.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','reference','legacy-training','output']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
