#!/usr/bin/env python3
"""Assistant-generation-only TV under an explicit, fixed sampling profile.

No dense logits are persisted. Top-k is applied before nucleus truncation;
therefore omitted tokens have EXACTLY zero post-sampler probability.
"""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

PROFILE = dict(temperature=1.0, top_k=20, top_p=0.95, order=['temperature','top_k','top_p'],
               repetition_penalty=1.0, min_p=0.0, grammar=None)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def distribution(logits):
    import numpy as np
    x=np.asarray(logits,dtype=np.float64)
    if len(x)<20 or not np.isfinite(x).all(): raise ValueError('invalid logits')
    # Tie rule is identical to Rust: lower token ID first.
    ids=np.lexsort((np.arange(len(x)),-x))[:20]
    weights=np.exp(x[ids]-x[ids[0]]); weights/=weights.sum()
    n=min(20,int(np.searchsorted(np.cumsum(weights),.95,side='left'))+1)
    weights=weights[:n]; weights/=weights.sum()
    return [[int(i),float(p)] for i,p in zip(ids[:n],weights)]

def validate_distribution(rows):
    d=dict(rows)
    if len(d)!=len(rows) or not d or len(d)>20: raise ValueError('invalid sparse support')
    if any(not isinstance(i,int) or i<0 or not math.isfinite(p) or p<=0 for i,p in rows):
        raise ValueError('invalid token probability')
    if abs(sum(d.values())-1)>1e-8: raise ValueError('probability mass is missing')
    return d

def tv(a,b):
    p,q=validate_distribution(a),validate_distribution(b)
    return .5*sum(abs(p.get(i,0)-q.get(i,0)) for i in p.keys()|q.keys())

def load_bridge(library,bridge,model):
    C.CDLL(str(library),mode=C.RTLD_GLOBAL)
    b=C.CDLL(str(bridge))
    ptr=C.c_void_p; ints=C.POINTER(C.c_int32)
    signatures={
        'tv_open':([C.c_char_p,C.c_int32],ptr), 'tv_close':([ptr],None),
        'tv_reset':([ptr],None),'tv_vocab':([ptr],C.c_int32),
        'tv_tokenize':([ptr,C.c_char_p,C.c_int32,ints,C.c_int32],C.c_int32),
        'tv_piece':([ptr,C.c_int32,C.c_void_p,C.c_int32],C.c_int32),
        'tv_eog':([ptr,C.c_int32],C.c_bool), 'tv_eval':([ptr,ints,C.c_int32],C.c_int32),
        'tv_logits':([ptr],C.POINTER(C.c_float))}
    for name,(args,res) in signatures.items():
        f=getattr(b,name); f.argtypes=args; f.restype=res
    s=b.tv_open(os.fsencode(model),2048)
    if not s: raise RuntimeError('Q8 model load failed')
    return b,s

def capture(a):
    import numpy as np
    from jinja2 import Environment, StrictUndefined
    root=Path(a.teacher_root); out=Path(a.output); out.mkdir(exist_ok=False,parents=True)
    expected='f5c702d8820d36fb55985bb238fc83ee3a313e920f4b752a437c3a6a9e14e4c8'
    model=root/'Qwen3.8-27B-Q8_0.gguf'
    if sha(model)!=expected: raise ValueError('Q8 checksum mismatch')
    config=json.loads((root/'generation_config.json').read_text())
    if any(config.get(k)!=PROFILE[k] for k in ['temperature','top_k','top_p']):
        raise ValueError('profile differs from supplied generation_config')
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(root/"tokenizer.json"))
    cases=json.loads(Path(a.cases).read_text())
    env=Environment(undefined=StrictUndefined)
    def fail(msg): raise ValueError(msg)
    env.globals['raise_exception']=fail
    template=env.from_string((root/'chat_template.jinja').read_text())
    meta=dict(format='ctox.assistant-sampling-tv.v1',profile=PROFILE,seed=a.seed,max_tokens=a.max_tokens,
        teacher_sha256=expected,teacher_library_sha256=sha(a.library),bridge_sha256=sha(a.bridge),
        tokenizer_sha256=sha(root/'tokenizer.json'),template_sha256=sha(root/'chat_template.jinja'),
        cases_sha256=sha(a.cases),enable_thinking=False,trajectory='sampled-Q8',
        phase='assistant_generation',forced_prefix_included=False,measured_precision='Q8 via llama.cpp versus native Q2/Q4 calibration graph; state and KV differences not isolated',
        intended_samples=len(cases))
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    b,s=load_bridge(a.library,a.bridge,model)
    try:
        vocab=b.tv_vocab(s)
        with (out/'teacher.jsonl').open('x') as tf,(out/'requests.jsonl').open('x') as rf,(out/'answers.jsonl').open('x') as af:
            for case_no,case in enumerate(cases):
                rng=np.random.default_rng(a.seed+case_no)
                prompt=template.render(messages=case['messages'],tools=case.get('tools',[]),enable_thinking=False,add_generation_prompt=True)
                encoded=prompt.encode(); tokens=(C.c_int32*2048)()
                n=b.tv_tokenize(s,encoded,len(encoded),tokens,2048)
                if n<=0 or n+a.max_tokens>2048: raise ValueError('prompt/context limit')
                ids=list(tokens[:n])
                if ids!=tokenizer.encode(prompt,add_special_tokens=False).ids: raise ValueError("GGUF/HF tokenizer mismatch")
                prompt_n=n; output_bytes=bytearray(); pieces=[]; generated=[]
                b.tv_reset(s)
                if b.tv_eval(s,tokens,n)!=0: raise RuntimeError('Q8 prefill failed')
                for offset in range(a.max_tokens):
                    logits=np.ctypeslib.as_array(b.tv_logits(s),shape=(vocab,))
                    d=distribution(logits)
                    token=int(rng.choice([i for i,p in d],p=[p for i,p in d]))
                    prefix_sha=hashlib.sha256(np.asarray(ids,dtype='<u4').tobytes()).hexdigest()
                    row=dict(id=case['id'],offset=offset,phase='assistant_generation',bucket=case['bucket'],
                        prefix_sha256=prefix_sha,sampled_token=token,distribution=d)
                    tf.write(json.dumps(row,separators=(',',':'))+'\n'); tf.flush()
                    piece=C.create_string_buffer(4096); length=b.tv_piece(s,token,piece,4096)
                    if length<0: raise RuntimeError('token piece too long')
                    part=piece.raw[:length]; pieces.append(part.hex()); output_bytes.extend(part)
                    ids.append(token); generated.append(token)
                    if b.tv_eog(s,token): break
                    one=(C.c_int32*1)(token)
                    if b.tv_eval(s,one,1)!=0: raise RuntimeError('Q8 decode failed')
                stop='eog' if b.tv_eog(s,generated[-1]) else 'length'
                rf.write(json.dumps(dict(id=case['id'],token_ids=ids,prompt_tokens=prompt_n,generated_tokens=len(generated),vocab_size=vocab))+'\n'); rf.flush()
                af.write(json.dumps(dict(id=case['id'],bucket=case['bucket'],prompt=prompt,answer=output_bytes.decode(errors='replace'),
                    pieces_hex=pieces,stop=stop))+'\n'); af.flush()
                print(f"captured {case_no+1}/{len(cases)} {case['id']} {len(generated)} output tokens ({stop})",flush=True)
    finally: b.tv_close(s)
    (out/'capture.complete').write_text(sha(out/'teacher.jsonl')+'\n')

def summary(vals):
    vals=sorted(vals); n=len(vals)
    if not n: return {'tokens':0}
    return dict(tokens=n,mean_percent=100*statistics.mean(vals),median_percent=100*statistics.median(vals),
        p90_percent=100*vals[math.ceil(.90*n)-1],p95_percent=100*vals[math.ceil(.95*n)-1],max_percent=100*vals[-1],
        tokens_above_percent={str(t):sum(x>t/100 for x in vals) for t in [1,5,10,25,50]})

def compare(a):
    root=Path(a.output)
    if not (root/'capture.complete').exists(): raise ValueError('capture incomplete')
    rows=[json.loads(x) for x in (root/'teacher.jsonl').read_text().splitlines()]
    student=[json.loads(x) for x in (root/'student.jsonl').read_text().splitlines()]
    answers={x['id']:x for x in map(json.loads,(root/'answers.jsonl').read_text().splitlines())}
    requests={x['id']:x for x in map(json.loads,(root/'requests.jsonl').read_text().splitlines())}
    if len(rows)!=len(student): raise ValueError('incomplete/misaligned student output')
    per_case={}; buckets={}; details=[]; spans={}; positions=set()
    import numpy as np
    for r,s in zip(rows,student):
        key=(r['id'],r['offset'])
        if key!=(s['id'],s['offset']) or key in positions or r['phase']!='assistant_generation': raise ValueError('invalid alignment/phase')
        positions.add(key); req=requests[r['id']]; off=r['offset']
        if not 0<=off<req['generated_tokens']: raise ValueError('prompt or excess position')
        prefix=req['token_ids'][:req['prompt_tokens']+off]
        if hashlib.sha256(np.asarray(prefix,dtype='<u4').tobytes()).hexdigest()!=r['prefix_sha256']: raise ValueError('prefix mismatch')
        if req['token_ids'][req['prompt_tokens']+off]!=r['sampled_token']: raise ValueError('trajectory mismatch')
        v=tv(r['distribution'],s['distribution'])
        per_case.setdefault(r['id'],[]).append(v); buckets.setdefault(r['bucket'],[]).append(v)
        answer=answers[r['id']]; pieces=[bytes.fromhex(x) for x in answer['pieces_hex']]
        start=sum(map(len,pieces[:off])); end=start+len(pieces[off]); raw=b''.join(pieces)
        # Label by overlap with the actually generated XML tool-call span.
        cursor=0; in_call=False
        while True:
            lo=raw.find(b'<tool_call>',cursor)
            if lo<0: break
            hi=raw.find(b'</tool_call>',lo); hi=len(raw) if hi<0 else hi+len(b'</tool_call>')
            if start<hi and end>lo: in_call=True; break
            cursor=hi
        kind='tool_call' if in_call else ('code_response' if r['bucket']=='code' else 'other_response')
        spans.setdefault(kind,[]).append(v)
        details.append(dict(id=r['id'],offset=off,bucket=r['bucket'],span=kind,tv=v,
            reference_token=r['sampled_token'],reference_piece=pieces[off].decode(errors='replace')))
    if len(per_case)!=len(requests) or any(len(per_case[k])!=v['generated_tokens'] for k,v in requests.items()): raise ValueError('missing generation positions')
    meta=json.loads((root/'metadata.json').read_text())
    result=dict(format='ctox.assistant-sampling-tv-result.v1',metadata=meta,overall=summary([r['tv'] for r in details]),
        buckets={k:summary(v) for k,v in buckets.items()},actual_spans={k:summary(v) for k,v in spans.items()},
        per_answer={k:summary(v) for k,v in per_case.items()},answer_weighted_mean_percent=100*statistics.mean(map(statistics.mean,per_case.values())),
        length_truncated_answers=[k for k,v in answers.items() if v['stop']=='length'],
        source_sha256={p:sha(root/p) for p in ['teacher.jsonl','student.jsonl','requests.jsonl','answers.jsonl']},
        limitations=['Small hand-authored diagnostic cohort, not representative population quality',
                    'Q8 sampled trajectories only; candidate free-running drift not measured',
                    'Q8 reference is not BF16 ground truth; runtime/state/kernel differences are included',
                    'Fixed temperature=1, top-k=20 then top-p=.95; no grammar constraints',
                    'Code bucket denotes a code request; tool_call span denotes actually emitted call markup',
                    'No tool executed and no semantic correctness claim'])
    with (root/'result.json').open('x') as f: json.dump(result,f,indent=2); f.write('\n')
    with (root/'positions.jsonl').open('x') as f:
        for r in details: f.write(json.dumps(r)+'\n')
    print(json.dumps(result['overall'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['capture','compare']); p.add_argument('--output',required=True)
    p.add_argument('--teacher-root'); p.add_argument('--library'); p.add_argument('--bridge'); p.add_argument('--cases')
    p.add_argument('--seed',type=int,default=20261016); p.add_argument('--max-tokens',type=int,default=128)
    a=p.parse_args(); {'capture':capture,'compare':compare}[a.command](a)
