#!/usr/bin/env python3
"""Describe actual tool-call spans and paired uncertainty without selecting models."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sampling_tv'))
from measure import tv,summary,sha

def tool_positions(ids,opening,closing,role_start):
    result=set();active=False;i=0
    while i<len(ids):
        if ids[i]==role_start:active=False
        if ids[i:i+len(opening)]==opening:active=True
        if active and ids[i:i+len(closing)]==closing:
            result.update(range(i,i+len(closing)));active=False;i+=len(closing);continue
        if active:result.add(i)
        i+=1
    return result

def paired_summary(rows):
    import numpy as np
    chats={}
    for identity,before,after in rows:chats.setdefault(identity,[]).append((before,after))
    if not chats:return dict(baseline=summary([]),candidate=summary([]),chats=0)
    sums=np.array([[sum(x[1]-x[0] for x in values),len(values)] for values in chats.values()])
    draws=np.random.default_rng(20261016).integers(0,len(sums),size=(2000,len(sums)))
    selected=sums[draws].sum(1);delta=100*selected[:,0]/selected[:,1]
    return dict(baseline=summary([r[1] for r in rows]),candidate=summary([r[2] for r in rows]),chats=len(chats),
        paired_delta_percentage_points=100*sum(r[2]-r[1] for r in rows)/len(rows),
        paired_chat_bootstrap_95_percentile_interval_pp=np.quantile(delta,[.025,.975]).tolist())

def run(a):
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(a.tokenizer));opening=tok.encode('<tool_call>',add_special_tokens=False).ids;closing=tok.encode('</tool_call>',add_special_tokens=False).ids
    chats={r['id']:r for r in map(json.loads,(a.root/'chats.jsonl').read_text().splitlines())}
    spans={key:tool_positions(r['token_ids'],opening,closing,tok.token_to_id('<|im_start|>')) for key,r in chats.items()}
    teacher=list(map(json.loads,(a.root/'teacher.jsonl').read_text().splitlines()))
    baseline=list(map(json.loads,(a.root/'student.jsonl').read_text().splitlines()))
    candidate=list(map(json.loads,(a.root/'head-native.jsonl').read_text().splitlines()))
    if not len(teacher)==len(baseline)==len(candidate):raise ValueError('row count differs')
    groups={}
    for i,(t,b,c) in enumerate(zip(teacher,baseline,candidate)):
        if (t['id'],t['position'],i)!=(b['id'],b['position'],b['row']) or c['row']!=i:raise ValueError('row identity differs')
        kind='tool_call' if t['position']+1 in spans[t['id']] else 'other_assistant'
        row=(t['id'],tv(t['distribution'],b['distribution']),tv(t['distribution'],c['distribution']))
        for key in [t['split'],t['split']+'/'+kind]:groups.setdefault(key,[]).append(row)
    result=dict(groups={k:paired_summary(v) for k,v in groups.items()},
        scope='Selected assistant positions; tool_call includes delimiters, name, arguments; not semantic correctness',
        uncertainty='Paired bootstrap by whole conversation, 2000 resamples; descriptive of this bounded cohort',
        hashes={name:sha(a.root/name) for name in ['chats.jsonl','teacher.jsonl','student.jsonl','head-native.jsonl']})
    (a.root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True);run(p.parse_args())
