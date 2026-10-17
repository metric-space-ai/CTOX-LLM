#!/usr/bin/env python3
"""Evaluate one tuning-selected candidate on frozen heldout and generation cohorts."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import replay,verify

def read(path):return list(map(json.loads,path.read_text().splitlines()))
def write(path,rows):path.write_text(''.join(json.dumps(r)+'\n' for r in rows))

def run(a):
    candidates=[]
    for strength in [.25,.5,1.]:
        result=json.loads((a.root/f'candidate-{strength}'/'tv.json').read_text())
        candidates.append((result['paired']['tune']['paired_delta_percentage_points'],strength))
    delta,strength=min(candidates)
    selection=dict(strength=strength,tuning_delta_pp=delta,policy='lowest overall tuning TV among fixed .25/.5/1 strengths; no heldout selection',target_met=False)
    (a.root/'selection-candidate.json').write_text(json.dumps(selection,indent=2)+'\n')
    if delta>=0:
        print('All candidates rejected: no overall native tuning TV improvement',flush=True);return
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(a.tokenizer));enc=lambda s:tok.encode(s,add_special_tokens=False).ids
    def annotated(rows,chats):
        spans={r['id']:verify.chat_analysis.tool_positions(r['token_ids'],enc('<tool_call>'),enc('</tool_call>'),tok.token_to_id('<|im_start|>')) for r in chats}
        return [dict(r,is_tool_call=r['position']+1 in spans[r['id']]) for r in rows]
    heldout=[r for r in read(a.frozen/'chats.jsonl') if r['split']=='heldout']
    identities={r['id'] for r in heldout}
    used={r['id'] for r in read(a.root/'chats.jsonl')}
    if len(heldout)!=16 or used&identities:raise ValueError('heldout cohort changed or overlaps fit/tuning')
    teacher=[r for r in read(a.frozen/'teacher.jsonl') if r['id'] in identities]
    baseline=[r for r in read(a.frozen/'student.jsonl') if r['id'] in identities]
    cohorts=[('heldout',heldout,annotated(teacher,heldout),baseline)]
    requests=read(a.legacy/'requests.jsonl');lookup={r['id']:r for r in requests}
    generated=[dict(r,positions=list(range(r['prompt_tokens']-1,len(r['token_ids'])-1))) for r in requests]
    def convert(path):
        return [dict(r,position=lookup[r['id']]['prompt_tokens']+r['offset']-1,split='generation') for r in read(path)]
    teacher=convert(a.legacy/'teacher.jsonl');baseline=convert(a.legacy/'student.jsonl')
    if sum(len(r['positions']) for r in generated)!=973:raise ValueError('original generation cohort changed')
    cohorts.append(('generation',generated,annotated(teacher,generated),baseline))
    reports={}
    for name,chats,teacher,baseline in cohorts:
        d=a.root/name;d.mkdir(exist_ok=False)
        for fn,rows in [('chats.jsonl',chats),('teacher.jsonl',teacher),('baseline.jsonl',baseline)]:write(d/fn,rows)
        replay.run(SimpleNamespace(artifact=a.artifact,overrides=a.root/'scale-overrides.npz',native=a.native,module=a.module,
            chats=d/'chats.jsonl',output=d/'candidate',strength=strength,timeout=1500))
        verify.run(SimpleNamespace(reference=d/'teacher.jsonl',baseline=d/'baseline.jsonl',candidate=d/'candidate/student.jsonl',output=d/'tv.json'))
        reports[name]=json.loads((d/'tv.json').read_text())
    result=dict(selection=selection,cohorts=reports,target_met=False,
        limitation='Frozen holdout is excluded from this fit/selection but was evaluated in earlier experiments; it is not a new blind test. Source-chat and Q8-generated prefixes are distinct populations')
    (a.root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','frozen','legacy','tokenizer','artifact','native','module']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
