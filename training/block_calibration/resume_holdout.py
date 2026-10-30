#!/usr/bin/env python3
"""Finish a pinned native holdout without recomputing completed conversations."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import replay,verify

SAMPLER={'temperature':1,'top_k':20,'top_p':.95}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def load(path):return [json.loads(s) for s in path.read_text().splitlines()]

def partition(chats,rows):
    expected={c['id']:set(c['positions']) for c in chats}
    if len(expected)!=len(chats) or any(not c['positions'] or len(set(c['positions']))!=len(c['positions']) for c in chats):
        raise ValueError('duplicate chat or invalid target positions')
    seen={key:set() for key in expected}
    for row in rows:
        key,pos=row['id'],row['position']
        if key not in expected or pos not in expected[key] or pos in seen[key]:raise ValueError('unknown or duplicate prefix')
        seen[key].add(pos)
    complete={key for key in expected if seen[key]==expected[key]}
    return [c for c in chats if c['id'] not in complete],[r for r in rows if r['id'] in complete]

def merge_rows(chats,old,new):
    remaining,complete=partition(chats,old)
    unfinished,_=partition(remaining,new)
    if unfinished:raise ValueError('resumed conversations incomplete')
    rows={(r['id'],r['position']):r for r in complete+new}
    return [rows[c['id'],p] for c in chats for p in c['positions']]

def validate_bindings(contract,paths):
    if contract.get('sampler')!=SAMPLER or set(contract.get('sha256',{}))!=set(paths):
        raise ValueError('incomplete resume identity contract')
    for name,path in paths.items():
        if sha(path)!=contract['sha256'][name]:raise ValueError('resume identity changed: '+name)

def run(a):
    contract=json.loads(a.contract.read_text())
    paths={k:getattr(a,k) for k in ['artifact','overrides','native','module','chats','previous_rows','reference','baseline']}
    paths['replay_source']=Path(replay.__file__)
    validate_bindings(contract,paths)
    chats=load(a.chats);old=load(a.previous_rows)
    remaining,complete=partition(chats,old)
    a.output.mkdir(parents=True,exist_ok=False)
    request=a.output/'remaining-chats.jsonl';request.write_text(''.join(json.dumps(c)+'\n' for c in remaining))
    (a.output/'binding.json').write_text(json.dumps(dict(contract=contract,contract_sha256=sha(a.contract),remaining_chats=len(remaining),complete_positions=len(complete)),indent=2)+'\n')
    new=[]
    if remaining:
        replay.run(SimpleNamespace(artifact=a.artifact,overrides=a.overrides,native=a.native,module=a.module,
            chats=request,output=a.output/'replay',strength=1.,timeout=a.timeout,scale_scope='recovery'))
        new=load(a.output/'replay/student.jsonl')
    merged=merge_rows(chats,old,new);candidate=a.output/'student.jsonl'
    candidate.write_text(''.join(json.dumps(r)+'\n' for r in merged))
    verify.run(SimpleNamespace(reference=a.reference,baseline=a.baseline,candidate=candidate,output=a.output/'tv.json'))
    result=json.loads((a.output/'tv.json').read_text())
    result['limitation']='Complete frozen holdout, excluded from this fit/tuning but used in earlier experiments; not fresh blind acceptance or an answer-error rate'
    result['resume_contract_sha256']=sha(a.contract)
    (a.output/'tv.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['artifact','overrides','native','module','chats','previous-rows','reference','baseline','contract','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--timeout',type=int,default=1800)
    run(p.parse_args())
