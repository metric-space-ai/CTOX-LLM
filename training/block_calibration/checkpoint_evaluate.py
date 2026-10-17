#!/usr/bin/env python3
"""Screen two preregistered checkpoint variants, then use the original generation guard."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import replay,verify

def run(a):
    a.output.mkdir(exist_ok=False);scores=[]
    for name in ['body-only','all-scales']:
        output=a.output/name
        replay.run(SimpleNamespace(artifact=a.artifact,overrides=a.scales/(name+'.npz'),native=a.native,module=a.module,
            chats=a.previous/'tune-chats.jsonl',output=output,strength=1.,timeout=900,scale_scope='recovery'))
        verify.run(SimpleNamespace(reference=a.previous/'tune-teacher.jsonl',baseline=a.previous/'student.jsonl',candidate=output/'student.jsonl',output=output/'tv.json'))
        result=json.loads((output/'tv.json').read_text());scores.append((result['paired']['tune']['paired_delta_percentage_points'],name))
    delta,name=min(scores)
    selection=dict(variant=name,tuning_delta_pp=delta,policy='best of fixed body-only/all-scales variants by full-chat tuning TV',target_met=False)
    (a.output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    if delta>=0:
        print('Historical checkpoint rejected on full-chat tuning',flush=True);return
    frozen=a.previous/'generation';output=a.output/'generation'
    if not (frozen/'tv.json').exists():raise ValueError('original generation guard has not completed')
    replay.run(SimpleNamespace(artifact=a.artifact,overrides=a.scales/(name+'.npz'),native=a.native,module=a.module,
        chats=frozen/'chats.jsonl',output=output,strength=1.,timeout=1200,scale_scope='recovery'))
    verify.run(SimpleNamespace(reference=frozen/'teacher.jsonl',baseline=frozen/'baseline.jsonl',candidate=output/'student.jsonl',output=output/'tv.json'))
    (a.output/'evaluation.json').write_text(json.dumps(dict(selection=selection,generation=json.loads((output/'tv.json').read_text()),
        target_met=False,limitation='Historical calibration on short windows; new independent full-chat holdout is still required before any promotion'),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['previous','scales','artifact','native','module','output']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
