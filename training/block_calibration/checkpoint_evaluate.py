#!/usr/bin/env python3
"""Screen two preregistered checkpoint variants, then use the original generation guard."""
import argparse,json,subprocess
from pathlib import Path
from types import SimpleNamespace
import replay,verify

def holdout(a):
    selection=json.loads((a.output/'selection.json').read_text())
    name=selection['variant']
    if name not in {'body-only','all-scales'} or selection['tuning_delta_pp']>=0:
        raise ValueError('holdout requires a valid improving tuning selection')
    evaluation=json.loads((a.output/'evaluation.json').read_text())
    if evaluation['selection']!=selection or 'generation' not in evaluation:
        raise ValueError('generation guard must complete for the same selection')
    frozen=a.previous/'heldout';output=a.output/'heldout'
    replay.run(SimpleNamespace(artifact=a.artifact,overrides=a.scales/(name+'.npz'),native=a.native,module=a.module,
        chats=frozen/'chats.jsonl',output=output,strength=1.,timeout=1200,scale_scope='recovery'))
    verify.run(SimpleNamespace(reference=frozen/'teacher.jsonl',baseline=frozen/'baseline.jsonl',candidate=output/'student.jsonl',output=output/'tv.json'))
    evaluation['heldout']=json.loads((output/'tv.json').read_text())
    evaluation['limitation']='Frozen complete-chat holdout excludes this fit and tuning, but was used in earlier experiments; no fresh blind acceptance or model promotion'
    (a.output/'evaluation.json').write_text(json.dumps(evaluation,indent=2)+'\n')

def run(a):
    if getattr(a,'holdout_only',False):return holdout(a)
    a.output.mkdir(exist_ok=False);scores=[];failures=[]
    for name in ['body-only','all-scales']:
        output=a.output/name
        try:
            replay.run(SimpleNamespace(artifact=a.artifact,overrides=a.scales/(name+'.npz'),native=a.native,module=a.module,
                chats=a.previous/'tune-chats.jsonl',output=output,strength=1.,timeout=900,scale_scope='recovery'))
        except subprocess.CalledProcessError as e:
            failure=dict(variant=name,native_exit=e.returncode,quality_assessed=False)
            failures.append(failure);(output/'native-failure.json').write_text(json.dumps(failure)+'\n')
            continue
        verify.run(SimpleNamespace(reference=a.previous/'tune-teacher.jsonl',baseline=a.previous/'student.jsonl',candidate=output/'student.jsonl',output=output/'tv.json'))
        result=json.loads((output/'tv.json').read_text());scores.append((result['paired']['tune']['paired_delta_percentage_points'],name))
    if not scores:
        (a.output/'evaluation.json').write_text(json.dumps(dict(failed_variants=failures,target_met=False,status='no valid native candidate'))+'\n');return
    delta,name=min(scores)
    selection=dict(variant=name,tuning_delta_pp=delta,failed_variants=failures,policy='best valid candidate among fixed body-only/all-scales variants by full-chat tuning TV',target_met=False)
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
    holdout(a)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['previous','scales','artifact','native','module','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--holdout-only',action='store_true',help='resume only the frozen holdout after the original generation guard completed')
    run(p.parse_args())
