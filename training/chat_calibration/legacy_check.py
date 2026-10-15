#!/usr/bin/env python3
"""Keep the original sampled-Q8 TV benchmark as an unchanged regression check."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sampling_tv'))
from measure import tv,summary,sha

def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    requests=[json.loads(l) for l in (a.reference/'requests.jsonl').read_text().splitlines()]
    teacher=[json.loads(l) for l in (a.reference/'teacher.jsonl').read_text().splitlines()]
    old=[json.loads(l) for l in (a.reference/'student.jsonl').read_text().splitlines()]
    rows=[];native_requests=[]
    for r in requests:
        positions=list(range(r['prompt_tokens']-1,len(r['token_ids'])-1))
        native_requests.append(dict(id=r['id'],token_ids=r['token_ids'],positions=positions))
    request=a.output/'requests.jsonl';request.write_text(''.join(json.dumps(r)+'\n' for r in native_requests))
    with request.open() as inp:
        subprocess.run([str(a.student),str(a.artifact),str(a.cubin),str(a.output/'student')],stdin=inp,check=True)
    subprocess.run([str(a.head),str(a.artifact),str(a.cubin),str(a.output/'student.f32'),str(a.scales),str(a.output/'candidate.jsonl')],check=True)
    baseline=[json.loads(l) for l in (a.output/'student.jsonl').read_text().splitlines()]
    candidate=[json.loads(l) for l in (a.output/'candidate.jsonl').read_text().splitlines()]
    if not len(teacher)==len(old)==len(baseline)==len(candidate):raise ValueError('legacy row count differs')
    position_by_id={r['id']:r['prompt_tokens']-1 for r in requests};groups={};oldvalues=[];parity=[]
    for t,o,b,c in zip(teacher,old,baseline,candidate):
        if (t['id'],t['offset'])!=(o['id'],o['offset']) or (t['id'],t['offset']+position_by_id[t['id']])!=(b['id'],b['position']):raise ValueError('legacy alignment differs')
        if c['row']!=b['row']:raise ValueError('candidate row differs')
        parity.append(tv(o['distribution'],b['distribution']))
        oldvalues.append(tv(t['distribution'],o['distribution']))
        for label in ['overall',t['bucket']]:groups.setdefault(label,[]).append(tv(t['distribution'],c['distribution']))
    if max(parity)>1e-6:raise ValueError('legacy baseline replay differs')
    result=dict(format='ctox.original-tv-calibration-regression.v1',baseline=summary(oldvalues),native_replay_parity=summary(parity),
        candidate={k:summary(v) for k,v in groups.items()},below_5_percent=summary(groups['overall'])['mean_percent']<5,
        reference_hashes={n:sha(a.reference/n) for n in ['requests.jsonl','teacher.jsonl','student.jsonl']},
        scales_hashes={str(p):sha(p) for p in [Path(str(a.scales)+'.s_in.f16'),Path(str(a.scales)+'.s_out.f16')]})
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['reference','output','student','head','artifact','cubin','scales']:p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
