#!/usr/bin/env python3
"""Join the untouched reference rows with final native calibrated projections."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sampling_tv'))
from measure import tv,summary,sha

def run(a):
    teacher=[json.loads(x) for x in (a.root/'teacher.jsonl').read_text().splitlines()]
    student=[json.loads(x) for x in (a.root/'head-native.jsonl').read_text().splitlines()]
    baseline=[json.loads(x) for x in (a.root/'student.jsonl').read_text().splitlines()]
    if len(teacher)!=len(student) or len(teacher)!=len(baseline):raise ValueError('incomplete native validation')
    groups={};before={}
    for i,(t,s,b) in enumerate(zip(teacher,student,baseline)):
        if s['row']!=i or b['row']!=i:raise ValueError('misaligned native rows')
        for group in [t['split'],t['split']+'/'+t['bucket']]:
            groups.setdefault(group,[]).append(tv(t['distribution'],s['distribution']))
            before.setdefault(group,[]).append(tv(t['distribution'],b['distribution']))
    result=dict(format='ctox.complete-chat-native-head-validation.v1',baseline={k:summary(v) for k,v in before.items()},
        candidate={k:summary(v) for k,v in groups.items()},
        heldout_screen_below_5_percent=summary(groups['heldout'])['mean_percent']<5,
        overall_target_met=False,reason='Original Q8-generated diagnostic and broader free generation still required',
        hashes={name:sha(a.root/name) for name in ['teacher.jsonl','student.jsonl','head-native.jsonl','head.s_in.f16','head.s_out.f16']})
    (a.root/'native-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args())
