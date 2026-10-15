#!/usr/bin/env python3
"""Choose the predeclared head variants exclusively by tuning TV."""
import argparse,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sampling_tv'))
from measure import sha

def run(a):
    reports=[json.loads((r/'fit-result.json').read_text()) for r in a.candidates]
    if len(reports)!=2 or [r['tv_weight'] for r in reports]!=[0.,1.]:raise ValueError('expected predeclared KL and KL+TV variants')
    if any(r['sources']!=reports[0]['sources'] or r['student_sha256']!=reports[0]['student_sha256'] for r in reports):raise ValueError('candidate data or model bindings differ')
    scores=[r['after']['tune']['tv']['mean_percent'] for r in reports]
    chosen=min(range(len(scores)),key=lambda i:(scores[i],i));root=a.candidates[chosen]
    a.output.mkdir(parents=True,exist_ok=False)
    for name in ['chats.jsonl','teacher.jsonl','teacher.complete.json','teacher-targets.npy','student.jsonl','student.f32']:
        (a.output/name).symlink_to((root/name).resolve())
    for name in ['head.s_in.f16','head.s_out.f16']:shutil.copyfile(root/name,a.output/name)
    decision=dict(selection='minimum tuning TV; first candidate wins exact ties',tune_percent=scores,selected=str(root),
        candidate_report_hashes=[sha(r/'fit-result.json') for r in a.candidates],heldout_used_for_selection=False)
    (a.output/'selection.json').write_text(json.dumps(decision,indent=2)+'\n');print(json.dumps(decision),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidates',type=Path,nargs=2,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
