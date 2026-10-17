#!/usr/bin/env python3
"""Compare aligned native candidate distributions; never grade prompt tokens."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sampling_tv'))
from measure import tv,summary,sha

def run(a):
    reference=[json.loads(x) for x in a.reference.read_text().splitlines()]
    candidate=[json.loads(x) for x in a.candidate.read_text().splitlines()]
    if len(reference)!=len(candidate):raise ValueError('incomplete candidate')
    grouped={};tools={};rows=[]
    for t,s in zip(reference,candidate):
        if (t['id'],t['position'])!=(s['id'],s['position']):raise ValueError('prefix alignment differs')
        split=t.get('split','evaluation');value=tv(t['distribution'],s['distribution']);grouped.setdefault(split,[]).append(value)
        if t.get('is_tool_call',False):tools.setdefault(split,[]).append(value)
        rows.append(dict(id=t['id'],position=t['position'],is_tool_call=t.get('is_tool_call'),tv=value))
    result=dict(summary={k:summary(v) for k,v in grouped.items()},actual_tool_call_summary={k:summary(v) for k,v in tools.items()},rows=rows,
        reference_sha256=sha(a.reference),candidate_sha256=sha(a.candidate),phase='assistant targets only',
        target_met=False,limitation='tuning result is not independent heldout acceptance')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summary']),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['reference','candidate','output']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
