#!/usr/bin/env python3
"""Resume flushed complete conversations, preserving stopped-run evidence."""
import argparse,json,re,shutil
from pathlib import Path

def run(a):
    if not (a.previous/'terminal.json').exists():raise ValueError('previous capture still active')
    original=list(map(json.loads,(a.previous/'result/chats.jsonl').read_text().splitlines()))
    completed=re.findall(r'^student ([0-9a-f]{64}) (\d+) positions$',(a.previous/'native.log').read_text(),re.M)
    if [x[0] for x in completed]!=[r['id'] for r in original[:len(completed)]]:raise ValueError('completed chats are not a contiguous prefix')
    n=sum(int(x[1]) for x in completed)
    if n!=sum(len(r['positions']) for r in original[:len(completed)]):raise ValueError('completed position count differs')
    done=a.output/'completed';replay=a.output/'replay';done.mkdir();replay.mkdir()
    with (a.previous/'result/student.jsonl').open() as inp,(done/'student.jsonl').open('x') as out:
        for i in range(n):
            line=inp.readline();r=json.loads(line)
            if r['row']!=i:raise ValueError('completed rows misaligned')
            out.write(line)
    size=n*5120*4
    with (a.previous/'result/student.f32').open('rb') as inp,(done/'student.f32').open('xb') as out:
        while size:
            chunk=inp.read(min(size,1024*1024))
            if not chunk:raise ValueError('completed hidden rows truncated')
            out.write(chunk);size-=len(chunk)
    supplement={r['id']:r for r in map(json.loads,a.supplement.read_text().splitlines())}
    done_chats=original[:len(completed)];requests=[]
    for r in original:
        extra=supplement.get(r['id'],{}).get('positions',[])
        positions=sorted(extra if r in done_chats else set(r['positions'])|set(extra))
        if positions:requests.append(dict(r,positions=positions))
    (done/'chats.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in done_chats))
    (replay/'chats.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in requests))
    result=dict(reused_chats=len(completed),reused_positions=n,replay_chats=len(requests),replay_tokens=sum(r['positions'][-1]+1 for r in requests))
    (a.output/'resume.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['previous','supplement','output']:p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
