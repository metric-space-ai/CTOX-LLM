#!/usr/bin/env python3
"""Merge identical-context Q8 captures and independently resumed native captures."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sampling_tv'))
from measure import sha

def run(a):
    import numpy as np
    roots=[a.base,a.supplement];captures=[];reference={};allrows={};native={};native_arrays=[]
    for k,root in enumerate(roots):
        complete=json.loads((root/'teacher.complete.json').read_text())
        for key,name in [('targets_sha256','teacher-targets.npy'),('chats_sha256','chats.jsonl')]:
            if sha(root/name)!=complete[key]:raise ValueError('capture changed: '+name)
        if k:
            for key in ['vocab','teacher_sha256','profile','library_sha256','bridge_sha256','tokenizer_sha256','template_sha256']:
                if complete[key]!=captures[0][0][key]:raise ValueError('runtime differs: '+key)
        chats=list(map(json.loads,(root/'chats.jsonl').read_text().splitlines()))
        if not k:reference={r['id']:r for r in chats}
        for r in chats:
            if r['id'] not in reference or r['token_ids']!=reference[r['id']]['token_ids'] or r['rendered']!=reference[r['id']]['rendered']:raise ValueError('full context differs')
            if k and (r['split']!='train' or reference[r['id']]['split']!='train'):raise ValueError('supplement touches evaluation')
        teachers=list(map(json.loads,(root/'teacher.jsonl').read_text().splitlines()));targets=np.load(root/'teacher-targets.npy',mmap_mode='r')
        if len(teachers)!=len(targets) or complete['rows']!=len(teachers):raise ValueError('incomplete teacher capture')
        expected={(r['id'],p) for r in chats for p in r['positions']};observed=set()
        for i,t in enumerate(teachers):
            key=(t['id'],t['position'])
            if t['row']!=i or key in allrows:raise ValueError('misaligned or duplicate teacher position')
            if t['split']!=reference[t['id']]['split']:raise ValueError('split differs')
            allrows[key]=(k,i,t);observed.add(key)
        if observed!=expected:raise ValueError('selected teacher positions differ')
        captures.append((complete,targets))
    for k,root in enumerate(a.native_captures):
        requests=list(map(json.loads,(root/'chats.jsonl').read_text().splitlines()))
        for r in requests:
            if r['token_ids']!=reference[r['id']]['token_ids']:raise ValueError('native full context differs')
        students=list(map(json.loads,(root/'student.jsonl').read_text().splitlines()))
        hidden=np.memmap(root/'student.f32',dtype='<f4',mode='r',shape=(len(students),5120));native_arrays.append(hidden)
        expected={(r['id'],p) for r in requests for p in r['positions']};observed=set()
        for i,s in enumerate(students):
            key=(s['id'],s['position'])
            if s['row']!=i or key in native:raise ValueError('misaligned or duplicate native position')
            native[key]=(k,i,s);observed.add(key)
        if observed!=expected:raise ValueError('native capture incomplete')
    if set(native)!=set(allrows):raise ValueError('native and teacher positions differ')
    chats=[];ordered=[]
    for r in reference.values():
        positions=sorted(pos for identity,pos in allrows if identity==r['id'])
        if r['split']!='train' and positions!=r['positions']:raise ValueError('evaluation selection changed')
        chats.append(dict(r,positions=positions));ordered.extend((r['id'],pos) for pos in positions)
    n=len(ordered);vocab=captures[0][0]['vocab']
    if n*vocab*2>600*1024**2:raise ValueError('merged targets exceed budget')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'chats.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chats))
    targets=np.lib.format.open_memmap(a.output/'teacher-targets.npy',mode='w+',dtype='<f2',shape=(n,vocab))
    with (a.output/'teacher.jsonl').open('x') as tfile,(a.output/'student.jsonl').open('x') as sfile,(a.output/'student.f32').open('xb') as hidden:
        for row,key in enumerate(ordered):
            k,i,t=allrows[key];nk,ni,s=native[key]
            targets[row]=captures[k][1][i];hidden.write(native_arrays[nk][ni].tobytes())
            tfile.write(json.dumps(dict(t,row=row))+'\n');sfile.write(json.dumps(dict(s,row=row))+'\n')
    targets.flush();del targets
    result=dict(captures[0][0],rows=n,targets_sha256=sha(a.output/'teacher-targets.npy'),chats_sha256=sha(a.output/'chats.jsonl'),
        capture_mode='merge of disjoint positions under identical runtime; evaluation unchanged',
        parents=[dict(path=str(r),complete_sha256=sha(r/'teacher.complete.json')) for r in roots],
        native_parents=[dict(path=str(r),requests_sha256=sha(r/'chats.jsonl'),student_sha256=sha(r/'student.jsonl'),hidden_sha256=sha(r/'student.f32')) for r in a.native_captures])
    (a.output/'teacher.complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'merged_positions':n,'evaluation_changed':False}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['base','supplement','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--native-captures',type=Path,nargs='+',required=True);run(p.parse_args())
