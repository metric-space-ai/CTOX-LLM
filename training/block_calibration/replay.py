#!/usr/bin/env python3
"""Apply scale overrides to a sealed RAM-only pack and replay the native engine.

No persistent 8GB pack copy; quantized payloads and tensor offsets stay intact.
The memfd is released when the bounded child completes.
"""
import argparse,hashlib,json,os,struct,subprocess,sys
from pathlib import Path

def patched_manifest(manifest,overrides):
    import numpy as np
    doc=json.loads(json.dumps(manifest));changes={}
    entries={x['name']:x for x in doc['tensors']}
    for name,value in overrides.items():
        if not name.endswith('.mlp.down_proj.weight.s_out') or name not in entries:raise ValueError('only existing FFN output scales may change')
        e=entries[name];v=np.asarray(value,dtype='<f2')
        if e['dtype']!='f16' or list(v.shape)!=e['shape'] or not np.isfinite(v).all() or not ((v>=.25)&(v<=4)).all():raise ValueError('invalid scale shape/range')
        payload=v.tobytes()
        if len(payload)!=e['length']:raise ValueError('scale length')
        e['sha256']=hashlib.sha256(payload).hexdigest();changes[e['offset']]=payload
    if not changes:raise ValueError('empty calibration candidate')
    return doc,changes

def run(a):
    import numpy as np,fcntl
    header_struct=struct.Struct('<8sIIQQII24x')
    with a.artifact.open('rb') as source:
        header=list(header_struct.unpack(source.read(64)));length,offset=header[3:5]
        if header[0]!=b'CTOXQ2Q4':raise ValueError('pack magic')
        manifest=json.loads(source.read(length));overrides=dict(np.load(a.overrides))
        if a.strength!=1:
            entries={e['name']:e for e in manifest['tensors']}
            for name,value in overrides.items():
                e=entries[name];source.seek(offset+e['offset']);old=np.frombuffer(source.read(e['length']),dtype='<f2').astype('f4')
                overrides[name]=(old+(value.astype('f4')-old)*a.strength).astype('<f2')
        doc,changes=patched_manifest(manifest,overrides)
        data=json.dumps(doc,separators=(',',':'),sort_keys=True).encode()
        if 64+len(data)>offset:raise ValueError('manifest exceeds original reserved area')
        header[3]=len(data)
        fd=os.memfd_create('ctox-calibration',os.MFD_ALLOW_SEALING)
        try:
            source.seek(0)
            with os.fdopen(os.dup(fd),'wb',closefd=True) as target:
                for chunk in iter(lambda:source.read(8*1024*1024),b''):target.write(chunk)
                target.seek(0);target.write(header_struct.pack(*header));target.write(data);target.write(bytes(offset-64-len(data)))
                for start,payload in changes.items():target.seek(offset+start);target.write(payload)
                target.flush()
            fcntl.fcntl(fd,fcntl.F_ADD_SEALS,fcntl.F_SEAL_WRITE|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_GROW|fcntl.F_SEAL_SEAL)
            a.output.mkdir(parents=True,exist_ok=False)
            with a.chats.open() as request:
                command=[str(a.native),f'/proc/self/fd/{fd}',str(a.module),str(a.output/'student')]
                subprocess.run(command,stdin=request,pass_fds=(fd,),check=True,timeout=a.timeout)
            (a.output/'override.json').write_text(json.dumps(dict(strength=a.strength,changed_tensors=list(overrides),
                overrides_sha256=hashlib.sha256(a.overrides.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(data).hexdigest(),
                backing='sealed memfd; all original quantized tensor payloads preserved'))+'\n')
        finally:os.close(fd)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['artifact','overrides','native','module','chats','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--strength',type=float,default=1);p.add_argument('--timeout',type=int,default=1200)
    a=p.parse_args()
    if not 0<a.strength<=1:raise SystemExit('invalid strength')
    run(a)
