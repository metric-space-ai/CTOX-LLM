#!/usr/bin/env python3
"""Capture Q8 block boundaries at assistant targets with complete preceding context."""
import argparse,ctypes as C,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sampling_tv'))
from measure import load_bridge,distribution,sha

def run(a):
    import numpy as np
    records=[json.loads(x) for x in a.chats.read_text().splitlines()]
    model=a.teacher_root/'Qwen3.8-27B-Q8_0.gguf'
    if sha(model)!='f5c702d8820d36fb55985bb238fc83ee3a313e920f4b752a437c3a6a9e14e4c8':raise ValueError('Q8 changed')
    b,s=load_bridge(a.library,a.bridge,model)
    b.block_enable.argtypes=[C.c_void_p];b.block_enable.restype=None
    b.block_hidden.argtypes=[C.c_void_p,C.c_int32,C.c_int32];b.block_hidden.restype=C.POINTER(C.c_float)
    rowno=0
    try:
        b.block_enable(s);vocab=b.tv_vocab(s)
        with (a.output/'teacher.f32').open('xb') as hf,(a.output/'teacher.jsonl').open('x') as jf:
            for r in records:
                b.tv_reset(s);cursor=0
                for pos in r['positions']:
                    chunk=r['token_ids'][cursor:pos+1];arr=(C.c_int32*len(chunk))(*chunk)
                    if b.tv_eval(s,arr,len(chunk))!=0:raise RuntimeError('Q8 decode')
                    local_row=(len(chunk)-1)%128
                    for layer in range(65):
                        ptr=b.block_hidden(s,layer,local_row)
                        if not ptr:raise RuntimeError(f'missing hidden layer {layer}')
                        values=np.ctypeslib.as_array(ptr,shape=(5120,)).copy()
                        if not np.isfinite(values).all():raise ValueError('nonfinite state')
                        hf.write(values.astype('<f4').tobytes())
                    logits=np.ctypeslib.as_array(b.tv_logits(s),shape=(vocab,))
                    jf.write(json.dumps(dict(id=r['id'],position=pos,row=rowno,split=r['split'],bucket=r['bucket'],distribution=distribution(logits)))+'\n')
                    rowno+=1;cursor=pos+1
                hf.flush();jf.flush();print(f'Q8 blocks {r["id"][:12]} {len(r["positions"])} positions',flush=True)
    finally:b.tv_close(s)
    (a.output/'capture.json').write_text(json.dumps(dict(rows=rowno,shape=[rowno,65,5120],
        boundary='layer0 embedding; layer1..63 raw next-block inputs;64 final normalized hidden',
        chats_sha256=sha(a.chats),teacher_sha256=sha(a.output/'teacher.f32'),library_sha256=sha(a.library),bridge_sha256=sha(a.bridge)))+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['chats','output','teacher-root','library','bridge']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
