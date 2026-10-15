#!/usr/bin/env python3
"""Q8 targets on complete chats; dense targets are bounded temporary FP16 rows."""
import argparse,ctypes as C,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sampling_tv'))
from measure import load_bridge,distribution,sha,PROFILE

def run(a):
    import numpy as np
    from tokenizers import Tokenizer
    root=a.output;records=[json.loads(x) for x in a.chats.read_text().splitlines()]
    expected='f5c702d8820d36fb55985bb238fc83ee3a313e920f4b752a437c3a6a9e14e4c8'
    model=a.teacher_root/'Qwen3.8-27B-Q8_0.gguf'
    if sha(model)!=expected:raise ValueError('Q8 hash mismatch')
    tokenizer=Tokenizer.from_file(str(a.teacher_root/'tokenizer.json'))
    b,s=load_bridge(a.library,a.bridge,model)
    try:
        vocab=b.tv_vocab(s)
        n=sum(len(r['positions']) for r in records)
        if n*vocab*2>600*1024**2:raise ValueError('target cache exceeds 600 MiB budget')
        targets=np.lib.format.open_memmap(root/'teacher-targets.npy',mode='w+',dtype='<f2',shape=(n,vocab))
        rowno=0;piece_cache={};tokenizer_differences=[]
        with (root/'teacher.jsonl').open('x') as f:
            for rec in records:
                ids=rec['token_ids'];encoded=rec['rendered'].encode();tmp=(C.c_int32*2048)()
                count=b.tv_tokenize(s,encoded,len(encoded),tmp,2048)
                if ids!=tokenizer.encode(rec['rendered'],add_special_tokens=False).ids:
                    raise ValueError('official tokenizer/selection changed')
                if count<0 or list(tmp[:count])!=ids:
                    tokenizer_differences.append(dict(id=rec['id'],hf_tokens=len(ids),gguf_tokens=count))
                # HF defines canonical input IDs. Different segmentation is permitted
                # only if Q8's vocabulary decodes those exact IDs to identical bytes.
                for token in set(ids)-piece_cache.keys():
                    piece=C.create_string_buffer(4096);length=b.tv_piece(s,token,piece,4096)
                    if length<0:raise ValueError('token piece exceeds buffer')
                    piece_cache[token]=piece.raw[:length]
                if b''.join(piece_cache[t] for t in ids)!=encoded:
                    raise ValueError('GGUF vocabulary does not round-trip official input IDs')
                b.tv_reset(s);cursor=0
                for pos in rec['positions']:
                    chunk=ids[cursor:pos+1];arr=(C.c_int32*len(chunk))(*chunk)
                    if not chunk or b.tv_eval(s,arr,len(chunk))!=0:raise RuntimeError('teacher decode')
                    x=np.ctypeslib.as_array(b.tv_logits(s),shape=(vocab,)).copy()
                    d=distribution(x);targets[rowno]=x-x.max()
                    f.write(json.dumps(dict(id=rec['id'],position=pos,split=rec['split'],bucket=rec['bucket'],row=rowno,
                        distribution=d,prefix_sha256=hashlib.sha256(np.asarray(ids[:pos+1],dtype='<u4').tobytes()).hexdigest()))+'\n')
                    rowno+=1;cursor=pos+1
                targets.flush();f.flush();print(f'teacher {rec["split"]} {rec["id"][:12]} {len(ids)} context tokens',flush=True)
        del targets
    finally:b.tv_close(s)
    (root/'teacher.complete.json').write_text(json.dumps(dict(rows=rowno,vocab=vocab,teacher_sha256=expected,
        targets_sha256=sha(root/'teacher-targets.npy'),chats_sha256=sha(a.chats),profile=PROFILE,
        cache_precision='maximum-centered fp16 logits; exact float64 post-sampler reference retained',
        trajectory='source-chat teacher forcing',phase='assistant_only',
        input_tokenizer='official HF; Q8 vocabulary byte round-trip verified for every complete chat',
        gguf_segmentation_differences=tokenizer_differences))+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['chats','output','teacher-root','library','bridge']:p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
