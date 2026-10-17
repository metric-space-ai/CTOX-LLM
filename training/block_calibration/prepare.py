#!/usr/bin/env python3
"""Select training/tuning chats only, without model-based sample filtering."""
import argparse,collections,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'chat_calibration'))
from prepare_tool_supplement import tool_positions
# Import by filename to avoid this module's name shadowing chat prepare.py.
import importlib.util
spec=importlib.util.spec_from_file_location('chat_prepare',Path(__file__).resolve().parents[1]/'chat_calibration/prepare.py')
chat=importlib.util.module_from_spec(spec);spec.loader.exec_module(chat)

def run(a):
    from tokenizers import Tokenizer
    from hashlib import sha256
    tok=Tokenizer.from_file(str(a.tokenizer));enc=lambda s:tok.encode(s,add_special_tokens=False).ids
    start,end=tok.token_to_id('<|im_start|>'),tok.token_to_id('<|im_end|>')
    src=[json.loads(x) for x in a.chats.read_text().splitlines()];selected=[];counts=collections.Counter()
    tool_count=0
    for r in sorted(src,key=lambda x:x['id']):
        limit={'train':4,'tune':2}.get(r['split'],0)
        key=(r['split'],r['bucket'])
        if counts[key]>=limit:continue
        eligible=set(chat.assistant_positions(r['token_ids'],start,end,enc('assistant\n'),enc('<think>\n\n</think>\n\n')))
        tool=sorted({i-1 for i in tool_positions(r['token_ids'],enc('<tool_call>'),enc('</tool_call>'),start)}&eligible)
        general=chat.selected_positions(sorted(eligible),8)
        p=sorted(set(general+(chat.selected_positions(tool,8) if tool else [])))
        selected.append(dict(r,positions=p));counts[key]+=1;tool_count+=len(set(p)&set(tool))
    if any(counts[s,b]!=n for s,n in [('train',4),('tune',2)] for b in ['tool','general']):raise ValueError('cohort incomplete')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'chats.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    report=dict(source_sha256=sha256(a.chats.read_bytes()).hexdigest(),counts={str(k):v for k,v in counts.items() if v},
        tokens=sum(len(r['token_ids']) for r in selected),positions=sum(len(r['positions']) for r in selected),tool_positions=tool_count,
        heldout_used=False,policy='first identities per train/tune bucket; 8 uniformly spaced assistant positions plus up to8 real tool-call positions per chat; full context retained')
    (a.output/'selection.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['chats','tokenizer','output']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
