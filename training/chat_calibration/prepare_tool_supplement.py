#!/usr/bin/env python3
"""Add tool-call training targets without changing any tuning/evaluation position."""
import argparse,json
from pathlib import Path
from prepare import assistant_positions,selected_positions
from analyze import tool_positions

def run(a):
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(a.tokenizer));start=tok.token_to_id('<|im_start|>');end=tok.token_to_id('<|im_end|>')
    enc=lambda text:tok.encode(text,add_special_tokens=False).ids
    records=[]
    for r in map(json.loads,a.chats.read_text().splitlines()):
        if r['split']!='train':continue
        eligible=set(assistant_positions(r['token_ids'],start,end,enc('assistant\n'),enc('<think>\n\n</think>\n\n')))
        spans=tool_positions(r['token_ids'],enc('<tool_call>'),enc('</tool_call>'),start)
        positions=sorted((set(i-1 for i in spans)&eligible)-set(r['positions']))
        if positions:records.append(dict(r,positions=selected_positions(positions,16),supplement='training tool spans only'))
    if not records:raise ValueError('no additional training tool positions')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'chats.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    result=dict(chats=len(records),additional_train_positions=sum(len(r['positions']) for r in records),tune_changed=False,heldout_changed=False,
        policy='up to16 uniformly spaced additional actual tool-call positions per existing training conversation; no model-based selection')
    (a.output/'selection.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['chats','tokenizer','output']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
