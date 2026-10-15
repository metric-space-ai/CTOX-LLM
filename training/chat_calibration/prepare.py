#!/usr/bin/env python3
"""Select complete real conversations, retaining every input token and assistant masks.

Source/identity and rendered-conversation hashes are disjoint across splits.
The bounded screen samples output positions uniformly, never by model agreement.
"""
import argparse, collections, hashlib, json, re
from pathlib import Path


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def assistant_positions(ids, start_id, end_id, header, forced_prefix=()):
    positions=[]; active=False; i=0
    while i<len(ids):
        if ids[i]==start_id:
            active=ids[i+1:i+1+len(header)]==header
            i+=1+len(header) if active else 1
            if active and forced_prefix and ids[i:i+len(forced_prefix)]==list(forced_prefix):
                i+=len(forced_prefix)
            continue
        if active and i>0: positions.append(i-1)  # prefix position predicts assistant token
        if ids[i]==end_id: active=False
        i+=1
    return positions


def selected_positions(positions, limit):
    if not positions: raise ValueError('no assistant targets')
    if len(positions)<=limit:return positions
    return [positions[(i*len(positions))//limit] for i in range(limit)]


def normalize(r):
    messages=r.get('messages')
    if not isinstance(messages,list) or len(messages)<2: raise ValueError('missing_chat')
    messages=json.loads(json.dumps(messages)); has_calls=False
    for m in messages:
        if m.get('role') not in ['system','user','assistant','tool']:raise ValueError('role')
        content=m.get('content') or ''
        if not isinstance(content,str):content=json.dumps(content,ensure_ascii=False)
        if any(t in content for t in ['<|im_start|>','<|im_end|>']):raise ValueError('embedded_role_marker')
        if m['role']=='user' and content.strip() in ['','-','...']:raise ValueError('missing_user_prompt')
        m['content']=content
        calls=m.get('tool_calls') or []
        normalized=[]
        for c in calls:
            f=c.get('function',c);args=f.get('arguments',{})
            if isinstance(args,str):args=json.loads(args)
            if not isinstance(args,dict) or not isinstance(f.get('name'),str):raise ValueError('invalid_call')
            normalized.append({'name':f['name'],'arguments':args}); has_calls=True
        m['tool_calls']=normalized
        m['reasoning_content']=m.get('reasoning_content') or ''
    tools=r.get('tools') or []
    if has_calls and not tools:raise ValueError('missing_tool_schema')
    if tools:
        names={t.get('function',t).get('name') for t in tools}
        if any(c['name'] not in names for m in messages for c in m.get('tool_calls',[])):
            raise ValueError('undeclared_tool')
    if messages[-1]['role']!='assistant':raise ValueError('unfinished_chat')
    serialized=json.dumps([messages,tools],ensure_ascii=False)
    if any(marker in serialized for marker in ['<|im_start|>','<|im_end|>']):
        raise ValueError('embedded_role_marker')
    return messages,tools,has_calls


def run(a):
    from jinja2 import Environment,StrictUndefined
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(a.teacher_root/'tokenizer.json'))
    env=Environment(undefined=StrictUndefined)
    def fail(msg):raise ValueError(msg)
    env.globals['raise_exception']=fail
    template=env.from_string((a.teacher_root/'chat_template.jinja').read_text())
    start=tok.token_to_id('<|im_start|>');end=tok.token_to_id('<|im_end|>')
    header=tok.encode('assistant\n',add_special_tokens=False).ids
    quotas={'train':a.train//2,'tune':a.tune//2,'heldout':a.heldout//2}
    if any(n%2 for n in [a.train,a.tune,a.heldout]):raise ValueError('even split sizes required')
    counts=collections.Counter();excluded=collections.Counter();seen=set();records=[];scanned=0
    with a.corpus.open() as f:
        for line in f:
            scanned+=1
            if scanned>a.max_scan:break
            r=json.loads(line)
            try:messages,tools,calls=normalize(r)
            except (ValueError,TypeError,KeyError) as e:excluded[str(e)]+=1;continue
            identity=digest(r.get('source_repo','')+':'+str(r.get('source_id',r['id'])))
            split=['train']*6+['tune']*2+['heldout']*2
            split=split[int(identity[:8],16)%10]
            bucket='tool' if calls else 'general'
            if counts[split,bucket]>=quotas[split]:continue
            try:text=template.render(messages=messages,tools=tools,add_generation_prompt=False,enable_thinking=False)
            except Exception as e:excluded['template_error:'+str(e)[:100]]+=1;continue
            raw_text_sha256=digest(text)
            text=tok.normalizer.normalize_str(text) if tok.normalizer is not None else text
            encoded=tok.encode(text,add_special_tokens=False);ids=encoded.ids
            if len(ids)>a.context:excluded['over_context']+=1;continue
            forced=tok.encode('<think>\n\n</think>\n\n',add_special_tokens=False).ids
            p=assistant_positions(ids,start,end,header,forced)
            if len(p)<8:excluded['too_few_assistant_tokens']+=1;continue
            h=digest(text)
            # First user prompt also prevents near-identical answer variants crossing splits.
            user=digest(next(m['content'] for m in messages if m['role']=='user'))
            keys=[('source',identity),('chat',h),('user',user)]
            if any(k in seen for k in keys):excluded['duplicate']+=1;continue
            seen.update(keys);counts[split,bucket]+=1
            records.append(dict(id=r['id'],split=split,bucket=bucket,source_repo=r['source_repo'],
                source_id=r.get('source_id'),source_revision=r.get('source_revision'),
                text_sha256=h,raw_text_sha256=raw_text_sha256,user_sha256=user,token_ids=ids,
                positions=selected_positions(p,a.positions),assistant_tokens=len(p),
                messages=messages,tools=tools,rendered=text))
            if all(counts[s,b]>=n for s,n in quotas.items() for b in ['tool','general']):break
    if not all(counts[s,b]==n for s,n in quotas.items() for b in ['tool','general']):
        raise ValueError(f'insufficient admitted conversations: {counts}; exclusions={excluded}')
    a.output.mkdir(parents=True,exist_ok=False)
    with (a.output/'chats.jsonl').open('x') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    report=dict(format='ctox.complete-chat-calibration.v1',corpus=str(a.corpus),scanned=scanned,
        counts={f'{s}/{b}':n for (s,b),n in counts.items()},excluded=dict(excluded),
        context=a.context,positions_per_chat=a.positions,seed_policy='source identity SHA256 buckets 60/20/20',
        tokenizer_normalizer=str(tok.normalizer),normalized_chats=sum(r['text_sha256']!=r['raw_text_sha256'] for r in records),
        selection_policy='first eligible per bucket; complete chats only; no model-based filtering',
        tokens=sum(len(r['token_ids']) for r in records),positions=sum(len(r['positions']) for r in records),
        sha256=digest((a.output/'chats.jsonl').read_text()),
        limitations=['bounded short-chat screen, not all-length or deployment-population coverage',
            'source continuations are teacher-forced, not free-running model generations'])
    (a.output/'selection.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True);p.add_argument('--teacher-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--context',type=int,default=2048)
    p.add_argument('--train',type=int,default=32);p.add_argument('--tune',type=int,default=8);p.add_argument('--heldout',type=int,default=16)
    p.add_argument('--positions',type=int,default=16);p.add_argument('--max-scan',type=int,default=20000)
    run(p.parse_args())
