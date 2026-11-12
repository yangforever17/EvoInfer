"""Alternating parent/candidate long-context generation and attention attribution."""
import json,os,time
from collections import deque
import torch
from nanovllm import LLM,SamplingParams
from nanovllm.layers.attention import Attention
from long_attention_parent import Attention as Parent

CANDIDATE=Attention.forward

def request(llm,prompt,variant,n):
    before=llm.spec_stats
    llm.add_request(prompt,SamplingParams(temperature=0,max_tokens=n,ignore_eos=True))
    torch.cuda.synchronize(); start=time.perf_counter()
    output,_=llm.step()
    torch.cuda.synchronize(); mid=time.perf_counter()
    while not llm.is_finished():
        finished,_=llm.step(); output.extend(finished)
    torch.cuda.synchronize(); end=time.perf_counter()
    tokens=output[0][1]
    return dict(variant=variant,prefill_s=mid-start,decode_s=end-mid,total_s=end-start,n=len(tokens),tokens=tokens,**{k:v-before[k] for k,v in llm.spec_stats.items()})

def main():
    llm=LLM(os.environ['EVO_MODEL'],**json.load(open('evo_config.json')))
    llm.scheduler.free_state_slots=deque([0])
    runner=llm.model_runner
    caches={v:dict(graphs={},step_graphs={},graph_pool=None) for v in ['parent','candidate']}
    def switch(v):
        Attention.forward=Parent.forward if v=='parent' else CANDIDATE
        for key,val in caches[v].items():setattr(runner,key,val)
    def save(v):
        for key in caches[v]:caches[v][key]=getattr(runner,key)
    sources=[
        ('def merge_intervals(intervals):\n    result = []\n    for start, end in sorted(intervals):\n        if result and start <= result[-1][1]:\n            result[-1][1] = max(end, result[-1][1])\n        else:\n            result.append([start, end])\n    return result\n\n','Review the interval merging algorithm above. Explain its correctness, edge cases and time complexity.'),
        ('Example: Solve x squared minus 5x plus 6 equals zero. Factor it as (x-2)(x-3), so the roots are 2 and 3. Substituting both roots verifies the result.\n','Explain how to solve x squared minus 7x plus 12 equals zero, then derive the general quadratic formula.'),
        ('Distributed systems must account for delayed messages, failed processes and partial network outages. Replication and consensus help preserve consistency but introduce coordination costs.\n','Summarize the tradeoffs between consistency, availability, and latency in distributed storage.')]
    prompts=[]
    for (body,question),length in zip(sources,[8192,16384,30000]):
        ids=llm.tokenizer.encode(body)*2000
        suffix=llm.tokenizer.encode('\n'+question)
        prompts.append(ids[:length-len(suffix)]+suffix)
    for variant in caches:
        switch(variant); request(llm,prompts[0],variant,8); save(variant)
    rows=[]
    for repeat in range(2):
      for i,prompt in enumerate(prompts):
        pair=[]
        for variant in (['parent','candidate'] if repeat==0 else ['candidate','parent']):
            switch(variant)
            row=request(llm,prompt,variant,256);save(variant)
            row.update(repeat=repeat,prompt=i);rows.append(row);pair.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='tokens'}),flush=True)
        print('PAIR',repeat,i,'tokens_equal',pair[0]['tokens']==pair[1]['tokens'],flush=True)
    for variant in caches:
        sub=[r for r in rows if r['variant']==variant]
        print('SUMMARY',variant,json.dumps(dict(decode_tps=sum(r['n']-1 for r in sub)/sum(r['decode_s'] for r in sub),e2e_tps=sum(r['n'] for r in sub)/sum(r['total_s'] for r in sub))),flush=True)
if __name__=='__main__':main()
