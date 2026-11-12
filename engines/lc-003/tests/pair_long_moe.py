"""Long-context paired prefill/generation with real MoE checks; evo run only."""
import json
import os
import time
from collections import deque
import torch
from nanovllm import LLM, SamplingParams
import nanovllm.models.qwen3_5_moe as model
from nanovllm.layers.moe_prefill import grouped_prefill as candidate
from parent_moe_prefill_834b import grouped_prefill as parent

CHECK = False
CHECKED = 0
MISMATCH = 0
MAXDIFF = 0.0

def checked(*args):
    global CHECKED, MISMATCH, MAXDIFF
    out = candidate(*args)
    if CHECK:
        ref = parent(*args)
        CHECKED += out.numel()
        MISMATCH += (out != ref).sum().item()
        MAXDIFF = max(MAXDIFF, (out-ref).abs().max().item())
    return out

def request(llm, prompt, variant, count):
    model.grouped_prefill = parent if variant == 'parent' else checked
    stats = llm.spec_stats
    llm.add_request(prompt, SamplingParams(temperature=0,max_tokens=count,ignore_eos=True))
    torch.cuda.synchronize(); start=time.perf_counter()
    output,_=llm.step()
    torch.cuda.synchronize(); prefill=time.perf_counter()-start
    start=time.perf_counter()
    while not llm.is_finished():
        result,_=llm.step(); output.extend(result)
    torch.cuda.synchronize(); decode=time.perf_counter()-start
    tokens=output[0][1]
    return dict(variant=variant,prefill_s=prefill,decode_s=decode,total_s=prefill+decode,
                n=len(tokens),decode_n=len(tokens)-1,tokens=tokens,
                **{k:v-stats[k] for k,v in llm.spec_stats.items()})

def main():
    global CHECK
    llm=LLM(os.environ['EVO_MODEL'],**json.load(open('evo_config.json')))
    llm.scheduler.free_state_slots=deque([0])
    snippets=[
        ('def merge_intervals(intervals):\n    result = []\n    for lo, hi in sorted(intervals):\n        if result and lo <= result[-1][1]:\n            result[-1][1] = max(hi, result[-1][1])\n        else:\n            result.append([lo, hi])\n    return result\n', 'Review this source code and explain its correctness and complexity.'),
        ('A quadratic polynomial x squared minus five x plus six factors as (x-2)(x-3). Its roots are 2 and 3. The discriminant is 25-24=1.\n', 'Explain how to derive the quadratic formula and apply it to three examples.'),
        ('A compiler builds a syntax tree, checks types, lowers the program to an intermediate representation, then performs constant folding and register allocation.\n', 'Discuss the tradeoffs in optimizing a compiler for speed and memory.')]
    prompts=[]
    for length,(snippet,question) in zip([8192,16384,32768],snippets):
        base=llm.tokenizer.encode(snippet)
        end=llm.tokenizer.encode(question)
        prompts.append((base*((length//len(base))+1))[:length-len(end)]+end)
    for p in prompts:
        CHECK=True
        request(llm,p,'candidate',8)
        CHECK=False
        request(llm,p,'parent',8)
    print('REAL_CHECK',json.dumps(dict(elements=CHECKED,mismatch=MISMATCH,maxdiff=MAXDIFF)),flush=True)
    if os.environ.get('PROFILE'):
        model.grouped_prefill=parent
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            request(llm,prompts[1],'parent',1)
        print(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=25),flush=True)
    rows=[]
    for repeat in range(2):
        for i,p in enumerate(prompts):
            pair=[]
            for variant in (['parent','candidate'] if repeat==0 else ['candidate','parent']):
                row=request(llm,p,variant,256)
                row.update(repeat=repeat,prompt=i)
                rows.append(row); pair.append(row)
                print('PAIR',json.dumps({k:v for k,v in row.items() if k!='tokens'}),flush=True)
            print('EQUAL',repeat,i,pair[0]['tokens']==pair[1]['tokens'],pair[0]['accepted']==pair[1]['accepted'],flush=True)
    for v in ['parent','candidate']:
        sub=[r for r in rows if r['variant']==v]
        print('SUMMARY',v,json.dumps(dict(decode_tps=sum(r['decode_n'] for r in sub)/sum(r['decode_s'] for r in sub),
            e2e_tps=sum(r['n'] for r in sub)/sum(r['total_s'] for r in sub),prefill_s=sum(r['prefill_s'] for r in sub)/len(sub))),flush=True)

if __name__=='__main__': main()
