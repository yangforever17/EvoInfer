"""Episode-parent/candidate real activations and alternating warm generation."""
import json,os,time
import torch
from nanovllm import LLM,SamplingParams
import nanovllm.models.qwen3_5_moe as model_module
from nanovllm.layers.moe_decode import moe_decode as candidate
from int4_parent_decode import moe_decode as parent
from pair_default_device import PROMPTS

@torch.inference_mode()
def request(llm,prompt,limit=256):
    before=llm.spec_stats
    llm.add_request(prompt,SamplingParams(temperature=0,max_tokens=limit,ignore_eos=True))
    torch.cuda.synchronize();start=time.perf_counter()
    out,_=llm.step()
    torch.cuda.synchronize();mid=time.perf_counter()
    while not llm.is_finished():
        done,_=llm.step();out.extend(done)
    torch.cuda.synchronize();end=time.perf_counter()
    tokens=out[0][1]
    return dict(prefill_s=mid-start,decode_s=end-mid,total_s=end-start,n=len(tokens),decode_n=len(tokens)-1,tokens=tokens,**{k:v-before[k] for k,v in llm.spec_stats.items()})

def main():
    # Shared worker scheduling can race another model load. Do not start a
    # 27-GiB model while another process already occupies the assigned device.
    deadline = time.monotonic() + 600
    while torch.cuda.mem_get_info()[0] < 29 * 2**30:
        if time.monotonic() >= deadline:
            raise RuntimeError('Assigned GPU did not have 29 GiB free within 10 minutes')
        print('WAIT assigned GPU free GiB', torch.cuda.mem_get_info()[0] / 2**30, flush=True)
        time.sleep(10)
    llm=LLM(os.environ['EVO_MODEL'],**json.load(open('evo_config.json')))
    counts=dict(calls=0,elements=0,different=0,max_abs=0.)
    def check(*args,**kwargs):
        a=parent(*args,**kwargs);b=candidate(*args,**kwargs)
        counts['calls']+=1;counts['elements']+=a.numel();counts['different']+=(a!=b).sum().item()
        counts['max_abs']=max(counts['max_abs'],(a-b).abs().max().item())
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        return b
    runner=llm.model_runner
    runner.enforce_eager=True
    model_module.moe_decode=check
    for prompt in PROMPTS[:3]:request(llm,prompt,16)
    print('REAL',counts,flush=True)
    runner.enforce_eager=False
    caches={v:dict(graphs={},step_graphs={},graph_pool=None) for v in ('parent','candidate')}
    def switch(v):
        model_module.moe_decode=parent if v=='parent' else candidate
        for k,val in caches[v].items():setattr(runner,k,val)
    def save(v):
        for k in caches[v]:caches[v][k]=getattr(runner,k)
    for v in caches:
        switch(v)
        for p in PROMPTS:request(llm,p,8)
        save(v)
    rows=[]
    for rep in range(2):
        for i,p in enumerate(PROMPTS):
            pair=[]
            for v in (('parent','candidate') if rep==0 else ('candidate','parent')):
                switch(v);r=request(llm,p);save(v)
                r.update(variant=v,repeat=rep,prompt=i);rows.append(r);pair.append(r)
                print('ROW',json.dumps({k:v for k,v in r.items() if k!='tokens'}),flush=True)
            print('EQUAL',rep,i,pair[0]['tokens']==pair[1]['tokens'],pair[0]['accepted']==pair[1]['accepted'],flush=True)
            assert pair[0]['tokens'] == pair[1]['tokens'], (rep, i, 'tokens')
            assert pair[0]['accepted'] == pair[1]['accepted'], (rep, i, 'acceptance')
            assert pair[0]['steps'] == pair[1]['steps'], (rep, i, 'steps')
    for v in caches:
        rs=[r for r in rows if r['variant']==v]
        print('SUMMARY',v,json.dumps(dict(decode_tps=sum(r['decode_n'] for r in rs)/sum(r['decode_s'] for r in rs),e2e_tps=sum(r['n'] for r in rs)/sum(r['total_s'] for r in rs),steps=sum(r['steps'] for r in rs),accepted=sum(r['accepted'] for r in rs))),flush=True)
if __name__=='__main__':main()
