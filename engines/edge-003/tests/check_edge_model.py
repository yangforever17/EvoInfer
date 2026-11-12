"""Full checkpoint load, real expert equivalence, and eager/graph generation."""
import json, os, time
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.layers.moe_decode import moe_decode
from nanovllm.layers.moe_prefill import grouped_prefill


def dequant(m):
    q=m.qweight
    e,n,k=q.shape
    # One layer at a time fits alongside the compressed model.
    codes=torch.stack((q&15,q>>4),-1).reshape(e,n,k*2).to(torch.int8)-8
    return (codes.float()*m.scales.float().repeat_interleave(128,-1)).bfloat16()


@torch.inference_mode()
def main():
    start=time.time()
    llm=LLM(os.environ['EVO_MODEL'],**json.load(open('evo_config.json')))
    print('LOADED seconds',time.time()-start,'allocated GiB',torch.cuda.memory_allocated()/2**30,flush=True)
    # Check real checkpoint values for target and MTP experts, using a small subset.
    for m in (llm.model_runner.model.layers[0].mlp,llm.model_runner.model.mtp.layer.mlp):
        from types import SimpleNamespace
        gu=m.experts_gate_up; down=m.experts_down
        gref=dequant(SimpleNamespace(qweight=gu.qweight[:8],scales=gu.scales[:8]))
        dref=dequant(SimpleNamespace(qweight=down.qweight[:8],scales=down.scales[:8]))
        for t in (1,4,32):
            x=torch.randn(t,2048,device='cuda',dtype=torch.bfloat16)
            ids=torch.stack([torch.randperm(8,device='cuda') for _ in range(t)])
            w=torch.rand(t,8,device='cuda').softmax(-1).bfloat16()
            if t<=8:
                a=moe_decode(x,gref,dref,ids,w)
                b=moe_decode(x,gu.qweight,down.qweight,ids,w,gate_up_scales=gu.scales,down_scales=down.scales)
            else:
                a=grouped_prefill(x,ids,w,gref,dref)
                b=grouped_prefill(x,ids,w,gu.qweight,down.qweight,gu.scales,down.scales)
            print("REAL expert diffs",t,(a!=b).sum().item(),(a-b).abs().max().item(),flush=True)
            torch.testing.assert_close(a,b,rtol=.02,atol=.02)
        del gref,dref
    print('PASS real checkpoint target/MTP expert numerical checks',flush=True)
    tok=llm.tokenizer
    prompts=['Explain why the sky is blue.', 'Write a Python function to merge sorted lists.', '用中文解释什么是机器学习。']
    prompts += [tok.encode('The river flows past the old town. '*80)[:n] for n in (255,256,257)]
    for idx,prompt in enumerate(prompts):
        outputs=[]
        for eager in (True,False,False):
            llm.model_runner.enforce_eager=eager
            before=dict(llm.spec_stats)
            out=llm.generate([prompt],SamplingParams(temperature=0,max_tokens=32,ignore_eos=True),use_tqdm=False)[0]
            outputs.append(out['token_ids'])
            print('GEN',idx,'eager',eager,'stats',{k:v-before[k] for k,v in llm.spec_stats.items()},'text',out['text'][:120],flush=True)
        assert outputs[0]==outputs[1]==outputs[2], ('graph mismatch',idx)
    print('PASS eager/graph/replay generations including KV boundaries',flush=True)
    # LLM owns cleanup through its registered atexit callback.

if __name__=='__main__': main()
