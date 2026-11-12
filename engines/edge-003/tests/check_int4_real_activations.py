"""Compare every eager small MoE call to canonical selected-expert BF16 GEMVs."""
import json,os
import torch
from nanovllm import LLM,SamplingParams
import nanovllm.models.qwen3_5_moe as model_module
from nanovllm.layers.moe_decode import moe_decode


def selected(q,s,ids):
    q=q[ids.flatten()]
    s=s[ids.flatten()]
    e,n,k=q.shape
    w=torch.empty(e,n,k*2,device=q.device,dtype=torch.bfloat16)
    # Bound scratch memory while retaining exact torch FP32 multiply -> BF16.
    for j in range(e):
        codes=torch.stack((q[j]&15,q[j]>>4),-1).reshape(n,k*2).int()-8
        w[j]=(codes.float()*s[j].float().repeat_interleave(128,-1)).bfloat16()
    return w


@torch.inference_mode()
def main():
    cfg=json.load(open('evo_config.json'))
    cfg['enforce_eager']=True
    llm=LLM(os.environ['EVO_MODEL'],**cfg)
    counts={'calls':0,'elements':0,'different':0,'max_abs':0.}
    def check(x,gu,down,ids,rw,shared=None,sg=None,gs=None,ds=None):
        out=moe_decode(x,gu,down,ids,rw,shared,sg,gs,ds)
        gref=selected(gu,gs,ids)
        dref=selected(down,ds,ids)
        local_ids=torch.arange(ids.numel(),device=ids.device).reshape_as(ids)
        ref=moe_decode(x,gref,dref,local_ids,rw,shared,sg)
        counts['calls']+=1; counts['elements']+=out.numel()
        counts['different']+=(out!=ref).sum().item()
        counts['max_abs']=max(counts['max_abs'],(out-ref).abs().max().item())
        torch.testing.assert_close(out,ref,rtol=.01,atol=.008)
        return out
    model_module.moe_decode=check
    for p in ('Explain why stars shine.', '计算 17 乘以 23，并解释计算步骤。', 'Write a Python binary search function.'):
        llm.generate([p],SamplingParams(temperature=0,max_tokens=16,ignore_eos=True),use_tqdm=False)
        print('REAL',counts,flush=True)
    print('PASS real target/MTP activations',counts,flush=True)

if __name__=='__main__':main()
