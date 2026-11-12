"""Exact grouped-MoE regression for the long-prefill dispatch boundary.

Run through evo run. Exercises skewed routing, empty experts, partial row tiles,
non-contiguous inputs, and the unchanged short-token fallback.
"""
import torch
from nanovllm.layers.moe_prefill import grouped_prefill
from parent_moe_prefill_834b import grouped_prefill as parent


def main():
    torch.manual_seed(834)
    torch.set_grad_enabled(False)
    e,h,i,k=256,2048,512,8
    w1=torch.randn(e,2*i,h,device='cuda',dtype=torch.bfloat16)*0.02
    w2=torch.randn(e,h,i,device='cuda',dtype=torch.bfloat16)*0.02
    checked=0
    for t,skew in [(9,False),(8191,False),(8192,False),(8193,True)]:
        x=torch.randn(t,h*2,device='cuda',dtype=torch.bfloat16)[:,::2]
        if skew:
            ti=torch.arange(k,device='cuda').expand(t,-1).contiguous()
            tw=torch.full((t,k),1/k,device='cuda',dtype=torch.bfloat16)
        else:
            logits=torch.randn(t,e,device='cuda')
            tw,ti=logits.softmax(-1).topk(k,-1)
            tw=(tw/tw.sum(-1,keepdim=True)).bfloat16()
        ref=parent(x,ti,tw,w1,w2)
        out=grouped_prefill(x,ti,tw,w1,w2)
        torch.testing.assert_close(out,ref,rtol=0,atol=0)
        checked+=out.numel()
        print('EXACT',t,'skew',skew,'elements',out.numel(),flush=True)
    print('PASS',checked,flush=True)

if __name__=='__main__': main()
