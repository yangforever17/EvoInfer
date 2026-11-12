"""Canonical BF16 equivalence, changing-route graphs, and component timings.
Run only through evo run. Dense references cover one layer, never the full model.
"""
import torch
import triton
from nanovllm.layers.moe_decode import moe_decode
from nanovllm.layers.moe_prefill import grouped_prefill


def weights(e, n, k):
    q = torch.randint(0, 256, (e,n,k//2), dtype=torch.uint8, device='cuda')
    s = (torch.rand(e,n,k//128, device='cuda') * .01).bfloat16()
    codes = torch.stack((q & 15, q >> 4), dim=-1).reshape(e,n,k).int()-8
    w = (codes.float() * s.float().repeat_interleave(128,-1)).bfloat16()
    return q, s, w


def time_graph(fn):
    for _ in range(3): fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): fn()
    return triton.testing.do_bench(graph.replay)


@torch.inference_mode()
def main():
    import triton.testing
    torch.manual_seed(123)
    for seed in range(3):
        torch.manual_seed(seed)
        e,h,i,k=256,2048,512,8
        gu,gs,gref=weights(e,2*i,h)
        down,ds,dref=weights(e,h,i)
        for t in (1,2,4,8,9,32,200,512):
            x=torch.randn(t,h,device='cuda',dtype=torch.bfloat16)
            ids=torch.stack([torch.randperm(e,device='cuda')[:k] for _ in range(t)])
            rw=torch.rand(t,k,device='cuda').softmax(-1).bfloat16()
            shared=torch.randn_like(x)
            sg=torch.randn(t,1,device='cuda',dtype=x.dtype)
            if t<=8:
                ref=lambda: moe_decode(x,gref,dref,ids,rw,shared,sg)
                cand=lambda: moe_decode(x,gu,down,ids,rw,shared,sg,gs,ds)
            else:
                ref=lambda: grouped_prefill(x,ids,rw,gref,dref)
                cand=lambda: grouped_prefill(x,ids,rw,gu,down,gs,ds)
            a,b=ref(),cand()
            print("DIFF eager",seed,t,(a!=b).sum().item(),(a-b).abs().max().item(),flush=True)
            torch.testing.assert_close(a,b,rtol=.01,atol=.008)
            if t<=8:
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g): out=cand()
                for _ in range(3):
                    x.normal_(); ids.copy_(torch.stack([torch.randperm(e,device='cuda')[:k] for _ in range(t)]))
                    rw.copy_(torch.rand_like(rw).softmax(-1)); g.replay()
                    torch.testing.assert_close(out,cand(),rtol=0,atol=0)
                    a=ref()
                    print("DIFF graph-ref",seed,t,(a!=out).sum().item(),(a-out).abs().max().item(),flush=True)
                    torch.testing.assert_close(out,a,rtol=.01,atol=.008)
            if seed==0 and t<=8:
                print('shape',t,'graph ms bf16/int4',time_graph(ref),time_graph(cand),flush=True)
        print('PASS seed',seed,flush=True)

if __name__=='__main__': main()
