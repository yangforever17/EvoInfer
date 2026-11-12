"""Compare packed decode against episode parent, including changing-route graphs."""
import torch
import triton.testing
from check_int4_experts import weights, time_graph
from int4_parent_decode import moe_decode as parent
from nanovllm.layers.moe_decode import moe_decode as candidate

@torch.inference_mode()
def main():
    for seed in range(3):
        torch.manual_seed(seed)
        e,h,i,k=256,2048,512,8
        gu,gs,_=weights(e,2*i,h)
        down,ds,_=weights(e,h,i)
        for t in (1,2,4,8):
            x=torch.randn(t,h,device='cuda',dtype=torch.bfloat16)
            ids=torch.stack([torch.randperm(e,device='cuda')[:k] for _ in range(t)])
            rw=torch.rand(t,k,device='cuda').softmax(-1).bfloat16()
            shared=torch.randn_like(x)
            sg=torch.randn(t,1,device='cuda',dtype=x.dtype)
            args=(x,gu,down,ids,rw,shared,sg,gs,ds)
            ref=lambda:parent(*args)
            cand=lambda:candidate(*args)
            a,b=ref(),cand()
            print('DIFF',seed,t,(a!=b).sum().item(),(a-b).abs().max().item(),flush=True)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=cand()
            for _ in range(3):
                x.normal_();ids.copy_(torch.stack([torch.randperm(e,device='cuda')[:k] for _ in range(t)]))
                g.replay()
                torch.testing.assert_close(out,cand(),rtol=0,atol=0)
                torch.testing.assert_close(out,ref(),rtol=0,atol=0)
            if seed==0:
                for order in ((ref,cand),(cand,ref)):
                    print('TIME',t,[time_graph(f)*1000 for f in order],flush=True)
    print('PASS',flush=True)
if __name__=='__main__':main()
