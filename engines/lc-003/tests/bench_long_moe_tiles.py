"""Grouped prefill tile screen on realistic long-context shapes; evo run only."""
import json
import torch
import triton
from nanovllm.layers.moe_prefill import _grouped_mm

def main():
    torch.manual_seed(834)
    torch.set_grad_enabled(False)
    configs = [(32,64,32,4,3),(32,128,32,4,3),(64,64,32,4,3),
               (64,128,32,4,3),(64,128,64,4,3),(64,128,64,8,3),
               (128,128,32,8,3),(128,128,64,8,3),(32,128,64,4,3)]
    for t in [8192,16384,32768]:
        # Uneven but nonempty routes, as in real prefill expert histograms.
        ids = torch.randint(256,(t*8,),device='cuda')
        counts = torch.bincount(ids,minlength=256)
        offsets = torch.cat((counts.new_zeros(1),counts.cumsum(0)))
        for n,k in [(1024,2048),(2048,512)]:
            x = torch.randn(t*8,k,device='cuda',dtype=torch.bfloat16)
            w = torch.randn(256,n,k,device='cuda',dtype=torch.bfloat16)*0.02
            y = torch.empty(t*8,n,device='cuda',dtype=torch.bfloat16)
            ref = None
            for bm,bn,bk,nw,ns in configs:
                tiles = torch.cat((counts.new_zeros(1),((counts+bm-1)//bm).cumsum(0)))
                def run():
                    _grouped_mm[(triton.cdiv(t*8,bm)+256,triton.cdiv(n,bn))](
                        x,w,y,offsets,tiles,256,n,k,bm,bn,bk,num_warps=nw,num_stages=ns)
                run()
                if ref is None: ref = y.clone()
                mismatch = (y != ref).sum().item()
                maxdiff = (y-ref).abs().max().item()
                for _ in range(3): run()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10): run()
                end.record(); end.synchronize()
                print(json.dumps(dict(t=t,n=n,k=k,cfg=[bm,bn,bk,nw,ns],ms=start.elapsed_time(end)/10,mismatch=mismatch,maxdiff=maxdiff)),flush=True)
            del x,w,y,ref

if __name__ == '__main__': main()
