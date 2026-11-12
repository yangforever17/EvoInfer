import torch
import triton
import triton.language as tl
from nanovllm.layers.moe_decode import _load_expert, _gate_activate
from check_int4_experts import weights

@triton.jit
def unpack(Q,S,W,N:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
    x=tl.program_id(0)*B+tl.arange(0,B)
    w=_load_expert(Q,S,0,x//K,x%K,N,K,True)
    tl.store(W+x,w,x<N*K)

def main():
    q,s,w=weights(1,1024,2048)
    y=torch.empty_like(w)
    unpack[(triton.cdiv(w.numel(),1024),)](q,s,y,1024,2048,1024)
    torch.testing.assert_close(y,w,rtol=0,atol=0)
    print('PASS canonical BF16 weight values',flush=True)
    x=torch.randn(1,2048,device='cuda',dtype=torch.bfloat16)
    ids=torch.zeros(8,device='cuda',dtype=torch.int64)
    out=torch.empty(8,512,device='cuda',dtype=torch.bfloat16)
    for quant in (False,True):
        c=_gate_activate[(8,128)](q if quant else w,x,ids,out,512,2048,8,4,2048,s if quant else None,quant,num_warps=4,enable_fp_fusion=False)
        print('QUANT',quant,flush=True)
        print('\n'.join(line for line in c.asm['ttgir'].splitlines() if line.startswith('#blocked')),flush=True)

if __name__=='__main__':main()
