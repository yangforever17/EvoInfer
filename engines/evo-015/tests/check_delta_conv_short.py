"""Exact convolution, rollback input, and changing CUDA graph regression."""
import torch
import torch.nn.functional as F
from nanovllm.layers.delta_conv_short import delta_conv_short
from nanovllm.layers.delta_qkv import delta_qkv


def main():
    count = 0
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for t in range(1, 9):
            for hk,hv,kd,vd in ((16,32,128,128),(8,32,64,128)):
                c = 2*hk*kd+hv*vd
                mixed = torch.randn(t,c,device='cuda',dtype=dtype)
                state = torch.randn(c,3,device='cuda',dtype=dtype)
                weight = torch.randn(c,1,4,device='cuda',dtype=dtype)
                def ref():
                    conv = torch.cat([state,mixed.t()],-1)
                    out = F.silu(F.conv1d(conv[None],weight,groups=c)).squeeze(0).t()
                    return (*delta_qkv(out,hk,hv,kd,vd),conv)
                def fused():return delta_conv_short(mixed,state,weight,hk,hv,kd,vd)
                for _ in range(2):fused()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):actual=fused()
                for _ in range(10):
                    mixed.normal_();state.normal_();weight.normal_()
                    graph.replay()
                    for a,b in zip(actual,ref()):
                        torch.testing.assert_close(a,b,atol=0,rtol=0)
                        count += a.numel()
                if dtype==torch.bfloat16 and t==4 and hk==16:
                    for name,fn in [('parent',ref),('fused',fused)]:
                        g=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            for _ in range(100):fn()
                        a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        a.record()
                        for _ in range(10):g.replay()
                        b.record();b.synchronize()
                        print(name,'us',a.elapsed_time(b),flush=True)
        print(dtype,'PASS',flush=True)
    print('PASS exact elements',count,flush=True)


if __name__=='__main__':main()
