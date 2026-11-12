"""Run with evo run -- python tests/check_delta_gates.py."""
import torch
import torch.nn.functional as F
from nanovllm.layers.delta_gates import delta_gates

def ref(a,b,log,bias): return -log.float().exp()*F.softplus(a.float()+bias), b.sigmoid()
for dtype in (torch.bfloat16,torch.float16,torch.float32):
    for seed in range(4):
        torch.manual_seed(seed)
        for t in (1,4,8,4096):
            a,b=[torch.randn(t,32,device='cuda',dtype=dtype)*8 for _ in range(2)]
            log=torch.randn(32,device='cuda')
            bias=torch.randn(32,device='cuda',dtype=dtype)
            actual=delta_gates(a,b,log,bias)
            for name,x,y in zip(('g','beta'),actual,ref(a,b,log,bias)):
                if not torch.equal(x,y):
                    print(dtype,seed,t,name,'mismatches',int((x!=y).sum()),'max',float((x-y).abs().max()),flush=True)
                torch.testing.assert_close(x,y,rtol=0,atol=0)
            if t==4:
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): actual=delta_gates(a,b,log,bias)
                a.normal_(); b.normal_(); log.normal_(); bias.normal_()
                graph.replay()
                for x,y in zip(actual,ref(a,b,log,bias)): torch.testing.assert_close(x,y,rtol=0,atol=0)
    print(dtype,'bitwise equal including graph mutations',flush=True)
