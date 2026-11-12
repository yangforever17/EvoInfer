"""Run with evo run -- python tests/check_raw_gate_norm.py."""
import torch
import torch.nn.functional as F
from nanovllm.layers.norm_apply import norm_apply,squared_float
for dtype in (torch.bfloat16,torch.float16,torch.float32):
    for rows,d in ((32,128),(128,128),(256,128)):
        for seed in range(8):
            torch.manual_seed(seed)
            x=torch.randn(rows,d,device='cuda',dtype=dtype)
            gate=torch.randn_like(x)*8
            weight=torch.randn(d,device='cuda',dtype=dtype)
            mean=squared_float(x).mean(-1,keepdim=True)
            expected=norm_apply(x,weight,mean,1e-6,F.silu(gate.float()))
            actual=norm_apply(x,weight,mean,1e-6,gate,gate_is_raw=True)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): actual=norm_apply(x,weight,mean,1e-6,gate,gate_is_raw=True)
        gate.normal_(); x.normal_(); weight.normal_()
        graph.replay()
        expected=norm_apply(x,weight,mean,1e-6,F.silu(gate.float()))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    print(dtype,'raw gated norm bitwise equal',flush=True)
