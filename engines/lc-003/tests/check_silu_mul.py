"""Run with evo run -- python tests/check_silu_mul.py."""
import torch
import torch.nn.functional as F
from nanovllm.layers.silu_mul import silu_mul
for dtype in (torch.bfloat16,torch.float16,torch.float32):
    for t in (1,4,32,64):
        torch.manual_seed(t)
        both=torch.randn(t,1024,device='cuda',dtype=dtype)*8
        x,up=both.chunk(2,-1)
        for _ in range(4):
            torch.testing.assert_close(silu_mul(x,up),F.silu(x)*up,rtol=0,atol=0)
            both.normal_()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): actual=silu_mul(x,up)
        both.normal_(); graph.replay()
        torch.testing.assert_close(actual,F.silu(x)*up,rtol=0,atol=0)
    print(dtype,'SiLU product bitwise equal',flush=True)
# Every BF16 bit pattern, including subnormal values and both signs.
x=torch.arange(65536,dtype=torch.int32,device='cuda').to(torch.int16).view(torch.bfloat16).reshape(-1,512)
up=torch.randn_like(x)
torch.testing.assert_close(silu_mul(x,up),F.silu(x)*up,rtol=0,atol=0,equal_nan=True)
print('All BF16 bit patterns pass',flush=True)
