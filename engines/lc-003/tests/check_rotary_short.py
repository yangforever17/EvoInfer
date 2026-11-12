"""Run with evo run -- python tests/check_rotary_short.py."""
import torch
from nanovllm.layers.rotary_short import rotary_short


def reference(positions, q, k, cos, sin, r):
    c = cos[positions].to(q.dtype).unsqueeze(1)
    s = sin[positions].to(q.dtype).unsqueeze(1)
    outputs = []
    for x in (q, k):
        a, b = x[..., :r].chunk(2, -1)
        rotated = x[..., :r] * c + torch.cat((-b, a), -1) * s
        outputs.append(torch.cat((rotated, x[..., r:]), -1))
    return outputs


for dtype in (torch.bfloat16, torch.float16, torch.float32):
    for t in (1, 2, 4, 8):
        for r in (32, 64, 128):
            torch.manual_seed(t + r)
            q = torch.randn(t, 16, 256, device='cuda', dtype=dtype)[..., :128]
            k = torch.randn(t, 2, 128, device='cuda', dtype=dtype)
            positions = torch.randint(0, 8192, (t,), device='cuda')
            angles = torch.randn(8192, r, device='cuda')
            cos, sin = angles.cos(), angles.sin()
            for _ in range(3):
                expected = reference(positions, q, k, cos, sin, r)
                actual = rotary_short(positions, q, k, cos, sin, r)
                for a, b in zip(actual, expected): torch.testing.assert_close(a,b,atol=0,rtol=0)
                q.mul_(100); k.mul_(100)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): actual = rotary_short(positions,q,k,cos,sin,r)
            q.normal_(); k.normal_(); positions.random_(0,8192)
            graph.replay()
            for a,b in zip(actual,reference(positions,q,k,cos,sin,r)):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
    print(dtype, 'bitwise equal including strided inputs and graph mutations',flush=True)
