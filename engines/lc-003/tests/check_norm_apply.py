"""evo run -- python tests/check_norm_apply.py"""
import torch
import torch.nn.functional as F
from nanovllm.layers.norm_apply import norm_apply, squared_float
for dtype in (torch.bfloat16, torch.float16, torch.float32):
    for rows, dim in ((1, 2048), (4, 2048), (8, 2048), (32, 128), (128, 128), (64, 256)):
        for seed in range(3):
            torch.manual_seed(seed)
            # Noncontiguous rows match q/k normalization views.
            x = torch.randn(rows, dim * 2, device='cuda', dtype=dtype)[:, :dim]
            w = torch.randn(dim, device='cuda', dtype=dtype)
            xf = x.float()
            assert torch.equal(squared_float(x), xf.pow(2))
            scale = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
            ref = (xf * scale * (1.0 + w.float())).to(dtype)
            actual = norm_apply(x, w, xf.pow(2).mean(-1, keepdim=True), eps=1e-6)
            assert torch.equal(actual, ref), (dtype, rows, dim, 'zero rsqrt')
            gate = F.silu(torch.randn_like(x).float())
            ref = ((w * (xf * scale).to(dtype)) * gate).to(dtype)
            actual = norm_apply(x, w, xf.pow(2).mean(-1, keepdim=True), 1e-6, gate)
            assert torch.equal(actual, ref), (dtype, rows, dim, 'gated rsqrt')
    print(dtype, 'all pointwise norm outputs bitwise equal')

# A q-projection view has a gap between heads and a leading token dimension.
x = torch.randn(4, 16, 256, device='cuda', dtype=torch.bfloat16)[..., :128]
w = torch.randn(128, device='cuda', dtype=x.dtype)
mean_square = torch.empty(4, 16, 1, device='cuda')
norm_apply(x, w, mean_square, 1e-6)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    actual = norm_apply(x, w, mean_square, 1e-6)
for _ in range(5):
    x.normal_()
    w.normal_()
    xf = x.float()
    mean_square.copy_(xf.pow(2).mean(-1, keepdim=True))
    scale = torch.rsqrt(mean_square + 1e-6)
    graph.replay()
    expected = (xf * scale * (1 + w.float())).to(x.dtype)
    assert torch.equal(actual, expected)
print('Strided 3D graph inputs: bitwise equal after every update')

# Capture the complete gated path, including the unchanged PyTorch reduction.
x = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
w = torch.randn(128, device='cuda', dtype=x.dtype)
gate = torch.randn_like(x)
def fused():
    mean = squared_float(x).mean(-1, keepdim=True)
    return norm_apply(x, w, mean, 1e-6, F.silu(gate.float()))
fused()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    actual = fused()
for _ in range(5):
    x.normal_()
    gate.normal_()
    w.normal_()
    graph.replay()
    xf = x.float()
    normalized = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
    expected = ((w * normalized) * F.silu(gate.float())).to(x.dtype)
    assert torch.equal(actual, expected)
print('Complete gated norm graph: bitwise equal after every update')
