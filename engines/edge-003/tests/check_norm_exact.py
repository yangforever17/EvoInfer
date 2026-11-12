"""Bitwise norm, reduction, mutable CUDA graph and graph timing regression.

Run with evo run -- python tests/check_norm_exact.py.
"""
import torch
from nanovllm.layers.norm_exact import norm_exact
from nanovllm.layers.norm_apply import norm_apply, squared_float


def reference(x, w, gate=None):
    mean = squared_float(x).mean(-1, keepdim=True)
    return norm_apply(x, w, mean, 1e-6, gate, gate_is_raw=True)


def check(x, w, gate):
    expected = reference(x, w, gate)
    actual, mean = norm_exact(x, w, 1e-6, gate, return_mean=True)
    ref_mean = x.float().square().mean(-1, keepdim=True)
    assert torch.equal(mean, ref_mean), (x.shape, x.dtype, 'mean', (mean-ref_mean).abs().max().item())
    assert torch.equal(actual, expected), (x.shape, x.dtype, gate is not None, 'output', (actual-expected).abs().max().item())


for dtype in (torch.bfloat16, torch.float16, torch.float32):
    for dim in (128, 256, 2048):
        for rows in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256):
            for seed in range(10):
                torch.manual_seed(seed)
                x = torch.randn(rows, dim*2, device='cuda', dtype=dtype)[:, :dim]
                w = torch.randn(dim, device='cuda', dtype=dtype)
                gate = torch.randn_like(x)
                check(x, w, None)
                check(x, w, gate)
    print(dtype, 'random and strided reductions/output exact', flush=True)

for shape in ((4, 16, 256), (4, 2048), (128, 128)):
    x = torch.randn(*shape, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(shape[-1], device='cuda', dtype=x.dtype)
    gate = torch.randn_like(x)
    norm_exact(x, w, 1e-6, gate)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = norm_exact(x, w, 1e-6, gate)
    for _ in range(10):
        x.normal_(); w.normal_(); gate.normal_()
        graph.replay()
        assert torch.equal(actual, reference(x, w, gate))
    for name, fn in [('parent', reference), ('fused', lambda x,w,g: norm_exact(x,w,1e-6,g))]:
        fn(x,w,gate)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(100):
                fn(x,w,gate)
        a,b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        a.record()
        for _ in range(20):
            g.replay()
        b.record(); b.synchronize()
        print(shape, name, a.elapsed_time(b)*1000/2000, 'us', flush=True)
print('PASS', flush=True)
