"""Validate existing exact norm against prefill expressions and time eager dispatch."""
import time
import torch
from nanovllm.layers.norm_exact import norm_exact


def reference(x, w, gate=None):
    xf = x.float()
    mean = xf.pow(2).mean(-1, keepdim=True)
    normalized = xf * torch.rsqrt(mean + 1e-6)
    if gate is None:
        out = (normalized * (1.0 + w.float())).type_as(x)
    else:
        out = ((w * normalized.to(x.dtype)) * torch.nn.functional.silu(gate.float())).to(x.dtype)
    return out, mean


def measure(fn):
    for _ in range(4):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(40):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-start)*1e6/40


def main():
    torch.set_default_device(None)
    torch.manual_seed(42)
    for t in [9, 33, 278, 1024, 4096]:
        for dim, heads, gated in [(2048, 1, False), (256, 16, False), (128, 32, True)]:
            x = torch.randn(t, heads, dim*2, device='cuda', dtype=torch.bfloat16)[..., :dim]
            w = torch.randn(dim, device='cuda', dtype=x.dtype)
            gate = torch.randn(t, heads, dim, device='cuda', dtype=x.dtype) if gated else None
            for seed in range(3):
                x.normal_(); w.normal_()
                if gate is not None:
                    gate.normal_()
                expected, ref_mean = reference(x,w,gate)
                actual, mean = norm_exact(x,w,1e-6,gate,return_mean=True)
                assert torch.equal(mean, ref_mean), (t, dim, 'mean', (mean-ref_mean).abs().max().item())
                assert torch.equal(actual, expected), (t, dim, 'output', (actual-expected).abs().max().item())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = norm_exact(x,w,1e-6,gate)
            for _ in range(3):
                x.normal_(); w.normal_()
                if gate is not None:
                    gate.normal_()
                graph.replay()
                assert torch.equal(actual, reference(x,w,gate)[0]), (t, dim, 'graph')
            times = []
            for repeat in range(2):
                fns = [('parent', lambda: reference(x,w,gate)),
                       ('candidate', lambda: norm_exact(x,w,1e-6,gate))]
                if repeat:
                    fns.reverse()
                times.append({name: measure(fn) for name, fn in fns})
            print('EXACT', t, dim, heads, gated, 'eager_us', times, flush=True)
    print('PASS', flush=True)


if __name__ == '__main__':
    main()
