"""Exact regression against the unfused indirect MoE; run with evo run.

The parent already uses FP32 GEMV reductions rather than cuBLAS. These
fusions must preserve that parent's BF16 outputs, including graph replays.
"""
import torch
import triton
import triton.language as tl
from nanovllm.layers.moe_decode import (
    expert_gemv, moe_decode, normalize_routes, _combine_shared, _silu,
)


@triton.jit
def silu_check(X, Y, B: tl.constexpr):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + offsets).to(tl.float32)
    tl.store(Y + offsets, _silu(x))


def unfused(x, gate_up, down, indices, weights, shared, shared_gate):
    t, k = indices.shape
    gate, up = expert_gemv(gate_up, x, indices, k, True).chunk(2, -1)
    activation = torch.nn.functional.silu(gate) * up
    h = expert_gemv(down, activation, indices, k, False)
    routed = (h.view(t, k, -1) * weights.unsqueeze(-1)).sum(1)
    return routed + torch.sigmoid(shared_gate) * shared


@torch.inference_mode()
def main():
    e, h, i, k = 256, 2048, 512, 8
    for seed in range(6):
        torch.manual_seed(seed)
        logits = torch.randn(65536, e, device='cuda', dtype=torch.bfloat16) * (seed + 1)
        weights = logits.softmax(-1, dtype=torch.float32).topk(k, -1).values
        expected = weights / weights.sum(-1, keepdim=True)
        torch.testing.assert_close(normalize_routes(weights, torch.float32), expected, rtol=0, atol=0)
        torch.testing.assert_close(normalize_routes(weights, torch.bfloat16), expected.bfloat16(), rtol=0, atol=0)
    print('PASS: exact FP32 route normalization to BF16', flush=True)
    for seed in range(3):
        torch.manual_seed(seed)
        gu = torch.randn(e, 2 * i, h, device='cuda', dtype=torch.bfloat16) * .02
        down = torch.randn(e, h, i, device='cuda', dtype=torch.bfloat16) * .02
        for t in (1, 2, 4, 8):
            for overlap in (False, True):
                x = torch.randn(t, h, device='cuda', dtype=torch.bfloat16)
                indices = torch.stack([torch.randperm(e, device='cuda')[:k] for _ in range(t)])
                if overlap:
                    indices[:] = indices[0].clone()
                topw = torch.randn(t, k, device='cuda').softmax(-1)
                weights = (topw / topw.sum(-1, keepdim=True)).bfloat16()
                shared = torch.randn_like(x)
                shared_gate = torch.randn(t, 1, device='cuda', dtype=x.dtype) * 4
                args = (x, gu, down, indices, weights, shared, shared_gate)
                torch.testing.assert_close(moe_decode(*args), unfused(*args), rtol=0, atol=0)
                if seed == 0 and not overlap:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = moe_decode(x, gu, down, indices,
                                              normalize_routes(topw, x.dtype), shared, shared_gate)
                    x.normal_()
                    indices.add_(17).remainder_(e)
                    topw.copy_(torch.randn_like(topw).softmax(-1))
                    weights.copy_((topw / topw.sum(-1, keepdim=True)).bfloat16())
                    shared.normal_()
                    shared_gate.normal_()
                    graph.replay()
                    torch.testing.assert_close(captured, unfused(*args), rtol=0, atol=0)
                    del graph, captured
        print(f'seed {seed}: exact eager and graph comparisons passed', flush=True)

    # Wide exponent ranges expose differences in the eight-route sum's tree.
    for t in (1, 2, 4, 8):
        weighted = torch.randn(t, k, h, device='cuda', dtype=torch.bfloat16)
        weighted *= torch.exp2(torch.randint(-20, 20, weighted.shape, device='cuda').float()).bfloat16()
        shared = torch.randn(t, h, device='cuda', dtype=torch.bfloat16)
        gate = torch.randn(t, 1, device='cuda', dtype=torch.bfloat16)
        out = torch.empty_like(shared)
        _combine_shared[(t, triton.cdiv(h, 256))](
            weighted, shared, gate, out, h, 256, enable_fp_fusion=False)
        torch.testing.assert_close(out, weighted.sum(1) + torch.sigmoid(gate) * shared, rtol=0, atol=0)

    # Check every BF16 gate value, including sigmoid's tails and infinities.
    gate = torch.arange(65536, device='cuda', dtype=torch.int32).short().view(torch.bfloat16).view(-1, 1)
    silu = torch.empty_like(gate)
    silu_check[(256,)](gate, silu, 256)
    torch.testing.assert_close(silu, torch.nn.functional.silu(gate), rtol=0, atol=0, equal_nan=True)
    weighted = torch.zeros(65536, k, 32, device='cuda', dtype=torch.bfloat16)
    shared = torch.ones(65536, 32, device='cuda', dtype=torch.bfloat16)
    out = torch.empty_like(shared)
    _combine_shared[(65536, 1)](weighted, shared, gate, out, 32, 32, enable_fp_fusion=False)
    torch.testing.assert_close(out, torch.sigmoid(gate).expand_as(out), rtol=0, atol=0, equal_nan=True)
    print('PASS: fused reduction and exhaustive BF16 SiLU/shared sigmoid', flush=True)


if __name__ == '__main__':
    main()
