"""GPU numerical regression; run with evo run -- python tests/check_moe_decode.py.

The GEMV reduction tree differs from cuBLAS, so this checks bounded BF16
error, not bitwise identity. Full-model strict evaluation is also required.
"""
import torch
import torch.nn.functional as F

from nanovllm.layers.moe_decode import expert_gemv, moe_decode


def reference(x, gate_up, down, indices, weights):
    t, k = indices.shape
    repeated = x.repeat_interleave(k, dim=0).unsqueeze(-1)
    gu = torch.bmm(gate_up[indices.flatten()], repeated).squeeze(-1)
    gate, up = gu.chunk(2, dim=-1)
    act = F.silu(gate) * up
    h = torch.bmm(down[indices.flatten()], act.unsqueeze(-1)).squeeze(-1)
    return (h.view(t, k, -1) * weights.unsqueeze(-1)).sum(1), gu, act, h


@torch.inference_mode()
def main():
    device = 'cuda'
    e, h, i, k = 256, 2048, 512, 8
    max_error = 0.0
    for seed in range(3):
        torch.manual_seed(seed)
        gu = torch.randn(e, 2 * i, h, device=device, dtype=torch.bfloat16) * 0.02
        down = torch.randn(e, h, i, device=device, dtype=torch.bfloat16) * 0.02
        for t in (1, 2, 4, 8):
            for overlap in (False, True):
                x = torch.randn(t, h, device=device, dtype=torch.bfloat16)
                indices = torch.stack([torch.randperm(e, device=device)[:k] for _ in range(t)])
                if overlap:
                    indices[:] = indices[0].clone()
                weights = torch.softmax(torch.randn(t, k, device=device), -1).to(x.dtype)
                expected, gu_ref, act, h_ref = reference(x, gu, down, indices, weights)
                gu_out = expert_gemv(gu, x, indices, k, True)
                h_out = expert_gemv(down, act, indices, k, False)
                torch.testing.assert_close(gu_out, gu_ref, rtol=0.01, atol=0.0001)
                torch.testing.assert_close(h_out, h_ref, rtol=0.01, atol=0.0001)
                actual = moe_decode(x, gu, down, indices, weights)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.004)
                max_error = max(max_error, (actual - expected).abs().max().item())
                if seed == 0 and not overlap:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = moe_decode(x, gu, down, indices, weights)
                    # Replay must use updated inputs, routing, and routing weights.
                    x.normal_()
                    indices.add_(17).remainder_(e)
                    weights.copy_(weights.roll(1, dims=1))
                    graph.replay()
                    eager = moe_decode(x, gu, down, indices, weights)
                    torch.testing.assert_close(captured, eager, rtol=0, atol=0)
                    del graph, captured
        print(f'seed {seed}: passed all batch sizes and route-overlap cases', flush=True)
    print(f'PASS; maximum absolute routed-output error: {max_error}', flush=True)


if __name__ == '__main__':
    main()
