"""Bitwise projection/gate and changing-input CUDA graph regression."""
import torch
from nanovllm.layers.delta_gate_projection import delta_gate_projection
from nanovllm.layers.delta_gates import delta_gates


def main():
    torch.manual_seed(1729)
    count = different = 0
    max_g = max_beta = 0.
    for t in (1, 2, 4, 8):
        for h in (16, 32, 64):
            x = torch.randn(t, 2048, device='cuda', dtype=torch.bfloat16)
            a = torch.randn(h, 2048, device='cuda', dtype=torch.bfloat16) * .02
            b = torch.randn_like(a) * .02
            log = torch.randn(h, device='cuda')
            bias = torch.randn(h, device='cuda', dtype=torch.bfloat16)
            def fused():
                return delta_gate_projection(x, a, b, log, bias)
            for _ in range(2):
                fused()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = fused()
            for _ in range(500 if (t, h) == (4, 32) else 20):
                x.normal_()
                a.normal_(std=.02); b.normal_(std=.02)
                log.normal_(); bias.normal_()
                ref = delta_gates(torch.nn.functional.linear(x, a),
                                  torch.nn.functional.linear(x, b), log, bias)
                eager = fused()
                graph.replay()
                for u, v in zip(captured, eager):
                    torch.testing.assert_close(u, v, rtol=0, atol=0)
                for u, v in zip(eager, ref):
                    torch.testing.assert_close(u, v, rtol=0, atol=0)
                count += sum(v.numel() for v in ref)
                different += sum((u != v).sum().item() for u, v in zip(eager, ref))
                max_g = max(max_g, (eager[0] - ref[0]).abs().max().item())
                max_beta = max(max_beta, (eager[1] - ref[1]).abs().max().item())
    print('PASS elements', count, 'different', different, 'max_g', max_g,
          'max_beta', max_beta, flush=True)


if __name__ == '__main__':
    main()
