"""Packed MoE input projections and exact strided routing; run through evo run.

Packing changes cuBLAS's reduction scheduling, so independent projection
comparisons allow BF16 rounding differences. Graph replay and fused routing
must be bitwise exact. Strict end-to-end correctness is checked by evo eval.
"""
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from nanovllm.models.qwen3_5_moe import SparseMoE
from nanovllm.layers.moe_router import route_logits
from nanovllm.layers.moe_decode import _combine_shared


@torch.inference_mode()
def main():
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for seed in range(8):
            torch.manual_seed(seed)
            for scale in (.1, 1, 5, 20, 80):
                backing = torch.randn(4096, 1280, device='cuda', dtype=dtype)
                logits = backing[:, 1024:]
                logits.mul_(scale)
                values, indices = logits.softmax(-1, dtype=torch.float32).topk(8, -1)
                reference = (values / values.sum(-1, keepdim=True)).to(dtype), indices
                for actual, expected in zip(route_logits(logits), reference):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print(f'PASS exact strided softmax/top8: {dtype}', flush=True)

    logits = torch.zeros(4, 1280, device='cuda', dtype=torch.bfloat16)[:, 1024:]
    route_logits(logits)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = route_logits(logits)
    for step in range(20):
        logits.copy_(torch.randint(1 + step * 10, logits.shape, device='cuda').float() / (1 + step))
        graph.replay()
        values, indices = logits.softmax(-1, dtype=torch.float32).topk(8, -1)
        for actual, expected in zip(output, ((values / values.sum(-1, keepdim=True)).bfloat16(), indices)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    del graph, output

    # Scalar shared gates may be a column of the packed projection output.
    for t in (1, 2, 4, 8):
        h = torch.randn(t, 8, 2048, device='cuda', dtype=torch.bfloat16)
        shared = torch.randn(t, 2048, device='cuda', dtype=h.dtype)
        gate = torch.randn(t, 1344, device='cuda', dtype=h.dtype)[:, 1280:1281]
        out = torch.empty_like(shared)
        _combine_shared[(t, 8)](h, shared, gate, out, 2048, 256, gate.stride(0), enable_fp_fusion=False)
        torch.testing.assert_close(out, h.sum(1) + torch.sigmoid(gate) * shared, rtol=0, atol=0)

    cfg = SimpleNamespace(num_experts=256, num_experts_per_tok=8, hidden_size=2048,
                          moe_intermediate_size=512, shared_expert_intermediate_size=512)
    with torch.device('cuda'):
        module = SparseMoE(cfg).bfloat16()
    for parameter in module.parameters():
        parameter.normal_(std=.02)
    module.short_input_weight = F.pad(torch.cat((module.shared_gate_proj.weight,
        module.shared_up_proj.weight, module.gate.weight, module.shared_expert_gate.weight)),
        (0, 0, 0, 63))
    for t in (1, 2, 4, 8):
        x = torch.randn(t, 2048, device='cuda', dtype=torch.bfloat16)
        packed = F.linear(x, module.short_input_weight)[:, :1281].split((512, 512, 256, 1), -1)
        separate = (module.shared_gate_proj(x), module.shared_up_proj(x), module.gate(x), module.shared_expert_gate(x))
        for actual, expected in zip(packed, separate):
            torch.testing.assert_close(actual, expected, rtol=.008, atol=.0005)
        module(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = module(x)
        for seed in range(4):
            x.normal_()
            graph.replay()
            torch.testing.assert_close(captured, module(x), rtol=0, atol=0)
        del graph, captured
    print('PASS packed projections, strided shared gate, and changed graph inputs', flush=True)


if __name__ == '__main__':
    main()
