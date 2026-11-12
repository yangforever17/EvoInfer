"""GPU regression: evo run -- python tests/check_moe_router.py.

Check threshold ties, sorted route order, normalized weights, and graph updates.
"""
import torch
from nanovllm.layers.moe_router import route_topk


def reference(probs, dtype):
    values, indices = torch.topk(probs, 8, dim=-1)
    return (values / values.sum(-1, keepdim=True)).to(dtype), indices


def check(probs):
    for dtype in (torch.float32, torch.bfloat16):
        actual = route_topk(probs, dtype)
        expected = reference(probs, dtype)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


for seed in range(3):
    torch.manual_seed(seed)
    for rows in (1, 4, 8, 4096):
        for bins in (1, 2, 8, 32, 128, 512, 2048):
            logits = torch.randint(bins, (rows, 256), device="cuda").float()
            check(torch.softmax(logits / bins * 10, -1))
        for scale in (1, 5, 20, 80):
            logits = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16) * scale
            check(torch.softmax(logits, -1, dtype=torch.float32))
    print(f"seed={seed}: indices and normalized weights bitwise equal", flush=True)

probs = torch.empty(4, 256, device="cuda", dtype=torch.float32)
probs.fill_(1 / 256)
route_topk(probs, torch.bfloat16)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    actual = route_topk(probs, torch.bfloat16)
for step in range(20):
    logits = torch.randint(1 + step * 10, (4, 256), device="cuda").float()
    probs.copy_(torch.softmax(logits / (1 + step), -1))
    graph.replay()
    for a, b in zip(actual, reference(probs, torch.bfloat16)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
print("Changing CUDA graph inputs: bitwise equal", flush=True)
