"""Run with evo run -- python tests/check_state_settle.py."""
import torch
from nanovllm.layers.state_settle import settle_state

for length in (1, 2, 4, 8):
    states = torch.randn(length, 32, 128, 128, device='cuda')
    conv = torch.randn(8192, 3 + length, device='cuda', dtype=torch.bfloat16)
    out = torch.empty(3, 32, 128, 128, device='cuda')
    cout = torch.empty(3, 8192, 3, device='cuda', dtype=torch.bfloat16)
    n = torch.ones(1, device='cuda', dtype=torch.long)
    settle_state(states, conv, n, out[1], cout[1])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        settle_state(states, conv, n, out[1], cout[1])
    for accepted in range(1, length + 1):
        states.normal_()
        conv.normal_()
        out.zero_()
        cout.zero_()
        n.fill_(accepted)
        graph.replay()
        assert torch.equal(out[1], states[accepted - 1])
        assert torch.equal(cout[1], conv[:, accepted:accepted + 3])
        assert out[0].count_nonzero() == out[2].count_nonzero() == 0
        assert cout[0].count_nonzero() == cout[2].count_nonzero() == 0
    print(f'length={length}: every prefix bitwise equal; neighboring slots intact')
