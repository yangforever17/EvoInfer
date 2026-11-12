"""GPU regression: evo run -- python tests/check_delta_verify.py.

Compare every output and rollback prefix with independent one-token FLA calls.
"""
import torch
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
from nanovllm.layers.delta_verify import delta_verify


for seed in range(3):
    torch.manual_seed(seed)
    for length in (1, 2, 4, 5, 6, 8):
        heads, key_dim, value_dim = 32, 128, 128
        q, k, v = [
            torch.randn(length, heads, key_dim, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        ]
        g = -torch.rand(length, heads, device="cuda")
        beta = torch.rand(length, heads, device="cuda", dtype=torch.bfloat16)
        initial = torch.randn(heads, key_dim, value_dim, device="cuda")
        state = initial[None].clone()
        outputs, states = [], []
        for token in range(length):
            output, state = fused_recurrent_gated_delta_rule(
                q[None, token:token + 1], k[None, token:token + 1],
                v[None, token:token + 1], g=g[None, token:token + 1],
                beta=beta[None, token:token + 1], initial_state=state,
                output_final_state=True, use_qk_l2norm_in_kernel=True,
            )
            outputs.append(output[0])
            states.append(state[0])
        output, snapshots = delta_verify(q, k, v, g, beta, initial)
        torch.testing.assert_close(output, torch.cat(outputs), rtol=0, atol=0)
        torch.testing.assert_close(snapshots, torch.stack(states), rtol=0, atol=0)
        print(f"seed={seed}, length={length}: bitwise equal", flush=True)

print("All outputs and rollback snapshots are bitwise equal.")
