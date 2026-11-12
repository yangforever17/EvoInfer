"""Pack repeated DeltaNet Q/K and V without changing recurrence memory layout."""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack(X, Q, K, V, T: tl.constexpr, HK: tl.constexpr, HV: tl.constexpr,
          KD: tl.constexpr, VD: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr,
          B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    part = tl.program_id(1)
    if part < 2:
        t, h, d = i // (HV * KD), (i // KD) % HV, i % KD
        channel = part * HK * KD + (h // (HV // HK)) * KD + d
        x = tl.load(X + t * S0 + channel * S1, i < T * HV * KD, 0)
        if part == 0:
            tl.store(Q + i, x, i < T * HV * KD)
        else:
            tl.store(K + i, x, i < T * HV * KD)
    else:
        t, h, d = i // (HV * VD), (i // VD) % HV, i % VD
        channel = 2 * HK * KD + h * VD + d
        x = tl.load(X + t * S0 + channel * S1, i < T * HV * VD, 0)
        tl.store(V + i, x, i < T * HV * VD)


def delta_qkv(mixed, key_heads, value_heads, key_dim, value_dim):
    t = mixed.shape[0]
    q = torch.empty((t, value_heads, key_dim), device=mixed.device, dtype=mixed.dtype)
    k = torch.empty_like(q)
    v = torch.empty((t, value_heads, value_dim), device=mixed.device, dtype=mixed.dtype)
    _pack[(triton.cdiv(max(q.numel(), v.numel()), 256), 3)](
        mixed, q, k, v, t, key_heads, value_heads, key_dim, value_dim, *mixed.stride(), 256)
    return q, k, v
