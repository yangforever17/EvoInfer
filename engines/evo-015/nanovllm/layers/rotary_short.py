"""Short-batch rotary application preserving separate activation-dtype products."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rotate(X, Y, Positions, Cos, Sin, H: tl.constexpr, D: tl.constexpr,
            R: tl.constexpr, T: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr,
            S2: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    t, h, d = i // (H * D), (i // D) % H, i % D
    valid = i < T * H * D
    x = tl.load(X + t * S0 + h * S1 + d * S2, valid, 0).to(tl.float32)
    other_d = tl.where(d < R // 2, d + R // 2, d - R // 2)
    other = tl.load(X + t * S0 + h * S1 + other_d * S2, valid & (d < R), 0).to(tl.float32)
    other = tl.where(d < R // 2, -other, other)
    pos = tl.load(Positions + t, valid, 0)
    c = tl.load(Cos + pos * R + d, valid & (d < R), 0).to(X.dtype.element_ty).to(tl.float32)
    s = tl.load(Sin + pos * R + d, valid & (d < R), 0).to(X.dtype.element_ty).to(tl.float32)
    a = (x * c).to(X.dtype.element_ty).to(tl.float32)
    b = (other * s).to(X.dtype.element_ty).to(tl.float32)
    y = tl.where(d < R, a + b, x)
    tl.store(Y + i, y, valid)


def rotary_short(positions, q, k, cos, sin, rotary_dim):
    outputs = []
    for x in (q, k):
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        _rotate[(triton.cdiv(x.numel(), 256),)](
            x, out, positions, cos, sin, x.shape[1], x.shape[2], rotary_dim,
            x.shape[0], *x.stride(), 256, enable_fp_fusion=False)
        outputs.append(out)
    return tuple(outputs)
