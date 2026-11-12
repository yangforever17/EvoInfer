"""SiLU and product with the same intermediate rounding as two PyTorch ops."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _silu_mul(X, U, Y, D: tl.constexpr, N: tl.constexpr,
              XS: tl.constexpr, US: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + (i // D) * XS + i % D, i < N, 0).to(tl.float32)
    u = tl.load(U + (i // D) * US + i % D, i < N, 0).to(tl.float32)
    activated = tl.div_rn(x, 1. + libdevice.exp(-x)).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + i, activated * u, i < N)


def silu_mul(x, up):
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _silu_mul[(triton.cdiv(x.numel(), 256),)](
        x, up, out, x.shape[-1], x.numel(), x.stride(-2), up.stride(-2), 256,
        enable_fp_fusion=False)
    return out
