"""Pointwise DeltaNet gates with explicit PyTorch dtype boundaries."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _gates(A, B, Log, Bias, G, Beta, N: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + i, i < N, 0).to(tl.float32)
    b = tl.load(B + i, i < N, 0).to(tl.float32)
    bias = tl.load(Bias + i % H, i < N, 0).to(tl.float32)
    log = tl.load(Log + i % H, i < N, 0).to(tl.float32)
    beta = tl.div_rn(1., 1. + libdevice.exp(-b))
    z = a + bias
    softplus = tl.where(z > 20., z, libdevice.log1p(libdevice.exp(z)))
    g = -libdevice.exp(log) * softplus
    tl.store(Beta + i, beta, i < N)
    tl.store(G + i, g, i < N)


def delta_gates(a, b, a_log, bias):
    beta = torch.empty_like(b)
    g = torch.empty(a.shape, device=a.device, dtype=torch.float32)
    _gates[(triton.cdiv(a.numel(), 128),)](
        a, b, a_log, bias, g, beta, a.numel(), a.shape[-1], 128,
        enable_fp_fusion=False)
    return g, beta
