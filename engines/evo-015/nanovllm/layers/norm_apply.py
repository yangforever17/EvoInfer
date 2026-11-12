"""Fuse pointwise norm application while retaining PyTorch's reduction."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _apply(X, W, MeanSquare, Gate, Out, D: tl.constexpr, N: tl.constexpr,
           XROW: tl.constexpr, GATED: tl.constexpr, RAW_GATE: tl.constexpr,
           EPS: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    row, col = i // D, i % D
    x = tl.load(X + row * XROW + col, i < N, other=0).to(tl.float32)
    mean_square = tl.load(MeanSquare + row, i < N, other=0)
    scale = tl.rsqrt(mean_square + EPS)
    w = tl.load(W + col, i < N, other=0).to(tl.float32)
    normalized = x * scale
    if GATED:
        normalized = normalized.to(Out.dtype.element_ty).to(tl.float32)
        weighted = (w * normalized).to(Out.dtype.element_ty).to(tl.float32)
        gate = tl.load(Gate + i, i < N, other=0).to(tl.float32)
        if RAW_GATE:
            gate = tl.div_rn(gate, 1.0 + libdevice.exp(-gate))
        result = weighted * gate
    else:
        result = normalized * (1.0 + w)
    tl.store(Out + i, result, i < N)


def norm_apply(x, weight, mean_square, eps, gate=None, gate_is_raw=False):
    """Apply a PyTorch mean-square reduction, retaining every dtype boundary.

    Inputs have contiguous channels and a constant row stride (including the
    strided head views produced by q_proj). Gated normalization rounds both
    the normalized activation and its weighted product back to x.dtype.
    With gate_is_raw, compute SiLU in FP32 inside this kernel; otherwise gate
    already contains the FP32 SiLU output.
    """
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    _apply[(triton.cdiv(x.numel(), 256),)](
        x, weight, mean_square, gate if gate is not None else x, out,
        x.shape[-1], x.numel(), x.stride(-2), gate is not None, gate_is_raw, eps, 256,
        enable_fp_fusion=False)
    return out


@triton.jit
def _square(X, Out, D: tl.constexpr, N: tl.constexpr,
            XROW: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + (i // D) * XROW + i % D, i < N, other=0).to(tl.float32)
    tl.store(Out + i, x * x, i < N)


def squared_float(x):
    out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    _square[(triton.cdiv(x.numel(), 256),)](
        x, out, x.shape[-1], x.numel(), x.stride(-2), 256)
    return out
