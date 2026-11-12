"""Short RMSNorm with the FP32 CUDA TensorIterator mean reduction order."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _norm(X, W, Gate, Out, Mean, D: tl.constexpr, XROW: tl.constexpr,
          WIDTH: tl.constexpr, GATED: tl.constexpr, EPS: tl.constexpr,
          SAVE_MEAN: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, WIDTH)
    a = tl.full((WIDTH,), 0., tl.float32)
    b = tl.full((WIDTH,), 0., tl.float32)
    c = tl.full((WIDTH,), 0., tl.float32)
    d = tl.full((WIDTH,), 0., tl.float32)
    # Four adjacent FP32 inputs per thread, four accumulators, followed by
    # their left-to-right combination (ATen CUDA vectorized mean).
    for base in tl.static_range(0, D, WIDTH * 4):
        v0 = tl.load(X + row * XROW + base + lane * 4).to(tl.float32)
        v1 = tl.load(X + row * XROW + base + lane * 4 + 1).to(tl.float32)
        v2 = tl.load(X + row * XROW + base + lane * 4 + 2).to(tl.float32)
        v3 = tl.load(X + row * XROW + base + lane * 4 + 3).to(tl.float32)
        a = a + v0 * v0
        b = b + v1 * v1
        c = c + v2 * v2
        d = d + v3 * v3
    total = ((a + b) + c) + d
    # Both shared-memory and CUDA warp reductions use descending offsets.
    # Cross-warp groups must combine BEFORE lanes, unlike a generic tl.sum.
    for level in tl.static_range(9, 5, -1):
        if WIDTH >= (1 << level):
            total = tl.sum(tl.reshape(total, (2, 1 << (level - 1))), 0)
    mean = tl.sum(total, 0) * (1.0 / D)
    if SAVE_MEAN:
        tl.store(Mean + row, mean)
    scale = tl.rsqrt(mean + EPS)
    col = tl.arange(0, D)
    x = tl.load(X + row * XROW + col).to(tl.float32)
    w = tl.load(W + col).to(tl.float32)
    normalized = x * scale
    if GATED:
        normalized = normalized.to(Out.dtype.element_ty).to(tl.float32)
        weighted = (w * normalized).to(Out.dtype.element_ty).to(tl.float32)
        gate = tl.load(Gate + row * D + col).to(tl.float32)
        gate = tl.div_rn(gate, 1.0 + libdevice.exp(-gate))
        result = weighted * gate
    else:
        result = normalized * (1.0 + w)
    tl.store(Out + row * D + col, result)


def norm_exact(x, weight, eps, gate=None, return_mean=False):
    """Fuse square, mean and application for the model's short norm widths.

    The reduction runs on a logically contiguous FP32 square, exactly as in
    squared_float(x).mean(-1). Input rows may have padding (attention Q).
    """
    dim = x.shape[-1]
    assert dim in (128, 256, 2048) and x.stride(-1) == 1
    rows = x.numel() // dim
    dim0 = dim // 4
    height = min(1 << (rows.bit_length() - 1), 16)
    width = min(dim0, 512 // height)
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    mean = torch.empty((*x.shape[:-1], 1), device=x.device, dtype=torch.float32) if return_mean else out
    _norm[(rows,)](x, weight, gate if gate is not None else x, out, mean,
                   dim, x.stride(-2), width, gate is not None, eps, return_mean,
                   num_warps=4, enable_fp_fusion=False)
    return (out, mean) if return_mean else out
