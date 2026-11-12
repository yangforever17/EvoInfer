"""Short attention normalization/rotary and gate application fusions."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _one(X, W, Out, Positions, Cos, Sin, row, H: tl.constexpr,
         D: tl.constexpr, R: tl.constexpr, STRIDE: tl.constexpr,
         WIDTH: tl.constexpr, EPS: tl.constexpr):
    lane = tl.arange(0, WIDTH)
    a = tl.full((WIDTH,), 0., tl.float32)
    b = tl.full((WIDTH,), 0., tl.float32)
    c = tl.full((WIDTH,), 0., tl.float32)
    d = tl.full((WIDTH,), 0., tl.float32)
    for base in tl.static_range(0, D, WIDTH * 4):
        v0 = tl.load(X + row * STRIDE + base + lane * 4).to(tl.float32)
        v1 = tl.load(X + row * STRIDE + base + lane * 4 + 1).to(tl.float32)
        v2 = tl.load(X + row * STRIDE + base + lane * 4 + 2).to(tl.float32)
        v3 = tl.load(X + row * STRIDE + base + lane * 4 + 3).to(tl.float32)
        a += v0 * v0
        b += v1 * v1
        c += v2 * v2
        d += v3 * v3
    total = ((a + b) + c) + d
    for level in tl.static_range(9, 5, -1):
        if WIDTH >= (1 << level):
            total = tl.sum(tl.reshape(total, (2, 1 << (level - 1))), 0)
    scale = tl.rsqrt(tl.sum(total, 0) * (1. / D) + EPS)
    col = tl.arange(0, D)
    x = tl.load(X + row * STRIDE + col).to(tl.float32)
    w = tl.load(W + col).to(tl.float32)
    normalized = ((x * scale) * (1. + w)).to(X.dtype.element_ty).to(tl.float32)
    other_col = tl.where(col < R, tl.where(col < R // 2, col + R // 2, col - R // 2), col)
    other = tl.gather(normalized, other_col, 0)
    other = tl.where(col < R // 2, -other, other)
    pos = tl.load(Positions + row // H)
    cosine = tl.load(Cos + pos * R + col, col < R, 0).to(X.dtype.element_ty).to(tl.float32)
    sine = tl.load(Sin + pos * R + col, col < R, 0).to(X.dtype.element_ty).to(tl.float32)
    left = (normalized * cosine).to(X.dtype.element_ty).to(tl.float32)
    right = (other * sine).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Out + row * D + col, tl.where(col < R, left + right, normalized))


@triton.jit
def _norm_rotary(Q, K, QW, KW, QOut, KOut, Positions, Cos, Sin,
                 T: tl.constexpr, QH: tl.constexpr, KH: tl.constexpr,
                 D: tl.constexpr, R: tl.constexpr, QS: tl.constexpr,
                 KS: tl.constexpr, QWIDTH: tl.constexpr, KWIDTH: tl.constexpr,
                 EPS: tl.constexpr):
    row = tl.program_id(0)
    if row < T * QH:
        _one(Q, QW, QOut, Positions, Cos, Sin, row, QH, D, R, QS, QWIDTH, EPS)
    else:
        _one(K, KW, KOut, Positions, Cos, Sin, row - T * QH, KH, D, R, KS, KWIDTH, EPS)


def attention_norm_rotary(positions, q, k, q_weight, k_weight, cos, sin, rotary_dim, eps):
    t, qh, d = q.shape
    kh = k.shape[1]
    assert d in (128, 256) and q.stride(-1) == k.stride(-1) == 1
    def width(rows):
        return min(d // 4, 512 // min(1 << (rows.bit_length() - 1), 16))
    qo = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ko = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    _norm_rotary[(t * (qh + kh),)](
        q, k, q_weight, k_weight, qo, ko, positions, cos, sin,
        t, qh, kh, d, rotary_dim, q.stride(1), k.stride(1),
        width(t * qh), width(t * kh), eps, num_warps=4, enable_fp_fusion=False)
    return qo, ko


@triton.jit
def _gate(O, Gate, Out, N: tl.constexpr, D: tl.constexpr, GS: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    gate = tl.load(Gate + (i // D) * GS + i % D, i < N, 0).to(tl.float32)
    sigmoid = tl.div_rn(1., 1. + libdevice.exp(-gate)).to(Out.dtype.element_ty).to(tl.float32)
    value = tl.load(O + i, i < N, 0).to(tl.float32)
    tl.store(Out + i, value * sigmoid, i < N)


def attention_gate(o, gate):
    out = torch.empty_like(o)
    _gate[(triton.cdiv(o.numel(), 256),)](o, gate, out, o.numel(), gate.shape[-1], gate.stride(-2), 256,
                                        enable_fp_fusion=False)
    return out
