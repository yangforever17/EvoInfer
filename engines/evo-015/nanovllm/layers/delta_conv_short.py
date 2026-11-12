"""Exact short depthwise convolution, SiLU, and repeated Q/K/V packing."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _convolve(Mixed, State, W, Out, i, part: tl.constexpr,
              T: tl.constexpr, C: tl.constexpr, HK: tl.constexpr,
              HV: tl.constexpr, KD: tl.constexpr, VD: tl.constexpr,
              CK: tl.constexpr, MS: tl.constexpr):
    D: tl.constexpr = KD if part < 2 else VD
    t = i // (HV * D)
    h = (i // D) % HV
    d = i % D
    if part < 2:
        channel = part * HK * KD + (h // (HV // HK)) * KD + d
    else:
        channel = 2 * HK * KD + h * VD + d
    valid = i < T * HV * D
    acc = tl.full(i.shape, 0., tl.float32)
    for j in tl.static_range(CK):
        position = t + j
        old = tl.load(State + channel * (CK - 1) + position,
                      valid & (position < CK - 1), 0).to(tl.float32)
        new = tl.load(Mixed + (position - CK + 1) * MS + channel,
                      valid & (position >= CK - 1), 0).to(tl.float32)
        value = tl.where(position < CK - 1, old, new)
        weight = tl.load(W + channel * CK + j, valid, 0).to(tl.float32)
        acc = tl.fma(value, weight, acc)
    acc = acc.to(Out.dtype.element_ty).to(tl.float32)
    activated = tl.div_rn(acc, 1. + libdevice.exp(-acc))
    tl.store(Out + i, activated, valid)


@triton.jit
def _conv_pack(Mixed, State, W, Q, K, V, Conv,
               T: tl.constexpr, C: tl.constexpr, HK: tl.constexpr,
               HV: tl.constexpr, KD: tl.constexpr, VD: tl.constexpr,
               CK: tl.constexpr, MS: tl.constexpr, B: tl.constexpr):
    pid = tl.program_id(0)
    qb: tl.constexpr = tl.cdiv(T * HV * KD, B)
    vb: tl.constexpr = tl.cdiv(T * HV * VD, B)
    lane = tl.arange(0, B)
    if pid < qb:
        _convolve(Mixed, State, W, Q, pid * B + lane, 0, T, C, HK, HV, KD, VD, CK, MS)
    elif pid < 2 * qb:
        _convolve(Mixed, State, W, K, (pid - qb) * B + lane, 1, T, C, HK, HV, KD, VD, CK, MS)
    elif pid < 2 * qb + vb:
        _convolve(Mixed, State, W, V, (pid - 2 * qb) * B + lane, 2, T, C, HK, HV, KD, VD, CK, MS)
    else:
        i = (pid - 2 * qb - vb) * B + lane
        channel, position = i // (CK - 1 + T), i % (CK - 1 + T)
        valid = channel < C
        old = tl.load(State + channel * (CK - 1) + position,
                      valid & (position < CK - 1), 0)
        new = tl.load(Mixed + (position - CK + 1) * MS + channel,
                      valid & (position >= CK - 1), 0)
        tl.store(Conv + i, tl.where(position < CK - 1, old, new), valid)


def delta_conv_short(mixed, state, weight, key_heads, value_heads, key_dim, value_dim):
    t, c = mixed.shape
    ck = weight.shape[-1]
    q = torch.empty((t, value_heads, key_dim), device=mixed.device, dtype=mixed.dtype)
    k = torch.empty_like(q)
    v = torch.empty((t, value_heads, value_dim), device=mixed.device, dtype=mixed.dtype)
    conv = torch.empty((c, ck - 1 + t), device=mixed.device, dtype=mixed.dtype)
    blocks = 2 * triton.cdiv(q.numel(), 256) + triton.cdiv(v.numel(), 256) + triton.cdiv(conv.numel(), 256)
    _conv_pack[(blocks,)](mixed, state, weight, q, k, v, conv,
                          t, c, key_heads, value_heads, key_dim, value_dim,
                          ck, mixed.stride(0), 256, num_warps=4)
    return q, k, v, conv
