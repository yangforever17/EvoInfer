"""Grouped expert GEMMs for prefill, with the original BF16 rounding boundaries."""
import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_mm(X, W, Y, Offsets, Tiles, E: tl.constexpr,
                N: tl.constexpr, K: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    col = tl.program_id(1)
    # Find the expert owning this tile. Empty experts own no tiles.
    lo = 0
    hi = E
    while lo < hi:
        mid = (lo + hi) // 2
        end = tl.load(Tiles + mid + 1)
        if tile >= end:
            lo = mid + 1
        else:
            hi = mid
    if lo < E:
        expert = lo
        begin = tl.load(Offsets + expert)
        end = tl.load(Offsets + expert + 1)
        first_tile = tl.load(Tiles + expert)
        rows = begin + (tile - first_tile) * BM + tl.arange(0, BM)
        cols = col * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        acc = tl.full((BM, BN), 0, tl.float32)
        for base in range(tl.cdiv(K, BK)):
            k = base * BK + ks
            a = tl.load(X + rows[:, None] * K + k[None, :],
                        (rows[:, None] < end) & (k[None, :] < K), 0)
            b = tl.load(W + expert * N * K + cols[None, :] * K + k[:, None],
                        (cols[None, :] < N) & (k[:, None] < K), 0)
            acc = tl.dot(a, b, acc)
        tl.store(Y + rows[:, None] * N + cols[None, :], acc,
                 (rows[:, None] < end) & (cols[None, :] < N))


@triton.jit
def _combine(Y, Order, TopI, TopW, Out, T: tl.constexpr, H: tl.constexpr,
             TOPK: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0)
    dims = tl.program_id(1) * B + tl.arange(0, B)
    # Order is the inverse route permutation; accumulate in ascending expert order.
    routes = tl.arange(0, triton.next_power_of_2(TOPK))
    experts = tl.load(TopI + token * TOPK + routes, routes < TOPK, 2147483647)
    result = tl.full((B,), 0, tl.float32)
    for _ in range(TOPK):
        expert = tl.min(experts, 0)
        route = tl.min(tl.where(experts == expert, routes, 2147483647), 0)
        sorted_row = tl.load(Order + token * TOPK + route)
        weight = tl.load(TopW + token * TOPK + route).to(tl.float32)
        h = tl.load(Y + sorted_row * H + dims, dims < H, 0).to(tl.float32)
        weighted = (h * weight).to(Out.dtype.element_ty).to(tl.float32)
        result = (result + weighted).to(Out.dtype.element_ty).to(tl.float32)
        experts = tl.where(routes == route, 2147483647, experts)
    tl.store(Out + token * H + dims, result, dims < H)


def grouped_prefill(x, topi, topw, gate_up, down):
    t, h = x.shape
    e, n, _ = gate_up.shape
    topk = topi.shape[1]
    flat = topi.flatten()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=e)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    bm = 32
    tiles = torch.cat((counts.new_zeros(1), ((counts + bm - 1) // bm).cumsum(0)))
    routed = x[order // topk]
    gu = torch.empty((t * topk, n), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(t * topk, bm) + e, triton.cdiv(n, 64))
    _grouped_mm[grid](routed, gate_up, gu, offsets, tiles, e, n, h, bm, 64, 32)
    gate, up = gu.chunk(2, dim=-1)
    activation = torch.nn.functional.silu(gate) * up
    y = torch.empty((t * topk, h), device=x.device, dtype=x.dtype)
    _grouped_mm[(grid[0], triton.cdiv(h, 64))](
        activation, down, y, offsets, tiles, e, h, n // 2, bm, 64, 32)
    inverse = torch.empty_like(order)
    inverse.scatter_(0, order, torch.arange(order.numel(), device=x.device))
    out = torch.empty_like(x)
    _combine[(t, triton.cdiv(h, 256))](y, inverse, topi, topw, out, t, h, topk, 256,
                                     enable_fp_fusion=False)
    return out
