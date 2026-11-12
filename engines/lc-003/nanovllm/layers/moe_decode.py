"""Indirect expert GEMVs for small decode and verification batches.

Each program computes a few output rows for one token/expert route, avoiding
materializing selected weight matrices. Activation and weighting are fused with
the projections, retaining the unfused BF16 rounding boundaries. Inputs and
expert weights are contiguous.
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


@triton.jit
def _normalize_routes(W, Out, K: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0)
    cols = tl.arange(0, B)
    w = tl.load(W + token * K + cols, cols < K, 0)
    total = tl.sum(w, 0)
    tl.store(Out + token * K + cols, tl.div_rn(w, total), cols < K)


def normalize_routes(weights, dtype):
    """Renormalize FP32 top-k probabilities, then round to the router dtype."""
    out = torch.empty_like(weights, dtype=dtype)
    _normalize_routes[(weights.shape[0],)](
        weights, out, weights.shape[1], triton.next_power_of_2(weights.shape[1]),
        num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _silu(x):
    return tl.div_rn(x, 1.0 + libdevice.exp(-x))


@triton.jit
def _gate_activate(W, X, Ids, Y, I: tl.constexpr, D: tl.constexpr,
                   TOPK: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    route = tl.program_id(0)
    expert = tl.load(Ids + route)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    x = tl.load(X + (route // TOPK) * D + cols, cols < D, 0).to(tl.float32)
    base = W + expert * 2 * I * D + rows[:, None] * D + cols[None, :]
    mask = (rows[:, None] < I) & (cols[None, :] < D)
    wg = tl.load(base, mask, 0).to(tl.float32)
    wu = tl.load(base + I * D, mask, 0).to(tl.float32)
    # Retain the original GEMV reduction layout and both projection stores.
    gate = tl.sum(wg * x[None, :], 1).to(Y.dtype.element_ty).to(tl.float32)
    up = tl.sum(wu * x[None, :], 1).to(Y.dtype.element_ty).to(tl.float32)
    # CUDA exp/division match eager SiLU; tl.exp/approximate division can differ
    # at BF16 rounding boundaries. Round SiLU before multiplying by up.
    silu = _silu(gate).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + route * I + rows, silu * up, rows < I)


@triton.jit
def _expert_gemv(W, X, Ids, Y, N: tl.constexpr, D: tl.constexpr,
                 TOPK: tl.constexpr, REPEAT_X: tl.constexpr,
                 ROWS: tl.constexpr, COLS: tl.constexpr,
                 Routing=None, WEIGHTED: tl.constexpr = False):
    route = tl.program_id(0)
    expert = tl.load(Ids + route)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    token = route // TOPK if REPEAT_X else route
    x = tl.load(X + token * D + cols, cols < D, 0).to(tl.float32)
    weights = tl.load(W + expert * N * D + rows[:, None] * D + cols[None, :],
                      (rows[:, None] < N) & (cols[None, :] < D), 0).to(tl.float32)
    # FP32 accumulation; the store preserves the original BF16 GEMM boundary.
    result = tl.sum(weights * x[None, :], axis=1)
    if WEIGHTED:
        result = result.to(Y.dtype.element_ty).to(tl.float32)
        result = result * tl.load(Routing + route).to(tl.float32)
    tl.store(Y + route * N + rows, result, rows < N)


def expert_gemv(weights, x, indices, top_k, repeat_x):
    _, n, d = weights.shape
    out = torch.empty((indices.numel(), n), device=x.device, dtype=x.dtype)
    _expert_gemv[(indices.numel(), triton.cdiv(n, 4))](
        weights, x, indices, out, n, d, top_k, repeat_x,
        4, triton.next_power_of_2(d), num_warps=4)
    return out


@triton.jit
def _combine_shared(H, Shared, SharedGate, Out, N: tl.constexpr,
                    B: tl.constexpr, GATE_STRIDE: tl.constexpr = 1):
    token = tl.program_id(0)
    cols = tl.program_id(1) * B + tl.arange(0, B)
    base = H + token * 8 * N + cols
    # PyTorch's noncontiguous eight-element sum uses four FP32 accumulators:
    # (h0+h4), (h1+h5), (h2+h6), (h3+h7), combined left to right.
    a0 = tl.load(base, cols < N, 0).to(tl.float32) + tl.load(base + 4 * N, cols < N, 0).to(tl.float32)
    a1 = tl.load(base + N, cols < N, 0).to(tl.float32) + tl.load(base + 5 * N, cols < N, 0).to(tl.float32)
    a2 = tl.load(base + 2 * N, cols < N, 0).to(tl.float32) + tl.load(base + 6 * N, cols < N, 0).to(tl.float32)
    a3 = tl.load(base + 3 * N, cols < N, 0).to(tl.float32) + tl.load(base + 7 * N, cols < N, 0).to(tl.float32)
    routed = (((a0 + a1) + a2) + a3).to(Out.dtype.element_ty).to(tl.float32)
    gate = tl.load(SharedGate + token * GATE_STRIDE).to(tl.float32)
    sigmoid = tl.div_rn(1.0, 1.0 + libdevice.exp(-gate)).to(Out.dtype.element_ty).to(tl.float32)
    shared = tl.load(Shared + token * N + cols, cols < N, 0).to(tl.float32)
    shared = (sigmoid * shared).to(Out.dtype.element_ty).to(tl.float32)
    tl.store(Out + token * N + cols, routed + shared, cols < N)


def moe_decode(x, gate_up, down, indices, routing_weights, shared=None, shared_gate=None):
    """Small-batch experts, optionally including the gated shared output.

    The eight-route epilogue matches PyTorch's reduction order and separately
    rounds the sum, sigmoid, and shared product before the final addition.
    """
    t, k = indices.shape
    _, n, d = down.shape
    activated = torch.empty((t * k, d), device=x.device, dtype=x.dtype)
    _gate_activate[(t * k, triton.cdiv(d, 4))](
        gate_up, x, indices, activated, d, n, k, 4,
        triton.next_power_of_2(n), num_warps=4, enable_fp_fusion=False)
    h = torch.empty((t * k, n), device=x.device, dtype=x.dtype)
    _expert_gemv[(t * k, triton.cdiv(n, 4))](
        down, activated, indices, h, n, d, k, False,
        4, triton.next_power_of_2(d), routing_weights, True,
        num_warps=4, enable_fp_fusion=False)
    if shared is not None and k == 8:
        out = torch.empty_like(x)
        _combine_shared[(t, triton.cdiv(n, 256))](
            h, shared, shared_gate, out, n, 256, shared_gate.stride(0),
            enable_fp_fusion=False)
        return out
    out = h.view(t, k, -1).sum(1)
    if shared is not None:
        out = out + torch.sigmoid(shared_gate) * shared
    return out
