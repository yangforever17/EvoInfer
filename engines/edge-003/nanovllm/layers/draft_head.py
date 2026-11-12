"""Draft-only shortlisted projection without materializing selected weights."""
import torch
import triton
import triton.language as tl


@triton.jit
def _shortlist_logits(W, X, Ids, Y, N: tl.constexpr, D: tl.constexpr,
                      ROWS: tl.constexpr, COLS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    ids = tl.load(Ids + rows, rows < N, 0)
    x = tl.load(X + cols, cols < D, 0).to(tl.float32)
    w = tl.load(W + ids[:, None] * D + cols[None, :],
                (rows[:, None] < N) & (cols[None, :] < D), 0).to(tl.float32)
    y = tl.sum(w * x[None, :], 1)
    tl.store(Y + rows, y, rows < N)


def shortlist_logits(hidden, weight, candidates):
    """BF16-rounded FP32 dot products; only draft proposals consume these."""
    out = torch.empty((1, candidates.numel()), dtype=hidden.dtype, device=hidden.device)
    _shortlist_logits[(triton.cdiv(candidates.numel(), 4),)](
        weight, hidden, candidates, out, candidates.numel(), hidden.shape[-1],
        4, triton.next_power_of_2(hidden.shape[-1]), num_warps=4,
        enable_fp_fusion=False)
    return out
