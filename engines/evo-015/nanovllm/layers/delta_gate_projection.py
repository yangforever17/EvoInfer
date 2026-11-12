"""DeltaNet gate projections with the short cuBLAS split accumulation order."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from nanovllm.layers.delta_gates import delta_gates


@triton.jit
def _project_parts(X, A, B, Parts):
    split = tl.program_id(0)
    which = tl.program_id(1)
    m = tl.arange(0, 16)
    n = tl.arange(0, 32)
    k = split * 64 + tl.arange(0, 64)
    weights = A if which == 0 else B
    x = tl.load(X + m[:, None] * 2048 + k[None, :], m[:, None] < 4, 0)
    w = tl.load(weights + n[None, :] * 2048 + k[:, None])
    partial = tl.dot(x, w)
    tl.store(Parts + which * 32 * 128 + split * 128 + m[:, None] * 32 + n[None, :],
             partial, m[:, None] < 4)


@triton.jit
def _finish_gates(Parts, Log, Bias, G, Beta):
    i = tl.arange(0, 128)
    a = tl.full((128,), 0., tl.float32)
    b = tl.full((128,), 0., tl.float32)
    # cuBLAS uses 32 independent 64-element tensor-core products followed by
    # an increasing-index FP32 sum, not a tree reduction over split results.
    for split in tl.static_range(32):
        a += tl.load(Parts + split * 128 + i)
        b += tl.load(Parts + 32 * 128 + split * 128 + i)
    a = a.to(Beta.dtype.element_ty).to(tl.float32)
    b = b.to(Beta.dtype.element_ty).to(tl.float32)
    bias = tl.load(Bias + i % 32).to(tl.float32)
    log = tl.load(Log + i % 32).to(tl.float32)
    z = a + bias
    softplus = tl.where(z > 20., z, libdevice.log1p(libdevice.exp(z)))
    tl.store(G + i, -libdevice.exp(log) * softplus)
    tl.store(Beta + i, tl.div_rn(1., 1. + libdevice.exp(-b)))


def delta_gate_projection(x, a_weight, b_weight, a_log, bias):
    # The exact cuBLAS algorithm was verified for this four-token BF16 shape.
    # Other shapes retain their original projection implementation.
    if (x.shape != (4, 2048) or a_weight.shape != (32, 2048)
            or x.dtype != torch.bfloat16 or not x.is_contiguous()):
        return delta_gates(torch.nn.functional.linear(x, a_weight),
                           torch.nn.functional.linear(x, b_weight), a_log, bias)
    parts = torch.empty((2, 32, 4, 32), device=x.device, dtype=torch.float32)
    g = torch.empty((4, 32), device=x.device, dtype=torch.float32)
    beta = torch.empty((4, 32), device=x.device, dtype=x.dtype)
    _project_parts[(32, 2)](x, a_weight, b_weight, parts, num_warps=4)
    _finish_gates[(1,)](parts, a_log, bias, g, beta, num_warps=4,
                        enable_fp_fusion=False)
    return g, beta
