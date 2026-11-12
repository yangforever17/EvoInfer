"""Short Gated DeltaNet recurrence with a rollback snapshot after each token.

Recurrence and launch layout follow flash-linear-attention's MIT-licensed
fused_recurrent_gated_delta_rule (Copyright 2023-2026 Songlin Yang,
Yu Zhang, Zhiyuan Li). Snapshots are written directly, without Python's
per-token launches, clones, and stack.
"""
# MIT License (flash-linear-attention)
# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import triton
import triton.language as tl
from fla.ops.utils.op import exp


@triton.jit
def _delta_verify(Q, Kp, Vp, G, Beta, H0, O, States,
                  T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                  V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                  SCALE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, BV)
    hv = pid // nv
    ik = tl.arange(0, BK)
    iv = (pid % nv) * BV + tl.arange(0, BV)
    sh = hv * K * V + ik[:, None] * V + iv[None, :]
    mask = (ik[:, None] < K) & (iv[None, :] < V)
    state = tl.load(H0 + sh, mask=mask, other=0).to(tl.float32)
    for t in range(T):
        q = tl.load(Q + (t * H + hv) * K + ik, ik < K, 0).to(tl.float32)
        k = tl.load(Kp + (t * H + hv) * K + ik, ik < K, 0).to(tl.float32)
        v = tl.load(Vp + (t * H + hv) * V + iv, iv < V, 0).to(tl.float32)
        q = q / tl.sqrt(tl.sum(q * q) + 1e-6)
        k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
        q = q * SCALE
        beta = tl.load(Beta + t * H + hv).to(tl.float32)
        g = tl.load(G + t * H + hv).to(tl.float32)
        state *= exp(g)
        v = beta * (v - tl.sum(state * k[:, None], 0))
        state += k[:, None] * v
        o = tl.sum(state * q[:, None], 0)
        tl.store(O + (t * H + hv) * V + iv, o, iv < V)
        tl.store(States + t * H * K * V + sh, state, mask)


def delta_verify(q, k, v, g, beta, initial_state):
    """Inputs [T,H,D]; return output [T,H,V] and states [T,H,K,V]."""
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    t, h, kd = q.shape
    vd = v.shape[-1]
    out = torch.empty_like(v)
    states = torch.empty((t, h, kd, vd), dtype=torch.float32, device=q.device)
    _delta_verify[(h * triton.cdiv(vd, 8),)](
        q, k, v, g, beta, initial_state, out, states,
        t, h, kd, vd, triton.next_power_of_2(kd), 8, kd ** -0.5,
        num_warps=1, num_stages=3)
    return out, states
