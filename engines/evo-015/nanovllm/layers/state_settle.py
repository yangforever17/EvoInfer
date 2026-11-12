"""Copy an accepted speculative prefix directly into persistent recurrent state."""
import triton
import triton.language as tl


@triton.jit
def _settle(States, Conv, N, StateOut, ConvOut,
            S: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
            CS0: tl.constexpr, CS1: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    n = tl.load(N)
    state = tl.load(States + (n - 1) * S + i, i < S, other=0)
    tl.store(StateOut + i, state, i < S)
    # Conv is [channels, kernel-1 + query_length].
    conv = tl.load(Conv + (i // K) * CS0 + (n + i % K) * CS1,
                   i < C * K, other=0)
    tl.store(ConvOut + i, conv, i < C * K)


def settle_state(states, conv, accepted, state_out, conv_out):
    """No arithmetic on model values: both destinations are exact copies."""
    size = state_out.numel()
    channels, width = conv_out.shape
    _settle[(triton.cdiv(max(size, channels * width), 1024),)](
        states, conv, accepted, state_out, conv_out,
        size, channels, width, *conv.stride(), 1024)
