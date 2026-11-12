"""Short-batch MoE routing with PyTorch CUDA's threshold and tie ordering."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _route_topk(Probs, Weights, Indices, N: tl.constexpr, K: tl.constexpr,
                B: tl.constexpr, STRIDE: tl.constexpr = 256,
                LOGITS: tl.constexpr = False):
    row = tl.program_id(0)
    experts = tl.arange(0, B)
    remaining = tl.load(Probs + row * STRIDE + experts, experts < N, float("-inf")).to(tl.float32)
    lanes = tl.arange(0, 32)
    if LOGITS:
        maximum = tl.max(remaining, 0)
        probabilities = libdevice.exp(remaining - maximum)
        # ATen's persistent softmax: eight elements per lane, accumulated
        # sequentially, then a descending XOR warp reduction.
        total = tl.full((32,), 0., tl.float32)
        for j in tl.static_range(8):
            total = total + tl.gather(probabilities, lanes + j * 32, 0)
        for distance in tl.static_range(5):
            total = total + tl.gather(total, lanes ^ (16 >> distance), 0)
        denominator = tl.sum(tl.where(lanes == 0, total, 0.), 0)
        probabilities = tl.div_rn(probabilities, denominator)
        remaining = probabilities
    selected = tl.full((32,), 0, tl.int32)
    values = tl.full((32,), float("-inf"), tl.float32)
    for rank in range(K):
        largest = tl.max(remaining, 0)
        index = tl.min(tl.where(remaining == largest, experts, 2147483647), 0)
        selected = tl.where(lanes == rank, index, selected)
        values = tl.where(lanes == rank, largest, values)
        remaining = tl.where(experts == index, float("-inf"), remaining)

    # gatherTopK writes values above the threshold in expert-index order,
    # followed by the first threshold-equal experts, also in index order.
    threshold = tl.min(tl.where(lanes < K, values, float("inf")), 0)
    order = tl.where(lanes < K, selected + tl.where(values == threshold, N, 0),
                     2147483647)
    selected = tl.sort(order, descending=False) % N
    if LOGITS:
        values = tl.where(lanes < K, tl.gather(probabilities, selected, 0), float("-inf"))
    else:
        values = tl.load(Probs + row * STRIDE + selected, lanes < K, float("-inf"))

    # CUDA sorts a top-8 result in a padded 32-element bitonic network. Equal
    # keys swap in descending stages; a stable sort would change route order.
    for level in tl.static_range(1, 6):
        for sub in tl.static_range(level):
            distance = 1 << (level - sub - 1)
            other = lanes ^ distance
            other_value = tl.gather(values, other, 0)
            other_index = tl.gather(selected, other, 0)
            direction = (lanes & (1 << level)) != 0
            swap = tl.where((lanes & distance) == 0,
                            (values > other_value) == direction,
                            (other_value > values) == direction)
            values = tl.where(swap, other_value, values)
            selected = tl.where(swap, other_index, selected)

    denominator = tl.sum(tl.where(lanes < K, values, 0.), 0)
    tl.store(Weights + row * K + lanes, tl.div_rn(values, denominator), lanes < K)
    tl.store(Indices + row * K + lanes, selected, lanes < K)


def route_topk(probs, dtype):
    """Select top-8 of 256 FP32 probabilities and normalize to model dtype.

    Softmax stays in PyTorch. Threshold ties select the same experts and the
    padded bitonic network preserves their order in the downstream reduction.
    """
    rows, experts = probs.shape
    assert experts == 256 and probs.is_contiguous()
    weights = torch.empty((rows, 8), device=probs.device, dtype=dtype)
    indices = torch.empty((rows, 8), device=probs.device, dtype=torch.int64)
    _route_topk[(rows,)](probs, weights, indices, experts, 8, 256, num_warps=1)
    return weights, indices


def route_logits(logits):
    """Fuse FP32 softmax and top-8 routing, including strided packed logits."""
    rows, experts = logits.shape
    assert experts == 256 and logits.stride(-1) == 1
    weights = torch.empty((rows, 8), device=logits.device, dtype=logits.dtype)
    indices = torch.empty((rows, 8), device=logits.device, dtype=torch.int64)
    _route_topk[(rows,)](logits, weights, indices, experts, 8, 256,
                       logits.stride(0), True, num_warps=1,
                       enable_fp_fusion=False)
    return weights, indices
