import torch
from torch import nn
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    _VLLM_FA = False
except ImportError:  # use the FlashAttention build that ships with vLLM
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    flash_attn_with_kvcache = None
    _VLLM_FA = True
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def paged_verify_attention(q, k_cache, v_cache, context, scale):
    """Expose each verification row as a singleton to FA2's split-KV path.

    Bottom-right causal attention for row i sees L - T + i + 1 keys.
    All rows share the original page table; no KV data is copied. Lengths
    are computed on device so captured graphs follow the growing history.
    """
    count = q.shape[0]
    cu = torch.arange(count + 1, dtype=torch.int32, device=q.device)
    lengths = context.context_lens + torch.arange(
        1 - count, 1, dtype=torch.int32, device=q.device)
    return flash_attn_varlen_func(
        q, k_cache, v_cache, max_seqlen_q=1, cu_seqlens_q=cu,
        max_seqlen_k=context.max_seqlen_k, seqused_k=lengths,
        softmax_scale=scale, causal=True,
        block_table=context.block_tables.expand(count, -1))


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel() and context.slot_mapping is not None:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if _VLLM_FA:
            if context.block_tables is None:   # warmup / no cache: plain varlen causal attention
                return flash_attn_varlen_func(q, k, v, max_seqlen_q=context.max_seqlen_q,
                                              cu_seqlens_q=context.cu_seqlens_q,
                                              max_seqlen_k=context.max_seqlen_q,
                                              cu_seqlens_k=context.cu_seqlens_q,
                                              softmax_scale=self.scale, causal=True)
            if (not context.is_prefill and 1 < q.shape[0] <= 8
                    and context.block_tables.shape[0] == 1
                    and context.max_seqlen_q == q.shape[0]
                    and context.max_seqlen_k >= 8192):
                return paged_verify_attention(q, k_cache, v_cache, context, self.scale)
            # General paged path, including multi-request batches.
            return flash_attn_varlen_func(q, k_cache, v_cache, max_seqlen_q=context.max_seqlen_q,
                                          cu_seqlens_q=context.cu_seqlens_q,
                                          max_seqlen_k=context.max_seqlen_k,
                                          seqused_k=context.context_lens,
                                          softmax_scale=self.scale, causal=True,
                                          block_table=context.block_tables)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
