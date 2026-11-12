# Long-context verification attention

Parent: `8745c8c88d`, episode `11100216-codex-longdecode-7302`.

The FA2 paged variable-length path cannot split a multi-token verification query
across the KV history. With 16 query heads, 2 KV heads, and head dimension 256,
a four-token query takes roughly 381/756/1509 microseconds per layer at
8K/16K/32K cached tokens on the assigned A100.

The candidate exposes each verification row as a singleton sequence. All rows
share the same page table (a stride-zero expanded view). For T queries and final
KV length L, row i receives length L-T+i+1, exactly matching bottom-right causal
masking. This enables the existing FA2 split-KV implementation. Cache writes,
model state, full target logits, draft policy, and acceptance logic are unchanged.
Device-computed lengths follow changing positions during CUDA graph replay.
Prefill, singleton queries and multi-request batches retain the original path.

This is mathematically equivalent but **not bitwise equivalent**: FlashAttention
changes its reduction implementation. Correctness therefore requires the strict
judge, not just a small component error.

## Local evidence

- `tests/check_long_attention.py`: graph replay, 600 invocations per timing.
  Four-token attention 381/756/1509 -> 61/103/170 us at 8K/16K/32K.
  Random BF16 maximum absolute parent difference <= 0.000244141.
- `tests/check_paged_verify.py`: 557,056 output elements checked against parent
  and independent FP32 attention; query widths 2/3/4/8; lengths
  255/256/257/8191/8192/8193/16384/32768. Mutates page tables, lengths, Q/K/V
  after graph capture. All passed.
- `tests/pair_long_attention.py`: one loaded model, separate parent/candidate
  graph caches and pools, warmed slot 0, reversed-order repeats, code/math/prose
  at 8K/16K/30K, 256 generated tokens. Parent/candidate aggregate decode
  162.379/280.240 tok/s (+72.6%), end-to-end 75.762/94.849 tok/s (+25.2%).
  Every prompt improved in both orders. Outputs differ, so these are
  end-to-end comparisons, not identical-work controls. Parent steps per prompt
  87/70/85 versus candidate 83/70/86; accepted 170/186/170 versus 172/188/170.

- `tests/check_real_long_attention.py`: eager real Q/K/V comparisons on
  8K/16K/30K English, Chinese, and code prompts, 64 generated tokens each.
  8,650,752 elements checked; max absolute difference 0.0625, maximum
  normalized RMS difference 0.001658 (0.166%). All finite and within the
  regression's 1% normalized RMS threshold. This supplements, not replaces,
  the strict token judge.

## Strict evaluation

`evo eval --dev`, run `1110-022255-11100216-codex-longdecode-7302-e1-43ae`:
PASS, zero hard violations, zero soft violations, 30 allowed near-tie flips
across 6,144 tokens. Decode 273.55 tok/s (+71.4% vs stored parent 159.64),
end-to-end 184.32 tok/s (+47.4%); mean TTFT 1.82 s, MTP acceptance 0.716,
3.148 tokens/step. Per-request decode 269.90/266.78/238.48/297.47/280.19/297.93.
Only one of ten allowed judged evaluations used. Hidden-holdout acceptance is
reserved for submission; these are dev and local measurements.

Next opportunity: the existing singleton FA2 auto split heuristic leaves room
for tuning, but explicit split counts use an internal operation and also change
numerics. Preserve this simple public-API change before trying that separately.
Prefill MoE tiling is being optimized independently by another episode.
