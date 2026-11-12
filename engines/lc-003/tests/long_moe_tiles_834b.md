# Long-context MoE tile tuning (11100215-codex-prefill-834b)

Parent: `8745c8c88d`. Configuration is unchanged (max length 33792, draft depth 3).
Only `grouped_prefill` changes: for T >= 8192, use BM=64, BN=128, BK=64
instead of 32/64/32. Sorting, stable route grouping, BF16 gate/up, SiLU and
product boundaries, down projection, weighting, and ascending-expert BF16
accumulation are unchanged. Short prefill and all decode paths are unchanged.

## Measurements

All GPU work ran through `evo run`; logs are in the repository root.

- `bench_long_moe_tiles.py`: nine tile configurations, 8K/16K/32K tokens,
  both projection shapes, warmed CUDA-event timing (10 invocations). Every
  configuration matched parent output bitwise. Selected combined projection
  times: 8K 8.434 -> 3.581 ms; 16K 15.203 -> 6.290 ms;
  32K 28.758 -> 12.648 ms. Synthetic timings exclude routing/activation/combine.
- `check_long_moe_tiles.py`: full MoE outputs exact over 50,350,080 elements.
  Includes 8191/8192/8193-token dispatch boundary, non-contiguous inputs,
  partial row tiles, all routes concentrated in eight experts, and T=9 fallback.
- `pair_long_moe.py`: loaded the configured model once and changed only the
  eager grouped-prefill function; captured decode functions are identical.
  Warmed both variants, used slot 0 consistently, then A/B and B/A for code,
  math, and compiler prompts at 8192/16384/32768 tokens with 256 new tokens.
  All six pairs had identical tokens, accepted counts, and steps.
  Real activation comparison: 4,815,060,992 MoE output elements, zero mismatches.

| Local paired aggregate | Parent | Candidate |
| --- | ---: | ---: |
| Mean prefill seconds | 1.852700 | 1.452197 |
| Decode tokens/s | 148.253562 | 148.301243 |
| End-to-end tokens/s | 71.653968 | 80.714570 |

End-to-end gain: 12.64%; decode +0.03%; prefill latency -21.62%.
Per-prompt prefill savings were about 0.187/0.359/0.656 seconds and reproduced
in reversed order. These local aggregates are not the evaluator's metrics.

The parent 16K prefill profile (including request setup and draft initialization)
reported 1482 ms GPU time: grouped GEMMs 616 ms (41.58%), dense GEMMs 208 ms
(14.06%), attention 176 ms (11.86%), copies 126 ms (8.53%), DeltaNet 102 ms
(6.88%), depthwise convolution 71.5 ms (4.83%).

## Evaluation

Strict full-dev evaluation `1110-022149-11100215-codex-prefill-834b-e1-2527`:
PASS over 6144 tokens, zero hard violations, zero soft violations, 37 allowed
near-tie flips. Decode 161.17 tokens/s (+1.0% versus stored parent 159.64),
end-to-end 131.18 tokens/s (+4.9% versus stored parent 125.01), mean TTFT 1.46 s
(parent 1.78 s), MTP acceptance 0.719 and tokens/step 3.156. Used 1/10 evals.
Stored-baseline gains are distinct from the controlled local paired results.

Final hidden-holdout acceptance belongs to the evaluator. Potential next
opportunities: prefill activation/copy traffic, preserving explicit BF16
boundaries. The latest board shows sibling agents already validating long-context
attention and prefill depthwise convolution; combine independently validated
changes before repeating those optimizations.
