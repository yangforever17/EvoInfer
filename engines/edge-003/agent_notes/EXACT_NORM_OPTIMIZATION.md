# Exact short-batch RMSNorm fusion

The parent already captures verification, acceptance, recurrent-state settlement,
and MTP drafting in one CUDA graph. A decode-only profile of 34 steps measured
422.3 ms of GPU work; 5,168 square/mean/application triples accounted for
36.44 ms (8.63%). The CPU's synchronous output copy mostly waits for GPU work.

`nanovllm/layers/norm_exact.py` combines those three launches for widths 128,
256 and 2048. `ZeroCenteredRMSNorm.forward` and `GatedRMSNorm.forward` use it
under their existing short-input conditions; other widths and large inputs
retain the parent implementation.

Numerical details:

- Compute FP32 squares with four independent accumulators corresponding to
  ATen's four-element input vectorization, then combine them left to right.
- Match ATen's row-count-dependent block width. Fold cross-warp groups in
  descending offsets **before** the final 32-lane reduction. A generic flat
  reduction can change rounding.
- Disable floating-point contraction. Keep zero-centered weights in FP32 and
  retain both model-dtype rounding boundaries in gated normalization.
- Compute raw-gate SiLU using the parent's FP32 exponential and rounded divide.
- Read constant-stride rows directly, including attention Q views with padding.

Validation: `evo run -- python tests/check_norm_exact.py` compares both FP32
means and final outputs bitwise for BF16, FP16 and FP32, ten row counts, three
widths, ten seeds, gated/ungated and strided inputs. It also changes inputs and
weights beneath captured graphs. All checks pass.

Final staged-reduction graph microbenchmarks (microseconds per gated norm):

| Shape | Parent | Fused |
| --- | ---: | ---: |
| 64 x 256 | 6.13 | 1.74 |
| 4 x 2048 | 6.48 | 2.88 |
| 128 x 128 | 5.82 | 1.81 |

The initial exact prototype's same-loaded-model alternating comparison used
three ordinary prompts and three repetitions, with separate graph caches.
All nine completion pairs were token-identical; 3,153,920 real norm elements
were also checked bitwise. Decode improved 220.566 to 232.027 tok/s (+5.20%),
and end-to-end improved 193.980 to 202.637 tok/s (+4.46%).

Fresh parent strict dev: 244.48 decode / 194.71 end-to-end tok/s, mean TTFT
0.27 s, zero hard/soft violations in 3,072 tokens. Stored-parent measurements
were 250.26 / 198.55; paired comparisons help distinguish gains from drift.

Final staged-reduction paired comparison: 220.747 to 233.172 decode tok/s
(+5.63%), 195.066 to 204.560 end-to-end tok/s (+4.87%). All nine completion
pairs remain token-identical; 3,153,920 real norm elements remain bitwise exact.

First candidate strict dev: **264.64 decode / 209.48 end-to-end tok/s**,
mean TTFT 0.26 s, acceptance 0.759, 3.277 tokens/step. This is +5.8% decode
and +5.5% end-to-end against the stored parent. Correctness passes with zero
hard/soft violations and 14 allowed near ties in 3,072 tokens.

Launch tuning rejected one- and two-warp variants: they were slower than four
warps. Eight warps only slightly helped some 2048-wide cases and regressed
smaller widths; the implementation uses four warps consistently.

Repeat candidate strict dev: **258.60 decode / 205.78 end-to-end tok/s**,
mean TTFT 0.26 s, acceptance 0.733, 3.199 tokens/step. Stored-parent gains
are +3.3% / +3.6%; fresh-parent gains are +5.8% / +5.7%. Correctness again
passes with zero hard/soft violations and 13 allowed near ties in 3,072 tokens.
Across the two candidate dev runs, all 6,144 judged tokens pass. Three of ten
evaluation calls were used, including the fresh parent baseline. Hidden
holdout and final paired parent retiming remain the orchestrator's decision.

Next experiment: fuse residual addition into post-attention normalization,
retaining the BF16 addition boundary and the exact norm reduction order.
Dense projection packing is being investigated independently on the board.
