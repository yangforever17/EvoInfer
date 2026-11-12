# Complementary overhead merge

Episode `11090400-codex-merge-813d`, parent `87d4828e67`, sibling
`11090305-codex-overhead-d801` (`2bb6fe090d`).

The parent already fuses routed-expert gate/up activation, route weighting,
eight-route/shared-output reduction, and route normalization. The sibling's
`moe_decode.py` changes are therefore deliberately excluded. The parent's
prefill implementation, recurrence arithmetic, full speculative-step graphs,
and three-token draft depth are retained.

Complementary ports are short-batch rotary fusion, DeltaNet gate fusion,
raw-gate SiLU inside gated normalization, shared-expert SiLU/product fusion,
and contiguous Q/K/V packing with removal of the singleton output concatenation.
These retain explicit dtype rounding boundaries. Packing supplies the same
contiguous inputs to the unchanged recurrence; the sibling's rejected
fully strided recurrence experiment is not reused.

## Baseline evidence

The supplied final parent measured 205.35 decode tok/s and 167.40 end-to-end
tok/s on dev, with 0.29 s TTFT; holdout decode was 207.82 tok/s.
`evo history` was reviewed before changes. A baseline ordinary 400-token prompt
profile (64 generated tokens, including prefill and graph initialization for a
new state slot) measured 372.632 ms CUDA kernel time and 659.055 ms CPU time.
Dense GEMMs and expert kernels dominate; pointwise and copy kernels remain
measurable. This profile is not presented as isolated steady-state decode time.

## Strict dev evaluations

All evaluations use 12 prompts and 256 generated tokens per prompt.

| Port stage | Decode tok/s | End-to-end tok/s | TTFT | Hard/soft violations |
| --- | ---: | ---: | ---: | ---: |
| Supplied parent | 205.35 | 167.40 | 0.29 s | — |
| Rotary | 210.30 | 172.95 | 0.27 s | 0 / 0 |
| Rotary + gates, gated norm, shared activation | 216.79 | 176.34 | 0.28 s | 0 / 0 |
| Add Q/K/V packing and remove singleton cat (final) | 220.53 | 179.75 | 0.27 s | 0 / 0 |

Rotary evaluation: `1109-040241-11090400-codex-merge-813d-e1-a890`,
15 allowed near ties / 3072 tokens, acceptance 0.783, 3.349 tokens/step.
Pointwise evaluation: `1109-040645-11090400-codex-merge-813d-e2-4da3`,
19 allowed near ties / 3072 tokens, acceptance 0.771, 3.314 tokens/step.

## Isolated component checks

GPU tests run exclusively through `evo run`. Rotary, DeltaNet gates, raw-gate
normalization, shared SiLU/product, and the existing norm regression all passed
bitwise comparisons for BF16/FP16/FP32 and changed CUDA-graph inputs. SiLU/product
also passed all 65,536 BF16 bit patterns, including subnormals (NaNs compared as
NaNs).

Repeated four-token CUDA-graph timings, in microseconds per invocation:

| Component | Original | Fused |
| --- | ---: | ---: |
| DeltaNet gates | 13.57–13.64 | 3.48–3.49 |
| Gated-norm application, including gate activation | 7.75–7.88 | 3.39–3.44 |
| Shared-expert SiLU/product | 4.37–4.40 | 3.39–3.42 |

These tiny graph timings include replay submission overhead; integration timings
are the deciding evidence. Each pointwise component improves its isolated timing.

Packing checks passed for lengths 1, 2, 4, and 8, multiple head ratios and
head dimensions, and changed graph inputs. Both recurrence outputs and every
FP32 rollback snapshot were bitwise equal. Graphs containing 32 invocations
(to amortize host submission) measured:

| Packing tokens | Original Q/K/V preparation | Packed | Removed singleton cat |
| --- | ---: | ---: | ---: |
| 1 | 4.27 µs | 1.41 µs | 1.36 µs |
| 4 | 6.95 µs | 1.61–1.62 µs | 1.46 µs |

## Paired integration benchmark

One loaded model, separate graph caches per variant, one reused state slot,
three ordinary/synthetic prompts, three repetitions, 256 generated tokens,
alternating configuration order. Every completion was token-identical across
all four stages, including block-boundary crossings. Decode counts the 255
post-prefill tokens; end-to-end counts all 256. All stages use the parent's
already-fused routed-expert kernels.

| Stage | Decode tok/s | End-to-end tok/s | Mean TTFT |
| --- | ---: | ---: | ---: |
| Parent | 206.27 | 183.80 | 0.15655 s |
| Rotary | 211.63 | 188.19 | 0.15535 s |
| Add pointwise fusions | 220.45 | 195.00 | 0.15612 s |
| Add packing and remove singleton cat | 223.61 | 197.54 | 0.15555 s |

The complete port improves paired decode by 8.41% and end-to-end by 7.47%.
Packing adds 1.43% decode and 1.30% end-to-end over the pointwise stage.
Each stage consistently reduces decode time on each prompt/repetition.


## Final decision and validation

Keep all complementary ports. Final strict dev evaluation
`1109-041131-11090400-codex-merge-813d-e3-7c0a` measured **220.53 decode tok/s**
and **179.75 end-to-end tok/s**, improvements of **7.39%** and **7.38%** over
the supplied parent. Mean TTFT is 0.27 s, acceptance 0.776, and tokens/step
3.328. Correctness passed: zero hard violations, zero soft violations,
15 allowed near ties in 3072 tokens. Across all three port-stage evaluations,
there were zero hard or soft violations in 9216 generated tokens.

All three permitted evaluations were used. The final packing stage improves
both strict-dev throughput metrics over the pointwise stage; the paired
benchmark independently supports its benefit. Hidden holdout was not inspected
or measured in this episode and remains for the submission pipeline.

Python syntax compilation and `git diff --check` pass. Numerical and dynamic
CUDA-graph tests are retained in `tests/check_{rotary_short,delta_gates,
raw_gate_norm,silu_mul,delta_qkv}.py`; the existing norm tests also passed.
Temporary profiling, timing, and parent-source comparison scripts were removed.
All GPU execution used `evo run` or `evo eval`.

Next idea: fuse convolution-output SiLU into Q/K/V packing while preserving the
explicit BF16 SiLU rounding boundary and exact recurrence inputs. Validate
rollback states bitwise and use paired timing before strict evaluation.
