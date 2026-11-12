# Short-batch MoE input projections and routing

Episode `11091415-codex-moe-bb42`, parent `31baa335ce`.

The parent launches four independent input projections in every MoE: shared
expert gate, shared expert up, router logits, and the shared expert scalar gate.
For at most eight tokens, pack their weights once after checkpoint loading and
compute them in one BF16 GEMM. Pad the output width to a multiple of 64 (1344
rows for this model); the unpadded 1281-row shape selects a slower cuBLAS kernel.
Checkpoint parameters and the larger-batch prefill computations are retained.
The packed cache costs about 215 MiB across 41 MoE modules.

Fuse FP32 softmax into the existing top-eight router. Read the packed logits
with their actual row stride. Preserve CUDA's sequential eight-element lane
sum, descending XOR warp reduction, correctly rounded division, top-k threshold
selection and equal-key sorting. This removes the separate cast/contiguous copy
and softmax launches. The existing SiLU product accepts strided inputs; extend
the shared-output epilogue to read the scalar gate with its actual row stride.

Routed expert gate/up and down GEMVs, BF16 activation and weighting boundaries,
eight-route accumulation order, speculative depth and state handling are unchanged.
Packing changes cuBLAS reduction scheduling, so the whole model is not claimed
bitwise equivalent to the parent. Strict token-level evaluation is decisive.

## Measurements

All GPU work used `evo run` or `evo eval`. The unchanged-parent profile with
400 input tokens and 64 output tokens (including prefill and graph capture)
reported 310.41 ms CUDA time. Routed expert gate/up consumed 39.67 ms / 776 calls,
down 22.80 ms / 776, router top-k 3.67 ms / 776, and softmax 2.10 ms / 817.

CUDA-graph component timings at four tokens:

| Component | Parent/separate | Packed/fused |
| --- | ---: | ---: |
| Shared gate/up + router projections | 19.31 us | 7.21 us |
| Routing from strided packed logits | 10.47 us | 5.44 us |
| Packed triple + scalar projection | 11.22 us | 7.19 us |

Unpadded scalar packing was rejected: a 1281-row GEMM took 19.16 us at T=4.
Padding to 1344 rows restores the fast kernel. A custom shared GEMV was considered
but not implemented after the packed cuBLAS measurements. Earlier history's
slower grouped expert reuse was not repeated.

Fresh strict dev results (12 prompts, 256 generated tokens each):

| Variant | Decode tok/s | End-to-end tok/s | TTFT | Hard / soft violations |
| --- | ---: | ---: | ---: | ---: |
| Unchanged parent | 252.41 | 199.40 | 0.27 s | 0 / 0 |
| Packed shared gate/up/router | 257.12 | 202.08 | 0.28 s | 0 / 0 |
| Above + fused strided routing | 260.31 | 204.92 | 0.27 s | 0 / 0 |
| Above + padded scalar gate (final) | 264.41 | 208.07 | 0.27 s | 0 / 0 |
| Final repeat | 264.24 | 208.44 | 0.26 s | 0 / 0 |

Evaluation IDs: `1109-141747-11091415-codex-moe-bb42-e1-db4b`,
`1109-142343-11091415-codex-moe-bb42-e2-6859`, and
`1109-142805-11091415-codex-moe-bb42-e3-bc99`.
Final padded variant: `1109-143237-11091415-codex-moe-bb42-e4-4db9`.
The final variant improves decode 4.75% and end-to-end 4.35% over the fresh
parent, with acceptance 0.740 and 3.220 tokens per step.

The triple-packing real-activation check compared 13,170,688 MoE output elements:
27,783 differed, maximum absolute difference 0.0078125. There were 160 differing
router logits and six differing route-index positions. These are numerical
measurements, not a substitute for the strict token judge.

Same-loaded-model alternating paired tests, three ordinary prompts and three
repeats, using separate CUDA graph caches and the same state slot:

- Triple packing: 237.50 to 244.59 decode (+2.99%), 204.52 to 208.85 end-to-end
  (+2.12%). All completion pairs differed, so this is not an identical-output
  latency control.
- Marginal padded scalar packing after fused routing: 251.44 to 254.83 decode
  (+1.35%), 219.30 to 222.04 end-to-end (+1.25%). The compiler explanation was
  token-identical in all three repeats and improved about 1.7%; the other two
  prompts differed.

- Full final candidate versus parent: 239.06 to 254.54 decode (+6.48%),
  209.26 to 221.13 end-to-end (+5.67%). The interval-merging completions were
  identical in all three repeats and improved about 6.5%; the compiler and
  water-cycle completions differed. Every measured run improved decode.

The broader six-prompt check (Chinese, French, algebra, SQL, long-context
summarization, and creative prose; two alternating repeats) measured 236.12 to
252.06 decode (+6.75%) and 203.93 to 218.71 end-to-end (+7.25%). Every decode
comparison improved, from about 1.8% on prose to 10.7% on French. All broader
completion pairs differed; these throughput results do not establish token
correctness beyond the strict dev evaluations.

## Regression coverage

`tests/check_moe_projection.py` checks exact fused routing against PyTorch on
491,520 strided rows across BF16/FP16/FP32, changing CUDA graph inputs including
threshold ties, and exact strided scalar-gate combination. It checks packed
projection differences within BF16 rounding tolerance and exact eager-versus-
graph outputs after changing all input tokens at T=1/2/4/8. Existing router and
MoE fusion regressions cover the retained probability-input API and epilogue.

The final packed-projection regression and the existing router/MoE fusion
regressions all pass, including exhaustive BF16 SiLU/shared sigmoid checks.
Hidden holdout is not accessed; final acceptance belongs to EvoInfer.

## Final validation

The unchanged final candidate repeated at **264.24 decode / 208.44 end-to-end
tok/s**, TTFT 0.26 s, acceptance 0.738 and 3.212 tokens/step. Evaluation ID:
`1109-143505-11091415-codex-moe-bb42-e5-ba5a`. This is +4.69% decode and +4.53%
end-to-end over the fresh parent. The two final evaluations have zero hard or
soft violations across 6,144 tokens (21 and 14 allowed near-tie flips). All four
candidate dev evaluations pass with zero hard/soft violations across 12,288
tokens. Five of ten evaluations were used, including the unchanged baseline.

Keep the padded four-projection GEMM, exact strided router and scalar-gate stride
support. Python compilation and `git diff --check` pass. Temporary profiling
and paired-benchmark scripts were removed. No hidden holdout data was accessed.

Next: compose the independently measured exact norm fusion, then profile the
remaining shared down-projection and routed expert memory traffic before
changing their arithmetic. Do not assume wider tiles or grouped expert reuse
will help; earlier episodes measured regressions there.
