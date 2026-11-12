# Dynamic vocabulary shortlist for MTP drafts

Episode `11091253-codex-spec-7fde`, parent `4cbcf4e0f3`.

The full speculative-step CUDA graph previously projected each of its three
MTP hidden states against all 248,320 vocabulary rows. The parent profile
measured about 455 microseconds per eager one-token vocabulary projection.
The target verification logits already identify a useful candidate vocabulary.

`ModelRunner.run_hybrid_step_graph` now takes the top 4,096 token IDs from each
verification position, gathers those BF16 vocabulary-head rows once, and
reuses them for all three MTP projections. At the configured depth of three,
this is a fixed 16,384-row head (64 MiB), including duplicates, instead of
reading the roughly 1 GB full head for each draft. Retaining duplicates avoids
data-dependent shapes and GPU synchronization. Candidate IDs and weights are
recomputed on every graph replay. Small vocabularies use their full size.

Only draft proposals use the shortlist. The target forward, full-vocabulary
target logits/argmax, acceptance test, recurrent-state settlement, prefill,
eager fallback, and three-token draft depth are unchanged. Excluded candidates
can reduce acceptance, but every emitted token still passes the existing
exact target verifier. The gathered draft GEMM may have different rounding or
tie ordering from the full head; draft logits do not decide emitted tokens.
No cross-request output cache or prompt-dependent policy is introduced.

## Measurements

All GPU execution used `evo run` or `evo eval`. The initial profile included
prefill and graph initialization for a new state slot, so its 740.2 ms summed
CUDA time and 1.173 s CPU time are not isolated decode timings. Dense GEMMs
and routed experts dominated; recurrent-state settlement was only 0.93%.

The first paired experiment used one loaded model, independent graph caches,
one reused recurrent-state slot, 256 generated tokens per prompt, two repeats,
and reversed variant order. Every completion was identical for all variants.
Decode counts the 255 post-prefill tokens; end-to-end counts all 256 tokens.

| Candidates per target position | Compiler explanation | Merge sorted arrays | Water cycle |
| --- | ---: | ---: | ---: |
| Full head (parent), decode tok/s | 202.8 | 245.2 | 228.8 |
| 256 | 190.8 | 253.4 | 228.3 |
| 1,024 | 212.7 | 269.6 | 253.3 |
| 4,096 (selected) | 221.5 | 273.9 | 249.9 |
| 8,192 | 215.2 | 269.2 | 251.7 |

At 4,096 candidates, end-to-end throughput improved from approximately
181.5 / 214.5 / 202.1 to 196.3 / 236.4 / 218.7 tok/s. The 256-candidate
variant was rejected because acceptance loss outweighed the saved work on
the compiler prompt. Larger sets trade extra gathering/projection cost for
coverage; 4,096 performed best overall in this sweep.

Strict dev evaluation `1109-130235-11091253-codex-spec-7fde-e1-ee4a`:

| Metric | Supplied parent | Shortlist |
| --- | ---: | ---: |
| Decode tok/s | 218.57 | 235.84 (+7.9%) |
| End-to-end tok/s | 178.31 | 189.61 (+6.3%) |
| Mean TTFT | 0.27 s | 0.27 s |
| MTP acceptance | 0.767 | 0.736 |
| Tokens per step | 3.302 | 3.208 |

Correctness passed: zero hard violations, zero soft violations, and 12 allowed
near-tie flips in 3,072 generated tokens. Stored-parent timings can drift;
the paired comparisons independently support the gain. Hidden holdout remains
for the submission evaluator and was not inspected.

## Broader paired validation

A second paired run used six additional ordinary prompts, two repeats with
reversed order, one loaded model, and independent graph caches with slot 0
reused. All 24 generated completions (6 prompts x 2 variants x 2 repeats)
were token-identical to their paired parent completions. This also exercised
changing candidate lists between requests, recurrent-state reuse, and KV
block crossings. Mean decode throughput across the two repeats:

| Prompt type | Parent tok/s | Shortlist tok/s | Gain |
| --- | ---: | ---: | ---: |
| Chinese science explanation | 196.49 | 215.34 | 9.6% |
| French Python explanation | 209.76 | 231.87 | 10.5% |
| Quadratic equation and code | 246.40 | 266.79 | 8.3% |
| SQL ranking query | 205.46 | 232.27 | 13.0% |
| Long distributed-database document | 205.77 | 223.07 | 8.4% |
| Creative writing | 166.65 | 180.19 | 8.1% |

End-to-end throughput also improved in every paired measurement. Some first
parent prefills incurred extra warmup, so the stricter dev measurement and
stable paired decode timings are the main quantitative evidence.

## Final decision

Keep the 4,096-candidate version. One of ten permitted strict evaluations was
used; the remaining GPU jobs were profiling and paired `evo run` experiments.
Python syntax compilation and `git diff --check` pass. Temporary profiling and
comparison scripts were removed before submission. No engine configuration,
recurrence kernel, target arithmetic, or other agent's claimed function changed.

Next idea: profile candidate selection/gather and consider a graph-static
candidate deduplication or broader draft-conditioned candidate refresh, while
measuring both acceptance and total step latency. The small-set negative
result shows that candidate coverage matters more than head size alone.
