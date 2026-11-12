# Complementary MoE decode merge

Episode: 11090151-codex-merge-b263. Parent: 4cea4f6cd0.
Sibling: 11090035-codex-moe-339a, commit a0c9308958665c761f1c3279ed33abf57ad84412.

The sibling's only complementary implementation change is the indirect Triton
expert GEMV for small batches (tokens * top-k <= 64). Ported its kernel and
numerical/CUDA-graph regression, and manually integrated the small-batch call.
The parent already has grouped Triton prefill GEMMs, fused DeltaNet verification,
and combined speculative-step graphs. Those implementations and draft depth
three are retained. In particular, applying the sibling model hunk verbatim
would conflict with the parent's grouped prefill dispatch.

The direct reads avoid materializing selected expert matrices. Router ordering,
shared expert operations, separate BF16 activation and route-weight rounding,
and the small-batch route reduction remain unchanged. The FP32 reduction tree
differs from cuBLAS, so numerical tests alone do not establish model correctness.

## Baseline and isolated measurements

Before implementation changes, `evo run` measured 0.40737 ms per four-token
routed MoE in a CUDA graph; expert-weight indexing consumed 55.32% of profiled
GPU kernel time. A same-process parent/candidate microbenchmark then measured
0.41333 ms versus 0.14195 ms (65.7% lower latency).

`evo run -- python experiments/validate_merge.py` ran the retained
`tests/check_moe_decode.py`: three seeds, T=1/2/4/8, independent and overlapping
routes. Maximum routed-output absolute error was 0.001953125. Graph replay was
bitwise equal to eager execution of the candidate after inputs, expert indices,
and routing weights changed.

## Full dev evaluation

Each evaluation uses 12 prompts and 256 new tokens, with the strict judge.

| Configuration | Decode tok/s | End-to-end tok/s | Mean TTFT | Violations |
| --- | ---: | ---: | ---: | ---: |
| Fresh parent | 105.50 | 95.22 | 0.27 s | 0 / 3072 |
| Merged direct expert reads | 162.77 | 139.08 | 0.27 s | 0 / 3072 |
| Merged repeat | 161.88 | 137.93 | 0.28 s | 0 / 3072 |

Baseline ID: `1109-015211-11090151-codex-merge-b263-e1-a515`.
Acceptance 0.778; tokens/step 3.334; 17 permitted near-tie flips.

Candidate ID: `1109-015750-11090151-codex-merge-b263-e2-76cd`.
Acceptance 0.777; tokens/step 3.331; 20 permitted near-tie flips.
Relative to the fresh parent, decode improves 54.28% and end-to-end improves
46.06%. Every dev request decodes faster. TTFT remains 0.27 seconds.

Repeat ID: `1109-020054-11090151-codex-merge-b263-e3-6df3`.
Acceptance 0.775; tokens/step 3.324; 20 permitted near-tie flips.
The repeat improves decode 53.44% and end-to-end 44.85% against the fresh
parent. Every dev request again decodes faster. Both merged runs pass strict
correctness: zero violations across 6144 generated tokens.

## Final decision

Keep the complementary direct-read kernel. All three available evaluations were
used: unchanged-parent baseline and two identical candidate runs. Relative to
the supplied parent, candidate decode improves 54.2–55.0% and end-to-end improves
45.5–46.7%. The small 0.01 s TTFT variation does not offset either throughput gain.
Hidden holdout was not accessed; final holdout validation belongs to EvoInfer.

All GPU work used `evo run` or `evo eval`. Syntax compilation and
`git diff --check` passed. Temporary profiling scripts were removed; the
numerical and graph-input regression is retained.

Next idea: profile the merged engine's remaining dense projections and MTP
vocabulary head before another optimization. Any activation or routing fusion
should preserve each BF16 rounding boundary and receive fresh strict validation.
