# Complementary router merge

Episode `11091327-codex-merge-f8b4`, parent `f5db2ec820`.
Sibling `11090306-codex-free-04c1` (`b8908987c447d481805748f66fd314a883ece731`).

The parent already fuses routing-weight normalization, expert activations,
shared-expert combination, DeltaNet gates, and Q/K/V packing, and skips the
singleton DeltaNet output concatenation. Its MTP shortlist and combined
speculative graph remain unchanged, with three draft tokens.

Port only the sibling's exact CUDA-order top-8 selection and normalization.
This replaces top-k selection plus the parent's normalization launch for at
most eight tokens, 256 experts, and top-8. Other shapes retain the parent
path. FP32 softmax remains in PyTorch. Threshold ties and equal-key sorting
preserve the route order used by the expert reduction.

## Component measurements

All GPU measurements use `evo run` or `evo eval`.
`tests/check_moe_router.py` passes bitwise index and FP32/BF16 weight comparisons
for ties, concentrated probabilities, random logits, and mutated graph inputs.

CUDA-graph microbenchmarks, microseconds per routing operation:

| Tokens | Parent top-k + fused normalization | Sibling router |
| --- | ---: | ---: |
| 1 | 14.16 | 5.09 |
| 4 | 19.09 | 5.35 |
| 8 | 15.58 | 5.25 |

The sibling's DeltaNet copy-removal path was measured separately against the
parent's already-fused Q/K/V pack plus recurrence. Outputs and FP32 rollback
snapshots were bitwise equal in this comparison.

| Tokens | Parent preparation + recurrence | Sibling preparation + recurrence |
| --- | ---: | ---: |
| 1 | 6.31 us | 4.48 us |
| 4 | 11.90 us | 15.19 us |
| 8 | 19.00 us | 22.45 us |

The sibling path is slower at the four-token speculative verification shape,
so it is not integrated. The parent already handles singleton output copies.

## Strict dev evaluation

Fresh unchanged parent: 237.23 decode tok/s, 190.77 end-to-end tok/s,
0.27 s mean TTFT, acceptance 0.741, 3.222 tokens/step. Correctness passed:
zero hard/soft violations, 19 near ties in 3072 tokens.
Evaluation ID: `1109-132836-11091327-codex-merge-f8b4-e1-f4fd`.

The supplied parent final reference is 242.74 decode / 193.52 end-to-end tok/s;
the fresh baseline is used to distinguish this session's timing drift.

## Paired ordinary-prompt comparison

One loaded model, separate graph caches, one reused state slot, three ordinary
prompts (compiler explanation, interval merging, water cycle), 256 output tokens,
three rounds with reversed order in the middle round. All nine completion pairs
were token-identical. Synchronization bracketed each measured step; first-token
work was separated from decode. Throughput is total tokens divided by total time.

| Variant | Decode tok/s | End-to-end tok/s |
| --- | ---: | ---: |
| Parent | 230.43 | 202.69 |
| Router | 240.51 | 210.57 |

Decode improves 4.37%; end-to-end improves 3.89%. This benchmark isolates the
router change and excludes initial graph capture from the measured warm runs.
The strict evaluator retains normal scheduler behavior.

Router-only strict dev: **249.06 decode tok/s**, **197.61 end-to-end tok/s**,
0.27 s mean TTFT, acceptance 0.749, 3.246 tokens/step. Correctness passed:
zero hard/soft violations, 20 near ties in 3072 tokens.
Evaluation ID: `1109-133211-11091327-codex-merge-f8b4-e2-fb75`.
Relative to the fresh baseline: +4.99% decode, +3.58% end-to-end.
Relative to the stored final parent: +2.60% decode, +2.11% end-to-end.

Repeat router-only strict dev: **247.78 decode tok/s**, **197.25 end-to-end tok/s**,
0.27 s mean TTFT, acceptance 0.744, 3.232 tokens/step. Correctness passed:
zero hard/soft violations, 16 near ties in 3072 tokens.
Evaluation ID: `1109-133500-11091327-codex-merge-f8b4-e3-9de9`.
Relative to fresh parent: +4.45% decode, +3.40% end-to-end.
Relative to stored final parent: +2.08% decode, +1.93% end-to-end.

## Decision and limits

Keep only the router port. The paired comparison and both strict dev runs
support a gain, with zero hard/soft violations across 6144 candidate tokens.
The stored-parent gains are below 3%, so this is not a claim of final acceptance:
EvoInfer must retime the parent and evaluate the hidden holdout. The holdout
was not accessed. All three available evaluations were used (parent, candidate,
candidate repeat). Python compilation and `git diff --check` pass.
Temporary benchmarks were removed; the router regression is retained.

Next: measure an exact CUDA-order softmax fusion with this router, preserving
its reduction order and tie handling; alternatively compose the independently
validated norm fusion and measure its marginal gain on this faster parent.
