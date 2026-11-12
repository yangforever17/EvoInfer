# Complementary sibling merge

Episode: 11061604-codex-merge-f23d. Parent: 48df782113.
Sibling: 11061409-codex-spec-d20f (eaee7c94b7).

The parent already captures target verification, GPU prefix acceptance, state
settlement, and recursive MTP in one CUDA graph. Replacing that with the sibling's
separate verification and per-prefix MTP graphs would duplicate existing work.
The sibling's dynamic state-index plumbing is likewise unnecessary: the parent's
graphs are already specialized by state slot. Neither change was ported.

The complementary kernel is `delta_verify`: one short recurrence loop writes
all verification outputs and fp32 rollback snapshots. It replaces the parent's
per-token FLA calls, state clones, and concatenation/stack, preserving the pending
state interface, convolution windows, prefill path, and full-step graph.
The source retains the sibling's FLA MIT attribution.

## Baseline

The supplied final parent reference was 101.22 decode tok/s, 49.09 end-to-end
tok/s, and 2.70 s mean TTFT on dev; holdout decode was 94.78 tok/s.

A fresh unchanged-parent dev evaluation measured 101.22 decode tok/s,
54.36 end-to-end tok/s, 2.19 s TTFT, 0.782 MTP acceptance, and 3.345 tokens/step.
It failed strict correctness with one violation (long-10, token 24, logit gap
0.625), plus 19 permitted near-tie flips in 3072 tokens. This is a pre-existing
failure in the unchanged parent, not evidence against or for the port.
Evaluation ID: 1106-160611-11061604-codex-merge-f23d-e1-f05c.

Eight profiled decode steps used 257.63 ms CUDA time. Expert-weight gathers took
64.42 ms (25.0%); FLA recurrence kernels alone took 4.71 ms (1.83%), excluding
state clones and concatenation. Direct uninstrumented decode on an ordinary
computer-execution prompt was 93.97 tok/s. GPU work used only `evo run` and
`evo eval`; EvoInfer's GPU lock serializes queued jobs.

## Kernel validation

`evo run -- python experiments/compare_recurrence.py` ran the retained
`tests/check_delta_verify.py` regression. All outputs and every fp32 rollback
snapshot were bitwise equal to independent one-token FLA calls for three seeds
and sequence lengths 1, 2, 4, 5, 6, and 8.

CUDA-graph microbenchmarks (500 replays, microseconds per recurrence):

| Verification tokens | Parent | Fused |
| --- | ---: | ---: |
| 4 | 38.06 | 9.93 |
| 5 | 48.26 | 11.46 |

This saves about 0.84 ms over 30 linear-attention layers at draft depth three,
before considering any change in graph capture overhead.

## Evaluation results

Fused recurrence, depth three, quick evaluation:
105.35 decode tok/s (+7.6% against stored quick parent), 39.26 end-to-end tok/s
(+2.9%), 2.06 s TTFT, acceptance 0.785, 3.354 tokens/step. Strict correctness
passed: zero violations and six near-tie flips in 768 tokens.
Evaluation ID: 1106-162116-11061604-codex-merge-f23d-e1-5062.
These small quick-mode gains require corroboration because of timing drift.

## Paired ordinary-prompt comparison

One loaded model, one state slot, warmed graphs, four ordinary prompts, 128 new
tokens per prompt. The script swapped only recurrence implementation and draft
depth between configurations, cleared graphs between configurations, timed actual
decode steps directly, and compared all depth-three output tokens exactly.
It did not read evaluation prompts. All depth-three outputs and acceptance
counts were identical: 159 steps, 477 drafts, 353 accepted drafts.

| Recurrence | Draft depth | Decode tok/s | End-to-end tok/s |
| --- | ---: | ---: | ---: |
| Parent | 3 | 99.11 | 66.10 |
| Fused | 3 | 103.12 | 67.82 |
| Fused | 4 | 102.85 | 64.71 |
| Fused, repeat | 3 | 103.38 | 67.91 |

Fusion improved direct decode by 4.0–4.3% and warmed end-to-end throughput by
2.6–2.7%. Depth four did not improve decode over fused depth three and reduced
end-to-end throughput by 4.6%. Its total prefill time across the four prompts
was 2.97 s versus 2.62 s for depth three. These independent measurements favor
retaining the parent's depth three; strict dev evaluation remains decisive.

Fused recurrence, depth three, full dev evaluation:
**103.92 decode tok/s**, **56.28 end-to-end tok/s**, **2.10 s mean TTFT**.
Decode improved 2.7% against both the supplied parent and fresh baseline.
End-to-end improved 14.7% against the supplied parent (49.09) and 3.5% against
the fresh baseline (54.36). Acceptance was 0.767 and tokens/step was 3.300.
Strict correctness passed: zero violations and 15 permitted near-tie flips
in 3072 tokens. Evaluation ID: 1106-163318-11061604-codex-merge-f23d-e2-1b4a.
The paired repeat supports a small genuine gain, but host timing drift makes the
fresh-baseline comparison more conservative than the stored-reference uplift.

Fused recurrence, depth four, full dev evaluation:
105.46 decode tok/s (+4.2% against parent), 57.40 end-to-end tok/s (+16.9%
against stored parent), 2.04 s TTFT, 0.708 acceptance, and 3.832 tokens/step.
Strict correctness passed: zero violations and 17 permitted near-tie flips
in 3072 tokens. Evaluation ID: 1106-164025-11061604-codex-merge-f23d-e3-c06e.

## Final selection

Keep the fused recurrence only, with the parent's **three draft tokens** and
combined CUDA graph. Depth four's full-dev gains over fused depth three were
only 1.5% decode and 2.0% end-to-end, while the paired ordinary-prompt run showed
no decode gain and a 4.6% end-to-end regression. That inconsistent evidence does
not establish a complementary benefit across workloads, so depth four was
reverted. The fusion itself improved the paired runs and both dev metrics, and
passed the strict judge at the selected depth.

Four `evo eval` invocations were used (baseline dev, fused quick, fused dev at
three drafts, fused dev at four drafts). Queued invocations received overlapping
CLI evaluation ordinals; the four unique evaluation IDs above identify them.
All temporary profiling scripts were removed; the numerical GPU regression is
retained. Python compilation and `git diff --check` passed. Hidden holdout was
not accessed; its performance and final correctness remain for the orchestrator.

Next: remove selected-expert weight gathers, which occupied 25% of profiled GPU
time. Preserve router ordering and accumulation numerics, and measure prefill
and decode separately. This episode did not attempt that independent kernel
change or replace the parent's existing graph orchestration.
