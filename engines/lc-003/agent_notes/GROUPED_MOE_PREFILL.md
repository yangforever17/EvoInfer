# Grouped expert GEMMs for prefill

Episode `11090036-codex-prefill-e559`, parent `f9426c8f96`.

The parent already groups routes by expert, but still launches gate/up GEMM,
SiLU, multiplication, down GEMM, weighting, and index-add separately for each
active expert. A synthetic 1,024-token request profile recorded 51,553 kernel
launches, 14,962 matrix multiplications and 6,633 index-add calls. Self CPU time
was 1.592 s versus 415 ms self CUDA time. The profile includes speculative graph
initialization and is diagnostic, not a steady-state throughput measurement.

`nanovllm/layers/moe_prefill.py` replaces this loop with two grouped Triton
matrix multiplications. Routes are stably sorted. GPU prefix sums assign row
tiles to experts, including empty experts, and a bounded launch grid avoids
reading route counts back to Python. SiLU and its product remain separate PyTorch
operations with BF16 intermediates. A final kernel weights expert outputs,
rounds to BF16, and accumulates in ascending expert order with BF16 rounding
after each addition. The router, shared expert, small-token decode path,
DeltaNet, attention, and speculative settings are unchanged.

GEMMs accumulate in FP32, but their reduction order can differ from cuBLAS.
Therefore this is not a bitwise-equivalent replacement; full-model strict
correctness evaluation is required. Temporary storage scales with the number
of routes and hidden width, including sorted inputs and expert outputs.

## Isolated measurement

All 256 experts, top-8 routing, model dimensions 2048/512, random BF16 inputs and
weights, ten synchronized iterations after warmup on the assigned A100:

| Prompt tokens | Parent loop | Grouped GEMMs |
| --- | ---: | ---: |
| 256 | 29.28 ms | 2.10 ms |
| 1,024 | 30.18 ms | 2.95 ms |
| 4,096 | 28.08 ms | 5.99 ms |
| 6,000 | 28.66 ms | 7.85 ms |

Output RMS differences were 5.3e-5–6.9e-5; maximum absolute differences were
0.00195–0.00391. The independent regression test in
`tests/check_moe_prefill.py` additionally checks exact expert-order accumulation
and sparse/skewed routing with empty experts.

## Full dev evaluation

Evaluation `1109-004333-11090036-codex-prefill-e559-e1-141d` (12 prompts,
256 generated tokens per prompt):

- Strict correctness PASS: zero violations, 14 permitted near-tie flips / 3,072 tokens.
- Mean TTFT: 0.28 s versus stored parent 1.20 s (about 77% lower).
- End-to-end throughput: 94.93 versus 71.25 tok/s (+33.2%).
- Decode throughput: 105.28 versus 106.42 tok/s (-1.1%, below the evaluator's
  stated host-load noise threshold).
- MTP acceptance: 0.776; tokens per step: 3.329.
- One of ten allowed evaluations used. Hidden holdout is reserved for submission.

The GPU regression passed all tested sizes (9, 33, 256, 1,024, and 6,000 tokens),
including bitwise equality of the independent accumulation kernel against the
parent's ascending-expert `index_add_` sequence. The 33-token case concentrates
routing in eight experts and exercises empty expert ranges and partial tiles.

## Paired synthetic requests

One loaded model, one recurrent-state slot, warm graphs, alternating AB/BA order
at each prompt length, two repetitions per implementation, 256 output tokens:

| Prompt tokens | Parent mean prefill | Grouped mean prefill |
| --- | ---: | ---: |
| 200 | 0.851 s | 0.156 s |
| 512 | 1.009 s | 0.155 s |
| 2,048 | 1.255 s | 0.256 s |
| 6,000 | 1.589 s | 0.629 s |

Mean prefill fell from 1.176 s to 0.299 s (-74.6%). Aggregate end-to-end throughput
was 65.72 versus 94.77 tok/s. Aggregate decode was 93.79 versus 106.16 tok/s.
However, synthetic continuations differed at token indices 174, 106, 247, and 1,
respectively; these are **not** controlled identical-output decode comparisons
or independent correctness results. Use the strict dev judge for correctness.
The unchanged small-token path and these measurements give no evidence of a
systematic decode cost regression, but hidden holdout performance remains for
the final evaluator to determine.

## Follow-up

Try tuning the grouped GEMM tile sizes and expert-weight reuse for long prompts.
A fixed-size route histogram could also replace `torch.bincount`, which may still
synchronize internally despite removing the explicit Python count readback.
Keep the BF16 activation boundaries and sequential expert accumulation, and rerun
the strict judge for any change to GEMM reduction order.
