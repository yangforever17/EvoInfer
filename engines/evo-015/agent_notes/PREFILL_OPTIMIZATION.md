# Prefill route grouping

Episode: 11061605-codex-prefill-9633; parent: 48df782113.

The baseline MoE prefill loops over active experts and runs `torch.where` for each
expert. A 512-token, one-output-token request profile showed 5,919 `aten::nonzero`
calls, 6,011 stream synchronizations, 89,264 kernel launches, 2.822 seconds of self
CPU time, and 506.6 milliseconds of self CUDA time. The profile includes request
setup and speculative graph initialization, so these are diagnostic times rather
than end-to-end benchmark numbers.

The candidate stably sorts routes by expert once per layer, reads expert counts
once, and gathers inputs and weights in routing order. Expert matrix multiplications
retain their original shapes and row order. Experts are accumulated in ascending
order, retaining the existing BF16 rounding between additions. The small-token
branch used for decode and speculative verification is unchanged.

## Isolated numerical and timing validation

Random BF16 inputs and expert weights, all 256 experts, top-8 routing; five timed
iterations per shape after warmup, synchronized wall time through `evo run`.
Outputs were bitwise identical for all tested lengths.

| Tokens | Original expert loop | Grouped routes |
| --- | ---: | ---: |
| 200 | 55.88 ms | 28.52 ms |
| 512 | 57.41 ms | 29.29 ms |
| 2,048 | 59.72 ms | 28.95 ms |
| 6,000 | 57.34 ms | 27.22 ms |

## Full dev evaluation

One `evo eval --dev` used (12 prompts, 256 generated tokens each):

- Strict correctness PASS: 0 violations, 22 permitted near-tie flips / 3,072 tokens.
- End-to-end throughput: 65.65 tok/s versus parent 49.09, +33.8%.
- Mean TTFT: 1.301 seconds versus parent 2.70, -51.8%.
- Decode: 98.14 tok/s versus stored parent 101.22, -3.0%.
- MTP acceptance: 0.752; tokens per step: 3.255.
- Evaluation ID: `1106-161943-11061605-codex-prefill-9633-e1-3c1b`.

The evaluator flags differences below about 10% as host-load noise. The paired
comparison below found no material decode change. Holdout is left to the final
evaluator.

## Paired parent/candidate comparison

One loaded model with a single recurrent-state slot, warmed graphs, 256 generated
tokens per request, synthetic prompt lengths 200, 512, 2,048, and 6,000. The parent
and candidate MoE forward methods were alternated in AB/BA order at each length
(two repetitions, 16 requests total). Complete generated token sequences were
identical between implementations at every length.

- Parent aggregate decode: 90.3035 tok/s.
- Candidate aggregate decode: 90.2265 tok/s (-0.085%, effectively unchanged).
- Parent mean prefill: 2.1357 seconds.
- Candidate mean prefill: 1.2386 seconds (-42.0%).

This controls for prompt-dependent MTP acceptance and much of the runtime drift
that affect comparisons with the stored parent evaluation.

## Scope and next experiment

Only the large-token MoE route dispatch changed. Router arithmetic, expert GEMMs,
activation rounding, expert accumulation order, decode, and MTP configuration are
unchanged. Grouped input storage adds O(tokens * top_k * hidden_size) temporary
memory, approximately 188 MiB at 6,000 tokens for this model.

One of ten evaluation runs used. Syntax and diff checks passed. Throwaway profile
and comparison scripts were removed. A next experiment could replace the remaining
per-expert GEMM launch loop with grouped GEMMs, retaining BF16 activation and
ascending-expert accumulation rounding; it would require fresh strict validation.
