# Decode launch overhead experiment

Episode: 11061409-codex-overhead-9e68. Parent: 85d499b414.

## Implementation

Hybrid single-request speculative decoding uses one CUDA graph for target
verification, greedy prefix acceptance, recurrent-state settlement, MTP catch-up,
and recursive drafting. Only the packed tokens and acceptance count are read back
to the host. Graphs are keyed by recurrent-state slot and verification length.

MTP catch-up processes the fixed verification window and selects the last accepted
hidden state. Its unused suffix cannot affect the accepted prefix because attention
is causal. Recursive drafts overwrite speculative KV entries before using them.
Capture saves and restores recurrent states. Persistent query offsets, input
buffers, and block tables remain alive for replay; positions and tables are updated
for every step. Graph initialization occurs during prefill. Batched execution and
explicit `enforce_eager=True` retain the eager path. Plain single-request decode has
a target-forward graph with the existing commit path.

## Measurements

- Parent final dev: 23.81 tok/s; holdout: 22.34 tok/s.
- Baseline profile, five steps: about 20,000 CUDA launches, 2.253 s self CPU
  time versus 0.167 s self CUDA time. Profiling overhead makes these diagnostic
  figures, not an uninstrumented throughput measurement.
- Evaluation 1: capture crashed with an illegal memory access. The query-offset
  tensor was not kept alive; retaining it in graph storage fixed subsequent runs.
- Evaluation 2, target-only graphs: quick 50.06 tok/s, +117.8% against the comparable
  parent quick run; correctness passed with zero violations and 8 near-tie flips.
- Evaluation 3, separate target and MTP graphs: quick 98.67 tok/s, +329.2%; correctness
  passed with zero violations and 3 near-tie flips. The final implementation replaces
  separate MTP graphs with full-step capture.

A separate fixed-length sweep used three ordinary prompts (hash tables, binary
search, and a lighthouse story), 128 output tokens each, excluding prefill:

| Execution | Draft tokens | Aggregate decode tok/s |
| --- | ---: | ---: |
| Separate graphs | 3 | 71.70 |
| Full-step graph | 3 | 81.69 |
| Full-step graph | 2 | 88.39 |
| Full-step graph | 1 | 81.11 |
| Target-only graph | 0 | 53.51 |

Full-step and separate graphs at depth 3 produced identical tokens in this sweep.
Changing draft depth changed some outputs, so depth selection requires the strict
judge rather than relying on these speed measurements alone.

Additional eager/graph comparisons passed for ordinary and synthetic prompts,
including lengths 1, 255, and 510, repeated state-slot reuse, batched fallback,
and non-speculative decode across an exact 256-token KV-block boundary.

Evaluation 4 tested full-step capture at depth 2 on dev: 102.50 tok/s (+330.4%),
but the judge rejected it with one violation at token 221 (logit gap 0.625)
and 13 near-tie flips over 3072 tokens. Depth 2 was reverted despite its speed.
The depth-3 configuration retains the parent verification shape.

Evaluation 5 is the submitted configuration: full-step capture, depth 3,
**129.96 tok/s on dev versus 23.81 for the parent (+445.7%, 5.46x)**.
The strict judge passed with zero violations and 20 allowed near-tie flips over
3072 tokens. Mean TTFT was 3.67 s, MTP acceptance 0.765, and tokens/step 3.296.
Graph initialization adds work during prefill. Holdout performance is left to the
final evaluation. Five of ten available evaluations were used. Throwaway profiling
and comparison scripts were removed.

## Next experiment

With Python launch overhead removed, profile GPU memory traffic from per-token
expert-weight gathers and the intermediate recurrent-state clones. An in-place
expert-weight kernel is a likely next target; preserve router and accumulation
numerics and validate it independently.
