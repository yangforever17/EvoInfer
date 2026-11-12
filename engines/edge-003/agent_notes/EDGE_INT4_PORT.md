# RTX 5090 INT4 expert port

The checkpoint's top-level `quantization_config` is propagated to the text config
before allocating the model. `evo_int4` experts use resident uint8 qweights and
BF16 per-output-row, K-group-128 scales. Main gate/up and down tensors load
directly; MTP's individual gate/up/down tensors load into the corresponding
expert slices. The loader checks shapes, packed dtypes, and all MTP slices.
Vision weights remain skipped. Unquantized checkpoints retain their BF16 path.

`moe_decode._load_expert` unpacks low/even and high/odd nibbles, subtracts eight,
multiplies in FP32, and rounds **each weight to BF16 before multiplication with
activations**. Both the indirect decode GEMVs and grouped prefill GEMMs use this
helper. No expanded expert cache or full-model BF16 expert allocation is needed.
Routing, activation/output rounding boundaries, shared experts, MTP depth three,
verification, rollback, and graph ownership are retained.

Packed loads change Triton's chosen reduction layout (gate GEMV sizePerThread
1 versus BF16's 8 on this installation). Thus canonical weights are bitwise
exact, but expert outputs are not claimed bitwise equal to BF16 kernels.
Graph replays are bitwise equal to eager packed execution for changing inputs,
routes and routing weights. The unchanged model judge must approve arithmetic.

Configuration: one concurrent request, 8,192-token maximum context/batch, memory
utilization 0.90. The original 4,096-token limit could not schedule the full dev
set's longer prompts: hybrid prefill requires the entire prompt in one batch.
The larger limit also sizes graph block tables for long prompts and generation.
Initialization with the new configuration allocated 26.37 GiB including
KV/recurrent storage; the long-context regression peaked at 27.21 GiB.

## Validation

Run GPU checks exclusively with `evo run -- python tests/<script>.py`:

- `check_int4_values.py`: canonical weight equality and compiler-layout evidence.
- `check_int4_experts.py`: 256 experts, three seeds, token counts
  1/2/4/8/9/32/200/512; numerical BF16-reference checks and exact changing-route
  graph replay checks. All pass.
- `check_edge_model.py`: real main/MTP checkpoint weights; six prompts including
  255/256/257-token KV-boundary cases, each eager then twice graphed, 32 generated
  tokens. All token sequences match. One prompt has differing eager/graph draft
  acceptance counts, so this is token equivalence, not identical speculative work.
- `check_edge_fallback.py`: MTP disabled, two concurrent requests, three prompts,
  eager/graph/replay. All token sequences match.
- `check_edge_long_context.py`: 4,097- and 5,376-token prompts, 256 generated
  tokens each, eager/first graph/replay. All tokens match, KV capacity covers
  the configured context plus MTP lookahead, and all KV blocks are released
  between requests. Passed with the 8,192-token configuration.
- `check_int4_real_activations.py`: selected-expert BF16 reference on real target
  and MTP inputs, without expanding the entire model. Passed 948 calls /
  7,841,792 output elements, 5,997 differing values, max absolute error 0.00390625.

Synthetic complete MoE graph times with 256 experts (BF16 / INT4 milliseconds):
T=1: 0.04557 / 0.03293; T=2: 0.08329 / 0.05258;
T=4: 0.14006 / 0.08462; T=8: 0.26996 / 0.16170.
These are component measurements, not whole-engine paired gains.

First `evo eval` benchmark completed: quick dev aggregate 363.40 decode tok/s,
297.06 e2e tok/s. Its judge did not run because edge_worker.py's result-export
call `rsync(*files, destination)` conflicts with the helper signature
`rsync(src, dst, *extra)` when both bench files exist. The worker reported
`edge_error`, then evaluate.py raised `KeyError: wall_s`. Framework repair was
requested; these numbers are unjudged and establish no acceptance verdict.
That initial attempt had no judge result. No evaluation rules or framework
files were changed.

After the framework export repair and the context-limit fix, full
`evo eval --dev` run `1110-135811-11101334-edge-root-e1-039f` completed all
12 prompts at 256 new tokens: **333.44 decode tok/s, 277.82 e2e tok/s**,
mean TTFT 0.16 s, MTP acceptance 0.745, 3.234 tokens/step. The strict judge
**passed** across 3,072 tokens: zero hard violations, one allowed soft
violation (gap 0.625), and 17 near-tie flips. This continuation used one of
12 evaluation runs. Hidden-holdout correctness/performance remains for the
orchestrator; these full-dev aggregates are not comparable to the earlier
short-prompt quick aggregate and do not establish a paired speed gain.

Next opportunity: vectorized packed loads and an explicit reduction layout,
then prefill tile tuning on sm_120. Preserve canonical BF16 rounding and measure
MTP singleton and four-token target verification separately.

## Warm decode profile

`profile_edge_int4.py`, after warming the complete-step graph, captured 32 steps
(96 drafted / 53 accepted). CUDA activity totaled 281.33 ms; profiler overhead
means this is attribution, not throughput. BF16 CUTLASS WMMA GEMMs accounted for
46.76%, expert gate/activation 20.22%, and expert down GEMV 14.28%. DeltaNet
verification was 2.57%, norms 2.15%, routing 1.69%, settlement 1.23%. Allocated
memory with warm step graphs was 27.03 GiB, peak 27.40 GiB. Dense BF16 GEMM
selection and vectorized INT4 expert loads are the largest remaining targets.
