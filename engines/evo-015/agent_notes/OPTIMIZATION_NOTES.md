# Decode launch overhead: episode 11090218-codex-overhead-2e4d

Parent: `4cea4f6cd0`. Configuration unchanged: three speculative tokens, max model length 8192.

## Evidence and changes

The parent already captures verification, acceptance, settlement, and MTP in a single CUDA graph.
A 20-step synthetic-prompt profile measured 614.81 ms of summed GPU kernel time and 717.00 ms
of profiled CPU time. Selected-expert weight gathering accounted for 24.95% of GPU time, but
history records a hidden-set correctness rejection for the previous indirect-GEMV port.
This episode retains all model GEMMs, routing, and recurrence arithmetic.

- `state_settle.py` copies the accepted recurrent snapshot and convolution window directly to
  persistent state, removing index-select temporaries and their follow-up copies.
- `norm_apply.py` fuses small pointwise kernels surrounding RMS normalization. PyTorch still
  performs the mean reduction and SiLU. The square/cast, epsilon/rsqrt, and final products are
  fused with explicit dtype boundaries; FP contraction is disabled for norm application.
  Large prefill normalization retains the parent path.
- `model_runner.py` reuses a pinned input buffer for IDs and positions and transfers it once
  per replay. Block tables are refreshed only when their contents change. The existing output
  readback completes before the next host-buffer update. Graphs retain all buffer owners.

## Measurements

Stored parent dev: 104.99 decode tok/s, 94.82 end-to-end tok/s, mean TTFT 0.27 s.

| Dev evaluation | Change | Decode tok/s | End-to-end tok/s | Strict violations |
| --- | --- | ---: | ---: | ---: |
| 1 | Direct state settlement | 106.05 | 95.52 | 0 / 3072 |
| 2 | Add pointwise norm application | 110.54 | 99.21 | 0 / 3072 |
| 3 | Also fuse square/cast and epsilon/rsqrt | 116.39 | 103.44 | 0 / 3072 |
| 4 | Final: add persistent pinned inputs and cached block tables | 116.25 | 103.72 | 0 / 3072 |

Paired tests use one loaded model, separate parent/candidate graph caches, alternating execution
order, three prompts, three repetitions, and 256 generated tokens per request. Prompts include
a 400-token synthetic sequence and two ordinary explanatory questions. They are not evaluation
or holdout prompts. Every parent/candidate completion was token-identical.

| Paired variant | Parent decode | Candidate decode | Parent end-to-end | Candidate end-to-end |
| --- | ---: | ---: | ---: | ---: |
| Settlement + initial norm application | 107.72 | 114.60 (+6.4%) | 101.57 | 107.54 (+5.9%) |
| Final, including persistent pinned inputs | 107.69 | 119.23 (+10.7%) | 101.55 | 111.67 (+10.0%) |

Final paired mean TTFT: parent 0.15299 s, candidate 0.15372 s. The paired benchmark exercises
changing block tables, block-boundary crossings, and graph reuse across requests.

## Regression checks

GPU checks must run through EvoInfer:

```
evo run -- python tests/check_state_settle.py
evo run -- python tests/check_norm_apply.py
```

State settlement is bitwise equal for every accepted prefix at query lengths 1, 2, 4, and 8,
including mutated graph inputs and untouched neighboring state slots. Norm checks compare with
the original PyTorch expressions in BF16, FP16, and FP32, including strided head views and graph
replay. The integration benchmark also checks full generated-token equality against the parent.

Next idea: remove remaining purely structural copies in DeltaNet (head repetition and contiguous
Q/K/V copies) by consuming their original strides, keeping the recurrence reduction unchanged.

Final strict dev evaluation: decode +10.7%, end-to-end +9.4%, mean TTFT 0.28 s,
MTP acceptance 0.777 and 3.332 tokens/step; 15 allowed near-tie flips, zero violations.
Four of ten permitted evaluations used. Hidden holdout is reserved for submission.
