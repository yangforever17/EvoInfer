# Prefill CPU dispatch measurements

Episode `11091900-codex-free-d6ff`, parent `e9f328408d`.

The parent calls `torch.set_default_device('cpu')` after loading and warming the
model. This leaves a Python `DeviceContext` active, intercepting tensor operations
even when they already specify a device. `torch.set_default_device(None)` removes
that mode and retains ordinary CPU allocation defaults.

## Parent attribution

`tests/profile_prefill_routes.py`, through `evo run`, measures an ordinary compiler
explanation prompt (278 tokens), prefill plus one output token, after warming the
same recurrent slot and graph. Three unprofiled requests took 152.2–156.8 ms.
The profiler recorded 154.97 ms self CPU time, 77.52 ms self CUDA time, and 3,991
CUDA launches. Grouped MoE GEMMs accounted for 42.55 ms CUDA time. The DeltaNet
chunk wrapper accounted for 41.49 ms self CPU time versus 3.09 ms total CUDA time.
A separate Python profile found 17,278 default-device `__torch_function__` calls.

The initial routing-metadata hypothesis was dropped: 41 `bincount` calls took
5.64 ms total CPU time, and all stream synchronization accounted for only 1.67 ms.
These profile timings attribute work; they are not throughput measurements.

## Default-device paired measurement

`tests/pair_default_device.py` loads the configured model once, warms both modes,
and alternates parent/candidate then candidate/parent on six ordinary prompts
(prose, code, Chinese, math, fiction, and a longer policy discussion), each with
256 output tokens. Slot 0 is reused to isolate warm prefill and decode. The
captured computation is unchanged between modes. Timing separates prefill from
decode; the prefill-produced token is excluded from decode throughput.

| Local aggregate | Parent | Device mode removed |
| --- | ---: | ---: |
| Decode tokens/s | 263.590 | 263.919 |
| End-to-end tokens/s | 224.664 | 233.702 |
| Mean prefill | 172.07 ms | 129.21 ms |

End-to-end gain was 4.02%; decode was effectively flat (+0.125%). All 12 paired
completions and accepted-token counts were identical.

`tests/check_default_device.py` additionally passed CPU/explicit-CUDA allocation
checks, ten sequential request pairs starting with separate empty graph caches
and graph pools, cycling all eight state slots, and a mixed two-request batch.
All paired tokens and acceptance counts were identical. The first request includes
compilation and is not a comparable timing. Later fresh-slot prefills were roughly
380 ms parent versus 260–280 ms candidate; warm-slot prefills also improved.

## Existing norm kernel screen

`tests/check_prefill_norm.py` compares `norm_exact` against the actual large-input
norm expressions at lengths 9, 33, 278, 1024, and 4096, with widths 2048, 256,
and 128, padded input rows, zero-centered and gated norms. All FP32 reduction
means and BF16 outputs were bitwise identical, including three mutations after
CUDA graph capture per shape.

With the device mode already removed, alternating eager component timings at
278 tokens were about 100–125 us parent versus 23–24 us fused; at 4096 tokens,
267/500/852 us parent versus 26/65/140 us fused for the three respective shapes.
These include eager dispatch, appropriate for prefill; no decode speedup is
inferred from them. Whole-request marginal measurements follow separately.

## Combined paired measurement

`tests/pair_prefill_norm.py` preserves the parent norm expressions and alternates
parent, device-only, and combined implementations, then reverses the order.
Prefill-only norm switching leaves the short/captured path unchanged. Before
timing, 366,481,408 real prefill norm elements from the six ordinary prompts
matched the parent bitwise. All twelve timed triples matched complete tokens and
accepted-token counts.

| Local aggregate | Parent | Device only | Device + prefill norms |
| --- | ---: | ---: | ---: |
| Decode tokens/s | 265.379 | 265.593 | 265.609 |
| End-to-end tokens/s | 228.063 | 236.292 | 239.164 |
| Mean prefill | 161.61 ms | 123.29 ms | 110.34 ms |

Combined end-to-end gain was 4.868%, with decode +0.087%. Norm fusion's marginal
end-to-end gain over device-only was 1.216%. The production change removes the
CPU device mode in `ModelRunner.__init__` and extends the existing `norm_exact`
dispatch to prefill in `ZeroCenteredRMSNorm.forward` and `GatedRMSNorm.forward`.
It introduces no new kernels or configuration changes.

The two reversed passes gained 4.815% and 4.921% end-to-end. Every prompt improved
in aggregate (1.98–5.96%); the long policy prompt had the smallest gain.

## Full dev judge

Evaluation `1109-191117-11091900-codex-free-d6ff-e1-e5ab` tested the final production
diff with the unchanged configuration and strict judge:

- Decode: 280.93 tokens/s versus stored parent 280.95 (effectively unchanged).
- End-to-end: 232.21 tokens/s versus stored parent 219.65, +5.7%.
- Mean TTFT: 0.20 s versus stored parent 0.26 s.
- MTP acceptance: 0.742; tokens/step: 3.226.
- Correctness PASS: zero hard violations, zero soft violations, and 18 permitted
  near-tie flips across 3,072 tokens.
- One of ten permitted evaluations used. Hidden-holdout acceptance remains the
  final evaluator's decision. The paired measurements above provide the control
  against stored-baseline timing drift.

Next opportunity: prefill grouped expert GEMMs dominate measured GPU time
(42.55/77.52 ms), while FLA chunk dispatch still has host overhead. Measure the
remaining candidate profile before trying more eager pointwise fusions or
specialized prefill kernels. The discarded routing-metadata idea has only small
standalone headroom on this workload.

Raw logs are `prefill_routes_profile.log`, `prefill_routes_cpu_profile.log`,
`default_device_pairs.log`, `default_device_lifecycle.log`,
`prefill_norm_components.log`, `prefill_norm_pairs.log`, and
`prefill_dispatch_eval1.log`.
