# Exact gate and attention launch reduction

Episode `11091858-codex-overhead-f360`, parent `e9f328408d`.

## Profile

Forty warmed speculative steps: 431.171 ms CUDA (10.779 ms/step).
Expert gate/up 22.89%, expert down 12.45%, all split-K reductions 3.63%,
Delta verification 3.25%, norms 3.34%. The synchronous result transfer waits
for the graph; removing Python loops alone has little headroom.

The DeltaNet a/b projections execute two tensor-core GEMMs, two split-K
reductions, and a gate epilogue per layer. With 30 layers, these small launches
are a useful target even though expert bandwidth dominates total work.

## Rejected direct-reduction variant

One fused FP32 GEMV plus gate kernel reduced T4 component time from
12.75–13.20 us to 2.60–3.23 us. But changing reduction order affected 9 of
67,200 synthetic gate outputs (maximum g difference 0.00582; beta 0.001953).
A six-prompt alternating paired run, repeated twice, measured:

- Decode 274.987 -> 274.327 tok/s (-0.24%).
- End-to-end 233.850 -> 233.519 tok/s (-0.14%).
- Speculative steps 1,024 -> 1,056; only 2/12 completions identical.

A ~3.3% step-time saving was canceled by changed draft acceptance. This
variant was never enabled in the production model and is not retained.

## Exact gate projection

Profiling cuBLAS on BF16 `[4,2048] @ [32,2048].T` identified
`ampere_s16816gemm_bf16_64x64_ldg8_stages_64x5_tn`, grid `[1,1,32]`, followed
by `splitKreduce_kernel`. It uses 32 independent 64-element tensor-core
products, followed by an increasing-index FP32 sum. A tree sum over splits
or an ordinary scalar reduction does not reproduce its rounding.

The new implementation launches both a/b sets of partials together, then
fuses their sequential sum, BF16 projection rounding, and gate epilogue.
Other shapes/dtypes use the original projections. T4H32 component time:
12.61 -> 3.84 us. The expanded mutable graph test checks 190,080 gate outputs, all exact,
including 500 changes to inputs, weights, biases, and decay parameters at
the optimized shape.

Six ordinary prompts, two repetitions, alternating parent/candidate order,
one loaded model, separate graph caches, and a reused slot:

- Decode 276.592 -> 283.407 tok/s (+2.46%).
- End-to-end 237.130 -> 242.197 tok/s (+2.14%).
- All 12 completion pairs identical, 1,024 speculative steps on both sides.

## Attention fusion

Combine Q/K zero-centered normalization and partial rotary in one launch,
retaining the parent's ATen reduction ordering and every BF16 rounding
boundary. A second kernel combines sigmoid and multiplication, reading the
strided gate projection directly instead of materializing its reshape.

Initial norm/rotary regression: 6,137,856 values match bitwise across BF16,
FP16, FP32, short lengths, padded Q rows, and changing graph inputs.
The strided gate extension also passes. A combined six-prompt paired run,
repeated twice with alternating order, measured 276.671 -> 286.686 decode
(+3.62%) and 236.584 -> 244.137 end-to-end tok/s (+3.19%). All 12 completion
pairs and all 1,024 speculative steps are identical. A separate comparison
of batch sizes one, two, and three, a four-token prefill, and sequential
CUDA-graph reuse passes all seven completion comparisons exactly.

## Complementary convolution fusion

Earlier episodes found exact short convolution fusion useful but too small
in isolation. Here it is tested as an addition to the independently measured
gate and attention changes, to improve the margin above the 3% threshold.

The kernel writes both contiguous repeated Q/K/V values and the complete
convolution buffer used by rollback. FP32 convolution uses the same four
ordered FMAs; the activation retains its BF16 convolution-output boundary.
Component T4 time is 10.34 -> 2.19 us. Numerical and mutable-graph tests pass
19,998,720 values bitwise across BF16, FP16, and FP32.
Three-way six-prompt comparison, two alternating-order repetitions:

| Variant | Decode tok/s | End-to-end tok/s | Steps |
| --- | ---: | ---: | ---: |
| Parent | 275.350 | 236.265 | 1,024 |
| Exact gates + attention | 285.084 | 243.897 | 1,024 |
| Add convolution fusion | 290.645 | 247.350 | 1,024 |

All 12 completions match the parent at both stages. The complete change
improves decode by 5.55% and end-to-end by 4.69%; convolution adds 1.95%
and 1.42% respectively. Final dispatch uses convolution fusion only for
non-prefill queries of at most eight tokens with a four-element filter.
Longer queries and prefill retain the original implementation.

Expanded tests for every length 1–8 pass 46,080,000 values bitwise. The final
mixed-batch comparison also passes all seven completions exactly. A longer-context paired comparison at 140, 522, and 1,548 input tokens,
two reversed-order repetitions each, passes all six completions exactly
and takes 534 speculative steps on both variants. Decode improves
256.696 -> 271.735 tok/s (+5.86%); end-to-end improves 215.087 -> 224.885
tok/s (+4.56%). Together with the ordinary-prompt comparison, all 18 final
completion pairs are identical.

## Strict evaluation

Fresh unmodified parent dev (evaluation 1): 292.27 decode, 220.88 end-to-end
tok/s, mean TTFT 0.29 s. Zero hard/soft violations in 3,072 tokens.
Stored parent: 280.95 decode / 219.65 end-to-end. This drift reinforces the
need for paired measurements. Gate/attention candidate dev (evaluation 2): 290.85 decode / 224.83
end-to-end tok/s, TTFT 0.26 s, acceptance 0.747, zero hard/soft violations
in 3,072 tokens. This improves over the stored parent by 3.5% / 2.4%, but
not over the fresh parent's decode result (whose first request measured
520.83 tok/s versus 313.62 for the candidate). The unchanged repeat (evaluation 3) measured 290.82 decode / 224.39
end-to-end tok/s, TTFT 0.26 s, and again zero hard/soft violations in 3,072
tokens. Engine loading took 459 s under shared-host load. Thus the two
candidate measurements agree, while paired comparisons provide the cleaner
evidence of the kernel saving. The final convolution-inclusive evaluation (`1109-194929-11091858-codex-overhead-f360-e3-6638`,
fourth actual evaluation call) passes with zero hard/soft violations in 3,072
tokens: **295.76 decode / 228.56 end-to-end tok/s**, mean TTFT 0.26 s,
acceptance 0.743. This is +5.3% / +4.1% versus the stored parent and
+1.19% / +3.48% versus the fresh parent. The final unchanged repeat
(`1109-200044-11091858-codex-overhead-f360-e5-0f28`) passes with zero hard/soft
violations in another 3,072 tokens: **297.44 decode / 228.77 end-to-end tok/s**,
TTFT 0.26 s, acceptance 0.750. This is +1.77% / +3.57% versus the fresh
parent and +5.9% / +4.2% versus the stored parent. Both final evaluations
pass all 6,144 judged tokens with zero hard or soft violations.
The CLI reused the `e3` ordinal because the prior call was still judging;
full run IDs distinguish the snapshots. Five evaluation calls were used, within the ten-call budget.

## Final state

Retain all three complementary fusions and their regression tests. Temporary
profiling, arithmetic-order, paired-comparison, and batch scripts were removed.
Python syntax compilation and `git diff --check` pass. GPU work ran only through
`evo run` / `evo eval`. Configuration, draft depth, target logits, recurrent
rollback policy, and large-prefill arithmetic are unchanged. Hidden holdout and
final parent retiming remain the submission pipeline's decision.

Next: apply the exact split-order methodology to other small projection
epilogues; ordinary FP32 GEMV replacements can lose more through changed
speculative acceptance than they save in kernel time.
