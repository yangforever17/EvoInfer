# Direct draft shortlist projection (RTX 5090)

Parent: `6af9511a0e`. Configuration unchanged (depth 3, INT4 experts).

The parent gathers 16,384 BF16 head rows (four target positions times
4,096 candidates) and runs three singleton projections. The candidate reads
those same rows directly, accumulating FP32 and rounding output to BF16.
Candidate ordering, duplicates and argmax tie handling remain unchanged.
This is a draft-only numerical change: its reduction need not match cuBLAS.
Target logits, verification, recurrent settlement and MTP state are unchanged.

Component graph timing (`evo run -- python tests/check_draft_head.py`):
parent gather + three projections 115.48 / 118.38 us; candidate 34.93 / 34.88 us,
200 replays per sample in ABBA order. Eight dynamic hidden/ID replay cases
matched argmax; maximum BF16 logit difference 0.03125 (not bitwise exact).

Paired generation (`evo run -- python tests/pair_draft_head.py`): six ordinary
prompts (explanation, code, Chinese, math, story, longer policy context),
256 tokens, AB then BA, separate graph caches and pools. All 12 pairs had
identical complete tokens and acceptance counts. Every pair improved decode.

| Metric | Parent | Candidate |
| --- | ---: | ---: |
| Decode tokens/s | 366.745 | 372.405 (+1.543%) |
| End-to-end tokens/s | 327.344 | 331.587 (+1.296%) |
| Average step ms | 7.7256 | 7.6082 |
| Decode tokens/step | 2.8333 | 2.8333 |
| Accepted/drafted | 0.61543 | 0.61543 |
| Mean prefill seconds | 0.08675 | 0.08731 |

Per-request timing records are in `draft_head_pair_timings.json`. The frozen
parent method is retained in `parent_spec_runner.py` for reproducibility.
These are local paired measurements, not hidden-holdout acceptance.

Strict full dev: `1110-141845-11101414-codex-spec-43bc-e1-2be9` PASS,
3,072 tokens, zero hard/soft violations, 11 allowed near-tie flips.
395.21 decode tok/s, 324.89 end-to-end tok/s, mean TTFT 0.14 s,
MTP acceptance 0.741 and 3.224 tokens/step. One of ten eval runs used.
Stored-parent gains (18.3% / 17.6%) are much larger than the controlled
paired gains and must not be attributed solely to this patch. Hidden holdout
and promotion remain for final evaluation; this small measured improvement
is submitted for pooling if below the promotion threshold.

Next opportunity: batch the 30 independent recurrent-state settlements,
preserving exact accepted-prefix snapshots and convolution windows. Earlier
A100 evidence suggests this is complementary, but it needs fresh 5090 timing.
