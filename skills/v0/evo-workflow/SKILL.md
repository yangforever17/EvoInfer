---
name: evo-workflow
description: How an optimization episode works in EvoInfer - goal, rules, the evo CLI, and how to finish. Read first.
---
# EvoInfer optimization episode

You are improving an LLM inference engine (a nano-vllm fork serving Qwen3.5-35B-A3B on one A100-80GB).
**Objective:** lower single-request greedy latency - decode tokens/s and end-to-end tokens/s (prefill + 256 new
tokens) - on the evaluation prompts, without changing what the model computes. A change is accepted if one of the
two improves by > 3% on dev and the hidden holdout set and neither regresses (parent re-timed on the same GPU).

## Rules (hard)
- Correctness is judged token by token against a frozen reference (teacher-forced logit gap = reference top-1
  logit minus the logit of your token). Gap <= 0.5: near-tie (max 2% of tokens). 0.5 < gap <= 1.0: soft violation
  (max 0.1% of tokens - legitimate bf16 reordering produces ~0.004%). Gap > 1.0: violation; any one = rejected,
  however fast. Bugs (wrong state, wrong cache) show up as many large gaps; a lone 0.6 is numerics.
- Do not touch anything outside your worktree. Never kill or signal processes you did not start; other users
  share this machine. Use only the GPU `evo` gives you.
- Do not special-case the evaluation prompts, cache outputs across requests, or read the holdout set.
- `evo_config.json` holds the engine kwargs used by the evaluator (e.g. `num_speculative_tokens`); you may change it.

## Loop
1. `evo history` - learn from earlier experiments, including failures. Do not repeat a rejected idea unless
   you know why it failed and what is different now.
2. Measure before changing: profile with `evo run -- python your_script.py` (worktree is on PYTHONPATH,
   model path in `$EVO_MODEL`). Find where decode time actually goes. `evo run` executes on the GPU host
   inside a mirror of your worktree: put scripts INSIDE the worktree (files in /tmp are not shipped), and
   keep throwaway scripts out of your final commit.
3. Coordinate: `evo board` shows what the other parallel agents have claimed, their notes, and results that
   finished during your episode. Before implementing, `evo claim -m "..." --where path:function` for every
   function you will edit; if refused, pick another bottleneck. Post reusable measurements with `evo note`, and
   `evo release -m "<why>"` when you abandon an idea. Your claims are released when you submit.
4. Make ONE coherent change at a time; keep it small enough to verify.
5. `evo eval` (quick: 6 prompts x 128 tokens, with correctness) - compare against the parent numbers.
   `evo eval --dev` is closer to the final evaluation. Evaluations are budgeted; do not waste them.
6. Keep what helps, revert what does not. Finish with
   `evo submit -m "<hypothesis> | <what changed> | <measured effect> | <what you would try next>"`.
   The message is read by future agents - make it specific (numbers, files, failure reasons).

## Engine map
- `nanovllm/models/qwen3_5_moe.py` - model: Gated DeltaNet (30 layers), gated attention (10), MoE, MTP.
- `nanovllm/engine/model_runner.py` - prefill / verification / MTP drafting (`run_hybrid`, `mtp_propose`).
- `nanovllm/engine/scheduler.py`, `block_manager.py` - batching, KV blocks, lookahead for drafts.
- `nanovllm/layers/attention.py` - paged FlashAttention (vLLM build), KV store kernel.
- `nanovllm/layers/` also holds kernels earlier agents added (fused MoE decode GEMVs, grouped prefill GEMMs,
  DeltaNet multi-token verification); read their `*.md` notes in the repo root before touching them.
