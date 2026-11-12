# Reproducing the paper

Run `python3 orchestrator.py setup` first. Each script takes a comma-separated GPU list and gives one agent per GPU per
round. They are thin wrappers around `orchestrator.py init` and `orchestrator.py evolve`, so read them to see the exact
settings. Agent runs are stochastic: expect the same path (CUDA graphs first, then MoE kernels, prefill and kernel
fusion), but different numbers. Three Codex runs under the same budget ended at 5.24x, 6.68x and 7.04x.

| Paper | Script | What it does | Paper result |
|---|---|---|---|
| E1 main line | `e1_main_line.sh 0,1 6` | evolve the naive seed (`baseline`) with Codex | 17.0 → 297.5 tok/s decode in ~20 agent-hours (`engines/evo-015`) |
| E2 long context | `e2_long_context.sh 0,1,2,3` | lineage `lc` from `paper-evo-015`, 8K-31K prompts | 1.60x / 1.63x default vLLM end-to-end (`engines/lc-003`) |
| E2 edge | `e2_edge.sh 0,1,2` | 4-bit experts, porting agent, then lineage `edge` | 1.61x / 1.79x default vLLM (`engines/edge-003`) |
| E3 policy models | `e3_policy.sh pq qwen 0,1` | one lineage per policy from the seed, 3 rounds × 2 agents × 75 min × ≤ 8 evals | Qwen3.8-Max 6.43x, GLM-5.3 4.31x, DeepSeek-V4-Pro 1.52x, Codex 6.32x (3 runs) |
| E3 seed engine | `e3_seed_engine.sh 0,1` | port Qwen3.5 into mini-sglang, then evolve | 65.0 → 118.3 tok/s (1.82x) |
| E3 target models | `e3_target_model.sh ol olmoe 0,1` | port another model into the seed, then evolve | OLMoE 2.24x, Gemma 4 9.12x |
| E4 meta level | `e4_skills.sh evo-013 0,1,2,3` | meta-agent v0 → v1, paired arms from one parent | same yield, v1 used 34% of the agent time |
| No-harness baselines | `no_harness/bare.py`, `no_harness/dsh.py` | the same model without EvoInfer: a bare tool loop, or DeepSeek Harness | bare loop 1.0x; DeepSeek Harness 7.37x (one run, 200 of 450 minutes) |

Notes:

- **Skills.** The scripts use the hand-written skills `skills/v0`. The paper's long-context and edge lineages used the
  meta-evolved skills v1, which are not part of this release. Generate your own with `meta.py evolve`.
- **Gateway policies.** `e3_policy.sh` with a gateway policy needs a `[policies.<name>]` entry and its environment
  variables (see `docs/configuration.md`).
- **No-harness baselines.**
  - They need an OpenAI-compatible endpoint (`EVO_LLM_BASE_URL`, `EVO_LLM_API_KEY`) and bubblewrap (Linux).
  - They start from the root of a lineage (`--seed-lineage`, default: the main line) and use the same `evo` commands
    as the agents.
  - `dsh.py` also needs `npm i -g @deepseek-ai/dsh` (Node ≥ 24.9).
- **vLLM reference.** The paper compares end-to-end throughput with vLLM 0.30 in its default configuration: no
  speculative decoding, `max_num_seqs` 128 (the hybrid model has too few Mamba state blocks for 256), the same prompts
  and token budget. This release does not include that script.
