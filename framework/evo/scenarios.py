"""Deployment scenarios an engine lineage can be evolved for. Each scenario fixes the prompt sets, the evaluation modes
(workloads, prompts per workload (0 = all), new tokens), whether requests run one at a time or as one batch, and the
metric(s) a change must improve. Prompt sets ship in data/ (built by build_workloads.py, build_scenarios.py and
build_olmoe_workloads.py).

Optional keys:
  model        key in [models] of evoinfer.toml for the checkpoint the engine runs (default "qwen3_5")
  judge        "engine" (default): score against the frozen seed engine; "hf": against the transformers implementation
  judge_model  key in [models] for the judge's reference weights (default: same as model)
  need_gb      free GPU memory the benchmark needs (default [gpus] need_gb)
  judge_need_gb  free memory the judge needs, if it differs (the judge then picks its own GPU from the pool)
  gate         per-model calibration of the correctness gate (evo.judge_gate defaults otherwise)
To add a scenario, add an entry here and to PRIMARY / SCEN_GOAL in orchestrator.py (docs/adding-a-scenario.md)."""
import os
from evo.paths import WORKLOADS, PRIVATE

W = lambda n: os.path.join(WORKLOADS, n)
P = lambda n: os.path.join(PRIVATE, n)


def modes(prefix, quick_limit, quick_new, new):
    dev, hold = ("dev", W(f"{prefix}dev.jsonl")), ("holdout", P(f"{prefix}holdout.jsonl"))
    return {"quick": ([dev], quick_limit, quick_new), "dev": ([dev], 0, new), "full": ([dev, hold], 0, new)}


SCENARIOS = {
    # single request, short-to-medium prompts (27-5K tokens), 256 new tokens; accept on decode or e2e tok/s
    "base": {"modes": modes("", 6, 128, 256), "batch": False, "primary": ("decode", "e2e")},
    # single request, 8K-31K-token prompts, 1024 new tokens; accept on end-to-end tok/s (prefill + decode)
    "lc": {"modes": modes("lc_", 3, 256, 1024), "batch": False, "primary": ("e2e",)},
    # 32 requests submitted at once, 256 new tokens each; accept on batch throughput (generated tokens / wall time)
    "cc": {"modes": modes("cc_", 16, 128, 256), "batch": True, "primary": ("e2e",)},
    # edge device: a 32 GB consumer GPU (the paper used an RTX 5090), routed experts in 4-bit (evo.quantize_int4), base
    # prompt sets, single request. The judge scores against a BF16 checkpoint holding exactly the same 4-bit expert values,
    # which needs an 80 GB GPU: put both kinds of GPU in [gpus] pool, or run the edge lineage on an 80 GB GPU.
    "edge": {"modes": modes("", 6, 128, 256), "batch": False, "primary": ("decode", "e2e"),
             "model": "qwen3_5_int4", "judge_model": "qwen3_5_int4_ref", "need_gb": 29, "judge_need_gb": 74},
    # a second target model: OLMoE-1B-7B (16 layers, 64-expert top-8 MoE, standard attention, 4K context, no draft head).
    # Base prompt sets cut to its context (build_olmoe_workloads.py), single request, 256 new tokens
    "olmoe": {"modes": modes("olmoe_", 6, 128, 256), "batch": False, "primary": ("decode", "e2e"),
              "model": "olmoe", "judge": "hf", "need_gb": 30,
              # gate calibrated on 18.9K tokens of two legitimate computations (transformers, incremental decoding with
              # sdpa and with eager attention, against the full-sequence forward): flips 0.5-0.95%, 6 gaps in (0.5, 1.0],
              # none above 1.0. The legitimate tail is heavier than the main model's
              "gate": {"hard": 1.5, "max_soft_frac": 0.003}},
    # a third target model: Gemma 4 26B-A4B-it, text path (30 layers, sliding-window + full attention, 128-expert top-8
    # MoE block, 262K vocabulary). The base prompt sets fit its context unchanged; BF16 weights are 49 GB
    "gemma4": {"modes": modes("", 6, 128, 256), "batch": False, "primary": ("decode", "e2e"),
               "model": "gemma4", "judge": "hf", "need_gb": 62,
               # gate calibrated on 17.3K tokens of legitimate computations (transformers, incremental decoding with sdpa
               # and eager attention, against the full-sequence forward): flips 0.77%, 0.11% of tokens above 0.5 and 4
               # gaps above 1.0 (1.125, 1.25, 1.5, 1.875). The main model's gate would reject the reference itself
               "gate": {"hard": 3.0, "max_soft_frac": 0.005}},
}


def current():
    return os.environ.get("EVO_SCENARIO") or "base"


def model_path(scenario, key="model"):
    """Checkpoint directory for a scenario's engine model (key="model") or judge reference (key="judge_model")."""
    from evo import config
    sc = SCENARIOS[scenario]
    name = sc.get(key) or sc.get("model") or "qwen3_5"
    path = config.model(name)
    if not path:
        raise SystemExit(f"scenario {scenario!r} needs [models] {name} in evoinfer.toml (or $EVO_MODEL_{name.upper()})")
    return path
