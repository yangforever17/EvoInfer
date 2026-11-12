# Extending EvoInfer

## A new deployment scenario

A scenario fixes the prompt sets, the evaluation modes, batching, and the metrics a change must improve.

1. **Prompt sets.** Write `data/workloads/<prefix>dev.jsonl` and `data/holdout/<prefix>holdout.jsonl`, one JSON object
   per line: `{"id": "...", "kind": "...", "text": "..."}` (`prompt_len` is optional). The bench applies the model's
   chat template and prepends a random nonce. `framework/evo/build_scenarios.py` shows how the long-context and
   concurrency sets were built.
2. **The scenario.** Add an entry to `SCENARIOS` in `framework/evo/scenarios.py`:
   ```python
   "mobile": {"modes": modes("mobile_", 6, 128, 256),   # quick mode: 6 prompts, 128 new tokens; dev/full: 256
              "batch": False, "primary": ("decode", "e2e")},
   ```
   Optional keys: `model`, `judge`, `judge_model`, `need_gb`, `judge_need_gb`, `gate` (see the module docstring).
3. **The orchestrator.** Add the scenario to `PRIMARY` (the same metrics) and a paragraph to `SCEN_GOAL` in
   `orchestrator.py`. That paragraph is what every agent of the lineage reads about its goal. New focus areas go into
   `FOCUS`.
4. **A lineage.** Create and evolve it:
   ```bash
   python3 orchestrator.py --lineage mob init --scenario mobile --from paper-evo-015 --gpu 0 --config '{"max_model_len": 4096}'
   python3 orchestrator.py --lineage mob evolve --gpus 0,1 --foci "overhead free|free free"
   ```
   `tests/test_config_and_scenarios.py` checks that steps 1-3 agree.

## A new target model

1. Add `[models] <key>` and a scenario with `"model": "<key>", "judge": "hf"`. The judge then uses the transformers
   implementation in BF16 as the reference.
2. **Calibrate the gate.** Run the reference against itself in two legitimate ways (incremental decoding with sdpa and
   with eager attention against the full-sequence forward) and look at the gap distribution. Set the scenario's
   `gate` so that the legitimate tail passes. For Gemma 4 the main model's gate would reject the reference itself.
3. Add a `MODEL_INFO` entry in `orchestrator.py`: a description for the porting agent, and the transformers source
   directory it may read.
4. Port and evolve: `experiments/e3_target_model.sh <lineage> <scenario> <gpus>`. A porting agent adds the model to
   the seed engine, and the root must pass the judge before evolution starts.

## A new seed engine

The benchmark only needs this, with the engine directory on `sys.path`:

```python
from nanovllm import LLM, SamplingParams
llm = LLM(model_path, **json.load(open("evo_config.json")))
outs = llm.generate(list_of_token_id_lists, SamplingParams(temperature=0, max_tokens=n), use_tqdm=False)
# outs[i]["token_ids"]: generated ids only; llm.spec_stats (optional): cumulative speculative-decoding counters
```

Any engine can sit behind a thin `nanovllm/` adapter. Import it and start a lineage:

```bash
python3 orchestrator.py import-engine path/to/engine --tag myengine-src
python3 orchestrator.py --lineage me init --scenario base --from myengine-src --gpu 0 \
    --port-minutes 240 --port-kind msgl --engine-paths "nanovllm python evo_config.json" \
    --engine-note "One paragraph every agent reads about this engine."
```

`--engine-paths` lists what counts as "the engine" (a submission that changes nothing there is not evaluated).
`--port-kind msgl` is the porting prompt used for mini-sglang; write your own in `PORT_PROMPTS` for other engines.

## New skills

Skills are Markdown files with a frontmatter `name` / `description`, under `skills/<version>/<skill>/SKILL.md`.
`evo-workflow/SKILL.md` is the entry point every agent reads first. Write a new version by hand or with
`meta.py evolve`, then compare it against the old one with paired arms (`experiments/e4_skills.sh`) before making it
the default in `skills/CURRENT`.

## A new agent CLI

Codex, Claude Code and DeepSeek Harness are wired in `orchestrator.py`; another CLI needs the same five pieces:

1. **A kind** in `AGENT_KINDS`, its binary and default model in `[agents]` (`<kind>_bin`, `<kind>_model`, defaults in
   `framework/evo/config.py`) and, if it can be used without configuration, a built-in entry in `POLICIES`.
2. **Where its skills go**: `agent_dir(kind)` names the agent-local directory (`.evo`, `.claude`, `.dsh`); add it to
   `LOCAL_ONLY` so it never reaches the GPU host. `make_local` copies the skills to `<dir>/skills` and writes an
   `AGENTS.md` pointer for CLIs that read one.
3. **How it runs**: a branch in `run_agent` (DeepSeek Harness has its own `run_dsh`). The CLI gets the prompt, runs in
   the worktree with `bin/` first on `PATH` and `EVO_LOCAL_WORKTREE` set, and must confine its commands to the
   worktree: the `evo` queue in `.evo/jobs` is the only way out. Wrap the run in `JobExecutor`.
4. **Keys**: never in the agent's environment. If the CLI calls a model API itself, point it at a
   `evo.llm_forward.Forwarder` on loopback (OpenAI `bearer` or Anthropic `x-api-key` auth) and remove the key variable
   from the environment it gets, as `run_dsh` does.
5. **Cost**: `parse_cost(kind, log)` returns `{"usage": {input_tokens, cached_input_tokens, output_tokens}, "turns": n}`
   from the CLI's own event log, or from `Forwarder.usage`.

Check a new CLI end to end before spending GPU time: run it against a fake model endpoint that first asks for one
shell call (`evo board; echo x > ~/probe`) and then answers DONE, with a fake job server answering `.evo/jobs`. The
DeepSeek Harness integration was checked this way with dsh 0.2.2 on Linux, for both `api = "deepseek"` and
`api = "openai"`. The skills appeared in the first request, `evo board` made the round trip through the queue, and
the write outside the worktree was refused (read-only file system). The real key reached only the fake upstream,
and the logs, the patch file and dsh's environment never contained it. Two things to know about dsh's sandbox:
it replaces `/tmp` with an empty directory, so the EvoInfer checkout must not live under `/tmp`; and its shell
tool stops commands after 60 s unless configured (`run_dsh` raises the limit to 40 minutes for `evo eval`).
