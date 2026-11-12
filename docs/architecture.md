# Architecture

EvoInfer treats self-evolution as a harness-design problem. The policy model (the LLM inside Codex, Claude Code or DeepSeek Harness)
stays fixed. What evolves is the inference engine and the knowledge the agents work from. The harness decides what an
agent can do, what feedback it gets, what it knows and how much budget it has.

## Two sides

| | Agent side (`orchestrator.py`, `bin/evo`) | GPU host (`framework/evo/`) |
|---|---|---|
| Runs | agent CLIs, one worktree per episode under `runs/<episode>/engine` | benchmarks, judges, `evo run` commands |
| State | transcripts, agent worktrees | `<home>/engine.git`, `records/`, `slots/`, `judge_engine/`, prompt sets |
| Talks to the other side | `sh()` / `rsync()` (local shell, or SSH when `[host] mode = "ssh"`) | nothing; it only answers |

Agents never talk to the GPU host directly. `bin/evo` writes each call to `<worktree>/.evo/jobs/<id>.req` and waits.
The orchestrator's `JobExecutor` (one per episode) then:

1. mirrors the worktree to the GPU-host copy `<home>/agents/<episode>/engine`;
2. runs `python -m evo.cli` there with the episode's settings (GPU, budget, parent result, lineage), which the agent
   cannot change;
3. writes the output and exit code back.

So the agent's sandbox needs only its worktree: no network, no SSH keys, no API keys of the GPU host.

## An episode

`orchestrator.py episode --gpu G --focus F` (or `evolve`, which runs several):

1. **Worktree.** A clone of the archive at the lineage's current best (or `--parent TAG`), plus the skills of the
   chosen version (`.evo/skills/` for Codex with an `AGENTS.md` pointer, `.claude/skills/` for Claude Code,
   `.dsh/skills/` plus `AGENTS.md` for DeepSeek Harness). Skills and
   the job queue are agent-side only and never reach the evaluated tree.
2. **Prompt.** The parent's measured performance, the scenario's goal, the focus, the coordination rules and the budget.
3. **The agent works.** It profiles with `evo run`, checks candidates with `evo eval` (quick mode: 6 dev prompts,
   compared with the parent's stored quick result), claims the functions it will edit on the board, and finishes with
   `evo submit -m "<hypothesis> | <change> | <measured effect> | <next idea>"`.
4. **Final evaluation** (`measure`). The orchestrator runs it, not the agent.
   - A submission that leaves the engine files (`nanovllm/`, `evo_config.json`) unchanged is recorded as `no_change`
     without an evaluation.
   - Otherwise the parent is re-timed on the GPU, and the candidate is evaluated on the same GPU right after it:
     all dev and holdout prompts, with the judge.
   - When the verdict would flip within ±1 standard error of the paired noise, it measures again, at most 3 times
     in total, and decides on the mean.
5. **Verdict and bookkeeping.** One row in `records/experiments.jsonl`. An accepted version is promoted to the next tag
   (`evo-NNN` or `<lineage>-NNN`). The promotion is atomic per lineage.

## The fitness function

**Correctness** (`evo.judge`, `evo.judge_hf`, `evo.judge_gate`).

1. The candidate's prompt and output are run through the frozen reference: the seed engine for Qwen3.5, the
   transformers implementation for other models.
2. For every generated position the judge computes the gap `g = max logit − logit(emitted token)`.
3. The gap is classified with the gate calibrated from legitimate numerical noise:

| Class | Gap | Limit |
|---|---|---|
| near-tie flip | `0 < g ≤ 0.5` | ≤ 2% of tokens |
| soft violation | `0.5 < g ≤ 1.0` | ≤ 0.1% of tokens |
| violation | `g > 1.0` | none |

Because every position is scored on the candidate's own prefix, one legitimate early flip does not cascade into
failures. Exact-match testing would reject 422 of 423 correct engines; task accuracy would miss errors that only
change one token.

**Speed** (`evo.evaluate`, `evo.bench_runner`).

- The harness times `generate()` itself. Decode time is total time minus the warmer of two bracketing 1-token runs.
- Every prompt gets a random nonce, so nothing can be cached.
- The timings must fit inside the externally measured wall clock.
- The paired gain is `Δ = m(candidate) / m(parent re-timed just before) − 1` on dev and on holdout. On a shared
  machine unpaired comparisons mis-judged 27% of null changes; paired ones 1 in 11.

**Verdict** (`verdict_of`, `noise_threshold` = 3% by default):

| verdict | condition |
|---|---|
| `rejected_correctness` | the judge failed |
| `rejected_regression` | any metric fell by more than the threshold on dev or holdout |
| `accepted` | a primary metric of the scenario gained more than the threshold on dev **and** holdout |
| `pooled` | correct and > 0.5% on both, but below the threshold |
| `no_gain` | otherwise |
| `accepted_sibling` | accepted, but another episode of the same round was promoted first |

## Population level

- **Recombination** (`integrate`, after each round). Accepted siblings are 3-way merged into the new best, largest
  gain first, and re-evaluated. On a conflict, a short merge episode (`--focus merge`, 30 minutes, 3 evaluations) reads
  the sibling's diff and report and ports only what is complementary.
- **Sub-threshold accumulation** (`bundle`). Pooled changes are merged together and evaluated once as a bundle. If the
  bundle is accepted, all of its members enter the line; otherwise each member uses up one of 2 attempts.
- **Coordination** (`evo.board`). Before implementing, an agent claims `path:function` regions. Overlapping claims are
  refused. Accepted siblings waiting for integration keep standing claims on the functions they changed
  (`evo.diffsyms`), so nobody redoes them.

## Meta level

`meta.py evolve --base v0 --new v1` gives a Codex meta-agent a digest of every episode: messages, commands, `evo`
outputs, edits, every evaluation and verdict, and the board. It edits a copy of the skills under four rules:

- every change cites evidence;
- no untested recipes;
- the fitness function stays out of reach;
- the total size of the skills is capped.

Compare skill versions with paired episodes from one parent (`episode --skills v1 --arm skills-v1`, see
`experiments/e4_skills.sh`). Arms have their own board and history and are never promoted.

## Lineages and scenarios

A lineage (`--lineage NAME`) is an independent evolution with its own archive (`records/archive-NAME.json`), tags,
board and history. `init` creates it from any version or engine tag. It commits optional deployment settings, runs an
optional porting agent (`--port-minutes`, `--port-kind edge|msgl|model`), evaluates the root with the judge, and records
it as `NAME-000`. Scenarios are defined in `framework/evo/scenarios.py`; see [extending.md](extending.md).

## Where the state lives (GPU host, `<home>`)

| Path | Content |
|---|---|
| `engine.git` | bare repo, every version as a tag (`baseline`, `paper-*`, `evo-NNN`, `<lineage>-NNN`) |
| `records/experiments.jsonl` | one row per episode / merge / bundle, including verdicts, gains, cost and summary |
| `records/archive*.json` | per lineage: accepted versions, current best, integrated and pooled bookkeeping |
| `records/evals/*.json` | every evaluation result |
| `records/board*.jsonl` | coordination boards |
| `slots/<eval>/` | logs, bench and judge outputs of each evaluation |
| `agents/<episode>/engine` | the evaluated tree of each episode |
