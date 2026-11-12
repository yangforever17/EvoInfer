#!/usr/bin/env python3
"""EvoInfer orchestrator: evolves an LLM inference engine with coding agents.

One episode = one agent (Codex or Claude Code) improving the current best engine of a lineage within a budget. After
the agent stops, the orchestrator runs the paired final evaluation (dev + hidden holdout, correctness judge) and
decides: accepted versions become the new best, accepted siblings are merged at the end of a round, and correct
changes below the noise threshold are pooled and bundled.

  python3 orchestrator.py doctor                    # check config, GPU host, models, agent CLIs
  python3 orchestrator.py setup                     # create the engine archive, judge and prompt sets on the GPU host
  python3 orchestrator.py init --scenario base --from baseline --gpu 0    # evaluate the seed: the main line's root
  python3 orchestrator.py evolve --rounds 3 --gpus 0,1                    # rounds of 2 parallel episodes + integrate
  python3 orchestrator.py episode --gpu 0 --focus overhead                # a single episode
  python3 orchestrator.py status

Agents always run on this machine, each in its own worktree under runs/<episode>/engine. They reach the GPU host only
through the `evo` command (bin/evo), which hands each call to this process (JobExecutor) through a queue inside the
worktree. So agents need neither network access nor credentials, and their sandbox can stay limited to the worktree.
The GPU host is this machine ([host] mode = "local") or another one reached over SSH (mode = "ssh"); see
evoinfer.example.toml and docs/configuration.md.
"""
import argparse, base64, glob, json, os, shlex, shutil, subprocess, sys, threading, time, uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "framework"))
from evo import config as C   # noqa: E402  (stdlib only: safe to import on the agent side)

CFG = C.CFG
MODE = CFG["host"]["mode"]
if MODE not in ("local", "ssh"):
    raise SystemExit(f"[host] mode must be 'local' or 'ssh', not {MODE!r}")
REMOTE = C.home(expand=(MODE == "local"))        # EvoInfer home on the GPU host ("~" is kept for remote shells)
REMOTE_SH = REMOTE.replace("~", "$HOME", 1) if REMOTE.startswith("~") else REMOTE   # same, for stored paths
_ssh_cfg = os.path.expanduser(CFG["host"]["ssh_config"]) if CFG["host"]["ssh_config"] else ""
SSH = (["ssh"] + (["-F", _ssh_cfg] if _ssh_cfg else []) + ["-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=20",
       CFG["host"]["ssh_host"]]) if MODE == "ssh" else []
RSYNC_E = (f"ssh {'-F ' + shlex.quote(_ssh_cfg) + ' ' if _ssh_cfg else ''}-o ServerAliveInterval=30") if MODE == "ssh" else ""
PY = CFG["host"]["python"] or ("venv/bin/python" if MODE == "ssh" or os.path.exists(os.path.join(REMOTE, "venv/bin/python")) else sys.executable)
if MODE == "local":
    PY = os.path.expanduser(PY)
LOCAL_ONLY = [".claude", ".evo", "AGENTS.md", ".git"]  # agent-side only: skills, job queue, notes (the GPU-host tree keeps its own git history)
RUNS = os.path.join(HERE, "runs")
SKILL_VERSIONS = os.path.join(HERE, "skills")   # v0, v1, ... ; CURRENT names the default
AG = CFG["agents"]


def remote_argv(cmd):
    """argv that runs a shell command on the GPU host."""
    return SSH + [cmd] if MODE == "ssh" else ["bash", "-c", cmd]


def rp(path):
    """A GPU-host path as an rsync endpoint."""
    return f"{CFG['host']['ssh_host']}:{path}" if MODE == "ssh" else os.path.expanduser(path)


def rsync(*args, check=True):
    return subprocess.run(["rsync", "-az"] + (["-e", RSYNC_E] if RSYNC_E else []) + list(args), check=check)


def shq(path):
    """Shell-quote a GPU-host path, keeping a leading ~ or $HOME expandable."""
    for pre in ("~/", "$HOME/"):
        if path.startswith(pre):
            return '"$HOME"/' + shlex.quote(path[len(pre):])
    return shlex.quote(path)


EVO_PY = f"PYTHONPATH=framework {shq(PY)}"   # framework commands, run with cwd = REMOTE on the GPU host


# A policy (--policy) = an agent CLI plus the model behind it; the harness (actions through `evo`, feedback, skills,
# budget) is the same for all of them. "codex" and "claude" use the CLIs' own login and default (or [agents]) model.
# More policies come from [policies.<name>] in evoinfer.toml - e.g. Claude Code against an Anthropic-compatible
# gateway serving another model, with the endpoint and key read from environment variables (never from a file here).
POLICIES = {"codex": {"cli": "codex"}, "claude": {"cli": "claude"}}
POLICIES.update({k: dict(v) for k, v in CFG.get("policies", {}).items()})
POLICY = ""       # --policy; "" = whatever --agent says with that CLI's default model


def policy_model(pol, kind):
    return pol.get("model") or (AG["codex_model"] if kind == "codex" else AG["claude_model"]) or None


def gateway_env(pol):
    """Environment that points Claude Code at the policy's gateway (values are never printed or logged)."""
    base, key = os.environ.get(pol["base_url_env"]), os.environ.get(pol["api_key_env"])
    if not base or not key:
        raise SystemExit(f"policy needs ${pol['base_url_env']} and ${pol['api_key_env']} in the environment")
    small = pol.get("small_model", pol["model"])
    proxy = os.environ.get(pol["proxy_env"]) if pol.get("proxy_env") else None
    via = {"HTTPS_PROXY": proxy, "https_proxy": proxy} if proxy else {}
    return {**via, "ANTHROPIC_BASE_URL": base, "ANTHROPIC_AUTH_TOKEN": key, "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_MODEL": pol["model"], "ANTHROPIC_DEFAULT_OPUS_MODEL": pol["model"], "ANTHROPIC_DEFAULT_SONNET_MODEL": pol["model"],
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": small, "ANTHROPIC_SMALL_FAST_MODEL": small,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "API_TIMEOUT_MS": "600000"}


FOCUS = {
    "moe": "Focus on the MoE layers (routing + expert computation) on the decode/verification path.",
    "spec": "Focus on speculative decoding: MTP drafting cost, verification, linear-attention state rollback, draft depth.",
    "overhead": "Focus on host/launch overhead of the decode step (syncs, Python loops, CUDA graphs).",
    "free": "Choose whatever you believe is the largest remaining bottleneck - measure first.",
    # long-context lineage
    "prefill": "Focus on time-to-first-token of the 8K-32K-token prompts: prefill attention, chunked DeltaNet, MoE over many tokens, memory.",
    "longdecode": "Focus on decode with a long history: full-attention layers over 8K-32K cached tokens, KV-cache layout, MTP verification at long context.",
    # edge lineage (one 32 GB consumer GPU, 4-bit routed experts)
    "quant": "Focus on the 4-bit routed-expert path: dequantization cost, fused 4-bit GEMV/GEMM kernels for decode, verification and prefill on the edge GPU.",
    "edgemem": "Focus on the 32 GB memory budget of the edge GPU: where memory and bandwidth go (weights, KV/state, workspaces, CUDA graphs) and what that costs in speed.",
    # concurrency lineage
    "batching": "Focus on batching and scheduling: how many sequences run together, the speculative-decoding policy under batching, CUDA graphs for larger batches.",
    "batchmoe": "Focus on MoE and dense-layer kernels when 8-32 sequences decode together (many more tokens per expert than at batch 1).",
    "merge": "A sibling episode improved the same ancestor independently; its diff and report are in {sib_dir}/. "
             "Port the parts that are complementary to the current engine (do not re-apply what the parent already does "
             "differently), measure each part, and keep only what helps.",
}


def sh(cmd, check=True, retries=3, **kw):
    """Run a shell command on the GPU host (cwd: home); retried on SSH connection failures."""
    for i in range(retries):
        p = subprocess.run(remote_argv(cmd), capture_output=True, text=True, **kw)
        if p.returncode == 0 or not check:
            return p.stdout
        if MODE != "ssh" or p.returncode != 255:   # the command failed (not a connection problem)
            raise RuntimeError(f"GPU-host command failed ({p.returncode}): {cmd}\n{p.stderr[-2000:]}")
        time.sleep(10 * (i + 1))
    raise RuntimeError(f"ssh failed: {cmd}\n{p.stderr[-2000:]}")


def remote_json(path):
    return json.loads(sh(f"cat {path}"))


def current_skills():
    return os.environ.get("EVO_SKILLS") or open(os.path.join(SKILL_VERSIONS, "CURRENT")).read().strip()


LIN = ""          # engine lineage (--lineage): its own archive, tags, board and history; "" = the main line
SCEN = "base"     # deployment scenario the lineage is evolved for (framework/evo/scenarios.py)
PRIMARY = {"base": ("decode", "e2e"), "lc": ("e2e",), "cc": ("e2e",), "edge": ("decode", "e2e"),   # metrics a change must improve (mirrors scenarios.py)
           "olmoe": ("decode", "e2e"), "gemma4": ("decode", "e2e")}
# Other target models. Each is a scenario of its own in framework/evo/scenarios.py ("model", "judge": "hf").
MODEL_INFO = {
    "olmoe": {"name": "OLMoE-1B-7B-0125-Instruct", "ctx": "about 3.6K",
              "desc": "It is a 16-layer decoder-only transformer (hidden size 2048): standard causal self-attention with 16 heads and "
                      "rotary position embeddings, RMSNorm, and in every layer a 64-expert top-8 mixture-of-experts MLP (SiLU, expert "
                      "width 1024); 50K vocabulary, 4096-token context. It has no draft head.",
              "ref": ["olmoe"]},
    "gemma4": {"name": "Gemma 4 26B-A4B-it (text path only)", "ctx": "about 5K",
               "desc": "It is a 30-layer decoder-only transformer (hidden size 2816) that alternates sliding-window attention layers "
                       "(window 1024) with full-attention layers, uses per-layer-type rotary embeddings, RMSNorm, GELU MLPs with a "
                       "128-expert top-8 mixture-of-experts block, tied embeddings and final-logit soft-capping; 262K vocabulary. Only "
                       "the text path is needed (prompts are token ids; the vision tower is not used). It has no draft head.",
               "ref": ["gemma4"]},
}
arch_file = lambda: f"archive-{LIN}.json" if LIN else "archive.json"
tag_prefix = lambda: LIN or "evo"
in_line = lambda r: (r.get("lineage") or "") == LIN and (not r.get("arm") or r.get("arm") == LIN)


def board_event(ev, arm=None):
    arm = arm or LIN
    ev = dict(ev, ts=time.time())
    f = f"board-{arm}.jsonl" if arm else "board.jsonl"
    subprocess.run(remote_argv(f"cat >> {REMOTE}/records/{f}"), input=json.dumps(ev) + "\n", text=True, check=False)


def append_row(row):
    if LIN:
        row = dict(row, lineage=LIN, scenario=SCEN, arm=row.get("arm") or LIN)
    subprocess.run(remote_argv(f"cat >> {REMOTE}/records/experiments.jsonl"), input=json.dumps(row) + "\n", text=True, check=True)


QUEUE_STATUS = {"accepted": "ACCEPTED, waiting to be merged into the best engine",
                "pooled": "POOLED (correct, gain below the bar alone), waiting to be bundled with other pooled changes"}


def queue_claim(eid, parent_commit, commit, summary, kind="accepted", repo=None):
    """Against cross-generation duplication: an accepted sibling (or a pooled change) stays on the board as a
    standing claim on the functions it changed until it is merged, so new agents neither redo it nor edit the same code."""
    try:
        where = json.loads(sh(f"cd {REMOTE} && {EVO_PY} -m evo.diffsyms {REMOTE}/agents/{repo or eid}/engine {parent_commit} {commit}"))
    except Exception as e:
        print(f"[{eid}] queue claim skipped: {e}", flush=True); return
    board_event({"type": "claim", "agent": f"queue:{eid}", "cid": "q" + eid[-4:],
                 "hypothesis": f"{QUEUE_STATUS[kind]}: {summary[:400]}", "where": where})
    print(f"[{eid}] standing claim on {len(where)} regions", flush=True)


def archive():
    return json.loads(sh(f"cat {REMOTE}/records/{arch_file()}"))


SCEN_GOAL = {
    "base": """Goal: lower single-request greedy latency - decode tokens/s and end-to-end tokens/s (prefill + 256 new tokens) -
while keeping outputs correct (the judge is strict). A change is accepted if it clearly improves one of the two
on both dev and the hidden holdout set without regressing the other.""",
    "lc": """Deployment scenario: LONG CONTEXT. One request at a time, greedy, 8K-32K-token prompts (source code to review, many-shot
math) and up to 1024 new tokens. Goal: lower end-to-end latency - end-to-end tokens/s (prefill + decode of the whole
request) - while keeping outputs correct (the judge is strict). Time-to-first-token of the long prompt and decode over
a long KV history both count. A change is accepted if end-to-end tokens/s improves on both dev and the hidden
holdout set without regressing decode tokens/s. Short-prompt performance is not part of this scenario.""",
    "cc": """Deployment scenario: CONCURRENT REQUESTS. 32 requests (short-to-medium prompts, 256 new tokens each, greedy) are
submitted at the same time. Goal: higher serving throughput - total generated tokens / wall time of the batch - while
keeping every request's output correct (the judge is strict, per request). Engine settings in evo_config.json (e.g.
max_num_seqs, num_speculative_tokens) and the scheduler are in scope. A change is accepted if throughput improves on
both dev and the hidden holdout batch. Single-request latency is not part of this scenario.""",
    "edge": """Deployment scenario: EDGE DEVICE. One consumer GPU with 32 GB (e.g. an RTX 5090, PCIe), serving one request at a
time, greedy, short-to-medium prompts and 256 new tokens. The model does not fit in BF16, so the routed MoE experts are
stored in 4-bit (format: the `quantization_config` in the model's config.json and evo.quantize_int4 in the framework);
everything else is BF16. `evo eval` and `evo run` execute on the edge GPU (EVO_MODEL points at the 4-bit model). Correctness is
judged against the model with exactly those 4-bit expert values (the judge is strict). Goal: lower single-request
latency - decode tokens/s and end-to-end tokens/s - on the edge GPU. A change is accepted if it clearly improves one of the
two on both dev and the hidden holdout set without regressing the other.""",
}
for _k, _m in MODEL_INFO.items():
    SCEN_GOAL[_k] = f"""Target model of this lineage: {_m['name']}. One request at a time, greedy, short-to-medium prompts (up to
{_m['ctx']} tokens) and 256 new tokens. Goal: lower single-request latency - decode tokens/s and end-to-end tokens/s (prefill +
256 new tokens) - while keeping outputs correct: every generated token is judged strictly against the Hugging Face
transformers implementation of the model in BF16. A change is accepted if it clearly improves one of the two on both
dev and the hidden holdout set without regressing the other."""


def parent_line(pm):
    d, f = pm["dev"], lambda x, k=".2f": "n/a" if x is None else format(x, k)
    if SCEN == "cc":
        return f"Parent on the final evaluation: batch throughput dev {f(d['e2e_tok_s'])} tok/s, holdout {f(pm['holdout']['e2e_tok_s'])} tok/s\n" \
               f"(MTP accept rate {d.get('mtp_accept_rate')}, tokens/step {d.get('tokens_per_step')})."
    return f"""Parent performance on the final evaluation: decode dev {f(d['decode_tok_s'])} tok/s, holdout {f(pm['holdout']['decode_tok_s'])} tok/s;
end-to-end incl. prefill dev {f(d.get('e2e_tok_s'))} tok/s, holdout {f(pm['holdout'].get('e2e_tok_s'))} tok/s; mean TTFT dev {f(d.get('mean_ttft_s'))}s
(MTP accept rate {d.get('mtp_accept_rate')}, tokens/step {d.get('tokens_per_step')})."""


def engine_desc():
    if ENGINE_PATHS == DEFAULT_ENGINE_PATHS:
        return "nanovllm/ and evo_config.json"
    return "the engine (" + ", ".join(ENGINE_PATHS.split()) + ")"


def engine_note():
    return "\n" + ENGINE_NOTE if ENGINE_NOTE else ""


def build_prompt(eid, kind, wt, parent, focus, max_evals, minutes):
    pm = parent["result"]["metrics"]
    skills_dir = ".claude/skills" if kind == "claude" else ".evo/skills"
    return f"""You are an optimization agent in EvoInfer, episode {eid}.
Your working directory is a git worktree of an LLM inference engine (branch agent/{eid}, parent commit {parent['commit'][:10]}).
Before anything else read the skills in {skills_dir}/ - start with {skills_dir}/evo-workflow/SKILL.md - and run `evo history`.

{parent_line(pm)}
{SCEN_GOAL[SCEN]} A correct change with a smaller gain that is
positive on both sets is not lost: it is pooled and later evaluated bundled with other pooled changes, so submit
such a change instead of reverting it. A submission that leaves {engine_desc()} unchanged is
recorded without evaluation.{engine_note()}
{FOCUS[focus].format(sib_dir=skills_dir.split('/')[0] + '/sibling')}
Other agents are optimizing the same engine in parallel right now. Coordinate through the board:
- `evo board` early (after `evo history`) and again whenever `evo eval` says there are new entries;
- before implementing a change, `evo claim -m "<hypothesis + planned change>" --where path:function ...` naming every
  function/class you will edit. A refused claim means another agent is already doing it: choose another bottleneck
  rather than duplicating it (duplicates are wasted and conflict at merge time);
- `evo note -m "..."` for measurements others can reuse (e.g. a profile breakdown); `evo release -m "<why>"` when you
  drop an idea, so nobody repeats it.
Budget: at most {max_evals} `evo eval` runs and about {minutes} minutes. GPU work only through `evo eval` / `evo run`.
Finish with `evo submit -m "<hypothesis> | <change> | <measured effect> | <next idea>"` even if nothing helped
(a well-documented negative result is valuable to the next agent)."""


def setup_worktree(eid, kind, parent, skills):
    rwt = f"{REMOTE}/agents/{eid}/engine"
    sh(f"rm -rf {rwt} && git clone -q {REMOTE}/engine.git {rwt} && cd {rwt} && git checkout -q -b agent/{eid} {parent['commit']}"
       f" && printf '.claude/\\n.evo/\\nAGENTS.md\\n' >> .git/info/exclude")
    return make_local(eid, kind, rwt, skills)


def make_local(eid, kind, rwt, skills):
    """Mirror the GPU-host worktree to this machine and add the skills (agent-local, never shipped back)."""
    local = os.path.join(RUNS, eid, "engine")
    os.makedirs(os.path.dirname(local), exist_ok=True)
    rsync(rp(f"{rwt}/"), local + "/")
    src = os.path.join(SKILL_VERSIONS, skills)
    ignore = shutil.ignore_patterns("CHANGELOG.md")       # the meta-agent's rationale is for the next meta-agent, not for workers
    if kind == "claude":
        shutil.copytree(src, os.path.join(local, ".claude", "skills"), dirs_exist_ok=True, ignore=ignore)
    else:
        shutil.copytree(src, os.path.join(local, ".evo", "skills"), dirs_exist_ok=True, ignore=ignore)
        with open(os.path.join(local, "AGENTS.md"), "w") as f:
            f.write("# EvoInfer agent notes\nSkills for this task live in `.evo/skills/*/SKILL.md`; read "
                    "`.evo/skills/evo-workflow/SKILL.md` first. GPU work only through the `evo` command.\n")
    return local


def add_sibling(local, kind, sib_eid):
    """Copy a sibling episode's diff (against its own parent) and report into the agent-local area."""
    row = [json.loads(l) for l in sh(f"grep '\"id\": \"{sib_eid}\"' {REMOTE}/records/experiments.jsonl").splitlines()][-1]
    d = os.path.join(local, ".claude" if kind == "claude" else ".evo", "sibling"); os.makedirs(d, exist_ok=True)
    rwt = f"{REMOTE}/agents/{sib_eid}/engine"
    open(os.path.join(d, "sibling.patch"), "w").write(sh(f"cd {rwt} && git diff {row['parent']} {row['commit']}"))
    for f in sh(f"cd {rwt} && git ls-files '*.md' | grep -iv readme || true").split():
        open(os.path.join(d, os.path.basename(f)), "w").write(sh(f"cat {rwt}/{f}"))
    open(os.path.join(d, "summary.json"), "w").write(json.dumps({k: row.get(k) for k in ("id", "summary", "dev_tok_s", "holdout_tok_s",
                                                                                     "e2e_speedup", "correctness", "commit")}, indent=1))


def parse_cost(kind, log):
    if kind == "claude":
        cost = {"usd": None, "turns": None}
        seen = {}
        for line in open(log, errors="replace"):
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if not isinstance(ev, dict):
                continue
            if ev.get("type") == "assistant":
                m = ev.get("message") or {}
                if isinstance(m, dict) and isinstance(m.get("usage"), dict):
                    seen[m.get("id")] = m["usage"].get("input_tokens") or 0
            if ev.get("type") == "result":
                cost = {"usd": ev.get("total_cost_usd"), "turns": ev.get("num_turns"), "usage": ev.get("usage"),
                        "models": list(ev.get("modelUsage") or {})}
        if cost.get("turns") is None and seen:
            # killed at the time limit: the CLI never wrote its totals. The stream shows only the main thread's
            # input tokens (about a quarter of what the CLI reports for finished episodes), so this is a lower bound.
            cost.update(usage=None, messages=len(seen), partial_input_tokens=sum(seen.values()))
        return cost
    usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    turns = tools = 0
    for line in open(log, errors="replace"):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict):
            continue
        if ev.get("type") == "turn.completed" and ev.get("usage"):
            turns += 1
            for k in usage:
                usage[k] += ev["usage"].get(k, 0)
        if ev.get("type") == "item.completed" and (ev.get("item") or {}).get("type") == "command_execution":
            tools += 1
    return {"usage": usage, "turns": max(turns, tools)}


def remote_worktree(eid):
    return f"{REMOTE}/agents/{eid}/engine"


def sync_worktree(eid, local):
    """Mirror the agent's worktree to its GPU-host copy (agent-side files - skills, job queue, git - stay here)."""
    excl = sum((["--exclude", x] for x in LOCAL_ONLY + ["__pycache__"]), [])
    rwt = remote_worktree(eid)
    if MODE == "local":
        os.makedirs(rwt, exist_ok=True)
    else:
        excl.append(f"--rsync-path=mkdir -p {shq(rwt)} && rsync")
    rsync("--delete", *excl, local + "/", rp(rwt + "/"), check=False)


def gpu_python():
    """Absolute path of the GPU host's Python, for running it without a shell (local mode)."""
    return PY if os.path.isabs(PY) or "/" not in PY else os.path.join(REMOTE, PY)


class JobExecutor:
    """Serves an agent's `evo` calls outside the agent's sandbox. bin/evo writes <worktree>/.evo/jobs/<id>.req (the
    arguments, NUL-separated and base64-encoded) and waits for <id>.rc; the executor mirrors the worktree to the GPU host,
    runs `python -m evo.cli` there with the episode's settings (which the agent cannot change) and writes <id>.out and
    <id>.rc. Each call runs in its own thread, so a long `evo eval` does not block `evo board`."""

    def __init__(self, eid, local, settings):
        self.eid, self.local, self.settings = eid, local, settings
        self.jobs = os.path.join(local, ".evo", "jobs"); os.makedirs(self.jobs, exist_ok=True)
        self.stop_ev, self.sync_lock = threading.Event(), threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self):
        self.thread.start(); return self

    def __exit__(self, *exc):
        self.stop_ev.set(); self.thread.join(timeout=5)

    def _loop(self):
        while not self.stop_ev.is_set():
            for req in sorted(glob.glob(os.path.join(self.jobs, "*.req"))):
                taken = req[:-4] + ".taken"
                try:
                    os.rename(req, taken)
                except OSError:
                    continue
                threading.Thread(target=self._handle, args=(taken,), daemon=True).start()
            self.stop_ev.wait(1.0)

    def _handle(self, taken):
        jid = os.path.basename(taken)[:-6]
        argv_b64 = open(taken).read().strip()
        try:
            with self.sync_lock:
                sync_worktree(self.eid, self.local)
            out, rc = self._run(jid, argv_b64)
        except Exception as e:   # never leave the agent waiting
            out, rc = f"[evo] internal error: {e}\n", 70
        for ext, data in (("out", out), ("rc", f"{rc}\n")):   # .rc last: bin/evo reads .out once .rc exists
            tmp = os.path.join(self.jobs, f"{jid}.{ext}.tmp")
            open(tmp, "w").write(data); os.rename(tmp, os.path.join(self.jobs, f"{jid}.{ext}"))

    def _env(self, argv_b64):
        e = dict(self.settings, EVO_AGENT_ID=self.eid, EVO_ARGV_B64=argv_b64, EVO_WORKTREE=remote_worktree(self.eid))
        return {k: str(v) for k, v in e.items() if v is not None}

    def _run(self, jid, argv_b64):
        e = self._env(argv_b64)
        if MODE == "local":
            e["EVO_PARENT_RESULT"] = os.path.expandvars(e.get("EVO_PARENT_RESULT", ""))
            p = subprocess.run([gpu_python(), "-m", "evo.cli"], cwd=REMOTE, capture_output=True, text=True,
                               env=dict(os.environ, PYTHONPATH=os.path.join(REMOTE, "framework"), **e))
            return p.stdout + p.stderr, p.returncode
        # ssh: run detached on the GPU host and poll, so an SSH drop during a long evaluation loses nothing
        jd = f"{REMOTE}/agents/{self.eid}/jobs"
        assign = " ".join(f"{k}={shq(v) if k in ('EVO_WORKTREE', 'EVO_PARENT_RESULT') else shlex.quote(v)}" for k, v in e.items())
        sh(f"mkdir -p {shq(jd)} && cd {shq(REMOTE)} && setsid nohup env {assign} PYTHONPATH=framework "
           f"bash -c '{PY.replace('~', '$HOME', 1)} -m evo.cli > {jd}/{jid}.out 2>&1; echo $? > {jd}/{jid}.rc' > /dev/null 2>&1 < /dev/null &")
        while True:
            time.sleep(10)
            rc = sh(f"cat {jd}/{jid}.rc 2>/dev/null || true", check=False).strip()
            if rc:
                return sh(f"cat {jd}/{jid}.out", check=False), int(rc)


def run_agent(kind, eid, local, gpu, parent_result, prompt, max_evals, minutes, arm=""):
    settings = dict(EVO_ARM=arm or LIN, EVO_SCENARIO=SCEN, EVO_GPU=gpu, EVO_PARENT_RESULT=parent_result,
                    EVO_MAX_EVALS=max_evals, EVO_EPISODE_START=int(time.time()), EVO_ISOLATED=os.environ.get("EVO_ISOLATED", ""))
    env = dict(os.environ, EVO_AGENT_ID=eid, EVO_LOCAL_WORKTREE=local, PATH=os.path.join(HERE, "bin") + os.pathsep + os.environ["PATH"])
    env.pop("CLAUDECODE", None)
    pol = POLICIES.get(POLICY, {})
    if pol.get("base_url_env"):
        env.update(gateway_env(pol))
    model = policy_model(pol, kind)
    if kind == "claude":
        # `evo eval` on a slow engine runs far longer than Claude Code's default 2-minute shell limit
        env.update(BASH_DEFAULT_TIMEOUT_MS="2400000", BASH_MAX_TIMEOUT_MS="3600000")
        allowed = ["Read", "Edit", "Write", "Glob", "Grep", "Skill", "TodoWrite", "Bash(evo:*)", "Bash(git status:*)",
                   "Bash(git diff:*)", "Bash(git log:*)", "Bash(git show:*)", "Bash(git checkout:*)", "Bash(git stash:*)",
                   "Bash(ls:*)", "Bash(cat:*)", "Bash(head:*)", "Bash(tail:*)", "Bash(wc:*)", "Bash(grep:*)", "Bash(rg:*)",
                   "Bash(python3 -c:*)"]
        log = os.path.join(RUNS, eid, "claude.jsonl")
        cmd = [AG["claude_bin"], "-p", prompt, "--output-format", "stream-json", "--verbose",
               "--allowedTools", " ".join(allowed), "--permission-mode", "dontAsk"] + (["--model", model] if model else [])
    else:
        # sandboxed: writes limited to the worktree (+tmp); `evo` reaches the GPU host through the job queue in .evo/jobs
        log = os.path.join(RUNS, eid, "codex.jsonl")
        cmd = [AG["codex_bin"], "exec", "--json", "--sandbox", "workspace-write", "--skip-git-repo-check", "-C", local, prompt]
        if model:
            cmd[2:2] = ["-m", model]
    t0 = time.time()
    with JobExecutor(eid, local, settings), open(log, "w") as f:
        try:
            subprocess.run(cmd, cwd=local, env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                           timeout=minutes * 60 + 600)
        except subprocess.TimeoutExpired:
            pass
    sync_worktree(eid, local)
    return parse_cost(kind, log), time.time() - t0


ENGINE_PATHS = "nanovllm evo_config.json"   # what the bench runner loads; tests, notes and scripts do not change the engine
DEFAULT_ENGINE_PATHS = ENGINE_PATHS         # a lineage with another seed engine records its own (archive: engine_paths, engine_note)
ENGINE_NOTE = ""
# Per-metric SD of one paired measurement, estimated from 14 evaluations of unchanged engines (null experiments)
NOISE_SD = {"decode": {"dev": 0.013, "holdout": 0.019}, "e2e": {"dev": 0.011, "holdout": 0.016}}
Z = 1.0              # re-measure while the accept decision flips within +-Z standard errors
MAX_MEASURE = 3      # paired measurements per candidate (the first one also runs the judge)
POOL_MIN = 0.005     # correct changes whose gain is positive on both splits but under the bar go to the pool
MAX_POOL_TRIES = 2   # bundle attempts (failed or conflicting) before a pooled change is dropped


def commit_submission(eid):
    rdir = f"{REMOTE}/agents/{eid}"
    sub = sh(f"cat {REMOTE}/records/agents/{eid}/submission.json 2>/dev/null || true").strip()
    if not sub:   # agent never submitted: commit whatever is in the worktree and evaluate it anyway
        sh(f"cd {rdir}/engine && git add -A && git -c user.name=evo-{eid} -c user.email=evo@local commit -q --allow-empty -m 'auto-commit (no submission)'")
        sub = json.dumps({"commit": sh(f"cd {rdir}/engine && git rev-parse HEAD").strip(), "message": "(no submission)"})
    sub = json.loads(sub)
    # the evaluated tree and the promoted commit must be identical: commit the GPU-host tree as it is now
    msg = shlex.quote(sub["message"])
    sh(f"cd {rdir}/engine && git add -A && (git diff --cached --quiet || git -c user.name=evo-{eid} -c user.email=evo@local commit -q -m {msg})")
    sub["commit"] = sh(f"cd {rdir}/engine && git rev-parse HEAD").strip()
    return sub


def engine_changed(repo_eid, parent_commit, commit="HEAD"):
    out = sh(f"cd {REMOTE}/agents/{repo_eid}/engine && (git diff --quiet {parent_commit} {commit} -- {ENGINE_PATHS} && echo same || echo changed)")
    return out.strip().endswith("changed")


def final_eval(eid, gpu, parent, judge=True):
    rdir = f"{REMOTE}/agents/{eid}"
    sub = commit_submission(eid)
    # paired measurement: re-time the parent on the same GPU right before the candidate (host load drifts by
    # up to ~30% over hours), and compare against that fresh number; the archived parent result is kept as secondary
    pdir = f"{REMOTE}/parents/{parent['commit'][:10]}"
    sh(f"[ -d {pdir} ] || (git clone -q {REMOTE}/engine.git {pdir} && git -C {pdir} checkout -q {parent['commit']})")
    ev = f"cd {REMOTE} && {EVO_PY} -m evo.evaluate --scenario {SCEN}"
    for attempt in range(2):
        # the parent may move to an idle pool GPU; the candidate is then pinned to the GPU the parent was timed on
        pref = json.loads(sh(f"{ev} {pdir} --gpu {gpu} --mode full --no-judge --label {eid}-parentref", timeout=4 * 3600))
        if pref["status"] == "ok":
            ref, cgpu = f"{REMOTE}/records/evals/{pref['eval_id']}.json", f"{pref['gpu']} --strict-gpu"
        else:
            ref, cgpu = parent["result_path"], gpu
        nj = "" if judge else " --no-judge"
        res = json.loads(sh(f"{ev} {rdir}/engine --gpu {cgpu} --mode full --ref {ref} --label {eid}-final{nj}", timeout=4 * 3600))
        if res["status"] not in ("judge_error", "gpu_busy"):
            break
    res["paired_parent_eval"] = pref["eval_id"] if pref["status"] == "ok" else None
    return sub, res


def verdict_of(correct, vs, e2e, thr):
    if not correct:
        return "rejected_correctness"
    if any(min(vs.get(w, -1), e2e.get(w, -1)) < -thr for w in ("dev", "holdout")):   # a metric regressed beyond noise
        return "rejected_regression"
    by = {"decode": vs, "e2e": e2e}
    gains = {w: tuple(by[m].get(w, -1) for m in PRIMARY[SCEN]) for w in ("dev", "holdout")}   # the scenario's target metric(s)
    if all(max(g) > thr for g in gains.values()):
        return "accepted"
    if all(max(g) > POOL_MIN for g in gains.values()):
        return "pooled"         # correct and positive everywhere, below the bar: evaluated later together with others
    return "no_gain"


def decide(res, thr):
    if res["status"] != "ok":
        return res["status"]
    return verdict_of(res["metrics"].get("correctness", {}).get("passed"), res.get("vs_ref", {}), res.get("vs_ref_e2e", {}), thr)


def accept_uncertain(res, n, thr):
    """Would the accept decision flip if every gain moved by Z standard errors of an n-measurement mean?"""
    if res.get("status") != "ok" or not res["metrics"].get("correctness", {}).get("passed"):
        return False
    shift = lambda k, m, s: {w: v + s * Z * NOISE_SD[m][w] / n ** 0.5 for w, v in res.get(k, {}).items() if w in NOISE_SD[m]}
    hi = verdict_of(True, shift("vs_ref", "decode", 1), shift("vs_ref_e2e", "e2e", 1), thr)
    lo = verdict_of(True, shift("vs_ref", "decode", -1), shift("vs_ref_e2e", "e2e", -1), thr)
    return hi == "accepted" and lo != "accepted"


def measure(eid, gpu, parent, thr):
    """Paired final evaluation with two refinements: an engine identical to its parent is not evaluated, and the
    accept decision is sequential - extra paired measurements (no judge, same tree) are taken while the verdict could
    flip within +-Z standard errors, symmetric for gains just under the bar and regressions just past it. The
    decision uses the mean over all measurements; `remeasured` keeps the individual ones."""
    sub = commit_submission(eid)
    if not engine_changed(eid, parent["commit"], sub["commit"]):
        print(f"[{eid}] engine unchanged vs parent ({ENGINE_PATHS}) - no evaluation", flush=True)
        return sub, {"status": "no_change", "eval_id": None}, "no_change"
    sub, res = final_eval(eid, gpu, parent)
    runs = [res]
    while len(runs) < MAX_MEASURE and accept_uncertain(res, len(runs), thr):
        print(f"[{eid}] uncertain after {len(runs)} (dev {res['vs_ref'].get('dev')}, holdout {res['vs_ref'].get('holdout')}, "
              f"e2e {res.get('vs_ref_e2e')}) - re-measuring", flush=True)
        _, r2 = final_eval(eid, gpu, parent, judge=False)
        if r2.get("status") != "ok":
            break
        runs.append(r2)
        mean = lambda k: {w: round(sum(r[k][w] for r in runs) / len(runs), 4) for w in runs[0][k] if all(w in r.get(k, {}) for r in runs)}
        res = dict(runs[0], vs_ref=mean("vs_ref"), vs_ref_e2e=mean("vs_ref_e2e"),
                   remeasured={"evals": [r["eval_id"] for r in runs], "vs_ref": [r["vs_ref"] for r in runs],
                               "vs_ref_e2e": [r["vs_ref_e2e"] for r in runs]})
    verdict = decide(res, thr)
    if len(runs) > 1:
        print(f"[{eid}] after {len(runs)} measurements: dev {res['vs_ref'].get('dev')}, holdout {res['vs_ref'].get('holdout')} -> {verdict}", flush=True)
    return sub, res, verdict


def resolve_parent(arch, tag):
    if not tag:
        return arch["best"]
    return next(e for e in arch["accepted"] if e.get("tag") == tag)


def episode(a):
    arch = archive()
    parent = resolve_parent(arch, getattr(a, "parent", None))
    skills = getattr(a, "skills", None) or current_skills()
    arm = getattr(a, "arm", None) or ""
    eid = f"{time.strftime('%m%d%H%M')}-{POLICY or a.agent}-{a.focus}-{uuid.uuid4().hex[:4]}"
    print(f"[{eid}] parent {parent['commit'][:10]} dev={parent['result']['metrics']['dev']['decode_tok_s']}", flush=True)
    local = setup_worktree(eid, a.agent, parent, skills)
    if SCEN in MODEL_INFO:   # another target model: its transformers source stays at hand for every agent
        add_host_reference(local, SCEN)
    if a.focus == "merge":
        add_sibling(local, a.agent, a.sibling)
    prompt = build_prompt(eid, a.agent, local, parent, a.focus, a.max_evals, a.minutes)
    cost, wall = run_agent(a.agent, eid, local, a.gpu, parent["agent_ref_path"], prompt, a.max_evals, a.minutes, arm)
    print(f"[{eid}] agent finished after {wall / 60:.1f} min, cost {cost}", flush=True)
    board_event({"type": "end", "agent": eid}, arm)
    n_evals = int(sh(f"ls {REMOTE}/records/agents/{eid}/eval-*.json 2>/dev/null | wc -l").strip() or 0)
    if n_evals == 0 and (cost.get("turns") or 0) <= 2:
        if wall >= a.minutes * 60 and (cost.get("messages") or 0) > 2:
            # worked until the hard limit without one evaluation: a failed episode, recorded so that its time counts
            print(f"[{eid}] agent timed out without an evaluation ({cost.get('messages')} messages) - recorded as timeout", flush=True)
            append_row({"id": eid, "agent_kind": a.agent, "policy": POLICY or a.agent,
                        "model": policy_model(POLICIES.get(POLICY, {}), a.agent),
                        "focus": a.focus, "skills_version": skills, "arm": arm or None, "parent_tag": parent.get("tag"),
                        "remeasured": None, "parent": parent["commit"], "commit": None, "summary": "(timed out without an evaluation)",
                        "verdict": "timeout", "dev_speedup": None, "holdout_speedup": None, "e2e_speedup": None,
                        "dev_tok_s": None, "holdout_tok_s": None, "correctness": None, "final_eval": None,
                        "agent_evals": 0, "agent_wall_min": round(wall / 60, 1), "cost": cost, "ended": time.strftime("%F %T")})
            return
        print(f"[{eid}] agent did not run (no evaluations, {cost.get('turns')} turns) - skipping final evaluation", flush=True)
        return
    sub, res, verdict = measure(eid, a.gpu, parent, arch["noise_threshold"])
    promote_ok = not arm and not getattr(a, "no_promote", False)
    # check-and-promote must be atomic per lineage: promote() takes minutes (quick-ref eval) before it updates the
    # archive, so a concurrent episode finishing meanwhile would otherwise also see the old best and collide on the version tag
    lock = lineage_lock() if promote_ok and verdict == "accepted" else None
    try:
        _finish_episode(a, eid, arm, skills, parent, sub, res, verdict, n_evals, cost, wall, promote_ok)
    finally:
        if lock:
            lock.close()


def lineage_lock():
    import fcntl
    f = open(os.path.join(RUNS, f".promote-{LIN or 'main'}.lock"), "w")
    fcntl.flock(f, fcntl.LOCK_EX)
    return f


def _finish_episode(a, eid, arm, skills, parent, sub, res, verdict, n_evals, cost, wall, promote_ok):
    if promote_ok and verdict == "accepted" and archive()["best"]["commit"] != parent["commit"]:
        verdict = "accepted_sibling"   # a concurrent episode already replaced the parent: keep for a later merge episode
    vs = res.get("vs_ref", {})
    row = {"id": eid, "agent_kind": a.agent, "policy": POLICY or a.agent,
           "model": policy_model(POLICIES.get(POLICY, {}), a.agent),
           "focus": a.focus, "skills_version": skills, "arm": arm or None,
           "parent_tag": parent.get("tag"), "remeasured": res.get("remeasured"),
           "parent": parent["commit"], "commit": sub["commit"], "summary": sub["message"], "verdict": verdict,
           "dev_speedup": vs.get("dev"), "holdout_speedup": vs.get("holdout"), "e2e_speedup": res.get("vs_ref_e2e"),
           "dev_tok_s": res.get("metrics", {}).get("dev", {}).get("decode_tok_s"),
           "holdout_tok_s": res.get("metrics", {}).get("holdout", {}).get("decode_tok_s"),
           "correctness": res.get("metrics", {}).get("correctness"), "final_eval": res["eval_id"],
           "agent_evals": n_evals, "agent_wall_min": round(wall / 60, 1), "cost": cost, "ended": time.strftime("%F %T")}
    append_row(row)
    if verdict == "accepted" and promote_ok:
        promote(eid, a.gpu, sub["commit"], res)
    elif verdict == "accepted_sibling":
        queue_claim(eid, parent["commit"], sub["commit"], sub["message"])
    elif verdict == "pooled" and promote_ok:
        queue_claim(eid, parent["commit"], sub["commit"], sub["message"], kind="pooled")
    print(json.dumps(row, indent=1), flush=True)


def promote(eid, gpu, commit, res, n=None):
    rdir = f"{REMOTE}/agents/{eid}"
    n = len(archive()["accepted"]) + 1 if n is None else n
    sh(f"cd {rdir}/engine && git push -q {REMOTE}/engine.git agent/{eid} && git -C {REMOTE}/engine.git tag {tag_prefix()}-{n:03d} {commit}")
    # quick-mode reference for the next agents' `evo eval` comparisons
    q = json.loads(sh(f"cd {REMOTE} && {EVO_PY} -m evo.evaluate --scenario {SCEN} {rdir}/engine --gpu {gpu} --mode quick --label {tag_prefix()}-{n:03d}-quickref", timeout=3 * 3600))
    sh(f"cd {REMOTE}/records && python3 - <<'EOF'\n"
       f"import json\n"
       f"a = json.load(open('{arch_file()}'))\n"
       f"full = json.load(open('evals/{res['eval_id']}.json')); quick = json.load(open('evals/{q['eval_id']}.json'))\n"
       f"ref = dict(full); ref['quick_ref'] = quick\n"
       f"json.dump(ref, open('agent_ref_{tag_prefix()}-{n:03d}.json', 'w'))\n"
       f"entry = {{'tag': '{tag_prefix()}-{n:03d}', 'commit': '{commit}', 'episode': '{eid}', 'result_path': '{REMOTE_SH}/records/evals/{res['eval_id']}.json',"
       f" 'agent_ref_path': '{REMOTE_SH}/records/agent_ref_{tag_prefix()}-{n:03d}.json', 'result': full}}\n"
       f"a['accepted'].append(entry); a['best'] = entry\n"
       f"json.dump(a, open('{arch_file()}', 'w'), indent=1)\nEOF")


def auto_merge(a):
    """Framework-level crossover, no agent: 3-way merge an earlier episode's tree into the current best,
    then the normal paired final evaluation decides. Conflicts are left to a `--focus merge` agent episode."""
    arch = archive(); parent = arch["best"]
    src = a.episode
    srow = [json.loads(l) for l in sh(f"grep '\"id\": \"{src}\"' {REMOTE}/records/experiments.jsonl").splitlines()][-1]
    eid = f"{time.strftime('%m%d%H%M')}-automerge-{uuid.uuid4().hex[:4]}"
    rwt = f"{REMOTE}/agents/{eid}/engine"
    print(f"[{eid}] merging {src} ({srow['commit'][:10]}) into {parent.get('tag')} {parent['commit'][:10]}", flush=True)
    sh(f"mkdir -p {REMOTE}/records/agents/{eid} && git clone -q {REMOTE}/engine.git {rwt} && cd {rwt} && git checkout -q -b agent/{eid} {parent['commit']}"
       f" && git fetch -q {REMOTE}/agents/{src}/engine {srow['commit']}")
    out = subprocess.run(remote_argv(f"cd {rwt} && git -c user.name=evo-{eid} -c user.email=evo@local merge --no-edit -m 'auto-merge {src}' FETCH_HEAD"),
                         capture_output=True, text=True)
    msg = f"auto-merge of {src} into {parent.get('tag')} | {srow['summary']}"
    if out.returncode != 0:
        sh(f"cd {rwt} && git merge --abort", check=False)
        verdict, res, sub = "merge_conflict", {}, {"commit": None, "message": msg}
        print(out.stdout[-1500:], out.stderr[-1500:])
    else:
        subprocess.run(remote_argv(f"cat > {REMOTE}/records/agents/{eid}/submission.json"), input=json.dumps({"commit": "", "message": msg}),
                       text=True, check=True)
        sub, res, verdict = measure(eid, a.gpu, parent, arch["noise_threshold"])
    vs = res.get("vs_ref", {})
    row = {"id": eid, "agent_kind": "none", "focus": "automerge", "source_episode": src, "skills_version": None,
           "remeasured": res.get("remeasured"),
           "parent": parent["commit"], "commit": sub["commit"], "summary": msg, "verdict": verdict,
           "dev_speedup": vs.get("dev"), "holdout_speedup": vs.get("holdout"), "e2e_speedup": res.get("vs_ref_e2e"),
           "dev_tok_s": res.get("metrics", {}).get("dev", {}).get("decode_tok_s"),
           "holdout_tok_s": res.get("metrics", {}).get("holdout", {}).get("decode_tok_s"),
           "correctness": res.get("metrics", {}).get("correctness"), "final_eval": res.get("eval_id"),
           "agent_evals": 0, "agent_wall_min": 0, "cost": None, "ended": time.strftime("%F %T")}
    append_row(row)
    if verdict == "accepted" and archive()["best"]["commit"] == parent["commit"]:
        promote(eid, a.gpu, sub["commit"], res)
    print(json.dumps(row, indent=1), flush=True)
    return verdict, row


def mark_integrated(ids):
    sh(f"cd {REMOTE}/records && python3 - <<'EOF'\n"
       f"import json\na = json.load(open('{arch_file()}'))\n"
       f"a['integrated'] = sorted(set(a.get('integrated', [])) | set({json.dumps(list(ids))}))\n"
       f"json.dump(a, open('{arch_file()}', 'w'), indent=1)\nEOF")


def integrate(a):
    """After a parallel round: bring every accepted sibling into the best line, largest gain first.
    Clean 3-way merge -> framework merge + paired eval; conflict -> small merge-agent episode."""
    rows = [json.loads(l) for l in sh(f"cat {REMOTE}/records/experiments.jsonl").splitlines()]
    latest = {}
    for r in rows:
        latest[r["id"]] = r
    done = set(archive().get("integrated", []))
    sibs = [r for r in latest.values() if r["verdict"] == "accepted_sibling" and r["id"] not in done and in_line(r)]
    sibs.sort(key=lambda r: -max(r.get("dev_speedup") or 0, r.get("holdout_speedup") or 0))
    for r in sibs:
        print(f"[integrate] {r['id']} ({r['focus']}, dev {r.get('dev_speedup')}, holdout {r.get('holdout_speedup')})", flush=True)
        v, mrow = auto_merge(argparse.Namespace(episode=r["id"], gpu=a.gpu))
        outcome = f"auto-merge {mrow['id']}: {v} (dev {mrow.get('dev_speedup')}, holdout {mrow.get('holdout_speedup')})"
        if v == "merge_conflict":
            before = archive()["best"].get("tag")
            episode(argparse.Namespace(agent=a.agent, gpu=a.gpu, focus="merge", sibling=r["id"], max_evals=3, minutes=30))
            after = archive()["best"].get("tag")
            outcome += f"; merge agent {'promoted ' + after if after != before else 'found no gain'}"
        mark_integrated([r["id"]])
        board_event({"type": "release", "agent": f"queue:{r['id']}", "cid": None,
                     "reason": f"{r['id']} integration finished - {outcome}. Best is now {archive()['best'].get('tag')}."})
    bundle(argparse.Namespace(gpu=a.gpu, min=2))


def latest_rows():
    latest = {}
    for l in sh(f"cat {REMOTE}/records/experiments.jsonl").splitlines():
        r = json.loads(l); latest[r["id"]] = r
    return latest


def repo_of(r):
    """Agent tree holding the row's commit (a re-pooled row points back at the episode that produced it)."""
    return r.get("repo_episode") or r["id"]


def pool_score(r):
    e = r.get("e2e_speedup") or {}
    return min(max(r.get("dev_speedup") or 0, e.get("dev", 0)), max(r.get("holdout_speedup") or 0, e.get("holdout", 0)))


def bump_pool_tries(ids):
    sh(f"cd {REMOTE}/records && python3 - <<'EOF'\n"
       f"import json\na = json.load(open('{arch_file()}'))\nt = a.setdefault('pool_attempts', {{}})\n"
       f"for i in {json.dumps(list(ids))}: t[i] = t.get(i, 0) + 1\n"
       f"json.dump(a, open('{arch_file()}', 'w'), indent=1)\nEOF")


def bundle(a):
    """Sub-threshold accumulation: 3-way merge every pooled change (correct, positive on both splits, below the bar
    alone) into the current best, largest gain first, skipping the ones that conflict, and evaluate the bundle once
    with the normal paired, sequential measurement. Accepted -> promoted, members integrated; otherwise every member
    used one of its MAX_POOL_TRIES attempts."""
    arch = archive(); parent = arch["best"]
    done, tries = set(arch.get("integrated", [])), arch.get("pool_attempts", {})
    pool = [r for r in latest_rows().values() if r["verdict"] == "pooled" and in_line(r)
            and r["id"] not in done and tries.get(r["id"], 0) < MAX_POOL_TRIES]
    if len(pool) < a.min:
        print(f"[bundle] {len(pool)} pooled change(s), need {a.min} - nothing to do", flush=True); return None
    pool.sort(key=lambda r: -pool_score(r))
    eid = f"{time.strftime('%m%d%H%M')}-bundle-{uuid.uuid4().hex[:4]}"
    rwt = f"{REMOTE}/agents/{eid}/engine"
    print(f"[{eid}] bundling {[r['id'] for r in pool]} into {parent.get('tag')} {parent['commit'][:10]}", flush=True)
    sh(f"mkdir -p {REMOTE}/records/agents/{eid} && git clone -q {REMOTE}/engine.git {rwt} && cd {rwt} && git checkout -q -b agent/{eid} {parent['commit']}")
    merged, skipped = [], []
    for r in pool:
        out = subprocess.run(remote_argv(f"cd {rwt} && git fetch -q {REMOTE}/agents/{repo_of(r)}/engine {r['commit']} && "
                                    f"git -c user.name=evo-{eid} -c user.email=evo@local merge --no-edit -m 'bundle {r['id']}' FETCH_HEAD"),
                             capture_output=True, text=True)
        if out.returncode != 0:
            sh(f"cd {rwt} && git merge --abort", check=False); skipped.append(r["id"])
            print(f"[{eid}] {r['id']} conflicts with the bundle so far - skipped", flush=True)
        else:
            merged.append(r["id"])
    msg = f"bundle of {', '.join(merged)} into {parent.get('tag')} | " + " || ".join(r["summary"].split(" | ")[0] for r in pool if r["id"] in merged)
    if merged:
        subprocess.run(remote_argv(f"cat > {REMOTE}/records/agents/{eid}/submission.json"), input=json.dumps({"commit": "", "message": msg}),
                       text=True, check=True)
        sub, res, verdict = measure(eid, a.gpu, parent, arch["noise_threshold"])
        if verdict == "pooled":      # a bundle is the pool's last chance, never a new pool member
            verdict = "no_gain"
    else:
        sub, res, verdict = {"commit": None}, {}, "merge_conflict"
    vs = res.get("vs_ref", {})
    row = {"id": eid, "agent_kind": "none", "focus": "bundle", "members": merged, "skipped": skipped, "skills_version": None,
           "remeasured": res.get("remeasured"), "parent": parent["commit"], "parent_tag": parent.get("tag"), "commit": sub["commit"],
           "summary": msg, "verdict": verdict, "dev_speedup": vs.get("dev"), "holdout_speedup": vs.get("holdout"),
           "e2e_speedup": res.get("vs_ref_e2e"), "dev_tok_s": res.get("metrics", {}).get("dev", {}).get("decode_tok_s"),
           "holdout_tok_s": res.get("metrics", {}).get("holdout", {}).get("decode_tok_s"),
           "correctness": res.get("metrics", {}).get("correctness"), "final_eval": res.get("eval_id"),
           "agent_evals": 0, "agent_wall_min": 0, "cost": None, "ended": time.strftime("%F %T")}
    append_row(row)
    if verdict == "accepted" and archive()["best"]["commit"] == parent["commit"]:
        promote(eid, a.gpu, sub["commit"], res)
        mark_integrated(merged)
        for m in merged:
            board_event({"type": "release", "agent": f"queue:{m}", "cid": None,
                         "reason": f"{m} merged in bundle {eid} (dev {vs.get('dev')}, holdout {vs.get('holdout')}). Best is now {archive()['best'].get('tag')}."})
    else:
        bump_pool_tries(merged + skipped)
        if len(merged) == 1:         # it was measured on its own on the current best: no point in bundling it again
            bump_pool_tries(merged * MAX_POOL_TRIES)
        board_event({"type": "note", "agent": "framework", "where": [],
                     "text": f"bundle {eid} of {merged}: {verdict} (dev {vs.get('dev')}, holdout {vs.get('holdout')}, e2e {res.get('vs_ref_e2e')})."})
    print(json.dumps(row, indent=1), flush=True)
    return verdict


def repool(a):
    """Apply the pooled verdict to earlier no_gain rows whose parent is the given tag or a later archive version
    (stored means, no GPU work). Re-pooled rows enter the main line's pool even if they came from an ablation arm."""
    arch = archive()
    tags = [e["tag"] for e in arch["accepted"]]
    commits = {e["commit"] for e in arch["accepted"][tags.index(a.since):]}
    pooled = {r.get("repo_episode") for r in latest_rows().values() if r["verdict"] == "pooled"}
    for r in latest_rows().values():
        if r["verdict"] != "no_gain" or r["parent"] not in commits or r["id"] in pooled or not r.get("final_eval"):
            continue
        v = verdict_of((r.get("correctness") or {}).get("passed"), {"dev": r["dev_speedup"], "holdout": r["holdout_speedup"]},
                       r.get("e2e_speedup") or {}, arch["noise_threshold"])
        changed = v == "pooled" and engine_changed(r["id"], r["parent"], r["commit"])
        print(f"{r['id']}: {v}{'' if v != 'pooled' else (' (engine changed)' if changed else ' (engine unchanged - skipped)')}")
        if v == "pooled" and changed and not a.dry_run:
            new = dict(r, id=f"{r['id']}-pool", repo_episode=r["id"], arm=None, source_arm=r.get("arm"), verdict="pooled",
                       summary=f"re-pooled from {r['id']} | {r['summary']}", ended=time.strftime("%F %T"))
            append_row(new)
            queue_claim(new["id"], r["parent"], r["commit"], r["summary"], kind="pooled", repo=r["id"])


def regate(a):
    """Re-apply the current judge gate (framework/evo/judge.py:gate) to episodes rejected on correctness; the judged
    gaps are stored, so no GPU work. Passing episodes become accepted_sibling (to be challenged / merged)."""
    sys.path.insert(0, os.path.join(HERE, "framework")); from evo.judge_gate import gate, TAU, HARD
    rows = [json.loads(l) for l in sh(f"cat {REMOTE}/records/experiments.jsonl").splitlines()]
    latest = {r["id"]: r for r in rows}
    for r in latest.values():
        if r["verdict"] != "rejected_correctness" or not r.get("final_eval"):
            continue
        j = json.loads(sh(f"cat {REMOTE}/slots/{r['final_eval']}/judge.json"))
        gaps = [g for w in j["workloads"].values() for q in w for _, g in q["near_tie_flips"] + q["violations"] + q.get("soft_violations", [])]
        tok = j["tokens"]; flips = sum(g <= TAU for g in gaps); soft = sum(TAU < g <= HARD for g in gaps); hard = sum(g > HARD for g in gaps)
        ok = gate(tok, flips, soft, hard)
        res = json.loads(sh(f"cat {REMOTE}/records/evals/{r['final_eval']}.json"))
        res["metrics"]["correctness"].update(passed=ok, soft_violations=soft, violations=hard)
        verdict = decide(res, archive()["noise_threshold"])
        if verdict == "accepted":
            verdict = "accepted_sibling"
        print(f"{r['id']}: soft={soft} hard={hard} -> {verdict}")
        if verdict != r["verdict"] and not a.dry_run:
            new = dict(r, verdict=verdict, correctness=res["metrics"]["correctness"], regated="v2", ended=time.strftime("%F %T"))
            append_row(new)
            if verdict == "accepted_sibling":
                queue_claim(r["id"], r["parent"], r["commit"], r.get("summary", ""))


def challenge(a):
    """Paired head-to-head of an earlier episode's tree against the current best (alternative implementations of
    the same change cannot be merged); the winner becomes the new best."""
    parent = archive()["best"]
    src = a.episode
    print(f"[{src}] challenging {parent.get('tag')} {parent['commit'][:10]}", flush=True)
    sub, res, verdict = measure(src, a.gpu, parent, archive()["noise_threshold"])
    vs = res.get("vs_ref", {})
    row = {"id": f"{src}-challenge", "agent_kind": "none", "focus": "challenge", "source_episode": src, "skills_version": None,
           "parent": parent["commit"], "commit": sub["commit"], "summary": f"challenge of {parent.get('tag')} | {sub['message']}",
           "verdict": verdict, "dev_speedup": vs.get("dev"), "holdout_speedup": vs.get("holdout"), "e2e_speedup": res.get("vs_ref_e2e"),
           "dev_tok_s": res.get("metrics", {}).get("dev", {}).get("decode_tok_s"),
           "holdout_tok_s": res.get("metrics", {}).get("holdout", {}).get("decode_tok_s"),
           "correctness": res.get("metrics", {}).get("correctness"), "final_eval": res.get("eval_id"),
           "agent_evals": 0, "agent_wall_min": 0, "cost": None, "ended": time.strftime("%F %T")}
    append_row(row)
    if verdict == "accepted" and archive()["best"]["commit"] == parent["commit"]:
        promote(src, a.gpu, sub["commit"], res)
    print(json.dumps(row, indent=1), flush=True)


PORT_PROMPT = """You are a porting agent in EvoInfer, episode {eid}.
Your working directory is a git worktree of an LLM inference engine (nano-vllm extended to Qwen3.5-35B-A3B: hybrid Gated
DeltaNet + gated attention, 256-expert top-8 MoE, MTP speculative decoding, heavily optimized for one 80 GB GPU in BF16).
Before anything else read the skills in .evo/skills/ - start with .evo/skills/evo-workflow/SKILL.md.

{goal}

Your task is to PORT this engine to the edge scenario, so that `evo eval` runs and passes the judge:
- Load the 4-bit checkpoint at $EVO_MODEL. Every routed-expert weight W [..., N, K] (main layers `...mlp.experts.gate_up_proj`
  / `down_proj`, MTP `mtp.layers.0.mlp.experts.<i>.<gate|up|down>_proj`) is replaced by `<name>.qweight` (uint8 [..., N, K/2],
  two signed 4-bit values per byte stored as q + 8, low nibble = even k) and `<name>.scales` (bf16 [..., N, K/128]). The
  canonical value is W[n, k] = bf16(float32(q[n, k]) * float32(scales[n, k // 128])); the judge's reference model holds exactly
  these values in BF16. Everything else in the checkpoint is unchanged BF16; vision-tower weights can be skipped.
- Keep the routed experts in 4-bit in GPU memory (dequantizing them all would not fit in 32 GB). The simplest correct
  approach is fine as a start - e.g. dequantize only the experts a step routes to, or a straightforward Triton 4-bit
  kernel - as long as every code path (prefill, decode, MTP drafting and verification, CUDA graphs) works and is correct.
- Keep speculative decoding (MTP) and the existing optimizations wherever they still apply; adjust evo_config.json
  (memory utilization, max_model_len, ...) to the 32 GB edge GPU.
- Use `evo run -- python ...` for experiments on the edge GPU and `evo eval` to check speed and correctness. If time is
  left after a correct port, make it faster; later agents will optimize further, so correctness and clarity come first.
Budget: at most {max_evals} `evo eval` runs and about {minutes} minutes. GPU work only through `evo eval` / `evo run`.
Finish with `evo submit -m "<what was ported> | <how 4-bit experts are computed> | <measured dev decode/e2e tok/s, judge result> | <next idea>"`."""


MSGL_PORT_PROMPT = """You are a porting agent in EvoInfer, episode {eid}.
Your working directory is a git worktree of mini-sglang (python/minisgl: a compact LLM serving engine - scheduler, radix
prefix cache, paged KV cache, flashinfer attention backends, CUDA-graph decode) plus an empty evo_config.json. It does
not support the target model yet.
Before anything else read the skills in .evo/skills/ - start with .evo/skills/evo-workflow/SKILL.md.

{goal}

Your task is to PORT the model Qwen3.5-35B-A3B (checkpoint at $EVO_MODEL on the evaluation host) into this engine, so
that `evo eval` runs and passes the judge:
- The model is a hybrid of Gated DeltaNet layers (linear attention with a recurrent state) and gated full-attention
  layers, with a 256-expert top-8 MoE MLP, and an MTP head for speculative decoding. A complete, correct but unoptimized
  implementation in another engine (nano-vllm) is in .evo/reference/nanovllm_baseline/ - a read-only reference: the
  model in nanovllm/models/qwen3_5_moe.py, the layers, the weight loading in utils/loader.py, the decode and speculation
  loop in engine/model_runner.py. Port the model's math and weight mapping exactly, and integrate it with THIS engine's
  structure (minisgl's model registry, layers, KV cache and attention backend, scheduler and engine loop) rather than
  copying the other engine wholesale.
- The benchmark and the judge load the engine only through a thin adapter that you must create at the worktree root:
  a package `nanovllm/` such that, with the worktree root on sys.path, `from nanovllm import LLM, SamplingParams` works;
  `LLM(model_path, **json.load(open("evo_config.json")))` loads the model on the visible GPU; and
  `llm.generate(prompts, SamplingParams(temperature=0, max_tokens=n), use_tqdm=False)`, where prompts is a list of
  token-id lists, returns a list of dicts with the key "token_ids" (generated ids only, in request order, stopping at
  EOS or max_tokens as the reference does). `llm.spec_stats` (a dict of cumulative counters) is optional. Keep the
  adapter thin: engine logic belongs in python/minisgl. Generation runs in-process on one GPU; the API server,
  tokenizer workers and multi-process parts of minisgl are not needed.
- Correctness first. Greedy outputs are judged strictly against a frozen reference implementation of the model (BF16
  weights, logit gap at every generated position). The simplest correct port is the goal: speculative decoding with
  the MTP head is optional here and can be left to later agents - say in your submission whether it is implemented.
- Use only the packages already installed in the evaluation host's Python environment; nothing can be installed.
  Avoid or replace code paths that import missing packages (check with `evo run -- python -c "import <pkg>"`).
- Use `evo run -- python ...` for experiments on the GPU and `evo eval` to check speed and correctness. The dev set has
  prompts of up to ~5K tokens (`evo eval --dev` runs all of them with the judge); the final evaluation also runs a
  hidden holdout set of similar lengths, 256 new tokens per request.
Budget: at most {max_evals} `evo eval` runs and about {minutes} minutes. GPU work only through `evo eval` / `evo run`.
Finish with `evo submit -m "<what was ported> | <how it is integrated into minisgl> | <measured dev decode/e2e tok/s, judge result> | <next idea>"`."""
MODEL_PORT_PROMPT = """You are a porting agent in EvoInfer, episode {eid}.
Your working directory is a git worktree of an LLM inference engine: nano-vllm (a compact engine - scheduler, paged KV
cache, model runner with prefill and decode) as it was extended to another model, Qwen3.5-35B-A3B (hybrid Gated DeltaNet
+ gated attention layers, a 256-expert MoE, MTP speculative decoding). That implementation is correct but unoptimized.
The engine does not support the target model of this lineage yet.
Before anything else read the skills in .evo/skills/ - start with .evo/skills/evo-workflow/SKILL.md.

{goal}

Your task is to PORT the target model (checkpoint at $EVO_MODEL on the evaluation host) into this engine, so that
`evo eval` runs and passes the judge:
- {model_desc} The reference is the Hugging Face transformers implementation: its source and the checkpoint's
  config.json are in .evo/reference/ (read-only). Port the model's math and weight mapping exactly, integrated with THIS
  engine's structure (a model file under nanovllm/models, the layers, the loader, the KV cache and the model runner).
  The engine must implement the model itself: do not import transformers' model code at run time.
- The interface the benchmark uses stays as it is in this engine: `from nanovllm import LLM, SamplingParams`;
  `LLM(model_path, **json.load(open("evo_config.json")))` loads the model on the visible GPU;
  `llm.generate(prompts, SamplingParams(temperature=0, max_tokens=n), use_tqdm=False)`, where prompts is a list of
  token-id lists, returns a list of dicts with the key "token_ids" (generated ids only, in request order, stopping at
  EOS or max_tokens). `llm.spec_stats` is optional.
- Correctness first. The simplest correct port is the goal. The Qwen3.5-specific parts (DeltaNet recurrent state, MTP
  drafting and verification) do not apply to this model: bypass or remove them, and keep whatever engine
  infrastructure still applies. Say in your submission what you kept and what you removed.
- Use only the packages already installed in the evaluation host's Python environment; nothing can be installed.
- Use `evo run -- python ...` for experiments on the GPU and `evo eval` to check speed and correctness. The dev set has
  prompts of up to {ctx} tokens (`evo eval --dev` runs all of them with the judge); the final evaluation also runs a
  hidden holdout set of similar lengths, 256 new tokens per request.
Budget: at most {max_evals} `evo eval` runs and about {minutes} minutes. GPU work only through `evo eval` / `evo run`.
Finish with `evo submit -m "<what was ported> | <what was kept / removed> | <measured dev decode/e2e tok/s, judge result> | <next idea>"`."""
PORT_PROMPTS = {"edge": PORT_PROMPT, "msgl": MSGL_PORT_PROMPT, "model": MODEL_PORT_PROMPT}


class _Keep(dict):
    def __missing__(self, k):
        return "{" + k + "}"


def add_host_reference(local, scen):
    """Agent-local, read-only reference for a model port: the transformers source of the model and its config.json."""
    dst = os.path.join(local, ".evo", "reference"); os.makedirs(dst, exist_ok=True)
    tf = sh(f"cd {REMOTE} && {EVO_PY} -c \"import os, transformers; print(os.path.join(os.path.dirname(transformers.__file__), 'models'))\"").strip()
    for src in MODEL_INFO[scen]["ref"]:
        rsync("--exclude", "__pycache__", rp(f"{tf}/{src}"), dst + "/transformers/")
    model = sh(f"cd {REMOTE} && {EVO_PY} -c \"from evo.scenarios import model_path; print(model_path('{scen}'))\"").strip()
    for f in ("config.json", "generation_config.json"):
        rsync(rp(f"{model}/{f}"), dst + "/", check=False)


def add_reference(eid, local, tag, paths, name):
    """Agent-local, read-only copy of another engine version's sources (never shipped back: .evo is local-only)."""
    ref = f"{REMOTE}/agents/{eid}/reference"
    sh(f"rm -rf {ref} && mkdir -p {ref} && git -C {REMOTE}/engine.git archive {tag} {paths} | tar -x -C {ref}")
    dst = os.path.join(local, ".evo", "reference", name); os.makedirs(dst, exist_ok=True)
    rsync(rp(f"{ref}/"), dst + "/")


def init_lineage(a):
    """Start an engine lineage for a deployment scenario from a main-line version: commit the scenario's engine
    settings (--config) on top of it, evaluate that root in full with the judge, and record it as <lineage>-000 in
    archive-<lineage>.json. Episodes, integrate, bundle and status with --lineage then work inside the lineage."""
    global SCEN, ENGINE_PATHS, ENGINE_NOTE
    ENGINE_PATHS, ENGINE_NOTE = a.engine_paths or ENGINE_PATHS, a.engine_note or ""
    port_prompt = PORT_PROMPTS[a.port_kind]
    if a.port_kind == "model":
        mi = MODEL_INFO[a.scenario]
        port_prompt = port_prompt.format_map(_Keep(model_desc=mi["desc"], ctx=mi["ctx"]))
        ENGINE_NOTE = ENGINE_NOTE or (f"The target model of this lineage is {mi['name']}, ported into nano-vllm. {mi['desc']} The skills were written while "
                                      "optimizing another model (Qwen3.5-35B-A3B): their methods apply, the model-specific details (MTP speculation, "
                                      "linear-attention state) do not. The transformers source of the model is in .evo/reference/ when present.")
    if sh(f"test -e {REMOTE}/records/{arch_file()} && echo exists || true").strip():
        raise SystemExit(f"{arch_file()} already exists")
    SCEN = a.scenario
    has_main = bool(sh(f"test -e {REMOTE}/records/archive.json && echo yes || true").strip())
    # without --lineage this creates the main line (archive.json); a lineage inherits the main line's noise threshold
    main = json.loads(sh(f"cat {REMOTE}/records/archive.json")) if has_main else {"accepted": [], "noise_threshold": a.noise_threshold}
    src = next((e for e in main["accepted"] if e["tag"] == a.from_tag), None)
    if src is None:   # not a main-line version: any engine tag, e.g. `baseline` (the naive port every lineage descends from)
        src = {"tag": a.from_tag, "commit": sh(f"git -C {REMOTE}/engine.git rev-parse --verify -q '{a.from_tag}^{{commit}}'").strip()}
    eid = a.resume or f"{time.strftime('%m%d%H%M')}-{LIN}-root"
    rwt = f"{REMOTE}/agents/{eid}/engine"
    cfg = json.loads(a.config or "{}")
    if not a.resume:
        sh(f"mkdir -p {REMOTE}/records/agents/{eid} && git clone -q {REMOTE}/engine.git {rwt} && cd {rwt} && git checkout -q -b agent/{eid} {src['commit']}")
    if cfg and not a.resume:
        sh(f"cd {rwt} && python3 - <<'EOF'\nimport json\nc = json.load(open('evo_config.json'))\nc.update(json.loads({json.dumps(json.dumps(cfg))}))\n"
           f"json.dump(c, open('evo_config.json', 'w'), indent=2)\nEOF")
    msg = f"{LIN or 'main'} lineage root: {a.from_tag} for scenario {SCEN}" + (f", settings {cfg}" if cfg else "")
    if not a.resume:
        sh(f"cd {rwt} && git add -A && git -c user.name=evo-{eid} -c user.email=evo@local commit -q --allow-empty -m {shlex.quote(msg)}")
    commit = sh(f"cd {rwt} && git rev-parse HEAD").strip()
    if a.port_minutes:   # the scenario needs code changes before the root can run at all: a porting agent makes them
        if a.resume:   # continue the same porting agent's worktree after a failed root evaluation
            local = os.path.join(RUNS, eid, "engine")
            prev = json.loads(sh(f"cat {REMOTE}/records/agents/{eid}/submission.json"))
            fails = sorted(sh(f"ls {REMOTE}/records/evals/ | grep '{eid}-full' || true").split())
            tail_ = json.loads(sh(f"cat {REMOTE}/records/evals/{fails[-1]}"))["log_tail"][-2500:] if fails else "(none)"
            sh(f"rm -f {REMOTE}/records/agents/{eid}/submission.json")
            last = json.loads(sh(f"cat {REMOTE}/records/evals/{fails[-1]}")) if fails else {}
            if a.port_kind != "edge":
                prompt = port_prompt.format(eid=eid, goal=SCEN_GOAL[SCEN], max_evals=a.max_evals, minutes=a.port_minutes) + f"""

CONTINUATION. You already worked on this port in this worktree; your submission was:
  {prev['message']}
The orchestrator's full evaluation of your submission (all dev + holdout prompts, 256 new tokens, judge) did not pass:
status {last.get('status')}, correctness {json.dumps((last.get('metrics') or {}).get('correctness'))[:1500]}
Log tail:
{tail_}
Fix the port so the full evaluation runs and passes the judge, verify with `evo eval --dev`, then submit."""
            else:
              prompt = PORT_PROMPT.format(eid=eid, goal=SCEN_GOAL[SCEN], max_evals=a.max_evals, minutes=a.port_minutes) + f"""

CONTINUATION. You already worked on this port in this worktree; your submission was:
  {prev['message']}
The orchestrator's full evaluation of your submission (all dev + holdout prompts, 256 new tokens) crashed. Log tail:
{tail_}
The dev set has prompts of up to ~5K tokens (see the workloads); the final evaluation also runs a hidden holdout set of
similar lengths. Fix the port so the full evaluation runs, verify with `evo eval --dev` (all dev prompts, judge), then submit."""
        else:
            local = make_local(eid, "codex", rwt, a.skills or current_skills())
            prompt = port_prompt.format(eid=eid, goal=SCEN_GOAL[SCEN], max_evals=a.max_evals, minutes=a.port_minutes)
            if a.port_kind == "msgl":
                add_reference(eid, local, "baseline", "nanovllm evo_config.json", "nanovllm_baseline")
            if a.port_kind == "model":
                add_host_reference(local, SCEN)
        cost, wall = run_agent("codex", eid, local, a.gpu, "", prompt, a.max_evals, a.port_minutes)
        print(f"[{eid}] port agent finished after {wall / 60:.1f} min, cost {cost}", flush=True)
        board_event({"type": "end", "agent": eid})
        sub = commit_submission(eid); commit = sub["commit"]
        msg = f"{LIN or 'main'} lineage root: {a.from_tag} ported to scenario {SCEN} | {sub['message']}"
    ev = f"cd {REMOTE} && {EVO_PY} -m evo.evaluate --scenario {SCEN}"
    res = json.loads(sh(f"{ev} {rwt} --gpu {a.gpu} --mode full --label {eid}-full", timeout=4 * 3600))
    print(json.dumps({k: res.get(k) for k in ("status", "eval_id", "gpu", "metrics", "log_tail")}, indent=1), flush=True)
    if res["status"] != "ok" or not res["metrics"].get("correctness", {}).get("passed"):
        raise SystemExit(f"root evaluation failed - lineage not created (worktree {rwt}, commit {commit[:10]})")
    subprocess.run(remote_argv(f"cat > {REMOTE}/records/{arch_file()}"), text=True, check=True, input=json.dumps(
        dict({"scenario": SCEN, "from": a.from_tag, "from_commit": src["commit"], "settings": cfg,
              "noise_threshold": main["noise_threshold"], "accepted": [], "integrated": []},
             **({"engine_paths": ENGINE_PATHS, "engine_note": ENGINE_NOTE} if ENGINE_PATHS != DEFAULT_ENGINE_PATHS or ENGINE_NOTE else {})), indent=1))
    promote(eid, a.gpu, commit, res, n=0)
    append_row({"id": eid, "agent_kind": "none", "focus": "lineage-root", "parent": src["commit"], "parent_tag": a.from_tag,
                "commit": commit, "summary": msg, "verdict": "root", "final_eval": res["eval_id"],
                "dev_tok_s": res["metrics"]["dev"]["decode_tok_s"], "holdout_tok_s": res["metrics"]["holdout"]["decode_tok_s"],
                "correctness": res["metrics"]["correctness"], "agent_evals": 0, "agent_wall_min": 0, "cost": None,
                "ended": time.strftime("%F %T")})
    print(f"[{LIN or 'main'}] lineage created: {tag_prefix()}-000 = {commit[:10]}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# Getting started: doctor, setup, evolve, eval
# ---------------------------------------------------------------------------------------------------------------------
SEED_TAG = "baseline"                                         # the naive port every lineage descends from
SHIPPED = {"evo-015": "paper-evo-015", "lc-003": "paper-lc-003", "edge-003": "paper-edge-003"}   # engines/<dir> -> tag


def _check(ok, what, hint=""):
    print(f"  {'ok ' if ok else 'XX '} {what}" + (f"\n       -> {hint}" if hint and not ok else ""), flush=True)
    return ok


def doctor(a):
    """Check everything a run needs, from the agent side to the GPU host, and say how to fix what is missing."""
    good = True
    print(f"config: {C.config_path()}" + ("" if os.path.exists(C.config_path()) else "  (not found: using defaults)"))
    print(f"GPU host: {MODE}" + (f" ({CFG['host']['ssh_host']})" if MODE == "ssh" else "") + f", home {REMOTE}")
    print("agent side (this machine):")
    for tool in ("git", "rsync", "bash"):
        good &= _check(shutil.which(tool) is not None, tool, f"install {tool}")
    have = [k for k in ("codex", "claude") if shutil.which(AG[f"{k}_bin"])]
    good &= _check(bool(have), f"agent CLI: {', '.join(have) or 'none'}", "install the Codex CLI (`npm i -g @openai/codex`) or Claude Code")
    print("GPU host:")
    if MODE == "ssh" and not _check(subprocess.run(remote_argv("true"), capture_output=True).returncode == 0,
                                    f"ssh {CFG['host']['ssh_host']}", "check [host] ssh_host / ssh_config and that `ssh <host> true` works"):
        return 1
    sh(f"mkdir -p {REMOTE}")
    probe = ("import json, sys\nout = {'python': sys.version.split()[0]}\n"
             "import importlib, importlib.metadata as md\n"
             "for m, dist in (('torch', 'torch'), ('transformers', 'transformers'), ('triton', 'triton'), ('fla', 'flash-linear-attention'),\n"
             "                ('vllm', 'vllm'), ('xxhash', 'xxhash'), ('safetensors', 'safetensors')):\n"
             "    try:\n        importlib.import_module(m)\n    except Exception:\n        out[m] = None; continue\n"
             "    try:\n        out[m] = md.version(dist)\n    except Exception:\n        out[m] = 'installed'\n"
             "try:\n    import torch; out['cuda'] = torch.cuda.is_available(); out['gpus'] = torch.cuda.device_count()\n"
             "except Exception:\n    out['cuda'] = False\nprint(json.dumps(out))\n")
    p = subprocess.run(remote_argv(f"cd {REMOTE} && {shq(PY)} - <<'EOF'\n{probe}EOF"), capture_output=True, text=True)
    if not _check(p.returncode == 0, f"python ({PY})", "run scripts/setup_gpu_host.sh on the GPU host, or set [host] python"):
        return 1
    v = json.loads(p.stdout.strip().splitlines()[-1])
    good &= _check(tuple(int(x) for x in v["python"].split(".")[:2]) >= (3, 11), f"python {v['python']} (>= 3.11)")
    for m in ("torch", "transformers", "triton", "fla", "vllm", "xxhash", "safetensors"):
        good &= _check(v.get(m) is not None, f"{m} {v.get(m) or 'missing'}", "pip install -r requirements-gpu.txt")
    good &= _check(bool(v.get("cuda")), f"CUDA available, {v.get('gpus', 0)} GPU(s)")
    smi = subprocess.run(remote_argv("nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits"),
                         capture_output=True, text=True)
    rows = {int(r.split(",")[0]): [x.strip() for x in r.split(",")] for r in smi.stdout.strip().splitlines() if r.strip()}
    for g in C.gpu_pool():
        r = rows.get(g)
        good &= _check(r is not None and float(r[3]) / 1024 >= CFG["gpus"]["need_gb"] * 0.9,
                       f"GPU {g}: {r[1] + ', ' + str(round(float(r[3]) / 1024)) + ' GB free' if r else 'not found'}",
                       f"[gpus] pool lists GPUs EvoInfer may use; an evaluation needs about {CFG['gpus']['need_gb']} GB free")
    print("models:")
    for k, path in CFG["models"].items():
        if not path:
            if k == "qwen3_5":
                good &= _check(False, f"{k}: not set", "set [models] qwen3_5 to the Qwen3.5-35B-A3B checkpoint directory")
            continue
        ok = subprocess.run(remote_argv(f"test -f {shq(path)}/config.json"), capture_output=True).returncode == 0
        good &= _check(ok, f"{k}: {path}", "the directory must contain config.json and the safetensors shards")
    print("setup:")
    for what, cmd in (("engine archive", f"test -d {REMOTE}/engine.git"), ("judge engine", f"test -d {REMOTE}/judge_engine"),
                      ("prompt sets", f"test -f {REMOTE}/workloads/dev.jsonl"), ("main line (archive.json)", f"test -f {REMOTE}/records/archive.json")):
        _check(subprocess.run(remote_argv(cmd), capture_output=True).returncode == 0, what,
               "python3 orchestrator.py setup" if what != "main line (archive.json)" else
               "python3 orchestrator.py init --scenario base --from baseline --gpu <gpu>")
    print("all required checks passed" if good else "some required checks failed (see -> hints)")
    return 0 if good else 1


def sync_framework():
    """Copy the framework, the config and the prompt sets to the GPU host's home (idempotent)."""
    sh(f"mkdir -p {REMOTE}/framework {REMOTE}/workloads {REMOTE}/private {REMOTE}/records")
    rsync("--delete", "--exclude", "__pycache__", os.path.join(HERE, "framework") + "/", rp(f"{REMOTE}/framework/"))
    if os.path.exists(C.config_path()):
        rsync(C.config_path(), rp(f"{REMOTE}/evoinfer.toml"))
    rsync(os.path.join(HERE, "data", "workloads") + "/", rp(f"{REMOTE}/workloads/"))
    rsync(os.path.join(HERE, "data", "holdout") + "/", rp(f"{REMOTE}/private/"))


def _import_engine(src_dir, tag, msg):
    """Commit an engine directory from this repo as its own root commit in the GPU host's archive, tagged `tag`."""
    tmp = f"{REMOTE}/import-{tag}"
    sh(f"rm -rf {tmp} && mkdir -p {tmp}")
    rsync("--exclude", "__pycache__", src_dir + "/", rp(tmp + "/"))
    sh(f"cd {tmp} && git init -q && git add -A && git -c user.name=evoinfer -c user.email=evo@local commit -q -m {shlex.quote(msg)}"
       f" && git tag {tag} && git push -q {REMOTE}/engine.git refs/tags/{tag} && cd .. && rm -rf {tmp}")


def setup(a):
    """Prepare the GPU host: framework + config + prompt sets, the engine archive (seed tagged `baseline`, plus the
    paper's evolved engines as paper-* tags), and the frozen judge engine."""
    print(f"[setup] GPU host {MODE}, home {REMOTE}", flush=True)
    sync_framework()
    if not sh(f"test -d {REMOTE}/engine.git && echo yes || true").strip():
        sh(f"git init -q --bare {REMOTE}/engine.git")
        _import_engine(os.path.join(HERE, "engines", "seed"), SEED_TAG, "seed: nano-vllm with a naive Qwen3.5-35B-A3B port")
        for d, tag in SHIPPED.items():
            if os.path.isdir(os.path.join(HERE, "engines", d)):
                _import_engine(os.path.join(HERE, "engines", d), tag, f"{d}: evolved engine from the EvoInfer paper")
        print(f"[setup] engine archive: tags {SEED_TAG}, {', '.join(SHIPPED.values())}", flush=True)
    if not sh(f"test -d {REMOTE}/judge_engine && echo yes || true").strip():
        sh(f"git clone -q {REMOTE}/engine.git {REMOTE}/judge_engine && git -C {REMOTE}/judge_engine checkout -q {SEED_TAG}")
        print("[setup] judge engine: frozen checkout of the seed", flush=True)
    print("[setup] done. Next: python3 orchestrator.py init --scenario base --from baseline --gpu <gpu>", flush=True)


def import_engine(a):
    """Add an engine directory (your own seed engine, e.g. a checkout of another serving engine) to the archive as a tag;
    start a lineage from it with `init --from <tag>` (with --port-minutes if a porting agent must add the model first)."""
    if sh(f"git -C {REMOTE}/engine.git rev-parse -q --verify refs/tags/{a.tag} || true").strip():
        raise SystemExit(f"tag {a.tag} already exists")
    if not os.path.exists(os.path.join(a.engine_dir, "evo_config.json")):
        raise SystemExit(f"{a.engine_dir}/evo_config.json missing: add one ({{}} is fine) - the benchmark passes it to LLM(...)")
    _import_engine(os.path.abspath(a.engine_dir), a.tag, a.message or f"imported engine {a.tag}")
    print(f"[import] {a.engine_dir} -> tag {a.tag}")


DEFAULT_FOCI = ["overhead free", "spec moe", "free free"]   # per round; later rounds repeat the last entry


def evolve(a):
    """The paper's protocol: per round, one episode per GPU in parallel (staggered), then integrate (merge accepted
    siblings, bundle pooled changes). Episodes run as subprocesses so each has its own logs under runs/."""
    gpus = [int(g) for g in a.gpus.split(",")]
    foci = [f.split() for f in (a.foci.split("|") if a.foci else DEFAULT_FOCI)]
    base = [sys.executable, os.path.abspath(__file__)] + (["--lineage", LIN] if LIN else []) + (["--policy", POLICY] if POLICY else [])
    tag = LIN or "main"
    os.makedirs(RUNS, exist_ok=True)
    for R in range(1, a.rounds + 1):
        F = foci[min(R - 1, len(foci) - 1)]
        procs = []
        for i, g in enumerate(gpus):
            focus = F[i % len(F)]
            log = os.path.join(RUNS, f"evolve-{tag}-r{R}-{i}-{focus}.log")
            cmd = base + ["episode", "--agent", a.agent, "--gpu", str(g), "--focus", focus, "--max-evals", str(a.max_evals),
                          "--minutes", str(a.minutes)] + (["--skills", a.skills] if a.skills else [])
            print(f"[{tag}] round {R}: episode on GPU {g}, focus {focus} -> {os.path.relpath(log, HERE)}", flush=True)
            procs.append(subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=dict(os.environ, EVO_NO_SYNC="1")))
            if i < len(gpus) - 1:
                time.sleep(45)   # staggered start: the second agent sees the first one's board claims
        for p in procs:
            p.wait()
        print(f"[{tag}] round {R}: episodes done, integrating", flush=True)
        subprocess.run(base + ["integrate", "--gpu", str(gpus[0]), "--agent", a.agent], env=dict(os.environ, EVO_NO_SYNC="1"),
                       stdout=open(os.path.join(RUNS, f"evolve-{tag}-r{R}-integrate.log"), "w"), stderr=subprocess.STDOUT)
        b = archive()["best"]
        print(f"[{tag}] round {R}: best {b.get('tag')} dev decode {b['result']['metrics']['dev']['decode_tok_s']} tok/s", flush=True)


def eval_engine(a):
    """Benchmark an engine directory (e.g. engines/evo-015) with the full evaluation and judge, and compare it with
    the lineage's root when there is one. Nothing is promoted."""
    src = os.path.abspath(a.engine_dir)
    name = os.path.basename(src.rstrip("/"))
    dst = f"{REMOTE}/xeval/{name}"
    sh(f"mkdir -p {dst}")
    rsync("--delete", "--exclude", "__pycache__", src + "/", rp(dst + "/"))
    scen = a.scenario or SCEN
    ref = ""
    if sh(f"test -f {REMOTE}/records/{arch_file()} && echo yes || true").strip():
        root = archive()["accepted"][0]
        ref = f" --ref {root['result_path']}"
    res = json.loads(sh(f"cd {REMOTE} && {EVO_PY} -m evo.evaluate --scenario {scen} {dst} --gpu {a.gpu} --mode {a.mode} --label xeval-{name}{ref}",
                        timeout=4 * 3600))
    if res["status"] != "ok":
        print(f"{name}: {res['status']}\n{res.get('log_tail', '')[-3000:]}"); return 1
    m = res["metrics"]
    print(f"{name} ({scen}, {a.mode}, GPU {res['gpu']}):")
    for w in ("dev", "holdout"):
        if w in m:
            x = res.get("vs_ref", {}).get(w)
            print(f"  {w:8s} decode {m[w]['decode_tok_s']:8.2f} tok/s   e2e {m[w]['e2e_tok_s']:8.2f} tok/s"
                  + (f"   ({x + 1:.2f}x the root's decode)" if x is not None else ""))
    c = m.get("correctness") or {}
    print(f"  judge: {'PASS' if c.get('passed') else 'FAIL'} ({c.get('violations')} violations, {c.get('soft_violations')} soft, "
          f"{c.get('near_tie_flips')} near-tie flips / {c.get('tokens')} tokens)")
    print(f"  full result: {REMOTE}/records/evals/{res['eval_id']}.json")
    return 0


def status(a):
    if not sh(f"test -f {REMOTE}/records/{arch_file()} && echo yes || true").strip():
        raise SystemExit(f"no {arch_file()} yet: run `python3 orchestrator.py init --scenario base --from baseline --gpu <gpu>`")
    arch = archive()
    print(f"best: {arch['best'].get('tag', 'baseline')} {arch['best']['commit'][:10]} dev={arch['best']['result']['metrics']['dev']['decode_tok_s']}"
          f" holdout={arch['best']['result']['metrics']['holdout']['decode_tok_s']}  noise_thr={arch['noise_threshold']}")
    print(sh(f"cd {REMOTE} && {EVO_PY} -m evo.cli history -n 50"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", default="", help="work inside a scenario lineage (archive-<name>.json) instead of the main line")
    ap.add_argument("--policy", default="", choices=[""] + list(POLICIES), help="agent CLI + model (overrides --agent)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("episode"); e.add_argument("--agent", choices=["claude", "codex"], default="codex")
    e.add_argument("--gpu", type=int, required=True); e.add_argument("--focus", choices=list(FOCUS), default="free")
    e.add_argument("--sibling", help="episode id whose changes a merge episode should port")
    e.add_argument("--max-evals", type=int, default=10); e.add_argument("--minutes", type=int, default=90)
    e.add_argument("--skills", help="skill version under skills/ (default: skills/CURRENT)")
    e.add_argument("--parent", help="archive tag to start from (default: current best)")
    e.add_argument("--arm", help="ablation arm: own board and history view, never promoted")
    e.add_argument("--no-promote", action="store_true")
    it = sub.add_parser("integrate"); it.add_argument("--gpu", type=int, required=True)
    it.add_argument("--agent", choices=["claude", "codex"], default="codex")
    m = sub.add_parser("merge"); m.add_argument("--episode", required=True); m.add_argument("--gpu", type=int, required=True)
    c = sub.add_parser("challenge"); c.add_argument("--episode", required=True); c.add_argument("--gpu", type=int, required=True)
    g = sub.add_parser("regate"); g.add_argument("--dry-run", action="store_true")
    b = sub.add_parser("bundle"); b.add_argument("--gpu", type=int, required=True); b.add_argument("--min", type=int, default=2)
    rpl = sub.add_parser("repool"); rpl.add_argument("--since", required=True, help="archive tag, e.g. evo-013")
    rpl.add_argument("--dry-run", action="store_true")
    sub.add_parser("status")
    sub.add_parser("doctor", help="check the config, the GPU host, the models and the agent CLIs")
    sub.add_parser("setup", help="prepare the GPU host: framework, prompt sets, engine archive, judge engine")
    ev = sub.add_parser("evolve", help="run rounds of parallel episodes + integrate (the paper's protocol)")
    ev.add_argument("--rounds", type=int, default=3); ev.add_argument("--gpus", required=True, help="one episode per GPU per round, e.g. 0,1")
    ev.add_argument("--agent", choices=["claude", "codex"], default="codex"); ev.add_argument("--minutes", type=int, default=75)
    ev.add_argument("--max-evals", type=int, default=8); ev.add_argument("--skills")
    ev.add_argument("--foci", help='focus per agent per round, rounds separated by "|", e.g. "overhead free|spec moe|free free"')
    ie = sub.add_parser("import-engine", help="add an engine directory to the archive as a tag (a new seed engine)")
    ie.add_argument("engine_dir"); ie.add_argument("--tag", required=True); ie.add_argument("-m", "--message")
    xe = sub.add_parser("eval", help="benchmark + judge an engine directory, e.g. engines/evo-015")
    xe.add_argument("engine_dir"); xe.add_argument("--gpu", type=int, required=True)
    xe.add_argument("--mode", default="full", choices=["quick", "dev", "full"]); xe.add_argument("--scenario")
    i = sub.add_parser("init", help="create the main line, or (with --lineage) a lineage for a deployment scenario")
    i.add_argument("--scenario", choices=list(PRIMARY), required=True); i.add_argument("--from", dest="from_tag", required=True)
    i.add_argument("--config", help="JSON engine settings merged into evo_config.json of the root"); i.add_argument("--gpu", type=int, required=True)
    i.add_argument("--port-minutes", type=int, default=0, help="run a Codex porting agent on the root first (scenario needs code changes)")
    i.add_argument("--max-evals", type=int, default=12); i.add_argument("--skills")
    i.add_argument("--resume", help="episode id of an earlier porting agent whose root failed: continue in its worktree")
    i.add_argument("--port-kind", choices=list(PORT_PROMPTS), default="edge", help="what the porting agent does (edge: 4-bit port; msgl: model into mini-sglang; model: another target model into nano-vllm)")
    i.add_argument("--engine-paths", help="paths that make up this lineage's engine (default: nanovllm evo_config.json)")
    i.add_argument("--engine-note", help="one paragraph about this lineage's engine, added to every agent prompt")
    i.add_argument("--noise-threshold", type=float, default=0.03, help="acceptance bar for the main line (3%%: ~2x the paired noise)")
    a = ap.parse_args()
    LIN, POLICY = a.lineage, a.policy
    if POLICY and hasattr(a, "agent"):
        a.agent = POLICIES[POLICY]["cli"]
    if a.cmd in ("episode", "init", "integrate", "eval", "evolve") and not os.environ.get("EVO_NO_SYNC"):
        sync_framework()   # edits to framework/, evoinfer.toml or data/ reach the GPU host before any GPU work
    if LIN and a.cmd not in ("init", "doctor", "setup", "import-engine"):
        _arch = archive()
        SCEN = _arch["scenario"]
        ENGINE_PATHS, ENGINE_NOTE = _arch.get("engine_paths", ENGINE_PATHS), _arch.get("engine_note", "")
    rc = {"episode": episode, "merge": auto_merge, "integrate": integrate, "challenge": challenge, "regate": regate, "status": status,
          "bundle": bundle, "repool": repool, "init": init_lineage, "doctor": doctor, "setup": setup, "evolve": evolve,
          "eval": eval_engine, "import-engine": import_engine}[a.cmd](a)
    sys.exit(rc if isinstance(rc, int) else 0)
