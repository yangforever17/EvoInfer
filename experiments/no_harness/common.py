"""Shared pieces of the no-harness baselines (bare.py, dsh.py): the same task text, seed, limits, sandbox, snapshots and
final measurement, so the two differ only in the agent around the model.

Model endpoint (OpenAI-compatible chat completions): $EVO_LLM_BASE_URL (e.g. https://host/v1) and $EVO_LLM_API_KEY.
The key never enters the agent's environment. Sandbox: bubblewrap (Linux), `apt install bubblewrap`.
"""
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import orchestrator as O  # noqa: E402

CMD_TIMEOUT = 2400       # seconds per command (a full `evo eval --dev` takes ~10 minutes on a slow engine)
SNAP_EVERY = 75 * 60     # one snapshot per 75 minutes = one episode of the matching evolution arm
HOME = os.path.expanduser("~")

TASK = """Your working directory is an LLM inference engine: nano-vllm extended to Qwen3.5-35B-A3B (hybrid Gated DeltaNet +
gated attention layers, 256-expert top-8 MoE, an MTP head used for speculative decoding), serving one request at a
time in BF16 on one 80 GB GPU. The benchmark loads the engine from nanovllm/ and evo_config.json in this directory.

Make it faster: higher single-request greedy decode tokens/s and end-to-end tokens/s (prefill + 256 new tokens). The
greedy outputs must stay those of the model in BF16 (no quantization, approximation or skipped computation).

Use the GPU only through these commands:
- `evo eval` benchmarks this directory and checks correctness against a reference implementation
  (`evo eval --dev`: all dev prompts at full length, slower);
- `evo run -- <command>` runs a command on the GPU with this directory on PYTHONPATH.
Work only in this directory; do not look for other code or files.
Budget: about {minutes} minutes and at most {max_evals} `evo eval` runs. When you are finished, leave the engine in its
best working state and reply DONE."""
NUDGE = ("You have about {left} minutes and {evals} `evo eval` runs left. Continue if you can still make the engine "
         "faster; otherwise reply DONE.")


def endpoint():
    base, key = os.environ.get("EVO_LLM_BASE_URL"), os.environ.get("EVO_LLM_API_KEY")
    if not base or not key:
        raise SystemExit("set EVO_LLM_BASE_URL (OpenAI-compatible, e.g. https://host/v1) and EVO_LLM_API_KEY")
    return base.rstrip("/"), key


def records():
    return f"{O.REMOTE}/records/no_harness.jsonl"


def record(row):
    O.subprocess.run(O.remote_argv(f"cat >> {records()}"), input=json.dumps(row) + "\n", text=True, check=True)


def seed(lineage):
    """The root of an evolution lineage (default: the main line), so a baseline is compared with that lineage."""
    arch = json.loads(O.sh(f"cat {O.REMOTE}/records/{'archive-' + lineage + '.json' if lineage else 'archive.json'}"))
    s = arch["accepted"][0]
    return s, f"{O.REMOTE}/agents/{s['episode']}/engine"


def prepare(eid, seed_row):
    """A GPU-host worktree at the seed and its local mirror (the agent works in the mirror)."""
    rwt = O.remote_worktree(eid)
    local = os.path.join(O.RUNS, eid, "engine")
    os.makedirs(os.path.dirname(local), exist_ok=True)
    O.sh(f"mkdir -p {O.REMOTE}/records/agents/{eid} && rm -rf {rwt} && git clone -q {O.REMOTE}/engine.git {rwt} && cd {rwt}"
         f" && git checkout -q -b agent/{eid} {seed_row['commit']} && printf '.evo/\\n' >> .git/info/exclude")
    O.rsync(O.rp(f"{rwt}/"), local + "/")
    return rwt, local


def executor(eid, local, gpu, seed_row, max_evals, arm):
    """Serves the agent's `evo` calls (bin/evo) with the baseline's settings: isolated history, its own arm."""
    O.SCEN, O.LIN = "base", ""
    return O.JobExecutor(eid, local, dict(EVO_ARM=arm, EVO_ISOLATED="1", EVO_SCENARIO="base", EVO_GPU=gpu,
                                          EVO_PARENT_RESULT=seed_row["agent_ref_path"], EVO_MAX_EVALS=max_evals,
                                          EVO_EPISODE_START=int(time.time())))


def agent_env(eid, local, extra_path=()):
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower() and not k.startswith("EVO_LLM_")}
    env.update(EVO_AGENT_ID=eid, EVO_LOCAL_WORKTREE=local, HOME=HOME,
               PATH=os.pathsep.join([os.path.join(O.HERE, "bin"), *extra_path, env.get("PATH", "/usr/bin:/bin")]))
    return env


def sandbox(writable, chdir, ro=()):
    """bubblewrap: the whole file system read-only, the home directory and /tmp replaced by empty ones; visible are only
    the writable directory, the `evo` wrapper and `ro`. Other experiments, the skills and the GPU host's records are not."""
    sb = ["bwrap", "--ro-bind", "/", "/", "--tmpfs", HOME, "--tmpfs", "/tmp", "--dev", "/dev", "--proc", "/proc",
          "--ro-bind", os.path.join(O.HERE, "bin", "evo"), os.path.join(O.HERE, "bin", "evo")]
    for p in ro:
        sb += ["--ro-bind", p, p]
    return sb + ["--bind", writable, writable, "--chdir", chdir, "--die-with-parent", "--"]


def n_evals(eid):
    return int(O.sh(f"ls {O.REMOTE}/records/agents/{eid}/eval-*.json 2>/dev/null | wc -l").strip() or 0)


def snapshot(local, eid, k):
    rwt = f"{O.REMOTE}/agents/{eid}-c{k}/engine"
    O.sh(f"mkdir -p {rwt}")
    O.rsync("--delete", "--exclude", "__pycache__", "--exclude", ".git", "--exclude", ".evo", local + "/", O.rp(f"{rwt}/"), check=False)
    return rwt


def full_eval(rwt, label, ref, gpu, strict=False, judge=True):
    ev = f"cd {O.REMOTE} && {O.EVO_PY} -m evo.evaluate --scenario base"
    g = f"{gpu} --strict-gpu" if strict else gpu
    return json.loads(O.sh(f"{ev} {rwt} --gpu {g} --mode full{' --ref ' + ref if ref else ''} --label {label}{'' if judge else ' --no-judge'}",
                           timeout=4 * 3600))


def summary(res):
    m = res.get("metrics") or {}
    return {"status": res.get("status"), "eval_id": res.get("eval_id"),
            "decode": {s: (m.get(s) or {}).get("decode_tok_s") for s in ("dev", "holdout")},
            "e2e": {s: (m.get(s) or {}).get("e2e_tok_s") for s in ("dev", "holdout")},
            "vs_seed": res.get("vs_ref"), "vs_seed_e2e": res.get("vs_ref_e2e"), "correctness": m.get("correctness")}


def post_run(eid, rwt, snaps, wall, seed_row, seed_dir, gpu, arm):
    """Snapshot curve (unpaired, against the seed's stored result), end state, then a paired final against the seed."""
    for k, s in enumerate(snaps, 1):
        record(dict(summary(full_eval(s, f"{eid}-c{k}-full", seed_row["result_path"], gpu)), id=f"{eid}-c{k}", arm=arm,
                    kind="snapshot", at_min=75 * k))
    res = full_eval(rwt, f"{eid}-end-full", seed_row["result_path"], gpu)
    record(dict(summary(res), id=f"{eid}-end", arm=arm, kind="end", at_min=round(wall / 60, 1)))
    pref = full_eval(seed_dir, f"{eid}-seedref", None, gpu, strict=True, judge=False)
    ref = f"{O.REMOTE_SH}/records/evals/{pref['eval_id']}.json" if pref.get("status") == "ok" else seed_row["result_path"]
    final = full_eval(rwt, f"{eid}-final", ref, gpu, strict=True)
    record(dict(summary(final), id=f"{eid}-final", arm=arm, kind="paired-final", seed_ref_eval=pref.get("eval_id")))
    return final
