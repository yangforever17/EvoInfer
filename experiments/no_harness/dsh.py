"""No-harness baseline 2: the same model as the bare agent inside a general-purpose agent framework, DeepSeek Harness
(`dsh`, npm @deepseek-ai/dsh), in its one-shot headless mode, instead of EvoInfer. Everything else equals bare.py: the
seed, the task text, `evo eval` / `evo run` as the only GPU access, 450 agent-minutes and 48 evaluations in total,
snapshots every 75 minutes, a paired final against the seed.

dsh runs as shipped (its tools, subagents, compaction, system prompt) with three settings: the model route (a custom
OpenAI-compatible provider pointing at a local forwarder), the default model, and the bash tool timeout raised from
60 s to 2400 s (the same per-command limit as the bare loop; `evo eval` takes ~10 minutes). dsh's own sandbox is set to
danger-full-access because the whole dsh process runs inside the same bubblewrap box as the bare agent's commands
(home and /tmp empty, only the run directory writable). The forwarder runs outside the box and adds the API key, so the
key never enters the agent's environment.

When dsh finishes its task early, the session is resumed (`--session-id`) with the remaining budget, at most 3 times,
as the bare loop nudges. Records: <home>/records/no_harness.jsonl (arm nh-dsh); events runs/<eid>/dsh-<k>.jsonl.

  npm i -g @deepseek-ai/dsh        (Node >= 24.9; or set EVO_DSH_BIN to its path)
  EVO_LLM_BASE_URL=... EVO_LLM_API_KEY=... python3 experiments/no_harness/dsh.py --gpu 0 --model qwen3.8-max
  (--smoke: a tiny task, no evaluations afterwards)

The paper's run (Qwen3.8-Max; the endpoint's quota ran out after 200.7 of 450 minutes): 7.37x / 7.97x decode.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import time

import common as C
from evo.llm_forward import Forwarder

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, required=True); ap.add_argument("--model", required=True)
ap.add_argument("--minutes", type=int, default=450); ap.add_argument("--max-evals", type=int, default=48)
ap.add_argument("--seed-lineage", default="", help="lineage whose root is the seed (default: the main line)")
ap.add_argument("--smoke", action="store_true", help="tiny task, no evaluations afterwards")
A = ap.parse_args()
MODEL, ARM = A.model, "nh-dsh"
DSH = shutil.which(os.environ.get("EVO_DSH_BIN", "dsh")) or os.environ.get("EVO_DSH_BIN", "dsh")
TASK = C.TASK if not A.smoke else ("List this directory, then run `evo run -- nvidia-smi -L` and `evo eval --help`, "
                                   "report what they printed, and reply DONE.")


def install_dirs():
    """Read-only binds for dsh and node when they live under the (hidden) home directory."""
    out = []
    for exe in (DSH, shutil.which("node")):
        if not exe:
            continue
        p = os.path.realpath(exe)
        root = p.split("/node_modules/")[0] if "/node_modules/" in p else os.path.dirname(os.path.dirname(p))
        if root.startswith(C.HOME) and root not in out:
            out.append(root)
    return out


# ---------------------------------------------------------------- the model route: a key-adding local forwarder
BASE, KEY = C.endpoint()
FW = Forwarder(BASE, KEY, auth="bearer", strip_prefix="/v1")   # honours $HTTPS_PROXY of this process, not the agent's
USAGE = FW.usage


# ---------------------------------------------------------------- dsh inside bubblewrap
def write_patch(path, port):
    with open(path, "w") as f:
        f.write(f"""# dsh.py: the only settings changed from the shipped headless profile
- id: llm-pi-ai
  config:
    providers:
      evo-gw:
        displayName: EvoInfer model endpoint
        apiKeyEnv: DSH_GW_KEY
        api: openai-completions
        baseURL: http://127.0.0.1:{port}/v1
        models:
          - id: {MODEL}
            contextWindow: 262144
            maxTokens: 32768
- id: agent-default-model
  config:
    provider: evo-gw
    model: {MODEL}
- id: bash-sandbox
  config:
    timeoutMs: {C.CMD_TIMEOUT * 1000}
""")


def dsh_cmd(local, run_dir, patch, session):
    return C.sandbox(run_dir, local, ro=install_dirs()) + [
        DSH, "--profile", "headless", "--patch", patch, "--json", "--permission-mode", "danger-full-access",
        "--approval-policy", "never"] + (["--session-id", session] if session else []) + ["-"]


def run_once(k, text, session, local, run_dir, patch, env, deadline, snap):
    out_path = os.path.join(run_dir, f"dsh-{k}.jsonl")
    with open(out_path, "w") as out, open(os.path.join(run_dir, f"dsh-{k}.err"), "w") as err:
        p = subprocess.Popen(dsh_cmd(local, run_dir, patch, session), cwd=local, env=env, stdin=subprocess.PIPE,
                             stdout=out, stderr=err, text=True, start_new_session=True)
        p.stdin.write(text); p.stdin.close()
        killed = False
        while p.poll() is None:
            snap()
            if time.time() >= deadline and not killed:
                os.killpg(p.pid, signal.SIGTERM); killed = True; kill_at = time.time()
            if killed and time.time() - kill_at > 60:
                os.killpg(p.pid, signal.SIGKILL)
            time.sleep(10)
    sid, final = session, None
    for ln in open(out_path):
        try:
            ev = json.loads(ln)
        except ValueError:
            continue
        if ev.get("type") == "session":
            sid = ev.get("sessionId") or sid
        elif ev.get("type") == "final":
            final = ev
            sid = ev.get("sessionId") or sid
    return p.returncode, sid, final, killed


def main():
    seed, seed_dir = C.seed(A.seed_lineage)
    eid = f"{time.strftime('%m%d%H%M')}-dsh-{MODEL.split('-')[0]}" + ("-smoke" if A.smoke else "")
    rwt, local = C.prepare(eid, seed)
    run_dir = os.path.dirname(local)
    os.makedirs(os.path.join(run_dir, "dshhome"), exist_ok=True)
    port = FW.start()
    patch = os.path.join(run_dir, "dsh.patch.yml")
    write_patch(patch, port)
    env = C.agent_env(eid, local)
    env.update(DSH_HOME=os.path.join(run_dir, "dshhome"), DSH_GW_KEY="local-forwarder")
    t0 = time.time(); deadline = t0 + A.minutes * 60; snaps = []

    def snap():
        while not A.smoke and len(snaps) < 6 and time.time() - t0 >= (len(snaps) + 1) * C.SNAP_EVERY:
            snaps.append(C.snapshot(local, eid, len(snaps) + 1))
            print(f"[{eid}] snapshot {len(snaps)} at {(time.time() - t0) / 60:.0f} min, usage {USAGE}", flush=True)

    print(f"[{eid}] start {time.strftime('%F %T')} dsh + {MODEL}, {A.minutes} min, {A.max_evals} evals, GPU {A.gpu}, forwarder :{port}", flush=True)
    text, session, runs = TASK.format(minutes=A.minutes, max_evals=A.max_evals), None, []
    with C.executor(eid, local, A.gpu, seed, A.max_evals, ARM):
        for k in range(1, 5):                                   # the task, then at most 3 resumptions
            rc, session, final, killed = run_once(k, text, session, local, run_dir, patch, env, deadline, snap)
            reason = (final or {}).get("reason")
            runs.append({"k": k, "rc": rc, "session": session, "killed": killed, "reason": reason,
                         "t_min": round((time.time() - t0) / 60, 1)})
            print(f"[{eid}] dsh run {k} exit {rc} after {(time.time() - t0) / 60:.1f} min, killed={killed}, reason={json.dumps(reason)[:200]}", flush=True)
            left = deadline - time.time()
            if killed or left <= 60 or not session or "DONE" in str(final or "") or k == 4:
                break
            text = C.NUDGE.format(left=int(left / 60), evals=A.max_evals - C.n_evals(eid))
        snap()
    wall = time.time() - t0
    C.O.sync_worktree(eid, local)
    n_ev = C.n_evals(eid)
    print(f"[{eid}] finished after {wall / 60:.1f} min, {n_ev} evals, usage {USAGE}", flush=True)
    if A.smoke:
        return
    C.record({"id": eid, "arm": ARM, "kind": "run", "harness": "deepseek-harness", "model": MODEL, "seed": seed["tag"],
              "agent_wall_min": round(wall / 60, 1), "agent_evals": n_ev, "usage": dict(USAGE), "dsh_runs": runs,
              "snapshots": snaps, "ended": time.strftime("%F %T")})
    final = C.post_run(eid, rwt, snaps, wall, seed, seed_dir, A.gpu, ARM)
    print(f"[{eid}] paired final vs seed: {json.dumps(C.summary(final))}", flush=True)


if __name__ == "__main__":
    main()
