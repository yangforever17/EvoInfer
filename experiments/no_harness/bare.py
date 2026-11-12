"""No-harness baseline 1, the bare agent: one LLM in a minimal tool loop - a single `bash` tool and a short prompt, no
agent CLI, no skills, no history, no board, no parallel agents, no recombination, no accept/reject. Every command runs
in bubblewrap with only the engine directory writable.

Same seed as the evolution arm it is compared with, the same benchmark and correctness check (`evo eval`), and the
same caps as one arm in total (6 x 75 = 450 agent-minutes, 6 x 8 = 48 evaluations). Snapshots every 75 minutes are
evaluated afterwards (full, judge) for the record; at the end the directory is measured paired against the seed.
Records: <home>/records/no_harness.jsonl on the GPU host; transcript runs/<eid>/bare.jsonl.

  EVO_LLM_BASE_URL=... EVO_LLM_API_KEY=... python3 experiments/no_harness/bare.py --gpu 0 --model qwen3.8-max
  (--seed-lineage pq to start from an arm's root; --minutes 3 for a smoke test)

The paper's run (Qwen3.8-Max, 450 min): 1.0x - the model found the bottleneck but never changed the code.
"""
import argparse
import json
import os
import subprocess
import time
import urllib.request

import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, required=True); ap.add_argument("--model", required=True)
ap.add_argument("--minutes", type=int, default=450); ap.add_argument("--max-evals", type=int, default=48)
ap.add_argument("--seed-lineage", default="", help="lineage whose root is the seed (default: the main line)")
A = ap.parse_args()
ARM = "nh-bare"
OUT_MAX = 8000           # characters of command output returned to the model
SYSTEM = "You are a software engineer working in a Linux shell. Use the bash tool to run commands. Work autonomously."
TOOLS = [{"type": "function", "function": {
    "name": "bash", "description": "Run a bash command in the working directory and return its output (stdout+stderr, truncated).",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
BASE, KEY = C.endpoint()


def chat(messages):
    body = json.dumps({"model": A.model, "messages": messages, "tools": TOOLS, "tool_choice": "auto"}).encode()
    for i in range(6):
        try:
            req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                         headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.load(r)
        except Exception as e:      # transient endpoint errors
            print(f"[bare] API error {type(e).__name__} {getattr(e, 'code', '')}; retry {i + 1}", flush=True)
            time.sleep(30 * (i + 1))
    raise RuntimeError("model endpoint unreachable")


def run_cmd(cmd, local, env):
    try:
        p = subprocess.run(C.sandbox(local, local) + ["bash", "-c", cmd], cwd=local, env=env, capture_output=True, text=True,
                           timeout=C.CMD_TIMEOUT, stdin=subprocess.DEVNULL)
        out = (p.stdout + p.stderr) + f"\n[exit code {p.returncode}]"
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")) + f"\n[timed out after {C.CMD_TIMEOUT}s]"
    return out if len(out) <= OUT_MAX else out[:2000] + f"\n[... {len(out) - OUT_MAX} characters omitted ...]\n" + out[-(OUT_MAX - 2000):]


def compact(messages, keep=24):
    """Keep the context bounded the simplest way: old command outputs are elided, the rest stays."""
    for m in messages[2:-keep]:
        if m["role"] == "tool" and not m["content"].startswith("[elided"):
            m["content"] = "[elided old output] " + m["content"][:300]


def main():
    seed, seed_dir = C.seed(A.seed_lineage)
    eid = f"{time.strftime('%m%d%H%M')}-bare-{A.model.split('-')[0]}"
    rwt, local = C.prepare(eid, seed)
    env = C.agent_env(eid, local)
    log = open(os.path.join(C.O.RUNS, eid, "bare.jsonl"), "w")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": C.TASK.format(minutes=A.minutes, max_evals=A.max_evals)}]
    for m in messages:
        log.write(json.dumps(m) + "\n")
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "commands": 0}
    t0 = time.time(); snaps = []; nudged = 0
    print(f"[{eid}] start {time.strftime('%F %T')} model {A.model}, {A.minutes} min, {A.max_evals} evals, GPU {A.gpu}", flush=True)
    with C.executor(eid, local, A.gpu, seed, A.max_evals, ARM):
        while True:
            left = A.minutes * 60 - (time.time() - t0)
            while len(snaps) < 6 and time.time() - t0 >= (len(snaps) + 1) * C.SNAP_EVERY:
                snaps.append(C.snapshot(local, eid, len(snaps) + 1))
                print(f"[{eid}] snapshot {len(snaps)} at {(time.time() - t0) / 60:.0f} min", flush=True)
            if left <= 0:
                print(f"[{eid}] time budget used", flush=True); break
            compact(messages)
            r = chat(messages)
            u = r.get("usage") or {}
            usage["prompt_tokens"] += u.get("prompt_tokens", 0); usage["completion_tokens"] += u.get("completion_tokens", 0); usage["calls"] += 1
            msg = r["choices"][0]["message"]
            keep = {"role": "assistant", "content": msg.get("content") or ""}
            if msg.get("tool_calls"):
                keep["tool_calls"] = msg["tool_calls"]
            messages.append(keep); log.write(json.dumps(keep) + "\n"); log.flush()
            if not msg.get("tool_calls"):
                if "DONE" in (msg.get("content") or "") or nudged >= 3:
                    print(f"[{eid}] agent stopped after {(time.time() - t0) / 60:.1f} min", flush=True); break
                nudged += 1
                nxt = {"role": "user", "content": C.NUDGE.format(left=int(left / 60), evals=A.max_evals - C.n_evals(eid))}
                messages.append(nxt); log.write(json.dumps(nxt) + "\n"); continue
            for tc in msg["tool_calls"]:
                try:
                    cmd = json.loads(tc["function"]["arguments"]).get("command", "")
                except ValueError:
                    cmd = ""
                out = run_cmd(cmd, local, env) if cmd else "error: the bash tool needs a JSON argument {\"command\": \"...\"}"
                usage["commands"] += 1
                tm = {"role": "tool", "tool_call_id": tc["id"], "content": out}
                messages.append(tm); log.write(json.dumps({**tm, "command": cmd, "t_min": round((time.time() - t0) / 60, 1)}) + "\n"); log.flush()
    wall = time.time() - t0
    C.O.sync_worktree(eid, local)
    n_ev = C.n_evals(eid)
    print(f"[{eid}] finished after {wall / 60:.1f} min, {n_ev} evals, usage {usage}", flush=True)
    C.record({"id": eid, "arm": ARM, "kind": "run", "model": A.model, "seed": seed["tag"], "agent_wall_min": round(wall / 60, 1),
              "agent_evals": n_ev, "usage": usage, "snapshots": snaps, "ended": time.strftime("%F %T")})
    final = C.post_run(eid, rwt, snaps, wall, seed, seed_dir, A.gpu, ARM)
    print(f"[{eid}] paired final vs seed: {json.dumps(C.summary(final))}", flush=True)


if __name__ == "__main__":
    main()
