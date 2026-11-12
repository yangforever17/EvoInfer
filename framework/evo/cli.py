"""`evo` - the only interface optimization agents need. Runs on the GPU host; agents reach it through bin/evo.
Environment (set by the orchestrator's job executor, never by the agent): EVO_AGENT_ID, EVO_GPU, EVO_WORKTREE,
EVO_PARENT_RESULT, EVO_MAX_EVALS, EVO_EPISODE_START, EVO_ARM, EVO_SCENARIO, EVO_ISOLATED."""
import argparse, glob, json, os, subprocess, sys, time, uuid
from evo.paths import RECORDS, SLOTS_DIR
from evo.evaluate import evaluate, acquire_gpu, NEED_FREE_GB
from evo.scenarios import SCENARIOS, current as current_scenario, model_path
from evo import board as B

AGENT = os.environ.get("EVO_AGENT_ID", "manual")
GPU = int(os.environ.get("EVO_GPU", "-1"))
WT = os.environ.get("EVO_WORKTREE", os.getcwd())
ADIR = os.path.join(RECORDS, "agents", AGENT); os.makedirs(ADIR, exist_ok=True)
ARM = os.environ.get("EVO_ARM", "")


def experiments():
    """Experiment rows this agent may see: the main line, plus its own ablation arm (arms never see each other)."""
    path = os.path.join(RECORDS, "experiments.jsonl")
    rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    if os.environ.get("EVO_ISOLATED"):   # no-harness baselines: no shared history, only its own arm
        return [r for r in rows if ARM and r.get("arm") == ARM]
    return [r for r in rows if not ARM or not r.get("arm") or r.get("arm") == ARM]

def fmt(res, parent):
    if res["status"] != "ok":
        return f"eval {res['eval_id']}: {res['status'].upper()}\n{res.get('log_tail', '')[-3000:]}"
    m = res["metrics"]; d = m["dev"]; lines = []
    vs = res.get("vs_ref", {}).get("dev")
    pv = parent["metrics"]["dev"]["decode_tok_s"] if parent and res["mode"] in ("dev", "full") else None
    lines.append(f"eval {res['eval_id']} ({res['mode']}): dev decode {d['decode_tok_s']:.2f} tok/s"
                 + (f" ({vs:+.1%} vs parent {pv:.2f})" if vs is not None and pv else (f" ({vs:+.1%} vs parent)" if vs is not None else ""))
                 + (f" | mean TTFT {d['mean_ttft_s']:.2f}s" if d.get("mean_ttft_s") is not None else " | batched: all requests submitted at once")
                 + f" | MTP accept {d['mtp_accept_rate']} | tokens/step {d['tokens_per_step']}")
    if res["mode"] == "quick" and parent and "quick_dev_tok_s" in parent:
        lines.append(f"parent quick-mode reference: {parent['quick_dev_tok_s']:.2f} tok/s")
    if "e2e_tok_s" in d:
        ve = res.get("vs_ref_e2e", {}).get("dev")
        lines.append((f"batch throughput (generated tokens / batch wall time) {d['e2e_tok_s']:.2f} tok/s" if d.get("mean_ttft_s") is None
                      else f"end-to-end (prefill + decode) {d['e2e_tok_s']:.2f} tok/s") + (f" ({ve:+.1%} vs parent)" if ve is not None else ""))
    if d.get("mean_ttft_s") is not None:
        lines.append(f"per-request decode tok/s: {d['per_request_tok_s']}")
    lines.append("note: quick-mode comparisons use a stored parent measurement; host load drifts, so differences under ~10% are noise")
    c = m.get("correctness")
    if c:
        lines.append(f"correctness: {'PASS' if c['passed'] else 'FAIL'} ({c['violations']} violations (gap > 1.0), {c.get('soft_violations', 0)} soft (0.5-1.0, max 0.1% of tokens), "
                     f"{c['near_tie_flips']} near-tie flips / {c['tokens']} tokens)"
                     + (f" examples (workload, request, [(token_idx, logit_gap)]): {c['violation_examples']}" if c["violation_examples"] else ""))
    else:
        lines.append("correctness: NOT CHECKED (--no-judge)")
    lines.append(f"engine load {res['load_s']}s; logs: {SLOTS_DIR}/{res['eval_id']}/")
    return "\n".join(lines)

def parent_result():
    p = os.path.expandvars(os.environ.get("EVO_PARENT_RESULT", ""))
    return json.load(open(p)) if p and os.path.exists(p) else None

def cmd_eval(a):
    n = len([f for f in glob.glob(os.path.join(ADIR, "eval-*.json"))
             if json.load(open(f)).get("status") != "gpu_busy"])   # GPU contention does not use up the budget
    cap = int(os.environ.get("EVO_MAX_EVALS", "12"))
    if n >= cap:
        print(f"evaluation budget exhausted ({n}/{cap}). Submit your best version with `evo submit`."); return 1
    mode = "dev" if a.dev else "quick"
    parent = parent_result()
    ref = parent.get("quick_ref") if (parent and mode == "quick") else parent
    res = evaluate(WT, GPU, mode, ref, f"{AGENT}-e{n + 1}", judge=not a.no_judge)
    json.dump(res, open(os.path.join(ADIR, f"eval-{len(glob.glob(os.path.join(ADIR, 'eval-*.json'))) + 1:02d}.json"), "w"), indent=1)
    print(fmt(res, parent)); print(f"[{n + 1}/{cap} evaluations used]")
    board_hint()
    return 0


def board_hint():
    seen_f = os.path.join(ADIR, "board_seen")
    seen = int(open(seen_f).read()) if os.path.exists(seen_f) else 0
    new = [e for e in B.events()[seen:] if e.get("agent") != AGENT]
    if new:
        print(f"board: {len(new)} new entries from other agents since you last looked - run `evo board`")


def cmd_board(a):
    evs = B.events()
    open(os.path.join(ADIR, "board_seen"), "w").write(str(len(evs)))
    claims = B.active_claims(evs)
    print("== Work in progress (active claims) ==")
    for c in sorted(claims, key=lambda c: c["ts"]):
        me = " (yours)" if c["agent"] == AGENT else (" (queued for integration: accepted or pooled change)" if B.is_queue(c["agent"]) else "")
        print(f"- [{c['cid']}] {c['agent']}{me}, {B.ago(c['ts'])}: {c['hypothesis']}\n    where: {', '.join(c['where'])}")
    if not claims:
        print("(none)")
    notes = [e for e in evs if (e["type"] == "note" and e.get("text")) or (e["type"] == "release" and e.get("reason"))][-a.n:]
    print("\n== Notes and abandoned ideas from running/recent agents ==")
    for e in notes:
        tag = ("integration result" if B.is_queue(e["agent"]) else "abandoned") if e["type"] == "release" else "note"
        print(f"- {e['agent']} ({tag}, {B.ago(e['ts'])}): {e.get('text') or e.get('reason')}")
    if not notes:
        print("(none)")
    import time
    t0 = os.environ.get("EVO_EPISODE_START", "")
    start = time.strftime("%F %T", time.localtime(int(t0))) if t0.isdigit() else ""
    done = [r for r in experiments() if start and r.get("ended", "") >= start]
    print("\n== Experiments finished since your episode started ==")
    for r in done:
        print(f"- {r['id']} verdict={r['verdict']} dev={r.get('dev_speedup')} holdout={r.get('holdout_speedup')}: {r.get('summary', '')[:300]}")
    if not done:
        print("(none)")
    return 0


def cmd_claim(a):
    where = [w for w in a.where if w.strip()]
    if not where:
        print("give at least one --where path[:function_or_class]"); return 2
    others = [c for c in B.active_claims() if c["agent"] != AGENT]
    clashes = [(c, B.overlaps(where, c["where"])) for c in others]
    clashes = [(c, h) for c, h in clashes if h]
    if clashes and not a.force:
        print("CLAIM REFUSED - overlaps work another agent is doing right now:")
        for c, h in clashes:
            print(f"- {c['agent']} [{c['cid']}]: {c['hypothesis']}\n    overlapping: {h}")
            if B.is_queue(c["agent"]):
                print("    (this change is accepted or pooled and will be merged into the best engine; building on the same"
                      " function now means redoing it or a merge conflict)")
        print("Pick a different bottleneck, or if your change is genuinely independent of theirs re-run with "
              "--force \"<why it does not duplicate or conflict>\". Two agents editing the same function produce merge conflicts.")
        B.append({"type": "refused", "agent": AGENT, "hypothesis": a.message, "where": where,
                  "against": [c["cid"] for c, _ in clashes]})
        return 3
    c = B.new_claim(AGENT, a.message + (f" [forced: {a.force}]" if a.force else ""), where)
    print(f"claimed [{c['cid']}]: {a.message} @ {', '.join(where)}")
    return 0


def cmd_release(a):
    B.append({"type": "release", "agent": AGENT, "cid": a.cid, "reason": a.message})
    print("released" + (f" [{a.cid}]" if a.cid else " all your claims"))
    return 0


def cmd_note(a):
    B.append({"type": "note", "agent": AGENT, "text": a.message})
    print("posted")
    return 0

def cmd_run(a):
    sc = SCENARIOS[current_scenario()]
    g, lock, waited = acquire_gpu(GPU, need=float(os.environ.get("EVO_RUN_NEED_GB", sc.get("need_gb", NEED_FREE_GB))))
    if g is None:
        print(f"no GPU with enough free memory (other users' jobs) after {waited / 60:.0f} min; try again later."); return 75
    if g != GPU:
        print(f"[evo] GPU {GPU} was busy; running on GPU {g} instead", flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONPATH=WT, EVO_MODEL=model_path(current_scenario()),
               PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
    cmd = a.cmd
    if os.environ.get("EVO_ISOLATED"):   # no-harness baselines: the command sees only its own directory, the Python install and the model
        home, venv = os.path.expanduser("~"), os.path.dirname(os.path.dirname(sys.executable))
        uv = os.path.join(home, ".local/share/uv/python")
        cmd = ["bwrap", "--ro-bind", "/", "/", "--tmpfs", home, "--tmpfs", "/tmp", "--dev-bind", "/dev", "/dev", "--proc", "/proc",
               "--ro-bind", venv, venv] + (["--ro-bind", uv, uv] if os.path.isdir(uv) else []) + [
               "--bind", WT, WT, "--chdir", WT, "--die-with-parent", "--"] + cmd
    try:
        return subprocess.run(cmd, cwd=WT, env=env, timeout=a.timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"command timed out after {a.timeout}s"); return 124

def cmd_history(a):
    rows = experiments()
    if not rows: print("no previous experiments"); return 0
    for r in rows[-a.n:]:
        print(f"- {r['id']} [{r['agent_kind']}] verdict={r['verdict']} dev={r.get('dev_speedup')} holdout={r.get('holdout_speedup')}"
              f" parent={r['parent'][:8]}\n  {r.get('summary', '')[:600]}")
    return 0

def cmd_submit(a):
    subprocess.run(["git", "add", "-A"], cwd=WT, check=True)
    subprocess.run(["git", "-c", f"user.name=evo-{AGENT}", "-c", "user.email=evo@local", "commit", "-q", "--allow-empty", "-m", a.message], cwd=WT)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=WT, capture_output=True, text=True).stdout.strip()
    json.dump({"commit": head, "message": a.message}, open(os.path.join(ADIR, "submission.json"), "w"), indent=1)
    B.append({"type": "release", "agent": AGENT, "cid": None, "reason": None})
    print(f"submitted {head[:10]}. The orchestrator will run the full evaluation (dev + hidden holdout). You can stop now.")
    return 0

def main():
    ap = argparse.ArgumentParser(prog="evo"); sub = ap.add_subparsers(dest="sub", required=True)
    e = sub.add_parser("eval", help="benchmark + correctness-check the current worktree on your GPU")
    e.add_argument("--dev", action="store_true", help="all dev prompts at the final evaluation's length (slower, closer to the final eval)")
    e.add_argument("--no-judge", action="store_true", help="skip the correctness judge (faster; never trust such numbers)")
    r = sub.add_parser("run", help="run a command on your GPU with your worktree on PYTHONPATH (profiling etc.)")
    r.add_argument("--timeout", type=int, default=1800); r.add_argument("cmd", nargs=argparse.REMAINDER)
    h = sub.add_parser("history", help="earlier experiments (accepted and rejected)"); h.add_argument("-n", type=int, default=20)
    s = sub.add_parser("submit", help="commit and hand over for the final evaluation"); s.add_argument("-m", "--message", required=True)
    b = sub.add_parser("board", help="what other agents are working on right now, their notes, and fresh results")
    b.add_argument("-n", type=int, default=20)
    c = sub.add_parser("claim", help="announce the change you are about to implement (refused if it overlaps another agent's)")
    c.add_argument("-m", "--message", required=True, help="hypothesis + planned change")
    c.add_argument("--where", action="append", default=[], help="path[:function_or_class] you will edit; repeatable")
    c.add_argument("--force", default="", help="override an overlap, with the reason")
    rl = sub.add_parser("release", help="give up a claim; the message tells others what you learned")
    rl.add_argument("-m", "--message", required=True); rl.add_argument("--cid")
    n = sub.add_parser("note", help="post a measurement or finding for the other agents"); n.add_argument("-m", "--message", required=True)
    argv = None
    if os.environ.get("EVO_ARGV_B64"):   # arguments shipped from bin/evo without shell quoting
        import base64
        argv = [x for x in base64.b64decode(os.environ.pop("EVO_ARGV_B64")).decode().split("\0") if x != ""]
    a = ap.parse_args(argv)
    if a.sub == "run":
        a.cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
        if not a.cmd: ap.error("evo run -- <command>")
    sys.exit({"eval": cmd_eval, "run": cmd_run, "history": cmd_history, "submit": cmd_submit, "board": cmd_board,
              "claim": cmd_claim, "release": cmd_release, "note": cmd_note}[a.sub](a))

if __name__ == "__main__":
    main()
