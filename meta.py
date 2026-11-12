#!/usr/bin/env python3
"""Meta level: a meta-agent evolves the optimization agents' skills from what the earlier episodes did.

  python3 meta.py evolve --base v0 --new v1 [--minutes 60]
  python3 meta.py evolve --base v0 --new v1 --dry-run        # only build the meta-agent's inputs, to inspect them

Inputs it gets (in its workspace, read-only by convention): every main-line experiment row, the coordination board,
one digest per episode (the agent's messages, commands, `evo` outputs, edits, every evaluation's result), and the
reports in reports/ (optional). Output: an edited copy of the skills plus CHANGELOG.md, saved as skills/<new>.
The meta-agent never touches the engine or the evaluator; compare skill versions with paired episodes
(orchestrator `episode --skills <version> --arm <name>`), see docs/architecture.md.
"""
import argparse, glob, json, os, re, shutil, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orchestrator import AG, HERE, RUNS, SKILL_VERSIONS, REMOTE, remote_argv, sh, parse_cost

MAX_SKILL_BYTES = 60_000


def trajectory(eid):
    log = os.path.join(RUNS, eid, "codex.jsonl")
    if not os.path.exists(log):
        return None
    wt = os.path.join(RUNS, eid, "engine")
    out = []
    for line in open(log, errors="replace"):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict) or ev.get("type") != "item.completed":
            continue
        it = ev.get("item") or {}
        kind = it.get("type")
        if kind == "agent_message":
            out.append("AGENT: " + it.get("text", "")[:1500])
        elif kind == "command_execution":
            cmd = it.get("command", "")
            m = re.match(r"^/bin/bash -lc (['\"])(.*)\1$", cmd, re.S)
            cmd = (m.group(2) if m else cmd).replace("\n", " ")
            o, rc = (it.get("aggregated_output") or "").strip(), it.get("exit_code")
            if re.search(r"\bevo (eval|submit|claim|release|note|run)\b", cmd):
                body = o.splitlines()[:14] if "evo run" not in cmd else o.splitlines()[-8:]
            elif rc not in (0, None):
                body = o.splitlines()[-4:]
            else:
                body = []
            out.append(f"$ {cmd[:400]}  [exit {rc}]" + "".join("\n    " + b[:300] for b in body))
        elif kind == "file_change":
            out.append("EDIT: " + ", ".join(f"{c.get('kind')} {os.path.relpath(c.get('path', ''), wt)}" for c in it.get("changes", [])))
    return out


def remote_evals():
    """Every agent-run evaluation (status, gains, correctness), keyed by episode."""
    code = r'''
import glob, json, os
out = {}
for f in sorted(glob.glob(os.path.expanduser("__RECORDS__/agents/*/eval-*.json"))):
    r = json.load(open(f)); c = (r.get("metrics") or {}).get("correctness") or {}
    out.setdefault(f.split("/")[-2], []).append({"mode": r.get("mode"), "status": r.get("status"),
        "dev_decode": ((r.get("metrics") or {}).get("dev") or {}).get("decode_tok_s"), "vs_parent": r.get("vs_ref"),
        "vs_parent_e2e": r.get("vs_ref_e2e"), "correct": c.get("passed"), "violations": c.get("violations"),
        "soft": c.get("soft_violations"), "queued_s": r.get("queued_s")})
print(json.dumps(out))
'''
    return json.loads(sh(f"python3 - <<'EOF'\n{code.replace('__RECORDS__', REMOTE + '/records')}\nEOF"))


def build_inputs(ws):
    inp = os.path.join(ws, "inputs"); os.makedirs(os.path.join(inp, "episodes"), exist_ok=True)
    rows = [json.loads(l) for l in sh(f"cat {REMOTE}/records/experiments.jsonl").splitlines()]
    rows = [r for r in rows if not r.get("arm")]
    open(os.path.join(inp, "experiments.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in rows))
    open(os.path.join(inp, "board.jsonl"), "w").write(sh(f"cat {REMOTE}/records/board.jsonl"))
    for rep in sorted(glob.glob(os.path.join(HERE, "reports", "*.md"))):   # optional write-ups of earlier rounds
        shutil.copy(rep, inp)
    evals = remote_evals()
    latest = {}
    for r in rows:
        latest[r["id"]] = r
    table = ["| episode | focus | verdict | dev | holdout | evals (status) | wall min | output tokens |", "|---|---|---|---|---|---|---|---|"]
    for eid, r in latest.items():
        ev = evals.get(eid, [])
        st = ",".join(e["status"] if e["status"] != "ok" else ("ok" if e["correct"] in (True, None) else "WRONG") for e in ev)
        out_tok = ((r.get("cost") or {}).get("usage") or {}).get("output_tokens")
        table.append(f"| {eid} | {r['focus']} | {r['verdict']} | {r.get('dev_speedup')} | {r.get('holdout_speedup')} | {len(ev)} ({st}) | {r.get('agent_wall_min')} | {out_tok} |")
        traj = trajectory(eid)
        if traj is None and r["agent_kind"] == "none":
            continue
        with open(os.path.join(inp, "episodes", f"{eid}.md"), "w") as f:
            f.write(f"# {eid}\n\n```json\n{json.dumps(r, indent=1)}\n```\n\n## Evaluations run by the agent\n")
            for i, e in enumerate(ev, 1):
                f.write(f"{i}. {json.dumps(e)}\n")
            f.write("\n## Trajectory (agent messages, commands with exit code, evo outputs, edits)\n\n")
            f.write("\n".join(traj or ["(no transcript)"]) + "\n")
    open(os.path.join(inp, "overview.md"), "w").write(
        "# All main-line experiments (latest row per id)\n\nverdicts: accepted = promoted to best; accepted_sibling = beat its "
        "parent but the best had moved (merged later by `integrate`); no_gain; rejected_*; merge_conflict.\n"
        "dev / holdout = paired speedup vs parent (decode). Evaluation status WRONG = failed the correctness judge.\n\n"
        + "\n".join(table) + "\n")
    return len(latest)


PROMPT = """You are the meta-agent of EvoInfer. Optimization agents (Codex, like you) improve an LLM inference engine
(nano-vllm fork, Qwen3.5-35B-A3B: Gated DeltaNet + gated attention, 256-expert MoE, MTP speculative decoding; one
80 GB GPU) in episodes of <= {max_evals} GPU evaluations and ~90 minutes. Before each episode they read the skills in
`skills/` (start: skills/evo-workflow/SKILL.md). Your job is to improve those skills so that FUTURE episodes do better.

Better means, measured on new episodes from the current best engine:
more episodes whose change is accepted (> 3% paired gain on dev AND hidden holdout, no regression, correctness pass),
larger accepted gains, and less waste per episode (evaluations that crash, fail the judge, or measure noise;
time spent re-deriving what earlier agents already measured; duplicated or conflicting work).

Evidence (read it; do not trust your priors over it):
- inputs/overview.md - every episode: focus, verdict, gains, evaluation statuses, cost.
- inputs/episodes/<id>.md - per episode: its record, every evaluation it ran, and its trajectory (messages,
  commands, `evo` outputs, edits). Compare accepted episodes with no_gain / rejected ones: what did the successful
  ones do differently (how they measured, how they kept numerics exact, how they scoped the change, when they
  stopped)? Where did time and evaluations go in the failed ones?
- inputs/experiments.jsonl, inputs/board.jsonl - raw records; inputs/*.md - the operators' write-ups of earlier rounds, if any (reports/*.md).

Rules:
1. Edit only files under skills/. You may rewrite, split, merge, add, or delete skills. Each skill is a directory
   with SKILL.md starting with YAML frontmatter (`name`, `description`). You may add small helper scripts next to a
   SKILL.md (e.g. a paired A/B timing harness, an exactness checker) that agents can copy into their worktree; test
   that they at least parse (`python3 -m py_compile`). No GPU is available to you.
2. Every change must be grounded in the evidence: cite episode ids. Prefer durable, procedural knowledge (how to
   measure, how to keep bf16 numerics exact, how to scope and verify a change, how to use the board and the eval
   budget, known pitfalls of this engine) over a list of optimizations already done - agents already see the
   history via `evo history`, and the engine they start from already contains every accepted change.
   Known-dead ends with their measured reason ARE worth keeping, compactly.
3. Do not weaken or reinterpret the rules of the evaluation (correctness gate, no special-casing prompts, no
   touching other processes or GPUs). They are enforced by the evaluator regardless of what skills say.
4. Keep the total size of skills/ under {max_kb} KB and keep evo-workflow/SKILL.md the short entry point
   (other skills are read on demand). Agents have limited attention: cut what does not change behaviour.
5. Write skills/CHANGELOG.md: one entry per change - what changed, the evidence (episode ids, numbers), and the
   effect you expect on future episodes. This file is not shown to the optimization agents.

Work for at most ~{minutes} minutes. Finish by printing a short summary of your changes."""


def validate(skills):
    total, problems = 0, []
    for root, _, files in os.walk(skills):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    for d in sorted(os.listdir(skills)):
        p = os.path.join(skills, d)
        if not os.path.isdir(p):
            continue
        sk = os.path.join(p, "SKILL.md")
        if not os.path.exists(sk):
            problems.append(f"{d}: no SKILL.md")
        elif not re.match(r"^---\nname: .+\ndescription: .+\n---", open(sk).read()):
            problems.append(f"{d}: bad frontmatter")
    if not os.path.exists(os.path.join(skills, "evo-workflow", "SKILL.md")):
        problems.append("evo-workflow/SKILL.md missing")
    if not os.path.exists(os.path.join(skills, "CHANGELOG.md")):
        problems.append("CHANGELOG.md missing")
    if total > MAX_SKILL_BYTES:
        problems.append(f"skills total {total} bytes > {MAX_SKILL_BYTES}")
    return total, problems


def evolve(a):
    dst = os.path.join(SKILL_VERSIONS, a.new)
    if os.path.exists(dst):
        sys.exit(f"{dst} exists")
    ws = os.path.join(RUNS, f"meta-{a.new}")
    shutil.rmtree(ws, ignore_errors=True); os.makedirs(ws)
    shutil.copytree(os.path.join(SKILL_VERSIONS, a.base), os.path.join(ws, "skills"))
    n = build_inputs(ws)
    print(f"[meta-{a.new}] {n} experiments digested into {ws}/inputs", flush=True)
    prompt = PROMPT.format(max_evals=10, max_kb=MAX_SKILL_BYTES // 1000, minutes=a.minutes)
    cmd = [AG["codex_bin"], "exec", "--json", "--sandbox", "workspace-write",
           "--skip-git-repo-check", "-C", ws, prompt]
    log = os.path.join(ws, "codex.jsonl"); t0 = time.time()
    with open(log, "w") as f:
        try:
            subprocess.run(cmd, cwd=ws, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=a.minutes * 60 + 900)
        except subprocess.TimeoutExpired:
            pass
    cost, wall = parse_cost("codex", log), time.time() - t0
    size, problems = validate(os.path.join(ws, "skills"))
    print(f"[meta-{a.new}] finished after {wall / 60:.1f} min, cost {cost}, skills {size} bytes, problems {problems}", flush=True)
    if problems:
        sys.exit("not saved: " + "; ".join(problems))
    shutil.copytree(os.path.join(ws, "skills"), dst)
    diff = subprocess.run(["diff", "-ruN", os.path.join(SKILL_VERSIONS, a.base), dst], capture_output=True, text=True).stdout
    open(os.path.join(ws, f"{a.base}-to-{a.new}.diff"), "w").write(diff)
    row = {"version": a.new, "base": a.base, "experiments_seen": n, "bytes": size, "wall_min": round(wall / 60, 1),
           "cost": cost, "diff_lines": diff.count("\n"), "ended": time.strftime("%F %T")}
    subprocess.run(remote_argv(f"cat >> {REMOTE}/records/skills.jsonl"), input=json.dumps(row) + "\n", text=True, check=True)
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evolve"); e.add_argument("--base", required=True); e.add_argument("--new", required=True)
    e.add_argument("--minutes", type=int, default=60)
    e.add_argument("--dry-run", action="store_true", help="only build the inputs")
    a = ap.parse_args()
    if a.dry_run:
        ws = os.path.join(RUNS, f"meta-{a.new}-dry"); shutil.rmtree(ws, ignore_errors=True); os.makedirs(ws)
        print(build_inputs(ws), "experiments ->", ws)
    else:
        evolve(a)
