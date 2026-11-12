"""Evaluate a candidate engine directory: snapshot -> bench (candidate code) -> judge (frozen reference).
Usage: python -m evo.evaluate <engine_dir> --gpu N [--mode quick|dev|full] [--ref result.json] [--label x] [--scenario s]"""
import argparse, fcntl, json, os, shutil, statistics, subprocess, sys, time, uuid
from evo import config
from evo.paths import VENV_PY, FRAMEWORK, JUDGE_ENGINE, SLOTS_DIR, RECORDS, LOCKS

from evo.scenarios import SCENARIOS, current as current_scenario, model_path
MODES = SCENARIOS["base"]["modes"]   # workloads, prompts per workload (0 = all), new tokens - per scenario in scenarios.py

NEED_FREE_GB = float(config.CFG["gpus"]["need_gb"])        # Qwen3.5-35B-A3B: BF16 weights (67 GB) + KV cache + activations
MAX_GPU_WAIT_S = int(config.CFG["gpus"]["max_wait_min"] * 60)
REASSIGN_AFTER_S = int(config.CFG["gpus"]["reassign_after_min"] * 60)   # preferred GPU still busy -> any idle pool GPU
GPU_POOL = config.gpu_pool()

def gpu_free_gb(gpu):
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(gpu)],
                         capture_output=True, text=True).stdout.strip()
    return int(out) / 1024 if out else 0.0

def wait_for_gpu(gpu, need=NEED_FREE_GB, max_wait=MAX_GPU_WAIT_S):
    """Other users share this machine: wait until the GPU has room instead of crashing with OOM."""
    t0 = time.time()
    while True:
        free = gpu_free_gb(gpu)
        if free >= need:
            return True, free
        if time.time() - t0 > max_wait:
            return False, free
        time.sleep(30)

def _try_lock(g, need):
    """Non-blocking: the GPU's evo lock (one evo job per GPU) and enough free memory (other users' jobs)."""
    if gpu_free_gb(g) < need:
        return None
    fh = open(os.path.join(LOCKS, f"gpu{g}.lock"), "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close(); return None
    if gpu_free_gb(g) < need:
        fcntl.flock(fh, fcntl.LOCK_UN); fh.close(); return None
    return fh


def acquire_gpu(pref, need=NEED_FREE_GB, strict=False, max_wait=MAX_GPU_WAIT_S, reassign_after=REASSIGN_AFTER_S):
    """Returns (gpu, lock_handle, waited_s) or (None, None, waited_s). Waiting behind another evo job on the
    preferred GPU does not count toward max_wait (that is our own queue); waiting on other users' memory does.
    Unless strict (paired measurements must stay on one GPU), any idle pool GPU is taken after reassign_after."""
    t0 = time.time(); budget_t0 = t0
    while True:
        waited = time.time() - t0
        cands = [pref] + ([g for g in GPU_POOL if g != pref] if not strict and waited >= reassign_after else [])
        for g in cands:
            fh = _try_lock(g, need)
            if fh:
                return g, fh, round(waited, 1)
        own_queue = False
        probe = open(os.path.join(LOCKS, f"gpu{pref}.lock"), "w")
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(probe, fcntl.LOCK_UN)
        except BlockingIOError:
            own_queue = True
        probe.close()
        if own_queue:
            budget_t0 = time.time()
        if time.time() - budget_t0 > max_wait:
            return None, None, round(waited, 1)
        time.sleep(30)


def run(cmd, env, log, timeout):
    with open(log, "w") as f:
        p = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode

def tail(path, n=40):
    try: return "".join(open(path, errors="replace").readlines()[-n:])
    except OSError: return ""

def summarize(bench, judge):
    out = {}
    for name, reqs in bench["workloads"].items():
        toks = sum(len(r["output_ids"]) - 1 for r in reqs)
        dec = sum(r["decode_s"] for r in reqs)
        spec = {k: sum(r["spec"].get(k, 0) for r in reqs) for k in ("steps", "drafted", "accepted")}
        batch = bench.get("batch")
        out[name] = {"decode_tok_s": round(toks / dec, 3),
                     "e2e_tok_s": round(sum(len(r["output_ids"]) for r in reqs) / sum(r["total_s"] for r in reqs), 3), "per_request_tok_s": [round((len(r["output_ids"]) - 1) / r["decode_s"], 2) for r in reqs],
                     "mean_ttft_s": None if batch else round(statistics.mean(r["prefill_s"] for r in reqs), 3), "tokens": toks,
                     "mtp_accept_rate": round(spec["accepted"] / spec["drafted"], 3) if spec["drafted"] else None,
                     "tokens_per_step": round((spec["steps"] + spec["accepted"]) / spec["steps"], 3) if spec["steps"] else None}
    if judge is not None:
        out["correctness"] = {k: judge.get(k, 0) for k in ("passed", "tokens", "near_tie_flips", "soft_violations", "violations")}
        out["correctness"]["violation_examples"] = [(w, r["id"], (r["violations"] + r.get("soft_violations", []))[:3])
                                                    for w, rs in judge["workloads"].items()
                                                    for r in rs if r["violations"] or r.get("soft_violations")][:5]
    return out

def run_retry(cmd, env, log, timeout, tries=3):
    """Re-run when the process died on a transient host problem (rendezvous port race with other users' jobs)."""
    for i in range(tries):
        rc = run(cmd, env, log, timeout)
        if rc == 0 or "EADDRINUSE" not in tail(log):
            return rc
        time.sleep(5 * (i + 1))
    return rc


def evaluate(engine_dir, gpu, mode="quick", ref=None, label="eval", judge=True, timeout=5400, strict_gpu=False, scenario=None):
    eid = f"{time.strftime('%m%d-%H%M%S')}-{label}-{uuid.uuid4().hex[:4]}"
    slot = os.path.join(SLOTS_DIR, eid); snap = os.path.join(slot, "engine")
    shutil.copytree(engine_dir, snap, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    scenario = scenario or current_scenario()
    workloads, limit, max_new = SCENARIOS[scenario]["modes"][mode]
    sc = SCENARIOS[scenario]
    model = model_path(scenario)                                # a scenario may name another target model ...
    need = sc.get("need_gb", NEED_FREE_GB)                      # ... and how much free GPU memory it needs
    res = {"eval_id": eid, "mode": mode, "scenario": scenario, "gpu": gpu, "engine_dir": engine_dir, "status": "ok"}
    bench_args = ["--max-new", str(max_new), "--limit", str(limit), "--nonce", uuid.uuid4().hex[:8]]
    for n, p in workloads: bench_args += ["--workload", f"{n}={p}"]
    if sc["batch"]: bench_args.append("--batch")
    lock = None
    try:
        got, lock, waited = acquire_gpu(gpu, need=need, strict=strict_gpu)
        res["queued_s"] = waited
        if got is None:
            res.update(status="gpu_busy", log_tail=f"No GPU had {need} GB free (preferred GPU {gpu}: {gpu_free_gb(gpu):.1f} GB; "
                       f"other users' jobs) within {MAX_GPU_WAIT_S // 60} min. This did not test your code.")
            return finish(res, slot)
        if got != gpu:
            res.update(gpu=got, reassigned_from=gpu)
        jgpu = got
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(got), PYTHONPATH=FRAMEWORK, TOKENIZERS_PARALLELISM="false")
        t0 = time.time()
        cmd = [VENV_PY, "-m", "evo.bench_runner", "--engine", snap, "--model", model, "--out", os.path.join(slot, "bench.json")] + bench_args
        rc = run_retry(cmd, env, os.path.join(slot, "bench.log"), timeout)
        wall = time.time() - t0
        if rc != 0 or not os.path.exists(os.path.join(slot, "bench.json")):
            res.update(status="crash", log_tail=tail(os.path.join(slot, "bench.log"))); return finish(res, slot)
        bench = json.load(open(os.path.join(slot, "bench.json")))
        timed = sum(r["total_s"] + r.get("prefill_runs_s", 2 * r["prefill_s"]) for w in bench["workloads"].values() for r in w)
        if timed > wall:   # timings must fit inside the externally measured wall clock
            res.update(status="invalid_timing"); return finish(res, slot)
        jr = None
        if judge:
            if "judge_need_gb" in sc:   # the reference needs more memory than the bench (edge): any pool GPU with room
                fcntl.flock(lock, fcntl.LOCK_UN); lock.close(); lock = None
                jgpu, lock, _ = acquire_gpu(gpu, need=sc["judge_need_gb"], reassign_after=0)
                if jgpu is None:
                    res.update(status="judge_error", log_tail=f"no GPU with {sc['judge_need_gb']} GB free for the judge"); return finish(res, slot)
                res["judge_gpu"] = jgpu
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(jgpu), PYTHONPATH=FRAMEWORK, TOKENIZERS_PARALLELISM="false")
            gate_args = [x for k, v in sc.get("gate", {}).items() for x in (f"--{k.replace('_', '-')}", str(v))]
            ref_model = model_path(scenario, "judge_model")
            jcmd = ([VENV_PY, "-m", "evo.judge_hf", "--model", ref_model] + gate_args if sc.get("judge") == "hf" else
                    [VENV_PY, "-m", "evo.judge", "--judge-engine", JUDGE_ENGINE, "--model", ref_model] + gate_args)
            rc = run_retry(jcmd + ["--bench", os.path.join(slot, "bench.json"), "--out", os.path.join(slot, "judge.json")],
                           env, os.path.join(slot, "judge.log"), timeout)
            if rc != 0:
                res.update(status="judge_error", log_tail=tail(os.path.join(slot, "judge.log"))); return finish(res, slot)
            jr = json.load(open(os.path.join(slot, "judge.json")))
        res["metrics"] = summarize(bench, jr)
        res["engine_config"] = bench.get("engine_config"); res["load_s"] = round(bench.get("load_s", 0), 1)
        res["bench_wall_s"] = round(wall, 1)
        if ref:
            r = json.load(open(ref)) if isinstance(ref, str) else ref
            res["vs_ref"] = {w: round(res["metrics"][w]["decode_tok_s"] / r["metrics"][w]["decode_tok_s"] - 1, 4)
                             for w in res["metrics"] if w in r.get("metrics", {}) and "decode_tok_s" in res["metrics"][w]}
            res["vs_ref_e2e"] = {w: round(res["metrics"][w]["e2e_tok_s"] / r["metrics"][w]["e2e_tok_s"] - 1, 4)
                                 for w in res["metrics"] if "e2e_tok_s" in r.get("metrics", {}).get(w, {}) and "e2e_tok_s" in res["metrics"][w]}
        return finish(res, slot)
    except subprocess.TimeoutExpired:
        res.update(status="timeout"); return finish(res, slot)
    finally:
        if lock:
            fcntl.flock(lock, fcntl.LOCK_UN); lock.close()

def finish(res, slot):
    os.makedirs(os.path.join(RECORDS, "evals"), exist_ok=True)
    json.dump(res, open(os.path.join(slot, "result.json"), "w"), indent=1)
    json.dump(res, open(os.path.join(RECORDS, "evals", res["eval_id"] + ".json"), "w"), indent=1)
    shutil.rmtree(os.path.join(slot, "engine"), ignore_errors=True)   # keep logs + results, drop the code copy
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("engine_dir"); ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--mode", default="quick", choices=list(MODES)); ap.add_argument("--ref")
    ap.add_argument("--label", default="eval"); ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--strict-gpu", action="store_true", help="never move to another GPU (paired measurements)")
    ap.add_argument("--scenario", choices=list(SCENARIOS), help="default: $EVO_SCENARIO or base")
    a = ap.parse_args()
    print(json.dumps(evaluate(os.path.abspath(a.engine_dir), a.gpu, a.mode, a.ref, a.label, not a.no_judge,
                              strict_gpu=a.strict_gpu, scenario=a.scenario), indent=1))
