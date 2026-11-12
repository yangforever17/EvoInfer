"""Runs INSIDE the candidate's environment (PYTHONPATH = candidate snapshot). Wall-clock timing of
generate() per request; greedy; a per-run nonce is prepended to every prompt so outputs cannot be cached.
Output: JSON with per-request token ids and timings for each workload."""
import argparse, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--workload", action="append", required=True, help="name=path.jsonl")
ap.add_argument("--max-new", type=int, default=256); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--nonce", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--batch", action="store_true", help="submit each workload as one batch and time the whole batch")
a = ap.parse_args()
sys.path.insert(0, a.engine)
import torch
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams

cfg_path = os.path.join(a.engine, "evo_config.json")
cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
tok = AutoTokenizer.from_pretrained(a.model)
t_load = time.perf_counter()
llm = LLM(a.model, **cfg)
t_load = time.perf_counter() - t_load

def ids_for(text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
    return list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)

def gen_many(batch, n):
    torch.cuda.synchronize(); t = time.perf_counter()
    out = llm.generate(batch, SamplingParams(temperature=0, max_tokens=n), use_tqdm=False)
    torch.cuda.synchronize()
    return [o["token_ids"] for o in out], time.perf_counter() - t

def gen(ids, n):
    out, t = gen_many([ids], n)
    return out[0], t

result = {"engine_config": cfg, "load_s": t_load, "max_new": a.max_new, "batch": a.batch, "workloads": {}}
warm = ids_for("Warm-up request " + a.nonce)
gen(warm, 32)
if a.batch:   # also warm the batched path (larger graphs / kernels) outside the timed region
    gen_many([ids_for(f"Warm-up request {a.nonce} {k}") for k in range(8)], 16)
for spec in a.workload:
    name, path = spec.split("=", 1)
    rows = [json.loads(l) for l in open(path)]
    if a.limit: rows = rows[: a.limit]
    reqs = []
    if a.batch:
        ids = [ids_for(f"[request {a.nonce}-{name}-{i}]\n" + r["text"]) for i, r in enumerate(rows)]
        s0 = llm.spec_stats if hasattr(llm, "spec_stats") else {}
        outs, t_batch = gen_many(ids, a.max_new)
        s1 = llm.spec_stats if hasattr(llm, "spec_stats") else {}
        for i, (r, p, out) in enumerate(zip(rows, ids, outs)):
            # the batch wall time is split evenly so per-request sums equal it (throughput = tokens / batch wall)
            reqs.append({"id": r["id"], "kind": r["kind"], "prompt_ids": p, "output_ids": out, "prefill_s": 0.0,
                         "prefill_runs_s": 0.0, "total_s": t_batch / len(rows), "decode_s": t_batch / len(rows),
                         "spec": {k: s1.get(k, 0) - s0.get(k, 0) for k in s1} if i == 0 else {}})
        print(f"{name}: batch of {len(rows)}, {sum(len(o) for o in outs)} tok in {t_batch:.2f}s = "
              f"{sum(len(o) for o in outs) / t_batch:.2f} tok/s", flush=True)
        result["workloads"][name] = reqs
        continue
    for i, r in enumerate(rows):
        ids = ids_for(f"[request {a.nonce}-{name}-{i}]\n" + r["text"])
        s0 = llm.spec_stats if hasattr(llm, "spec_stats") else {}
        _, t_p1 = gen(ids, 1)
        out, t_total = gen(ids, a.max_new)
        s1 = llm.spec_stats if hasattr(llm, "spec_stats") else {}
        _, t_p2 = gen(ids, 1)
        # prefill cost = the warmer of two 1-token runs (bracketing the timed run), so a one-off cold start in the
        # first prefill (kernel autotune, allocator growth) is not subtracted from the decode time
        t_prefill = min(t_p1, t_p2)
        reqs.append({"id": r["id"], "kind": r["kind"], "prompt_ids": ids, "output_ids": out,
                     "prefill_s": t_prefill, "prefill_runs_s": t_p1 + t_p2, "total_s": t_total,
                     "decode_s": max(t_total - t_prefill, 1e-6),
                     "spec": {k: s1.get(k, 0) - s0.get(k, 0) for k in s1}})
        print(f"{name} {r['id']}: {len(out)} tok, {(len(out) - 1) / max(t_total - t_prefill, 1e-6):.2f} tok/s", flush=True)
    result["workloads"][name] = reqs
json.dump(result, open(a.out, "w"))
print("BENCH_DONE", flush=True)
