"""Teacher-forced correctness judge using the FROZEN seed engine (PYTHONPATH = judge_engine) as the reference.
For each generated token: is it the reference's argmax given the candidate's own prefix? If not, the logit gap
(reference top-1 logit minus logit of the emitted token) is classified by evo.judge_gate."""
import argparse, json, sys
from evo.judge_gate import add_gate_args, report
ap = argparse.ArgumentParser(); ap.add_argument("--judge-engine", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--bench", required=True); ap.add_argument("--out", required=True)
add_gate_args(ap)
a = ap.parse_args()
sys.path.insert(0, a.judge_engine)
import torch
from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.context import set_context, reset_context

bench = json.load(open(a.bench))
maxlen = max(len(r["prompt_ids"]) + len(r["output_ids"]) for w in bench["workloads"].values() for r in w) + 16
runner = ModelRunner(Config(a.model, max_model_len=max(maxlen, 4096), gpu_memory_utilization=0.95), 0, [])
model = runner.model


@torch.inference_mode()
def score(prompt, gen):
    seq = prompt + gen
    x = torch.tensor(seq, device="cuda"); pos = torch.arange(len(seq), device="cuda")
    cu = torch.tensor([0, len(seq)], dtype=torch.int32, device="cuda")
    set_context(True, cu, cu, len(seq), len(seq), query_lens=[len(seq)], state_slots=[0])
    h = model(x, pos); reset_context()
    gaps = []
    for s in range(len(prompt) - 1, len(seq) - 1, 512):
        e = min(s + 512, len(seq) - 1)
        lg = model.compute_logits(h[s:e]).float()
        g = torch.tensor(seq[s + 1:e + 1], device="cuda")
        gaps.append(lg.max(-1).values - lg.gather(1, g[:, None])[:, 0])
    return torch.cat(gaps).tolist()


rep = report(bench, score, a)
json.dump(rep, open(a.out, "w"))
print("JUDGE_DONE", json.dumps({k: rep[k] for k in ("tokens", "near_tie_flips", "soft_violations", "violations", "passed")}), flush=True)
