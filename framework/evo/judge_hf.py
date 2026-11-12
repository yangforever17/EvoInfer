"""Teacher-forced correctness judge whose FROZEN reference is outside any engine: the Hugging Face transformers
implementation of the target model, BF16 weights. Used by scenarios with "judge": "hf" (a target model the frozen seed
engine does not implement). Scoring and gate are the same as evo.judge (evo.judge_gate)."""
import argparse, json
from evo.judge_gate import add_gate_args, report
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True)
ap.add_argument("--bench", required=True); ap.add_argument("--out", required=True)
add_gate_args(ap)
a = ap.parse_args()
import torch, transformers
from transformers import AutoModelForCausalLM

bench = json.load(open(a.bench))
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()


@torch.inference_mode()
def score(prompt, gen):
    seq = prompt + gen
    lg = model(input_ids=torch.tensor([seq], device="cuda"), use_cache=False).logits[0, len(prompt) - 1:len(seq) - 1].float()
    g = torch.tensor(seq[len(prompt):], device="cuda")
    return (lg.max(-1).values - lg.gather(1, g[:, None])[:, 0]).tolist()


rep = report(bench, score, a, reference=f"transformers {transformers.__version__}, bf16")
json.dump(rep, open(a.out, "w"))
print("JUDGE_DONE", json.dumps({k: rep[k] for k in ("tokens", "near_tie_flips", "soft_violations", "violations", "passed")}), flush=True)
