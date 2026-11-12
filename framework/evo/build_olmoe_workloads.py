"""Prompt sets for the second target model (scenario "olmoe"): the base dev / holdout prompts, with the long ones cut
so that prompt + 256 new tokens fit OLMoE's 4096-token context. The question after a code block is kept; the code
block itself is shortened. Writes olmoe_dev.jsonl (workloads) and olmoe_holdout.jsonl (private)."""
import json, os
from transformers import AutoTokenizer
from evo.paths import WORKLOADS, PRIVATE
from evo.scenarios import model_path

tok = AutoTokenizer.from_pretrained(model_path("olmoe"))
LIMIT = 3600      # prompt tokens incl. chat template and the per-run request tag; leaves room for 256 new tokens


def n_tok(text):
    ids = tok.apply_chat_template([{"role": "user", "content": "[request 00000000-holdout-00]\n" + text}], add_generation_prompt=True, tokenize=True)
    return len(ids["input_ids"]) if hasattr(ids, "keys") else len(ids)


def fit(text):
    if n_tok(text) <= LIMIT:
        return text
    cut = text.rfind("\n```\n")                       # end of the (last) code block: keep what follows it
    head, tail = (text[:cut], "\n# ... (file truncated)" + text[cut:]) if cut > 0 else (text, "")
    lo, hi = 0, len(head)
    while hi - lo > 64:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if n_tok(head[:mid] + tail) <= LIMIT else (lo, mid)
    return head[:lo] + tail


for src, dst in ((os.path.join(WORKLOADS, "dev.jsonl"), os.path.join(WORKLOADS, "olmoe_dev.jsonl")),
                 (os.path.join(PRIVATE, "holdout.jsonl"), os.path.join(PRIVATE, "olmoe_holdout.jsonl"))):
    rows = [json.loads(l) for l in open(src)]
    with open(dst, "w") as f:
        for r in rows:
            t = fit(r["text"]); r = dict(r, text=t, prompt_len=n_tok(t), cut=t != r["text"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(os.path.basename(dst), r["id"], r["prompt_len"], "cut" if r["cut"] else "")
