"""Build the base prompt sets (data/workloads/dev.jsonl, data/holdout/holdout.jsonl ship prebuilt; this script
regenerates them). dev = visible to agents (quick/full eval); holdout = acceptance only.
Inputs under <home>/data: humaneval/*/*.parquet (openai_humaneval) and gsm8k_test.jsonl (GSM8K test split).

  PYTHONPATH=framework python -m evo.build_workloads"""
import glob, json, os, random
import pandas as pd
from transformers import AutoTokenizer
from evo.paths import MODEL, WORKLOADS, PRIVATE, ROOT, JUDGE_ENGINE

tok = AutoTokenizer.from_pretrained(MODEL)
he = pd.read_parquet(glob.glob(os.path.join(ROOT, "data/humaneval/*/*.parquet"))[0]).to_dict("records")
gsm = [json.loads(l) for l in open(os.path.join(ROOT, "data/gsm8k_test.jsonl"))]
src = lambda p: open(os.path.join(JUDGE_ENGINE, p)).read()

def code(i): return ("code", f"Complete the following Python function. Return the full implementation.\n```python\n{he[i]['prompt']}```")
def math(i): return ("math", gsm[i]["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}.")
ZH = ["用三句话解释为什么天空是蓝色的，然后给出一个适合小学生的比喻。",
      "请比较 Rust 和 Go 在并发模型上的差异，并分别给出一个简短的代码示例。",
      "帮我写一封邮件，向导师说明下周组会的实验进度：MoE 推理延迟降低了 30%，但推测解码的接受率下降了。"]
AGENT = ["You are a coding agent working in a repository. The test suite fails with:\n"
         "```\nFAILED tests/test_scheduler.py::test_preempt - AssertionError: assert seq.block_table == []\n```\n"
         "The relevant code is below. Find the bug, explain it, and propose a minimal patch as a unified diff.\n```python\n"
         + src("nanovllm/engine/block_manager.py") + "\n```",
         "You are a coding agent. Read this module and write three pytest unit tests that would catch "
         "regressions in it. Explain what each test checks.\n```python\n" + src("nanovllm/engine/scheduler.py") + "\n```"]
def long(path, q): return ("long", f"Here is a source file from an LLM inference engine.\n```python\n{src(path)}\n```\n{q}")

dev = [code(0), code(2), code(11), code(37), math(0), math(5), math(17), ("zh", ZH[0]), ("zh", ZH[1]),
       long("nanovllm/engine/model_runner.py", "Explain how speculative decoding is executed step by step, and point out the biggest performance problems."),
       long("nanovllm/models/qwen3_5_moe.py", "Summarize the architecture this file implements and list three concrete ways to make decoding faster."),
       ("agent", AGENT[0])]
holdout = [code(53), code(91), code(120), math(42), math(77), ("zh", ZH[2]), ("agent", AGENT[1]),
           long("nanovllm/layers/attention.py", "Describe the KV-cache layout and how a paged attention call is made.")]

def render(items):
    out = []
    for i, (kind, text) in enumerate(items):
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
        ids = list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)
        out.append({"id": f"{kind}-{i}", "kind": kind, "text": text, "prompt_len": len(ids)})
    return out

for name, items, d in [("dev", dev, WORKLOADS), ("holdout", holdout, PRIVATE)]:
    rows = render(items)
    with open(os.path.join(d, f"{name}.jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, [(r["id"], r["prompt_len"]) for r in rows])
