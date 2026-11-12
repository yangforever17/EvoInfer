"""Prompt sets for the scenario lineages (dev = visible to agents, holdout = acceptance only).

lc (long context): 8K-32K-token prompts, long answers (1024 new tokens), one request at a time.
  - repo:     several real Python source files from the venv (transformers / torch), then a detailed question;
  - manyshot: a long run of solved GSM8K problems, then five new problems to solve step by step.
cc (concurrency): 32 short-to-medium prompts submitted at once (256 new tokens each), batch throughput.
None of the texts overlaps the base dev/holdout sets."""
import glob, json, os, random
from transformers import AutoTokenizer
from evo.paths import MODEL, WORKLOADS, PRIVATE, ROOT, JUDGE_ENGINE
import pandas as pd

tok = AutoTokenizer.from_pretrained(MODEL)
import sysconfig; SITE = sysconfig.get_paths()["purelib"]   # source files of the installed packages (transformers, torch)
n_tok = lambda s: len(tok(s)["input_ids"])
gsm = [json.loads(l) for l in open(os.path.join(ROOT, "data/gsm8k_test.jsonl"))]
he = pd.read_parquet(glob.glob(os.path.join(ROOT, "data/humaneval/*/*.parquet"))[0]).to_dict("records")
BASE_HE, BASE_GSM = {0, 2, 11, 37, 53, 91, 120}, {0, 5, 17, 42, 77}


def files(pattern):
    return sorted(p for p in glob.glob(os.path.join(SITE, pattern), recursive=True) if os.path.getsize(p) > 4000)


def repo(paths, target, question):
    parts, used = [], 0
    for p in paths:
        body = open(p, errors="replace").read()
        t = n_tok(body)
        if used >= 0.92 * target: break
        if used + t > target:
            if used: continue                      # too big for the remaining space: try the next file
            body = tok.decode(tok(body)["input_ids"][:target]); t = target
        parts.append(f"### File: {os.path.relpath(p, SITE)}\n```python\n{body}\n```"); used += t
    return ("repo", "You are reviewing part of a Python code base.\n\n" + "\n\n".join(parts) + "\n\n" + question)


def manyshot(start, target, new):
    shots, used, i = [], 0, start
    while used < target:
        q = gsm[i]; i += 1
        if i - 1 in BASE_GSM: continue
        s = f"Problem: {q['question']}\nSolution: {q['answer']}\n"
        shots.append(s); used += n_tok(s)
    probs = "\n".join(f"{k + 1}. {gsm[j]['question']}" for k, j in enumerate(new))
    return ("manyshot", "Here are solved math word problems.\n\n" + "\n".join(shots) +
            f"\nNow solve the following {len(new)} new problems. For each one, reason step by step in detail, "
            f"then give the final answer in \\boxed{{}}.\n{probs}")


Q_REPO = ["Write a detailed technical walkthrough of how these files fit together: the main classes, how data flows between "
          "them, and the design decisions behind them. Then list five concrete bugs or performance risks with the exact "
          "function names, and propose a fix for each.",
          "Explain, step by step, what happens at runtime when the main entry point of this code is used, naming every function "
          "that is called along the way. Then suggest how you would refactor the code to make it easier to test, with code sketches."]
random.seed(0)
tf = files("transformers/models/*/modeling_*.py"); th = files("torch/nn/**/*.py"); tg = files("transformers/generation/*.py")
random.shuffle(tf); random.shuffle(th)
lc_dev = [repo(tf[0:40], 8000, Q_REPO[0]), repo(th[0:200], 16000, Q_REPO[1]), repo(tf[40:120], 31000, Q_REPO[0]),
          manyshot(100, 8000, [300, 301, 302, 303, 304]), manyshot(400, 16000, [305, 306, 307, 308, 309]),
          manyshot(700, 31000, [310, 311, 312, 313, 314])]
lc_holdout = [repo(tg + th[200:400], 12000, Q_REPO[1]), repo(tf[120:220], 24000, Q_REPO[1]),
              manyshot(1000, 12000, [320, 321, 322, 323, 324]), manyshot(1100, 28000, [325, 326, 327, 328, 329])]

ZH = ["请用通俗的语言解释什么是混合专家模型（MoE），以及它为什么能在参数量很大的情况下保持推理速度。",
      "写一段 Python 代码，读取一个 CSV 文件并统计每一列的缺失值比例，然后解释代码的每一步。",
      "给一家咖啡店写一份开业宣传文案，要求有标题、三段正文和一句口号。",
      "比较 TCP 和 UDP 的主要区别，并举出各自适合的三个应用场景。",
      "请把下面这句话翻译成英文并解释其中的语法结构：学而不思则罔，思而不学则殆。",
      "设计一个简单的待办事项应用的数据库表结构，并说明每个字段的用途。",
      "解释一下梯度下降算法的原理，并说明学习率过大或过小分别会出现什么问题。",
      "帮我规划一次三天两晚的杭州旅行，包含每天的景点、交通和餐饮建议。"]
src = lambda p: open(os.path.join(JUDGE_ENGINE, p)).read()
MID = [("mid", f"Here is a source file.\n```python\n{src(p)}\n```\n{q}") for p, q in [
    ("nanovllm/engine/scheduler.py", "Explain the scheduling policy and how preemption works."),
    ("nanovllm/engine/block_manager.py", "Explain how blocks are allocated and freed, and how prefix caching works."),
    ("nanovllm/layers/sampler.py", "Explain what this sampler does and suggest two improvements."),
    ("nanovllm/engine/sequence.py", "Describe the Sequence class and its lifecycle."),
    ("nanovllm/layers/linear.py", "Explain how tensor-parallel linear layers are implemented here."),
    ("nanovllm/utils/loader.py", "Explain how the weights are loaded and mapped to modules.")]]
he_ids = [i for i in range(len(he)) if i not in BASE_HE]; gsm_ids = [i for i in range(200, 400) if i not in BASE_GSM]
code = lambda i: ("code", f"Complete the following Python function. Return the full implementation.\n```python\n{he[i]['prompt']}```")
math = lambda i: ("math", gsm[i]["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}.")
cc_dev = [code(i) for i in he_ids[0:12]] + [math(i) for i in gsm_ids[0:12]] + [("zh", z) for z in ZH[0:5]] + MID[0:3]
cc_holdout = [code(i) for i in he_ids[12:24]] + [math(i) for i in gsm_ids[12:24]] + [("zh", z) for z in ZH[5:8]] + \
             [code(i) for i in he_ids[24:26]] + MID[3:6]


def render(items):
    out = []
    for i, (kind, text) in enumerate(items):
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
        ids = list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)
        out.append({"id": f"{kind}-{i}", "kind": kind, "text": text, "prompt_len": len(ids)})
    return out


for name, items, d in [("lc_dev", lc_dev, WORKLOADS), ("lc_holdout", lc_holdout, PRIVATE),
                       ("cc_dev", cc_dev, WORKLOADS), ("cc_holdout", cc_holdout, PRIVATE)]:
    rows = render(items)
    with open(os.path.join(d, f"{name}.jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows), [(r["id"], r["prompt_len"]) for r in rows])
