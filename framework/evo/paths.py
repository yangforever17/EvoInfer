"""Fixed locations on the GPU host, all under the EvoInfer home (config [host] home). Nothing here is writable by the
candidate engine's code paths: agents only ever see their own worktree."""
import os
import sys

from evo import config

ROOT = config.home()
MODEL = config.model("qwen3_5")                          # the main target model (Qwen3.5-35B-A3B, BF16)
VENV_PY = (os.path.expanduser(config.CFG["host"]["python"]) if config.CFG["host"]["python"]
           else (os.path.join(ROOT, "venv/bin/python") if os.path.exists(os.path.join(ROOT, "venv/bin/python")) else sys.executable))
FRAMEWORK = os.path.join(ROOT, "framework")             # synced from the repo by `orchestrator.py setup`
ENGINE_REPO = os.path.join(ROOT, "engine.git")          # bare repo: the archive of every engine version
JUDGE_ENGINE = os.path.join(ROOT, "judge_engine")       # frozen seed checkout used as the correctness reference
WORKLOADS = os.path.join(ROOT, "workloads")             # dev prompt sets (agents may read them)
PRIVATE = os.path.join(ROOT, "private")                 # holdout prompt sets (acceptance only, never shown to agents)
SLOTS_DIR = os.path.join(ROOT, "slots")                 # per-evaluation snapshots and logs
AGENTS_DIR = os.path.join(ROOT, "agents")               # agent worktrees as evaluated on the GPU host
RECORDS = os.path.join(ROOT, "records")                 # experiments.jsonl, archives, boards, eval results
LOCKS = os.path.join(ROOT, "locks")                     # one evo job per GPU
GPUS = config.gpu_pool()
for d in (SLOTS_DIR, AGENTS_DIR, RECORDS, LOCKS, PRIVATE, WORKLOADS):
    os.makedirs(d, exist_ok=True)
