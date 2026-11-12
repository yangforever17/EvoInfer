#!/bin/bash
# E3 - generality across policy models: one lineage per policy, each restarted from the naive seed under the same
# harness and budget (3 rounds x 2 agents x 75 minutes x <= 8 evaluations, skills v0, foci overhead+free | spec+moe |
# free+free). POLICY is codex, claude, or a [policies.<name>] entry of evoinfer.toml (e.g. Claude Code against a gateway
# serving another model - see evoinfer.example.toml). Paper: Qwen3.8-Max 6.43x, GLM-5.3 4.31x, DeepSeek-V4-Pro 1.52x;
# Codex (GPT-6-Astra), three runs: 6.32x on average (5.24-7.04x).
#   usage: experiments/e3_policy.sh LINEAGE POLICY GPUS          e.g. experiments/e3_policy.sh pq qwen 0,1
set -euo pipefail
cd "$(dirname "$0")/.."
L=${1:?"usage: $0 LINEAGE POLICY GPUS"}; P=${2:?}; GPUS=${3:?}
python3 orchestrator.py --lineage "$L" status > /dev/null 2>&1 || \
  python3 orchestrator.py --lineage "$L" init --scenario base --from baseline --gpu "${GPUS%%,*}"
python3 orchestrator.py --lineage "$L" --policy "$P" evolve --rounds 3 --gpus "$GPUS" --minutes 75 --max-evals 8 --skills v0
python3 orchestrator.py --lineage "$L" status | head -1
