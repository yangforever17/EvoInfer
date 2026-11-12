#!/bin/bash
# E2 - scenario-conditioned evolution, long context: a lineage `lc` forked from an evolved engine (default: the paper's
# evo-015, tag paper-evo-015) with the long-context deployment settings, evolved on 8K-31K-token prompts (1024 new
# tokens, metric end-to-end tok/s). The paper ran 2 rounds x 4 agents (lc-003: 1.60x / 1.63x the end-to-end throughput
# of default vLLM; the key change made FlashAttention split the KV dimension during MTP verification, +72% decode).
# Note: the paper's agents used the evolved skills v1, which this repo does not ship; this script uses v0.
#   usage: experiments/e2_long_context.sh GPUS [FROM_TAG]        e.g. experiments/e2_long_context.sh 0,1,2,3
set -euo pipefail
cd "$(dirname "$0")/.."
GPUS=${1:?"usage: $0 GPUS [FROM_TAG]"}; FROM=${2:-paper-evo-015}
python3 orchestrator.py --lineage lc status > /dev/null 2>&1 || python3 orchestrator.py --lineage lc init --scenario lc \
  --from "$FROM" --gpu "${GPUS%%,*}" --config '{"max_model_len": 33792, "gpu_memory_utilization": 0.95}'
python3 orchestrator.py --lineage lc evolve --rounds 2 --gpus "$GPUS" --agent codex --minutes 90 --max-evals 10 \
  --foci "prefill longdecode overhead free"
python3 orchestrator.py --lineage lc status | head -1
