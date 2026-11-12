#!/bin/bash
# E2 - scenario-conditioned evolution, edge device: routed experts in 4-bit so the model fits a 32 GB GPU (the paper used
# an RTX 5090). A porting agent first adapts an evolved engine to the 4-bit checkpoint (root edge-000 must pass the judge),
# then the lineage evolves (paper: 2 rounds x 3 agents; edge-003 at 1.61x / 1.79x default vLLM on the same weights).
#
# Prerequisites (once): quantize, then set [models] qwen3_5_int4 and qwen3_5_int4_ref in evoinfer.toml:
#   PYTHONPATH=framework python -m evo.quantize_int4 /path/Qwen3.5-35B-A3B /path/int4 /path/int4-ref
# The judge's reference (BF16 with the 4-bit values) needs an 80 GB GPU; put both kinds of GPU in [gpus] pool, or run
# the whole lineage on 80 GB GPUs. Note: the paper's agents used skills v1 (not shipped); this script uses v0.
#   usage: experiments/e2_edge.sh GPUS [FROM_TAG]
set -euo pipefail
cd "$(dirname "$0")/.."
GPUS=${1:?"usage: $0 GPUS [FROM_TAG]"}; FROM=${2:-paper-evo-015}
python3 orchestrator.py --lineage edge status > /dev/null 2>&1 || python3 orchestrator.py --lineage edge init --scenario edge \
  --from "$FROM" --gpu "${GPUS%%,*}" --port-minutes 90 --port-kind edge --max-evals 12
python3 orchestrator.py --lineage edge evolve --rounds 2 --gpus "$GPUS" --agent codex --minutes 90 --max-evals 10 \
  --foci "quant spec overhead"
python3 orchestrator.py --lineage edge status | head -1
