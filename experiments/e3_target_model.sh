#!/bin/bash
# E3 - generality across target models: a lineage for another model starts from the naive nano-vllm port (written for
# Qwen3.5-35B-A3B). A porting agent first adds the new model (root must pass the judge - here the transformers
# implementation in BF16, with a gate calibrated for that model in framework/evo/scenarios.py), then the lineage evolves
# with the E3 protocol; these models have no draft head, so round 2 is moe+free instead of spec+moe.
# Scenarios: olmoe (OLMoE-1B-7B-0125-Instruct), gemma4 (Gemma 4 26B-A4B-it); set [models] olmoe / gemma4 first.
# Paper: OLMoE 2.24x, Gemma 4 9.12x decode over their ported roots.
#   usage: experiments/e3_target_model.sh LINEAGE SCENARIO GPUS [PORT_MINUTES]     e.g. ... ol olmoe 0,1
set -euo pipefail
cd "$(dirname "$0")/.."
L=${1:?"usage: $0 LINEAGE SCENARIO GPUS [PORT_MINUTES]"}; S=${2:?}; GPUS=${3:?}; PM=${4:-240}
python3 orchestrator.py --lineage "$L" status > /dev/null 2>&1 || python3 orchestrator.py --lineage "$L" init --scenario "$S" \
  --from baseline --gpu "${GPUS%%,*}" --port-minutes "$PM" --port-kind model --max-evals 12 --skills v0
python3 orchestrator.py --lineage "$L" evolve --rounds 3 --gpus "$GPUS" --minutes 75 --max-evals 8 --skills v0 \
  --foci "overhead free|moe free|free free"
python3 orchestrator.py --lineage "$L" status | head -1
