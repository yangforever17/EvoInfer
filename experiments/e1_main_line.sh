#!/bin/bash
# E1 - the main line: evolve the naive seed (engine tag `baseline`) for the base scenario with Codex agents.
# The paper's main line ran 44 episodes (~20 agent-hours, 90-minute episodes, <= 10 evaluations each, 2-4 agents in
# parallel) and reached 17.0 -> 297.5 tok/s decode (17.5x); its result ships as engines/evo-015.
#   usage: experiments/e1_main_line.sh GPUS [ROUNDS]        e.g. experiments/e1_main_line.sh 0,1 6
set -euo pipefail
cd "$(dirname "$0")/.."
GPUS=${1:?"usage: $0 GPUS [ROUNDS]   (comma-separated GPU indices, one agent per GPU per round)"}; ROUNDS=${2:-6}
python3 orchestrator.py status > /dev/null 2>&1 || python3 orchestrator.py init --scenario base --from baseline --gpu "${GPUS%%,*}"
python3 orchestrator.py evolve --rounds "$ROUNDS" --gpus "$GPUS" --agent codex --minutes 90 --max-evals 10 --skills v0 \
  --foci "overhead moe spec free|moe spec overhead free|free free free free"
python3 orchestrator.py status | head -1
