#!/bin/bash
# E4 - the meta level: the meta-agent rewrites the skills from the main line's records (v0 -> v1), then a paired
# comparison from one parent: frozen v0 (arm skills-v0) against evolved v1 (arm skills-v1), the same focus areas side by
# side, 2 rounds x 2 pairs. Arms have their own boards and history views and are never promoted.
# Paper: same yield (1 accepted version each), v1 used 34% of the agent time and 1 of 20 evaluations.
#   usage: experiments/e4_skills.sh PARENT_TAG GPUS          (4 GPUs, e.g. experiments/e4_skills.sh evo-013 0,1,2,3)
set -euo pipefail
cd "$(dirname "$0")/.."
PARENT=${1:?"usage: $0 PARENT_TAG GPUS"}; IFS=, read -ra G <<< "${2:?}"
[ "${#G[@]}" -ge 4 ] || { echo "needs 4 GPUs"; exit 1; }
[ -d skills/v1 ] || python3 meta.py evolve --base v0 --new v1 --minutes 60
mkdir -p runs
for R in 1 2; do
  for PAIR in "moe spec" "overhead free"; do
    set -- $PAIR; i=0
    for F in "$1" "$2"; do
      for V in v0 v1; do
        python3 orchestrator.py episode --agent codex --gpu "${G[$i]}" --focus "$F" --parent "$PARENT" --skills $V \
          --arm skills-$V --max-evals 10 --minutes 90 > "runs/e4-r$R-$F-$V.log" 2>&1 &
        i=$((i + 1)); sleep 45
      done
    done
    wait
  done
done
echo "done: compare the arms with python3 orchestrator.py status (rows with arm skills-v0 / skills-v1)"
