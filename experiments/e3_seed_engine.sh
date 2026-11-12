#!/bin/bash
# E3 - generality across seed engines: lineage `ms` starts from mini-sglang (upstream source only). A porting agent
# first implements Qwen3.5-35B-A3B in it behind a thin `nanovllm` adapter (root ms-000 must pass the judge), then the
# lineage evolves with the E3 protocol. Paper: port in 21 minutes, then 65.0 -> 118.3 tok/s decode (1.82x).
#   usage: experiments/e3_seed_engine.sh GPUS [PORT_MINUTES]
set -euo pipefail
cd "$(dirname "$0")/.."
GPUS=${1:?"usage: $0 GPUS [PORT_MINUTES]"}; PM=${2:-240}
NOTE="This lineage evolves a different engine, mini-sglang: the engine code is under python/minisgl, and nanovllm/ is only the thin adapter the benchmark imports (keep engine logic out of it). The skills were written for another engine (nano-vllm): their methods apply, their file paths do not."
if ! python3 orchestrator.py --lineage ms status > /dev/null 2>&1; then
  if [ ! -d runs/mini-sglang ]; then
    git clone -q https://github.com/sgl-project/mini-sglang runs/mini-sglang
    git -C runs/mini-sglang checkout -q 9a91cfa        # the upstream revision the paper started from (MIT)
    rm -rf runs/mini-sglang/.git && echo '{}' > runs/mini-sglang/evo_config.json
  fi
  python3 orchestrator.py import-engine runs/mini-sglang --tag msgl-src -m "mini-sglang 9a91cfa (upstream source)" || true
  python3 orchestrator.py --lineage ms init --scenario base --from msgl-src --gpu "${GPUS%%,*}" --port-minutes "$PM" \
    --port-kind msgl --max-evals 12 --skills v0 --engine-paths "nanovllm python evo_config.json" --engine-note "$NOTE"
fi
python3 orchestrator.py --lineage ms evolve --rounds 3 --gpus "$GPUS" --minutes 75 --max-evals 8 --skills v0
python3 orchestrator.py --lineage ms status | head -1
