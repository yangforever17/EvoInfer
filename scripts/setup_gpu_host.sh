#!/bin/bash
# Create the Python environment EvoInfer evaluates engines with, on the GPU host, at <home>/venv
# (home: [host] home in evoinfer.toml, default ~/.evoinfer). Uses uv if available, else python3 -m venv + pip.
#   bash scripts/setup_gpu_host.sh [HOME_DIR]
# With [host] mode = "ssh", copy this script and requirements-gpu.txt to the GPU host and run it there,
# or point [host] python at an existing environment that has the packages in requirements-gpu.txt.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
H="${1:-${EVO_HOME:-$HOME/.evoinfer}}"
REQ="$HERE/requirements-gpu.txt"; [ -f "$REQ" ] || REQ="$(dirname "$0")/requirements-gpu.txt"
mkdir -p "$H"
if command -v uv > /dev/null; then
  uv venv --python 3.12 "$H/venv"
  VIRTUAL_ENV="$H/venv" uv pip install -r "$REQ"
else
  python3 -m venv "$H/venv"
  "$H/venv/bin/pip" install -U pip
  "$H/venv/bin/pip" install -r "$REQ"
fi
"$H/venv/bin/python" - <<'EOF'
import torch, transformers, fla
from vllm.vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache  # noqa: F401
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(), torch.cuda.device_count(), "GPU(s)",
      "| transformers", transformers.__version__, "| fla", fla.__version__)
EOF
echo "environment ready: $H/venv/bin/python"
