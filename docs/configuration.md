# Configuration

All settings live in `evoinfer.toml` at the repo root. Copy it from `evoinfer.example.toml`. `$EVO_CONFIG` points
elsewhere. `orchestrator.py setup` and every GPU command copy it to the GPU host, so both sides read the same file.
Every key has a default; only `[models] qwen3_5` must be set.

## One machine (default)

```toml
[host]
mode = "local"
home = "~/.evoinfer"

[gpus]
pool = [0, 1]

[models]
qwen3_5 = "/models/Qwen3.5-35B-A3B"
```

Run `bash scripts/setup_gpu_host.sh` once to create `~/.evoinfer/venv`, or set `[host] python` to an existing
environment that has the packages in `requirements-gpu.txt`.

## Agents on one machine, GPUs on another

```toml
[host]
mode = "ssh"
ssh_host = "gpu-box"              # `ssh gpu-box true` must work without a password prompt
# ssh_config = "~/.ssh/config"
home = "~/.evoinfer"              # on the GPU host
# python = "~/miniforge3/envs/evo/bin/python"   # on the GPU host; default <home>/venv/bin/python
```

The GPU host needs Python with `requirements-gpu.txt` and git; the agent side needs rsync and ssh. Long evaluations are
started detached on the GPU host and polled, so a dropped SSH connection loses nothing.

## GPUs

| key | default | meaning |
|---|---|---|
| `pool` | `[0]` | GPU indices EvoInfer may use; `$EVO_GPU_POOL=0,1` overrides it |
| `need_gb` | `74` | free memory an evaluation waits for (other users' jobs are respected) |
| `max_wait_min` | `30` | after this, the evaluation reports `gpu_busy`; it does not count against the agent's budget |
| `reassign_after_min` | `5` | an unpaired evaluation moves to any idle pool GPU after this |

Each GPU runs one EvoInfer job at a time (file locks in `<home>/locks`). Paired measurements stay on one GPU.

## Models

Checkpoint directories on the GPU host. `$EVO_MODEL_<KEY>` (e.g. `EVO_MODEL_QWEN3_5`) overrides an entry.

| key | used by |
|---|---|
| `qwen3_5` | scenarios `base`, `lc`, `cc`, and the judge |
| `qwen3_5_int4`, `qwen3_5_int4_ref` | scenario `edge`; create both with `PYTHONPATH=framework python -m evo.quantize_int4 SRC OUT_INT4 OUT_REF` |
| `olmoe`, `gemma4` | scenarios `olmoe`, `gemma4` |

## Agents and policies

```toml
[agents]
codex_bin = "codex"
codex_model = ""          # empty: the CLI's default model
claude_bin = "claude"
claude_model = ""
dsh_bin = "dsh"                  # DeepSeek Harness: npm i -g @deepseek-ai/dsh (Node >= 24.9)
dsh_model = "deepseek-v4-pro"    # model of the built-in dsh policy
```

`--agent codex|claude|dsh` picks the CLI. Codex and Claude Code use their own login; DeepSeek Harness uses DeepSeek's
official API with the key in `$DEEPSEEK_API_KEY`. A *policy* (`--policy NAME`) is an agent CLI plus a model. Policies
`codex`, `claude` and `dsh` are built in. You can add more, e.g. Claude Code against an Anthropic-compatible gateway
that serves another model:

```toml
[policies.qwen]
cli = "claude"
model = "qwen3.8-max"
small_model = "qwen3.8-flash"
base_url_env = "QWEN_ANTHROPIC_BASE"   # names of environment variables, not values
api_key_env = "QWEN_API_KEY"
```

```bash
export QWEN_ANTHROPIC_BASE=https://... QWEN_API_KEY=...
python3 orchestrator.py --lineage pq --policy qwen evolve --gpus 0,1
```

or DeepSeek Harness against any endpoint:

```toml
[policies.dsh-qwen]
cli = "dsh"
api = "openai"                   # OpenAI-compatible chat completions; "deepseek" (default): DeepSeek's Messages API
model = "qwen3.8-max"
base_url_env = "QWEN_OPENAI_BASE"   # e.g. https://host/v1; or base_url = "..." (a URL is not a secret)
api_key_env = "QWEN_API_KEY"
context_window = 262144          # api = "openai" only; defaults 131072 / 32768
max_tokens = 32768
# proxy_env = "MY_HTTPS_PROXY"   # optional: variable holding an HTTPS proxy for the model requests
```

Keys are only read from the environment; they are never written to records or logs. Claude Code gets its gateway key
in its environment. DeepSeek Harness never sees a key: the orchestrator runs a forwarder on `127.0.0.1`
(`framework/evo/llm_forward.py`) that adds the key to each request, and dsh is given only that loopback address and
a placeholder. The forwarder also counts the tokens that `status` and the records report for dsh episodes.

## Environment variables

| variable | effect |
|---|---|
| `EVO_CONFIG` | path of the config file |
| `EVO_HOME` | overrides `[host] home` |
| `EVO_GPU_POOL` | overrides `[gpus] pool` |
| `EVO_MODEL_<KEY>` | overrides a `[models]` entry |
| `EVO_SKILLS` | default skill version (otherwise `skills/CURRENT`) |
| `EVO_NO_SYNC` | skip copying the framework to the GPU host before a command (used internally by `evolve`) |
