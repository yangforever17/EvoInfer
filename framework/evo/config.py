"""EvoInfer configuration: one TOML file shared by the orchestrator (agent side) and the framework (GPU side).

Lookup order: $EVO_CONFIG, then `evoinfer.toml` next to the `framework/` directory this module lives in (the repo root
when you run from a checkout, the EvoInfer home on the GPU host after `orchestrator.py setup` has synced it), then the
built-in defaults below. Every key is optional; see evoinfer.example.toml for what each one means.
Standard library only (Python >= 3.11 for tomllib; `pip install tomli` on 3.9/3.10)."""
import copy
import os

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

ROOT_OF_FRAMEWORK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULTS = {
    "host": {
        "mode": "local",          # "local": agents and GPUs on this machine; "ssh": GPUs on another machine
        "ssh_host": "",           # ssh destination (alias from your ssh config, or user@host) when mode = "ssh"
        "ssh_config": "",         # optional ssh config file passed with -F
        "home": "~/.evoinfer",    # state on the GPU host: engine archive, eval slots, records, judge engine, venv
        "python": "",             # Python on the GPU host with torch/vllm/fla installed; default <home>/venv/bin/python
    },
    "gpus": {
        "pool": [0],              # GPU indices EvoInfer may use on the GPU host
        "need_gb": 74,            # free memory an evaluation needs on one GPU (Qwen3.5-35B-A3B BF16: 67 GB + KV cache)
        "max_wait_min": 30,       # give up on a busy GPU after this long (shared machines)
        "reassign_after_min": 5,  # move a non-paired evaluation to any idle pool GPU after this long
    },
    "models": {                   # checkpoint directories on the GPU host
        "qwen3_5": "",            # Qwen3.5-35B-A3B (BF16) - the main target model
        "qwen3_5_int4": "",       # edge scenario: routed experts in 4-bit (evo.quantize_int4)
        "qwen3_5_int4_ref": "",   # edge scenario: BF16 holding exactly the 4-bit expert values (the judge's reference)
        "olmoe": "",              # OLMoE-1B-7B-0125-Instruct
        "gemma4": "",             # Gemma 4 26B-A4B-it
    },
    "agents": {
        "codex_bin": "codex",
        "codex_model": "",        # empty: the Codex CLI's default model
        "claude_bin": "claude",
        "claude_model": "",       # empty: the Claude Code CLI's default model
    },
    # A policy = agent CLI + model. Gateway policies run Claude Code against an Anthropic-compatible endpoint whose URL
    # and key are read from the environment variables named here (never from this file).
    "policies": {},
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def config_path():
    return os.environ.get("EVO_CONFIG") or os.path.join(ROOT_OF_FRAMEWORK, "evoinfer.toml")


def load(path=None):
    path = path or config_path()
    if not os.path.exists(path):
        return copy.deepcopy(DEFAULTS)
    if tomllib is None:
        raise SystemExit(f"reading {path} needs Python >= 3.11 (tomllib) or `pip install tomli`")
    with open(path, "rb") as f:
        return _merge(DEFAULTS, tomllib.load(f))


CFG = load()


def home(expand=True):
    h = os.environ.get("EVO_HOME") or CFG["host"]["home"]
    return os.path.expanduser(h) if expand else h


def model(key):
    """Checkpoint path for a key in [models]; $EVO_MODEL_<KEY> overrides it."""
    return os.environ.get(f"EVO_MODEL_{key.upper()}") or CFG["models"].get(key, "")


def gpu_pool():
    env = os.environ.get("EVO_GPU_POOL")
    return [int(x) for x in env.split(",") if x.strip()] if env else [int(g) for g in CFG["gpus"]["pool"]]
