import json
import os

import pytest

from evo import config as C

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_example_config_parses_and_merges(tmp_path):
    if C.tomllib is None:
        pytest.skip("needs Python >= 3.11 or tomli")
    cfg = C.load(os.path.join(REPO, "evoinfer.example.toml"))
    assert cfg["host"]["mode"] == "local" and cfg["gpus"]["need_gb"] == 74
    p = tmp_path / "c.toml"; p.write_text('[gpus]\npool = [2, 3]\n')
    cfg = C.load(str(p))
    assert cfg["gpus"]["pool"] == [2, 3] and cfg["gpus"]["need_gb"] == 74       # unspecified keys keep defaults


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("EVO_GPU_POOL", "4,5")
    monkeypatch.setenv("EVO_MODEL_QWEN3_5", "/m/qwen")
    assert C.gpu_pool() == [4, 5] and C.model("qwen3_5") == "/m/qwen"


def test_scenarios_have_prompt_sets_and_goals():
    import orchestrator as O
    from evo.scenarios import SCENARIOS
    data = {"workloads": os.path.join(REPO, "data", "workloads"), "private": os.path.join(REPO, "data", "holdout")}
    for name, sc in SCENARIOS.items():
        assert tuple(sc["primary"]) == tuple(O.PRIMARY[name]), name      # orchestrator mirrors the scenario's metrics
        assert name in O.SCEN_GOAL, name
        for split, path in sc["modes"]["full"][0]:
            kind = "private" if split == "holdout" else "workloads"
            f = os.path.join(data[kind], os.path.basename(path))
            assert os.path.exists(f), f
            rows = [json.loads(l) for l in open(f)]
            assert rows and all({"id", "kind", "text"} <= set(r) for r in rows), f


def test_shipped_engines_have_the_benchmark_interface():
    for d in ("seed", "evo-015", "lc-003", "edge-003"):
        root = os.path.join(REPO, "engines", d)
        assert os.path.exists(os.path.join(root, "evo_config.json")), d
        init = open(os.path.join(root, "nanovllm", "__init__.py")).read()
        assert "LLM" in init and "SamplingParams" in init, d
