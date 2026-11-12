"""The acceptance rule (eq. 4 in the paper) and the sequential re-measurement trigger."""
import orchestrator as O

THR = 0.03


def v(dec, e2e, correct=True):
    return O.verdict_of(correct, dict(zip(("dev", "holdout"), dec)), dict(zip(("dev", "holdout"), e2e)), THR)


def test_base_scenario(monkeypatch):
    monkeypatch.setattr(O, "SCEN", "base")
    assert v((0.10, 0.08), (0.05, 0.04)) == "accepted"
    assert v((0.0, -0.01), (0.25, 0.30)) == "accepted"                  # end-to-end gain alone counts (prefill)
    assert v((0.10, 0.02), (0.0, 0.0)) == "pooled"                      # holdout below the bar: pooled, not lost
    assert v((0.10, 0.0), (0.0, 0.0)) == "no_gain"
    assert v((0.20, -0.04), (0.1, 0.1)) == "rejected_regression"        # any metric regressing beyond noise
    assert v((0.20, 0.20), (0.2, 0.2), correct=False) == "rejected_correctness"


def test_long_context_scores_e2e_only(monkeypatch):
    monkeypatch.setattr(O, "SCEN", "lc")
    assert v((0.20, 0.20), (0.01, 0.01)) == "pooled"
    assert v((0.0, 0.0), (0.05, 0.05)) == "accepted"


def test_accept_uncertain(monkeypatch):
    monkeypatch.setattr(O, "SCEN", "base")
    res = lambda d, h: {"status": "ok", "metrics": {"correctness": {"passed": True}},
                        "vs_ref": {"dev": d, "holdout": h}, "vs_ref_e2e": {"dev": 0.0, "holdout": 0.0}}
    assert O.accept_uncertain(res(0.035, 0.035), 1, THR)               # just above the bar: measure again
    assert not O.accept_uncertain(res(0.20, 0.20), 1, THR)             # clear gain
    assert not O.accept_uncertain(res(0.0, 0.0), 1, THR)               # clearly no gain
