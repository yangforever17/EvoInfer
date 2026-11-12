import argparse

from evo import judge_gate as J


def args(**kw):
    ap = argparse.ArgumentParser(); J.add_gate_args(ap)
    return ap.parse_args([f"--{k.replace('_', '-')}={v}" for k, v in kw.items()])


def test_gate_thresholds():
    assert J.gate(10_000, flips=0, soft=0, hard=0)
    assert not J.gate(10_000, flips=0, soft=0, hard=1)                  # any violation rejects
    assert J.gate(10_000, flips=200, soft=10, hard=0)                   # at the limits: 2% flips, 0.1% soft
    assert not J.gate(10_000, flips=201, soft=0, hard=0)
    assert not J.gate(10_000, flips=0, soft=11, hard=0)


def test_report_classifies_gaps():
    bench = {"workloads": {"dev": [{"id": "a", "prompt_ids": [1, 2], "output_ids": [3, 4, 5, 6]}]}}
    gaps = [0.0, 0.25, 0.75, 1.5]                                        # argmax, near tie, soft, violation
    rep = J.report(bench, lambda p, g: gaps, args())
    row = rep["workloads"]["dev"][0]
    assert [i for i, _ in row["near_tie_flips"]] == [1]
    assert [i for i, _ in row["soft_violations"]] == [2]
    assert [i for i, _ in row["violations"]] == [3]
    assert rep["tokens"] == 4 and not rep["passed"]


def test_report_per_model_gate():
    bench = {"workloads": {"dev": [{"id": "a", "prompt_ids": [1], "output_ids": [2] * 1000}]}}
    gaps = [0.0] * 998 + [1.25, 1.875]                                  # legitimate tail of a model like Gemma 4
    assert not J.report(bench, lambda p, g: gaps, args())["passed"]     # the default gate rejects it
    assert J.report(bench, lambda p, g: gaps, args(hard=3.0, max_soft_frac=0.005))["passed"]
