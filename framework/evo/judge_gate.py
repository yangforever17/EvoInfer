"""Correctness gate shared by both judges and the orchestrator (no torch import, so it is unit-testable anywhere).

For every generated token the judge computes the logit gap g = max_v z(v) - z(emitted) under the frozen reference, given
the candidate's own prefix. Gaps are classified as
  0 < g <= TAU        near-tie flip   (legitimate floating-point reordering; allowed up to MAX_FLIP_FRAC of tokens)
  TAU < g <= HARD     soft violation  (allowed up to MAX_SOFT_FRAC of tokens)
  g > HARD            violation       (any one rejects)
"""
TAU = 0.5            # bf16 logits are quantized (~0.125 at |logit|~20); ties within a few ulps are legitimate
MAX_FLIP_FRAC = 0.02
# gate v2, calibrated on 220K judged tokens of 71 legitimate evaluations of Qwen3.5-35B-A3B: the numerical tail decays
# ~3-4x per 0.125 and never exceeded 0.75; a known state-corruption bug had 31/768 tokens above 1.0.
HARD = 1.0
MAX_SOFT_FRAC = 0.001  # legitimate rate ~0.004%


def gate(tokens, flips, soft, hard, max_flip_frac=MAX_FLIP_FRAC, max_soft_frac=MAX_SOFT_FRAC):
    return hard == 0 and soft <= max_soft_frac * max(tokens, 1) and flips <= max_flip_frac * max(tokens, 1)


def add_gate_args(ap):
    """The gate is calibrated per target model (scenario "gate" in scenarios.py); defaults are gate v2."""
    ap.add_argument("--tau", type=float, default=TAU); ap.add_argument("--hard", type=float, default=HARD)
    ap.add_argument("--max-flip-frac", type=float, default=MAX_FLIP_FRAC)
    ap.add_argument("--max-soft-frac", type=float, default=MAX_SOFT_FRAC)


def report(bench, score, a, **extra):
    """Score every request of a bench result (score(prompt_ids, output_ids) -> list of gaps) and apply the gate."""
    rep = {"tau": a.tau, "hard": a.hard, "max_flip_frac": a.max_flip_frac, "max_soft_frac": a.max_soft_frac, "gate": "v2",
           "workloads": {}, **extra}
    tot = flips = soft = viol = 0
    for name, reqs in bench["workloads"].items():
        rows = []
        for r in reqs:
            gaps = score(r["prompt_ids"], r["output_ids"])
            f = [(i, round(g, 3)) for i, g in enumerate(gaps) if 0 < g <= a.tau]
            sv = [(i, round(g, 3)) for i, g in enumerate(gaps) if a.tau < g <= a.hard]
            v = [(i, round(g, 3)) for i, g in enumerate(gaps) if g > a.hard]
            rows.append({"id": r["id"], "tokens": len(gaps), "near_tie_flips": f, "soft_violations": sv, "violations": v})
            tot += len(gaps); flips += len(f); soft += len(sv); viol += len(v)
        rep["workloads"][name] = rows
    rep.update(tokens=tot, near_tie_flips=flips, soft_violations=soft, violations=viol,
               passed=gate(tot, flips, soft, viol, a.max_flip_frac, a.max_soft_frac))
    return rep
