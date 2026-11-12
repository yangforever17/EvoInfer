# Contributing

Issues and pull requests are welcome: new scenarios, target models, seed engines, agent CLIs, and fixes.

- **Tests.** `python3 -m pytest tests` needs no GPU, model or agent CLI. It covers the correctness gate, the verdict
  rule, the board, diff-to-claim mapping, config and scenario integrity, and the agent ↔ GPU-host job queue. Please
  add a test with every change to that logic.
- **The fitness function is the contract.** Changes to `framework/evo/judge*.py`, `bench_runner.py`, `evaluate.py` or
  the verdict rule in `orchestrator.py` change what "better" and "correct" mean for every lineage. Describe the
  calibration or evidence in the pull request, as `judge_gate.py` and `scenarios.py` do for the current gates.
- **Agents must not be able to change their own evaluation.** Anything an agent can write (its worktree) must not be
  read by the evaluator except the engine itself.
- **Style.** Standard library only on the agent side (`orchestrator.py`, `meta.py`, `framework/evo/config.py`). Keep
  comments about *why*.
- **Before opening a pull request**, run `python3 orchestrator.py doctor` and, if you touched evaluation code, one
  `python3 orchestrator.py eval engines/evo-015 --gpu <gpu>`. Its result should still pass the judge.
