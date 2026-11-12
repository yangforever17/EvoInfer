# Engines

Each directory is a complete [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) tree (MIT, see its `LICENSE`).
It is loaded through `from nanovllm import LLM, SamplingParams` with the settings in `evo_config.json`.
`orchestrator.py setup` imports them into the engine archive as tags.

| Directory | Tag | What it is | Measured in the paper (decode tok/s on an A100 80 GB unless noted) |
|---|---|---|---|
| `seed/` | `baseline` | the naive port of Qwen3.5-35B-A3B that every lineage starts from: eager PyTorch, per-expert Python loop, per-token DeltaNet recurrence, Python-side MTP acceptance. Its greedy outputs match transformers. | 17-28 (host-bound, so it varies with host load) |
| `evo-015/` | `paper-evo-015` | the end of the main line (15 versions, ~20 agent-hours) | 297.5 / 309.3 |
| `lc-003/` | `paper-lc-003` | long-context lineage from evo-015: split-KV MTP verification at ≥ 8K tokens, long-prefill MoE tiles (run with `max_model_len` 33792) | 279 at 30K-token prompts (evo-015: 138) |
| `edge-003/` | `paper-edge-003` | edge lineage from evo-015: routed experts in 4-bit for a 32 GB GPU (needs the 4-bit checkpoint, see `evo.quantize_int4`) | end-to-end 1.61x / 1.79x default vLLM on the same 4-bit weights (RTX 5090) |

`agent_notes/` holds the Markdown notes the agents wrote while evolving the engine, and `tests/` their regression
checks; both are kept as they were. Benchmark any of them with the full evaluation and judge:

```bash
python3 orchestrator.py eval engines/evo-015 --gpu 0
python3 orchestrator.py eval engines/lc-003 --gpu 0 --scenario lc
```
