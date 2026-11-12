# Prompt sets

`workloads/*dev.jsonl` are visible to agents (`evo eval`); `holdout/*holdout.jsonl` are used only by the final
evaluation and never shown to agents. `orchestrator.py setup` copies them to `<home>/workloads` and `<home>/private` on
the GPU host. One JSON object per line: `id`, `kind`, `text`, `prompt_len` (Qwen3.5 tokens incl. chat template).

| Files | Scenario | Content |
|---|---|---|
| `dev.jsonl` (12), `holdout.jsonl` (8) | `base`, `edge`, `gemma4` | HumanEval and GSM8K problems, Chinese requests, an agent-style bug fix, and nano-vllm source files with questions; 27-5K tokens |
| `lc_dev.jsonl` (6), `lc_holdout.jsonl` (4) | `lc` | source files of transformers / PyTorch with review questions, and many-shot GSM8K; 8K-31K tokens |
| `cc_dev.jsonl` (32), `cc_holdout.jsonl` (32) | `cc` | code, math, Chinese and source-reading prompts (26-1.3K tokens), submitted as one batch |
| `olmoe_dev.jsonl`, `olmoe_holdout.jsonl` | `olmoe` | the base sets with long code blocks cut to fit a 4K context |

Builders: `framework/evo/build_workloads.py`, `build_scenarios.py`, `build_olmoe_workloads.py`. They need HumanEval
(`openai_humaneval`, MIT) and the GSM8K test split (MIT) under `<home>/data`. The source code quoted in the prompts
keeps the licenses of nano-vllm (MIT), transformers (Apache-2.0) and PyTorch (BSD-3).
