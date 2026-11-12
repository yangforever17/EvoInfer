---
name: moe-decode
description: Ideas and pitfalls for MoE layers at batch size 1-8 tokens (decode and speculative verification).
---
# MoE at tiny batch

Qwen3.5-35B-A3B: 256 experts, top-8, expert FFN 2048->512->2048 (gate_up fused [256,1024,2048], down
[256,2048,512]) plus a gated shared expert. At decode each layer touches 8 experts (verification with 4 tokens
touches up to 32 distinct experts - real prompts route far more diversely than synthetic ones).
- The baseline gathers the selected experts' weights (`experts_gate_up[flat]`) and runs bmm: it COPIES
  ~25 MB per layer per token before using it. A kernel that reads the expert weights in place (grouped
  GEMV / fused MoE in Triton) removes that traffic.
- Python loops over experts with `torch.where` / `.tolist()` force host synchronization per layer.
- Router: softmax in fp32 -> top-8 -> renormalize -> cast to bf16. Keep this order or results change.
- Fusions worth testing: gate_up + SiLU*mul; down + weighted sum over k; shared expert alongside routed experts.
- Validate numerics against the original MoE on random inputs (max abs diff) before running `evo eval`.
