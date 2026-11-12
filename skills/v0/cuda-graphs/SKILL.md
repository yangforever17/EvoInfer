---
name: cuda-graphs
description: Capturing decode / verification steps in CUDA graphs safely (static buffers, no host syncs, fixed shapes).
---
# CUDA graphs for decode

Eager decode here launches thousands of small kernels per token; a graph replays them with one launch.
- Everything inside the capture must be shape-static and sync-free: no `.item()`, `.tolist()`, `torch.where`
  with data-dependent output size, Python branching on tensor values, or new allocations of varying size.
- Use persistent input/output buffers; copy inputs in before `graph.replay()`, read outputs after.
- Capture per shape you need (e.g. 1 token for plain decode, 1+k tokens for verification, 1 token for MTP).
- Per-sequence Python loops (linear-attention state handling, MoE expert loops) must be vectorized first.
- Warm up (and let Triton autotune) before capture. Recapture if `max_model_len` / block tables change shape.
- Check correctness after capturing: stale buffers silently reuse old data.
