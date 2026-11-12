---
name: gpu-profiling
description: Find where decode time goes on the A100 - CPU-bound vs GPU-bound, torch.profiler, CUDA-event microbenchmarks.
---
# Profiling decode

First decide whether a decode step is **CPU-bound** (Python + kernel-launch overhead; GPU idle between
kernels) or **GPU-bound** (kernels themselves slow). Compare wall time per step with summed CUDA kernel time.

```python
# prof.py - run with: evo run -- python prof.py
import os, torch
from torch.profiler import profile, ProfilerActivity
from nanovllm import LLM, SamplingParams
llm = LLM(os.environ["EVO_MODEL"], max_model_len=4096, num_speculative_tokens=3)
ids = list(range(1000, 1400)); sp = SamplingParams(temperature=0, max_tokens=32)
llm.generate([ids], sp, use_tqdm=False)                       # warm-up / compile
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    llm.generate([ids], sp, use_tqdm=False)
print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))
```
- Many `cudaLaunchKernel` / tiny kernels / `aten::nonzero` / `.tolist()` / `.item()` => host-bound:
  remove syncs, batch work, then CUDA graphs.
- For a single op, time it in isolation with `torch.cuda.Event(enable_timing=True)` over 100 iterations.
- Bytes moved per decode token bounds the speed: active weights ~3B params (~6 GB bf16) + lm_head (1 GB)
  at ~1.5-1.8 TB/s => a few ms per target forward is the floor.
- Prefill and decode differ; the metric is decode tokens/s (TTFT is reported but not optimized).
