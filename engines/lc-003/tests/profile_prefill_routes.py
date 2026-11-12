"""Warm parent prefill attribution; run only via evo run."""
import json
import os
import time
import cProfile
import pstats
from collections import deque

import torch
from nanovllm import LLM, SamplingParams


def main():
    llm = LLM(os.environ['EVO_MODEL'], **json.load(open('evo_config.json')))
    llm.scheduler.free_state_slots = deque([0])
    sp = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
    prompt = ('Explain how a compiler translates source code into machine instructions. '
              'Discuss parsing, type checking, optimization, and register allocation. ')*12
    print('prompt_tokens', len(llm.tokenizer.encode(prompt)), flush=True)
    for _ in range(2):
        llm.generate([prompt], sp, use_tqdm=False)
    for _ in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        llm.generate([prompt], sp, use_tqdm=False)
        torch.cuda.synchronize()
        print('warm_prefill_s', time.perf_counter()-start, flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                             torch.profiler.ProfilerActivity.CUDA]) as prof:
        llm.generate([prompt], sp, use_tqdm=False)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=35), flush=True)
    print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=35), flush=True)
    prof.export_chrome_trace('prefill_routes_trace.json')
    cpu = cProfile.Profile()
    cpu.enable()
    llm.generate([prompt], sp, use_tqdm=False)
    torch.cuda.synchronize()
    cpu.disable()
    pstats.Stats(cpu).strip_dirs().sort_stats('tottime').print_stats(35)
    pstats.Stats(cpu).strip_dirs().sort_stats('cumtime').print_stats(35)


if __name__ == '__main__':
    main()
