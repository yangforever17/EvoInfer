"""Attribute warm 5090 speculative decode; timing here includes profiler overhead."""
import os,json
import torch
from nanovllm import LLM,SamplingParams

llm=LLM(os.environ['EVO_MODEL'],**json.load(open('evo_config.json')))
prompt='Explain how a compiler translates source code into machine instructions.'
llm.generate([prompt],SamplingParams(temperature=0,max_tokens=32,ignore_eos=True),use_tqdm=False)
llm.add_request(prompt,SamplingParams(temperature=0,max_tokens=256,ignore_eos=True))
llm.step()
torch.cuda.synchronize()
before=llm.spec_stats
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(32):
        if llm.is_finished():break
        llm.step()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=35),flush=True)
print('STATS',{k:v-before[k] for k,v in llm.spec_stats.items()},flush=True)
print('MEMORY GiB',torch.cuda.memory_allocated()/2**30,'peak',torch.cuda.max_memory_allocated()/2**30,flush=True)
