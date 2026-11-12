"""INT4 checkpoint loading without MTP and batched eager/graph fallback."""
import os,json
import torch
from nanovllm import LLM,SamplingParams

cfg=json.load(open('evo_config.json'))
cfg.update(num_speculative_tokens=0,max_num_seqs=2)
llm=LLM(os.environ['EVO_MODEL'],**cfg)
assert llm.model_runner.model.mtp is None
prompts=['Explain photosynthesis in simple terms.', 'Write a Python palindrome checker.', '为什么一年有四季？']
outputs=[]
for eager in (True,False,False):
    llm.model_runner.enforce_eager=eager
    result=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=32,ignore_eos=True),use_tqdm=False)
    outputs.append([r['token_ids'] for r in result])
    print('GEN no MTP batch2 eager',eager,flush=True)
assert outputs[0]==outputs[1]==outputs[2]
print('PASS no-MTP, batched eager/graph/replay token equality',flush=True)
