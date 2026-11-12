"""Compare parent/candidate on identical real Q/K/V outside CUDA graphs."""
import json, os
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.layers.attention import Attention
from nanovllm.utils.context import get_context
from long_attention_parent import Attention as Parent

CANDIDATE=Attention.forward
COUNT=0
MAX_ERROR=0.
MAX_NORMALIZED_RMS=0.

def checked(self,q,k,v):
    global COUNT,MAX_ERROR,MAX_NORMALIZED_RMS
    ctx=get_context()
    out=CANDIDATE(self,q,k,v)
    if (not ctx.is_prefill and 1<q.shape[0]<=8 and ctx.block_tables.shape[0]==1
            and ctx.max_seqlen_k>=8192):
        ref=Parent.forward(self,q,k,v)
        diff=(out.float()-ref.float())
        assert torch.isfinite(out).all()
        normalized=(diff.square().mean().sqrt()/ref.float().square().mean().sqrt().clamp_min(1.e-6)).item()
        assert normalized<.01, normalized
        COUNT+=out.numel();MAX_ERROR=max(MAX_ERROR,diff.abs().max().item())
        MAX_NORMALIZED_RMS=max(MAX_NORMALIZED_RMS,normalized)
    return out

def main():
    global COUNT
    cfg=json.load(open('evo_config.json'));cfg['enforce_eager']=True
    llm=LLM(os.environ['EVO_MODEL'],**cfg)
    Attention.forward=checked
    for length,text in [(8192,'Explain why a hash table has average constant time lookup and describe collision resolution. '),(16384,'用中文解释分布式系统的一致性和可用性，并给出具体例子。'),(30000,'Review this Python code: def square(x): return x*x. Explain integer overflow, floating point behavior and test cases. ')]:
        ids=llm.tokenizer.encode(text)*3000
        llm.generate([ids[:length]],SamplingParams(temperature=0,max_tokens=64,ignore_eos=True),use_tqdm=False)
        print('REAL',length,'elements',COUNT,'max_abs',MAX_ERROR,'max_normalized_rms',MAX_NORMALIZED_RMS,flush=True)
if __name__=='__main__':main()
