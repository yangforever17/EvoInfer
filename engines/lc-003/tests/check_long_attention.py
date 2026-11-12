"""Paged FA2 split-KV screening; execute through evo run."""
import json, os
import torch
from nanovllm.layers.attention import flash_attn_varlen_func

def direct(q,k,v,cu,lens,tables,splits):
    return torch.ops._vllm_fa2_C.varlen_fwd(q,k,v,None,cu,torch.empty_like(cu),lens,None,tables,None,q.shape[0],33798,0.,q.shape[-1]**-.5,False,True,-1,-1,0.,False,splits,None)[0]

def bench(fn):
    for _ in range(3): fn()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20): out=fn()
    g.replay()
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(30): g.replay()
    b.record(); b.synchronize()
    return a.elapsed_time(b)*1000/600

def main():
    cfg=json.load(open(os.environ['EVO_MODEL']+'/config.json')); cfg=cfg.get('text_config',cfg)
    print({k:cfg[k] for k in ['num_attention_heads','num_key_value_heads','head_dim']},flush=True)
    h,hk,d=cfg['num_attention_heads'],cfg['num_key_value_heads'],cfg['head_dim']
    torch.manual_seed(31)
    k=torch.randn(132,256,hk,d,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    tables=torch.randperm(132,device='cuda',dtype=torch.int32)[None]
    for t in [1,4]:
      q=torch.randn(t,h,d,device='cuda',dtype=torch.bfloat16)
      cu=torch.tensor([0,t],device='cuda',dtype=torch.int32)
      for length in [8192,16384,32768]:
        lens=torch.tensor([length],device='cuda',dtype=torch.int32)
        ref=flash_attn_varlen_func(q,k,v,max_seqlen_q=t,cu_seqlens_q=cu,max_seqlen_k=33798,seqused_k=lens,block_table=tables,causal=True)
        for split in [0]:
          try:
            fn=lambda:direct(q,k,v,cu,lens,tables,split)
            out=fn(); diff=(out.float()-ref.float()).abs()
            print(t,length,split,'us',bench(fn),'max',diff.max().item(),'unequal',(out!=ref).sum().item(),flush=True)
          except Exception as e: print('ERROR',t,length,split,str(e),flush=True)
        singleton_cu=torch.arange(t+1,device='cuda',dtype=torch.int32)
        singleton_lens=lens + torch.arange(1-t,1,device='cuda',dtype=torch.int32)
        singleton_tables=tables.expand(t,-1)
        fn=lambda:flash_attn_varlen_func(q,k,v,max_seqlen_q=1,cu_seqlens_q=singleton_cu,max_seqlen_k=33798,seqused_k=singleton_lens,block_table=singleton_tables,causal=True)
        out=fn(); diff=(out.float()-ref.float()).abs()
        print(t,length,'singleton','us',bench(fn),'max',diff.max().item(),'unequal',(out!=ref).sum().item(),flush=True)
if __name__=='__main__': main()
