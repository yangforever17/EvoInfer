"""Causal and dynamic graph regression for singleton verification attention."""
import torch
from nanovllm.layers.attention import paged_verify_attention, flash_attn_varlen_func
from nanovllm.utils.context import Context

def main():
    torch.manual_seed(7032)
    k=torch.randn(132,256,2,256,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    checked=0
    for count in [2,3,4,8]:
        q=torch.randn(count,16,256,device='cuda',dtype=torch.bfloat16)
        tables=torch.randperm(132,device='cuda',dtype=torch.int32)[None]
        lengths=torch.tensor([8192],device='cuda',dtype=torch.int32)
        cu=torch.tensor([0,count],device='cuda',dtype=torch.int32)
        ctx=Context(max_seqlen_q=count,max_seqlen_k=33798,context_lens=lengths,block_tables=tables)
        fn=lambda:paged_verify_attention(q,k,v,ctx,256**-.5)
        for _ in range(3):fn()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): out=fn()
        # The captured graph must consume new lengths, tables and activations.
        for length in [255,256,257,8191,8192,8193,16384,32768]:
            lengths.fill_(length)
            tables.copy_(torch.randperm(132,device='cuda',dtype=torch.int32)[None])
            q.normal_(); k.normal_(); v.normal_()
            graph.replay()
            ref=flash_attn_varlen_func(q,k,v,max_seqlen_q=count,cu_seqlens_q=cu,max_seqlen_k=33798,seqused_k=lengths,block_table=tables,causal=True)
            torch.testing.assert_close(out,ref,atol=.002,rtol=.02)
            # Independent FP32 attention oracle includes exactly the causal prefix.
            keys=k[tables[0].long()].reshape(-1,2,256).repeat_interleave(8,dim=1)
            vals=v[tables[0].long()].reshape(-1,2,256).repeat_interleave(8,dim=1)
            for i in range(count):
                n=length-count+i+1
                scores=torch.einsum('hd,nhd->hn',q[i].float(),keys[:n].float())*.0625
                oracle=torch.einsum('hn,nhd->hd',scores.softmax(-1),vals[:n].float())
                torch.testing.assert_close(out[i].float(),oracle,atol=.002,rtol=.02)
            checked+=out.numel()
        print('PASS',count,'dynamic lengths/table/activations, block boundaries',flush=True)
    print('CHECKED',checked,flush=True)
if __name__=='__main__':main()
