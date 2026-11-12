"""Run with evo run -- python tests/check_moe_prefill.py.

Checks grouped GEMMs and exact BF16 expert-order accumulation independently.
Full-model correctness additionally requires evo eval.
"""
import torch, time
import triton
from nanovllm.layers.moe_prefill import _combine
from nanovllm.layers.moe_prefill import grouped_prefill

torch.manual_seed(42)
torch.set_grad_enabled(False)
e,h,i,k=256,2048,512,8
w1=torch.randn(e,2*i,h,device='cuda',dtype=torch.bfloat16)*0.02
w2=torch.randn(e,h,i,device='cuda',dtype=torch.bfloat16)*0.02

def reference(x,ti,tw):
    flat=ti.flatten(); order=torch.argsort(flat,stable=True)
    counts=torch.bincount(flat,minlength=e).tolist()
    tokens=order//k; rx=x[tokens]; rw=tw.flatten()[order,None]
    out=torch.zeros_like(x); start=0
    for expert,count in enumerate(counts):
        if not count: continue
        end=start+count
        gate,up=torch.nn.functional.linear(rx[start:end],w1[expert]).chunk(2,-1)
        v=torch.nn.functional.linear(torch.nn.functional.silu(gate)*up,w2[expert])
        out.index_add_(0,tokens[start:end],(v*rw[start:end]).to(out.dtype))
        start=end
    return out
for t in [9,33,256,1024,6000]:
    x=torch.randn(t,h,device='cuda',dtype=torch.bfloat16)
    logits=torch.randn(t,e,device='cuda'); tw,ti=logits.softmax(-1).topk(k,-1)
    tw=(tw/tw.sum(-1,keepdim=True)).bfloat16()
    if t == 33:
        # Concentrate all routes in a few experts, leaving empty expert ranges.
        ti = (torch.rand(t, k, device='cuda').argsort(-1) * 31).long()
    order = torch.argsort(ti.flatten(), stable=True)
    inverse = torch.empty_like(order)
    inverse.scatter_(0, order, torch.arange(order.numel(), device='cuda'))
    y = torch.randn(t*k,h,device='cuda',dtype=torch.bfloat16)
    expected = torch.zeros_like(x)
    for expert in range(e):
        rows = torch.where(ti.flatten()[order] == expert)[0]
        routes = order[rows]
        expected.index_add_(0, routes//k, y[rows]*tw.flatten()[routes,None])
    combined = torch.empty_like(x)
    _combine[(t,triton.cdiv(h,256))](y,inverse,ti,tw,combined,t,h,k,256,enable_fp_fusion=False)
    torch.testing.assert_close(combined,expected,rtol=0,atol=0)
    a=reference(x,ti,tw); b=grouped_prefill(x,ti,tw,w1,w2)
    assert (a-b).float().square().mean().sqrt().item() < 1e-4
    assert (a-b).abs().max().item() <= 0.0078125
    print(t,'max', (a-b).abs().max().item(),'rms',(a-b).float().square().mean().sqrt().item(), 'equal',(a==b).float().mean().item(),flush=True)
    for name,fn in [('reference',lambda: reference(x,ti,tw)),('grouped',lambda:grouped_prefill(x,ti,tw,w1,w2))]:
        for _ in range(3): fn()
        torch.cuda.synchronize(); start=time.perf_counter()
        for _ in range(10): fn()
        torch.cuda.synchronize()
        print(t,name,(time.perf_counter()-start)*100, 'ms',flush=True)
