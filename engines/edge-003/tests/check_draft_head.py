import torch
from nanovllm.layers.draft_head import shortlist_logits

torch.manual_seed(42)
w = torch.randn(32768, 2048, device='cuda', dtype=torch.bfloat16)
x = torch.randn(1, 2048, device='cuda', dtype=torch.bfloat16)
ids = torch.randint(32768, (16384,), device='cuda')
def parent():
    head=w.index_select(0,ids)
    return [torch.nn.functional.linear(x,head) for _ in range(3)]
def candidate():
    return [shortlist_logits(x,w,ids) for _ in range(3)]
for f in [parent,candidate]:
    for _ in range(3): f()
graphs=[]
for f in [parent,candidate]:
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): y=f()
    graphs.append((g,y))
for i in range(8):
    x.normal_();ids.random_(32768)
    for g,y in graphs:g.replay()
    a,b=graphs[0][1][0],graphs[1][1][0]
    assert a.argmax().item() == b.argmax().item()
    torch.testing.assert_close(a, b, atol=0.125, rtol=0.01)
    print('CHECK',i,'max_error',(a-b).abs().max().item(),'argmax',a.argmax().item()==b.argmax().item(),flush=True)
for j in [0,1,1,0]:
    g=graphs[j][0]
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(200):g.replay()
    b.record();b.synchronize()
    print('TIME_US',j,a.elapsed_time(b)*1000/200,flush=True)
