"""Run with evo run -- python tests/check_delta_qkv.py."""
import torch
from nanovllm.layers.delta_qkv import delta_qkv
from nanovllm.layers.delta_verify import delta_verify

def reference(x,hk,hv,kd,vd):
    t=x.shape[0]
    q,k,v=x.split([hk*kd,hk*kd,hv*vd],-1)
    q,k=[a.reshape(t,hk,kd).repeat_interleave(hv//hk,1) for a in (q,k)]
    return q.contiguous(),k.contiguous(),v.reshape(t,hv,vd).contiguous()
for t in (1,2,4,8):
    for hk,hv,kd,vd in ((16,32,128,128),(32,32,128,128),(8,32,64,128)):
        torch.manual_seed(t)
        x=torch.randn(2*hk*kd+hv*vd,t,device='cuda',dtype=torch.bfloat16).t()
        actual=delta_qkv(x,hk,hv,kd,vd)
        expected=reference(x,hk,hv,kd,vd)
        for a,b in zip(actual,expected):
            assert a.is_contiguous()
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): actual=delta_qkv(x,hk,hv,kd,vd)
        x.normal_(); graph.replay()
        expected=reference(x,hk,hv,kd,vd)
        for a,b in zip(actual,expected): torch.testing.assert_close(a,b,rtol=0,atol=0)
        g=-torch.rand(t,hv,device='cuda')
        beta=torch.rand(t,hv,device='cuda',dtype=x.dtype)
        state=torch.randn(hv,kd,vd,device='cuda')
        for a,b in zip(delta_verify(*actual,g,beta,state),delta_verify(*expected,g,beta,state)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    print('length',t,'pack and recurrent states bitwise equal',flush=True)
