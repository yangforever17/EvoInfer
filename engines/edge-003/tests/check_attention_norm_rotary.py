"""Bitwise short attention fusion and mutable CUDA graph regression."""
import torch
from nanovllm.layers.norm_exact import norm_exact
from nanovllm.layers.rotary_short import rotary_short
from nanovllm.layers.attention_norm_rotary import attention_norm_rotary, attention_gate


def main():
    count = 0
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for t in (1, 2, 3, 4, 8):
            for d in (128, 256):
                for qh, kh in ((16, 2), (16, 4), (32, 4)):
                    q = torch.randn(t, qh, 2*d, device='cuda', dtype=dtype)[..., :d]
                    k = torch.randn(t, kh, d, device='cuda', dtype=dtype)
                    qw = torch.randn(d, device='cuda', dtype=dtype)
                    kw = torch.randn_like(qw)
                    pos = torch.randint(0, 1024, (t,), device='cuda')
                    angles = torch.randn(1024, d//4, device='cuda')
                    cos, sin = angles.cos(), angles.sin()
                    def ref():
                        return rotary_short(pos, norm_exact(q,qw,1e-6),norm_exact(k,kw,1e-6),cos,sin,d//4)
                    def fused():
                        return attention_norm_rotary(pos,q,k,qw,kw,cos,sin,d//4,1e-6)
                    for _ in range(2): fused()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph): got = fused()
                    for _ in range(4):
                        q.normal_(); k.normal_(); qw.normal_(); kw.normal_(); pos.random_(0,1024)
                        graph.replay()
                        for a,b in zip(got,ref()):
                            torch.testing.assert_close(a,b,atol=0,rtol=0)
                            count += a.numel()
                    o = torch.randn(t,qh*d,device='cuda',dtype=dtype)
                    gate = torch.randn(t,qh,2*d,device='cuda',dtype=dtype)[..., d:]
                    torch.testing.assert_close(attention_gate(o,gate),o*gate.reshape(t,-1).sigmoid(),atol=0,rtol=0)
        print(dtype,'PASS',flush=True)
    print('PASS exact elements',count,flush=True)


if __name__ == '__main__': main()
