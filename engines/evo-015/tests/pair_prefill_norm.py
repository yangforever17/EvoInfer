"""Measure marginal prefill norm fusion; short/captured paths stay unchanged."""
import json
import os
from collections import deque
import torch
from nanovllm import LLM
from nanovllm.models.qwen3_5_moe import ZeroCenteredRMSNorm, GatedRMSNorm
from nanovllm.layers.norm_exact import norm_exact
from nanovllm.layers.norm_apply import norm_apply, squared_float
from pair_default_device import PROMPTS, request

VALIDATE = False
CHECKED = 0


def parent_zero(self, x):
    """Parent e9f328408d expression, retained for reproducibility."""
    if x.shape[0] <= 8:
        if x.shape[-1] in (128, 256, 2048):
            return norm_exact(x, self.weight, self.eps)
        return norm_apply(x, self.weight, squared_float(x).mean(-1, keepdim=True), self.eps)
    xf = x.float()
    scale = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
    return ((xf * scale) * (1.0 + self.weight.float())).type_as(x)


def parent_gated(self, x, gate):
    """Parent e9f328408d BF16 boundaries and raw FP32 SiLU."""
    if x.numel() <= 32768:
        if x.shape[-1] in (128, 256, 2048):
            return norm_exact(x, self.weight, self.eps, gate)
        return norm_apply(x, self.weight, squared_float(x).mean(-1, keepdim=True),
                          self.eps, gate, gate_is_raw=True)
    xf = x.float()
    scale = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
    out = self.weight * (xf * scale).to(x.dtype)
    return (out * torch.nn.functional.silu(gate.float())).to(x.dtype)


PARENT_ZERO = parent_zero
PARENT_GATED = parent_gated


def zero(self, x):
    global CHECKED
    if x.shape[0] <= 8:
        return PARENT_ZERO(self, x)
    result = norm_exact(x, self.weight, self.eps)
    if VALIDATE:
        assert torch.equal(result, PARENT_ZERO(self, x)), ('real_zero', x.shape)
        CHECKED += x.numel()
    return result


def gated(self, x, gate):
    global CHECKED
    if x.numel() <= 32768:
        return PARENT_GATED(self, x, gate)
    result = norm_exact(x, self.weight, self.eps, gate)
    if VALIDATE:
        assert torch.equal(result, PARENT_GATED(self, x, gate)), ('real_gated', x.shape)
        CHECKED += x.numel()
    return result


def main():
    global VALIDATE
    llm = LLM(os.environ['EVO_MODEL'], **json.load(open('evo_config.json')))
    llm.scheduler.free_state_slots = deque([0])
    for prompt in PROMPTS:
        ZeroCenteredRMSNorm.forward, GatedRMSNorm.forward = zero, gated
        VALIDATE = True
        request(llm, prompt, 'combined', 8)
        VALIDATE = False
        ZeroCenteredRMSNorm.forward, GatedRMSNorm.forward = PARENT_ZERO, PARENT_GATED
        request(llm, prompt, 'parent', 8)
    print('REAL_PREFILL_NORM_ELEMENTS_EXACT', CHECKED, flush=True)
    rows = []
    for repeat in range(2):
        for i, prompt in enumerate(PROMPTS):
            pair = []
            variants = ['parent', 'device_only', 'combined']
            if repeat:
                variants.reverse()
            for variant in variants:
                ZeroCenteredRMSNorm.forward, GatedRMSNorm.forward = (
                    (zero, gated) if variant == 'combined' else (PARENT_ZERO, PARENT_GATED))
                row = request(llm, prompt, variant)
                row.update(repeat=repeat, prompt=i)
                rows.append(row)
                pair.append(row)
                print(json.dumps({k:v for k,v in row.items() if k != 'tokens'}), flush=True)
            assert all(r['tokens'] == pair[0]['tokens'] for r in pair)
            assert all(r['accepted'] == pair[0]['accepted'] for r in pair)
    for variant in ['parent', 'device_only', 'combined']:
        subset = [r for r in rows if r['variant'] == variant]
        print('SUMMARY', variant, json.dumps(dict(
            decode_tps=sum(r['decode_n'] for r in subset)/sum(r['decode_s'] for r in subset),
            e2e_tps=sum(r['n'] for r in subset)/sum(r['total_s'] for r in subset),
            prefill_s=sum(r['prefill_s'] for r in subset)/len(subset))), flush=True)
    print('ALL 12 TRIPLES TOKEN AND ACCEPTANCE IDENTICAL', flush=True)


if __name__ == '__main__':
    main()
