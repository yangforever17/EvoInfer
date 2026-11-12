"""Paired CPU DeviceContext overhead screen, via evo run only."""
import json
import os
import time
from collections import deque
import torch
from nanovllm import LLM, SamplingParams

PROMPTS = [
    'Explain how a compiler translates source code into machine instructions. Discuss parsing, type checking, optimization, and register allocation.',
    'Write a Python function that merges overlapping intervals. Explain its correctness and time complexity with examples.',
    '用中文解释为什么天空是蓝色的，并讨论日落时天空颜色的变化。',
    'Solve the equation x squared minus five x plus six equals zero. Then explain how to solve a general quadratic equation.',
    'Write a short story about a lighthouse keeper who receives a letter from the future.',
    ('A city is considering replacing diesel buses with electric buses. Its goals include reliable service, clean air, and affordable fares. The council must consider charging infrastructure, battery lifetime, and route scheduling.\n'*24)
    + 'Summarize the tradeoffs and propose a phased implementation plan.',
]


def request(llm, prompt, variant, max_tokens=256):
    torch.set_default_device('cpu' if variant == 'parent' else None)
    before = llm.spec_stats
    llm.add_request(prompt, SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True))
    torch.cuda.synchronize()
    start = time.perf_counter()
    output, _ = llm.step()
    torch.cuda.synchronize()
    prefill = time.perf_counter()-start
    prefill_tokens = 1
    decode_start = time.perf_counter()
    while not llm.is_finished():
        finished, _ = llm.step()
        output.extend(finished)
    torch.cuda.synchronize()
    decode = time.perf_counter()-decode_start
    tokens = output[0][1]
    stats = {k: v-before[k] for k, v in llm.spec_stats.items()}
    return dict(variant=variant, prefill_s=prefill, decode_s=decode,
                total_s=prefill+decode, n=len(tokens), decode_n=len(tokens)-prefill_tokens,
                tokens=tokens, **stats)


def main():
    llm = LLM(os.environ['EVO_MODEL'], **json.load(open('evo_config.json')))
    llm.scheduler.free_state_slots = deque([0])
    print('default_device', torch.get_default_device(), flush=True)
    for prompt in PROMPTS:
        for variant in ['parent', 'candidate']:
            request(llm, prompt, variant, 8)
    rows = []
    for repeat in range(2):
        for i, prompt in enumerate(PROMPTS):
            pair = []
            for variant in (['parent', 'candidate'] if repeat == 0 else ['candidate', 'parent']):
                row = request(llm, prompt, variant)
                row.update(repeat=repeat, prompt=i)
                rows.append(row)
                pair.append(row)
                print(json.dumps({k:v for k,v in row.items() if k != 'tokens'}), flush=True)
            assert pair[0]['tokens'] == pair[1]['tokens'], (repeat, i, 'token mismatch')
            assert pair[0]['accepted'] == pair[1]['accepted'], (repeat, i, 'acceptance mismatch')
    for variant in ['parent', 'candidate']:
        subset = [r for r in rows if r['variant'] == variant]
        print('SUMMARY', variant, json.dumps(dict(
            decode_tps=sum(r['decode_n'] for r in subset)/sum(r['decode_s'] for r in subset),
            e2e_tps=sum(r['n'] for r in subset)/sum(r['total_s'] for r in subset),
            prefill_s=sum(r['prefill_s'] for r in subset)/len(subset))), flush=True)
    print('ALL 12 PAIRS TOKEN AND ACCEPTANCE IDENTICAL', flush=True)


if __name__ == '__main__':
    main()
