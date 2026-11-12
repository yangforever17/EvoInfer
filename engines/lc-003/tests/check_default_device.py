"""Check allocation defaults and independent cold graph/request lifecycles."""
import json
import os
from collections import deque
import torch
from nanovllm import LLM, SamplingParams
from pair_default_device import PROMPTS, request


def main():
    for device in ['cpu', None]:
        torch.set_default_device(device)
        assert torch.empty(1).device.type == 'cpu'
        assert torch.arange(3).device.type == 'cpu'
        assert torch.tensor([1, 2]).device.type == 'cpu'
        assert torch.empty(1, device='cuda').device.type == 'cuda'
    llm = LLM(os.environ['EVO_MODEL'], **json.load(open('evo_config.json')))
    runner = llm.model_runner
    states = {v: dict(graphs={}, step_graphs={}, graph_pool=None,
                      slots=deque(range(runner.config.max_num_seqs)))
              for v in ['parent', 'candidate']}
    for i in range(10):
        pair = []
        for variant in (['parent','candidate'] if i % 2 == 0 else ['candidate','parent']):
            state = states[variant]
            for key in ['graphs', 'step_graphs', 'graph_pool']:
                setattr(runner, key, state[key])
            llm.scheduler.free_state_slots = state['slots']
            row = request(llm, PROMPTS[i % len(PROMPTS)], variant, 32)
            for key in ['graphs', 'step_graphs', 'graph_pool']:
                state[key] = getattr(runner, key)
            pair.append(row)
            print('LIFECYCLE', i, json.dumps({k:v for k,v in row.items() if k != 'tokens'}), flush=True)
        assert pair[0]['tokens'] == pair[1]['tokens'], (i, 'tokens')
        assert pair[0]['accepted'] == pair[1]['accepted'], (i, 'accepted')
    outputs = []
    for variant in ['parent', 'candidate']:
        torch.set_default_device('cpu' if variant == 'parent' else None)
        llm.scheduler.free_state_slots = states[variant]['slots']
        outputs.append(llm.generate(PROMPTS[:2], SamplingParams(temperature=0, max_tokens=32,
                                                             ignore_eos=True), use_tqdm=False))
    assert [r['token_ids'] for r in outputs[0]] == [r['token_ids'] for r in outputs[1]]
    print('PASS: CPU defaults, 10 lifecycle pairs, and mixed batch exact', flush=True)


if __name__ == '__main__':
    main()
