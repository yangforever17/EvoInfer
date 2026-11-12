"""Exercise long whole-prompt prefill, MTP graphs, and sequential state reuse."""
import json
import os
import time

import torch
from nanovllm import LLM, SamplingParams


def main():
    llm = LLM(os.environ['EVO_MODEL'], **json.load(open('evo_config.json')))
    cfg = llm.model_runner.config
    capacity = cfg.num_kvcache_blocks * cfg.kvcache_block_size
    assert capacity >= cfg.max_model_len + 2 * cfg.num_speculative_tokens, capacity
    print('KV capacity', capacity, 'allocated GiB', torch.cuda.memory_allocated() / 2**30, flush=True)
    source = llm.tokenizer.encode('The river flows past the old town. Explain how rivers shape the landscape. ' * 600)
    for length in (4097, 5376):
        prompt = source[:length]
        assert len(prompt) == length
        outputs = []
        for eager in (True, False, False):
            llm.model_runner.enforce_eager = eager
            start = time.perf_counter()
            result = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=256, ignore_eos=True), use_tqdm=False)[0]
            outputs.append(result['token_ids'])
            assert len(outputs[-1]) == 256
            assert llm.is_finished()
            assert not llm.scheduler.block_manager.used_block_ids
            print('length', length, 'eager', eager, 'seconds', time.perf_counter() - start, flush=True)
        assert outputs[0] == outputs[1] == outputs[2], ('eager/graph mismatch', length)
    print('PASS long prefill, 256-token MTP eager/graph/replay, and KV release; peak GiB',
          torch.cuda.max_memory_allocated() / 2**30, flush=True)


if __name__ == '__main__':
    main()
