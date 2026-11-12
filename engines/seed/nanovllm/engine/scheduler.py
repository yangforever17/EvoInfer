from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = set(config.eos) if isinstance(config.eos, (list, tuple)) else {config.eos}
        self.block_size = config.kvcache_block_size
        self.hybrid = config.is_hybrid
        # verification feeds 1 + k tokens per step; after accepting up to k of them, MTP drafting
        # writes up to k - 1 further positions, so keep 2k positions of KV slots ahead
        self.lookahead = 2 * config.num_speculative_tokens
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size,
                                          enable_prefix_caching=not self.hybrid)
        self.free_state_slots = deque(range(config.max_num_seqs)) if self.hybrid else None
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1 or (self.hybrid and not self.free_state_slots):
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and (scheduled_seqs or self.hybrid):
                break   # hybrid models prefill whole prompts; others allow chunking for the first seq
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
                if self.hybrid:
                    seq.state_slot = self.free_state_slots.popleft()
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            if self.lookahead:
                for seq in scheduled_seqs:   # MTP drafting after prefill writes ahead of the prompt
                    self.block_manager.may_append(seq, self.lookahead)
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq, self.lookahead):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1 + len(seq.draft_tokens)
                seq.is_prefill = False
                self.block_manager.may_append(seq, self.lookahead)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def _release(self, seq: Sequence):
        self.block_manager.deallocate(seq)
        if self.hybrid and seq.state_slot >= 0:
            self.free_state_slots.append(seq.state_slot)
            seq.state_slot = -1
        seq.draft_tokens = []

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self._release(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[list[int]], is_prefill: bool):
        for seq, new_tokens in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            if is_prefill:
                seq.num_cached_tokens += seq.num_scheduled_tokens
                seq.num_scheduled_tokens = 0
                if seq.num_cached_tokens < seq.num_tokens:
                    continue
            seq.num_scheduled_tokens = 0
            for token_id in new_tokens:
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id in self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    break
            seq.num_cached_tokens = seq.num_tokens - 1   # the last token is fed at the next step
            if seq.is_finished:
                self._release(seq)
                self.running.remove(seq)
