import pickle
import socket
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.state_settle import settle_state
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.hybrid = config.is_hybrid
        self.num_spec = config.num_speculative_tokens
        self.spec_stats = {"steps": 0, "drafted": 0, "accepted": 0}

        port = _free_port() if self.world_size == 1 else 2333
        dist.init_process_group("nccl", f"tcp://127.0.0.1:{port}", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        if self.hybrid:
            from nanovllm.models.qwen3_5_moe import Qwen3_5MoeForCausalLM
            self.model = Qwen3_5MoeForCausalLM(hf_config, use_mtp=self.num_spec > 0)
            self.model.load_weights(config.model)
        else:
            self.model = Qwen3ForCausalLM(hf_config)
            load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        # Remove the initialization-only DeviceContext. Setting "cpu" keeps a
        # TorchFunctionMode that intercepts every eager prefill tensor operation.
        torch.set_default_device(None)
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
            if self.hybrid:
                del self.step_graphs
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    # ------------------------------------------------------------------ memory
    @torch.inference_mode()
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        if self.hybrid:
            ids = torch.zeros(seq_len, dtype=torch.int64)
            pos = torch.arange(seq_len, dtype=torch.int64)
            cu = torch.tensor([0, seq_len], dtype=torch.int32)
            set_context(True, cu, cu, seq_len, seq_len, query_lens=[seq_len], state_slots=[0])
            hidden = self.model(ids, pos)
            self.model.compute_logits(hidden[-1:])
            if self.model.mtp is not None:
                self.model.compute_logits(self.model.mtp(self.model.embed(ids), hidden, pos)[-1:])
            reset_context()
        else:
            num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
            seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
            for seq in seqs:
                seq.num_scheduled_tokens = seq_len
            self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        if self.hybrid:   # recurrent-state pool for the linear-attention layers, one slot per sequence
            S = config.max_num_seqs
            for m in self.model.linear_attention_modules():
                m.conv_state = torch.zeros(S, m.conv_dim, m.conv_kernel - 1)
                m.ssm_state = torch.zeros(S, m.num_v_heads, m.head_k_dim, m.head_v_dim, dtype=torch.float32)
            attn_modules = self.model.attention_modules()
        else:
            attn_modules = [m for m in self.model.modules() if hasattr(m, "k_cache") and hasattr(m, "v_cache")]
        num_layers = len(attn_modules)
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * num_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0, "not enough GPU memory for the KV cache"
        self.kv_cache = torch.empty(2, num_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        for layer_id, module in enumerate(attn_modules):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]

    # ------------------------------------------------------------------ input preparation
    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def slot(self, seq: Sequence, pos: int) -> int:
        return seq.block_table[pos // self.block_size] * self.block_size + pos % self.block_size

    def set_hybrid_context(self, seqs, starts, lens, is_prefill):
        """Context for feeding lens[i] tokens of seqs[i] at positions starts[i] .. starts[i]+lens[i]-1."""
        cu, slots, ctx_lens = [0], [], []
        for seq, s, n in zip(seqs, starts, lens):
            cu.append(cu[-1] + n)
            slots.extend(self.slot(seq, p) for p in range(s, s + n))
            ctx_lens.append(s + n)
        t = lambda x, d: torch.tensor(x, dtype=d, pin_memory=True).cuda(non_blocking=True)
        set_context(is_prefill, cu_seqlens_q=t(cu, torch.int32), max_seqlen_q=max(lens),
                    max_seqlen_k=max(ctx_lens), slot_mapping=t(slots, torch.int32),
                    context_lens=t(ctx_lens, torch.int32), block_tables=self.prepare_block_tables(seqs),
                    query_lens=list(lens), state_slots=[seq.state_slot for seq in seqs])
        return cu

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    # ------------------------------------------------------------------ execution
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[list[int]]:
        """Returns, per sequence, the list of newly committed tokens."""
        if self.hybrid:
            return self.run_hybrid(seqs, is_prefill)
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return [[t] for t in token_ids] if token_ids is not None else None

    @torch.inference_mode()
    def run_hybrid(self, seqs: list[Sequence], is_prefill: bool) -> list[list[int]]:
        if (not is_prefill and not self.enforce_eager and len(seqs) == 1
                and 0 < self.num_spec <= 7 and len(seqs[0].draft_tokens) == self.num_spec):
            seq = seqs[0]
            assert seq.temperature == 0, "MTP speculative decoding supports greedy only"
            packed = self.run_hybrid_step_graph(
                seq, [seq.last_token] + seq.draft_tokens, len(seq) - 1).tolist()
            n = packed[-1]
            seq.draft_tokens = packed[self.num_spec + 1:-1]
            self.spec_stats["steps"] += 1
            self.spec_stats["drafted"] += self.num_spec
            self.spec_stats["accepted"] += n - 1
            return [packed[:n]]
        if self.num_spec:
            assert all(seq.temperature == 0 for seq in seqs), "MTP speculative decoding supports greedy only"
        else:
            temperatures = self.prepare_sample(seqs)
        if is_prefill:
            starts = [0] * len(seqs)
            lens = [len(seq) for seq in seqs]
            tokens_in = [t for seq in seqs for t in seq.token_ids]
        else:
            starts = [len(seq) - 1 for seq in seqs]
            lens = [1 + len(seq.draft_tokens) for seq in seqs]
            tokens_in = [t for seq in seqs for t in [seq.last_token] + seq.draft_tokens]
        input_ids = torch.tensor(tokens_in, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor([p for s, n in zip(starts, lens) for p in range(s, s + n)],
                                 dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu = self.set_hybrid_context(seqs, starts, lens, is_prefill)
        hidden = self.run_hybrid_model(input_ids, positions)
        reset_context()

        if is_prefill:
            last = torch.tensor([c - 1 for c in cu[1:]], device=hidden.device)
            logits = self.model.compute_logits(hidden[last])
            new_tokens = (logits.argmax(-1) if self.num_spec else self.sampler(logits, temperatures)).tolist()
            out = [[t] for t in new_tokens]
            if self.num_spec:
                for i, seq in enumerate(seqs):
                    h = hidden[cu[i]:cu[i + 1]]
                    self.mtp_propose(seq, h, seq.token_ids[1:] + [new_tokens[i]], 0)
            # Initialize graphs during prefill using valid next-step inputs.
            # The helpers preserve recurrent states; provisional KV writes are
            # overwritten by the real next step before they can be consumed.
            if not self.enforce_eager and len(seqs) == 1 and 0 < self.num_spec <= 7:
                self.run_hybrid_step_graph(
                    seqs[0], [new_tokens[0]] + seqs[0].draft_tokens, len(seqs[0]), warmup_only=True)
            elif (not self.enforce_eager and len(seqs) == 1 and 1 + self.num_spec <= 8
                    and (len(seqs[0]) + self.num_spec) // self.block_size < len(seqs[0].block_table)):
                seq = seqs[0]
                ids = torch.tensor([new_tokens[0]] + seq.draft_tokens, device=hidden.device)
                pos = torch.arange(len(seq), len(seq) + ids.numel(), device=hidden.device)
                self.set_hybrid_context(seqs, [len(seq)], [ids.numel()], False)
                self.run_hybrid_model(ids, pos)
                for module in self.model.linear_attention_modules():
                    module.pending = []
                reset_context()
            return out

        logits = self.model.compute_logits(hidden)
        if not self.num_spec:
            new_tokens = self.sampler(logits, temperatures).tolist()
            self.model.commit_linear_states([1] * len(seqs))
            return [[t] for t in new_tokens]
        preds = logits.argmax(-1).tolist()
        out, accepted = [], []
        for i, seq in enumerate(seqs):
            p = preds[cu[i]:cu[i + 1]]
            drafts = seq.draft_tokens
            a = 0
            while a < len(drafts) and drafts[a] == p[a]:
                a += 1
            out.append(drafts[:a] + [p[a]])
            accepted.append(a + 1)
            self.spec_stats["steps"] += 1
            self.spec_stats["drafted"] += len(drafts)
            self.spec_stats["accepted"] += a
        self.model.commit_linear_states(accepted)
        for i, seq in enumerate(seqs):
            n = accepted[i]
            self.mtp_propose(seq, hidden[cu[i]:cu[i] + n], out[i], starts[i])
        return out

    def run_hybrid_step_graph(self, seq, tokens, start, warmup_only=False):
        """One replay for verify, prefix acceptance, state settlement and drafting.

        MTP catch-up computes the entire verification window, then selects the
        last accepted hidden state. Causality makes the unused suffix harmless;
        recursive drafts overwrite its KV entries before they can be consumed.
        """
        m = len(tokens)
        key = (seq.state_slot, m)
        if key not in self.step_graphs:
            modules = self.model.linear_attention_modules()
            device = self.model.embed_tokens.device
            # One persistent pinned transfer for IDs and positions per replay.
            host_inputs = torch.tensor([tokens, list(range(start, start + m))],
                                       dtype=torch.int64, device="cpu", pin_memory=True)
            inputs = host_inputs.to(device)
            ids, pos = inputs.unbind(0)
            max_len = self.config.max_model_len + 2 * self.num_spec
            tables = torch.zeros((1, (max_len + self.block_size - 1) // self.block_size),
                                 dtype=torch.int32, device=device)
            tables[:, :len(seq.block_table)].copy_(self.prepare_block_tables([seq]))
            cu = torch.tensor([0, m], dtype=torch.int32, device=device)
            cu_one = torch.tensor([0, 1], dtype=torch.int32, device=device)
            slot = seq.state_slot
            # Warmup and capture execute state writes. Restore the prompt state
            # before the first real replay (and after prefill-only initialization).
            saved = [(module.conv_state[slot].clone(), module.ssm_state[slot].clone())
                     for module in modules]

            def context(p, offsets):
                slots = (tables[0][p // self.block_size] * self.block_size + p % self.block_size).int()
                set_context(False, cu_seqlens_q=offsets, max_seqlen_q=p.numel(),
                            max_seqlen_k=max_len, slot_mapping=slots,
                            context_lens=(p[-1:] + 1).int(), block_tables=tables,
                            query_lens=[p.numel()], state_slots=[slot])

            def forward():
                context(pos, cu)
                hidden = self.model(ids, pos)
                logits = self.model.compute_logits(hidden)
                preds = logits.argmax(-1)
                # Reuse the target's likely vocabulary for all recursive drafts.
                # Duplicates are harmless and keep the graph shape static. Only
                # draft proposals are restricted; verification remains exact.
                candidates = logits.topk(min(4096, logits.shape[-1]), dim=-1,
                                         sorted=False).indices.flatten()
                draft_head = self.model.lm_head.index_select(0, candidates)
                n = (preds[:-1] == ids[1:]).long().cumprod(0).sum().reshape(1) + 1
                for module in modules:
                    _, conv, states = module.pending[0]
                    settle_state(states, conv, n, module.ssm_state[slot], module.conv_state[slot])
                    module.pending = []
                h = self.model.mtp(self.model.embed(preds), hidden, pos)
                h = h.index_select(0, n - 1)
                drafts = [candidates[torch.nn.functional.linear(h, draft_head).argmax(-1)]]
                for j in range(1, self.num_spec):
                    p = pos[:1] + n - 1 + j
                    context(p, cu_one)
                    h = self.model.mtp(self.model.embed(drafts[-1]), h, p)
                    drafts.append(candidates[torch.nn.functional.linear(h, draft_head).argmax(-1)])
                reset_context()
                return torch.cat([preds, *drafts, n])

            for _ in range(2):
                forward()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.graph_pool):
                packed = forward()
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            for module, (conv, state) in zip(modules, saved):
                module.conv_state[slot].copy_(conv)
                module.ssm_state[slot].copy_(state)
            self.step_graphs[key] = dict(
                graph=graph, inputs=inputs, host_inputs=host_inputs,
                host_view=host_inputs.numpy(), tables=tables, cu=cu, cu_one=cu_one,
                packed=packed, block_table=tuple(seq.block_table))
        if warmup_only:
            return
        buffers = self.step_graphs[key]
        # run_hybrid reads the output synchronously before the next invocation,
        # so the previous H2D copy has completed before this host buffer is reused.
        buffers["host_view"][0] = tokens
        buffers["host_view"][1] = range(start, start + m)
        buffers["inputs"].copy_(buffers["host_inputs"], non_blocking=True)
        block_table = tuple(seq.block_table)
        if buffers["block_table"] != block_table:
            buffers["tables"][:, :len(block_table)].copy_(self.prepare_block_tables([seq]))
            buffers["block_table"] = block_table
        buffers["graph"].replay()
        return buffers["packed"]

    def run_hybrid_model(self, input_ids, positions):
        """Capture the sync-free small-token forward, retaining rollback buffers.

        A graph is specific to the recurrent-state slot and query length. All
        changing attention metadata lives in persistent device buffers. Graphs
        share a pool because their outputs are consumed before the next replay.
        Larger/batched forwards retain the eager implementation.
        """
        ctx = get_context()
        if (self.enforce_eager or ctx.is_prefill or len(ctx.query_lens) != 1
                or input_ids.numel() > 8):
            return self.model(input_ids, positions)
        key = (ctx.state_slots[0], input_ids.numel())
        modules = self.model.linear_attention_modules()
        if key not in self.graphs:
            ids, pos = input_ids.clone(), positions.clone()
            slots = ctx.slot_mapping.clone()
            lengths = ctx.context_lens.clone()
            cu = ctx.cu_seqlens_q.clone()
            max_len = self.config.max_model_len + 2 * self.num_spec
            tables = torch.zeros((1, (max_len + self.block_size - 1) // self.block_size),
                                 dtype=torch.int32, device=input_ids.device)
            tables[:, :ctx.block_tables.size(1)].copy_(ctx.block_tables)
            set_context(False, cu_seqlens_q=cu,
                        max_seqlen_q=input_ids.numel(), max_seqlen_k=max_len,
                        slot_mapping=slots, context_lens=lengths, block_tables=tables,
                        query_lens=ctx.query_lens, state_slots=ctx.state_slots)
            for _ in range(2):
                for module in modules:
                    module.pending = []
                self.model(ids, pos)
            for module in modules:
                module.pending = []
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.graph_pool):
                hidden = self.model(ids, pos)
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            pending = [module.pending for module in modules]
            self.graphs[key] = (graph, ids, pos, slots, lengths, tables, hidden, pending, cu)
        graph, ids, pos, slots, lengths, tables, hidden, pending, cu = self.graphs[key]
        ids.copy_(input_ids)
        pos.copy_(positions)
        slots.copy_(ctx.slot_mapping)
        lengths.copy_(ctx.context_lens)
        tables[:, :ctx.block_tables.size(1)].copy_(ctx.block_tables)
        graph.replay()
        # Python forward is skipped during replay, so restore the captured
        # intermediate-state references for the ordinary acceptance/commit path.
        for module, states in zip(modules, pending):
            module.pending = states
        return hidden

    def mtp_propose(self, seq: Sequence, hidden: torch.Tensor, next_tokens: list[int], start: int):
        """Run the MTP layer over entries start .. start+m-1 (target hidden h_p paired with token_{p+1}),
        then draft `num_spec` tokens greedily, reusing the single MTP layer recursively."""
        mtp, m = self.model.mtp, len(next_tokens)
        ids = torch.tensor(next_tokens, dtype=torch.int64, device=hidden.device)
        pos = torch.arange(start, start + m, dtype=torch.int64, device=hidden.device)
        self.set_hybrid_context([seq], [start], [m], False)
        h = mtp(self.model.embed(ids), hidden, pos)[-1:]
        reset_context()
        draft = self.model.compute_logits(h).argmax(-1)
        drafts = [draft]
        for j in range(1, self.num_spec):
            p = start + m - 1 + j
            self.set_hybrid_context([seq], [p], [1], False)
            ids = draft
            h = mtp(self.model.embed(ids), h, torch.tensor([p], dtype=torch.int64, device=hidden.device))
            reset_context()
            draft = self.model.compute_logits(h).argmax(-1)
            drafts.append(draft)
        seq.draft_tokens = torch.cat(drafts).tolist()

    @torch.inference_mode()
    def capture_cudagraph(self):
        if self.hybrid:
            # Hybrid graphs are initialized during prefill with valid KV tables.
            self.graphs = {}
            self.step_graphs = {}
            self.graph_pool = None
            return
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
