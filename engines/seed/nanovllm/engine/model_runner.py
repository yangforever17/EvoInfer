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
        torch.set_default_device("cpu")
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
        temperatures = self.prepare_sample(seqs)
        if self.num_spec:
            assert bool((temperatures == 0).all()), "MTP speculative decoding baseline supports greedy only"
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
        hidden = self.model(input_ids, positions)
        reset_context()

        if is_prefill:
            last = torch.tensor([c - 1 for c in cu[1:]], device=hidden.device)
            new_tokens = self.sampler(self.model.compute_logits(hidden[last]), temperatures).tolist()
            out = [[t] for t in new_tokens]
            if self.num_spec:
                for i, seq in enumerate(seqs):
                    h = hidden[cu[i]:cu[i + 1]]
                    self.mtp_propose(seq, h, seq.token_ids[1:] + [new_tokens[i]], 0)
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

    def mtp_propose(self, seq: Sequence, hidden: torch.Tensor, next_tokens: list[int], start: int):
        """Run the MTP layer over entries start .. start+m-1 (target hidden h_p paired with token_{p+1}),
        then draft `num_spec` tokens greedily, reusing the single MTP layer recursively."""
        mtp, m = self.model.mtp, len(next_tokens)
        ids = torch.tensor(next_tokens, dtype=torch.int64, device=hidden.device)
        pos = torch.arange(start, start + m, dtype=torch.int64, device=hidden.device)
        self.set_hybrid_context([seq], [start], [m], False)
        h = mtp(self.model.embed(ids), hidden, pos)[-1:]
        reset_context()
        draft = int(self.model.compute_logits(h).argmax(-1))
        drafts = [draft]
        for j in range(1, self.num_spec):
            p = start + m - 1 + j
            self.set_hybrid_context([seq], [p], [1], False)
            ids = torch.tensor([draft], dtype=torch.int64, device=hidden.device)
            h = mtp(self.model.embed(ids), h, torch.tensor([p], dtype=torch.int64, device=hidden.device))
            reset_context()
            draft = int(self.model.compute_logits(h).argmax(-1))
            drafts.append(draft)
        seq.draft_tokens = drafts

    @torch.inference_mode()
    def capture_cudagraph(self):
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
