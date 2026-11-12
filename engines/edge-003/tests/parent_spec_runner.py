"""Frozen parent 6af9511 step for the paired draft-head benchmark."""
import torch
from nanovllm.layers.state_settle import settle_state
from nanovllm.utils.context import set_context, reset_context


class ModelRunner:
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
