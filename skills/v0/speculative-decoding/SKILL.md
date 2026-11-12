---
name: speculative-decoding
description: MTP speculative decoding in this engine - cost model, draft depth, verification, linear-attention rollback.
---
# MTP speculative decoding

Step = verify (target forward on 1+k tokens) + accept (greedy prefix match) + MTP catch-up over accepted
tokens + (k-1) recursive MTP drafts. Throughput = tokens_per_step / step_time; both appear in `evo eval`.
- Draft depth k (`num_speculative_tokens` in evo_config.json) trades acceptance against draft + verify cost.
  Measure; do not guess. The best k changes as the step gets cheaper.
- Every MTP draft runs a full lm_head (248k x 2048, ~1 GB read). Drafts only need to be good guesses -
  they do not affect correctness - but the TARGET's logits must stay exact.
- Linear-attention (Gated DeltaNet) state must be rolled back to the accepted prefix. The baseline runs
  verification token-by-token and clones every state (`GatedDeltaNet.forward` / `commit`). A kernel that
  processes the k+1 tokens in one launch and keeps per-step states (or recomputes only the accepted prefix)
  is much cheaper.
- Host round trips per step (`.tolist()`, Python accept loop, per-sequence MTP calls) cost far more than the
  math at this size. Moving accept/settle onto the GPU enables capturing the whole step in a CUDA graph.
- Positions: MTP entry p pairs target hidden h_p with token_{p+1}; its KV uses the same block table.
