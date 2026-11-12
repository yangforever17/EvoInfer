"""Qwen3.5-MoE (text part) for nano-vllm: hybrid Gated DeltaNet / gated full attention, 256-expert MoE
with a gated shared expert, and the native MTP draft layer.

This is the deliberately straightforward *baseline* implementation that the agents evolve:
semantics follow transformers' modeling_qwen3_5_moe.py; no fused kernels beyond flash-attn and fla.
Single GPU (TP=1) only.
"""
import os
from glob import glob

import torch
import torch.nn.functional as F
from torch import nn
from safetensors import safe_open
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

from nanovllm.layers.attention import Attention
from nanovllm.utils.context import get_context


# ----------------------------------------------------------------------------- norms

class ZeroCenteredRMSNorm(nn.Module):
    """Qwen3.5 RMSNorm: weight stored zero-centered, applied as (1 + w), computed in fp32."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * (1.0 + self.weight.float())).type_as(x)


class GatedRMSNorm(nn.Module):
    """RMSNorm(x) * silu(gate) used at the output of Gated DeltaNet."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        out = self.weight * xf.to(dtype)
        out = out * F.silu(gate.float())
        return out.to(dtype)


def linear(x, w):
    return F.linear(x, w)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

    def forward(self, x):
        return F.linear(x, self.weight)


# ----------------------------------------------------------------------------- rotary

class PartialRotary(nn.Module):
    """Text-only M-RoPE: with identical t/h/w positions it reduces to standard RoPE on the first
    `rotary_dim` channels (rotate-half layout). cos/sin are cast to the activation dtype like HF."""

    def __init__(self, head_dim: int, rotary_dim: int, base: float, max_position: int):
        super().__init__()
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, positions, q, k):
        # q: [T, H, D], k: [T, KVH, D]
        cos = self.cos_cache[positions].to(q.dtype).unsqueeze(1)
        sin = self.sin_cache[positions].to(q.dtype).unsqueeze(1)
        r = self.rotary_dim
        q_rot, q_pass = q[..., :r], q[..., r:]
        k_rot, k_pass = k[..., :r], k[..., r:]
        q_rot = q_rot * cos + self._rotate_half(q_rot) * sin
        k_rot = k_rot * cos + self._rotate_half(k_rot) * sin
        return torch.cat([q_rot, q_pass], -1), torch.cat([k_rot, k_pass], -1)


# ----------------------------------------------------------------------------- attention

class GatedAttention(nn.Module):
    def __init__(self, cfg, rotary: PartialRotary):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_proj = Linear(cfg.hidden_size, self.num_heads * self.head_dim * 2)
        self.k_proj = Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim)
        self.o_proj = Linear(self.num_heads * self.head_dim, cfg.hidden_size)
        self.q_norm = ZeroCenteredRMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.rotary = rotary
        self.attn = Attention(self.num_heads, self.head_dim, self.head_dim ** -0.5, self.num_kv_heads)

    def forward(self, positions, x):
        T = x.shape[0]
        q, gate = self.q_proj(x).view(T, self.num_heads, 2 * self.head_dim).chunk(2, dim=-1)
        gate = gate.reshape(T, -1)
        q = self.q_norm(q)
        k = self.k_norm(self.k_proj(x).view(T, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).view(T, self.num_kv_heads, self.head_dim)
        q, k = self.rotary(positions, q, k)
        o = self.attn(q.contiguous(), k.contiguous(), v.contiguous())
        o = o.reshape(T, -1) * torch.sigmoid(gate)
        return self.o_proj(o)


# ----------------------------------------------------------------------------- gated deltanet

class GatedDeltaNet(nn.Module):
    """Linear attention layer. Recurrent state lives in a pool indexed by sequence slot:
    conv_state [S, conv_dim, K-1] (model dtype) and ssm_state [S, HV, K, V] (fp32).

    Modes (from the forward context):
      * prefill: whole prompt per sequence, zero initial state (no prefix cache / chunked prefill).
      * decode/verify: each sequence feeds q_len >= 1 tokens on top of its saved state. Per-step
        states are kept in `self.pending` so the runner can roll back to the accepted prefix.
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_k_heads = cfg.linear_num_key_heads
        self.num_v_heads = cfg.linear_num_value_heads
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel = cfg.linear_conv_kernel_dim
        self.in_proj_qkv = Linear(cfg.hidden_size, self.conv_dim)
        self.in_proj_z = Linear(cfg.hidden_size, self.value_dim)
        self.in_proj_b = Linear(cfg.hidden_size, self.num_v_heads)
        self.in_proj_a = Linear(cfg.hidden_size, self.num_v_heads)
        self.conv1d_weight = nn.Parameter(torch.empty(self.conv_dim, 1, self.conv_kernel))
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, dtype=torch.float32))
        self.norm = GatedRMSNorm(self.head_v_dim, cfg.rms_norm_eps)
        self.out_proj = Linear(self.value_dim, cfg.hidden_size)
        self.conv_state = torch.tensor([])
        self.ssm_state = torch.tensor([])
        self.pending = []  # [(slot, conv_ext [C, K-1+q], states [q, HV, K, V])]

    def _qkv_heads(self, mixed, T):
        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(T, self.num_k_heads, self.head_k_dim)
        k = k.reshape(T, self.num_k_heads, self.head_k_dim)
        v = v.reshape(T, self.num_v_heads, self.head_v_dim)
        rep = self.num_v_heads // self.num_k_heads
        if rep > 1:
            q = q.repeat_interleave(rep, dim=1)
            k = k.repeat_interleave(rep, dim=1)
        return q, k, v

    def forward(self, positions, x):
        ctx = get_context()
        T = x.shape[0]
        mixed = self.in_proj_qkv(x)                       # [T, C]
        z = self.in_proj_z(x).reshape(T, self.num_v_heads, self.head_v_dim)
        beta = self.in_proj_b(x).sigmoid()                 # [T, HV]
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias)
        w = self.conv1d_weight                             # [C, 1, K]
        Km1 = self.conv_kernel - 1
        outs = []
        start = 0
        for i, qlen in enumerate(ctx.query_lens):
            slot = ctx.state_slots[i]
            xs = mixed[start:start + qlen].t()             # [C, q]
            if ctx.is_prefill:
                conv_in = F.pad(xs, (Km1, 0))
            else:
                conv_in = torch.cat([self.conv_state[slot], xs], dim=-1)
            conv_out = F.silu(F.conv1d(conv_in.unsqueeze(0), w, groups=self.conv_dim)).squeeze(0)  # [C, q]
            conv_out = conv_out.t().to(x.dtype)
            q, k, v = self._qkv_heads(conv_out, qlen)
            gi, bi = g[start:start + qlen], beta[start:start + qlen]
            if ctx.is_prefill:
                o, state = chunk_gated_delta_rule(
                    q[None], k[None], v[None], g=gi[None], beta=bi[None],
                    initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
                if self.ssm_state.numel():
                    self.ssm_state[slot].copy_(state[0])
                    self.conv_state[slot].copy_(conv_in[:, -Km1:])
                outs.append(o[0])
            else:
                state = self.ssm_state[slot][None].clone()
                step_states, step_out = [], []
                for t in range(qlen):  # token by token so every intermediate state is available
                    o, state = fused_recurrent_gated_delta_rule(
                        q[None, t:t + 1], k[None, t:t + 1], v[None, t:t + 1],
                        g=gi[None, t:t + 1], beta=bi[None, t:t + 1],
                        initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)
                    step_out.append(o[0])
                    step_states.append(state[0].clone())
                outs.append(torch.cat(step_out, 0))
                self.pending.append((slot, conv_in, torch.stack(step_states)))
            start += qlen
        core = torch.cat(outs, 0)                          # [T, HV, V]
        core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(core.reshape(T, -1))

    def commit(self, accepted: list[int]):
        """accepted[i] = number of fed tokens kept for the i-th pending sequence (>= 1)."""
        Km1 = self.conv_kernel - 1
        for (slot, conv_in, states), n in zip(self.pending, accepted):
            self.ssm_state[slot].copy_(states[n - 1])
            self.conv_state[slot].copy_(conv_in[:, n:n + Km1])
        self.pending = []


# ----------------------------------------------------------------------------- MoE

class SparseMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_tok
        H, I = cfg.hidden_size, cfg.moe_intermediate_size
        self.gate = Linear(H, self.num_experts)
        self.experts_gate_up = nn.Parameter(torch.empty(self.num_experts, 2 * I, H))
        self.experts_down = nn.Parameter(torch.empty(self.num_experts, H, I))
        Is = cfg.shared_expert_intermediate_size
        self.shared_gate_proj = Linear(H, Is)
        self.shared_up_proj = Linear(H, Is)
        self.shared_down_proj = Linear(Is, H)
        self.shared_expert_gate = Linear(H, 1)

    def forward(self, x):
        shared = self.shared_down_proj(F.silu(self.shared_gate_proj(x)) * self.shared_up_proj(x))
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        topw, topi = torch.topk(probs, self.top_k, dim=-1)
        topw = (topw / topw.sum(-1, keepdim=True)).to(logits.dtype)
        shared = torch.sigmoid(self.shared_expert_gate(x)) * shared
        T = x.shape[0]
        if T * self.top_k <= 64:
            # few tokens (decode / verification): gather the selected experts' weights and use bmm.
            # No host synchronization, but every selected expert's weights are copied once.
            flat = topi.flatten()
            xr = x.repeat_interleave(self.top_k, dim=0).unsqueeze(-1)              # [T*k, H, 1]
            gate, up = torch.bmm(self.experts_gate_up[flat], xr).squeeze(-1).chunk(2, dim=-1)
            h = torch.bmm(self.experts_down[flat], (F.silu(gate) * up).unsqueeze(-1)).squeeze(-1)
            h = (h.view(T, self.top_k, -1) * topw.unsqueeze(-1)).sum(1)
            return h + shared
        out = torch.zeros_like(x)
        flat = topi.flatten()
        for e in torch.unique(flat).tolist():
            tok, slot = torch.where(topi == e)
            h = x[tok]
            gate, up = F.linear(h, self.experts_gate_up[e]).chunk(2, dim=-1)
            h = F.linear(F.silu(gate) * up, self.experts_down[e])
            out.index_add_(0, tok, (h * topw[tok, slot, None]).to(out.dtype))
        return out + shared


# ----------------------------------------------------------------------------- layers / model

class DecoderLayer(nn.Module):
    def __init__(self, cfg, layer_type: str, rotary: PartialRotary):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(cfg)
        else:
            self.self_attn = GatedAttention(cfg, rotary)
        self.mlp = SparseMoE(cfg)
        self.input_layernorm = ZeroCenteredRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = ZeroCenteredRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, positions, x):
        h = self.input_layernorm(x)
        h = self.linear_attn(positions, h) if self.layer_type == "linear_attention" else self.self_attn(positions, h)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MTP(nn.Module):
    """Native multi-token-prediction layer. Entry at position p consumes (target hidden h_p, token_{p+1})
    and predicts token_{p+2}; recursive drafting feeds back its own (normed) output hidden."""

    def __init__(self, cfg, rotary):
        super().__init__()
        H = cfg.hidden_size
        self.pre_fc_norm_embedding = ZeroCenteredRMSNorm(H, cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = ZeroCenteredRMSNorm(H, cfg.rms_norm_eps)
        self.fc = Linear(2 * H, H)
        self.layer = DecoderLayer(cfg, "full_attention", rotary)
        self.norm = ZeroCenteredRMSNorm(H, cfg.rms_norm_eps)

    def forward(self, embeds, hidden, positions):
        x = self.fc(torch.cat([self.pre_fc_norm_embedding(embeds), self.pre_fc_norm_hidden(hidden)], -1))
        return self.norm(self.layer(positions, x))


class Qwen3_5MoeForCausalLM(nn.Module):
    def __init__(self, cfg, use_mtp: bool = False):
        super().__init__()
        self.cfg = cfg
        rp = cfg.rope_parameters
        self.rotary = PartialRotary(cfg.head_dim, int(cfg.head_dim * rp.get("partial_rotary_factor", 1.0)),
                                    rp["rope_theta"], min(cfg.max_position_embeddings, 1 << 17))
        self.embed_tokens = nn.Parameter(torch.empty(cfg.vocab_size, cfg.hidden_size))
        self.layers = nn.ModuleList([DecoderLayer(cfg, t, self.rotary) for t in cfg.layer_types])
        self.norm = ZeroCenteredRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Parameter(torch.empty(cfg.vocab_size, cfg.hidden_size))
        self.mtp = MTP(cfg, self.rotary) if use_mtp else None

    # attention layers in KV-cache order: target full-attention layers, then the MTP layer
    def attention_modules(self):
        mods = [l.self_attn.attn for l in self.layers if l.layer_type == "full_attention"]
        if self.mtp is not None:
            mods.append(self.mtp.layer.self_attn.attn)
        return mods

    def linear_attention_modules(self):
        return [l.linear_attn for l in self.layers if l.layer_type == "linear_attention"]

    def embed(self, input_ids):
        return F.embedding(input_ids, self.embed_tokens)

    def forward(self, input_ids, positions):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(positions, x)
        return self.norm(x)

    def compute_logits(self, hidden):
        return F.linear(hidden, self.lm_head)

    def commit_linear_states(self, accepted: list[int]):
        for m in self.linear_attention_modules():
            m.commit(accepted)

    # ------------------------------------------------------------------ weights
    @torch.no_grad()
    def load_weights(self, path: str):
        params = dict(self.named_parameters())
        loaded = set()

        def put(name, tensor, index=None):
            p = params[name]
            dst = p.data if index is None else p.data[index]
            dst.copy_(tensor.to(dst.dtype))
            loaded.add(name)

        I = self.cfg.moe_intermediate_size
        for file in sorted(glob(os.path.join(path, "*.safetensors"))):
            with safe_open(file, "pt", "cpu") as f:
                for name in f.keys():
                    if name.startswith("model.visual") or (name.startswith("mtp.") and self.mtp is None):
                        continue
                    t = f.get_tensor(name)
                    if name == "lm_head.weight":
                        put("lm_head", t); continue
                    if name == "model.language_model.embed_tokens.weight":
                        put("embed_tokens", t); continue
                    if name.startswith("model.language_model."):
                        n = name[len("model.language_model."):]
                    elif name.startswith("mtp."):
                        n = name.replace("mtp.layers.0.", "mtp.layer.")
                    else:
                        raise KeyError(name)
                    n = (n.replace("linear_attn.conv1d.weight", "linear_attn.conv1d_weight")
                          .replace("mlp.experts.gate_up_proj", "mlp.experts_gate_up")
                          .replace("mlp.experts.down_proj", "mlp.experts_down")
                          .replace("mlp.shared_expert.gate_proj", "mlp.shared_gate_proj")
                          .replace("mlp.shared_expert.up_proj", "mlp.shared_up_proj")
                          .replace("mlp.shared_expert.down_proj", "mlp.shared_down_proj"))
                    if ".mlp.experts." in n:  # MTP stores experts one by one
                        pre, rest = n.split(".mlp.experts.")
                        e, proj, _ = rest.split(".")
                        e = int(e)
                        if proj == "gate_proj":
                            put(f"{pre}.mlp.experts_gate_up", t, (e, slice(0, I)))
                        elif proj == "up_proj":
                            put(f"{pre}.mlp.experts_gate_up", t, (e, slice(I, 2 * I)))
                        else:
                            put(f"{pre}.mlp.experts_down", t, e)
                        continue
                    put(n, t)
        missing = [n for n in params if n not in loaded]
        assert not missing, f"unloaded parameters: {missing[:8]}"
