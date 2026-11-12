"""Qwen3.5-MoE (text part) for nano-vllm: hybrid Gated DeltaNet / gated full attention, 256-expert MoE
with a gated shared expert, and the native MTP draft layer.

Semantics follow transformers' modeling_qwen3_5_moe.py. Short verification uses
a fused recurrence with per-token rollback snapshots; prefill uses FLA.
Single GPU (TP=1) only.
"""
import os
from glob import glob

import torch
import torch.nn.functional as F
from torch import nn
from safetensors import safe_open
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from nanovllm.layers.delta_verify import delta_verify
from nanovllm.layers.moe_router import route_logits
from nanovllm.layers.delta_qkv import delta_qkv
from nanovllm.layers.delta_gates import delta_gates
from nanovllm.layers.delta_gate_projection import delta_gate_projection
from nanovllm.layers.delta_conv_short import delta_conv_short
from nanovllm.layers.attention_norm_rotary import attention_norm_rotary, attention_gate
from nanovllm.layers.silu_mul import silu_mul
from nanovllm.layers.rotary_short import rotary_short
from nanovllm.layers.norm_apply import norm_apply, squared_float
from nanovllm.layers.norm_exact import norm_exact
from nanovllm.layers.moe_prefill import grouped_prefill
from nanovllm.layers.moe_decode import moe_decode, normalize_routes

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
        if x.shape[-1] in (128, 256, 2048):
            return norm_exact(x, self.weight, self.eps)
        if x.shape[0] <= 8:
            mean_square = squared_float(x).mean(-1, keepdim=True)
            return norm_apply(x, self.weight, mean_square, self.eps)
        xf = x.float()
        scale = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        xf = xf * scale
        return (xf * (1.0 + self.weight.float())).type_as(x)


class GatedRMSNorm(nn.Module):
    """RMSNorm(x) * silu(gate) used at the output of Gated DeltaNet."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if x.shape[-1] in (128, 256, 2048):
            return norm_exact(x, self.weight, self.eps, gate)
        if x.numel() <= 32768:
            mean_square = squared_float(x).mean(-1, keepdim=True)
            return norm_apply(x, self.weight, mean_square, self.eps, gate, gate_is_raw=True)
        xf = x.float()
        scale = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        xf = xf * scale
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
        if q.shape[0] <= 8:
            return rotary_short(positions, q, k, self.cos_cache, self.sin_cache, self.rotary_dim)
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
        short = T <= 8 and self.head_dim in (128, 256)
        if short:
            k = self.k_proj(x).view(T, self.num_kv_heads, self.head_dim)
            v = self.v_proj(x).view(T, self.num_kv_heads, self.head_dim)
            q, k = attention_norm_rotary(
                positions, q, k, self.q_norm.weight, self.k_norm.weight,
                self.rotary.cos_cache, self.rotary.sin_cache,
                self.rotary.rotary_dim, self.q_norm.eps)
        else:
            gate = gate.reshape(T, -1)
            q = self.q_norm(q)
            k = self.k_norm(self.k_proj(x).view(T, self.num_kv_heads, self.head_dim))
            v = self.v_proj(x).view(T, self.num_kv_heads, self.head_dim)
            q, k = self.rotary(positions, q, k)
        o = self.attn(q.contiguous(), k.contiguous(), v.contiguous())
        o = o.reshape(T, -1)
        o = attention_gate(o, gate) if short else o * torch.sigmoid(gate)
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
        if not ctx.is_prefill and T == 4:
            g, beta = delta_gate_projection(
                x, self.in_proj_a.weight, self.in_proj_b.weight, self.A_log, self.dt_bias)
        else:
            b, a = self.in_proj_b(x), self.in_proj_a(x)
            if not ctx.is_prefill:
                g, beta = delta_gates(a, b, self.A_log, self.dt_bias)
            else:
                beta = b.sigmoid()
                g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        w = self.conv1d_weight                             # [C, 1, K]
        Km1 = self.conv_kernel - 1
        outs = []
        start = 0
        for i, qlen in enumerate(ctx.query_lens):
            slot = ctx.state_slots[i]
            xs = mixed[start:start + qlen].t()             # [C, q]
            if not ctx.is_prefill and qlen <= 8 and self.conv_kernel == 4:
                q, k, v, conv_in = delta_conv_short(
                    mixed[start:start + qlen], self.conv_state[slot], w,
                    self.num_k_heads, self.num_v_heads, self.head_k_dim, self.head_v_dim)
            else:
                if ctx.is_prefill:
                    conv_in = F.pad(xs, (Km1, 0))
                else:
                    conv_in = torch.cat([self.conv_state[slot], xs], dim=-1)
                conv_out = F.silu(F.conv1d(conv_in.unsqueeze(0), w, groups=self.conv_dim)).squeeze(0)
                conv_out = conv_out.t().to(x.dtype)
                if ctx.is_prefill:
                    q, k, v = self._qkv_heads(conv_out, qlen)
                else:
                    q, k, v = delta_qkv(conv_out, self.num_k_heads, self.num_v_heads,
                                        self.head_k_dim, self.head_v_dim)
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
                o, states = delta_verify(q, k, v, gi, bi, self.ssm_state[slot])
                outs.append(o)
                self.pending.append((slot, conv_in, states))
            start += qlen
        # The recurrence already returns contiguous [T, HV, V] output.
        core = outs[0] if not ctx.is_prefill and len(outs) == 1 else torch.cat(outs, 0)
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

class Int4Experts(nn.Module):
    """Resident evo_int4 storage; weights are rounded to BF16 inside the kernels."""
    def __init__(self, experts, n, k):
        super().__init__()
        assert k % 128 == 0
        self.qweight = nn.Parameter(torch.empty(experts, n, k // 2, dtype=torch.uint8),
                                    requires_grad=False)
        self.scales = nn.Parameter(torch.empty(experts, n, k // 128, dtype=torch.bfloat16),
                                   requires_grad=False)


class SparseMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_tok
        H, I = cfg.hidden_size, cfg.moe_intermediate_size
        self.gate = Linear(H, self.num_experts)
        quant = getattr(cfg, "quantization_config", None)
        self.int4 = quant is not None
        if self.int4:
            if (quant.get("quant_method") != "evo_int4" or quant.get("bits") != 4
                    or quant.get("group_size") != 128 or quant.get("zero_point") != 8
                    or quant.get("sym") is not True
                    or quant.get("pack") != "uint8, low nibble = even k"):
                raise ValueError(f"Unsupported expert quantization: {quant}")
            self.experts_gate_up = Int4Experts(self.num_experts, 2 * I, H)
            self.experts_down = Int4Experts(self.num_experts, H, I)
        else:
            self.experts_gate_up = nn.Parameter(torch.empty(self.num_experts, 2 * I, H))
            self.experts_down = nn.Parameter(torch.empty(self.num_experts, H, I))
        Is = cfg.shared_expert_intermediate_size
        self.shared_gate_proj = Linear(H, Is)
        self.shared_up_proj = Linear(H, Is)
        self.shared_down_proj = Linear(Is, H)
        self.shared_expert_gate = Linear(H, 1)
        self.register_buffer("short_input_weight", None, persistent=False)

    def forward(self, x):
        if x.shape[0] <= 8:
            size = self.shared_gate_proj.weight.shape[0]
            projected = F.linear(x, self.short_input_weight)
            gate, up, logits, shared_gate = projected[:, :2 * size + self.num_experts + 1].split(
                (size, size, self.num_experts, 1), dim=-1)
            shared = self.shared_down_proj(silu_mul(gate, up))
        else:
            shared = self.shared_down_proj(F.silu(self.shared_gate_proj(x)) * self.shared_up_proj(x))
            logits = self.gate(x)
            shared_gate = self.shared_expert_gate(x)
        T = x.shape[0]
        if T <= 8 and self.num_experts == 256 and self.top_k == 8:
            topw, topi = route_logits(logits)
        else:
            probs = torch.softmax(logits, dim=-1, dtype=torch.float)
            topw, topi = torch.topk(probs, self.top_k, dim=-1)
            if T * self.top_k <= 64:
                topw = normalize_routes(topw, logits.dtype)
            else:
                topw = (topw / topw.sum(-1, keepdim=True)).to(logits.dtype)
        gu = self.experts_gate_up.qweight if self.int4 else self.experts_gate_up
        down = self.experts_down.qweight if self.int4 else self.experts_down
        gs = self.experts_gate_up.scales if self.int4 else None
        ds = self.experts_down.scales if self.int4 else None
        if T * self.top_k <= 64:
            return moe_decode(x, gu, down, topi, topw, shared, shared_gate, gs, ds)
        shared = torch.sigmoid(shared_gate) * shared
        return grouped_prefill(x, topi, topw, gu, down, gs, ds) + shared


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
        mtp_slices = set()

        def put(name, tensor, index=None):
            p = params[name]
            dst = p.data if index is None else p.data[index]
            if tensor.shape != dst.shape or (name.endswith((".qweight", ".scales"))
                                             and tensor.dtype != dst.dtype):
                raise ValueError(f"Checkpoint mismatch for {name}: {tensor.shape}/{tensor.dtype}, "
                                 f"expected {dst.shape}/{dst.dtype}")
            dst.copy_(tensor)
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
                        e, proj, suffix = rest.split(".")
                        tail = "" if suffix == "weight" else "." + suffix
                        e = int(e)
                        mtp_slices.add((e, proj, suffix))
                        if proj == "gate_proj":
                            put(f"{pre}.mlp.experts_gate_up{tail}", t, (e, slice(0, I)))
                        elif proj == "up_proj":
                            put(f"{pre}.mlp.experts_gate_up{tail}", t, (e, slice(I, 2 * I)))
                        else:
                            put(f"{pre}.mlp.experts_down{tail}", t, e)
                        continue
                    put(n, t)
        if self.mtp is not None:
            suffixes = ("qweight", "scales") if self.mtp.layer.mlp.int4 else ("weight",)
            expected = {(e, p, s) for e in range(self.cfg.num_experts)
                        for p in ("gate_proj", "up_proj", "down_proj") for s in suffixes}
            assert mtp_slices == expected, "missing or unexpected MTP expert slices"
        missing = [n for n in params if n not in loaded]
        assert not missing, f"unloaded parameters: {missing[:8]}"
        # Keep checkpoint parameters and prefill GEMMs unchanged. Packing once
        # removes three input-projection launches from each short-batch MoE.
        # Pad to 64 output rows: the extra scalar gate would otherwise select
        # a much slower unaligned cuBLAS kernel. Consumers read strided views.
        for module in self.modules():
            if isinstance(module, SparseMoE):
                packed = torch.cat((module.shared_gate_proj.weight,
                                    module.shared_up_proj.weight, module.gate.weight,
                                    module.shared_expert_gate.weight), dim=0)
                module.short_input_weight = F.pad(packed, (0, 0, 0, (-packed.shape[0]) % 64))
