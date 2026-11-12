"""Edge-scenario weights: symmetric 4-bit, group-size-128 quantization of the routed MoE experts (main layers + MTP).

  python -m evo.quantize_int4 SRC_BF16_DIR OUT_INT4_DIR OUT_REF_DIR [--device cuda:0]

Format (OUT_INT4_DIR, read by the edge engine):
  every routed-expert weight W [..., N, K] (K = input dim) is replaced by
    <name>.qweight  uint8 [..., N, K/2]   two signed 4-bit values per byte, stored as q + 8 in 0..15;
                                          low nibble = even k, high nibble = odd k
    <name>.scales   bf16  [..., N, K/128] one scale per 128 consecutive k
  (<name> drops a trailing ".weight", e.g. mtp.layers.0.mlp.experts.3.gate_proj.qweight;
   fused tensors keep their name, e.g. model.language_model.layers.0.mlp.experts.gate_up_proj.qweight)
  canonical value: W[n, k] = bf16( float32(q[n, k]) * float32(scales[n, k // 128]) ),  q in -8..7
  everything else (attention, DeltaNet, shared experts, router, embeddings, lm_head, norms, MTP non-expert) is unchanged BF16.
OUT_REF_DIR is a plain BF16 checkpoint with exactly those canonical expert values: the judge's reference model, so that
correctness is judged against the quantized model itself (not against BF16)."""
import argparse, json, os, re, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

G = 128
EXPERT = re.compile(r"(\.mlp\.experts\.(gate_up_proj|down_proj)$)|(\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$)")


def quant(w):
    """w: bf16 [..., N, K] -> (packed uint8 [..., N, K/2], scales bf16 [..., N, K/G], canonical bf16 [..., N, K])."""
    K = w.shape[-1]; assert K % G == 0
    x = w.float().reshape(*w.shape[:-1], K // G, G)
    s = (x.abs().amax(-1) / 7.0).to(torch.bfloat16)
    s = torch.where(s == 0, torch.ones_like(s), s)
    q = torch.clamp(torch.round(x / s.float().unsqueeze(-1)), -8, 7)
    deq = (q * s.float().unsqueeze(-1)).to(torch.bfloat16).reshape(w.shape)
    u = (q.to(torch.int16) + 8).to(torch.uint8).reshape(*w.shape[:-1], K)
    packed = (u[..., 0::2] | (u[..., 1::2] << 4)).contiguous()
    return packed, s.contiguous(), deq


def base(name):
    return name[: -len(".weight")] if name.endswith(".weight") else name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out_q"); ap.add_argument("out_ref"); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    for d in (a.out_q, a.out_ref):
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(a.src):
            if not f.endswith(".safetensors") and f != "model.safetensors.index.json" and os.path.isfile(os.path.join(a.src, f)):
                shutil.copy2(os.path.join(a.src, f), d)
    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
    qmap, n_q, err = {}, 0, []
    for shard in sorted(set(idx["weight_map"].values())):
        tq, tr = {}, {}
        with safe_open(os.path.join(a.src, shard), "pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if EXPERT.search(k):
                    p, s, deq = quant(t.to(a.device))
                    tq[base(k) + ".qweight"] = p.cpu(); tq[base(k) + ".scales"] = s.cpu(); tr[k] = deq.cpu()
                    qmap[base(k) + ".qweight"] = qmap[base(k) + ".scales"] = shard
                    err.append(((deq.float() - t.to(a.device).float()).norm() / t.to(a.device).float().norm()).item()); n_q += 1
                else:
                    tq[k] = t; tr[k] = t; qmap[k] = shard
        save_file(tq, os.path.join(a.out_q, shard), metadata={"format": "pt"})
        save_file(tr, os.path.join(a.out_ref, shard), metadata={"format": "pt"})
        print(shard, len(tq), "tensors", flush=True)
    json.dump({"metadata": {}, "weight_map": qmap}, open(os.path.join(a.out_q, "model.safetensors.index.json"), "w"), indent=1)
    shutil.copy2(os.path.join(a.src, "model.safetensors.index.json"), a.out_ref)
    qc = {"quant_method": "evo_int4", "bits": 4, "group_size": G, "sym": True, "zero_point": 8,
          "modules": "routed MoE experts (main + MTP)", "pack": "uint8, low nibble = even k"}
    for d, extra in ((a.out_q, {"quantization_config": qc}), (a.out_ref, {"evo_reference_for": qc})):
        c = json.load(open(os.path.join(d, "config.json"))); c.update(extra)
        json.dump(c, open(os.path.join(d, "config.json"), "w"), indent=2)
    print(json.dumps({"quantized_tensors": n_q, "mean_rel_err": sum(err) / len(err), "max_rel_err": max(err)}))


if __name__ == "__main__":
    main()
