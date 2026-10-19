# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from sglang_tpu.glm_flash_megakernel.kernel import decode as D

_F8 = torch.float8_e4m3fn


def slab_plan(spec, hidden: int) -> dict[str, tuple[int, int, tuple[int, ...]]]:
    """tag -> (row offset, rows, shape) in the bf16 slab; KDA and DSA share one region."""
    nc = (2 + spec.hc_mult) * spec.hc_mult
    m, off = {}, 0

    def add(tag, shape):
        nonlocal off
        n = 1
        for d in shape:
            n *= d
        rows = -(-n // hidden)
        m[tag] = (off, rows, tuple(shape))
        off += rows

    add("input_layernorm", (hidden,))
    add("post_attention_layernorm", (hidden,))
    add("hc_attn_fn", (nc, spec.hc_mult * hidden))
    add("hc_ffn_fn", (nc, spec.hc_mult * hidden))
    common = off
    add("f_a_proj", (spec.linear_head_dim, hidden))
    add("g_a_proj", (spec.linear_head_dim, hidden))
    add("o_norm", (spec.linear_head_dim,))
    kda_end, off = off, common
    add("q_a_layernorm", (spec.q_lora_rank,))
    add("kv_a_layernorm", (spec.kv_lora_rank,))
    add("indexer_wk", (spec.index_head_dim, hidden))
    add("indexer_k_norm_w", (spec.index_head_dim,))
    add("indexer_k_norm_b", (spec.index_head_dim,))
    add("indexer_weights_proj", (spec.index_n_heads, hidden))
    add("indexer_ape", (spec.index_kpool, spec.index_head_dim))
    add("indexer_compress_gate", (spec.index_head_dim, hidden))
    off = max(off, kda_end)
    for tag in ("enorm", "hnorm", "shared_head_norm"):
        add(tag, (hidden,))
    m["_rows"] = (off, 0, ())
    return m


def slab_f32_plan(spec) -> dict[str, tuple[int, int]]:
    """tag -> (offset, n) in row 0 of the f32 slab; identical for every layer."""
    nc = (2 + spec.hc_mult) * spec.hc_mult
    m, off = {}, 0
    for tag, n in (
        ("hc_attn_base", nc),
        ("hc_attn_scale", 3),
        ("hc_ffn_base", nc),
        ("hc_ffn_scale", 3),
        ("router_bias", spec.n_routed_experts),
    ):
        m[tag] = (off, n)
        off += n
    return m


def slab_maps(spec):
    """(small_map, small_f32_map) for decode.SlabIndex."""
    plan = slab_plan(spec, spec.hidden_size)
    plan.pop("_rows")
    fplan = slab_f32_plan(spec)
    f32 = {}
    for L in range(spec.num_hidden_layers):
        tags = ["hc_attn_base", "hc_attn_scale", "hc_ffn_base", "hc_ffn_scale"]
        if spec.mlp_layer_types[L] == "sparse":
            tags.append("router_bias")
        f32[L] = {t: fplan[t] for t in tags}
    return plan, f32


def pack_kmajor(w_kn: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """[L, K, N] -> K-major ring slots [L, n, rows, cols], as decode.pack_kmajor on [L, N, K]."""
    L, K, N = w_kn.shape
    k = D.km_blocks(K, N, rows, cols)
    if N % (cols * k):
        raise ValueError(f"cannot pack [{K}, {N}] into [{rows}, {cols}] slots")
    n = N // (cols * k)
    fp8 = w_kn.dtype == _F8
    w = w_kn.view(torch.uint8) if fp8 else w_kn
    out = w.reshape(L, K, n, k, cols).permute(0, 2, 3, 1, 4).reshape(L, n, k * K, cols)
    if k * K < rows:
        out = torch.nn.functional.pad(out, (0, 0, 0, rows - k * K))
    out = out.contiguous()
    return out.view(_F8) if fp8 else out


def _quant_channel(w_nk: torch.Tensor):
    """bf16 [N, K] -> (fp8 [K, N], f32 scale [N]): per-output-channel amax / 448."""
    w = w_nk.float()
    s = torch.clamp(w.abs().amax(dim=1) / 448.0, min=1e-12)
    return (w / s[:, None]).to(_F8).t().contiguous(), s


def _scale(layer) -> torch.Tensor:
    return layer.weight_scale.reshape(-1).float()


def _gather_vocab(w: torch.Tensor) -> torch.Tensor:
    from sglang.srt.distributed import tensor_model_parallel_all_gather

    return tensor_model_parallel_all_gather(w.contiguous(), dim=0)


def build_weights(
    model,
    spec,
    layout,
    cfg: D.KernelConfig,
    tp: int,
    rank: int,
) -> dict:
    H = spec.hidden_size
    layers = model.model.layers
    dev = model.model.norm.weight.device
    bf16, f32 = torch.bfloat16, torch.float32
    cols, rows8, rows16 = cfg.km_cols, 2 * H, H
    qsh = spec.kda_qkv_dim // tp
    hk = spec.linear_num_heads // tp
    dh = spec.linear_head_dim
    out: dict[str, torch.Tensor] = {}

    embed = _gather_vocab(model.model.embed_tokens.weight)
    hsh = H // tp
    out["embed_tokens"] = embed[:, rank * hsh : (rank + 1) * hsh].contiguous()
    del embed
    lm = _gather_vocab(model.lm_head.weight)
    vsh = layout.vocab_padded // tp
    lm = torch.nn.functional.pad(lm, (0, 0, 0, layout.vocab_padded - lm.shape[0]))
    out["lm_head"] = pack_kmajor(
        lm[rank * vsh : (rank + 1) * vsh].t()[None], rows16, cols
    )
    del lm
    out["final_norm"] = model.model.norm.weight.reshape(1, H).to(bf16)

    # Row num_hidden_layers (the MTP layer) stays zero.
    plan = slab_plan(spec, H)
    n_rows = plan.pop("_rows")[0]
    fplan = slab_f32_plan(spec)
    smalls = torch.zeros(
        (spec.num_hidden_layers + 1, n_rows * H), dtype=bf16, device=dev
    )
    smalls_f = torch.zeros((spec.num_hidden_layers + 1, H), dtype=f32, device=dev)

    def put(L, tag, t):
        off, _, shape = plan[tag]
        flat = t.reshape(-1).to(bf16)
        smalls[L, off * H : off * H + flat.numel()] = flat

    def putf(L, tag, t):
        off, n = fplan[tag]
        smalls_f[L, off : off + n] = t.reshape(-1).float()

    for L in range(spec.num_hidden_layers):
        lay = layers[L]
        put(L, "input_layernorm", lay.input_layernorm.weight)
        put(L, "post_attention_layernorm", lay.post_attention_layernorm.weight)
        for tag in ("hc_attn", "hc_ffn"):
            put(L, f"{tag}_fn", getattr(lay, f"{tag}_fn"))
            putf(L, f"{tag}_base", getattr(lay, f"{tag}_base"))
            putf(L, f"{tag}_scale", getattr(lay, f"{tag}_scale"))
        if spec.mlp_layer_types[L] == "sparse":
            putf(L, "router_bias", lay.mlp.gate.e_score_correction_bias)

    kda = {n: [] for n in ("qkv", "b", "fgb", "o", "dt", "conv", "a_log")}
    for L in spec.kda_layers:
        a = layers[L].self_attn
        if hasattr(a, "fused_qkvbfg_a_proj"):
            w = a.fused_qkvbfg_a_proj.weight
            qkv, b = w[: 3 * qsh], w[3 * qsh : 3 * qsh + hk]
            fa, ga = w[3 * qsh + hk : 3 * qsh + hk + dh], w[3 * qsh + hk + dh :]
        else:
            qkv, b = a.qkv_proj.weight, a.b_proj.weight
            fa, ga = a.f_a_proj.weight, a.g_a_proj.weight
        put(L, "f_a_proj", fa)
        put(L, "g_a_proj", ga)
        put(L, "o_norm", a.o_norm.weight)
        if hasattr(a, "fused_fg_b_proj"):
            fgb = a.fused_fg_b_proj.weight.reshape(2 * qsh, dh)
        else:
            fgb = torch.cat([a.f_b_proj.weight, a.g_b_proj.weight])
        kda["qkv"].append(qkv)
        kda["b"].append(b)
        kda["fgb"].append(fgb)
        kda["o"].append(a.o_proj.weight)
        kda["dt"].append(a.dt_bias.reshape(1, qsh).float())
        kda["conv"].append(a.qkv_conv1d.weight[:, 0, :].t().to(bf16))
        kda["a_log"].append(a.A_log.reshape(1, hk).float())
    q8, sq, o8, so = [], [], [], []
    for qkv, o in zip(kda["qkv"], kda["o"]):
        w, s = _quant_channel(qkv)
        q8.append(w)
        sq.append(s)
        w, s = _quant_channel(o)
        o8.append(w)
        so.append(s)
    out["kda_qkv"] = pack_kmajor(torch.stack(q8), rows8, cols)
    out["kda_o"] = pack_kmajor(torch.stack(o8), rows8, cols)
    out["kda_scale"] = torch.stack([torch.cat([a_, b_]) for a_, b_ in zip(sq, so)])[
        :, None
    ]
    out["kda_fg_b"] = torch.stack(kda["fgb"]).to(bf16)
    out["kda_dt_bias"] = torch.stack(kda["dt"])
    out["kda_conv"] = torch.stack(kda["conv"])
    out["kda_b"] = torch.stack(kda["b"]).to(bf16)
    out["kda_a_log"] = torch.stack(kda["a_log"])

    # Scale rows follow decode.dsa_scale_widths order.
    width = D.proj_scale_cols(spec, tp)
    qakv, qb, do, kvb, iwq, dsc = [], [], [], [], [], []
    for L in spec.dsa_layers:
        a = layers[L].self_attn
        put(L, "q_a_layernorm", a.q_a_layernorm.weight)
        put(L, "kv_a_layernorm", a.kv_a_layernorm.weight)
        ix = a.indexer
        put(L, "indexer_wk", ix.wk.weight)
        put(L, "indexer_k_norm_w", ix.k_norm.weight)
        put(L, "indexer_k_norm_b", ix.k_norm.bias)
        put(L, "indexer_weights_proj", ix.weights_proj.weight)
        put(L, "indexer_ape", ix.index_kpool_compress_ape)
        put(L, "indexer_compress_gate", ix.index_kpool_compress_gate)
        qakv.append(a.fused_qkv_a_proj_with_mqa.weight)
        qb.append(a.q_b_proj.weight)
        do.append(a.o_proj.weight)
        s = torch.cat(
            [_scale(a.fused_qkv_a_proj_with_mqa), _scale(a.q_b_proj), _scale(a.o_proj)]
        )
        dsc.append(torch.nn.functional.pad(s, (0, width - s.numel())))
        kvb.append(a.kv_b_proj.weight)
        iwq.append(ix.wq_b.weight.t())
    out["dsa_qakv"] = pack_kmajor(torch.stack(qakv), rows8, cols)
    out["dsa_q_b"] = pack_kmajor(torch.stack(qb), rows8, cols)
    out["dsa_o"] = pack_kmajor(torch.stack(do), rows8, cols)
    out["dsa_scale"] = torch.stack(dsc)[:, None]
    out["dsa_kv_b"] = torch.stack(kvb).to(bf16)
    out["indexer_wq_b"] = pack_kmajor(torch.stack(iwq).to(bf16), rows16, cols)

    dgu, ddn, dnsc = [], [], []
    for L in range(spec.first_k_dense_replace):
        m = layers[L].mlp
        dgu.append(m.gate_up_proj.weight)
        ddn.append(m.down_proj.weight)
        s = torch.cat([_scale(m.gate_up_proj), _scale(m.down_proj)])
        dnsc.append(torch.nn.functional.pad(s, (0, width - s.numel())))
    out["dense_gate_up"] = pack_kmajor(torch.stack(dgu), rows8, cols)
    out["dense_down"] = pack_kmajor(torch.stack(ddn), rows8, cols)
    out["dense_scale"] = torch.stack(dnsc)[:, None]

    E = spec.n_routed_experts
    moe = [
        L for L in range(spec.num_hidden_layers) if spec.mlp_layer_types[L] == "sparse"
    ]
    router, e_gate_up, e_down, egs, eds = [], [], [], [], []
    sgu, sgs, sdn, sds = [], [], [], []
    for m_i, L in enumerate(moe):
        m = layers[L].mlp
        router.append(m.gate.weight.to(bf16))
        ex = m.experts
        e_gate_up.append(ex.w13_weight.data)
        e_down.append(ex.w2_weight.data)
        egs.append(ex.w13_weight_scale.reshape(E, -1).float())
        eds.append(ex.w2_weight_scale.reshape(E, -1).float())
        sh = m.shared_experts
        sgu.append(sh.gate_up_proj.weight)
        sgs.append(_scale(sh.gate_up_proj))
        sdn.append(sh.down_proj.weight)
        sds.append(_scale(sh.down_proj))
    out["moe_router"] = torch.stack(router)
    # sglang's own per-layer expert tensors, not copies.
    out["expert_gate_up"] = e_gate_up
    out["expert_down"] = e_down
    out["expert_gate_up_scale"] = torch.cat(egs)
    out["expert_down_scale"] = torch.cat(eds)
    out["shared_gate_up"] = torch.stack(sgu)
    out["shared_gate_up_scale"] = torch.stack(sgs)
    out["shared_down"] = torch.stack(sdn)
    out["shared_down_scale"] = torch.stack(sds)

    out["smalls"] = smalls.reshape(spec.num_hidden_layers + 1, n_rows, H)
    out["smalls_f32"] = smalls_f.reshape(spec.num_hidden_layers + 1, 1, H)
    return out
