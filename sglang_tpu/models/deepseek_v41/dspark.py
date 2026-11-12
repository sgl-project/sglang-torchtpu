# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 DSpark draft on TPU: the model-side pieces the hooks install.

The draft is the checkpoint's three ``mtp.*`` MoE stages
(``DeepseekV4ForCausalLMDSpark``), every stage a ratio-0 CSA2 SWA layer, as in
vllm-torchtpu ``models/vllm/deepseek_v41/v2/dspark.py``:

- attention: the backend's ``prepare_mqa_qkv`` (q norm, RoPE, kv norm + RoPE)
  and its non-causal block attention; CSA2 already inverse-rotates the output,
  so only ``wo_a`` / ``wo_b`` remain.
- draft tokens: greedy Markov chain in the draft graph,
  ``z_j = lm_head(norm(h_j)) + markov_w2(markov_w1(d_{j-1}))``, ``d_0`` the anchor.
- context KV: ``main_norm(main_proj(target aux hidden))`` projected by each
  stage's ``wkv``, kv-normed, RoPE'd and published into the stage's SWA pages.
"""

from __future__ import annotations

import torch

from sglang_tpu.models.deepseek_v41.attention_backend import apply_rope


def is_dsv41_dspark_draft(model) -> bool:
    return (
        type(model).__name__ == "DeepseekV4ForCausalLMDSpark"
        and model.config.model_type == "deepseek_v41"
    )


def dspark_attention_forward(attn, positions, hidden_states, forward_batch):
    """``DSparkAttention.forward`` on the V4.1 TPU backend."""
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    q_lora, _ = attn.wq_a(hidden_states)
    _, q, kv = backend.prepare_mqa_qkv(attn, hidden_states, q_lora, None, positions)
    o = backend.forward(
        q,
        kv,
        kv,
        attn.attn,
        forward_batch,
        save_kv_cache=False,
        compress_ratio=0,
        attn_sink=attn._local_attn_sink(),
    )
    o = o[:, : attn.n_local_heads]
    num_tokens = o.shape[0]
    groups = attn.n_local_groups
    o = o.reshape(num_tokens, groups, -1)
    wo_a = attn.wo_a.weight.view(groups, attn.o_lora_rank, -1)
    o = torch.einsum("tgd,grd->tgr", o, wo_a)
    out, _ = attn.wo_b(o.reshape(num_tokens, -1))
    return out


def block_width(model) -> int:
    """Rows per request in the draft block: anchor drafts read every row."""
    return model.gamma if model.sample_from_anchor else model.gamma + 1


def greedy_markov_drafts(
    model, hidden_states: torch.Tensor, input_ids: torch.Tensor
) -> torch.Tensor:
    """[bs, gamma] greedy drafts from the [bs * width, H] block hidden.

    The Markov embedding is a one-hot matmul instead of a gather: an unsharded
    BF16 table gathered inside the TP1/DP8 compiled graph returns wrong rows
    (see tpu_dsv41_int32_graph_inputs). With one 1.0 per row it is exact.
    """
    width = block_width(model)
    gamma = model.gamma
    hidden = hidden_states.view(-1, width, hidden_states.shape[-1])
    bs = hidden.shape[0]
    draft_hidden = hidden if model.sample_from_anchor else hidden[:, 1:]
    head = model.markov_head
    vocab = head.vocab_size
    base_logits, _ = model.compute_base_logits(draft_hidden.reshape(bs * gamma, -1))
    base_logits = base_logits[..., :vocab].float().view(bs, gamma, vocab)
    w1 = head.markov_w1.weight[:vocab]
    w2 = head.markov_w2.weight[:vocab]
    ids = torch.arange(vocab, device=hidden.device, dtype=torch.int32)
    prev = input_ids.view(bs, width)[:, 0].to(torch.int32)
    drafts = []
    for step in range(gamma):
        one_hot = (prev.unsqueeze(-1) == ids).to(w1.dtype)
        latent = one_hot @ w1
        bias = (latent.to(w2.dtype) @ w2.T).float()
        prev = torch.argmax(base_logits[:, step] + bias, dim=-1).to(torch.int32)
        drafts.append(prev)
    return torch.stack(drafts, dim=1).to(torch.int64)


def dspark_forward(original, model, input_ids, positions, forward_batch, *args, **kw):
    """``DeepseekV4ForCausalLMDSpark.forward`` plus the in-graph greedy drafts
    TpuBlockSpecWorker reads from ``next_draft_tokens``."""
    out = original(model, input_ids, positions, forward_batch, *args, **kw)
    hidden = out.hidden_states
    if hidden is not None and hidden.numel():
        out.next_draft_tokens = greedy_markov_drafts(model, hidden, input_ids)
    return out


def swa_commit_records(
    cache_loc: torch.Tensor,
    cache_loc_2d: torch.Tensor | None,
    commit_lens: torch.Tensor | None,
) -> torch.Tensor:
    """[T] int32 SWA records of the rows to write; -1 skips a row.

    Extend writes every row (padding rows carry slot 0, which publish_swa drops
    with the null page); verify commit writes the first commit_lens columns.
    """
    if cache_loc_2d is None:
        return cache_loc.to(torch.int32)
    cols = torch.arange(cache_loc_2d.shape[1], device=cache_loc_2d.device)
    committed = cols.unsqueeze(0) < commit_lens.to(cols.dtype).unsqueeze(-1)
    records = torch.where(committed, cache_loc_2d, torch.full_like(cache_loc_2d, -1))
    return records.reshape(-1).to(torch.int32)


def write_context_kv(
    model,
    pool,
    *,
    target_hidden: torch.Tensor,
    positions: torch.Tensor,
    records: torch.Tensor,
) -> torch.Tensor:
    """Publish every stage's context KV; returns the last pool write's fence tail."""
    from sglang_tpu.models.deepseek_v41.ops import csa2

    main_x = model.project_target_hidden(target_hidden)
    positions = positions.to(torch.int32)
    pages = None
    for stage in model.stages:
        attn = stage.self_attn
        kv, _ = attn.wkv(main_x)
        kv = apply_rope(attn.kv_norm(kv), positions, attn.freqs_cis)
        pages = pool.get_swa_pages(attn.layer_id)
        pages.copy_(csa2.publish_swa(pages, records, kv))
    assert pages is not None, "the draft has no stages"
    # A reduction, not a view: the KV free fence reads it after later donations.
    return pages[:1, :1].to(torch.float32).sum(dim=(1, 2))


def commit_engram_history(
    hasher,
    *,
    bonus_tokens: torch.Tensor,
    accept_tokens: torch.Tensor,
    accept_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
) -> None:
    """Advance each request's Engram history past its committed verify tokens.

    The verify window is [r, d_1..d_N]; a request commits r and its correct
    drafts (accept_lens tokens; accept_tokens holds the correct drafts first),
    so its history becomes the last n - 1 of history + committed, oldest first,
    as core EngramHasher.commit_after_verify. Padded rows (accept_lens 0) write
    the hasher's pad row.
    """
    history = hasher.history
    n1 = hasher.max_ngram_size - 1
    req = req_pool_indices.to(torch.int64)
    window = torch.cat(
        [
            history[req],
            bonus_tokens.view(-1, 1).to(history.dtype),
            accept_tokens.clamp_min(0).to(history.dtype),
        ],
        dim=1,
    )
    cols = accept_lens.to(torch.int64).unsqueeze(-1) + torch.arange(
        n1, device=window.device
    )
    values = window.gather(1, cols)
    rows = torch.where(accept_lens > 0, req, torch.full_like(req, hasher.pad_row))
    history.copy_(history.index_put((rows,), values))
