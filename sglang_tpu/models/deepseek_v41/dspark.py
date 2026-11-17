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
- base logits: this rank's full-vocab rows (no TP gather) at attention TP 1.
- context KV: ``main_norm(main_proj(target aux hidden))`` projected by each
  stage's ``wkv``, kv-normed, RoPE'd and published into the stage's SWA pages.
"""

from __future__ import annotations

import torch

from sglang_tpu.models.deepseek_v41.attention_backend import apply_rope


def serves_dspark_draft() -> bool:
    """Whether the forward runs on the CSA2 backend these draft pieces call into."""
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang_tpu.models.deepseek_v41.attention_backend import DSV41TPUAttnBackend

    return isinstance(get_attn_backend(), DSV41TPUAttnBackend)


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


def tpu_compute_base_logits(
    model, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``compute_base_logits`` of a full-vocab head: this rank's rows, no gather.

    Under attention DP the target lm_head shards over the attention-TP group;
    at attention TP 1 every rank holds the full vocab for its own rows. Core's
    gather_and_crop_vocab gathers over the full TP group instead, so cropping
    keeps TP rank 0's logits on every rank.
    """
    from sglang.srt.models.dspark import project_through_lm_head

    lm_head = model.lm_head
    x_post_hc = model.collapse_hc_head(x)
    x = model.stages[-1].norm(x_post_hc)
    logits = project_through_lm_head(x, lm_head)[..., : int(lm_head.org_vocab_size)]
    return logits, x_post_hc


def greedy_markov_drafts(
    model, hidden_states: torch.Tensor, input_ids: torch.Tensor, *, width: int
) -> torch.Tensor:
    """[bs, gamma] greedy drafts from the [bs * width, H] block hidden.

    The Markov embedding is a one-hot matmul instead of a gather: an unsharded
    BF16 table gathered inside the TP1/DP8 compiled graph returns wrong rows,
    with int32 and int64 ids alike (DeepseekV4DecoderTpuCompilePolicy). With one
    1.0 per row it is exact.
    """
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
        width = int(forward_batch.spec_info.draft_token_num)
        out.next_draft_tokens = greedy_markov_drafts(
            model, hidden, input_ids, width=width
        )
    return out


def write_target_hidden_kv(
    model,
    *,
    main_hidden: torch.Tensor,
    swa_loc: torch.Tensor,
    positions: torch.Tensor,
    pool,
) -> None:
    """``write_target_hidden_kv`` into the CSA2 SWA pages; -1 in swa_loc skips a row."""
    from sglang_tpu.models.deepseek_v41.ops import csa2

    main_x = model.project_target_hidden(main_hidden)
    positions = positions.to(torch.int32)
    records = swa_loc.to(torch.int32)
    for stage in model.stages:
        attn = stage.self_attn
        kv, _ = attn.wkv(main_x)
        kv = apply_rope(attn.kv_norm(kv), positions, attn.freqs_cis)
        pages = pool.get_swa_pages(attn.layer_id)
        pages.copy_(csa2.publish_swa(pages, records, kv))
