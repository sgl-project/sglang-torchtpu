# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp (Qwen3.8-Flash-Next) on TPU: dense full attention in place of QSA.

dense_qsa_config
  Why: TPU has no QSA kernel. Core keys every QSA path (the CUDA-only
       QwenSparseAttnBackend, the QSA KV pool, page_size 64, each full-attention
       layer's indexer) on parse_qsa_profile(config), so a config that keeps
       the indexer fields cannot start on TPU.
  How: AFTER get_config: for a qwen4_exp config, record indexer_budget as
       tpu_dense_qsa_budget and clear indexer_n_heads, so parse_qsa_profile
       returns None everywhere. The full-attention layers then run dense on the
       TPU attention backend and the checkpoint's indexer weights load as
       unused. QSA attends to the indexer_budget highest-scoring tokens of each
       query row, so dense attention equals it while every sequence fits that
       budget.
  Upstream PR or issue: not filed.
  Remove when: sglang_tpu has a QSA backend (block top-k indexer + sparse GQA).

check_dense_qsa_context
  Why: past the budget, dense attention attends to tokens QSA would drop, so
       outputs diverge from the model's.
  How: AFTER ModelConfig.__init__: refuse a context_len above
       tpu_dense_qsa_budget unless SGLANG_TPU_QWEN4_DENSE_QSA=approx, which is
       for plumbing and performance work whose results are not QSA-exact.
  Upstream PR or issue: not filed.
  Remove when: dense_qsa_config is removed.

load_ple_table_on_host
  Why: core fills the PLE n-gram table (51.2 GB fp8 on Qwen3.8-Flash-Next)
       slice by slice from its checkpoint shards; on TPU each slice copy
       rewrites the whole table, so a TP=1 pipeline stage needs two tables of
       HBM and fails to compile.
  How: AROUND Qwen4ExpForConditionalGeneration.load_weights: take the
       ngram_embedding.shard_<i> weights out of the stream, gather this rank's
       rows in host memory, and upload the table once after the load.
  Upstream PR or issue: not filed.
  Remove when: TPU loads large row-sharded tables without whole-table copies.

gather_fp8_ple_rows
  Why: a TP=1 stage's PLE lookup F.embedding(ids, fp8_table).to(bf16)
       compiled to a whole-table conversion (140.98G of HLO temporaries on
       PP0 of Qwen3.8-Flash-Next).
  How: AROUND Qwen4ExpNGramEmbedding._embed_ngram_ids: at TP=1 with an fp8
       table, index_select the rows of a uint8 view and convert only those.
  Upstream PR or issue: not filed.
  Remove when: TPU lowers fp8 gathers without converting the operand.

hash_ngrams_int32
  Why: the PLE n-gram hash multiplies and XORs int64 values; it compiled for
       prefill, but decode-sized graphs aborted the TPU compiler
       (lowering_util.cc "Unsupported conversion from vmreg/vreg to U64").
  How: AROUND Qwen4ExpNGramEmbedding._hash_contexts: the same hash in int32
       limbs (bit-exact; products stay below 2^63 by construction) with a
       Horner reduction modulo each head's prime vocab size.
  Upstream PR or issue: not filed.
  Remove when: TPU compiles 64-bit integer vector arithmetic at every shape.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DENSE_QSA_ENV = "SGLANG_TPU_QWEN4_DENSE_QSA"
BUDGET_ATTR = "tpu_dense_qsa_budget"


def _text_config(config: Any) -> Any:
    return getattr(config, "text_config", None) or config


def dense_qsa_config(config: Any, *args: Any, **kwargs: Any) -> Any:
    """AFTER get_config. get_config is cached, so this runs again on the same
    object; recording the budget first keeps it idempotent."""
    if getattr(config, "model_type", None) != "qwen4_exp":
        return config
    text_config = _text_config(config)
    budget = getattr(text_config, "indexer_budget", None)
    if budget is None or getattr(text_config, BUDGET_ATTR, None) is not None:
        return config
    budget = int(budget)
    setattr(text_config, BUDGET_ATTR, budget)
    text_config.indexer_n_heads = None
    logger.info(
        "Qwen4-Exp on TPU: QSA runs as dense full attention, exact up to the "
        "%d-token indexer budget.",
        budget,
    )
    return config


def check_dense_qsa_context(result: None, model_config: Any, *args, **kwargs) -> None:
    """AFTER ModelConfig.__init__."""
    budget = getattr(_text_config(model_config.hf_config), BUDGET_ATTR, None)
    if budget is None:
        return
    context_len = int(model_config.context_len)
    if context_len <= budget:
        return
    if os.environ.get(DENSE_QSA_ENV, "") == "approx":
        logger.warning(
            "Qwen4-Exp on TPU: context_len %d exceeds the %d-token QSA budget; "
            "dense attention there is NOT QSA-exact (%s=approx).",
            context_len,
            budget,
            DENSE_QSA_ENV,
        )
        return
    raise ValueError(
        f"Qwen4-Exp on TPU runs its QSA layers as dense attention, which matches "
        f"QSA only up to the {budget}-token indexer budget; got context_len "
        f"{context_len}. Pass --context-length {budget} or less, or set "
        f"{DENSE_QSA_ENV}=approx for runs that do not need QSA-exact outputs."
    )


def load_ple_table_on_host(original: Any, model: Any, weights: Any, *args, **kwargs):
    """AROUND Qwen4ExpForConditionalGeneration.load_weights.

    The checkpoint stores the PLE n-gram table as row shards
    (``...ngram_embedding.shard_<i>.weight``; 51.2 GB of fp8 for
    Qwen3.8-Flash-Next) that core copies into row slices of the device table
    one at a time. On TPU each slice copy is a functional update of the whole
    table, so it needs two tables of HBM: a TP=1 pipeline stage holding all of
    it fails to compile (CompileTimeHbmOom in copy__as_strided_inverse). Gather
    this rank's rows on the host instead and upload the table once.
    """
    import re

    import torch

    from sglang.srt.models.qwen4_exp import Qwen4ExpNGramEmbedding

    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, Qwen4ExpNGramEmbedding)
        and not isinstance(module.ngram_embedding.weight, torch.nn.UninitializedParameter)
        and module.ngram_embedding.weight.device.type != "meta"
    }
    if not modules:
        return original(model, weights, *args, **kwargs)
    text_config = _text_config(model.config)
    num_shards = int(
        getattr(
            text_config,
            "split_ngram_parts",
            getattr(model.config, "split_ngram_parts", 512),
        )
    )
    pattern = re.compile(r"^(.*)\.ngram_embedding\.shard_(\d+)\.weight$")
    tables: dict[str, torch.Tensor] = {}

    def host_rows(prefix: str, index: int, loaded: torch.Tensor) -> None:
        emb = modules[prefix].ngram_embedding
        table = tables.get(prefix)
        if table is None:
            table = tables[prefix] = torch.zeros(
                emb.weight.shape, dtype=emb.weight.dtype, device="cpu"
            )
        shard_rows = (emb.org_vocab_size + num_shards - 1) // num_shards
        row_start = index * shard_rows
        row_end = row_start + int(loaded.shape[0])
        tp_start = emb.shard_indices.org_vocab_start_index
        tp_end = emb.shard_indices.org_vocab_end_index
        start, end = max(row_start, tp_start), min(row_end, tp_end)
        if start < end:
            table[start - tp_start : end - tp_start].copy_(
                loaded[start - row_start : end - row_start].to(
                    device="cpu", dtype=table.dtype
                )
            )

    def filtered():
        for name, loaded in weights:
            match = pattern.match(name.replace("model.language_model.", "model."))
            if match is not None and match.group(1) in modules:
                host_rows(match.group(1), int(match.group(2)), loaded)
                continue
            yield name, loaded

    loaded_params = original(model, filtered(), *args, **kwargs)
    for module in modules.values():
        # The int32 hash's host constants, before any trace reads them.
        _ngram_hash_limb_plan(module)
    for prefix, table in tables.items():
        emb = modules[prefix].ngram_embedding
        device = emb.weight.device
        # Release the init-time table before the upload: two would not fit.
        emb.weight.data = torch.empty(0, dtype=table.dtype, device=device)
        emb.weight.data = table.to(device)
        if loaded_params is not None:
            loaded_params.add(f"{prefix}.ngram_embedding.weight")
        logger.info(
            "Qwen4-Exp on TPU: uploaded PLE table %s %s %s in one transfer.",
            prefix,
            tuple(table.shape),
            table.dtype,
        )
    tables.clear()
    return loaded_params


def gather_fp8_ple_rows(
    original: Any,
    ngram: Any,
    ngram_ids: Any,
    forward_batch: Any,
    physical_tokens: int,
    *args,
    **kwargs,
):
    """AROUND Qwen4ExpNGramEmbedding._embed_ngram_ids.

    With the whole fp8 PLE table on one device (a TP=1 pipeline stage), the
    core lookup ``F.embedding(ids, fp8_table).to(bf16)`` compiles to a
    whole-table conversion on TPU: the stage-0 forward needed 140.98G of HLO
    temporaries for a 51.2 GB table. Gather the rows through a uint8 view of the
    table, which moves bytes only, and convert just the gathered rows.
    """
    import torch

    emb = ngram.ngram_embedding
    weight = getattr(emb, "weight", None)
    if (
        not isinstance(weight, torch.Tensor)
        or weight.dtype != torch.float8_e4m3fn
        or getattr(emb, "tp_size", 1) != 1
        or weight.device.type == "meta"
    ):
        return original(ngram, ngram_ids, forward_batch, physical_tokens, *args, **kwargs)
    lookup_ids, semantic_tokens = ngram._prepare_embedding_lookup(
        ngram_ids, forward_batch, physical_tokens
    )
    flat = lookup_ids.reshape(-1).to(torch.int32)
    rows = torch.index_select(weight.view(torch.uint8), 0, flat)
    embeddings = rows.view(torch.float8_e4m3fn).to(emb.output_dtype or torch.bfloat16)
    embeddings = embeddings.reshape(*lookup_ids.shape, weight.shape[-1])
    embeddings = embeddings * emb.weight_scale
    return ngram._finish_embedding_lookup(
        embeddings, semantic_tokens, forward_batch, physical_tokens
    )


def _ngram_hash_limb_plan(ngram: Any) -> dict:
    """Host constants of the int32 n-gram hash, cached on the module.

    Reads the module's buffers on the host, so build it eagerly (after the
    weights load, see load_ple_table_on_host), never inside a trace.

    Products token * multiplier stay below 2^63 by construction (each
    multiplier is at most int64_max // vocab), so they are split into
    ``bits``-bit limbs small enough that token * limb + carry fits int32.
    """
    import math

    import torch

    plan = getattr(ngram, "_tpu_hash_plan", None)
    if plan is not None:
        return plan
    vocab = int(ngram.unigram_vocab_size)
    token_bits = max(1, math.ceil(math.log2(vocab + 1)))
    bits = 30 - token_bits
    if bits < 6:
        raise NotImplementedError(f"vocab {vocab} is too large for the int32 n-gram hash")
    multipliers = [int(m) for m in ngram.layer_multipliers.cpu().tolist()]
    num_m = max(1, math.ceil(max(m.bit_length() for m in multipliers) / bits))
    num_p = math.ceil(63 / bits) + 1
    mask = (1 << bits) - 1
    m_limbs = [[(m >> (bits * k)) & mask for k in range(num_m)] for m in multipliers]
    sizes = [int(v) for v in ngram.ngram_heads_vocab_sizes.cpu().tolist()]
    offsets = [int(o) for o in ngram.ngram_heads_offsets.cpu().tolist()]
    if max(sizes) >= 1 << 25 or max(o + s for o, s in zip(offsets, sizes)) >= 1 << 31:
        raise NotImplementedError("n-gram head vocab too large for the int32 n-gram hash")
    plan = dict(bits=bits, mask=mask, num_p=num_p, m_limbs=m_limbs)
    # Module buffers, so direct_compile lifts them as model inputs.
    device = ngram.layer_multipliers.device
    if hasattr(ngram, "register_buffer"):
        ngram.register_buffer(
            "_tpu_hash_sizes",
            torch.tensor(sizes, dtype=torch.int32, device=device),
            persistent=False,
        )
        ngram.register_buffer(
            "_tpu_hash_offsets",
            torch.tensor(offsets, dtype=torch.int32, device=device),
            persistent=False,
        )
    else:  # plain objects in unit tests
        ngram._tpu_hash_sizes = torch.tensor(sizes, dtype=torch.int32, device=device)
        ngram._tpu_hash_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    ngram._tpu_hash_plan = plan
    return plan


def ngram_hash_int32(ngram: Any, contexts: Any) -> Any:
    """Qwen4ExpNGramEmbedding._hash_contexts in int32 arithmetic only.

    Matches the int64 reference exactly: for n-gram order g the hash of a
    window is XOR_{k<g} tok_k * m_k (tok_k the token k positions before the
    window's last, or EOS when an EOS lies between), reduced mod each head's
    vocab size plus the head's offset.
    """
    import torch

    plan = _ngram_hash_limb_plan(ngram)
    device = contexts.device
    eos = int(ngram.eos_token_id)
    ctx = contexts.to(torch.int32)
    last = ctx.shape[1] - 1
    bits, mask, num_p = plan["bits"], plan["mask"], plan["num_p"]

    # tok_k: the token k back from the last, or EOS when an EOS lies in
    # positions last-k+1 .. last-1 (the reference's segment rule; an EOS at
    # last-k itself yields EOS either way).
    tokens = [ctx[:, last]]
    blocked = torch.zeros_like(ctx[:, last], dtype=torch.bool)
    for k in range(1, ngram.ngram_size):
        if k > 1:
            blocked = blocked | (ctx[:, last - k + 1] == eos)
        src = ctx[:, last - k]
        tokens.append(torch.where(blocked, torch.full_like(src, eos), src))

    def product_limbs(token, m_limbs):
        limbs, carry = [], torch.zeros_like(token)
        for k in range(num_p):
            acc = carry + (token * m_limbs[k] if k < len(m_limbs) else 0)
            limbs.append(acc & mask)
            carry = acc >> bits
        return limbs

    sizes = ngram._tpu_hash_sizes
    offsets = ngram._tpu_hash_offsets
    blocks = []
    mixed = None
    for pos in range(ngram.ngram_size):
        limbs = product_limbs(tokens[pos], plan["m_limbs"][pos])
        mixed = limbs if mixed is None else [a ^ b for a, b in zip(mixed, limbs)]
        if pos == 0:
            continue
        g = pos + 1
        start = (g - 2) * ngram.heads_per_ngram
        end = start + ngram.heads_per_ngram
        v = sizes[start:end].view(1, -1)
        r = torch.zeros((ctx.shape[0], end - start), dtype=torch.int32, device=device)
        # Horner over 6-bit chunks, most significant first: r < v < 2^25.
        chunks_per_limb = bits // 6 if bits % 6 == 0 else None
        for limb in reversed(mixed):
            if chunks_per_limb is not None:
                parts = [(limb >> (6 * c)) & 63 for c in reversed(range(chunks_per_limb))]
                widths = [6] * chunks_per_limb
            else:
                parts = [(limb >> b) & 1 for b in reversed(range(bits))]
                widths = [1] * bits
            for part, width in zip(parts, widths):
                r = torch.remainder((r << width) | part.view(-1, 1), v)
        blocks.append(r + offsets[start:end].view(1, -1))
    return torch.cat(blocks, dim=-1)


def hash_ngrams_int32(original: Any, ngram: Any, contexts: Any, *args, **kwargs):
    """AROUND Qwen4ExpNGramEmbedding._hash_contexts: the int32 hash on TPU.

    The int64 reference compiled for prefill but aborted the TPU compiler on
    decode-sized graphs (lowering_util.cc 'Unsupported conversion from
    vmreg/vreg to U64'); int32 limbs avoid 64-bit vector values entirely.
    """
    return ngram_hash_int32(ngram, contexts)
