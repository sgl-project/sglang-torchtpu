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
