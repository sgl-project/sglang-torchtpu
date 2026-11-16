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
