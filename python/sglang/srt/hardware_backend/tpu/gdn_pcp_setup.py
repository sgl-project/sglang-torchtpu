# SPDX-License-Identifier: Apache-2.0
"""Construct the context-parallel Gated-DeltaNet backend and its state pool.

Kept out of the attention registry so the registry only has to ask one
question -- "is this configuration context-parallel GDN?" -- and gets either a
fully wired backend or None.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.hardware_backend.tpu._logging import get_logger

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = get_logger(__name__)


def pcp_gdn_active() -> bool:
    """True when linear-attention layers must run the context-parallel path."""
    from sglang.srt.layers.cp.base import is_pcp
    from sglang.srt.layers.cp.utils import enable_cp_v2
    from sglang.srt.runtime_context import get_parallel

    if not (enable_cp_v2() and is_pcp()):
        return False
    return int(get_parallel().attn_cp_size) > 1


def _release_dense_state_buffers(mamba_pool) -> None:
    """Drop the dense conv/recurrent storage the pooled path replaces.

    The memory solver already charged for these, and the fused kernel reads
    and writes the pooled buffer instead, so leaving them allocated doubles
    the GDN state footprint for nothing. Slot bookkeeping lives in the
    allocator and does not need the storage; every remaining reader of these
    tensors is unreachable on this path (the dense backend raises, radix
    copy-on-write raises, clear_slots is a no-op, and PD transfer is refused).
    """
    cache = getattr(mamba_pool, "mamba_cache", None)
    if cache is None:
        return
    freed = 0
    tensors = list(cache.conv) + [cache.temporal]
    for t in tensors:
        freed += t.numel() * t.element_size()
        t.resize_(0)
    logger.info(
        "Released %.3f GB of dense GDN state; the context-parallel path uses "
        "the pooled buffer.",
        freed / (1024**3),
    )


def build_pcp_gdn_backend(runner: ModelRunner) -> Optional[object]:
    """Build the context-parallel GDN backend, or None if it does not apply.

    Allocates the byte-addressed state pool alongside it: the fused kernel
    reads and writes state through one indexed buffer rather than the dense
    pool's two tensors, and the pool has to exist before the first forward.
    """
    if not pcp_gdn_active():
        return None

    from sglang.srt.configs.hybrid_arch import mambaish_config
    from sglang.srt.hardware_backend.tpu.attention_gdn_pallas_pcp_backend import (
        TpuGDNPcpAttnBackend,
    )
    from sglang.srt.hardware_backend.tpu.mamba_pool_pooled import TpuPooledGDNState
    from sglang.srt.layers.cp.pcp_sequence_layout import resolve_pcp_interleave_size
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.server_args import get_global_server_args

    cfg = mambaish_config(runner.model_config)
    if cfg is None:
        raise RuntimeError(
            "Context-parallel GDN was selected for a model with no linear "
            "attention layers."
        )

    parallel = get_parallel()
    pcp_size = int(parallel.attn_cp_size)
    server_args = get_global_server_args()
    interleave_size = resolve_pcp_interleave_size(int(server_args.page_size))

    # cache_params.shape is already divided by attn_tp_size * attn_cp_size
    # (see configs/qwen3_next.py), so these are this rank's head shard.
    shape = cfg.mamba2_cache_params.shape
    n_v_local, d_v, d_k = shape.temporal
    conv_dim_local = shape.conv[0][0]
    kernel_size = shape.conv[0][1] + 1

    mamba_pool = runner.req_to_token_pool.mamba_pool
    pooled = TpuPooledGDNState(
        size=mamba_pool.size,
        mamba_layer_ids=list(cfg.mamba2_cache_params.layers),
        n_v=n_v_local,
        conv_dim=conv_dim_local,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        device=str(runner.device),
    )

    _release_dense_state_buffers(mamba_pool)

    backend = TpuGDNPcpAttnBackend(
        runner, pcp_size=pcp_size, interleave_size=interleave_size
    )
    backend.attach_pooled_state(pooled)
    backend.attach_to_model(runner.model)
    logger.info(
        "Context-parallel GDN backend built: pcp_size=%d, interleave=%d, "
        "local heads n_v=%d conv_dim=%d, pool block tokens=%d.",
        pcp_size,
        interleave_size,
        n_v_local,
        conv_dim_local,
        pooled.pool_block_tokens,
    )
    return backend
