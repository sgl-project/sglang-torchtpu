# SPDX-License-Identifier: Apache-2.0
"""Batched Pallas ragged-paged-attention backend for the TPU torch_tpu platform.

The kernel reads either of two KV cache layouts, chosen once here from
``SGLANG_TPU_USE_SEQ_ALONG_LANE_KV``:

``HEAD_ALONG_SUBLANE`` (default)
    Token-major fused 5D buffer, ``num_kv_heads * 2`` padded up to
    ``kv_packing``.
``SEQ_ALONG_LANE``
    Token index on the lane axis and ``num_kv_heads * 2`` unpadded, so at one
    KV head per rank an fp8 cache is 256 bytes per token instead of 512. That
    halving is what lets long context and large batches fit; it needs
    ``TpuSeqAlongLaneKVPool`` (selected by the platform from the same flag) and
    a page size of at least 256.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here (see
# attention_pallas_rpa_base for why torch_tpu's jax_op needs real annotations).

from sglang_tpu.attention_pallas_rpa_base import (
    BaseTpuPallasRpaAttnBackend,
    PallasForwardMetadata,
    RpaKernelSpec,
    _resolve_decode_query_size,
)
from sglang_tpu.cp_forward import get_tpu_cp_layout
from sglang_tpu.envs import envs

__all__ = [
    "PallasForwardMetadata",
    "TpuPallasRpaBatchedAttnBackend",
    "kv_layout",
    "uses_seq_along_lane_kv",
]


def uses_seq_along_lane_kv() -> bool:
    """Whether the KV cache is in the seq-along-lane layout.

    The kernel, the pool that allocates the buffer and the page-size floor all
    have to agree, so they all ask here instead of reading the flag separately.
    """
    return envs.SGLANG_TPU_USE_SEQ_ALONG_LANE_KV.get()


def kv_layout():
    """The layout as the kernel's own enum.

    Read at trace time, not per token, so a plain lookup is fine here.
    """
    from sglang_tpu.attention.pallas.batched_rpa import configs

    if uses_seq_along_lane_kv():
        return configs.KVLayout.SEQ_ALONG_LANE
    return configs.KVLayout.HEAD_ALONG_SUBLANE


def _import_batched_kernel():
    from sglang_tpu.attention.pallas.batched_rpa.wrapper import (
        ragged_paged_attention,
    )

    return ragged_paged_attention


def _invoke_batched_kernel(
    jax_rpa,
    *,
    queries,
    keys,
    values,
    kv_cache_fused,
    kv_lens,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    distribution,
    attention_sink,
    sm_scale,
    sliding_window,
    soft_cap,
    k_scale,
    v_scale,
    decode_query_size=1,
    use_causal_mask=True,
):
    if not use_causal_mask:
        raise NotImplementedError(
            "pallas_tpu_rpa_batched supports only causal attention; "
            "use pallas_tpu_rpa_v3."
        )
    # cu_kv_lens is not consumed by the batched kernel.
    # It builds its schedule inline and has no per-mode dispatch to skip, so
    # the argument the shared shell passes for kernels that do is
    # discarded here.
    del cu_kv_lens

    def run(kv_cache, dist, prefill_block_sizes=None):
        return jax_rpa(
            queries,
            keys,
            values,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            dist,
            attention_sink=attention_sink,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            k_scale=k_scale,
            v_scale=v_scale,
            use_causal_mask=True,
            # The kernel owns the KV write; the backend never writes out of band.
            skip_kv_update=False,
            kv_layout=kv_layout(),
            prefill_block_sizes=prefill_block_sizes,
        )

    if decode_query_size == 1:
        return run(kv_cache_fused, distribution)
    # The DECODE range [0, distribution[0]) holds the width-wide sequences (a verify
    # batch, or a mixed-verify extend's tails, ordered first). The kernel's DECODE pass
    # takes one row per sequence, so they run as MIXED with verify-sized q blocks; the
    # remaining sequences then run as MIXED with the prefill blocks.
    import jax.numpy as jnp

    out_verify, kv_cache_fused = run(
        kv_cache_fused,
        jnp.zeros_like(distribution).at[2].set(distribution[0]),
        _verify_blocks(
            decode_query_size=decode_query_size, sliding_window=sliding_window
        ),
    )
    out_rest, kv_cache_fused = run(kv_cache_fused, distribution.at[0].set(0))
    verify_rows = jnp.arange(queries.shape[0]) < cu_q_lens[distribution[0]]
    return jnp.where(verify_rows[:, None, None], out_verify, out_rest), kv_cache_fused


# k block for verify-as-MIXED, measured on v7 at head_dim 64, 2 KV heads, fp8, 16-32 x 8 rows
# at 150K: 4096 beats 2048 by 10% and 8192 is no better; SWA layers stop at the window.
_VERIFY_BKV = 4096


def _verify_blocks(*, decode_query_size: int, sliding_window: int | None):
    """MIXED-mode blocks for width-wide verify sequences: a q block of that width (the
    default block search sizes q blocks for prefill chunks)."""
    from sglang_tpu.attention.pallas.batched_rpa import configs

    page = _MIN_PAGE_SIZE
    bkv = _VERIFY_BKV
    if sliding_window:
        bkv = min(bkv, max(page, -(-sliding_window // page) * page))
    # One lane: more lanes help at 16 sequences (-8%) but are 9x slower at 32.
    return configs.BlockSizes(
        bq_sz=decode_query_size,
        bq_c_sz=decode_query_size,
        bkv_sz=bkv,
        batch_size=1,
        n_buffer=3,
    )


# SEQ_ALONG_LANE tiles the page axis onto lanes, so a page has to be a multiple
# of the 128-lane width, and 256 is what the benchmarks run at. The
# token-major layout has no such constraint but keeps the same floor, because a
# smaller page turns each KV fetch into many tiny DMAs.
_MIN_PAGE_SIZE = 256

RPA_BATCHED_SPEC = RpaKernelSpec(
    op_name="sglang_tpu_pallas::ragged_paged_attention_batched",
    import_kernel=_import_batched_kernel,
    invoke_kernel=_invoke_batched_kernel,
    supports_attention_sink=True,
    uses_cu_kv_lens=False,
    extend_distribution_mask=(0, 0, 1),  # [0, 0, n] (MIXED-only)
    min_page_size=_MIN_PAGE_SIZE,
)


class TpuPallasRpaBatchedAttnBackend(BaseTpuPallasRpaAttnBackend):
    """Batched RPA attention backend (``--attention-backend pallas_tpu_rpa_batched``)."""

    kernel_spec = RPA_BATCHED_SPEC

    def __init__(self, model_runner, speculative_step_id=0):
        super().__init__(model_runner, speculative_step_id)
        from sglang.srt.managers.schedule_batch import mixed_tail_width
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        if (
            envs.SGLANG_TPU_RPA_BATCHED_VERIFY_BLOCKS.get()
            or mixed_tail_width(model_runner.spec_algorithm) > 1
        ):
            # Verify batches (and mixed-verify tails) then reach the kernel as
            # uniform-width DECODE, which _invoke_batched_kernel runs as MIXED with
            # verify-sized blocks.
            self.decode_query_size = _resolve_decode_query_size(
                supports_decode_query_size=True,
                spec_enabled=not model_runner.spec_algorithm.is_none(),
            )
        self._pp_schedule_runner = model_runner if parallel.pp_size > 1 else None
        self._cp_schedule_runner = model_runner if parallel.attn_cp_size > 1 else None

    def init_forward_metadata(self, forward_batch):
        cp_layout = get_tpu_cp_layout(forward_batch)
        if cp_layout is not None:
            from sglang_tpu.pp_attention_capacity import (
                check_cp_schedule,
            )

            # The kernel runs the context-parallel rows, not the global lengths.
            check_cp_schedule(self._cp_schedule_runner, self, cp_layout.host)
        elif self._pp_schedule_runner is not None:
            from sglang_tpu.pp_attention_capacity import (
                check_batch_schedule,
            )

            check_batch_schedule(self._pp_schedule_runner, self, forward_batch)
        super().init_forward_metadata(forward_batch)
