# SPDX-License-Identifier: Apache-2.0
"""FlyWheel attention backend (``--attention-backend flywheel``).

Runs the flywheel-tpu kernels (https://github.com/heavyball-research/flywheel-tpu)
through ``sglang_tpu.attention.flywheel_attention``, on RPA v3's shared base:
the same fused KV pool, page table and ``distribution``, with its own custom
op. In one step:

* a DMA writer copies the K/V of the extend rows into the pages;
* the decode kernel appends and attends the decode rows;
* ``lax.switch`` runs the kernel the host picked on the extend rows: paged
  extend, dense prefill, or ragged prefill at a block span.

The host builds the writer's page pieces and the kernel index per step
(``_plan_step``), so the compiled graph holds every kernel and never
recompiles for a new batch.

The fused pool, ``[num_blocks, page_size, align(2 * num_kv_heads, packing) /
packing, packing, head_dim]``, is flywheel's cache layout ``[num_blocks,
page_size, 2 * num_kv_heads, head_dim]`` with K and V interleaved per head, in
bf16 and fp8 alike, so the op reshapes it in place.

Neither registering the backend nor validating its server args imports
flywheel_tpu; constructing the backend does.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here (see
# attention_pallas_rpa_base for why torch_tpu's jax_op needs real annotations).

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang_tpu.attention_pallas_rpa_base import (
    BaseTpuPallasRpaAttnBackend,
    RpaKernelSpec,
    _get_pages_per_seq,
    _pool_kv_pack_heads,
)
from sglang_tpu.pallas_export import tpu_info_for_cpu_export

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

FLYWHEEL_BACKEND = "flywheel"

# flywheel_tpu's PAGE_SIZE_MULTIPLE: its kernels index the cache in whole
# 128-token pages. Restated so validating server args imports no flywheel_tpu.
FLYWHEEL_PAGE_SIZE_MULTIPLE = 128

# The head_dims flywheel_attention's decode kernel runs (its _DECODE_HEAD_DIMS).
FLYWHEEL_HEAD_DIMS = (128, 256)

_OP_NAME = "sglang_tpu_flywheel::flywheel_attention"
_flywheel_op = None


def _import_flywheel_attention():
    try:
        from sglang_tpu.attention.flywheel_attention import flywheel_tpu_attention
    except ModuleNotFoundError as e:
        if e.name != "flywheel_tpu":
            raise
        raise ImportError(
            "--attention-backend flywheel needs flywheel-tpu; install it with "
            "`pip install 'sglang[flywheel]'` from a TPU build."
        ) from e

    return flywheel_tpu_attention


def _invoke_from_shared_shell(*args, **kwargs):
    raise NotImplementedError(
        "The flywheel backend takes per-step operands the shared RPA op shell "
        "does not pass; it builds its own op (TpuFlywheelAttnBackend._build_kernel_op)."
    )


FLYWHEEL_SPEC = RpaKernelSpec(
    op_name=_OP_NAME,
    import_kernel=_import_flywheel_attention,
    invoke_kernel=_invoke_from_shared_shell,
    supports_attention_sink=False,
    uses_cu_kv_lens=False,
    # Decode rows first: distribution is (decode_end, prefill_end, mixed_end).
    is_batched=False,
    min_page_size=FLYWHEEL_PAGE_SIZE_MULTIPLE,
)


class FlywheelStepOperands(NamedTuple):
    """One step's flywheel operands on the device, shared by every layer."""

    # i32[3 * bound]: the KV writer's (cache_row_start, new_row_start, length)
    # pieces, and i32[1] their count.
    kv_write_slices: torch.Tensor
    kv_write_num_slices: torch.Tensor
    # i32[1]: the step's index into flywheel_attention.step_kernels(num_tokens).
    kernel_index: torch.Tensor


def fused_cache_flywheel_attention(
    queries,  # [T, num_q_heads, head_dim]
    keys,  # [T, num_kv_heads, head_dim]
    values,
    kv_cache_fused,  # [num_blocks, page_size, num_kv_heads, 2, head_dim]
    kv_lens,  # i32[max_num_seqs]
    page_indices,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens,  # i32[max_num_seqs + 1]
    distribution,  # i32[3]: (decode_end, prefill_end, mixed_end)
    kv_write_slices,  # i32[3 * bound]
    kv_write_num_slices,  # i32[1]
    kernel_index,  # i32[1]
    *,
    sm_scale: float,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
    """flywheel_tpu_attention on the fused pool's buffer, in JAX.

    Returns the output rows and the updated buffer in the pool's shape. An
    fp8 pool takes K/V already quantized by the caller, and k_scale/v_scale
    dequantize them.
    """
    flywheel_tpu_attention = _import_flywheel_attention()
    num_blocks, page_size = kv_cache_fused.shape[:2]
    head_dim = kv_cache_fused.shape[-1]
    output, kv_cache = flywheel_tpu_attention(
        queries,
        keys,
        values,
        kv_cache_fused.reshape(num_blocks, page_size, -1, head_dim),
        kv_lens,
        page_indices,
        cu_q_lens,
        kv_write_slices,
        kv_write_num_slices,
        distribution,
        kernel_index,
        sm_scale=sm_scale,
        k_descale=k_scale,
        v_descale=v_scale,
    )
    return output, kv_cache.reshape(kv_cache_fused.shape)


def plan_flywheel_step(
    *,
    is_decode: bool,
    q_lens: np.ndarray,  # i32[live]
    kv_lens: np.ndarray,  # i32[live], this step's tokens included
    block_table: np.ndarray,  # i32[live, pages_per_seq]
    num_tokens: int,
    max_num_seqs: int,
    page_size: int,
) -> tuple[np.ndarray, int, int]:
    """One step's KV write pieces (i32[3, bound]), their count and kernel index.

    The bound depends on the token and batch buckets only, so a compiled step
    sees the same shapes for every batch. A decode step writes nothing: the
    decode kernel appends its rows. An extend step has no decode rows, so the
    writer takes every live row.
    """
    from sglang_tpu.attention import flywheel_attention as fw

    slices = np.zeros(
        (3, fw.kv_write_slice_bound(num_tokens, max_num_seqs, page_size)), np.int32
    )
    if is_decode or len(q_lens) == 0:
        return slices, 0, 0  # step_kernels' first entry: the decode step
    cu_q_lens = np.zeros(len(q_lens) + 1, np.int32)
    np.cumsum(q_lens, out=cu_q_lens[1:])
    num_slices = fw.build_kv_write_slices(
        cu_q_lens, kv_lens, block_table, page_size=page_size, out=slices
    )
    kernel_index = fw.step_kernel_index(0, cu_q_lens, kv_lens, num_tokens)
    return slices, num_slices, kernel_index


def _get_flywheel_op():
    """Register fused_cache_flywheel_attention as a torch custom op, once per process."""
    global _flywheel_op
    if _flywheel_op is not None:
        return _flywheel_op

    import jax
    from torch_tpu._internal import pallas as _torch_pallas

    def _wrapped(
        queries: jax.Array,
        keys: jax.Array,
        values: jax.Array,
        kv_cache_fused: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        kv_write_slices: jax.Array,
        kv_write_num_slices: jax.Array,
        kernel_index: jax.Array,
        sm_scale: float = 1.0,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        return fused_cache_flywheel_attention(
            queries,
            keys,
            values,
            kv_cache_fused,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            kv_write_slices,
            kv_write_num_slices,
            kernel_index,
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{_OP_NAME}",
        static_argnums=(11, 12, 13),  # sm_scale, k_scale, v_scale
        donate_argnums=(3,),
        input_output_aliases={3: 1},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        _OP_NAME, mutates_args=("kv_cache_fused",), device_types=["tpu"]
    )
    def flywheel_op(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        kv_cache_fused: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        kv_write_slices: torch.Tensor,
        kv_write_num_slices: torch.Tensor,
        kernel_index: torch.Tensor,
        sm_scale: float,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ) -> torch.Tensor:
        with tpu_info_for_cpu_export():
            output, _ = jax_kernel(
                queries,
                keys,
                values,
                kv_cache_fused,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                kv_write_slices,
                kv_write_num_slices,
                kernel_index,
                sm_scale,
                k_scale,
                v_scale,
            )
        return output

    @flywheel_op.register_fake
    def _(queries, *args, **kwargs):
        return torch.empty_like(queries)

    _flywheel_op = flywheel_op
    return flywheel_op


_KV_CACHE_DTYPES = ("auto", "bfloat16", "bf16", "fp8_e4m3", "fp8_e5m2")


def validate_flywheel_server_args(cfg) -> None:
    """Reject, in one error, the server configurations the backend cannot serve.

    ``cfg`` is the resolving view of the server args. Model properties
    (MLA, head_dim, hybrid layers) are checked when the backend is built.
    """
    from sglang.srt.arg_groups.overrides import attention_backends_of

    problems = []
    backends = set(attention_backends_of(cfg))
    if backends != {FLYWHEEL_BACKEND}:
        problems.append(
            "the prefill and decode attention backends must both be flywheel, "
            f"got {sorted(backends)}"
        )
    if cfg.page_size % FLYWHEEL_PAGE_SIZE_MULTIPLE:
        problems.append(
            f"--page-size {cfg.page_size} is not a multiple of "
            f"{FLYWHEEL_PAGE_SIZE_MULTIPLE}"
        )
    if cfg.kv_cache_dtype not in _KV_CACHE_DTYPES:
        problems.append(
            f"--kv-cache-dtype {cfg.kv_cache_dtype} is not one of {_KV_CACHE_DTYPES}"
        )
    if cfg.dtype not in ("auto", "bfloat16", "bf16"):
        problems.append(f"--dtype {cfg.dtype} is not bfloat16")
    unsupported = {
        "speculative decoding": cfg.speculative_algorithm is not None,
        "P/D disaggregation": cfg.disaggregation_mode != "null",
        "pipeline parallelism": cfg.pp_size > 1,
        "decode context parallelism": cfg.dcp_size > 1,
        "prefill context parallelism": cfg.attn_cp_size > 1,
        "DP attention": cfg.attn_dp_size > 1,
        # A mixed batch puts its decode rows last; the decode kernel takes them first.
        "--enable-mixed-chunk": cfg.enable_mixed_chunk,
        "hierarchical cache": cfg.enable_hierarchical_cache,
        "--enable-unified-memory": cfg.enable_unified_memory,
    }
    problems += [f"{name} is not supported" for name, bad in unsupported.items() if bad]
    if problems:
        raise ValueError(
            "--attention-backend flywheel cannot serve this configuration: "
            + "; ".join(problems)
            + "."
        )


class TpuFlywheelAttnBackend(BaseTpuPallasRpaAttnBackend):
    """FlyWheel attention backend (``--attention-backend flywheel``)."""

    kernel_spec = FLYWHEEL_SPEC

    def __init__(
        self, model_runner: "ModelRunner", speculative_step_id: int = 0
    ) -> None:
        _check_model(model_runner)
        super().__init__(model_runner, speculative_step_id)
        if _pool_kv_pack_heads(self.token_to_kv_pool) > 1:
            raise ValueError(
                "--attention-backend flywheel does not support a KV pool that "
                "packs heads (SGLANG_TPU_KV_PACK_HEADS)."
            )
        from sglang_tpu.attention import flywheel_attention

        self._fw = flywheel_attention
        self._step_operands: FlywheelStepOperands | None = None

    def _build_kernel_op(self):
        op = _get_flywheel_op()

        # The base forward's call to the shared RPA op shell, with this step's
        # flywheel operands appended. Features the kernels lack fail at trace.
        def run(
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
            decode_query_size,
            *,
            use_causal_mask,
            layer_idx,
        ):
            self._fw.check_supported(
                queries,
                sinks=attention_sink,
                attention_chunk_size=sliding_window,
                attn_logits_soft_cap=soft_cap,
                q_scale=None,
                k_scale=k_scale,
                v_scale=v_scale,
                update_kv_cache=True,
                use_causal_mask=use_causal_mask,
            )
            if layer_idx is not None:
                raise NotImplementedError(
                    "The flywheel attention backend does not support block-major KV."
                )
            step = self._step_operands
            return op(
                queries,
                keys,
                values,
                kv_cache_fused,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                step.kv_write_slices,
                step.kv_write_num_slices,
                step.kernel_index,
                sm_scale,
                k_scale,
                v_scale,
            )

        return run

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_mixed():
            raise NotImplementedError(
                "The flywheel attention backend does not serve mixed batches: "
                "their decode rows come last, and its decode kernel takes them first."
            )
        super().init_forward_metadata(forward_batch)
        self._step_operands = self._plan_step(forward_batch)

    def _plan_step(self, forward_batch: ForwardBatch) -> FlywheelStepOperands:
        """This step's flywheel operands, planned on the host and uploaded."""
        bs = int(forward_batch.seq_lens.shape[0])
        live = self._non_empty_batch_size(forward_batch)
        is_decode = forward_batch.forward_mode.is_decode()
        if is_decode or live == 0:
            q_lens = kv_lens = np.zeros(0, np.int32)
            block_table = np.zeros((0, 0), np.int32)
        else:
            q_lens = np.asarray(
                self._extend_query_lens(forward_batch, bs)[:live], np.int32
            )
            kv_lens = np.asarray(forward_batch.seq_lens_cpu[:live], np.int32)
            block_table = self._host_block_table(forward_batch, live)
        slices, num_slices, kernel_index = plan_flywheel_step(
            is_decode=is_decode,
            q_lens=q_lens,
            kv_lens=kv_lens,
            block_table=block_table,
            # The token bucket: positions is [T], or [3, T] under M-RoPE.
            num_tokens=forward_batch.positions.shape[-1],
            max_num_seqs=bs,
            page_size=self.page_size,
        )
        device = self.forward_metadata.kv_lens.device
        return FlywheelStepOperands(
            kv_write_slices=self._value_metadata(
                name="flywheel_kv_write_slices",
                value=torch.from_numpy(slices.reshape(-1)),
                device=device,
            ),
            kv_write_num_slices=self._value_metadata(
                name="flywheel_kv_write_num_slices",
                value=torch.tensor([num_slices], dtype=torch.int32),
                device=device,
            ),
            kernel_index=self._value_metadata(
                name="flywheel_kernel_index",
                value=torch.tensor([kernel_index], dtype=torch.int32),
                device=device,
            ),
        )

    def _host_block_table(self, forward_batch: ForwardBatch, live: int) -> np.ndarray:
        """i32[live, pages_per_seq]: the live requests' page ids, on the host."""
        pages_per_seq = _get_pages_per_seq(self.max_context_len, self.page_size)
        page_cols = self._host_page_columns(pages_per_seq, self.page_size)
        rows = forward_batch.req_pool_indices_cpu[:live].to(torch.int64)
        slots = self.req_to_token_pool.req_to_token[rows[:, None], page_cols[None, :]]
        return (slots // self.page_size).to(torch.int32).numpy()


def _check_model(model_runner: "ModelRunner") -> None:
    """Reject the model features the flywheel kernels lack, in one error."""
    model_config = model_runner.model_config
    problems = []
    if model_runner.use_mla_backend:
        problems.append("MLA models are not supported")
    if model_config.head_dim not in FLYWHEEL_HEAD_DIMS:
        problems.append(
            f"head_dim {model_config.head_dim} is not one of {FLYWHEEL_HEAD_DIMS}"
        )
    if model_config.is_hybrid_swa:
        problems.append("hybrid sliding-window models are not supported")
    if mambaish_config(model_config) is not None:
        problems.append("hybrid linear-attention models are not supported")
    if problems:
        raise ValueError(
            "--attention-backend flywheel cannot serve this model: "
            + "; ".join(problems)
            + "."
        )
