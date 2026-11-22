# SPDX-License-Identifier: Apache-2.0
"""Torch custom-op wrapper for the distributed TPU TopK Pallas kernel."""

import functools
from collections.abc import Callable
from threading import Lock

import torch

from sglang_tpu.kernels.distributed_topk import (
    SUPPORTED_WORLD_SIZES,
)
from sglang_tpu.kernels.distributed_topk import (
    distributed_top_k as distributed_top_k_kernel,
)

_OPS: dict[int, Callable] = {}
_OPS_LOCK = Lock()


@functools.cache
def _get_mesh(world_size: int):
    from torch_tpu._internal.pallas import get_pallas_mesh

    return get_pallas_mesh(("model",), (world_size,))


def _build_op(world_size: int) -> Callable:
    import jax
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    mesh = _get_mesh(world_size)

    def _local(
        local_logits: jax.Array,
        org_vocab_size: int,
        global_vocab_size: int,
        org_vocab_padded_count: int,
        k: int,
    ) -> tuple[jax.Array, jax.Array]:
        return distributed_top_k_kernel(
            local_logits,
            world_size=world_size,
            org_vocab_size=org_vocab_size,
            global_vocab_size=global_vocab_size,
            org_vocab_padded_count=org_vocab_padded_count,
            k=k,
        )

    sharded = jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(P(None, "model"), None, None, None, None),
        out_specs=(P(), P()),
        check_vma=False,
    )

    def _wrapped(
        local_logits: jax.Array,
        org_vocab_size: int,
        global_vocab_size: int,
        org_vocab_padded_count: int,
        k: int,
    ) -> tuple[jax.Array, jax.Array]:
        return sharded(
            local_logits,
            org_vocab_size,
            global_vocab_size,
            org_vocab_padded_count,
            k,
        )

    return cpu_export_jax_op(
        f"sglang_tpu_pallas::distributed_top_k_tp{world_size}",
        _wrapped,
        mesh=mesh,
        input_partition_specs=(
            P(None, "model"),
            None,
            None,
            None,
            None,
        ),
        output_partition_specs=(P(), P()),
    )


def get_pallas_distributed_top_k_op(world_size: int) -> Callable:
    """Return the lazily registered custom op for one supported topology."""

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Pallas distributed TopK does not support TP{world_size}")
    op = _OPS.get(world_size)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = _OPS.get(world_size)
        if op is None:
            op = _build_op(world_size)
            _OPS[world_size] = op
    return op


def register_pallas_distributed_top_k_op(world_size: int) -> None:
    """Register the current TP8/TP16 topology before full-graph tracing."""

    if world_size in SUPPORTED_WORLD_SIZES:
        get_pallas_distributed_top_k_op(world_size)
        from sglang_tpu.envs import envs as tpu_envs

        if tpu_envs.SGLANG_TPU_MUSE_FUSED_FP8_LM_HEAD_TOPK.get():
            from sglang_tpu.kernels.fp8_lm_head_fused import (
                get_fp8_lm_head_distributed_topk_op,
                get_fp8_lm_head_gumbel_max_op,
            )

            get_fp8_lm_head_distributed_topk_op(world_size)
            gumbel_env = getattr(
                tpu_envs, "SGLANG_TPU_MUSE_FUSED_GUMBEL_MAX_LM_HEAD", None
            )
            if gumbel_env is not None and gumbel_env.get():
                get_fp8_lm_head_gumbel_max_op(world_size)
        if tpu_envs.SGLANG_TPU_MUSE_FAST_DISTRIBUTED_TOPK.get():
            from sglang_tpu.kernels.fp8_lm_head_fused import (
                get_lossless_distributed_topk_vmem_ag_op,
            )

            get_lossless_distributed_topk_vmem_ag_op(world_size)


def pallas_distributed_top_k(
    local_logits: torch.Tensor,
    *,
    org_vocab_size: int,
    global_vocab_size: int,
    org_vocab_padded_count: int,
    k: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact global TopK using the native mesh for vocabulary ownership."""

    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.kernels.distributed_topk import MAX_DISTRIBUTED_TOP_K

    rows = int(local_logits.shape[0])
    min_fast_rows = 128 if world_size == 8 else 64
    b384_env = getattr(tpu_envs, "SGLANG_TPU_MUSE_DECODE_B384", None)
    lift_env = getattr(tpu_envs, "SGLANG_TPU_MUSE_VERIFY_BRIDGE_LIFT", None)
    max_fast_rows = (
        384
        if (
            (b384_env is not None and b384_env.get())
            or (lift_env is not None and lift_env.get())
        )
        else 256
    )
    if (
        tpu_envs.SGLANG_TPU_MUSE_FAST_DISTRIBUTED_TOPK.get()
        and min_fast_rows <= rows <= max_fast_rows
        and 1 <= k <= MAX_DISTRIBUTED_TOP_K
    ):
        from sglang_tpu.kernels.fp8_lm_head_fused import (
            get_lossless_distributed_topk_vmem_ag_op,
        )

        values, token_ids = get_lossless_distributed_topk_vmem_ag_op(world_size)(
            local_logits,
            org_vocab_size,
            global_vocab_size,
            org_vocab_padded_count,
            k,
        )
        return values.float(), token_ids

    values, token_ids = get_pallas_distributed_top_k_op(world_size)(
        local_logits,
        org_vocab_size,
        global_vocab_size,
        org_vocab_padded_count,
        k,
    )
    return values.float(), token_ids


def fused_fp8_lm_head_distributed_top_k(
    hidden_states: torch.Tensor,
    lm_head,
    *,
    top_k: int,
    logit_scale: float | None,
    group,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Single-kernel fused FP8 LM head + exact global Top-K (``1 <= k <= 64``).

    Returns ``None`` whenever any precondition fails so callers seamlessly fall back
    to the unfused LM head + distributed Top-K path.
    """

    from sglang_tpu.kernels.distributed_topk import MAX_DISTRIBUTED_TOP_K
    from sglang_tpu.top_k import (
        TpuVocabShard,
        _group_spans_world,
        _is_canonical_pallas_shard,
        _pallas_shards_cover_vocab,
    )

    if (
        hidden_states.device.type != "tpu"
        or hidden_states.ndim != 2
        or not (0 < top_k <= MAX_DISTRIBUTED_TOP_K)
        or (hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"))
        or getattr(lm_head, "bias", None) is not None
    ):
        return None
    w8k = getattr(lm_head, "fp8_w8k", None)
    scale = getattr(lm_head, "fp8_scale", None)
    if w8k is None or scale is None:
        return None
    eff_scale = 1.0 if logit_scale is None else float(logit_scale)
    if eff_scale <= 0.0:
        return None
    world_size = int(getattr(group, "world_size", 0))
    model_rank = getattr(group, "rank_in_group", None)
    if (
        world_size not in SUPPORTED_WORLD_SIZES
        or not isinstance(model_rank, int)
        or not _group_spans_world(group)
    ):
        return None
    try:
        shard = TpuVocabShard.from_lm_head(lm_head)
    except (AttributeError, TypeError, ValueError):
        return None
    if (
        not _is_canonical_pallas_shard(shard, model_rank, world_size)
        or not _pallas_shards_cover_vocab(shard, world_size)
        or shard.width < top_k
        or w8k.shape[0] != hidden_states.shape[-1]
        or w8k.shape[1] < shard.width
    ):
        return None

    from sglang_tpu.envs import envs as tpu_envs

    b384_env = getattr(tpu_envs, "SGLANG_TPU_MUSE_DECODE_B384", None)
    lift_env = getattr(tpu_envs, "SGLANG_TPU_MUSE_VERIFY_BRIDGE_LIFT", None)
    max_fused_rows = (
        384
        if (
            (b384_env is not None and b384_env.get())
            or (lift_env is not None and lift_env.get())
        )
        else 256
    )

    rows = int(hidden_states.shape[0])
    if rows == 0:
        return (
            hidden_states.new_empty((0, top_k), dtype=torch.float32),
            hidden_states.new_empty((0, top_k), dtype=torch.int32),
        )
    if rows > max_fused_rows:
        return None

    from sglang_tpu.kernels.fp8_lm_head_fused import (
        get_fp8_lm_head_distributed_topk_op,
    )

    x_in = hidden_states
    pad_bucket = rows
    if not torch.compiler.is_compiling():
        from sglang_tpu.row_pad import pad_rows, row_bucket

        cand_bucket = 384 if (256 < rows <= 384 and max_fused_rows >= 384) else row_bucket(rows)
        if cand_bucket != rows and cand_bucket <= max_fused_rows:
            x_in = pad_rows(hidden_states, cand_bucket)
            pad_bucket = cand_bucket

    values, token_ids = get_fp8_lm_head_distributed_topk_op(world_size)(
        x_in,
        w8k,
        scale,
        shard.width,
        shard.org_vocab_size,
        shard.global_vocab_size,
        shard.org_vocab_padded_count,
        top_k,
        eff_scale,
    )
    if pad_bucket != rows:
        values = values[:rows]
        token_ids = token_ids[:rows]
    return values.float(), token_ids


def fused_fp8_lm_head_gumbel_max(
    hidden_states: torch.Tensor,
    lm_head,
    temperatures: torch.Tensor,
    *,
    logit_scale: float | None,
    group,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """1-kernel fused FP8 LM head + on-device Gumbel-max categorical draw."""
    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.top_k import (
        TpuVocabShard,
        _group_spans_world,
        _is_canonical_pallas_shard,
        _pallas_shards_cover_vocab,
    )

    if (
        hidden_states.device.type != "tpu"
        or hidden_states.ndim != 2
        or temperatures is None
        or (hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"))
        or getattr(lm_head, "bias", None) is not None
    ):
        return None
    w8k = getattr(lm_head, "fp8_w8k", None)
    scale = getattr(lm_head, "fp8_scale", None)
    if w8k is None or scale is None:
        return None
    eff_scale = 1.0 if logit_scale is None else float(logit_scale)
    if eff_scale <= 0.0:
        return None
    world_size = int(getattr(group, "world_size", 0))
    model_rank = getattr(group, "rank_in_group", None)
    if (
        world_size not in SUPPORTED_WORLD_SIZES
        or not isinstance(model_rank, int)
        or not _group_spans_world(group)
    ):
        return None
    try:
        shard = TpuVocabShard.from_lm_head(lm_head)
    except (AttributeError, TypeError, ValueError):
        return None
    if (
        not _is_canonical_pallas_shard(shard, model_rank, world_size)
        or not _pallas_shards_cover_vocab(shard, world_size)
        or w8k.shape[0] != hidden_states.shape[-1]
        or w8k.shape[1] < shard.width
    ):
        return None

    b384_env = getattr(tpu_envs, "SGLANG_TPU_MUSE_DECODE_B384", None)
    lift_env = getattr(tpu_envs, "SGLANG_TPU_MUSE_VERIFY_BRIDGE_LIFT", None)
    max_fused_rows = (
        384
        if (
            (b384_env is not None and b384_env.get())
            or (lift_env is not None and lift_env.get())
        )
        else 256
    )
    rows = int(hidden_states.shape[0])
    if rows == 0:
        return (
            hidden_states.new_empty((0, 1), dtype=torch.float32),
            hidden_states.new_empty((0, 1), dtype=torch.int32),
        )
    if rows > max_fused_rows or int(temperatures.shape[0]) < rows:
        return None

    from sglang_tpu.kernels.fp8_lm_head_fused import get_fp8_lm_head_gumbel_max_op

    t_in = temperatures[:rows].reshape(rows, 1).float()
    x_in = hidden_states
    pad_bucket = rows
    if not torch.compiler.is_compiling():
        from sglang_tpu.row_pad import pad_rows, row_bucket

        cand_bucket = 384 if (256 < rows <= 384 and max_fused_rows >= 384) else row_bucket(rows)
        if cand_bucket != rows and cand_bucket <= max_fused_rows:
            x_in = pad_rows(hidden_states, cand_bucket)
            t_in = torch.nn.functional.pad(t_in, (0, 0, 0, cand_bucket - rows), value=1.0)
            pad_bucket = cand_bucket

    rng_seed = torch.rand((1, 128), dtype=torch.float32, device=hidden_states.device)
    values, token_ids = get_fp8_lm_head_gumbel_max_op(world_size)(
        x_in,
        w8k,
        scale,
        t_in,
        rng_seed,
        shard.width,
        shard.org_vocab_size,
        shard.global_vocab_size,
        shard.org_vocab_padded_count,
        eff_scale,
    )
    if pad_bucket != rows:
        values = values[:rows]
        token_ids = token_ids[:rows]
    return values.float(), token_ids
