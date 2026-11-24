# SPDX-License-Identifier: Apache-2.0
"""Run MiniMax-H3's DiT block stack as one megakernel call per denoise step.

Runtime patch, tracked debt:
- Why: the torch_tpu block path runs each of the 50 blocks as several XLA
  programs with Ulysses all-to-alls between them; the Pallas megakernel
  (``kernel.stack``) runs every block of a step in one call, with the
  Ulysses exchange inside it.
- How: with ``SGLANG_TPU_ENABLE_H3_MEGAKERNEL=1``, :func:`install` wraps
  ``MiniMaxH3DiTModel.post_load_weights`` to decide, identically on every
  rank, whether the kernel can run (:func:`unsupported_reason`). If it can,
  the op is registered, the blocks' projections are packed into the kernel's
  layout (originals freed, staged through host memory) and ``model.blocks``
  becomes ``ModuleList([H3MegakernelStack])``, the seam Cache-DiT also uses:
  sglang's own forward then calls the stack once a step with the block
  keyword arguments. The stack builds the AdaLN tables from each block's
  resident ``adaln_proj`` on every call, the one-hot and RoPE tables from
  sglang's indices and RoPE cache, and passes the live length from
  ``cu_seqlens`` on device (one program for every prompt length of a padded
  row count). ``packed_sequence.minimax_h3_padded_seq_len`` pads the packed
  sequence to a row count the kernel tiles well (``tiling.padded_rows``).
  With ``SGLANG_TPU_OPT_H3_MEGAKERNEL_STATIC_STRUCTURE=1`` the host bounds
  also pick a static loop structure (``kernel.stack``'s
  ``static_structure``): one program per padded row count and loop
  structure, the generic walk at other row counts.
  Otherwise sglang's blocks run unchanged. Packed models refuse weight
  updates, LoRA, Cache-DiT, Spectrum and TeaCache. With
  ``SGLANG_TPU_OPT_H3_MEGAKERNEL_VOUCH=1`` the op is the vouched variant
  (``op.local_vouched_stack``): each step after a request's first also
  tries at a fixed max the heads the request's previous step's peaks vouch
  for and checks its own peaks on device, recomputing the step with the
  proven layers only if a tried head left the bound. Those heads live on
  the stack from the start to the end of one request's denoise loop. With
  both options the vouched op also takes the static structure and both its
  attempts run the structured program. With
  ``SGLANG_TPU_OPT_H3_MEGAKERNEL_SHARD_ADALN=1`` on every rank, each rank
  keeps the ``adaln_proj`` of a contiguous share of the blocks (the other
  blocks' weights are freed before packing); each call projects its share's
  tables and one all-gather over the sequence-parallel group (every rank)
  puts all of them in block order: the replicated path's tables bit for bit
  for an AdaLN input that is the same on every rank (checked on the device
  at startup, on a fixed input). With
  ``SGLANG_TPU_OPT_H3_MEGAKERNEL_ADALN_PRECOMPUTE=1`` the stack projects every
  step's AdaLN rows of a request in one GEMM a block when the denoise loop
  stages its timesteps (``MiniMaxH3DiTModel.prepare_adaln_plans``) and builds
  each step's tables from them (the step found by its timesteps tensor), if
  the first batch of each row count gave the per-step projections' bits. The
  kernel lever flags
  (SGLANG_TPU_OPT_H3_MEGAKERNEL_LEAN_CUT, _SPLIT_ITEMS, _EPILOGUE_CHUNKS,
  _WIDE_KV, _WIDE_KV_ONLINE) shape the structured calls at
  ``op.LEVER_SHAPES``' row counts.
- Upstream PR or issue: none.
- Remove when: the torch_tpu block path reaches the kernel's speed, or core
  grows a platform hook for the DiT block stack.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import math
import os
import re
import weakref

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from sglang_tpu.envs import envs
from sglang_tpu.multimodal_gen.h3_megakernel import pack

logger = logging.getLogger(__name__)

# model.__dict__ entry: the H3MegakernelStack once packed, None once refused.
_STATE = "_tpu_h3_megakernel"
# One registered op a process: a second registration of a name fails.
_OPS: dict = {}
# kernel.py's fixed-max softmax bound in log2 score units, and the slack for
# the bf16 roundings of q and k (_FIXED_BOUND, _BOUND_SLACK).
_FIXED_BOUND = 109.0
_BOUND_SLACK = 1.025
# The support vote of a rank that can run the kernel and asks for sharded
# AdaLN projections, in place of None (_ranks_agree_sharded).
_SHARD_VOTE = (
    "asks for sharded AdaLN projections (SGLANG_TPU_OPT_H3_MEGAKERNEL_SHARD_ADALN)"
)


def requested() -> bool:
    """The hook is installed for the kernel or for DiT dumps (debug)."""
    return (
        envs.SGLANG_TPU_ENABLE_H3_MEGAKERNEL.get()
        or envs.SGLANG_TPU_DEBUG_H3_DUMP_DIR.get() is not None
    )


def _dump_only() -> bool:
    """Dumps without the kernel: sglang's blocks run, the hook only saves."""
    return not envs.SGLANG_TPU_ENABLE_H3_MEGAKERNEL.get()


def _structured() -> bool:
    """The static-structure option: loops shaped by the host live length."""
    return envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_STATIC_STRUCTURE.get()


def _levers() -> frozenset[str]:
    """The kernel levers whose flags are set (``op.LEVER_SHAPES``' names);
    the structured calls take them."""
    flags = {
        "lean_cut": envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_LEAN_CUT,
        "split_items": envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_SPLIT_ITEMS,
        "epilogue_chunks": envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_EPILOGUE_CHUNKS,
        "wide_kv": envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_WIDE_KV,
        "wide_kv_online": envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_WIDE_KV_ONLINE,
        "modulate_pieces": envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_MODULATE_PIECES,
        "prefetch_down": envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_PREFETCH_DOWN,
        "uniform_adaln": envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_UNIFORM_ADALN,
        "zero_ahead": envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_ZERO_AHEAD,
        "mask_scores": envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_MASK_SCORES,
    }
    return frozenset(name for name, flag in flags.items() if flag.get())


def _tiling_overrides() -> tuple:
    """The structured calls' tiling bounds
    (SGLANG_TPU_OPT_H3_MEGAKERNEL_TILING, ``op.parse_tiling``)."""
    from sglang_tpu.multimodal_gen.h3_megakernel import op as stack_op

    return stack_op.parse_tiling(envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_TILING.get())


def static_structure(
    used: int,
    rows: int,
    devices: int,
    config,
    cut_kv_compute: int = 0,
    tiling=None,
) -> int:
    """The static structure the call passes for ``used`` live rows at
    ``rows`` a device on ``devices``: the first live length whose
    ``kernel.loop_structure`` (the call's tiling and ``cut_kv_compute``) is
    ``used``'s. Every live length of a structure passes the same value, so
    one program serves them all; the live length itself stays a traced
    operand (the masks). ``tiling``: the structured call's (default
    ``op.structured_tiling``'s)."""
    from sglang_tpu.multimodal_gen.h3_megakernel import kernel
    from sglang_tpu.multimodal_gen.h3_megakernel import op as stack_op

    def structure(value: int) -> tuple[int, ...]:
        return kernel.loop_structure(
            value,
            rows,
            devices,
            config,
            tiling or stack_op.structured_tiling(devices, rows),
            cut_kv_compute=cut_kv_compute,
        )

    # A structure holds an interval of live lengths within one of the
    # narrowed tail's 128-row steps (query tiles start on multiples of 128):
    # bisect that step for its first, a few calls of about 60 us each.
    want = structure(used)
    first, last = (used - 1) // 128 * 128 + 1, used
    while first < last:
        middle = (first + last) // 2
        if structure(middle) == want:
            last = middle
        else:
            first = middle + 1
    return first


def _shard_adaln() -> bool:
    """This rank asks for sharded AdaLN projections (the ranks agree in
    :func:`enable`)."""
    return envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_SHARD_ADALN.get()


def _precompute_adaln() -> bool:
    """Every step's AdaLN projections of a request in one GEMM a block."""
    return envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_ADALN_PRECOMPUTE.get()


# Rows a step of a batched AdaLN projection -> whether it gave this rank's
# blocks the per-step projections' bits at every step (checked once each).
_ADALN_BATCH_EXACT: dict[int, bool] = {}


def _adaln_input(model, unique_timesteps: torch.Tensor) -> torch.Tensor:
    """``MiniMaxH3DiTModel.forward``'s AdaLN input for a step's timesteps."""
    t_emb = model._time_embedding(unique_timesteps.view(-1))
    return nn.functional.silu(t_emb).to(torch.bfloat16)


def _step_rows(proj, rows: torch.Tensor, start: int, count: int, adaln_input):
    """A block's projection of one step from its batched rows: what
    ``proj(adaln_input)`` returns (TP 1, as the support check requires)."""
    return proj.split_output(rows[start : start + count])


def _vouch_margin() -> float | None:
    """The margin of the vouched heads (``SGLANG_TPU_OPT_H3_MEGAKERNEL_VOUCH``),
    or None: the proven layers only."""
    if not envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_VOUCH.get():
        return None
    return float(envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_VOUCH_MARGIN.get())


def proven_layers(q_norm: torch.Tensor, k_norm: torch.Tensor, config) -> list[int]:
    """The layers whose attention the kernel runs at a fixed max.

    ``kernel.fixed_max_layers``'s bound, from the ``[L, 1, D]`` norm weights
    alone: RMSNorm leaves q and k at norm ``sqrt(D)`` before their norm
    weights and RoPE rotates dim ``d`` with ``d + R/2``, so by Cauchy-Schwarz
    no score exceeds ``D`` times a rotated pair's largest q weight times its
    largest k weight (or an unrotated dim's q weight times its k weight). A
    layer whose bound, in log2 units and with the bf16 rounding slack, stays
    at or under 109 takes ``exp2(s - 17)`` without a running max, exactly,
    for every input. The kernel derives the same flags itself (the call
    passes ``fixed=None``); this host copy is for the startup log.
    """
    half = config.rope_dims // 2
    q, k = (
        np.abs(w.detach().float().cpu().numpy().reshape(w.shape[0], -1))
        for w in (q_norm, k_norm)
    )

    def pairs(w):
        return np.maximum(w[:, :half], w[:, half : 2 * half])

    rotated = (pairs(q) * pairs(k)).max(axis=1)
    plain = (q[:, 2 * half :] * k[:, 2 * half :]).max(axis=1, initial=0.0)
    bound = (
        np.maximum(rotated, plain)
        * np.float32(math.sqrt(config.head_dim))
        * np.float32(math.log2(math.e))
    )
    proven = bound * np.float32(_BOUND_SLACK) <= np.float32(_FIXED_BOUND)
    return [layer for layer, ok in enumerate(proven.tolist()) if ok]


def _server_args():
    from sglang.multimodal_gen.runtime.server_args import get_global_server_args

    try:
        return get_global_server_args()
    except ValueError:
        return None


def _sparse_backend(server_args) -> str | None:
    """The H3 DiT's selected attention backend, if it is a sparse one."""
    resolve = getattr(
        server_args.pipeline_config, "resolve_transformer_attention_backend", None
    )
    backend = resolve(server_args) if resolve is not None else None
    if backend is not None and backend.is_sparse:
        return str(backend)
    return None


def unsupported_reason(model) -> str | None:
    """Why this rank cannot run ``model``'s blocks through the kernel, or None."""
    from torch.distributed.tensor import DTensor

    from sglang.multimodal_gen import envs as gen_envs
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_ring_ctx,
        get_tp_world_size,
        get_ulysses_ctx,
        get_world_rank,
        get_world_size,
    )
    from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
    from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
        RESIDENT,
    )
    from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
        is_layerwise_offloaded_module,
    )
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as dit
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if not current_platform.is_tpu():
        return "not a TPU platform"
    ulysses, ulysses_rank = get_ulysses_ctx()
    if get_ring_ctx()[0] != 1 or get_tp_world_size() != 1:
        return "needs ring degree 1 and TP 1"
    if ulysses < 2 or ulysses != get_world_size():
        return f"needs Ulysses over the whole world ({ulysses} of {get_world_size()})"
    if ulysses_rank != get_world_rank():
        return "Ulysses rank differs from the world (Pallas mesh) rank"
    if model.adaln_cache is not None or model.adaln_t_table is not None:
        return "AdaLN cache or curve AdaLN"
    server_args = _server_args()
    if server_args is not None:
        if getattr(server_args, "minimax_h3_adaln_cache_path", None) or getattr(
            server_args, "minimax_h3_adaln_online", False
        ):
            return "AdaLN cache (--minimax-h3-adaln-cache-path / --minimax-h3-adaln-online)"
        for flag in (
            "use_fsdp_inference",
            "dit_cpu_offload",
            "dit_layerwise_offload",
            "enable_torch_compile",
        ):
            if getattr(server_args, flag, False):
                return f"--{flag.replace('_', '-')}"
        if getattr(server_args, "lora_path", None):
            return "LoRA"
        if server_args.residency_mode("transformer") != RESIDENT:
            return "the transformer is not resident on device"
        if server_args.pipeline_config.uses_subblock_attention(server_args):
            return "sub-block sparse attention backend"
        sparse = _sparse_backend(server_args)
        if sparse is not None:
            # The kernel runs dense attention.
            return f"sparse attention backend {sparse}"
    if gen_envs.SGLANG_CACHE_DIT_ENABLED or is_layerwise_offloaded_module(model):
        return "Cache-DiT or layerwise offload"
    for block in model.blocks:
        if type(block) is not dit.MiniMaxH3DiTBlock or block.adaln_proj is None:
            return f"unsupported block {type(block).__name__}"
        if block.attn.hybrid is not None or block.attn.to_gate_compress is not None:
            return "hybrid or sparse attention"
        for module in (*_projections(block), block.adaln_proj.linear):
            weight = module.weight
            if (
                isinstance(weight, DTensor)
                or weight.dtype != torch.bfloat16
                or weight.device.type == "cpu"
                or not isinstance(module.quant_method, UnquantizedLinearMethod)
            ):
                return (
                    "projections are not plain bf16 device tensors "
                    "(FSDP, offload, quantization)"
                )
        if any(getattr(m, "bias", None) is not None for m in _projections(block)):
            return "projection bias"
    from sglang_tpu.multimodal_gen.h3_megakernel import tiling as tl

    config = pack.config_from_arch(model.arch, len(model.blocks))
    try:
        tl.Tiling().fit(config, 128, ulysses)
    except ValueError as error:
        return str(error)
    rows = envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_ROWS.get()
    if rows is not None:
        from sglang_tpu.multimodal_gen.h3_megakernel import op as stack_op

        try:
            stack_op.shape_tiling(ulysses, rows, tl.Tiling()).fit(config, rows, ulysses)
        except ValueError as error:
            return f"SGLANG_TPU_OPT_H3_MEGAKERNEL_ROWS={rows}: {error}"
    if _structured():
        try:
            _tiling_overrides()
        except ValueError as error:
            return f"SGLANG_TPU_OPT_H3_MEGAKERNEL_TILING: {error}"
    margin = _vouch_margin()
    if margin is not None and not (math.isfinite(margin) and margin >= 0):
        return (
            f"SGLANG_TPU_OPT_H3_MEGAKERNEL_VOUCH_MARGIN={margin}: needs a finite "
            "margin >= 0"
        )
    return None


def _projections(block) -> tuple[nn.Module, ...]:
    return (block.attn.qkv_proj, block.attn.out_proj, block.mlp.fc1, block.mlp.fc2)


def _gather(value) -> list:
    """``value`` from every world rank, over the CPU group."""
    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_world_group,
    )

    group = get_world_group()
    if group.world_size <= 1:
        return [value]
    values = [None] * group.world_size
    dist.all_gather_object(values, value, group=group.cpu_group)
    return values


def _ranks_agree(reason: str | None) -> str | None:
    """Every rank must take the same path: a rank outside the kernel leaves
    the others waiting at its barrier."""
    return next(
        (f"rank {rank}: {why}" for rank, why in enumerate(_gather(reason)) if why),
        None,
    )


def _ranks_agree_sharded(reason: str | None) -> tuple[str | None, bool]:
    """:func:`_ranks_agree` for a rank asking for sharded AdaLN projections:
    ``(refusal, shard)``.

    Such a rank votes ``_SHARD_VOTE`` where it would vote None. A rank without
    the option votes as before and reads that vote as a refusal, so every rank
    decides alike: the ranks shard when all of them asked, and all of them
    leave the kernel off when only some did (they would otherwise disagree on
    the per-step all-gather).
    """
    votes = _gather(reason or _SHARD_VOTE)
    refusal = next(
        (
            f"rank {rank}: {why}"
            for rank, why in enumerate(votes)
            if why and why != _SHARD_VOTE
        ),
        None,
    )
    if refusal is None and any(why != _SHARD_VOTE for why in votes):
        refusal = "SGLANG_TPU_OPT_H3_MEGAKERNEL_SHARD_ADALN is set on some ranks only"
    return refusal, refusal is None


def _try(what: str, step, *args, **kwargs) -> tuple[object, str | None]:
    """``step``'s result and None, or None and why it raised: a rank must not
    leave its peers waiting in the next collective."""
    try:
        return step(*args, **kwargs), None
    except Exception as error:  # noqa: BLE001 - reported to every rank
        logger.exception("MiniMax H3 megakernel: %s failed", what)
        return None, f"{what}: {type(error).__name__}: {error}"


def _host_available() -> int:
    """Bytes this process may still allocate on the host: the smaller of the
    machine's available memory and its cgroup's headroom (a pod's limit)."""
    import psutil

    available = psutil.virtual_memory().available
    for limit_path, usage_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
    ):
        try:
            with open(limit_path) as limit, open(usage_path) as usage:
                limit_text, usage_text = limit.read().strip(), usage.read().strip()
        except OSError:
            continue
        if limit_text.isdigit() and usage_text.isdigit():
            available = min(available, max(0, int(limit_text) - int(usage_text)))
        break
    return available


def _host_waves(need: int) -> tuple[int, int]:
    """(this rank's wave, waves): the ranks of a host stage their blocks
    through its memory as many at a time as fit in 80% of what is free."""
    import socket

    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_world_group,
    )

    infos = _gather((socket.gethostname(), need, _host_available()))
    me, mine, waves = get_world_group().rank_in_group, 0, 1
    for host in sorted({info[0] for info in infos}):
        ranks = [rank for rank, info in enumerate(infos) if info[0] == host]
        free = min(infos[rank][2] for rank in ranks)
        most = max(max(infos[rank][1] for rank in ranks), 1)
        per_wave = max(1, int(0.8 * free) // most)
        waves = max(waves, -(-len(ranks) // per_wave))
        if me in ranks:
            mine = ranks.index(me) // per_wave
    return mine, waves


def _pack_on_every_rank(blocks, config) -> dict[str, torch.Tensor]:
    """``pack.pack_blocks`` through host memory, a wave of ranks at a time.

    A rank that fails still joins every barrier, and then every rank raises:
    the ranks that packed have freed their blocks, so there is no torch path
    to fall back to.
    """
    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_world_group,
    )

    group = get_world_group()
    mine, waves = _host_waves(pack.device_bytes(blocks))
    weights, error = None, None
    for wave in range(waves):
        if wave == mine:
            weights, error = _try(
                "packing", pack.pack_blocks, blocks, config, stage_on_host=True
            )
        if group.world_size > 1:
            dist.barrier(group=group.cpu_group)
    failed = _ranks_agree(error)
    if failed is not None:
        raise RuntimeError(f"MiniMax H3 megakernel: packing failed ({failed})")
    return weights


def padded_seq_len(
    used: int, *, devices: int, config, rows: int | None, structured: bool = False
) -> int:
    """Packed rows for ``used`` live rows: ``devices`` times ``rows`` when
    the request fits them, else the kernel's ``padded_rows`` (rows a device
    grow by 128 until its tiles are large enough; 4,992 a device at 8
    devices for 1344x768x124). ``structured`` (the static-structure option):
    ``op.STRUCTURED_ROWS`` replaces some of ``padded_rows``' row counts."""
    from sglang_tpu.multimodal_gen.h3_megakernel import op as stack_op
    from sglang_tpu.multimodal_gen.h3_megakernel import tiling as tl

    if rows is not None:
        if used <= rows * devices:
            return rows * devices
        logger.warning(
            "MiniMax H3 megakernel: %d rows do not fit SGLANG_TPU_OPT_H3_MEGAKERNEL_ROWS"
            "=%d a device; padding by the kernel's rule instead",
            used,
            rows,
        )
    padded = tl.padded_rows(used, devices, stack_op.ROWS_TILING, config)
    if structured:
        padded = devices * stack_op.STRUCTURED_ROWS.get(
            (devices, padded // devices), padded // devices
        )
    return padded


def _use_kernel_padding(devices: int, config) -> None:
    """Point the packed builders' padding at the kernel's row count."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        packed_sequence,
    )

    packed_sequence.minimax_h3_padded_seq_len = functools.partial(
        padded_seq_len,
        devices=devices,
        config=config,
        rows=envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_ROWS.get(),
        structured=_structured(),
    )


@dataclasses.dataclass(frozen=True)
class AdalnShard:
    """This rank's share of the blocks' AdaLN projections
    (``SGLANG_TPU_OPT_H3_MEGAKERNEL_SHARD_ADALN``): blocks ``[first, end)`` of
    ``layers``, ``per_rank`` a rank (``pack.adaln_block_range``); the
    projections' ``(expand_ratio, modality_num, hidden_size)``, which shape a
    rank's zero tables; and the projection bytes it kept and freed."""

    rank: int
    devices: int
    layers: int
    first: int
    end: int
    per_rank: int
    table_shape: tuple[int, int, int]
    kept: int
    freed: int


def _shard_group():
    """The group the AdaLN tables are all-gathered over: sglang's
    sequence-parallel group. Under the support check (Ulysses over the whole
    world, ring 1, TP 1) it holds every rank, and its device group already
    runs a device all-gather every step (the DiT's output gather in
    ``MiniMaxH3DiTModel.forward``)."""
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_sp_group,
    )

    return get_sp_group()


def _all_gather_tables(
    local: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Every rank's ``local`` concatenated along dim 0 in rank order, over the
    sequence-parallel group's device group (:func:`_shard_group`):
    ``GroupCoordinator.all_gather``'s device path (dim 0), without its CPU
    shared-memory branch. ``out``: the ``[ranks * k, ...]`` result buffer,
    else a new one."""
    from sglang.multimodal_gen.runtime.distributed.utils import all_gather_single

    group = _shard_group()
    if out is None:
        out = local.new_empty((group.world_size * local.shape[0], *local.shape[1:]))
    all_gather_single(out, local, group=group.device_group)
    return out


def _shard_probe(proj) -> torch.Tensor:
    """The startup check's AdaLN input: two fixed random rows on the
    projections' device, the same on every rank and unrelated to requests."""
    weight = proj.linear.weight
    generator = torch.Generator().manual_seed(0)
    probe = torch.randn(2, weight.shape[1], generator=generator)
    return probe.to(dtype=weight.dtype, device=weight.device)


def _shard_adaln_projs(blocks, devices: int):
    """``(shard, (tables, local, out))``: keep this rank's share of
    ``blocks``' AdaLN projections and free the other blocks' (their
    parameters drop to empty tensors, as packing drops the projections it
    copies, so nothing that still holds the blocks keeps the memory).

    Also everything the startup check needs before its all-gather, so that
    nothing a rank could fail on runs between the vote on this step and that
    collective: ``tables``, every block's replicated tables of a fixed input
    (:func:`_shard_probe`), built before the freeing; ``local``, this rank's
    share of that input's tables, built after it from the kept projections;
    and ``out``, the all-gather's result buffer. The device is synchronized
    last, so an asynchronous device error surfaces here."""
    group = _shard_group()
    if group.world_size != devices:
        raise RuntimeError(
            f"the sequence-parallel group has {group.world_size} ranks, the world "
            f"{devices}"
        )
    rank = group.rank_in_group
    first, end, per_rank = pack.adaln_block_range(len(blocks), devices, rank)
    proj = blocks[0].adaln_proj
    table_shape = (proj.expand_ratio, proj.modality_num, proj.hidden_size)
    probe = _shard_probe(proj)
    with torch.no_grad():
        tables = pack.adaln_tables([block.adaln_proj for block in blocks], probe)
    # No copy into a weight may still be running when its memory goes.
    pack._sync(probe.device)
    kept = freed = 0
    for index, block in enumerate(blocks):
        for param in block.adaln_proj.parameters():
            size = param.numel() * param.element_size()
            if first <= index < end:
                kept += size
            else:
                freed += size
                param.data = torch.empty(0, dtype=param.dtype, device=param.device)
    with torch.no_grad():
        local = pack.adaln_share(
            [block.adaln_proj for block in blocks[first:end]],
            probe,
            per_rank=per_rank,
            table_shape=table_shape,
        )
    out = local.new_empty((devices * per_rank, *local.shape[1:]))
    pack._sync(probe.device)
    shard = AdalnShard(
        rank=rank,
        devices=devices,
        layers=len(blocks),
        first=first,
        end=end,
        per_rank=per_rank,
        table_shape=table_shape,
        kept=kept,
        freed=freed,
    )
    return shard, (tables, local, out)


def _check_shard(shard: AdalnShard, tables, local, out) -> str | None:
    """The startup check's collective: None if the per-step all-gather of
    every rank's ``local`` gives ``tables`` bit for bit (every block's,
    whichever rank holds it), else why not. It starts with the all-gather:
    every rank enters it, as all of them passed :func:`_shard_adaln_projs`."""
    got = _all_gather_tables(local, out)[: shard.layers]
    if got.shape != tables.shape or got.dtype != tables.dtype:
        return f"gathered tables {tuple(got.shape)} {got.dtype}, want {tuple(tables.shape)} {tables.dtype}"
    if not torch.equal(got.view(torch.int16), tables.view(torch.int16)):
        return "the gathered AdaLN tables differ from the replicated ones"
    return None


def _shard_on_every_rank(blocks, devices: int) -> AdalnShard | None:
    """Shard the blocks' AdaLN projections over the ranks (all of them asked;
    :func:`enable`), before packing: the freed projections are neither
    staged nor moved back. Then check, on the device, that the per-step
    all-gather gives the replicated tables of a fixed input bit for bit.

    Two steps, each a ``_try`` and a vote on the CPU group: sharding (which
    builds the check's inputs) and the check. A rank's failure in the first
    step reaches every rank through its vote before any rank enters the
    device all-gather of the second, so no rank waits in a collective its
    peers skip; a failure or a difference in the second is voted on too. A
    rank that fails has freed some projections, so every rank raises. None,
    with a log line, on one rank."""
    if devices == 1:
        logger.info(
            "MiniMax H3 megakernel: SGLANG_TPU_OPT_H3_MEGAKERNEL_SHARD_ADALN is a "
            "no-op on one rank; every block's AdaLN projection stays"
        )
        return None
    result, error = _try(
        "sharding the AdaLN projections", _shard_adaln_projs, blocks, devices
    )
    failed = _ranks_agree(error)
    if failed is None:
        shard, check = result
        del result
        why, error = _try(
            "checking the sharded AdaLN tables", _check_shard, shard, *check
        )
        del check
        failed = _ranks_agree(error or why)
    if failed is not None:
        raise RuntimeError(
            f"MiniMax H3 megakernel: sharding the AdaLN projections failed ({failed})"
        )
    first, end = shard.first, shard.end
    logger.info(
        "MiniMax H3 megakernel: AdaLN projections sharded over %d ranks, %d "
        "blocks a rank: rank %d keeps %s (%.2f GiB on the device) and freed "
        "%.2f GiB; each step all-gathers the %d ranks' tables over the "
        "sequence-parallel group (checked: the replicated tables of a fixed "
        "input, bit for bit)",
        devices,
        shard.per_rank,
        shard.rank,
        f"blocks {first}-{end - 1}" if end > first else "no block",
        shard.kept / 2**30,
        shard.freed / 2**30,
        devices,
    )
    return shard


class _Vouching:
    """One request's vouched heads, on device, from the start to the end of
    its denoise loop: ``heads`` ([L, heads] int32), the heads its next step
    tries at a fixed max (its last step's vouched heads; zeros before its
    first step), and ``tally`` ([2] int32), its recomputed steps and the
    heads its steps ran at a fixed max beyond the proven layers."""

    __slots__ = ("request_id", "heads", "tally")

    def __init__(self, request_id, heads: torch.Tensor, tally: torch.Tensor):
        self.request_id = request_id
        self.heads = heads
        self.tally = tally


class H3MegakernelStack(nn.Module):
    """``model.blocks[0]`` once packed: every block of a step in one op call.

    It takes the block keyword arguments sglang's forward passes each block
    and holds each block's ``adaln_proj`` (resident, run on every call) and
    the packed weights (non-persistent buffers). ``structured`` (the
    static-structure option): the op also takes the static structure of each
    call's host live length (``structure``). With ``vouch_margin`` the op is
    the vouched variant and the stack holds the running request's vouched
    heads between :meth:`begin_request` and :meth:`end_request`. With
    ``adaln_shard`` it holds this rank's share of the projections only and
    each call all-gathers the tables (``pack.sharded_adaln_tables``).
    ``levers``: the structured calls' kernel levers (None: the flags', if
    structured); ``tiling_overrides``: their tiling bounds (None: the
    flag's, if structured).
    """

    def __init__(
        self,
        model,
        adaln_projs,
        weights,
        op,
        config,
        devices: int = 1,
        structured: bool = False,
        vouch_margin=None,
        adaln_shard: AdalnShard | None = None,
        levers=None,
        tiling_overrides=None,
    ) -> None:
        super().__init__()
        self.adaln = nn.ModuleList(adaln_projs)
        self.adaln_shard = adaln_shard
        for name in pack.WEIGHT_ORDER:
            self.register_buffer(name, weights[name], persistent=False)
        self.head_dim = config.head_dim
        self.heads = config.heads
        self.layers = config.layers
        # Plain attributes, not submodules: the op is a torch custom op and
        # the model owns this module.
        self.__dict__["op"] = op
        self.__dict__["_model"] = weakref.ref(model)
        self.__dict__["config"] = config
        self.devices = devices
        self.structured = structured
        if levers is None:
            levers = _levers() if structured else frozenset()
        self.levers = frozenset(levers)
        if tiling_overrides is None:
            tiling_overrides = _tiling_overrides() if structured else ()
        self.tiling_overrides = tuple(tiling_overrides)
        # (rows a device, static structure) pairs logged once each.
        self.__dict__["_structures"] = set()
        self.calls = 0
        self.vouch_margin = vouch_margin
        self._vouching: _Vouching | None = None
        # A request's batched AdaLN rows (precompute_adaln): this rank's blocks'
        # projections of every step's rows, and each step's (first row, rows)
        # by the id of its timesteps tensor; the step of the next forward.
        self._adaln_batch = None
        self._adaln_step: tuple[int, int] | None = None

    def _fresh_vouching(self, request_id) -> _Vouching:
        """No heads to try yet: the proven layers only."""
        device = self.q_norm.device
        return _Vouching(
            request_id,
            torch.zeros((self.layers, self.heads), dtype=torch.int32, device=device),
            torch.zeros(2, dtype=torch.int32, device=device),
        )

    def begin_request(self, request_id) -> None:
        """A request's denoise loop starts: whatever an earlier one left is
        dropped, and its first step runs the proven layers only."""
        self._vouching = None
        self._adaln_batch, self._adaln_step = None, None
        if self.vouch_margin is not None:
            self._vouching = self._fresh_vouching(request_id)

    def end_request(self) -> tuple[int, int] | None:
        """The request's (recomputed steps, heads its steps ran at a fixed
        max beyond the proven layers), read from device once, or None
        without vouching; its vouched heads are dropped."""
        vouching, self._vouching = self._vouching, None
        self._adaln_batch, self._adaln_step = None, None
        if vouching is None:
            return None
        try:
            redone, ran = vouching.tally.tolist()
        except Exception:  # noqa: BLE001 - a failed loop's own error wins
            logger.warning("MiniMax H3 megakernel: no tally of the vouched heads")
            return None
        return int(redone), int(ran)

    def _vouching_for_call(self) -> _Vouching:
        """The running request's vouching, or a fresh one (kept nowhere) for
        a forward outside a request's denoise loop."""
        vouching = self._vouching
        if vouching is None:
            return self._fresh_vouching(None)
        current = _current_request_id()
        if current is not None and current != vouching.request_id:
            raise RuntimeError(
                f"MiniMax H3 megakernel: a forward of request {current} while "
                f"request {vouching.request_id} holds the vouched heads"
            )
        return vouching

    def precompute_adaln(self, model, step_timesteps) -> None:
        """A request's AdaLN projections of every step at once: one GEMM a
        block of this rank's over the steps' stacked AdaLN inputs, each
        computed as ``MiniMaxH3DiTModel.forward`` computes it (a step has one
        row a unique timestep: H3's first step, at t = 0 for video and audio,
        has one). Kept only if the batch gives every step the per-step
        projection's bits: the first batch of each pattern of step rows is
        checked on the device, one host read."""
        self._adaln_batch, self._adaln_step = None, None
        if not step_timesteps or not self.adaln:
            return
        inputs = [_adaln_input(model, t) for t in step_timesteps]
        counts = [x.shape[0] for x in inputs]
        starts = [sum(counts[:step]) for step in range(len(counts))]
        exact = _ADALN_BATCH_EXACT.get(tuple(counts))
        if exact is False:
            return
        stacked = torch.cat(inputs)
        batch = [proj.project_local(stacked) for proj in self.adaln]
        if exact is None:
            differ = stacked.new_zeros((), dtype=torch.int64)
            for proj, rows in zip(self.adaln, batch):
                for start, x in zip(starts, inputs):
                    one = proj.project_local(x)
                    differ += (rows[start : start + x.shape[0]] != one).sum()
            differ = int(differ)
            exact = _ADALN_BATCH_EXACT[tuple(counts)] = differ == 0
            (logger.info if exact else logger.warning)(
                "MiniMax H3 megakernel: AdaLN projections of %d steps (%d rows) "
                "in one GEMM a block: %s (rank %d)",
                len(inputs),
                stacked.shape[0],
                "bitwise the per-step projections, kept"
                if exact
                else f"{differ} elements differ from the per-step projections; "
                "per-step for these steps",
                dist.get_rank() if dist.is_initialized() else 0,
            )
        if exact:
            rows = zip(step_timesteps, starts, counts)
            self._adaln_batch = (batch, {id(t): (a, n) for t, a, n in rows})

    def adaln_step(self, unique_timesteps) -> None:
        """The next forward is the batched plan's step of this timesteps
        tensor (``forward_kwargs`` passes the plan's own), or no step of it."""
        batch = self._adaln_batch
        self._adaln_step = None if batch is None else batch[1].get(id(unique_timesteps))

    def packed_bytes(self) -> int:
        return sum(
            getattr(self, name).numel() * getattr(self, name).element_size()
            for name in pack.WEIGHT_ORDER
        )

    def operands(
        self,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """``(onehot, cos, sin, tables, used)`` for this forward. Sharded,
        every rank all-gathers the tables here on every call, the same
        collective in the same place on all of them."""
        shard = self.adaln_shard
        projs, step, self._adaln_step = self.adaln, self._adaln_step, None
        if step is not None and self._adaln_batch is not None:
            projs = [
                functools.partial(_step_rows, proj, rows, *step)
                for proj, rows in zip(self.adaln, self._adaln_batch[0])
            ]
        if shard is None:
            tables = pack.adaln_tables(projs, adaln_input)
        else:
            tables = pack.sharded_adaln_tables(
                projs,
                adaln_input,
                per_rank=shard.per_rank,
                layers=shard.layers,
                table_shape=shard.table_shape,
                gather=_all_gather_tables,
            )
        onehot = pack.table_onehot(combined_indices)
        cos, sin = pack.rope_tables(cos_sin_cache, self.head_dim)
        # [0, used, total]: rows from used on are trailing padding.
        used = cu_seqlens[1:2].to(torch.int32)
        return onehot, cos, sin, tables, used

    def structure(self, cu_seqlens_host, rows: int) -> int:
        """The static structure of a call over ``rows`` a device: that of the
        host live length (``static_structure``) at the row counts the option
        covers (``op.STRUCTURED_SHAPES``); else 0, the generic walk, as also
        without host bounds that match the rows."""
        from sglang_tpu.multimodal_gen.h3_megakernel import op as stack_op

        total = rows * self.devices
        bounds = tuple(int(bound) for bound in cu_seqlens_host or ())
        shape, why = 0, None
        if (self.devices, rows) not in stack_op.STRUCTURED_SHAPES:
            why = f"no static structure at {rows} rows a device, {self.devices} devices"
        elif len(bounds) != 3 or bounds[2] != total or not 0 < bounds[1] <= total:
            why = f"host bounds {cu_seqlens_host} give no live length"
        else:
            shape = static_structure(
                bounds[1],
                rows,
                self.devices,
                self.config,
                cut_kv_compute=stack_op.cut_kv_compute(self.devices, rows, self.levers),
                tiling=stack_op.structured_tiling(
                    self.devices, rows, overrides=self.tiling_overrides
                ),
            )
        if (rows, shape) not in self._structures:
            self._structures.add((rows, shape))
            if why is None:
                logger.info(
                    "MiniMax H3 megakernel: %d rows a device; live length %d "
                    "runs the loops of static live length %d (one program for "
                    "every live length of that structure)",
                    rows,
                    bounds[1],
                    shape,
                )
            else:
                logger.info("MiniMax H3 megakernel: %s; the generic walk", why)
        return shape

    def _check_model(self) -> None:
        model = self._model()
        if model is None:
            return
        blocks = model._modules.get("blocks")
        if (
            not isinstance(blocks, nn.ModuleList)
            or len(blocks) != 1
            or blocks[0] is not self
        ):
            raise RuntimeError(
                "MiniMax H3 megakernel: the DiT's blocks changed after packing "
                "(Cache-DiT?); unset SGLANG_TPU_ENABLE_H3_MEGAKERNEL to use it"
            )
        backend = getattr(model, "_resolved_attention_backend", None)
        if backend is not None and backend.is_sparse:
            raise RuntimeError(
                f"MiniMax H3 megakernel runs dense attention; {backend} is sparse"
            )

    def _rows_supported(self, rows: int) -> bool:
        """A multiple of 128 rows a rank, or a row count whose tiling
        op.SHAPE_TILINGS defines (8 x 4,800 rows: query tiles that
        cross device boundaries)."""
        from sglang_tpu.multimodal_gen.h3_megakernel import op as stack_op

        return rows % 128 == 0 or (self.devices, rows) in stack_op.SHAPE_TILINGS

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int = 0,
        subblock_sparse_query_block_mask: torch.Tensor | None = None,
        ulysses_active: bool = False,
        ring_active: bool = False,
        adaln_params: tuple[torch.Tensor, ...] | None = None,
        block_adaln_params: tuple[tuple[torch.Tensor, ...], ...] | None = None,
    ) -> torch.Tensor:
        """Every block on this rank's rows. The live length comes from
        ``cu_seqlens`` on device; the host bounds only pick the static
        structure under the static-structure option."""
        del max_seqlen
        self._check_model()
        if (
            ring_active
            or not ulysses_active
            or adaln_params is not None
            or block_adaln_params is not None
            or subblock_sparse_query_block_mask is not None
            or cu_seqlens.shape[0] != 3
            or not self._rows_supported(hidden.shape[0])
            or hidden.dtype != torch.bfloat16
        ):
            raise RuntimeError(
                "MiniMax H3 megakernel takes one bf16 segment plus trailing "
                "padding, a multiple of 128 rows a rank (or a row count with "
                "its own tiling, op.SHAPE_TILINGS), under Ulysses without "
                "ring, sparse attention or cached AdaLN; got "
                f"{hidden.shape[0]} {hidden.dtype} rows and cu_seqlens of "
                f"{cu_seqlens.shape[0]}"
            )
        onehot, cos, sin, tables, used = self.operands(
            adaln_input, combined_indices, rope_cache[0], cu_seqlens
        )
        weights = tuple(getattr(self, name) for name in pack.WEIGHT_ORDER)
        structure = (
            (self.structure(cu_seqlens_host, hidden.shape[0]),)
            if self.structured
            else ()
        )
        if self.vouch_margin is None:
            out = self.op(
                hidden.contiguous(),
                onehot,
                cos,
                sin,
                tables,
                used,
                *weights,
                *structure,
            )
        else:
            vouching = self._vouching_for_call()
            out, heads, tally = self.op(
                hidden.contiguous(),
                onehot,
                cos,
                sin,
                tables,
                used,
                vouching.heads,
                vouching.tally,
                *weights,
                *structure,
            )
            # Device tensors only: the next step reads them on device.
            vouching.heads, vouching.tally = heads, tally
        self.calls += 1
        return out


def is_packed(model) -> bool:
    return model.__dict__.get(_STATE) is not None


def _world_size() -> int:
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_world_size,
    )

    return get_world_size()


def _op_for(config, devices: int, structured: bool = False):
    """The op of ``config``'s blocks; one a set of options."""
    margin = _vouch_margin()
    levers = _levers() if structured else frozenset()
    overrides = _tiling_overrides() if structured else ()
    # Profiling only: the kernel's xprof trace regions (kernel.stack trace).
    trace = max(0, envs.SGLANG_TPU_MK_TRACE_REGIONS.get())
    key = (config, devices, structured, margin, levers, overrides, trace)
    if key not in _OPS:
        from sglang_tpu.multimodal_gen.h3_megakernel import op as stack_op

        _OPS[key] = stack_op.build_stack_op(
            config,
            devices,
            structured=structured,
            vouch_margin=margin,
            **({"levers": levers} if levers else {}),
            **({"tiling_overrides": overrides} if overrides else {}),
            **({"trace": trace} if trace else {}),
        )
    return _OPS[key]


def _layer_ranges(layers: list[int]) -> str:
    """``[0, 1, 2, 5]`` -> ``"0-2,5"``."""
    spans, start = [], None
    for index, layer in enumerate(layers):
        if start is None:
            start = layer
        if index + 1 == len(layers) or layers[index + 1] != layer + 1:
            spans.append(str(start) if start == layer else f"{start}-{layer}")
            start = None
    return ",".join(spans) or "none"


def enable(model) -> bool:
    """Pack ``model``'s blocks and register the op, if every rank can."""
    if model.__dict__.get(_STATE, False) is not False:
        return is_packed(model)
    reason, error = _try("the support check", unsupported_reason, model)
    shard = _shard_adaln()
    if shard:
        reason, shard = _ranks_agree_sharded(error or reason)
    else:
        reason = _ranks_agree(error or reason)
    devices, config, stack_op = None, None, None
    structured = _structured()
    if reason is None:
        devices = _world_size()
        config = pack.config_from_arch(model.arch, len(model.blocks))
        # Register the op first: packing frees the blocks' projections.
        stack_op, error = _try(
            "registering the op", _op_for, config, devices, structured
        )
        reason = _ranks_agree(error)
    if reason is not None:
        logger.warning("MiniMax H3 megakernel off (%s); blocks run in torch.", reason)
        model.__dict__[_STATE] = None
        return False
    blocks = list(model.blocks)
    adaln_shard = _shard_on_every_rank(blocks, devices) if shard else None
    weights = _pack_on_every_rank(blocks, config)
    margin = _vouch_margin()
    levers = _levers() if structured else frozenset()
    if adaln_shard is None:
        adaln_projs = [block.adaln_proj for block in blocks]
    else:
        adaln_projs = [
            block.adaln_proj for block in blocks[adaln_shard.first : adaln_shard.end]
        ]
    stack = H3MegakernelStack(
        model,
        adaln_projs,
        weights,
        stack_op,
        config,
        devices=devices,
        structured=structured,
        vouch_margin=margin,
        adaln_shard=adaln_shard,
        levers=levers,
        tiling_overrides=_tiling_overrides() if structured else (),
    )
    model.blocks = nn.ModuleList([stack])
    model.__dict__[_STATE] = stack
    _use_kernel_padding(devices, config)
    proven = proven_layers(weights["q_norm"], weights["k_norm"], config)
    online = [layer for layer in range(config.layers) if layer not in proven]
    rows = envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_ROWS.get()
    vouched = (
        ""
        if margin is None
        else "; from each request's second step, also at a fixed max the online "
        "layers' heads the request's previous step's q/k peaks bound within it by "
        f"margin {margin:g}, checked on each step's own peaks (the step recomputed "
        "with the proven layers when one is out)"
    )
    logger.info(
        "MiniMax H3 megakernel on: %d blocks, Ulysses %d, %.2f GiB packed bf16 "
        "weights a device, rows a device %s; fixed-max softmax on the layers the "
        "q/k norm weights prove (%d/%d): %s; online softmax: %s%s",
        config.layers,
        devices,
        stack.packed_bytes() / 2**30,
        rows if rows is not None else "by padded_rows",
        len(proven),
        config.layers,
        _layer_ranges(proven),
        _layer_ranges(online),
        vouched,
    )
    if structured:
        _log_structured(devices, levers)
    return True


def _log_structured(devices: int, levers=frozenset()) -> None:
    from sglang_tpu.multimodal_gen.h3_megakernel import op as stack_op

    shapes = sorted(
        rows for count, rows in stack_op.STRUCTURED_SHAPES if count == devices
    )
    notes = [f"at {', '.join(map(str, shapes)) or 'no'} rows a device"]
    notes += [
        f"{new} rows a device where padded_rows gives {old}"
        for (count, old), new in stack_op.STRUCTURED_ROWS.items()
        if count == devices
    ]
    if stack_op.stack_kwargs(devices, 0).get("prefetch_down") is False:
        notes.append(
            "no phase C down-weight prefetch"
            + (" (but the prefetch_down lever)" if "prefetch_down" in levers else "")
        )
    for lever in sorted(levers):
        at = sorted(
            rows for count, rows in stack_op.LEVER_SHAPES[lever] if count == devices
        )
        notes.append(
            f"kernel lever {lever} at {', '.join(map(str, at)) or 'no'} rows a device"
        )
    overrides = _tiling_overrides()
    if overrides:
        notes.append(
            "tiling bounds " + ", ".join(f"{name}={value}" for name, value in overrides)
        )
    logger.info(
        "MiniMax H3 megakernel: static loop structure from each call's host "
        "live length: narrowed tail, cut key tile and dense walk (%s)",
        "; ".join(notes),
    )


def _packed_dit(module):
    """The packed H3 DiT behind ``module`` (compile/parallel wrappers
    unwrapped), or None."""
    for _ in range(4):
        if isinstance(getattr(module, "__dict__", None), dict) and is_packed(module):
            return module
        module = getattr(module, "_orig_mod", None) or getattr(module, "module", None)
        if module is None:
            return None
    return None


def request_refusal(stage, batch) -> str | None:
    """Why a packed DiT cannot serve ``batch``: step caching or Cache-DiT."""
    params = batch.sampling_params
    if getattr(params, "enable_spectrum", False):
        return "Spectrum (step caching)"
    if getattr(params, "enable_teacache", False):
        return "TeaCache (step caching)"
    enable = getattr(params, "enable_cache_dit", None)
    quality = getattr(params, "quality", "lossless")
    if enable is not False and (
        enable or quality == "high" or stage._cache_dit_requested_for_batch(batch)
    ):
        return "Cache-DiT"
    return None


def _request_id(batch) -> str:
    return str(getattr(batch, "request_id", None))


def _current_request_id() -> str | None:
    """The request of the forward context's batch, if it has one."""
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        get_forward_context_or_none,
    )

    context = get_forward_context_or_none()
    batch = None if context is None else context.forward_batch
    return None if batch is None else _request_id(batch)


def _vouch_summary(tally: tuple[int, int] | None, calls: int) -> str:
    """The request log's vouching part ("" without vouching)."""
    if tally is None:
        return ""
    redone, ran = tally
    return (
        f"; {ran / max(calls, 1):.1f} heads a step at a fixed max beyond the "
        f"proven layers, {redone} step(s) recomputed with the proven layers only"
    )


_dump_counts: dict = {}


def _maybe_dump(model, kwargs: dict, out) -> None:
    """Debug (SGLANG_TPU_DEBUG_H3_DUMP_DIR): save a request's first DiT
    forwards' inputs and velocities from world rank 0."""
    root = envs.SGLANG_TPU_DEBUG_H3_DUMP_DIR.get()
    if not root:
        return
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_world_rank,
    )
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        get_forward_context_or_none,
    )

    if get_world_rank() != 0:
        return
    context = get_forward_context_or_none()
    batch = context.forward_batch if context is not None else None
    request = _request_id(batch) if batch is not None else "none"
    step = _dump_counts.get(request, 0)
    _dump_counts[request] = step + 1
    if step >= envs.SGLANG_TPU_DEBUG_H3_DUMP_STEPS.get():
        return
    data = {
        name: kwargs[name].detach().float().cpu()
        for name in ("x", "audio_x", "unique_timesteps", "prompt_embeds")
        if isinstance(kwargs.get(name), torch.Tensor)
    }
    data["video"], data["audio"] = (t.detach().float().cpu() for t in out)
    psp = kwargs.get("packed_seq_params")
    cu_seqlens = (
        psp.get("cu_seqlens_q")
        if isinstance(psp, dict)
        else getattr(psp, "cu_seqlens_q", None)
    )
    if isinstance(cu_seqlens, torch.Tensor):
        data["cu_seqlens"] = cu_seqlens.detach().cpu()
    data["meta"] = {
        "tag": "mk" if is_packed(model) else "torch",
        "request_id": request,
        "step": step,
        "timestep": None if context is None else context.current_timestep,
        "warmup": bool(getattr(batch, "is_warmup", False)),
    }
    os.makedirs(root, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", request)[:80]
    path = os.path.join(root, f"h3_{data['meta']['tag']}_{safe}_step{step:02d}.pt")
    torch.save(data, path)
    logger.warning("MiniMax H3 megakernel debug: dumped %s", path)


def _packed_error(what: str) -> ValueError:
    return ValueError(
        "MiniMax H3 megakernel packed the DiT blocks' weights at load; "
        f"{what} cannot reach them. Restart without "
        "SGLANG_TPU_ENABLE_H3_MEGAKERNEL to use it."
    )


def _refuse_when_packed(method, what: str):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        if is_packed(self):
            raise _packed_error(what)
        return method(self, *args, **kwargs)

    return wrapped


def install() -> None:
    """Route ``MiniMaxH3DiTModel``'s block stack through the kernel when every
    rank can run it (decided after the weights load); with the dump directory
    set, save each request's first DiT forwards."""
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as dit
    from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import (
        LoRAPipeline,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
        MiniMaxH3DenoisingStage,
    )

    cls = dit.MiniMaxH3DiTModel
    if getattr(cls, "_tpu_h3_megakernel_installed", False):
        return
    post_load_weights = cls.post_load_weights
    torch_forward = cls.forward
    prepare_adaln_plans = cls.prepare_adaln_plans
    reject_lora = LoRAPipeline._reject_lora_on_packed_weights
    run_full_loop = MiniMaxH3DenoisingStage._run_full_loop

    @functools.wraps(post_load_weights)
    def _post_load_weights(self) -> None:
        post_load_weights(self)
        if not _dump_only():
            enable(self)

    @functools.wraps(torch_forward)
    def _forward(self, **kwargs):
        stack = self.__dict__.get(_STATE)
        if isinstance(stack, H3MegakernelStack):
            stack.adaln_step(kwargs.get("unique_timesteps"))
        out = torch_forward(self, **kwargs)
        _maybe_dump(self, kwargs, out)
        return out

    @functools.wraps(prepare_adaln_plans)
    def _prepare_adaln_plans(self, step_timesteps):
        # Every step's timesteps are staged: the stack's batched AdaLN rows.
        slots = prepare_adaln_plans(self, step_timesteps)
        stack = self.__dict__.get(_STATE)
        if slots is None and isinstance(stack, H3MegakernelStack):
            stack.precompute_adaln(self, step_timesteps)
        return slots

    @functools.wraps(reject_lora)
    def _reject_lora_on_packed_weights(self) -> None:
        # set_lora and convert_to_lora_layers check here before any layer is
        # replaced.
        for name in ("transformer", "transformer_2"):
            module = self.modules.get(name)
            if module is not None and _packed_dit(module) is not None:
                raise _packed_error("LoRA")
        reject_lora(self)

    @functools.wraps(run_full_loop)
    def _run_full_loop(self, batch, server_args) -> None:
        model = _packed_dit(self.transformer)
        if model is None:
            return run_full_loop(self, batch, server_args)
        # Before the loop: Cache-DiT would mount on the blocks first thing.
        refusal = request_refusal(self, batch)
        if refusal is not None:
            raise ValueError(
                f"MiniMax H3 megakernel: a request with {refusal} cannot run "
                "on the packed DiT blocks; send it without, or restart without "
                "SGLANG_TPU_ENABLE_H3_MEGAKERNEL"
            )
        stack = model.__dict__[_STATE]
        start = stack.calls
        # The vouched heads live from here to the end of this request's loop.
        stack.begin_request(_request_id(batch))
        try:
            return run_full_loop(self, batch, server_args)
        finally:
            # After the loop: the one host read of the tally comes just before
            # the stage's own output check waits for the device. The tally
            # follows the "(warmup)" mark, which log readers match.
            tally = stack.end_request()
            logger.info(
                "MiniMax H3 megakernel: request %s ran the block-stack op %d times%s%s",
                _request_id(batch),
                stack.calls - start,
                " (warmup)" if getattr(batch, "is_warmup", False) else "",
                _vouch_summary(tally, stack.calls - start),
            )

    cls.post_load_weights = _post_load_weights
    if envs.SGLANG_TPU_DEBUG_H3_DUMP_DIR.get() or _precompute_adaln():
        cls.forward = _forward
    if _precompute_adaln():
        cls.prepare_adaln_plans = _prepare_adaln_plans
    cls.validate_weight_update_source = _refuse_when_packed(
        cls.validate_weight_update_source, "weight updates"
    )
    cls.validate_lora_layers = _refuse_when_packed(cls.validate_lora_layers, "LoRA")
    LoRAPipeline._reject_lora_on_packed_weights = _reject_lora_on_packed_weights
    MiniMaxH3DenoisingStage._run_full_loop = _run_full_loop
    cls._tpu_h3_megakernel_installed = True
