# SPDX-License-Identifier: Apache-2.0
"""PCP (Prefill-Context-Parallel) mesh helpers.

Builds a JAX Mesh whose device ordering mirrors the sglang attention CP
c10d group's rank ordering so ``pltpu.make_async_remote_copy`` operates
on the intended torus dimension. Consumed by the PCP streaming RPA
kernel adapter (``attention_pallas_pcp_streaming_op.py``).

``get_pcp_group()`` returns sglang's ``get_attn_cp_group()`` — a c10d
subgroup covering CP-sharded ranks with ``.device_group`` /
``.rank_in_group`` / ``.world_size`` / ``.all_gather`` / ``.all_reduce``.
``PcpGroupLayout`` is a ``msgspec.Struct`` per
.claude/rules/no-dataclasses.md.
"""

from __future__ import annotations

from typing import Any, Optional

import msgspec
import torch


class PcpGroupLayout(msgspec.Struct, frozen=True):
    """PCP process group order plus the matching TPU global device ids."""

    ranks: tuple[int, ...]
    device_ids: tuple[int, ...]
    rank_in_group: int
    world_size: int

    @property
    def device_id(self) -> int:
        return self.device_ids[self.rank_in_group]

    @property
    def prev_rank_in_group(self) -> int:
        return (self.rank_in_group - 1) % self.world_size

    @property
    def next_rank_in_group(self) -> int:
        return (self.rank_in_group + 1) % self.world_size


_LAYOUT_CACHE: dict[tuple[int, tuple[int, ...], int, int], PcpGroupLayout] = {}
_MESH_CACHE: dict[tuple[str, tuple[int, ...]], Any] = {}


def get_pcp_group() -> Optional[Any]:
    """Return sglang's attention-CP subgroup, or None before distributed init.

    Wraps ``get_attn_cp_group`` from sglang.srt.distributed.parallel_state
    which returns a ``GroupCoordinator`` with the interface the helpers
    depend on (``ranks``, ``rank_in_group``, ``world_size``,
    ``device_group``, ``cpu_group``, ``all_gather``, ``all_reduce``).

    Import-time failures are propagated (an ImportError here would be a
    real refactor bug in ``parallel_state``); only the runtime pre-init
    ``AssertionError`` is swallowed and translated to ``None``.
    """
    from sglang.srt.distributed.parallel_state import get_attn_cp_group as _get_group

    try:
        return _get_group()
    except AssertionError:
        return None


def get_pcp_rank() -> int:
    group = get_pcp_group()
    return 0 if group is None else int(group.rank_in_group)


def get_pcp_world_size() -> int:
    group = get_pcp_group()
    return 1 if group is None else int(group.world_size)


def all_gather_equal_tokens(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Gather equal-shaped tensors across the attention CP group."""
    group = get_pcp_group()
    if group is None or int(group.world_size) == 1:
        return tensor
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    gathered = group.all_gather(tensor, dim=dim)
    if isinstance(gathered, torch.Tensor):
        return gathered
    return torch.cat(list(gathered), dim=dim)


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Sum a tensor across the attention CP group."""
    group = get_pcp_group()
    if group is None or int(group.world_size) == 1:
        return tensor
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return group.all_reduce(tensor)


def _get_current_global_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def _get_tpu_global_device_id() -> int:
    # Only the narrow set of exceptions raised BEFORE TorchTPU/PJRT is
    # initialized should trigger the fallback path. Any other exception
    # (e.g. a real refactor bug in ``tpu_distributed.global_device_id``)
    # must surface — silently returning 0 would corrupt the PCP device
    # ordering across ranks and produce mysteriously wrong outputs.
    try:
        from torch_tpu._internal.distributed import tpu_distributed
    except ImportError:
        return _fallback_tpu_global_device_id()

    try:
        return int(tpu_distributed.global_device_id())
    except (AssertionError, RuntimeError):
        # TorchTPU present but PJRT not initialized yet — unit tests exercise
        # layout code in this state.
        return _fallback_tpu_global_device_id()


def _fallback_tpu_global_device_id() -> int:
    """JAX local-device fallback for pre-PJRT unit-test contexts.

    Never used at server startup (torch_tpu is available and initialized by
    the launcher); only reachable from single-process tests that construct
    a PCP mesh before ``TorchTPU`` PJRT is up.
    """
    try:
        import jax
    except ImportError:
        return 0
    local_devices = jax.local_devices()
    if not local_devices:
        return 0
    return int(local_devices[0].id)


def _collect_rank_to_device_id(
    group: Any, global_rank: int, device_id: int
) -> dict[int, int]:
    gathered: list[Optional[tuple[int, int]]] = [None] * int(group.world_size)
    torch.distributed.all_gather_object(
        gathered, (global_rank, device_id), group=group.cpu_group
    )
    rank_to_device_id: dict[int, int] = {}
    for item in gathered:
        if item is None:
            raise RuntimeError("PCP device id gather returned an empty entry.")
        rank, gathered_device_id = item
        rank_to_device_id[int(rank)] = int(gathered_device_id)
    return rank_to_device_id


def get_pcp_group_layout() -> PcpGroupLayout:
    """Return the current PCP group's rank order and TPU device-id order.

    The order of ``device_ids`` mirrors ``get_pcp_group().ranks``. This is
    the order a PCP Pallas mesh must use so TP+PCP layouts form one PCP
    ring per TP lane and ``pltpu.make_async_remote_copy`` runs on adjacent
    ICI torus links.
    """
    group = get_pcp_group()
    global_rank = _get_current_global_rank()
    device_id = _get_tpu_global_device_id()
    if group is None or int(group.world_size) == 1:
        return PcpGroupLayout(
            ranks=(global_rank,),
            device_ids=(device_id,),
            rank_in_group=0,
            world_size=1,
        )

    ranks = tuple(int(rank) for rank in group.ranks)
    rank_in_group = int(group.rank_in_group)
    cache_key = (global_rank, ranks, rank_in_group, device_id)
    cached = _LAYOUT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rank_to_device_id = _collect_rank_to_device_id(group, global_rank, device_id)
    missing = [rank for rank in ranks if rank not in rank_to_device_id]
    if missing:
        raise RuntimeError(
            "Could not map all PCP ranks to TPU global device ids: "
            f"missing={missing}, ranks={ranks}, gathered={rank_to_device_id}"
        )
    device_ids = tuple(rank_to_device_id[rank] for rank in ranks)
    layout = PcpGroupLayout(
        ranks=ranks,
        device_ids=device_ids,
        rank_in_group=rank_in_group,
        world_size=int(group.world_size),
    )
    _LAYOUT_CACHE[cache_key] = layout
    return layout


def get_or_create_pcp_mesh(axis_name: str = "pcp") -> Any:
    """Build a JAX Mesh ordered by the current attention CP group.

    Primary path — 1-D ``(pcp,)`` mesh when ``attn_tp_size == 1`` (the
    common ``attn_cp_size == tp_size`` production config, and the
    single-device unit-test case). The 2-D fallback below was originally
    the only path; it silently corrupted the PCP kernel's ring targeting
    when
    ``attn_tp > 1`` because ``pl.DeviceIdType.MESH`` interactions with the
    replicated ``attn_tp`` axis produced wrong device coordinates.

    Fallback — 2-D ``(attn_tp, pcp)`` mesh when both TP and CP > 1. Under
    sglang's TPU worker layout each Python process owns one TPU chip
    (``TPU_VISIBLE_CHIPS`` per rank), but ``jax.devices()`` still returns
    all ``tp_size`` chips (single-controller SPMD; per-process
    ``num_partitions == tp_size``). A shard_map with a mesh smaller than
    ``num_partitions`` triggers ``num_partitions == 1 || n == num_partitions``
    in the XLA verifier. Building a full ``attn_tp * attn_cp_size`` mesh
    satisfies the check; ``shard_map`` shards only on the ``pcp`` axis and
    XLA emits one all_to_all replica-group per ``attn_tp`` slice.
    """
    from sglang.srt.distributed.parallel_state import get_attn_tp_group

    layout = get_pcp_group_layout()
    attn_tp_group = get_attn_tp_group()
    attn_tp_size = int(attn_tp_group.world_size)

    # 1-D mesh path — applies when attn_tp_size == 1 (i.e.,
    # attn_cp_size == tp_size), so `layout.world_size
    # == num_partitions == tp_size` and the XLA verifier check
    # (`num_partitions == n`) passes with a single-axis mesh.
    # Also covers the single-device unit-test case.
    if layout.world_size == 1 or attn_tp_size == 1:
        cache_key = (axis_name, layout.device_ids)
        cached = _MESH_CACHE.get(cache_key)
        if cached is not None:
            return cached

        import jax
        import numpy as np
        from jax.sharding import Mesh

        devices_by_id = {int(device.id): device for device in jax.devices()}
        mesh_devices = np.asarray(
            [devices_by_id[device_id] for device_id in layout.device_ids]
        )
        mesh = Mesh(mesh_devices, axis_names=(axis_name,))
        _MESH_CACHE[cache_key] = mesh
        return mesh

    # Gather every CP group's device_ids across the attn_tp dimension so
    # each worker can assemble the same 2D mesh.
    gathered_cp_device_ids: list = [None] * attn_tp_size
    torch.distributed.all_gather_object(
        gathered_cp_device_ids,
        tuple(layout.device_ids),
        group=attn_tp_group.cpu_group,
    )
    # attn_tp_group rank == mesh's "tp" axis coordinate.
    cp_device_ids_by_tp: tuple[tuple[int, ...], ...] = tuple(
        tuple(int(x) for x in item) for item in gathered_cp_device_ids
    )
    cache_key = (axis_name, cp_device_ids_by_tp)
    cached = _MESH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import jax
    import numpy as np
    from jax.sharding import Mesh

    from sglang.srt.environ import envs

    if envs.SGLANG_TPU_PCP_DEBUG_MESH.get():
        print(
            f"[PCP MESH DEBUG] jax.devices()={len(jax.devices())} "
            f"jax.local_devices()={len(jax.local_devices())} "
            f"cp_device_ids_by_tp={cp_device_ids_by_tp} "
            f"attn_tp_rank={attn_tp_group.rank_in_group}",
            flush=True,
        )
    devices_by_id = {
        int(getattr(device, "id", idx)): device
        for idx, device in enumerate(jax.devices())
    }
    missing = sorted(
        {
            device_id
            for cp_devices in cp_device_ids_by_tp
            for device_id in cp_devices
            if device_id not in devices_by_id
        }
    )
    if missing:
        raise RuntimeError(
            "Cannot build PCP mesh because JAX does not expose all attn_tp x "
            f"pcp TPU device ids: missing={missing}, "
            f"cp_device_ids_by_tp={cp_device_ids_by_tp}, "
            f"jax_device_ids={sorted(devices_by_id)}"
        )
    # Shape (attn_tp, pcp): row r holds the CP group's device ordering for
    # attn_tp rank r; columns match ``layout.rank_in_group``.
    mesh_devices = np.asarray(
        [
            [devices_by_id[device_id] for device_id in cp_devices]
            for cp_devices in cp_device_ids_by_tp
        ]
    )
    mesh = Mesh(mesh_devices, axis_names=("attn_tp", axis_name))
    _MESH_CACHE[cache_key] = mesh
    return mesh
