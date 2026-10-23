"""Layer-addressed Raiden transfers from a pipeline to DP, TP, or matching PP stages."""

from __future__ import annotations

import concurrent.futures
import hashlib
import inspect
import json
import logging
import math
import os
import threading
import time
from types import SimpleNamespace
from typing import Any, Sequence

import msgspec

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.network import NetworkAddress
from sglang_tpu.disaggregation.raiden import pp_swa
from sglang_tpu.disaggregation.raiden.host_pool import (
    StagingAdmission,
    ranks_per_node,
)
from sglang_tpu.disaggregation.raiden.pipeline_final_chunks import FinalChunks
from sglang_tpu.disaggregation.raiden.pp_common import (
    _HEAD_MAJOR_KV_RANK,
    _kv_rank,
    _new_transfer_state,
    group_ranks,
    is_swa_pool,
    parse_req_id,
    part_destination_blocks,
    part_req_id,
    part_spans,
    part_uuid,
    pool_kind,
    stage_group,
    stage_req_id,
    stage_uuid,
)
from sglang_tpu.disaggregation.raiden.serde import (
    MessageTag,
    read_aux_blob,
    write_aux_blob,
)

logger = logging.getLogger(__name__)

# Error texts of tpu-sync kv_cache/reshard/request_block_registry.cc.
_REGISTRATION_CANCELLED = "Request block registration was cancelled"
_MISSING_REGISTRATION = "Missing producer block registration"
_EARLY_COMPLETION = "Only an empty producer registration may complete before"


def pool_manifest(groups: Any) -> list[dict]:
    """Name each local full-attention or sliding-window tensor by model layer."""
    pools = []
    for kind, tensors, layer_ids, num_draft in (
        ("fa", groups.full, groups.full_layer_ids, groups.num_draft_full),
        ("swa", groups.swa, groups.swa_layer_ids, groups.num_draft_swa),
    ):
        if len(tensors) != len(layer_ids):
            raise ValueError(f"Raiden {kind} pools need one global layer ID per tensor")
        num_target = len(tensors) - num_draft
        for index, (tensor, layer) in enumerate(zip(tensors, layer_ids)):
            # Draft layer IDs restart at 0, so they need their own namespace.
            tag_kind = kind if index < num_target else f"draft.{kind}"
            _kv_rank(tensor)
            size = math.prod(int(n) for n in tensor.shape[1:]) * tensor.element_size()
            pools.append(
                dict(
                    tag=f"{tag_kind}.l{int(layer)}",
                    storage_index=len(pools),
                    base_offset_bytes=0,
                    block_stride_bytes=size,
                    num_blocks=int(tensor.shape[0]),
                    dtype_tag=str(tensor.dtype).removeprefix("torch."),
                    regions=[
                        dict(
                            name="kv",
                            offset_bytes=0,
                            stride_bytes=size,
                            unit_bytes=size,
                            num_units=1,
                        )
                    ],
                )
            )
    if not groups.full:
        raise ValueError(
            "Raiden pipeline transfer needs full-attention KV in every stage"
        )
    if len({_kv_rank(tensor) for tensor in [*groups.full, *groups.swa]}) != 1:
        raise ValueError("Raiden pipeline KV layers must share one page layout")
    tags = [pool["tag"] for pool in pools]
    if len(tags) != len(set(tags)):
        raise ValueError("Raiden pipeline transfer requires distinct global layer IDs")
    return pools


def materialized_storage_layout(tensor, getter):
    """Read the layout of the live storage backing an FP8 cache dtype view."""
    layout = getter(tensor)
    if layout is None and str(tensor.dtype) in (
        "torch.float8_e4m3fn",
        "torch.float8_e5m2",
    ):
        import torch

        # An FP8 dtype view can retain uint8 base storage. Matching that dtype
        # lets the layout getter inspect the native storage reference.
        layout = getter(tensor.view(torch.uint8))
    return layout


def layout_fingerprint(pools: Sequence[dict], tensors: Sequence[Any], getter) -> str:
    """Compare per-class page geometry and materialized layout across layer
    subsets; the per-rank head count is left out so a tensor-parallel consumer
    matches its producer."""
    classes = {}
    schema = "sglang-fused-kv-v2"
    for pool, tensor in zip(pools, tensors):
        layout = materialized_storage_layout(tensor, getter)
        if layout is None:
            raise ValueError("Raiden pipeline KV must have a materialized TPU layout")
        major, tiles, bits = layout
        if _kv_rank(tensor) == _HEAD_MAJOR_KV_RANK:
            # Axis 1, the per-rank 2 * kv_heads, is left out.
            schema = "sglang-rpa-seq-kv-v1"
            page_shape = [int(tensor.shape[2]), int(tensor.shape[3])]
        else:
            page_shape = [
                int(tensor.shape[1]),
                int(tensor.shape[3]),
                int(tensor.shape[4]),
            ]
        signature = dict(
            page_shape=page_shape,
            dtype=pool["dtype_tag"],
            minor_to_major=[int(n) for n in major],
            tiles=[[int(n) for n in tile] for tile in tiles],
            element_bits=int(bits or tensor.element_size() * 8),
        )
        # A draft page matches its target class, so a stage without the draft
        # fingerprints like one with it.
        kind = pool_kind(pool["tag"]).removeprefix("draft.")
        if kind in classes and classes[kind] != signature:
            raise ValueError(f"Raiden {kind} layers have differing page layouts")
        classes[kind] = signature
    value = json.dumps(dict(schema=schema, classes=classes), sort_keys=True)
    return hashlib.sha256(value.encode()).hexdigest()


def destination_blocks(
    pools: Sequence[dict],
    full: list[int],
    swa: list[int],
    source_full_count: int,
    source_swa_count: int,
) -> dict:
    """Clip a cached prefix and omit the decode-only trailing token page."""
    tags, blocks, counts, skips = [], [], [], []
    for pool in pools:
        is_swa = is_swa_pool(pool["tag"])
        local = list(swa if is_swa else full)
        remote_count = source_swa_count if is_swa else source_full_count
        if len(local) > remote_count:
            local = local[:remote_count]
        if not local:
            continue
        tags.append(pool["tag"])
        blocks.extend(local)
        counts.append(len(local))
        skips.append((remote_count - len(local)) * pool["block_stride_bytes"])
    return dict(
        transfer_pool_tags=tags,
        dst_device_block_ids=blocks,
        dst_block_counts=counts,
        dst_skip_bytes=skips,
    )


def legacy_slot_host_bytes(tensors, max_blocks, getter):
    """Match the legacy native constructor's uniform host-storage page size."""
    first = tensors[0]
    layout = materialized_storage_layout(first, getter)
    if layout is None:
        raise ValueError("Legacy Raiden pools require materialized page layouts")
    major, tiles, _ = layout
    major = tuple(int(n) for n in major)
    if sorted(major) != list(range(len(first.shape))) or major[-1] != 0:
        raise ValueError("Legacy Raiden pools require block-major page layouts")
    page_bytes = math.prod(first.shape[1:]) * first.element_size()
    uniform = all(
        tuple(tensor.shape) == tuple(first.shape)
        and tensor.element_size() == first.element_size()
        for tensor in tensors
    )
    if not uniform and tiles:
        tile = tuple(int(n) for n in tiles[0])
        if len(tile) != 2 or min(tile) <= 0:
            raise ValueError("Legacy Raiden pools require a two-dimensional tile")
        height, width = tile
        minor_width, minor_height = major[:2]
        elements = (
            ((int(first.shape[minor_width]) + width - 1) // width)
            * ((int(first.shape[minor_height]) + height - 1) // height)
            * height
            * width
        )
        for index, dimension in enumerate(first.shape):
            if index not in (0, minor_width, minor_height):
                elements *= int(dimension)
        page_bytes = elements * first.element_size()
    return int(page_bytes) * len(tensors) * max_blocks


def legacy_mirror_slots(pools, slot_bytes: int, ceiling: int, budget: float):
    """Bound full mirrors plus one temporary replacement of an initial storage."""
    if {pool["storage_index"] for pool in pools} != set(range(len(pools))):
        raise ValueError("Legacy Raiden pipeline pools need one pool per storage")
    if any(pool["base_offset_bytes"] != 0 for pool in pools):
        raise ValueError("Legacy Raiden pipeline pools need zero storage offsets")
    extents = [pool["block_stride_bytes"] * pool["num_blocks"] for pool in pools]
    extents = [((extent + 4095) // 4096) * 4096 for extent in extents]
    initial_per_slot = slot_bytes // len(pools)
    for slots in range(ceiling, 0, -1):
        initial = ((initial_per_slot * slots + 4095) // 4096) * 4096
        final = sum(max(initial, extent) for extent in extents)
        # The native allocator creates and copies a replacement before freeing
        # that storage's original buffer; other storages retain their mirrors.
        peak = final + (initial if any(n > initial for n in extents) else 0)
        if peak <= budget:
            return slots, peak
    raise ValueError(
        "Raiden pipeline full mirrors need at least "
        f"{peak / 1024**3:.2f} GiB per rank including pool replacement, but "
        f"SGLANG_RAIDEN_HOST_POOL_GB allows {budget / 1024**3:.2f} GiB; "
        "raise the host budget, lower max-total-tokens, or use a wheel with "
        "bounded pool staging"
    )


def warn_if_staging_exceeds_premapped(staging_bytes: int, premapped_bytes: int) -> bool:
    """Full-mirror staging is carved out of libtpu's premapped DMA buffer; staging
    past the buffer is DMA-mapped as each copy lands, and those mappings stall the
    decode steps. Returns whether the staging exceeds the buffer."""
    if staging_bytes <= premapped_bytes:
        return False
    logger.warning(
        "Raiden host staging of %.1f GiB per rank exceeds libtpu's premapped DMA "
        "buffer of %.1f GiB; set TPU_PREMAPPED_BUFFER_SIZE above the staging size.",
        staging_bytes / 1024**3,
        premapped_bytes / 1024**3,
    )
    return True


class Receive(msgspec.Struct):
    full: list[int]
    swa: list[int]
    aux_index: int | None
    ranks: set[int]
    sources: dict[int, dict]
    future: concurrent.futures.Future | None = None
    accepted: bool = False
    deadline: float = 0.0
    endpoints: tuple[str, ...] = ()
    next_attempt: float = 0.0
    armed_until: float = 0.0
    # Early parts by index and stage groups by group: _new_transfer_state().
    parts: dict[int, dict] = {}
    groups: dict[int, dict] = {}
    expected_parts: int | None = None
    # Stage metadata by source stage, and the stages whose first-token
    # metadata arrived.
    stage: dict[int, dict] = {}
    aux_ranks: set[int] = set()
    # Staging reservations held: the room for the closing transfer,
    # (room, "p", index) per early part and (room, "g", group) per group.
    admitted_keys: set = set()
    # Window twins (pp_swa) of the closing transfer, key (), and of each
    # stage group, key ("g", group): _new_transfer_state() and their
    # coordinate arguments. A twin is staged with its full-attention transfer.
    twins: dict[tuple, dict] = {}


class StagingPlan(msgspec.Struct, frozen=True):
    slots: int
    mode: str
    options: dict = {}
    mirror_peak: int = 0
    native_slot_bytes: int = 0


class PipelineTransfer:
    """One named-pool engine per worker, sharing its pipeline's reshard
    controller; a hybrid pool's sliding-window layers get a second engine
    (pp_swa)."""

    # The window link, when sliding-window layers have their own engine.
    swa: pp_swa.SwaLink | None = None

    def __init__(self, manager, groups):
        from tpu_sync.api.torch.reshard_client import ReshardClient
        from tpu_sync.rpc import raiden_service_pb2

        self.manager = manager
        self.producer = manager.disaggregation_mode == DisaggregationMode.PREFILL
        self._check_support(raiden_service_pb2)
        self.hbm_mem_type = raiden_service_pb2.MEMORY_TYPE_HBM
        self.lock = threading.RLock()
        # A hybrid pool's sliding-window layers get an engine of their own.
        self.pools, swa_pools = pp_swa.split_manifest(pool_manifest(groups))
        self.tensors = list(groups.full)
        if swa_pools:
            pp_swa.require_bounded_staging()
        # One worker per pipeline stage or attention-TP rank. Rank 0 hosts the
        # shared controller and, on a tensor-parallel decode, plans every
        # transfer for all ranks.
        self.rank = manager.pp_rank * manager.attn_tp_size + manager.attn_tp_rank
        self.workers = manager.pp_size * manager.attn_tp_size
        self.client_type = ReshardClient
        self._init_overlap_state()
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1 if self.producer else self.coord_workers,
            thread_name_prefix="raiden-pipeline-transfer",
        )
        self.sent = {}
        self.replies = {}
        self.release_requests = {}
        self.receives: dict[int, Receive] = {}
        self.cleared = set()
        self.done_send = set()
        self.done_recv = set()
        self.failed = set()
        self.retired = set()
        self.controller, dispatch, swa_addresses = self._rendezvous(swa=bool(swa_pools))
        self.client = ReshardClient(self.controller)
        self.unit_fields = dict(
            job_name="sglang",
            job_replica_id=self.controller,
            data_name="prefill" if self.producer else "decode",
            data_replica_idx=self.rank,
        )
        self.unit = SimpleNamespace(**self.unit_fields)
        # A tensor-parallel decode receives one transfer into every rank's
        # engine; a stage or DP replica receives into its own unit only.
        self.dst_units = [
            SimpleNamespace(
                **(
                    self.unit_fields
                    | {"data_replica_idx": self.rank - manager.attn_tp_rank + rank}
                )
            )
            for rank in range(manager.attn_tp_size)
        ]
        self.max_blocks = manager._max_request_blocks()
        swa_slots = swa_blocks = reserved = 0
        if swa_pools:
            # The window engine stages the window pages; this one, full pages.
            swa_slots, swa_blocks, reserved = pp_swa.plan_slots(
                manager, list(groups.swa), self.tensors, self.max_blocks
            )
        staging = self._plan_staging(reserved_bytes=reserved)
        # A twin's window pages are admitted with its full-attention transfer.
        self.admission = StagingAdmission(
            staging.slots * self.max_blocks,
            swa_slots * swa_blocks if swa_pools else None,
        )
        self.engine = self._start_engine(
            tensors=self.tensors,
            max_blocks=self.max_blocks,
            slots=staging.slots,
            dispatch=dispatch,
        )
        fingerprint = self._register_pools(
            engine=self.engine,
            client=self.client,
            unit=self.unit,
            pools=self.pools,
            tensors=self.tensors,
            staging_options=staging.options,
        )
        if swa_pools:
            self.swa = pp_swa.build(
                self,
                pools=swa_pools,
                tensors=list(groups.swa),
                addresses=swa_addresses,
                slots=swa_slots,
                max_blocks=swa_blocks,
            )
        self._log_ready(staging=staging, fingerprint=fingerprint)

    def _init_overlap_state(self) -> None:
        """Early-part and stage-group configuration and state, both off by default."""
        self.early_parts = envs.SGLANG_RAIDEN_PIPELINE_EARLY_PARTS.get()
        self.part_min_pages = max(1, envs.SGLANG_RAIDEN_PIPELINE_PART_PAGES.get())
        self.stage_parts = (
            envs.SGLANG_RAIDEN_PIPELINE_STAGE_PARTS.get() and self.producer
        )
        self.stage_group_size = max(1, envs.SGLANG_RAIDEN_PIPELINE_STAGE_GROUP.get())
        self.coord_workers = max(1, envs.SGLANG_RAIDEN_PIPELINE_COORD_WORKERS.get())
        self.early_close = (
            envs.SGLANG_RAIDEN_PIPELINE_EARLY_CLOSE.get() and self.producer
        )
        if self.early_close and self.stage_parts:
            raise ValueError(
                "SGLANG_RAIDEN_PIPELINE_EARLY_CLOSE moves each stage's share in the "
                "closing transfer, which stage groups replace; unset "
                "SGLANG_RAIDEN_PIPELINE_STAGE_PARTS"
            )
        if self.early_close and get_parallel().pp_async_batch_depth < 1:
            raise ValueError(
                "SGLANG_RAIDEN_PIPELINE_EARLY_CLOSE needs --pp-async-batch-depth 1: "
                "the last stage closes its share on the deferred output send"
            )
        # Producer: this stage's final chunks until their KV is handed off.
        self.final_chunks = FinalChunks(self) if self.producer else None
        # A ReshardClient owns one socket, so each coordinating thread has its own.
        self._thread_clients = threading.local()
        # Producer: a room's early parts, and this stage's closing part.
        self.early: dict[int, dict] = {}
        self.stage: dict[int, dict] = {}
        self.fence_executor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="raiden-stage-fence"
            )
            if self.stage_parts
            else None
        )

    def _check_support(self, raiden_service_pb2) -> None:
        """Reject topologies and native builds the pipeline transfer cannot serve."""
        manager = self.manager
        if manager.attn_cp_size != 1:
            raise ValueError("Raiden pipeline transfer requires attention CP=1")
        if self.producer and (manager.attn_tp_size != 1 or manager.pp_size <= 1):
            raise ValueError(
                "Raiden pipeline transfer requires a pipeline producer with "
                "attention TP=1"
            )
        if manager.attn_tp_size != 1 and manager.pp_size != 1:
            raise ValueError(
                "Raiden pipeline transfer takes attention TP>1 only on a decode "
                "without pipeline stages"
            )
        if (
            "wire_pool_indices"
            not in raiden_service_pb2.StartTransferRequest.DESCRIPTOR.fields_by_name
        ):
            raise RuntimeError(
                "Raiden pipeline transfer requires tpu-sync PP pool-resharding support"
            )

    def _rendezvous(self, *, swa: bool) -> tuple[str, str, tuple[str, str] | None]:
        """Host the shared reshard controller on rank 0 and hand every worker its
        (reshard service, Raiden controller) addresses, plus the window store's
        when sliding-window layers have their own engine (``swa``)."""
        from tpu_sync.api.torch.reshard_store import RaidenId, ReshardStore

        manager = self.manager
        self.store = None
        self.swa_store = None
        addresses = None
        if self.rank == 0:

            def store(data_name: str):
                return ReshardStore(
                    RaidenId(
                        job_name="sglang",
                        job_replica_id=f"{manager.local_ip}:{manager._control_port}",
                        data_name=data_name,
                        data_replica_idx=0,
                    ),
                    store_server_ip=manager.local_ip,
                )

            def store_addresses(owned) -> tuple[str, str]:
                return (
                    NetworkAddress(
                        manager.local_ip, owned.reshard_service_port
                    ).to_host_port_str(),
                    owned.raiden_controller_address,
                )

            self.store = store("pipeline")
            addresses = store_addresses(self.store)
            if swa:
                with pp_swa.private_native_servers():
                    self.swa_store = store("pipeline-swa")
                addresses += store_addresses(self.swa_store)
        if manager.pp_size > 1:
            addresses = get_parallel().pp_group.broadcast_object(addresses, src=0)
        if manager.attn_tp_size > 1:
            addresses = get_parallel().attn_tp_group.broadcast_object(addresses, src=0)
        # Rank 0 decides from its own layers; every worker must agree with it.
        if swa != (len(addresses) == 4):
            raise ValueError(
                "Raiden pipeline workers disagree on the sliding-window engine: "
                f"rank {self.rank} has {'' if swa else 'no '}sliding-window "
                f"layers but rank 0 has {'none' if swa else 'some'}; with hybrid "
                "SWA memory every pipeline stage needs sliding-window layers, so "
                "pick a stage layout that gives each stage some, or "
                f"{pp_swa.DISABLE_HYBRID_SWA}"
            )
        return addresses[0], addresses[1], (addresses[2:] if swa else None)

    def _plan_staging(self, *, reserved_bytes: int) -> StagingPlan:
        """Stage through bounded per-pool leases when the installed Raiden
        supports them, and through checked full mirrors otherwise;
        ``reserved_bytes`` of the budget belong to the window engine."""
        from tpu_sync.api.torch.kv_cache_manager import KVCacheManager

        slots = self.manager._num_slots(
            self.tensors, self.max_blocks, reserved_bytes=reserved_bytes
        )
        parameters = inspect.signature(KVCacheManager.register_pools).parameters
        if not {"staging_leases", "staging_blocks_per_pool"} <= parameters.keys():
            return self._plan_full_mirrors(slots=slots, parameters=parameters)
        return StagingPlan(
            slots=slots,
            mode="leases",
            options=dict(
                staging_leases=slots,
                staging_blocks_per_pool=[self.max_blocks] * len(self.pools),
            ),
        )

    def _plan_full_mirrors(self, *, slots: int, parameters) -> StagingPlan:
        """Bound legacy full host mirrors by this rank's share of the host budget."""
        from torch_tpu._internal.compile import tpu_torch_compile

        manager = self.manager
        if os.environ.get("RAIDEN_SHM_KEY"):
            raise ValueError(
                "Legacy pipeline full mirrors do not support RAIDEN_SHM_KEY"
            )
        if set(parameters) != {"self", "pools"}:
            raise RuntimeError("Unsupported Raiden register_pools signature")
        from torch_tpu._internal.sync.sync import synchronize

        synchronize(self.tensors, wait=True)
        initial_slot_bytes = legacy_slot_host_bytes(
            self.tensors,
            self.max_blocks,
            tpu_torch_compile.get_device_layout_if_materialized,
        )
        local_ranks = ranks_per_node(
            manager.attn_tp_size,
            manager.attn_dp_size,
            manager.system_dp_size,
            get_parallel().nnodes,
            manager.attn_cp_size,
            manager.pp_size,
        )
        budget = (
            float(os.environ.get("SGLANG_RAIDEN_HOST_POOL_GB", "128"))
            * 1024**3
            / local_ranks
        )
        slots, mirror_peak = legacy_mirror_slots(
            pools=self.pools,
            slot_bytes=initial_slot_bytes,
            ceiling=slots,
            budget=budget,
        )
        premapped = os.environ.get("TPU_PREMAPPED_BUFFER_SIZE")
        if premapped is not None:
            warn_if_staging_exceeds_premapped(
                staging_bytes=mirror_peak, premapped_bytes=int(premapped)
            )
        return StagingPlan(
            slots=slots,
            mode="full_mirrors",
            mirror_peak=mirror_peak,
            native_slot_bytes=initial_slot_bytes,
        )

    def _start_engine(
        self,
        *,
        tensors: list[Any],
        max_blocks: int,
        slots: int,
        dispatch: str,
        port_offset: int = 0,
    ):
        from tpu_sync.api.torch.kv_cache_manager import KVCacheManager

        manager = self.manager
        return KVCacheManager(
            kv_caches=tensors,
            node_id=manager._local_rank,
            local_control_port=manager._control_port + port_offset,
            max_blocks=max_blocks,
            num_slots=slots,
            timeout_s=manager._timeout_s,
            listener_port=0,
            parallelism=self.workers,
            raiden_controller_address=dispatch,
            worker_id=f"worker_{self.rank}",
        )

    def _register_pools(
        self, *, engine, client, unit, pools, tensors, staging_options: dict
    ) -> str:
        """Register the named layer pools natively and this worker's unit with
        the controller; returns the layout fingerprint peers must match."""
        from torch_tpu._internal.compile import tpu_torch_compile

        engine.register_pools(pools, **staging_options)
        fingerprint = layout_fingerprint(
            pools,
            tensors,
            tpu_torch_compile.get_device_layout_if_materialized,
        )
        client.register_work_unit(
            unit=unit,
            shards=[engine.transfer_address],
            control_plane_rpc_address=engine.listener_address,
            pool_manifest=pools,
            layout_fingerprint=fingerprint,
            page_tokens=self.manager.kv_args.page_size,
            transfer_parallelism=self.workers,
            transfer_rank=self.rank,
        )
        return fingerprint

    def _log_ready(self, *, staging: StagingPlan, fingerprint: str) -> None:
        manager = self.manager
        logger.info(
            "Raiden pipeline ready: mode=%s pp=%d/%d tp=%d/%d local_rank=%d "
            "page_tokens=%d staging=%s slots=%d native_slot_bytes=%d mirror_peak_bytes=%d "
            "layout_fingerprint=%s pools=%s",
            "prefill" if self.producer else "decode",
            manager.pp_rank,
            manager.pp_size,
            manager.attn_tp_rank,
            manager.attn_tp_size,
            manager._local_rank,
            manager.kv_args.page_size,
            staging.mode,
            staging.slots,
            staging.native_slot_bytes,
            staging.mirror_peak,
            fingerprint,
            json.dumps(self.pools, sort_keys=True),
        )

    def _send(self, endpoint, frames):
        self.manager._send_multipart_locked(endpoint, frames)

    @staticmethod
    def _validate_room(room: int) -> None:
        if not 0 < room < (1 << 63):
            raise ValueError(
                "Raiden pipeline bootstrap_room must be a positive signed 64-bit "
                "integer for the native transfer UUID"
            )

    # ----- producer: early parts -------------------------------------------

    def offer_part(self, room: int, start: int, block_ids: list[int]) -> None:
        """Offer full-attention pages that are final before the prefill ends.

        Pages gather until they make part_min_pages; each such run registers
        as its own native transfer once the decode has named the room, and the
        decode pulls it while the pipeline keeps computing. Pages still
        gathering when the prefill ends ride the closing transfer.
        """
        if not self.early_parts or not block_ids:
            return
        self._validate_room(room)
        mgr = self.manager
        with self.lock:
            if (
                mgr.is_aborted(room)
                or room in self.sent
                or room in self.stage
                or room in self.failed
            ):
                return
            state = self.early.setdefault(room, dict(offered=0, pending=None, parts=[]))
            p_start, p_blocks = state["pending"] or (start, [])
            if p_start + len(p_blocks) != start:
                raise ValueError(
                    f"Raiden pipeline room {room} offered pages out of order"
                )
            p_blocks = p_blocks + [int(b) for b in block_ids]
            if len(p_blocks) < self.part_min_pages:
                state["pending"] = (p_start, p_blocks)
                return
            state["pending"] = None
            part = dict(
                index=len(state["parts"]),
                start=p_start,
                full=p_blocks,
                registered=False,
                replied=set(),
                done=False,
                expires=time.monotonic() + mgr._timeout_s,
            )
            state["parts"].append(part)
            state["offered"] = p_start + len(p_blocks)
            self._reply_part(room, part)

    def _dst_attn_tp(self, room) -> int:
        widths = {int(meta["attn_tp_size"]) for meta in self.replies[room].values()}
        if len(widths) != 1:
            raise ValueError(
                "Raiden pipeline consumers of one request disagree on attention TP"
            )
        return widths.pop()

    def _register_blocks(
        self,
        room,
        *,
        req_id: str,
        uuid: int,
        full: list[int],
        swa: list[int],
        with_swa: bool,
    ) -> bool:
        """Register one transfer's pages, sliced for the consumer's attention
        TP, and its window twin; a registration that cannot be made fails the
        room."""
        try:
            dst_attn_tp = self._dst_attn_tp(room)
            pack = getattr(self.manager, "_kv_pack_heads", 1)
            if pack > 1 and dst_attn_tp != self.manager.attn_tp_size:
                # TODO(#829): head_slice_spans treats a packed row as one head.
                raise ValueError(
                    f"KV pools that pack {pack} heads per row need equal attention "
                    "TP on both sides; unset SGLANG_TPU_KV_PACK_HEADS"
                )
            spans = part_spans(
                self.pools,
                self.tensors,
                full,
                swa,
                dst_attn_tp,
                int(self.manager.kv_args.total_kv_head_num),
                int(self.manager.kv_args.total_swa_kv_head_num),
                with_swa=with_swa,
            )
            self.client.register_request_blocks(
                req_id=req_id,
                uuid=uuid,
                unit=self.unit,
                block_ids=full,
                pool_spans=spans,
            )
            if with_swa:
                pp_swa.register_twin(
                    self, req_id=req_id, uuid=uuid, swa=swa, dst_attn_tp=dst_attn_tp
                )
        except Exception as error:
            self._fail_room(
                room, f"Raiden pipeline registration of {req_id} failed: {error}"
            )
            return False
        return True

    def _source_metadata(self, **fields) -> bytes:
        mgr = self.manager
        return json.dumps(
            dict(
                controller=self.controller,
                unit=self.unit_fields,
                rank=mgr.pp_rank,
                pp_size=mgr.pp_size,
                pool_tags=self._pool_tags(),
                **fields,
            )
        ).encode()

    def _pool_tags(self) -> list[str]:
        """Every layer tag of this worker, on either engine."""
        tags = [pool["tag"] for pool in self.pools]
        return tags + (self.swa.tags if self.swa is not None else [])

    def _announce(self, room, item: dict, payload: bytes, aux: bytes, key: str) -> None:
        """Send a PIPELINE_READY frame to every consumer not yet told."""
        for endpoint in self.replies[room]:
            if endpoint in item[key]:
                continue
            self._send(
                endpoint, [MessageTag.PIPELINE_READY, str(room).encode(), payload, aux]
            )
            item[key].add(endpoint)

    def _reply_part(self, room, part) -> None:
        if not self.replies.get(room) or room in self.failed:
            return
        if not part["registered"]:
            if not self._register_blocks(
                room,
                req_id=part_req_id(room, part["index"]),
                uuid=part_uuid(room, part["index"]),
                full=part["full"],
                swa=[],
                with_swa=False,
            ):
                return
            part["registered"] = True
        payload = self._source_metadata(
            part=part["index"], start=part["start"], full=len(part["full"])
        )
        self._announce(room, part, payload, b"", "replied")

    # ----- producer: stage groups ------------------------------------------

    def make_fence(self, output: Any) -> concurrent.futures.Future:
        """A completion fence for the forward that produced ``output``.

        One element of the output is copied to the host on a worker thread; the
        copy can only finish once the producing execution, which also wrote
        this stage's KV, has completed. torch_tpu's materialization query is
        true as soon as the output buffer exists, so it cannot fence the writes.
        """
        probe = output.reshape(-1)[:1]
        return self.fence_executor.submit(probe.cpu)

    def stage_ready(
        self, room: int, full: list[int], swa: list[int], fence: Any
    ) -> None:
        """This stage launched the room's final chunk: full/swa are its page
        ids here and fence the make_fence() future of that forward."""
        if not self.stage_parts:
            return
        self._validate_room(room)
        mgr = self.manager
        with self.lock:
            if mgr.is_aborted(room) or room in self.stage or room in self.failed:
                return
            self.stage[room] = self._new_stage_entry(full, swa, fence)
            self._advance_stage(room)

    def advance_stages(self) -> None:
        """Register every stage part whose fence has completed. The scheduler
        calls this on each launch: a request reaches the transfer queue's poll
        only after its result point, which is too late."""
        if not self.stage_parts:
            return
        with self.lock:
            for room, entry in list(self.stage.items()):
                if not entry["ready"] and room not in self.failed:
                    self._advance_stage(room)

    def _new_stage_entry(self, full, swa, fence) -> dict:
        return dict(
            full=[int(b) for b in full],
            swa=[int(b) for b in swa],
            fence=fence,
            ready=False,
            registered=False,
            replied=set(),
            aux=None,
            aux_replied=set(),
            done=False,
            expires=time.monotonic() + self.manager._timeout_s,
        )

    def _advance_stage(self, room) -> None:
        entry = self.stage[room]
        if not entry["ready"]:
            fence = entry["fence"]
            # A probe that failed never fences: the part waits for finalize.
            if fence is not None and (not fence.done() or fence.exception()):
                return
            entry["fence"] = None
            entry["ready"] = True
            state = self.early.get(room)
            start = state["offered"] if state else 0
            if entry["full"] and start >= len(entry["full"]):
                # The closing transfer keeps one full-attention page.
                start = len(entry["full"]) - 1
            entry["start"] = start
            entry["parts"] = len(state["parts"]) if state else 0
            if state:
                # Pages still gathering for an early part move with this one.
                state["pending"] = None
        self._reply_stage(room)

    def _reply_stage(self, room) -> None:
        entry = self.stage[room]
        if not entry["ready"] or not self.replies.get(room) or room in self.failed:
            return
        mgr = self.manager
        group = stage_group(mgr.pp_rank, self.stage_group_size)
        if not entry["registered"]:
            full = entry["full"][entry["start"] :]
            if not self._register_blocks(
                room,
                req_id=stage_req_id(room, group),
                uuid=stage_uuid(room, group),
                full=full,
                swa=entry["swa"],
                with_swa=True,
            ):
                return
            entry["registered"] = True
        payload = self._source_metadata(
            stage=True,
            group=group,
            group_ranks=group_ranks(group, self.stage_group_size, mgr.pp_size),
            start=entry["start"],
            full=len(entry["full"]) - entry["start"],
            swa=len(entry["swa"]),
            parts=entry["parts"],
            **pp_swa.source_fields(self),
        )
        self._announce(room, entry, payload, b"", "replied")
        if entry["aux"] is not None:
            aux_meta = json.dumps(
                dict(rank=mgr.pp_rank, pp_size=mgr.pp_size, aux_only=True)
            ).encode()
            self._announce(room, entry, aux_meta, entry["aux"], "aux_replied")

    def _early_settled(self, room) -> bool:
        state = self.early.get(room)
        return state is None or all(part["done"] for part in state["parts"])

    def _expire_parts(self) -> None:
        """Cancel unclaimed early and stage parts of aborted, failed or expired rooms."""
        mgr = self.manager
        now = time.monotonic()
        group = stage_group(mgr.pp_rank, self.stage_group_size)
        for room, entry in self.stage.items():
            if not entry["done"] and self._cancel_unclaimed(
                room,
                entry,
                expired=now >= entry["expires"],
                req_id=stage_req_id(room, group),
                uuid=stage_uuid(room, group),
            ):
                entry["fence"] = None
        for room, state in self.early.items():
            for part in state["parts"]:
                if not part["done"]:
                    self._cancel_unclaimed(
                        room,
                        part,
                        expired=now >= part["expires"],
                        req_id=part_req_id(room, part["index"]),
                        uuid=part_uuid(room, part["index"]),
                    )

    def _cancel_unclaimed(
        self, room, item: dict, *, expired: bool, req_id, uuid
    ) -> bool:
        """Settle a producer part of a dead room unless the decode claimed it,
        in which case the native transfer settles it."""
        aborted = room in self.manager._aborted
        if not (aborted or expired or room in self.failed):
            return False
        if not pp_swa.cancel_unclaimed(
            self, req_id, uuid, registered=item["registered"]
        ):
            return False
        item["done"] = True
        if aborted or expired:
            self.failed.add(room)
        return True

    # ----- producer: closing transfer --------------------------------------

    def _closing_entry(self, room, full: list[int], swa: list[int], aux: bytes):
        """The room's closing transfer: the pages its early parts left, the
        window and the first-token metadata. Hold the manager's state lock."""
        state = self.early.get(room)
        start = state["offered"] if state else 0
        if full and start >= len(full):
            # Keep the full-attention page the decode's completion
            # is keyed on in the closing transfer.
            start = len(full) - 1
        # The pages register with the consumer's head geometry, so
        # registration waits for the first consumer request.
        return dict(
            full=full[start:],
            start=start,
            parts=len(state["parts"]) if state else 0,
            swa=swa,
            aux=aux,
            expires=time.monotonic() + self.manager._timeout_s,
            registered=False,
            replied=set(),
        )

    def close_early(
        self, room: int, full: list[int], swa: list[int], aux: bytes = b""
    ) -> None:
        """SGLANG_RAIDEN_PIPELINE_EARLY_CLOSE: register and announce this
        stage's share of the closing transfer before its result point. Its
        pages may still be written: tpu-sync reads registered pages only when
        the decode coordinates the transfer (request_block_registry.cc), which
        it does once every stage has announced."""
        self._validate_room(room)
        mgr = self.manager
        with self.lock:
            with mgr._state_lock:
                if room in mgr._aborted or room in self.sent or room in self.failed:
                    return
                self.sent[room] = self._closing_entry(
                    room, [int(b) for b in full], [int(b) for b in swa], aux
                )
            self._reply(room)

    def finalize_send(self, room: int, aux_index: int | None) -> bool:
        self._validate_room(room)
        mgr = self.manager
        late_aux = False
        mismatch = None
        with self.lock:
            with mgr._state_lock:
                if room in mgr._aborted:
                    return False
                full = list(mgr._prefill.send_blocks.get(room, []))
                swa = list(mgr._prefill.send_swa_blocks.get(room, []))
                aux = (
                    b""
                    if aux_index is None
                    else read_aux_blob(
                        mgr.kv_args.aux_data_ptrs, mgr.kv_args.aux_item_lens, aux_index
                    )
                )
                if self.stage_parts:
                    entry = self.stage.get(room)
                    if entry is None:
                        entry = self.stage[room] = self._new_stage_entry(
                            full, swa, None
                        )
                    # The forward has certainly completed at the result point.
                    entry["fence"] = None
                    entry["aux"] = aux
                elif room in self.sent:
                    # Closed early: check its pages, and give a last-stage share
                    # sent without first-token metadata the result point's.
                    item = self.sent[room]
                    if item["full"] != full[item["start"] :] or item["swa"] != swa:
                        mismatch = (
                            f"the pages registered before the result point "
                            f"({len(item['full'])} full, {len(item['swa'])} window) "
                            f"differ from the result point's "
                            f"({len(full) - item['start']}, {len(swa)})"
                        )
                        self._abort_room(room, mismatch)
                    elif not item["aux"] and aux and mgr.pp_rank == mgr.pp_size - 1:
                        item["aux"] = aux
                        late_aux = True
                else:
                    self.sent[room] = self._closing_entry(room, full, swa, aux)
                mgr.update_status(room, KVPoll.Transferring)
            if mismatch is not None:
                for endpoint in self.replies.get(room, {}):
                    mgr._reply_room_failed(
                        room=room, endpoint=endpoint, reason=mismatch
                    )
                return False
            # Registration calls the controller, so it runs outside the
            # manager's state lock.
            if self.stage_parts:
                self._advance_stage(room)
            else:
                if late_aux:
                    self._announce_aux(room)
                self._reply(room)
            return room not in self.failed

    def _abort_room(self, room, reason: str) -> None:
        """Fail a room as an abort does: the producer poll cancels its closing
        transfer if unclaimed, and a claimed one keeps the room's pages until
        it settles. Hold the manager's state lock."""
        mgr = self.manager
        logger.error("Raiden pipeline rank%d room %s: %s", self.rank, room, reason)
        mgr._mark_aborted_locked(room)
        mgr._fail_room_locked(room, reason, preserve_reason=True)
        mgr.update_status(room, KVPoll.Failed)

    def _announce_aux(self, room) -> None:
        """Send the first-token metadata alone to every decode already told of
        the room's closing share; later ones get it with the share."""
        item = self.sent[room]
        mgr = self.manager
        meta = json.dumps(dict(rank=mgr.pp_rank, pp_size=mgr.pp_size, aux_only=True))
        for endpoint in item["replied"]:
            self._send(
                endpoint,
                [
                    MessageTag.PIPELINE_READY,
                    str(room).encode(),
                    meta.encode(),
                    item["aux"],
                ],
            )

    def _fail_room(self, room, reason: str) -> None:
        """Fail one room and record why; its sender or receiver concludes it on
        the next poll."""
        with self.manager._state_lock:
            self._fail_room_held(room, reason)

    def _fail_room_held(self, room, reason: str) -> None:
        """_fail_room for a caller already holding the manager's state lock."""
        logger.error("Raiden pipeline rank%d room %s: %s", self.rank, room, reason)
        self.failed.add(room)
        self.manager._fail_room_locked(room, reason, preserve_reason=True)

    def _reply(self, room):
        item = self.sent.get(room)
        if item is None or not self.replies.get(room) or room in self.failed:
            return
        if not item["registered"]:
            if not self._register_blocks(
                room,
                req_id=str(room),
                uuid=room,
                full=item["full"],
                swa=item["swa"],
                with_swa=True,
            ):
                self.retired.add(room)
                return
            item["registered"] = True
        payload = self._source_metadata(
            full=len(item["full"]),
            swa=len(item["swa"]),
            start=item["start"],
            parts=item["parts"],
            **pp_swa.source_fields(self),
        )
        self._announce(room, item, payload, item["aux"], "replied")

    # ----- consumer --------------------------------------------------------

    def register_decode(self, room, full, swa, aux_index, infos):
        self._validate_room(room)
        mgr = self.manager
        ranks = {int(info["pp_rank"]) for info in infos}
        if len(ranks) != len(infos):
            raise ValueError(
                "Raiden pipeline sources must have distinct pipeline ranks"
            )
        with self.lock:
            self.receives[room] = Receive(
                full=list(full),
                swa=list(swa),
                aux_index=aux_index,
                ranks=ranks,
                sources={},
                deadline=time.monotonic() + mgr._timeout_s,
                endpoints=tuple(
                    NetworkAddress(info["rank_ip"], int(info["rank_port"])).to_tcp()
                    for info in infos
                ),
            )
        endpoint = NetworkAddress(mgr.local_ip, mgr.rank_port).to_tcp()
        consumer = json.dumps(
            dict(attn_tp_size=mgr.attn_tp_size, attn_tp_rank=mgr.attn_tp_rank)
        ).encode()
        for info in infos:
            peer = NetworkAddress(info["rank_ip"], int(info["rank_port"])).to_tcp()
            self._send(
                peer,
                [
                    MessageTag.PIPELINE_REQ,
                    str(room).encode(),
                    endpoint.encode(),
                    consumer,
                ],
            )

    def on_message(self, msg):
        mgr = self.manager
        room = int(msg[1])
        if msg[0] in (MessageTag.PIPELINE_RELEASE, MessageTag.PIPELINE_REQ):
            # Producer frames call the controller, so they run outside the
            # manager's state lock.
            with self.lock:
                with mgr._state_lock:
                    failed_reason = mgr._tombstone_reason_locked(room)
                if failed_reason is None:
                    self._on_producer_message(room, msg)
            if failed_reason is not None and msg[0] == MessageTag.PIPELINE_REQ:
                mgr._require_frames(msg, MessageTag.PIPELINE_REQ, 4)
                mgr._reply_room_failed(
                    room=room, endpoint=msg[2].decode(), reason=failed_reason
                )
            return
        with self.lock, mgr._state_lock:
            if room in mgr._aborted:
                return
            mgr._require_frames(msg, MessageTag.PIPELINE_READY, 4)
            request = self.receives.get(room)
            if request is None:
                return
            meta = json.loads(msg[2])
            rank = int(meta["rank"])
            if rank not in request.ranks:
                raise ValueError(f"Unexpected pipeline source rank {rank}")
            pp_size = int(meta["pp_size"])
            expected = set(range(pp_size)) if mgr.pp_size == 1 else {mgr.pp_rank}
            if request.ranks != expected or mgr.pp_size not in (1, pp_size):
                raise ValueError(
                    "Raiden decode must use DP or the same pipeline stage count"
                )
            if meta.get("aux_only"):
                # A stage-groups room's first-token metadata, from the last stage.
                self._write_aux(request, msg[3])
                request.aux_ranks.add(rank)
                return
            if mgr.pp_size > 1 and set(meta["pool_tags"]) != set(self._pool_tags()):
                raise ValueError(
                    "Matching pipelines must own the same layers per stage"
                )
            if meta.get("part") is None:
                pp_swa.check_agreement(self, meta)
            if meta.get("stage"):
                request.stage[rank] = meta
            elif meta.get("part") is not None:
                part = request.parts.setdefault(
                    int(meta["part"]), _new_transfer_state()
                )
                part["sources"][rank] = meta
            else:
                request.sources[rank] = meta
                # A DP/TP decode hears every stage; only the last one holds the
                # sampled token and the draft hidden states.
                if mgr.pp_size > 1 or rank == pp_size - 1:
                    self._write_aux(request, msg[3])

    def _write_aux(self, request, blob: bytes) -> None:
        mgr = self.manager
        if request.aux_index is not None and blob:
            write_aux_blob(
                mgr.kv_args.aux_data_ptrs,
                mgr.kv_args.aux_item_lens,
                request.aux_index,
                blob,
            )

    def _on_producer_message(self, room, msg):
        mgr = self.manager
        if msg[0] == MessageTag.PIPELINE_RELEASE:
            mgr._require_frames(msg, MessageTag.PIPELINE_RELEASE, 3)
            if not self.producer:
                raise ValueError("A pipeline release must target a prefill worker")
            decode_pp_size = int(msg[2])
            if decode_pp_size not in (1, mgr.pp_size):
                raise ValueError("Pipeline release has an incompatible stage count")
            self.release_requests[room] = dict(pp_size=decode_pp_size, next_attempt=0.0)
            self._try_release(room)
            return
        mgr._require_frames(msg, MessageTag.PIPELINE_REQ, 4)
        if not self.producer:
            raise ValueError("A pipeline request must target a prefill worker")
        self.replies.setdefault(room, {})[msg[2].decode()] = json.loads(msg[3])
        for part in self.early.get(room, {}).get("parts", []):
            self._reply_part(room, part)
        if room in self.stage:
            self._reply_stage(room)
        self._reply(room)

    def maybe_start_pull(self, room):
        mgr = self.manager
        with self.lock, mgr._state_lock:
            if room in mgr._aborted:
                return False
            request = self.receives.get(room)
            if request is None:
                return False
            if not self._start_parts(room, request):
                return False
            if request.stage:
                return self._start_stages(room, request)
            if set(request.sources) != request.ranks:
                return False
            if request.accepted or request.future is not None:
                pp_swa.resume_twin(self, request, key=())
                return True
            if time.monotonic() < request.next_attempt:
                return False
            sources, args, twin = self._plan_closing(room, request)
            reason = pp_swa.unplaceable(args, twin)
            if reason is not None:
                self._fail_room_held(room, reason)
                return False
            if not args["transfer_pool_tags"]:
                if mgr.attn_tp_rank == 0:
                    for endpoint in request.endpoints:
                        self._send(
                            endpoint,
                            [
                                MessageTag.PIPELINE_RELEASE,
                                str(room).encode(),
                                str(mgr.pp_size).encode(),
                            ],
                        )
                self.done_recv.add(room)
                return True
            if not self._admit(room, request, window=twin is not None):
                return False
            mgr.update_status(room, KVPoll.Transferring)
            ranks = list(range(int(sources[0]["pp_size"])))
            pp_swa.start_twin(
                self, room, request, key=(), twin=twin, metas=sources, ranks=ranks
            )
            if mgr.attn_tp_rank != 0:
                # Rank 0 plans one transfer into every attention-TP rank's
                # engine; this rank's engine reports its own landing.
                request.accepted = True
                return True
            request.future = self._submit_pull(
                sources, ranks, req_id=str(room), uuid=room, args=args
            )
            return True

    def _plan_closing(self, room, request) -> tuple[list[dict], dict, dict | None]:
        """The stages' READY metadata of the closing transfer, by rank, and the
        destination pages of the transfer and of its twin."""
        sources = [request.sources[rank] for rank in sorted(request.ranks)]
        addresses = {source["controller"] for source in sources}
        # A producer without early parts leaves out start and parts.
        geometries = {
            (
                source["full"],
                source["swa"],
                source.get("start", 0),
                source.get("parts", 0),
            )
            for source in sources
        }
        if len(addresses) != 1 or len(geometries) != 1:
            raise ValueError(
                "Raiden pipeline sources disagree on request geometry or controller"
            )
        source_full, source_swa, start, parts = geometries.pop()
        request.expected_parts = parts
        if set(request.parts) != set(range(parts)):
            raise ValueError(
                f"Raiden pipeline room {room} closed with {parts} early parts "
                f"but announced {sorted(request.parts)}"
            )
        args = part_destination_blocks(
            self.pools,
            request.full,
            request.swa,
            start=start,
            count=source_full,
            source_swa_count=source_swa,
        )
        twin = pp_swa.twin_destination(self, request, swa_count=source_swa)
        if twin is not None and not (
            args["transfer_pool_tags"] or twin["transfer_pool_tags"]
        ):
            twin = None  # the decode already holds every page
        return sources, args, twin

    def _submit_pull(self, metas, ranks, *, req_id: str, uuid: int, args: dict):
        """Coordinate one transfer from the source stages ``ranks``, announced
        in their READY metadata ``metas``, into every attention-TP rank of
        this decode."""
        address = metas[0]["controller"]
        return self.executor.submit(
            self._coordinate,
            address,
            src_units=[
                SimpleNamespace(**(metas[0]["unit"] | {"data_replica_idx": rank}))
                for rank in ranks
            ],
            dst_units=self.dst_units,
            req_id=req_id,
            uuid=uuid,
            use_block_chunks=True,
            is_sender=True,
            dst_mem_type=self.hbm_mem_type,
            src_controller_address=address,
            dst_controller_address=self.controller,
            **args,
        )

    def _admit(self, room, request, key: tuple = (), *, window: bool = False) -> bool:
        """Reserve host staging for one native transfer of the room and, with
        ``window``, for its twin on the window engine. Raiden seats every
        transfer in its own staging slot, so a room moving early parts or
        several stage groups holds one reservation per transfer."""
        key = (room, *key) if key else room
        if key in request.admitted_keys:
            return True
        window_pages = self.swa.max_blocks if window else 0
        if not self.admission.admit(key, self.max_blocks, window_pages):
            return False
        request.admitted_keys.add(key)
        return True

    def _release_admission(self, room, request, key: tuple = ()) -> None:
        key = (room, *key) if key else room
        if key in request.admitted_keys:
            request.admitted_keys.discard(key)
            self.admission.release(key)

    def _release_all_admission(self, room, request) -> None:
        for key in request.admitted_keys:
            self.admission.release(key)
        request.admitted_keys.clear()
        # A transfer of the room still at the head of the line would otherwise
        # block every later admission.
        waiting = self.admission.waiting
        if waiting == room or (isinstance(waiting, tuple) and waiting[0] == room):
            self.admission.forget(waiting)

    def _start_parts(self, room, request) -> bool:
        """Start every early part all stages have announced. False only when
        staging admission has to be retried."""
        mgr = self.manager
        for index in sorted(request.parts):
            part = request.parts[index]
            if part["accepted"] or part["future"] is not None:
                continue
            if part["done"]:
                # Rank 0 started it for this attention-TP rank, and it landed.
                part["accepted"] = True
                continue
            if set(part["sources"]) != request.ranks:
                continue
            if time.monotonic() < part["next_attempt"]:
                continue
            sources = [part["sources"][rank] for rank in sorted(request.ranks)]
            addresses = {source["controller"] for source in sources}
            geometries = {(source["start"], source["full"]) for source in sources}
            if len(addresses) != 1 or len(geometries) != 1:
                raise ValueError(
                    "Raiden pipeline sources disagree on an early part's pages"
                )
            start, count = geometries.pop()
            if not self._admit(room, request, ("p", index)):
                return False
            if mgr.attn_tp_rank != 0:
                part["accepted"] = True
                continue
            part["future"] = self._submit_pull(
                sources,
                range(int(sources[0]["pp_size"])),
                req_id=part_req_id(room, index),
                uuid=part_uuid(room, index),
                args=part_destination_blocks(
                    self.pools,
                    request.full,
                    request.swa,
                    start=start,
                    count=count,
                    source_swa_count=None,
                ),
            )
        return True

    def _coordinate(self, address: str, **kwargs):
        """coordinate_transfer on this worker thread's own client."""
        clients = self._thread_clients.__dict__.setdefault("by_address", {})
        if address not in clients:
            clients[address] = self.client_type(address)
        return clients[address].coordinate_transfer(**kwargs)

    def _start_stages(self, room, request) -> bool:
        """Start each stage group's transfer once all of its stages have
        announced. True once every stage has announced."""
        mgr = self.manager
        parts = {int(meta["parts"]) for meta in request.stage.values()}
        if len(parts) != 1:
            raise ValueError("Raiden pipeline stages disagree on early part count")
        request.expected_parts = parts.pop()
        by_group: dict[int, list[dict]] = {}
        for meta in request.stage.values():
            by_group.setdefault(int(meta["group"]), []).append(meta)
        for group, metas in sorted(by_group.items()):
            ranks = sorted(int(meta["rank"]) for meta in metas)
            if ranks != sorted(int(r) for r in metas[0]["group_ranks"]):
                continue  # a stage of this group is still computing
            state = request.groups.setdefault(group, _new_transfer_state())
            if state["accepted"] or state["future"] is not None:
                pp_swa.resume_twin(self, request, key=("g", group))
                continue
            if state["done"]:
                # Rank 0 started it for this attention-TP rank, and it landed.
                state["accepted"] = True
                continue
            if time.monotonic() < state["next_attempt"]:
                continue
            geometries = {
                (int(m["start"]), int(m["full"]), int(m["swa"])) for m in metas
            }
            if len(geometries) != 1:
                raise ValueError("Raiden pipeline stages of a group disagree on pages")
            start, count, swa_count = geometries.pop()
            group_tags = set().union(*(set(m["pool_tags"]) for m in metas))
            args = part_destination_blocks(
                self.pools,
                request.full,
                request.swa,
                start=start,
                count=count,
                source_swa_count=swa_count,
                only_tags=group_tags,
            )
            twin = pp_swa.twin_destination(
                self, request, swa_count=swa_count, only_tags=group_tags
            )
            reason = pp_swa.unplaceable(args, twin)
            if reason is not None:
                self._fail_room_held(room, f"group {group}: {reason}")
                return False
            if not self._admit(room, request, ("g", group), window=twin is not None):
                return False
            metas = sorted(metas, key=lambda m: int(m["rank"]))
            pp_swa.start_twin(
                self,
                room,
                request,
                key=("g", group),
                twin=twin,
                metas=metas,
                ranks=ranks,
            )
            if mgr.attn_tp_rank != 0:
                state["accepted"] = True
                continue
            state["future"] = self._submit_pull(
                metas,
                ranks,
                req_id=stage_req_id(room, group),
                uuid=stage_uuid(room, group),
                args=args,
            )
        if set(request.stage) != request.ranks:
            return False
        mgr.update_status(room, KVPoll.Transferring)
        return True

    def _stages_landed(self, request) -> bool:
        """Every group landed, the last stage's metadata arrived and the early
        parts landed."""
        if not request.stage or set(request.stage) != request.ranks:
            return False
        groups = {int(meta["group"]) for meta in request.stage.values()}
        return (
            set(request.groups) == groups
            and all(state["done"] for state in request.groups.values())
            and max(request.ranks) in request.aux_ranks
            and (not request.parts or self._parts_landed(request))
        )

    def _parts_landed(self, request) -> bool:
        return request.expected_parts is not None and all(
            request.parts[index]["done"] for index in range(request.expected_parts)
        )

    def _receive_settled(self, request) -> bool:
        """No coordination of the room is outstanding and its parts landed."""
        return (
            request.future is None
            and all(
                state["future"] is None
                for state in [*request.groups.values(), *request.twins.values()]
            )
            and (not request.parts or self._parts_landed(request))
        )

    def _try_release(self, room):
        release = self.release_requests.get(room)
        if (
            release is None
            or room not in self.sent
            or room in self.done_send | self.failed
            or time.monotonic() < release["next_attempt"]
        ):
            return
        # An unregistered room has nothing native to claim or release.
        registered = self.sent[room]["registered"]
        if registered and release["pp_size"] == 1:
            if not pp_swa.cancel_unclaimed(self, str(room), room, registered=True):
                return
        elif registered:
            try:
                pp_swa.complete_twin(self, str(room))
                self.client.complete_request_blocks(str(room), room, self.unit)
            except RuntimeError as error:
                if _EARLY_COMPLETION not in str(error):
                    raise
                release["next_attempt"] = time.monotonic() + 0.05
                return
        self.done_send.add(room)
        self.retired.add(room)

    def active(self, room):
        with self.lock:
            if self.producer:
                parts = self.early.get(room, {}).get("parts", [])
                entry = self.stage.get(room)
                return (
                    (room in self.sent and room not in self.done_send | self.failed)
                    or any(part["registered"] and not part["done"] for part in parts)
                    or (entry is not None and entry["registered"] and not entry["done"])
                )
            request = self.receives.get(room)
            return request is not None and (
                request.future is not None
                or (request.accepted and room not in self.done_recv | self.failed)
                or time.monotonic() < request.armed_until
                or any(
                    state["future"] is not None
                    or time.monotonic() < state["armed_until"]
                    or (
                        state["accepted"]
                        and not state["done"]
                        and room not in self.failed
                    )
                    for state in [
                        *request.parts.values(),
                        *request.groups.values(),
                        *request.twins.values(),
                    ]
                )
            )

    def _stage_landed(self, room: int, group: int, *, failed: bool = False) -> None:
        """A stage group's transfer settled natively: it landed, or failed."""
        if self.producer:
            entry = self.stage.get(room)
            if (
                entry is not None
                and not entry["done"]
                and group == stage_group(self.manager.pp_rank, self.stage_group_size)
            ):
                entry["done"] = True
                self.client.complete_request_blocks(
                    stage_req_id(room, group), stage_uuid(room, group), self.unit
                )
                pp_swa.complete_twin(self, stage_req_id(room, group))
            return
        request = self.receives.get(room)
        if request is not None:
            # Rank 0 starts a group for every attention-TP rank, so it can land
            # here before this rank has reached it in _start_stages.
            state = request.groups.setdefault(group, _new_transfer_state())
            state["armed_until"] = 0.0
            if not failed:
                state["done"] = True
            self._release_admission(room, request, ("g", group))
            self.admission.forget((room, "g", group))

    def _part_landed(self, room: int, index: int, *, failed: bool = False) -> None:
        """An early part's transfer settled natively: it landed, or failed."""
        if self.producer:
            parts = self.early.get(room, {}).get("parts", [])
            if index < len(parts) and not parts[index]["done"]:
                parts[index]["done"] = True
                self.client.complete_request_blocks(
                    part_req_id(room, index), part_uuid(room, index), self.unit
                )
            return
        request = self.receives.get(room)
        if request is not None:
            # As for groups: it can land before every stage announced it here.
            state = request.parts.setdefault(index, _new_transfer_state())
            state["armed_until"] = 0.0
            if not failed:
                state["done"] = True
            self._release_admission(room, request, ("p", index))
            # Landed before this rank admitted it: it no longer waits in line.
            self.admission.forget((room, "p", index))

    def _settle_coordination(self, room, request) -> None:
        """Take a finished coordinate_transfer. A registration the producer has
        yet to make is retried until the room's deadline; any other error
        fails the room."""
        future, request.future = request.future, None
        retry = self._settle_future(room, request, future, req_id=str(room))
        if retry is None:
            request.accepted = True
        elif retry:
            if not pp_swa.twin_live(request, key=()):
                self._release_admission(room, request)
            request.next_attempt = time.monotonic() + 0.05

    def _settle_transfer(
        self, room, request, state: dict, *, key: tuple | None, req_id: str
    ):
        """_settle_coordination for one early part, stage group or twin; a twin
        (``key`` None) is staged with its full-attention transfer."""
        future, state["future"] = state["future"], None
        retry = self._settle_future(room, request, future, req_id=req_id, state=state)
        if retry is None:
            state["accepted"] = True
        elif retry:
            if key is not None and not pp_swa.twin_live(request, key=key):
                self._release_admission(room, request, key)
            state["next_attempt"] = time.monotonic() + 0.05

    def _settle_future(
        self, room, request, future, *, req_id: str, state: dict | None = None
    ) -> bool | None:
        """None when the controller accepted the transfer, True to retry it,
        False when it failed the room. The transfer, ``state`` or else the
        closing transfer, stays armed while a push of it can still land."""
        mgr = self.manager
        error = future.exception()
        if error is None and future.result():
            self._arm(room, request, state)
            return None
        text = "the controller did not accept it" if error is None else str(error)
        missing = _MISSING_REGISTRATION in text
        if missing and not (
            room in mgr._aborted or time.monotonic() >= request.deadline
        ):
            return True
        if not missing and _REGISTRATION_CANCELLED not in text:
            # The controller arms the receivers before the senders, so a push
            # can still land in the room's pages until the native deadline.
            self._arm(room, request, state)
        self._fail_room(room, f"Raiden pipeline transfer {req_id} failed: {text}")
        return False

    def _arm(self, room, request, state: dict | None) -> None:
        """Hold the room's pages for one native deadline, while a push of the
        transfer can land in them, unless its landing has been reported."""
        until = time.monotonic() + self.manager._timeout_s
        if state is None:
            if room not in self.done_recv:
                request.armed_until = until
        elif not state["done"]:
            state["armed_until"] = until

    def poll(self) -> tuple[set[str], set[str], set[str]]:
        """Settle native completions and deadlines; returns the sent, received
        and failed rooms for the manager to record."""
        mgr = self.manager
        with self.lock:
            status = mgr._poll_engine_status(name="pipeline", engine=self.engine)
            if self.swa is not None:
                status = pp_swa.merge(
                    self,
                    status,
                    mgr._poll_engine_status(
                        name="pipeline-swa", engine=self.swa.engine
                    ),
                )
            if status is not None:
                self._record_native_status(*status)
            if self.producer:
                self._poll_producer()
            else:
                for room, request in self.receives.items():
                    self._poll_receive(room, request)
            sent_rooms = {
                str(room) for room in self.done_send if self._early_settled(room)
            }
            received_rooms = {
                str(room)
                for room in self.done_recv
                if room in self.receives and self._receive_settled(self.receives[room])
            }
            failed_rooms = {str(room) for room in self.failed}
            for room in list(self.cleared):
                if not self.active(room):
                    self._forget(room)
        return sent_rooms, received_rooms, failed_rooms

    def _record_native_status(self, sent, received, failed) -> None:
        mgr = self.manager
        for transfers, done in ((sent, self.done_send), (received, self.done_recv)):
            for t in transfers:
                room, key = parse_req_id(mgr._norm_req_id(t.req_id))
                if key is None:
                    done.add(room)
                    if room in self.receives:
                        self.receives[room].armed_until = 0.0
                elif isinstance(key, tuple):
                    self._stage_landed(room, key[1])
                else:
                    self._part_landed(room, key)
        for t in failed:
            room, key = parse_req_id(mgr._norm_req_id(t.req_id))
            self.failed.add(room)
            if key is None:
                if room in self.receives:
                    self.receives[room].armed_until = 0.0
            elif isinstance(key, tuple):
                self._stage_landed(room, key[1], failed=True)
            else:
                self._part_landed(room, key, failed=True)

    def _poll_producer(self) -> None:
        mgr = self.manager
        self._expire_parts()
        for room, entry in list(self.stage.items()):
            if not entry["ready"] and room not in self.failed:
                self._advance_stage(room)
            if entry["done"] and entry["aux"] is not None:
                self.done_send.add(room)
        for room in (self.done_send | self.failed) - self.retired:
            if room in self.sent:
                self.client.complete_request_blocks(str(room), room, self.unit)
                pp_swa.complete_twin(self, str(room))
            self.retired.add(room)
        for room, item in self.sent.items():
            self._try_release(room)
            if room in self.done_send | self.failed:
                continue
            if room in mgr._aborted or time.monotonic() >= item["expires"]:
                if pp_swa.cancel_unclaimed(
                    self, str(room), room, registered=item["registered"]
                ):
                    if room in self.release_requests and room not in mgr._aborted:
                        self.done_send.add(room)
                    else:
                        self.failed.add(room)
                    self.retired.add(room)

    def _poll_receive(self, room, request) -> None:
        mgr = self.manager
        if request.future is not None and request.future.done():
            self._settle_coordination(room, request)
        for key, state in request.twins.items():
            if state["future"] is not None and state["future"].done():
                self._settle_transfer(
                    room,
                    request,
                    state,
                    key=None,
                    req_id=pp_swa.twin_id(room, key),
                )
        for group, state in request.groups.items():
            if state["future"] is not None and state["future"].done():
                self._settle_transfer(
                    room,
                    request,
                    state,
                    key=("g", group),
                    req_id=stage_req_id(room, group),
                )
        for index, state in request.parts.items():
            if state["future"] is not None and state["future"].done():
                self._settle_transfer(
                    room,
                    request,
                    state,
                    key=("p", index),
                    req_id=part_req_id(room, index),
                )
        if request.stage:
            if room not in self.failed:
                # As from maybe_start_pull: _start_stages can fail the room.
                with mgr._state_lock:
                    self._start_stages(room, request)
            if self._stages_landed(request):
                self.done_recv.add(room)
        transfers = [*request.parts.values(), *request.groups.values()]
        if (
            (request.accepted or any(state["accepted"] for state in transfers))
            and mgr.attn_tp_rank != 0
            and room not in self.done_recv | self.failed
            and time.monotonic() >= request.deadline + mgr._timeout_s
        ):
            # Rank 0 submits by its deadline and the native transfer deadline
            # bounds the landing after that.
            self.failed.add(room)
        settled = room in self.failed or (
            room in self.done_recv
            and (not request.parts or self._parts_landed(request))
        )
        now = time.monotonic()
        if (
            settled
            and request.future is None
            and now >= request.armed_until
            and all(
                state["future"] is None and now >= state["armed_until"]
                for state in [*transfers, *request.twins.values()]
            )
        ):
            self._release_all_admission(room, request)

    def clear(self, room):
        with self.lock:
            if self.active(room):
                self.cleared.add(room)
            else:
                self._forget(room)

    def _forget(self, room):
        self.sent.pop(room, None)
        self.early.pop(room, None)
        self.stage.pop(room, None)
        self.replies.pop(room, None)
        self.release_requests.pop(room, None)
        request = self.receives.pop(room, None)
        if request is not None:
            self._release_all_admission(room, request)
        self.done_send.discard(room)
        self.done_recv.discard(room)
        self.failed.discard(room)
        self.retired.discard(room)
        self.cleared.discard(room)
        self.admission.forget(room)
        pp_swa.forget(self, room)
