"""TPU prefill/decode disaggregation backend backed by tpu-raiden.

Raiden moves KV pages directly between TPU hosts (D2H -> network -> H2D) and
tracks transfer completion itself. SGLang only provides rendezvous: the decode
worker tells the target prefill worker "I want room R, reply to <my endpoint>",
the prefill worker replies with its page block-ids + raiden endpoint (and the
small aux/first-token blob), and the decode worker then issues a raiden pull.

This is the reverse direction from the push-based RDMA backends (mooncake/nixl):
raiden is consumer-pull, so the per-request block-ids flow prefill -> decode
(raiden's ``start_read`` needs the remote block-ids on the consumer side). The
existing ZMQ rank sockets and HTTP bootstrap server from ``CommonKVManager`` are
reused unchanged for the rendezvous.

When decode TP < prefill TP (e.g. 8TP->4TP2DP), a decode rank's kv-head slice
spans several prefill ranks, so a whole-block pull cannot place them. That case
is planned by the raiden controller: every rank registers a tagged pool table
over its KV tensors, each prefill rank declares the byte spans it contributes to
the decode pages, and the controller builds the strided schedule, arms the
decode receiver, and dispatches the senders. Raiden stages device->host and
auto-H2Ds into decode HBM on its own. Span geometry is built in
``reshard_schedule.py``; the pool table in ``pool_manifest.py``.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll, StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.raiden.serde import (
    MessageTag,
    pack_blocks,
    pack_ints,
    pack_unit,
    read_aux_blob,
    unpack_blocks,
    unpack_ints,
    unpack_unit,
    write_aux_blob,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# Bounds the wait for the other prefill ranks of a group to publish their span
# declarations. The controller rejects a plan whose sources are not all
# registered, and that rejection happens before any receiver is armed, so it is
# retried; every other failure surfaces immediately.
_MISSING_REGISTRATION_ERROR = "Missing producer block registration"


@dataclass(frozen=True)
class RaidenKVCacheGroups:
    """Page-compatible cache groups registered with independent Raiden engines."""

    full: List[Any]
    swa: List[Any]


def get_raiden_kv_caches(pool) -> RaidenKVCacheGroups:
    """Group per-layer device KV tensors by their page-index namespace.

    The TPU fused KV pool stores one [num_blocks, page_size, ...] tensor per
    layer in ``kv_buffer`` and exposes them via ``get_fused_kv_buffer``. Hybrid
    SWA pools allocate full-attention and sliding-window pages independently, so
    they must not share one Raiden block-id list.
    """
    if hasattr(pool, "get_fused_kv_buffer"):
        start = pool.start_layer
        buffers = [pool.get_fused_kv_buffer(start + i) for i in range(pool.layer_num)]
    elif hasattr(pool, "kv_buffer"):
        buffers = list(pool.kv_buffer)
    else:
        raise RuntimeError(
            f"Raiden backend requires a fused KV pool exposing kv_buffer; got "
            f"{type(pool).__name__}"
        )
    layers_mapping = getattr(pool, "layers_mapping", None)
    if layers_mapping:
        full = []
        swa = []
        for offset, buf in enumerate(buffers):
            layer_id = pool.start_layer + offset
            (swa if layers_mapping[layer_id][1] else full).append(buf)
    else:
        full, swa = buffers, []

    for name, group in (("full", full), ("SWA", swa)):
        block_counts = {int(buf.shape[0]) for buf in group}
        if len(block_counts) > 1:
            raise RuntimeError(
                f"Raiden requires uniform {name} KV block counts; got "
                f"{sorted(block_counts)}"
            )
    return RaidenKVCacheGroups(full=full, swa=swa)


class RaidenKVManager(CommonKVManager):
    """KV manager for the raiden prefill/decode transport.

    FIXME: the ZMQ rendezvous scaffolding here (per-rank listener socket,
    handshake message routing, poll draining) overlaps the tpu backend's
    ``base.py`` control plane. Consolidate onto a shared base so raiden reuses it
    instead of reimplementing the coordination.
    """

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)

        self._engine: Optional[Any] = None  # raiden KVCacheManager
        self._swa_engine: Optional[Any] = None
        self._kv_caches: Optional[list] = None
        self._swa_kv_caches: Optional[list] = None
        self.raiden_endpoint: Optional[str] = None
        self.swa_raiden_endpoint: Optional[str] = None
        # Each rank reserves control+data ports for full KV and SWA KV.
        base_port = int(os.environ.get("SGLANG_RAIDEN_KV_PORT", "30000"))
        self._local_rank = self.attn_dp_rank * self.attn_tp_size + self.attn_tp_rank
        self._control_port = base_port + 4 * self._local_rank
        self._timeout_s = float(os.environ.get("SGLANG_RAIDEN_TIMEOUT_S", "120"))

        # Resharding for asymmetric TP (decode TP < prefill TP) is planned by a
        # raiden controller shared by the prefill and decode instances. Without
        # one, only the symmetric pull path is available.
        self._controller_addr = os.environ.get(
            "SGLANG_RAIDEN_CONTROLLER_ADDR", ""
        ).strip()
        self._job_name = os.environ.get("SGLANG_RAIDEN_JOB_NAME", "sglang").strip()
        self._reshard_enabled = bool(self._controller_addr)
        # Prefill only: the decode instance's attention TP. It fixes how many
        # prefill ranks feed one decode rank, which the controller admits at
        # work-unit registration, long before the first request names a peer.
        self._peer_attn_tp = int(os.environ.get("SGLANG_RAIDEN_PEER_ATTN_TP", "0"))
        self._manifest: Optional[Any] = None
        self._facade: Optional[Any] = None
        self._work_unit: Optional[Any] = None

        # Completion sets, drained from raiden and keyed by str(room). Persist
        # until clear() so repeated poll() calls before the framework advances
        # keep reporting the same terminal state.
        self._completed_send: Set[str] = set()
        self._completed_recv: Set[str] = set()
        self._failed: Set[str] = set()
        self._engine_lock = threading.Lock()
        self._push_lock = threading.Lock()
        self._state_lock = threading.Lock()

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # room -> accumulated page block-ids, aux blob, decode reply addr
            self._send_blocks: Dict[int, List[int]] = {}
            self._send_swa_blocks: Dict[int, List[int]] = {}
            self._send_aux: Dict[int, bytes] = {}
            self._decode_addr: Dict[int, Tuple[str, int]] = {}
            self._replied: Set[int] = set()
        else:
            # room -> local page block-ids, remote page block-ids, endpoint
            self._local_blocks: Dict[int, List[int]] = {}
            self._local_swa_blocks: Dict[int, List[int]] = {}
            self._decode_aux_index: Dict[int, int] = {}
            self._remote_blocks: Dict[int, List[int]] = {}
            self._remote_swa_blocks: Dict[int, List[int]] = {}
            self._remote_endpoint: Dict[int, str] = {}
            self._remote_swa_endpoint: Dict[int, str] = {}

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # room -> decode meta from RESHARD_REQ
            self._reshard_req: Dict[int, dict] = {}
            self._reshard_finalized: Set[int] = set()  # send accumulation complete
            self._reshard_registered: Set[int] = set()  # spans declared to controller
            self._reshard_submitted: Set[int] = set()
            # Declaring spans and submitting the plan are blocking control-plane
            # RPCs, so they run off the _listen dispatch thread.
            self._push_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(
                    1, int(os.environ.get("SGLANG_RAIDEN_PUSH_WORKERS", "1"))
                ),
                thread_name_prefix="raiden-reshard-submit",
            )

        self._listener = threading.Thread(target=self._listen, daemon=True)
        self._listener.start()

    # ----- resharding topology / identity -----------------------------------

    def _reshard_ratio(self) -> int:
        """Prefill ranks feeding one decode rank, from the configured peer TP."""
        from sglang.srt.disaggregation.raiden.reshard_schedule import reshard_ratio

        if self._peer_attn_tp <= 0:
            raise RuntimeError(
                "Raiden resharding needs the decode instance's attention TP on "
                "the prefill side: set SGLANG_RAIDEN_PEER_ATTN_TP."
            )
        return reshard_ratio(self.attn_tp_size, self._peer_attn_tp)

    def _work_unit_fields(self) -> Dict[str, Any]:
        role = (
            "prefill"
            if self.disaggregation_mode == DisaggregationMode.PREFILL
            else "decode"
        )
        return dict(
            job_name=f"{self._job_name}-{role}",
            job_replica_id=str(self.attn_dp_rank),
            data_name="kv_cache",
            data_replica_idx=int(self.attn_tp_rank),
        )

    def _source_units(self) -> List[Any]:
        """Work units of the prefill ranks feeding this rank's decode rank."""
        from tpu_raiden.rpc.raiden_controller import RaidenId

        from sglang.srt.disaggregation.raiden.reshard_schedule import (
            decode_rank_for,
            source_ranks_for,
        )

        ratio = self._reshard_ratio()
        decode_rank = decode_rank_for(self.attn_tp_rank, ratio)
        fields = self._work_unit_fields()
        return [
            RaidenId(**{**fields, "data_replica_idx": rank})
            for rank in source_ranks_for(decode_rank, ratio)
        ]

    def _make_geometry(self, src_attn_tp: int, dst_attn_tp: int):
        """ReshardGeometry for a (src_attn_tp -> dst_attn_tp) transfer, sized from
        the registered pool manifest (the K+V head-pair granule matches on both
        sides) and the global kv-head count.
        """
        from sglang.srt.disaggregation.raiden.reshard_schedule import ReshardGeometry

        # Distinct model kv-head count (model_config.get_total_num_kv_heads()),
        # set on both the prefill and decode KVArgs.
        total_kv_heads = int(self.kv_args.total_kv_head_num)
        return ReshardGeometry(
            src_attn_tp, dst_attn_tp, self._manifest.head_pair_bytes, total_kv_heads
        )

    # ----- engine lifecycle -------------------------------------------------

    def set_kv_caches(self, kv_caches: RaidenKVCacheGroups) -> None:
        """Hand raiden the per-layer device KV tensors and build the engine.

        Called by prefill.py/decode.py right after construction (the KV pool is
        only available there). Raiden indexes each tensor's major dim as a
        block/page, which matches the TPU fused KV pool layout
        ([num_blocks, page_size, ...]).
        """
        self._kv_caches = list(kv_caches.full)
        self._swa_kv_caches = list(kv_caches.swa)
        if not self._kv_caches:
            raise RuntimeError(
                "Raiden requires at least one full-attention KV cache tensor"
            )
        self._build_engine()

    def _build_engine(self) -> None:
        if self._engine is not None or self._kv_caches is None:
            return
        # TPU-only import, deferred so the disaggregation module imports cleanly
        # on hosts without tpu_raiden. Raiden owns the transport, host staging,
        # D2H/H2D DMA, and completion tracking; it indexes each KV tensor's major
        # dim as a page, matching the TPU fused KV pool ([num_blocks, page, ...]).
        from tpu_raiden.api.torch.kv_cache_manager import KVCacheManager

        max_blocks = self._max_request_blocks()
        num_slots = self._num_slots()
        self.raiden_endpoint = f"{self.local_ip}:{self._control_port}"
        kwargs = dict(
            kv_caches=self._kv_caches,
            node_id=self._local_rank,
            local_control_port=self._control_port,
            max_blocks=max_blocks,
            num_slots=num_slots,
            timeout_s=self._timeout_s,
        )
        if self._reshard_enabled:
            # The controller arms this rank over the native listener, and reads
            # back its parallelism when it admits a plan's source set.
            kwargs.update(
                listener_port=0,
                parallelism=self._transfer_parallelism(),
            )
        self._engine = KVCacheManager(**kwargs)
        logger.info(
            "RaidenKVManager rank%d engine up | endpoint=%s max_blocks=%d "
            "num_slots=%d",
            self.attn_tp_rank,
            self.raiden_endpoint,
            max_blocks,
            num_slots,
        )
        if self._reshard_enabled:
            self._register_pools_and_work_unit()
        if self._swa_kv_caches:
            self.swa_raiden_endpoint = f"{self.local_ip}:{self._control_port + 2}"
            swa_max_blocks = self._max_request_blocks()
            swa_num_slots = self._num_slots()
            self._swa_engine = KVCacheManager(
                kv_caches=self._swa_kv_caches,
                node_id=self._local_rank,
                local_control_port=self._control_port + 2,
                max_blocks=swa_max_blocks,
                num_slots=swa_num_slots,
                timeout_s=self._timeout_s,
            )
            logger.info(
                "RaidenKVManager rank%d SWA engine up | endpoint=%s "
                "max_blocks=%d num_slots=%d",
                self.attn_tp_rank,
                self.swa_raiden_endpoint,
                swa_max_blocks,
                swa_num_slots,
            )

    def _transfer_parallelism(self) -> int:
        """Ranks the controller admits as one source set for a reshard plan."""
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            return self._reshard_ratio()
        return 1

    def _transfer_rank(self) -> int:
        """This rank's position in its source set (contiguous from zero)."""
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            return self.attn_tp_rank % self._reshard_ratio()
        return 0

    def _register_pools_and_work_unit(self) -> None:
        """Publish this rank's block pools and transfer identity to the controller.

        The pool table gives raiden a byte-level address space over the KV
        tensors -- one pool per layer, all under one tag, so a transfer that
        names the tag moves every layer. The controller matches the layout
        fingerprint and pool tags across a transfer's peers before it plans.
        """
        from tpu_raiden.rpc.raiden_controller import (
            RaidenControllerClientFacade,
            RaidenId,
        )

        from sglang.srt.disaggregation.raiden.pool_manifest import build_pool_manifest

        manifest = build_pool_manifest(
            self._kv_caches, total_kv_heads=int(self.kv_args.total_kv_head_num)
        )
        summary = self._engine.register_pools(manifest.pool_dicts())

        unit = RaidenId(**self._work_unit_fields())
        facade = RaidenControllerClientFacade(self._controller_addr)
        facade.register_work_unit(
            unit=unit,
            shards=[self._engine.transfer_address],
            control_plane_rpc_address=self._engine.listener_address,
            pool_manifest=manifest.pool_dicts(),
            layout_fingerprint=manifest.layout_fingerprint,
            page_tokens=manifest.page_tokens,
            transfer_parallelism=self._transfer_parallelism(),
            transfer_rank=self._transfer_rank(),
        )
        self._manifest = manifest
        self._facade = facade
        self._work_unit = unit
        logger.info(
            "RaidenKVManager rank%d registered with controller=%s | unit=%s "
            "data=%s listener=%s pools=%s rank=%d/%d fingerprint=%s",
            self.attn_tp_rank,
            self._controller_addr,
            unit,
            self._engine.transfer_address,
            self._engine.listener_address,
            summary,
            self._transfer_rank(),
            self._transfer_parallelism(),
            manifest.layout_fingerprint,
        )

    def _max_request_blocks(self) -> int:
        override = int(os.environ.get("SGLANG_RAIDEN_MAX_BLOCKS", "0"))
        if override > 0:
            return override
        max_len = self.server_args.context_length or int(
            os.environ.get("SGLANG_RAIDEN_MAX_MODEL_LEN", "8192")
        )
        return max(1, math.ceil(max_len / self.kv_args.page_size))

    def _num_slots(self) -> int:
        return max(1, int(os.environ.get("SGLANG_RAIDEN_NUM_SLOTS", "16")))

    def _ensure_engine(self) -> Any:
        if self._engine is None:
            with self._engine_lock:
                self._build_engine()
        if self._engine is None:
            raise RuntimeError(
                "Raiden engine not initialized: set_kv_caches() must be called "
                "with the KV pool tensors before any transfer."
            )
        return self._engine

    @staticmethod
    def _norm_req_id(rid: str) -> str:
        # Resharded transfers surface as "resharded_transfer_<room>" from
        # poll_stats; strip the prefix so completion maps back to str(room).
        prefix = "resharded_transfer_"
        return rid[len(prefix) :] if rid.startswith(prefix) else rid

    @staticmethod
    def _swa_req_id(room: int) -> str:
        return f"swa:{room}"

    def _drain_engine(self) -> None:
        if self._engine is None and self._swa_engine is None:
            return
        for engine in (self._engine, self._swa_engine):
            if engine is None:
                continue
            done_send, done_recv, failed = engine.poll_stats()
            if done_send or done_recv or failed:
                with self._state_lock:
                    self._completed_send.update(self._norm_req_id(r) for r in done_send)
                    self._completed_recv.update(self._norm_req_id(r) for r in done_recv)
                    self._failed.update(self._norm_req_id(r) for r in failed)
                if failed:
                    logger.error(
                        "RaidenKVManager rank%d failed=%s", self.attn_tp_rank, failed
                    )

    def register_to_bootstrap(self):
        # Topology/handshake registration is backend-agnostic; raiden adds no
        # new bootstrap fields (its endpoint rides in the per-request reply).
        super().register_to_bootstrap()

    # ----- rendezvous (ZMQ over the existing per-rank PULL socket) -----------

    def _listen(self) -> None:
        while True:
            try:
                msg = self.server_socket.recv_multipart()
            except Exception as e:
                logger.error("RaidenKVManager listener error: %s", e)
                continue
            try:
                if msg[0] == MessageTag.PULL_REQ:
                    self._on_pull_req(msg)
                elif msg[0] == MessageTag.BLOCKS:
                    self._on_blocks(msg)
                elif msg[0] == MessageTag.RESHARD_REQ:
                    self._on_reshard_req(msg)
                elif msg[0] == MessageTag.RESHARD_AUX:
                    self._on_reshard_aux(msg)
                else:
                    logger.warning("RaidenKVManager unknown header: %s", msg[0])
            except Exception as e:
                logger.error("RaidenKVManager message handling error: %s", e)

    @staticmethod
    def _require_frames(msg: List[bytes], tag: MessageTag, n: int) -> None:
        """Reject a malformed header before it is indexed past its length."""
        if len(msg) != n:
            raise ValueError(
                f"malformed {tag.name} message: expected {n} frames, got {len(msg)}"
            )

    def _on_pull_req(self, msg: List[bytes]) -> None:
        self._require_frames(msg, MessageTag.PULL_REQ, 4)
        room = int(msg[1].decode("ascii"))
        decode_ip = msg[2].decode("ascii")
        decode_port = int(msg[3].decode("ascii"))
        with self._state_lock:
            self._decode_addr[room] = (decode_ip, decode_port)
        self._maybe_reply(room)

    def _on_blocks(self, msg: List[bytes]) -> None:
        self._require_frames(msg, MessageTag.BLOCKS, 7)
        room = int(msg[1].decode("ascii"))
        remote_blocks = unpack_blocks(msg[2])
        endpoint = msg[3].decode("ascii")
        remote_swa_blocks = unpack_blocks(msg[4])
        swa_endpoint = msg[5].decode("ascii")
        aux_blob = msg[6]
        aux_index = None
        with self._state_lock:
            self._remote_blocks[room] = remote_blocks
            self._remote_endpoint[room] = endpoint
            self._remote_swa_blocks[room] = remote_swa_blocks
            self._remote_swa_endpoint[room] = swa_endpoint
            aux_index = self._decode_aux_index.get(room)
        if aux_index is not None and aux_blob:
            write_aux_blob(
                self.kv_args.aux_data_ptrs,
                self.kv_args.aux_item_lens,
                aux_index,
                aux_blob,
            )

    # ----- prefill side ------------------------------------------------------

    def _swa_blocks_from_state_indices(
        self, state_indices: Optional[List]
    ) -> List[int]:
        if not self._swa_kv_caches:
            return []
        try:
            index = self.kv_args.state_types.index(StateType.SWA)
        except ValueError as exc:
            raise RuntimeError(
                "Raiden has SWA KV tensors but StateType.SWA is not registered"
            ) from exc
        if (
            state_indices is None
            or index >= len(state_indices)
            or state_indices[index] is None
        ):
            raise RuntimeError("Raiden SWA transfer is missing its page indices")
        return [int(x) for x in state_indices[index]]

    def append_send_blocks(
        self,
        room: int,
        block_ids: List[int],
        swa_block_ids: Optional[List[int]] = None,
    ) -> None:
        with self._state_lock:
            self._send_blocks.setdefault(room, []).extend(block_ids)
            if swa_block_ids is not None:
                self._send_swa_blocks[room] = list(swa_block_ids)

    def finalize_send(self, room: int, aux_index: Optional[int]) -> None:
        self._ensure_engine()
        with self._state_lock:
            if aux_index is not None:
                self._send_aux[room] = read_aux_blob(
                    self.kv_args.aux_data_ptrs,
                    self.kv_args.aux_item_lens,
                    aux_index,
                )
            self._reshard_finalized.add(room)
            is_reshard = room in self._reshard_req
        # The request type is selected by the decode handshake, not by whether
        # resharding is globally available. Symmetric TP uses PULL_REQ even when
        # a controller is configured; asymmetric TP uses RESHARD_REQ.
        if is_reshard:
            self._maybe_declare_spans(room)
        else:
            self._maybe_reply(room)

    def _maybe_reply(self, room: int) -> None:
        with self._state_lock:
            if room in self._replied:
                return
            if (
                room not in self._reshard_finalized
                or room not in self._send_blocks
                or (self._swa_engine is not None and room not in self._send_swa_blocks)
                or room not in self._decode_addr
            ):
                return
            swa_blocks = list(self._send_swa_blocks.get(room, []))
            blocks = list(self._send_blocks[room])
            addr = self._decode_addr[room]
            aux_blob = self._send_aux.get(room, b"")
            self._replied.add(room)
        if self._swa_engine is not None:
            self._swa_engine.register_read(self._swa_req_id(room), room, swa_blocks)
        self._ensure_engine().register_read(str(room), room, blocks)
        endpoint = f"tcp://{addr[0]}:{addr[1]}"
        payload = [
            MessageTag.BLOCKS,
            str(room).encode("ascii"),
            pack_blocks(blocks),
            (self.raiden_endpoint or "").encode("ascii"),
            pack_blocks(swa_blocks),
            (self.swa_raiden_endpoint or "").encode("ascii"),
            aux_blob,
        ]
        with self._push_lock:
            self._connect(endpoint).send_multipart(payload)

    # ----- decode side -------------------------------------------------------

    def register_decode_request(
        self,
        room: int,
        local_blocks: List[int],
        local_swa_blocks: List[int],
        aux_index: Optional[int],
        bootstrap_infos: List[dict],
    ) -> None:
        if len(bootstrap_infos) != 1:
            if self._swa_engine is not None:
                raise NotImplementedError(
                    "Raiden cannot reshard hybrid SWA KV caches: decode rank "
                    f"received {len(bootstrap_infos)} prefill sources. Use "
                    "symmetric attention TP for prefill and decode."
                )
            # Asymmetric attention TP (decode attn TP < prefill attn TP): one
            # decode rank is fed by several prefill ranks, each declaring its
            # head-slice into the wider decode block.
            if self._reshard_enabled:
                if not local_blocks:
                    # Full prefix-cache hit: decode already holds every page,
                    # nothing to receive.
                    with self._state_lock:
                        self._completed_recv.add(str(room))
                    return
                self._request_reshard(room, local_blocks, aux_index, bootstrap_infos)
                self.update_status(room, KVPoll.WaitingForInput)
                return
            raise NotImplementedError(
                "Raiden asymmetric attention TP (got "
                f"{len(bootstrap_infos)} prefill sources for one decode rank) "
                "is planned by a raiden controller, but none is configured; "
                "set SGLANG_RAIDEN_CONTROLLER_ADDR on both instances."
            )
        with self._state_lock:
            self._local_blocks[room] = local_blocks
            self._local_swa_blocks[room] = local_swa_blocks
            if aux_index is not None:
                self._decode_aux_index[room] = aux_index
        payload = [
            MessageTag.PULL_REQ,
            str(room).encode("ascii"),
            self.local_ip.encode("ascii"),
            str(self.rank_port).encode("ascii"),
        ]
        for info in bootstrap_infos:
            endpoint = f"tcp://{info['rank_ip']}:{info['rank_port']}"
            with self._push_lock:
                self._connect(endpoint).send_multipart(payload)

    @staticmethod
    def _pair_blocks(
        remote: List[int], local: List[int]
    ) -> Tuple[List[int], List[int]]:
        local = list(local)
        remote = list(remote)
        if not local:
            remote = []
        elif len(remote) > len(local):
            remote = remote[-len(local) :]
        elif len(remote) < len(local):
            local = local[: len(remote)]
        return remote, local

    def maybe_start_pull(self, room: int) -> bool:
        """Issue the raiden pull once the prefill reply has arrived.

        Returns True if the pull was (or had already been) issued.
        """
        with self._state_lock:
            if (
                room not in self._remote_blocks
                or room not in self._remote_endpoint
                or (
                    self._swa_engine is not None and room not in self._remote_swa_blocks
                )
                or (
                    self._swa_engine is not None
                    and room not in self._remote_swa_endpoint
                )
            ):
                return False
            local = list(self._local_blocks.get(room, []))
            remote = list(self._remote_blocks[room])
            endpoint = self._remote_endpoint[room]
            local_swa = list(self._local_swa_blocks.get(room, []))
            remote_swa = list(self._remote_swa_blocks.get(room, []))
            swa_endpoint = self._remote_swa_endpoint.get(room, "")
        # start_read needs equal-length, position-paired remote/local. Reconcile
        # the two independent ways they can differ:
        #   - decode prefix-cache hit: decode already holds the head pages and
        #     allocated fewer local pages, so pull only the matching tail of
        #     remote (remote[-len(local):]).
        #   - reserved decode token: decode allocated one extra trailing page that
        #     prefill produced no KV for, so drop it from local.
        # Empty local => full prefix hit; pull nothing (and avoid remote[-0:],
        # which would select the whole list).
        remote, local = self._pair_blocks(remote, local)
        engine = self._ensure_engine()
        engine.start_read(str(room), room, endpoint, remote, local)
        if self._swa_engine is not None:
            remote_swa, local_swa = self._pair_blocks(remote_swa, local_swa)
            self._swa_engine.start_read(
                self._swa_req_id(room), room, swa_endpoint, remote_swa, local_swa
            )
        return True

    # ----- asymmetric-TP resharding (controller-planned push) ---------------
    #
    # Data flows prefill -> decode. Handshake (decode-initiated), per decode rank
    # r fed by prefill ranks {r*ratio .. (r+1)*ratio-1}:
    #   1. decode -> each source: RESHARD_REQ (its work unit + pages + attn TP)
    #   2. each source declares its byte spans to the controller
    #   3. the group's first source asks the controller to run the transfer; the
    #      controller arms the decode receiver, then dispatches every sender
    #   4. senders stage device->host and push; the receiver auto-H2Ds into HBM
    # The first source also forwards the aux/first-token blob over RESHARD_AUX.
    # Both sides observe completion through poll_stats().

    def _request_reshard(
        self,
        room: int,
        local_blocks: List[int],
        aux_index: Optional[int],
        bootstrap_infos: List[dict],
    ) -> None:
        with self._state_lock:
            if aux_index is not None:
                self._decode_aux_index[room] = aux_index
        payload = [
            MessageTag.RESHARD_REQ,
            str(room).encode("ascii"),
            self.local_ip.encode("ascii"),
            str(self.rank_port).encode("ascii"),
            pack_unit(self._work_unit_fields()),
            pack_ints(list(local_blocks)),
            str(self.attn_tp_size).encode("ascii"),
        ]
        for info in bootstrap_infos:
            endpoint = f"tcp://{info['rank_ip']}:{info['rank_port']}"
            with self._push_lock:
                self._connect(endpoint).send_multipart(payload)

    def _on_reshard_req(self, msg: List[bytes]) -> None:
        self._require_frames(msg, MessageTag.RESHARD_REQ, 7)
        room = int(msg[1].decode("ascii"))
        decode_ip = msg[2].decode("ascii")
        decode_port = int(msg[3].decode("ascii"))
        decode_unit_fields = unpack_unit(msg[4])
        decode_blocks = unpack_ints(msg[5])
        dst_attn_tp = int(msg[6].decode("ascii"))
        with self._state_lock:
            self._reshard_req[room] = dict(
                reply_addr=(decode_ip, decode_port),
                decode_unit_fields=decode_unit_fields,
                decode_blocks=decode_blocks,
                dst_attn_tp=dst_attn_tp,
            )
        self._maybe_declare_spans(room)

    def _on_reshard_aux(self, msg: List[bytes]) -> None:
        self._require_frames(msg, MessageTag.RESHARD_AUX, 3)
        room = int(msg[1].decode("ascii"))
        aux_blob = msg[2]
        with self._state_lock:
            aux_index = self._decode_aux_index.get(room)
        if aux_index is not None and aux_blob:
            write_aux_blob(
                self.kv_args.aux_data_ptrs,
                self.kv_args.aux_item_lens,
                aux_index,
                aux_blob,
            )

    def _maybe_declare_spans(self, room: int) -> None:
        """Declare this rank's byte contribution once its pages are final.

        A RESHARD_REQ can arrive while ``send()`` is still accumulating pages,
        so both halves gate the declaration.
        """
        with self._state_lock:
            if room in self._reshard_registered:
                return
            if room not in self._reshard_req or room not in self._reshard_finalized:
                return
            self._reshard_registered.add(room)
        self._push_executor.submit(self._declare_and_submit, room)

    def _declare_and_submit(self, room: int) -> None:
        # A control-plane failure marks the room failed so the sender poll
        # surfaces it instead of the request hanging until the decode timeout.
        try:
            self._declare_spans(room)
            self._maybe_submit_transfer(room)
        except Exception as e:
            logger.error("raiden reshard failed for room %s: %s", room, e)
            with self._state_lock:
                self._failed.add(str(room))

    def _declare_spans(self, room: int) -> None:
        from sglang.srt.disaggregation.raiden.pool_manifest import TAG_KV
        from sglang.srt.disaggregation.raiden.reshard_schedule import (
            build_span_registration,
        )

        if self._facade is None:
            raise RuntimeError(
                "A decode rank asked this prefill rank to reshard, but no "
                "raiden controller is configured here: set "
                "SGLANG_RAIDEN_CONTROLLER_ADDR on the prefill instance too."
            )
        with self._state_lock:
            req_meta = dict(self._reshard_req[room])
            prefill_blocks = list(self._send_blocks[room])
            aux_blob = self._send_aux.get(room, b"")

        decode_blocks = req_meta["decode_blocks"]
        geom = self._make_geometry(self.attn_tp_size, req_meta["dst_attn_tp"])
        # A decode prefix hit leaves decode holding fewer (tail) pages than the
        # prefill produced; keep the last n so both sides pair the same pages.
        n = len(decode_blocks)
        if len(prefill_blocks) > n:
            prefill_blocks = prefill_blocks[-n:]

        registration = build_span_registration(
            tag=TAG_KV,
            geom=geom,
            prefill_rank=self.attn_tp_rank,
            decode_rank=self.attn_tp_rank // geom.ratio,
            src_block_ids=prefill_blocks,
            live_tokens=[self.kv_args.page_size] * len(prefill_blocks),
        )
        self._facade.register_request_blocks(
            req_id=str(room),
            uuid=room,
            unit=self._work_unit,
            block_ids=prefill_blocks,
            pool_spans=[registration],
        )
        logger.debug(
            "reshard declare room=%s prefill_rank=%s decode_rank=%s n=%d "
            "src[:4]=%s dst[:4]=%s spans=%d bytes=%d",
            room,
            self.attn_tp_rank,
            self.attn_tp_rank // geom.ratio,
            len(prefill_blocks),
            prefill_blocks[:4],
            decode_blocks[:4],
            len(registration.spans),
            registration.declared_bytes,
        )
        if self._transfer_rank() == 0:
            endpoint = "tcp://{}:{}".format(*req_meta["reply_addr"])
            payload = [MessageTag.RESHARD_AUX, str(room).encode("ascii"), aux_blob]
            with self._push_lock:
                self._connect(endpoint).send_multipart(payload)

    def _maybe_submit_transfer(self, room: int) -> None:
        """Ask the controller to plan and run the transfer for one decode page set.

        Only the group's first source submits: one plan covers every source, and
        the controller arms the decode receiver before it dispatches any sender.
        """
        if self._transfer_rank() != 0:
            return
        with self._state_lock:
            if room in self._reshard_submitted:
                return
            self._reshard_submitted.add(room)
            req_meta = dict(self._reshard_req[room])

        from tpu_raiden.rpc.raiden_controller import RaidenId, RaidenMemoryType

        from sglang.srt.disaggregation.raiden.pool_manifest import TAG_KV

        dst_unit = RaidenId(**req_meta["decode_unit_fields"])
        kwargs = dict(
            src_units=self._source_units(),
            dst_units=[dst_unit],
            req_id=str(room),
            uuid=room,
            dst_device_block_ids=list(req_meta["decode_blocks"]),
            dst_mem_type=RaidenMemoryType.HBM,
            use_block_chunks=True,
            is_sender=True,
            transfer_pool_tags=[TAG_KV],
        )
        deadline = self._monotonic() + self._timeout_s
        while True:
            try:
                self._facade.start_transfer(**kwargs)
                return
            except RuntimeError as exc:
                expired = self._monotonic() >= deadline
                if expired or _MISSING_REGISTRATION_ERROR not in str(exc):
                    raise
                time.sleep(0.001)

    @staticmethod
    def _monotonic() -> float:
        return time.monotonic()

    # ----- completion queries (drained by poll()) ---------------------------

    def is_send_done(self, room: int) -> bool:
        rid = str(room)
        return rid in self._completed_send and (
            self._swa_engine is None or self._swa_req_id(room) in self._completed_send
        )

    def is_recv_done(self, room: int) -> bool:
        rid = str(room)
        return rid in self._completed_recv and (
            self._swa_engine is None or self._swa_req_id(room) in self._completed_recv
        )

    def is_failed(self, room: int) -> bool:
        return str(room) in self._failed or self._swa_req_id(room) in self._failed

    def clear_room(self, room: int) -> None:
        rid = str(room)
        swa_rid = self._swa_req_id(room)
        was_reshard = False
        with self._state_lock:
            self._completed_send.discard(rid)
            self._completed_send.discard(swa_rid)
            self._completed_recv.discard(rid)
            self._completed_recv.discard(swa_rid)
            self._failed.discard(rid)
            self._failed.discard(swa_rid)
            if hasattr(self, "_reshard_registered"):
                was_reshard = room in self._reshard_registered
            for d in (
                getattr(self, "_send_blocks", None),
                getattr(self, "_send_swa_blocks", None),
                getattr(self, "_send_aux", None),
                getattr(self, "_decode_addr", None),
                getattr(self, "_local_blocks", None),
                getattr(self, "_local_swa_blocks", None),
                getattr(self, "_decode_aux_index", None),
                getattr(self, "_remote_blocks", None),
                getattr(self, "_remote_swa_blocks", None),
                getattr(self, "_remote_endpoint", None),
                getattr(self, "_remote_swa_endpoint", None),
                getattr(self, "_reshard_req", None),
            ):
                if d is not None:
                    d.pop(room, None)
            if hasattr(self, "_replied"):
                self._replied.discard(room)
            for s in (
                getattr(self, "_reshard_finalized", None),
                getattr(self, "_reshard_registered", None),
                getattr(self, "_reshard_submitted", None),
            ):
                if s is not None:
                    s.discard(room)
        if was_reshard:
            # Vote this rank terminal so the controller can retire the request's
            # declarations once every source has finished with them. It is a
            # control-plane RPC, so it stays off the scheduler thread.
            self._push_executor.submit(self._complete_request_blocks, room)

    def _complete_request_blocks(self, room: int) -> None:
        try:
            self._facade.complete_request_blocks(str(room), room, self._work_unit)
        except Exception as e:
            logger.error(
                "raiden reshard completion vote failed for room %s: %s", room, e
            )


class RaidenKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: RaidenKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
        req_has_disagg_prefill_dp_rank: bool = False,
    ):
        super().__init__(
            mgr,
            bootstrap_addr,
            bootstrap_room,
            dest_tp_ranks,
            pp_rank,
            req_has_disagg_prefill_dp_rank=req_has_disagg_prefill_dp_rank,
        )
        self._last_chunk = False
        self._started = False
        # Raiden is consumer-pull: the prefill need not wait for any decode
        # handshake before computing (decode pulls once prefill registers its
        # blocks). Advance straight to WaitingForInput so the bootstrap queue
        # releases this request into prefill. Don't override a Failed/dummy state
        # already set by CommonKVSender.__init__.
        if (
            not self.kv_mgr.is_dummy_cp_rank
            and self.kv_mgr.check_status(self.bootstrap_room) == KVPoll.Bootstrapping
        ):
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        super().init(num_kv_indices, aux_index)

    def should_send_kv_chunk(self, num_pages: int, last_chunk: bool) -> bool:
        self._last_chunk = last_chunk
        return num_pages > 0 or last_chunk

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
        num_kv_tokens: Optional[int] = None,
    ):
        if self.kv_mgr.is_dummy_cp_rank:
            return
        self._record_transfer_indices(kv_indices, state_indices)
        swa_blocks = (
            self.kv_mgr._swa_blocks_from_state_indices(state_indices)
            if state_indices is not None
            else None
        )
        self.kv_mgr.append_send_blocks(
            self.bootstrap_room, [int(x) for x in kv_indices], swa_blocks
        )
        if self._last_chunk and not self._started:
            self._started = True
            self.kv_mgr.finalize_send(self.bootstrap_room, self.aux_index)
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Transferring)

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        if self.kv_mgr.is_dummy_cp_rank:
            self.conclude_state = KVPoll.Success
            return KVPoll.Success
        self.kv_mgr._drain_engine()
        if self.kv_mgr.is_failed(self.bootstrap_room):
            self.conclude_state = KVPoll.Failed
            return KVPoll.Failed
        if self.kv_mgr.is_send_done(self.bootstrap_room):
            self.conclude_state = KVPoll.Success
            return KVPoll.Success
        return self.kv_mgr.check_status(self.bootstrap_room)

    def failure_exception(self):
        reason = self.kv_mgr.failure_records.get(
            self.bootstrap_room, "Raiden send failed"
        )
        raise Exception(reason)

    def clear(self) -> None:
        super().clear()
        self.kv_mgr.clear_room(self.bootstrap_room)


class RaidenKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: RaidenKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room)
        self._started = False

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
    ):
        if self.conclude_state == KVPoll.Failed:
            return
        self.kv_mgr.register_decode_request(
            self.bootstrap_room,
            [int(x) for x in kv_indices],
            self.kv_mgr._swa_blocks_from_state_indices(state_indices),
            aux_index,
            self.bootstrap_infos,
        )
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        self.kv_mgr._drain_engine()
        if self.kv_mgr.is_failed(self.bootstrap_room):
            self.conclude_state = KVPoll.Failed
            return KVPoll.Failed
        if self.kv_mgr.is_recv_done(self.bootstrap_room):
            self.conclude_state = KVPoll.Success
            return KVPoll.Success
        if not self._started and self.kv_mgr.maybe_start_pull(self.bootstrap_room):
            self._started = True
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Transferring)
        return self.kv_mgr.check_status(self.bootstrap_room)

    def failure_exception(self):
        reason = self.kv_mgr.failure_records.get(
            self.bootstrap_room, "Raiden receive failed"
        )
        raise Exception(reason)

    def clear(self) -> None:
        super().clear()
        self.kv_mgr.clear_room(self.bootstrap_room)


class RaidenKVBootstrapServer(CommonKVBootstrapServer):
    pass
