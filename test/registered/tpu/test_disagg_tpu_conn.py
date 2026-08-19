"""Control and data protocol tests for TPU P/D disaggregation transfer backend."""

from __future__ import annotations

import logging
import threading
import time
import unittest
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import torch
import zmq

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.tpu import transfer as T
from sglang.srt.disaggregation.tpu.conn import (
    MessageTag,
    PendingChunk,
    PinnedHostPool,
    PrefillTransferInfo,
    TpuKVManager,
    _KvStats,
    _tcp_endpoint,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")
envs.SGLANG_TPU_ENABLE_DISAGGREGATION_PERFETTO_TRACING.set(True)

logger = logging.getLogger(__name__)


class _TestRpaV3KVPool:
    """Mock KV pool representing TPU fused RPA tensors with byte storage."""

    def __init__(
        self,
        num_blocks: int,
        page_size: int,
        num_lanes: int,
        kv_packing: int,
        head_dim: int,
        n_layers: int,
        dtype: torch.dtype = torch.uint8,
    ):
        self.kv_buffer = [
            torch.zeros(
                (num_blocks, page_size, num_lanes, kv_packing, head_dim),
                dtype=dtype,
            )
            for _ in range(n_layers)
        ]


_ACTIVE_MANAGERS: list[TpuKVManager] = []


def _cleanup_tpu_kv_manager(m: TpuKVManager) -> None:
    """Explicit typed cleanup for TpuKVManager instances."""
    m._closed = True
    if hasattr(m, "_serve_thread") and m._serve_thread is not None:
        try:
            m._serve_thread.join(timeout=0.5)
        except Exception as e:
            logger.debug("Error joining serve thread: %s", e)
    if hasattr(m, "_push_pool") and m._push_pool is not None:
        try:
            m._push_pool.shutdown(wait=True, cancel_futures=False)
        except Exception as e:
            logger.debug("Error shutting down push pool: %s", e)
    for sock in list(m._socket_cache.values()):
        try:
            sock.close(linger=0)
        except Exception as e:
            logger.debug("Error closing socket cache: %s", e)
    for sock in list(m._monitor_cache.values()):
        try:
            sock.close(linger=0)
        except Exception as e:
            logger.debug("Error closing monitor cache: %s", e)
    try:
        m.server_socket.close(linger=0)
    except Exception as e:
        logger.debug("Error closing server socket: %s", e)
    if hasattr(m, "_stage_pool") and m._stage_pool is not None:
        try:
            m._stage_pool.clear()
        except Exception as e:
            logger.debug("Error clearing stage pool: %s", e)
    if hasattr(m, "_zmq_ctx") and m._zmq_ctx is not None:
        try:
            m._zmq_ctx.destroy(linger=0)
        except Exception as e:
            logger.debug("Error destroying zmq context: %s", e)


def _create_tpu_kv_manager(
    mode: DisaggregationMode,
    pool: _TestRpaV3KVPool,
    aux_tensors: list[torch.Tensor],
    attn_tp_size: int = 1,
    engine_rank: int = 0,
    total_kv_head_num: int = 0,
) -> TpuKVManager:
    """Create a TpuKVManager without background recv threads for deterministic testing."""
    m = object.__new__(TpuKVManager)
    _ACTIVE_MANAGERS.append(m)
    m._name = f"TpuKVManager-{mode.value}-r{engine_rank}"
    m.disaggregation_mode = mode
    m.kv_pool = None
    m.attn_dp_size = 1
    m.attn_dp_rank = 0
    m.attn_tp_rank = engine_rank
    m.attn_tp_size = attn_tp_size
    m._initialize_trace_id()

    m.request_status = {}
    m.failure_records = {}
    m.failure_lock = threading.Lock()
    m._endpoint_locks = defaultdict(threading.Lock)
    m.bootstrap_timeout = 60
    m.waiting_timeout = 60

    if total_kv_head_num == 0 and pool is not None and hasattr(pool, "kv_buffer"):
        lanes = pool.kv_buffer[0].shape[2]
        packing = pool.kv_buffer[0].shape[3]
        total_kv_head_num = (lanes * (2 if packing == 4 else 1)) * attn_tp_size

    m.kv_args = SimpleNamespace(
        aux_data_ptrs=[t.data_ptr() for t in aux_tensors],
        aux_item_lens=[t[0].nbytes for t in aux_tensors],
        engine_rank=engine_rank,
        total_kv_head_num=total_kv_head_num,
    )
    m._aux_keepalive = aux_tensors

    ctx = zmq.Context()
    m._zmq_ctx = ctx
    m._socket_cache = {}
    m._monitor_cache = {}
    m._socket_lock = threading.Lock()
    m._socket_send_locks = {}

    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.setsockopt(zmq.LINGER, 0)
    port = sock.bind_to_random_port("tcp://127.0.0.1")
    m.server_socket = sock
    m.local_ip = "127.0.0.1"
    m.rank_port = port

    if mode == DisaggregationMode.PREFILL:
        m.transfer_infos = {}
        m.req_to_decode_prefix_len = {}
        m._push_pool = ThreadPoolExecutor(max_workers=4)
        m._push_progress = {}
        m._push_progress_lock = threading.Lock()
        m._gather_stats = _KvStats("gather", 0)
        m._async_d2h = False
        m._stage_pool = PinnedHostPool(max_bytes=64 * 1024 * 1024)
    else:
        m.decode_states = {}
        m.decode_states_lock = threading.Lock()
        m.decode_pending = defaultdict(deque)
        m.reasm_pending = defaultdict(lambda: defaultdict(list))
        m.decode_pending_lock = threading.Lock()
        m._stage_pool = PinnedHostPool(max_bytes=64 * 1024 * 1024)
        m._scatter_stats = _KvStats("scatter", 0)
        m._assemble_stats = _KvStats("assemble", 0)
        m._push_to_recv_stats = _KvStats("push_to_recv", 0)

    m.attach_kv_pool(pool)
    return m


def _make_trsfr_msg(
    room: int,
    decode_manager: TpuKVManager,
    num_pages: int,
    dst_tp_rank: int,
    decode_attn_tp_size: int,
    prefix_len: int = 0,
) -> list[bytes]:
    """Construct the raw TRSFR multipart message."""
    return [
        MessageTag.TRSFR,
        str(room).encode(),
        b"127.0.0.1",
        str(decode_manager.rank_port).encode(),
        b"1",
        str(num_pages).encode(),
        b"1",
        str(prefix_len).encode(),
        str(dst_tp_rank).encode(),
        str(decode_attn_tp_size).encode(),
    ]


def _drain_and_handle(decode_manager: TpuKVManager, count: int) -> None:
    """Drain `count` messages from decode's server_socket and invoke _handle_kvdata."""
    for _ in range(count):
        msg = decode_manager.server_socket.recv_multipart()
        decode_manager._handle_kvdata(msg)


class TestTpuKVManagerProtocol(CustomTestCase):
    TAIL = (2, 2, 16)
    DTYPE = torch.bfloat16

    def tearDown(self):
        while _ACTIVE_MANAGERS:
            m = _ACTIVE_MANAGERS.pop()
            _cleanup_tpu_kv_manager(m)
        super().tearDown()

    def _make_pair(self, layers: int = 2):
        ppool = _TestRpaV3KVPool(16, 4, 2, 2, 16, layers, dtype=self.DTYPE)
        dpool = _TestRpaV3KVPool(16, 4, 2, 2, 16, layers, dtype=self.DTYPE)
        for buf in ppool.kv_buffer:
            buf.copy_(
                (torch.arange(buf.numel(), dtype=torch.float32) % 100)
                .reshape(buf.shape)
                .to(self.DTYPE)
            )
        paux = [torch.arange(8 * 16, dtype=torch.int32).reshape(8, 16)]
        daux = [torch.zeros(8, 16, dtype=torch.int32)]
        prefill = _create_tpu_kv_manager(DisaggregationMode.PREFILL, ppool, paux)
        decode = _create_tpu_kv_manager(DisaggregationMode.DECODE, dpool, daux)
        return prefill, decode, ppool, dpool, paux, daux

    def _assert_pages_equal(self, ppool, dpool, src_pages, dst_pages):
        for li in range(len(ppool.kv_buffer)):
            for sp, dp in zip(src_pages, dst_pages):
                self.assertTrue(
                    torch.equal(ppool.kv_buffer[li][sp], dpool.kv_buffer[li][dp]),
                    f"layer {li}: prefill page {sp} != decode page {dp}",
                )

    def test_trsfr_marks_waiting_for_input(self):
        prefill, decode, *_ = self._make_pair()
        prefill._serve(MessageTag.TRSFR, prefill._handle_transfer)
        room = 4242
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(_tcp_endpoint("127.0.0.1", prefill.rank_port))
        push.send_multipart(
            [
                MessageTag.TRSFR,
                str(room).encode(),
                b"127.0.0.1",
                str(decode.rank_port).encode(),
                b"1",
                b"3",
                b"1",
                b"0",
                b"0",
                b"1",
            ]
        )
        deadline = time.time() + 5
        while time.time() < deadline:
            if prefill.request_status.get(room) == KVPoll.WaitingForInput:
                break
            time.sleep(0.02)
        push.close(linger=0)
        self.assertEqual(prefill.request_status.get(room), KVPoll.WaitingForInput)
        self.assertIn(room, prefill.transfer_infos)
        self.assertEqual(prefill.req_to_decode_prefix_len[room], 0)

    def test_single_chunk_kv_and_aux_roundtrip(self):
        prefill, decode, ppool, dpool, paux, daux = self._make_pair()
        room = 7
        src_pages = np.array([2, 5, 9], dtype=np.int64)
        dst_pages = np.array([0, 11, 4], dtype=np.int64)
        decode.register_decode_state(room, dst_pages, aux_index=3)
        ep = _tcp_endpoint("127.0.0.1", decode.rank_port)
        prefill.transfer_infos[room] = {
            ep: prefill.get_prefill_transfer_info(room, ep, 3)
        }

        prefill.gather_and_push(room, src_pages, 0, True, aux_index=6)
        msg = decode.server_socket.recv_multipart()
        decode._handle_kvdata(msg)
        self.assertNotIn(room, decode.request_status)
        decode.process_decode_pending(room)

        self._assert_pages_equal(ppool, dpool, src_pages, dst_pages)
        self.assertTrue(torch.equal(daux[0][3], paux[0][6]))
        self.assertEqual(prefill.request_status[room], KVPoll.Success)
        self.assertEqual(decode.request_status[room], KVPoll.Success)

    def test_chunked_transfer(self):
        prefill, decode, ppool, dpool, paux, daux = self._make_pair()
        room = 99
        src_pages = np.array([1, 2, 3, 8], dtype=np.int64)
        dst_pages = np.array([10, 0, 6, 2], dtype=np.int64)
        decode.register_decode_state(room, dst_pages, aux_index=2)
        ep = _tcp_endpoint("127.0.0.1", decode.rank_port)
        prefill.transfer_infos[room] = {
            ep: prefill.get_prefill_transfer_info(room, ep, 4)
        }

        prefill.gather_and_push(room, src_pages[:2], 0, False, aux_index=None)
        decode._handle_kvdata(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)
        self.assertNotIn(room, decode.request_status)

        prefill.gather_and_push(room, src_pages[2:], 2, True, aux_index=5)
        decode._handle_kvdata(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)

        self._assert_pages_equal(ppool, dpool, src_pages, dst_pages)
        self.assertTrue(torch.equal(daux[0][2], paux[0][5]))
        self.assertEqual(decode.request_status[room], KVPoll.Success)

    def test_out_of_order_chunks_wait_for_all(self):
        prefill, decode, ppool, dpool, paux, daux = self._make_pair()
        room = 4040
        src_pages = np.array([1, 2, 3, 8], dtype=np.int64)
        dst_pages = np.array([10, 0, 6, 2], dtype=np.int64)
        decode.register_decode_state(room, dst_pages, aux_index=2)
        ep = _tcp_endpoint("127.0.0.1", decode.rank_port)
        prefill.transfer_infos[room] = {
            ep: prefill.get_prefill_transfer_info(room, ep, 4)
        }

        prefill.gather_and_push(room, src_pages[2:], 2, True, aux_index=5)
        decode._handle_kvdata(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)
        self.assertNotEqual(decode.request_status.get(room), KVPoll.Success)

        prefill.gather_and_push(room, src_pages[:2], 0, False, aux_index=None)
        decode._handle_kvdata(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)

        self.assertEqual(decode.request_status[room], KVPoll.Success)
        self._assert_pages_equal(ppool, dpool, src_pages, dst_pages)
        self.assertTrue(torch.equal(daux[0][2], paux[0][5]))

    def test_prefill_success_waits_for_all_pushes(self):
        prefill, *_ = self._make_pair()
        room = 5050
        prefill._note_push_submitted(room, total_units=4)
        prefill._note_push_submitted(room, total_units=4)
        prefill._note_push_done(room, push_lane_units=2)
        self.assertNotEqual(prefill.request_status.get(room), KVPoll.Success)
        prefill._note_push_done(room, push_lane_units=2)
        self.assertEqual(prefill.request_status[room], KVPoll.Success)

    def test_prefill_failure_suppresses_late_success(self):
        prefill, *_ = self._make_pair()
        room = 6060
        prefill.request_status[room] = KVPoll.WaitingForInput
        prefill._note_push_submitted(room, total_units=4)
        prefill._note_push_failed(room)
        prefill.update_status(room, KVPoll.Failed)
        prefill._note_push_done(room, push_lane_units=2)
        self.assertEqual(prefill.request_status[room], KVPoll.Failed)

    def test_page_count_mismatch_fails(self):
        prefill, decode, _ppool, _dpool, *_ = self._make_pair()
        room = 13
        decode.request_status[room] = KVPoll.WaitingForInput
        decode.register_decode_state(
            room, np.array([0, 1], dtype=np.int64), aux_index=0
        )
        ep = _tcp_endpoint("127.0.0.1", decode.rank_port)
        prefill.transfer_infos[room] = {
            ep: prefill.get_prefill_transfer_info(room, ep, 3)
        }
        prefill.gather_and_push(room, np.array([2, 5, 9], dtype=np.int64), 0, True, 1)
        decode._handle_kvdata(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)
        self.assertEqual(decode.request_status[room], KVPoll.Failed)

    def test_threaded_happy_path(self):
        prefill, decode, ppool, dpool, paux, daux = self._make_pair()
        decode._serve_router(
            {
                MessageTag.KVDATA: decode._handle_kvdata,
            },
            copy=False,
        )
        room = 555
        src_pages = np.array([1, 7, 12], dtype=np.int64)
        dst_pages = np.array([3, 9, 0], dtype=np.int64)
        decode.register_decode_state(room, dst_pages, aux_index=4)
        ep = _tcp_endpoint("127.0.0.1", decode.rank_port)
        prefill.transfer_infos[room] = {
            ep: prefill.get_prefill_transfer_info(room, ep, 3)
        }
        prefill.gather_and_push(room, src_pages, 0, True, aux_index=2)

        deadline = time.time() + 5
        while time.time() < deadline:
            decode.process_decode_pending(room)
            if decode.request_status.get(room) == KVPoll.Success:
                break
            time.sleep(0.02)
        self.assertEqual(decode.request_status.get(room), KVPoll.Success)
        self._assert_pages_equal(ppool, dpool, src_pages, dst_pages)
        self.assertTrue(torch.equal(daux[0][4], paux[0][2]))


class TestTpuReshardingSimulation(CustomTestCase):
    """Deterministic step-by-step simulation of TPU P/D KV resharding."""

    def tearDown(self):
        while _ACTIVE_MANAGERS:
            m = _ACTIVE_MANAGERS.pop()
            _cleanup_tpu_kv_manager(m)
        super().tearDown()

    def test_fanin_2_prefill_into_1_decode_fp8_odd_head(self):
        n_layers = 2
        page_size = 3
        num_blocks = 4
        head_dim = 4
        kv_packing = 4
        total_num_head = 2
        src_tp_size = 2
        dst_tp_size = 1
        num_pages = 2
        dummy_byte = 0xEE

        p_pools = [
            _TestRpaV3KVPool(
                num_blocks,
                page_size,
                1,
                kv_packing,
                head_dim,
                n_layers,
            )
            for _ in range(src_tp_size)
        ]
        num_valid = num_blocks * page_size * 1 * 2 * head_dim
        for r, pool in enumerate(p_pools):
            for li, buf in enumerate(pool.kv_buffer):
                buf.fill_(dummy_byte)
                shift = r * 100 + li * 40
                valid_vals = (
                    ((torch.arange(num_valid, dtype=torch.int64) + shift) % 256)
                    .to(torch.uint8)
                    .reshape(num_blocks, page_size, 1, 2, head_dim)
                )
                buf[:, :, :, :2, :] = valid_vals

        p_aux = [
            [torch.full((num_blocks, 16), 100, dtype=torch.int32)]
            for _ in range(src_tp_size)
        ]
        prefills = [
            _create_tpu_kv_manager(
                DisaggregationMode.PREFILL,
                p_pools[r],
                p_aux[r],
                attn_tp_size=src_tp_size,
                engine_rank=r,
                total_kv_head_num=total_num_head,
            )
            for r in range(src_tp_size)
        ]

        d_pools = [
            _TestRpaV3KVPool(
                num_blocks,
                page_size,
                1,
                kv_packing,
                head_dim,
                n_layers,
            )
            for _ in range(dst_tp_size)
        ]
        d_aux = [
            [torch.zeros(num_blocks, 16, dtype=torch.int32)] for _ in range(dst_tp_size)
        ]
        decodes = [
            _create_tpu_kv_manager(
                DisaggregationMode.DECODE,
                d_pools[r],
                d_aux[r],
                attn_tp_size=dst_tp_size,
                engine_rank=r,
                total_kv_head_num=total_num_head,
            )
            for r in range(dst_tp_size)
        ]

        room = 6413157384759490229
        src_pages = np.array([1, 3], dtype=np.int64)
        dst_pages = np.array([0, 2], dtype=np.int64)
        aux_src_idx = 2
        aux_dst_idx = 1

        decodes[0].register_decode_state(room, dst_pages, aux_index=aux_dst_idx)

        for r in range(src_tp_size):
            prefills[r]._handle_transfer(
                _make_trsfr_msg(
                    room,
                    decodes[0],
                    num_pages,
                    dst_tp_rank=0,
                    decode_attn_tp_size=dst_tp_size,
                )
            )
            self.assertEqual(
                prefills[r].request_status.get(room), KVPoll.WaitingForInput
            )

        for r in range(src_tp_size):
            prefills[r].gather_and_push(
                room,
                src_pages,
                index_start=0,
                is_last_chunk=True,
                aux_index=aux_src_idx,
            )

        _drain_and_handle(decodes[0], count=src_tp_size)
        decodes[0].process_decode_pending(room)
        self.assertEqual(decodes[0].request_status[room], KVPoll.Success)

        for li in range(n_layers):
            for p_i in range(num_pages):
                src_b = src_pages[p_i]
                dst_b = dst_pages[p_i]
                expected_page = torch.cat(
                    [
                        p_pools[0].kv_buffer[li][src_b, :, :, :2, :],
                        p_pools[1].kv_buffer[li][src_b, :, :, :2, :],
                    ],
                    dim=2,
                )
                torch.testing.assert_close(
                    decodes[0].layer_buffers[li][dst_b],
                    expected_page,
                )

        untouched_pages = [b for b in range(num_blocks) if b not in dst_pages]
        for buf in decodes[0].layer_buffers:
            for ub in untouched_pages:
                self.assertTrue((buf[ub] == 0).all().item())

        self.assertTrue((d_aux[0][0][aux_dst_idx] == 100).all().item())

    def test_fanin_2_prefill_into_1_decode_fp8_even_head(self):
        n_layers = 2
        page_size = 3
        num_blocks = 4
        head_dim = 4
        kv_packing = 4
        total_num_head = 4
        src_tp_size = 2
        dst_tp_size = 1
        num_pages = 2

        p_pools = [
            _TestRpaV3KVPool(
                num_blocks,
                page_size,
                1,
                kv_packing,
                head_dim,
                n_layers,
            )
            for _ in range(src_tp_size)
        ]
        num_valid_p = num_blocks * page_size * 1 * kv_packing * head_dim
        for r, pool in enumerate(p_pools):
            for li, buf in enumerate(pool.kv_buffer):
                shift = r * 100 + li * 40
                valid_vals = (
                    ((torch.arange(num_valid_p, dtype=torch.int64) + shift) % 256)
                    .to(torch.uint8)
                    .reshape(num_blocks, page_size, 1, kv_packing, head_dim)
                )
                buf.copy_(valid_vals)

        p_aux = [
            [torch.full((num_blocks, 16), 100, dtype=torch.int32)]
            for _ in range(src_tp_size)
        ]
        prefills = [
            _create_tpu_kv_manager(
                DisaggregationMode.PREFILL,
                p_pools[r],
                p_aux[r],
                attn_tp_size=src_tp_size,
                engine_rank=r,
                total_kv_head_num=total_num_head,
            )
            for r in range(src_tp_size)
        ]

        d_pools = [
            _TestRpaV3KVPool(
                num_blocks,
                page_size,
                2,
                kv_packing,
                head_dim,
                n_layers,
            )
            for _ in range(dst_tp_size)
        ]
        d_aux = [
            [torch.zeros(num_blocks, 16, dtype=torch.int32)] for _ in range(dst_tp_size)
        ]
        decodes = [
            _create_tpu_kv_manager(
                DisaggregationMode.DECODE,
                d_pools[r],
                d_aux[r],
                attn_tp_size=dst_tp_size,
                engine_rank=r,
                total_kv_head_num=total_num_head,
            )
            for r in range(dst_tp_size)
        ]

        room = 8413157384759490222
        src_pages = np.array([1, 3], dtype=np.int64)
        dst_pages = np.array([0, 2], dtype=np.int64)
        aux_src_idx = 2
        aux_dst_idx = 1

        decodes[0].register_decode_state(room, dst_pages, aux_index=aux_dst_idx)

        for r in range(src_tp_size):
            prefills[r]._handle_transfer(
                _make_trsfr_msg(
                    room,
                    decodes[0],
                    num_pages,
                    dst_tp_rank=0,
                    decode_attn_tp_size=dst_tp_size,
                )
            )
            self.assertEqual(
                prefills[r].request_status.get(room), KVPoll.WaitingForInput
            )

        for r in range(src_tp_size):
            prefills[r].gather_and_push(
                room,
                src_pages,
                index_start=0,
                is_last_chunk=True,
                aux_index=aux_src_idx,
            )

        _drain_and_handle(decodes[0], count=src_tp_size)
        decodes[0].process_decode_pending(room)
        self.assertEqual(decodes[0].request_status[room], KVPoll.Success)

        for li in range(n_layers):
            for p_i in range(num_pages):
                src_b = src_pages[p_i]
                dst_b = dst_pages[p_i]
                expected_page = torch.cat(
                    [
                        p_pools[0].kv_buffer[li][src_b],
                        p_pools[1].kv_buffer[li][src_b],
                    ],
                    dim=1,
                )
                torch.testing.assert_close(
                    decodes[0].layer_buffers[li][dst_b],
                    expected_page,
                )

        untouched_pages = [b for b in range(num_blocks) if b not in dst_pages]
        for buf in decodes[0].layer_buffers:
            for ub in untouched_pages:
                self.assertTrue((buf[ub] == 0).all().item())

        self.assertTrue((d_aux[0][0][aux_dst_idx] == 100).all().item())

    def test_fanin_2_prefill_into_1_decode_fp16_odd_head(self):
        n_layers = 2
        page_size = 3
        num_blocks = 4
        head_dim = 4
        kv_packing = 2
        total_num_head = 2
        src_tp_size = 2
        dst_tp_size = 1
        num_pages = 2

        p_pools = [
            _TestRpaV3KVPool(
                num_blocks,
                page_size,
                1,
                kv_packing,
                head_dim,
                n_layers,
            )
            for _ in range(src_tp_size)
        ]
        num_valid_p = num_blocks * page_size * 1 * kv_packing * head_dim
        for r, pool in enumerate(p_pools):
            for li, buf in enumerate(pool.kv_buffer):
                shift = r * 100 + li * 40
                valid_vals = (
                    ((torch.arange(num_valid_p, dtype=torch.int64) + shift) % 256)
                    .to(torch.uint8)
                    .reshape(num_blocks, page_size, 1, kv_packing, head_dim)
                )
                buf.copy_(valid_vals)

        p_aux = [
            [torch.full((num_blocks, 16), 100, dtype=torch.int32)]
            for _ in range(src_tp_size)
        ]
        prefills = [
            _create_tpu_kv_manager(
                DisaggregationMode.PREFILL,
                p_pools[r],
                p_aux[r],
                attn_tp_size=src_tp_size,
                engine_rank=r,
                total_kv_head_num=total_num_head,
            )
            for r in range(src_tp_size)
        ]

        d_pools = [
            _TestRpaV3KVPool(
                num_blocks,
                page_size,
                2,
                kv_packing,
                head_dim,
                n_layers,
            )
            for _ in range(dst_tp_size)
        ]
        d_aux = [
            [torch.zeros(num_blocks, 16, dtype=torch.int32)] for _ in range(dst_tp_size)
        ]
        decodes = [
            _create_tpu_kv_manager(
                DisaggregationMode.DECODE,
                d_pools[r],
                d_aux[r],
                attn_tp_size=dst_tp_size,
                engine_rank=r,
                total_kv_head_num=total_num_head,
            )
            for r in range(dst_tp_size)
        ]

        room = 7413157384759490111
        src_pages = np.array([1, 3], dtype=np.int64)
        dst_pages = np.array([0, 2], dtype=np.int64)
        aux_src_idx = 2
        aux_dst_idx = 1

        decodes[0].register_decode_state(room, dst_pages, aux_index=aux_dst_idx)

        for r in range(src_tp_size):
            prefills[r]._handle_transfer(
                _make_trsfr_msg(
                    room,
                    decodes[0],
                    num_pages,
                    dst_tp_rank=0,
                    decode_attn_tp_size=dst_tp_size,
                )
            )
            self.assertEqual(
                prefills[r].request_status.get(room), KVPoll.WaitingForInput
            )

        for r in range(src_tp_size):
            prefills[r].gather_and_push(
                room,
                src_pages,
                index_start=0,
                is_last_chunk=True,
                aux_index=aux_src_idx,
            )

        _drain_and_handle(decodes[0], count=src_tp_size)
        decodes[0].process_decode_pending(room)
        self.assertEqual(decodes[0].request_status[room], KVPoll.Success)

        for li in range(n_layers):
            for p_i in range(num_pages):
                src_b = src_pages[p_i]
                dst_b = dst_pages[p_i]
                expected_page = torch.cat(
                    [
                        p_pools[0].kv_buffer[li][src_b],
                        p_pools[1].kv_buffer[li][src_b],
                    ],
                    dim=1,
                )
                torch.testing.assert_close(
                    decodes[0].layer_buffers[li][dst_b],
                    expected_page,
                )

        untouched_pages = [b for b in range(num_blocks) if b not in dst_pages]
        for buf in decodes[0].layer_buffers:
            for ub in untouched_pages:
                self.assertTrue((buf[ub] == 0).all().item())

        self.assertTrue((d_aux[0][0][aux_dst_idx] == 100).all().item())

    def test_fan_out_one_prefill_into_two_decode(self):
        layers = 2
        ppool = _TestRpaV3KVPool(16, 4, 4, 2, 16, layers, dtype=torch.bfloat16)
        d0pool = _TestRpaV3KVPool(16, 4, 2, 2, 16, layers, dtype=torch.bfloat16)
        d1pool = _TestRpaV3KVPool(16, 4, 2, 2, 16, layers, dtype=torch.bfloat16)
        for buf in ppool.kv_buffer:
            buf.copy_(
                (torch.arange(buf.numel(), dtype=torch.float32) % 100)
                .reshape(buf.shape)
                .to(torch.bfloat16)
            )
        paux = [torch.arange(8 * 16, dtype=torch.int32).reshape(8, 16)]
        d0aux = [torch.zeros(8, 16, dtype=torch.int32)]
        d1aux = [torch.zeros(8, 16, dtype=torch.int32)]

        prefill = _create_tpu_kv_manager(DisaggregationMode.PREFILL, ppool, paux, 1, 0)
        d0 = _create_tpu_kv_manager(DisaggregationMode.DECODE, d0pool, d0aux, 2, 0)
        d1 = _create_tpu_kv_manager(DisaggregationMode.DECODE, d1pool, d1aux, 2, 1)

        room = 901
        src_pages = np.array([1, 7, 12], dtype=np.int64)
        d0_pages = np.array([3, 9, 0], dtype=np.int64)
        d1_pages = np.array([2, 5, 8], dtype=np.int64)
        d0.register_decode_state(room, d0_pages, aux_index=4)
        d1.register_decode_state(room, d1_pages, aux_index=1)
        ep0 = _tcp_endpoint("127.0.0.1", d0.rank_port)
        ep1 = _tcp_endpoint("127.0.0.1", d1.rank_port)
        prefill.transfer_infos[room] = {
            ep0: prefill.get_prefill_transfer_info(
                room,
                ep0,
                3,
                dst_tp_rank=0,
                decode_attn_tp_size=2,
            ),
            ep1: prefill.get_prefill_transfer_info(
                room,
                ep1,
                3,
                dst_tp_rank=1,
                decode_attn_tp_size=2,
            ),
        }

        prefill.gather_and_push(room, src_pages, 0, True, aux_index=6)
        d0._handle_kvdata(d0.server_socket.recv_multipart())
        d1._handle_kvdata(d1.server_socket.recv_multipart())
        d0.process_decode_pending(room)
        d1.process_decode_pending(room)

        self.assertEqual(d0.request_status[room], KVPoll.Success)
        self.assertEqual(d1.request_status[room], KVPoll.Success)
        for li in range(layers):
            for sp, dp in zip(src_pages, d0_pages):
                self.assertTrue(
                    torch.equal(
                        ppool.kv_buffer[li][sp][:, 0:2],
                        d0pool.kv_buffer[li][dp][:, 0:2],
                    ),
                    f"fan-out layer {li}: src lanes[0:2] -> decode0 mismatch",
                )
            for sp, dp in zip(src_pages, d1_pages):
                self.assertTrue(
                    torch.equal(
                        ppool.kv_buffer[li][sp][:, 2:4],
                        d1pool.kv_buffer[li][dp][:, 0:2],
                    ),
                    f"fan-out layer {li}: src lanes[2:4] -> decode1 mismatch",
                )
        self.assertTrue(torch.equal(d0aux[0][4], paux[0][6]))
        self.assertTrue(torch.equal(d1aux[0][1], paux[0][6]))

    def test_fan_in_fp8_gqa_replica_no_reassemble(self):
        layers, page_size, head_dim = 2, 4, 16
        p0pool = _TestRpaV3KVPool(
            16, page_size, 1, 4, head_dim, layers, dtype=torch.bfloat16
        )
        p1pool = _TestRpaV3KVPool(
            16, page_size, 1, 4, head_dim, layers, dtype=torch.bfloat16
        )
        dpool = _TestRpaV3KVPool(
            16, page_size, 1, 4, head_dim, layers, dtype=torch.bfloat16
        )
        for buf in p0pool.kv_buffer:
            buf.copy_(
                (torch.arange(buf.numel(), dtype=torch.float32) % 100)
                .reshape(buf.shape)
                .to(torch.bfloat16)
            )
        for b0, b1 in zip(p0pool.kv_buffer, p1pool.kv_buffer):
            b1.copy_(b0)
        paux = [torch.arange(8 * 16, dtype=torch.int32).reshape(8, 16)]
        daux = [torch.zeros(8, 16, dtype=torch.int32)]

        kw = {"total_kv_head_num": 4}
        p0 = _create_tpu_kv_manager(
            DisaggregationMode.PREFILL, p0pool, paux, 8, 0, **kw
        )
        p1 = _create_tpu_kv_manager(
            DisaggregationMode.PREFILL, p1pool, paux, 8, 1, **kw
        )
        decode = _create_tpu_kv_manager(
            DisaggregationMode.DECODE, dpool, daux, 4, 0, **kw
        )

        room = 904
        src_pages = np.array([2, 5, 9], dtype=np.int64)
        dst_pages = np.array([0, 11, 4], dtype=np.int64)
        decode.register_decode_state(room, dst_pages, aux_index=3)
        ep = _tcp_endpoint("127.0.0.1", decode.rank_port)
        info_kw = {"dst_tp_rank": 0, "decode_attn_tp_size": 4}
        p0.transfer_infos[room] = {
            ep: p0.get_prefill_transfer_info(room, ep, 3, **info_kw)
        }
        p1.transfer_infos[room] = {
            ep: p1.get_prefill_transfer_info(room, ep, 3, **info_kw)
        }

        info0 = p0.transfer_infos[room][ep]
        info1 = p1.transfer_infos[room][ep]
        self.assertEqual(info0.resharding_direction, T.ReshardDirection.FAN_IN)
        self.assertEqual(info0.src_heads, info0.dst_heads)
        self.assertFalse(PendingChunk(0, 1, [b""], [], info0).require_reassemble)
        self.assertFalse(info0.is_src_replica)
        self.assertTrue(info1.is_src_replica)

        p0.gather_and_push(room, src_pages, 0, True, aux_index=6)
        decode._handle_kvdata(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)
        self.assertEqual(decode.request_status[room], KVPoll.Success)
        for li in range(layers):
            for sp, dp in zip(src_pages, dst_pages):
                self.assertTrue(
                    torch.equal(p0pool.kv_buffer[li][sp], dpool.kv_buffer[li][dp]),
                    f"gqa-replica layer {li}: decode lane != prefill lane (whole copy)",
                )
        self.assertTrue(torch.equal(daux[0][3], paux[0][6]))

        self.assertTrue(info1.if_skip_sending)
        p1.gather_and_push(room, src_pages, 0, True, aux_index=6)
        with self.assertRaises(zmq.error.Again):
            decode.server_socket.recv_multipart(flags=zmq.NOBLOCK)
        with decode.decode_pending_lock:
            self.assertEqual(len(decode.decode_pending[room]), 0)
        for li in range(layers):
            for sp, dp in zip(src_pages, dst_pages):
                self.assertTrue(
                    torch.equal(p0pool.kv_buffer[li][sp], dpool.kv_buffer[li][dp])
                )


class TestTpuReshardTransferTimeFailure(CustomTestCase):
    PAGE = 4
    HEAD_DIM = 16
    DTYPE = torch.bfloat16

    def tearDown(self):
        while _ACTIVE_MANAGERS:
            m = _ACTIVE_MANAGERS.pop()
            _cleanup_tpu_kv_manager(m)
        super().tearDown()

    def _fp8_pool(self, lanes: int, layers: int = 2) -> _TestRpaV3KVPool:
        return _TestRpaV3KVPool(
            16, self.PAGE, lanes, 4, self.HEAD_DIM, layers, dtype=self.DTYPE
        )

    def _prefill(self, attn_tp_size: int, lanes: int, engine_rank: int = 0):
        aux = [torch.zeros(8, 16, dtype=torch.int32)]
        return _create_tpu_kv_manager(
            DisaggregationMode.PREFILL,
            self._fp8_pool(lanes),
            aux,
            attn_tp_size,
            engine_rank,
        )

    def _record_and_gather(self, m, room, **info_kw):
        ep = _tcp_endpoint("127.0.0.1", m.rank_port)
        m.transfer_infos[room] = {
            ep: m.get_prefill_transfer_info(room, ep, 3, **info_kw)
        }
        m.update_status(room, KVPoll.WaitingForInput)
        src_pages = np.array([0, 1, 2], dtype=np.int64)
        m.gather_and_push(room, src_pages, 0, True, aux_index=0)

    def test_handshake_does_not_preflight_fail(self):
        m = self._prefill(attn_tp_size=4, lanes=1, engine_rank=0)
        room = 4240
        ep = _tcp_endpoint("127.0.0.1", 55550)
        m.get_prefill_transfer_info(room, ep, 3, dst_tp_rank=0, decode_attn_tp_size=8)
        self.assertNotIn(room, m.failure_records)
        self.assertNotIn(room, m.request_status)

    def test_fp8_odd_head_supported_at_transfer(self):
        m = self._prefill(attn_tp_size=4, lanes=1, engine_rank=0)
        room = 4242
        self._record_and_gather(m, room, dst_tp_rank=0, decode_attn_tp_size=8)
        self.assertIn(m.check_status(room), (KVPoll.WaitingForInput, KVPoll.Success))

    def test_zero_total_fails_cleanly_at_transfer(self):
        m = self._prefill(attn_tp_size=4, lanes=1, engine_rank=0)
        m.kv_args.total_kv_head_num = 0
        room = 4243
        self._record_and_gather(m, room, dst_tp_rank=0, decode_attn_tp_size=8)
        self.assertEqual(m.check_status(room), KVPoll.Failed)
        self.assertIn(room, m.failure_records)


class TestPendingChunkReassembly(CustomTestCase):
    def tearDown(self):
        while _ACTIVE_MANAGERS:
            m = _ACTIVE_MANAGERS.pop()
            _cleanup_tpu_kv_manager(m)
        super().tearDown()

    def test_process_incoming_chunk_no_reassemble(self):
        pending_queue = deque()
        reasm_dict = defaultdict(list)

        info = PrefillTransferInfo(
            room=1,
            decode_endpoint="127.0.0.1:10000",
            num_pages=2,
            kv_packing=4,
            src_tp_rank=0,
            dst_tp_rank=0,
            src_attn_tp_size=1,
            dst_attn_tp_size=1,
            total_num_head=2,
            n_layers=2,
            page_size=1,
        )
        chunk = PendingChunk(
            index_start=0,
            num_pages=2,
            layer_blobs=[b"data1", b"data2"],
            aux_blobs=[],
            prefill_transfer_info=info,
            from_assemble=False,
        )

        PendingChunk.process_incoming_chunk(pending_queue, reasm_dict, chunk)

        self.assertEqual(len(pending_queue), 1)
        self.assertIs(pending_queue[0], chunk)
        self.assertEqual(len(reasm_dict), 0)

    def test_process_incoming_chunk_fan_in_reassemble(self):
        pending_queue = deque()
        reasm_dict = defaultdict(list)

        info0 = PrefillTransferInfo(
            room=1,
            decode_endpoint="127.0.0.1:10000",
            num_pages=2,
            kv_packing=4,
            src_tp_rank=0,
            dst_tp_rank=0,
            src_attn_tp_size=2,
            dst_attn_tp_size=1,
            total_num_head=2,
            n_layers=1,
            page_size=1,
        )
        info1 = PrefillTransferInfo(
            room=1,
            decode_endpoint="127.0.0.1:10000",
            num_pages=2,
            kv_packing=4,
            src_tp_rank=1,
            dst_tp_rank=0,
            src_attn_tp_size=2,
            dst_attn_tp_size=1,
            total_num_head=2,
            n_layers=1,
            page_size=1,
        )

        data0 = b"\x01" * (2 * 2 * 16)
        data1 = b"\x02" * (2 * 2 * 16)

        chunk0 = PendingChunk(
            index_start=0,
            num_pages=2,
            layer_blobs=[data0],
            aux_blobs=[],
            prefill_transfer_info=info0,
        )
        chunk1 = PendingChunk(
            index_start=0,
            num_pages=2,
            layer_blobs=[data1],
            aux_blobs=[],
            prefill_transfer_info=info1,
        )

        stats = _KvStats("assemble", 0)

        PendingChunk.process_incoming_chunk(
            pending_queue, reasm_dict, chunk0, stats=stats
        )
        self.assertEqual(len(pending_queue), 0)
        self.assertEqual(len(reasm_dict[chunk0.assemble_key]), 1)
        self.assertEqual(stats.n, 0)

        PendingChunk.process_incoming_chunk(
            pending_queue, reasm_dict, chunk0, stats=stats
        )
        self.assertEqual(len(pending_queue), 0)
        self.assertEqual(len(reasm_dict[chunk0.assemble_key]), 1)
        self.assertEqual(stats.n, 0)

        PendingChunk.process_incoming_chunk(
            pending_queue, reasm_dict, chunk1, stats=stats
        )
        self.assertEqual(len(pending_queue), 1)
        self.assertNotIn(chunk0.assemble_key, reasm_dict)
        self.assertEqual(stats.n, 1)
        self.assertGreater(stats.nbytes, 0)

        merged = pending_queue.popleft()
        self.assertTrue(merged.from_assemble)
        self.assertEqual(merged.prefill_transfer_info.src_attn_tp_size, 1)

    def test_stage_buf_fan_in_reassemble(self):
        pool = PinnedHostPool(max_bytes=10 * 1024 * 1024)
        pending_queue = deque()
        reasm_dict = defaultdict(list)

        info0 = PrefillTransferInfo(
            room=1,
            decode_endpoint="127.0.0.1:10000",
            num_pages=2,
            kv_packing=4,
            src_tp_rank=0,
            dst_tp_rank=0,
            src_attn_tp_size=2,
            dst_attn_tp_size=1,
            total_num_head=2,
            n_layers=1,
            page_size=1,
        )
        info1 = PrefillTransferInfo(
            room=1,
            decode_endpoint="127.0.0.1:10000",
            num_pages=2,
            kv_packing=4,
            src_tp_rank=1,
            dst_tp_rank=0,
            src_attn_tp_size=2,
            dst_attn_tp_size=1,
            total_num_head=2,
            n_layers=1,
            page_size=1,
        )

        nbytes = 2 * 2 * 16
        data0 = b"\x01" * nbytes
        data1 = b"\x02" * nbytes

        sb0 = pool.acquire(nbytes)
        sb0.copy_bytes(data0)
        sb1 = pool.acquire(nbytes)
        sb1.copy_bytes(data1)

        chunk0 = PendingChunk(
            index_start=0,
            num_pages=2,
            layer_blobs=[sb0],
            aux_blobs=[],
            prefill_transfer_info=info0,
        )
        chunk1 = PendingChunk(
            index_start=0,
            num_pages=2,
            layer_blobs=[sb1],
            aux_blobs=[],
            prefill_transfer_info=info1,
        )

        PendingChunk.process_incoming_chunk(pending_queue, reasm_dict, chunk0)
        PendingChunk.process_incoming_chunk(pending_queue, reasm_dict, chunk1)

        self.assertEqual(len(pending_queue), 1)
        merged = pending_queue.popleft()
        self.assertTrue(merged.from_assemble)
        self.assertIsInstance(merged.layer_blobs[0], bytes)

    def test_stage_buf_fan_out_reassemble(self):
        pool = PinnedHostPool(max_bytes=10 * 1024 * 1024)
        pending_queue = deque()
        reasm_dict = defaultdict(list)

        info = PrefillTransferInfo(
            room=1,
            decode_endpoint="127.0.0.1:10000",
            num_pages=2,
            kv_packing=4,
            src_tp_rank=0,
            dst_tp_rank=1,
            src_attn_tp_size=1,
            dst_attn_tp_size=2,
            total_num_head=2,
            n_layers=1,
            page_size=1,
        )

        nbytes = 2 * 2 * 16
        data = b"\x03" * nbytes
        sb = pool.acquire(nbytes)
        sb.copy_bytes(data)

        chunk = PendingChunk(
            index_start=0,
            num_pages=2,
            layer_blobs=[sb],
            aux_blobs=[],
            prefill_transfer_info=info,
        )

        PendingChunk.process_incoming_chunk(pending_queue, reasm_dict, chunk)

        self.assertEqual(len(pending_queue), 1)
        merged = pending_queue.popleft()
        self.assertTrue(merged.from_assemble)
        self.assertIsInstance(merged.layer_blobs[0], bytes)

    def test_reassemble_chunks_benchmark(self):
        sizes_gb = [0.1, 0.5, 1.0]

        page_size, head_dim, kvp = 128, 128, 4
        per_page = page_size * kvp * head_dim
        for size_gb in sizes_gb:
            size_bytes = int(size_gb * 1024 * 1024 * 1024)
            num_pages = max(1, size_bytes // per_page)
            actual_bytes = num_pages * per_page

            info0 = PrefillTransferInfo(
                room=100,
                decode_endpoint="127.0.0.1:10000",
                num_pages=num_pages,
                kv_packing=kvp,
                src_tp_rank=0,
                dst_tp_rank=0,
                src_attn_tp_size=2,
                dst_attn_tp_size=1,
                total_num_head=2,
                n_layers=1,
                page_size=page_size,
            )
            info1 = PrefillTransferInfo(
                room=100,
                decode_endpoint="127.0.0.1:10000",
                num_pages=num_pages,
                kv_packing=kvp,
                src_tp_rank=1,
                dst_tp_rank=0,
                src_attn_tp_size=2,
                dst_attn_tp_size=1,
                total_num_head=2,
                n_layers=1,
                page_size=page_size,
            )

            blob0 = np.ones(actual_bytes, dtype=np.uint8).tobytes()
            blob1 = np.full(actual_bytes, 2, dtype=np.uint8).tobytes()

            c0 = PendingChunk(
                index_start=0,
                num_pages=num_pages,
                layer_blobs=[blob0],
                aux_blobs=[],
                prefill_transfer_info=info0,
            )
            c1 = PendingChunk(
                index_start=0,
                num_pages=num_pages,
                layer_blobs=[blob1],
                aux_blobs=[],
                prefill_transfer_info=info1,
            )

            out_chunk = PendingChunk.reassemble_chunks([c0, c1])
            self.assertTrue(out_chunk.from_assemble)
            self.assertEqual(len(out_chunk.layer_blobs[0]), actual_bytes)


class TestFanInOddHeadReassembleLayout(CustomTestCase):
    KVP = 4
    DUMMY = 0xEE

    def tearDown(self):
        while _ACTIVE_MANAGERS:
            m = _ACTIVE_MANAGERS.pop()
            _cleanup_tpu_kv_manager(m)
        super().tearDown()

    def setUp(self):
        self._pool = PinnedHostPool(max_bytes=10 * 1024 * 1024)

    def _fan_in_info(
        self, src_rank: int, num_pages: int, page_size: int
    ) -> PrefillTransferInfo:
        return PrefillTransferInfo(
            room=1,
            decode_endpoint="127.0.0.1:10000",
            num_pages=num_pages,
            kv_packing=self.KVP,
            src_tp_rank=src_rank,
            dst_tp_rank=0,
            src_attn_tp_size=8,
            dst_attn_tp_size=4,
            total_num_head=8,
            n_layers=1,
            page_size=page_size,
        )

    def _generate_input_lane(
        self, num_pages: int, page_size: int, head_num: int, head_dim: int
    ):
        adjusted_head_num = max(1, head_num // 2)
        total_size = num_pages * page_size * adjusted_head_num * head_dim * self.KVP
        return np.arange(total_size, dtype=np.uint8).reshape(
            num_pages, page_size, adjusted_head_num, head_dim, self.KVP
        )

    def _reassemble(
        self, num_pages: int, page_size: int, head_num: int, head_dim: int
    ) -> tuple[np.ndarray, np.ndarray]:
        a0 = self._generate_input_lane(
            num_pages, page_size, head_num, head_dim
        ).tobytes()
        a1 = self._generate_input_lane(
            num_pages, page_size, head_num, head_dim
        ).tobytes()
        info_kw = {"num_pages": num_pages, "page_size": page_size}
        c0 = PendingChunk(0, num_pages, [a0], [], self._fan_in_info(0, **info_kw))
        c1 = PendingChunk(0, num_pages, [a1], [], self._fan_in_info(1, **info_kw))
        merged = PendingChunk.reassemble_chunks([c0, c1])
        adjusted_head_num = max(1, head_num // 2)
        out = np.frombuffer(merged.layer_blobs[0], dtype=np.uint8).reshape(
            num_pages, page_size, adjusted_head_num, head_dim, self.KVP
        )
        src0 = np.frombuffer(a0, dtype=np.uint8).reshape(out.shape)
        src1 = np.frombuffer(a1, dtype=np.uint8).reshape(out.shape)
        exp = np.empty_like(out)
        exp[:, :, :, :, 0:2] = src0[:, :, :, :, 0:2]
        exp[:, :, :, :, 2:4] = src1[:, :, :, :, 0:2]
        return out, exp

    def test_page_size_3_is_correct(self):
        out, exp = self._reassemble(num_pages=2, page_size=3, head_num=1, head_dim=3)
        self.assertEqual(
            int((out == self.DUMMY).sum()), 0, "dummy leaked into decode lane"
        )
        np.testing.assert_array_equal(out, exp)


if __name__ == "__main__":
    unittest.main()
