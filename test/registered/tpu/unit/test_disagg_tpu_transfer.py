"""Unit + protocol tests for the ``tpu`` P/D disaggregation transfer backend.

Two layers of coverage:

1. **Data plane** (``transfer.py``) — gather/scatter page round-trips across
   dtypes and the ctypes aux-row codec. Pure CPU tensors, no TPU needed.

2. **Control + data protocol** (``conn.py``) — drives ``TpuKVManager``'s real
   ZMQ message paths (``_handle_trsfr``, ``gather_and_push``,
   ``_handle_kvdata_ch``, ``process_decode_pending``) over a localhost ZMQ
   loopback, asserting the KV bytes land in the right decode pages, aux is
   written on the last chunk only, and the ``KVPoll`` status machine advances
   correctly. Managers are built with ``object.__new__`` + the minimal
   CommonKVManager state our methods touch, so the test needs neither a TPU
   nor torch.distributed nor a bootstrap server.

A final TPU-gated test exercises ``index_select`` / ``index_copy_`` on real
device tensors via the actual fused KV pool.
"""

from __future__ import annotations

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

import importlib.util
import threading
import time
import unittest
from array import array
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import torch
import zmq

from sglang.srt.disaggregation import decode_schedule_batch_mixin
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
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
from sglang.srt.managers.schedule_batch import Range
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase

# Enable Perfetto tracing for tests to exercise tracing code paths
envs.SGLANG_TPU_ENABLE_DISAGGREGATION_PERFETTO_TRACING.set(True)


def _skip_if_no_tpu() -> bool:
    try:
        return importlib.util.find_spec("torch_tpu") is None
    except Exception:
        return True


class _FakePool:
    """Stand-in for TpuFusedMHATokenToKVPool: a list of block-indexed buffers
    ``[num_blocks, page_size, ...]`` (CPU)."""

    def __init__(self, num_blocks, page_size, tail, layers, dtype):
        self.kv_buffer = [
            torch.zeros((num_blocks, page_size, *tail), dtype=dtype)
            for _ in range(layers)
        ]


def _fill_random(pool):
    for buf in pool.kv_buffer:
        # randn then cast preserves a meaningful bit pattern for low-precision
        buf.copy_(torch.randn(buf.shape, dtype=torch.float32).to(buf.dtype))


# --------------------------------------------------------------------------
# 1. Data-plane unit tests (CPU only)
# --------------------------------------------------------------------------


class TestTpuTransferDataPlane(CustomTestCase):
    def _roundtrip_for_dtype(self, dtype):
        src = _FakePool(8, 4, (2, 2, 16), layers=3, dtype=dtype)
        dst = _FakePool(8, 4, (2, 2, 16), layers=3, dtype=dtype)
        _fill_random(src)
        sbufs = T.get_layer_page_buffers(src)
        dbufs = T.get_layer_page_buffers(dst)

        src_pages = np.array([1, 4, 6], dtype=np.int64)
        dst_pages = np.array([7, 0, 3], dtype=np.int64)  # deliberately remapped
        blobs = T.gather_pages_to_host(sbufs, src_pages)
        self.assertEqual(len(blobs), 3)
        T.scatter_pages_from_host(dbufs, dst_pages, blobs, dst_lane_slice=None)

        for li in range(3):
            for sp, dp in zip(src_pages, dst_pages):
                self.assertTrue(
                    torch.equal(src.kv_buffer[li][sp], dst.kv_buffer[li][dp]),
                    f"layer {li} page {sp}->{dp} mismatch for {dtype}",
                )

    def test_roundtrip_bf16(self):
        self._roundtrip_for_dtype(torch.bfloat16)

    def test_roundtrip_float32(self):
        self._roundtrip_for_dtype(torch.float32)

    def test_roundtrip_fp8(self):
        self._roundtrip_for_dtype(torch.float8_e4m3fn)

    # ---- MHA resharding: lane-slice math + sliced gather/scatter -----------

    def _make_info(
        self,
        src_tp: int,
        dst_tp: int,
        engine_rank: int,
        dst_rank: int,
        total_heads: int,
        kvp: int,
        n_layers: int = 2,
    ) -> PrefillTransferInfo:
        return PrefillTransferInfo(
            room=0,
            decode_endpoint="",
            num_pages=0,
            src_tp_rank=engine_rank,
            dst_tp_rank=dst_rank,
            src_attn_tp_size=src_tp,
            dst_attn_tp_size=dst_tp,
            total_num_head=total_heads,
            kv_packing=kvp,
            n_layers=n_layers,
            page_size=1,
        )

    def test_compute_lane_slice_directions(self):
        # total_kv_heads=4, kv_packing=2 (bf16): 1 KV head == 1 packed lane.
        kvp, total = 2, 4
        # equal 2->2: whole per-rank head set at offset 0.
        info = self._make_info(2, 2, 0, 0, total, kvp)
        self.assertEqual(info.src_lane_idx, 0)
        self.assertEqual(info.n_lane_to_send, 2)
        self.assertEqual(info.dst_head_start, 0)
        self.assertFalse(info.if_skip_sending)
        self.assertTrue(info.if_send_aux_data)

        # fan-in 4->2: rank0 -> dst lane 0, rank1 -> dst lane 1 (each sends 1 lane).
        info = self._make_info(4, 2, 0, 0, total, kvp)
        self.assertEqual(info.src_lane_idx, 0)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertEqual(info.dst_head_start, 0)
        self.assertTrue(info.if_send_aux_data)

        info = self._make_info(4, 2, 1, 0, total, kvp)
        self.assertEqual(info.src_lane_idx, 0)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertEqual(info.dst_head_start, 1)
        self.assertFalse(info.if_send_aux_data)

        # fan-out 2->4: one prefill rank feeds decode rank0 (src lane0) and
        # rank1 (src lane1), both landing at dst offset 0.
        info = self._make_info(2, 4, 0, 0, total, kvp)
        self.assertEqual(info.src_lane_idx, 0)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertEqual(info.dst_head_start, 0)
        self.assertTrue(info.if_send_aux_data)

        info = self._make_info(2, 4, 0, 1, total, kvp)
        self.assertEqual(info.src_lane_idx, 1)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertEqual(info.dst_head_start, 0)
        self.assertTrue(info.if_send_aux_data)

    def test_compute_lane_slice_gqa_prefill_replication(self):
        # total_kv_heads=2 < prefill_tp=4: src_replication=2. For FAN_IN,
        # primary ranks 0/2 send while replicated ranks 1/3 are skipped.
        kvp, total = 2, 2
        info = self._make_info(4, 2, 0, 0, total, kvp)
        self.assertEqual(info.src_lane_idx, 0)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertFalse(info.if_skip_sending)
        self.assertTrue(info.if_send_aux_data)

        info = self._make_info(4, 2, 1, 0, total, kvp)
        self.assertTrue(info.if_skip_sending)
        self.assertTrue(info.if_send_aux_data)

        info = self._make_info(4, 2, 2, 0, total, kvp)
        self.assertEqual(info.src_lane_idx, 0)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertFalse(info.if_skip_sending)
        self.assertTrue(info.if_send_aux_data)

        info = self._make_info(4, 2, 3, 0, total, kvp)
        self.assertTrue(info.if_skip_sending)
        self.assertTrue(info.if_send_aux_data)

    def test_compute_lane_slice_gqa_decode_replication(self):
        # total_kv_heads=4 < decode_tp=8: dst_heads=1, decode_replication=2.
        # Decode ranks 0/1 are replicas of head 0, ranks 2/3 replicas of head 1, etc.
        # prefill_tp=2: prefill rank 0 has heads 0 and 1 (lanes 0 and 1).
        kvp, total = 2, 4
        # Prefill rank 0 sends head 0 (lane 0) to decode ranks 0 and 1
        info = self._make_info(2, 8, 0, 0, total, kvp)
        self.assertEqual(info.src_lane_idx, 0)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertEqual(info.dst_head_start, 0)
        self.assertTrue(info.if_send_aux_data)

        info = self._make_info(2, 8, 0, 1, total, kvp)
        self.assertEqual(info.src_lane_idx, 0)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertEqual(info.dst_head_start, 0)
        self.assertTrue(info.if_send_aux_data)

        # Prefill rank 0 sends head 1 (lane 1) to decode ranks 2 and 3
        info = self._make_info(2, 8, 0, 2, total, kvp)
        self.assertEqual(info.src_lane_idx, 1)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertEqual(info.dst_head_start, 0)
        self.assertTrue(info.if_send_aux_data)

        info = self._make_info(2, 8, 0, 3, total, kvp)
        self.assertEqual(info.src_lane_idx, 1)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertEqual(info.dst_head_start, 0)
        self.assertTrue(info.if_send_aux_data)

    def test_compute_lane_slice_gqa_fan_out_prefill_replication(self):
        # prefill_tp=4, decode_tp=8, total_kv_heads=2:
        # Under always-send strategy, all ranks send and none skip.
        kvp, total = 2, 2
        # Rank 0 (primary for head 0) sends to decode rank 0
        info = self._make_info(4, 8, 0, 0, total, kvp)
        self.assertEqual(info.src_lane_idx, 0)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertFalse(info.if_skip_sending)

        # Rank 1 also sends
        info = self._make_info(4, 8, 1, 0, total, kvp)
        self.assertFalse(info.if_skip_sending)

    def test_compute_lane_slice_fp8_odd_head_fan_out(self):
        # FP8 (kv_packing=4), total_num_head=4, src_tp=2, dst_tp=4.
        # Each prefill rank has 2 heads, each decode rank has 1 head.
        # n_head_to_send = 1, so 2 sub-heads < kv_packing=4.
        # n_lane_to_send must be at least 1 lane (ceiling division).
        kvp, total = 4, 4
        info = self._make_info(2, 4, 1, 3, total, kvp)
        self.assertEqual(info.src_heads, 2)
        self.assertEqual(info.dst_heads, 1)
        self.assertEqual(info.n_head_to_send, 1)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertEqual(info.src_lane_idx, 0)

    def test_lane_slice_gather_scatter_roundtrip(self):
        # 5D fused layout [nb, ps, lanes, kv_packing, head_dim]; send src lanes
        # [0:2] into dst lanes [2:4], leaving other lanes untouched.
        src = _FakePool(8, 4, (4, 2, 16), layers=2, dtype=torch.bfloat16)
        dst = _FakePool(8, 4, (4, 2, 16), layers=2, dtype=torch.bfloat16)
        _fill_random(src)
        sbufs = T.get_layer_page_buffers(src)
        dbufs = T.get_layer_page_buffers(dst)
        src_pages = np.array([1, 4, 6], dtype=np.int64)
        dst_pages = np.array([7, 0, 3], dtype=np.int64)

        blobs = T.gather_pages_to_host(sbufs, src_pages, lane_slice=(0, 2))
        T.scatter_pages_from_host(dbufs, dst_pages, blobs, dst_lane_slice=(2, 2))
        for li in range(2):
            for sp, dp in zip(src_pages, dst_pages):
                self.assertTrue(
                    torch.equal(
                        src.kv_buffer[li][sp][:, 0:2], dst.kv_buffer[li][dp][:, 2:4]
                    ),
                    f"layer {li} lane slice {sp}->{dp} mismatch",
                )
            # untouched dst lanes [0:2] stay zero
            self.assertEqual(
                torch.count_nonzero(dst.kv_buffer[li][:, :, 0:2]).item(), 0
            )

    def test_rejects_non_fused_pool(self):
        class TokenPool:  # k_buffer/v_buffer style, no kv_buffer
            k_buffer = [torch.zeros(4, 2, 8)]
            v_buffer = [torch.zeros(4, 2, 8)]

        with self.assertRaises(NotImplementedError):
            T.get_layer_page_buffers(TokenPool())

    def test_bytes_roundtrip_helpers(self):
        t = torch.randn(5, 3, dtype=torch.float32).to(torch.bfloat16)
        blob = T.tensor_to_bytes(t)
        back = T.bytes_to_tensor(blob, torch.bfloat16, (5, 3))
        self.assertTrue(torch.equal(t, back))

    def test_aux_codec_roundtrip(self):
        # Two CPU aux buffers shaped [size, K]; row index == aux_index.
        a = torch.arange(4 * 16, dtype=torch.int32).reshape(4, 16)
        b = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8) + 0.5
        src_args = SimpleNamespace(
            aux_data_ptrs=[a.data_ptr(), b.data_ptr()],
            aux_item_lens=[a[0].nbytes, b[0].nbytes],
        )
        a2 = torch.zeros_like(a)
        b2 = torch.zeros_like(b)
        dst_args = SimpleNamespace(
            aux_data_ptrs=[a2.data_ptr(), b2.data_ptr()],
            aux_item_lens=[a2[0].nbytes, b2[0].nbytes],
        )
        blobs = T.read_aux_row(src_args, aux_index=2)
        T.write_aux_row(dst_args, aux_index=1, blobs=blobs)
        self.assertTrue(torch.equal(a2[1], a[2]))
        self.assertTrue(torch.equal(b2[1], b[2]))
        # untouched rows stay zero
        self.assertTrue(torch.equal(a2[0], torch.zeros(16, dtype=torch.int32)))


# --------------------------------------------------------------------------
# 2. Protocol tests over a real ZMQ loopback (CPU only)
# --------------------------------------------------------------------------


def _bare_manager(
    mode,
    pool,
    aux_tensors,
    attn_tp_size=1,
    engine_rank=0,
    total_kv_head_num=0,
):
    """Construct a TpuKVManager with only the CommonKVManager state our data
    /control-plane methods touch — bypassing distributed/bootstrap init."""
    m = object.__new__(TpuKVManager)
    m._name = "TpuKVManager"
    m.disaggregation_mode = mode
    m.kv_pool = None
    m.attn_dp_size = 1
    m.attn_dp_rank = 0
    m.attn_tp_rank = 0
    m._initialize_trace_id()
    m.request_status = {}
    m.failure_records = {}
    m.failure_lock = threading.Lock()
    m._endpoint_locks = defaultdict(threading.Lock)
    m.bootstrap_timeout = 60
    m.waiting_timeout = 60
    # attn_tp identity for resharding slice math (default single-rank, equal TP).
    m.attn_tp_size = attn_tp_size
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
    m._aux_keepalive = aux_tensors  # keep buffers alive for ctypes

    ctx = zmq.Context.instance()
    m._zmq_ctx = ctx
    m._socket_cache = {}
    m._monitor_cache = {}
    m._socket_lock = threading.Lock()
    # Read by _send_multipart_locked.
    m._socket_send_locks = {}

    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    port = sock.bind_to_random_port("tcp://127.0.0.1")
    m.server_socket = sock
    m.local_ip = "127.0.0.1"
    m.rank_port = port

    if mode == DisaggregationMode.PREFILL:
        m.transfer_infos = {}
        m.req_to_decode_prefix_len = {}
        m._push_pool = ThreadPoolExecutor(max_workers=2)
        # Per-room push completion tracking (out-of-order-drain safe Success).
        m._push_progress = {}
        m._push_progress_lock = threading.Lock()
        # _init_prefill sets these in production; _bare_manager bypasses it, so
        # mirror its opt-in-feature defaults: stats off, single push channel.
        m._gather_stats = None
        m._n_channels = 1
        m._chan_socks = {}
        m._chan_socks_lock = threading.Lock()
    else:
        m.decode_states = {}
        m.decode_states_lock = threading.Lock()
        m.decode_pending = defaultdict(deque)
        m.reasm_pending = defaultdict(lambda: defaultdict(list))
        m.decode_pending_lock = threading.Lock()
        m._chan_reasm = {}
        m._chan_reasm_lock = threading.Lock()
        m._stage_pool = PinnedHostPool(max_bytes=m._stage_pool_budget())
        m._scatter_stats = _KvStats("scatter", 0)
        m._assemble_stats = _KvStats("assemble", 0)
        m._push_to_recv_stats = _KvStats("push_to_recv", 0)
    m.attach_kv_pool(pool)
    return m


class TestTpuKVManagerProtocol(CustomTestCase):
    TAIL = (2, 2, 16)
    DTYPE = torch.bfloat16

    def _make_pair(self, layers=2):
        ppool = _FakePool(16, 4, self.TAIL, layers, self.DTYPE)
        dpool = _FakePool(16, 4, self.TAIL, layers, self.DTYPE)
        _fill_random(ppool)
        # decode aux buffers: [size, K] int32 (mimics MetadataBuffers.output_ids)
        paux = [torch.arange(8 * 16, dtype=torch.int32).reshape(8, 16)]
        daux = [torch.zeros(8, 16, dtype=torch.int32)]
        prefill = _bare_manager(DisaggregationMode.PREFILL, ppool, paux)
        decode = _bare_manager(DisaggregationMode.DECODE, dpool, daux)
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
                b"1",  # aux present
                b"3",  # num pages
                b"1",  # required_dst_info_num
                b"0",  # decode_prefix_len
                b"0",  # dst_tp_rank
                b"1",  # decode_attn_tp_size
            ]
        )
        deadline = time.time() + 5
        while time.time() < deadline:
            if prefill.request_status.get(room) == KVPoll.WaitingForInput:
                break
            time.sleep(0.02)
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

        # prefill aux row 6 should land in decode aux row 3.
        # gather (main thread) + push (bg); decode bg recv stashes; decode
        # main-thread process_decode_pending scatters.
        prefill.gather_and_push(room, src_pages, 0, True, aux_index=6)
        msg = decode.server_socket.recv_multipart()
        decode._handle_kvdata_ch(msg)
        self.assertNotIn(room, decode.request_status)  # not scattered yet
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

        # chunk 0: pages [0:2], not last
        prefill.gather_and_push(room, src_pages[:2], 0, False, aux_index=None)
        decode._handle_kvdata_ch(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)
        self.assertNotIn(room, decode.request_status)  # not Success yet
        # chunk 1: pages [2:4], last, carries aux
        prefill.gather_and_push(room, src_pages[2:], 2, True, aux_index=5)
        decode._handle_kvdata_ch(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)

        self._assert_pages_equal(ppool, dpool, src_pages, dst_pages)
        self.assertTrue(torch.equal(daux[0][2], paux[0][5]))
        self.assertEqual(decode.request_status[room], KVPoll.Success)

    def test_out_of_order_chunks_wait_for_all(self):
        """Regression: chunks are pushed by a pool of workers over parallel
        channels, so the ``is_last`` chunk can arrive/scatter *before* an
        earlier one. The decode must not conclude Success off ``is_last`` alone
        — it waits until every page has landed. Here we deliver the last chunk
        first and assert the room stays un-concluded until the first arrives."""
        prefill, decode, ppool, dpool, paux, daux = self._make_pair()
        room = 4040
        src_pages = np.array([1, 2, 3, 8], dtype=np.int64)
        dst_pages = np.array([10, 0, 6, 2], dtype=np.int64)
        decode.register_decode_state(room, dst_pages, aux_index=2)
        ep = _tcp_endpoint("127.0.0.1", decode.rank_port)
        prefill.transfer_infos[room] = {
            ep: prefill.get_prefill_transfer_info(room, ep, 4)
        }

        # Push chunk 1 (pages [2:4], IS last, carries aux) FIRST, deliver + scatter.
        prefill.gather_and_push(room, src_pages[2:], 2, True, aux_index=5)
        decode._handle_kvdata_ch(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)
        # is_last scattered, but pages [0:2] have not arrived -> NOT Success yet.
        self.assertNotEqual(decode.request_status.get(room), KVPoll.Success)

        # Now the earlier chunk 0 (pages [0:2], not last) arrives.
        prefill.gather_and_push(room, src_pages[:2], 0, False, aux_index=None)
        decode._handle_kvdata_ch(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)

        self.assertEqual(decode.request_status[room], KVPoll.Success)
        self._assert_pages_equal(ppool, dpool, src_pages, dst_pages)
        self.assertTrue(torch.equal(daux[0][2], paux[0][5]))

    def test_prefill_success_waits_for_all_pushes(self):
        """Prefill counterpart: pushes run on ``_push_pool`` and can drain out of
        order. The sender concludes Success only once the drained page count
        reaches the room total, not when any single (e.g. last) push finishes."""
        prefill, *_ = self._make_pair()
        room = 5050
        # Room total is 4 lane-page units, delivered as two 2-unit chunks.
        prefill._note_push_submitted(room, total_units=4)
        prefill._note_push_submitted(room, total_units=4)  # idempotent
        # One chunk's push drains -> only 2/4 units -> not complete yet.
        prefill._note_push_done(room, push_lane_units=2)
        self.assertNotEqual(prefill.request_status.get(room), KVPoll.Success)
        # The other chunk's push drains -> 4/4 -> complete.
        prefill._note_push_done(room, push_lane_units=2)
        self.assertEqual(prefill.request_status[room], KVPoll.Success)

    def test_prefill_failure_suppresses_late_success(self):
        """If one chunk's push fails, a sibling push draining afterwards must
        not overwrite the recorded failure with Success."""
        prefill, *_ = self._make_pair()
        room = 6060
        prefill.request_status[room] = KVPoll.WaitingForInput
        prefill._note_push_submitted(room, total_units=4)
        # A chunk fails (drops the progress entry + records Failed).
        prefill._note_push_failed(room)
        prefill.update_status(room, KVPoll.Failed)
        # A sibling push drains afterwards: must be a no-op (entry gone), not
        # resurrect the room as Success.
        prefill._note_push_done(room, push_lane_units=2)
        self.assertEqual(prefill.request_status[room], KVPoll.Failed)

    def test_parallel_channels_roundtrip(self):
        """#3: with N>1 channels the prefill shards layers across N parallel
        sockets and the decode reassembles them byte-exact (4 layers / 2 ch)."""
        prefill, decode, ppool, dpool, paux, daux = self._make_pair(layers=4)
        prefill._n_channels = 2
        prefill._chan_socks = {}
        prefill._chan_socks_lock = threading.Lock()
        decode._chan_reasm = {}
        decode._chan_reasm_lock = threading.Lock()

        room = 321
        src_pages = np.array([1, 3, 5], dtype=np.int64)
        dst_pages = np.array([0, 2, 4], dtype=np.int64)
        decode.register_decode_state(room, dst_pages, aux_index=2)
        ep = _tcp_endpoint("127.0.0.1", decode.rank_port)
        prefill.transfer_infos[room] = {
            ep: prefill.get_prefill_transfer_info(room, ep, 3)
        }

        prefill.gather_and_push(room, src_pages, 0, True, aux_index=5)
        # 2 channels -> 2 messages; route each into the reassembly buffer.
        for _ in range(2):
            decode._handle_kvdata_ch(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)

        self._assert_pages_equal(ppool, dpool, src_pages, dst_pages)
        self.assertTrue(torch.equal(daux[0][2], paux[0][5]))
        self.assertEqual(decode.request_status[room], KVPoll.Success)

    def test_page_count_mismatch_fails(self):
        prefill, decode, ppool, dpool, *_ = self._make_pair()
        room = 13
        # Realistic precondition: the receiver has already advanced the room
        # to WaitingForInput (update_status(Failed) is a no-op for unknown
        # rooms by design).
        decode.request_status[room] = KVPoll.WaitingForInput
        # decode only has 2 dst pages but prefill will send 3
        decode.register_decode_state(
            room, np.array([0, 1], dtype=np.int64), aux_index=0
        )
        ep = _tcp_endpoint("127.0.0.1", decode.rank_port)
        prefill.transfer_infos[room] = {
            ep: prefill.get_prefill_transfer_info(room, ep, 3)
        }
        prefill.gather_and_push(room, np.array([2, 5, 9], dtype=np.int64), 0, True, 1)
        decode._handle_kvdata_ch(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)
        self.assertEqual(decode.request_status[room], KVPoll.Failed)

    def test_threaded_happy_path(self):
        """Exercise the real background recv loop + push pool, with the
        decode H2D scatter driven on the (test's) main thread like poll()."""
        prefill, decode, ppool, dpool, paux, daux = self._make_pair()
        # bg recv -> stashes host bytes only (inlined _init_decode recv loop).
        decode._serve_router(
            {
                MessageTag.KVDATA_CH: decode._handle_kvdata_ch,
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
            decode.process_decode_pending(room)  # main-thread scatter (like poll)
            if decode.request_status.get(room) == KVPoll.Success:
                break
            time.sleep(0.02)
        self.assertEqual(decode.request_status.get(room), KVPoll.Success)
        self._assert_pages_equal(ppool, dpool, src_pages, dst_pages)
        self.assertTrue(torch.equal(daux[0][4], paux[0][2]))


# --------------------------------------------------------------------------
# 2a. MHA resharding protocol (heterogeneous attn_tp) — CPU only
# --------------------------------------------------------------------------


class TestTpuReshardingProtocol(CustomTestCase):
    """Drive the real gather_and_push / _handle_kvdata_ch / process_decode_pending
    paths across heterogeneous attn_tp, asserting each prefill rank's KV-head
    slice lands in the right decode lanes, aux is written exactly once, and the
    KVPoll status machine reaches Success. ``kv_packing=2`` (bf16) so 1 KV head
    == 1 packed lane; ``total_kv_heads=4``."""

    DTYPE = torch.bfloat16
    PAGE = 4
    HEAD_DIM = 16
    KVP = 2
    TOTAL = 4

    def _pool(self, lanes: int, layers: int = 2) -> _FakePool:
        return _FakePool(
            16, self.PAGE, (lanes, self.KVP, self.HEAD_DIM), layers, self.DTYPE
        )

    def _mgr(self, mode, pool, aux, attn_tp_size: int, engine_rank: int):
        return _bare_manager(mode, pool, aux, attn_tp_size, engine_rank)

    def test_fan_in_two_prefill_into_one_decode(self):
        # prefill_tp=2 (each rank holds 2 KV heads == 2 lanes) -> decode_tp=1
        # (holds all 4 heads == 4 lanes). rank0 fills decode lanes [0:2], rank1
        # fills [2:4]; only rank0 (dst offset 0) writes aux.
        layers = 2
        p0pool, p1pool = self._pool(2, layers), self._pool(2, layers)
        dpool = self._pool(4, layers)
        _fill_random(p0pool)
        _fill_random(p1pool)
        paux = [torch.arange(8 * 16, dtype=torch.int32).reshape(8, 16)]
        daux = [torch.zeros(8, 16, dtype=torch.int32)]

        p0 = self._mgr(DisaggregationMode.PREFILL, p0pool, paux, 2, 0)
        p1 = self._mgr(DisaggregationMode.PREFILL, p1pool, paux, 2, 1)
        decode = self._mgr(DisaggregationMode.DECODE, dpool, daux, 1, 0)

        room = 900
        src_pages = np.array([2, 5, 9], dtype=np.int64)
        dst_pages = np.array([0, 11, 4], dtype=np.int64)
        decode.register_decode_state(room, dst_pages, aux_index=3)
        ep = _tcp_endpoint("127.0.0.1", decode.rank_port)
        info_kw = dict(dst_tp_rank=0, decode_attn_tp_size=1)
        p0.transfer_infos[room] = {
            ep: p0.get_prefill_transfer_info(room, ep, 3, **info_kw)
        }
        p1.transfer_infos[room] = {
            ep: p1.get_prefill_transfer_info(room, ep, 3, **info_kw)
        }

        # Both prefill ranks pass aux_index; only rank0 should actually send it.
        p0.gather_and_push(room, src_pages, 0, True, aux_index=6)
        decode._handle_kvdata_ch(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)
        self.assertNotEqual(decode.request_status.get(room), KVPoll.Success)  # half
        p1.gather_and_push(room, src_pages, 0, True, aux_index=6)
        decode._handle_kvdata_ch(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)

        self.assertEqual(decode.request_status[room], KVPoll.Success)
        for li in range(layers):
            for sp, dp in zip(src_pages, dst_pages):
                self.assertTrue(
                    torch.equal(
                        p0pool.kv_buffer[li][sp][:, 0:2],
                        dpool.kv_buffer[li][dp][:, 0:2],
                    ),
                    f"fan-in layer {li}: rank0 -> decode lanes[0:2] mismatch",
                )
                self.assertTrue(
                    torch.equal(
                        p1pool.kv_buffer[li][sp][:, 0:2],
                        dpool.kv_buffer[li][dp][:, 2:4],
                    ),
                    f"fan-in layer {li}: rank1 -> decode lanes[2:4] mismatch",
                )
        # aux written exactly once (by rank0), equal to prefill row 6.
        self.assertTrue(torch.equal(daux[0][3], paux[0][6]))

    def test_fan_out_one_prefill_into_two_decode(self):
        # prefill_tp=1 (holds all 4 heads == 4 lanes) -> decode_tp=2 (each rank
        # holds 2 heads == 2 lanes). decode rank0 gets src lanes [0:2], decode
        # rank1 gets src lanes [2:4]; both land at their own offset 0 and both
        # receive aux.
        layers = 2
        ppool = self._pool(4, layers)
        d0pool, d1pool = self._pool(2, layers), self._pool(2, layers)
        _fill_random(ppool)
        paux = [torch.arange(8 * 16, dtype=torch.int32).reshape(8, 16)]
        d0aux = [torch.zeros(8, 16, dtype=torch.int32)]
        d1aux = [torch.zeros(8, 16, dtype=torch.int32)]

        prefill = self._mgr(DisaggregationMode.PREFILL, ppool, paux, 1, 0)
        d0 = self._mgr(DisaggregationMode.DECODE, d0pool, d0aux, 2, 0)
        d1 = self._mgr(DisaggregationMode.DECODE, d1pool, d1aux, 2, 1)

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
        d0._handle_kvdata_ch(d0.server_socket.recv_multipart())
        d1._handle_kvdata_ch(d1.server_socket.recv_multipart())
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
        # both decode ranks receive aux at their own index.
        self.assertTrue(torch.equal(d0aux[0][4], paux[0][6]))
        self.assertTrue(torch.equal(d1aux[0][1], paux[0][6]))

    def test_fan_in_fp8_odd_head_end_to_end(self):
        # FP8 odd-head FAN-IN over the real gather -> reassemble -> scatter path
        # with page_size>1 (the production repro). prefill_tp=2 (1 real head/rank
        # == 1 packed lane, kv_packing=4, dummy slots [2:4]) -> decode_tp=1 (2
        # heads == 1 full lane). rank0's head must land in decode slots [0:2],
        # rank1's in [2:4]; the prefill dummy slots must not leak through.
        layers, page_size, head_dim = 2, 4, 16
        tail = (1, 4, head_dim)  # 1 lane, kv_packing=4 (fp8)
        p0pool = _FakePool(16, page_size, tail, layers, torch.bfloat16)
        p1pool = _FakePool(16, page_size, tail, layers, torch.bfloat16)
        dpool = _FakePool(16, page_size, tail, layers, torch.bfloat16)
        _fill_random(p0pool)
        _fill_random(p1pool)
        paux = [torch.arange(8 * 16, dtype=torch.int32).reshape(8, 16)]
        daux = [torch.zeros(8, 16, dtype=torch.int32)]

        kw = dict(total_kv_head_num=2)  # 1 real head/rank; shape would infer 2
        p0 = _bare_manager(DisaggregationMode.PREFILL, p0pool, paux, 2, 0, **kw)
        p1 = _bare_manager(DisaggregationMode.PREFILL, p1pool, paux, 2, 1, **kw)
        decode = _bare_manager(DisaggregationMode.DECODE, dpool, daux, 1, 0, **kw)

        room = 902
        src_pages = np.array([2, 5, 9], dtype=np.int64)
        dst_pages = np.array([0, 11, 4], dtype=np.int64)
        decode.register_decode_state(room, dst_pages, aux_index=3)
        ep = _tcp_endpoint("127.0.0.1", decode.rank_port)
        info_kw = dict(dst_tp_rank=0, decode_attn_tp_size=1)
        p0.transfer_infos[room] = {
            ep: p0.get_prefill_transfer_info(room, ep, 3, **info_kw)
        }
        p1.transfer_infos[room] = {
            ep: p1.get_prefill_transfer_info(room, ep, 3, **info_kw)
        }

        # First source alone must not conclude the room (2 chunks reassemble).
        p0.gather_and_push(room, src_pages, 0, True, aux_index=6)
        decode._handle_kvdata_ch(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)
        self.assertNotEqual(decode.request_status.get(room), KVPoll.Success)
        p1.gather_and_push(room, src_pages, 0, True, aux_index=6)
        decode._handle_kvdata_ch(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)

        self.assertEqual(decode.request_status[room], KVPoll.Success)
        for li in range(layers):
            for sp, dp in zip(src_pages, dst_pages):
                # buffer page: [page_size, n_lane=1, kv_packing=4, head_dim]
                self.assertTrue(
                    torch.equal(
                        p0pool.kv_buffer[li][sp][:, 0, 0:2, :],
                        dpool.kv_buffer[li][dp][:, 0, 0:2, :],
                    ),
                    f"fan-in fp8 layer {li}: rank0 head -> decode slots[0:2]",
                )
                self.assertTrue(
                    torch.equal(
                        p1pool.kv_buffer[li][sp][:, 0, 0:2, :],
                        dpool.kv_buffer[li][dp][:, 0, 2:4, :],
                    ),
                    f"fan-in fp8 layer {li}: rank1 head -> decode slots[2:4]",
                )
        self.assertTrue(torch.equal(daux[0][3], paux[0][6]))

    def test_fan_in_fp8_non_odd_head_end_to_end(self):
        # FP8 non-odd FAN-IN (path A: no reassembly, just lane-sliced scatter).
        # prefill_tp=2 (4 heads/rank == 2 packed lanes, no dummy) -> decode_tp=1
        # (8 heads == 4 lanes). rank0 fills decode lanes [0:2], rank1 [2:4]. Each
        # source writes a disjoint whole-lane range, so require_reassemble is
        # False and process_decode_pending scatters directly.
        layers, page_size, head_dim = 2, 4, 16
        p0pool = _FakePool(16, page_size, (2, 4, head_dim), layers, torch.bfloat16)
        p1pool = _FakePool(16, page_size, (2, 4, head_dim), layers, torch.bfloat16)
        dpool = _FakePool(16, page_size, (4, 4, head_dim), layers, torch.bfloat16)
        _fill_random(p0pool)
        _fill_random(p1pool)
        paux = [torch.arange(8 * 16, dtype=torch.int32).reshape(8, 16)]
        daux = [torch.zeros(8, 16, dtype=torch.int32)]

        kw = dict(total_kv_head_num=8)
        p0 = _bare_manager(DisaggregationMode.PREFILL, p0pool, paux, 2, 0, **kw)
        p1 = _bare_manager(DisaggregationMode.PREFILL, p1pool, paux, 2, 1, **kw)
        decode = _bare_manager(DisaggregationMode.DECODE, dpool, daux, 1, 0, **kw)

        room = 903
        src_pages = np.array([2, 5, 9], dtype=np.int64)
        dst_pages = np.array([0, 11, 4], dtype=np.int64)
        decode.register_decode_state(room, dst_pages, aux_index=3)
        ep = _tcp_endpoint("127.0.0.1", decode.rank_port)
        info_kw = dict(dst_tp_rank=0, decode_attn_tp_size=1)
        p0.transfer_infos[room] = {
            ep: p0.get_prefill_transfer_info(room, ep, 3, **info_kw)
        }
        p1.transfer_infos[room] = {
            ep: p1.get_prefill_transfer_info(room, ep, 3, **info_kw)
        }

        # Prove this config takes path A (no merge): FAN_IN but not reassembled.
        info0 = p0.transfer_infos[room][ep]
        self.assertEqual(info0.resharding_direction, T.ReshardDirection.FAN_IN)
        probe = PendingChunk(0, 1, [b""], [], info0)
        self.assertFalse(probe.require_reassemble)

        p0.gather_and_push(room, src_pages, 0, True, aux_index=6)
        decode._handle_kvdata_ch(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)
        self.assertNotEqual(decode.request_status.get(room), KVPoll.Success)
        p1.gather_and_push(room, src_pages, 0, True, aux_index=6)
        decode._handle_kvdata_ch(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)

        self.assertEqual(decode.request_status[room], KVPoll.Success)
        for li in range(layers):
            for sp, dp in zip(src_pages, dst_pages):
                # buffer page: [page_size, n_lanes, kv_packing, head_dim]
                self.assertTrue(
                    torch.equal(
                        p0pool.kv_buffer[li][sp][:, 0:2],
                        dpool.kv_buffer[li][dp][:, 0:2],
                    ),
                    f"fan-in fp8 non-odd layer {li}: rank0 -> decode lanes[0:2]",
                )
                self.assertTrue(
                    torch.equal(
                        p1pool.kv_buffer[li][sp][:, 0:2],
                        dpool.kv_buffer[li][dp][:, 2:4],
                    ),
                    f"fan-in fp8 non-odd layer {li}: rank1 -> decode lanes[2:4]",
                )
        self.assertTrue(torch.equal(daux[0][3], paux[0][6]))

    @unittest.skip(
        "Resource temporarily unavailable in CI due to zmq socket contention"
    )
    def test_fan_in_fp8_gqa_replica_no_reassemble(self):
        # total_num_head=4, prefill_tp=8 -> decode_tp=4: each rank holds exactly 1
        # head (fp8 lane == [head, dummy]), so src_heads == dst_heads. This is
        # FAN_IN by attn_tp but needs NO resharding -- a direct whole-lane copy.
        # The extra prefill rank is a pure GQA replica. Guards the fix that stops
        # reassembly from packing two replicas into [head, head].
        layers, page_size, head_dim = 2, 4, 16
        tail = (1, 4, head_dim)  # 1 lane, kv_packing=4 (fp8)
        p0pool = _FakePool(16, page_size, tail, layers, torch.bfloat16)
        p1pool = _FakePool(16, page_size, tail, layers, torch.bfloat16)
        dpool = _FakePool(16, page_size, tail, layers, torch.bfloat16)
        _fill_random(p0pool)
        # rank 1 is a GQA replica of rank 0: identical KV data.
        for b0, b1 in zip(p0pool.kv_buffer, p1pool.kv_buffer):
            b1.copy_(b0)
        paux = [torch.arange(8 * 16, dtype=torch.int32).reshape(8, 16)]
        daux = [torch.zeros(8, 16, dtype=torch.int32)]

        kw = dict(total_kv_head_num=4)  # 1 head/rank on both sides
        # prefill ranks 0 and 1 are GQA replicas of head 0 -> decode rank 0.
        p0 = _bare_manager(DisaggregationMode.PREFILL, p0pool, paux, 8, 0, **kw)
        p1 = _bare_manager(DisaggregationMode.PREFILL, p1pool, paux, 8, 1, **kw)
        decode = _bare_manager(DisaggregationMode.DECODE, dpool, daux, 4, 0, **kw)

        room = 904
        src_pages = np.array([2, 5, 9], dtype=np.int64)
        dst_pages = np.array([0, 11, 4], dtype=np.int64)
        decode.register_decode_state(room, dst_pages, aux_index=3)
        ep = _tcp_endpoint("127.0.0.1", decode.rank_port)
        info_kw = dict(dst_tp_rank=0, decode_attn_tp_size=4)
        p0.transfer_infos[room] = {
            ep: p0.get_prefill_transfer_info(room, ep, 3, **info_kw)
        }
        p1.transfer_infos[room] = {
            ep: p1.get_prefill_transfer_info(room, ep, 3, **info_kw)
        }

        # No repack: FAN_IN but src_heads == dst_heads, so require_reassemble False.
        info0 = p0.transfer_infos[room][ep]
        info1 = p1.transfer_infos[room][ep]
        self.assertEqual(info0.resharding_direction, T.ReshardDirection.FAN_IN)
        self.assertEqual(info0.src_heads, info0.dst_heads)
        self.assertFalse(PendingChunk(0, 1, [b""], [], info0).require_reassemble)
        # rank0 is the group leader; rank1 is the pure GQA replica to be deduped.
        self.assertFalse(info0.is_src_replica)
        self.assertTrue(info1.is_src_replica)

        # A single replica fully fills the decode rank (no waiting to merge), and
        # the whole lane is copied verbatim -- dummy slots [2:4] are NOT polluted
        # with a second copy of the head.
        p0.gather_and_push(room, src_pages, 0, True, aux_index=6)
        decode._handle_kvdata_ch(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)
        self.assertEqual(decode.request_status[room], KVPoll.Success)
        for li in range(layers):
            for sp, dp in zip(src_pages, dst_pages):
                self.assertTrue(
                    torch.equal(p0pool.kv_buffer[li][sp], dpool.kv_buffer[li][dp]),
                    f"gqa-replica layer {li}: decode lane != prefill lane (whole copy)",
                )
        self.assertTrue(torch.equal(daux[0][3], paux[0][6]))

        # The replica arriving afterwards is deduped at prefill, so no redundant
        # message is sent or enqueued for a second scatter.
        self.assertTrue(info1.if_skip_sending)
        p1.gather_and_push(room, src_pages, 0, True, aux_index=6)
        with decode.decode_pending_lock:
            self.assertEqual(len(decode.decode_pending[room]), 0)
        decode.process_decode_pending(room)
        for li in range(layers):
            for sp, dp in zip(src_pages, dst_pages):
                self.assertTrue(
                    torch.equal(p0pool.kv_buffer[li][sp], dpool.kv_buffer[li][dp])
                )


class TestTpuReshardTransferTimeFailure(CustomTestCase):
    """An unsupported fp8 reshard is NOT pre-flighted at handshake — a Failed set
    on the TRSFR recv thread would be clobbered by the sender's later
    WaitingForInput/Transferring (update_status uses max() and KVPoll.Failed == 0).
    It instead fails cleanly at transfer time in ``gather_and_push`` — the state
    machine's last step, where Failed sticks — mirroring how mooncake fails a
    transfer in send_kvcache. Pools use tail ``(lanes, 4, hd)`` so
    ``_on_pool_attached`` reads ``_kv_packing=4`` (fp8) from the shape."""

    PAGE = 4
    HEAD_DIM = 16
    DTYPE = torch.bfloat16  # _kv_packing comes from shape[3], not the dtype

    def _fp8_pool(self, lanes: int, layers: int = 2) -> _FakePool:
        return _FakePool(16, self.PAGE, (lanes, 4, self.HEAD_DIM), layers, self.DTYPE)

    def _prefill(self, attn_tp_size: int, lanes: int, engine_rank: int = 0):
        aux = [torch.zeros(8, 16, dtype=torch.int32)]
        return _bare_manager(
            DisaggregationMode.PREFILL,
            self._fp8_pool(lanes),
            aux,
            attn_tp_size,
            engine_rank,
        )

    def _record_and_gather(self, m, room, **info_kw):
        # The handshake just records the endpoint (no pre-flight); the reshard
        # compatibility is only exercised when gather_and_push resolves the slice.
        ep = _tcp_endpoint("127.0.0.1", m.rank_port)
        m.transfer_infos[room] = {
            ep: m.get_prefill_transfer_info(room, ep, 3, **info_kw)
        }
        # The sender advances this room to WaitingForInput before send() calls
        # gather; seed it so the terminal Failed lands (update_status won't
        # create a fresh entry for Failed — the anti-resurrection guard). The
        # transfer-time Failed overwriting this WaitingForInput is exactly the
        # "Failed sticks at the end of the state machine" property under test.
        m.update_status(room, KVPoll.WaitingForInput)
        src_pages = np.array([0, 1, 2], dtype=np.int64)
        m.gather_and_push(room, src_pages, 0, True, aux_index=0)

    def test_handshake_does_not_preflight_fail(self):
        # record_decode_endpoint must NOT touch request_status/failure_records
        # for an unsupported config — failing there gets clobbered by max().
        m = self._prefill(attn_tp_size=4, lanes=1, engine_rank=0)
        room = 4240
        ep = _tcp_endpoint("127.0.0.1", 55550)
        m.get_prefill_transfer_info(room, ep, 3, dst_tp_rank=0, decode_attn_tp_size=8)
        self.assertNotIn(room, m.failure_records)
        self.assertNotIn(room, m.request_status)

    def test_fp8_odd_head_supported_at_transfer(self):
        # FP8 odd-head (1 head) resharding is supported via PendingChunk reassembly.
        m = self._prefill(attn_tp_size=4, lanes=1, engine_rank=0)
        room = 4242
        self._record_and_gather(m, room, dst_tp_rank=0, decode_attn_tp_size=8)
        self.assertIn(m.check_status(room), (KVPoll.WaitingForInput, KVPoll.Success))

    def test_zero_total_fails_cleanly_at_transfer(self):
        # kv_args.total_kv_head_num=0: also an unsupported reshard, failed cleanly.
        m = self._prefill(attn_tp_size=4, lanes=1, engine_rank=0)
        m.kv_args.total_kv_head_num = 0
        room = 4243
        self._record_and_gather(m, room, dst_tp_rank=0, decode_attn_tp_size=8)
        self.assertEqual(m.check_status(room), KVPoll.Failed)
        self.assertIn(room, m.failure_records)


# --------------------------------------------------------------------------
# 2b. Pinned host staging pool (#1) — CPU only
# --------------------------------------------------------------------------


class TestPinnedHostPool(CustomTestCase):
    def test_acquire_release_reuse_and_budget(self):
        from sglang.srt.disaggregation.tpu.host_staging import PinnedHostPool

        # Budget holds exactly two 128-byte buffers.
        pool = PinnedHostPool(max_bytes=256)
        a = pool.acquire(128)
        ptr_a = a._t.data_ptr()
        self.assertEqual(a._t.numel(), 128)
        a.release()
        # same-size acquire reuses the freed buffer
        b = pool.acquire(128)
        self.assertEqual(b._t.data_ptr(), ptr_a)
        # different size -> different bucket, fresh buffer
        c = pool.acquire(256)
        self.assertNotEqual(c._t.data_ptr(), ptr_a)
        b.release()
        c.release()  # 256-byte buffer alone already hits the 256B budget edge
        # Retention is byte-bounded: releasing more than the budget drops extras.
        xs = [pool.acquire(128) for _ in range(3)]
        for x in xs:
            x.release()
        self.assertLessEqual(pool._pooled_bytes, 256)
        self.assertLessEqual(len(pool._free[128]), 2)

    def test_munlock_on_evict_and_clear(self):
        from sglang.srt.disaggregation.tpu.host_staging import PinnedHostPool

        pool = PinnedHostPool(max_bytes=128)  # room for one 128B buffer
        a = pool.acquire(128)
        b = pool.acquire(128)
        a.release()  # retained (within budget)
        b.release()  # over budget -> dropped, and unlocked
        self.assertFalse(b._locked)  # evicted buffer must not stay mlock'd
        pool.clear()  # munlocks + drops the retained one too
        self.assertFalse(a._locked)
        self.assertEqual(pool._pooled_bytes, 0)

    def test_copy_bytes_and_view(self):
        from sglang.srt.disaggregation.tpu.host_staging import PinnedHostPool

        pool = PinnedHostPool()
        t = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
        st = pool.acquire(t.numel() * t.element_size())
        st.copy_bytes(T.tensor_to_bytes(t))
        self.assertTrue(torch.equal(st.view(torch.bfloat16, (2, 3)), t))
        self.assertEqual(len(st.mv()), t.numel() * t.element_size())


# --------------------------------------------------------------------------
# 3. Real-device gather/scatter (requires a TPU)
# --------------------------------------------------------------------------


@unittest.skipIf(_skip_if_no_tpu(), "Requires torch_tpu / a TPU device")
class TestTpuTransferOnDevice(CustomTestCase):
    def test_device_gather_scatter_roundtrip(self):
        from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool

        kwargs = dict(
            size=64,
            page_size=32,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=128,
            layer_num=2,
            device="tpu",
            enable_memory_saver=False,
        )
        src = TpuFusedMHATokenToKVPool(**kwargs)
        dst = TpuFusedMHATokenToKVPool(**kwargs)
        for buf in src.kv_buffer:
            buf.copy_(
                torch.randn(buf.shape, dtype=torch.float32).to(buf.dtype).to("tpu")
            )

        sbufs = T.get_layer_page_buffers(src)
        dbufs = T.get_layer_page_buffers(dst)
        src_pages = np.array([0, 1], dtype=np.int64)
        dst_pages = np.array([1, 0], dtype=np.int64)

        blobs = T.gather_pages_to_host(sbufs, src_pages)
        T.scatter_pages_from_host(dbufs, dst_pages, blobs, dst_lane_slice=None)

        for li in range(2):
            for sp, dp in zip(src_pages, dst_pages):
                a = src.kv_buffer[li][sp].to("cpu").float()
                b = dst.kv_buffer[li][dp].to("cpu").float()
                self.assertTrue(torch.equal(a, b), f"layer {li} page {sp}->{dp}")

    def test_stage_path_roundtrip(self):
        """Zero-copy pinned-staging path (#1/#4): gather_pages_to_stage ->
        memoryview bytes (the wire) -> scatter_stage_from_host, byte-exact."""
        from sglang.srt.disaggregation.tpu.host_staging import PinnedHostPool
        from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool

        kwargs = dict(
            size=64,
            page_size=32,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=128,
            layer_num=2,
            device="tpu",
            enable_memory_saver=False,
        )
        src = TpuFusedMHATokenToKVPool(**kwargs)
        dst = TpuFusedMHATokenToKVPool(**kwargs)
        for buf in src.kv_buffer:
            buf.copy_(
                torch.randn(buf.shape, dtype=torch.float32).to(buf.dtype).to("tpu")
            )

        sbufs = T.get_layer_page_buffers(src)
        dbufs = T.get_layer_page_buffers(dst)
        src_pages = np.array([0, 1], dtype=np.int64)
        dst_pages = np.array([1, 0], dtype=np.int64)

        pool = PinnedHostPool()
        stages = T.gather_pages_to_stage(sbufs, src_pages, pool)
        # Simulate the wire: copy the pinned memoryviews out to plain bytes.
        blobs = [bytes(st.mv()) for st in stages]
        for st in stages:
            st.release()
        T.scatter_stage_from_host(dbufs, dst_pages, blobs, pool, dst_lane_slice=None)

        for li in range(2):
            for sp, dp in zip(src_pages, dst_pages):
                a = src.kv_buffer[li][sp].to("cpu").float()
                b = dst.kv_buffer[li][dp].to("cpu").float()
                self.assertTrue(torch.equal(a, b), f"stage layer {li} {sp}->{dp}")

    def test_scatter_from_prestaged_buffers(self):
        """One-copy recv path: ``conn._recv_layers`` copies the wire bytes once,
        straight into pooled StageBufs, and ``scatter_stage_from_host`` consumes
        those StageBufs directly (no second copy), then releases them to the
        pool. Byte-exact, and the buffers return to the pool for reuse."""
        from sglang.srt.disaggregation.tpu.host_staging import PinnedHostPool, StageBuf
        from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool

        kwargs = dict(
            size=64,
            page_size=32,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=128,
            layer_num=2,
            device="tpu",
            enable_memory_saver=False,
        )
        src = TpuFusedMHATokenToKVPool(**kwargs)
        dst = TpuFusedMHATokenToKVPool(**kwargs)
        for buf in src.kv_buffer:
            buf.copy_(
                torch.randn(buf.shape, dtype=torch.float32).to(buf.dtype).to("tpu")
            )

        sbufs = T.get_layer_page_buffers(src)
        dbufs = T.get_layer_page_buffers(dst)
        src_pages = np.array([0, 1], dtype=np.int64)
        dst_pages = np.array([1, 0], dtype=np.int64)

        pool = PinnedHostPool()
        blobs = [
            bytes(st.mv()) for st in T.gather_pages_to_stage(sbufs, src_pages, pool)
        ]
        # Mirror _recv_layers: copy each wire blob once into a pooled StageBuf.
        prestaged = [pool.acquire(len(b)) for b in blobs]
        for st, b in zip(prestaged, blobs):
            st.copy_bytes(b)
        self.assertTrue(all(isinstance(st, StageBuf) for st in prestaged))

        T.scatter_stage_from_host(
            dbufs, dst_pages, prestaged, pool, dst_lane_slice=None
        )

        for li in range(2):
            for sp, dp in zip(src_pages, dst_pages):
                a = src.kv_buffer[li][sp].to("cpu").float()
                b = dst.kv_buffer[li][dp].to("cpu").float()
                self.assertTrue(torch.equal(a, b), f"prestaged layer {li} {sp}->{dp}")
        # scatter_stage_from_host released the pre-staged buffers back to the pool.
        self.assertTrue(pool._free[prestaged[0].nbytes])


# --------------------------------------------------------------------------
# 4. Manager staging path end-to-end on device (requires a TPU)
#
# The CPU protocol tests above always take the bytes fallback: their fake pool
# is CPU, so gather_and_push / _recv_layers / process_decode_pending skip the
# pooled-staging branch (guarded on ``device.type != "cpu"``). These tests drive
# the *default production* data path through the manager on real device tensors:
# pooled pinned staging + zero-copy send + one-copy recv + H2D scatter, plus the
# async D2H gather (conn.py:280-284, gather_pages_to_stage_async) and the
# parallel-channel zero-copy send — none of which were previously exercised.
# --------------------------------------------------------------------------


@unittest.skipIf(_skip_if_no_tpu(), "Requires torch_tpu / a TPU device")
class TestTpuKVManagerStagingOnDevice(CustomTestCase):
    # 4 blocks (size/page_size), 4 layers so 2 channels shard 2 layers each.
    POOL_KWARGS = dict(
        size=128,
        page_size=32,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=128,
        layer_num=4,
        device="tpu",
        enable_memory_saver=False,
    )

    def _make_device_pair(self, async_d2h, n_channels):
        from sglang.srt.disaggregation.tpu.host_staging import PinnedHostPool
        from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool

        src = TpuFusedMHATokenToKVPool(**self.POOL_KWARGS)
        dst = TpuFusedMHATokenToKVPool(**self.POOL_KWARGS)
        for buf in src.kv_buffer:
            buf.copy_(
                torch.randn(buf.shape, dtype=torch.float32).to(buf.dtype).to("tpu")
            )
        # aux buffers stay on host (like MetadataBuffers); ctypes codec is CPU.
        paux = [torch.arange(8 * 16, dtype=torch.int32).reshape(8, 16)]
        daux = [torch.zeros(8, 16, dtype=torch.int32)]

        prefill = _bare_manager(DisaggregationMode.PREFILL, src, paux)
        decode = _bare_manager(DisaggregationMode.DECODE, dst, daux)
        # Turn on the pooled-staging data path on both sides (what _init_prefill
        # / _init_decode set in production; _bare_manager omits them).
        prefill._stage_pool = PinnedHostPool()
        prefill._async_d2h = async_d2h
        prefill._n_channels = n_channels
        prefill._chan_socks = {}
        prefill._chan_socks_lock = threading.Lock()
        decode._stage_pool = PinnedHostPool()
        decode._chan_reasm = {}
        decode._chan_reasm_lock = threading.Lock()
        return prefill, decode, src, dst, paux, daux

    def _recv_and_scatter(self, decode, n_channels, room):
        if n_channels > 1:
            for _ in range(n_channels):
                decode._handle_kvdata_ch(decode.server_socket.recv_multipart())
        else:
            decode._handle_kvdata_ch(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)

    def _run(self, async_d2h, n_channels):
        prefill, decode, src, dst, paux, daux = self._make_device_pair(
            async_d2h, n_channels
        )
        room = 700 + n_channels * 2 + int(async_d2h)
        src_pages = np.array([0, 1, 2], dtype=np.int64)
        dst_pages = np.array([2, 0, 1], dtype=np.int64)  # deliberately remapped
        decode.register_decode_state(room, dst_pages, aux_index=3)
        ep = _tcp_endpoint("127.0.0.1", decode.rank_port)
        n_pages = len(src_pages)
        prefill.transfer_infos[room] = {
            ep: prefill.get_prefill_transfer_info(room, ep, n_pages)
        }

        prefill.gather_and_push(room, src_pages, 0, True, aux_index=6)
        self._recv_and_scatter(decode, n_channels, room)

        for li in range(len(src.kv_buffer)):
            for sp, dp in zip(src_pages, dst_pages):
                a = src.kv_buffer[li][sp].to("cpu").float()
                b = dst.kv_buffer[li][dp].to("cpu").float()
                self.assertTrue(torch.equal(a, b), f"layer {li} page {sp}->{dp}")
        self.assertTrue(torch.equal(daux[0][3], paux[0][6]))
        self.assertEqual(decode.request_status[room], KVPoll.Success)
        # Prove the pooled-staging branch actually ran (not the bytes fallback):
        # scatter acquired + released StageBufs, so the pool holds free buffers.
        self.assertTrue(
            any(decode._stage_pool._free.values()),
            "decode never used the staging pool -> took the bytes fallback",
        )

    def test_manager_staging_sync(self):
        """Pooled staging + zero-copy send, synchronous D2H, single channel."""
        self._run(async_d2h=False, n_channels=1)

    def test_manager_staging_async(self):
        """Async D2H gather (gather_pages_to_stage_async + conn.py:280 branch)."""
        self._run(async_d2h=True, n_channels=1)

    def test_manager_staging_async_channels(self):
        """The full default production config: async D2H + 2 zero-copy channels
        + pooled staging, reassembled across channels into the decode pages."""
        self._run(async_d2h=True, n_channels=2)


class TestPrepareForPrebuiltTpuPadding(CustomTestCase):
    def test_prepare_for_prebuilt_padding(self):
        class MockScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):
            def __init__(self):
                self.device = torch.device("cpu")
                self.forward_mode = None
                self.return_logprob = False
                self.model_config = SimpleNamespace(vocab_size=32000)

                req = SimpleNamespace()
                req.req_pool_idx = 1
                req.prefix_indices = list(range(10))
                req.get_fill_ids = lambda: array("q", range(110))
                req.extend_range = Range(10, 110)  # length = 100
                req.origin_input_ids = list(range(110))
                req.output_ids = []
                req.retracted_stain = False
                req.cached_tokens = 0
                req.cached_tokens_device = 0
                req.already_computed = 0
                req.is_retracted = False
                req.multimodal_inputs = None
                req.sampling_params = SamplingParams()
                # Read by SamplingBatchInfo.from_schedule_batch (Req default).
                req.return_sampling_mask = False

                self.reqs = [req]
                self.req_to_token_pool = SimpleNamespace(
                    req_to_token=torch.zeros((10, 500), dtype=torch.int32)
                )
                self.tree_cache = None

        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        batch = MockScheduleBatch()
        orig_is_tpu = decode_schedule_batch_mixin._is_tpu
        try:
            decode_schedule_batch_mixin._is_tpu = True
            batch.prepare_for_prebuilt()
            self.assertIsNotNone(batch.out_cache_loc)
            self.assertEqual(batch.out_cache_loc.shape[0], 100)
            self.assertEqual(batch.out_cache_loc.device.type, "cpu")

            # Test non-TPU path
            decode_schedule_batch_mixin._is_tpu = False
            batch.prepare_for_prebuilt()
            self.assertIsNotNone(batch.out_cache_loc)
            self.assertEqual(batch.out_cache_loc.shape[0], 100)
        finally:
            decode_schedule_batch_mixin._is_tpu = orig_is_tpu


class TestTpuPagedTokenToKVPoolAllocatorCpuAssertion(CustomTestCase):
    def test_free_cpu_tensor_succeeds(self):
        from sglang.srt.hardware_backend.tpu.allocator_tpu import (
            TpuPagedTokenToKVPoolAllocator,
        )

        allocator = TpuPagedTokenToKVPoolAllocator(
            size=128,
            page_size=16,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        allocator.clear()

        # Freeing CPU tensor should succeed
        cpu_index = torch.tensor([16, 17, 18], dtype=torch.int32, device="cpu")
        allocator.free(cpu_index)

        # Free group mode on CPU
        allocator.free_group_begin()
        allocator.free(torch.tensor([32, 33], dtype=torch.int32, device="cpu"))
        allocator.free_group_end()

    def test_free_non_cpu_tensor_converts_to_cpu(self):
        from sglang.srt.hardware_backend.tpu.allocator_tpu import (
            TpuPagedTokenToKVPoolAllocator,
        )

        allocator = TpuPagedTokenToKVPoolAllocator(
            size=128,
            page_size=16,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        allocator.clear()

        class FakeDeviceTensor:
            device = SimpleNamespace(type="cuda")

            def cpu(self):
                return torch.tensor([16, 17, 18], dtype=torch.int32, device="cpu")

        fake_dev_tensor = FakeDeviceTensor()
        allocator.free(fake_dev_tensor)


class TestDecodeReqToTokenPoolTpuHostResident(CustomTestCase):
    def test_tpu_host_resident(self):
        from sglang.srt.disaggregation.decode import DecodeReqToTokenPool

        pool = DecodeReqToTokenPool(
            size=16,
            max_context_len=512,
            device="tpu:0",
            enable_memory_saver=False,
            pre_alloc_size=16,
        )
        self.assertTrue(pool.host_resident)
        self.assertEqual(pool.req_to_token.device.type, "cpu")


class TestPendingChunkReassembly(CustomTestCase):
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

        # FAN-IN 2 prefill ranks -> 1 decode rank (each prefill rank has 1 head)
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

        # 2 pages, 1 lane (valid + dummy = 2 halves), each half = 16 bytes
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

        # Step 1: Push chunk0 from rank 0
        PendingChunk.process_incoming_chunk(
            pending_queue, reasm_dict, chunk0, stats=stats
        )
        self.assertEqual(len(pending_queue), 0)
        self.assertEqual(len(reasm_dict[chunk0.assemble_key]), 1)
        self.assertEqual(stats.n, 0)

        # Step 2: Duplicate push chunk0 from rank 0 (should be ignored for reasm bucket)
        PendingChunk.process_incoming_chunk(
            pending_queue, reasm_dict, chunk0, stats=stats
        )
        self.assertEqual(len(pending_queue), 0)
        self.assertEqual(len(reasm_dict[chunk0.assemble_key]), 1)
        self.assertEqual(stats.n, 0)

        # Step 3: Push chunk1 from rank 1 -> triggers reassembly
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
        """Benchmark reassemble_chunks for 0.1GB, 0.5GB, and 1GB chunk sizes."""
        # Sizes in GB to test: 0.1GB (100MB), 0.5GB (500MB), 1.0GB (1000MB)
        sizes_gb = [0.1, 0.5, 1.0]

        print("\n" + "=" * 80)
        print(" [Benchmark] PendingChunk.reassemble_chunks Performance ")
        print("=" * 80)

        # Production RPA v3 fp8 geometry (matches the repro.log): one packed lane
        # of page_size x kv_packing x head_dim; a source carries 1 real head in
        # slots [0:2] and dummy in [2:4]. reassemble slices [0:2] (strided) from
        # 2 sources and concatenates on the kv_packing axis into a full lane.
        page_size, head_dim, kvp = 128, 128, 4
        per_page = page_size * kvp * head_dim  # bytes/page for one source (fp8=1B)
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

            # Generate dummy bytes array for layer blob
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

            # Warmup
            _ = PendingChunk.reassemble_chunks([c0, c1])

            # Measure timing
            t0 = time.perf_counter()
            out_chunk = PendingChunk.reassemble_chunks([c0, c1])
            t1 = time.perf_counter()

            cost_sec = t1 - t0
            cost_ms = cost_sec * 1000.0
            total_input_bytes = actual_bytes * 2
            gbps = (total_input_bytes / (1024**3)) / cost_sec if cost_sec > 0 else 0.0

            print(
                f" Single Chunk Input: {size_gb:.1f} GB ({actual_bytes / 1e6:.1f} MB) | "
                f"Total Input Processed: {total_input_bytes / 1e6:.1f} MB | "
                f"Time: {cost_ms:8.2f} ms | "
                f"Throughput: {gbps:6.2f} GB/s"
            )
            self.assertTrue(out_chunk.from_assemble)
            self.assertEqual(len(out_chunk.layer_blobs[0]), actual_bytes)

        print("=" * 80 + "\n")


class TestFanInOddHeadReassembleLayout(CustomTestCase):
    """Regression guard for FP8 odd-head FAN-IN reassembly on the fused 5D packed
    layout (num_pages, page_size, n_lanes, kv_packing, head_dim).

    A KV head occupies kv_packing slots [0:2] (K,V) per lane; on an odd-head rank
    slots [2:4] are dummy padding. FAN-IN 2:1 strips each source's dummy and
    interleaves two ranks' real heads into one decode lane: rank(dst_head_start=0)
    -> slots [0:2], rank(dst_head_start=1) -> slots [2:4].

    reassemble_chunks must split on the kv_packing (slot) axis, not the flat page
    bytes — the two coincide only when page_size == 1. The prior flat-half split
    dropped the second rank's head and leaked dummy bytes for page_size > 1 (every
    real deployment; the production repro has page_size=128).

    These tests mirror the production repro: src_attn_tp=8 -> dst_attn_tp=4
    (assemble 2 chunks), kv_packing=4, n_lane_to_send=1, page_size > 1.
    """

    KVP = 4  # fp8
    DUMMY = 0xEE

    def setUp(self):
        self._pool = PinnedHostPool(max_bytes=10 * 1024 * 1024)

    def _fan_in_info(
        self, src_rank: int, num_pages: int, page_size: int
    ) -> PrefillTransferInfo:
        # total_num_head=8, src_attn_tp=8 (1 head/rank), dst_attn_tp=4 (2 heads).
        # 2 prefill ranks feed each decode rank (assemble_min_chunks=2). ranks
        # 0 and 1 both map to dst rank 0: rank0->dst_head_start 0, rank1->1.
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
        total_size = num_pages * page_size * adjusted_head_num * self.KVP * head_dim
        return np.arange(total_size, dtype=np.uint8).reshape(
            num_pages, page_size, adjusted_head_num, self.KVP, head_dim
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
            num_pages, page_size, adjusted_head_num, self.KVP, head_dim
        )
        # Hand-built expected lane: rank0's real head (its slots [0:2]) -> decode
        # slots [0:2]; rank1's real head -> decode slots [2:4]; dummy stripped.
        src0 = np.frombuffer(a0, dtype=np.uint8).reshape(out.shape)
        src1 = np.frombuffer(a1, dtype=np.uint8).reshape(out.shape)
        exp = np.empty_like(out)
        exp[:, :, :, 0:2, :] = src0[:, :, :, 0:2, :]
        exp[:, :, :, 2:4, :] = src1[:, :, :, 0:2, :]
        return out, exp

    def test_page_size_3_is_correct(self):
        # Sanity + pinpoint: the flat-half split only coincides with the packed
        # slot split when page_size == 1, so this configuration is correct today.
        out, exp = self._reassemble(num_pages=2, page_size=3, head_num=1, head_dim=3)
        self.assertEqual(
            int((out == self.DUMMY).sum()), 0, "dummy leaked into decode lane"
        )
        print(out)
        np.testing.assert_array_equal(out, exp)


if __name__ == "__main__":
    unittest.main()
