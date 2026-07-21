"""Unit + protocol tests for the ``tpu`` P/D disaggregation transfer backend.

Two layers of coverage:

1. **Data plane** (``transfer.py``) — gather/scatter page round-trips across
   dtypes and the ctypes aux-row codec. Pure CPU tensors, no TPU needed.

2. **Control + data protocol** (``conn.py``) — drives ``TpuKVManager``'s real
   ZMQ message paths (``_handle_trsfr``, ``gather_and_push``,
   ``_handle_kvdata``, ``process_decode_pending``) over a localhost ZMQ
   loopback, asserting the KV bytes land in the right decode pages, aux is
   written on the last chunk only, and the ``KVPoll`` status machine advances
   correctly. Managers are built with ``object.__new__`` + the minimal
   CommonKVManager state our methods touch, so the test needs neither a TPU
   nor torch.distributed nor a bootstrap server.

A final TPU-gated test exercises ``index_select`` / ``index_copy_`` on real
device tensors via the actual fused KV pool.
"""

from __future__ import annotations

import importlib.util
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
    TAG_TRSFR,
    TpuKVManager,
    _PrefillTransferInfo,
    _tcp_endpoint,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.test.test_utils import CustomTestCase

# Enable Perfetto tracing for tests to exercise tracing code paths
envs.SGLANG_ENABLE_PERFETTO_TRACING.set(True)


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
        specs = T.layer_page_specs(sbufs)

        src_pages = np.array([1, 4, 6], dtype=np.int64)
        dst_pages = np.array([7, 0, 3], dtype=np.int64)  # deliberately remapped
        blobs = T.gather_pages_to_host(sbufs, src_pages)
        self.assertEqual(len(blobs), 3)
        T.scatter_pages_from_host(dbufs, dst_pages, blobs, specs)

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


def _bare_manager(mode, pool, aux_tensors):
    """Construct a TpuKVManager with only the CommonKVManager state our data
    /control-plane methods touch — bypassing distributed/bootstrap init."""
    m = object.__new__(TpuKVManager)
    m._name = "TpuKVManager"
    m.kv_pool = None
    m.disaggregation_mode = mode
    m.request_status = {}
    m.failure_records = {}
    m.failure_lock = threading.Lock()
    m._endpoint_locks = defaultdict(threading.Lock)
    m.bootstrap_timeout = 60
    m.waiting_timeout = 60
    m.kv_args = SimpleNamespace(
        aux_data_ptrs=[t.data_ptr() for t in aux_tensors],
        aux_item_lens=[t[0].nbytes for t in aux_tensors],
    )
    m._aux_keepalive = aux_tensors  # keep buffers alive for ctypes

    ctx = zmq.Context.instance()
    m._zmq_ctx = ctx
    m._socket_cache = {}
    m._monitor_cache = {}
    m._socket_lock = threading.Lock()

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
    else:
        m.decode_states = {}
        m.decode_states_lock = threading.Lock()
        m.decode_pending = defaultdict(deque)
        m.decode_pending_lock = threading.Lock()
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
        prefill._start_prefill_thread()
        room = 4242
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(_tcp_endpoint("127.0.0.1", prefill.rank_port))
        push.send_multipart(
            [
                TAG_TRSFR,
                str(room).encode(),
                b"127.0.0.1",
                str(decode.rank_port).encode(),
                b"1",  # aux present
                b"3",  # num pages
                b"1",  # required_dst_info_num
                b"0",  # decode_prefix_len
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
        prefill.transfer_infos[room] = {ep: _PrefillTransferInfo(room, ep, 3)}

        # prefill aux row 6 should land in decode aux row 3.
        # gather (main thread) + push (bg); decode bg recv stashes; decode
        # main-thread process_decode_pending scatters.
        prefill.gather_and_push(room, src_pages, 0, True, aux_index=6)
        msg = decode.server_socket.recv_multipart()
        decode._handle_kvdata(msg)
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
        prefill.transfer_infos[room] = {ep: _PrefillTransferInfo(room, ep, 4)}

        # chunk 0: pages [0:2], not last
        prefill.gather_and_push(room, src_pages[:2], 0, False, aux_index=None)
        decode._handle_kvdata(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)
        self.assertNotIn(room, decode.request_status)  # not Success yet
        # chunk 1: pages [2:4], last, carries aux
        prefill.gather_and_push(room, src_pages[2:], 2, True, aux_index=5)
        decode._handle_kvdata(decode.server_socket.recv_multipart())
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
        prefill.transfer_infos[room] = {ep: _PrefillTransferInfo(room, ep, 4)}

        # Push chunk 1 (pages [2:4], IS last, carries aux) FIRST, deliver + scatter.
        prefill.gather_and_push(room, src_pages[2:], 2, True, aux_index=5)
        decode._handle_kvdata(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)
        # is_last scattered, but pages [0:2] have not arrived -> NOT Success yet.
        self.assertNotEqual(decode.request_status.get(room), KVPoll.Success)

        # Now the earlier chunk 0 (pages [0:2], not last) arrives.
        prefill.gather_and_push(room, src_pages[:2], 0, False, aux_index=None)
        decode._handle_kvdata(decode.server_socket.recv_multipart())
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
        # Room total is 4 pages, delivered as two 2-page chunks.
        prefill._note_push_submitted(room, total_pages=4)
        prefill._note_push_submitted(room, total_pages=4)  # idempotent
        # One chunk's push drains -> only 2/4 pages -> not complete yet.
        prefill._note_push_done(room, n_pages=2)
        self.assertNotEqual(prefill.request_status.get(room), KVPoll.Success)
        # The other chunk's push drains -> 4/4 -> complete.
        prefill._note_push_done(room, n_pages=2)
        self.assertEqual(prefill.request_status[room], KVPoll.Success)

    def test_prefill_failure_suppresses_late_success(self):
        """If one chunk's push fails, a sibling push draining afterwards must
        not overwrite the recorded failure with Success."""
        prefill, *_ = self._make_pair()
        room = 6060
        prefill.request_status[room] = KVPoll.WaitingForInput
        prefill._note_push_submitted(room, total_pages=4)
        # A chunk fails (drops the progress entry + records Failed).
        prefill._note_push_failed(room)
        prefill.update_status(room, KVPoll.Failed)
        # A sibling push drains afterwards: must be a no-op (entry gone), not
        # resurrect the room as Success.
        prefill._note_push_done(room, n_pages=2)
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
        prefill.transfer_infos[room] = {ep: _PrefillTransferInfo(room, ep, 3)}

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
        prefill.transfer_infos[room] = {ep: _PrefillTransferInfo(room, ep, 3)}
        prefill.gather_and_push(room, np.array([2, 5, 9], dtype=np.int64), 0, True, 1)
        decode._handle_kvdata(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)
        self.assertEqual(decode.request_status[room], KVPoll.Failed)

    def test_threaded_happy_path(self):
        """Exercise the real background recv loop + push pool, with the
        decode H2D scatter driven on the (test's) main thread like poll()."""
        prefill, decode, ppool, dpool, paux, daux = self._make_pair()
        decode._start_decode_thread()  # bg recv -> stashes host bytes only
        room = 555
        src_pages = np.array([1, 7, 12], dtype=np.int64)
        dst_pages = np.array([3, 9, 0], dtype=np.int64)
        decode.register_decode_state(room, dst_pages, aux_index=4)
        ep = _tcp_endpoint("127.0.0.1", decode.rank_port)
        prefill.transfer_infos[room] = {ep: _PrefillTransferInfo(room, ep, 3)}
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
        specs = T.layer_page_specs(sbufs)
        src_pages = np.array([0, 1], dtype=np.int64)
        dst_pages = np.array([1, 0], dtype=np.int64)

        blobs = T.gather_pages_to_host(sbufs, src_pages)
        T.scatter_pages_from_host(dbufs, dst_pages, blobs, specs)

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
        specs = T.layer_page_specs(sbufs)
        src_pages = np.array([0, 1], dtype=np.int64)
        dst_pages = np.array([1, 0], dtype=np.int64)

        pool = PinnedHostPool()
        stages = T.gather_pages_to_stage(sbufs, src_pages, pool)
        # Simulate the wire: copy the pinned memoryviews out to plain bytes.
        blobs = [bytes(st.mv()) for st in stages]
        for st in stages:
            st.release()
        T.scatter_stage_from_host(dbufs, dst_pages, blobs, specs, pool)

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
        specs = T.layer_page_specs(sbufs)
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

        T.scatter_stage_from_host(dbufs, dst_pages, prestaged, specs, pool)

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
            decode._handle_kvdata(decode.server_socket.recv_multipart())
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
        prefill.transfer_infos[room] = {ep: _PrefillTransferInfo(room, ep, n_pages)}

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


if __name__ == "__main__":
    unittest.main()
