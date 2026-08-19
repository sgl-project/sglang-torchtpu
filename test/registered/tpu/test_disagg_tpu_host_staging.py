"""Unit + device tests for the TPU P/D host staging pool (PinnedHostPool / StageBuf)."""

from __future__ import annotations

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit")

import importlib.util
import threading
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.tpu import transfer as T
from sglang.srt.disaggregation.tpu.conn import _tcp_endpoint
from sglang.srt.disaggregation.tpu.host_staging import PinnedHostPool
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.test_utils import CustomTestCase


def _skip_if_no_tpu() -> bool:
    try:
        return importlib.util.find_spec("torch_tpu") is None
    except Exception:
        return True


class TestPinnedHostPool(CustomTestCase):
    def test_acquire_release_reuse_and_budget(self):
        pool = PinnedHostPool(max_bytes=256)
        a = pool.acquire(128)
        ptr_a = a._t.data_ptr()
        self.assertEqual(a._t.numel(), 128)
        a.release()
        b = pool.acquire(128)
        self.assertEqual(b._t.data_ptr(), ptr_a)
        c = pool.acquire(256)
        self.assertNotEqual(c._t.data_ptr(), ptr_a)
        b.release()
        c.release()
        xs = [pool.acquire(128) for _ in range(3)]
        for x in xs:
            x.release()
        self.assertLessEqual(pool._pooled_bytes, 256)
        self.assertLessEqual(len(pool._free[128]), 2)

    def test_munlock_on_evict_and_clear(self):
        pool = PinnedHostPool(max_bytes=128)
        a = pool.acquire(128)
        b = pool.acquire(128)
        a.release()
        b.release()
        self.assertFalse(b._locked)
        pool.clear()
        self.assertFalse(a._locked)
        self.assertEqual(pool._pooled_bytes, 0)

    def test_copy_bytes_and_view(self):
        pool = PinnedHostPool()
        t = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
        st = pool.acquire(t.numel() * t.element_size())
        st.copy_bytes(T.tensor_to_bytes(t))
        self.assertTrue(torch.equal(st.view(torch.bfloat16, (2, 3)), t))
        self.assertEqual(len(st.mv()), t.numel() * t.element_size())


@unittest.skipIf(_skip_if_no_tpu(), "Requires torch_tpu / a TPU device")
class TestTpuKVManagerStagingOnDevice(CustomTestCase):
    POOL_KWARGS = {
        "size": 128,
        "page_size": 32,
        "dtype": torch.bfloat16,
        "head_num": 2,
        "head_dim": 128,
        "layer_num": 4,
        "device": "tpu",
        "enable_memory_saver": False,
    }

    def _make_device_pair(self, async_d2h, n_channels):
        from sglang.srt.disaggregation.tpu.conn import TpuKVManager
        from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool

        src = TpuFusedMHATokenToKVPool(**self.POOL_KWARGS)
        dst = TpuFusedMHATokenToKVPool(**self.POOL_KWARGS)
        for buf in src.kv_buffer:
            buf.copy_(
                torch.randn(buf.shape, dtype=torch.float32).to(buf.dtype).to("tpu")
            )
        paux = [torch.arange(8 * 16, dtype=torch.int32).reshape(8, 16)]
        daux = [torch.zeros(8, 16, dtype=torch.int32)]

        def _bare_mgr(mode, pool, aux_tensors):
            from collections import defaultdict, deque
            from concurrent.futures import ThreadPoolExecutor

            import zmq

            from sglang.srt.disaggregation.tpu.conn import _KvStats

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
            m.attn_tp_size = 1
            m.kv_args = SimpleNamespace(
                aux_data_ptrs=[t.data_ptr() for t in aux_tensors],
                aux_item_lens=[t[0].nbytes for t in aux_tensors],
                engine_rank=0,
                total_kv_head_num=0,
            )
            m._aux_keepalive = aux_tensors

            ctx = zmq.Context.instance()
            m._zmq_ctx = ctx
            m._socket_cache = {}
            m._monitor_cache = {}
            m._socket_lock = threading.Lock()
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
                m._push_progress = {}
                m._push_progress_lock = threading.Lock()
                m._gather_stats = None
            else:
                m.decode_states = {}
                m.decode_states_lock = threading.Lock()
                m.decode_pending = defaultdict(deque)
                m.reasm_pending = defaultdict(lambda: defaultdict(list))
                m.decode_pending_lock = threading.Lock()
                m._stage_pool = PinnedHostPool(max_bytes=m._stage_pool_budget())
                m._scatter_stats = _KvStats("scatter", 0)
                m._assemble_stats = _KvStats("assemble", 0)
                m._push_to_recv_stats = _KvStats("push_to_recv", 0)
            m.attach_kv_pool(pool)
            return m

        prefill = _bare_mgr(DisaggregationMode.PREFILL, src, paux)
        decode = _bare_mgr(DisaggregationMode.DECODE, dst, daux)

        prefill._stage_pool = PinnedHostPool()
        prefill._async_d2h = async_d2h
        decode._stage_pool = PinnedHostPool()
        return prefill, decode, src, dst, paux, daux

    def _recv_and_scatter(self, decode, room):
        decode._handle_kvdata(decode.server_socket.recv_multipart())
        decode.process_decode_pending(room)

    def _run(self, async_d2h):
        prefill, decode, src, dst, paux, daux = self._make_device_pair(async_d2h)
        room = 700 + int(async_d2h)
        src_pages = np.array([0, 1, 2], dtype=np.int64)
        dst_pages = np.array([2, 0, 1], dtype=np.int64)
        decode.register_decode_state(room, dst_pages, aux_index=3)
        ep = _tcp_endpoint("127.0.0.1", decode.rank_port)
        n_pages = len(src_pages)
        prefill.transfer_infos[room] = {
            ep: prefill.get_prefill_transfer_info(room, ep, n_pages)
        }

        prefill.gather_and_push(room, src_pages, 0, True, aux_index=6)
        self._recv_and_scatter(decode, room)

        for li in range(len(src.kv_buffer)):
            for sp, dp in zip(src_pages, dst_pages):
                a = src.kv_buffer[li][sp].to("cpu").float()
                b = dst.kv_buffer[li][dp].to("cpu").float()
                self.assertTrue(torch.equal(a, b), f"layer {li} page {sp}->{dp}")
        self.assertTrue(torch.equal(daux[0][3], paux[0][6]))
        self.assertEqual(decode.request_status[room], KVPoll.Success)
        self.assertTrue(
            any(decode._stage_pool._free.values()),
            "decode never used the staging pool -> took the bytes fallback",
        )

    def test_manager_staging_sync(self):
        self._run(async_d2h=False)

    def test_manager_staging_async(self):
        self._run(async_d2h=True)


if __name__ == "__main__":
    unittest.main()
