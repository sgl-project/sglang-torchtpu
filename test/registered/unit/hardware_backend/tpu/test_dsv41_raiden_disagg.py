# SPDX-License-Identifier: Apache-2.0
"""CPU protocol, DMA, C2 pair-state, and HostEngram tests for DeepSeek-V4.1 P/D on Raiden."""

from __future__ import annotations

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

import ctypes
import socket
import sys
import time
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll, StateType
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    get_kv_transfer_buf_infos,
    setup_state_kv_args,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.raiden.common import (
    PORTS_PER_RANK,
    get_raiden_kv_caches,
)
from sglang_tpu.disaggregation.raiden.conn import (
    RaidenKVBootstrapServer,
    RaidenKVManager,
    RaidenKVReceiver,
    RaidenKVSender,
)
from sglang_tpu.dsv4.dsv4_memory_pool import DSV4TPUTokenToKVPool
from sglang_tpu.dsv41.memory_pool import DSV41TPUTokenToKVPool, TPUPairStatePool


def _free_port_block(count: int = 8) -> int:
    for _ in range(64):
        sockets = []
        try:
            s0 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s0.bind(("127.0.0.1", 0))
            base = s0.getsockname()[1]
            sockets.append(s0)
            ok = True
            for offset in range(1, count):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    s.bind(("127.0.0.1", base + offset))
                    sockets.append(s)
                except OSError:
                    ok = False
                    break
            if ok:
                return base
        finally:
            for s in sockets:
                s.close()
    raise RuntimeError(f"Could not find {count} contiguous free ports")


class _InMemoryRaidenRegistry:
    def __init__(self) -> None:
        self.engines: dict[int, _InMemoryKVCacheManager] = {}


class _InMemoryKVCacheManager:
    def __init__(
        self,
        *,
        registry: _InMemoryRaidenRegistry,
        kv_caches: list[torch.Tensor],
        node_id: int,
        local_control_port: int,
        max_blocks: int,
        num_slots: int,
        timeout_s: float = 30.0,
        parallelism: int = 1,
        num_staging_layers: int | None = None,
    ) -> None:
        del node_id, timeout_s, parallelism, num_staging_layers
        self.registry = registry
        self.kv_caches = list(kv_caches)
        self.port = int(local_control_port)
        self.max_blocks = int(max_blocks)
        self.num_slots = int(num_slots)
        self._initial_layout = [
            (tuple(buf.shape), buf.dtype, buf.data_ptr()) for buf in self.kv_caches
        ]
        self._assert_buffers_intact()

        self.registered_reads: dict[tuple[str, int], list[int]] = {}
        self._completed_send: list[str] = []
        self._completed_recv: list[str] = []
        self._failed: list[str] = []
        registry.engines[self.port] = self

    def _assert_buffers_intact(self) -> None:
        for idx, (buf, expected) in enumerate(
            zip(self.kv_caches, self._initial_layout, strict=True)
        ):
            if buf._base is not None:
                raise AssertionError(f"Registered buffer[{idx}] is a view")
            actual = (tuple(buf.shape), buf.dtype, buf.data_ptr())
            if actual != expected:
                raise AssertionError(
                    f"Registered buffer[{idx}] reallocated after registration: "
                    f"{expected} -> {actual}"
                )

    def register_read(self, req_id: str, uuid: int, block_ids: list[int]) -> bool:
        self._assert_buffers_intact()
        blocks = [int(b) for b in block_ids]
        if len(blocks) > self.max_blocks:
            raise AssertionError(
                f"register_read({req_id}) passed {len(blocks)} blocks > "
                f"max_blocks={self.max_blocks}"
            )
        self.registered_reads[(req_id, int(uuid))] = blocks
        return True

    def start_read(
        self,
        req_id: str,
        uuid: int,
        endpoint: str,
        remote_blocks: list[int],
        local_blocks: list[int],
    ) -> bool:
        self._assert_buffers_intact()
        peer_port = int(endpoint.rsplit(":", 1)[1])
        peer = self.registry.engines[peer_port]
        peer._assert_buffers_intact()

        remote = [int(b) for b in remote_blocks]
        local = [int(b) for b in local_blocks]
        if len(remote) != len(local):
            raise AssertionError(f"Mismatched remote={remote} vs local={local}")
        if len(local) > self.max_blocks:
            raise AssertionError(
                f"start_read({req_id}) passed {len(local)} blocks > "
                f"max_blocks={self.max_blocks}"
            )
        published = peer.registered_reads.get((req_id, int(uuid)))
        if published is None or not set(remote).issubset(set(published)):
            raise AssertionError(
                f"start_read({req_id}, uuid={uuid}) expected remote {remote} to be "
                f"a subset of published {published}"
            )
        for src_buf, dst_buf in zip(peer.kv_caches, self.kv_caches, strict=True):
            dst_buf[local] = src_buf[remote].clone()

        peer._completed_send.append(req_id)
        self._completed_recv.append(req_id)
        return True

    def poll_stats_with_details(self):
        def _settled(ids: list[str]):
            return [
                SimpleNamespace(
                    req_id=r,
                    completed_ns=1_000_000,
                    code=None,
                    phase=None,
                    peer=None,
                    elapsed_ns=None,
                    reason=None,
                )
                for r in ids
            ]

        sent = _settled(self._completed_send)
        recv = _settled(self._completed_recv)
        failed = _settled(self._failed)
        self._completed_send.clear()
        self._completed_recv.clear()
        self._failed.clear()
        return SimpleNamespace(sent=sent, received=recv, failed=failed)

    def poll_stats(self):
        details = self.poll_stats_with_details()
        return (
            [t.req_id for t in details.sent],
            [t.req_id for t in details.received],
            [t.req_id for t in details.failed],
            1_000_000,
        )


def _fake_tpu_sync_modules(registry: _InMemoryRaidenRegistry) -> dict[str, object]:
    pkg = types.ModuleType("tpu_sync")
    api = types.ModuleType("tpu_sync.api")
    torch_mod = types.ModuleType("tpu_sync.api.torch")
    mgr_mod = types.ModuleType("tpu_sync.api.torch.kv_cache_manager")

    def _factory(**kwargs):
        return _InMemoryKVCacheManager(registry=registry, **kwargs)

    mgr_mod.KVCacheManager = _factory
    return {
        "tpu_sync": pkg,
        "tpu_sync.api": api,
        "tpu_sync.api.torch": torch_mod,
        "tpu_sync.api.torch.kv_cache_manager": mgr_mod,
    }


def _make_dsv41_pool(
    testcase: unittest.TestCase,
    *,
    num_req_slots: int = 6,
    full_pages: int = 15,  # Deliberately odd to guard C1/C2/Index page-count parity
    swa_pages: int = 7,
    bootstrap_port: int = 8998,
) -> DSV41TPUTokenToKVPool:
    from sglang.srt.runtime_context import get_context, get_parallel

    override = get_context().override_server_args(
        host="127.0.0.1",
        disaggregation_bootstrap_port=bootstrap_port,
        dsv4_attn_backend="tpu_dsv4",
        page_size=256,
        tp_size=1,
        dp_size=1,
        ep_size=1,
        enable_dp_attention=True,
    )
    override.install()
    testcase.addCleanup(override.restore)

    parallel_ctx = get_parallel().override(
        tp_size=1,
        attn_tp_size=1,
        attn_tp_rank=0,
        attn_cp_size=1,
        attn_cp_rank=0,
        attn_dcp_size=1,
        attn_dcp_rank=0,
        attn_dp_size=1,
        attn_dp_rank=0,
        dp_size=1,
        pp_size=1,
        pp_group=None,
        dist_init_addr=None,
        enable_dsa_cache_layer_split=False,
    )
    parallel_ctx.__enter__()
    testcase.addCleanup(parallel_ctx.__exit__, None, None, None)

    # Mix SWA (0), C2 (2, two KV sources + one consumer), and C1 (1, one KV source).
    compression_ratios = [0, 2, 2, 1, 2, 0]
    kv_source_layers = [1, 3, 4]
    pool = DSV4TPUTokenToKVPool(
        max_num_reqs=num_req_slots - 1,
        num_req_slots=num_req_slots,
        swa_size=swa_pages * 256,
        c4_size=0,
        c128_size=0,
        c4_state_pool_size=0,
        c128_state_pool_size=0,
        page_size=256,
        swa_page_size=256,
        sliding_window=128,
        dtype=torch.bfloat16,
        c4_state_dtype=torch.float32,
        c128_state_dtype=torch.float32,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        indexer_head_dim=128,
        layer_num=len(compression_ratios),
        device="cpu",
        enable_memory_saver=False,
        compression_ratios=compression_ratios,
        kv_source_layers=kv_source_layers,
        start_layer=0,
        end_layer=len(compression_ratios),
        full_size=full_pages * 256,
    )
    testcase.assertIsInstance(pool, DSV41TPUTokenToKVPool)
    return pool


def _fill_pages(buf: torch.Tensor, pages: list[int], seed: int) -> None:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    shape = (len(pages), *buf.shape[1:])
    if buf.dtype.is_floating_point:
        buf[pages] = torch.randn(shape, dtype=buf.dtype, generator=gen)
    else:
        buf[pages] = torch.randint(
            -100000, 100000, shape, dtype=buf.dtype, generator=gen
        )


class TestDSV41RaidenDisaggregation(CustomTestCase):
    def _build_kv_args_and_aux(
        self, pool: DSV41TPUTokenToKVPool, *, port: int
    ) -> tuple[KVArgs, ctypes.Array]:
        ptrs, lens, item_lens = get_kv_transfer_buf_infos(pool)
        aux_buf = (ctypes.c_int32 * 8)()
        kv_args = KVArgs()
        kv_args.engine_rank = 0
        kv_args.kv_data_ptrs = ptrs
        kv_args.kv_data_lens = lens
        kv_args.kv_item_lens = item_lens
        kv_args.kv_cache_dtype_str = "bf16"
        setup_state_kv_args(kv_args, pool)
        kv_args.aux_data_ptrs = [ctypes.addressof(aux_buf)]
        kv_args.aux_data_lens = [ctypes.sizeof(aux_buf)]
        kv_args.aux_item_lens = [ctypes.sizeof(ctypes.c_int32)]
        kv_args.kv_head_num = 1
        kv_args.total_kv_head_num = 8
        kv_args.page_size = 256
        kv_args.sliding_window_size = 128
        kv_args.prefill_pp_size = 1
        kv_args.prefill_attn_cp_size = 1
        kv_args.prefill_start_layer = 0
        kv_args.pp_rank = 0
        kv_args.system_dp_rank = 0
        kv_args.preferred_port = port
        return kv_args, aux_buf

    def _close_manager(self, mgr: RaidenKVManager) -> None:
        executor = mgr.__dict__.get("_push_executor")
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        for sock in list(getattr(mgr, "_socket_cache", {}).values()):
            sock.close(linger=0)

    def _spawn_pd_pair(
        self,
        *,
        full_pages: int = 33,
        swa_pages: int = 7,
        context_length: int = 8192,
    ):
        registry = _InMemoryRaidenRegistry()
        patcher = patch.dict(sys.modules, _fake_tpu_sync_modules(registry))
        patcher.start()
        self.addCleanup(patcher.stop)

        bootstrap_port = _free_port_block(1)
        prefill_pool = _make_dsv41_pool(
            self,
            num_req_slots=6,
            full_pages=full_pages,
            swa_pages=swa_pages,
            bootstrap_port=bootstrap_port,
        )
        decode_pool = _make_dsv41_pool(
            self,
            num_req_slots=6,
            full_pages=full_pages,
            swa_pages=swa_pages,
            bootstrap_port=bootstrap_port,
        )

        _bootstrap = RaidenKVBootstrapServer(host="127.0.0.1", port=bootstrap_port)
        assert _bootstrap is not None
        bootstrap_addr = f"127.0.0.1:{bootstrap_port}"

        p_raiden_base = _free_port_block(PORTS_PER_RANK + 2)
        d_raiden_base = _free_port_block(PORTS_PER_RANK + 2)

        p_kv_args, p_aux = self._build_kv_args_and_aux(
            prefill_pool, port=p_raiden_base + PORTS_PER_RANK
        )
        d_kv_args, d_aux = self._build_kv_args_and_aux(
            decode_pool, port=d_raiden_base + PORTS_PER_RANK
        )

        p_server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="prefill",
            disaggregation_bootstrap_port=bootstrap_port,
            context_length=context_length,
            page_size=256,
        )
        d_server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_bootstrap_port=bootstrap_port,
            context_length=context_length,
            page_size=256,
        )

        with patch.dict("os.environ", {"SGLANG_RAIDEN_KV_PORT": str(p_raiden_base)}):
            prefill_mgr = RaidenKVManager(
                p_kv_args, DisaggregationMode.PREFILL, p_server_args
            )
            prefill_mgr.set_kv_caches(get_raiden_kv_caches(prefill_pool))
        self.addCleanup(self._close_manager, prefill_mgr)

        with patch.dict("os.environ", {"SGLANG_RAIDEN_KV_PORT": str(d_raiden_base)}):
            decode_mgr = RaidenKVManager(
                d_kv_args, DisaggregationMode.DECODE, d_server_args
            )
            decode_mgr.set_kv_caches(get_raiden_kv_caches(decode_pool))
        self.addCleanup(self._close_manager, decode_mgr)

        decode_mgr.try_ensure_parallel_info(bootstrap_addr)
        return (
            prefill_pool,
            decode_pool,
            prefill_mgr,
            decode_mgr,
            p_aux,
            d_aux,
            bootstrap_addr,
        )

    def _complete_transfer(
        self,
        *,
        sender: RaidenKVSender,
        receiver: RaidenKVReceiver,
        prefill_pool: DSV41TPUTokenToKVPool,
        decode_pool: DSV41TPUTokenToKVPool,
        prefill_req_idx: int,
        decode_req_idx: int,
        src_full_pages: list[int],
        dst_full_pages: list[int],
        src_swa_pages: list[int],
        dst_swa_pages: list[int],
        seq_len: int,
        chunk_sizes: list[int] | None = None,
        prefill_before_pull_req: bool = False,
    ) -> None:
        chunks = chunk_sizes or [len(src_full_pages)]
        self.assertEqual(sum(chunks), len(src_full_pages))
        sender.init(num_kv_indices=len(src_full_pages), aux_index=prefill_req_idx)
        receiver.init(prefill_dp_rank=0)

        def _drive_decode_metadata() -> None:
            decode_pool.clear_request_scoped_state(decode_req_idx)
            receiver.send_metadata(
                np.asarray(dst_full_pages, dtype=np.int32),
                aux_index=decode_req_idx,
                state_indices=[
                    np.asarray(dst_swa_pages, dtype=np.int32),
                    decode_pool.request_state_transfer_indices(decode_req_idx, seq_len),
                ],
            )

        if not prefill_before_pull_req:
            _drive_decode_metadata()

        cursor = 0
        for step_idx, chunk_len in enumerate(chunks):
            is_last = step_idx == len(chunks) - 1
            chunk_pages = np.asarray(
                src_full_pages[cursor : cursor + chunk_len], dtype=np.int32
            )
            cursor += chunk_len
            self.assertTrue(sender.should_send_kv_chunk(len(chunk_pages), is_last))
            state_indices = (
                [
                    np.asarray(src_swa_pages, dtype=np.int32),
                    prefill_pool.request_state_transfer_indices(
                        prefill_req_idx, seq_len
                    ),
                ]
                if is_last
                else None
            )
            sender.send(chunk_pages, state_indices=state_indices)
            if not is_last and not prefill_before_pull_req:
                receiver.poll()

        if prefill_before_pull_req:
            _drive_decode_metadata()

        deadline = time.monotonic() + 5.0
        send_done = recv_done = False
        while time.monotonic() < deadline and not (send_done and recv_done):
            if not send_done:
                s_status = sender.poll()
                self.assertNotEqual(s_status, KVPoll.Failed)
                send_done = s_status == KVPoll.Success
            if not recv_done:
                r_status = receiver.poll()
                self.assertNotEqual(r_status, KVPoll.Failed)
                recv_done = r_status == KVPoll.Success
            if not (send_done and recv_done):
                time.sleep(0.005)
        self.assertTrue(send_done and recv_done)
        sender.clear()
        receiver.clear()

    def test_get_raiden_kv_caches_uniform_blocks_and_root_ownership(self):
        pool = _make_dsv41_pool(self, num_req_slots=6, full_pages=15, swa_pages=7)
        groups = get_raiden_kv_caches(pool)

        # C1: 1 source * (k + scale + index) = 3; C2: 2 sources * (k + scale + index) = 6 -> 9
        self.assertEqual(len(groups.full), 9)
        self.assertEqual({int(b.shape[0]) for b in groups.full}, {16})
        # All 6 layers have SWA word pools
        self.assertEqual(len(groups.swa), 6)
        self.assertEqual({int(b.shape[0]) for b in groups.swa}, {8})
        # 2 C2 source layers own TPUPairStatePool (6+1 = 7 slots)
        self.assertEqual(len(groups.state), 2)
        self.assertEqual({int(b.shape[0]) for b in groups.state}, {7})
        self.assertEqual(groups.state_type, StateType.DSV4_REQUEST_STATE)
        self.assertEqual(groups.state_max_blocks, 1)
        self.assertIsNone(groups.read_state_meta)
        self.assertIsNone(groups.write_state_meta)

        for buf in [*groups.full, *groups.swa, *groups.state]:
            self.assertIsNone(buf._base)
            self.assertGreaterEqual(buf.ndim, 3)
            self.assertEqual(buf.shape[-2] % 8, 0)
            self.assertEqual(buf.shape[-1] % 128, 0)

        kv_args, _ = self._build_kv_args_and_aux(pool, port=39000)
        self.assertIn(StateType.SWA, kv_args.state_types)
        self.assertIn(StateType.DSV4_REQUEST_STATE, kv_args.state_types)

        orig_scale = pool.kv_pools[1].scale_pages[0]
        pool.kv_pools[1].scale_pages[0] = torch.zeros((16, 1, 256), dtype=torch.float32)
        try:
            with self.assertRaisesRegex(RuntimeError, "minor tile dims"):
                get_raiden_kv_caches(pool)
        finally:
            pool.kv_pools[1].scale_pages[0] = orig_scale

        orig = pool.swa_kv_pool.k_pages[0]
        pool.swa_kv_pool.k_pages[0] = orig[:]
        try:
            with self.assertRaisesRegex(RuntimeError, "root tensors"):
                get_raiden_kv_caches(pool)
        finally:
            pool.swa_kv_pool.k_pages[0] = orig

    def test_live_raiden_pd_roundtrip_and_pointer_stability(self):
        (
            prefill_pool,
            decode_pool,
            prefill_mgr,
            decode_mgr,
            p_aux,
            d_aux,
            bootstrap_addr,
        ) = self._spawn_pd_pair()

        self.assertEqual(prefill_mgr._state_engine_max_blocks, 1)
        self.assertEqual(decode_mgr._state_engine_max_blocks, 1)

        groups_p = get_raiden_kv_caches(prefill_pool)
        groups_d = get_raiden_kv_caches(decode_pool)
        initial_d_ptrs = [
            b.data_ptr() for b in (*groups_d.full, *groups_d.swa, *groups_d.state)
        ]

        src_full = [2, 5, 9]
        dst_full = [30, 14, 7]
        src_swa = [6, 3]
        dst_swa = [1, 5]
        prefill_req_idx, decode_req_idx, canary_req_idx = 4, 2, 3

        for idx, buf in enumerate(groups_p.full):
            _fill_pages(buf, src_full, seed=100 + idx)
        for idx, buf in enumerate(groups_p.swa):
            _fill_pages(buf, src_swa, seed=200 + idx)
        for idx, buf in enumerate(groups_p.state):
            _fill_pages(buf, [prefill_req_idx], seed=300 + idx)
        for idx, buf in enumerate(groups_d.state):
            _fill_pages(buf, [canary_req_idx], seed=900 + idx)
        canary_snapshots = [buf[canary_req_idx].clone() for buf in groups_d.state]

        p_aux[prefill_req_idx] = 77777
        d_aux[decode_req_idx] = -1

        decode_pool.clear_c4_req_states([decode_req_idx])
        self.assertEqual(
            [b.data_ptr() for b in (*groups_d.full, *groups_d.swa, *groups_d.state)],
            initial_d_ptrs,
        )

        sender = RaidenKVSender(prefill_mgr, bootstrap_addr, 5151)
        receiver = RaidenKVReceiver(decode_mgr, bootstrap_addr, 5151)
        self._complete_transfer(
            sender=sender,
            receiver=receiver,
            prefill_pool=prefill_pool,
            decode_pool=decode_pool,
            prefill_req_idx=prefill_req_idx,
            decode_req_idx=decode_req_idx,
            src_full_pages=src_full,
            dst_full_pages=dst_full,
            src_swa_pages=src_swa,
            dst_swa_pages=dst_swa,
            seq_len=517,
        )

        self.assertEqual(d_aux[decode_req_idx], 77777)
        for src_buf, dst_buf in zip(groups_p.full, groups_d.full, strict=True):
            torch.testing.assert_close(dst_buf[dst_full], src_buf[src_full])
        for src_buf, dst_buf in zip(groups_p.swa, groups_d.swa, strict=True):
            torch.testing.assert_close(dst_buf[dst_swa], src_buf[src_swa])
        for src_buf, dst_buf, canary in zip(
            groups_p.state, groups_d.state, canary_snapshots, strict=True
        ):
            torch.testing.assert_close(
                dst_buf[decode_req_idx], src_buf[prefill_req_idx]
            )
            torch.testing.assert_close(dst_buf[canary_req_idx], canary)

        self.assertEqual(
            [b.data_ptr() for b in (*groups_d.full, *groups_d.swa, *groups_d.state)],
            initial_d_ptrs,
        )

    def test_multi_chunk_streaming_and_late_pull_roundtrip(self):
        (
            prefill_pool,
            decode_pool,
            prefill_mgr,
            decode_mgr,
            p_aux,
            d_aux,
            bootstrap_addr,
        ) = self._spawn_pd_pair(full_pages=52, swa_pages=8, context_length=13312)
        groups_p = get_raiden_kv_caches(prefill_pool)
        groups_d = get_raiden_kv_caches(decode_pool)

        for idx, (chunk_sizes, late_pull) in enumerate(
            (([32, 12], False), ([20, 24], True))
        ):
            with self.subTest(chunk_sizes=chunk_sizes, late_pull=late_pull):
                num_pages = sum(chunk_sizes)
                src_full = [1 + i for i in range(num_pages)]
                dst_full = [50 - i for i in range(num_pages)]
                src_swa = [5, 2]
                dst_swa = [3, 6]
                prefill_req_idx, decode_req_idx = 4 - idx, 1 + idx

                for b_idx, buf in enumerate(groups_p.full):
                    _fill_pages(buf, src_full, seed=400 + 20 * idx + b_idx)
                for b_idx, buf in enumerate(groups_p.swa):
                    _fill_pages(buf, src_swa, seed=500 + 20 * idx + b_idx)
                for b_idx, buf in enumerate(groups_p.state):
                    _fill_pages(buf, [prefill_req_idx], seed=600 + 20 * idx + b_idx)

                p_aux[prefill_req_idx] = 88000 + idx
                d_aux[decode_req_idx] = -1
                room = 5300 + idx
                sender = RaidenKVSender(prefill_mgr, bootstrap_addr, room)
                receiver = RaidenKVReceiver(decode_mgr, bootstrap_addr, room)
                self._complete_transfer(
                    sender=sender,
                    receiver=receiver,
                    prefill_pool=prefill_pool,
                    decode_pool=decode_pool,
                    prefill_req_idx=prefill_req_idx,
                    decode_req_idx=decode_req_idx,
                    src_full_pages=src_full,
                    dst_full_pages=dst_full,
                    src_swa_pages=src_swa,
                    dst_swa_pages=dst_swa,
                    seq_len=num_pages * 256,
                    chunk_sizes=chunk_sizes,
                    prefill_before_pull_req=late_pull,
                )
                self.assertEqual(d_aux[decode_req_idx], 88000 + idx)
                for src_buf, dst_buf in zip(groups_p.full, groups_d.full, strict=True):
                    torch.testing.assert_close(dst_buf[dst_full], src_buf[src_full])
                for src_buf, dst_buf in zip(groups_p.swa, groups_d.swa, strict=True):
                    torch.testing.assert_close(dst_buf[dst_swa], src_buf[src_swa])
                for src_buf, dst_buf in zip(
                    groups_p.state, groups_d.state, strict=True
                ):
                    torch.testing.assert_close(
                        dst_buf[decode_req_idx], src_buf[prefill_req_idx]
                    )

    def test_pair_compress_odd_and_even_pd_handoff_matches_unified(self):
        from sglang_tpu.dsv41.attention_backend import pair_compress

        (
            prefill_pool,
            decode_pool,
            prefill_mgr,
            decode_mgr,
            _p_aux,
            _d_aux,
            bootstrap_addr,
        ) = self._spawn_pd_pair()

        for idx, prompt_len in enumerate((517, 518)):
            with self.subTest(prompt_len=prompt_len):
                decode_steps = 4
                total_len = prompt_len + decode_steps
                gen = torch.Generator(device="cpu").manual_seed(2026 + idx)
                all_kv = torch.randn(
                    (total_len, 512), dtype=torch.float32, generator=gen
                )
                all_score = torch.randn(
                    (total_len, 512), dtype=torch.float32, generator=gen
                )

                # 0. Exact reference via DeepseekV41Compressor.pool_pairs across all 512 features
                from sglang.srt.layers.attention.dsv4.dsv41_sparse import (
                    DeepseekV41Compressor,
                )

                num_pairs = total_len // 2
                ref_emitted = DeepseekV41Compressor.pool_pairs(
                    all_kv[: 2 * num_pairs].view(num_pairs, 2, 512),
                    all_score[: 2 * num_pairs].view(num_pairs, 2, 512),
                )

                # 1. Unified single-instance reference
                unified_state = TPUPairStatePool(
                    num_req_slots=6,
                    ring_size=2,
                    head_dim=512,
                    device="cpu",
                    enable_memory_saver=False,
                )
                u_req = 2
                u_emitted = []
                u_prefill = pair_compress(
                    all_kv[:prompt_len],
                    all_score[:prompt_len],
                    torch.full((prompt_len,), u_req, dtype=torch.int32),
                    torch.arange(prompt_len, dtype=torch.int32),
                    torch.ones(prompt_len, dtype=torch.bool),
                    unified_state.state_buffer,
                    unified_state.ring_size,
                )
                for pos in range(prompt_len):
                    if pos % 2 == 1:
                        u_emitted.append(u_prefill[pos])
                for step in range(decode_steps):
                    pos = prompt_len + step
                    u_dec = pair_compress(
                        all_kv[pos : pos + 1],
                        all_score[pos : pos + 1],
                        torch.tensor([u_req], dtype=torch.int32),
                        torch.tensor([pos], dtype=torch.int32),
                        torch.tensor([True]),
                        unified_state.state_buffer,
                        unified_state.ring_size,
                    )
                    if pos % 2 == 1:
                        u_emitted.append(u_dec[0])
                torch.testing.assert_close(
                    torch.stack(u_emitted), ref_emitted, atol=1e-6, rtol=1e-6
                )

                # 2. Disaggregated multi-chunk Prefill (req 3) -> Raiden -> Decode (req 1)
                p_req, d_req = 3, 1
                p_state = prefill_pool.get_attention_compress_states(1)
                d_state = decode_pool.get_attention_compress_states(1)
                assert isinstance(p_state, TPUPairStatePool)
                assert isinstance(d_state, TPUPairStatePool)

                d_emitted = []
                nan_kv = torch.full((3, 512), float("nan"), dtype=torch.float32)
                nan_score = torch.full((3, 512), float("nan"), dtype=torch.float32)
                for c0, c1 in ((0, 256), (256, prompt_len)):
                    n_tok = c1 - c0
                    # Append bucket-padding tokens filled with NaN (valid=False)
                    # to guard against sink-row NaN pollution across chunks.
                    p_out = pair_compress(
                        torch.cat([all_kv[c0:c1], nan_kv], dim=0),
                        torch.cat([all_score[c0:c1], nan_score], dim=0),
                        torch.cat(
                            [
                                torch.full((n_tok,), p_req, dtype=torch.int32),
                                torch.zeros(3, dtype=torch.int32),
                            ]
                        ),
                        torch.cat(
                            [
                                torch.arange(c0, c1, dtype=torch.int32),
                                torch.zeros(3, dtype=torch.int32),
                            ]
                        ),
                        torch.cat(
                            [
                                torch.ones(n_tok, dtype=torch.bool),
                                torch.zeros(3, dtype=torch.bool),
                            ]
                        ),
                        p_state.state_buffer,
                        p_state.ring_size,
                    )
                    self.assertFalse(torch.isnan(p_out[:n_tok]).any())
                    self.assertTrue(
                        torch.equal(
                            p_state.state_buffer[-1, -1, :512],
                            torch.zeros(512, dtype=torch.float32),
                        )
                    )
                    self.assertTrue(
                        torch.isneginf(p_state.state_buffer[-1, -1, 512:1024]).all()
                    )
                    for offset, pos in enumerate(range(c0, c1)):
                        if pos % 2 == 1:
                            d_emitted.append(p_out[offset])

                room = 6200 + idx
                sender = RaidenKVSender(prefill_mgr, bootstrap_addr, room)
                receiver = RaidenKVReceiver(decode_mgr, bootstrap_addr, room)
                self._complete_transfer(
                    sender=sender,
                    receiver=receiver,
                    prefill_pool=prefill_pool,
                    decode_pool=decode_pool,
                    prefill_req_idx=p_req,
                    decode_req_idx=d_req,
                    src_full_pages=[1, 2, 3],
                    dst_full_pages=[12, 11, 10],
                    src_swa_pages=[2],
                    dst_swa_pages=[4],
                    seq_len=prompt_len,
                )

                for step in range(decode_steps):
                    pos = prompt_len + step
                    d_out = pair_compress(
                        all_kv[pos : pos + 1],
                        all_score[pos : pos + 1],
                        torch.tensor([d_req], dtype=torch.int32),
                        torch.tensor([pos], dtype=torch.int32),
                        torch.tensor([True]),
                        d_state.state_buffer,
                        d_state.ring_size,
                    )
                    if pos % 2 == 1:
                        d_emitted.append(d_out[0])

                self.assertEqual(len(d_emitted), len(u_emitted))
                torch.testing.assert_close(
                    torch.stack(d_emitted),
                    torch.stack(u_emitted),
                    atol=1e-6,
                    rtol=1e-6,
                )

    def test_host_engram_first_and_subsequent_disagg_decode_steps(self):
        from sglang_tpu.engram import PendingTokens, capture_inputs

        ngram = 5
        prompt = list(range(100, 100 + 517))
        first_token = 9001
        second_token = 9002

        # Unified reference across positions 0..518
        full_req = SimpleNamespace(
            origin_input_ids=prompt,
            output_ids=[first_token, second_token],
        )
        ref = capture_inputs([full_req], [517], [2], ngram=ngram)

        # Disaggregated Decode step 1: prebuilt request carries prompt + [first_token]
        disagg_req = SimpleNamespace(
            origin_input_ids=list(prompt),
            output_ids=[first_token],
        )
        step1 = capture_inputs([disagg_req], [517], [1], ngram=ngram, pending=None)
        np.testing.assert_array_equal(step1.tokens[0], ref.tokens[0])
        np.testing.assert_array_equal(step1.positions, [517])

        # Disaggregated Decode step 2 under overlap: second_token comes from PendingTokens
        pending = PendingTokens(
            [disagg_req],
            [518],
            SimpleNamespace(copy_done=None, next_token_ids=[second_token]),
        )
        step2 = capture_inputs([disagg_req], [518], [1], ngram=ngram, pending=pending)
        np.testing.assert_array_equal(step2.tokens[0], ref.tokens[1])
        np.testing.assert_array_equal(step2.positions, [518])

    def test_tp_sharded_engram_safetensors_slice_reader(self):
        import json
        import struct
        import tempfile
        from pathlib import Path

        import sglang_tpu.dsv41.loader as dsv41_loader
        from sglang.srt.model_loader.utils import should_async_load
        from sglang.srt.model_loader.weight_utils import RUNAI_STREAMER_TENSOR_ATTR
        from sglang_tpu.dsv41.loader import read_tp_sharded_safetensors_rows
        from sglang_tpu.tpu_platform import _sync_weight_stream

        # 50 % 8 == 2 (mirrors 384016682 % 8 == 2 in DeepSeek-V4.1-Flash)
        num_rows = 50
        gen = torch.Generator(device="cpu").manual_seed(42)
        w_u8 = torch.randint(0, 256, (num_rows, 256), dtype=torch.uint8, generator=gen)
        s_u8 = torch.randint(0, 256, (num_rows, 8), dtype=torch.uint8, generator=gen)
        w_ref = w_u8.view(torch.float8_e4m3fn)
        s_ref = s_u8.view(torch.float8_e8m0fnu)

        header = json.dumps(
            {
                "layers.14.engram.embed.weight": {
                    "dtype": "F8_E4M3",
                    "shape": [num_rows, 256],
                    "data_offsets": [0, w_u8.nbytes],
                },
                "layers.14.engram.embed.scale": {
                    "dtype": "F8_E8M0",
                    "shape": [num_rows, 8],
                    "data_offsets": [w_u8.nbytes, w_u8.nbytes + s_u8.nbytes],
                },
            }
        ).encode("utf-8")

        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "model-00048-of-00048.safetensors"
            with open(shard_path, "wb") as f:
                f.write(struct.pack("<Q", len(header)))
                f.write(header)
                f.write(w_u8.numpy().tobytes())
                f.write(s_u8.numpy().tobytes())
            payload = shard_path.read_bytes()

            class _FakeBlob:
                def __init__(self, data: bytes) -> None:
                    self._data = data
                    self.calls = 0

                def download_as_bytes(
                    self, *, start: int, end: int, raw_download: bool = False
                ) -> bytes:
                    assert raw_download is True
                    assert 0 <= start <= end < len(self._data)
                    self.calls += 1
                    if self.calls == 3:
                        raise ConnectionError("simulated transient GCS reset")
                    return self._data[start : end + 1]

            fake_blob = _FakeBlob(payload)
            for source_kind, target in (
                ("local", shard_path),
                ("gcs", "gs://test-bucket/ckpt/model-00048-of-00048.safetensors"),
            ):
                with (
                    self.subTest(source_kind=source_kind),
                    patch.object(dsv41_loader, "_READ_CHUNK_BYTES", 256),
                    patch.object(dsv41_loader, "_SEQ_STEP_BYTES", 128),
                    patch.object(dsv41_loader, "sleep", lambda _: None),
                    patch.object(
                        dsv41_loader, "_open_gcs_blob", return_value=fake_blob
                    ),
                ):
                    assembled_w = torch.empty_like(w_ref)
                    assembled_s = torch.empty_like(s_ref)
                    tp_size = 8
                    for tp_rank in range(tp_size):
                        r0 = num_rows * tp_rank // tp_size
                        r1 = num_rows * (tp_rank + 1) // tp_size
                        raw_items = list(
                            read_tp_sharded_safetensors_rows(
                                target,
                                [
                                    "layers.14.engram.embed.weight",
                                    "layers.14.engram.embed.scale",
                                ],
                                tp_size=tp_size,
                                tp_rank=tp_rank,
                                streamer_attr=RUNAI_STREAMER_TENSOR_ATTR,
                            )
                        )
                        self.assertEqual(len(raw_items), 2)
                        synced = list(
                            _sync_weight_stream((n, p) for n, p, _ in raw_items)
                        )
                        self.assertEqual(len(synced), 2)
                        for (name, proxy, nbytes), (_, synced_proxy) in zip(
                            raw_items, synced, strict=True
                        ):
                            self.assertIs(proxy, synced_proxy)
                            self.assertEqual(
                                proxy.numel() * proxy.element_size(), nbytes
                            )
                            self.assertFalse(should_async_load(proxy))
                            if name.endswith(".weight"):
                                self.assertEqual(nbytes, (r1 - r0) * 256)
                                assembled_w[r0:r1].copy_(proxy[r0:r1])
                                with self.assertRaises(IndexError):
                                    _ = proxy[0:num_rows]
                            else:
                                self.assertEqual(nbytes, (r1 - r0) * 8)
                                assembled_s[r0:r1].copy_(proxy[r0:r1])

                    self.assertTrue(
                        torch.equal(
                            assembled_w.view(torch.uint8), w_ref.view(torch.uint8)
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            assembled_s.view(torch.uint8), s_ref.view(torch.uint8)
                        )
                    )
            self.assertGreater(fake_blob.calls, 8)

    def test_aot_autograd_compile_preserves_root_buffers_without_base_error(self):
        from sglang_tpu.dsv41.attention_backend import pair_compress
        from sglang_tpu.dsv41.custom_ops import publish_source

        pool = _make_dsv41_pool(self, num_req_slots=6, full_pages=8, swa_pages=4)
        groups = get_raiden_kv_caches(pool)
        initial_ptrs = [
            b.data_ptr() for b in (*groups.full, *groups.swa, *groups.state)
        ]

        def _compressed(words, scales, records, latent, after):
            assert words._base is None and words.ndim == 3
            assert scales._base is None and scales.ndim == 3
            return words + 11, scales + 2.5

        def _index(words, records, key, after, page_rec, stride=0):
            assert words._base is None and words.ndim == 3
            return words + 7

        def _swa(swa_words, records, k, page_size):
            assert swa_words._base is None and swa_words.ndim == 3
            return swa_words + 13

        def _pair_state(state_buf, kv, score, req, pos, valid, ring_size):
            assert state_buf._base is None and state_buf.ndim == 3
            return state_buf + 5.0, kv + 1.0

        ops = SimpleNamespace(
            compressed=_compressed,
            index=_index,
            swa=_swa,
            pair_state=_pair_state,
        )

        def _step(kv, score, req, pos, valid):
            state = pool.get_attention_compress_states(1)
            pooled = pair_compress(
                kv,
                score,
                req,
                pos,
                valid,
                state.state_buffer,
                state.ring_size,
                ops,
            )
            publish_source(ops, pool, 1, pos, pooled, pooled[:, :128], pos)
            swa_root = pool.swa_root(1)
            new_swa = ops.swa(swa_root, pos, pooled, pool.page_size)
            swa_root.copy_(new_swa)
            # Verify in-graph post-write reads resolve through the updated roots
            return (
                pooled,
                pool.low_ratio_cache(1)[0, 0],
                pool.low_ratio_scales(1)[0],
                pool.low_ratio_cache(1, index=True)[0, 0],
                pool.get_swa_buffer(1)[0, 0],
                state.state_buffer[0, 0, 0],
            )

        compiled = torch.compile(_step, backend="aot_eager", fullgraph=True)
        kv = torch.ones((4, 512), dtype=torch.float32)
        score = torch.zeros((4, 512), dtype=torch.float32)
        req = torch.full((4,), 2, dtype=torch.int32)
        pos = torch.arange(4, dtype=torch.int32)
        valid = torch.ones(4, dtype=torch.bool)

        _, w0, s0, idx0, swa0, st0 = compiled(kv, score, req, pos, valid)
        self.assertEqual(int(w0.item()), 11)
        self.assertAlmostEqual(float(s0.item()), 2.5)
        self.assertEqual(int(idx0.item()), 7)
        self.assertEqual(int(swa0.item()), 13)
        self.assertAlmostEqual(float(st0.item()), 5.0)
        self.assertEqual(
            [b.data_ptr() for b in (*groups.full, *groups.swa, *groups.state)],
            initial_ptrs,
        )

    def test_multi_tile_sparse_core_selection_has_no_constant_hbm_operands(self):
        try:
            import jax
            import jax.numpy as jnp
        except ImportError:
            self.skipTest("jax is not installed")

        import math
        import re
        from unittest.mock import patch

        from sglang_tpu.kernels.deepseek_v41.index_selection.fp8.cache import (
            PackedFP8Layout,
        )
        from sglang_tpu.kernels.deepseek_v41.index_selection.selection import (
            prepare_fp8_candidates,
            select_fp8_indexer_jax,
            select_fp8_prepared,
        )
        from sglang_tpu.kernels.deepseek_v41.index_selection.streaming_topk import (
            _block_visit_order,
        )
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        for num_blocks in (1, 2, 3, 4, 12, 127, 128, 512, 65536):
            tail, stride, offset = _block_visit_order(num_blocks)
            if num_blocks <= 2:
                visited = [0, 1][:num_blocks]
            else:
                self.assertEqual(math.gcd(stride, tail), 1)
                visited = [0] + [
                    1 + (step * stride + offset) % tail for step in range(tail)
                ]
            self.assertEqual(visited[0], 0)
            self.assertEqual(sorted(visited), list(range(num_blocks)))

        def _assert_no_sc_constant_operands(mlir_text: str) -> int:
            sc_calls = 0
            consts: set[str] = set()
            for line in mlir_text.splitlines():
                stripped = line.strip()
                if stripped.startswith(("func.func ", "} // end")):
                    consts.clear()
                m_const = re.match(r"(%\S+)\s*=\s*stablehlo\.constant\b", stripped)
                if m_const:
                    consts.add(m_const.group(1))
                if (
                    "stablehlo.custom_call @tpu_custom_call(" in stripped
                    and "DEVICE_TYPE_SPARSECORE" in stripped
                ):
                    sc_calls += 1
                    m_args = re.search(r"@tpu_custom_call\(([^)]*)\)", stripped)
                    self.assertIsNotNone(m_args)
                    args = [a.strip() for a in m_args.group(1).split(",") if a.strip()]
                    const_args = [a for a in args if a in consts]
                    self.assertEqual(
                        const_args,
                        [],
                        f"SparseCore custom_call received constant operand(s) {const_args}: {stripped[:200]}",
                    )
            return sc_calls

        layout = PackedFP8Layout(512, page_stride=72, page_size=64)
        layout_args = (
            layout.pages,
            layout.page_stride,
            layout.page_offset,
            layout.page_size,
        )
        with (
            patch(
                "sglang_tpu.pallas_export.get_pallas_tpu_device_kind",
                return_value="TPU7x",
            ),
            patch("jax.default_backend", return_value="tpu"),
            tpu_info_for_cpu_export(),
            jax._src.config.export_ignore_forward_compatibility(True),
        ):
            exp_dense = jax.export.export(
                jax.jit(
                    lambda q, w, b, p, r, l, c: select_fp8_indexer_jax(
                        q,
                        (b,),
                        p,
                        w,
                        l,
                        c,
                        r,
                        jnp.max(l, initial=0).reshape(1),
                        layout_args,
                        512,
                        8,
                        True,
                        False,
                        True,
                        32768,
                    )
                ),
                platforms=("tpu",),
            )(
                jax.ShapeDtypeStruct((102, 32, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((102, 32), jnp.float32),
                jax.ShapeDtypeStruct((512 * 72, 128), jnp.int32),
                jax.ShapeDtypeStruct((8, 512), jnp.int32),
                jax.ShapeDtypeStruct((102,), jnp.int32),
                jax.ShapeDtypeStruct((102,), jnp.int32),
                jax.ShapeDtypeStruct((102, 128), jnp.int32),
            )
            self.assertGreaterEqual(
                _assert_no_sc_constant_operands(exp_dense.mlir_module()), 2
            )

            def _prep_and_select(q, w, b, p, r, l, c):
                prepared = prepare_fp8_candidates(
                    (b,),
                    p,
                    c,
                    r,
                    layout_args,
                    8,
                    32768,
                )
                return select_fp8_prepared(
                    q,
                    w,
                    l,
                    prepared,
                    512,
                    8,
                    True,
                    32768,
                )

            exp_prep = jax.export.export(
                jax.jit(_prep_and_select),
                platforms=("tpu",),
            )(
                jax.ShapeDtypeStruct((128, 32, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((128, 32), jnp.float32),
                jax.ShapeDtypeStruct((512 * 72, 128), jnp.int32),
                jax.ShapeDtypeStruct((8, 512), jnp.int32),
                jax.ShapeDtypeStruct((128,), jnp.int32),
                jax.ShapeDtypeStruct((128,), jnp.int32),
                jax.ShapeDtypeStruct((128, 128), jnp.int32),
            )
            self.assertGreaterEqual(
                _assert_no_sc_constant_operands(exp_prep.mlir_module()), 2
            )

            from sglang_tpu.kernels.deepseek_v41.csa2.cache import (
                gather_csa2_kv_sc,
                publish_csa2_kv,
            )

            for page_records in (256, 128):
                num_pages = 8
                exp_pub = jax.export.export(
                    jax.jit(publish_csa2_kv, donate_argnums=(0, 1)),
                    platforms=("tpu",),
                )(
                    jax.ShapeDtypeStruct((num_pages * page_records, 128), jnp.int32),
                    jax.ShapeDtypeStruct((num_pages * 1024,), jnp.float32),
                    jax.ShapeDtypeStruct((64,), jnp.int32),
                    jax.ShapeDtypeStruct((64, 512), jnp.bfloat16),
                )
                self.assertEqual(
                    _assert_no_sc_constant_operands(exp_pub.mlir_module()), 1
                )
                exp_gat = jax.export.export(
                    jax.jit(gather_csa2_kv_sc),
                    platforms=("tpu",),
                )(
                    jax.ShapeDtypeStruct((num_pages * page_records, 128), jnp.int32),
                    jax.ShapeDtypeStruct((num_pages * 1024,), jnp.float32),
                    jax.ShapeDtypeStruct((64, 512), jnp.int32),
                )
                self.assertEqual(
                    _assert_no_sc_constant_operands(exp_gat.mlir_module()), 1
                )


if __name__ == "__main__":
    unittest.main()
