# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU end-to-end protocol + in-memory DMA round-trip tests for DeepSeek-V4 P/D on Raiden.

Guards the full contract between ``DSV4TPUTokenToKVPool`` and the 3-engine
Raiden transport (``full``, ``swa``, ``state``):

1. ``get_raiden_kv_caches`` groups all 7 root buffer families into uniform-block
   ``full``, ``swa``, and 32-page ``state`` groups, rejecting view tensors.
2. Live ZMQ + HTTP bootstrap ``RaidenKVSender`` -> ``RaidenKVReceiver`` transfers
   across single-chunk, multi-chunk (``> CHUNK_ANNOUNCE_TOKENS``), and
   prefill-before-``PULL_REQ`` orderings copy every tensor slice byte-for-byte
   between distinct ``req_pool_idx`` slots without invalidating registered buffer
   pointers (even after ``clear_request_scoped_state``), synchronize
   ``(shift_c4, shift_c128)`` via ``Blocks.state_meta``, and clean up room state.
3. Host staging admission sizes ``state_pool_pages = num_slots * 32`` so tight
   slot budgets (e.g. ``num_slots = 3 < 32`` on 92-layer Pro) admit 32-page
   state reads and throttle concurrency instead of raising ``ValueError``.
"""

from __future__ import annotations

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

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
from sglang_tpu.dsv4.dsv4_memory_pool import (
    _TPU_STATE_SLOT_STRIDE_PAGES,
    DSV4TPUTokenToKVPool,
    TPUCompressStatePool,
)


def _free_port_block(count: int = 8) -> int:
    """Reserve a contiguous block of free local TCP ports for test isolation."""
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
    """Shared registry enabling real slice-by-slice DMA copies between engines."""

    def __init__(self) -> None:
        self.engines: dict[int, _InMemoryKVCacheManager] = {}


class _InMemoryKVCacheManager:
    """Drop-in CPU ``tpu_sync.api.torch.kv_cache_manager.KVCacheManager``.

    Validates root ownership, stable ``data_ptr()`` across registration and DMA,
    ``max_blocks`` bounds, and copies tensor pages byte-for-byte on ``start_read``.
    """

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
    ) -> None:
        del node_id, timeout_s, parallelism
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


def _make_dsv4_pool(
    testcase: unittest.TestCase,
    *,
    num_req_slots: int = 5,
    full_pages: int = 48,
    swa_pages: int = 8,
    bootstrap_port: int = 8998,
) -> DSV4TPUTokenToKVPool:
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

    # Mix SWA (0), C128 (128), and C4 (4) layers as in DeepSeek-V4 Flash / Pro.
    compression_ratios = [0, 0, 128, 4, 128, 4]
    pool = DSV4TPUTokenToKVPool(
        max_num_reqs=num_req_slots - 1,
        num_req_slots=num_req_slots,
        swa_size=swa_pages * 256,
        c4_size=full_pages * 64,
        c128_size=full_pages * 2,
        c4_state_pool_size=32,
        c128_state_pool_size=256,
        page_size=256,
        swa_page_size=256,
        sliding_window=512,
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
        start_layer=0,
        end_layer=len(compression_ratios),
        full_size=full_pages * 256,
    )
    return pool


def _fill_pattern(t: torch.Tensor, pages: list[int], seed: int) -> None:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    t[pages] = torch.randint(
        0, 256, (len(pages), *t.shape[1:]), dtype=torch.uint8, generator=gen
    )


class TestDSV4RaidenDisaggregation(CustomTestCase):
    def _build_kv_args_and_aux(
        self, pool: DSV4TPUTokenToKVPool, *, port: int
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
        kv_args.sliding_window_size = 512
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
        full_pages: int = 48,
        swa_pages: int = 8,
        context_length: int = 12288,
    ):
        registry = _InMemoryRaidenRegistry()
        patcher = patch.dict(sys.modules, _fake_tpu_sync_modules(registry))
        patcher.start()
        self.addCleanup(patcher.stop)

        bootstrap_port = _free_port_block(1)
        prefill_pool = _make_dsv4_pool(
            self,
            num_req_slots=5,
            full_pages=full_pages,
            swa_pages=swa_pages,
            bootstrap_port=bootstrap_port,
        )
        decode_pool = _make_dsv4_pool(
            self,
            num_req_slots=5,
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

    def test_get_raiden_kv_caches_groups_and_rejects_views(self):
        pool = _make_dsv4_pool(self, num_req_slots=5, full_pages=16, swa_pages=6)
        groups = get_raiden_kv_caches(pool)

        # 2 C4 layers * (nope + rope + indexer) + 2 C128 layers = 8 full buffers
        self.assertEqual(len(groups.full), 8)
        # All 6 layers carry an SWA sub-buffer in DSV4
        self.assertEqual(len(groups.swa), 6)
        # 4 non-SWA compressor state pools + 2 C4 indexer state pools = 6 state buffers
        self.assertEqual(len(groups.state), 6)
        self.assertEqual(groups.state_type, StateType.DSV4_REQUEST_STATE)
        self.assertEqual(groups.state_max_blocks, _TPU_STATE_SLOT_STRIDE_PAGES)
        self.assertIsNone(groups.committed_state_row)
        self.assertIsNotNone(groups.read_state_meta)
        self.assertIsNotNone(groups.write_state_meta)

        # Injecting a view must fail closed so PJRT buffer registration cannot break.
        assert isinstance(pool.swa_kv_pool.k_buffer[0], torch.Tensor)
        orig = pool.swa_kv_pool.k_buffer[0]
        pool.swa_kv_pool.k_buffer[0] = orig[:]
        try:
            with self.assertRaisesRegex(RuntimeError, "root tensors"):
                get_raiden_kv_caches(pool)
        finally:
            pool.swa_kv_pool.k_buffer[0] = orig

    def _populate_and_verify_transfer(
        self,
        *,
        num_pages: int,
        chunk_sizes: list[int],
        prefill_before_pull_req: bool,
    ) -> None:
        (
            prefill_pool,
            decode_pool,
            prefill_mgr,
            decode_mgr,
            p_aux,
            d_aux,
            bootstrap_addr,
        ) = self._spawn_pd_pair()

        self.assertEqual(
            prefill_mgr._state_engine_max_blocks, _TPU_STATE_SLOT_STRIDE_PAGES
        )
        self.assertEqual(
            decode_mgr._state_engine_max_blocks, _TPU_STATE_SLOT_STRIDE_PAGES
        )

        room = 4242 + sum(chunk_sizes) + int(prefill_before_pull_req)
        prefill_req_idx = 3
        decode_req_idx = 1
        seq_len = num_pages * 256

        # Non-contiguous physical pages on Prefill vs Decode to prove translation.
        src_full_pages = [2 + i for i in range(num_pages)]
        dst_full_pages = [45 - i for i in range(num_pages)]
        src_swa_pages = [6, 4]
        dst_swa_pages = [2, 5]
        canary_decode_req_idx = 2

        groups_p = get_raiden_kv_caches(prefill_pool)
        groups_d = get_raiden_kv_caches(decode_pool)
        for idx, buf in enumerate(groups_p.full):
            _fill_pattern(buf, src_full_pages, seed=100 + idx)
        for idx, buf in enumerate(groups_p.swa):
            _fill_pattern(buf, src_swa_pages, seed=200 + idx)

        p_state_rows = [
            int(x)
            for x in prefill_pool.request_state_transfer_indices(
                prefill_req_idx, seq_len
            )
        ]
        d_state_rows = [
            int(x)
            for x in decode_pool.request_state_transfer_indices(decode_req_idx, seq_len)
        ]
        canary_rows = [
            int(x)
            for x in decode_pool.request_state_transfer_indices(
                canary_decode_req_idx, seq_len
            )
        ]
        for idx, buf in enumerate(groups_p.state):
            _fill_pattern(buf, p_state_rows, seed=300 + idx)
        for idx, buf in enumerate(groups_d.state):
            _fill_pattern(buf, canary_rows, seed=900 + idx)
        canary_snapshots = [buf[canary_rows].clone() for buf in groups_d.state]

        # Prefill finishes with non-zero compressor ring shifts; Decode starts with
        # dirty stale shifts that decode.py clears before send_metadata.
        prefill_pool._req_state_shifts[4][prefill_req_idx] = 5
        prefill_pool._req_state_shifts[128][prefill_req_idx] = 19
        decode_pool._req_state_shifts[4][decode_req_idx] = 7
        decode_pool._req_state_shifts[128][decode_req_idx] = 29
        decode_pool._req_state_shifts[4][canary_decode_req_idx] = 3
        decode_pool._req_state_shifts[128][canary_decode_req_idx] = 21

        p_aux[prefill_req_idx] = 98765
        d_aux[decode_req_idx] = -1

        sender = RaidenKVSender(prefill_mgr, bootstrap_addr, room)
        receiver = RaidenKVReceiver(decode_mgr, bootstrap_addr, room)
        sender.init(num_kv_indices=num_pages, aux_index=prefill_req_idx)
        receiver.init(prefill_dp_rank=0)

        def _drive_decode_metadata() -> None:
            decode_pool.clear_request_scoped_state(decode_req_idx)
            self.assertEqual(int(decode_pool._req_state_shifts[4][decode_req_idx]), 0)
            self.assertEqual(int(decode_pool._req_state_shifts[128][decode_req_idx]), 0)
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
        for step_idx, chunk_len in enumerate(chunk_sizes):
            is_last = step_idx == len(chunk_sizes) - 1
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
                # Let Decode pull intermediate chunks mid-prefill as in production.
                receiver.poll()

        if prefill_before_pull_req:
            _drive_decode_metadata()

        deadline = time.monotonic() + 5.0
        p_status = KVPoll.Transferring
        d_status = KVPoll.Transferring
        while time.monotonic() < deadline:
            p_status = sender.poll()
            d_status = receiver.poll()
            if p_status == KVPoll.Success and d_status == KVPoll.Success:
                break
            time.sleep(0.005)

        self.assertEqual(p_status, KVPoll.Success)
        self.assertEqual(d_status, KVPoll.Success)

        # Every full, SWA, and 32-page request-state tensor slice matches bit-for-bit.
        for src_buf, dst_buf in zip(groups_p.full, groups_d.full, strict=True):
            self.assertTrue(
                torch.equal(src_buf[src_full_pages], dst_buf[dst_full_pages])
            )
        for src_buf, dst_buf in zip(groups_p.swa, groups_d.swa, strict=True):
            self.assertTrue(torch.equal(src_buf[src_swa_pages], dst_buf[dst_swa_pages]))
        for src_buf, dst_buf in zip(groups_p.state, groups_d.state, strict=True):
            self.assertTrue(torch.equal(src_buf[p_state_rows], dst_buf[d_state_rows]))

        # Neighboring request slots on Decode were never overwritten.
        for expected, dst_buf in zip(canary_snapshots, groups_d.state, strict=True):
            self.assertTrue(torch.equal(expected, dst_buf[canary_rows]))
        self.assertEqual(
            int(decode_pool._req_state_shifts[4][canary_decode_req_idx]), 3
        )
        self.assertEqual(
            int(decode_pool._req_state_shifts[128][canary_decode_req_idx]), 21
        )

        # Aux first-token metadata and post-prefill compressor shifts landed on Decode.
        self.assertEqual(d_aux[decode_req_idx], 98765)
        self.assertEqual(int(decode_pool._req_state_shifts[4][decode_req_idx]), 5)
        self.assertEqual(int(decode_pool._req_state_shifts[128][decode_req_idx]), 19)

        # First decode step's build_state_block_table agrees on ring shift & modulo wrap.
        for ratio, expected_shift in ((4, 5), (128, 19)):
            table = decode_pool.build_state_block_table(
                ratio=ratio,
                req_pool_indices_cpu=torch.tensor([decode_req_idx], dtype=torch.long),
                seq_lens_cpu=torch.tensor([seq_len + 1], dtype=torch.int32),
                extend_seq_lens_cpu=None,
                is_decode=True,
                bs=1,
                non_empty_bs=1,
                device="cpu",
            )
            assert table is not None
            self.assertEqual(
                int(table[0, 0].item()),
                1 + decode_req_idx * _TPU_STATE_SLOT_STRIDE_PAGES,
            )
            self.assertEqual(int(table[0, 2].item()), expected_shift)
            self.assertEqual(int(table[0, 4].item()), expected_shift)
            self.assertIsInstance(
                decode_pool._state_pool_proto[ratio], TPUCompressStatePool
            )

        # clear() sweeps all room bookkeeping including send_state_meta.
        sender.clear()
        receiver.clear()
        self.assertNotIn(room, prefill_mgr._prefill.send_state_meta)
        self.assertNotIn(room, prefill_mgr._prefill.send_state_rows)
        self.assertNotIn(room, decode_mgr._decode.local_state_rows)
        self.assertNotIn(room, decode_mgr._decode.remote_state_rows)

    def test_single_chunk_pd_roundtrip(self):
        self._populate_and_verify_transfer(
            num_pages=3, chunk_sizes=[3], prefill_before_pull_req=False
        )

    def test_multi_chunk_streaming_pd_roundtrip(self):
        # 40 pages = 10,240 tokens > CHUNK_ANNOUNCE_TOKENS (8,192 = 32 pages):
        # exercises live mid-prefill CHUNK reads (state_indices=None) + closing read.
        self._populate_and_verify_transfer(
            num_pages=40, chunk_sizes=[18, 16, 6], prefill_before_pull_req=False
        )

    def test_prefill_finishes_before_pull_req_roundtrip(self):
        # Exercises finalize_send snapshotting send_state_meta before PULL_REQ arrives.
        self._populate_and_verify_transfer(
            num_pages=4, chunk_sizes=[4], prefill_before_pull_req=True
        )

    def test_tight_state_slot_budget_admits_32_page_reads_without_overflow(self):
        # Simulates 92-layer DeepSeek-V4 Pro where per-engine host budget fits
        # num_slots = 3 (< 32 pages per read). state_pool_pages must be 3 * 32 = 96.
        registry = _InMemoryRaidenRegistry()
        pool = _make_dsv4_pool(self, num_req_slots=5, full_pages=16, swa_pages=6)
        port_base = _free_port_block(PORTS_PER_RANK + 2)
        kv_args, _ = self._build_kv_args_and_aux(pool, port=port_base + PORTS_PER_RANK)
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            context_length=4096,
            page_size=256,
        )
        with (
            patch.dict(sys.modules, _fake_tpu_sync_modules(registry)),
            patch.dict("os.environ", {"SGLANG_RAIDEN_KV_PORT": str(port_base)}),
            patch.object(RaidenKVManager, "_num_slots", return_value=3),
        ):
            mgr = RaidenKVManager(kv_args, DisaggregationMode.DECODE, server_args)
            mgr.set_kv_caches(get_raiden_kv_caches(pool))
        self.addCleanup(self._close_manager, mgr)

        self.assertEqual(mgr._state_engine_max_blocks, 32)
        self.assertEqual(mgr._pulling.state_capacity, 96)

        # Up to 3 concurrent 32-page state reads fit; the 4th waits safely in line.
        self.assertTrue(mgr._pulling.admit("100", 1, 1, state_pages=32))
        self.assertTrue(mgr._pulling.admit("101", 1, 1, state_pages=32))
        self.assertTrue(mgr._pulling.admit("102", 1, 1, state_pages=32))
        self.assertFalse(mgr._pulling.admit("103", 1, 1, state_pages=32))
        mgr._pulling.release("100")
        self.assertTrue(mgr._pulling.admit("103", 1, 1, state_pages=32))


if __name__ == "__main__":
    unittest.main()
