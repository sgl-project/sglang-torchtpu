# SPDX-License-Identifier: Apache-2.0
"""CPU protocol, DMA, C2 pair-state, HostEngram, AOTAutograd, and TPU7x export tests
for DeepSeek-V4.1-Flash P/D disaggregation on Raiden."""

from __future__ import annotations

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

import ctypes
import importlib.util
import re
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
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.disaggregation.utils import (
    get_kv_transfer_buf_infos,
    setup_state_kv_args,
)
from sglang.srt.mem_cache.static_kv_storage import StaticKVStorage
from sglang.srt.model_executor.forward_mode import ForwardMode
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
from sglang_tpu.models.deepseek_v41.attention_backend import DSV41TPUAttnBackend
from sglang_tpu.models.deepseek_v41.memory_pool import (
    DSV41TPUTokenToKVPool,
    TPUPairStatePool,
    get_tpu_dsv41_bytes_per_full_token,
    get_tpu_dsv41_c2_state_fixed_bytes,
    index_bytes_per_record,
    index_page_stride,
    main_bytes_per_record,
)

_HAS_JAX = importlib.util.find_spec("jax") is not None


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
    disaggregation_mode: str = "null",
    disaggregation_transfer_backend: str = "raiden",
) -> DSV41TPUTokenToKVPool:
    from sglang.srt.runtime_context import get_context, get_parallel

    override = get_context().override_server_args(
        host="127.0.0.1",
        disaggregation_mode=disaggregation_mode,
        disaggregation_transfer_backend=disaggregation_transfer_backend,
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
    return DSV41TPUTokenToKVPool(
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


def _fill_pages(buf: torch.Tensor, pages: list[int], seed: int) -> None:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    shape = (len(pages), *buf.shape[1:])
    if buf.dtype.is_floating_point:
        buf[pages] = torch.randn(shape, dtype=buf.dtype, generator=gen)
    else:
        buf[pages] = torch.randint(
            -100000, 100000, shape, dtype=buf.dtype, generator=gen
        )


def _cpu_write_rows(
    rows: torch.Tensor,
    slots: torch.Tensor,
    values: torch.Tensor,
    after: torch.Tensor,
) -> torch.Tensor:
    """CPU reference matching ``ops/csa2.py:_write_rows`` semantics."""
    del after
    out = rows.clone()
    flat = out.view(-1, out.shape[-1])
    if flat.shape[1] > values.shape[1]:
        pad = torch.zeros(
            (values.shape[0], flat.shape[1] - values.shape[1]),
            dtype=values.dtype,
            device=values.device,
        )
        values = torch.cat([values, pad], dim=-1)
    valid = (slots >= 0) & (slots < flat.shape[0])
    if bool(valid.any()):
        flat[slots[valid].to(torch.long)] = values[valid].to(flat.dtype)
    return out


class TestDSV41RaidenDisaggregation(CustomTestCase):
    def _build_kv_args_and_aux(
        self, pool: DSV41TPUTokenToKVPool, *, port: int
    ) -> tuple[KVArgs, ctypes.Array]:
        # PD peers register a static pool through the storage the scheduler
        # builds over its allocator (`kv_storage_of`).
        allocator = SimpleNamespace(storage=None, state=None, get_kvcache=lambda: pool)
        storage = StaticKVStorage(None, allocator)
        ptrs, lens, item_lens = get_kv_transfer_buf_infos(storage)
        aux_buf = (ctypes.c_int32 * 8)()
        kv_args = KVArgs()
        kv_args.engine_rank = 0
        kv_args.kv_data_ptrs = ptrs
        kv_args.kv_data_lens = lens
        kv_args.kv_item_lens = item_lens
        kv_args.kv_cache_dtype_str = "bf16"
        setup_state_kv_args(kv_args, storage)
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
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="raiden",
        )
        decode_pool = _make_dsv41_pool(
            self,
            num_req_slots=6,
            full_pages=full_pages,
            swa_pages=swa_pages,
            bootstrap_port=bootstrap_port,
            disaggregation_mode="decode",
            disaggregation_transfer_backend="raiden",
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
        pool = _make_dsv41_pool(
            self,
            num_req_slots=6,
            full_pages=15,
            swa_pages=7,
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="raiden",
        )
        groups = get_raiden_kv_caches(pool)

        # C1: 1 source * (k + scale + index) = 3; C2: 2 sources * (k + scale + index) = 6 -> 9
        self.assertEqual(len(groups.full), 9)
        self.assertEqual({int(b.shape[0]) for b in groups.full}, {16})
        # All 6 layers have 256-token SWA page roots (7 + 1 reserved = 8).
        self.assertEqual(len(groups.swa), 6)
        self.assertEqual({int(b.shape[0]) for b in groups.swa}, {8})
        # 2 C2 source layers own TPUPairStatePool (6 + 1 spare sentinel = 7 slots).
        self.assertEqual(len(groups.state), 2)
        self.assertEqual({int(b.shape[0]) for b in groups.state}, {7})
        self.assertEqual(groups.state_type, StateType.DSV4_REQUEST_STATE)
        self.assertEqual(groups.state_max_blocks, 1)
        self.assertIsNone(groups.read_state_meta)
        self.assertIsNone(groups.write_state_meta)

        for buf in [*groups.full, *groups.swa, *groups.state]:
            self.assertIsNone(buf._base)
            self.assertEqual(buf.ndim, 3)
            self.assertEqual(buf.shape[-2] % 8, 0)
            self.assertEqual(buf.shape[-1] % 128, 0)

        kv_args, _ = self._build_kv_args_and_aux(pool, port=39000)
        self.assertIn(StateType.SWA, kv_args.state_types)
        self.assertIn(StateType.DSV4_REQUEST_STATE, kv_args.state_types)

        # Tile-alignment guard catches non-(8, 128) minor dims.
        orig_scale = pool.kv_pools[1].scale_pages[0]
        pool.kv_pools[1].scale_pages[0] = torch.zeros((16, 1, 256), dtype=torch.float32)
        try:
            with self.assertRaisesRegex(RuntimeError, "minor tile dims"):
                get_raiden_kv_caches(pool)
        finally:
            pool.kv_pools[1].scale_pages[0] = orig_scale

        # Root-ownership guard catches views.
        orig = pool.swa_kv_pool.k_pages[0]
        pool.swa_kv_pool.k_pages[0] = orig[:]
        try:
            with self.assertRaisesRegex(RuntimeError, "root tensors"):
                get_raiden_kv_caches(pool)
        finally:
            pool.swa_kv_pool.k_pages[0] = orig

        # Unsupported P/D transfer backends are rejected up front.
        with self.assertRaisesRegex(
            ValueError, "disaggregation-transfer-backend raiden"
        ):
            _make_dsv41_pool(
                self,
                num_req_slots=4,
                full_pages=4,
                swa_pages=4,
                disaggregation_mode="prefill",
                disaggregation_transfer_backend="mooncake",
            )

    def test_plugin_budget_and_aligned_pool_geometry(self):
        self.assertEqual(main_bytes_per_record(1), 528)
        self.assertEqual(main_bytes_per_record(2), 544)
        self.assertEqual(index_page_stride(1), 36)
        self.assertEqual(index_page_stride(2), 40)
        self.assertEqual(index_bytes_per_record(1), 144)
        self.assertEqual(index_bytes_per_record(2), 160)

        pool = _make_dsv41_pool(self, num_req_slots=6, full_pages=15, swa_pages=7)
        self.assertEqual(pool.kv_pools[1].scale_step, 4)
        self.assertEqual(pool.kv_pools[2].scale_step, 8)
        self.assertEqual(pool.index_pools[1].layout.page_stride, 36)
        self.assertEqual(pool.index_pools[2].layout.page_stride, 40)
        self.assertEqual(pool.index_pools[1].k_pages[0].shape, (16, 72, 128))
        self.assertEqual(pool.index_pools[2].k_pages[0].shape, (16, 40, 128))
        self.assertEqual(pool.swa_kv_pool.k_pages[0].shape, (8, 256, 512))
        self.assertEqual(pool.swa_kv_pool.kv_buffer[0].shape, (32, 64, 512))

        groups = get_raiden_kv_caches(pool)
        actual_root_bytes = sum(
            buf.nbytes for buf in (*groups.full, *groups.swa, *groups.state)
        )
        self.assertEqual(pool.get_kv_size_bytes()[0], actual_root_bytes)

        # Request-state transfer indices always return 1 block regardless of seq_len parity.
        for seq_len in (0, 1, 2, 512, 517):
            np.testing.assert_array_equal(
                pool.request_state_transfer_indices(3, seq_len),
                np.array([3], dtype=np.int32),
            )

        # Configurator budget hooks match aligned pool bytes and pass V4 through.
        fake_v41_cfg = SimpleNamespace(
            stage_compress_ratios=[0, 2, 2, 1, 2, 0],
            _unified=False,
            swa_cap_tokens=None,
            swa_ratio=0.5,
            bytes_per_swa_token=float(6 * 1024),
            stage_owner_layers={1: [3], 2: [1, 4]},
            num_layers=lambda r: {1: 1, 2: 2}.get(r, 0),
            _get_num_req_slots=lambda n: n + 1,
            ring_sizes={2: 2},
            attn_head_dim=512,
        )
        self.assertEqual(
            get_tpu_dsv41_bytes_per_full_token(lambda _: -1.0, fake_v41_cfg),
            0.5 * 6 * 1024 + (528 + 144) * 1 + (544 + 160) / 2 * 2,
        )
        expected_c2_bytes = ((6 + 1) * 8 * 1024 * 4) * 2
        self.assertEqual(
            get_tpu_dsv41_c2_state_fixed_bytes(lambda *_: -1, fake_v41_cfg, 5),
            expected_c2_bytes,
        )
        self.assertEqual(
            sum(b.nbytes for b in groups.state),
            DSV41TPUTokenToKVPool.c2_state_fixed_bytes(
                num_req_slots=6, ring_size=2, head_dim=512, num_layers=2
            ),
        )

        fake_v4_cfg = SimpleNamespace(stage_compress_ratios=[0, 4, 128])
        self.assertEqual(
            get_tpu_dsv41_bytes_per_full_token(lambda _: 123.5, fake_v4_cfg), 123.5
        )
        self.assertEqual(
            get_tpu_dsv41_c2_state_fixed_bytes(lambda _, n: n * 7, fake_v4_cfg, 9), 63
        )

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

        # Allocation/reuse clear hooks must never reallocate registered buffers.
        decode_pool.clear_request_scoped_state(decode_req_idx)
        decode_pool.clear_request_slots([decode_req_idx, canary_req_idx])
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

                room = 6200 + idx
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
                    seq_len=num_pages * 256 - (idx % 2),
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
        """Real DSV41TPUAttnBackend._pool_pairs/_pair_partners + TPUPairStatePool
        across odd/even prompt handoffs, stale reused slots, and NaN padding."""
        from sglang.srt.layers.attention.dsv4.dsv41_sparse import (
            DeepseekV41Compressor,
        )

        (
            prefill_pool,
            decode_pool,
            prefill_mgr,
            decode_mgr,
            _p_aux,
            _d_aux,
            bootstrap_addr,
        ) = self._spawn_pd_pair(full_pages=8, swa_pages=4, context_length=2048)
        unified_pool = _make_dsv41_pool(
            self, num_req_slots=6, full_pages=8, swa_pages=4
        )

        def _make_stub_backend(pool: DSV41TPUTokenToKVPool) -> DSV41TPUAttnBackend:
            backend = DSV41TPUAttnBackend.__new__(DSV41TPUAttnBackend)
            backend.token_to_kv_pool = pool
            backend.csa2_ops = SimpleNamespace(write_rows=_cpu_write_rows)
            return backend

        p_backend = _make_stub_backend(prefill_pool)
        d_backend = _make_stub_backend(decode_pool)
        u_backend = _make_stub_backend(unified_pool)

        layer = SimpleNamespace(layer_id=1, compressor=DeepseekV41Compressor)
        gen = torch.Generator(device="cpu").manual_seed(2026)

        for case_idx, prompt_len in enumerate((7, 8)):
            with self.subTest(prompt_len=prompt_len):
                total_len = prompt_len + 4
                kv_all = torch.randn(
                    (total_len, 512), dtype=torch.float32, generator=gen
                )
                score_all = torch.randn(
                    (total_len, 512), dtype=torch.float32, generator=gen
                )

                prefill_req_idx, decode_req_idx, unified_req_idx = 4, 2, 1

                # Pollute decode_pool's reused slot with NaNs/junk to prove
                # no-op clear_request_scoped_state never leaks stale data.
                d_state = decode_pool.get_attention_compress_states(layer.layer_id)
                self.assertIsInstance(d_state, TPUPairStatePool)
                d_state.state_buffer[decode_req_idx].fill_(float("nan"))

                def _run_step(backend, req_idx, start_pos, length, with_nan_pad=False):
                    kv = kv_all[start_pos : start_pos + length]
                    sc = score_all[start_pos : start_pos + length]
                    pos = torch.arange(start_pos, start_pos + length, dtype=torch.int32)
                    req = torch.full((length,), req_idx, dtype=torch.int32)
                    valid = torch.ones((length,), dtype=torch.bool)
                    if with_nan_pad:
                        kv = torch.cat(
                            [
                                kv,
                                torch.full((2, 512), float("nan"), dtype=torch.float32),
                            ],
                            dim=0,
                        )
                        sc = torch.cat(
                            [
                                sc,
                                torch.full((2, 512), float("nan"), dtype=torch.float32),
                            ],
                            dim=0,
                        )
                        pos = torch.cat([pos, torch.tensor([0, 1], dtype=torch.int32)])
                        req = torch.cat(
                            [req, torch.tensor([req_idx, req_idx], dtype=torch.int32)]
                        )
                        valid = torch.cat([valid, torch.tensor([False, False])])
                    meta = SimpleNamespace(
                        positions=pos,
                        token_req_pool=req,
                        valid=valid,
                        compress_valid=valid,
                    )
                    pooled, group_pos = backend._pool_pairs(
                        layer=layer, kv=kv, score=sc, metadata=meta
                    )
                    return pooled[:length], group_pos[:length]

                # 1) Prefill prompt on prefill_pool and unified_pool.
                p_pooled, p_gpos = _run_step(
                    p_backend, prefill_req_idx, 0, prompt_len, with_nan_pad=True
                )
                u_pooled, u_gpos = _run_step(
                    u_backend, unified_req_idx, 0, prompt_len, with_nan_pad=False
                )
                odd_mask = (torch.arange(prompt_len) % 2) == 1
                torch.testing.assert_close(p_pooled[odd_mask], u_pooled[odd_mask])
                torch.testing.assert_close(p_gpos, u_gpos)

                # 2) Live Raiden transfer of pair state from prefill -> decode.
                room = 7100 + case_idx
                sender = RaidenKVSender(prefill_mgr, bootstrap_addr, room)
                receiver = RaidenKVReceiver(decode_mgr, bootstrap_addr, room)
                self._complete_transfer(
                    sender=sender,
                    receiver=receiver,
                    prefill_pool=prefill_pool,
                    decode_pool=decode_pool,
                    prefill_req_idx=prefill_req_idx,
                    decode_req_idx=decode_req_idx,
                    src_full_pages=[1],
                    dst_full_pages=[2],
                    src_swa_pages=[1],
                    dst_swa_pages=[2],
                    seq_len=prompt_len,
                )

                # 3) Step through 4 decode tokens and compare every odd publish step.
                for step in range(4):
                    pos = prompt_len + step
                    d_out, d_gp = _run_step(
                        d_backend, decode_req_idx, pos, 1, with_nan_pad=True
                    )
                    u_out, u_gp = _run_step(
                        u_backend, unified_req_idx, pos, 1, with_nan_pad=False
                    )
                    torch.testing.assert_close(d_gp, u_gp)
                    if pos % 2 == 1:
                        self.assertFalse(bool(torch.isnan(d_out).any()))
                        torch.testing.assert_close(d_out, u_out)

                # Spare sentinel slot stays pristine (kv=0, score=-inf).
                sentinel = d_state.state_buffer[d_state.num_req_slots]
                torch.testing.assert_close(
                    sentinel[:, :512], torch.zeros_like(sentinel[:, :512])
                )
                self.assertTrue(bool(torch.isneginf(sentinel[:, 512:1024]).all()))

    @unittest.skipUnless(_HAS_JAX, "jax is required for JAX op verification")
    def test_pair_compress_jax_write_rows_3d_layout(self):
        """Verify the JAX _write_rows op directly on CPU with 3D root layout."""
        import jax.numpy as jnp

        from sglang_tpu.models.deepseek_v41.ops.csa2 import _write_rows

        rows = jnp.zeros((7, 8, 1024), dtype=jnp.float32)
        slots = jnp.array([2 * 8 + 0, -1, 2 * 8 + 1], dtype=jnp.int32)
        vals = jnp.arange(3 * 1024, dtype=jnp.float32).reshape(3, 1024) + 1.0
        updated = _write_rows(rows, slots, vals, jnp.zeros((3, 1024), jnp.float32))
        self.assertEqual(updated.shape, (7, 8, 1024))
        np.testing.assert_allclose(np.asarray(updated[2, 0]), np.asarray(vals[0]))
        np.testing.assert_allclose(np.asarray(updated[2, 1]), np.asarray(vals[2]))
        np.testing.assert_allclose(np.asarray(updated[6]), 0.0)

    def test_host_engram_disagg_decode_steps(self):
        """PREBUILT history seeding + engram_host CPU decode hashing matches unified."""
        from sglang.srt.layers.engram import EngramHasher, EngramLayout
        from sglang.srt.model_executor.model_runner_components.ngram_embedding_manager import (
            NgramEmbeddingManager,
        )
        from sglang_tpu.models.deepseek_v41 import engram_host

        layout = EngramLayout(
            max_ngram_size=3,
            layer_ids=(1, 4),
            num_embeddings=(256, 256),
            primes=(((17, 19), (23, 29)), ((31, 37), (41, 43))),
            n_heads=2,
            head_dim=16,
        )

        def _make_hasher():
            with patch(
                "sglang.srt.layers.engram.build_compressed_token_map",
                return_value=(list(range(64)), 64),
            ):
                h = EngramHasher(layout, None, pad_id=0, compressed_vocab_size=64)
            h.init_history(num_req_slots=6, device=torch.device("cpu"))
            engram_host.move_hasher_to_host(h)
            return h

        unified_hasher = _make_hasher()
        decode_hasher = _make_hasher()

        # Test both short prompt (len=1 < n-1=2) and normal prompt (len=5 >= n-1).
        prompts = {
            "short": ([11], 22, 33),
            "normal": ([5, 9, 13, 17, 21], 25, 29),
        }
        for name, (prompt_ids, tok1, tok2) in prompts.items():
            with self.subTest(case=name):
                u_slot, d_slot = 1, 3
                L = len(prompt_ids)

                # Unified: EXTEND prompt -> DECODE tok1 -> DECODE tok2.
                extend_batch = SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND,
                    spec_info=None,
                    positions=torch.arange(L, dtype=torch.int32),
                    req_pool_indices=torch.tensor([u_slot], dtype=torch.int32),
                    out_cache_loc=torch.arange(1, L + 1, dtype=torch.int32),
                    extend_seq_lens=torch.tensor([L], dtype=torch.int32),
                    extend_start_loc=torch.tensor([0], dtype=torch.int32),
                    engram_history=torch.zeros((1, 2), dtype=torch.int32),
                )
                engram_host.host_hash_ids(
                    unified_hasher,
                    torch.tensor(prompt_ids, dtype=torch.int64),
                    extend_batch,
                )

                # Disagg decode: seed via NgramEmbeddingManager.prepare_for_forward(PREBUILT).
                mgr = NgramEmbeddingManager(
                    enabled=False,
                    table=None,
                    n=3,
                    engram_hasher=decode_hasher,
                )
                prebuilt_req = SimpleNamespace(
                    origin_input_ids=list(prompt_ids),
                    output_ids=[tok1],
                    kv=SimpleNamespace(req_pool_idx=d_slot),
                )
                prebuilt_batch = SimpleNamespace(
                    forward_mode=ForwardMode.PREBUILT,
                    reqs=[prebuilt_req],
                )
                mgr.prepare_for_forward(prebuilt_batch, chunked_req=None)
                self.assertEqual(decode_hasher.history.device.type, "cpu")

                # Compare decode step 1 (tok1) and step 2 (tok2), with a padded
                # bucket of 2 slots on decode (real + padding slot -> spare row).
                for step_idx, tok in enumerate((tok1, tok2)):
                    pos = L + step_idx
                    u_dec = SimpleNamespace(
                        forward_mode=ForwardMode.DECODE,
                        spec_info=None,
                        positions=torch.tensor([pos], dtype=torch.int32),
                        req_pool_indices=torch.tensor([u_slot], dtype=torch.int32),
                        out_cache_loc=torch.tensor(
                            [L + 1 + step_idx], dtype=torch.int32
                        ),
                        extend_seq_lens=None,
                        extend_start_loc=None,
                        engram_history=None,
                    )
                    d_dec = SimpleNamespace(
                        forward_mode=ForwardMode.DECODE,
                        spec_info=None,
                        positions=torch.tensor([pos, 0], dtype=torch.int32),
                        req_pool_indices=torch.tensor(
                            [d_slot, decode_hasher.pad_row], dtype=torch.int32
                        ),
                        out_cache_loc=torch.tensor(
                            [L + 1 + step_idx, 0], dtype=torch.int32
                        ),
                        extend_seq_lens=None,
                        extend_start_loc=None,
                        engram_history=None,
                    )
                    u_ids = engram_host.host_hash_ids(
                        unified_hasher, torch.tensor([tok], dtype=torch.int64), u_dec
                    )
                    d_ids = engram_host.host_hash_ids(
                        decode_hasher,
                        torch.tensor([tok, 0], dtype=torch.int64),
                        d_dec,
                    )
                    torch.testing.assert_close(d_ids[:1], u_ids)

    def test_aot_autograd_compile_preserves_root_buffers_without_base_error(self):
        """AOTAutograd fullgraph compile mutates root buffers and reads fresh views."""
        pool = _make_dsv41_pool(self, num_req_slots=4, full_pages=4, swa_pages=2)
        layer_c2 = 1
        state = pool.get_attention_compress_states(layer_c2)
        self.assertIsInstance(state, TPUPairStatePool)

        def step(
            swa_root,
            words_root,
            scales_root,
            index_root,
            state_root,
            swa_val,
            word_val,
            scale_val,
            index_val,
            state_val,
        ):
            swa_root.copy_(swa_root + swa_val)
            words_root.copy_(words_root + word_val)
            scales_root.copy_(scales_root + scale_val)
            index_root.copy_(index_root + index_val)
            state_root.copy_(state_root + state_val)

            swa_view = swa_root.view(-1, 64, 512)
            words_view = words_root.view(-1, 128)
            scales_view = scales_root.view(-1)
            index_view = index_root.view(-1, 128)
            state_view = state_root.view(-1, state.feat_stride)[
                torch.tensor([8], dtype=torch.int64)
            ]
            return (
                swa_view[4, 0, 0],
                words_view[128, 0],
                scales_view[1024],
                index_view[40, 0],
                state_view[0, 0],
            )

        compiled = torch.compile(step, backend="aot_eager", fullgraph=True)
        swa_root = pool.get_swa_root(layer_c2)
        words_root, scales_root = pool.get_main_kv_roots(layer_c2)
        index_root = pool.get_index_k_root(layer_c2)
        state_root = state.state_buffer

        ptrs_before = [
            t.data_ptr()
            for t in (swa_root, words_root, scales_root, index_root, state_root)
        ]
        outs = compiled(
            swa_root,
            words_root,
            scales_root,
            index_root,
            state_root,
            torch.tensor(2.0, dtype=torch.bfloat16),
            torch.tensor(3, dtype=torch.int32),
            torch.tensor(4.5, dtype=torch.float32),
            torch.tensor(5, dtype=torch.int32),
            torch.tensor(6.25, dtype=torch.float32),
        )
        self.assertEqual(float(outs[0]), 2.0)
        self.assertEqual(int(outs[1]), 3)
        self.assertAlmostEqual(float(outs[2]), 4.5)
        self.assertEqual(int(outs[3]), 5)
        self.assertAlmostEqual(float(outs[4]), 6.25)

        # Fresh views through pool accessors observe the mutated roots, and pointers stay fixed.
        self.assertEqual(float(pool.get_swa_pages(layer_c2)[4, 0, 0]), 2.0)
        w_view, s_view = pool.get_main_kv(layer_c2)
        self.assertEqual(int(w_view[128, 0]), 3)
        self.assertAlmostEqual(float(s_view[1024]), 4.5)
        self.assertEqual(int(pool.get_index_k(layer_c2)[40, 0]), 5)
        self.assertAlmostEqual(
            float(
                state.get_state_by_state_loc(torch.tensor([8], dtype=torch.int64)).kv[
                    0, 0
                ]
            ),
            6.25,
        )
        self.assertEqual(
            [
                t.data_ptr()
                for t in (swa_root, words_root, scales_root, index_root, state_root)
            ],
            ptrs_before,
        )

    @unittest.skipUnless(_HAS_JAX, "jax is required for TPU7x MLIR export tests")
    def test_tpu7x_mlir_export_aligned_kernels_and_no_sc_constant_operands(self):
        """TPU7x CPU export of aligned indexer, CSA2 main KV, SWA 256-page, and pair-state ops."""
        import jax
        import jax.numpy as jnp

        from sglang_tpu.kernels.deepseek_v41.index_selection.fp8.cache import (
            PackedFP8Layout,
        )
        from sglang_tpu.kernels.deepseek_v41.index_selection.selection import (
            prepare_fp8_candidates,
            select_fp8_indexer_jax,
            select_fp8_prepared,
        )
        from sglang_tpu.models.deepseek_v41.kernels.csa2.cache import (
            gather_csa2_kv_sc,
        )
        from sglang_tpu.models.deepseek_v41.ops.csa2 import (
            _publish_main,
            _publish_swa,
            _write_rows,
        )
        from sglang_tpu.models.deepseek_v41.ops.indexer import _write_index_k
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        def _assert_no_sc_constant_operands(mlir: str) -> int:
            consts: set[str] = set()
            sc_calls = 0
            for line in mlir.splitlines():
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
                    match = re.search(r"@tpu_custom_call\(([^)]*)\)", stripped)
                    self.assertIsNotNone(match, stripped[:200])
                    args = [a.strip() for a in match.group(1).split(",") if a.strip()]
                    const_args = [a for a in args if a in consts]
                    self.assertEqual(
                        const_args,
                        [],
                        f"SparseCore custom_call received constant operand(s) "
                        f"{const_args}: {stripped[:200]}",
                    )
            return sc_calls

        with (
            patch(
                "sglang_tpu.pallas_export.get_pallas_tpu_device_kind",
                return_value="TPU7x",
            ),
            patch("jax.default_backend", return_value="tpu"),
            tpu_info_for_cpu_export(),
            jax._src.config.export_ignore_forward_compatibility(True),
        ):
            # 1) Aligned packed FP8 indexer selection & write for ratio 1 (stride 36)
            #    and ratio 2 (stride 40).
            for ratio in (1, 2):
                stride = index_page_stride(ratio)
                subpages = (256 // ratio) // 128
                full_pages = 256
                pages = full_pages * subpages
                rows_per_full = subpages * stride
                layout = PackedFP8Layout(pages, page_stride=stride, page_size=128)
                layout_args = (
                    layout.pages,
                    layout.page_stride,
                    layout.page_offset,
                    layout.page_size,
                )
                width = 256 * 128

                exp_dense = jax.export.export(
                    jax.jit(
                        lambda q, w, b, p, r, l, c, la=layout_args: (
                            select_fp8_indexer_jax(
                                q,
                                (b,),
                                p,
                                w,
                                l,
                                c,
                                r,
                                jnp.max(l, initial=0).reshape(1),
                                la,
                                512,
                                8,
                                True,
                                False,
                                True,
                                width,
                            )
                        )
                    ),
                    platforms=("tpu",),
                )(
                    jax.ShapeDtypeStruct((102, 32, 128), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((102, 32), jnp.float32),
                    jax.ShapeDtypeStruct((pages * stride, 128), jnp.int32),
                    jax.ShapeDtypeStruct((8, 256), jnp.int32),
                    jax.ShapeDtypeStruct((102,), jnp.int32),
                    jax.ShapeDtypeStruct((102,), jnp.int32),
                    jax.ShapeDtypeStruct((102, 128), jnp.int32),
                )
                self.assertGreaterEqual(
                    _assert_no_sc_constant_operands(exp_dense.mlir_module()), 1
                )

                def _prep_and_select(q, w, b, p, r, l, c, la=layout_args):
                    prepared = prepare_fp8_candidates((b,), p, c, r, la, 8, width)
                    return select_fp8_prepared(q, w, l, prepared, 512, 8, True, width)

                exp_prep = jax.export.export(
                    jax.jit(_prep_and_select),
                    platforms=("tpu",),
                )(
                    jax.ShapeDtypeStruct((128, 32, 128), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((128, 32), jnp.float32),
                    jax.ShapeDtypeStruct((pages * stride, 128), jnp.int32),
                    jax.ShapeDtypeStruct((8, 256), jnp.int32),
                    jax.ShapeDtypeStruct((128,), jnp.int32),
                    jax.ShapeDtypeStruct((128,), jnp.int32),
                    jax.ShapeDtypeStruct((128, 128), jnp.int32),
                )
                self.assertGreaterEqual(
                    _assert_no_sc_constant_operands(exp_prep.mlir_module()), 1
                )

                exp_write_idx = jax.export.export(
                    jax.jit(
                        lambda bank, recs, keys, la=layout_args: _write_index_k(
                            bank, recs, keys, *la
                        ),
                        donate_argnums=(0,),
                    ),
                    platforms=("tpu",),
                )(
                    jax.ShapeDtypeStruct((full_pages, rows_per_full, 128), jnp.int32),
                    jax.ShapeDtypeStruct((64,), jnp.int32),
                    jax.ShapeDtypeStruct((64, 128), jnp.bfloat16),
                )
                self.assertIn("func.func", exp_write_idx.mlir_module())

            # 2) CSA2 main KV publish (3D roots) & SparseCore gather (deep=False & True).
            for page_records in (256, 128):
                num_pages = 8
                exp_pub = jax.export.export(
                    jax.jit(_publish_main, donate_argnums=(0, 1)),
                    platforms=("tpu",),
                )(
                    jax.ShapeDtypeStruct((num_pages, page_records, 128), jnp.int32),
                    jax.ShapeDtypeStruct((num_pages, 8, 128), jnp.float32),
                    jax.ShapeDtypeStruct((64,), jnp.int32),
                    jax.ShapeDtypeStruct((64, 512), jnp.bfloat16),
                )
                self.assertEqual(
                    _assert_no_sc_constant_operands(exp_pub.mlir_module()), 1
                )

                exp_gat = jax.export.export(
                    jax.jit(lambda w, s, r: gather_csa2_kv_sc(w, s, r, deep=False)),
                    platforms=("tpu",),
                )(
                    jax.ShapeDtypeStruct((num_pages * page_records, 128), jnp.int32),
                    jax.ShapeDtypeStruct((num_pages * 1024,), jnp.float32),
                    jax.ShapeDtypeStruct((64, 512), jnp.int32),
                )
                self.assertEqual(
                    _assert_no_sc_constant_operands(exp_gat.mlir_module()), 1
                )

                exp_gat_deep = jax.export.export(
                    jax.jit(lambda w, s, r: gather_csa2_kv_sc(w, s, r, deep=True)),
                    platforms=("tpu",),
                )(
                    jax.ShapeDtypeStruct((num_pages * page_records, 128), jnp.int32),
                    jax.ShapeDtypeStruct((num_pages * 1024,), jnp.float32),
                    jax.ShapeDtypeStruct((64, 512), jnp.int32),
                )
                self.assertIn(
                    "csa2_banked_word_scale_gather", exp_gat_deep.mlir_module()
                )

            # 3) SWA publish on 3D 256-token root for both prefill (64) and fused decode (96).
            for tokens in (64, 96):
                exp_swa = jax.export.export(
                    jax.jit(_publish_swa, donate_argnums=(0,)),
                    platforms=("tpu",),
                )(
                    jax.ShapeDtypeStruct((8, 256, 512), jnp.bfloat16),
                    jax.ShapeDtypeStruct((tokens,), jnp.int32),
                    jax.ShapeDtypeStruct((tokens, 512), jnp.bfloat16),
                )
                mlir = exp_swa.mlir_module()
                self.assertEqual(_assert_no_sc_constant_operands(mlir), 1)
                if tokens == 96:
                    self.assertIn("decode_swa_sort_pack", mlir)

            # 4) Pair-state 3D root write_rows export.
            exp_pair = jax.export.export(
                jax.jit(_write_rows, donate_argnums=(0,)),
                platforms=("tpu",),
            )(
                jax.ShapeDtypeStruct((7, 8, 1024), jnp.float32),
                jax.ShapeDtypeStruct((64,), jnp.int32),
                jax.ShapeDtypeStruct((64, 1024), jnp.float32),
                jax.ShapeDtypeStruct((64, 1024), jnp.float32),
            )
            self.assertIn("func.func", exp_pair.mlir_module())

    def test_raiden_kv_manager_disables_shared_staging_for_heterogeneous_full_pool(
        self,
    ):
        """DSV4.1 full-KV roots are heterogeneous, so Raiden must keep full mirrors."""
        import sys
        import types
        from contextlib import ExitStack
        from unittest.mock import patch

        from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager

        pool = _make_dsv41_pool(self, num_req_slots=4, full_pages=4, swa_pages=2)
        groups = get_raiden_kv_caches(pool)

        engines = []

        class _FakeEngine:
            def __init__(self, **kwargs):
                engines.append(kwargs)

        fake_mod = types.ModuleType("tpu_sync.api.torch.kv_cache_manager")
        fake_mod.KVCacheManager = _FakeEngine

        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {fake_mod.__name__: fake_mod}))
            manager = object.__new__(RaidenKVManager)
            manager._engine = None
            manager._kv_caches = list(groups.full)
            manager._swa_kv_caches = list(groups.swa)
            manager._state_kv_caches = tuple(groups.state)
            manager._state_max_blocks = groups.state_max_blocks
            manager._dynamic_host_staging = True
            manager._staging_layers_total = 8
            manager._slot_ceiling = lambda: 32
            manager._max_request_blocks = lambda: 37
            manager._swa_max_request_blocks = lambda: 2
            manager.attn_tp_rank = 0
            manager.attn_tp_size = 1
            manager.attn_dp_size = 8
            manager.system_dp_size = manager.attn_cp_size = 1
            manager.pp_size = 1
            manager.server_args = SimpleNamespace(nnodes=1)
            manager.local_ip = "127.0.0.1"
            manager._local_rank = 0
            manager._control_port = 31000
            manager._timeout_s = 120
            manager._transfer_parallelism = 4
            manager.chunked_transfer = True
            manager._build_engine()

        self.assertEqual(len(engines), 3)
        # Full engine has heterogeneous KV/scale/indexer roots -> full mirror.
        self.assertEqual(engines[0]["num_staging_layers"], len(groups.full))
        # SWA engine is homogeneous -> shares bounded staging layers safely.
        self.assertEqual(engines[1]["num_staging_layers"], min(len(groups.swa), 8))
        # State engine always keeps full mirrors.
        self.assertIsNone(engines[2].get("num_staging_layers"))

    def test_dspark_write_target_hidden_kv_publishes_to_swa_root(self):
        """DSpark draft SWA publish must donate/write the 256-token root, not a 64-token view."""
        import sglang_tpu.models.deepseek_v41.ops as dsv41_ops
        from sglang_tpu.models.deepseek_v41 import dspark

        pool = _make_dsv41_pool(self, num_req_slots=4, full_pages=4, swa_pages=2)
        stages = [
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    layer_id=layer_id,
                    wkv=lambda x, offset=layer_id: (x + float(offset + 1), None),
                    kv_norm=lambda x: x,
                    freqs_cis=None,
                )
            )
            for layer_id in (0, 1)
        ]
        model = SimpleNamespace(project_target_hidden=lambda x: x, stages=stages)
        seen_roots = []
        ptrs_before = [pool.get_swa_root(i).data_ptr() for i in (0, 1)]

        def _fake_publish_swa(pages, records, kv):
            self.assertIsNone(
                pages._base, "publish_swa must receive a base root tensor, not a view"
            )
            self.assertEqual(pages.shape[1:], (256, 512))
            seen_roots.append(id(pages))
            out = pages.clone()
            flat = out.view(-1, 512)
            for rec, row in zip(records.tolist(), kv):
                if rec >= 0:
                    flat[rec] = row
            return out

        fake_csa2 = types.ModuleType("sglang_tpu.models.deepseek_v41.ops.csa2")
        fake_csa2.publish_swa = _fake_publish_swa

        main_hidden = torch.arange(3 * 512, dtype=torch.bfloat16).reshape(3, 512) / 512
        swa_loc = torch.tensor([256, -1, 258], dtype=torch.int64)
        positions = torch.tensor([0, 1, 2], dtype=torch.int64)
        with (
            patch.object(dspark, "apply_rope", side_effect=lambda x, pos, freqs: x),
            patch.dict(
                sys.modules,
                {"sglang_tpu.models.deepseek_v41.ops.csa2": fake_csa2},
            ),
            patch.object(dsv41_ops, "csa2", fake_csa2, create=True),
        ):
            dspark.write_target_hidden_kv(
                model,
                main_hidden=main_hidden,
                swa_loc=swa_loc,
                positions=positions,
                pool=pool,
            )

        self.assertEqual(seen_roots, [id(pool.get_swa_root(i)) for i in (0, 1)])
        self.assertEqual([pool.get_swa_root(i).data_ptr() for i in (0, 1)], ptrs_before)
        for layer_id in (0, 1):
            view = pool.get_swa_pages(layer_id).view(-1, 512)
            expected = main_hidden + float(layer_id + 1)
            torch.testing.assert_close(view[256], expected[0])
            torch.testing.assert_close(view[257], torch.zeros_like(view[257]))
            torch.testing.assert_close(view[258], expected[2])


if __name__ == "__main__":
    unittest.main()
