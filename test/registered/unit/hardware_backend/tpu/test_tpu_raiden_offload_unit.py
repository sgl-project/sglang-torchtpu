"""Unit tests for the TPU raiden KV-offload host pool and future-event.

These cover the hardware-agnostic logic only (page-granular allocation,
token->page id conversion, the RaidenFuture event adapter). The actual raiden
DMA path requires a TPU + the built raiden extension and is validated on-device.

The host pool is constructed via ``object.__new__`` so its ``__init__`` (which
builds a real ``RaidenKVTransfer`` over a torch_tpu device pool) is bypassed; we
inject a fake transfer that records the page ids it is handed.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import ast
import collections
import inspect
import os
import sys
import textwrap
import threading
import time
import types
import unittest
from collections import defaultdict
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import zmq
from google.protobuf.wrappers_pb2 import Int64Value

from sglang.srt.disaggregation.base.conn import BaseKVSender, KVArgs, StateType
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeReqToTokenPool
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    TransferBackend,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache import hicache_auto_size as sizing
from sglang.srt.mem_cache.hicache_auto_size import auto_size_hicache
from sglang.srt.mem_cache.pool_host import base as host_base
from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.raiden.conn import (
    DEFAULT_KV_PORT,
    RaidenKVManager,
    RaidenKVSender,
    get_raiden_kv_caches,
)
from sglang_tpu.disaggregation.raiden.host_pool import StagingAdmission
from sglang_tpu.disaggregation.raiden.serde import (
    Blocks,
    Chunk,
    MessageTag,
    PullReq,
    ReshardBlocks,
    ReshardChunk,
    ReshardChunkReady,
    ReshardReady,
    ReshardReq,
)
from sglang_tpu.disaggregation.raiden.state import DecodeState, PrefillState
from sglang_tpu.kv_pool_host import (
    TPU_DMA_LAYOUT,
    TpuDMAHostKVPool,
)
from sglang_tpu.raiden_transfer import RaidenFutureEvent, RaidenKVTransfer
from sglang_tpu.tpu_dma_hicache import NOOP_EVENT, CombinedFutureEvent


def _polled(sent=(), received=(), failed=(), completed_ns=None):
    """Fake poll_stats_with_details() result for the given req ids."""

    def settled(ids):
        return [
            SimpleNamespace(
                req_id=r,
                completed_ns=completed_ns,
                elapsed_ns=None,
                peer=None,
                code=None,
                reason=None,
                phase=None,
            )
            for r in ids
        ]

    return SimpleNamespace(
        sent=settled(sent), received=settled(received), failed=settled(failed)
    )


def _reshard_chunk_frames(room, index, start, total_pages, blocks, src_rank):
    return ReshardChunk(room, index, start, total_pages, blocks, src_rank).to_frames()


def _reshard_blocks_frames(room, blocks, src_rank):
    return ReshardBlocks(room, blocks, src_rank, b"", []).to_frames()


class _FakeTransfer:
    """Records (src, dst) page ids passed to backup/load."""

    def __init__(self):
        self.backups = []
        self.loads = []

    def backup(self, device_pages, host_pages):
        self.backups.append((list(device_pages), list(host_pages)))
        return RaidenFutureEvent(None)  # immediately "ready"

    def load(self, host_pages, device_pages):
        self.loads.append((list(host_pages), list(device_pages)))
        return RaidenFutureEvent(None)


def _make_host_pool(page_size=4, page_num=3):
    pool = object.__new__(TpuDMAHostKVPool)
    pool.page_size = page_size
    pool.page_num = page_num
    pool.size = page_size * page_num
    pool.layout = TPU_DMA_LAYOUT
    import threading

    pool.lock = threading.RLock()
    pool.transfer = _FakeTransfer()
    pool.clear()
    return pool


class TestDecodeReqToTokenPool(CustomTestCase):
    def test_write_matches_req_to_token_pool_api(self):
        pool = DecodeReqToTokenPool(
            size=4,
            max_context_len=8,
            device="tpu:0",
            enable_memory_saver=False,
            pre_alloc_size=4,
        )
        values = torch.tensor([11, 12, 13], dtype=torch.int32)

        pool.write((2, slice(1, 4)), values)

        torch.testing.assert_close(pool.req_to_token[2, 1:4], values)


class TestRaidenHostMemoryBudget(CustomTestCase):
    def _device_pool(self):
        return SimpleNamespace(
            size=8,
            store_dtype=torch.bfloat16,
            start_layer=0,
            end_layer=2,
            layer_num=2,
            get_fused_kv_buffer=Mock(
                return_value=torch.empty((3, 4, 1, 2, 128), dtype=torch.bfloat16)
            ),
        )

    def _host_pool(self, *, host_size=0, pool_label="kv"):
        return TpuDMAHostKVPool(
            self._device_pool(),
            host_to_device_ratio=2,
            host_size=host_size,
            page_size=4,
            layout="page_first_fused",
            pool_label=pool_label,
        )

    def test_oversized_pool_fails_before_raiden_allocation(self):
        """HiCache must reject padded, page-rounded bytes above the per-rank budget."""
        # 20 tokens after page rounding, 512 bytes/layer/token, two layers.
        requested = 20_480
        for host_size in (0, 16_384 / 1e9):
            for budget in (requested - 1, requested):
                with (
                    self.subTest(host_size=host_size, budget=budget),
                    patch.object(host_base, "ranks_per_host", return_value=8),
                    patch.object(
                        host_base,
                        "available_host_memory_bytes",
                        return_value=host_base.HICACHE_HOST_MEMORY_RESERVE_BYTES
                        + budget * 8,
                    ),
                    patch("sglang_tpu.raiden_transfer.RaidenKVTransfer") as transfer,
                ):
                    if budget < requested:
                        with self.assertRaisesRegex(
                            ValueError, "Not enough host memory available"
                        ):
                            self._host_pool(host_size=host_size)
                        transfer.assert_not_called()
                    else:
                        pool = self._host_pool(host_size=host_size)
                        self.assertEqual(pool.available_size(), 20)
                        self.assertEqual(pool.size_per_token, 1024)
                        transfer.assert_called_once()

    def test_explicit_hybrid_pools_share_one_budget(self):
        """Full and SWA pools must not each spend the same host-memory allowance."""
        from sglang_tpu.platform import TpuOOTPlatform
        from sglang_tpu.tpu_platform import TpuSRTPlatform

        for platform_cls in (TpuSRTPlatform, TpuOOTPlatform):
            for host_size in (0, 16_384 / 1e9):
                self._check_explicit_hybrid_budget(platform_cls(), host_size)

    def test_native_or_disabled_hicache_does_not_reserve_raiden_budget(self):
        """The Raiden policy must not consume headroom for other cache paths."""
        from sglang_tpu.platform import TpuOOTPlatform
        from sglang_tpu.tpu_platform import TpuSRTPlatform

        for platform_cls in (TpuSRTPlatform, TpuOOTPlatform):
            for io_backend, enabled in (("torch_tpu", True), ("raiden", False)):
                with (
                    self.subTest(platform=platform_cls.__name__, backend=io_backend),
                    patch.object(sizing, "current_platform", platform_cls()),
                    get_context().override_server_args(
                        hicache_io_backend=io_backend,
                        hicache_host_memory_fraction=None,
                    ),
                    patch.object(
                        host_base,
                        "available_host_memory_bytes",
                        side_effect=AssertionError("unexpected host-memory query"),
                    ),
                    auto_size_hicache(None, None, enabled=enabled),
                ):
                    self.assertIsNone(host_base._host_memory_budget.get())

    def _check_explicit_hybrid_budget(self, platform, host_size):
        with (
            self.subTest(platform=type(platform).__name__, host_size=host_size),
            patch.object(sizing, "current_platform", platform),
            get_context().override_server_args(
                enable_hierarchical_cache=True,
                hicache_ratio=2.0,
                hicache_size=host_size,
                hicache_io_backend="raiden",
                hicache_host_memory_fraction=None,
            ),
            patch.object(host_base, "ranks_per_host", return_value=8),
            patch.object(
                host_base,
                "available_host_memory_bytes",
                return_value=host_base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 30_000 * 8,
            ),
            patch("sglang_tpu.raiden_transfer.RaidenKVTransfer") as transfer,
        ):
            with auto_size_hicache(None, None, enabled=True):
                full = self._host_pool(host_size=host_size, pool_label="full")
                self.assertEqual(full.available_size(), 20)
                with self.assertRaisesRegex(ValueError, "Not enough host memory"):
                    self._host_pool(host_size=host_size, pool_label="swa")
                self.assertEqual(transfer.call_count, 1)
            # A failed startup must not leak its reservations into the next one.
            with auto_size_hicache(None, None, enabled=True):
                self.assertEqual(
                    self._host_pool(host_size=host_size).available_size(), 20
                )
        self.assertIsNone(host_base._host_memory_budget.get())


class TestRaidenHostPoolPaging(CustomTestCase):
    def test_smaller_host_pool_warns_instead_of_failing(self):
        device_pool = SimpleNamespace(
            size=32,
            store_dtype=torch.bfloat16,
            start_layer=0,
            end_layer=1,
            layer_num=1,
            get_fused_kv_buffer=Mock(
                return_value=torch.empty((8, 4), dtype=torch.bfloat16)
            ),
        )

        # Test the sizing warning independently of the CI runner's free RAM.
        with (
            patch.object(host_base, "ranks_per_host", return_value=1),
            patch.object(
                host_base,
                "available_host_memory_bytes",
                return_value=host_base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 1024,
            ),
            patch("sglang_tpu.raiden_transfer.RaidenKVTransfer") as transfer_cls,
            self.assertLogs("sglang_tpu.kv_pool_host", level="WARNING") as logs,
        ):
            pool = TpuDMAHostKVPool(
                device_pool,
                host_to_device_ratio=0.5,
                host_size=0,
                page_size=4,
                layout="page_first_fused",
            )

        self.assertEqual(pool.size, 20)
        self.assertLess(pool.size, device_pool.size)
        self.assertIn("L2 cache effectiveness is reduced", logs.output[0])
        device_pool.get_fused_kv_buffer.assert_called_once_with(0)
        transfer_cls.assert_called_once_with(device_pool, 5)

    def test_decode_kv_args_include_raiden_metadata(self):
        captured = {}

        class _KVPool:
            page_size = 128
            head_num = 4
            start_layer = 0
            layer_num = 1
            buffers = [torch.empty(3, 4)]

            def get_contiguous_buf_infos(self):
                return [11], [22], [33]

            def get_fused_kv_buffer(self, layer_id):
                return self.buffers[layer_id]

        class _DraftKVPool:
            start_layer = 0
            layer_num = 1
            buffers = [torch.empty(3, 4)]

            def get_contiguous_buf_infos(self):
                return [44], [55], [66]

            def get_fused_kv_buffer(self, layer_id):
                return self.buffers[layer_id]

        class _Manager:
            def __init__(self, kv_args, *_args):
                captured["kv_args"] = kv_args

            def set_kv_caches(self, kv_caches):
                captured["kv_caches"] = kv_caches

        def _get_kv_class(_backend, class_type):
            if class_type == KVClassType.KVARGS:
                return KVArgs
            return _Manager

        model_config = SimpleNamespace(
            num_hidden_layers=2,
            get_total_num_kv_heads=Mock(return_value=8),
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.transfer_backend = TransferBackend.RAIDEN
        queue.tp_rank = 1
        queue.pp_rank = 0
        queue.scheduler = SimpleNamespace(
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(kv_cache_dtype_str="bfloat16")
            ),
            enable_hisparse=False,
            sliding_window_size=4096,
            model_config=model_config,
            server_args=SimpleNamespace(),
        )
        queue.token_to_kv_pool = _KVPool()
        queue.draft_token_to_kv_pool = _DraftKVPool()
        queue.metadata_buffers = SimpleNamespace(
            get_buf_infos=Mock(return_value=([], [], []))
        )
        queue.is_mla_backend = False
        queue.enable_staging = False

        with (
            patch(
                "sglang.srt.disaggregation.decode.get_parallel",
                return_value=SimpleNamespace(attn_tp_size=2, dp_rank=2),
            ),
            patch(
                "sglang.srt.disaggregation.decode.get_kv_class",
                side_effect=_get_kv_class,
            ),
            patch("sglang.srt.disaggregation.decode.setup_state_kv_args"),
            # _init_kv_manager reads the IB device off the disagg bag; publish it.
            get_context().override_server_args(disaggregation_ib_device="mlx5_0"),
        ):
            queue._init_kv_manager()

        kv_args = captured["kv_args"]
        self.assertEqual(kv_args.ib_device, "mlx5_0")
        self.assertEqual(kv_args.page_size, 128)
        self.assertEqual(kv_args.sliding_window_size, 4096)
        self.assertEqual(kv_args.kv_head_num, 4)
        self.assertEqual(kv_args.total_kv_head_num, 8)
        self.assertEqual(
            captured["kv_caches"].full,
            [_KVPool.buffers[0], _DraftKVPool.buffers[0]],
        )
        self.assertEqual(captured["kv_caches"].swa, [])
        model_config.get_total_num_kv_heads.assert_called_once_with()

    def test_sender_matches_base_api_contract(self):
        for method_name in ("__init__", "send"):
            base_params = inspect.signature(
                getattr(BaseKVSender, method_name)
            ).parameters
            raiden_params = inspect.signature(
                getattr(RaidenKVSender, method_name)
            ).parameters
            self.assertEqual(list(raiden_params), list(base_params))

    def test_uniform_kv_caches_are_returned(self):
        class _Pool:
            start_layer = 0
            layer_num = 2
            buffers = [torch.empty(3, 4), torch.empty(3, 4)]

            def get_fused_kv_buffer(self, layer_id):
                return self.buffers[layer_id]

        pool = _Pool()
        groups = get_raiden_kv_caches(pool)
        self.assertEqual(groups.full, pool.buffers)
        self.assertEqual(groups.swa, [])

    def test_mixed_block_counts_require_uniform_swa_memory(self):
        class _Pool:
            start_layer = 0
            layer_num = 2
            buffers = [torch.empty(2, 4), torch.empty(3, 4)]

            def get_fused_kv_buffer(self, layer_id):
                return self.buffers[layer_id]

        with self.assertRaisesRegex(RuntimeError, "uniform full KV block counts"):
            get_raiden_kv_caches(_Pool())

    def test_hybrid_swa_caches_are_grouped_by_page_namespace(self):
        class _Pool:
            start_layer = 0
            layer_num = 4
            layers_mapping = {
                0: (0, False),
                1: (0, True),
                2: (1, False),
                3: (1, True),
            }
            buffers = [
                torch.empty(3, 4),
                torch.empty(2, 4),
                torch.empty(3, 4),
                torch.empty(2, 4),
            ]

            def get_fused_kv_buffer(self, layer_id):
                return self.buffers[layer_id]

        groups = get_raiden_kv_caches(_Pool())
        self.assertEqual(groups.full, [_Pool.buffers[0], _Pool.buffers[2]])
        self.assertEqual(groups.swa, [_Pool.buffers[1], _Pool.buffers[3]])

    def test_hybrid_pipeline_grouping_uses_global_layer_mapping(self):
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        class _LocalPool:
            def __init__(self, blocks):
                self.buffers = [torch.empty(blocks, 4), torch.empty(blocks, 4)]

            def get_fused_kv_buffer(self, layer_id):
                return self.buffers[layer_id]

        for first_layer in (0, 8, 31):
            with self.subTest(first_layer=first_layer):
                pool = object.__new__(SWAKVPool)
                pool.start_layer = 0
                pool.layer_num = 4
                pool.full_kv_pool = _LocalPool(3)
                pool.swa_kv_pool = _LocalPool(2)
                pool.layers_mapping = {
                    first_layer: (0, False),
                    first_layer + 2: (1, False),
                    first_layer + 1: (0, True),
                    first_layer + 3: (1, True),
                }
                groups = get_raiden_kv_caches(pool)
                self.assertEqual(groups.full, pool.full_kv_pool.buffers)
                self.assertEqual(groups.swa, pool.swa_kv_pool.buffers)
                self.assertEqual(groups.full_layer_ids, [first_layer, first_layer + 2])
                self.assertEqual(
                    groups.swa_layer_ids, [first_layer + 1, first_layer + 3]
                )

    def test_speculative_draft_caches_join_target_page_namespaces(self):
        class _TargetPool:
            start_layer = 0
            layer_num = 2
            layers_mapping = {0: (0, False), 1: (0, True)}
            buffers = [torch.empty(3, 4), torch.empty(2, 4)]

            def get_fused_kv_buffer(self, layer_id):
                return self.buffers[layer_id]

        class _DraftPool:
            start_layer = 0
            layer_num = 1
            buffers = [torch.empty(3, 4)]

            def get_fused_kv_buffer(self, layer_id):
                return self.buffers[layer_id]

        groups = get_raiden_kv_caches(_TargetPool(), draft_pool=_DraftPool())
        self.assertEqual(groups.full, [_TargetPool.buffers[0], _DraftPool.buffers[0]])
        self.assertEqual(groups.swa, [_TargetPool.buffers[1]])

    def test_hybrid_swa_max_blocks_is_sized_from_sliding_window(self):
        mgr = object.__new__(RaidenKVManager)
        mgr.attn_tp_rank = 0
        mgr._prefill = None
        mgr._decode = DecodeState()
        mgr._timeout_s = 120.0
        mgr._aborted = {}
        mgr._send_reads = {}
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr.request_status = {}
        mgr.kv_args = SimpleNamespace(page_size=128, sliding_window_size=4096)

        with patch.dict(
            os.environ,
            {"SGLANG_RAIDEN_SWA_MAX_BLOCKS": "0"},
        ):
            self.assertEqual(mgr._swa_max_request_blocks(), 33)

    def test_hybrid_swa_max_blocks_override_is_independent(self):
        mgr = object.__new__(RaidenKVManager)
        mgr.attn_tp_rank = 0
        mgr._prefill = None
        mgr._decode = DecodeState()
        mgr._timeout_s = 120.0
        mgr._aborted = {}
        mgr._send_reads = {}
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr.request_status = {}
        mgr.kv_args = SimpleNamespace(page_size=128, sliding_window_size=4096)

        with patch.dict(
            os.environ,
            {
                "SGLANG_RAIDEN_MAX_BLOCKS": "7",
                "SGLANG_RAIDEN_SWA_MAX_BLOCKS": "11",
            },
        ):
            self.assertEqual(mgr._swa_max_request_blocks(), 11)

    def test_symmetric_pull_rendezvous(self):
        class _Engine:
            def __init__(self):
                self.register_read = Mock()

        room = 17
        blocks = [3, 9]
        for finalize_first in (False, True):
            with self.subTest(finalize_first=finalize_first):
                mgr = object.__new__(RaidenKVManager)
                mgr.attn_tp_rank = 0
                mgr._prefill = PrefillState(announce_pages=1)
                mgr._decode = None
                mgr._timeout_s = 120.0
                mgr._aborted = {}
                mgr._send_reads = {}
                mgr.failure_records = {}
                mgr.failure_lock = threading.Lock()
                mgr.request_status = {}
                mgr._state_lock = threading.Lock()
                mgr._status_lock = threading.Lock()
                mgr._announce_lock = threading.Lock()
                mgr._read_leases = {}
                mgr._aborted = {}
                mgr._send_reads = {}
                mgr._prefill.decode_pages = {room: len(blocks)}
                mgr._prefill.decode_swa_pages = {room: 0}
                mgr._engine = _Engine()
                mgr._swa_engine = None
                mgr._swa_kv_caches = []
                mgr.swa_raiden_endpoint = None
                mgr.raiden_endpoint = "127.0.0.1:31020"
                socket = Mock()
                mgr._socket_send_locks = defaultdict(threading.Lock)
                mgr._socket_lock = threading.Lock()
                mgr._socket_cache = {}
                mgr._connect = Mock(
                    side_effect=lambda ep, is_ipv6=False: mgr._socket_cache.setdefault(
                        ep, socket
                    )
                )

                mgr.append_send_blocks(room, blocks)
                if finalize_first:
                    mgr.finalize_send(room, None)
                    mgr._engine.register_read.assert_not_called()
                    mgr._prefill.decode_addr[room] = ("127.0.0.1", 31002)
                    mgr._maybe_reply(room)
                else:
                    mgr._prefill.decode_addr[room] = ("127.0.0.1", 31002)
                    mgr._maybe_reply(room)
                    mgr._engine.register_read.assert_not_called()
                    mgr.finalize_send(room, None)

                mgr._engine.register_read.assert_called_once_with(
                    str(room), room, blocks
                )

    def test_hybrid_swa_rendezvous_registers_both_engines(self):
        room = 23
        full_blocks = [3, 9]
        swa_blocks = [5, 7]
        mgr = object.__new__(RaidenKVManager)
        mgr.attn_tp_rank = 0
        mgr._prefill = PrefillState(announce_pages=1)
        mgr._decode = None
        mgr._timeout_s = 120.0
        mgr._aborted = {}
        mgr._send_reads = {}
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr.request_status = {}
        mgr.kv_args = Mock(state_types=[StateType.SWA])
        mgr._state_lock = threading.Lock()
        mgr._status_lock = threading.Lock()
        mgr._announce_lock = threading.Lock()
        mgr._read_leases = {}
        mgr._aborted = {}
        mgr._send_reads = {}
        mgr._prefill.decode_pages = {room: len(full_blocks)}
        mgr._prefill.decode_swa_pages = {room: len(swa_blocks)}
        mgr._engine = Mock()
        mgr._swa_engine = Mock()
        mgr._swa_kv_caches = [object()]
        mgr.raiden_endpoint = "127.0.0.1:31020"
        mgr.swa_raiden_endpoint = "127.0.0.1:31022"
        mgr._completed_send = set()
        mgr._send_done_at = {}
        socket = Mock()
        mgr._socket_send_locks = defaultdict(threading.Lock)
        mgr._socket_lock = threading.Lock()
        mgr._socket_cache = {}
        mgr._connect = Mock(
            side_effect=lambda ep, is_ipv6=False: mgr._socket_cache.setdefault(
                ep, socket
            )
        )

        swa_from_state = mgr._swa_blocks_from_state_indices([swa_blocks])
        mgr.append_send_blocks(room, full_blocks, swa_from_state)
        mgr._prefill.decode_addr[room] = ("127.0.0.1", 31002)
        mgr.finalize_send(room, None)

        mgr._engine.register_read.assert_called_once_with(str(room), room, full_blocks)
        mgr._swa_engine.register_read.assert_called_once_with(
            mgr._swa_req_id(room), room, swa_blocks
        )
        payload = socket.send_multipart.call_args.args[0]
        response = Blocks.from_frames(payload)
        self.assertEqual(response.swa_endpoint, "127.0.0.1:31022")

        self.assertFalse(mgr.is_send_done(room))
        mgr._completed_send.update((str(room), mgr._swa_req_id(room)))
        self.assertTrue(mgr.is_send_done(room))

    def test_hybrid_swa_decode_starts_both_pulls_with_separate_pages(self):
        room = 29
        mgr = object.__new__(RaidenKVManager)
        mgr.attn_tp_rank = 0
        mgr._prefill = None
        mgr._decode = DecodeState()
        mgr._timeout_s = 120.0
        mgr._aborted = {}
        mgr._send_reads = {}
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr.request_status = {}
        mgr._state_lock = threading.Lock()
        mgr._status_lock = threading.Lock()
        mgr._engine = Mock()
        mgr._swa_engine = Mock()
        mgr._dynamic_host_staging = True
        mgr._pulling = StagingAdmission(2, 1)
        mgr._full_engine_max_blocks = 2
        mgr._swa_engine_max_blocks = 1
        mgr._decode.local_blocks = {room: [8, 9]}
        mgr._decode.local_swa_blocks = {room: [7]}
        mgr._decode.remote_blocks = {room: [1, 2, 3]}
        mgr._decode.remote_swa_blocks = {room: [4, 5]}
        mgr._decode.remote_endpoint = {room: "127.0.0.1:31020"}
        mgr._decode.remote_swa_endpoint = {room: "127.0.0.1:31022"}
        mgr.chunked_transfer = False
        mgr._failed = set()
        mgr._native_failed = set()

        self.assertTrue(mgr.maybe_start_pull(room))
        mgr._engine.start_read.assert_called_once_with(
            str(room), room, "127.0.0.1:31020", [2, 3], [8, 9]
        )
        mgr._swa_engine.start_read.assert_called_once_with(
            mgr._swa_req_id(room), room, "127.0.0.1:31022", [5], [7]
        )

    def _make_hybrid_swa_decode_manager(
        self,
        *,
        full_pages,
        swa_pages,
        full_max_blocks=128,
        swa_max_blocks=33,
    ):
        room = 29
        mgr = object.__new__(RaidenKVManager)
        mgr._prefill = None
        mgr._decode = DecodeState()
        mgr._timeout_s = 120.0
        mgr.attn_tp_rank = 0
        mgr.chunked_transfer = False
        mgr._state_lock = threading.Lock()
        mgr._engine = Mock()
        mgr._swa_engine = Mock()
        mgr._dynamic_host_staging = True
        mgr._pulling = StagingAdmission(full_max_blocks, swa_max_blocks)
        mgr._pulling_cleared = set()
        mgr._aborted = {}
        mgr._send_reads = {}
        mgr._status_lock = threading.Lock()
        mgr.request_status = {}
        mgr._failed_reasons = {}
        mgr._full_engine_max_blocks = full_max_blocks
        mgr._swa_engine_max_blocks = swa_max_blocks
        mgr._completed_recv = set()
        mgr._failed = set()
        mgr._native_failed = set()
        mgr.record_failure = Mock()
        mgr._decode.local_blocks = {room: list(range(1000, 1000 + full_pages))}
        mgr._decode.local_swa_blocks = {room: list(range(2000, 2000 + swa_pages))}
        mgr._decode.remote_blocks = {room: list(range(full_pages))}
        mgr._decode.remote_swa_blocks = {room: list(range(swa_pages))}
        mgr._decode.remote_endpoint = {room: "127.0.0.1:31020"}
        mgr._decode.remote_swa_endpoint = {room: "127.0.0.1:31022"}
        return room, mgr

    def test_hybrid_swa_decode_accepts_full_request_larger_than_swa_window(self):
        room, mgr = self._make_hybrid_swa_decode_manager(
            full_pages=128,
            swa_pages=33,
        )

        self.assertTrue(mgr.maybe_start_pull(room))
        mgr._engine.start_read.assert_called_once_with(
            str(room),
            room,
            "127.0.0.1:31020",
            list(range(128)),
            list(range(1000, 1128)),
        )
        mgr._swa_engine.start_read.assert_called_once_with(
            mgr._swa_req_id(room),
            room,
            "127.0.0.1:31022",
            list(range(33)),
            list(range(2000, 2033)),
        )

    def test_hybrid_swa_decode_rejects_oversized_full_attention_pull(self):
        room, mgr = self._make_hybrid_swa_decode_manager(
            full_pages=129,
            swa_pages=33,
        )

        self.assertFalse(mgr.maybe_start_pull(room))
        reason = mgr.record_failure.call_args.args[1]
        self.assertIn("full-attention", reason)
        self.assertIn("129 pages", reason)
        self.assertIn("holds 128", reason)
        self.assertIn("SGLANG_RAIDEN_MAX_BLOCKS", reason)
        mgr._engine.start_read.assert_not_called()
        mgr._swa_engine.start_read.assert_not_called()

    def test_hybrid_swa_decode_rejects_oversized_swa_pull(self):
        room, mgr = self._make_hybrid_swa_decode_manager(
            full_pages=128,
            swa_pages=34,
        )

        self.assertFalse(mgr.maybe_start_pull(room))
        reason = mgr.record_failure.call_args.args[1]
        self.assertIn("SWA", reason)
        self.assertIn("34 pages", reason)
        self.assertIn("holds 33", reason)
        self.assertIn("SGLANG_RAIDEN_SWA_MAX_BLOCKS", reason)
        mgr._engine.start_read.assert_not_called()
        mgr._swa_engine.start_read.assert_not_called()

    def test_decode_defers_pull_until_the_pool_has_room(self):
        first, second = 29, 30
        mgr = object.__new__(RaidenKVManager)
        mgr.attn_tp_rank = 0
        mgr._prefill = None
        mgr._decode = DecodeState()
        mgr._timeout_s = 120.0
        mgr._aborted = {}
        mgr._send_reads = {}
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr.request_status = {}
        mgr.chunked_transfer = False
        mgr._state_lock = threading.Lock()
        mgr._status_lock = threading.Lock()
        mgr._engine = Mock()
        mgr._swa_engine = None
        mgr._dynamic_host_staging = True
        mgr._pulling = StagingAdmission(1)
        mgr._pulling_cleared = set()
        mgr._full_engine_max_blocks = 4
        mgr._completed_recv = set()
        mgr._failed = set()
        mgr._native_failed = set()
        mgr._decode.local_blocks = {first: [8], second: [9]}
        mgr._decode.remote_blocks = {first: [1], second: [2]}
        mgr._decode.remote_endpoint = {r: "127.0.0.1:31020" for r in (first, second)}

        self.assertTrue(mgr.maybe_start_pull(first))
        self.assertFalse(mgr.maybe_start_pull(second))
        self.assertEqual(mgr._engine.start_read.call_count, 1)

        mgr._completed_recv.add(str(first))
        mgr._release_pull_slots_locked()
        self.assertTrue(mgr.maybe_start_pull(second))
        mgr._engine.start_read.assert_called_with(
            str(second), second, "127.0.0.1:31020", [2], [9]
        )

    @staticmethod
    def _decode_mgr(pool_pages, full_limit, swa_limit=None, swa_pool_pages=None):
        mgr = object.__new__(RaidenKVManager)
        mgr._prefill = None
        mgr._decode = DecodeState()
        mgr._aborted = {}
        mgr._send_reads = {}
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr.request_status = {}
        mgr.chunked_transfer = False
        mgr._state_lock = threading.Lock()
        mgr._status_lock = threading.Lock()
        mgr._engine = Mock()
        mgr._swa_engine = Mock() if swa_limit is not None else None
        mgr._dynamic_host_staging = True
        mgr._pulling = StagingAdmission(pool_pages, swa_pool_pages)
        mgr._timeout_s = 120.0
        mgr._pulling_cleared = set()
        mgr._full_engine_max_blocks = full_limit
        mgr._swa_engine_max_blocks = swa_limit or 0
        mgr._completed_send = set()
        mgr._send_done_at = {}
        mgr._completed_recv = set()
        mgr._failed = set()
        mgr._native_failed = set()
        mgr.attn_tp_rank = 0
        mgr.record_failure = Mock()
        # clear_room sweeps the reshard bookkeeping too.
        mgr._reshard_slot_of = {}
        mgr._reshard_slot_rooms = [None]
        return mgr

    @staticmethod
    def _announce(mgr, room, full_pages, swa_pages=0):
        mgr._decode.local_blocks[room] = list(range(100, 100 + full_pages))
        mgr._decode.remote_blocks[room] = list(range(full_pages))
        mgr._decode.remote_endpoint[room] = "127.0.0.1:31020"
        if mgr._swa_engine is not None:
            mgr._decode.local_swa_blocks[room] = list(range(200, 200 + swa_pages))
            mgr._decode.remote_swa_blocks[room] = list(range(swa_pages))
            mgr._decode.remote_swa_endpoint[room] = "127.0.0.1:31022"

    def test_hybrid_swa_checks_each_pool_against_its_own_page_limit(self):
        # gpt-oss at page 64: a 128-token window is 3 pages, the full pool 128.
        room = 31
        mgr = self._decode_mgr(
            pool_pages=512, full_limit=128, swa_limit=3, swa_pool_pages=12
        )
        self._announce(mgr, room, full_pages=40, swa_pages=3)

        self.assertTrue(mgr.maybe_start_pull(room))
        mgr._engine.start_read.assert_called_once()
        mgr._swa_engine.start_read.assert_called_once()
        mgr.record_failure.assert_not_called()

        too_wide = 32
        self._announce(mgr, too_wide, full_pages=4, swa_pages=5)
        self.assertFalse(mgr.maybe_start_pull(too_wide))
        self.assertIn(str(too_wide), mgr._failed)
        self.assertIn(
            "SGLANG_RAIDEN_SWA_MAX_BLOCKS", mgr.record_failure.call_args.args[1]
        )
        self.assertEqual(mgr._engine.start_read.call_count, 1)

    def test_cleared_room_keeps_its_staging_until_the_read_settles(self):
        first, second = 33, 34
        mgr = self._decode_mgr(pool_pages=2, full_limit=8)
        self._announce(mgr, first, full_pages=2)
        self._announce(mgr, second, full_pages=2)
        mgr._read_leases = {}

        self.assertTrue(mgr.maybe_start_pull(first))
        # The framework aborts the request while raiden still holds the read.
        mgr.clear_room(first)
        self.assertFalse(mgr.maybe_start_pull(second))

        mgr._completed_recv.add(str(first))
        with mgr._state_lock:
            mgr._release_pull_slots_locked()
        self.assertTrue(mgr.maybe_start_pull(second))
        self.assertNotIn(str(first), mgr._completed_recv)
        self.assertFalse(mgr._pulling_cleared)

    def test_cleared_settled_room_releases_its_staging_at_once(self):
        first, second = 35, 36
        mgr = self._decode_mgr(pool_pages=2, full_limit=8)
        self._announce(mgr, first, full_pages=2)
        self._announce(mgr, second, full_pages=2)
        mgr._read_leases = {}

        self.assertTrue(mgr.maybe_start_pull(first))
        mgr._completed_recv.add(str(first))
        mgr.clear_room(first)
        self.assertTrue(mgr.maybe_start_pull(second))
        self.assertNotIn(str(first), mgr._completed_recv)

    def test_hybrid_swa_staging_is_held_until_both_engines_settle(self):
        first, second = 37, 38
        mgr = self._decode_mgr(
            pool_pages=2, full_limit=8, swa_limit=3, swa_pool_pages=1
        )
        self._announce(mgr, first, full_pages=2, swa_pages=1)
        self._announce(mgr, second, full_pages=2, swa_pages=1)

        self.assertTrue(mgr.maybe_start_pull(first))
        # The full-attention read fails while the sliding-window read is
        # still in flight on the other engine.
        mgr._failed.add(str(first))
        mgr._native_failed.add(str(first))
        with mgr._state_lock:
            mgr._release_pull_slots_locked()
        self.assertFalse(mgr.maybe_start_pull(second))

        mgr._completed_recv.add(mgr._swa_req_id(first))
        with mgr._state_lock:
            mgr._release_pull_slots_locked()
        self.assertTrue(mgr.maybe_start_pull(second))

    def test_pool_seats_reads_by_their_pages(self):
        # Four 2-page reads fit a pool sized for one 8-page read; a fifth
        # waits for pages to free.
        rooms = [50, 51, 52, 53, 54]
        mgr = self._decode_mgr(pool_pages=8, full_limit=8)
        for room in rooms:
            self._announce(mgr, room, full_pages=2)

        self.assertTrue(all(mgr.maybe_start_pull(room) for room in rooms[:4]))
        self.assertFalse(mgr.maybe_start_pull(rooms[4]))
        self.assertEqual(mgr._engine.start_read.call_count, 4)

        mgr._completed_recv.add(str(rooms[0]))
        with mgr._state_lock:
            mgr._release_pull_slots_locked()
        self.assertTrue(mgr.maybe_start_pull(rooms[4]))

    def test_fixed_slots_seat_one_read_per_slot(self):
        first, second = 55, 56
        mgr = self._decode_mgr(pool_pages=8, full_limit=8)
        mgr._dynamic_host_staging = False
        self._announce(mgr, first, full_pages=2)
        self._announce(mgr, second, full_pages=2)

        self.assertTrue(mgr.maybe_start_pull(first))
        self.assertFalse(mgr.maybe_start_pull(second))

    def test_a_waiting_read_is_not_starved_by_smaller_ones(self):
        holder, big, small = 57, 58, 59
        mgr = self._decode_mgr(pool_pages=8, full_limit=8)
        self._announce(mgr, holder, full_pages=6)
        self._announce(mgr, big, full_pages=4)
        self._announce(mgr, small, full_pages=2)

        self.assertTrue(mgr.maybe_start_pull(holder))
        self.assertFalse(mgr.maybe_start_pull(big))
        # Two pages are free, but the smaller read waits behind the larger.
        self.assertFalse(mgr.maybe_start_pull(small))

        mgr._completed_recv.add(str(holder))
        with mgr._state_lock:
            mgr._release_pull_slots_locked()
        self.assertTrue(mgr.maybe_start_pull(big))
        self.assertTrue(mgr.maybe_start_pull(small))

    def test_clearing_a_waiting_room_frees_the_line(self):
        holder, big, small = 60, 61, 62
        mgr = self._decode_mgr(pool_pages=8, full_limit=8)
        mgr._read_leases = {}
        self._announce(mgr, holder, full_pages=6)
        self._announce(mgr, big, full_pages=4)
        self._announce(mgr, small, full_pages=2)

        self.assertTrue(mgr.maybe_start_pull(holder))
        self.assertFalse(mgr.maybe_start_pull(big))
        # The framework drops the waiting request; the smaller read may go.
        mgr.clear_room(big)
        self.assertTrue(mgr.maybe_start_pull(small))

    @staticmethod
    def _reshard_decode_mgr():
        mgr = object.__new__(RaidenKVManager)
        mgr.attn_tp_rank = 0
        mgr._prefill = None
        mgr._decode = DecodeState()
        mgr._timeout_s = 120.0
        mgr._aborted = {}
        mgr._send_reads = {}
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr.request_status = {}
        mgr._state_lock = threading.RLock()
        mgr._completed_recv = set()
        mgr._swa_engine = Mock()
        mgr._dynamic_host_staging = True
        mgr.update_status = Mock()
        return mgr

    def test_hybrid_swa_reshard_forwards_its_window_pages(self):
        """Asymmetric TP used to reject hybrid models outright. It now forwards
        the sliding-window pages too, so the second engine can be given its own
        reshard plan -- without them the window would never reach decode."""
        mgr = self._reshard_decode_mgr()
        mgr._register_reshard_request = Mock()

        mgr.register_decode_request(
            room=31,
            local_blocks=[8, 9],
            local_swa_blocks=[6, 7],
            aux_index=None,
            bootstrap_infos=[{}, {}],
        )

        mgr._register_reshard_request.assert_called_once_with(
            31, [8, 9], [6, 7], None, [{}, {}]
        )

    def test_hybrid_swa_reshard_prefix_hit_settles_both_engines(self):
        """A full prefix hit receives nothing on either engine. Settling only
        the full-attention id would hang is_recv_done, which requires the
        sliding-window id too whenever a second engine exists."""
        mgr = self._reshard_decode_mgr()
        mgr._register_reshard_request = Mock()

        mgr.register_decode_request(
            room=31,
            local_blocks=[],
            local_swa_blocks=[6, 7],
            aux_index=None,
            bootstrap_infos=[{}, {}],
        )

        mgr._register_reshard_request.assert_not_called()
        self.assertIn("31", mgr._completed_recv)
        self.assertIn(mgr._swa_req_id(31), mgr._completed_recv)

    def test_alloc_returns_page_aligned_token_runs(self):
        pool = _make_host_pool(page_size=4, page_num=3)
        idx = pool.alloc(8)
        self.assertEqual(idx.tolist(), [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(pool._to_page_ids(idx), [0, 1])
        self.assertEqual(pool.available_size(), 4)  # 1 page of 4 left

    def test_alloc_requires_page_multiple(self):
        pool = _make_host_pool(page_size=4, page_num=3)
        with self.assertRaises(AssertionError):
            pool.alloc(6)

    def test_alloc_returns_none_when_full(self):
        pool = _make_host_pool(page_size=4, page_num=3)
        self.assertIsNone(pool.alloc(16))  # 4 pages requested, only 3 exist

    def test_free_recycles_pages(self):
        pool = _make_host_pool(page_size=4, page_num=3)
        idx = pool.alloc(8)
        self.assertEqual(pool.available_size(), 4)
        pool.free(idx)
        self.assertEqual(pool.available_size(), 12)
        # A fresh full allocation must now succeed again.
        self.assertIsNotNone(pool.alloc(12))

    def test_to_page_ids_after_fragmentation(self):
        # Free a middle page, re-alloc, and confirm page ids stay whole/aligned.
        pool = _make_host_pool(page_size=4, page_num=3)
        a = pool.alloc(4)  # page 0
        b = pool.alloc(4)  # page 1
        pool.free(a)  # page 0 back to the tail of the free list
        c = pool.alloc(4)  # page 2 (front of free list), then page 0 next
        self.assertEqual(pool._to_page_ids(b), [1])
        self.assertEqual(pool._to_page_ids(c), [2])

    def test_backup_converts_token_indices_to_page_ids(self):
        pool = _make_host_pool(page_size=4, page_num=8)
        host_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64)
        device_idx = torch.tensor([8, 9, 10, 11, 20, 21, 22, 23], dtype=torch.int64)
        ev = pool.backup_from_device_all_layer(None, host_idx, device_idx, "raiden")
        self.assertTrue(ev.query())
        dev_pages, host_pages = pool.transfer.backups[0]
        self.assertEqual(dev_pages, [2, 5])  # 8//4, 20//4
        self.assertEqual(host_pages, [0, 1])

    def test_load_all_layer_converts_page_ids(self):
        pool = _make_host_pool(page_size=4, page_num=8)
        host_idx = torch.tensor([12, 13, 14, 15], dtype=torch.int64)
        device_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        pool.load_all_layer(None, host_idx, device_idx)
        host_pages, dev_pages = pool.transfer.loads[0]
        self.assertEqual(host_pages, [3])
        self.assertEqual(dev_pages, [0])

    def test_per_layer_load_is_rejected(self):
        pool = _make_host_pool()
        with self.assertRaises(NotImplementedError):
            pool.load_to_device_per_layer(None, None, None, 0, "raiden")

    def test_l3_seams_raise(self):
        pool = _make_host_pool()
        with self.assertRaises(NotImplementedError):
            pool.get_data_page(0)
        with self.assertRaises(NotImplementedError):
            pool.get_dummy_flat_data_page()
        with self.assertRaises(NotImplementedError):
            pool.set_from_flat_data_page(0, torch.empty(0))


class TestRaidenChunkedTransfer(CustomTestCase):
    """Chunk announcements and pulls on the symmetric pull path."""

    @staticmethod
    def _prefill():
        mgr = object.__new__(RaidenKVManager)
        mgr.attn_tp_rank = 0
        mgr._prefill = PrefillState(announce_pages=1)
        mgr._decode = None
        mgr._timeout_s = 120.0
        mgr._aborted = {}
        mgr._send_reads = {}
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr.request_status = {}
        mgr._state_lock = threading.Lock()
        mgr._status_lock = threading.Lock()
        mgr._announce_lock = threading.Lock()
        mgr._engine = Mock()
        mgr._engine.register_read = Mock(return_value=True)
        mgr._swa_engine = None
        mgr._local_rank = 3
        mgr._uuid_seq = 0
        mgr._read_leases = {}
        mgr._prefill.announce_pages = 16
        mgr.raiden_endpoint = "127.0.0.1:31020"
        mgr._socket = Mock()
        mgr._socket_send_locks = collections.defaultdict(threading.Lock)
        mgr._socket_lock = threading.Lock()
        mgr._socket_cache = {}
        mgr._connect = Mock(
            side_effect=lambda ep, is_ipv6=False: mgr._socket_cache.setdefault(
                ep, mgr._socket
            )
        )
        return mgr

    @staticmethod
    def _pull_req(room, decode_pages):
        return PullReq(room, "10.0.0.2", 31002, decode_pages).to_frames()

    def test_prefill_announces_only_the_pages_the_decode_will_pull(self):
        room = 31
        mgr = self._prefill()
        mgr._on_pull_req(self._pull_req(room, 40))
        self.assertEqual(mgr._prefill.decode_pages[room], 40)

        # 64 prefill pages; the decode already holds the first 24.
        blocks = list(range(100, 164))
        mgr.offer_chunk(room, 0, blocks[0:16], 64)
        mgr._engine.register_read.assert_not_called()
        mgr._socket.send_multipart.assert_not_called()

        mgr.offer_chunk(room, 16, blocks[16:32], 64)
        mgr._engine.register_read.assert_called_once()
        self.assertEqual(mgr._engine.register_read.call_args.args[2], blocks[24:32])
        msg = mgr._socket.send_multipart.call_args.args[0]
        chunk = Chunk.from_frames(msg)
        self.assertEqual(chunk.start, 24)
        self.assertEqual(chunk.total_pages, 64)
        self.assertEqual(chunk.blocks, blocks[24:32])

        mgr.offer_chunk(room, 32, blocks[32:48], 64)
        self.assertEqual(mgr._engine.register_read.call_count, 2)
        self.assertEqual(mgr._engine.register_read.call_args.args[2], blocks[32:48])

    def test_small_chunks_gather_into_one_announcement(self):
        room = 34
        mgr = self._prefill()
        mgr._on_pull_req(self._pull_req(room, 64))
        blocks = list(range(100, 164))
        mgr.offer_chunk(room, 0, blocks[0:8], 64)
        mgr._engine.register_read.assert_not_called()
        mgr.offer_chunk(room, 8, blocks[8:16], 64)
        mgr._engine.register_read.assert_called_once()
        self.assertEqual(mgr._engine.register_read.call_args.args[2], blocks[0:16])
        msg = mgr._socket.send_multipart.call_args.args[0]
        self.assertEqual(Chunk.from_frames(msg).start, 0)
        self.assertNotIn(room, mgr._prefill.pending_offer)

    def test_chunks_offered_before_the_pull_request_are_held_then_filtered(self):
        room = 32
        mgr = self._prefill()
        blocks = list(range(100, 164))
        mgr.offer_chunk(room, 0, blocks[0:16], 64)
        mgr.offer_chunk(room, 16, blocks[16:32], 64)
        mgr._engine.register_read.assert_not_called()

        mgr._on_pull_req(self._pull_req(room, 40))
        self.assertEqual(mgr._engine.register_read.call_count, 1)
        self.assertEqual(mgr._engine.register_read.call_args.args[2], blocks[24:32])
        self.assertNotIn(room, mgr._prefill.held_chunks)

    @staticmethod
    def _decode(room, local, pool_pages, swa_pool_pages=8):
        mgr = object.__new__(RaidenKVManager)
        mgr._prefill = None
        mgr._decode = DecodeState()
        mgr._aborted = {}
        mgr._send_reads = {}
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr.request_status = {}
        mgr._state_lock = threading.Lock()
        mgr._status_lock = threading.Lock()
        mgr._engine = Mock()
        mgr._engine.start_read = Mock(return_value=0)
        mgr._swa_engine = None
        mgr.chunked_transfer = True
        mgr._dynamic_host_staging = True
        mgr._pulling = StagingAdmission(pool_pages, swa_pool_pages)
        mgr._full_engine_max_blocks = 64
        mgr._swa_engine_max_blocks = 2
        mgr.attn_tp_rank = 0
        mgr.record_failure = Mock()
        mgr._decode.local_blocks = {room: local}
        mgr._decode.remote_total = {room: 64}
        mgr._timeout_s = 120.0
        mgr._completed_send = set()
        mgr._send_done_at = {}
        mgr._completed_recv = set()
        mgr._failed = set()
        mgr._native_failed = set()
        mgr._pulling_cleared = set()
        mgr._cleared_plans = {}
        mgr._stale_plans = []
        mgr._engine_poll_failing_since = {}
        # clear_room sweeps the reshard bookkeeping too.
        mgr._reshard_slot_of = {}
        mgr._reshard_slot_rooms = [None]
        return mgr

    def test_announced_and_closing_frames_preserve_page_pairs_across_prefix_hits(self):
        for prefix_pages in (0, 8, 24, 48):
            for offer_before_pull in (False, True):
                with self.subTest(prefix=prefix_pages, offer_first=offer_before_pull):
                    room = 61
                    # Raiden does not publish decode_prefix_len to the sender:
                    # announcements span the prompt, but decode owns its suffix.
                    remote = list(range(100, 164))
                    local = list(range(200, 264 - prefix_pages))
                    prefill = self._prefill()
                    decode = self._decode(room, local, pool_pages=64)
                    pull = self._pull_req(room, len(local))
                    if not offer_before_pull:
                        prefill._on_pull_req(pull)
                    for start in range(0, len(remote) - 16, 16):
                        prefill.offer_chunk(
                            room, start, remote[start : start + 16], len(remote)
                        )
                    if offer_before_pull:
                        prefill._on_pull_req(pull)
                    for call in prefill._socket.send_multipart.call_args_list:
                        self.assertGreaterEqual(
                            Chunk.from_frames(call.args[0]).start, prefix_pages
                        )
                        decode._on_chunk(call.args[0])
                    decode._on_blocks(
                        Blocks(
                            room, remote, prefill.raiden_endpoint, [], "", b""
                        ).to_frames()
                    )
                    self.assertFalse(decode.is_recv_done(room))
                    # Complete each native read explicitly; no sleeps or timing races.
                    for _ in range(4):
                        decode.maybe_start_pull(room)
                        for req_id in decode._decode.chunk_reads.get(room, []):
                            decode._completed_recv.add(req_id)
                        with decode._state_lock:
                            decode._release_pull_slots_locked()
                    pairs = [
                        pair
                        for call in decode._engine.start_read.call_args_list
                        for pair in zip(call.args[3], call.args[4])
                    ]
                    self.assertEqual(pairs, list(zip(remote[prefix_pages:], local)))
                    self.assertTrue(decode.is_recv_done(room))
                    self.assertEqual(len(decode._pulling), 0)

    def test_closing_read_carries_the_sliding_window_pages(self):
        room = 35
        local = list(range(200, 264))
        remote = list(range(100, 164))
        endpoint = "127.0.0.1:31020"
        mgr = self._decode(room, local, pool_pages=256)
        mgr._swa_engine = Mock()
        mgr._decode.remote_swa_blocks = {room: [4, 5]}
        mgr._decode.local_swa_blocks = {room: [7]}
        mgr._decode.remote_swa_endpoint = {room: "127.0.0.1:31022"}
        mgr._decode.final_seen.add(room)
        mgr._decode.chunk_queue[room] = [
            (str(room), room, list(range(48, 48 + 16)), remote[48:64], endpoint)
        ]

        self.assertTrue(mgr.maybe_start_pull(room))
        mgr._engine.start_read.assert_called_once_with(
            "35", room, endpoint, remote[48:64], local[48:64]
        )
        mgr._swa_engine.start_read.assert_called_once_with(
            "swa:35", room, "127.0.0.1:31022", [5], [7]
        )
        mgr._completed_recv.add("35")
        self.assertFalse(mgr.is_recv_done(room))
        mgr._completed_recv.add("swa:35")
        self.assertTrue(mgr.is_recv_done(room))

    def test_decode_keeps_one_chunk_read_in_flight_per_request(self):
        room = 33
        local = list(range(200, 264))
        remote = list(range(100, 164))
        endpoint = "127.0.0.1:31020"
        mgr = self._decode(room, local, pool_pages=256)
        mgr._decode.chunk_queue[room] = [
            ("33#0", 11, list(range(0, 0 + 16)), remote[0:16], endpoint),
            ("33#1", 12, list(range(16, 16 + 16)), remote[16:32], endpoint),
        ]

        self.assertTrue(mgr.maybe_start_pull(room))
        mgr._engine.start_read.assert_called_once_with(
            "33#0", 11, endpoint, remote[0:16], local[0:16]
        )
        # The second chunk waits for the first read even though slots are free.
        self.assertTrue(mgr.maybe_start_pull(room))
        self.assertEqual(mgr._engine.start_read.call_count, 1)
        self.assertEqual(len(mgr._pulling), 1)
        self.assertFalse(mgr.is_recv_done(room))

        with mgr._state_lock:
            mgr._completed_recv.add("33#0")
            mgr._release_pull_slots_locked()
        self.assertTrue(mgr.maybe_start_pull(room))
        self.assertEqual(mgr._engine.start_read.call_count, 2)
        self.assertEqual(mgr._engine.start_read.call_args.args[0], "33#1")
        self.assertEqual(mgr._engine.start_read.call_args.args[4], local[16:32])
        self.assertEqual(len(mgr._pulling), 1)

    def test_oversized_chunk_fails_the_request(self):
        room = 43
        local = list(range(200, 264))
        remote = list(range(100, 164))
        mgr = self._decode(room, local, pool_pages=256)
        mgr._full_engine_max_blocks = 8
        mgr._decode.chunk_queue[room] = [
            ("43#0", 11, list(range(0, 0 + 16)), remote[0:16], "127.0.0.1:31020")
        ]

        self.assertFalse(mgr.maybe_start_pull(room))
        mgr._engine.start_read.assert_not_called()
        self.assertTrue(mgr.is_failed(room))
        self.assertIn("SGLANG_RAIDEN_MAX_BLOCKS", mgr.record_failure.call_args.args[1])
        self.assertIn("16 pages", mgr.record_failure.call_args.args[1])

    def test_closing_read_checks_the_sliding_window_pages_against_their_limit(self):
        room = 44
        local = list(range(200, 264))
        remote = list(range(100, 164))
        mgr = self._decode(room, local, pool_pages=256)
        mgr._swa_engine = Mock()
        mgr._decode.remote_swa_blocks = {room: [4, 5, 6]}
        mgr._decode.local_swa_blocks = {room: [7, 8, 9]}
        mgr._decode.remote_swa_endpoint = {room: "127.0.0.1:31022"}
        mgr._decode.chunk_queue[room] = [
            (
                str(room),
                room,
                list(range(48, 48 + 16)),
                remote[48:64],
                "127.0.0.1:31020",
            )
        ]

        self.assertFalse(mgr.maybe_start_pull(room))
        mgr._engine.start_read.assert_not_called()
        mgr._swa_engine.start_read.assert_not_called()
        self.assertTrue(mgr.is_failed(room))
        self.assertIn(
            "SGLANG_RAIDEN_SWA_MAX_BLOCKS", mgr.record_failure.call_args.args[1]
        )

    def test_announced_reach_is_the_furthest_chunk_end(self):
        room = 45
        remote = list(range(100, 164))
        mgr = self._decode(room, list(range(200, 264)), pool_pages=256)

        def chunk(req_id, start, blocks):
            return Chunk(
                room, req_id, 11, start, 64, blocks, "127.0.0.1:31020"
            ).to_frames()

        # The second chunk overtakes the first on the wire.
        mgr._on_chunk(chunk("45#1", 16, remote[16:32]))
        mgr._on_chunk(chunk("45#0", 0, remote[0:16]))
        self.assertEqual(sorted(mgr._decode.announced_spans[room]), [(0, 16), (16, 32)])

    def test_closing_read_covers_a_chunk_lost_mid_range(self):
        # Chunk 0:16 lost its announcement race while 16:32 got through; the
        # closing read must pull 0:16 and 32:64, not just the tail.
        room = 46
        remote = list(range(100, 164))
        mgr = self._decode(room, list(range(200, 264)), pool_pages=256)
        mgr._decode.remote_blocks = {}
        mgr._decode.remote_endpoint = {}
        mgr._decode.remote_swa_blocks = {}
        mgr._decode.remote_swa_endpoint = {}
        mgr._decode.decode_aux_index = {}
        mgr._on_chunk(
            Chunk(
                room, "46#1", 11, 16, 64, remote[16:32], "127.0.0.1:31020"
            ).to_frames()
        )
        mgr._on_blocks(Blocks(room, remote, "127.0.0.1:31020", [], "", b"").to_frames())
        req_id, uuid, pages, blocks, _ = mgr._decode.chunk_queue[room][-1]
        self.assertEqual(req_id, str(room))
        self.assertEqual(pages, list(range(0, 16)) + list(range(32, 64)))
        self.assertEqual(blocks, remote[0:16] + remote[32:64])

    def test_discarded_waiter_does_not_block_closing_read(self):
        room = 48
        mgr = self._decode(room, list(range(200, 208)), pool_pages=8)
        mgr._decode.remote_blocks = {}
        mgr._decode.remote_endpoint = {}
        mgr._decode.remote_swa_blocks = {}
        mgr._decode.remote_swa_endpoint = {}
        mgr._decode.decode_aux_index = {}

        # Leave only two pages free, so the four-page chunk becomes the
        # head-of-line waiter.
        self.assertTrue(mgr._pulling.admit("holder", 6))
        mgr._on_chunk(
            Chunk(room, "48#0", 48, 0, 8, list(range(4)), "127.0.0.1:31020").to_frames()
        )
        self.assertFalse(mgr.maybe_start_pull(room))
        self.assertEqual(mgr._pulling.waiting, "48#0")

        # The final metadata moves the decode's retained suffix past that
        # chunk and queues a replacement closing read. BLOCKS itself does not
        # release staging capacity: the holder still owns its six pages.
        mgr._on_blocks(
            Blocks(room, list(range(20)), "127.0.0.1:31020", [], "", b"").to_frames()
        )
        self.assertEqual(mgr._pulling.pages(), 6)
        self.assertFalse(mgr.maybe_start_pull(room))
        self.assertEqual(mgr._pulling.pages(), 6)

        # Once the holder completes, all eight pages are available. The
        # discarded chunk must not remain as the head-of-line waiter.
        mgr._pulling.release("holder")
        self.assertEqual(mgr._pulling.pages(), 0)
        self.assertTrue(mgr.maybe_start_pull(room))

    def test_room_failure_does_not_settle_its_closing_read(self):
        # An abort shares its id with the closing read; only the engine's own
        # report may release the pages the read still writes.
        room = 47
        mgr = self._decode(room, list(range(200, 264)), pool_pages=256)
        mgr._pulling.admit(str(room), 16)
        with mgr._state_lock:
            mgr._fail_room_locked(room, "aborted mid-read")
        self.assertTrue(mgr.room_has_active_io(room))
        with mgr._state_lock:
            mgr._native_failed.add(str(room))
        self.assertFalse(mgr.room_has_active_io(room))

    def test_chunks_of_several_requests_share_the_pool_by_pages(self):
        rooms = [60, 61, 62]
        local = list(range(200, 264))
        remote = list(range(100, 164))
        endpoint = "127.0.0.1:31020"
        mgr = self._decode(rooms[0], local, pool_pages=32)
        for room in rooms:
            mgr._decode.local_blocks[room] = local
            mgr._decode.remote_total[room] = 64
            mgr._decode.chunk_queue[room] = [
                (f"{room}#0", room, list(range(0, 16)), remote[0:16], endpoint)
            ]

        # A 32-page pool seats two 16-page chunks; the third request waits.
        self.assertTrue(mgr.maybe_start_pull(rooms[0]))
        self.assertTrue(mgr.maybe_start_pull(rooms[1]))
        self.assertFalse(mgr.maybe_start_pull(rooms[2]))
        self.assertEqual(mgr._engine.start_read.call_count, 2)

        with mgr._state_lock:
            mgr._completed_recv.add(f"{rooms[0]}#0")
            mgr._release_pull_slots_locked()
        self.assertTrue(mgr.maybe_start_pull(rooms[2]))
        self.assertEqual(mgr._engine.start_read.call_count, 3)

    def test_failed_chunk_read_fails_the_request(self):
        room = 42
        local = list(range(200, 264))
        remote = list(range(100, 164))
        endpoint = "127.0.0.1:31020"
        mgr = self._decode(room, local, pool_pages=256)
        mgr._decode.chunk_queue[room] = [
            ("42#0", 11, list(range(0, 0 + 16)), remote[0:16], endpoint),
            ("42#1", 12, list(range(16, 16 + 16)), remote[16:32], endpoint),
        ]
        self.assertTrue(mgr.maybe_start_pull(room))
        self.assertFalse(mgr.is_failed(room))

        with mgr._state_lock:
            mgr._failed.add("42#0")
            mgr._native_failed.add("42#0")
            mgr._release_pull_slots_locked()
        self.assertTrue(mgr.is_failed(room))
        self.assertFalse(mgr.is_failed(420))
        self.assertEqual(len(mgr._pulling), 0)

        mgr.clear_room(room)
        self.assertFalse(mgr._failed)

    def test_cleared_room_keeps_its_staging_until_the_chunk_read_settles(self):
        first, second = 36, 37
        local = list(range(200, 264))
        remote = list(range(100, 164))
        endpoint = "127.0.0.1:31020"
        mgr = self._decode(first, local, pool_pages=16)
        mgr._decode.local_blocks[second] = local
        mgr._decode.remote_total[second] = 64
        mgr._decode.chunk_queue[first] = [
            ("36#0", 11, list(range(0, 0 + 16)), remote[0:16], endpoint)
        ]
        mgr._decode.chunk_queue[second] = [
            ("37#0", 21, list(range(0, 0 + 16)), remote[0:16], endpoint)
        ]

        self.assertTrue(mgr.maybe_start_pull(first))
        mgr._completed_send.add("36#0")
        # The framework aborts the request while raiden still holds the read.
        mgr.clear_room(first)
        self.assertFalse(mgr.maybe_start_pull(second))
        self.assertEqual(mgr._engine.start_read.call_count, 1)

        with mgr._state_lock:
            mgr._completed_recv.add("36#0")
            mgr._release_pull_slots_locked()
        self.assertTrue(mgr.maybe_start_pull(second))
        self.assertEqual(mgr._engine.start_read.call_args.args[0], "37#0")
        self.assertFalse(mgr._pulling_cleared)
        self.assertNotIn("36#0", mgr._completed_recv)
        self.assertNotIn("36#0", mgr._completed_send)

    def test_clearing_a_room_drops_its_settled_chunk_ids(self):
        room = 38
        local = list(range(200, 264))
        remote = list(range(100, 164))
        endpoint = "127.0.0.1:31020"
        mgr = self._decode(room, local, pool_pages=256)
        mgr._decode.chunk_queue[room] = [
            ("38#0", 11, list(range(0, 0 + 16)), remote[0:16], endpoint),
            ("38#1", 12, list(range(16, 16 + 16)), remote[16:32], endpoint),
        ]
        self.assertTrue(mgr.maybe_start_pull(room))
        with mgr._state_lock:
            mgr._completed_recv.add("38#0")
            mgr._release_pull_slots_locked()
        self.assertTrue(mgr.maybe_start_pull(room))
        with mgr._state_lock:
            mgr._completed_recv.update(("38#1", "380#0"))
            mgr._release_pull_slots_locked()

        mgr.clear_room(room)
        self.assertEqual(len(mgr._pulling), 0)
        # Only the cleared room's ids go; room 380 shares the prefix digits.
        self.assertEqual(mgr._completed_recv, {"380#0"})

    def test_hybrid_closing_read_holds_its_staging_until_both_engines_settle(self):
        first, second = 39, 40
        local = list(range(200, 264))
        remote = list(range(100, 164))
        endpoint = "127.0.0.1:31020"
        mgr = self._decode(first, local, pool_pages=64, swa_pool_pages=1)
        mgr._swa_engine = Mock()
        mgr._decode.remote_swa_blocks = {first: [4, 5], second: [4, 5]}
        mgr._decode.local_swa_blocks = {first: [7], second: [7]}
        mgr._decode.remote_swa_endpoint = {
            first: "127.0.0.1:31022",
            second: "127.0.0.1:31022",
        }
        mgr._decode.local_blocks[second] = local
        mgr._decode.remote_total[second] = 64
        mgr._decode.chunk_queue[first] = [
            (str(first), first, list(range(48, 48 + 16)), remote[48:64], endpoint)
        ]
        mgr._decode.chunk_queue[second] = [
            (str(second), second, list(range(48, 48 + 16)), remote[48:64], endpoint)
        ]

        self.assertTrue(mgr.maybe_start_pull(first))
        # The full-attention read fails while the sliding-window read is
        # still in flight on the other engine.
        with mgr._state_lock:
            mgr._failed.add(str(first))
            mgr._native_failed.add(str(first))
            mgr._release_pull_slots_locked()
        self.assertFalse(mgr.maybe_start_pull(second))
        self.assertEqual(mgr._engine.start_read.call_count, 1)

        with mgr._state_lock:
            mgr._completed_recv.add(mgr._swa_req_id(first))
            mgr._release_pull_slots_locked()
        self.assertTrue(mgr.maybe_start_pull(second))
        self.assertEqual(mgr._engine.start_read.call_args.args[0], str(second))
        self.assertEqual(mgr._swa_engine.start_read.call_count, 2)

    def test_cleared_hybrid_room_drops_both_reads_when_they_settle(self):
        room = 41
        local = list(range(200, 264))
        remote = list(range(100, 164))
        endpoint = "127.0.0.1:31020"
        mgr = self._decode(room, local, pool_pages=64, swa_pool_pages=1)
        mgr._swa_engine = Mock()
        mgr._decode.remote_swa_blocks = {room: [4, 5]}
        mgr._decode.local_swa_blocks = {room: [7]}
        mgr._decode.remote_swa_endpoint = {room: "127.0.0.1:31022"}
        mgr._decode.chunk_queue[room] = [
            (str(room), room, list(range(48, 48 + 16)), remote[48:64], endpoint)
        ]

        self.assertTrue(mgr.maybe_start_pull(room))
        with mgr._state_lock:
            mgr._completed_recv.add(str(room))
            mgr._release_pull_slots_locked()
        mgr.clear_room(room)
        self.assertEqual(len(mgr._pulling), 1)
        self.assertIn(str(room), mgr._completed_recv)

        with mgr._state_lock:
            mgr._completed_recv.add(mgr._swa_req_id(room))
            mgr._release_pull_slots_locked()
        self.assertEqual(len(mgr._pulling), 0)
        self.assertFalse(mgr._pulling_cleared)
        self.assertFalse(mgr._completed_recv)


class TestRaidenRoleStatePlacement(CustomTestCase):
    """Role handlers must only access fields owned by their state record.

    A decode-only field once lived in prefill initialization and caused an
    AttributeError on the chunked reshard path. Check all mixin methods so
    moving fields or methods cannot silently remove that coverage.
    """

    # Handlers named by the message each side receives; the reshard handshake is
    # decode-initiated, so RESHARD_REQ/READY/CHUNK_READY land on prefill.
    DECODE_SIDE = frozenset(
        {
            "_on_reshard_blocks",
            "_on_reshard_chunk",
            "_register_receiver_and_signal",
            "_register_reshard_request",
            "_register_reshard_swa_receiver",
            "_maybe_register_reshard_final",
            "_publish_reshard_plan",
            "register_decode_request",
        }
    )
    PREFILL_SIDE = frozenset(
        {
            "_on_reshard_req",
            "_on_reshard_ready",
            "_on_reshard_chunk_ready",
            "_run_reshard_push",
            "_run_reshard_swa_push",
            "_maybe_reshard_reply",
            "_announce_reshard_chunk",
        }
    )

    def test_role_state_is_declared_in_the_record_that_reads_it(self):
        from sglang_tpu.disaggregation.raiden.receiver import RaidenReceiverMixin
        from sglang_tpu.disaggregation.raiden.sender import RaidenSenderMixin

        roles = (
            ("_prefill", PrefillState, RaidenSenderMixin, self.PREFILL_SIDE),
            ("_decode", DecodeState, RaidenReceiverMixin, self.DECODE_SIDE),
        )
        state_fields = {
            role: {f.name for f in fields(state)} for role, state, _, _ in roles
        }
        self.assertIn("reshard_covered", state_fields["_decode"])
        self.assertNotIn("reshard_covered", state_fields["_prefill"])
        old_attributes = {
            "_" + name for names in state_fields.values() for name in names
        }
        for role, _, mixin, expected in roles:
            methods = dict(inspect.getmembers(mixin, inspect.isfunction))
            self.assertFalse(
                expected - methods.keys(),
                f"Missing role methods: {sorted(expected - methods.keys())}",
            )
            for name, method in methods.items():
                with self.subTest(role=role, method=name):
                    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
                    for node in ast.walk(tree):
                        if not isinstance(node, ast.Attribute):
                            continue
                        owner = node.value
                        if isinstance(owner, ast.Name) and owner.id == "self":
                            self.assertNotIn(node.attr, old_attributes)
                            if node.attr in state_fields:
                                self.assertEqual(node.attr, role)
                        if (
                            isinstance(owner, ast.Attribute)
                            and isinstance(owner.value, ast.Name)
                            and owner.value.id == "self"
                            and owner.attr in state_fields
                        ):
                            self.assertIn(node.attr, state_fields[owner.attr])

    @staticmethod
    def _manager(mode):
        def init_base(mgr, args, disaggregation_mode, *_):
            mgr.kv_args = args
            mgr.disaggregation_mode = disaggregation_mode
            mgr.attn_tp_rank = mgr.attn_dp_rank = mgr.attn_cp_rank = 0
            mgr.attn_tp_size = mgr.attn_dp_size = mgr.attn_cp_size = 1
            mgr.pp_rank, mgr.pp_size = 0, 1
            mgr.system_dp_rank = 0
            mgr.request_status = {}
            mgr.failure_records = {}
            mgr.failure_lock = threading.Lock()

        with (
            patch(
                "sglang.srt.disaggregation.common.conn.CommonKVManager.__init__",
                init_base,
            ),
            patch("sglang_tpu.disaggregation.raiden.conn.threading.Thread"),
            patch(
                "sglang_tpu.disaggregation.raiden.conn.concurrent.futures.ThreadPoolExecutor"
            ),
            patch.object(RaidenKVManager, "_start_heartbeat_checker_thread"),
        ):
            return RaidenKVManager(SimpleNamespace(page_size=128), mode, Mock())

    def test_constructor_creates_only_its_role_state(self):
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            with self.subTest(mode=mode):
                first, second = self._manager(mode), self._manager(mode)
                if mode == DisaggregationMode.PREFILL:
                    self.assertIsInstance(first._prefill, PrefillState)
                    self.assertIsNone(first._decode)
                    first._prefill.send_blocks[41] = [1]
                    self.assertEqual(second._prefill.send_blocks, {})
                else:
                    self.assertIsInstance(first._decode, DecodeState)
                    self.assertIsNone(first._prefill)
                    first._decode.local_blocks[41] = [1]
                    self.assertEqual(second._decode.local_blocks, {})

    def test_cleanup_preserves_other_rooms_and_native_io(self):
        room, other = 41, 42
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            with self.subTest(mode=mode):
                mgr = self._manager(mode)
                state = mgr._prefill if mgr._prefill is not None else mgr._decode
                # Populate every protocol collection so newly added fields
                # cannot silently escape the manager's cleanup sweep.
                for field in fields(state):
                    value = getattr(state, field.name)
                    if isinstance(value, dict):
                        if field.name == "announced_at":
                            value.update({f"{room}#0": 1.0, f"{other}#0": 1.0})
                        else:
                            value.update({room: object(), other: object()})
                    elif isinstance(value, set):
                        value.update((room, other))
                mgr._aborted[room] = 1.0
                if mode == DisaggregationMode.PREFILL:
                    mgr._send_reads[room] = {str(room)}
                    mgr._reshard_pushes_pending[room] = 1
                    pending_copy = (room, room, Mock())
                    mgr._stale_plans.append(pending_copy)
                else:
                    mgr._reshard_plans[room] = {str(room): room}

                mgr.clear_room(room)

                for field in fields(state):
                    value = getattr(state, field.name)
                    if isinstance(value, (dict, set)):
                        key = f"{room}#0" if field.name == "announced_at" else room
                        other_key = (
                            f"{other}#0" if field.name == "announced_at" else other
                        )
                        if field.name.startswith("victim_"):
                            # Victim sessions are independent of request rooms;
                            # even a matching producer sequence must survive.
                            self.assertIn(key, value, field.name)
                        else:
                            self.assertNotIn(key, value, field.name)
                        self.assertIn(other_key, value, field.name)
                self.assertIn(room, mgr._aborted)
                if mode == DisaggregationMode.PREFILL:
                    self.assertEqual(mgr._send_reads[room], {str(room)})
                    self.assertEqual(mgr._reshard_pushes_pending[room], 1)
                    self.assertIn(pending_copy, mgr._stale_plans)
                else:
                    self.assertNotIn(room, mgr._reshard_plans)
                    self.assertEqual(mgr._cleared_plans, {str(room): room})


class TestRaidenDefaultPort(CustomTestCase):
    def test_default_port_range_clears_the_serving_ports(self):
        """Each rank takes `base + 4*local_rank`, so the default base decides
        whether rank 0 lands on the HTTP serving port. At 30000 it did: the
        engine won the bind, uvicorn died with EADDRINUSE and took the process
        tree with it, so the server never went healthy and no transfer was ever
        attempted -- a bring-up crash that reads as a raiden failure (combo run
        33256868716). The ports below are the ones SGLang serves on by default.
        """
        reserved = range(DEFAULT_KV_PORT, DEFAULT_KV_PORT + 4 * 16)
        for port in (30000, 8000, 8998, 10011):
            self.assertNotIn(port, reserved)


class TestNoPushLock(CustomTestCase):
    def test_every_send_goes_through_the_endpoint_send_lock(self):
        # The per-endpoint send helper replaced the old global push lock; a
        # send site reintroducing it would crash in production while unit
        # fixtures could mask the missing constructor field.
        from sglang_tpu.disaggregation.raiden import conn, receiver, sender

        for module in (conn, sender, receiver):
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                self.assertNotIn("_push_lock\n", source)
                self.assertNotIn(
                    "self._push_lock",
                    source.replace("self._reshard_push_lock", ""),
                )


class _InlineExecutor:
    """Runs submitted pushes on the calling thread."""

    def submit(self, fn, *args):
        fn(*args)


class _ReadyFuture:
    def is_ready(self):
        return True

    def ok(self):
        return True


class _EntryList(list):
    def add(self):
        self.append(types.SimpleNamespace())
        return self[-1]


class _ScheduleMap(dict):
    def __missing__(self, key):
        self[key] = types.SimpleNamespace(entries=_EntryList())
        return self[key]


class _Pb2:
    """Stands in for raiden's protobuf module so the real plan builders run."""

    MEMORY_TYPE_DRAM = 1
    MEMORY_TYPE_HBM = 2

    @staticmethod
    def StartTransferRequest():
        return types.SimpleNamespace(shard_push_schedules=_ScheduleMap())


def _pb2_modules():
    rpc = types.ModuleType("tpu_sync.rpc")
    rpc.raiden_service_pb2 = _Pb2
    pkg = types.ModuleType("tpu_sync")
    pkg.rpc = rpc
    return {
        "tpu_sync": pkg,
        "tpu_sync.rpc": rpc,
        "tpu_sync.rpc.raiden_service_pb2": _Pb2,
    }


def _plan_blocks(req, prefill_rank):
    """(src, dst) block ids of one source's schedule in a built plan."""
    entries = req.shard_push_schedules[prefill_rank].entries
    return [e.src_block_id for e in entries], [e.dst_block_id for e in entries]


class TestRaidenChunkedResharding(CustomTestCase):
    """Chunk announcements and pushes on the resharding path."""

    ROOM = 41
    PREFILL_BLOCKS = list(range(100, 164))  # 64 pages, 4 chunks of 16
    DECODE_BLOCKS = list(range(200, 240))  # decode holds the last 40

    @staticmethod
    def _geometry(src_tp, dst_tp):
        from sglang_tpu.disaggregation.raiden.reshard_schedule import ReshardGeometry

        return ReshardGeometry(src_tp, dst_tp, 1024, 8)

    def _prefill(self):
        mgr = object.__new__(RaidenKVManager)
        mgr._prefill = PrefillState(announce_pages=1)
        mgr._decode = None
        mgr._aborted = {}
        mgr._send_reads = {}
        mgr._superseded_sender_reads = set()
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr.request_status = {}
        mgr.chunked_transfer = True
        mgr._state_lock = threading.Lock()
        mgr._status_lock = threading.Lock()
        mgr._announce_lock = threading.Lock()
        mgr._reshard_push_lock = threading.Lock()
        mgr._engine = Mock()
        mgr._engine.d2h = Mock(return_value=_ReadyFuture())
        # Device-shadow staging: a plan stages each device block into the
        # host block of the same id.
        mgr._engine.plan_host_blocks = Mock(side_effect=lambda uuid, blocks: blocks)
        mgr._swa_engine = None
        mgr._kv_caches = [object(), object()]
        mgr.kv_args = Mock(page_size=128)
        mgr.attn_tp_rank = 1
        mgr.attn_tp_size = 8
        mgr._local_rank = 1
        mgr._timeout_s = 5
        mgr._transfer_parallelism = 4
        mgr._push_executor = _InlineExecutor()
        mgr._make_geometry = self._geometry
        mgr._read_leases = {}
        mgr._uuid_seq = 0
        mgr._prefill.announce_pages = 16
        mgr._reshard_pushes_pending = {}
        mgr._completed_send = set()
        mgr._send_done_at = {}
        mgr._completed_recv = set()
        mgr._failed = set()
        mgr._native_failed = set()
        mgr._pulling = None
        mgr._aborted = {}
        mgr._send_reads = {}
        mgr._pulling_cleared = set()
        mgr._cleared_plans = {}
        mgr._stale_plans = []
        mgr._engine_poll_failing_since = {}
        mgr._socket = Mock()
        mgr._socket_send_locks = collections.defaultdict(threading.Lock)
        mgr._socket_lock = threading.Lock()
        mgr._socket_cache = {}
        mgr._connect = Mock(
            side_effect=lambda ep, is_ipv6=False: mgr._socket_cache.setdefault(
                ep, mgr._socket
            )
        )
        # clear_room sweeps the reshard bookkeeping too.
        mgr._reshard_slot_of = {}
        mgr._reshard_slot_rooms = [None]
        return mgr

    def _reshard_req(self):
        return ReshardReq(
            self.ROOM,
            "10.0.0.2",
            31002,
            "10.0.0.2:31000",
            self.DECODE_BLOCKS,
            4,
            [],
            "",
        ).to_frames()

    def _sent(self, mgr, tag):
        return [
            c.args[0]
            for c in mgr._socket.send_multipart.call_args_list
            if c.args[0][0] == tag
        ]

    def test_only_acknowledged_chunks_leave_the_closing_span(self):
        # An announced but never-acknowledged chunk keeps its pages in the
        # closing transfer: the frontier moves only when the decode asks for
        # the chunk's push, and a duplicate request moves nothing twice.
        mgr = self._prefill()
        room = self.ROOM
        mgr._prefill.reshard_req[room] = dict(
            reply_addr=("10.0.0.2", 5555), decode_blocks=list(range(64))
        )
        mgr._prefill.reshard_chunks[room] = {
            0: (0, list(range(16))),
            1: (16, list(range(16))),
        }
        mgr._push_executor = Mock()  # hold the push; only the accounting matters
        mgr._on_reshard_chunk_ready(ReshardChunkReady(41, 0, 7).to_frames())
        self.assertEqual(mgr._prefill.reshard_chunk_acked[room], {0})
        mgr._on_reshard_chunk_ready(ReshardChunkReady(41, 0, 7).to_frames())
        self.assertEqual(mgr._prefill.reshard_chunk_acked[room], {0})
        self.assertEqual(mgr._reshard_pushes_pending[room], 1)
        self.assertEqual(mgr._push_executor.submit.call_count, 1)

    def test_reshard_chunk_racing_the_closing_message_is_skipped(self):
        mgr = self._prefill()
        room = self.ROOM
        mgr._prefill.reshard_req[room] = dict(
            reply_addr=("10.0.0.2", 5555), decode_blocks=list(range(40))
        )
        mgr._prefill.reshard_replied.add(room)
        mgr._announce_reshard_chunk(room, 0, list(range(16)), 64)
        self.assertEqual(self._sent(mgr, MessageTag.RESHARD_CHUNK), [])
        self.assertNotIn(room, mgr._prefill.reshard_chunk_acked)

    def test_chunk_racing_the_closing_reply_is_skipped(self):
        # A chunk announced after the reply would arrive after it and outlive
        # the room's completion; its pages ride the closing transfer instead.
        mgr = self._prefill()
        mgr._ensure_engine = Mock()
        room = self.ROOM
        mgr._prefill.decode_addr[room] = ("10.0.0.2", 5555)
        mgr._prefill.decode_pages[room] = 40
        mgr._prefill.replied = {room}
        mgr._announce_chunk(room, 0, list(range(16)), 64)
        mgr._ensure_engine.assert_not_called()
        mgr._socket.send_multipart.assert_not_called()

    def test_prefill_announces_chunks_and_pushes_them_on_request(self):
        room, blocks = self.ROOM, self.PREFILL_BLOCKS
        mgr = self._prefill()
        # Two chunks commit before the decode has asked for the room.
        for start in (0, 16):
            mgr.append_send_blocks(room, blocks[start : start + 16])
            mgr.offer_chunk(room, start, blocks[start : start + 16], 64)
        self.assertEqual(self._sent(mgr, MessageTag.RESHARD_CHUNK), [])

        mgr._on_reshard_req(self._reshard_req())
        # The decode holds the first 24 pages: chunk 0 is never announced and
        # chunk 1 is trimmed to pages 24..32.
        chunks = self._sent(mgr, MessageTag.RESHARD_CHUNK)
        self.assertEqual(len(chunks), 1)
        chunk = ReshardChunk.from_frames(chunks[0])
        self.assertEqual(chunk.index, 0)
        self.assertEqual(chunk.start, 24)
        self.assertEqual(chunk.blocks, blocks[24:32])

        with patch.dict(sys.modules, _pb2_modules()):
            mgr._on_reshard_chunk_ready(ReshardChunkReady(41, 0, 777).to_frames())
            self.assertEqual(mgr._failed, set())
            mgr._engine.d2h.assert_called_once_with(
                blocks[24:32], blocks[24:32], [1] * 8
            )
            uuid, req = mgr._engine.register_active_plan.call_args.args
            self.assertEqual((uuid, req.uuid, req.req_id), (777, 777, "41#0"))
            self.assertEqual(
                _plan_blocks(req, 0), (blocks[24:32], self.DECODE_BLOCKS[0:8])
            )
            self.assertEqual(mgr._engine.push_registered_plan.call_count, 2)
            mgr._engine.unregister_active_plan.assert_called_once_with(777)
            self.assertFalse(mgr.is_send_done(room))

            # The prefill finishes; the closing push moves only the rest.
            mgr.append_send_blocks(room, blocks[32:64])
            mgr.finalize_send(room, None)
            self.assertEqual(len(self._sent(mgr, MessageTag.RESHARD_BLOCKS)), 1)
            mgr._on_reshard_ready(ReshardReady(41, list(range(32, 64))).to_frames())
            self.assertEqual(mgr._failed, set())
            uuid, req = mgr._engine.register_active_plan.call_args.args
            self.assertEqual((uuid, req.req_id), (room, "41"))
            self.assertEqual(
                _plan_blocks(req, 0), (blocks[32:64], self.DECODE_BLOCKS[8:40])
            )
        self.assertTrue(mgr.is_send_done(room))

    def _decode(self):
        mgr = object.__new__(RaidenKVManager)
        mgr._prefill = None
        mgr._decode = DecodeState()
        mgr._timeout_s = 120.0
        mgr._aborted = {}
        mgr._send_reads = {}
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr.request_status = {}
        mgr.chunked_transfer = True
        mgr._state_lock = threading.Lock()
        mgr._status_lock = threading.Lock()
        mgr._announce_lock = threading.Lock()
        mgr._engine = Mock(transfer_address="10.0.0.2:31000")
        mgr._swa_engine = None
        mgr.kv_args = Mock(page_size=128)
        mgr.attn_tp_rank = 0
        mgr.attn_tp_size = 4
        mgr._local_rank = 0
        mgr._uuid_seq = 0
        mgr._make_geometry = self._geometry
        mgr._decode.reshard_meta = {
            self.ROOM: dict(
                local_blocks=list(self.DECODE_BLOCKS),
                source_ranks=[0, 1],
                src_attn_tp=8,
                aux_index=None,
                ready_endpoints=["tcp://10.0.0.10:1", "tcp://10.0.0.11:1"],
            )
        }
        mgr._decode.reshard_sources = {self.ROOM: {}}
        # Same shape for the window pool; _on_reshard_blocks records both.
        mgr._decode.reshard_swa_sources = {self.ROOM: {}}
        mgr._reshard_plans = {}
        mgr._cleared_plans = {}
        mgr._stale_plans = []
        mgr._engine_poll_failing_since = {}
        mgr._pulling = None
        mgr._aborted = {}
        mgr._send_reads = {}
        mgr._pulling_cleared = set()
        mgr._decode.announced_at = {}
        mgr._timeout_s = 120.0
        mgr._completed_send = set()
        mgr._send_done_at = {}
        mgr._completed_recv = set()
        mgr._failed = set()
        mgr._native_failed = set()
        mgr._superseded_sender_reads = set()
        mgr._socket = Mock()
        mgr._socket_send_locks = collections.defaultdict(threading.Lock)
        mgr._socket_lock = threading.Lock()
        mgr._socket_cache = {}
        mgr._connect = Mock(
            side_effect=lambda ep, is_ipv6=False: mgr._socket_cache.setdefault(
                ep, mgr._socket
            )
        )
        # clear_room sweeps the reshard bookkeeping too.
        mgr._reshard_slot_of = {}
        mgr._reshard_slot_rooms = [None]
        return mgr

    def _decode_with_chunk_plan(self):
        """A decode whose sources have both announced chunk 0 of ROOM, so its
        receive plan is registered and the push is in flight."""
        mgr = self._decode()
        with patch.dict(sys.modules, _pb2_modules()):
            for rank in (0, 1):
                mgr._on_reshard_chunk(
                    _reshard_chunk_frames(41, 0, 24, 64, list(range(124, 132)), rank)
                )
        uuid = mgr._engine.register_active_plan.call_args.args[0]
        self.assertEqual(mgr._reshard_plans, {self.ROOM: {"41#0": uuid}})
        return mgr, uuid

    def test_chunk_plan_id_fits_positive_int64(self):
        for upper_bound in (False, True):
            with self.subTest(upper_bound=upper_bound):
                with patch(
                    "secrets.randbelow",
                    side_effect=lambda bound: bound - 1 if upper_bound else 0,
                ):
                    mgr, uuid = self._decode_with_chunk_plan()

                _, req = mgr._engine.register_active_plan.call_args.args
                self.assertGreater(uuid, 0)
                # StartTransferRequest.uuid uses protobuf's signed int64 range.
                self.assertEqual(Int64Value(value=uuid).value, req.uuid)

    def test_aborted_room_keeps_its_chunk_plan_until_the_push_settles(self):
        mgr, uuid = self._decode_with_chunk_plan()
        mgr._completed_send.add("41#0")

        # The framework aborts the request while the sources are still pushing.
        mgr.clear_room(self.ROOM)
        mgr._engine.unregister_active_plan.assert_not_called()
        self.assertEqual(mgr._cleared_plans, {"41#0": uuid})
        self.assertNotIn(self.ROOM, mgr._reshard_plans)
        self.assertFalse(mgr._completed_send)

        # An unrelated completion does not touch it.
        mgr._engine.poll_stats_with_details = Mock(
            return_value=_polled([], ["7#0"], [])
        )
        mgr._drain_engine()
        mgr._engine.unregister_active_plan.assert_not_called()

        # The push lands late: the plan goes, exactly once, with its record.
        mgr._engine.poll_stats_with_details = Mock(
            return_value=_polled([], ["41#0"], [])
        )
        mgr._drain_engine()
        mgr._engine.unregister_active_plan.assert_called_once_with(uuid)
        self.assertEqual(mgr._cleared_plans, {})
        self.assertNotIn("41#0", mgr._completed_recv)
        mgr._drain_engine()
        mgr._engine.unregister_active_plan.assert_called_once_with(uuid)

    def test_aborted_room_drops_its_chunk_plan_when_the_push_times_out(self):
        mgr, uuid = self._decode_with_chunk_plan()
        mgr.clear_room(self.ROOM)

        # Raiden reports the timeout and has already dropped the plan itself.
        mgr._engine.unregister_active_plan = Mock(
            side_effect=RuntimeError(
                f"KVCacheManager unregister_active_plan failed: Plan with "
                f"UUID {uuid} is not registered"
            )
        )
        mgr._engine.poll_stats_with_details = Mock(
            return_value=_polled([], [], ["41#0"])
        )
        mgr._drain_engine()
        mgr._engine.unregister_active_plan.assert_called_once_with(uuid)
        self.assertEqual(mgr._cleared_plans, {})
        self.assertFalse(mgr._failed)
        self.assertNotIn(self.ROOM, mgr.failure_records)

    def test_room_is_not_complete_while_closing_registration_is_pending(self):
        mgr, _ = self._decode_with_chunk_plan()
        mgr._completed_recv.add("41#0")
        seen_complete = []

        def registers_slowly(uuid, req, is_sender):
            # While the closing plan registers, the room must not look done:
            # its closing read is already owed.
            seen_complete.append(mgr.is_recv_done(self.ROOM))

        mgr._engine.register_active_plan = Mock(side_effect=registers_slowly)
        with patch.dict(sys.modules, _pb2_modules()):
            for rank in (0, 1):
                mgr._on_reshard_blocks(
                    _reshard_blocks_frames(
                        41,
                        list(range(100 + 200 * rank, 164 + 200 * rank)),
                        rank,
                    )
                )
        self.assertEqual(seen_complete, [False])
        self.assertFalse(mgr.is_recv_done(self.ROOM))
        mgr._completed_recv.add("41")
        self.assertTrue(mgr.is_recv_done(self.ROOM))

    def test_closing_plan_is_dropped_once_its_push_settles(self):
        mgr, chunk_uuid = self._decode_with_chunk_plan()
        with patch.dict(sys.modules, _pb2_modules()):
            for rank in (0, 1):
                mgr._on_reshard_blocks(
                    _reshard_blocks_frames(
                        41,
                        list(range(100 + 200 * rank, 164 + 200 * rank)),
                        rank,
                    )
                )
        self.assertEqual(
            mgr._reshard_plans[self.ROOM], {"41#0": chunk_uuid, "41": self.ROOM}
        )
        mgr._completed_recv.update({"41#0", "41"})
        self.assertTrue(mgr.is_recv_done(self.ROOM))
        mgr.clear_room(self.ROOM)
        self.assertEqual(
            sorted(
                c.args[0] for c in mgr._engine.unregister_active_plan.call_args_list
            ),
            sorted([chunk_uuid, self.ROOM]),
        )

    def test_clearing_a_room_drops_its_settled_chunk_plan_at_once(self):
        mgr, uuid = self._decode_with_chunk_plan()
        mgr._completed_recv.add("41#0")
        mgr.clear_room(self.ROOM)
        mgr._engine.unregister_active_plan.assert_called_once_with(uuid)
        self.assertEqual(mgr._cleared_plans, {})
        self.assertFalse(mgr._completed_recv)

    def test_reshard_decode_chunk_failure_after_closing_push_fails_room(self):
        """When closing push completes first on reshard decode, a later failure of
        a chunk receive plan must NOT be filtered out: the room must be marked failed
        and room_has_active_io must clear (preventing permanent hang / leak)."""
        mgr, _ = self._decode_with_chunk_plan()
        with patch.dict(sys.modules, _pb2_modules()):
            for rank in (0, 1):
                mgr._on_reshard_blocks(
                    _reshard_blocks_frames(
                        41,
                        list(range(100 + 200 * rank, 164 + 200 * rank)),
                        rank,
                    )
                )
        # Closing plan "41" settles in _completed_recv
        mgr._completed_recv.add("41")
        self.assertFalse(mgr.is_recv_done(self.ROOM))
        self.assertTrue(mgr.room_has_active_io(self.ROOM))

        # Raiden reports chunk plan "41#0" failed
        mgr._engine.poll_stats_with_details = Mock(
            return_value=_polled([], [], ["41#0"])
        )
        mgr._drain_engine()

        # The failure MUST NOT be masked by "41" in _completed_recv
        self.assertIn("41#0", mgr._failed)
        self.assertIn("41#0", mgr._native_failed)
        self.assertTrue(mgr.is_failed(self.ROOM))
        self.assertFalse(mgr.room_has_active_io(self.ROOM))

    def test_late_reshard_chunk_after_final_seen_is_dropped(self):
        mgr = self._decode()
        mgr._decode.final_seen.add(self.ROOM)
        with patch.dict(sys.modules, _pb2_modules()):
            mgr._on_reshard_chunk(
                _reshard_chunk_frames(41, 0, 24, 64, list(range(124, 132)), 0)
            )
        self.assertNotIn(self.ROOM, mgr._decode.reshard_chunk_sources)
        mgr._engine.register_active_plan.assert_not_called()

    def test_late_reshard_chunk_after_reshard_registered_is_dropped(self):
        mgr = self._decode()
        mgr._decode.reshard_registered.add(self.ROOM)
        with patch.dict(sys.modules, _pb2_modules()):
            mgr._on_reshard_chunk(
                _reshard_chunk_frames(41, 0, 24, 64, list(range(124, 132)), 0)
            )
        self.assertNotIn(self.ROOM, mgr._decode.reshard_chunk_sources)
        mgr._engine.register_active_plan.assert_not_called()

    def test_duplicate_reshard_blocks_after_final_seen_is_dropped(self):
        mgr = self._decode()
        mgr._decode.final_seen.add(self.ROOM)
        with patch.dict(sys.modules, _pb2_modules()):
            mgr._on_reshard_blocks(_reshard_blocks_frames(41, [100, 101], 0))
        self.assertEqual(mgr._decode.reshard_sources[self.ROOM], {})

    def test_duplicate_reshard_blocks_after_reshard_registered_is_dropped(self):
        mgr = self._decode()
        mgr._decode.reshard_registered.add(self.ROOM)
        with patch.dict(sys.modules, _pb2_modules()):
            mgr._on_reshard_blocks(_reshard_blocks_frames(41, [100, 101], 0))
        self.assertEqual(mgr._decode.reshard_sources[self.ROOM], {})

    def test_timed_out_stage_keeps_its_plan_until_the_copy_settles(self):

        room = self.ROOM
        blocks = self.PREFILL_BLOCKS
        mgr = self._prefill()
        mgr._timeout_s = 0.01

        class _PendingFuture:
            ready = False

            def is_ready(self):
                return self.ready

        fut = _PendingFuture()
        mgr._engine.plan_host_blocks = Mock(side_effect=lambda uuid, b: list(b))
        mgr._engine.d2h = Mock(return_value=fut)
        with patch.dict(sys.modules, _pb2_modules()):
            mgr._on_reshard_req(self._reshard_req())
            mgr._prefill.send_blocks[room] = list(blocks[:32])
            for start in (0, 16):
                mgr.offer_chunk(room, start, blocks[start : start + 16], 64)
            chunk_msg = self._sent(mgr, MessageTag.RESHARD_CHUNK)[0]
            # The push's D2H stage never becomes ready and times out.
            chunk = ReshardChunk.from_frames(chunk_msg)
            mgr._on_reshard_chunk_ready(
                ReshardChunkReady(41, chunk.index, 7).to_frames()
            )
        self.assertTrue(mgr.is_failed(room))
        # The copy still runs: the plan (and its staging) must survive it.
        mgr._engine.unregister_active_plan.assert_not_called()
        self.assertEqual([(u, r) for u, r, _ in mgr._stale_plans], [(7, self.ROOM)])

        mgr._engine.poll_stats_with_details = Mock(return_value=_polled([], [], []))
        mgr._drain_engine()
        mgr._engine.unregister_active_plan.assert_not_called()

        fut.ready = True
        mgr._drain_engine()
        mgr._engine.unregister_active_plan.assert_called_once_with(7)
        self.assertEqual(mgr._stale_plans, [])

    def test_timed_out_stage_is_reclaimed_without_another_poll(self):

        room = self.ROOM
        blocks = self.PREFILL_BLOCKS
        mgr = self._prefill()
        mgr._timeout_s = 0.01

        class _PendingFuture:
            ready = False

            def is_ready(self):
                return self.ready

        fut = _PendingFuture()
        mgr._engine.plan_host_blocks = Mock(side_effect=lambda uuid, b: list(b))
        mgr._engine.d2h = Mock(return_value=fut)
        with patch.dict(sys.modules, _pb2_modules()):
            mgr._on_reshard_req(self._reshard_req())
            mgr._prefill.send_blocks[room] = list(blocks[:32])
            for start in (0, 16):
                mgr.offer_chunk(room, start, blocks[start : start + 16], 64)
            chunk_msg = self._sent(mgr, MessageTag.RESHARD_CHUNK)[0]
            chunk = ReshardChunk.from_frames(chunk_msg)
            mgr._on_reshard_chunk_ready(
                ReshardChunkReady(41, chunk.index, 7).to_frames()
            )
        # The copy settles later with nobody polling; the plan still goes.
        fut.ready = True
        deadline = time.time() + 3
        while mgr._stale_plans and time.time() < deadline:
            time.sleep(0.02)
        mgr._engine.unregister_active_plan.assert_called_once_with(7)
        self.assertEqual(mgr._stale_plans, [])

    def test_push_asked_for_before_the_room_was_cleared_is_a_noop(self):

        room = self.ROOM
        blocks = self.PREFILL_BLOCKS
        mgr = self._prefill()
        mgr._push_executor = Mock()  # hold the push instead of running it
        with patch.dict(sys.modules, _pb2_modules()):
            mgr._on_reshard_req(self._reshard_req())
            mgr._prefill.send_blocks[room] = list(blocks[:32])
            for start in (0, 16):
                mgr.offer_chunk(room, start, blocks[start : start + 16], 64)
            chunk_msg = self._sent(mgr, MessageTag.RESHARD_CHUNK)[0]
            chunk = ReshardChunk.from_frames(chunk_msg)
            mgr._on_reshard_chunk_ready(
                ReshardChunkReady(41, chunk.index, 7).to_frames()
            )
            self.assertEqual(mgr._reshard_pushes_pending[room], 1)
            push = mgr._push_executor.submit.call_args.args

            mgr.clear_room(room)
            push[0](*push[1:])
        mgr._engine.register_active_plan.assert_not_called()
        self.assertFalse(mgr._failed)
        self.assertNotIn(room, mgr._reshard_pushes_pending)
        self.assertFalse(mgr._completed_send)

    def test_schedules_are_keyed_by_the_sender_engine_id(self):
        room = self.ROOM
        src = {8: list(range(100, 164)), 9: list(range(300, 364))}
        mgr = self._decode()

        def chunk(sender_id):
            return _reshard_chunk_frames(
                41, 0, 24, 64, src[sender_id][24:32], sender_id
            )

        # The sources sit on the prefill's second attention-DP replica, so
        # their engine ids are 8 and 9 while their attention-TP ranks (what
        # the decode expects announcements from) are 0 and 1.
        with patch.dict(sys.modules, _pb2_modules()):
            mgr._on_reshard_chunk(chunk(8))
            mgr._engine.register_active_plan.assert_not_called()
            mgr._on_reshard_chunk(chunk(9))
            self.assertEqual(mgr._failed, set())
            uuid, req = mgr._engine.register_active_plan.call_args.args
            self.assertEqual(set(req.shard_push_schedules), {8, 9})
            self.assertEqual(
                _plan_blocks(req, 8), (src[8][24:32], self.DECODE_BLOCKS[0:8])
            )
            self.assertEqual(
                _plan_blocks(req, 9), (src[9][24:32], self.DECODE_BLOCKS[0:8])
            )

    def test_room_cleared_during_plan_registration_is_not_resurrected(self):
        room = self.ROOM
        src = {0: list(range(100, 164)), 1: list(range(300, 364))}
        mgr = self._decode()

        # clear_room() lands while register_active_plan runs outside the lock.
        def clears_midway(uuid, req, is_sender):
            mgr.clear_room(room)

        mgr._engine.register_active_plan = Mock(side_effect=clears_midway)

        def chunk(rank):
            return _reshard_chunk_frames(41, 0, 24, 64, src[rank][24:32], rank)

        with patch.dict(sys.modules, _pb2_modules()):
            mgr._on_reshard_chunk(chunk(0))
            mgr._on_reshard_chunk(chunk(1))
        # The plan is dropped again, nothing is recreated, no push is asked for.
        uuid = mgr._engine.register_active_plan.call_args.args[0]
        mgr._engine.unregister_active_plan.assert_called_once_with(uuid)
        self.assertEqual(mgr._decode.chunk_reads.get(room, []), [])
        self.assertEqual(mgr._reshard_plans.get(room, {}), {})
        self.assertEqual(self._sent(mgr, MessageTag.RESHARD_CHUNK_READY), [])
        self.assertFalse(mgr.is_failed(room))

    def test_failed_plan_registration_fails_the_room(self):
        room = self.ROOM
        src = {0: list(range(100, 164)), 1: list(range(300, 364))}
        mgr = self._decode()
        mgr.record_failure = Mock()
        mgr._engine.register_active_plan = Mock(
            side_effect=RuntimeError("RESOURCE_EXHAUSTED: no host blocks")
        )

        def chunk(rank):
            return _reshard_chunk_frames(41, 0, 24, 64, src[rank][24:32], rank)

        with patch.dict(sys.modules, _pb2_modules()):
            mgr._on_reshard_chunk(chunk(0))
            mgr._on_reshard_chunk(chunk(1))
        self.assertTrue(mgr.is_failed(room))
        self.assertIn("RESOURCE_EXHAUSTED", mgr.record_failure.call_args.args[1])
        # The chunk is not admitted anywhere a completion could be faked from.
        self.assertEqual(mgr._decode.chunk_reads.get(room, []), [])
        self.assertEqual(mgr._reshard_plans.get(room, {}), {})
        self.assertEqual(self._sent(mgr, MessageTag.RESHARD_CHUNK_READY), [])

    def test_decode_registers_a_plan_per_chunk_once_every_source_announced(self):
        room = self.ROOM
        src = {0: list(range(100, 164)), 1: list(range(300, 364))}
        mgr = self._decode()

        def chunk(rank):
            return _reshard_chunk_frames(41, 0, 24, 64, src[rank][24:32], rank)

        with patch.dict(sys.modules, _pb2_modules()):
            mgr._on_reshard_chunk(chunk(0))
            mgr._engine.register_active_plan.assert_not_called()
            mgr._on_reshard_chunk(chunk(1))
            self.assertEqual(mgr._failed, set())
            uuid, req = mgr._engine.register_active_plan.call_args.args
            self.assertEqual((req.uuid, req.req_id), (uuid, "41#0"))
            self.assertEqual(
                _plan_blocks(req, 0), (src[0][24:32], self.DECODE_BLOCKS[0:8])
            )
            self.assertEqual(
                _plan_blocks(req, 1), (src[1][24:32], self.DECODE_BLOCKS[0:8])
            )
            ready = self._sent(mgr, MessageTag.RESHARD_CHUNK_READY)
            self.assertEqual(len(ready), 2)
            message = ReshardChunkReady.from_frames(ready[0])
            self.assertEqual([message.index, message.uuid], [0, uuid])
            self.assertFalse(mgr.is_recv_done(room))

            # Both sources finish: the closing plan covers only the tail.
            for rank in (0, 1):
                mgr._on_reshard_blocks(_reshard_blocks_frames(41, src[rank], rank))
            uuid, req = mgr._engine.register_active_plan.call_args.args
            self.assertEqual((uuid, req.req_id), (room, "41"))
            self.assertEqual(
                _plan_blocks(req, 0), (src[0][32:64], self.DECODE_BLOCKS[8:40])
            )
            self.assertEqual(
                _plan_blocks(req, 1), (src[1][32:64], self.DECODE_BLOCKS[8:40])
            )
            self.assertEqual(len(self._sent(mgr, MessageTag.RESHARD_READY)), 2)

        self.assertEqual(mgr._decode.chunk_reads[room], ["41#0", "41"])
        self.assertFalse(mgr.is_recv_done(room))
        mgr._completed_recv.update({"41#0", "41"})
        self.assertTrue(mgr.is_recv_done(room))

    def test_partial_source_chunk_is_folded_into_the_closing_plan(self):
        """A source that loses its chunk/final race sends no chunk.  The other
        source's unacknowledged announcement must not leave a hole."""
        room = self.ROOM
        src = {0: list(range(100, 164)), 1: list(range(300, 364))}
        mgr = self._decode()

        with patch.dict(sys.modules, _pb2_modules()):
            mgr._on_reshard_chunk(
                _reshard_chunk_frames(41, 0, 24, 64, src[0][24:32], 0)
            )
            self.assertEqual(self._sent(mgr, MessageTag.RESHARD_CHUNK_READY), [])

            for rank in (0, 1):
                mgr._on_reshard_blocks(_reshard_blocks_frames(41, src[rank], rank))

        uuid, req = mgr._engine.register_active_plan.call_args.args
        self.assertEqual((uuid, req.req_id), (room, "41"))
        self.assertEqual(_plan_blocks(req, 0), (src[0][24:64], self.DECODE_BLOCKS))
        self.assertEqual(_plan_blocks(req, 1), (src[1][24:64], self.DECODE_BLOCKS))
        self.assertNotIn(room, mgr._decode.reshard_chunk_sources)
        self.assertEqual(mgr._decode.chunk_reads[room], ["41"])

    def test_closing_plan_covers_a_reshard_chunk_lost_mid_range(self):
        # Chunk 0 (pages 24:32) lost its race; chunk 1 (pages 32:40) was
        # registered by both sources. The closing plan and the READY must
        # cover 24:32 and 40:64, not just the tail past the furthest chunk.
        room = self.ROOM
        src = {0: list(range(100, 164)), 1: list(range(300, 364))}
        mgr = self._decode()

        with patch.dict(sys.modules, _pb2_modules()):
            for rank in (0, 1):
                mgr._on_reshard_chunk(
                    _reshard_chunk_frames(41, 1, 32, 64, src[rank][32:40], rank)
                )
            for rank in (0, 1):
                mgr._on_reshard_blocks(_reshard_blocks_frames(41, src[rank], rank))

        missing = list(range(24, 32)) + list(range(40, 64))
        uuid, req = mgr._engine.register_active_plan.call_args.args
        self.assertEqual((uuid, req.req_id), (room, "41"))
        self.assertEqual(
            _plan_blocks(req, 0),
            (
                [src[0][pg] for pg in missing],
                self.DECODE_BLOCKS[0:8] + self.DECODE_BLOCKS[16:40],
            ),
        )
        readies = self._sent(mgr, MessageTag.RESHARD_READY)
        self.assertTrue(readies)
        self.assertEqual(ReshardReady.from_frames(readies[-1]).pages, missing)

    def test_closing_plan_waits_for_chunk_registration_to_publish(self):
        room = self.ROOM
        src = {0: list(range(100, 164)), 1: list(range(300, 364))}
        mgr = self._decode()
        seen_during_chunk_registration = []

        def register_with_final_arrival(uuid, req, is_sender):
            if req.req_id != "41#0":
                return
            for rank in (0, 1):
                mgr._on_reshard_blocks(_reshard_blocks_frames(41, src[rank], rank))
            seen_during_chunk_registration.append(
                (
                    room in mgr._decode.reshard_registered,
                    mgr.is_recv_done(room),
                )
            )

        mgr._engine.register_active_plan = Mock(side_effect=register_with_final_arrival)

        with patch.dict(sys.modules, _pb2_modules()):
            for rank in (0, 1):
                mgr._on_reshard_chunk(
                    _reshard_chunk_frames(41, 0, 24, 64, src[rank][24:32], rank)
                )

        self.assertEqual(seen_during_chunk_registration, [(False, False)])
        self.assertEqual(
            [c.args[1].req_id for c in mgr._engine.register_active_plan.call_args_list],
            ["41#0", "41"],
        )
        self.assertEqual(mgr._decode.chunk_reads[room], ["41#0", "41"])
        self.assertNotIn(room, mgr._decode.reshard_chunk_registering)


class _FakeFuture:
    def __init__(self, ready_after=0, error=None):
        self._calls = 0
        self._ready_after = ready_after
        self._error = error

    def is_ready(self):
        ready = self._calls >= self._ready_after
        self._calls += 1
        return ready

    def ok(self):
        return self._error is None

    def error_message(self):
        return self._error or ""


class TestRaidenKVTransferEmptyPageList(CustomTestCase):
    def test_empty_page_list_issues_no_dma(self):
        """A hybrid write can carry SWA pages but no Full pages. raiden reads an
        empty page list as the whole layer buffer, so none may reach it."""
        transfer = object.__new__(RaidenKVTransfer)
        transfer._mgr = Mock()
        for method, mgr_call in (
            (transfer.backup, transfer._mgr.d2h),
            (transfer.load, transfer._mgr.h2d),
        ):
            with self.subTest(method=method.__name__):
                self.assertIs(method([], []), NOOP_EVENT)
                mgr_call.assert_not_called()
                method([3], [5])
                mgr_call.assert_called_once_with([3], [5], [1])


class TestRaidenFutureEvent(CustomTestCase):
    def test_noop_event_is_always_ready(self):
        self.assertTrue(NOOP_EVENT.query())
        NOOP_EVENT.synchronize()  # must not block

    def test_query_reflects_future(self):
        fut = _FakeFuture(ready_after=2)
        ev = RaidenFutureEvent(fut)
        self.assertFalse(ev.query())
        self.assertFalse(ev.query())
        self.assertTrue(ev.query())

    def test_synchronize_polls_until_ready(self):
        ev = RaidenFutureEvent(_FakeFuture(ready_after=3))
        ev.synchronize()  # returns once is_ready() flips True

    def test_query_raises_once_pending_past_the_deadline(self):
        """A copy that never lands must stop the scheduler when polled, as a
        blocking wait does, instead of holding its pages and every later ack."""
        monotonic = "sglang_tpu.raiden_transfer.time.monotonic"
        with envs.SGLANG_TPU_RAIDEN_DMA_TIMEOUT_S.override(60.0):
            with patch(monotonic, return_value=1000.0):
                pending = RaidenFutureEvent(_FakeFuture(ready_after=10**6))
                landed = RaidenFutureEvent(_FakeFuture())
            with patch(monotonic, return_value=1059.0):
                self.assertFalse(pending.query())
            with patch(monotonic, return_value=1061.0):
                with self.assertRaisesRegex(TimeoutError, "still pending"):
                    pending.query()
                # A copy that landed is done, however late it is polled.
                self.assertTrue(landed.query())

    def test_record_and_wait_are_noops(self):
        ev = RaidenFutureEvent(_FakeFuture())
        ev.record()
        ev.wait()

    def test_failed_dma_raises_at_poll(self):
        # Dispatch runs on raiden's background thread, so a bad transfer is
        # reported through the future; treating a failed-but-ready future as
        # landed would hand the forward garbage KV.
        ev = RaidenFutureEvent(_FakeFuture(ready_after=1, error="bad offsets"))
        self.assertFalse(ev.query())  # still pending: no error yet
        with self.assertRaisesRegex(RuntimeError, "bad offsets"):
            ev.query()
        with self.assertRaisesRegex(RuntimeError, "bad offsets"):
            RaidenFutureEvent(_FakeFuture(error="bad offsets")).synchronize()


class TestCombinedFutureEvent(CustomTestCase):
    def test_ready_only_when_all_events_ready(self):
        """The combined event must gate on EVERY underlying DMA: an ack that
        reports ready while the draft pool's transfer is still in flight would
        free/reuse pages mid-DMA (silent corruption)."""
        fast = RaidenFutureEvent(_FakeFuture(ready_after=0))
        slow = RaidenFutureEvent(_FakeFuture(ready_after=2))
        ev = CombinedFutureEvent([fast, slow])
        self.assertFalse(ev.query())
        self.assertFalse(ev.query())
        self.assertTrue(ev.query())
        ev.synchronize()  # must not block once all are ready


class TestDecodeRadixPullRegistration(CustomTestCase):
    """Tests trimmed producer registration and zero-read completion on full hits."""

    ROOM = 41
    BLOCKS = [3, 9, 12, 20]
    SWA_BLOCKS = [5, 6, 7]

    def _make_producer(self, decode_pages):
        mgr = object.__new__(RaidenKVManager)
        mgr._prefill = PrefillState(announce_pages=1)
        mgr._decode = None
        mgr.attn_tp_rank = 0
        mgr._state_lock = threading.Lock()
        mgr._announce_lock = threading.Lock()
        mgr._aborted = {}
        mgr._send_reads = {}
        mgr._prefill.send_blocks = {self.ROOM: list(self.BLOCKS)}
        mgr._prefill.send_aux = {self.ROOM: b"aux"}
        mgr._prefill.decode_addr = {self.ROOM: ("127.0.0.1", 31002)}
        mgr._prefill.decode_pages = {self.ROOM: decode_pages}
        mgr._prefill.decode_swa_pages = {self.ROOM: 0}
        mgr._prefill.reshard_finalized = {self.ROOM}
        mgr._completed_send = set()
        mgr._send_done_at = {}
        mgr._failed = set()
        mgr.record_failure = Mock()
        mgr.update_status = Mock()
        mgr._engine = Mock()
        mgr._swa_engine = None
        mgr._swa_kv_caches = []
        mgr.raiden_endpoint = "127.0.0.1:31020"
        mgr.swa_raiden_endpoint = None
        socket = Mock()
        mgr._socket_send_locks = defaultdict(threading.Lock)
        mgr._socket_lock = threading.Lock()
        mgr._socket_cache = {}
        mgr._connect = Mock(
            side_effect=lambda ep, is_ipv6=False: mgr._socket_cache.setdefault(
                ep, socket
            )
        )
        return mgr, socket

    def _make_receiver(self, local_blocks):
        mgr = object.__new__(RaidenKVManager)
        mgr.attn_tp_rank = 0
        mgr._prefill = None
        mgr._decode = DecodeState()
        mgr.chunked_transfer = False
        mgr._state_lock = threading.Lock()
        mgr._status_lock = threading.Lock()
        mgr._aborted = {}
        mgr._send_reads = {}
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr.request_status = {}
        mgr._engine = Mock()
        mgr._swa_engine = None
        mgr._pulling = StagingAdmission(128, 8)
        mgr._pulling_cleared = set()
        mgr._dynamic_host_staging = True
        mgr._timeout_s = 120.0
        mgr._full_engine_max_blocks = 128
        mgr._completed_recv = set()
        mgr._failed = set()
        mgr._native_failed = set()
        mgr._decode.local_blocks = {self.ROOM: list(local_blocks)}
        mgr._decode.remote_blocks = {self.ROOM: list(self.BLOCKS)}
        mgr._decode.remote_endpoint = {self.ROOM: "127.0.0.1:31020"}
        return mgr

    def test_partial_hit_registers_the_tail_but_announces_everything(self):
        mgr, socket = self._make_producer(decode_pages=2)
        mgr._maybe_reply(self.ROOM)
        mgr._engine.register_read.assert_called_once_with(
            str(self.ROOM), self.ROOM, self.BLOCKS[-2:]
        )
        payload = socket.send_multipart.call_args.args[0]
        self.assertEqual(Blocks.from_frames(payload).blocks, self.BLOCKS)
        # Ensure send does not complete while a read is pending an engine ACK.
        self.assertFalse(mgr.is_send_done(self.ROOM))

    def test_full_hit_sends_blocks_and_completes_without_a_registration(self):
        mgr, socket = self._make_producer(decode_pages=0)
        mgr._maybe_reply(self.ROOM)
        mgr._engine.register_read.assert_not_called()
        reply = Blocks.from_frames(socket.send_multipart.call_args.args[0])
        self.assertEqual(reply.blocks, self.BLOCKS)
        self.assertEqual(reply.aux_blob, b"aux")
        self.assertTrue(mgr.is_send_done(self.ROOM))
        self.assertFalse(mgr.is_failed(self.ROOM))

    def test_full_hit_send_failure_fails_the_room_instead_of_completing(self):
        mgr, socket = self._make_producer(decode_pages=0)
        socket.send_multipart.side_effect = zmq.ZMQError(zmq.EHOSTUNREACH)
        mgr._maybe_reply(self.ROOM)
        self.assertFalse(mgr.is_send_done(self.ROOM))
        self.assertTrue(mgr.is_failed(self.ROOM))

    def test_blocks_send_failure_fails_the_room(self):
        mgr, socket = self._make_producer(decode_pages=len(self.BLOCKS))
        socket.send_multipart.side_effect = zmq.ZMQError(zmq.EHOSTUNREACH)
        mgr._maybe_reply(self.ROOM)
        self.assertFalse(mgr.is_send_done(self.ROOM))
        self.assertTrue(mgr.is_failed(self.ROOM))

    def test_receiver_full_hit_completes_without_a_read(self):
        mgr = self._make_receiver(local_blocks=[])
        self.assertTrue(mgr.maybe_start_pull(self.ROOM))
        mgr._engine.start_read.assert_not_called()
        self.assertTrue(mgr.is_recv_done(self.ROOM))
        self.assertFalse(mgr.is_failed(self.ROOM))

    def test_receiver_partial_hit_still_reads_the_tail(self):
        mgr = self._make_receiver(local_blocks=[8, 9])
        self.assertTrue(mgr.maybe_start_pull(self.ROOM))
        mgr._engine.start_read.assert_called_once_with(
            str(self.ROOM), self.ROOM, "127.0.0.1:31020", self.BLOCKS[-2:], [8, 9]
        )
        self.assertFalse(mgr.is_recv_done(self.ROOM))

    def _add_swa_producer(self, mgr):
        mgr._swa_engine = Mock()
        mgr._prefill.send_swa_blocks = {self.ROOM: list(self.SWA_BLOCKS)}
        mgr._prefill.decode_swa_pages = {self.ROOM: len(self.SWA_BLOCKS)}
        mgr.swa_raiden_endpoint = "127.0.0.1:31022"

    def test_swa_partial_hit_trims_the_window_registration(self):
        mgr, _ = self._make_producer(decode_pages=2)
        self._add_swa_producer(mgr)
        mgr._prefill.decode_swa_pages = {self.ROOM: 1}
        mgr._maybe_reply(self.ROOM)
        mgr._engine.register_read.assert_called_once_with(
            str(self.ROOM), self.ROOM, self.BLOCKS[-2:]
        )
        mgr._swa_engine.register_read.assert_called_once_with(
            f"swa:{self.ROOM}", self.ROOM, self.SWA_BLOCKS[-1:]
        )
        self.assertFalse(mgr.is_send_done(self.ROOM))

    def test_swa_full_hit_completes_both_engines_without_registration(self):
        mgr, socket = self._make_producer(decode_pages=0)
        self._add_swa_producer(mgr)
        mgr._prefill.decode_swa_pages = {self.ROOM: 0}
        mgr._maybe_reply(self.ROOM)
        mgr._engine.register_read.assert_not_called()
        mgr._swa_engine.register_read.assert_not_called()
        reply = Blocks.from_frames(socket.send_multipart.call_args.args[0])
        self.assertEqual(reply.swa_blocks, self.SWA_BLOCKS)
        self.assertTrue(mgr.is_send_done(self.ROOM))
        self.assertFalse(mgr.is_failed(self.ROOM))

    def test_full_hit_skips_the_window_registration_even_untrimmed(self):
        # Full-attention full hits issue no network reads. Suppress window
        # registrations regardless of window page counts to avoid orphaned leases.
        mgr, _ = self._make_producer(decode_pages=0)
        self._add_swa_producer(mgr)
        mgr._prefill.decode_swa_pages = {self.ROOM: len(self.SWA_BLOCKS)}
        mgr._maybe_reply(self.ROOM)
        mgr._swa_engine.register_read.assert_not_called()
        self.assertTrue(mgr.is_send_done(self.ROOM))

    def test_pull_req_carries_the_window_page_count(self):
        # A PULL_REQ without the window count is a decode without a window
        # pool; the sender then registers no window blocks.
        mgr, _ = self._make_producer(decode_pages=len(self.BLOCKS))
        mgr._prefill.held_chunks = {}
        mgr._prefill.send_total = {}
        mgr._on_pull_req(PullReq(self.ROOM, "127.0.0.1", 31002, 2).to_frames())
        self.assertEqual(mgr._prefill.decode_pages[self.ROOM], 2)
        self.assertEqual(mgr._prefill.decode_swa_pages[self.ROOM], 0)
        mgr._prefill.replied = set()
        mgr._on_pull_req(
            PullReq(self.ROOM, "127.0.0.1", 31002, 2, decode_swa_pages=1).to_frames()
        )
        self.assertEqual(mgr._prefill.decode_swa_pages[self.ROOM], 1)

    def _add_swa_receiver(self, mgr):
        mgr._swa_engine = Mock()
        mgr._swa_engine_max_blocks = 8
        mgr._decode.local_swa_blocks = {self.ROOM: []}
        mgr._decode.remote_swa_blocks = {self.ROOM: list(self.SWA_BLOCKS)}
        mgr._decode.remote_swa_endpoint = {self.ROOM: "127.0.0.1:31022"}

    def test_receiver_swa_full_hit_settles_the_window_key_too(self):
        mgr = self._make_receiver(local_blocks=[])
        self._add_swa_receiver(mgr)
        self.assertTrue(mgr.maybe_start_pull(self.ROOM))
        mgr._engine.start_read.assert_not_called()
        mgr._swa_engine.start_read.assert_not_called()
        self.assertTrue(mgr.is_recv_done(self.ROOM))

    def test_receiver_swa_partial_hit_reads_both_tails(self):
        mgr = self._make_receiver(local_blocks=[8, 9])
        self._add_swa_receiver(mgr)
        mgr._decode.local_swa_blocks = {self.ROOM: [2]}
        self.assertTrue(mgr.maybe_start_pull(self.ROOM))
        mgr._engine.start_read.assert_called_once_with(
            str(self.ROOM), self.ROOM, "127.0.0.1:31020", self.BLOCKS[-2:], [8, 9]
        )
        mgr._swa_engine.start_read.assert_called_once_with(
            f"swa:{self.ROOM}", self.ROOM, "127.0.0.1:31022", self.SWA_BLOCKS[-1:], [2]
        )
        self.assertFalse(mgr.is_recv_done(self.ROOM))


if __name__ == "__main__":
    unittest.main()
