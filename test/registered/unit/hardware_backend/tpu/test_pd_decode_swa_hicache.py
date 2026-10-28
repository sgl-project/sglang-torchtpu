"""Unit tests for the TPU PD decode SWA HiCache patch (pd_decode_swa_hicache).

Decode backs up and restores full-attention KV through host RAM, and a
request's SWA window always comes from the prefill transfer. With host SWA
windows, RAM also keeps the last window of each path, which only the prefix
offer restore (pd_decode_window_restore) loads back. Each wrapper is called
with the real upstream function (or a fake where upstream needs a full server),
so the tests pin both this behavior and the pass-through to upstream for every
other cache, except the restore hold, which every raiden PD decode cache gets.
Some tests drive a real unified tree: its host stack attach, its locks and
its SWA backup walk.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import contextlib
import inspect
import itertools
import re
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation import decode
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeRequest,
    DecodeTransferQueue,
)
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreResult,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    IncLockRefResult,
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.registry import TreeCacheBuildContext
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.base import CacheTransferPhase
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import HiCacheDraftMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import pd_decode_swa_hicache as tpu
from sglang_tpu import pd_decode_window_restore as restore
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.tpu_dma_hybrid_cache_controller import TpuDMAHybridCacheController

ASSEMBLER = "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler"
BUILDER = "sglang.srt.mem_cache.kv_cache_builder"
TPU = "sglang_tpu.pd_decode_swa_hicache"


@contextlib.contextmanager
def installed_patches():
    """Install the patches and restore the upstream functions afterwards."""
    originals = [
        (owner, name, getattr(owner, name)) for owner, name, _ in tpu._patch_targets()
    ]
    tpu.install()
    try:
        yield originals
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)
        hybrid_pool_assembler._STRATEGIES.remove(tpu._FULL_ONLY_SWA_STRATEGY)
        tpu._is_installed = False


def mark_full_only(cache):
    # As the full-only host stack builder does.
    tpu._full_only_host_caches.add(cache)
    tpu._restore_gated_caches.add(cache)


def swa_receipt(*, node_id, swa_uuid):
    return DecLockRefParams(
        node_id=node_id, component_lock_uuids={ComponentType.SWA: swa_uuid}
    )


def swa_uuid(receipt):
    return receipt.get_lock_uuid(ComponentType.SWA)


def make_req():
    return SimpleNamespace(
        rid="restore",
        origin_input_ids=list(range(512)),
        output_ids=[],
        extra_key=None,
        kv_rotation_base=None,
        cache_salt=None,
        prefix_indices=torch.arange(64),
        last_node=1,
        lock_receipt=swa_receipt(node_id=1, swa_uuid=17),
        swa_prefix_lock_released=True,
        swa_stale_window_floor=None,
        kv=ReqKvInfo(req_pool_idx=0, cache_protected_len=64),
        _compute_max_prefix_len=lambda n: n - 1,
        finished=lambda: False,
        refresh_fill_ids=lambda: None,
        full_untruncated_fill_ids=array("q", range(512)),
    )


def make_match(device_len=64, host_len=256):
    return SimpleNamespace(
        device_indices=torch.arange(device_len),
        last_device_node=2,
        last_host_node=3,
        best_match_node=3,
        host_hit_length=host_len,
        swa_host_hit_length=0,
        swa_branching_seqlen=None,
        mamba_host_hit_length=0,
        mamba_branching_seqlen=None,
        cache_protected_len=None,
    )


def full_only_cache(*, is_eagle=False):
    cache = MagicMock()
    cache.tree_core.is_eagle = is_eagle
    cache.swa_reprefill_tail_tokens.return_value = 0
    # No L3 storage backend in this mode.
    cache.query_storage_hit_length.return_value = 0
    mark_full_only(cache)
    return cache


def disagg(*, mode="decode", decode_radix=True, retraction_backup=None):
    return SimpleNamespace(
        disaggregation_mode=mode,
        disaggregation_decode_enable_radix_cache=decode_radix,
        disaggregation_decode_retraction_backup=retraction_backup,
    )


def memory(**overrides):
    fields = dict(
        hicache_io_backend="raiden",
        hicache_host_memory_mode="cache",
        hicache_storage_backend=None,
        hicache_size=0,
        hicache_write_policy="write_through",
        enable_unified_memory=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def make_ctx():
    # The registry resolves a non-CPU/CUDA allocator device to the Python core.
    # "meta" stands in for "tpu", which torch knows only with torch_tpu imported.
    params = SimpleNamespace(
        tree_core_backend=None,
        enable_session_radix_cache=False,
        tree_components=None,
        component_registry_override=None,
        token_to_kv_pool_allocator=SimpleNamespace(device="meta"),
    )
    return TreeCacheBuildContext(
        server_args=None,
        params=params,
        is_hybrid_swa=True,
        is_hybrid_ssm=False,
        enable_hierarchical_cache=False,
        disable_radix_cache=False,
        effective_chunked_prefill_size=None,
        tp_worker=None,
        model_config=None,
        tp_size=1,
        tp_rank=0,
        tp_group=None,
    )


def spec(*, eagle3=False, dflash=False, none=False):
    if none:
        return SpeculativeAlgorithm.NONE
    if eagle3:
        return SpeculativeAlgorithm.EAGLE3
    if dflash:
        return SpeculativeAlgorithm.DFLASH
    return SpeculativeAlgorithm.EAGLE


class _OptedInSpec(CustomSpecAlgo):
    def supports_pd_decode_swa_hicache(self) -> bool:
        return True


def plugin_spec(*, opted_in):
    cls = _OptedInSpec if opted_in else CustomSpecAlgo
    return cls("PLUGIN", factory=MagicMock())


def real_swa_cache(*, is_eagle=False):
    """A real FULL + SWA unified tree over mocked pools, without HiCache."""
    allocator = MagicMock(spec=SWATokenToKVPoolAllocator)
    allocator.device = torch.device("cpu")
    allocator.page_size = 64
    allocator._swa_req_ring = False
    allocator.translate_loc_from_full_to_swa.side_effect = lambda indices: indices
    pool = MagicMock(spec=ReqToTokenPool)
    pool.req_to_token = torch.arange(512, dtype=torch.int64).reshape(1, 512)
    pool.write.side_effect = pool.req_to_token.__setitem__
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator,
        page_size=64,
        sliding_window_size=128,
        is_eagle=is_eagle,
        tree_components=(ComponentType.FULL, ComponentType.SWA),
    )
    with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"):
        cache = UnifiedRadixCache(params)
    return cache, params


def build_kwargs(**overrides):
    kwargs = dict(
        enable_hierarchical_cache=True,
        tp_worker=SimpleNamespace(is_hybrid_swa=True),
        page_size=64,
        spec_algorithm=spec(none=True),
        hicache_draft_plan=None,
    )
    kwargs.update(overrides)
    return kwargs


class TestCacheConstruction(CustomTestCase):
    def test_upstream_flag_uses_are_known(self):
        # _build_kv_cache passes enable_hierarchical_cache=False to clear the
        # upstream SWA guard, so any new upstream read of the flag would see
        # False for full-only caches. Review _build_kv_cache when this fails.
        source = inspect.getsource(inspect.unwrap(kv_cache_builder.build_kv_cache))
        uses = [
            line.strip()
            for line in source.splitlines()
            if re.search(r"(?<![.\w])enable_hierarchical_cache\b", line)
        ]
        self.assertEqual(
            uses,
            [
                "enable_hierarchical_cache: bool,",
                "if enable_hierarchical_cache:",
                "enable_hierarchical_cache=enable_hierarchical_cache,",
                'enabled=enable_hierarchical_cache or retraction_backup == "host_pool",',
                'enable_hierarchical_cache or retraction_backup == "host_pool"',
            ],
        )

    def test_other_configs_keep_upstream_build(self):
        no_swa = build_kwargs(tp_worker=SimpleNamespace(is_hybrid_swa=False))
        for name, kwargs, mode in (
            ("hicache off", build_kwargs(enable_hierarchical_cache=False), "decode"),
            ("no SWA", no_swa, "decode"),
            ("prefill", build_kwargs(), "prefill"),
        ):
            with (
                self.subTest(name),
                patch(TPU + ".get_disagg", return_value=disagg(mode=mode)),
            ):
                upstream = MagicMock()
                tpu._build_kv_cache(upstream, **kwargs)
                upstream.assert_called_once_with(**kwargs)

    def test_full_only_build_passes_guard_then_reenables_hicache(self):
        # Upstream raises for decode SWA radix + HiCache before building the
        # tree. The build must pass it with HiCache off and turn HiCache and
        # host pool auto-sizing back on only in the tree cache factory.
        ctx = make_ctx()
        cache = MagicMock()

        def upstream_create(ctx):
            self.assertTrue(ctx.enable_hierarchical_cache)
            mark_full_only(cache)
            return cache

        def upstream_build(**kwargs):
            self.assertFalse(kwargs["enable_hierarchical_cache"])
            return tpu._create_tree_cache(upstream_create, ctx)

        with (
            patch(TPU + ".get_disagg", return_value=disagg()),
            patch(TPU + ".get_memory", return_value=memory()),
            patch(TPU + ".auto_size_hicache") as auto_size,
            patch(BUILDER + ".maybe_register_hicache_draft") as register_draft,
        ):
            self.assertIs(tpu._build_kv_cache(upstream_build, **build_kwargs()), cache)
            auto_size.assert_called_once_with(ctx.params, None, enabled=True)
            register_draft.assert_not_called()
            # Outside that build the factory is untouched.
            upstream = MagicMock()
            tpu._create_tree_cache(upstream, ctx)
            upstream.assert_called_once_with(ctx)
            auto_size.assert_called_once()

    def test_draft_plan_is_sized_and_registered_as_a_sidecar(self):
        # Upstream registers the draft sidecar only with the flag on, which
        # this build turns off to pass the SWA guard.
        for algorithm in (spec(eagle3=True), spec(dflash=True)):
            with self.subTest(algorithm=algorithm):
                self._check_draft_sidecar(algorithm)

    def _check_draft_sidecar(self, algorithm):
        ctx = make_ctx()
        cache = MagicMock()
        draft_plan = SimpleNamespace(mode=HiCacheDraftMode.SIDECAR)

        def upstream_build(**kwargs):
            self.assertIs(kwargs["hicache_draft_plan"], draft_plan)
            mark_full_only(cache)
            return tpu._create_tree_cache(lambda ctx: cache, ctx)

        with (
            patch(TPU + ".get_disagg", return_value=disagg()),
            patch(TPU + ".get_memory", return_value=memory()),
            patch(TPU + ".auto_size_hicache") as auto_size,
            patch(BUILDER + ".maybe_register_hicache_draft") as register_draft,
        ):
            tpu._build_kv_cache(
                upstream_build,
                **build_kwargs(spec_algorithm=algorithm, hicache_draft_plan=draft_plan),
            )
        auto_size.assert_called_once_with(ctx.params, draft_plan, enabled=True)
        register_draft.assert_called_once_with(tree_cache=cache, draft_plan=draft_plan)
        self.assertIsNone(tpu._building_draft_plan)

    def test_full_only_build_requires_the_full_only_stack(self):
        # E.g. a cache type the SWA host strategy does not build.
        def upstream_build(**kwargs):
            return tpu._create_tree_cache(lambda ctx: object(), make_ctx())

        with (
            patch(TPU + ".get_disagg", return_value=disagg()),
            patch(TPU + ".get_memory", return_value=memory()),
            patch(TPU + ".auto_size_hicache"),
            self.assertRaisesRegex(ValueError, "unified radix tree"),
        ):
            tpu._build_kv_cache(upstream_build, **build_kwargs())
        self.assertFalse(tpu._is_building_full_only_host_cache)

    def test_full_only_build_requires_the_python_tree_core(self):
        # The Rust tree core has its own SWA validator without this policy.
        def upstream_build(**kwargs):
            return tpu._create_tree_cache(MagicMock(), make_ctx())

        with (
            patch(TPU + ".get_disagg", return_value=disagg()),
            patch(TPU + ".get_memory", return_value=memory()),
            patch(TPU + ".auto_size_hicache"),
            patch(TPU + ".select_tree_core_backend", return_value="rust"),
            self.assertRaisesRegex(ValueError, "TREE_CORE_BACKEND"),
        ):
            tpu._build_kv_cache(upstream_build, **build_kwargs())
        self.assertFalse(tpu._is_building_full_only_host_cache)

    def test_builtin_algorithms_accepted(self):
        accepted = {
            algo
            for algo in SpeculativeAlgorithm
            if algo.supports_pd_decode_swa_hicache()
        }
        self.assertEqual(
            accepted,
            {
                SpeculativeAlgorithm.NONE,
                SpeculativeAlgorithm.EAGLE3,
                SpeculativeAlgorithm.DFLASH,
            },
        )

    def test_opted_in_plugin_algorithm_builds(self):
        cache = MagicMock()

        def upstream_build(**kwargs):
            mark_full_only(cache)
            return tpu._create_tree_cache(lambda ctx: cache, make_ctx())

        with (
            patch(TPU + ".get_disagg", return_value=disagg()),
            patch(TPU + ".get_memory", return_value=memory()),
            patch(TPU + ".auto_size_hicache"),
        ):
            built = tpu._build_kv_cache(
                upstream_build,
                **build_kwargs(spec_algorithm=plugin_spec(opted_in=True)),
            )
        self.assertIs(built, cache)

    def test_unsupported_configs_are_rejected(self):
        for overrides, mem, dis, tree_core, reason in (
            ({"page_size": 1}, {}, {}, "python", "page-size"),
            # MTP and the other drafts are not validated in this mode.
            (
                {"spec_algorithm": spec()},
                {},
                {},
                "python",
                "speculative algorithm .* is not supported",
            ),
            (
                {"spec_algorithm": plugin_spec(opted_in=False)},
                {},
                {},
                "python",
                "speculative algorithm .* is not supported",
            ),
            (
                {
                    "spec_algorithm": plugin_spec(opted_in=True),
                    "hicache_draft_plan": SimpleNamespace(mode=HiCacheDraftMode.PACKED),
                },
                {},
                {},
                "python",
                "packed",
            ),
            # Draft KV reaches RAM only as upstream's draft sidecar.
            ({"spec_algorithm": spec(eagle3=True)}, {}, {}, "python", "sidecar"),
            ({"spec_algorithm": spec(dflash=True)}, {}, {}, "python", "sidecar"),
            (
                {
                    "spec_algorithm": spec(eagle3=True),
                    "hicache_draft_plan": SimpleNamespace(mode=HiCacheDraftMode.PACKED),
                },
                {},
                {},
                "python",
                "sidecar",
            ),
            ({}, {"hicache_io_backend": "direct"}, {}, "python", "raiden"),
            (
                {},
                {"hicache_host_memory_mode": "buffer_only"},
                {},
                "python",
                "host-memory-mode",
            ),
            ({}, {"hicache_storage_backend": "file"}, {}, "python", "storage-backend"),
            # Upstream already auto-sizes the host pool for host_pool retraction.
            ({}, {}, {"retraction_backup": "host_pool"}, "python", "retraction"),
        ):
            with (
                self.subTest(reason),
                patch(TPU + ".get_disagg", return_value=disagg(**dis)),
                patch(TPU + ".get_memory", return_value=memory(**mem)),
                envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override(tree_core),
            ):
                upstream = MagicMock()
                with self.assertRaisesRegex(ValueError, reason):
                    tpu._build_kv_cache(upstream, **build_kwargs(**overrides))
                upstream.assert_not_called()


class TestFullOnlyHostStack(CustomTestCase):
    def build(self, hicache_size=8):
        # Sparse global layers: SWA at 0 and 2, full attention at 1 and 3.
        kvcache = SimpleNamespace(
            start_layer=0,
            layers_mapping={0: (0, True), 1: (0, False), 2: (1, True), 3: (1, False)},
            full_kv_pool=MagicMock(),
        )
        params = MagicMock(mtp_draft_device_pools=(), page_size=64)
        cache = MagicMock()
        # Autospec checks every call against the upstream signature.
        with (
            patch(TPU + ".get_memory", return_value=memory(hicache_size=hicache_size)),
            patch(ASSEMBLER + ".build_kv_host_pool", autospec=True) as host_pool,
            patch(ASSEMBLER + ".build_pool_entry", autospec=True) as entry,
            patch(TPU + ".HybridCacheController.create_for_backend") as controller,
            patch(TPU + ".HostPoolGroup", autospec=True) as group,
        ):
            result = tpu._build_full_only_host_stack(
                cache=cache,
                kvcache=kvcache,
                params=params,
                load_cache_event=None,
                storage_backend=None,
                storage_backend_extra_config=None,
                prefetch_threshold=256,
                model_name=None,
                enable_storage_metrics=False,
            )
        return cache, result, host_pool, entry, controller, group

    def test_one_full_pool_keeps_global_layer_span_and_whole_budget(self):
        cache, result, host_pool, entry, controller, group = self.build()
        self.assertEqual(host_pool.call_args.kwargs["host_size"], 8)
        self.assertEqual(len(group.call_args.args[0]), 1)
        self.assertEqual(entry.call_args.kwargs["name"], PoolName.KV)
        self.assertEqual(entry.call_args.kwargs["layer_mapping"], {1: 0, 3: 1})
        self.assertEqual(entry.call_args.kwargs["transfer_layer_id_max"], 4)
        self.assertEqual(controller.call_args.kwargs["transfer_layer_id_max"], 4)
        self.assertEqual(set(result.component_host_pools), {ComponentType.FULL})
        self.assertTrue(tpu.uses_full_only_host_cache(cache))
        self.assertIn(result.cache_controller, tpu._load_device_pages)

    def test_controller_args_match_the_controller_signature(self):
        # create_for_backend forwards *args / **kwargs to the controller.
        *_, controller, _ = self.build()
        inspect.signature(HybridCacheController.__init__).bind(
            None, *controller.call_args.args, **controller.call_args.kwargs
        )

    def test_ratio_sized_pool_when_no_fixed_budget(self):
        _, _, host_pool, *_ = self.build(hicache_size=0)
        self.assertIsNone(host_pool.call_args.kwargs["host_size"])


class TestFullOnlyStrategy(CustomTestCase):
    """The registered host stack strategy, through upstream's assembler."""

    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="scheduler")

    def attach(self, num_windows):
        cache, params = real_swa_cache()
        params.token_to_kv_pool_allocator.swa_attn_allocator = MagicMock()
        kvcache = MagicMock(spec=SWAKVPool)
        # SWA at layer 0, full attention at layer 1.
        kvcache.start_layer = 0
        kvcache.layers_mapping = {0: (0, True), 1: (0, False)}
        kvcache.full_kv_pool = MagicMock()
        # 1000 bytes per device token, padding pages included.
        kvcache.swa_kv_pool = MagicMock(size=4096)
        kvcache.swa_kv_pool.get_kv_size_bytes.return_value = (2048000, 2048000)
        cache.token_to_kv_pool_allocator.get_kvcache.return_value = kvcache
        components = {ComponentType.FULL, ComponentType.SWA}
        host_pools = {"full": MagicMock(), "swa": MagicMock()}
        with (
            installed_patches(),
            tpu_envs.SGLANG_TPU_PD_SWA_HOST_WINDOWS.override(num_windows),
            patch(TPU + ".get_memory", return_value=memory()),
            patch(
                ASSEMBLER + ".build_kv_host_pool",
                autospec=True,
                side_effect=lambda **kw: host_pools[kw["pool_label"]],
            ) as host_pool,
            patch(TPU + ".HybridCacheController.create_for_backend") as controller,
        ):
            # Outside a full-only build, upstream picks its own SWA stack.
            self.assertIsNot(
                hybrid_pool_assembler._select_strategy(kvcache, components),
                tpu._FULL_ONLY_SWA_STRATEGY,
            )
            with tpu._full_only_host_cache_build(None):
                hybrid_pool_assembler.attach_hybrid_pool_to_unified_cache(
                    cache,
                    params,
                    ServerArgs(model_path="dummy"),
                    load_cache_event=None,
                )
        self.assertTrue(tpu.uses_full_only_host_cache(cache))
        self.assertIs(cache.cache_controller, controller.return_value)
        self.assertIs(
            cache.components[ComponentType.FULL]._full_kv_pool_host,
            host_pools["full"],
        )
        return cache, host_pool, host_pools

    def test_zero_windows_attaches_one_full_pool(self):
        cache, _, _ = self.attach(num_windows=0)
        self.assertEqual(set(cache.host_pool_group.entry_map), {PoolName.KV})
        self.assertFalse(tpu.uses_window_host_cache(cache))
        # init_hicache derives has_swa_host_pool=False from this.
        self.assertIsNone(cache.components[ComponentType.SWA]._swa_kv_pool_host)

    def test_windows_add_an_swa_pool_sized_in_windows(self):
        cache, host_pool, host_pools = self.attach(num_windows=8)
        self.assertEqual(
            set(cache.host_pool_group.entry_map), {PoolName.KV, PoolName.SWA}
        )
        self.assertTrue(tpu.uses_window_host_cache(cache))
        self.assertIs(
            cache.components[ComponentType.SWA]._swa_kv_pool_host, host_pools["swa"]
        )
        sizes = {
            call.kwargs["pool_label"]: call.kwargs["host_size"]
            for call in host_pool.call_args_list
        }
        # The full pool keeps the full-only budget (--hicache-ratio here).
        self.assertIsNone(sizes["full"])
        # Window 128 at page 64 can straddle 3 pages: 8 x 3 x 64 tokens.
        self.assertAlmostEqual(sizes["swa"] * 1e9 / 1000, 8 * 3 * 64)

    def test_window_pages_cover_a_window_starting_mid_page(self):
        # Muse Spark: window 2048, page 128.
        self.assertEqual(
            tpu._swa_host_window_pages(sliding_window_size=2048, page_size=128), 17
        )
        self.assertEqual(
            tpu._swa_host_window_pages(sliding_window_size=128, page_size=64), 3
        )


class TestAdmissionAndAllocation(CustomTestCase):
    def queue(self, cache):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.tree_cache = cache
        queue.scheduler = SimpleNamespace(enable_decode_hicache=True)
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._pre_alloc_fill_len = MagicMock(return_value=512)
        queue._swa_tail_len = MagicMock(return_value=192)
        return queue

    def test_full_only_match_stops_at_the_swa_window(self):
        cache = full_only_cache()
        cache.match_prefix.return_value = make_match(device_len=64, host_len=256)
        cache.inc_lock_ref.return_value = IncLockRefResult(
            node_id=2, component_lock_uuids={ComponentType.SWA: 19}
        )
        req = make_req()
        result = tpu._match_prefix_and_lock(MagicMock(), self.queue(cache), req)
        # Device and host hits are both bounded by fill_len - swa_tail_len.
        self.assertEqual(len(cache.match_prefix.call_args.args[0].key), 320)
        self.assertEqual(result.decode_prefix_len, 320)
        self.assertEqual(swa_uuid(req.lock_receipt), 19)

    def test_eagle_match_still_reaches_the_swa_window(self):
        # Bigram keys need one token past the window start to match its page.
        cache = full_only_cache(is_eagle=True)
        cache.match_prefix.return_value = make_match(device_len=64, host_len=256)
        cache.inc_lock_ref.return_value = IncLockRefResult(node_id=2)
        tpu._match_prefix_and_lock(MagicMock(), self.queue(cache), make_req())
        self.assertEqual(len(cache.match_prefix.call_args.args[0].key), 321)

    def test_restore_rematch_adds_the_bigram_boundary_token(self):
        req = make_req()
        for cache, expected in (
            (full_only_cache(is_eagle=True), 321),
            (full_only_cache(is_eagle=False), 320),
            (MagicMock(), 320),
        ):
            upstream = MagicMock()
            tpu._restore_rematch(
                upstream, cache, req, req.origin_input_ids[:320], include_req=True
            )
            args, kwargs = upstream.call_args
            self.assertEqual((len(args[2]), kwargs), (expected, {"include_req": True}))

    def test_other_caches_keep_upstream_match(self):
        upstream = MagicMock()
        queue = self.queue(MagicMock())
        req = make_req()
        tpu._match_prefix_and_lock(upstream, queue, req)
        upstream.assert_called_once_with(queue, req)

    def alloc(self, *, prefix_len, total_prefix_len):
        allocator = MagicMock(
            page_size=64,
            device="cpu",
            spec=["page_size", "device", "alloc_extend_swa_tail"],
        )
        allocator.alloc_extend_swa_tail.return_value = torch.arange(192)
        req = make_req()
        # Runs the real upstream allocation path.
        tpu._alloc_for_decode_prealloc(
            decode.alloc_for_decode_prealloc,
            allocator,
            req=req,
            fill_len=512,
            delta_len=512 - total_prefix_len,
            prefix_len=prefix_len,
            total_prefix_len=total_prefix_len,
            prefix_indices=torch.arange(prefix_len),
            uses_swa_tail=True,
            swa_tail_len=192,
        )
        return allocator.alloc_extend_swa_tail.call_args.kwargs, req

    def test_fresh_suffix_is_allocated_after_the_host_restore_range(self):
        for prefix_len, last_loc in ((64, 63), (0, -1)):
            with self.subTest(prefix_len=prefix_len):
                kwargs, req = self.alloc(prefix_len=prefix_len, total_prefix_len=320)
                self.assertEqual(kwargs["prefix_lens"].tolist(), [320])
                self.assertEqual(kwargs["prefix_lens_cpu"].tolist(), [320])
                self.assertEqual(kwargs["last_loc"].tolist(), [last_loc])
                self.assertEqual(kwargs["extend_num_tokens"], 192)
                self.assertEqual(req.kv.get_evicted_seqlen(ComponentType.SWA), 320)

    def test_host_restore_range_must_end_on_a_page(self):
        with self.assertRaises(AssertionError):
            self.alloc(prefix_len=64, total_prefix_len=321)

    def test_no_host_restore_keeps_upstream_prefix(self):
        kwargs, _ = self.alloc(prefix_len=320, total_prefix_len=320)
        self.assertEqual(kwargs["prefix_lens"].tolist(), [320])
        self.assertEqual(kwargs["last_loc"].tolist(), [319])


class TestRestore(CustomTestCase):
    def setUp(self):
        self.queue = DecodeHiCacheTransferMixin()
        self.queue.tree_cache = full_only_cache()
        self.cache = self.queue.tree_cache
        self.cache.match_prefix.return_value = make_match()
        self.cache.init_load_back.return_value = (torch.arange(64, 320), 3)
        self.restored_receipt = swa_receipt(node_id=3, swa_uuid=23)
        self.cache.inc_lock_ref.return_value = MagicMock(
            to_dec_params=MagicMock(return_value=self.restored_receipt)
        )
        self.decode_req = DecodeRequest(
            req=make_req(),
            kv_receiver=MagicMock(),
            prefix_match=DecodePrefixMatch(
                prefix_indices=torch.arange(64),
                l2_host_hit_length=256,
                l3_storage_hit_length=0,
                last_device_node=1,
            ),
        )

    def restore(self):
        # Upstream's restore, which the commit below takes over from.
        return DecodeHiCacheTransferMixin._try_hicache_queue_load_back(
            self.queue, self.decode_req
        )

    def commit(self):
        tpu._commit_hicache_local_restore_to_req(
            DecodeHiCacheTransferMixin._commit_hicache_local_restore_to_req,
            self.queue,
            self.decode_req,
        )

    def test_commit_moves_the_lock_then_drops_its_swa_part(self):
        self.restore()
        self.commit()
        req = self.decode_req.req
        args, kwargs = self.cache.dec_lock_ref.call_args
        self.assertEqual((args[0], swa_uuid(args[1])), (1, 17))
        self.assertEqual(kwargs, {"skip_swa": True})
        write = self.cache.req_to_token_pool.write.call_args.args
        self.assertEqual(write[0], (0, slice(64, 320)))
        self.assertEqual(req.last_node, 3)
        self.assertIs(req.lock_receipt, self.restored_receipt)
        self.cache.dec_swa_lock_only.assert_called_once_with(3, self.restored_receipt)
        self.assertTrue(req.swa_prefix_lock_released)

    def test_other_caches_keep_upstream_commit(self):
        self.queue.tree_cache = MagicMock()
        upstream_commit = MagicMock()
        tpu._commit_hicache_local_restore_to_req(
            upstream_commit, self.queue, self.decode_req
        )
        upstream_commit.assert_called_once_with(self.queue, self.decode_req)
        self.queue.tree_cache.dec_swa_lock_only.assert_not_called()


class TestInflightLoadGate(CustomTestCase):
    """A request reusing pages of an un-acked load waits for the ack."""

    def setUp(self):
        self.queue = DecodeHiCacheTransferMixin()
        self.queue.tree_cache = self.cache = full_only_cache()
        self.cache.page_size = 64
        self.cache.ongoing_load_back = {}
        self.cache.is_load_back_event_done.return_value = True
        self.controller = self.cache.cache_controller
        self.controller.page_size = 64
        tpu._load_device_pages[self.controller] = {}

    def issue_load(self, node_id, kv_indices):
        # The load publishes kv_indices in the tree before its DMA lands.
        upstream = MagicMock(return_value=kv_indices)
        tpu._controller_load(
            upstream, self.controller, host_indices=torch.arange(256), node_id=node_id
        )
        self.cache.ongoing_load_back[node_id] = object()

    @staticmethod
    def request(prefix_indices, *, host_len=0):
        return DecodeRequest(
            req=make_req(),
            kv_receiver=MagicMock(),
            prefix_match=DecodePrefixMatch(prefix_indices, host_len, 0, 2),
        )

    def process(self, decode_reqs):
        tpu._process_hicache_local_restores(
            DecodeHiCacheTransferMixin._process_hicache_local_restores,
            self.queue,
            decode_reqs,
        )
        return [decode_req.hicache_restore_status for decode_req in decode_reqs]

    def test_device_hit_on_loading_pages_waits_for_the_ack(self):
        self.issue_load(3, torch.arange(64, 320))
        # Admitted after the load: a device hit on the loading pages.
        reuser = self.request(torch.arange(320))
        unrelated = self.request(torch.arange(1024, 1088))
        pending, ready = HiCacheRestoreResult.PENDING, HiCacheRestoreResult.READY
        self.assertEqual(self.process([reuser, unrelated]), [pending, ready])
        self.assertEqual(self.process([reuser]), [pending])
        # The DMA landed and loading_check acked the anchor.
        del self.cache.ongoing_load_back[3]
        self.assertEqual(self.process([reuser]), [ready])
        self.assertEqual(tpu._load_device_pages[self.controller], {})

    def test_finished_restore_waits_for_a_load_it_rematched(self):
        # Admitted before the load: its restore rematch reused loading pages
        # and its own DMA finished first.
        self.issue_load(3, torch.arange(64, 320))
        restorer = self.request(torch.arange(64), host_len=256)
        restorer.hicache_restored_node = 5
        restorer.hicache_restored_kv_indices = torch.arange(64, 320)
        restorer.hicache_load_consumer_index = 0
        self.assertEqual(self.process([restorer]), [HiCacheRestoreResult.PENDING])
        del self.cache.ongoing_load_back[3]
        self.assertEqual(self.process([restorer]), [HiCacheRestoreResult.READY])

    def test_other_caches_and_controllers_keep_upstream(self):
        self.issue_load(3, torch.arange(64, 320))
        self.queue.tree_cache = MagicMock()
        reuser = self.request(torch.arange(320))
        self.assertEqual(self.process([reuser]), [HiCacheRestoreResult.READY])
        other_controller = MagicMock(page_size=64)
        upstream = MagicMock(return_value=torch.arange(64))
        tpu._controller_load(
            upstream, other_controller, host_indices=torch.arange(64), node_id=4
        )
        self.assertNotIn(other_controller, tpu._load_device_pages)


class TestDenseRestoreGate(CustomTestCase):
    """Non-SWA models share the hazard: a restore publishes pages before they land."""

    def dense_cache(self, controller_spec=TpuDMAHybridCacheController):
        cache = MagicMock(spec=UnifiedRadixCache)
        cache.cache_controller = MagicMock(spec=controller_spec)
        cache.cache_controller.page_size = 64
        cache.page_size = 64
        cache.ongoing_load_back = {}
        return cache

    def build(self, cache, **disagg_overrides):
        with patch(TPU + ".get_disagg", return_value=disagg(**disagg_overrides)):
            self.assertIs(
                tpu._create_tree_cache(MagicMock(return_value=cache), make_ctx()),
                cache,
            )

    def test_reuser_of_a_dense_restore_waits_for_the_ack(self):
        cache = self.dense_cache()
        self.build(cache)
        queue = DecodeHiCacheTransferMixin()
        queue.tree_cache = cache
        tpu._controller_load(
            MagicMock(return_value=torch.arange(64, 320)),
            cache.cache_controller,
            host_indices=torch.arange(256),
            node_id=3,
        )
        cache.ongoing_load_back[3] = object()
        reuser = TestInflightLoadGate.request(torch.arange(320))

        def process():
            tpu._process_hicache_local_restores(
                DecodeHiCacheTransferMixin._process_hicache_local_restores,
                queue,
                [reuser],
            )
            return reuser.hicache_restore_status

        self.assertEqual(process(), HiCacheRestoreResult.PENDING)
        del cache.ongoing_load_back[3]
        self.assertEqual(process(), HiCacheRestoreResult.READY)

    def test_auxiliary_only_restore_is_not_tracked(self):
        # A Mamba-state restore returns no KV pages; tracking it must not
        # break the in-flight page mask for other requests.
        cache = self.dense_cache()
        self.build(cache)
        tpu._controller_load(
            MagicMock(return_value=torch.empty(0, dtype=torch.int64)),
            cache.cache_controller,
            host_indices=torch.empty(0, dtype=torch.int64),
            node_id=4,
        )
        cache.ongoing_load_back[4] = object()
        self.assertIsNone(tpu._inflight_page_mask(cache))

    def test_only_raiden_decode_prefix_caches_are_gated(self):
        for name, cache, overrides in (
            ("prefill", self.dense_cache(), dict(mode="prefill")),
            ("decode radix off", self.dense_cache(), dict(decode_radix=False)),
            ("other controller", self.dense_cache(HybridCacheController), {}),
        ):
            with self.subTest(name):
                self.build(cache, **overrides)
                self.assertNotIn(cache, tpu._restore_gated_caches)
                self.assertNotIn(cache.cache_controller, tpu._load_device_pages)


class TestSwaMatchValidator(CustomTestCase):
    def validator(self, full_only):
        cache = MagicMock()
        cache.token_to_kv_pool_allocator = object()
        if full_only:
            mark_full_only(cache)
        component = object.__new__(SWAComponent)
        component.cache = cache
        component.sliding_window_size = 128
        return tpu._create_swa_match_validator(
            SWAComponent.create_match_validator, component
        )

    @staticmethod
    def node(*, live, evicted=False, backuped=False):
        value = torch.arange(64) if live else None
        return SimpleNamespace(
            key=[0] * 64,
            component_data={
                ComponentType.SWA: SimpleNamespace(value=value, host_value=None)
            },
            evicted=evicted,
            backuped=backuped,
        )

    def test_partial_live_window_does_not_end_a_full_only_match(self):
        # The previous turn left a live SWA tail shorter than the window right
        # before the new cap. Its full-attention KV must stay reusable.
        for full_only in (False, True):
            with self.subTest(full_only=full_only):
                validator = self.validator(full_only)
                self.assertEqual(validator(self.node(live=False)), full_only)
                self.assertEqual(validator(self.node(live=True)), full_only)

    def test_full_only_tombstone_needs_resident_or_backed_up_full_kv(self):
        validator = self.validator(True)
        self.assertTrue(validator(self.node(live=False, evicted=True, backuped=True)))
        self.assertFalse(validator(self.node(live=False, evicted=True)))


class TestRealLockAccounting(CustomTestCase):
    """Restore, commit and abort with the real unified tree lock accounting."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # TestCase.enterClassContext needs Python 3.11, CI runs 3.10.
        patches = installed_patches()
        patches.__enter__()
        cls.addClassCleanup(patches.__exit__, None, None, None)

    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="scheduler")

    def make_cache(self, *, is_eagle=False):
        cache, _ = real_swa_cache(is_eagle=is_eagle)
        mark_full_only(cache)
        return cache

    def assert_unlocked(self, cache):
        nodes = list(cache.tree_core.root_node.children.values())
        self.assertTrue(nodes)
        while nodes:
            node = nodes.pop()
            for component_type in cache.tree_components:
                data = node.component_data[component_type]
                self.assertEqual(data.lock_ref, 0, (node.id, component_type))
                self.assertEqual(data.host_lock_ref, 0, (node.id, component_type))
            nodes.extend(node.children.values())

    def test_commit_finish_and_abort_balance_full_and_swa_locks(self):
        # EAGLE trees key bigrams, and the restore must still cover its last page.
        for is_eagle, commit in itertools.product((False, True), (False, True)):
            with self.subTest(is_eagle=is_eagle, commit=commit):
                cache = self.make_cache(is_eagle=is_eagle)

                def key(num_tokens):
                    # How insert_req keys num_tokens tokens of KV.
                    token_ids = array("q", range(num_tokens + is_eagle))
                    return RadixKey(token_ids, is_bigram=is_eagle)

                prefix = cache.insert(
                    InsertParams(key=key(64), value=torch.arange(64))
                ).last_device_node
                # Full KV for 320 tokens whose SWA window was tombstoned.
                cache.insert(
                    InsertParams(
                        key=key(320),
                        value=torch.arange(320),
                        component_evicted_seqlens={ComponentType.SWA: 320},
                    )
                )
                req = make_req()
                req.session = None
                req.origin_input_ids = array("q", req.origin_input_ids)
                req.output_ids = array("q")
                req.session_id = None
                req.swa_branching_seqlen = None
                req.kv.set_evicted_seqlen(ComponentType.SWA, 320)
                req.kv.kv_committed_len = req.kv.kv_allocated_len = 512
                req.owned_kv_len = lambda: 512
                req.skip_radix_cache_insert = False
                req.priority = None
                req.last_node = prefix
                req.lock_receipt = cache.inc_lock_ref(prefix).to_dec_params()
                cache.dec_swa_lock_only(prefix, req.lock_receipt)
                req.swa_prefix_lock_released = True
                decode_req = DecodeRequest(
                    req=req,
                    kv_receiver=MagicMock(),
                    prefix_match=DecodePrefixMatch(torch.arange(64), 256, 0, prefix),
                )
                queue = DecodeHiCacheTransferMixin()
                queue.tree_cache = cache
                # The tombstoned prefix is device-resident: the full-only match
                # accepts it and no DMA is needed.
                self.assertFalse(queue._try_hicache_queue_load_back(decode_req))
                self.assertEqual(
                    decode_req.hicache_restore_status, HiCacheRestoreResult.READY
                )
                if commit:
                    queue._commit_hicache_local_restore_to_req(decode_req)
                    self.assertTrue(req.swa_prefix_lock_released)
                    req.kv.cache_protected_len = 320
                    release_kv_cache(req, cache)
                    queue._clean_hicache_prefetch_resources(decode_req)
                else:
                    queue._clean_hicache_prefetch_resources(decode_req)
                    release_kv_cache(req, cache, is_insert=False)
                self.assert_unlocked(cache)


def mark_window(cache):
    # As the full + SWA window host stack builder does.
    mark_full_only(cache)
    tpu._window_host_caches.add(cache)


class TestSwaWindowRestoreGate(CustomTestCase):
    """Only the offer restore loads SWA KV back, a request's own restore never."""

    @staticmethod
    def component(*, window):
        cache = MagicMock()
        if window:
            mark_window(cache)
        else:
            mark_full_only(cache)
        component = object.__new__(SWAComponent)
        component.cache = cache
        return component

    def test_swa_load_back_runs_only_inside_a_window_restore(self):
        # A request's own restore anchors at its SWA window start, usually a
        # tombstone, and upstream's SWA LOAD_BACK asserts on it.
        load_back = CacheTransferPhase.LOAD_BACK
        upstream = MagicMock(return_value=["swa"])
        component = self.component(window=True)
        build = tpu._build_swa_hicache_transfers
        self.assertIsNone(build(upstream, component, "node", load_back))
        upstream.assert_not_called()
        backup = CacheTransferPhase.BACKUP_HOST
        self.assertEqual(build(upstream, component, "node", backup), ["swa"])
        with tpu.swa_window_restore():
            self.assertEqual(build(upstream, component, "node", load_back), ["swa"])
        self.assertFalse(tpu._swa_window_restore)
        # Without host windows upstream decides, as it has no SWA host pool.
        upstream.reset_mock()
        build(upstream, self.component(window=False), "node", load_back)
        upstream.assert_called_once()

    def test_swa_host_hit_is_reported_only_inside_a_window_restore(self):
        result = MatchResult(
            device_indices=torch.arange(64),
            last_device_node=1,
            last_host_node=2,
            best_match_node=2,
            swa_host_hit_length=128,
        )
        upstream = MagicMock(return_value=result)
        component = self.component(window=True)
        finalize = tpu._finalize_swa_match_result
        self.assertEqual(
            finalize(upstream, component, result=result).swa_host_hit_length, 0
        )
        with tpu.swa_window_restore():
            self.assertEqual(
                finalize(upstream, component, result=result).swa_host_hit_length, 128
            )
        self.assertEqual(
            finalize(
                upstream, self.component(window=False), result=result
            ).swa_host_hit_length,
            128,
        )


class TestPathEndSwaBackup(CustomTestCase):
    """Host RAM keeps the last SWA window of each path and nothing above it."""

    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="scheduler")
        # Window 128 tokens, page 64.
        self.cache, _ = real_swa_cache()
        self.cache.tree_core.has_swa_host_pool = True
        mark_window(self.cache)
        self.swa = self.cache.components[ComponentType.SWA]

    def insert(self, num_tokens, **kwargs):
        self.cache.insert(
            InsertParams(
                key=RadixKey(array("q", range(num_tokens))),
                value=torch.arange(num_tokens),
                **kwargs,
            )
        )
        # The mocked allocator leaves live SWA values empty: one slot per token.
        for node in self.path():
            data = node.component_data[ComponentType.SWA]
            if data.value is not None:
                data.value = torch.arange(len(node.key))

    def path(self):
        """The nodes from the root down the first child at each level."""
        nodes = []
        node = self.cache.tree_core.root_node
        while node.children:
            node = next(iter(node.children.values()))
            nodes.append(node)
        return nodes

    def collect(self, node):
        return tpu._collect_unbacked_swa_nodes(
            SWAComponent._collect_unbacked_swa_nodes, self.swa, node
        )

    def lengths(self):
        return [len(node.key) for node in self.path()]

    def test_leaf_backs_up_its_window(self):
        # Insert splits the last window off as its own leaf.
        self.insert(320)
        self.assertEqual(self.lengths(), [192, 128])
        _, leaf = self.path()
        self.assertEqual(self.collect(leaf), [leaf])

    def test_stale_internal_windows_are_not_backed_up(self):
        self.insert(128)
        self.insert(320)
        self.assertEqual(self.lengths(), [128, 64, 128])
        far, near, _ = self.path()
        # Upstream copies a window ending at each, though the path goes on
        # for a whole window past them (exactly one for ``near``).
        upstream = SWAComponent._collect_unbacked_swa_nodes
        self.assertEqual(upstream(self.swa, near), [near, far])
        self.assertEqual(upstream(self.swa, far), [far])
        self.assertEqual(self.collect(near), [])
        self.assertEqual(self.collect(far), [])

    def test_window_above_a_tombstone_is_not_a_path_end(self):
        # The path end lost its window, so the part above it is no window either.
        self.insert(128)
        self.insert(192, component_evicted_seqlens={ComponentType.SWA: 192})
        parent, child = self.path()
        self.assertIsNone(child.component_data[ComponentType.SWA].value)
        self.assertEqual(self.collect(parent), [])

    def test_split_leaf_keeps_the_whole_last_window(self):
        self.insert(256)
        # A shorter match splits the window leaf: the last window now spans
        # the new leaf and its new parent.
        self.cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", range(192)))))
        self.assertEqual(self.lengths(), [128, 64, 64])
        stale, split, leaf = self.path()
        self.assertEqual(self.collect(split), [split])
        self.assertEqual(self.collect(stale), [])
        self.assertEqual(self.collect(leaf), [leaf, split])
        # Once the split part is in RAM only the leaf is left.
        split.component_data[ComponentType.SWA].host_value = torch.arange(64)
        self.assertEqual(self.collect(leaf), [leaf])


class TestPlanWindowRestore(CustomTestCase):
    @staticmethod
    def node(*, parent, num_tokens, evicted):
        full = SimpleNamespace(host_value=torch.arange(num_tokens) if evicted else None)
        return SimpleNamespace(
            key=[0] * num_tokens,
            parent=parent,
            evicted=evicted,
            component_data={ComponentType.FULL: full},
        )

    def plan(self, *, on_device, prompt_len=300):
        root = SimpleNamespace()
        resident = self.node(parent=root, num_tokens=128, evicted=False)
        in_ram = self.node(parent=resident, num_tokens=128, evicted=True)
        cache = MagicMock()
        cache.tree_core.root_node = root
        cache.tree_core.is_eagle = False
        cache.tree_core.node_by_id.return_value = in_ram
        flags = []

        def match_prefix(params):
            flags.append(tpu._swa_window_restore)
            self.assertEqual(len(params.key), 256)
            return SimpleNamespace(best_match_node=5, swa_host_hit_length=128)

        cache.match_prefix.side_effect = match_prefix
        req = SimpleNamespace(
            origin_input_ids=list(range(prompt_len)), extra_key=None, cache_salt=None
        )
        with patch.object(
            restore, "deepest_device_window_prefix", return_value=on_device
        ):
            plan = restore._plan_restore(
                cache=cache, req=req, window_size=128, page_size=64
            )
        return plan, flags

    def test_loads_only_a_window_deeper_than_the_device_one(self):
        # The strict match ends 256 tokens in, the last 128 of them in RAM.
        plan, flags = self.plan(on_device=128)
        self.assertEqual((plan.node, plan.full_tokens, plan.swa_tokens), (5, 128, 128))
        self.assertEqual(flags, [True])
        self.assertIsNone(self.plan(on_device=256)[0])

    def test_prompt_shorter_than_a_page_needs_no_match(self):
        plan, flags = self.plan(on_device=0, prompt_len=64)
        self.assertIsNone(plan)
        self.assertEqual(flags, [])


class TestOfferWindowRestore(CustomTestCase):
    """The offer restore state machine and its lock pairing, on every rank alike."""

    def setUp(self):
        self.events = []
        self.load_ok = True
        self.cache = cache = MagicMock()
        mark_window(cache)
        cache.ongoing_load_back = {}
        cache.has_free_load_back_slot.return_value = True
        cache.full_evictable_size.return_value = 0
        cache.swa_evictable_size.return_value = 0
        cache.load_back.side_effect = self.load_back
        cache.inc_lock_ref.side_effect = self.lock
        cache.dec_lock_ref.side_effect = lambda node, receipt: self.events.append(
            ("unlock", node, receipt)
        )
        self.allocator = MagicMock(page_size=64)
        self.allocator.full_available_size.return_value = 10**6
        self.allocator.swa_available_size.return_value = 10**6
        self.queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        self.queue.tree_cache = cache
        self.queue.scheduler = SimpleNamespace(
            enable_decode_hicache=True,
            token_to_kv_pool_allocator=self.allocator,
            sliding_window_size=128,
        )
        self.plans = {}
        patcher = patch.object(
            restore,
            "_plan_restore",
            side_effect=lambda *, cache, req, window_size, page_size: self.plans[
                req.rid
            ],
        )
        self.plan_restore = patcher.start()
        self.addCleanup(patcher.stop)

    def load_back(self, node):
        # SWA load-back runs only under the flag.
        self.assertTrue(tpu._swa_window_restore)
        self.events.append(("load", node))
        if self.load_ok:
            self.cache.ongoing_load_back[node] = object()
        return self.load_ok

    def lock(self, node):
        self.events.append(("lock", node))
        return MagicMock(to_dec_params=MagicMock(return_value=("receipt", node)))

    def request(self, rid, *, node=None, status=HiCacheRestoreResult.READY):
        req = make_req()
        req.rid = rid
        self.plans[rid] = (
            None
            if node is None
            else restore._RestorePlan(node=node, full_tokens=256, swa_tokens=128)
        )
        return DecodeRequest(
            req=req,
            kv_receiver=MagicMock(prefix_offer_pending=True),
            prefix_match=DecodePrefixMatch(torch.arange(64), 0, 0, 1),
            hicache_restore_status=status,
        )

    def offer(self, decode_reqs):
        """One scheduler step, returns the rids offered in it."""
        offered = []

        def upstream(queue, reqs):
            for decode_req in reqs:
                # Upstream waits for the request's own restore.
                if (
                    decode_req.prefix_offer_sent
                    or decode_req.hicache_restore_status == HiCacheRestoreResult.PENDING
                ):
                    continue
                self.events.append(("offer", decode_req.req.rid))
                decode_req.prefix_offer_sent = True
                offered.append(decode_req.req.rid)

        restore._offer_resident_prefixes(upstream, self.queue, decode_reqs)
        return offered

    def unlocks(self):
        return [event for event in self.events if event[0] == "unlock"]

    def test_offer_waits_for_the_load_then_takes_over_its_lock(self):
        decode_req = self.request("a", node=7)
        self.assertEqual(self.offer([decode_req]), [])
        self.assertEqual(self.events, [("load", 7), ("lock", 7)])
        self.cache.ready_to_load_host_cache.assert_called_once()
        # Still loading on some rank: no offer, no second load.
        self.assertEqual(self.offer([decode_req]), [])
        self.assertEqual(len(self.events), 2)
        # loading_check reaped the ack on every rank.
        del self.cache.ongoing_load_back[7]
        self.assertEqual(self.offer([decode_req]), ["a"])
        # The restore lock goes in the step the offer locks the path.
        self.assertEqual(
            self.events[2:], [("offer", "a"), ("unlock", 7, ("receipt", 7))]
        )
        self.offer([decode_req])
        restore._release_prefix_offer_lock(MagicMock(), self.queue, decode_req)
        self.assertEqual(len(self.unlocks()), 1)
        self.cache.ready_to_load_host_cache.assert_called_once()

    def test_nothing_to_restore_offers_at_once(self):
        # The device already holds the deepest window.
        self.assertEqual(self.offer([self.request("a")]), ["a"])
        # The request's own restore goes first: no window restore meanwhile.
        pending = self.request("b", node=7, status=HiCacheRestoreResult.PENDING)
        self.assertEqual(self.offer([pending]), [])
        self.assertEqual(self.events, [("offer", "a")])
        self.cache.ready_to_load_host_cache.assert_not_called()

    def test_declined_or_unfitting_load_offers_the_device_prefix(self):
        self.load_ok = False
        self.assertEqual(self.offer([self.request("a", node=7)]), ["a"])
        self.assertEqual(self.events, [("load", 7), ("offer", "a")])
        self.load_ok = True
        # A whole window must fit the SWA pool, free or evictable.
        self.allocator.swa_available_size.return_value = 64
        self.assertEqual(self.offer([self.request("b", node=8)]), ["b"])
        self.assertEqual(self.events[2:], [("offer", "b")])
        self.assertNotIn("lock", [event[0] for event in self.events])
        self.cache.ready_to_load_host_cache.assert_not_called()

    def loads(self):
        return [event for event in self.events if event[0] == "load"]

    def test_loads_are_capped(self):
        reqs = [self.request(rid, node=node) for rid, node in zip("abc", (7, 8, 9))]
        # The third waits for a loading restore to land, not offered shallower.
        self.assertEqual(self.offer(reqs), [])
        self.assertEqual(self.loads(), [("load", 7), ("load", 8)])
        del self.cache.ongoing_load_back[7]
        self.assertEqual(self.offer(reqs), ["a"])
        self.assertEqual(self.loads()[2:], [("load", 9)])

    def test_load_waits_for_a_free_slot(self):
        # The slot count agrees on every rank, acks are reaped in lockstep.
        self.cache.has_free_load_back_slot.return_value = False
        decode_req = self.request("a", node=7)
        self.assertEqual(self.offer([decode_req]), [])
        self.assertEqual(self.loads(), [])
        self.cache.has_free_load_back_slot.return_value = True
        self.assertEqual(self.offer([decode_req]), [])
        self.assertEqual(self.loads(), [("load", 7)])

    def test_queue_exit_releases_a_loading_restore_once(self):
        decode_req = self.request("a", node=7)
        self.offer([decode_req])
        release = MagicMock()
        restore._release_prefix_offer_lock(release, self.queue, decode_req)
        release.assert_called_once_with(self.queue, decode_req)
        self.assertEqual(self.unlocks(), [("unlock", 7, ("receipt", 7))])
        restore._release_prefix_offer_lock(release, self.queue, decode_req)
        self.assertEqual(len(self.unlocks()), 1)

    def test_caches_without_host_windows_keep_upstream_offer(self):
        self.queue.tree_cache = full_only_cache()
        upstream = MagicMock()
        reqs = [self.request("a", node=7)]
        restore._offer_resident_prefixes(upstream, self.queue, reqs)
        upstream.assert_called_once_with(self.queue, reqs)
        self.plan_restore.assert_not_called()


class TestInstall(CustomTestCase):
    def test_install_wraps_every_target_once_and_restores(self):
        with installed_patches() as originals:
            tpu.install()
            for owner, name, original in originals:
                with self.subTest(target=f"{owner.__name__}.{name}"):
                    self.assertIs(getattr(owner, name).__wrapped__, original)
            self.assertEqual(
                hybrid_pool_assembler._STRATEGIES.count(tpu._FULL_ONLY_SWA_STRATEGY), 1
            )
        for owner, name, original in originals:
            self.assertIs(getattr(owner, name), original)


if __name__ == "__main__":
    unittest.main()
