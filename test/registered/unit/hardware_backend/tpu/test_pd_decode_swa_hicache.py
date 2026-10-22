"""Unit tests for the TPU PD decode SWA HiCache patch (pd_decode_swa_hicache).

Decode backs up and restores full-attention KV through host RAM, and the SWA
window always comes from the prefill transfer. Each wrapper is called with the
real upstream function (or a fake where upstream needs a full server), so the
tests pin both the full-only behavior and the pass-through to upstream for
every other cache, except the restore hold, which every raiden PD decode cache
gets. Two tests drive a real unified tree: its host stack attach
and its locks.
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
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeRequest
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
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import HiCacheDraftMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import pd_decode_swa_hicache as tpu
from sglang_tpu.raiden_hybrid_cache_controller import TpuRaidenHybridCacheController

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

    def test_only_a_full_only_build_attaches_one_full_pool(self):
        cache, params = real_swa_cache()
        kvcache = MagicMock(spec=SWAKVPool)
        # SWA at layer 0, full attention at layer 1.
        kvcache.start_layer = 0
        kvcache.layers_mapping = {0: (0, True), 1: (0, False)}
        kvcache.full_kv_pool = MagicMock()
        cache.token_to_kv_pool_allocator.get_kvcache.return_value = kvcache
        components = {ComponentType.FULL, ComponentType.SWA}
        with (
            installed_patches(),
            patch(TPU + ".get_memory", return_value=memory()),
            patch(ASSEMBLER + ".build_kv_host_pool", autospec=True) as host_pool,
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
        self.assertEqual(set(cache.host_pool_group.entry_map), {PoolName.KV})
        self.assertIs(
            cache.components[ComponentType.FULL]._full_kv_pool_host,
            host_pool.return_value,
        )
        # init_hicache derives has_swa_host_pool=False from this.
        self.assertIsNone(cache.components[ComponentType.SWA]._swa_kv_pool_host)


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

    def dense_cache(self, controller_spec=TpuRaidenHybridCacheController):
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
