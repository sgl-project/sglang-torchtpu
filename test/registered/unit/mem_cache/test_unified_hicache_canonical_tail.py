"""Canonical KV contents through real host-backed duplicate insertion."""

import contextlib
import unittest
from array import array
from dataclasses import dataclass
from unittest import mock

import torch
from test_unified_radix_cache_unittest import (
    CacheConfig,
    UnifiedRadixCacheSuite,
    _write_backup,
    build_fixture,
)

from sglang.srt.hardware_backend.tpu.kv_free_fence import KVFreeFence
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.srt.platforms import current_platform
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

FULL, SWA = ComponentType.FULL, ComponentType.SWA
MAMBA = ComponentType.MAMBA
PAGE = 4


class CappedRestoredWindow(SWAComponent):
    def recover_after_unevict(self, **kwargs):
        # Select the original independent-component cap only at its recovery
        # boundary. The shared allocator and all ownership methods stay real.
        with mock.patch.object(self, "_independent", True):
            return super().recover_after_unevict(**kwargs)


@dataclass
class CanonicalFixture:
    cache: object
    allocator: object
    req_pool: object
    cfg: CacheConfig

    def key(self, tokens):
        return RadixKey(array("q", tokens))

    def allocate(self, tokens, base):
        indices = UnifiedRadixCacheSuite._alloc(self, self.allocator, len(tokens))
        assert indices is not None
        self.seed(indices, base)
        return indices

    def seed(self, indices, base):
        pool = self.allocator.get_kvcache()
        pools = [(pool.full_kv_pool, indices, 0)]
        if self.cfg.has_swa:
            pools.append(
                (
                    pool.swa_kv_pool,
                    self.allocator.translate_loc_from_full_to_swa(indices),
                    1000,
                )
            )
        for component, locations, offset in pools:
            for buffer, extra in (
                (component.get_key_buffer(0), 0),
                (component.get_value_buffer(0), 500),
            ):
                values = torch.arange(len(indices), dtype=torch.float32)
                values += base + offset + extra
                buffer[locations.long()] = values[:, None, None]

    def read(self, indices):
        pool = self.allocator.get_kvcache()
        pools = [(pool.full_kv_pool, indices)]
        if self.cfg.has_swa:
            pools.append(
                (
                    pool.swa_kv_pool,
                    self.allocator.translate_loc_from_full_to_swa(indices),
                )
            )
        return tuple(
            buffer[locations.long()].clone()
            for component, locations in pools
            for buffer in (component.get_key_buffer(0), component.get_value_buffer(0))
        )

    def insert(self, tokens, indices, **kwargs):
        return self.cache.insert(
            InsertParams(key=self.key(tokens), value=indices, **kwargs)
        )

    def match(self, tokens):
        return self.cache.match_prefix(MatchPrefixParams(key=self.key(tokens)))

    def backup(self, tokens):
        node = self.node_for(tokens)
        assert _write_backup(self.cache, node, write_back=False) > 0
        self.cache.writing_check(write_back=True)
        assert not self.cache.ongoing_write_through
        return node

    def node_for(self, tokens):
        node = self.cache.tree_core.root_node
        key = self.key(tokens)
        while len(key):
            node = node.children[key.child_key(self.cfg.page_size)]
            prefix = node.key.match(key, page_size=self.cfg.page_size)
            assert prefix == len(node.key)
            key = key[prefix:]
        return node.id

    def host_bytes(self):
        return {
            name: tuple(
                buffer.clone()
                for buffer in (
                    entry.host_pool.kv_buffer
                    if isinstance(entry.host_pool.kv_buffer, list)
                    else [entry.host_pool.kv_buffer]
                )
            )
            for name, entry in self.cache.cache_controller.mem_pool_host.entry_map.items()
        }

    def nodes(self):
        return [
            self.cache.tree_core.node_by_id(node_id)
            for node_id in self.cache.tree_core.get_all_node_ids()
            if node_id != self.cache.root_node_handle()
        ]


@contextlib.contextmanager
def canonical_fixture(*, window=2 * PAGE, independent=False, mamba=False):
    """Use the shared real Python tree, independent pools and host controller."""
    import test_unified_radix_cache_unittest as fixtures

    import sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler as assembler
    import sglang.srt.mem_cache.pool_host.mamba as mamba_host_memory
    import sglang.srt.mem_cache.pool_host.mha as host_memory
    import sglang.srt.mem_cache.unified_radix_cache as cache_module

    cfg = CacheConfig(
        page_size=1 if mamba else PAGE,
        components=(FULL, MAMBA) if mamba else (FULL, SWA),
        num_layers=2,
        full_attention_layer_ids=(0,),
        sliding_window_size=None if mamba else window,
        kv_size=128,
        max_context_len=128,
        head_num=1,
        head_dim=8,
        dtype=torch.float32,
        mamba_cache_size=8,
        mamba_intermediate_size=16,
        mamba_num_heads=2,
        mamba_head_dim=4,
        mamba_state_size=4,
        mamba_conv_kernel=3,
    )
    with (
        mock.patch.object(fixtures, "get_device", return_value="cpu"),
        mock.patch.object(fixtures, "_TREE_CORE_TEST_BACKEND", "python"),
        mock.patch.dict(
            cache_module.COMPONENT_REGISTRY,
            {SWA: CappedRestoredWindow if independent else SWAComponent},
        ),
    ):
        cache, allocator, req_pool = build_fixture(cfg)
    host_class = current_platform.get_mha_kv_pool_host_cls

    def unpinned(pool):
        cls = host_class(pool)

        def create(*args, **kwargs):
            kwargs["pin_memory"] = False
            return cls(*args, **kwargs)

        return create

    args = ServerArgs(
        model_path="dummy",
        page_size=cfg.page_size,
        hicache_io_backend="direct",
        hicache_mem_layout="layer_first",
        hicache_write_policy="write_through",
        hicache_ratio=2,
    )
    set_global_server_args_for_scheduler(args)
    mamba_host_class = assembler.MambaPoolHost

    def unpinned_mamba(*args, **kwargs):
        kwargs.update(pin_memory=False, layout="page_first_direct")
        return mamba_host_class(*args, **kwargs)

    with (
        mock.patch.object(
            current_platform, "get_mha_kv_pool_host_cls", side_effect=unpinned
        ),
        mock.patch.object(assembler, "MambaPoolHost", side_effect=unpinned_mamba),
    ):
        cache.init_hicache(args, cache.cache_init_params)
    cache.write_through_threshold = 1 << 30
    cache.load_back_threshold = 0
    for entry in cache.cache_controller.mem_pool_host.entry_map.values():
        buffers = entry.host_pool.kv_buffer
        for buffer in buffers if isinstance(buffers, list) else [buffers]:
            buffer.zero_()

    def copy_rows(*, src_layers, dst_layers, src_indices, dst_indices, page_size):
        assert len(src_indices) == len(dst_indices)
        assert len(src_indices) % page_size == 0
        for source, destination in zip(src_layers, dst_layers, strict=True):
            destination[dst_indices.long()] = source[src_indices.long()]

    def copy_layer_to_page(*, src_ptrs, dst_ptrs, src_indices, dst_indices, page_size):
        assert page_size == 1 and len(dst_ptrs) == 1
        for layer, source in enumerate(src_ptrs):
            dst_ptrs[0][dst_indices.long(), layer, 0] = source[src_indices.long()]

    def copy_page_to_layer(
        *, src_ptrs, dst_ptrs, src_indices, dst_indices, layer_id, page_size
    ):
        assert page_size == 1 and len(src_ptrs) == len(dst_ptrs) == 1
        dst_ptrs[0][dst_indices.long()] = src_ptrs[0][src_indices.long(), layer_id, 0]

    try:
        # Only the GPU copy primitive is replaced. Index resolution, host
        # allocation, tree commits, controller queues and completion remain real.
        with (
            mock.patch.object(
                host_memory, "transfer_kv_direct", side_effect=copy_rows, create=True
            ),
            mock.patch.object(
                mamba_host_memory,
                "transfer_kv_direct",
                side_effect=copy_rows,
                create=True,
            ),
            mock.patch.object(
                mamba_host_memory,
                "transfer_kv_all_layer_direct_lf_pf",
                side_effect=copy_layer_to_page,
                create=True,
            ),
            mock.patch.object(
                mamba_host_memory,
                "transfer_kv_per_layer_direct_pf_lf",
                side_effect=copy_page_to_layer,
                create=True,
            ),
        ):
            yield CanonicalFixture(cache, allocator, req_pool, cfg)
    finally:
        fixtures._drop_hicache_atexit_pin(cache)
        cache.release_host_resources()


class TestCanonicalDuplicate(unittest.TestCase):
    def assert_contents(self, expected, actual):
        for name, want, got in zip(
            ("FULL K", "FULL V", "SWA K", "SWA V"), expected, actual, strict=True
        ):
            self.assertTrue(
                torch.equal(want, got),
                f"{name}: adopted fresh B instead of canonical host A",
            )

    def assert_host_unchanged(self, fixture, expected):
        actual = fixture.host_bytes()
        self.assertEqual(expected.keys(), actual.keys())
        for name in expected:
            for saved, current in zip(expected[name], actual[name], strict=True):
                self.assertTrue(torch.equal(saved, current))

    def assert_no_load_ticket(self, fixture):
        self.assertFalse(fixture.cache.ongoing_load_back)
        self.assertFalse(fixture.cache.cache_controller.load_queue)
        self.assertFalse(fixture.cache.cache_controller.ack_load_queue)

    def test_host_backed_duplicate_preserves_canonical_bytes(self):
        with canonical_fixture() as fixture:
            tokens = list(range(1, 2 * PAGE + 1))
            old = fixture.allocate(tokens, 100)
            expected = fixture.read(old)
            fixture.insert(tokens, old)
            node = fixture.match(tokens).last_device_node
            self.assertTrue(fixture.cache.backup_node_for_write_back(node))
            host_before = fixture.host_bytes()
            fixture.cache.evict(EvictParams(num_tokens=len(tokens)))
            self.assertTrue(fixture.cache.tree_core.is_full_device_evicted(node))
            incoming = fixture.allocate(tokens, 9000)
            incoming_before = fixture.read(incoming)
            self.assertTrue(
                all(not torch.equal(a, b) for a, b in zip(expected, incoming_before))
            )
            available = fixture.allocator.available_size()
            fixture.insert(tokens, incoming)
            adopted = fixture.match(tokens).device_indices
            self.assertEqual(adopted.tolist(), incoming.tolist())
            actual = fixture.read(adopted)
            self.assert_contents(expected, actual)
            self.assertEqual(fixture.allocator.available_size(), available)
            self.assert_host_unchanged(fixture, host_before)
            self.assert_no_load_ticket(fixture)
            fixture.cache.sanity_check()

    def test_resident_and_unbacked_duplicates_keep_their_owned_version(self):
        for backed in (False, True):
            with self.subTest(backed=backed), canonical_fixture() as fixture:
                tokens = list(range(1, 2 * PAGE + 1))
                old = fixture.allocate(tokens, 100)
                fixture.insert(tokens, old)
                if backed:
                    fixture.backup(tokens)
                else:
                    fixture.cache.evict(EvictParams(num_tokens=len(tokens)))
                incoming = fixture.allocate(tokens, 9000)
                expected = fixture.read(old if backed else incoming)
                fixture.insert(tokens, incoming)
                adopted = fixture.match(tokens).device_indices
                self.assertEqual(
                    adopted.tolist(), (old if backed else incoming).tolist()
                )
                self.assert_contents(expected, fixture.read(adopted))
                self.assert_no_load_ticket(fixture)
                fixture.cache.sanity_check()

    def test_missing_swa_backup_preserves_full_and_keeps_fresh_swa(self):
        with canonical_fixture() as fixture:
            tokens = list(range(1, 2 * PAGE + 1))
            old = fixture.allocate(tokens, 100)
            expected_full = fixture.read(old)[:2]
            fixture.allocator.free_swa(old)
            fixture.insert(tokens, old, component_evicted_seqlens={SWA: len(tokens)})
            node = fixture.backup(tokens)
            self.assertIsNone(
                fixture.cache.tree_core.node_by_id(node).component_data[SWA].host_value
            )
            fixture.cache.evict(EvictParams(num_tokens=len(tokens)))
            incoming = fixture.allocate(tokens, 9000)
            fresh_swa = fixture.read(incoming)[2:]
            fixture.insert(tokens, incoming)
            adopted = fixture.match(tokens).device_indices
            self.assert_contents(expected_full + fresh_swa, fixture.read(adopted))
            self.assert_no_load_ticket(fixture)
            fixture.cache.sanity_check()

    def test_host_backed_overlap_restores_only_the_shared_logical_slice(self):
        with canonical_fixture() as fixture:
            tokens = list(range(1, 2 * PAGE + 1))
            old = fixture.allocate(tokens, 100)
            expected = fixture.read(old)
            fixture.insert(tokens, old)
            fixture.backup(tokens)
            host_before = fixture.host_bytes()
            fixture.cache.evict(EvictParams(num_tokens=len(tokens)))
            branch = tokens[:PAGE] + list(range(100, 100 + PAGE))
            incoming = fixture.allocate(branch, 9000)
            fresh = fixture.read(incoming)
            fixture.insert(branch, incoming)
            adopted = fixture.match(branch).device_indices
            mixed = tuple(
                torch.cat((a[:PAGE], b[PAGE:]))
                for a, b in zip(expected, fresh, strict=True)
            )
            self.assert_contents(mixed, fixture.read(adopted))
            self.assert_host_unchanged(fixture, host_before)
            self.assert_no_load_ticket(fixture)
            fixture.cache.sanity_check()

    def test_swa_restore_respects_the_existing_page_aligned_floor(self):
        with canonical_fixture(window=PAGE) as fixture:
            tokens = list(range(1, 2 * PAGE + 1))
            old = fixture.allocate(tokens, 100)
            expected = fixture.read(old)
            fixture.insert(tokens, old)
            fixture.backup(tokens)
            host_before = fixture.host_bytes()
            fixture.cache.evict(EvictParams(num_tokens=len(tokens)))
            incoming = fixture.allocate(tokens, 9000)
            fixture.allocator.free_swa(incoming[:PAGE])
            fixture.insert(tokens, incoming, component_evicted_seqlens={SWA: PAGE})
            adopted = fixture.match(tokens).device_indices
            self.assertEqual(adopted.tolist(), incoming.tolist())
            actual = fixture.read(adopted)
            self.assert_contents(
                expected[:2] + tuple(a[PAGE:] for a in expected[2:]),
                actual[:2] + tuple(a[PAGE:] for a in actual[2:]),
            )
            self.assertTrue(
                (
                    fixture.allocator.translate_loc_from_full_to_swa(adopted[:PAGE])
                    == 0
                ).all()
            )
            self.assert_host_unchanged(fixture, host_before)
            fixture.cache.sanity_check()

    def _long_backed_duplicate(self, fixture):
        tokens = list(range(1, 3 * PAGE + 1))
        tombstone = fixture.allocate(tokens, 100)
        fixture.allocator.free_swa(tombstone)
        fixture.insert(tokens, tombstone, component_evicted_seqlens={SWA: len(tokens)})
        old = fixture.allocate(tokens, 100)
        fixture.insert(tokens, old)
        node = fixture.cache.tree_core.node_by_id(
            fixture.match(tokens).last_device_node
        )
        self.assertEqual(len(node.key), len(tokens))
        self.assertEqual(len(node.component_data[SWA].value), len(tokens))
        expected = fixture.read(fixture.match(tokens).device_indices)
        fixture.backup(tokens)
        host_before = fixture.host_bytes()
        fixture.cache.evict(EvictParams(num_tokens=len(tokens)))
        self.assertTrue(fixture.cache.tree_core.is_full_device_evicted(node.id))
        incoming = fixture.allocate(tokens, 9000)
        return tokens, expected, incoming, host_before

    def test_long_backed_overlap_pins_every_fragment_until_completion(self):
        with canonical_fixture(independent=True) as fixture:
            tokens, expected, incoming, host_before = self._long_backed_duplicate(
                fixture
            )
            order, covered_fragments = [], set()
            fixture.allocator.kv_free_fence = KVFreeFence(lambda: order.append("fence"))
            engine = fixture.cache.cache_controller.l2_transfer_engine
            fixture.cache.cache_controller.load_fence_stream = object()
            original_submit = engine.submit_host_to_device

            def submit(transfers, **kwargs):
                order.append("submit")
                completion = original_submit(transfers, **kwargs)
                original_finish = completion.finish_event

                def finish():
                    order.append("finish")
                    entries = fixture.cache.cache_controller.mem_pool_host.entry_map
                    for transfer in transfers:
                        ct = (
                            FULL
                            if transfer.host_pool is entries[PoolName.KV].host_pool
                            else SWA
                        )
                        needed = set(transfer.host_indices.tolist())
                        covered = set()
                        for node in fixture.nodes():
                            cd = node.component_data[ct]
                            if cd.host_value is None:
                                continue
                            overlap = needed.intersection(cd.host_value.tolist())
                            if overlap:
                                self.assertGreater(cd.host_lock_ref, 0)
                                covered.update(overlap)
                                if ct == SWA:
                                    covered_fragments.add(node.id)
                        self.assertEqual(covered, needed)
                    original_finish.synchronize()

                event = mock.Mock()
                event.synchronize.side_effect = finish
                return completion._replace(finish_event=event)

            with (
                mock.patch.object(engine, "submit_host_to_device", side_effect=submit),
                mock.patch.object(
                    engine.host_to_device_stream,
                    "wait_stream",
                    side_effect=lambda _: order.append("stream"),
                    create=True,
                ),
                mock.patch.object(
                    fixture.allocator.full_attn_allocator,
                    "alloc",
                    side_effect=AssertionError("restore allocated FULL"),
                ),
                mock.patch.object(
                    fixture.allocator.swa_attn_allocator,
                    "alloc",
                    side_effect=AssertionError("restore allocated SWA"),
                ),
            ):
                fixture.insert(tokens, incoming)
            self.assert_contents(
                expected, fixture.read(fixture.match(tokens).device_indices)
            )
            self.assertEqual(order, ["fence", "stream", "submit", "finish"])
            self.assertGreaterEqual(len(covered_fragments), 2)
            for node in fixture.nodes():
                for cd in node.component_data:
                    self.assertEqual(cd.host_lock_ref, 0)
            self.assert_host_unchanged(fixture, host_before)
            self.assert_no_load_ticket(fixture)
            fixture.cache.sanity_check()

    def test_partial_swa_eviction_preserves_backed_full_and_swa(self):
        for locked in (False, True):
            with self.subTest(locked=locked), canonical_fixture() as fixture:
                tokens = list(range(1, 2 * PAGE + 1))
                old = fixture.allocate(tokens, 100)
                expected = fixture.read(old)
                fixture.insert(tokens, old)
                node = fixture.backup(tokens)
                child_tokens = list(range(100, 100 + 2 * PAGE))
                child = fixture.allocate(child_tokens, 200)
                fixture.insert(
                    tokens + child_tokens,
                    torch.cat((old, child)),
                    prev_prefix_len=len(tokens),
                    component_evicted_seqlens={SWA: len(tokens)},
                )
                fixture.cache.evict(EvictParams(num_tokens=0, swa_num_tokens=PAGE))
                core_node = fixture.cache.tree_core.node_by_id(node)
                self.assertTrue(core_node.children)
                self.assertFalse(fixture.cache.tree_core.is_full_device_evicted(node))
                self.assertIsNone(core_node.component_data[SWA].value)
                self.assertIsNotNone(core_node.component_data[FULL].host_value)
                self.assertIsNotNone(core_node.component_data[SWA].host_value)
                receipt = (
                    fixture.cache.inc_lock_ref(node).to_dec_params() if locked else None
                )
                try:
                    incoming = fixture.allocate(tokens, 9000)
                    fixture.insert(tokens, incoming)
                    adopted = fixture.match(tokens).device_indices
                    self.assertEqual(
                        adopted.tolist(), (old if locked else incoming).tolist()
                    )
                    self.assert_contents(expected, fixture.read(adopted))
                    self.assert_no_load_ticket(fixture)
                    fixture.cache.sanity_check()
                finally:
                    if receipt is not None:
                        fixture.cache.dec_lock_ref(node, receipt)

    def test_split_host_receipts_release_on_restore_failure(self):
        for failure in ("second_lock", "submit", "finish"):
            with (
                self.subTest(failure=failure),
                canonical_fixture(independent=True) as fixture,
            ):
                tokens, _, incoming, host_before = self._long_backed_duplicate(fixture)
                engine = fixture.cache.cache_controller.l2_transfer_engine
                original_lock = fixture.cache.inc_host_lock_ref
                original_submit = engine.submit_host_to_device
                calls = 0

                def acquire(node, *, case=failure, acquire_lock=original_lock):
                    nonlocal calls
                    calls += 1
                    if case == "second_lock" and calls == 2:
                        raise RuntimeError("second host lock")
                    return acquire_lock(node)

                def submit(
                    transfers, *, case=failure, submit_load=original_submit, **kwargs
                ):
                    if case == "submit":
                        raise RuntimeError("host submission")
                    completion = submit_load(transfers, **kwargs)
                    original_finish = completion.finish_event

                    def finish():
                        original_finish.synchronize()
                        raise RuntimeError("host completion")

                    event = mock.Mock()
                    event.synchronize.side_effect = finish
                    return completion._replace(finish_event=event)

                with (
                    mock.patch.object(
                        fixture.cache, "inc_host_lock_ref", side_effect=acquire
                    ),
                    mock.patch.object(
                        engine, "submit_host_to_device", side_effect=submit
                    ),
                    self.assertRaisesRegex(
                        RuntimeError,
                        {
                            "second_lock": "second host lock",
                            "submit": "host submission",
                            "finish": "host completion",
                        }[failure],
                    ),
                ):
                    fixture.insert(tokens, incoming)
                for node in fixture.nodes():
                    for cd in node.component_data:
                        self.assertEqual(cd.host_lock_ref, 0)
                self.assert_host_unchanged(fixture, host_before)
                self.assert_no_load_ticket(fixture)

    def test_full_mamba_keeps_original_fresh_duplicate_behavior(self):
        with canonical_fixture(mamba=True) as fixture:
            tokens = list(range(1, 2 * PAGE + 1))
            pool = fixture.req_pool.mamba_pool

            def allocate_mamba(base):
                slot = fixture.req_pool.mamba_allocator.alloc(1)
                self.assertIsNotNone(slot)
                for buffer in [pool.mamba_cache.temporal, *pool.mamba_cache.conv]:
                    buffer[:, slot] = base
                return slot

            def read_mamba(slot):
                return tuple(
                    buffer[:, slot].clone()
                    for buffer in [pool.mamba_cache.temporal, *pool.mamba_cache.conv]
                )

            old = fixture.allocate(tokens, 100)
            old_state = allocate_mamba(100)
            fixture.insert(tokens, old, mamba_value=old_state)
            node = fixture.backup(tokens)
            host_before = fixture.host_bytes()
            fixture.cache.evict(EvictParams(num_tokens=len(tokens)))
            core_node = fixture.cache.tree_core.node_by_id(node)
            self.assertTrue(fixture.cache.tree_core.is_full_device_evicted(node))
            self.assertIsNone(core_node.component_data[MAMBA].value)
            self.assertIsNotNone(core_node.component_data[FULL].host_value)
            self.assertIsNotNone(core_node.component_data[MAMBA].host_value)
            incoming = fixture.allocate(tokens, 9000)
            incoming_state = allocate_mamba(9000)
            expected_full = fixture.read(incoming)
            expected_mamba = read_mamba(incoming_state)
            engine = fixture.cache.cache_controller.l2_transfer_engine
            with mock.patch.object(
                engine,
                "submit_host_to_device",
                side_effect=AssertionError("Mamba duplicate changed restore policy"),
            ):
                fixture.insert(tokens, incoming, mamba_value=incoming_state)
            adopted = fixture.match(tokens).device_indices
            self.assertEqual(adopted.tolist(), incoming.tolist())
            for expected, actual in zip(
                expected_full, fixture.read(adopted), strict=True
            ):
                self.assertTrue(torch.equal(expected, actual))
            actual_state = core_node.component_data[MAMBA].value
            self.assertEqual(actual_state.tolist(), incoming_state.tolist())
            for expected, actual in zip(
                expected_mamba, read_mamba(actual_state), strict=True
            ):
                self.assertTrue(torch.equal(expected, actual))
            self.assert_host_unchanged(fixture, host_before)
            self.assert_no_load_ticket(fixture)
            fixture.cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
