"""Unit tests for asymmetric attention-TP resharding and DP-attention support
in prefill KV offload to decode DRAM."""

from __future__ import annotations

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist

from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferDestination
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import raiden_transfer
from sglang_tpu.disaggregation.decode_dram_offload import (
    dram_offload_hooks,
    radix_host_ops,
)
from sglang_tpu.disaggregation.decode_dram_offload.decode_dram_victim_receiver import (
    DecodeDramVictimReceiver,
)
from sglang_tpu.disaggregation.decode_dram_offload.prefill_dram_prefix_puller import (
    PrefillDramPrefixPuller,
)
from sglang_tpu.disaggregation.decode_dram_offload.prefill_dram_victim_pusher import (
    PrefillDramVictimPusher,
)
from sglang_tpu.disaggregation.decode_dram_offload.rank_lockstep import (
    DramOffloadWorkerContext,
    RankLockstep,
)
from sglang_tpu.disaggregation.decode_dram_offload.reshard_host_store import (
    DecodeReshardBranchOfferer,
    DecodeReshardBranchReceiver,
    ReshardLayoutStore,
    ReshardPoolLayout,
    ReshardPoolSet,
)
from sglang_tpu.disaggregation.raiden.dram_offload_channel import DramOffloadChannel
from sglang_tpu.disaggregation.raiden.serde import (
    PrefixPull,
    ReshardReq,
    VictimPush,
)
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.testing.decode_dram_offload import (
    HYBRID_PAGE,
    FakeHostTransfer,
    _bare_decode_manager,
    _bare_prefill_manager,
    full_cache,
    host_hit,
    hybrid_cache,
    insert_tokens,
    pushed_req,
    sender_on,
)

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _InMemoryPool(FakeHostTransfer):
    """Fake RaidenKVTransfer that copies real byte payloads block-for-block
    between producer and consumer host pools."""

    _REGISTRY: dict[str, _InMemoryPool] = {}

    def __init__(self, endpoint: str, num_pages: int, page_bytes: int):
        super().__init__()
        self._endpoint = endpoint
        self.num_host_pages = num_pages
        self.page_bytes = page_bytes
        self.blocks: dict[int, bytes] = {
            i: b"\x00" * page_bytes for i in range(num_pages)
        }
        self._REGISTRY[endpoint] = self

    def host_control_address(self, local_ip: str) -> str:
        return self._endpoint

    def start_read(
        self,
        req_id: str,
        uuid: int,
        remote_endpoint,
        remote_block_ids: list[int],
        local_block_ids: list[int],
    ) -> None:
        super().start_read(
            req_id, uuid, remote_endpoint, remote_block_ids, local_block_ids
        )
        peer = self._REGISTRY[remote_endpoint]
        matched = [
            rid for rid, (ruuid, rblocks) in peer.registrations.items() if ruuid == uuid
        ]
        assert len(matched) == 1, (
            f"expected 1 registration for uuid={uuid}, got {matched}"
        )
        reg_id = matched[0]
        _, reg_blocks = peer.registrations[reg_id]
        reg_set = set(reg_blocks)
        for src, dst in zip(remote_block_ids, local_block_ids, strict=True):
            assert src in reg_set, f"block {src} not registered in {reg_blocks}"
            self.blocks[dst] = peer.blocks[src]
        peer.results[reg_id] = "sent"
        self.results[req_id] = "received"


class _SimLockstep(RankLockstep):
    """Lockstep helper over an in-process list of simulated ranks."""

    def __init__(self, rank: int, group_state: dict) -> None:
        super().__init__(None)
        self._rank = rank
        self._state = group_state

    def step_reduce(self, all_rows: list[list[list[int]]], op) -> list[list[list[int]]]:
        t = torch.tensor(all_rows, dtype=torch.int64)
        if op == dist.ReduceOp.MIN:
            reduced = t.min(dim=0).values.tolist()
        elif op == dist.ReduceOp.MAX:
            reduced = t.max(dim=0).values.tolist()
        else:
            raise ValueError(f"unsupported op {op}")
        return [reduced for _ in all_rows]


def _attach_tensor_layout(host_pool, *, rank: int, layers: int = 2, width: int = 16):
    buf = torch.zeros((max(1, host_pool.capacity), width), dtype=torch.bfloat16)
    host_pool.transfer._device_tensors = [[buf] for _ in range(layers)]
    return buf


class TestDpAttentionPortIsolation(CustomTestCase):
    def test_dp_attention_and_system_dp_workers_get_distinct_control_ports(self):
        seen_dp_attn = set()
        for dp_rank in range(2):
            for tp_rank in range(4):
                parallel = SimpleNamespace(
                    attn_tp_rank=tp_rank,
                    attn_tp_size=4,
                    attn_dp_rank=dp_rank,
                    attn_dp_size=2,
                    dp_rank=dp_rank,
                    attn_cp_rank=0,
                    attn_cp_size=1,
                    pp_rank=0,
                    pp_size=1,
                )
                with (
                    patch(
                        "sglang.srt.runtime_context.get_parallel",
                        return_value=parallel,
                    ),
                    patch.object(raiden_transfer, "_host_pools_built", 0),
                    patch.object(raiden_transfer, "_bound_control_ports", set()),
                ):
                    p0 = next(raiden_transfer.host_control_port_candidates())
                    p1 = next(raiden_transfer.host_control_port_candidates())
                self.assertNotIn(p0, seen_dp_attn)
                self.assertNotIn(p1, seen_dp_attn)
                self.assertNotEqual(p0, p1)
                seen_dp_attn.update((p0, p1))
        self.assertEqual(len(seen_dp_attn), 16)

        seen_sys_dp = set()
        for sys_dp_rank in range(2):
            for tp_rank in range(4):
                parallel = SimpleNamespace(
                    attn_tp_rank=tp_rank,
                    attn_tp_size=4,
                    attn_dp_rank=0,
                    attn_dp_size=1,
                    dp_rank=sys_dp_rank,
                    attn_cp_rank=0,
                    attn_cp_size=1,
                    pp_rank=0,
                    pp_size=1,
                )
                with (
                    patch(
                        "sglang.srt.runtime_context.get_parallel",
                        return_value=parallel,
                    ),
                    patch.object(raiden_transfer, "_host_pools_built", 0),
                    patch.object(raiden_transfer, "_bound_control_ports", set()),
                ):
                    p0 = next(raiden_transfer.host_control_port_candidates())
                self.assertNotIn(p0, seen_sys_dp)
                seen_sys_dp.add(p0)
        self.assertEqual(len(seen_sys_dp), 8)

    def test_in_process_bound_ports_are_never_reused_by_extra_reshard_pools(self):
        bound = {50001}
        with (
            patch(
                "sglang_tpu.raiden_transfer.get_free_port",
                side_effect=[50001, 50002, 50003],
            ),
            patch.object(raiden_transfer, "_host_pools_built", 2),
            patch.object(raiden_transfer, "_bound_control_ports", bound),
        ):
            ports = list(raiden_transfer.host_control_port_candidates())
        self.assertEqual(ports, [50002, 50003])


class TestReshardLayoutStore(CustomTestCase):
    def test_shared_prefix_dedup_lru_eviction_and_pin_protection(self):
        store = ReshardLayoutStore(page_size=2, pages=4)
        scope = (None, None)
        l1 = store.reserve(scope, [10, 11, 12, 13])
        self.assertEqual((l1.first_new_page, l1.new_pages), (0, [0, 1]))
        store.commit(l1)

        # Shares first page [10, 11], allocates only 1 new page for [14, 15].
        l2 = store.reserve(scope, [10, 11, 14, 15])
        self.assertEqual((l2.first_new_page, l2.new_pages), (1, [2]))
        store.commit(l2)
        self.assertEqual(store.used_pages, 3)

        # Pin branch 1 so eviction must reclaim leaf of branch 2 instead.
        m1 = store.match(scope, [10, 11, 12, 13, 99])
        self.assertEqual(m1.pages, [0, 1])
        store.pin(m1)

        l3 = store.reserve(scope, [20, 21, 22, 23])
        self.assertIsNotNone(l3)
        store.commit(l3)
        self.assertEqual(store.stats["evicted_pages"], 1)
        self.assertEqual(store.match(scope, [10, 11, 14, 15]).pages, [0])
        self.assertEqual(store.match(scope, [20, 21, 22, 23]).pages, l3.new_pages)

        with self.assertRaisesRegex(RuntimeError, "pinned"):
            store.clear()
        store.unpin(m1)
        store.clear()
        self.assertEqual(store.used_pages, 0)

    def test_hybrid_window_store_matches_only_branches_ending_with_a_window(self):
        store = ReshardLayoutStore(page_size=2, pages=6, window_pages=2, windows=2)
        scope = ("k", "s")
        l1 = store.reserve(scope, [1, 2, 3, 4, 5, 6], with_window=True)
        store.commit(l1)
        # Full extension holds a window; mid-branch prefix without a window rolls back.
        full_match = store.match(scope, [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual((full_match.pages, len(full_match.window)), ([0, 1, 2], 2))
        self.assertIsNone(store.match(scope, [1, 2, 3, 4, 9]))

    def test_concurrent_window_only_landing_on_interior_node_is_rejected_and_recycles(
        self,
    ):
        store = ReshardLayoutStore(page_size=2, pages=6, window_pages=2, windows=3)
        scope = ("k", "s")
        # Commit 3-page branch: depth-2 interior node [1, 2, 3, 4] has window=None.
        l_full = store.reserve(scope, [1, 2, 3, 4, 5, 6], with_window=True)
        store.commit(l_full)
        self.assertEqual(store.used_windows, 1)

        # First window-only landing on [1, 2, 3, 4] succeeds; concurrent second returns busy.
        l_win1 = store.reserve(scope, [1, 2, 3, 4], with_window=True)
        self.assertIsNotNone(l_win1)
        self.assertEqual(l_win1.new_pages, [])
        self.assertIsNone(store.reserve(scope, [1, 2, 3, 4], with_window=True))
        self.assertEqual(store.stats["busy"], 1)

        # Aborting recycles the window IDs and clears window_landing so retry succeeds.
        store.abort(l_win1)
        self.assertEqual(store.used_windows, 1)
        l_win2 = store.reserve(scope, [1, 2, 3, 4], with_window=True)
        self.assertIsNotNone(l_win2)
        store.commit(l_win2)
        self.assertEqual(store.used_windows, 2)
        self.assertEqual(len(store.match(scope, [1, 2, 3, 4, 9]).window), 2)
        store.clear()
        self.assertEqual((store.used_pages, store.used_windows), (0, 0))


class TestReshardWireAndEndToEndPair(CustomTestCase):
    """End-to-end asymmetric-TP pair tests over real ZMQ frames and byte-checked
    host block pools, covering ratio=2 (e.g. 8TP->4TP) and ratio=4 (8TP->2TP)."""

    def _build_asymmetric_pair(self, *, src_tp: int, dst_tp: int, hybrid: bool = False):
        ratio = src_tp // dst_tp
        page_size = HYBRID_PAGE if hybrid else 1
        page_bytes = 64

        # Build ZMQ-connected bare managers for every prefill and decode rank.
        endpoints: dict[str, any] = {}

        def send_locked(endpoint: str, frames: list[bytes]) -> None:
            endpoints[endpoint]._dispatch(frames)

        prefills = []
        for s in range(src_tp):
            pmgr = _bare_prefill_manager()
            pmgr.attn_tp_rank = s
            pmgr.attn_tp_size = src_tp
            pmgr.local_ip = "10.0.1.1"
            pmgr.rank_port = 4000 + s
            pmgr._send_multipart_locked = Mock(side_effect=send_locked)
            pmgr.dram_offload_channel = DramOffloadChannel(pmgr)
            pmgr._message_handlers.update(pmgr.dram_offload_channel.message_handlers())
            endpoints[f"tcp://10.0.1.1:{4000 + s}"] = pmgr

            cache, allocator = (
                hybrid_cache(window=128) if hybrid else full_cache(host_capacity=32)
            )
            host_pool = radix_host_ops.anchor_host_pool(cache)
            swa_pool = radix_host_ops.swa_host_pool(cache)
            host_pool.transfer = _InMemoryPool(
                f"10.0.1.1:{7000 + 2 * s}", 64, page_bytes
            )
            _attach_tensor_layout(host_pool, rank=s, layers=2, width=16)
            if swa_pool is not None:
                swa_pool.transfer = _InMemoryPool(
                    f"10.0.1.1:{7000 + 2 * s + 1}", 64, page_bytes
                )
                _attach_tensor_layout(swa_pool, rank=s, layers=2, width=16)
            pmgr.dram_offload_channel.bind_host_pools(host_pool, swa_pool)
            ctx = DramOffloadWorkerContext(
                cache=cache,
                host_pool=host_pool,
                page_size=page_size,
                channel=pmgr.dram_offload_channel,
                lockstep=RankLockstep(None),
                swa_host_pool=swa_pool,
                window_tokens=radix_host_ops.window_tokens(cache, page_size),
            )
            pusher = PrefillDramVictimPusher(ctx)
            pusher._namespace = "pf0"
            puller = PrefillDramPrefixPuller(ctx)
            puller._max_inflight_tokens = 1 << 20
            puller._max_inflight_swa_tokens = 1 << 20
            prefills.append(
                SimpleNamespace(
                    mgr=pmgr,
                    cache=cache,
                    allocator=allocator,
                    host_pool=host_pool,
                    swa_pool=swa_pool,
                    pusher=pusher,
                    puller=puller,
                )
            )

        decodes = []
        for d in range(dst_tp):
            dmgr = _bare_decode_manager()
            dmgr.attn_tp_rank = d
            dmgr.attn_tp_size = dst_tp
            dmgr.kv_args = SimpleNamespace(page_size=page_size, engine_rank=d)
            dmgr.local_ip = "10.0.2.1"
            dmgr.rank_port = 5000 + d
            dmgr._dynamic_host_staging = True
            dmgr._bounded_staging = False
            dmgr._state_engine = None
            dmgr._draft_engine = None
            dmgr._unified_swa_pool = None
            dmgr._deferred_ack_targets = {}
            dmgr._deferred_ack_poisoned_rooms = set()
            dmgr._staging_outstanding = {}
            dmgr._ensure_engine = lambda d=d: SimpleNamespace(
                transfer_address=f"10.0.2.1:{6000 + d}"
            )
            dmgr._send_multipart_locked = Mock(side_effect=send_locked)
            dmgr.dram_offload_channel = DramOffloadChannel(dmgr)
            dmgr._message_handlers.update(dmgr.dram_offload_channel.message_handlers())
            endpoints[f"tcp://10.0.2.1:{5000 + d}"] = dmgr

            built_idx = [0]

            def make_pool(layout: ReshardPoolLayout, pages: int, d=d, idx=built_idx):
                k = idx[0]
                idx[0] += 1
                return _InMemoryPool(
                    f"10.0.2.1:{8000 + d * 16 + k}", pages, layout.page_bytes
                )

            pools = ReshardPoolSet(
                rank=d,
                budget_bytes=ratio * page_bytes * 16,
                build_pool=make_pool,
            )
            dmgr.dram_offload_channel.bind_reshard_pools(pools)
            owned: set[int] = set()
            receiver = DecodeReshardBranchReceiver(
                channel=dmgr.dram_offload_channel,
                lockstep=RankLockstep(None),
                pools=pools,
                page_size=page_size,
            )
            offerer = DecodeReshardBranchOfferer(
                channel=dmgr.dram_offload_channel,
                lockstep=RankLockstep(None),
                receiver=receiver,
                owned_rooms=lambda o=owned: o,
            )
            decodes.append(
                SimpleNamespace(
                    mgr=dmgr,
                    pools=pools,
                    receiver=receiver,
                    offerer=offerer,
                    owned=owned,
                )
            )

        return prefills, decodes, page_size, page_bytes

    def test_full_and_partial_prefix_round_trip_across_8tp_to_4tp_and_2tp(self):
        with tpu_envs.SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM.override(True):
            for dst_tp in (4, 2):
                with self.subTest(dst_tp=dst_tp):
                    self._check_round_trip(dst_tp=dst_tp)

    def _check_round_trip(self, *, dst_tp: int):
        src_tp = 8
        ratio = src_tp // dst_tp
        prefills, decodes, page_size, page_bytes = self._build_asymmetric_pair(
            src_tp=src_tp, dst_tp=dst_tp, hybrid=False
        )
        tokens = list(range(500, 508))
        room1 = 11
        for s, pf in enumerate(prefills):
            sender_on(pf.mgr)
            pf.mgr.request_status[room1] = KVPoll.Bootstrapping

        # Turn 1 handshake: each decode rank sends RESHARD_REQ to its `ratio` writers.
        for d, dec in enumerate(decodes):
            infos = [
                {"rank_ip": "10.0.1.1", "rank_port": 4000 + d * ratio + w}
                for w in range(ratio)
            ]
            dec.mgr.register_decode_request(room1, [1, 2], [], None, infos)

        # Populate distinct per-rank byte payloads in prefill host pools and evict HBM.
        expected_bytes: dict[tuple[int, int], bytes] = {}
        for s, pf in enumerate(prefills):
            insert_tokens(pf.cache, pf.allocator, tokens)
            pf.pusher.note_decode_of(
                pushed_req(room=room1, tokens=tokens), recalled=False
            )
            pf.cache.evict(EvictParams(num_tokens=len(tokens)))
            pages = radix_host_ops.host_pages(
                pf.cache,
                pf.host_pool,
                radix_host_ops.leaf_of(
                    pf.cache,
                    pf.cache.tree_core.root_node.children[
                        next(iter(pf.cache.tree_core.root_node.children))
                    ].id,
                ),
            )
            for idx, pid in enumerate(pages):
                payload = f"rank={s}:page={idx}:".encode().ljust(page_bytes, b"x")
                pf.host_pool.transfer.blocks[pid] = payload
                expected_bytes[(s, idx)] = payload

        # Prefill pushes victim to decode's per-writer pools.
        for pf in prefills:
            pf.pusher.advance(serving=True)
            self.assertEqual(pf.pusher.stats["started"], 1)

        # Decode admits and lands the push across all `ratio` writer pools.
        for dec in decodes:
            for _ in range(16):
                dec.receiver.advance()
            self.assertEqual(dec.receiver.store.stats["landed"], 1)
            self.assertEqual(dec.receiver.store.used_pages, len(tokens))

        # Prefill retires settled pushes and evicts local host copies.
        for pf in prefills:
            pf.pusher.advance(serving=False)
            self.assertEqual(pf.pusher.stats["stored"], 1)
            pf.cache.evict_host(len(tokens))
            self.assertEqual(host_hit(pf.cache, tokens), 0)

        # Keep first 2 pages locally on prefill to exercise partial recall [2..8)!
        for s, pf in enumerate(prefills):
            local_idx = pf.host_pool.alloc(2)
            for idx, pid in enumerate(pf.host_pool._to_page_ids(local_idx)):
                pf.host_pool.transfer.blocks[pid] = expected_bytes[(s, idx)]
            radix_host_ops.insert_host_prefix(
                pf.cache, RadixKey(array("q", tokens[:2])), local_idx, 1
            )

        # Turn 2: decode offers stored reshard prefix in RESHARD_REQ; prefill recalls [2..8).
        room2 = 22
        turn2_tokens = tokens + [900, 901]
        req2 = SimpleNamespace(
            bootstrap_room=room2,
            origin_input_ids=turn2_tokens,
            extra_key=None,
            cache_salt=None,
            last_node=None,
        )
        for pf in prefills:
            pf.mgr.request_status[room2] = KVPoll.Bootstrapping
        for d, dec in enumerate(decodes):
            dec.owned.add(room2)
            dec.offerer.offer_reshard_prefix(
                room=room2, tokens=tokens, scope=(None, None)
            )
            infos = [
                {"rank_ip": "10.0.1.1", "rank_port": 4000 + d * ratio + w}
                for w in range(ratio)
            ]
            dec.mgr.register_decode_request(room2, [1, 2], [], None, infos)

        # Prefill asks for missing range [2, 8), receives PrefixReady, reads, and commits.
        for pf in prefills:
            self.assertTrue(pf.mgr.dram_offload_channel.prefix_pull_open_locked(room2))
            pf.puller.advance([req2], waiting_reqs=[])
            self.assertEqual(
                (pf.puller._plans[room2].first_page, pf.puller._plans[room2].num_pages),
                (2, 6),
            )
            pf.puller.advance([req2], waiting_reqs=[])
            pf.puller.advance([req2], waiting_reqs=[])
            self.assertTrue(pf.puller.insert_recalled_host_prefix(req2))
            self.assertEqual(pf.puller.stats["pulled_tokens"], 6)
            self.assertEqual(host_hit(pf.cache, turn2_tokens), len(tokens))

        # Verify every prefill rank recovered its exact shard's bytes!
        for s, pf in enumerate(prefills):
            leaf = next(iter(pf.cache.tree_core.root_node.children.values()))
            while leaf.children:
                leaf = next(iter(leaf.children.values()))
            recalled_pages = radix_host_ops.host_pages(pf.cache, pf.host_pool, leaf)
            self.assertEqual(len(recalled_pages), len(tokens))
            for idx, pid in enumerate(recalled_pages):
                self.assertEqual(
                    pf.host_pool.transfer.blocks[pid],
                    expected_bytes[(s, idx)],
                )

        # Decode stays non-idle while offer is pinned, then unpins cleanly once departed.
        for dec in decodes:
            self.assertTrue(bool(dec.offerer.offers))
            dec.owned.clear()
            dec.offerer.settle_offers()
            self.assertFalse(bool(dec.offerer.offers))
            dec.offerer.release_all()
            dec.receiver.release_all()

    def test_hybrid_swa_reshard_push_and_recall_round_trip(self):
        with tpu_envs.SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM.override(True):
            prefills, decodes, page_size, page_bytes = self._build_asymmetric_pair(
                src_tp=4, dst_tp=2, hybrid=True
            )
            ratio = 2
            tokens = list(range(1000, 1000 + 3 * page_size))
            room1 = 31
            for pf in prefills:
                pf.mgr.request_status[room1] = KVPoll.Bootstrapping
            for d, dec in enumerate(decodes):
                infos = [
                    {"rank_ip": "10.0.1.1", "rank_port": 4000 + d * ratio + w}
                    for w in range(ratio)
                ]
                dec.mgr.register_decode_request(room1, [1, 2, 3], [1, 2], None, infos)

            expected_full: dict[tuple[int, int], bytes] = {}
            expected_swa: dict[tuple[int, int], bytes] = {}
            for s, pf in enumerate(prefills):
                full_idx = pf.host_pool.alloc(len(tokens))
                swa_idx = pf.swa_pool.alloc(2 * page_size)
                for i, pid in enumerate(pf.host_pool._to_page_ids(full_idx)):
                    b = f"hf:{s}:{i}".encode().ljust(page_bytes, b"f")
                    pf.host_pool.transfer.blocks[pid] = b
                    expected_full[(s, i)] = b
                for i, pid in enumerate(pf.swa_pool._to_page_ids(swa_idx)):
                    b = f"hs:{s}:{i}".encode().ljust(page_bytes, b"s")
                    pf.swa_pool.transfer.blocks[pid] = b
                    expected_swa[(s, i)] = b
                node_id, _ = radix_host_ops.insert_host_prefix(
                    pf.cache,
                    RadixKey(array("q", tokens)),
                    full_idx,
                    page_size,
                    swa_indices=swa_idx,
                )
                req = pushed_req(room=room1, tokens=tokens)
                req.last_node = node_id
                pf.pusher.note_decode_of(req, recalled=False)
                # Trigger host pressure on SWA pool.
                pf.swa_pool.used = pf.swa_pool.capacity

            for pf in prefills:
                pf.pusher.advance(serving=True)
            for dec in decodes:
                for _ in range(16):
                    dec.receiver.advance()
                self.assertEqual(dec.receiver.store.used_windows, 1)
            for pf in prefills:
                pf.pusher.advance(serving=False)
                self.assertEqual(pf.pusher.stats["stored"], 1)
                pf.swa_pool.used = 2 * page_size
                pf.cache.evict_host(len(tokens))
                self.assertEqual(host_hit(pf.cache, tokens), 0)

            room2 = 32
            turn2 = tokens + list(range(2000, 2000 + page_size))
            req2 = SimpleNamespace(
                bootstrap_room=room2,
                origin_input_ids=turn2,
                extra_key=None,
                cache_salt=None,
                last_node=None,
            )
            for pf in prefills:
                pf.mgr.request_status[room2] = KVPoll.Bootstrapping
            for d, dec in enumerate(decodes):
                dec.owned.add(room2)
                dec.offerer.offer_reshard_prefix(
                    room=room2, tokens=tokens, scope=(None, None)
                )
                infos = [
                    {"rank_ip": "10.0.1.1", "rank_port": 4000 + d * ratio + w}
                    for w in range(ratio)
                ]
                dec.mgr.register_decode_request(room2, [1, 2, 3], [1, 2], None, infos)

            for s, pf in enumerate(prefills):
                pf.puller.advance([req2], waiting_reqs=[])
                pf.puller.advance([req2], waiting_reqs=[])
                pf.puller.advance([req2], waiting_reqs=[])
                self.assertTrue(pf.puller.insert_recalled_host_prefix(req2))
                self.assertEqual(pf.puller.stats["pulled_swa_tokens"], 2 * page_size)
                leaf = next(iter(pf.cache.tree_core.root_node.children.values()))
                while leaf.children:
                    leaf = next(iter(leaf.children.values()))
                full_pages = radix_host_ops.host_pages(pf.cache, pf.host_pool, leaf)
                swa_pages = radix_host_ops.window_host_pages(
                    pf.cache,
                    pf.swa_pool,
                    leaf,
                    tokens=len(tokens),
                    window_tokens=2 * page_size,
                )
                for i, pid in enumerate(full_pages):
                    self.assertEqual(
                        pf.host_pool.transfer.blocks[pid], expected_full[(s, i)]
                    )
                for i, pid in enumerate(swa_pages):
                    self.assertEqual(
                        pf.swa_pool.transfer.blocks[pid], expected_swa[(s, i)]
                    )

    def test_layout_mismatch_on_subsequent_push_is_declined_without_crash(self):
        with tpu_envs.SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM.override(True):
            prefills, decodes, _, _ = self._build_asymmetric_pair(
                src_tp=2, dst_tp=1, hybrid=False
            )
            dec = decodes[0]
            # First push builds store with page_bytes=64.
            for w in range(2):
                dec.mgr._dispatch(
                    VictimPush(
                        prefill="pf",
                        seq=1,
                        ack_endpoint=f"tcp://10.0.1.1:{4000 + w}",
                        tokens=[1, 2] if w == 0 else [],
                        blocks=[0, 1],
                        uuid=10 + w,
                        address=prefills[w].host_pool.transfer.host_control_address(""),
                        writer=w,
                        writers=2,
                        layout=[["bfloat16", [16]], ["bfloat16", [16]]],
                        page_bytes=64,
                    ).to_frames()
                )
                prefills[w].host_pool.transfer.register_read(f"r{w}", 10 + w, [0, 1])
            for _ in range(16):
                dec.receiver.advance()
            self.assertEqual(dec.receiver.store.stats["landed"], 1)

            # Second push arrives with incompatible tensor shape / page_bytes.
            for w in range(2):
                dec.mgr._dispatch(
                    VictimPush(
                        prefill="pf",
                        seq=2,
                        ack_endpoint=f"tcp://10.0.1.1:{4000 + w}",
                        tokens=[3, 4] if w == 0 else [],
                        blocks=[0, 1],
                        uuid=20 + w,
                        address=prefills[w].host_pool.transfer.host_control_address(""),
                        writer=w,
                        writers=2,
                        layout=[["bfloat16", [32]], ["bfloat16", [32]]],
                        page_bytes=128,
                    ).to_frames()
                )
            for _ in range(16):
                dec.receiver.advance()
            self.assertEqual(dec.receiver.stats["layout"], 1)
            self.assertFalse(prefills[0].mgr.dram_offload_channel.victim_reply(2))

    def test_wire_defaults_stay_backward_compatible_for_symmetric_messages(self):
        pull = PrefixPull(room=5, first_page=0, num_pages=4)
        self.assertNotIn(b"writer", pull.to_frames()[1])
        self.assertEqual(PrefixPull.from_frames(pull.to_frames()).writer, -1)

        req = ReshardReq(
            room=5,
            decode_ip="10.0.0.2",
            decode_port=5555,
            push_endpoint="10.0.0.2:31000",
            decode_blocks=[1, 2],
            dst_attn_tp=2,
            decode_swa_blocks=[],
            swa_push_endpoint="",
        )
        self.assertNotIn(b"prefix_blocks", req.to_frames()[1])
        decoded = ReshardReq.from_frames(req.to_frames())
        self.assertEqual((decoded.prefix_blocks, decoded.prefix_swa_blocks), ([], []))

    def test_symmetric_tp_hybrid_swa_victim_push_preserves_distinct_swa_uuid(self):
        with tpu_envs.SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM.override(True):
            endpoints: dict[str, any] = {}

            def send_locked(endpoint: str, frames: list[bytes]) -> None:
                endpoints[endpoint]._dispatch(frames)

            pmgr = _bare_prefill_manager()
            pmgr.attn_tp_rank, pmgr.attn_tp_size = 0, 4
            pmgr.local_ip, pmgr.rank_port = "10.0.3.1", 4300
            pmgr._send_multipart_locked = Mock(side_effect=send_locked)
            pmgr.dram_offload_channel = DramOffloadChannel(pmgr)
            pmgr._message_handlers.update(pmgr.dram_offload_channel.message_handlers())
            endpoints["tcp://10.0.3.1:4300"] = pmgr

            dmgr = _bare_decode_manager()
            dmgr.attn_tp_rank, dmgr.attn_tp_size = 0, 4
            dmgr.local_ip, dmgr.rank_port = "10.0.3.2", 5300
            dmgr._send_multipart_locked = Mock(side_effect=send_locked)
            dmgr.dram_offload_channel = DramOffloadChannel(dmgr)
            dmgr._message_handlers.update(dmgr.dram_offload_channel.message_handlers())
            endpoints["tcp://10.0.3.2:5300"] = dmgr

            p_cache, _ = hybrid_cache(window=128)
            d_cache, _ = hybrid_cache(window=128)
            p_full = radix_host_ops.anchor_host_pool(p_cache)
            p_swa = radix_host_ops.swa_host_pool(p_cache)
            d_full = radix_host_ops.anchor_host_pool(d_cache)
            d_swa = radix_host_ops.swa_host_pool(d_cache)
            p_full.transfer = _InMemoryPool("10.0.3.1:7300", 64, 32)
            p_swa.transfer = _InMemoryPool("10.0.3.1:7301", 64, 32)
            d_full.transfer = _InMemoryPool("10.0.3.2:8300", 64, 32)
            d_swa.transfer = _InMemoryPool("10.0.3.2:8301", 64, 32)
            pmgr.dram_offload_channel.bind_host_pools(p_full, p_swa)
            dmgr.dram_offload_channel.bind_host_pools(d_full, d_swa)

            p_ctx = DramOffloadWorkerContext(
                cache=p_cache,
                host_pool=p_full,
                page_size=HYBRID_PAGE,
                channel=pmgr.dram_offload_channel,
                lockstep=RankLockstep(None),
                swa_host_pool=p_swa,
                window_tokens=128,
            )
            d_ctx = DramOffloadWorkerContext(
                cache=d_cache,
                host_pool=d_full,
                page_size=HYBRID_PAGE,
                channel=dmgr.dram_offload_channel,
                lockstep=RankLockstep(None),
                swa_host_pool=d_swa,
                window_tokens=128,
            )
            pusher = PrefillDramVictimPusher(p_ctx)
            receiver = DecodeDramVictimReceiver(d_ctx)

            tokens = list(range(3000, 3000 + 3 * HYBRID_PAGE))
            room = 41
            pmgr._prefill.decode_addr[room] = ("10.0.3.2", 5300)
            full_idx = p_full.alloc(len(tokens))
            swa_idx = p_swa.alloc(128)
            for i, pid in enumerate(p_swa._to_page_ids(swa_idx)):
                p_swa.transfer.blocks[pid] = f"sym-swa-{i}".encode().ljust(32, b"z")
            node_id, _ = radix_host_ops.insert_host_prefix(
                p_cache,
                RadixKey(array("q", tokens)),
                full_idx,
                HYBRID_PAGE,
                swa_indices=swa_idx,
            )
            req = pushed_req(room=room, tokens=tokens)
            req.last_node = node_id
            pusher.note_decode_of(req, recalled=False)
            p_swa.used = p_swa.capacity

            pusher.advance(serving=True)
            for _ in range(16):
                receiver.advance()
            pusher.advance(serving=False)
            self.assertEqual(pusher.stats["stored"], 1)
            self.assertEqual(receiver.stats["landed"], 1)
            d_leaf = next(iter(d_cache.tree_core.root_node.children.values()))
            while d_leaf.children:
                d_leaf = next(iter(d_leaf.children.values()))
            d_swa_pages = radix_host_ops.window_host_pages(
                d_cache, d_swa, d_leaf, tokens=len(tokens), window_tokens=128
            )
            self.assertEqual(
                [d_swa.transfer.blocks[pid] for pid in d_swa_pages],
                [f"sym-swa-{i}".encode().ljust(32, b"z") for i in range(2)],
            )

    def test_send_kv_metadata_skips_reshard_offer_on_host_staged_receive(self):
        ext = SimpleNamespace(reshard_source=Mock(), source=Mock())
        queue = SimpleNamespace(tree_cache=Mock())
        decode_req = SimpleNamespace(
            req=SimpleNamespace(
                bootstrap_room=7,
                origin_input_ids=list(range(64)),
                extra_key=None,
                cache_salt=None,
            ),
            kv_receiver=SimpleNamespace(bootstrap_infos=[{}, {}]),
        )
        upstream = Mock()
        with patch.object(dram_offload_hooks, "_extension", return_value=ext):
            dram_offload_hooks._send_kv_metadata(
                upstream,
                queue,
                decode_req,
                [0],
                16,
                decode_prefix_len=0,
                destination=KVTransferDestination.HOST,
            )
        ext.reshard_source.offer_reshard_prefix.assert_not_called()
        ext.source.offer_host_prefix.assert_not_called()
        upstream.assert_called_once()

    def test_size_reshard_store_rejects_non_positive_ratio_and_honors_env(self):
        from sglang_tpu.disaggregation.decode_dram_offload.reshard_host_store import (
            size_reshard_store,
        )

        with self.assertRaises(ValueError):
            size_reshard_store(
                budget_bytes=1024,
                page_bytes=8,
                window_page_bytes=16,
                window_pages=2,
                pages_per_window_page=0,
            )
        p_default, w_default = size_reshard_store(
            budget_bytes=1024,
            page_bytes=8,
            window_page_bytes=16,
            window_pages=2,
            pages_per_window_page=4,
        )
        p_wide, w_wide = size_reshard_store(
            budget_bytes=1024,
            page_bytes=8,
            window_page_bytes=16,
            window_pages=2,
            pages_per_window_page=16,
        )
        self.assertGreater(p_wide, p_default)
        self.assertLess(w_wide, w_default)

    def test_reshard_pool_set_rejects_negative_ratio_env_and_honors_positive(self):
        from sglang_tpu.disaggregation.decode_dram_offload.rank_lockstep import (
            RankLockstep,
        )
        from sglang_tpu.disaggregation.decode_dram_offload.reshard_host_store import (
            ReshardPoolLayout,
            ReshardPoolSet,
        )

        full_layout = ReshardPoolLayout(
            writer=0, writers=1, tensors=(("uint8", (8,)),), page_bytes=8, window=False
        )
        swa_layout = ReshardPoolLayout(
            writer=0, writers=1, tensors=(("uint8", (16,)),), page_bytes=16, window=True
        )
        parts = {0: SimpleNamespace(layout=full_layout, swa_layout=swa_layout)}
        lockstep = RankLockstep()

        bad_pools = ReshardPoolSet(
            rank=0,
            budget_bytes=4096,
            build_pool=lambda layout, pages: SimpleNamespace(),
        )
        with patch.dict(
            "os.environ", {"SGLANG_PD_OFFLOAD_RESHARD_PAGES_PER_WINDOW_PAGE": "-1"}
        ):
            self.assertIsNone(bad_pools.build(parts, window_pages=2, lockstep=lockstep))
        self.assertIn("non-negative", bad_pools.refusal or "")

        p_default = ReshardPoolSet(
            rank=0,
            budget_bytes=4096,
            build_pool=lambda layout, pages: SimpleNamespace(),
        )
        with patch.dict(
            "os.environ", {"SGLANG_PD_OFFLOAD_RESHARD_PAGES_PER_WINDOW_PAGE": "0"}
        ):
            sizes_default = p_default.build(parts, window_pages=2, lockstep=lockstep)
        p_wide = ReshardPoolSet(
            rank=0,
            budget_bytes=4096,
            build_pool=lambda layout, pages: SimpleNamespace(),
        )
        with patch.dict(
            "os.environ", {"SGLANG_PD_OFFLOAD_RESHARD_PAGES_PER_WINDOW_PAGE": "16"}
        ):
            sizes_wide = p_wide.build(parts, window_pages=2, lockstep=lockstep)
        self.assertIsNotNone(sizes_default)
        self.assertIsNotNone(sizes_wide)
        self.assertGreater(sizes_wide[0], sizes_default[0])
        self.assertLess(sizes_wide[1], sizes_default[1])


if __name__ == "__main__":
    unittest.main()
