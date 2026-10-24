"""Layer subsets and transfer lifetimes for pipeline-parallel Raiden disaggregation."""

import concurrent.futures
import json
import sys
import threading
import time
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden.common import get_raiden_kv_caches
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.disaggregation.raiden.host_pool import (
    StagingAdmission,
    node_worker_rank,
    ranks_per_node,
)
from sglang_tpu.disaggregation.raiden.pipeline import (
    PipelineTransfer,
    Receive,
    destination_blocks,
    layout_fingerprint,
    legacy_mirror_slots,
    legacy_slot_host_bytes,
    materialized_storage_layout,
    pool_manifest,
    warn_if_staging_exceeds_premapped,
)
from sglang_tpu.disaggregation.raiden.pp_common import (
    head_slice_spans,
    source_spans,
)
from sglang_tpu.disaggregation.raiden.serde import MessageTag, RoomFailed

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _polled(sent=(), received=(), failed=(), completed_ns=None):
    """Fake poll_stats_with_details() result for the given req ids."""

    def settled(ids):
        return [SimpleNamespace(req_id=r, completed_ns=completed_ns) for r in ids]

    return SimpleNamespace(
        sent=settled(sent), received=settled(received), failed=settled(failed)
    )


class Buffer:
    dtype = "torch.float8_e4m3fn"

    def __init__(self, pages=64):
        self.shape = (pages, 128, 8, 2, 128)

    def element_size(self):
        return 1

    def view(self, dtype):
        return SimpleNamespace(dtype=str(dtype), shape=self.shape)


class Bf16Buffer(Buffer):
    """Eight kv-heads of head_dim 128 in bf16: one head's K+V is 512 bytes."""

    dtype = "torch.bfloat16"

    def element_size(self):
        return 2


class Bf16FourHeadBuffer(Bf16Buffer):
    """Four kv-heads: a TP8 decode holds every head on two ranks."""

    def __init__(self, pages=64):
        self.shape = (pages, 128, 4, 2, 128)


class Fp8Buffer(Buffer):
    """Eight kv-heads in fp8: two heads share every 32-bit pack."""

    def __init__(self, pages=64):
        self.shape = (pages, 128, 4, 4, 128)


class Bf16SixteenHeadBuffer(Bf16Buffer):
    """Sixteen kv-heads, as a Gemma-4-style sliding-window pool beside four
    full-attention heads."""

    def __init__(self, pages=64):
        self.shape = (pages, 128, 16, 2, 128)


def unequal_groups(swa_buffer):
    """Four full-attention kv-heads beside a sliding-window pool with more."""
    return SimpleNamespace(
        full=[Bf16FourHeadBuffer()],
        swa=[swa_buffer()],
        full_layer_ids=[6],
        swa_layer_ids=[7],
        num_draft_full=0,
        num_draft_swa=0,
    )


def groups(full=(0, 2), swa=(1, 3), pages=64, buffer=Buffer, draft_full=()):
    return SimpleNamespace(
        full=[buffer(pages) for _ in (*full, *draft_full)],
        swa=[buffer(pages) for _ in swa],
        full_layer_ids=[*full, *draft_full],
        swa_layer_ids=list(swa),
        num_draft_full=len(draft_full),
        num_draft_swa=0,
    )


def sent(expires=None, registered=True):
    return dict(
        full=[20, 21],
        swa=[30],
        aux=b"",
        expires=time.monotonic() + 120 if expires is None else expires,
        registered=registered,
        replied=set(),
        start=0,
        parts=0,
    )


def transfer(producer=False, pp_size=1, pp_rank=0, attn_tp_size=1, attn_tp_rank=0):
    pipe = PipelineTransfer.__new__(PipelineTransfer)
    pipe.manager = SimpleNamespace(
        _state_lock=threading.Lock(),
        _aborted={},
        _norm_req_id=lambda value: value.removeprefix("resharded_transfer_"),
        _require_frames=Mock(),
        _prefill=SimpleNamespace(send_blocks={}, send_swa_blocks={}),
        _timeout_s=120,
        _monotonic=time.monotonic,
        _engine_poll_failing_since={},
        _failed=set(),
        failure_lock=threading.Lock(),
        failure_records={},
        update_status=Mock(),
        kv_args=SimpleNamespace(total_kv_head_num=8, total_swa_kv_head_num=8),
        local_ip="10.0.0.2",
        rank_port=31000,
        pp_size=pp_size,
        pp_rank=pp_rank,
        attn_tp_size=attn_tp_size,
        attn_tp_rank=attn_tp_rank,
    )
    for method in (
        "_poll_engine_status",
        "_fail_room_locked",
        "is_aborted",
        "_tombstone_reason_locked",
        "_reply_room_failed",
    ):
        setattr(
            pipe.manager,
            method,
            types.MethodType(getattr(RaidenKVManager, method), pipe.manager),
        )
    pipe.producer = producer
    pipe._init_overlap_state()
    pipe.lock = threading.RLock()
    pipe.engine = SimpleNamespace(poll_stats_with_details=Mock(return_value=_polled()))
    layers = groups(full=(6,), swa=(7,), buffer=Bf16Buffer)
    pipe.pools = pool_manifest(layers)
    pipe.tensors = list(layers.full) + list(layers.swa)
    pipe.client = Mock()
    pipe.clients = {}
    pipe.client_type = Mock(return_value=Mock())
    pipe.controller = "decode:30001"
    pipe.rank = pp_rank * attn_tp_size + attn_tp_rank
    pipe.unit_fields = dict(
        job_name="sglang",
        job_replica_id="decode:30001",
        data_name="prefill" if producer else "decode",
        data_replica_idx=pipe.rank,
    )
    pipe.unit = SimpleNamespace(**pipe.unit_fields)
    pipe.dst_units = [
        SimpleNamespace(
            **(pipe.unit_fields | {"data_replica_idx": pp_rank * attn_tp_size + rank})
        )
        for rank in range(attn_tp_size)
    ]
    pipe.executor = Mock()
    pipe.executor.submit.return_value = concurrent.futures.Future()
    pipe.hbm_mem_type = 2
    pipe.max_blocks = 4
    pipe.admission = StagingAdmission(8)
    pipe.sent, pipe.replies, pipe.receives = {}, {}, {}
    pipe.release_requests = {}
    pipe.done_send, pipe.done_recv, pipe.failed = set(), set(), set()
    pipe.retired, pipe.cleared = set(), set()
    pipe._send = Mock()
    return pipe


def receive(rank=0, pp_size=8, full=(20, 21), swa=(30,)):
    metadata = dict(
        controller="prefill:30001",
        rank=rank,
        pp_size=pp_size,
        full=2,
        swa=1,
        unit=dict(
            job_name="sglang",
            job_replica_id="prefill:30001",
            data_name="prefill",
            data_replica_idx=rank,
        ),
    )
    return Receive(
        full=list(full),
        swa=list(swa),
        aux_index=None,
        ranks={rank},
        sources={rank: metadata},
        deadline=time.monotonic() + 120,
    )


class TestStorageLayout(unittest.TestCase):
    def test_fp8_dtype_alias_queries_the_existing_uint8_storage_view(self):
        storage = SimpleNamespace(dtype="torch.uint8")
        tensor = SimpleNamespace(
            dtype="torch.float8_e4m3fn", view=Mock(return_value=storage)
        )
        expected = ((4, 3, 2, 1, 0), ((8, 128),), 8)
        getter = Mock(side_effect=[None, expected])
        self.assertEqual(materialized_storage_layout(tensor, getter), expected)
        self.assertIs(getter.call_args_list[0].args[0], tensor)
        self.assertIs(getter.call_args_list[1].args[0], storage)
        self.assertEqual(str(tensor.view.call_args.args[0]), "torch.uint8")

    def test_materialized_fp8_base_is_queried_directly(self):
        tensor = SimpleNamespace(dtype="torch.float8_e4m3fn", view=Mock())
        expected = ((4, 3, 2, 1, 0), ((8, 128),), 8)
        self.assertEqual(
            materialized_storage_layout(tensor, lambda _: expected), expected
        )
        tensor.view.assert_not_called()

    def test_other_unmaterialized_dtypes_are_not_reinterpreted(self):
        tensor = SimpleNamespace(dtype="torch.bfloat16", view=Mock())
        self.assertIsNone(materialized_storage_layout(tensor, lambda _: None))
        tensor.view.assert_not_called()


class TestLayerPools(unittest.TestCase):
    def test_hybrid_tags_keep_global_layer_ids_and_local_storage_indices(self):
        pools = pool_manifest(groups(full=(8, 10), swa=(9, 11)))
        self.assertEqual(
            [p["tag"] for p in pools], ["fa.l8", "fa.l10", "swa.l9", "swa.l11"]
        )
        self.assertEqual([p["storage_index"] for p in pools], [0, 1, 2, 3])

    def test_draft_layers_get_their_own_namespace(self):
        pools = pool_manifest(groups(full=(0, 1), swa=(), draft_full=(0,)))
        self.assertEqual([p["tag"] for p in pools], ["fa.l0", "fa.l1", "draft.fa.l0"])
        spans = source_spans(pools, [100, 103], [12])
        self.assertEqual(spans[2].block_ids, [100, 103])

    def test_duplicate_layer_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "distinct global layer IDs"):
            pool_manifest(groups(full=(8, 8), swa=()))

    def test_stage_requires_a_full_attention_pool(self):
        with self.assertRaisesRegex(ValueError, "full-attention KV"):
            pool_manifest(groups(full=(), swa=(1,)))

    def test_every_layer_owns_all_token_pages_in_its_namespace(self):
        pools = pool_manifest(groups(full=(7, 8), swa=(9,)))
        spans = source_spans(pools, [100, 103], [12])
        self.assertEqual([s.block_ids for s in spans], [[100, 103], [100, 103], [12]])
        self.assertEqual([s.dst_space_version for s in spans], [1, 1, 1])
        for pool, entry in zip(pools, spans):
            stride = pool["block_stride_bytes"]
            count = len(entry.block_ids)
            self.assertEqual([s.dst_block_index for s in entry.spans], [0] * count)
            self.assertEqual(
                [s.dst_offset_bytes for s in entry.spans],
                [index * stride for index in range(count)],
            )
            self.assertEqual(entry.declared_bytes, count * stride)

    def test_global_spans_clip_full_and_swa_prefixes_by_bytes(self):
        pools = [
            dict(tag="fa.l8", block_stride_bytes=4096),
            dict(tag="swa.l9", block_stride_bytes=2048),
        ]
        spans = source_spans(pools, [100, 103, 109], [12, 16])
        destination = destination_blocks(pools, [23, 24], [30], 3, 2)
        self.assertEqual(destination["dst_skip_bytes"], [4096, 2048])
        # Global byte offsets rebase onto the uncached destination suffix.
        clipped = [
            [
                (
                    entry.block_ids[span.src_block_ordinal],
                    span.dst_offset_bytes - skip,
                    span.size_bytes,
                )
                for span in entry.spans
                if span.dst_offset_bytes >= skip
            ]
            for entry, skip in zip(spans, destination["dst_skip_bytes"])
        ]
        self.assertEqual(
            clipped, [[(103, 0, 4096), (109, 4096, 4096)], [(16, 0, 2048)]]
        )

    def test_destination_clips_full_prefix_and_window_independently(self):
        pools = pool_manifest(groups(full=(8,), swa=(9,)))
        result = destination_blocks(pools, [23, 24], [30], 4, 1)
        self.assertEqual(result["dst_device_block_ids"], [23, 24, 30])
        self.assertEqual(result["dst_block_counts"], [2, 1])
        self.assertEqual(
            result["dst_skip_bytes"], [2 * pools[0]["block_stride_bytes"], 0]
        )

    def test_destination_omits_the_decode_only_page(self):
        pools = pool_manifest(groups(full=(8,), swa=()))
        result = destination_blocks(pools, [23, 24, 25], [], 2, 0)
        self.assertEqual(result["dst_device_block_ids"], [23, 24])
        self.assertEqual(result["dst_skip_bytes"], [0])

    def test_head_slice_spans_route_one_slice_per_rank(self):
        layers = groups(full=(6,), swa=(7,), buffer=Bf16Buffer)
        pools = pool_manifest(layers)
        spans = head_slice_spans(
            pools, layers.full + layers.swa, [20, 21], [30], 8, 8, 8
        )
        self.assertEqual([len(entry.spans) for entry in spans], [16, 8])
        self.assertEqual(
            [entry.declared_bytes for entry in spans], [2 * 524288, 524288]
        )
        self.assertEqual(
            {span.dst_unit_ordinal for span in spans[0].spans}, set(range(8))
        )
        (span,) = [
            span
            for span in spans[0].spans
            if span.dst_unit_ordinal == 3 and span.src_block_ordinal == 1
        ]
        # Rank 3 takes head 3 of every token: 512 bytes every 4096, landing
        # contiguously in its own 65536-byte page.
        self.assertEqual(
            (span.src_offset_bytes, span.size_bytes, span.src_stride_bytes),
            (1536, 512, 4096),
        )
        self.assertEqual(
            (span.dst_block_index, span.dst_offset_bytes, span.dst_stride_bytes),
            (0, 65536, 512),
        )
        self.assertEqual((span.count, spans[0].dst_space_version), (128, 1))

    def test_swa_head_slices_use_the_swa_head_count(self):
        """MiMo-V2 (4 full / 8 SWA kv-heads) and Gemma-4 (4 / 16): each rank
        takes its own heads of each pool from the TP1 producer page."""
        head = 2 * 128 * 2  # one kv-head's K+V in bf16
        for swa_buffer, swa_heads in ((Bf16Buffer, 8), (Bf16SixteenHeadBuffer, 16)):
            for dst_tp in (2, 4):
                layers = unequal_groups(swa_buffer)
                entries = head_slice_spans(
                    pool_manifest(layers),
                    layers.full + layers.swa,
                    [20, 21],
                    [30],
                    dst_tp,
                    4,
                    swa_heads,
                )
                for entry, heads, pages in zip(entries, (4, swa_heads), (2, 1)):
                    with self.subTest(tag=entry.tag, heads=heads, dst_tp=dst_tp):
                        self.assertEqual(len(entry.spans), dst_tp * pages)
                        width = heads // dst_tp * head
                        for span in entry.spans:
                            rank = span.dst_unit_ordinal
                            self.assertEqual(
                                (
                                    span.src_offset_bytes,
                                    span.size_bytes,
                                    span.src_stride_bytes,
                                ),
                                (rank * width, width, heads * head),
                            )
                            self.assertEqual(
                                (span.dst_offset_bytes, span.dst_stride_bytes),
                                (span.src_block_ordinal * 128 * width, width),
                            )

    def test_head_slice_spans_declare_every_replica_of_a_shared_head(self):
        """With fewer kv-heads than decode ranks each head lands on several
        ranks, and the registry requires declared_bytes to equal the span sum:
        three 262144-byte pages reach a TP8 decode as 1572864 bytes."""
        layers = groups(full=(6,), swa=(), buffer=Bf16FourHeadBuffer)
        (entry,) = head_slice_spans(
            pool_manifest(layers), layers.full, [20, 21, 22], [], 8, 4, 4
        )
        self.assertEqual(entry.declared_bytes, 1572864)
        self.assertEqual(
            sum(span.size_bytes * span.count for span in entry.spans), 1572864
        )
        offsets = {
            span.dst_unit_ordinal: span.src_offset_bytes
            for span in entry.spans
            if span.src_block_ordinal == 0
        }
        self.assertEqual(offsets, {rank: rank // 2 * 512 for rank in range(8)})

    def test_fp8_head_slices_take_whole_packs(self):
        """FP8 packs two kv-heads into each 32-bit word; a rank's slice of every
        token is exactly its own heads' K and V bytes."""
        layers = groups(full=(6,), swa=(), buffer=Fp8Buffer)
        (tensor,) = layers.full
        _, page_tokens, packs, packing, head_dim = tensor.shape
        row_bytes = packs * packing * head_dim * tensor.element_size()
        heads_per_rank = 8 // 4
        rank_bytes = heads_per_rank * 2 * head_dim * tensor.element_size()
        (entry,) = head_slice_spans(
            pool_manifest(layers), layers.full, [20, 21], [], 4, 8, 8
        )
        self.assertEqual(len(entry.spans), 8)
        for span in entry.spans:
            rank = span.dst_unit_ordinal
            self.assertEqual(
                (span.src_offset_bytes, span.size_bytes, span.src_stride_bytes),
                (rank * rank_bytes, rank_bytes, row_bytes),
            )
            self.assertEqual(
                (span.dst_offset_bytes, span.dst_stride_bytes, span.count),
                (
                    span.src_block_ordinal * page_tokens * rank_bytes,
                    rank_bytes,
                    page_tokens,
                ),
            )

    def test_partial_fp8_head_packs_are_rejected(self):
        layers = groups(full=(6,), swa=(), buffer=Fp8Buffer)
        with self.assertRaisesRegex(ValueError, "Partial FP8"):
            head_slice_spans(pool_manifest(layers), layers.full, [20], [], 8, 8, 8)

    def test_fingerprint_ignores_the_per_rank_head_count(self):
        getter = lambda tensor: ((4, 3, 2, 1, 0), ((8, 128),), 8)
        producer = groups(full=(6,), swa=(), buffer=Bf16Buffer)
        consumer = groups(full=(6,), swa=(), buffer=Bf16Buffer)
        consumer.full[0].shape = (64, 128, 1, 2, 128)
        self.assertEqual(
            layout_fingerprint(pool_manifest(producer), producer.full, getter),
            layout_fingerprint(pool_manifest(consumer), consumer.full, getter),
        )

    def test_fingerprint_matches_an_uneven_stage_subset(self):
        full = groups(full=tuple(range(62)), swa=(), pages=64)
        stage = groups(full=tuple(range(55, 62)), swa=(), pages=192)
        getter = lambda tensor: ((4, 3, 2, 1, 0), ((8, 128),), 8)
        self.assertEqual(
            layout_fingerprint(pool_manifest(full), full.full, getter),
            layout_fingerprint(pool_manifest(stage), stage.full, getter),
        )
        self.assertNotEqual(
            layout_fingerprint(pool_manifest(full), full.full, getter),
            layout_fingerprint(
                pool_manifest(stage),
                stage.full,
                lambda tensor: ((4, 3, 2, 1, 0), ((4, 128),), 8),
            ),
        )

    def test_fingerprint_matches_between_stages_with_and_without_the_draft(self):
        """Only the last prefill stage and the decode hold the draft pool; every
        stage must still fingerprint like the decode, or the controller rejects
        each transfer."""
        getter = lambda tensor: ((4, 3, 2, 1, 0), ((8, 128),), 8)
        stage = groups(full=(0, 1), swa=())
        decode = groups(full=(0, 1, 2), swa=(), draft_full=(0,))
        self.assertEqual(
            layout_fingerprint(pool_manifest(stage), stage.full, getter),
            layout_fingerprint(pool_manifest(decode), decode.full, getter),
        )

    def test_pp_ranks_have_distinct_ports_and_share_the_host_budget(self):
        ranks = {
            node_worker_rank(0, 1, 0, 1, 0, pp_rank=stage, pp_size=8)
            for stage in range(8)
        }
        self.assertEqual(ranks, set(range(8)))
        self.assertEqual(ranks_per_node(1, 1, 1, 1, pp_size=8), 8)
        self.assertEqual(ranks_per_node(1, 1, 1, 2, pp_size=8), 4)


class TestLegacyPoolBudget(unittest.TestCase):
    @staticmethod
    def pools(extents):
        return [
            dict(
                storage_index=i,
                base_offset_bytes=0,
                block_stride_bytes=4096,
                num_blocks=n,
            )
            for i, n in enumerate(extents)
        ]

    def test_uniform_pools_use_logical_metadata_despite_physical_tile_padding(self):
        tensors = [Buffer(64), Buffer(64)]
        padded = lambda _: ((4, 3, 2, 1, 0), ((8, 128),), 8)
        self.assertEqual(
            legacy_slot_host_bytes(tensors, 4, padded), 128 * 8 * 2 * 128 * 2 * 4
        )

    def test_mixed_full_window_pools_charge_native_first_tile_padding(self):
        tensors = [Buffer(64), Buffer(9)]
        padded = lambda _: ((4, 3, 2, 1, 0), ((8, 128),), 8)
        self.assertEqual(
            legacy_slot_host_bytes(tensors, 4, padded), 128 * 8 * 8 * 128 * 2 * 4
        )
        unpadded = lambda _: ((4, 2, 3, 1, 0), ((8, 128),), 8)
        self.assertEqual(
            legacy_slot_host_bytes(tensors, 4, unpadded), 128 * 8 * 2 * 128 * 2 * 4
        )

    def test_unknown_and_non_block_major_layouts_are_rejected(self):
        tensors = [Buffer(64), Buffer(9)]
        with self.assertRaisesRegex(ValueError, "materialized page layouts"):
            legacy_slot_host_bytes(tensors, 4, lambda _: None)
        with self.assertRaisesRegex(ValueError, "block-major"):
            legacy_slot_host_bytes(tensors, 4, lambda _: ((0, 4, 3, 2, 1), (), 8))
        with self.assertRaisesRegex(ValueError, "two-dimensional tile"):
            legacy_slot_host_bytes(
                tensors, 4, lambda _: ((4, 3, 2, 1, 0), ((8, 8, 128),), 8)
            )

    def test_existing_uniform_buffers_cover_small_pools(self):
        slots, peak = legacy_mirror_slots(self.pools([2, 3]), 8 * 4096, 4, 32 * 4096)
        self.assertEqual((slots, peak), (4, 32 * 4096))

    def test_mixed_full_and_window_pools_include_replacement_peak(self):
        slots, peak = legacy_mirror_slots(self.pools([40, 2]), 8 * 4096, 4, 56 * 4096)
        self.assertEqual((slots, peak), (2, 56 * 4096))

    def test_a_mirror_that_cannot_fit_one_slot_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "including pool replacement"):
            legacy_mirror_slots(self.pools([40, 2]), 8 * 4096, 4, 47 * 4096)

    def test_two_growing_storages_bound_last_replacement(self):
        self.assertEqual(
            legacy_mirror_slots(self.pools([40, 60]), 8 * 4096, 4, 108 * 4096),
            (2, 108 * 4096),
        )

    def test_staging_above_the_premapped_buffer_warns(self):
        with self.assertLogs(
            "sglang_tpu.disaggregation.raiden.pipeline", level="WARNING"
        ) as logs:
            self.assertTrue(
                warn_if_staging_exceeds_premapped(
                    staging_bytes=33 << 30, premapped_bytes=4 << 30
                )
            )
        self.assertIn("33.0 GiB", logs.output[0])
        self.assertIn("TPU_PREMAPPED_BUFFER_SIZE", logs.output[0])

    def test_staging_within_the_premapped_buffer_is_silent(self):
        self.assertFalse(
            warn_if_staging_exceeds_premapped(
                staging_bytes=32 << 30, premapped_bytes=36 << 30
            )
        )

    def test_aliased_or_offset_storage_is_rejected(self):
        pools = self.pools([20, 30])
        pools[1]["storage_index"] = 0
        with self.assertRaisesRegex(ValueError, "one pool per storage"):
            legacy_mirror_slots(pools, 8, 1, 100)
        pools = self.pools([20])
        pools[0]["base_offset_bytes"] = 4
        with self.assertRaisesRegex(ValueError, "zero storage offsets"):
            legacy_mirror_slots(pools, 4, 1, 100)


class TestPipelineLifetimes(unittest.TestCase):
    def test_invalid_native_uuid_is_rejected_before_registration(self):
        source = transfer(producer=True)
        destination = transfer()
        for room in (0, -1, 1 << 63):
            with self.assertRaisesRegex(ValueError, "positive signed 64-bit"):
                source.finalize_send(room, None)
            with self.assertRaisesRegex(ValueError, "positive signed 64-bit"):
                destination.register_decode(room, [], [], None, [])
        source.client.register_request_blocks.assert_not_called()

    def test_pp_destination_enumerates_all_source_work_units(self):
        pipe = transfer(pp_size=8, pp_rank=3)
        pipe.receives[7] = receive(rank=3)
        self.assertTrue(pipe.maybe_start_pull(7))
        args = pipe.executor.submit.call_args.kwargs
        self.assertEqual(
            [u.data_replica_idx for u in args["src_units"]], list(range(8))
        )
        self.assertEqual(args["transfer_pool_tags"], ["fa.l6", "swa.l7"])
        self.assertEqual(args["dst_units"], [pipe.unit])
        self.assertEqual(args["dst_mem_type"], 2)

    def test_tp_rank_zero_plans_one_transfer_into_every_rank(self):
        pipe = transfer(attn_tp_size=2)
        pipe.receives[7] = receive()
        self.assertTrue(pipe.maybe_start_pull(7))
        args = pipe.executor.submit.call_args.kwargs
        self.assertEqual([u.data_replica_idx for u in args["dst_units"]], [0, 1])

    def test_tp_follower_waits_for_its_own_engine(self):
        pipe = transfer(attn_tp_size=2, attn_tp_rank=1)
        pipe.receives[7] = receive()
        self.assertTrue(pipe.maybe_start_pull(7))
        pipe.executor.submit.assert_not_called()
        self.assertTrue(pipe.active(7))
        pipe.engine.poll_stats_with_details.return_value = _polled(received=["7"])
        _, received, _ = pipe.poll()
        self.assertFalse(pipe.active(7))
        self.assertIn("7", received)

    def test_tp_follower_fails_after_the_planning_and_transfer_deadlines(self):
        pipe = transfer(attn_tp_size=2, attn_tp_rank=1)
        request = receive()
        request.deadline = time.monotonic() - 121
        pipe.receives[7] = request
        self.assertTrue(pipe.maybe_start_pull(7))
        _, _, failed = pipe.poll()
        self.assertIn("7", failed)
        self.assertFalse(pipe.active(7))

    def test_producer_registers_once_with_the_consumer_head_geometry(self):
        pipe = transfer(producer=True)
        pipe.manager._prefill.send_blocks[7] = [20, 21]
        pipe.manager._prefill.send_swa_blocks[7] = [30]
        self.assertTrue(pipe.finalize_send(7, None))
        pipe.client.register_request_blocks.assert_not_called()
        request = [MessageTag.PIPELINE_REQ, b"7", b"tcp://decode:31003"]
        pipe.on_message(request + [b'{"attn_tp_size": 8, "attn_tp_rank": 3}'])
        spans = pipe.client.register_request_blocks.call_args.kwargs["pool_spans"]
        self.assertEqual([len(entry.spans) for entry in spans], [16, 8])
        request[2] = b"tcp://decode:31000"
        pipe.on_message(request + [b'{"attn_tp_size": 8, "attn_tp_rank": 0}'])
        pipe.client.register_request_blocks.assert_called_once()
        self.assertEqual(
            [call.args[0] for call in pipe._send.call_args_list],
            ["tcp://decode:31003", "tcp://decode:31000"],
        )
        frames = pipe._send.call_args.args[1]
        self.assertEqual(frames[0], MessageTag.PIPELINE_READY)
        self.assertEqual(json.loads(frames[2])["full"], 2)

    def test_registration_slices_swa_pools_over_the_swa_head_count(self):
        pipe = transfer(producer=True)
        layers = unequal_groups(Bf16Buffer)
        pipe.pools = pool_manifest(layers)
        pipe.tensors = layers.full + layers.swa
        pipe.manager.kv_args = SimpleNamespace(
            total_kv_head_num=4, total_swa_kv_head_num=8
        )
        pipe.manager._prefill.send_blocks[7] = [20, 21]
        pipe.manager._prefill.send_swa_blocks[7] = [30]
        self.assertTrue(pipe.finalize_send(7, None))
        request = [MessageTag.PIPELINE_REQ, b"7", b"tcp://decode:31000"]
        pipe.on_message(request + [b'{"attn_tp_size": 4, "attn_tp_rank": 0}'])
        full, swa = pipe.client.register_request_blocks.call_args.kwargs["pool_spans"]
        # Rank 3 of 4: one of four full heads, two of eight SWA heads.
        self.assertEqual(
            {
                (s.src_offset_bytes, s.size_bytes)
                for s in full.spans
                if s.dst_unit_ordinal == 3
            },
            {(3 * 512, 512)},
        )
        self.assertEqual(
            {
                (s.src_offset_bytes, s.size_bytes)
                for s in swa.spans
                if s.dst_unit_ordinal == 3
            },
            {(3 * 1024, 1024)},
        )

    def test_failed_registration_fails_the_room_not_the_prefill_scheduler(self):
        """Registration runs on the prefill scheduler thread when the decode
        asked first; consumers that disagree on attention TP, a head slice the
        packing cannot split, and a registry error each used to escape
        finalize_send and stop the server."""
        fp8 = groups(full=(6,), swa=(), buffer=Fp8Buffer)
        cases = [
            ("disagree on attention TP", [8, 1], None, None),
            ("Partial FP8", [8], fp8, None),
            ("controller unreachable", [1], None, RuntimeError),
        ]
        for reason, widths, layers, registry_error in cases:
            with self.subTest(reason=reason):
                pipe = transfer(producer=True)
                if layers is not None:
                    pipe.pools = pool_manifest(layers)
                    pipe.tensors = list(layers.full)
                if registry_error is not None:
                    pipe.client.register_request_blocks.side_effect = registry_error(
                        reason
                    )
                pipe.manager._prefill.send_blocks[7] = [20, 21]
                pipe.replies[7] = {
                    f"tcp://decode:{port}": dict(attn_tp_size=width)
                    for port, width in enumerate(widths)
                }
                self.assertFalse(pipe.finalize_send(7, None))
                self.assertFalse(pipe.active(7))
                self.assertIn("7", pipe.manager._failed)
                self.assertIn(reason, pipe.manager.failure_records[7])
                pipe._send.assert_not_called()
                pipe.poll()
                pipe.client.complete_request_blocks.assert_not_called()

    def test_registration_runs_outside_the_manager_state_lock(self):
        """The registration RPC ran under the manager's state lock, stalling
        every scheduler-thread reader of that lock for the RPC's duration."""
        pipe = transfer(producer=True)
        held = []
        pipe.client.register_request_blocks.side_effect = lambda **_: held.append(
            pipe.manager._state_lock.locked()
        )
        request = [MessageTag.PIPELINE_REQ, b"7", b"tcp://decode:31000"]
        consumer = b'{"attn_tp_size": 1, "attn_tp_rank": 0}'
        pipe.manager._prefill.send_blocks[7] = [20, 21]
        pipe.finalize_send(7, None)
        pipe.on_message(request + [consumer])
        request[1] = b"8"
        pipe.on_message(request + [consumer])
        pipe.manager._prefill.send_blocks[8] = [22]
        pipe.finalize_send(8, None)
        self.assertEqual(held, [False, False])

    def test_request_for_a_failed_room_is_answered_with_room_failed(self):
        """A pipeline request for a room this stage already failed must tell
        the decode, or the decode holds the room's pages until its waiting
        timeout."""
        pipe = transfer(producer=True)
        pipe.manager._aborted[7] = time.time()
        pipe.manager._send_multipart_locked = Mock()
        consumer = b'{"attn_tp_size": 1, "attn_tp_rank": 0}'
        pipe.on_message(
            [MessageTag.PIPELINE_REQ, b"7", b"tcp://decode:31000", consumer]
        )
        pipe.client.register_request_blocks.assert_not_called()
        self.assertNotIn(7, pipe.replies)
        endpoint, frames = pipe.manager._send_multipart_locked.call_args.args
        self.assertEqual(endpoint, "tcp://decode:31000")
        self.assertEqual(RoomFailed.from_frames(frames).room, 7)

    def test_unrequested_room_expires_without_a_native_cancel(self):
        pipe = transfer(producer=True)
        pipe.sent[7] = sent(expires=0, registered=False)
        _, _, failed = pipe.poll()
        self.assertFalse(pipe.active(7))
        self.assertIn("7", failed)
        pipe.client.cancel_request_blocks_if_unclaimed.assert_not_called()
        pipe.client.complete_request_blocks.assert_not_called()

    def test_matching_pipeline_rejects_different_stage_layer_subsets(self):
        """A topology mismatch found on the listener thread fails the room at
        once instead of leaving it to wait out the transfer deadline."""
        from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager

        pipe = transfer(pp_size=8, pp_rank=3)
        pipe.receives[7] = receive(rank=3)
        meta = pipe.receives[7].sources[3] | {"pool_tags": ["fa.l12", "swa.l13"]}
        manager = RaidenKVManager.__new__(RaidenKVManager)
        manager.attn_tp_rank = 0
        manager._pipeline_enabled = True
        manager._pipeline_init_lock = threading.Lock()
        manager._pipeline = pipe
        manager._state_lock = threading.Lock()
        manager._status_lock = threading.Lock()
        manager._aborted = {}
        manager._failed = set()
        manager.failure_lock = threading.Lock()
        manager.failure_records = {}
        manager.request_status = {7: KVPoll.Transferring}
        with self.assertRaisesRegex(ValueError, "same layers per stage"):
            manager._dispatch(
                [MessageTag.PIPELINE_READY, b"7", json.dumps(meta).encode(), b""]
            )
        self.assertIn("7", manager._failed)
        self.assertIn(7, manager._aborted)
        self.assertEqual(manager.request_status[7], KVPoll.Failed)
        self.assertIn("same layers per stage", manager.failure_records[7])

    def test_dp_destination_takes_aux_from_the_last_stage_only(self):
        """Stages before the last carry no draft hidden states; their aux must not
        overwrite the last stage's."""
        pipe = transfer()
        pipe.manager.kv_args.aux_data_ptrs = pipe.manager.kv_args.aux_item_lens = []
        request = receive()
        request.ranks, request.aux_index = set(range(8)), 5
        pipe.receives[7] = request
        with patch("sglang_tpu.disaggregation.raiden.pipeline.write_aux_blob") as write:
            for rank in (7, 0):
                meta = receive(rank=rank).sources[rank]
                pipe.on_message(
                    [
                        MessageTag.PIPELINE_READY,
                        b"7",
                        json.dumps(meta).encode(),
                        f"aux{rank}".encode(),
                    ]
                )
        write.assert_called_once()
        self.assertEqual(write.call_args.args[2:], (5, b"aux7"))

    def test_dp_destination_waits_for_every_pipeline_source(self):
        pipe = transfer()
        request = receive()
        request.ranks = set(range(8))
        pipe.receives[7] = request
        self.assertFalse(pipe.maybe_start_pull(7))
        pipe.executor.submit.assert_not_called()

    def test_native_receive_completion_waits_for_submission_to_return(self):
        pipe = transfer()
        request = receive()
        pipe.receives[7] = request
        pipe.maybe_start_pull(7)
        pipe.engine.poll_stats_with_details.return_value = _polled(
            received=["resharded_transfer_7"]
        )
        _, received, _ = pipe.poll()
        self.assertTrue(pipe.active(7))
        self.assertNotIn("7", received)
        self.assertIn(7, pipe.admission)
        request.future.set_result(True)
        _, received, _ = pipe.poll()
        self.assertFalse(pipe.active(7))
        self.assertIn("7", received)
        self.assertNotIn(7, pipe.admission)

    def test_known_missing_source_registration_releases_admission_for_retry(self):
        pipe = transfer()
        pipe.receives[7] = receive()
        pipe.maybe_start_pull(7)
        pipe.receives[7].future.set_exception(
            RuntimeError("Missing producer block registration")
        )
        _, _, failed = pipe.poll()
        self.assertIsNone(pipe.receives[7].future)
        self.assertNotIn(7, pipe.admission)
        self.assertNotIn("7", failed)

    def test_cancelled_source_registration_fails_without_native_transfer(self):
        pipe = transfer()
        pipe.receives[7] = receive()
        pipe.maybe_start_pull(7)
        pipe.receives[7].future.set_exception(
            RuntimeError("Request block registration was cancelled")
        )
        _, _, failed = pipe.poll()
        self.assertFalse(pipe.active(7))
        self.assertNotIn(7, pipe.admission)
        self.assertIn("7", failed)

    def test_expired_missing_registration_fails_without_native_transfer(self):
        pipe = transfer()
        pipe.receives[7] = receive()
        pipe.receives[7].deadline = 0
        pipe.maybe_start_pull(7)
        pipe.receives[7].future.set_exception(
            RuntimeError("Missing producer block registration")
        )
        _, _, failed = pipe.poll()
        self.assertFalse(pipe.active(7))
        self.assertIn("7", failed)

    def test_unknown_submission_error_fails_the_room_and_holds_its_pages(self):
        """A planner rejection or an unreachable prefill controller used to
        escape the decode scheduler's poll. The controller arms receivers
        before senders, so the failed room keeps its pages and staging until
        the engine settles it."""
        for outcome in (RuntimeError("bad pool geometry"), False):
            with self.subTest(outcome=outcome):
                pipe = transfer()
                pipe.receives[7] = receive()
                pipe.maybe_start_pull(7)
                if outcome is False:
                    pipe.receives[7].future.set_result(False)
                else:
                    pipe.receives[7].future.set_exception(outcome)
                _, _, failed = pipe.poll()
                self.assertIn("7", failed)
                self.assertIsNone(pipe.receives[7].future)
                self.assertIn("7", pipe.manager._failed)
                if outcome is not False:
                    self.assertIn(str(outcome), pipe.manager.failure_records[7])
                self.assertTrue(pipe.active(7))
                self.assertIn(7, pipe.admission)
                pipe.engine.poll_stats_with_details.return_value = _polled(failed=["7"])
                pipe.poll()
                self.assertFalse(pipe.active(7))
                self.assertNotIn(7, pipe.admission)

    def test_unsettled_failed_room_is_released_at_the_native_deadline(self):
        pipe = transfer()
        pipe.receives[7] = receive()
        pipe.maybe_start_pull(7)
        pipe.receives[7].future.set_exception(RuntimeError("bad pool geometry"))
        pipe.poll()
        self.assertTrue(pipe.active(7))
        pipe.receives[7].armed_until = time.monotonic() - 1
        pipe.poll()
        self.assertFalse(pipe.active(7))
        self.assertNotIn(7, pipe.admission)

    def test_raising_engine_poll_is_retried_on_the_next_poll(self):
        """A raising poll_stats_with_details takes no completion; the pipeline
        polls through the manager's retry instead of stopping the scheduler."""
        pipe = transfer(attn_tp_size=2, attn_tp_rank=1)
        pipe.receives[7] = receive()
        pipe.maybe_start_pull(7)
        pipe.engine.poll_stats_with_details.side_effect = RuntimeError("injected")
        self.assertEqual(pipe.poll(), (set(), set(), set()))
        self.assertTrue(pipe.active(7))
        pipe.engine.poll_stats_with_details.side_effect = None
        pipe.engine.poll_stats_with_details.return_value = _polled(received=["7"])
        _, received, _ = pipe.poll()
        self.assertIn("7", received)

    def test_aborted_claimed_source_keeps_its_pages(self):
        pipe = transfer(producer=True)
        pipe.sent[7] = sent()
        pipe.manager._aborted[7] = time.time()
        pipe.client.cancel_request_blocks_if_unclaimed.return_value = False
        pipe.poll()
        self.assertTrue(pipe.active(7))
        pipe.clear(7)
        self.assertIn(7, pipe.sent)
        pipe.engine.poll_stats_with_details.return_value = _polled(sent=["7"])
        pipe.poll()
        self.assertFalse(pipe.active(7))
        self.assertNotIn(7, pipe.sent)
        pipe.client.complete_request_blocks.assert_called_once_with("7", 7, pipe.unit)

    def test_mixed_pp_cache_hits_vote_only_for_the_completed_stage(self):
        cached = transfer(producer=True, pp_size=2, pp_rank=0)
        pulling = transfer(producer=True, pp_size=2, pp_rank=1)
        for pipe in (cached, pulling):
            pipe.sent[7] = sent()
        cached.client.complete_request_blocks.side_effect = RuntimeError(
            "Only an empty producer registration may complete before the request block snapshot is claimed"
        )
        cached.on_message([MessageTag.PIPELINE_RELEASE, b"7", b"2"])
        self.assertTrue(cached.active(7))
        self.assertTrue(pulling.active(7))
        cached.client.cancel_request_blocks_if_unclaimed.assert_not_called()
        cached.client.complete_request_blocks.side_effect = None
        cached.release_requests[7]["next_attempt"] = 0
        cached.poll()
        self.assertFalse(cached.active(7))
        self.assertTrue(pulling.active(7))
        cached.client.complete_request_blocks.assert_called_with("7", 7, cached.unit)
        cached.client.cancel_request_blocks_if_unclaimed.assert_not_called()
        pulling.engine.poll_stats_with_details.return_value = _polled(sent=["7"])
        pulling.poll()
        self.assertFalse(pulling.active(7))
        pulling.client.complete_request_blocks.assert_called_once_with(
            "7", 7, pulling.unit
        )
        pulling.client.cancel_request_blocks_if_unclaimed.assert_not_called()

    def test_all_pp_cache_hits_cancel_only_after_unclaimed_expiry(self):
        for rank in range(2):
            pipe = transfer(producer=True, pp_size=2, pp_rank=rank)
            pipe.sent[7] = sent()
            pipe.client.complete_request_blocks.side_effect = RuntimeError(
                "Only an empty producer registration may complete before the request block snapshot is claimed"
            )
            pipe.on_message([MessageTag.PIPELINE_RELEASE, b"7", b"2"])
            self.assertTrue(pipe.active(7))
            pipe.client.cancel_request_blocks_if_unclaimed.assert_not_called()
            pipe.sent[7]["expires"] = 0
            pipe.client.cancel_request_blocks_if_unclaimed.return_value = True
            sent_rooms, _, failed = pipe.poll()
            self.assertFalse(pipe.active(7))
            self.assertIn("7", sent_rooms)
            self.assertNotIn("7", failed)
            pipe.client.cancel_request_blocks_if_unclaimed.assert_called_once_with(
                "7", 7
            )

    def test_native_source_completion_retires_exactly_once(self):
        pipe = transfer(producer=True)
        pipe.sent[7] = sent()
        pipe.engine.poll_stats_with_details.return_value = _polled(sent=["7"])
        pipe.poll()
        pipe.engine.poll_stats_with_details.return_value = _polled()
        pipe.poll()
        pipe.client.complete_request_blocks.assert_called_once_with("7", 7, pipe.unit)

    def test_full_prefix_hit_releases_source_without_starting_transfer(self):
        pipe = transfer()
        request = receive(full=(), swa=())
        request.endpoints = ("tcp://prefill:30002",)
        pipe.receives[7] = request
        self.assertTrue(pipe.maybe_start_pull(7))
        _, received, _ = pipe.poll()
        self.assertIn("7", received)
        pipe.executor.submit.assert_not_called()
        pipe._send.assert_called_once_with(
            "tcp://prefill:30002", [MessageTag.PIPELINE_RELEASE, b"7", b"1"]
        )


class TestPipelineInitialization(unittest.TestCase):
    def setUp(self):
        override = get_context().override_server_args(
            speculative_algorithm=None, nnodes=1
        )
        override.install()
        self.addCleanup(override.restore)

    def test_resolved_speculative_mode_initializes(self):
        get_context().override("test.speculative", speculative_algorithm="EAGLE3")
        self._initialize()

    def test_legacy_host_budget_uses_resolved_node_count(self):
        get_context().override("test.multihost", nnodes=2)
        with patch(
            "sglang_tpu.disaggregation.raiden.pipeline.ranks_per_node",
            wraps=ranks_per_node,
        ) as local_ranks:
            self._initialize(legacy=True)
        self.assertEqual(local_ranks.call_args.args[3], 2)

    def _initialize(self, legacy=False):
        events = []
        buffers = groups(full=(0,), swa=())
        manager = SimpleNamespace(
            disaggregation_mode="decode",
            attn_tp_size=1,
            attn_tp_rank=0,
            attn_dp_size=1,
            system_dp_size=1,
            attn_cp_size=1,
            pp_size=1,
            pp_rank=0,
            local_ip="10.0.0.1",
            _control_port=31000,
            _local_rank=0,
            _timeout_s=120,
            server_args=SimpleNamespace(speculative_algorithm=None, nnodes=1),
            kv_args=SimpleNamespace(page_size=128),
            _max_request_blocks=lambda: 4,
            _num_slots=lambda *args, **kwargs: 2,
        )

        def layout(tensor):
            self.assertTrue(tensor.materialized)
            events.append("layout")
            return ((4, 2, 3, 1, 0), ((8, 128),), 8)

        def synchronize(tensors, wait):
            events.append("materialize")
            self.assertTrue(wait)
            for tensor in tensors:
                tensor.materialized = True

        def engine(**kwargs):
            events.append("engine")
            for tensor in kwargs["kv_caches"]:
                tensor.materialized = True
            return SimpleNamespace(
                transfer_address="10.0.0.1:31001",
                listener_address="10.0.0.1:31002",
                register_pools=Mock(
                    side_effect=lambda *args, **kwargs: events.append("pools")
                ),
            )

        def register_pools(
            self, pools, *, staging_leases=0, staging_blocks_per_pool=None
        ):
            pass

        def legacy_register_pools(self, pools):
            pass

        engine.register_pools = legacy_register_pools if legacy else register_pools
        client = Mock()
        client.register_work_unit.side_effect = lambda **kwargs: events.append(
            "work_unit"
        )
        native = {
            "torch_tpu._internal.sync.sync": SimpleNamespace(synchronize=synchronize),
            "torch_tpu._internal.compile": SimpleNamespace(
                tpu_torch_compile=SimpleNamespace(
                    get_device_layout_if_materialized=layout
                )
            ),
            "tpu_sync.api.torch.kv_cache_manager": SimpleNamespace(
                KVCacheManager=engine
            ),
            "tpu_sync.api.torch.reshard_client": SimpleNamespace(
                ReshardClient=lambda address: client
            ),
            "tpu_sync.api.torch.reshard_store": SimpleNamespace(
                RaidenId=lambda **kwargs: kwargs,
                ReshardStore=lambda *args, **kwargs: SimpleNamespace(
                    reshard_service_port=32000,
                    raiden_controller_address="10.0.0.1:32001",
                ),
            ),
            "tpu_sync.rpc": SimpleNamespace(
                raiden_service_pb2=SimpleNamespace(
                    MEMORY_TYPE_HBM=2,
                    StartTransferRequest=SimpleNamespace(
                        DESCRIPTOR=SimpleNamespace(
                            fields_by_name={"wire_pool_indices": object()}
                        )
                    ),
                )
            ),
        }
        with patch.dict(sys.modules, native):
            pipe = PipelineTransfer(manager, buffers)
        expected = ["engine", "pools", "layout", "work_unit"]
        if legacy:
            expected = ["materialize", "layout"] + expected
        self.assertEqual(events, expected)
        if legacy:
            pipe.engine.register_pools.assert_called_once_with(pipe.pools)
        else:
            pipe.engine.register_pools.assert_called_once_with(
                pipe.pools, staging_leases=2, staging_blocks_per_pool=[4]
            )
        pipe.executor.shutdown(wait=False)

    def test_layout_is_measured_after_native_materializes_the_cache(self):
        self._initialize()

    def test_legacy_pool_api_uses_checked_full_mirrors(self):
        self._initialize(legacy=True)

    def test_bootstrap_messages_wait_for_pool_initialization(self):
        from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager

        manager = RaidenKVManager.__new__(RaidenKVManager)
        manager.attn_tp_rank = 0
        manager._pipeline_enabled = True
        manager._pipeline_init_lock = threading.Lock()
        manager._pipeline_messages = []
        manager._pipeline = None
        message = [MessageTag.PIPELINE_REQ, b"7", b"tcp://decode:31000", b"{}"]
        manager._dispatch(message)
        self.assertEqual(manager._pipeline_messages, [message])


class TestDraftPoolGroups(unittest.TestCase):
    def pool(self, layers=2):
        return SimpleNamespace(
            start_layer=0, kv_buffer=[Buffer() for _ in range(layers)]
        )

    def test_target_pool_alone_has_no_draft_pool(self):
        target = self.pool()
        self.assertEqual(get_raiden_kv_caches(target).num_draft_full, 0)
        self.assertEqual(
            get_raiden_kv_caches(target, draft_pool=target).num_draft_full, 0
        )

    def test_separate_draft_pool_is_appended_and_counted(self):
        groups = get_raiden_kv_caches(self.pool(), draft_pool=self.pool(layers=1))
        self.assertEqual(groups.num_draft_full, 1)
        self.assertEqual(groups.full_layer_ids, [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
