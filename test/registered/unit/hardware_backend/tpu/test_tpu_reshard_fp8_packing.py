"""End-to-end byte simulation of Raiden push resharding over the fused KV pool.

Regression test for fp8 resharding: with an fp8 KV cache
(kv_packing=4) and prefill attention TP 8 -> decode attention TP 4, the decode
rank's packs mix kv-heads from two prefill ranks, and the reshard schedule used
to copy whole prefill packs at a doubled per-token stride, scattering token
``t`` of prefill rank ``2j+h`` onto token ``2t+h`` of decode rank ``j``.

Each case builds the real per-rank fused KV tensors (``TpuFusedMHATokenToKVPool``
on CPU, the same ``[num_blocks, 64, packs, kv_packing, 128]`` shapes and strides
as the GPT-OSS-120B runtime dump), derives both sides' schedules the way
``conn.py`` / ``sender.py`` / ``receiver.py`` do (each side's geometry from its
OWN tensor), replays the schedule entries on byte arrays laid out like the
native host copy, and checks that every decode rank's (token, head, K/V,
head-dim) element holds the right prefill value.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden import reshard_schedule as rs
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

TOTAL_KV_HEADS = 8  # GPT-OSS-120B
HEAD_DIM = 64  # GPT-OSS-120B; the pool pads it to 128 lanes
PAGE_SIZE = 64
NUM_PAGES = 6
PAD = 0xFD  # a byte no KV value takes (values are < 251)

# Byte layout of one token row inside the buffer Raiden copies:
#   "packed": 32-bit words hold ``kv_packing`` slots of one head-dim lane
#             (row = [pack][lane d][slot]), slot s in bits [8s*itemsize, ...).
#             The native TPU layout: the RPA kernel bitcasts a pack row to
#             uint32 lanes and shifts K/V out of each word, and the raw D2H/H2D
#             host copies keep the device bytes.
#   "contiguous": the torch-stride order (row = [pack][slot][lane d]); kept
#             only to show which results depend on the packing.
PHYSICAL_LAYOUT = "packed"


class _Entry:
    pass


class _EntryList(list):
    def add(self):
        entry = _Entry()
        self.append(entry)
        return entry


class _Schedule:
    def __init__(self):
        self.entries = _EntryList()


class _ScheduleMap(dict):
    def __getitem__(self, key):
        if key not in self:
            self[key] = _Schedule()
        return dict.__getitem__(self, key)


class _Req:
    def __init__(self):
        self.shard_push_schedules = _ScheduleMap()


class _Pb2:
    MEMORY_TYPE_DRAM = 1
    MEMORY_TYPE_HBM = 2

    @staticmethod
    def StartTransferRequest():
        return _Req()


def _fused_layer(dtype, heads):
    pool = TpuFusedMHATokenToKVPool(
        size=PAGE_SIZE * (NUM_PAGES - 1),
        page_size=PAGE_SIZE,
        dtype=dtype,
        head_num=heads,
        head_dim=HEAD_DIM,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    return pool.get_fused_kv_buffer(0)


def _manager(tensor, local_tp):
    mgr = object.__new__(RaidenKVManager)
    mgr.attn_tp_rank = 0
    mgr.attn_tp_size = local_tp
    mgr._kv_caches = [tensor]
    mgr._swa_kv_caches = []
    mgr.kv_args = SimpleNamespace(total_kv_head_num=TOTAL_KV_HEADS, page_size=PAGE_SIZE)
    return mgr


def _heads_of_rank(tp, rank):
    """Global kv-heads a rank holds, in local slot order (sglang head sharding:
    heads split evenly, replicated when the model has fewer heads than ranks)."""
    per_rank = max(1, TOTAL_KV_HEADS // tp)
    replicas = max(1, tp // TOTAL_KV_HEADS)
    first = (rank % tp) // replicas * per_rank
    return list(range(first, first + per_rank))


def _byte_offsets(tensor, layout):
    """Byte offset within one page of every (token, slot, lane, byte)."""
    _, page, packs, packing, lanes = tensor.shape
    itemsize = tensor.element_size()
    token_width = tensor.stride(1) * itemsize
    assert packing * itemsize == 4
    t = np.arange(page)[:, None, None, None]
    slot = np.arange(packs * packing)[None, :, None, None]
    d = np.arange(lanes)[None, None, :, None]
    b = np.arange(itemsize)[None, None, None, :]
    pack, lane_slot = slot // packing, slot % packing
    if layout == "packed":
        inner = pack * (lanes * 4) + d * 4 + lane_slot * itemsize + b
    elif layout == "contiguous":
        inner = (pack * packing + lane_slot) * (lanes * itemsize) + d * itemsize + b
    else:
        raise AssertionError(layout)
    return t * token_width + inner  # [page, slots, lanes, itemsize]


def _values(block, heads, kv_slots, itemsize):
    """Expected byte of (token, slot, lane, byte) for global ``heads`` in page
    ``block``; slot 2i is head i's K, 2i+1 its V, the rest padding."""
    t = np.arange(PAGE_SIZE)[:, None, None, None]
    d = np.arange(128)[None, None, :, None]
    b = np.arange(itemsize)[None, None, None, :]
    out = np.full((PAGE_SIZE, kv_slots, 128, itemsize), PAD, dtype=np.int64)
    for local, head in enumerate(heads):
        for kv in (0, 1):
            value = (block * 17 + t * 29 + head * 37 + kv * 43 + d * 61 + b * 73) % 251
            out[:, 2 * local + kv] = value[:, 0]
    return out


def _page_image(tensor, block, heads, layout):
    page_bytes = tensor.stride(0) * tensor.element_size()
    image = np.full(page_bytes, PAD, dtype=np.uint8)
    offsets = _byte_offsets(tensor, layout)
    vals = _values(block, heads, offsets.shape[1], tensor.element_size())
    image[offsets.reshape(-1)] = vals.reshape(-1)
    return image


def _apply(entry, source_pages, dest_pages):
    src = source_pages[entry.src_block_id]
    dst = dest_pages[entry.dst_block_id]
    rep = np.arange(entry.count, dtype=np.int64)[:, None]
    byte = np.arange(entry.size_bytes, dtype=np.int64)[None, :]
    src_idx = entry.src_offset_bytes + rep * entry.src_stride_bytes + byte
    dst_idx = entry.dst_offset_bytes + rep * entry.dst_stride_bytes + byte
    assert src_idx.min() >= 0 and src_idx.max() < src.size, "src out of block"
    assert dst_idx.min() >= 0 and dst_idx.max() < dst.size, "dst out of block"
    dst[dst_idx.reshape(-1)] = src[src_idx.reshape(-1)]


class TestTpuReshardFp8Packing(unittest.TestCase):
    """Decode rank j must hold, at every token of every page, exactly the K/V
    of its own kv-heads as written by the prefill ranks."""

    # Paired pages: prefill page ids differ from decode page ids; the last page
    # is partial (35 live tokens) so tokens 31/32 cross the half-page mark and
    # 63 is the page's last slot on the full pages.
    SRC_BLOCKS = [1, 2, 4]
    DST_BLOCKS = [3, 5, 2]
    LIVE = [PAGE_SIZE, PAGE_SIZE, 35]

    def _transfer(self, dtype, src_tp, dst_tp, layout=PHYSICAL_LAYOUT):
        src_heads = max(1, TOTAL_KV_HEADS // src_tp)
        dst_heads = max(1, TOTAL_KV_HEADS // dst_tp)
        prefill_t = _fused_layer(dtype, src_heads)
        decode_t = _fused_layer(dtype, dst_heads)
        sender_geom = _manager(prefill_t, src_tp)._make_geometry(src_tp, dst_tp)
        receiver_geom = _manager(decode_t, dst_tp)._make_geometry(src_tp, dst_tp)
        ratio = rs.reshard_ratio(src_tp, dst_tp)
        results = {}
        for decode_rank in range(dst_tp):
            if src_tp >= dst_tp:
                senders = rs.source_ranks_for(decode_rank, ratio)
            else:
                senders = [decode_rank * src_tp // dst_tp]
            receiver = rs.build_receiver_request(
                _Pb2,
                uuid=7,
                req_id="7",
                peer="decode:1",
                decode_rank=decode_rank,
                geom=receiver_geom,
                sources=[(rank, self.SRC_BLOCKS) for rank in senders],
                dst_block_ids=self.DST_BLOCKS,
                live_tokens=self.LIVE,
            )
            dest_pages = {
                b: np.full(
                    decode_t.stride(0) * decode_t.element_size(), PAD, dtype=np.uint8
                )
                for b in range(NUM_PAGES)
            }
            for rank in senders:
                sender = rs.build_sender_request(
                    _Pb2,
                    uuid=7,
                    req_id="7",
                    peer="decode:1",
                    prefill_rank=rank,
                    decode_rank=decode_rank,
                    geom=sender_geom,
                    src_block_ids=self.SRC_BLOCKS,
                    dst_block_ids=self.DST_BLOCKS,
                    live_tokens=self.LIVE,
                )
                sent = [vars(e) for e in sender.shard_push_schedules[0].entries]
                recv = [vars(e) for e in receiver.shard_push_schedules[rank].entries]
                for e in sent + recv:
                    e.pop("dst_peer", None)
                self.assertEqual(sent, recv, f"sender/receiver disagree, rank {rank}")
                source_pages = {
                    b: _page_image(prefill_t, b, _heads_of_rank(src_tp, rank), layout)
                    for b in range(NUM_PAGES)
                }
                for entry in receiver.shard_push_schedules[rank].entries:
                    _apply(entry, source_pages, dest_pages)
            results[decode_rank] = (dest_pages, decode_t)
        return results

    def _check(self, dtype, src_tp, dst_tp, layout=PHYSICAL_LAYOUT):
        results = self._transfer(dtype, src_tp, dst_tp, layout)
        for decode_rank, (dest_pages, decode_t) in results.items():
            heads = _heads_of_rank(dst_tp, decode_rank)
            offsets = _byte_offsets(decode_t, layout)
            itemsize = decode_t.element_size()
            for src_b, dst_b, live in zip(self.SRC_BLOCKS, self.DST_BLOCKS, self.LIVE):
                got = dest_pages[dst_b][offsets]  # [page, slots, lanes, itemsize]
                want = _values(src_b, heads, offsets.shape[1], itemsize)
                for local, head in enumerate(heads):
                    for kv in (0, 1):
                        slot = 2 * local + kv
                        bad = np.argwhere(got[:live, slot] != want[:live, slot])
                        if bad.size:
                            token = int(bad[0][0])
                            self.fail(
                                f"{dtype} {src_tp}->{dst_tp} decode rank "
                                f"{decode_rank} page {dst_b} token {token} "
                                f"head {head} {'KV'[kv]}: got "
                                f"{got[token, slot, :4, 0].tolist()} want "
                                f"{want[token, slot, :4, 0].tolist()} "
                                f"({len(bad)} bad bytes)"
                            )
                # Spot-check the page-boundary tokens explicitly.
                for token in (0, 31, 32, 63):
                    if token < live:
                        np.testing.assert_array_equal(
                            got[token, : 2 * len(heads)], want[token, : 2 * len(heads)]
                        )

    def test_fp8_fan_in_8_to_4(self):
        """GPT-OSS-120B fp8, prefill TP8 -> decode TP8 DP2 (attention TP4).
        One fp8 head per prefill rank cannot be placed into a decode pack it
        shares with another rank's head by any strided copy: refuse it."""
        with self.assertRaisesRegex(
            ValueError, "Partial FP8 head packs are unsupported"
        ):
            self._check(torch.float8_e4m3fn, 8, 4)

    def test_fp8_fan_in_8_to_2(self):
        with self.assertRaisesRegex(
            ValueError, "Partial FP8 head packs are unsupported"
        ):
            self._check(torch.float8_e4m3fn, 8, 2)

    def test_partial_pack_message_tp8_to_attn_tp4(self):
        decode_t = _fused_layer(torch.float8_e4m3fn, 2)
        geometry = _manager(decode_t, 4)._make_geometry(8, 4)
        with self.assertRaisesRegex(ValueError, "splits a 4-slot pack"):
            geometry.check_decode_rank(0)

    def test_fp8_equal_8_to_8(self):
        self._check(torch.float8_e4m3fn, 8, 8)

    def test_fp8_replicated_16_to_8(self):
        """Multihost TP16 prefill -> attention-TP8 decode (heads replicated)."""
        self._check(torch.float8_e4m3fn, 16, 8)

    def test_bf16_all_shapes(self):
        for src_tp, dst_tp in ((8, 4), (8, 8), (8, 2), (16, 8)):
            with self.subTest(src_tp=src_tp, dst_tp=dst_tp):
                self._check(torch.bfloat16, src_tp, dst_tp)

    def test_real_runtime_shapes(self):
        """The pools match the diag3 runtime dump: [*, 64, 1, 4, 128] fp8,
        stride [32768, 512, 512, 128, 1] on both prefill (1 head) and decode
        (2 heads)."""
        for heads in (1, 2):
            t = _fused_layer(torch.float8_e4m3fn, heads)
            self.assertEqual(tuple(t.shape[1:]), (64, 1, 4, 128))
            self.assertEqual(t.stride(), (32768, 512, 512, 128, 1))
            self.assertEqual(t.dtype, torch.float8_e4m3fn)

    def test_geometry_requires_layout(self):
        """kv_packing / element_size have no bf16 defaults any more."""
        with self.assertRaises(TypeError):
            rs.ReshardGeometry(8, 4, 512, 8)

    def test_token_width_checked_against_strides(self):
        """A geometry whose computed per-token width disagrees with the local
        tensor's real row is refused -- here the old bf16-style sizing, a whole fp8
        pack (512 B) taken as one head."""
        decode_t = _fused_layer(torch.float8_e4m3fn, 2)
        row = decode_t.stride(1) * decode_t.element_size()
        geometry = rs.ReshardGeometry(8, 4, 256, 8, kv_packing=4, element_size=1)
        geometry.check_token_bytes(4, row, "decode")
        wrong = rs.ReshardGeometry(8, 4, 512, 8, kv_packing=4, element_size=1)
        with self.assertRaisesRegex(ValueError, "per-token KV row"):
            wrong.check_token_bytes(4, row, "decode")

    def test_bf16_entries_unchanged(self):
        """bf16 8->4 entries are byte-identical to the old geometry's."""
        decode_t = _fused_layer(torch.bfloat16, 2)
        geometry = _manager(decode_t, 4)._make_geometry(8, 4)
        self.assertEqual(geometry.entry_bytes(0, 0), (0, 512, 512, 0, 1024))
        self.assertEqual(geometry.entry_bytes(1, 0), (0, 512, 512, 512, 1024))

    def test_fp8_replicated_entry_unchanged(self):
        """TP16 -> attention TP8 fp8 (1 head -> 1 head) keeps the whole padded
        pack move it had before."""
        decode_t = _fused_layer(torch.float8_e4m3fn, 1)
        geometry = _manager(decode_t, 8)._make_geometry(16, 8)
        for prefill_rank in (0, 1, 14, 15):
            self.assertEqual(
                geometry.entry_bytes(prefill_rank, prefill_rank // 2),
                (0, 512, 512, 0, 512),
            )


class TestTpuFinerDecodeRefusal(unittest.TestCase):
    """A decode at a larger attention TP than its prefill pulls one prefill
    rank's whole pages (no head slicing), so it is refused unless both hold
    the same kv-heads."""

    def test_finer_decode_refusal(self):
        decode_t = _fused_layer(torch.float8_e4m3fn, 1)
        for total, prefill_tp, decode_tp, refused in (
            (8, 4, 8, True),  # 2 heads/prefill rank, 1/decode rank
            (8, 2, 8, True),
            (4, 4, 8, False),  # replicated: both hold 1 head
            (8, 8, 8, False),  # equal TP: pull
            (8, 16, 8, False),  # fan-in: reshard
        ):
            with self.subTest(total=total, prefill_tp=prefill_tp, decode_tp=decode_tp):
                mgr = _manager(decode_t, decode_tp)
                mgr.kv_args.total_kv_head_num = total
                mgr.is_mla_backend = False
                reason = mgr.finer_decode_refusal(prefill_tp)
                self.assertEqual(reason is not None, refused, reason)
                if refused:
                    self.assertIn("Cannot serve decode attention TP", reason)
        mgr = _manager(decode_t, 8)
        mgr.is_mla_backend = True
        self.assertIsNone(mgr.finer_decode_refusal(4))

    def test_receiver_fails_before_registering(self):
        from unittest import mock

        from sglang.srt.disaggregation.base.conn import KVPoll
        from sglang_tpu.disaggregation.raiden.receiver import RaidenKVReceiver

        mgr = _manager(_fused_layer(torch.float8_e4m3fn, 1), 8)
        mgr.is_mla_backend = False
        mgr.record_failure = mock.Mock()
        mgr.update_status = mock.Mock()
        mgr.register_decode_request = mock.Mock()
        receiver = object.__new__(RaidenKVReceiver)
        receiver.kv_mgr = mgr
        receiver.bootstrap_room = 11
        receiver.conclude_state = None
        receiver.prefill_info = SimpleNamespace(attn_tp_size=4)
        receiver.abort_notified = False
        receiver.bootstrap_infos = [{"rank_ip": "10.0.0.1", "rank_port": 1}]
        receiver._send_abort_notification = mock.Mock()
        receiver.send_metadata(np.array([1, 2], dtype=np.int32))
        mgr.register_decode_request.assert_not_called()
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)
        mgr.update_status.assert_called_once_with(11, KVPoll.Failed)
        self.assertIn("attention TP 8", mgr.record_failure.call_args[0][1])
        # The prefill is told right away instead of waiting for its timeout.
        receiver._send_abort_notification.assert_called_once_with()
        self.assertTrue(receiver.abort_notified)


if __name__ == "__main__":
    unittest.main()
