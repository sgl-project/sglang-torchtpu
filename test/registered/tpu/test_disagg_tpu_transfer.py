"""Unit tests for TPU disaggregation KV resharding transfer math and memory ops.

Focuses on TPU-specific KV buffer slicing, lane calculation, and host staging
buffer manipulation.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

import sglang.srt.disaggregation.tpu.transfer as T
from sglang.srt.disaggregation.tpu.host_staging import PinnedHostPool
from sglang.srt.disaggregation.tpu.transfer import PrefillTransferInfo
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit")


def _fill_random(fake_pool):
    for buf in fake_pool.kv_buffer:
        buf.copy_(torch.randn_like(buf))


class _FakePool:
    def __init__(self, n_blocks, page_size, lane_shape, layers=1, dtype=torch.float32):
        self.kv_buffer = [
            torch.zeros((n_blocks, page_size, *lane_shape), dtype=dtype)
            for _ in range(layers)
        ]


class TestTpuTransferDataPlane(CustomTestCase):
    def _make_info(
        self,
        src_tp: int,
        dst_tp: int,
        engine_rank: int,
        dst_rank: int,
        total_heads: int,
        kvp: int,
        n_layers: int = 2,
    ) -> PrefillTransferInfo:
        return PrefillTransferInfo(
            room=0,
            decode_endpoint="",
            num_pages=0,
            src_tp_rank=engine_rank,
            dst_tp_rank=dst_rank,
            src_attn_tp_size=src_tp,
            dst_attn_tp_size=dst_tp,
            total_num_head=total_heads,
            kv_packing=kvp,
            n_layers=n_layers,
            page_size=1,
        )

    def test_compute_lane_slice_directions(self):
        kvp, total = 2, 4
        info = self._make_info(2, 2, 0, 0, total, kvp)
        self.assertEqual(info.src_lane_idx, 0)
        self.assertEqual(info.n_lane_to_send, 2)
        self.assertEqual(info.dst_head_start, 0)
        self.assertFalse(info.if_skip_sending)
        self.assertTrue(info.if_send_aux_data)

        info = self._make_info(4, 2, 0, 0, total, kvp)
        self.assertEqual(info.src_lane_idx, 0)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertEqual(info.dst_head_start, 0)
        self.assertTrue(info.if_send_aux_data)

        info = self._make_info(4, 2, 1, 0, total, kvp)
        self.assertEqual(info.src_lane_idx, 0)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertEqual(info.dst_head_start, 1)
        self.assertFalse(info.if_send_aux_data)

        info = self._make_info(2, 4, 0, 0, total, kvp)
        self.assertEqual(info.src_lane_idx, 0)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertEqual(info.dst_head_start, 0)
        self.assertTrue(info.if_send_aux_data)

        info = self._make_info(2, 4, 0, 1, total, kvp)
        self.assertEqual(info.src_lane_idx, 1)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertEqual(info.dst_head_start, 0)
        self.assertTrue(info.if_send_aux_data)

    def test_compute_lane_slice_gqa_prefill_replication(self):
        # total_kv_heads=2 < prefill_tp=4: src_replication=2. For FAN_IN,
        # primary ranks 0/2 send while replicated ranks 1/3 are skipped.
        kvp, total = 2, 2
        info = self._make_info(4, 2, 0, 0, total, kvp)
        self.assertEqual(info.src_lane_idx, 0)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertFalse(info.is_src_replica)
        self.assertFalse(info.if_skip_sending)
        self.assertTrue(info.if_send_aux_data)

        info = self._make_info(4, 2, 1, 0, total, kvp)
        self.assertTrue(info.is_src_replica)
        self.assertTrue(info.if_skip_sending)

        info = self._make_info(4, 2, 2, 0, total, kvp)
        self.assertEqual(info.src_lane_idx, 0)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertFalse(info.is_src_replica)
        self.assertFalse(info.if_skip_sending)

        info = self._make_info(4, 2, 3, 0, total, kvp)
        self.assertTrue(info.is_src_replica)
        self.assertTrue(info.if_skip_sending)

    def test_compute_lane_slice_gqa_decode_replication(self):
        kvp, total = 2, 4
        info = self._make_info(2, 8, 0, 0, total, kvp)
        self.assertEqual(info.src_lane_idx, 0)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertEqual(info.dst_head_start, 0)
        self.assertTrue(info.if_send_aux_data)

        info = self._make_info(2, 8, 0, 1, total, kvp)
        self.assertEqual(info.src_lane_idx, 0)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertEqual(info.dst_head_start, 0)
        self.assertTrue(info.if_send_aux_data)

        info = self._make_info(2, 8, 0, 2, total, kvp)
        self.assertEqual(info.src_lane_idx, 1)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertEqual(info.dst_head_start, 0)
        self.assertTrue(info.if_send_aux_data)

        info = self._make_info(2, 8, 0, 3, total, kvp)
        self.assertEqual(info.src_lane_idx, 1)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertEqual(info.dst_head_start, 0)
        self.assertTrue(info.if_send_aux_data)

    def test_compute_lane_slice_gqa_fan_out_prefill_replication(self):
        kvp, total = 2, 2
        info = self._make_info(4, 8, 0, 0, total, kvp)
        self.assertEqual(info.src_lane_idx, 0)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertFalse(info.if_skip_sending)

        info = self._make_info(4, 8, 1, 0, total, kvp)
        self.assertFalse(info.if_skip_sending)

    def test_compute_lane_slice_fp8_odd_head_fan_out(self):
        kvp, total = 4, 4
        info = self._make_info(2, 4, 1, 3, total, kvp)
        self.assertEqual(info.src_heads, 2)
        self.assertEqual(info.dst_heads, 1)
        self.assertEqual(info.n_head_to_send, 1)
        self.assertEqual(info.n_lane_to_send, 1)
        self.assertEqual(info.src_lane_idx, 0)

    def test_lane_slice_gather_scatter_roundtrip(self):
        src = _FakePool(8, 4, (4, 2, 16), layers=2, dtype=torch.bfloat16)
        dst = _FakePool(8, 4, (4, 2, 16), layers=2, dtype=torch.bfloat16)
        _fill_random(src)
        sbufs = T.get_layer_page_buffers(src)
        dbufs = T.get_layer_page_buffers(dst)
        src_pages = np.array([1, 4, 6], dtype=np.int64)
        dst_pages = np.array([7, 0, 3], dtype=np.int64)

        blobs = T.gather_pages_to_host(sbufs, src_pages, lane_slice=(0, 2))
        pool = PinnedHostPool(max_bytes=16 * 1024 * 1024)
        T.scatter_stage_from_host(
            dbufs, dst_pages, blobs, pool=pool, dst_lane_slice=(2, 2)
        )
        for li in range(2):
            for sp, dp in zip(src_pages, dst_pages):
                self.assertTrue(
                    torch.equal(
                        src.kv_buffer[li][sp][:, 0:2], dst.kv_buffer[li][dp][:, 2:4]
                    ),
                    f"layer {li} lane slice {sp}->{dp} mismatch",
                )
            self.assertEqual(
                torch.count_nonzero(dst.kv_buffer[li][:, :, 0:2]).item(), 0
            )

    def test_rejects_non_fused_pool(self):
        class NonFusedPool:
            def __init__(self):
                self.k_buffer = [torch.zeros(4, 2, 8)]
                self.v_buffer = [torch.zeros(4, 2, 8)]

        with self.assertRaises(NotImplementedError):
            T.get_layer_page_buffers(NonFusedPool())

    def test_bytes_roundtrip_helpers(self):
        a = torch.arange(64, dtype=torch.int32).reshape(4, 16)
        b = torch.arange(128, dtype=torch.int64).reshape(4, 32)
        src_args = SimpleNamespace(
            aux_data_ptrs=[a.data_ptr(), b.data_ptr()],
            aux_item_lens=[a[0].nbytes, b[0].nbytes],
        )
        a2 = torch.zeros_like(a)
        b2 = torch.zeros_like(b)
        dst_args = SimpleNamespace(
            aux_data_ptrs=[a2.data_ptr(), b2.data_ptr()],
            aux_item_lens=[a2[0].nbytes, b2[0].nbytes],
        )
        blobs = T.read_aux_row(src_args, aux_index=2)
        T.write_aux_row(dst_args, aux_index=1, blobs=blobs)
        self.assertTrue(torch.equal(a2[1], a[2]))
        self.assertTrue(torch.equal(b2[1], b[2]))
        self.assertTrue(torch.equal(a2[0], torch.zeros(16, dtype=torch.int32)))


class TestHostStagingHelpers(CustomTestCase):
    def test_get_layer_page_buffers(self):
        pool = _FakePool(4, 8, (2, 4, 16), layers=3, dtype=torch.bfloat16)
        bufs = T.get_layer_page_buffers(pool)
        self.assertEqual(len(bufs), 3)
        self.assertEqual(bufs[0].shape, (4, 8, 2, 4, 16))

    def test_gather_pages_to_host_full_and_slice(self):
        pool = _FakePool(4, 8, (4, 2, 16), layers=2, dtype=torch.bfloat16)
        _fill_random(pool)
        bufs = T.get_layer_page_buffers(pool)
        pages = np.array([0, 2], dtype=np.int64)

        full_blobs = T.gather_pages_to_host(bufs, pages)
        self.assertEqual(len(full_blobs), 2)
        expected_bytes = 2 * 8 * 4 * 2 * 16 * 2
        self.assertEqual(len(bytes(full_blobs[0])), expected_bytes)

        sliced_blobs = T.gather_pages_to_host(bufs, pages, lane_slice=(1, 2))
        expected_sliced_bytes = 2 * 8 * 2 * 2 * 16 * 2
        self.assertEqual(len(bytes(sliced_blobs[0])), expected_sliced_bytes)

    def test_scatter_stage_from_host_slice(self):
        src_pool = _FakePool(4, 8, (4, 2, 16), layers=1, dtype=torch.bfloat16)
        dst_pool = _FakePool(4, 8, (4, 2, 16), layers=1, dtype=torch.bfloat16)
        _fill_random(src_pool)

        sbufs = T.get_layer_page_buffers(src_pool)
        dbufs = T.get_layer_page_buffers(dst_pool)
        pages = np.array([1], dtype=np.int64)

        blobs = T.gather_pages_to_host(sbufs, pages, lane_slice=(0, 2))
        pool = PinnedHostPool(max_bytes=1024 * 1024)
        T.scatter_stage_from_host(dbufs, pages, blobs, pool=pool, dst_lane_slice=(2, 2))

        self.assertTrue(
            torch.equal(
                src_pool.kv_buffer[0][1][:, 0:2], dst_pool.kv_buffer[0][1][:, 2:4]
            )
        )
        self.assertEqual(
            torch.count_nonzero(dst_pool.kv_buffer[0][1][:, 0:2]).item(), 0
        )


class TestDecodeReqToTokenPoolTpuHostResident(CustomTestCase):
    def test_tpu_host_resident(self):
        from sglang.srt.disaggregation.decode import DecodeReqToTokenPool

        pool = DecodeReqToTokenPool(
            size=16,
            max_context_len=512,
            device="tpu:0",
            enable_memory_saver=False,
            pre_alloc_size=16,
        )
        self.assertTrue(pool.host_resident)
        self.assertEqual(pool.req_to_token.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
