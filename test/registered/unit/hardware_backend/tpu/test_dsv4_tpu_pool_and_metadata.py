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
"""CPU-runnable unit tests for DeepSeek-V4 TPU memory pools and attention metadata.

``DSV4TPUTokenToKVPool`` and ``DSV4TPUAttnBackend._compute_metadata`` operate on
plain PyTorch tensors and CPU metadata arrays, so these tests run on CPU
(``device="cpu"``) without requiring TPU hardware. They guard:

1. **4D ``torch.uint8`` buffer shapes and ``_base is None`` donation invariant**:
   every SWA, CSA (NoPE + RoPE), HCA, Indexer K, and Compressor State buffer
   must be a genuine ``torch.uint8`` base tensor so ``torch_tpu`` can donate it
   in-place across Pallas kernel calls without defensive copies.
2. **6-slot ``state_block_table`` construction (`[bs, 6]` `int32`)**:
   ``(ring_base, b_prev, old_shift, b_tail_start, new_shift, scratch_base)``
   across decode, short extend (`<= window_blocks`), long initial prefill, and
   long continuation extend (`> window_blocks`), plus slot reset hooks.
3. **Bucket-padded ``DSV4TPUForwardMetadata`` construction**:
   when ``compiled_forward_shim`` pads ``seq_lens`` to bucket ``bs`` while
   keeping ``seq_lens_cpu`` at ``_tpu_real_b < bs``, ``_compute_metadata`` must
   still produce ``(bs,)``-shaped ``kv_lens`` and ``c128_kv_lens`` and
   ``(bs + 1,)``-shaped ``cu_seqlens_q`` so ``torch.compile(dynamic=False)``
   never sees shape drift across active request counts.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.dsv4.dsv4_attention_backend import (
    DSV4TPUAttnBackend,
)
from sglang_tpu.dsv4.dsv4_memory_pool import (
    DSV4TPUTokenToKVPool,
    TPUCompressStatePool,
)
from sglang_tpu.patches import _PALLAS_TPU_BACKENDS


def _make_dsv4_tpu_pool(
    testcase: unittest.TestCase, num_req_slots: int = 5
) -> DSV4TPUTokenToKVPool:
    from sglang.srt.runtime_context import get_context

    override = get_context().override_server_args(
        dsv4_attn_backend="tpu_dsv4", page_size=256
    )
    override.install()
    testcase.addCleanup(override.restore)

    compression_ratios = [0, 0, 128, 4, 128, 4]
    pool = DSV4TPUTokenToKVPool(
        max_num_reqs=num_req_slots - 1,
        num_req_slots=num_req_slots,
        swa_size=512,
        c4_size=256,
        c128_size=8,
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
        full_size=1024,
    )
    # Set up a non-identity full -> SWA page mapping:
    # Full page 0 -> SWA page 0, full page 1 -> SWA page 2, full page 2 -> SWA page 1
    mapping = torch.zeros(1024 + 256, dtype=torch.int64, device="cpu")
    mapping[256:512] = torch.arange(512, 768, dtype=torch.int64)
    mapping[512:768] = torch.arange(256, 512, dtype=torch.int64)
    pool.full_to_swa_index_mapping = mapping
    return pool


class TestDSV4TPUMemoryPool(CustomTestCase):
    def test_buffer_shapes_and_base_ownership(self):
        pool = _make_dsv4_tpu_pool(self, num_req_slots=5)

        # SWA: (512 + 256 + 1) // 256 = 3 pages, 512 rows/page, (4, 128) uint8
        for layer_id in range(6):
            swa_buf = pool.get_swa_buffer(layer_id)
            self.assertEqual(swa_buf.dtype, torch.uint8)
            self.assertEqual(tuple(swa_buf.shape), (3, 512, 4, 128))
            self.assertIsNone(swa_buf._base)

        # CSA (layers 3, 5): c4_page_size = 64, (256 + 64 + 1) // 64 = 5 pages
        for layer_id in (3, 5):
            nope_buf, rope_buf = pool.get_csa_kv_buffers(layer_id)
            self.assertEqual(nope_buf.dtype, torch.uint8)
            self.assertEqual(rope_buf.dtype, torch.uint8)
            self.assertEqual(tuple(nope_buf.shape), (5, 64, 4, 128))
            self.assertEqual(tuple(rope_buf.shape), (5, 16, 4, 128))
            self.assertIsNone(nope_buf._base)
            self.assertIsNone(rope_buf._base)

            idx_buf = pool.get_indexer_k_buffer(layer_id)
            self.assertEqual(idx_buf.dtype, torch.uint8)
            self.assertEqual(tuple(idx_buf.shape), (5, 16, 4, 256))
            self.assertIsNone(idx_buf._base)

            c4_state = pool.get_state_buffer(layer_id, from_indexer=False)
            c4_idx_state = pool.get_state_buffer(layer_id, from_indexer=True)
            # Uniform across C4, C4-Indexer, and C128:
            # 1 dummy page + 5 * 32 slot pages + 2048 scratch pages = 2209 pages
            self.assertEqual(tuple(c4_state.shape), (2209, 64, 4, 128))
            self.assertEqual(tuple(c4_idx_state.shape), (2209, 8, 4, 256))
            self.assertIsNone(c4_state._base)
            self.assertIsNone(c4_idx_state._base)

        # HCA (layers 2, 4): c128_page_size = 2, (8 + 2 + 1) // 2 = 5 pages
        for layer_id in (2, 4):
            hca_buf = pool.get_hca_buffer(layer_id)
            self.assertEqual(hca_buf.dtype, torch.uint8)
            self.assertEqual(tuple(hca_buf.shape), (5, 4, 4, 128))
            self.assertIsNone(hca_buf._base)

            c128_state = pool.get_state_buffer(layer_id, from_indexer=False)
            self.assertEqual(tuple(c128_state.shape), (2209, 64, 4, 128))
            self.assertIsNone(c128_state._base)

    def test_state_block_table_decode_and_extend_transitions(self):
        pool = _make_dsv4_tpu_pool(self, num_req_slots=5)
        c4_proto = pool._state_pool_proto[4]
        c128_proto = pool._state_pool_proto[128]
        self.assertIsInstance(c4_proto, TPUCompressStatePool)
        self.assertIsInstance(c128_proto, TPUCompressStatePool)
        self.assertEqual(c4_proto.ring_pages_per_req, 8)
        self.assertEqual(c128_proto.ring_pages_per_req, 32)
        slot_stride = c4_proto.slot_stride_pages
        self.assertEqual(slot_stride, 32)
        self.assertEqual(c128_proto.slot_stride_pages, 32)
        scratch_start = c4_proto.scratch_start_page  # 1 + 5 * 32 = 161
        self.assertEqual(scratch_start, 161)

        # 1. Short initial prefill (12 tokens -> b_prev = -1, b_end = 2, 3 blocks <= 4):
        # Fits inside persistent ring without scratch pages.
        table_short = pool.build_state_block_table(
            ratio=4,
            req_pool_indices_cpu=[1],
            seq_lens_cpu=[12],
            extend_seq_lens_cpu=[12],
            is_decode=False,
            bs=2,
            non_empty_bs=1,
            device="cpu",
        )
        self.assertEqual(tuple(table_short.shape), (2, 6))
        self.assertEqual(
            table_short[0].tolist(),
            [1 + 1 * slot_stride, 0, 0, 0, 0, 0],
        )
        self.assertEqual(table_short[1].tolist(), [0, 0, 0, 0, 0, 0])

        # 2. Long initial prefill on req 2 (64 tokens -> b_prev = -1, b_end = 15,
        # window_blocks = 4 -> b_tail_start = 12, num_scratch = 12):
        table_long = pool.build_state_block_table(
            ratio=4,
            req_pool_indices_cpu=[2],
            seq_lens_cpu=[64],
            extend_seq_lens_cpu=[64],
            is_decode=False,
            bs=1,
            non_empty_bs=1,
            device="cpu",
        )
        self.assertEqual(
            table_long[0].tolist(),
            [1 + 2 * slot_stride, -1, 0, 12, 0, scratch_start],
        )

        # 3. Long continuation chunk on req 2 (start_pos = 64, q_len = 32 -> seq_len = 96,
        # b_prev = 15, b_end = 23, b_end - b_prev = 8 > 4 -> b_tail_start = 20,
        # new_shift = (0 + 4 - 8) % 8 = 4 — still modulo 8 for C4!):
        table_cont = pool.build_state_block_table(
            ratio=4,
            req_pool_indices_cpu=[2],
            seq_lens_cpu=[96],
            extend_seq_lens_cpu=[32],
            is_decode=False,
            bs=1,
            non_empty_bs=1,
            device="cpu",
        )
        self.assertEqual(
            table_cont[0].tolist(),
            [1 + 2 * slot_stride, 15, 0, 20, 4, scratch_start],
        )
        self.assertEqual(int(pool._req_state_shifts[4][2].item()), 4)

        # 4. Subsequent decode on req 2 uses updated shift = 4:
        table_decode = pool.build_state_block_table(
            ratio=4,
            req_pool_indices_cpu=[2],
            seq_lens_cpu=[97],
            extend_seq_lens_cpu=None,
            is_decode=True,
            bs=2,
            non_empty_bs=1,
            device="cpu",
        )
        self.assertEqual(
            table_decode[0].tolist(),
            [1 + 2 * slot_stride, 0, 4, 0, 4, 0],
        )

        # 5. Clearing req 2 resets its ring shift to 0:
        pool.clear_c4_req_states([2])
        self.assertEqual(int(pool._req_state_shifts[4][2].item()), 0)
        self.assertEqual(int(pool._req_state_shifts[128][2].item()), 0)

    def test_pd_disagg_buf_infos_transfer_indices_and_pjrt_pointer_stability(self):
        from sglang.srt.disaggregation.base.conn import KVArgs, StateType
        from sglang.srt.disaggregation.utils import (
            get_kv_transfer_buf_infos,
            setup_state_kv_args,
        )

        pool = _make_dsv4_tpu_pool(self, num_req_slots=5)

        kv_ptrs, kv_lens, kv_items = get_kv_transfer_buf_infos(pool)
        # 2 C4 NoPE + 2 C4 RoPE + 2 C4 Indexer + 2 C128 HCA = 8 full-page tensors
        self.assertEqual(len(kv_ptrs), 8)
        self.assertEqual(
            kv_items,
            [64 * 4 * 128] * 2
            + [16 * 4 * 128] * 2
            + [16 * 4 * 256] * 2
            + [4 * 4 * 128] * 2,
        )

        kv_args = KVArgs()
        setup_state_kv_args(kv_args, pool, total_kv_layers=6)
        self.assertEqual(
            kv_args.state_types,
            [StateType.SWA, StateType.DSV4_REQUEST_STATE],
        )
        # 6 SWA layers on StateType.SWA + (4 attn state + 2 indexer state) on DSV4_REQUEST_STATE
        self.assertEqual(len(kv_args.state_data_ptrs[0]), 6)
        self.assertEqual(len(kv_args.state_data_ptrs[1]), 6)
        total_pd_bytes = (
            sum(kv_lens)
            + sum(kv_args.state_data_lens[0])
            + sum(kv_args.state_data_lens[1])
        )
        self.assertEqual(total_pd_bytes, pool.get_kv_size_bytes()[0])

        # Transfer indices + (shift_c4, shift_c128) round-trip from prefill slot 3 -> decode slot 1
        src_rows = pool.request_state_transfer_indices(3, seq_len=517)
        dst_rows = pool.request_state_transfer_indices(1, seq_len=517)
        self.assertEqual(src_rows.tolist(), list(range(1 + 3 * 32, 1 + 4 * 32)))
        self.assertEqual(dst_rows.tolist(), list(range(1 + 1 * 32, 1 + 2 * 32)))

        pool._req_state_shifts[4][3] = 5
        pool._req_state_shifts[128][3] = 19
        meta_blob = pool.pack_req_state_meta(src_rows)

        # Simulate decode.py pre-transfer clear_request_scoped_state followed by metadata unpack:
        # Must NOT touch underlying HBM state_buffer contents or reallocate pointers.
        sentinel_buf = pool.get_state_buffer(2, from_indexer=False)
        ptr_before = sentinel_buf.data_ptr()
        sentinel_buf[dst_rows[0], 0, 0, 0] = 0xAB
        pool._req_state_shifts[4][1] = 7
        pool._req_state_shifts[128][1] = 29

        pool.clear_request_scoped_state(1)
        self.assertEqual(sentinel_buf.data_ptr(), ptr_before)
        self.assertEqual(int(sentinel_buf[dst_rows[0], 0, 0, 0].item()), 0xAB)
        self.assertEqual(int(pool._req_state_shifts[4][1].item()), 0)
        self.assertEqual(int(pool._req_state_shifts[128][1].item()), 0)

        pool.unpack_req_state_meta(dst_rows, meta_blob)
        self.assertEqual(int(pool._req_state_shifts[4][1].item()), 5)
        self.assertEqual(int(pool._req_state_shifts[128][1].item()), 19)

    def test_deepseek_v4_pro_61_layers_with_trailing_mtp_compress_ratios(self):
        """Verify DeepSeek-V4-Pro's 61-layer config (`[128, 128, 4, ..., 4, 0, 0, 0]`).

        In DeepSeek-V4-Pro:
        - `num_hidden_layers = 61`, while `compress_ratios` has 64 entries
          (`[128] + [128, 4] * 30 + [0, 0, 0]`, where the 3 trailing `0`s are
          for MTP / DSpark draft layers 61..63).
        - Layers 0 and 1 are both HCA (`ratio=128`), and there are zero pure-SWA
          (`ratio=0`) layers in the 61 target layers (`31` HCA + `30` CSA).
        - `sliding_window = 128`.
        """
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args(
            dsv4_attn_backend="tpu_dsv4", page_size=256
        )
        override.install()
        self.addCleanup(override.restore)

        pro_compress_ratios = [128] + [128, 4] * 30 + [0, 0, 0]
        self.assertEqual(len(pro_compress_ratios), 64)

        pool = DSV4TPUTokenToKVPool(
            max_num_reqs=2,
            num_req_slots=3,
            swa_size=256,
            c4_size=256,
            c128_size=8,
            c4_state_pool_size=32,
            c128_state_pool_size=256,
            page_size=256,
            swa_page_size=256,
            sliding_window=128,
            dtype=torch.bfloat16,
            c4_state_dtype=torch.float32,
            c128_state_dtype=torch.float32,
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            indexer_head_dim=128,
            layer_num=61,
            device="cpu",
            enable_memory_saver=False,
            compression_ratios=pro_compress_ratios,
            full_size=1024,
        )

        # Must allocate exactly 61 SWA layers (excluding the 3 trailing MTP 0s),
        # 30 CSA layers, and 31 HCA layers.
        self.assertEqual(len(pool.swa_kv_pool.k_buffer), 61)
        self.assertEqual(len(pool.c4_kv_pool.nope_buffer), 30)
        self.assertEqual(len(pool.c4_kv_pool.rope_buffer), 30)
        self.assertEqual(len(pool.c4_indexer_kv_pool.index_k_with_scale_buffer), 30)
        self.assertEqual(len(pool.c128_kv_pool.k_buffer), 31)

        # Layer 0 and Layer 1 are both HCA (128), Layer 2 is the first CSA (4),
        # and Layer 60 is the last CSA (4).
        self.assertEqual(pool.layer_mapping[0].compress_ratio, 128)
        self.assertEqual(pool.layer_mapping[0].compress_layer_id, 0)
        self.assertEqual(pool.layer_mapping[1].compress_ratio, 128)
        self.assertEqual(pool.layer_mapping[1].compress_layer_id, 1)
        self.assertEqual(pool.layer_mapping[2].compress_ratio, 4)
        self.assertEqual(pool.layer_mapping[2].compress_layer_id, 0)
        self.assertEqual(pool.layer_mapping[59].compress_ratio, 128)
        self.assertEqual(pool.layer_mapping[59].compress_layer_id, 30)
        self.assertEqual(pool.layer_mapping[60].compress_ratio, 4)
        self.assertEqual(pool.layer_mapping[60].compress_layer_id, 29)
        for mtp_idx in (61, 62, 63):
            self.assertIsNone(pool.layer_mapping[mtp_idx])


class TestDSV4TPUAttentionMetadata(CustomTestCase):
    def _make_backend_stub(self, pool: DSV4TPUTokenToKVPool) -> DSV4TPUAttnBackend:
        backend = object.__new__(DSV4TPUAttnBackend)
        backend.device = torch.device("cpu")
        backend.page_size = 256
        backend.max_context_len = 1024
        backend.token_to_kv_pool = pool
        req_to_token = torch.zeros((5, 1024), dtype=torch.int32, device="cpu")
        # Req 1: full page 1 (256..512) then full page 2 (512..768)
        req_to_token[1, :256] = 256
        req_to_token[1, 256:512] = 512
        # Req 2: full page 2 (512..768) then full page 1 (256..512)
        req_to_token[2, :256] = 512
        req_to_token[2, 256:512] = 256
        backend.req_to_token = req_to_token
        return backend

    def test_extend_metadata_and_swa_table_translation(self):
        pool = _make_dsv4_tpu_pool(self, num_req_slots=5)
        backend = self._make_backend_stub(pool)

        fb = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            batch_size=2,
            seq_lens=torch.tensor([300, 512], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([300, 512], dtype=torch.int32),
            extend_seq_lens=torch.tensor([300, 256], dtype=torch.int32),
            extend_seq_lens_cpu=[300, 256],
            req_pool_indices=torch.tensor([1, 2], dtype=torch.int32),
            req_pool_indices_cpu=torch.tensor([1, 2], dtype=torch.int32),
            positions=torch.arange(556, dtype=torch.int64),
        )
        meta = backend._compute_metadata(fb)
        self.assertEqual(meta.distribution.tolist(), [0, 0, 2])
        self.assertEqual(meta.cu_seqlens_q.tolist(), [0, 300, 556])
        self.assertEqual(meta.kv_lens.tolist(), [300, 512])
        # c128_kv_lens = kv_lens // 128 -> [300 // 128, 512 // 128] = [2, 4]
        self.assertEqual(meta.c128_kv_lens.tolist(), [2, 4])
        # C4/C128 block tables (max_pages_per_req = 4):
        # Req 1 pages: [1, 2, 0, 0]; Req 2 pages: [2, 1, 0, 0]
        self.assertEqual(
            meta.c4_block_tables_flat.view(2, 4).tolist(),
            [[1, 2, 0, 0], [2, 1, 0, 0]],
        )
        self.assertEqual(
            meta.c128_block_tables_flat.view(2, 4).tolist(),
            [[1, 2, 0, 0], [2, 1, 0, 0]],
        )
        # SWA block table translated via full_to_swa_index_mapping (1 -> 2, 2 -> 1):
        self.assertEqual(
            meta.swa_block_tables_flat.view(2, 4).tolist(),
            [[2, 1, 0, 0], [1, 2, 0, 0]],
        )

    def test_bucket_padded_decode_metadata_preserves_static_shapes(self):
        """Regression test for compiled_forward_shim bucket padding (`_tpu_real_b < bs`).

        ``_pad_forward_batch_inputs`` pads ``seq_lens`` and ``req_pool_indices``
        to ``bs`` (e.g., 4) while leaving ``seq_lens_cpu`` at ``_tpu_real_b``
        (e.g., 2). ``_compute_metadata`` must still emit ``(bs,)``-shaped
        ``kv_lens`` and ``c128_kv_lens`` so ``torch.compile(dynamic=False)``
        never recompiles or fails a shape guard.
        """
        pool = _make_dsv4_tpu_pool(self, num_req_slots=5)
        backend = self._make_backend_stub(pool)

        fb = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=4,
            _tpu_real_b=2,
            seq_lens=torch.tensor([260, 128, 0, 0], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([260, 128], dtype=torch.int32),
            extend_seq_lens=None,
            extend_seq_lens_cpu=None,
            req_pool_indices=torch.tensor([1, 2, 0, 0], dtype=torch.int32),
            req_pool_indices_cpu=torch.tensor([1, 2, 0, 0], dtype=torch.int32),
            positions=torch.tensor([259, 127, 0, 0], dtype=torch.int64),
        )
        meta = backend._compute_metadata(fb)
        self.assertEqual(meta.distribution.tolist(), [2, 2, 2])
        self.assertEqual(tuple(meta.cu_seqlens_q.shape), (5,))
        self.assertEqual(meta.cu_seqlens_q.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(tuple(meta.kv_lens.shape), (4,))
        self.assertEqual(meta.kv_lens.tolist(), [260, 128, 0, 0])
        self.assertEqual(tuple(meta.c128_kv_lens.shape), (4,))
        self.assertEqual(meta.c128_kv_lens.tolist(), [2, 1, 0, 0])
        self.assertEqual(tuple(meta.c4_state_block_tables_flat.shape), (24,))
        self.assertEqual(tuple(meta.c128_state_block_tables_flat.shape), (24,))


class TestDSV4TPUPatches(CustomTestCase):
    def test_dsv4_in_pallas_tpu_backends(self):
        self.assertIn("dsv4", _PALLAS_TPU_BACKENDS)
        self.assertIn("tpu_dsv4", _PALLAS_TPU_BACKENDS)

    def test_register_dsv4_attention_backend(self):
        from sglang.srt.layers.attention.attention_registry import (
            ATTENTION_BACKENDS,
            create_dsv4_backend,
        )
        from sglang_tpu.tpu_platform import _register_attention_backends

        original = ATTENTION_BACKENDS.get("dsv4")
        try:
            _register_attention_backends()
            self.assertIn("dsv4", ATTENTION_BACKENDS)
            self.assertIsNot(ATTENTION_BACKENDS["dsv4"], create_dsv4_backend)
            self.assertEqual(
                ATTENTION_BACKENDS["dsv4"].__name__, "create_tpu_dsv4_backend"
            )
        finally:
            if original is not None:
                ATTENTION_BACKENDS["dsv4"] = original

    def test_prepare_mqa_qkv_respects_q_head_norm(self):
        from unittest.mock import patch

        pool = _make_dsv4_tpu_pool(self, num_req_slots=5)
        backend = object.__new__(DSV4TPUAttnBackend)
        backend.device = torch.device("cpu")
        backend.token_to_kv_pool = pool

        def _make_layer(q_head_norm: bool):
            return SimpleNamespace(
                n_local_heads=4,
                head_dim=16,
                q_lora_rank=8,
                q_head_norm=q_head_norm,
                eps=1e-6,
                freqs_cis=torch.ones((32, 8), dtype=torch.float32),
                q_norm=lambda x: x,
                wq_b=lambda x: (torch.ones((x.shape[0], 64), dtype=x.dtype), None),
                wkv=lambda x: (torch.ones((x.shape[0], 16), dtype=x.dtype), None),
                kv_norm=lambda x: x,
            )

        x_linear = torch.ones((2, 32), dtype=torch.bfloat16)
        q_lora = torch.ones((2, 8), dtype=torch.bfloat16)
        positions = torch.tensor([0, 1], dtype=torch.int64)

        with (
            patch(
                "sglang_tpu.dsv4.dsv4_attention_backend.apply_tpu_dsv4_qnorm_rope",
                side_effect=lambda q, pos, fc, eps: q * 2,
            ) as mock_qnorm_rope,
            patch(
                "sglang_tpu.dsv4.dsv4_attention_backend.apply_tpu_dsv4_rope",
                side_effect=lambda x, pos, fc, inverse=False: x * 3,
            ) as mock_rope,
        ):
            _, q_with_norm, kv_with_norm = backend.prepare_mqa_qkv(
                _make_layer(q_head_norm=True), x_linear, q_lora, None, positions
            )
            self.assertEqual(mock_qnorm_rope.call_count, 1)
            self.assertEqual(mock_rope.call_count, 1)  # Only KV uses plain rope
            self.assertEqual(tuple(q_with_norm.shape), (2, 4, 16))
            self.assertEqual(tuple(kv_with_norm.shape), (2, 16))

            mock_qnorm_rope.reset_mock()
            mock_rope.reset_mock()

            q_out = torch.zeros((2, 4, 16), dtype=torch.bfloat16)
            _, q_no_norm, _ = backend.prepare_mqa_qkv(
                _make_layer(q_head_norm=False),
                x_linear,
                q_lora,
                None,
                positions,
                q_out=q_out,
            )
            self.assertEqual(mock_qnorm_rope.call_count, 0)
            self.assertEqual(mock_rope.call_count, 2)  # Both Q and KV use plain rope
            self.assertTrue(torch.equal(q_out, q_no_norm))


if __name__ == "__main__":
    unittest.main()
