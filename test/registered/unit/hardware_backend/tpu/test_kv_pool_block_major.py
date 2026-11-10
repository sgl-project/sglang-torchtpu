"""CPU parity checks of the block-major TPU KV pool against the fused pool.

``kv_bundle[b, l]`` holds page ``b`` of layer ``l``;
the flat kernel page is ``b * layer_num + l``.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_pallas_rpa_base import _fused_kv_operands
from sglang_tpu.disaggregation.raiden.common import get_raiden_kv_caches
from sglang_tpu.kv_pool import (
    TpuBlockMajorMHATokenToKVPool,
    TpuFusedMHATokenToKVPool,
)
from sglang_tpu.tpu_platform import _check_block_major_supported

HEAD_NUM = 1
HEAD_DIM = 128
CHUNK_SIZE = 3
PAGE_SIZE = 4
SIZE = 32
LAYER_NUM = 3


def _make_pool(pool_cls, size=SIZE, page_size=PAGE_SIZE, layer_num=LAYER_NUM):
    pool = pool_cls(
        size=size,
        page_size=page_size,
        dtype=torch.bfloat16,
        head_num=HEAD_NUM,
        head_dim=HEAD_DIM,
        layer_num=layer_num,
        device="cpu",
        enable_memory_saver=False,
    )
    pool.cpu_offloading_chunk_size = CHUNK_SIZE
    return pool


def _fill_distinct(pool):
    # Every (layer, token row) gets its own value to detect slice aliasing.
    for layer in range(pool.layer_num):
        buf = pool.kv_buffer[layer]
        n_tokens = buf.shape[0] * buf.shape[1]
        values = torch.arange(n_tokens, dtype=torch.float32) + layer * 1000
        buf.copy_(
            values.view(buf.shape[0], buf.shape[1], 1, 1, 1)
            .expand(buf.shape)
            .to(buf.dtype)
        )


class TestBlockMajorLayout(CustomTestCase):
    def test_bundle_shape_and_layer_views(self):
        fused = _make_pool(TpuFusedMHATokenToKVPool)
        bm = _make_pool(TpuBlockMajorMHATokenToKVPool)
        per_layer_shape = fused.kv_buffer[0].shape
        self.assertEqual(
            bm.kv_bundle.shape,
            (per_layer_shape[0], LAYER_NUM, *per_layer_shape[1:]),
        )
        for layer in range(LAYER_NUM):
            self.assertEqual(bm.kv_buffer[layer].shape, per_layer_shape)
            self.assertEqual(
                bm.kv_buffer[layer].data_ptr(), bm.kv_bundle[:, layer].data_ptr()
            )
        bm.kv_buffer[1][2].fill_(7.0)
        self.assertTrue(torch.all(bm.kv_bundle[2, 1] == 7.0))
        self.assertTrue(torch.all(bm.kv_bundle[2, 0] == 0.0))

    def test_flat_page_remap_matches_bundled_kernel_formula(self):
        bm = _make_pool(TpuBlockMajorMHATokenToKVPool)
        _fill_distinct(bm)
        flat = bm.kv_bundle.view(bm.num_blocks * bm.layer_num, *bm.kv_bundle.shape[2:])
        for block in (0, 1, bm.num_blocks - 1):
            for layer in range(bm.layer_num):
                torch.testing.assert_close(
                    flat[block * bm.layer_num + layer],
                    bm.kv_buffer[layer][block],
                )

    def test_fp8_bundle_allocates(self):
        bm = _make_pool(TpuBlockMajorMHATokenToKVPool)
        fp8 = TpuBlockMajorMHATokenToKVPool(
            size=SIZE,
            page_size=PAGE_SIZE,
            dtype=torch.float8_e4m3fn,
            head_num=HEAD_NUM,
            head_dim=HEAD_DIM,
            layer_num=LAYER_NUM,
            device="cpu",
            enable_memory_saver=False,
        )
        self.assertEqual(fp8.kv_bundle.dtype, torch.float8_e4m3fn)
        self.assertEqual(fp8.kv_bundle.shape[:3], bm.kv_bundle.shape[:3])

    def test_logical_kv_view_matches_fused(self):
        fused = _make_pool(TpuFusedMHATokenToKVPool)
        bm = _make_pool(TpuBlockMajorMHATokenToKVPool)
        _fill_distinct(fused)
        _fill_distinct(bm)
        for layer in range(LAYER_NUM):
            torch.testing.assert_close(
                bm.get_key_buffer(layer), fused.get_key_buffer(layer)
            )
            torch.testing.assert_close(
                bm.get_value_buffer(layer), fused.get_value_buffer(layer)
            )

    def test_contiguous_buf_infos_is_the_bundle(self):
        bm = _make_pool(TpuBlockMajorMHATokenToKVPool)
        ptrs, lens, item_lens = bm.get_contiguous_buf_infos()
        self.assertEqual(ptrs, [bm.kv_bundle.data_ptr()])
        self.assertEqual(lens, [bm.kv_bundle.nbytes])
        self.assertEqual(item_lens, [bm.kv_bundle[0].nbytes])

    def test_offload_backends_are_raiden_only(self):
        self.assertEqual(
            TpuBlockMajorMHATokenToKVPool.kv_offload_io_backends, frozenset({"raiden"})
        )
        self.assertIn("torch_tpu", TpuFusedMHATokenToKVPool.kv_offload_io_backends)

    def test_get_state_round_trip_carries_layer_indices(self):
        """The compiled forward donates every state tensor and swaps in the
        returned ones, so the kernel operands must come from the swapped state:
        the bundle and one int32 layer-index tensor."""
        bm = _make_pool(TpuBlockMajorMHATokenToKVPool)
        state = bm.get_state()
        self.assertEqual(len(state), 2)
        self.assertIs(state[0], bm.kv_bundle)
        self.assertEqual(state[1].tolist(), list(range(LAYER_NUM)))
        self.assertEqual(state[1].dtype, torch.int32)
        new_state = [
            torch.zeros_like(state[0]),
            torch.arange(5, 5 + LAYER_NUM, dtype=torch.int32),
        ]
        bm.set_state(new_state)
        bundle, layer_idx = bm.get_fused_kv_operands(1)
        self.assertIs(bundle, new_state[0])
        self.assertEqual(int(layer_idx), 6)
        self.assertEqual(bm.kv_buffer[1].data_ptr(), bm.kv_bundle[:, 1].data_ptr())


class TestBlockMajorCpuCopy(CustomTestCase):
    """The row path produces the fused pool's payload and restores it bit for bit."""

    def _round_trip(self, indices, load_indices=None, row_mask=None):
        fused = _make_pool(TpuFusedMHATokenToKVPool)
        bm = _make_pool(TpuBlockMajorMHATokenToKVPool)
        _fill_distinct(fused)
        _fill_distinct(bm)
        idx = torch.tensor(indices, dtype=torch.long)
        fused_cpu = fused.get_cpu_copy(idx)
        bm_cpu = bm.get_cpu_copy(idx)
        self.assertEqual(len(bm_cpu), len(fused_cpu))
        for bm_chunks, fused_chunks in zip(bm_cpu, fused_cpu):
            self.assertEqual(len(bm_chunks), len(fused_chunks))
            for a, b in zip(bm_chunks, fused_chunks):
                torch.testing.assert_close(a, b)
        if row_mask is not None:
            mask = torch.tensor(row_mask, dtype=torch.bool)
            fused_cpu = fused.filter_cpu_copy(fused_cpu, mask)
            bm_cpu = bm.filter_cpu_copy(bm_cpu, mask)
        dst = (
            idx
            if load_indices is None
            else torch.tensor(load_indices, dtype=torch.long)
        )
        fused.kv_buffer = [torch.zeros_like(b) for b in fused.kv_buffer]
        bm.kv_bundle.zero_()
        fused.load_cpu_copy(fused_cpu, dst)
        bm.load_cpu_copy(bm_cpu, dst)
        for layer in range(LAYER_NUM):
            torch.testing.assert_close(bm.kv_buffer[layer], fused.kv_buffer[layer])

    def test_round_trip_full_pages(self):
        # Token range [4, 12) spans physical pages 1 and 2 entirely.
        self._round_trip(list(range(4, 12)))

    def test_round_trip_partial_pages(self):
        # Page 2 is covered whole, pages 1 and 3 partly.
        self._round_trip([5, 6, 8, 9, 10, 11, 13])

    def test_round_trip_relocated(self):
        """A retraction resume lands the saved rows in different pages."""
        self._round_trip(
            [5, 6, 8, 9, 10, 11, 13],
            load_indices=[17, 18, 20, 21, 22, 23, 25],
        )

    def test_round_trip_relocated_and_filtered(self):
        """The SWA member restores a row-filtered payload, as its window eviction produces."""
        self._round_trip(
            list(range(4, 12)),
            load_indices=[16, 17, 18, 19, 20, 21],
            row_mask=[True, True, False, False, True, True, True, True],
        )


class TestBlockMajorRaidenTransfer(CustomTestCase):
    """RaidenKVTransfer registers the bundle, never the strided layer views."""

    def _build_transfer(self, pool):
        import sys
        import types

        captured = {}

        class FakeKVCacheManager:
            def __init__(self, kv_caches, **kwargs):
                captured["kv_caches"] = kv_caches
                captured["kwargs"] = kwargs

        mod = types.ModuleType("tpu_sync.api.torch.kv_cache_manager")
        mod.KVCacheManager = FakeKVCacheManager
        stubs = {
            "tpu_sync": types.ModuleType("tpu_sync"),
            "tpu_sync.api": types.ModuleType("tpu_sync.api"),
            "tpu_sync.api.torch": types.ModuleType("tpu_sync.api.torch"),
            "tpu_sync.api.torch.kv_cache_manager": mod,
        }
        saved = {name: sys.modules.get(name) for name in stubs}
        sys.modules.update(stubs)
        try:
            from sglang_tpu.raiden_transfer import RaidenKVTransfer

            transfer = RaidenKVTransfer(pool, num_host_pages=8)
        finally:
            for name, prev in saved.items():
                if prev is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = prev
        return transfer, captured

    def test_block_major_registers_the_bundle(self):
        bm = _make_pool(TpuBlockMajorMHATokenToKVPool)
        transfer, captured = self._build_transfer(bm)
        self.assertEqual(len(transfer._device_tensors), 1)
        self.assertIs(transfer._device_tensors[0][0], bm.kv_bundle)

    def test_layer_major_registers_per_layer(self):
        fused = _make_pool(TpuFusedMHATokenToKVPool)
        transfer, captured = self._build_transfer(fused)
        self.assertEqual(len(transfer._device_tensors), LAYER_NUM)
        for layer in range(LAYER_NUM):
            self.assertIs(transfer._device_tensors[layer][0], fused.kv_buffer[layer])


def _swa_pool(pool_cls, full_ids=(1, 3), swa_ids=(0, 2)):
    return SWAKVPool(
        size=SIZE,
        size_swa=SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        head_num=HEAD_NUM,
        head_dim=HEAD_DIM,
        swa_attention_layer_ids=list(swa_ids),
        full_attention_layer_ids=list(full_ids),
        device="cpu",
        token_to_kv_pool_class=pool_cls,
    )


class TestBlockMajorRaidenGroups(CustomTestCase):
    def test_plain_pool_registers_the_bundle(self):
        bm = _make_pool(TpuBlockMajorMHATokenToKVPool)
        groups = get_raiden_kv_caches(bm)
        self.assertEqual(len(groups.full), 1)
        self.assertIs(groups.full[0], bm.kv_bundle)
        self.assertEqual(groups.full_layer_ids, [0])
        self.assertEqual(groups.swa, [])

    def test_swa_pool_registers_both_bundles(self):
        pool = _swa_pool(TpuBlockMajorMHATokenToKVPool)
        groups = get_raiden_kv_caches(pool)
        self.assertIs(groups.full[0], pool.full_kv_pool.kv_bundle)
        self.assertIs(groups.swa[0], pool.swa_kv_pool.kv_bundle)
        self.assertEqual((groups.full_layer_ids, groups.swa_layer_ids), ([1], [0]))

    def test_pipeline_manifest_names_each_bundle_by_its_layers(self):
        from sglang_tpu.disaggregation.raiden.pipeline import pool_manifest

        stage = get_raiden_kv_caches(_swa_pool(TpuBlockMajorMHATokenToKVPool))
        self.assertEqual(
            [pool["tag"] for pool in pool_manifest(stage)], ["fa.l1+2", "swa.l0+2"]
        )
        later = get_raiden_kv_caches(
            _swa_pool(TpuBlockMajorMHATokenToKVPool, (9, 11), (8, 10))
        )
        self.assertEqual(
            [pool["tag"] for pool in pool_manifest(later)], ["fa.l9+2", "swa.l8+2"]
        )

    def test_layer_major_pool_is_unchanged(self):
        fused = _make_pool(TpuFusedMHATokenToKVPool)
        groups = get_raiden_kv_caches(fused)
        self.assertEqual(len(groups.full), LAYER_NUM)

    def test_swa_bundle_routing(self):
        pool = _swa_pool(TpuBlockMajorMHATokenToKVPool)
        bundle, layer_idx = _fused_kv_operands(pool, 2)
        self.assertIs(bundle, pool.swa_kv_pool.kv_bundle)
        self.assertEqual(int(layer_idx), 1)
        bundle, layer_idx = _fused_kv_operands(pool, 1)
        self.assertIs(bundle, pool.full_kv_pool.kv_bundle)
        self.assertEqual(int(layer_idx), 0)

    def test_layer_major_wrapper_has_no_layer_index(self):
        fused = _make_pool(TpuFusedMHATokenToKVPool)
        wrapper = SimpleNamespace(get_fused_kv_buffer=fused.get_fused_kv_buffer)
        buf, layer_idx = _fused_kv_operands(wrapper, 1)
        self.assertIs(buf, fused.kv_buffer[1])
        self.assertIsNone(layer_idx)


class TestBlockMajorStartupCheck(CustomTestCase):
    """Every consumer that reads the pool layer by layer is refused before the pool exists."""

    _SUPPORTED = dict(
        attention_backend="pallas_tpu_rpa_v3",
        prefill_attention_backend=None,
        decode_attention_backend=None,
        speculative_algorithm=None,
        pp_size=1,
        disaggregation_mode="null",
        disaggregation_transfer_backend="mooncake",
        enable_hierarchical_cache=False,
        hicache_io_backend="kernel",
    )

    def _check(self, **overrides):
        _check_block_major_supported(
            SimpleNamespace(**{**self._SUPPORTED, **overrides})
        )

    def setUp(self):
        for target, value in (
            ("sglang.srt.arg_groups.overrides.model_config_of", SimpleNamespace()),
            ("sglang.srt.configs.hybrid_arch.mambaish_config", None),
        ):
            patcher = mock.patch(target, return_value=value)
            started = patcher.start()
            self.addCleanup(patcher.stop)
        self.mambaish_config = started

    def test_supported_configurations_pass(self):
        self._check()
        self._check(
            disaggregation_mode="decode", disaggregation_transfer_backend="raiden"
        )
        self._check(enable_hierarchical_cache=True, hicache_io_backend="raiden")

    def test_platform_hook_runs_the_check(self):
        from sglang.srt.model_executor.cuda_graph_config import (
            default_cuda_graph_config,
        )
        from sglang_tpu.envs import envs as tpu_envs
        from sglang_tpu.tpu_platform import TpuSRTPlatform

        args = SimpleNamespace(
            dtype=None,
            sampling_backend=None,
            disable_cuda_graph=False,
            cuda_graph_config=default_cuda_graph_config(),
            enable_torch_compile=False,
            disable_overlap_schedule=False,
            mem_fraction_static=None,
            tp_size=1,
            dp_size=1,
            attn_dp_size=1,
            enable_dp_attention=False,
            enable_dp_lm_head=False,
            enable_dynamic_chunking=False,
            page_size=None,
            kv_cache_dtype="auto",
            use_ray=False,
            max_total_tokens=None,
            speculative_eagle_topk=None,
            speculative_use_rejection_sampling=False,
            optimistic_prefill_attempts=0,
            max_queued_requests=None,
            enable_priority_scheduling=False,
            disaggregation_enable_kv_checksum=False,
            dcp_size=1,
            attn_cp_size=1,
            enable_prefill_cp=False,
            tpu_compile_tracer="dynamo",
            json_model_override_args=None,
            **{k: v for k, v in self._SUPPORTED.items() if k != "attention_backend"},
            attention_backend=None,
        )
        args.speculative_algorithm = "DFLASH"
        # override() does not restore the variable when the body raises.
        self.addCleanup(tpu_envs.SGLANG_TPU_USE_BLOCK_MAJOR_KV.clear)
        with tpu_envs.SGLANG_TPU_USE_BLOCK_MAJOR_KV.override(True):
            with mock.patch("sglang_tpu.patches._prepare_tpu_env"):
                with self.assertRaisesRegex(ValueError, "speculative"):
                    TpuSRTPlatform().apply_server_args_defaults(args)

    def test_other_attention_backend_is_refused(self):
        with self.assertRaisesRegex(ValueError, "pallas_tpu_rpa_v3"):
            self._check(attention_backend="pallas_tpu_rpa_batched")

    def test_split_attention_backend_is_refused(self):
        with self.assertRaisesRegex(ValueError, "pallas_tpu_rpa_batched"):
            self._check(prefill_attention_backend="pallas_tpu_rpa_batched")

    def test_speculative_decoding_is_refused(self):
        with self.assertRaisesRegex(ValueError, "speculative"):
            self._check(speculative_algorithm="DFLASH")

    def test_non_raiden_pd_is_refused(self):
        with self.assertRaisesRegex(ValueError, "raiden"):
            self._check(
                disaggregation_mode="prefill", disaggregation_transfer_backend="tpu"
            )

    def test_non_pipeline_consumer_of_a_pipeline_is_refused(self):
        from sglang_tpu.envs import envs as tpu_envs

        decode = dict(
            disaggregation_mode="decode", disaggregation_transfer_backend="raiden"
        )
        with tpu_envs.SGLANG_RAIDEN_PIPELINE_TRANSFER.override(True):
            with self.assertRaisesRegex(ValueError, "same stage layout"):
                self._check(**decode)
            self._check(**decode, pp_size=4)
            self._check(**{**decode, "disaggregation_mode": "prefill"}, pp_size=4)

    def test_torch_tpu_hicache_is_refused(self):
        with self.assertRaisesRegex(ValueError, "hicache-io-backend"):
            self._check(enable_hierarchical_cache=True, hicache_io_backend="torch_tpu")

    def test_hybrid_linear_model_is_refused(self):
        self.mambaish_config.return_value = object()
        with self.assertRaisesRegex(ValueError, "hybrid linear-attention"):
            self._check()


if __name__ == "__main__":
    unittest.main()
