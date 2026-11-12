"""Block top-k admission and native exact selection contracts.

CPU-only admission can be run by selecting TestBlockMaximumTopKAdmission.
Native tests require a TPU with SparseCore; no CPU interpretation substitute.
"""

import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import tpu_info
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import pallas_export
from sglang_tpu.kernels.deepseek_v41.index_selection import block_topk as impl

register_tpu_ci(
    est_time=60,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)


def ordered_bits(values):
    """Unsigned monotone IEEE keys; +0 ranks above -0, NaNs unsupported."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return bits ^ np.where(bits >> 31, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))


def exact_topk(scores, lengths, k):
    """Score descending / logical ID ascending, then ascending IDs and -1 tail."""
    scores = np.asarray(scores, dtype=np.float32)
    result = np.full((scores.shape[0], k), -1, np.int32)
    for row, length in enumerate(np.clip(lengths, 0, scores.shape[1])):
        ids = np.flatnonzero(scores[row, :length] > -np.inf)
        order = np.lexsort((ids, ~ordered_bits(scores[row, ids])))[:k]
        selected = np.sort(ids[order])
        result[row, : selected.size] = selected
    return result


def published_blocks(scores, lengths, count, block_size):
    """Visible block maxima, newest mandatory even when other blocks have +inf."""
    scores = np.asarray(scores, dtype=np.float32)
    if np.any(np.isnan(scores)):
        raise ValueError("NaN scores are unsupported")
    width = scores.shape[1]
    blocks = (width + block_size - 1) // block_size
    maxima = np.full((scores.shape[0], blocks), -np.inf, np.float32)
    for row, length in enumerate(np.clip(lengths, 0, width)):
        for block in range((int(length) + block_size - 1) // block_size):
            values = scores[
                row, block * block_size : min((block + 1) * block_size, length)
            ]
            # Integer-key maximum deliberately distinguishes signed zeros.
            maxima[row, block] = values[np.argmax(ordered_bits(values))]
    result = np.full((scores.shape[0], count), -1, np.int32)
    for row, length in enumerate(np.clip(lengths, 0, width)):
        if not length or not count:
            continue
        latest = (length - 1) // block_size
        maxima[row, latest] = -np.inf
        others = exact_topk(maxima[row : row + 1], [blocks], count - 1)[0]
        selected = np.sort(np.append(others[others >= 0], latest))
        result[row, : selected.size] = selected
    return result


def adversarial_scores(width, k, block_size):
    """Different failure modes per row, batched to share one kernel compile."""
    rng = np.random.default_rng(20261112)
    scores = rng.normal(size=(16, width)).astype(np.float32)
    lengths = np.array(
        [
            -7,
            0,
            1,
            k - 1,
            k,
            k + 1,
            block_size - 1,
            block_size,
            block_size + 1,
            width - 1,
            width,
            width + 7,
            width,
            width,
            width,
            width,
        ],
        np.int32,
    )
    scores[6] = np.arange(width, dtype=np.float32)
    scores[7] = -np.arange(width, dtype=np.float32)
    scores[8] = 1
    # Many blocks have the same maximum; only the lowest logical IDs win ties.
    scores[9] = (np.arange(width) // block_size % 5).astype(np.float32)
    scores[10] = -np.inf
    # Alternating zero signs also lie within each block, not just across blocks.
    scores[11] = np.float32(-0.0)
    scores[11, 1::2] = np.float32(0.0)
    scores[12, ::3] = -np.inf
    scores[13] = 2
    scores[13, -block_size:] = -100  # forced publication must not change token set
    scores[14] = -1
    scores[14, ::block_size] = 3
    scores[15] = -np.inf
    scores[15, -1] = -10
    scores[np.arange(width)[None, :] >= np.clip(lengths, 0, width)[:, None]] = -np.inf
    return scores, lengths


class TestBlockMaximumTopKAdmission(CustomTestCase):
    def test_shapes_and_vmem_admission(self):
        info = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1)
        with (
            patch.object(tpu_info, "is_tpu_device", return_value=True),
            patch.object(pltpu, "get_tpu_info", return_value=info),
            patch.object(plsc, "get_sparse_core_info", return_value=info.sparse_core),
        ):
            self.assertTrue(impl.is_supported(16, 16384, 512, 8, publish=128))
            for args, publish in [
                ((0, 16384, 512, 8), 0),
                ((16, 16383, 512, 8), 0),
                ((16, 16384, 511, 8), 0),
                ((16, 16384, 0, 8), 0),
                ((16, 16384, 16384, 8), 0),
                ((16, 16384, 512, 3), 0),
                # HBM-aligned but final full-span maxima gather would overrun.
                ((1, 2176, 128, 16), 0),
                ((16, 16384, 512, 0), 0),
                ((16, 4096, 512, 8), 0),
                ((16, 16384, 512, 8), -128),
                ((16, 16384, 512, 8), 1),
                ((16, 16384, 512, 8), 4096),
                # The maxima alone exceed per-subcore VMEM.
                ((16, 1 << 24, 512, 8), 0),
            ]:
                with self.subTest(args=args, publish=publish):
                    self.assertFalse(impl.is_supported(*args, publish=publish))

    def test_plain_cpu_is_not_accelerator_admission(self):
        with patch.object(tpu_info, "is_tpu_device", return_value=False):
            self.assertFalse(impl.is_supported(16, 16384, 512, 8))
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                impl.block_topk(
                    jnp.zeros((1, 16384), jnp.float32),
                    jnp.array([16384], jnp.int32),
                    k=512,
                    block_size=8,
                )

    def test_cpu_export_target_admission_is_scoped(self):
        # Cross-compilation keeps a CPU backend but must admit the TPU target.
        # Exercise the real metadata context, including restoration on failure.
        with (
            patch.object(jax, "default_backend", return_value="cpu"),
            patch.object(tpu_info, "is_tpu_device", return_value=False),
            patch.object(
                pallas_export, "get_pallas_tpu_device_kind", return_value="TPU7x"
            ),
        ):
            self.assertFalse(impl.is_supported(16, 16384, 512, 8))
            with pallas_export.tpu_info_for_cpu_export():
                self.assertEqual(jax.default_backend(), "cpu")
                self.assertTrue(impl.is_supported(16, 16384, 512, 8, publish=128))
            self.assertFalse(impl.is_supported(16, 16384, 512, 8))
            with self.assertRaisesRegex(RuntimeError, "export failed"):
                with pallas_export.tpu_info_for_cpu_export():
                    self.assertTrue(impl.is_supported(16, 16384, 512, 8))
                    raise RuntimeError("export failed")
            self.assertFalse(impl.is_supported(16, 16384, 512, 8))

    def test_public_fp8_selector_export_dispatch(self):
        from jax import export

        from sglang_tpu.kernels.deepseek_v41.index_selection.fp8.cache import (
            PackedFP8Layout,
        )
        from sglang_tpu.kernels.deepseek_v41.index_selection.selection import (
            select_fp8_indexer_jax,
        )

        # Complete scorer group traces serial and pipelined selection branches.
        width, k, rows = 8192, 512, 6
        layout = PackedFP8Layout(width // 128)
        layout_args = (
            layout.pages,
            layout.page_stride,
            layout.page_offset,
            layout.page_size,
        )
        specs = (
            jax.ShapeDtypeStruct((rows, 32, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((layout.pages * layout.page_rows, 128), jnp.int32),
            jax.ShapeDtypeStruct((1, width // 128), jnp.int32),
            jax.ShapeDtypeStruct((rows, 32), jnp.float32),
            jax.ShapeDtypeStruct((rows,), jnp.int32),
            jax.ShapeDtypeStruct((rows, 128), jnp.int32),
            jax.ShapeDtypeStruct((rows,), jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        )
        with patch.object(
            pallas_export, "get_torch_tpu_device_kind", return_value="TPU v7"
        ):
            with (
                pallas_export.tpu_info_for_cpu_export(),
                jax._src.config.export_ignore_forward_compatibility(True),
            ):
                # Actual packed scorer + public selector, not mocked dispatch.
                # Check source publication too: its existing path must coexist.
                for unordered, publish in ((True, False), (True, True), (False, False)):

                    def run(
                        q, bank, pages, weights, lengths, candidates, owners, maximum
                    ):
                        return select_fp8_indexer_jax(
                            q,
                            (bank,),
                            pages,
                            weights,
                            lengths,
                            candidates,
                            owners,
                            maximum,
                            layout_args,
                            k,
                            8,
                            publish,
                            False,
                            unordered,
                            width,
                        )

                    hlo = export.export(jax.jit(run), platforms=("tpu",))(
                        *specs
                    ).mlir_module()
                    with self.subTest(unordered=unordered, publish=publish):
                        self.assertEqual("block_topk" in hlo, unordered)


class TestBlockMaximumTopK(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if jax.default_backend() != "tpu":
            raise unittest.SkipTest("native SparseCore execution requires TPU")
        cls.info = pltpu.get_tpu_info()
        if cls.info.sparse_core is None:
            raise unittest.SkipTest("target has no SparseCore")

    def test_exact_contract_resident_and_streamed(self):
        k, block_size, rows = 512, 8, 16
        plans = {}
        # Bound the matrix: one representative of each physical execution plan.
        for width in (8192, 16384, 32768, 65536, 131072, 262144, 524288):
            if not impl.is_supported(rows, width, k, block_size, publish=128):
                continue
            blocks = impl.default_blocks(k, self.info.sparse_core.num_lanes)
            block_width = (
                (width // block_size + self.info.num_lanes - 1)
                // self.info.num_lanes
                * self.info.num_lanes
            )
            plan = impl._plan(rows, width, block_width, k, blocks, block_size, 128)
            plans.setdefault(plan[0], width)
        self.assertEqual(set(plans), {True, False}, "must exercise both VMEM plans")
        for resident, width in plans.items():
            scores, lengths = adversarial_scores(width, k, block_size)
            expected = exact_topk(scores, lengths, k)
            for publish in (0, 128):
                with self.subTest(resident=resident, width=width, publish=publish):
                    run = jax.jit(
                        lambda s, n: impl.block_topk(
                            s, n, k=k, block_size=block_size, publish=publish
                        )
                    )
                    result = jax.device_get(
                        run(jnp.asarray(scores), jnp.asarray(lengths))
                    )
                    if publish:
                        selected, published = result
                        np.testing.assert_array_equal(
                            published,
                            published_blocks(scores, lengths, publish, block_size),
                        )
                    else:
                        selected = result
                    # Full array equality checks order, exact membership and -1 suffix.
                    np.testing.assert_array_equal(selected, expected)

    def test_buffer_reuse_and_full_span_boundary(self):
        # More than three rows per vector subcore wraps both score slots and
        # exercises previous-output waits, including empty/nonempty rows.
        rows = (
            self.info.sparse_core.num_cores * self.info.sparse_core.num_subcores * 3 + 1
        )
        for width, block_size, k in [
            (8192, 16, 128),
            (65536, 8, 512),
            (131072, 8, 512),
        ]:
            with self.subTest(width=width, block_size=block_size):
                self.assertTrue(
                    impl.is_supported(rows, width, k, block_size, publish=128)
                )
                scores, lengths = adversarial_scores(width, k, block_size)
                owners = np.arange(rows) % len(scores)
                scores, lengths = scores[owners], lengths[owners]
                selected, published = jax.jit(
                    lambda s, n: impl.block_topk(
                        s, n, k=k, block_size=block_size, publish=128
                    )
                )(jnp.asarray(scores), jnp.asarray(lengths))
                np.testing.assert_array_equal(
                    np.asarray(selected), exact_topk(scores, lengths, k)
                )
                np.testing.assert_array_equal(
                    np.asarray(published),
                    published_blocks(scores, lengths, 128, block_size),
                )
        self.assertFalse(impl.is_supported(1, 2176, 128, 16))

    def test_positive_infinity_token_selection(self):
        width, k, block_size = 16384, 512, 8
        self.assertTrue(impl.is_supported(1, width, k, block_size))
        scores = np.full((1, width), -np.inf, np.float32)
        scores[0, ::7] = np.inf
        lengths = np.array([width], np.int32)
        selected, published = jax.jit(
            lambda s, n: impl.block_topk(s, n, k=k, block_size=block_size, publish=128)
        )(jnp.asarray(scores), jnp.asarray(lengths))
        np.testing.assert_array_equal(
            np.asarray(selected), exact_topk(scores, lengths, k)
        )
        np.testing.assert_array_equal(
            np.asarray(published), published_blocks(scores, lengths, 128, block_size)
        )


if __name__ == "__main__":
    unittest.main()
