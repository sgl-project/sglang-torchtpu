"""Unit tests for the packed-FP4 gmm default tiles in ``tuned_block_sizes``.

Pure tile-sizing math: ``get_tuned_block_sizes`` is a Python lookup keyed on
static shapes, so these run without a TPU. ``TUNED_BLOCK_SIZES`` holds no
``float4_e2m1fn`` entries, so every packed-FP4 gmm takes the default.
"""

import importlib.util
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Load the lookup module by path: importing it through the pallas package
# would load the TPU op chain, which CPU agents lack.
_REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "python" / "sglang").is_dir()
)
_SPEC = importlib.util.spec_from_file_location(
    "gmm_tuned_block_sizes", _REPO_ROOT / "sglang_tpu/moe/pallas/tuned_block_sizes.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
get_default_gmm_block_sizes = _MODULE.get_default_gmm_block_sizes
get_tuned_block_sizes = _MODULE.get_tuned_block_sizes

# gmm2 of a 128-local-expert layer with a 4096-wide hidden: K = intermediate
# (4096 here), N = hidden; m = chunk tokens x top-8.
_K, _N, _G = 4096, 4096, 128


def _tiles(m, rhs_dtype="float4_e2m1fn", block=512, k=_K, n=_N, g=_G, vmem=None):
    return get_tuned_block_sizes(
        vmem_limit_bytes=vmem,
        m=m,
        k=k,
        n=n,
        num_total_groups=g,
        num_current_groups=g,
        lhs_dtype="bfloat16",
        rhs_dtype=rhs_dtype,
        quant_block_size=block,
    )


class TestPackedFp4DefaultTiles(unittest.TestCase):
    def test_w4a8_caps_tile_m_and_widens_tile_n(self):
        # 2K / 4K / 8K-token chunks at top-8.
        for m in (16384, 32768, 65536):
            with self.subTest(m=m):
                # tile_k (2048) is the generic default's.
                self.assertEqual(_tiles(m), (256, 2048, 4096))

    def test_tile_m_never_grows_and_divides_m(self):
        for m in (128, 256, 640, 1024, 3072, 9600, 16384, 24576, 49152, 65536):
            with self.subTest(m=m):
                tm, _, _ = _tiles(m)
                self.assertLessEqual(tm, get_default_gmm_block_sizes(m, _K, _N, _G)[0])
                self.assertEqual(m % tm, 0)

    def test_tile_n_covers_n_up_to_4096(self):
        # The generic default caps tile_n at 2048; the packed-FP4 rule takes
        # the whole N up to 4096 (3072 instead of 1536 below).
        for n, expected in ((1024, 1024), (2048, 2048), (3072, 3072), (8192, 4096)):
            with self.subTest(n=n):
                self.assertEqual(_tiles(32768, n=n)[2], expected)

    def test_tile_n_falls_back_when_the_rhs_tile_overflows_vmem(self):
        # A 2048x4096 4-bit rhs tile is 4 MiB; calculate_tiling budgets a third
        # of the limit per rhs buffer.
        mib = 1024 * 1024
        self.assertEqual(_tiles(65536, vmem=int(64 * mib * 0.9)), (256, 2048, 4096))
        self.assertEqual(_tiles(65536, vmem=12 * mib), (256, 2048, 4096))
        self.assertEqual(_tiles(65536, vmem=12 * mib - 3), (256, 2048, 2048))

    def test_other_rhs_and_w4a16_blocks_keep_the_generic_default(self):
        cases = [
            dict(rhs_dtype="float8_e4m3fn", block=128),
            dict(rhs_dtype="bfloat16", block=_K),
            dict(
                rhs_dtype="float4_e2m1fn", block=128
            ),  # W4A16: rhs dequantized in VMEM
        ]
        for case in cases:
            with self.subTest(**case):
                self.assertEqual(
                    _tiles(65536, **case),
                    get_default_gmm_block_sizes(65536, _K, _N, _G),
                )


if __name__ == "__main__":
    unittest.main()
