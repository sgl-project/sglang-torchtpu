"""CPU tests for --attention-backend flywheel's startup contract."""

import subprocess
import sys
import types
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _server_args(**overrides):
    fields = dict(
        attention_backend="flywheel",
        prefill_attention_backend=None,
        decode_attention_backend=None,
        page_size=128,
        kv_cache_dtype="auto",
        dtype="bfloat16",
        speculative_algorithm=None,
        disaggregation_mode="null",
        pp_size=1,
        dcp_size=1,
        attn_cp_size=1,
        attn_dp_size=1,
        enable_mixed_chunk=False,
        enable_hierarchical_cache=False,
        enable_unified_memory=False,
    )
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


class TestFlywheelServerArgs(CustomTestCase):
    def test_accepts_the_served_configuration(self):
        from sglang_tpu.attention_flywheel import validate_flywheel_server_args

        validate_flywheel_server_args(_server_args())
        validate_flywheel_server_args(
            _server_args(page_size=256, kv_cache_dtype="bf16")
        )
        for kv_cache_dtype in ("fp8_e4m3", "fp8_e5m2"):
            validate_flywheel_server_args(_server_args(kv_cache_dtype=kv_cache_dtype))

    def test_rejects_what_the_kernels_cannot_serve(self):
        from sglang_tpu.attention_flywheel import validate_flywheel_server_args

        for overrides, message in (
            ({"page_size": 192}, "--page-size 192"),
            ({"kv_cache_dtype": "fp4_e2m1"}, "--kv-cache-dtype fp4_e2m1"),
            ({"dtype": "float32"}, "--dtype float32"),
            ({"decode_attention_backend": "pallas_tpu_rpa_v3"}, "must both be"),
            ({"speculative_algorithm": "EAGLE"}, "speculative decoding"),
            ({"disaggregation_mode": "prefill"}, "disaggregation"),
            ({"pp_size": 2}, "pipeline parallelism"),
            ({"dcp_size": 2}, "decode context parallelism"),
            ({"attn_cp_size": 2}, "prefill context parallelism"),
            ({"attn_dp_size": 2}, "DP attention"),
            ({"enable_mixed_chunk": True}, "--enable-mixed-chunk"),
            ({"enable_hierarchical_cache": True}, "hierarchical cache"),
            ({"enable_unified_memory": True}, "--enable-unified-memory"),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    validate_flywheel_server_args(_server_args(**overrides))

    def test_allocates_the_flywheel_pool(self):
        # flywheel's 4D pool; the 5D fused buffer costs a whole-layer relayout
        # per call at head_dim 256, and the generic MHA pool does not match.
        from sglang_tpu.kv_pool import TpuFlywheelKVPool
        from sglang_tpu.tpu_platform import _mha_kv_pool_cls

        self.assertIs(_mha_kv_pool_cls("flywheel"), TpuFlywheelKVPool)

    def test_startup_imports_no_flywheel_tpu(self):
        # The launcher validates server args, and CPU hosts import the plugin,
        # without flywheel_tpu or JAX; only constructing the backend loads them.
        code = (
            "import sys\n"
            "from sglang_tpu.attention_flywheel import validate_flywheel_server_args\n"
            "from sglang_tpu.tpu_platform import _rpa_kernel_spec\n"
            "assert _rpa_kernel_spec('flywheel').min_page_size == 128\n"
            "loaded = sorted(m for m in ('flywheel_tpu', 'jax') if m in sys.modules)\n"
            "assert not loaded, loaded\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
