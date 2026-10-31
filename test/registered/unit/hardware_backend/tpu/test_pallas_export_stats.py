# SPDX-License-Identifier: Apache-2.0

"""Unit tests for cpu_export_jax_op export stats."""

import os
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import pallas_export_stats
from sglang_tpu.pallas_export_stats import (
    ExportStats,
    get_export_stats,
    record_export_call,
    reset_export_stats,
)

# Pure Python, no JAX or TorchTPU needed.
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_LOGGER = pallas_export_stats.logger.name
_THRESHOLD_ENV = "SGLANG_TPU_PALLAS_EXPORT_STATIC_VARIANT_THRESHOLD"


class _Tensor:
    def __init__(self, *shape, dtype="float32"):
        self.shape = shape
        self.dtype = dtype


class _Op:
    """Fake _CpuExportJaxCallable cache: a new (static values, shapes) key exports."""

    def __init__(self, name, param_names, static_argnums):
        self.name = name
        self.param_names = param_names
        self.static_argnums = static_argnums
        self.keys = set()

    def __call__(self, *args, **kwargs):
        key = (
            tuple(
                arg if i in self.static_argnums else (arg.shape, arg.dtype)
                for i, arg in enumerate(args)
            ),
            tuple(sorted(kwargs.items())),
        )
        exported = key not in self.keys
        self.keys.add(key)
        record_export_call(
            self.name,
            exported=exported,
            param_names=self.param_names,
            static_argnums=self.static_argnums,
            args=args,
            kwargs=kwargs,
        )


class TestPallasExportStats(CustomTestCase):
    def setUp(self):
        env_patcher = patch.dict(os.environ, {}, clear=False)
        env_patcher.start()
        self.addCleanup(env_patcher.stop)
        os.environ.pop(_THRESHOLD_ENV, None)
        reset_export_stats()
        self.addCleanup(reset_export_stats)

    def test_per_layer_static_warns_once(self):
        op = _Op("test::per_layer", ("x", "layer_index"), (1,))
        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            for _ in range(2):
                for layer_index in range(62):
                    op(_Tensor(8, 128), layer_index)

        self.assertEqual(
            get_export_stats(op.name),
            ExportStats(
                hits=62,
                misses=62,
                static_variants=62,
                shape_variants=1,
                max_static_variants_per_shape=62,
                varying_static_args={"layer_index": 62},
            ),
        )
        # The default threshold is 16, so the 17th layer warns, once.
        self.assertEqual(len(logs.output), 1)
        self.assertIn("test::per_layer exported 17 kernels", logs.output[0])
        self.assertIn("layer_index=17", logs.output[0])

    def test_bounded_static_hits_without_warning(self):
        op = _Op("test::bounded", ("x", "schedule_id"), (1,))
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            for layer_index in range(62):
                op(_Tensor(8, 128), layer_index % 2)

        stats = get_export_stats(op.name)
        self.assertEqual((stats.misses, stats.hits), (2, 60))
        self.assertEqual(stats.varying_static_args, {"schedule_id": 2})

    def test_shape_variants_are_counted_separately(self):
        op = _Op("test::shapes", ("x", "eps"), (1,))
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            for tokens in range(1, 41):
                op(_Tensor(tokens, 128), 1e-6)
            op(_Tensor(40, 128, dtype="bfloat16"), 1e-6)

        stats = get_export_stats(op.name)
        self.assertEqual((stats.misses, stats.shape_variants), (41, 41))
        self.assertEqual(stats.static_variants, 1)

    def test_static_that_follows_the_shape_does_not_warn(self):
        # One static value per shape bucket is expected, which is why the
        # threshold counts variants per shape rather than in total.
        op = _Op("test::bucketed", ("x", "block"), (1,))
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            for bucket in range(1, 41):
                op(_Tensor(bucket * 8, 128), bucket)

        stats = get_export_stats(op.name)
        self.assertEqual(stats.static_variants, 40)
        self.assertEqual(stats.max_static_variants_per_shape, 1)

    def test_warning_names_only_args_that_vary_at_one_shape(self):
        op = _Op("test::bucketed_per_layer", ("x", "block", "layer_index"), (1, 2))
        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            for bucket in (1, 2):
                for layer_index in range(20):
                    op(_Tensor(bucket * 8, 128), bucket, layer_index)

        self.assertEqual(len(logs.output), 1)
        self.assertIn("varying static args: layer_index=17.", logs.output[0])

    def test_warning_hides_static_values(self):
        op = _Op("test::tagged", ("x", "tag"), (1,))
        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            for i in range(17):
                op(_Tensor(4), f"private-{i}")

        self.assertIn("tag=17", logs.output[0])
        self.assertNotIn("private", logs.output[0])

    def test_threshold_env(self):
        os.environ[_THRESHOLD_ENV] = "0"  # turns off the warning, not counting
        disabled = _Op("test::disabled", ("x", "layer_index"), (1,))
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            for layer_index in range(64):
                disabled(_Tensor(8), layer_index)
        self.assertEqual(get_export_stats(disabled.name).misses, 64)

        os.environ[_THRESHOLD_ENV] = "4"
        low = _Op("test::low_threshold", ("x", "layer_index"), (1,))
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            for layer_index in range(4):
                low(_Tensor(8), layer_index)
        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            low(_Tensor(8), 4)
        self.assertIn("(threshold 4)", logs.output[0])

    def test_kwargs_and_unnamed_args(self):
        op = _Op("test::unnamed", (), (1,))  # no names: falls back to argN
        for i in range(3):
            op(_Tensor(8), i, mode="a" if i < 2 else "b")

        self.assertEqual(
            get_export_stats(op.name).varying_static_args, {"arg1": 3, "mode": 2}
        )

    def test_debug_log_explains_each_miss(self):
        op = _Op("test::explained", ("x", "layer_index"), (1,))
        with self.assertLogs(_LOGGER, level="DEBUG") as logs:
            op(_Tensor(8), 0)
            op(_Tensor(8), 1)
            op(_Tensor(16), 1)
            op(_Tensor(16), 1)  # hit: not logged
            op(_Tensor(16), 0)

        causes = [line.split(": ", 1)[1] for line in logs.output]
        self.assertEqual(
            causes,
            [
                "new layer_index value, new input shape/dtype",
                "new layer_index value",
                "new input shape/dtype",
                "no new static value or input shape/dtype",
            ],
        )

    def test_hits_skip_signature_work(self):
        op = _Op("test::hits", ("x", "k"), (1,))
        op(_Tensor(8), 4)
        with patch.object(pallas_export_stats, "_signatures") as signatures:
            for _ in range(100):
                op(_Tensor(8), 4)

        signatures.assert_not_called()
        self.assertEqual(get_export_stats(op.name).hits, 100)

    def test_reset(self):
        os.environ[_THRESHOLD_ENV] = "1"
        op = _Op("test::reset", ("x", "layer_index"), (1,))
        with self.assertLogs(_LOGGER, level="WARNING"):
            op(_Tensor(8), 0)
            op(_Tensor(8), 1)

        reset_export_stats()
        self.assertIsNone(get_export_stats(op.name))
        op(_Tensor(8), 0)  # still cached by the op, so a hit after the reset
        self.assertEqual(get_export_stats(op.name).hits, 1)
        with self.assertLogs(_LOGGER, level="WARNING"):  # the warning re-arms
            op(_Tensor(8), 2)
            op(_Tensor(8), 3)


if __name__ == "__main__":
    unittest.main()
