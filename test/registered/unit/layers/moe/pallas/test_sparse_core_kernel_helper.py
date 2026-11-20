# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests verifying core_map_helper lowers directly through pl.kernel."""

import ast
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)

_HAS_JAX = importlib.util.find_spec("jax") is not None


class TestSparseCoreKernelHelper(CustomTestCase):
    def test_module_avoids_lax_empty_and_new_ref(self):
        helper_path = (
            Path(__file__).resolve().parents[6]
            / "sglang_tpu/moe/pallas/sparse_core/core_map_helper.py"
        )
        tree = ast.parse(
            helper_path.read_text(encoding="utf-8"), filename=str(helper_path)
        )

        called_attrs = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("kernel", called_attrs)
        self.assertIn("jit", called_attrs)
        self.assertNotIn("empty", called_attrs)
        self.assertNotIn("new_ref", called_attrs)
        self.assertNotIn("core_map", called_attrs)

    @unittest.skipUnless(_HAS_JAX, "Requires JAX")
    def test_kernel_delegates_to_pl_kernel_wrapped_in_api_jit(self):
        from jax import lax
        from jax._src import core as jax_core

        from sglang_tpu.moe.pallas.sparse_core import core_map_helper

        body = object()
        out_type = object()
        mesh = object()
        scratch_types = (object(),)
        compiler_params = object()
        cost_estimate = object()
        metadata = {"tag": "sc_test"}
        lowered = object()
        jitted = object()

        with (
            mock.patch.object(
                core_map_helper.pl, "kernel", return_value=lowered
            ) as mock_pl_kernel,
            mock.patch.object(
                core_map_helper.api, "jit", return_value=jitted
            ) as mock_api_jit,
            mock.patch.object(lax, "empty") as mock_empty,
            mock.patch.object(jax_core, "new_ref") as mock_new_ref,
        ):
            got = core_map_helper.kernel(
                body,
                out_type=out_type,
                mesh=mesh,
                scratch_types=scratch_types,
                compiler_params=compiler_params,
                interpret=True,
                cost_estimate=cost_estimate,
                debug=True,
                name="sc_custom_op",
                metadata=metadata,
            )

        self.assertIs(got, jitted)
        mock_pl_kernel.assert_called_once_with(
            body,
            out_type=out_type,
            mesh=mesh,
            scratch_types=scratch_types,
            compiler_params=compiler_params,
            interpret=True,
            cost_estimate=cost_estimate,
            debug=True,
            name="sc_custom_op",
            metadata=metadata,
        )
        mock_api_jit.assert_called_once_with(lowered)
        mock_empty.assert_not_called()
        mock_new_ref.assert_not_called()


if __name__ == "__main__":
    unittest.main()
