"""Tests for cost-balanced shared-compilation owners."""

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.compilation import shared_compile_cost
from sglang_tpu.compilation.spmd_compiler import SPMDCompiler

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)


def _compiler(size, estimate):
    compiler = object.__new__(SPMDCompiler)
    compiler.size = size
    compiler.verify_spmd_safe = False
    compiler.submissions = {}
    compiler._estimate = estimate
    compiler._estimated_load = [0.0] * size
    return compiler


def _owners(compiler, graphs):
    for graph in graphs:
        compiler.enqueue(graph)
    return [compiler.submissions[graph].owner for graph in graphs]


class TestSharedCompileOwners(CustomTestCase):
    def test_graph_family_replaces_numbers(self):
        self.assertEqual(
            shared_compile_cost.graph_family("target/model.forward:[[[8,2]]]"),
            "target/model.forward",
        )
        self.assertEqual(
            shared_compile_cost.graph_family("verify_16_202048_8_7_BLOCK_random1"),
            "verify_N_N_N_N_BLOCK_randomN",
        )
        self.assertEqual(
            shared_compile_cost.graph_family("sampler_tp16_k50_0p17_v13312"),
            "sampler_tpN_kN_NpN_vN",
        )

    def test_round_robin_owners_without_a_profile(self):
        compiler = _compiler(3, None)
        self.assertEqual(
            _owners(compiler, [f"g{i}:x" for i in range(7)]), [0, 1, 2, 0, 1, 2, 0]
        )

    def test_owners_go_to_the_least_estimated_load(self):
        costs = {"heavy": 10.0, "light": 1.0}
        compiler = _compiler(3, lambda graph: costs[graph.split(":")[0]])
        graphs = ["heavy:1", "light:1", "light:2", "heavy:2", "light:3", "heavy:3"]
        # Heavy graphs spread before light ones stack on a rank; ties take the
        # lowest rank.
        self.assertEqual(_owners(compiler, graphs), [0, 1, 2, 1, 2, 2])
        self.assertEqual(compiler._estimated_load, [10.0, 11.0, 12.0])

    def test_dropping_graphs_of_one_family_keeps_heavy_owners_spread(self):
        costs = {"forward": 18.0, "helper": 0.02}
        estimate = lambda graph: costs[graph.split(":")[0]]  # noqa: E731
        forwards = [f"forward:{bs}" for bs in range(4)]
        with_helpers = [g for bs in range(4) for g in (f"helper:{bs}", forwards[bs])]
        owners = dict(zip(with_helpers, _owners(_compiler(4, estimate), with_helpers)))
        # Round-robin would put all four forwards on ranks 1 and 3.
        self.assertEqual(sorted(owners[graph] for graph in forwards), [0, 1, 2, 3])
        self.assertEqual(_owners(_compiler(4, estimate), forwards), [0, 1, 2, 3])

    def test_invalid_costs_are_rejected(self):
        for cost in (-1.0, math.nan, math.inf):
            with self.subTest(cost=cost):
                with self.assertRaisesRegex(
                    ValueError, "Invalid shared TPU graph cost"
                ):
                    _compiler(2, lambda graph: cost).enqueue("g:1")


class TestSharedCompileCostProfile(CustomTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "profile.json"

    def test_profile_from_logs_round_trips(self):
        lines = [
            "[TP0] [TPU] [shared_compile] owner=0 trace=18.000s graph=target/model.forward:[[[8]]]",
            "[TP0] [TPU] [shared_compile] owner=1 trace=16.000s graph=target/model.forward:[[[16]]]",
            "[TP0] [TPU] [shared_compile] owner=2 trace=0.020s graph=helper_8:[[[8]]]",
            "[TP0] [TPU] [shared_compile] owner=3 trace=nans graph=untraced:[[[8]]]",
            "an unrelated line",
        ]
        profile = shared_compile_cost.build_cost_profile(lines)
        self.assertEqual(
            profile,
            {
                "schema_version": 1,
                "default_seconds": 16.0,
                "families": {"helper_N": 0.02, "target/model.forward": 17.0},
            },
        )
        self.path.write_text(json.dumps(profile))
        estimate = shared_compile_cost.load_cost_profile(self.path)
        self.assertEqual(estimate("target/model.forward:[[[32]]]"), 17.0)
        self.assertEqual(estimate("helper_64:[[[64]]]"), 0.02)
        self.assertEqual(estimate("unlisted:[[[1]]]"), 16.0)

    def test_profile_rejects_bad_schema_and_costs(self):
        self.path.write_text(json.dumps({"schema_version": 2}))
        with self.assertRaisesRegex(
            ValueError, "unsupported shared compile cost profile"
        ):
            shared_compile_cost.load_cost_profile(self.path)
        self.path.write_text(
            json.dumps(
                {"schema_version": 1, "default_seconds": 1, "families": {"x": -2}}
            )
        )
        with self.assertRaisesRegex(ValueError, "profile x is invalid"):
            shared_compile_cost.load_cost_profile(self.path)
        with self.assertRaisesRegex(ValueError, r"no \[shared_compile\] trace lines"):
            shared_compile_cost.build_cost_profile(["nothing here"])

    def test_estimator_follows_the_environment(self):
        env = "SGLANG_TPU_SHARED_COMPILE_COST_PROFILE"
        with patch.dict(os.environ, {env: ""}):
            self.assertIsNone(shared_compile_cost.estimator_from_env())
        self.path.write_text(
            json.dumps({"schema_version": 1, "default_seconds": 2, "families": {}})
        )
        with patch.dict(os.environ, {env: str(self.path)}):
            self.assertEqual(
                shared_compile_cost.estimator_from_env()("any:[[[1]]]"), 2.0
            )


if __name__ == "__main__":
    unittest.main()
