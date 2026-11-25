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
from sglang_tpu.compilation import shared_compile_cost, spmd_compiler
from sglang_tpu.compilation.spmd_compiler import SPMDCompiler

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)


def _compiler(size, profile):
    compiler = object.__new__(SPMDCompiler)
    compiler.size = size
    compiler.verify_spmd_safe = False
    compiler.submissions = {}
    compiler._profile = profile
    compiler._planned_owner, compiler._estimated_load = (
        profile.plan(size) if profile is not None else ({}, [0.0] * size)
    )
    return compiler


def _families(costs, default=0.0):
    return shared_compile_cost.CostProfile(default_seconds=default, families=costs)


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
        compiler = _compiler(3, _families({"heavy": 10.0, "light": 1.0}))
        graphs = ["heavy:1", "light:1", "light:2", "heavy:2", "light:3", "heavy:3"]
        # Heavy graphs spread before light ones stack on a rank; ties take the
        # lowest rank.
        self.assertEqual(_owners(compiler, graphs), [0, 1, 2, 1, 2, 2])
        self.assertEqual(compiler._estimated_load, [10.0, 11.0, 12.0])

    def test_dropping_graphs_of_one_family_keeps_heavy_owners_spread(self):
        profile = _families({"forward": 18.0, "helper": 0.02})
        forwards = [f"forward:{bs}" for bs in range(4)]
        with_helpers = [g for bs in range(4) for g in (f"helper:{bs}", forwards[bs])]
        owners = dict(zip(with_helpers, _owners(_compiler(4, profile), with_helpers)))
        # Round-robin would put all four forwards on ranks 1 and 3.
        self.assertEqual(sorted(owners[graph] for graph in forwards), [0, 1, 2, 3])
        self.assertEqual(_owners(_compiler(4, profile), forwards), [0, 1, 2, 3])

    def test_invalid_costs_are_rejected(self):
        for cost in (-1.0, math.nan, math.inf):
            with self.subTest(cost=cost):
                with self.assertRaisesRegex(
                    ValueError, "Invalid shared TPU graph cost"
                ):
                    _compiler(2, _families({}, default=cost)).enqueue("g:1")

    def test_listed_graphs_follow_the_plan_whatever_the_enqueue_order(self):
        # Regression: warmup enqueues cheap graphs first, so a greedy owner per
        # enqueue stacked the late heavy forwards (DSV4.1 c4: 734 s on one rank
        # for 527 s of average tracing). Listed graphs take their planned owner.
        costs = {f"g{i}:x": c for i, c in enumerate([1, 1, 2, 2, 8, 9, 10, 11])}
        profile = shared_compile_cost.CostProfile(
            default_seconds=1.0, families={}, graphs=costs
        )
        ascending = sorted(costs, key=costs.get)
        owners = dict(zip(ascending, _owners(_compiler(4, profile), ascending)))
        loads = [0.0] * 4
        for graph, owner in owners.items():
            loads[owner] += costs[graph]
        self.assertEqual(max(loads), 11.0)  # each heavy graph on its own rank
        self.assertEqual(
            owners,
            dict(zip(ascending, _owners(_compiler(4, profile), ascending[::-1])[::-1])),
        )

    def test_unlisted_graphs_go_to_the_least_planned_load(self):
        profile = shared_compile_cost.CostProfile(
            default_seconds=1.0, families={"new": 3.0}, graphs={"a:1": 10.0, "b:1": 4.0}
        )
        compiler = _compiler(2, profile)
        self.assertEqual(_owners(compiler, ["new:1", "a:1", "new:2"]), [1, 0, 1])
        self.assertEqual(compiler._estimated_load, [10.0, 10.0])

    def test_plan_is_deterministic_and_breaks_ties_by_name(self):
        graphs = {"b:1": 5.0, "a:1": 5.0, "c:1": 5.0}
        profile = shared_compile_cost.CostProfile(
            default_seconds=0.0, families={}, graphs=graphs
        )
        self.assertEqual(profile.plan(2), ({"a:1": 0, "b:1": 1, "c:1": 0}, [10.0, 5.0]))
        self.assertEqual(profile.plan(2), profile.plan(2))


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
                "repeat_fraction": 1.0,
                "schema_version": 2,
                "default_seconds": 16.0,
                "families": {"helper_N": 0.02, "target/model.forward": 17.0},
                "graphs": {
                    "helper_8:[[[8]]]": 0.02,
                    "target/model.forward:[[[16]]]": 16.0,
                    "target/model.forward:[[[8]]]": 18.0,
                },
            },
        )
        self.path.write_text(json.dumps(profile))
        loaded = shared_compile_cost.load_cost_profile(self.path)
        self.assertEqual(loaded.estimate("target/model.forward:[[[8]]]"), 18.0)
        self.assertEqual(loaded.estimate("target/model.forward:[[[32]]]"), 17.0)
        self.assertEqual(loaded.estimate("helper_64:[[[64]]]"), 0.02)
        self.assertEqual(loaded.estimate("unlisted:[[[1]]]"), 16.0)

    def test_schema_1_profiles_still_load(self):
        self.path.write_text(
            json.dumps(
                {"schema_version": 1, "default_seconds": 2, "families": {"f": 5}}
            )
        )
        loaded = shared_compile_cost.load_cost_profile(self.path)
        self.assertEqual((loaded.estimate("f:1"), loaded.graphs), (5.0, {}))

    def test_profile_rejects_bad_schema_and_costs(self):
        self.path.write_text(json.dumps({"schema_version": 3}))
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
            self.assertEqual(shared_compile_cost.profile_from_env(), (None, ""))
        self.path.write_text(
            json.dumps({"schema_version": 1, "default_seconds": 2, "families": {}})
        )
        with patch.dict(os.environ, {env: str(self.path)}):
            profile, digest = shared_compile_cost.profile_from_env()
        self.assertEqual(profile.estimate("any:[[[1]]]"), 2.0)
        self.assertEqual(len(digest), 64)


class TestShapeAffinity(CustomTestCase):
    """Regression: a rank's second graph of one shape traced in ~13% of the
    first (cached kernel lowering), so spreading same-shape graphs across ranks
    wasted tracing (DSV4.1 c4: 615 s on one rank vs a 568 s plan)."""

    def test_affinity_key_drops_the_last_shape_field(self):
        key = shared_compile_cost.affinity_key
        self.assertEqual(key('f:["EXTEND",64,2048,"dec0"]'), 'f:["EXTEND",64,2048]')
        self.assertEqual(key('f:["EXTEND",64,2048,256]'), key('f:["EXTEND",64,2048,0]'))
        self.assertNotEqual(key('f:["V",4,24,null]'), key('f:["V",8,48,null]'))
        self.assertEqual(key("helper_8:[[[8]]]"), "helper_8:[[[8]]]")

    def test_cheap_repeats_keep_a_shape_on_one_rank(self):
        graphs = {f'f:["E",2048,{v}]': 10.0 for v in range(3)}
        graphs.update({f'f:["E",1024,{v}]': 6.0 for v in range(3)})
        profile = shared_compile_cost.CostProfile(
            default_seconds=1.0, families={}, graphs=graphs, repeat_fraction=0.1
        )
        owners, loads = profile.plan(2)
        self.assertEqual({owners[g] for g in graphs if "2048" in g}, {0})
        self.assertEqual({owners[g] for g in graphs if "1024" in g}, {1})
        self.assertEqual([round(x, 6) for x in loads], [12.0, 7.2])

    def test_without_learned_reuse_the_plan_is_largest_first(self):
        graphs = {f'f:["E",2048,{v}]': 10.0 for v in range(3)}
        profile = shared_compile_cost.CostProfile(
            default_seconds=1.0, families={}, graphs=graphs
        )
        owners, loads = profile.plan(3)
        self.assertEqual(sorted(owners.values()), [0, 1, 2])
        self.assertEqual(loads, [10.0, 10.0, 10.0])

    def test_profile_learns_the_repeat_fraction_per_boot(self):
        line = "[TPU] [shared_compile] owner={} trace={}s graph={}"
        boot = [
            line.format(0, 200.0, 'f:["E",1024,0]'),
            line.format(0, 26.0, 'f:["E",1024,128]'),  # same owner, same key: a repeat
            line.format(1, 240.0, 'f:["E",1024,256]'),  # another owner: first touch
        ]
        profile = shared_compile_cost.build_cost_profile(boot + boot)  # two boots
        self.assertEqual(profile["repeat_fraction"], 0.118)  # 26 / mean(200, 240)
        self.assertNotIn('f:["E",1024,128]', profile["graphs"])
        self.assertEqual(profile["graphs"]['f:["E",1024,0]'], 200.0)
        self.assertEqual(
            shared_compile_cost.build_cost_profile(boot[:1])["repeat_fraction"], 1.0
        )


class TestPackagedProfiles(CustomTestCase):
    def test_a_packaged_profile_loads_by_name_and_plans_every_graph(self):
        env = "SGLANG_TPU_SHARED_COMPILE_COST_PROFILE"
        names = sorted(p.name for p in shared_compile_cost.PACKAGED_DIR.glob("*.json"))
        self.assertIn("dsv41_flash_agentx_c4.json", names)
        for name in names:
            with self.subTest(name=name), patch.dict(os.environ, {env: name}):
                profile, digest = shared_compile_cost.profile_from_env()
                owners, loads = profile.plan(8)
                self.assertEqual(set(owners), set(profile.graphs))
                self.assertEqual(len(digest), 64)


class TestSharedCompileProfileAgreement(CustomTestCase):
    """Every rank plans from the same profile, or every rank falls back to
    round-robin owners; a profile problem never fails startup."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.path = self.directory / "profile.json"
        self.path.write_text(
            json.dumps({"schema_version": 1, "default_seconds": 2, "families": {}})
        )

    def _start(
        self,
        profile,
        peer=lambda record: record,
        *,
        world_size=2,
        device_count=None,
        **kwargs,
    ):
        """Rank 0 of two; ``peer`` maps rank 0's gathered record to rank 1's."""
        env = {
            "SGLANG_TPU_SHARED_COMPILE_COST_PROFILE": profile,
            "VERIFY_SPMD_SAFE_DUMP_DIR": str(self.directory),
        }
        gather = lambda compiler, record: [record, peer(record)]  # noqa: E731
        with (
            patch.dict(os.environ, env),
            patch.object(spmd_compiler.dist, "get_rank", return_value=0),
            patch.object(
                spmd_compiler.dist,
                "get_world_size",
                side_effect=lambda group=None: world_size if group is None else 2,
            ),
            patch.object(
                spmd_compiler.device_mesh,
                "get_device_assignment",
                return_value=list(
                    range(world_size if device_count is None else device_count)
                ),
            ),
            patch.object(SPMDCompiler, "gather", gather),
        ):
            compiler = SPMDCompiler(object(), **kwargs)
        if compiler._serializer is not None:
            compiler._serializer.shutdown()
        return compiler

    def test_ranks_sharing_a_profile_load_it(self):
        compiler = self._start(str(self.path))
        self.assertEqual(compiler._profile.estimate("any:[[[1]]]"), 2.0)

    def test_tp_subgroup_shares_within_the_native_session(self):
        compiler = self._start(str(self.path), world_size=4)
        self.assertEqual((compiler.size, compiler.world_size), (2, 4))
        self.assertEqual(_owners(compiler, ["g:1", "g:2", "g:3"]), [0, 1, 0])

    def test_native_session_must_match_the_distributed_world(self):
        with self.assertRaisesRegex(ValueError, "session and distributed world"):
            self._start(str(self.path), world_size=4, device_count=2)

    def test_a_rank_that_cannot_load_the_profile_falls_back_everywhere(self):
        # Rank 0's profile is missing while rank 1 loaded one: both use
        # round-robin owners instead of failing startup.
        def peer(record):
            runtime, error, _ = record
            return runtime, error, ("0" * 64, "")

        with self.assertLogs(spmd_compiler.logger, "WARNING") as logs:
            compiler = self._start(str(self.directory / "missing.json"), peer=peer)
        self.assertIsNone(compiler._profile)
        self.assertEqual(_owners(compiler, ["g:1", "g:2", "g:3"]), [0, 1, 0])
        self.assertIn("rank 0: FileNotFoundError", logs.output[0])

    def test_ranks_with_different_profiles_fall_back_everywhere(self):
        def peer(record):
            runtime, error, _ = record
            return runtime, error, ("0" * 64, "")

        with self.assertLogs(spmd_compiler.logger, "WARNING"):
            compiler = self._start(str(self.path), peer=peer)
        self.assertIsNone(compiler._profile)
        self.assertEqual(compiler._planned_owner, {})

    def test_an_invalid_profile_falls_back_instead_of_failing(self):
        self.path.write_text(json.dumps({"schema_version": 99}))
        compiler = self._start(str(self.path))
        self.assertIsNone(compiler._profile)

    def test_verification_does_not_load_the_profile(self):
        compiler = self._start(
            str(self.directory / "missing.json"), verify_spmd_safe=True
        )
        self.assertIsNone(compiler._profile)


if __name__ == "__main__":
    unittest.main()
