"""Regression tests for the Ray TPU multihost launch path.

Runs anywhere (no TPU / no live Ray cluster required) -- exercises the pure,
CPU-side logic that the Ray engine relies on for correct TPU launch:

* ``_resolve_bundle_order_from_probes`` -- maps Ray placement-group bundles to
  physical ``TPU_WORKER_ID`` order. Regression coverage for the bug where
  ``bundle_for_node`` used engine-first + numeric bundle order while the
  slicebuilder endpoints / ``MASTER_ADDR`` assume physical ``TPU_WORKER_ID``
  order -- silently misplacing ranks for nnodes >= 3 or when the Engine is not
  on TPU worker 0.
* ``tpu_actor_runtime_env`` / ``compute_single_host_tpu_env`` -- the launcher
  stashes the TPU env on ``server_args`` and injects it into each actor via
  ``runtime_env``, because Ray does NOT propagate the driver's ``os.environ``
  mutations to actors.
* ``_actor_options`` -- shared actor options; GPU must stay byte-identical to
  the pre-refactor dict and never touch the TPU package.
* ``detect_tpu_nnodes`` -- auto-detect node count from the live Ray cluster.

Usage:
    python -m pytest test/registered/tpu/test_ray_bundle_order.py -v
    python test/registered/tpu/test_ray_bundle_order.py
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import pytest

# ray is an optional extra (project.optional-dependencies "ray"), but the sglang
# imports below pull in ray_utils -> `import ray`. Skip the whole module when the
# extra is absent so collection doesn't fail with ModuleNotFoundError; the
# stage-a-test-tpu-unit runner image installs sglang[ray] so it actually runs.
pytest.importorskip("ray")

from sglang.srt.hardware_backend.tpu.patches import compute_single_host_tpu_env
from sglang.srt.hardware_backend.tpu.ray_utils import (
    _resolve_bundle_order_from_probes,
    tpu_actor_runtime_env,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(
    est_time=30,
    suite="stage-a-test-tpu-unit",
)


def _server_args(**overrides):
    """Minimal server_args stand-in exposing the fields the launch helpers read."""
    fields = dict(
        device="cuda",
        tpu_worker_hostnames=None,
        torch_tpu_slicebuilder_addresses=None,
        torch_tpu_topology=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestResolveBundleOrderFromProbes(CustomTestCase):
    def test_nnodes2_bundles_already_in_worker_order(self):
        # bundle 0 -> worker 0 (engine), bundle 1 -> worker 1
        probes = [("0", "10.0.0.0"), ("1", "10.0.0.1")]
        bundle_for_node, rank0_ip = _resolve_bundle_order_from_probes(
            probes, engine_ip="10.0.0.0"
        )
        self.assertEqual(bundle_for_node, [0, 1])
        self.assertEqual(rank0_ip, "10.0.0.0")

    def test_nnodes2_bundles_reversed_vs_worker_order(self):
        # Ray happened to assign bundle 0 -> worker 1, bundle 1 -> worker 0.
        # Node k must still map to the bundle on physical worker k.
        probes = [("1", "10.0.0.1"), ("0", "10.0.0.0")]
        bundle_for_node, rank0_ip = _resolve_bundle_order_from_probes(
            probes, engine_ip="10.0.0.0"
        )
        # node 0 -> worker 0 -> bundle 1 ; node 1 -> worker 1 -> bundle 0
        self.assertEqual(bundle_for_node, [1, 0])
        self.assertEqual(rank0_ip, "10.0.0.0")

    def test_nnodes3_bundle_numeric_order_differs_from_worker_order(self):
        # The core nnodes>=3 regression: bundle index order != TPU_WORKER_ID order.
        # bundle 0 -> worker 2, bundle 1 -> worker 0 (engine), bundle 2 -> worker 1
        probes = [
            ("2", "10.0.0.2"),
            ("0", "10.0.0.0"),
            ("1", "10.0.0.1"),
        ]
        bundle_for_node, rank0_ip = _resolve_bundle_order_from_probes(
            probes, engine_ip="10.0.0.0"
        )
        # node 0 -> worker 0 -> bundle 1
        # node 1 -> worker 1 -> bundle 2
        # node 2 -> worker 2 -> bundle 0
        self.assertEqual(bundle_for_node, [1, 2, 0])
        self.assertEqual(rank0_ip, "10.0.0.0")

    def test_nnodes4_arbitrary_permutation(self):
        # 4-node slice with a fully scrambled bundle->worker mapping.
        # bundle idx:   0      1      2      3
        # worker id:    3      0      2      1
        probes = [
            ("3", "ip3"),
            ("0", "ip0"),
            ("2", "ip2"),
            ("1", "ip1"),
        ]
        bundle_for_node, rank0_ip = _resolve_bundle_order_from_probes(
            probes, engine_ip="ip0"
        )
        # node k -> bundle whose worker id == k
        # w0->bundle1, w1->bundle3, w2->bundle2, w3->bundle0
        self.assertEqual(bundle_for_node, [1, 3, 2, 0])
        self.assertEqual(rank0_ip, "ip0")

    def test_worker_ids_ordered_numerically_not_lexically(self):
        # A 10-node slice: worker id "10" must be treated as int 10 (after "9"),
        # not sorted lexically before "2".
        probes = [(str(i), f"ip{i}") for i in range(10)]
        # scramble bundle order so the mapping is non-trivial
        probes = probes[::-1]  # bundle 0 -> worker 9, ..., bundle 9 -> worker 0
        bundle_for_node, rank0_ip = _resolve_bundle_order_from_probes(
            probes, engine_ip="ip0"
        )
        # node k -> bundle (9 - k)
        self.assertEqual(bundle_for_node, [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        self.assertEqual(rank0_ip, "ip0")

    def test_raises_when_engine_not_on_worker0(self):
        # Engine (rank0 scheduler) lives on worker 1, but MASTER_ADDR would point
        # at worker 0 -> must fail loudly instead of hanging at rendezvous.
        probes = [("0", "10.0.0.0"), ("1", "10.0.0.1")]
        with self.assertRaises(RuntimeError) as ctx:
            _resolve_bundle_order_from_probes(probes, engine_ip="10.0.0.1")
        self.assertIn("worker 0", str(ctx.exception))

    def test_raises_when_worker_id_missing(self):
        probes = [("0", "10.0.0.0"), (None, "10.0.0.1")]
        with self.assertRaises(RuntimeError) as ctx:
            _resolve_bundle_order_from_probes(probes, engine_ip="10.0.0.0")
        self.assertIn("TPU_WORKER_ID missing", str(ctx.exception))

    def test_raises_on_duplicate_worker_id(self):
        probes = [("0", "10.0.0.0"), ("0", "10.0.0.9")]
        with self.assertRaises(RuntimeError) as ctx:
            _resolve_bundle_order_from_probes(probes, engine_ip="10.0.0.0")
        self.assertIn("duplicate", str(ctx.exception))

    def test_raises_on_noncontiguous_worker_ids(self):
        # worker ids {0, 2} for a 2-node slice -- something is wrong with discovery.
        probes = [("0", "10.0.0.0"), ("2", "10.0.0.2")]
        with self.assertRaises(RuntimeError) as ctx:
            _resolve_bundle_order_from_probes(probes, engine_ip="10.0.0.0")
        self.assertIn("contiguous", str(ctx.exception))

    def test_single_node(self):
        probes = [("0", "10.0.0.0")]
        bundle_for_node, rank0_ip = _resolve_bundle_order_from_probes(
            probes, engine_ip="10.0.0.0"
        )
        self.assertEqual(bundle_for_node, [0])
        self.assertEqual(rank0_ip, "10.0.0.0")


class TestTpuActorRuntimeEnv(CustomTestCase):
    """The launcher stashes the TPU env on server_args; tpu_actor_runtime_env maps
    those fields to the env vars injected into each actor via runtime_env. Ray does
    NOT propagate the driver's os.environ mutations, so this channel is the fix for
    the single-host slicebuilder-port mismatch (each actor otherwise picks its own).
    """

    def test_empty_when_no_tpu_fields(self):
        # GPU / non-Ray: nothing to inject -> no runtime_env key at all.
        self.assertEqual(tpu_actor_runtime_env(_server_args()), {})

    def test_multihost_injects_worker_hostnames(self):
        env = tpu_actor_runtime_env(_server_args(tpu_worker_hostnames="ipA,ipB"))
        self.assertEqual(
            env,
            {"runtime_env": {"env_vars": {"TPU_WORKER_HOSTNAMES": "ipA,ipB"}}},
        )

    def test_single_host_injects_empty_worker_hostnames(self):
        # nnodes<=1 on a multi-host slice: "" must be injected (not skipped as
        # falsy) so the actor overrides the slice-wide TPU_WORKER_HOSTNAMES it
        # inherits from `ray start` -- otherwise it takes the multihost branch
        # and hangs. None (GPU / discovery failed) stays uninjected.
        env = tpu_actor_runtime_env(_server_args(tpu_worker_hostnames=""))
        self.assertEqual(env["runtime_env"]["env_vars"]["TPU_WORKER_HOSTNAMES"], "")

    def test_single_host_injects_slicebuilder_and_topology(self):
        env = tpu_actor_runtime_env(
            _server_args(
                torch_tpu_slicebuilder_addresses="localhost:1,localhost:2",
                torch_tpu_topology="2,1,1",
            )
        )
        self.assertEqual(
            env["runtime_env"]["env_vars"],
            {
                "TORCH_TPU_SLICEBUILDER_ADDRESSES": "localhost:1,localhost:2",
                "TORCH_TPU_TOPOLOGY": "2,1,1",
            },
        )

    def test_single_host_topology_optional(self):
        # torch_tpu_topology may be unknown (None) -- only addresses get injected.
        env = tpu_actor_runtime_env(
            _server_args(torch_tpu_slicebuilder_addresses="localhost:1")
        )
        self.assertEqual(
            env["runtime_env"]["env_vars"],
            {"TORCH_TPU_SLICEBUILDER_ADDRESSES": "localhost:1"},
        )


class TestComputeSingleHostTpuEnv(CustomTestCase):
    """Single-host slicebuilder env is computed ONCE in the driver (random free
    ports) and shared with all actors, so all ranks agree on the endpoint list.
    """

    def test_one_endpoint_per_rank(self):
        env = compute_single_host_tpu_env(4)
        addrs = env["TORCH_TPU_SLICEBUILDER_ADDRESSES"].split(",")
        self.assertEqual(len(addrs), 4)
        self.assertTrue(all(a.startswith("localhost:") for a in addrs))

    def test_world_size_one_topology(self):
        env = compute_single_host_tpu_env(1)
        self.assertEqual(len(env["TORCH_TPU_SLICEBUILDER_ADDRESSES"].split(",")), 1)
        self.assertEqual(env["TORCH_TPU_TOPOLOGY"], "1,1,1")

    def test_ports_distinct(self):
        env = compute_single_host_tpu_env(8)
        ports = [
            a.split(":")[1] for a in env["TORCH_TPU_SLICEBUILDER_ADDRESSES"].split(",")
        ]
        self.assertEqual(len(set(ports)), len(ports))


class TestActorOptions(CustomTestCase):
    """_actor_options composes num_cpus + accelerator reservation, adding the TPU
    runtime_env only on device=="tpu". The GPU path must stay byte-identical to the
    pre-refactor {"num_cpus": 0, "num_gpus": 1} and must never import the TPU
    package (import is lazy + gated on device).

    The injected runtime_env only *delivers* the launcher-computed TPU env vars
    into the actor's os.environ; they are actually *applied* later inside the
    actor -- monkey_patch_torch_tpu_env / _prepare_tpu_env run after
    apply_tpu_patches and read them straight from os.environ.
    """

    def test_gpu_options_byte_identical(self):
        from sglang.srt.ray.engine import _actor_options

        self.assertEqual(
            _actor_options(_server_args(device="cuda")),
            {"num_cpus": 0, "num_gpus": 1},
        )

    def test_tpu_options_reserve_tpu_resource_no_env(self):
        from sglang.srt.ray.engine import _actor_options

        # No TPU env stashed -> tpu_actor_runtime_env is empty, so no runtime_env.
        self.assertEqual(
            _actor_options(_server_args(device="tpu")),
            {"num_cpus": 0, "num_gpus": 0, "resources": {"TPU": 1}},
        )

    def test_tpu_options_merge_runtime_env(self):
        from sglang.srt.ray.engine import _actor_options

        opts = _actor_options(
            _server_args(device="tpu", tpu_worker_hostnames="ipA,ipB")
        )
        self.assertEqual(opts["num_cpus"], 0)
        self.assertEqual(opts["resources"], {"TPU": 1})
        self.assertEqual(
            opts["runtime_env"]["env_vars"],
            {"TPU_WORKER_HOSTNAMES": "ipA,ipB"},
        )


class TestDetectTpuNnodes(CustomTestCase):
    """detect_tpu_nnodes counts alive Ray nodes advertising the TPU resource; any
    error falls back to 0 so the caller keeps server_args.nnodes.
    """

    def test_counts_only_alive_tpu_nodes(self):
        from sglang.srt.hardware_backend.tpu import ray_utils

        nodes = [
            {"Alive": True, "Resources": {"TPU": 4}},
            {"Alive": True, "Resources": {"TPU": 4}},
            {"Alive": False, "Resources": {"TPU": 4}},  # dead -> skip
            {"Alive": True, "Resources": {"CPU": 8}},  # no TPU -> skip
        ]
        with mock.patch.object(ray_utils.ray, "nodes", return_value=nodes):
            self.assertEqual(ray_utils.detect_tpu_nnodes(), 2)

    def test_returns_zero_on_error(self):
        from sglang.srt.hardware_backend.tpu import ray_utils

        with mock.patch.object(
            ray_utils.ray, "nodes", side_effect=RuntimeError("boom")
        ):
            self.assertEqual(ray_utils.detect_tpu_nnodes(), 0)


if __name__ == "__main__":
    unittest.main()
