"""CPU coverage of coordinate discovery and rank-consistent launch plans."""

import argparse
import itertools
import json
import os
import socket
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.hardware_backend.tpu import node_normalization as normalization
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import enter_scope

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _cpu_node_rank(local_rank, config, host_index, timeout, writer):
    if local_rank == 0:
        writer.send(1)
    writer.close()


def _cpu_probe_failure(local_rank, config, host_index, timeout, writer):
    if local_rank == 0:
        raise RuntimeError("probe failed")
    time.sleep(60)


def _cpu_probe_timeout(local_rank, config, host_index, timeout, writer):
    time.sleep(60)


class TestNodeNormalization(unittest.TestCase):
    def setUp(self):
        self.config = normalization.TopologyConfig(
            ("source", "destination"), (2, 2, 2, 2)
        )

    def test_all_four_host_permutations_preserve_rank_endpoint_identity(self):
        for permutation in itertools.permutations(range(4)):
            with self.subTest(permutation=permutation):
                config = replace(
                    self.config,
                    hosts=tuple(f"host-{i}" for i in permutation),
                    topology=(4, 2, 2, 2),
                )
                order = normalization.canonical_host_order(list(permutation))
                ordered_config = replace(
                    config, hosts=tuple(config.hosts[index] for index in order)
                )
                for original_index, physical_rank in enumerate(permutation):
                    env = ordered_config.environment(order.index(original_index))
                    self.assertEqual(env["NODE_RANK"], str(physical_rank))
                    self.assertEqual(env["MASTER_ADDR"], "host-0")
                    self.assertEqual(
                        env["TPU_WORKER_HOSTNAMES"], "host-0,host-1,host-2,host-3"
                    )
                    self.assertEqual(
                        env["TORCH_TPU_SLICEBUILDER_ADDRESSES"].split(",")[
                            physical_rank * 8
                        ],
                        f"host-{physical_rank}:8070",
                    )

    def test_invalid_topology_placement_and_ports_fail_closed(self):
        for changes in (
            {"hosts": ()},
            {"hosts": ("a", "a")},
            {"topology": (2, 2, 3, 2)},
            {"topology": (2, 0, 2, 2)},
            {"topology": (3, 2, 2, 2)},
            {"chips_per_host": (0, 2, 1)},
            {"devices_per_host": 4},
            {"bootstrap_port": 8070},
            {"master_port": 65536},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config, **changes)
        for ranks in ([0, 0], [0, 2], [0, -1], [0, True], [0, None]):
            with self.subTest(ranks=ranks), self.assertRaises(ValueError):
                normalization.canonical_host_order(ranks)

    def test_coordinate_worker_sets_rank_metadata_before_native_discovery(self):
        distributed = ModuleType("torch_tpu._internal.distributed")
        distributed.tpu_distributed = SimpleNamespace(
            local_device_coords=Mock(return_value=[0, 0, 1])
        )
        wrapper = ModuleType(
            "torch_tpu._internal.distributed.launchers.multihost_wrapper"
        )
        wrapper.calculate_topological_node_rank = Mock(return_value=1)
        with (
            patch.dict(os.environ, RANK="stale", LOCAL_RANK="stale"),
            patch.dict(
                sys.modules,
                {distributed.__name__: distributed, wrapper.__name__: wrapper},
            ),
            patch("torch.distributed.init_process_group") as init,
            patch("torch.distributed.barrier"),
            patch("torch.distributed.destroy_process_group"),
            patch("sglang.srt.utils.kill_itself_when_parent_died"),
        ):
            for local_rank in (0, 1):
                writer = Mock()
                normalization._coordinate_worker(local_rank, self.config, 1, 10, writer)
                self.assertEqual(os.environ["RANK"], str(8 + local_rank))
                self.assertEqual(os.environ["LOCAL_RANK"], str(local_rank))
                self.assertEqual(os.environ["WORLD_SIZE"], "16")
                self.assertEqual(os.environ["MASTER_ADDR"], "source")
                if local_rank == 0:
                    writer.send.assert_called_once_with(1)
                else:
                    writer.send.assert_not_called()
            self.assertEqual(init.call_count, 2)
            wrapper.calculate_topological_node_rank.assert_called_once_with(
                [0, 0, 1], "2,2,2,2", fallback_index=None, chips_per_host=(2, 2, 1)
            )
            for coords in ([], [0, 1], [0, 0, 2], [-1, 0, 0], [True, 0, 0]):
                with self.subTest(coords=coords), self.assertRaises(ValueError):
                    distributed.tpu_distributed.local_device_coords.return_value = (
                        coords
                    )
                    normalization._coordinate_worker(0, self.config, 1, 10, Mock())

    def test_spawn_probe_returns_rank_and_cleans_up_on_failure(self):
        import torch.multiprocessing as mp

        spawn = mp.spawn
        contexts = []

        def record_spawn(*args, **kwargs):
            context = spawn(*args, **kwargs)
            contexts.append(context)
            return context

        # Two CPU workers test process lifetime, not TPU topology correctness.
        config = replace(
            self.config,
            topology=(1, 1, 2, 2),
            chips_per_host=(1, 1, 1),
            devices_per_host=2,
        )
        # Failures must keep the worker's traceback even though the worker
        # exits without running the native teardown.
        for worker, timeout, error, message in (
            (_cpu_node_rank, 30, None, None),
            (_cpu_probe_failure, 30, RuntimeError, "worker 0:(.|\n)*probe failed"),
            (_cpu_probe_timeout, 0.1, TimeoutError, "timed out"),
        ):
            with (
                self.subTest(worker=worker.__name__),
                patch.object(normalization, "_coordinate_worker", worker),
                patch.object(mp, "spawn", side_effect=record_spawn),
            ):
                if error is None:
                    self.assertEqual(
                        normalization.query_node_rank(config, 0, timeout),
                        1,
                    )
                else:
                    with self.assertRaisesRegex(error, message):
                        normalization.query_node_rank(config, 0, timeout)
                self.assertTrue(all(not p.is_alive() for p in contexts[-1].processes))

    def test_hosts_agree_before_launch_using_real_cpu_store(self):

        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        config = replace(
            self.config, hosts=("localhost", "127.0.0.1"), bootstrap_port=port
        )
        local = threading.local()

        def host(index):
            local.rank = 1 - index
            return normalization.discover_host_order(config, index, 10)

        with patch.object(
            normalization,
            "query_node_rank",
            side_effect=lambda config, index, timeout: local.rank,
        ):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(host, (0, 1)))
        for order in results:
            self.assertEqual(order, [1, 0])

    def test_discovery_errors_are_acknowledged_before_owner_releases_store(self):
        for changes, expected_error in (
            (
                {"error": "TimeoutError: TPU coordinate discovery timed out"},
                RuntimeError,
            ),
            ({"config": {}}, ValueError),
            ({"rank": 0}, ValueError),
        ):
            with self.subTest(changes=changes):
                records = [
                    {"config": asdict(self.config), "rank": index, "error": None}
                    for index in range(2)
                ]
                records[1].update(changes)
                store = Mock()
                store.get.side_effect = [json.dumps(record) for record in records]
                with (
                    patch("torch.distributed.TCPStore", return_value=store),
                    patch.object(normalization, "query_node_rank", return_value=0),
                    self.assertRaises(expected_error),
                ):
                    normalization.discover_host_order(self.config, 0, 10)
                store.set.assert_any_call("done/0", "1")
                store.wait.assert_called_once_with(["done/0", "done/1"])


class TestAutomaticTopologyLaunch(unittest.TestCase):
    def setUp(self):
        from sglang.srt.server_args import ServerArgs

        self.server_args = ServerArgs(
            model_path="dummy",
            device="tpu",
            nnodes=2,
            tp_size=16,
            dist_init_addr="source:29500",
            nccl_port=29600,
        )
        enter_scope(
            self,
            patch.dict(
                os.environ,
                {
                    "TPU_WORKER_HOSTNAMES": "source,destination",
                    "TORCH_TPU_TOPOLOGY": "2,2,2,2",
                    "TORCH_TPU_DEVICES_PER_HOST": "8",
                    "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1",
                    "TORCH_TPU_BASE_PORT": "8070",
                    "SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE": "",
                },
            ),
        )
        os.environ.pop("SGLANG_TPU_TOPOLOGY_BOOTSTRAP_PORT", None)
        self.resolve = enter_scope(
            self,
            patch.object(
                normalization,
                "discover_host_order",
                return_value=[1, 0],
            ),
        )

    def test_automatic_launch_declares_ranks_once_and_dp_ports_follow(self):
        from sglang.srt.arg_groups.overrides import resolving_view
        from sglang.srt.server_args import PortArgs

        # Attention DP is declared by width now; the deprecated flag alone
        # resolves to attn_dp_size == dp_size == 1, i.e. no DP attention.
        self.server_args.attn_dp_size = 2
        normalization.prepare_tpu_environment(self.server_args)
        cfg = resolving_view(self.server_args)
        self.assertEqual(cfg.node_rank, 1)
        self.assertEqual(cfg.dist_init_addr, "destination:29500")
        self.assertEqual(cfg.tpu_worker_hostnames, "destination,source")
        self.assertEqual(cfg.torch_tpu_topology, "2,2,2,2")
        self.assertEqual(
            cfg.torch_tpu_slicebuilder_addresses.split(",")[0], "destination:8070"
        )
        # Preserve the requested inputs; publish/worker config uses declarations.
        self.assertEqual(self.server_args.node_rank, 0)
        self.assertEqual(self.server_args.dist_init_addr, "source:29500")
        self.assertEqual(os.environ["NODE_RANK"], "1")
        self.assertEqual(os.environ["MASTER_ADDR"], "destination")
        self.assertEqual(self.resolve.call_args.args[0].bootstrap_port, 29501)
        self.resolve.assert_called_once()
        with patch("sglang.srt.server_args.wait_port_available"):
            ports = PortArgs.init_new(self.server_args)
        self.assertEqual(ports.rpc_ipc_name, "tcp://destination:29503")
        self.assertEqual(ports.tokenizer_ipc_name, "tcp://destination:29501")
        # A later preparation of an already-normalized record is a no-op for
        # declarations, including when that record is currently published.
        self.resolve.return_value = [0, 1]
        with patch("sglang.srt.arg_groups.overrides.declare_resolution") as declare:
            normalization.prepare_tpu_environment(self.server_args)
        declare.assert_not_called()

    def test_cli_flag_defaults_on_and_opt_out_preserves_existing_inputs(self):
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        for flags, expected in (
            ([], True),
            (["--tpu-node-normalization"], True),
            (["--no-tpu-node-normalization"], False),
        ):
            with self.subTest(flags=flags):
                args = ServerArgs.from_cli_args(
                    parser.parse_args(["--model", "dummy", *flags])
                )
                self.assertIs(args.tpu_node_normalization, expected)
        self.server_args.tpu_node_normalization = False
        # Opt-out must also leave previously supported Ray/external launches alone.
        self.server_args.use_ray = True
        before = dict(os.environ)
        normalization.prepare_tpu_environment(self.server_args)
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(self.server_args.node_rank, 0)
        self.assertEqual(self.server_args.dist_init_addr, "source:29500")
        self.resolve.assert_not_called()

    def test_single_host_and_other_devices_never_probe(self):
        for device, nnodes in (("tpu", 1), ("cuda", 2), ("cpu", 2)):
            with self.subTest(device=device, nnodes=nnodes):
                self.server_args.device = device
                self.server_args.nnodes = nnodes
                before = dict(os.environ)
                normalization.prepare_tpu_environment(self.server_args)
                self.assertEqual(dict(os.environ), before)
        self.resolve.assert_not_called()

    def test_unsupported_launches_fail_before_discovery(self):
        for field, value in (
            ("use_ray", True),
            ("nnodes", 4),
            ("tp_size", 8),
            ("node_rank", 2),
            ("tpu_worker_hostnames", "only-one"),
            ("torch_tpu_topology", "2,2,1,2"),
            ("torch_tpu_slicebuilder_addresses", "elsewhere:8070"),
            ("dist_init_addr", None),
            ("dist_init_addr", "source"),
        ):
            with (
                self.subTest(field=field),
                patch.object(self.server_args, field, value),
            ):
                with self.assertRaises(ValueError):
                    normalization.prepare_tpu_environment(self.server_args)
        with (
            patch.dict(os.environ, SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE="env://"),
            self.assertRaises(ValueError),
        ):
            normalization.prepare_tpu_environment(self.server_args)
        self.resolve.assert_not_called()

    def test_existing_scheduler_launcher_receives_normalized_ranks(self):
        # Opt-out is covered by test_cli_flag_defaults_on_and_opt_out_...: it
        # leaves the launch inputs untouched before the Engine sees them.
        from sglang.srt.entrypoints import engine

        class ReachedSchedulerLaunch(Exception):
            pass

        args = type(self.server_args)(
            model_path="dummy",
            device="tpu",
            nnodes=2,
            tp_size=16,
            dist_init_addr="source:29500",
            tpu_node_normalization=True,
        )

        def launch(*_args, **_kwargs):
            self.assertEqual(engine.get_parallel().node_rank, 1)
            self.assertEqual(engine.get_parallel().dist_init_addr, "destination:29500")
            raise ReachedSchedulerLaunch

        previous = engine.snapshot_context()
        try:
            with (
                patch.dict(os.environ),
                patch.object(type(args), "resolve_once"),
                patch.object(type(args), "check_server_args"),
                patch.object(engine, "configure_logger"),
                patch.object(engine, "_set_envs_and_config"),
                patch.object(engine, "load_plugins"),
                patch.object(engine.PortArgs, "init_new"),
                patch.object(
                    engine.Engine, "_launch_scheduler_processes", side_effect=launch
                ) as existing_launcher,
                self.assertRaises(ReachedSchedulerLaunch),
            ):
                engine.Engine._launch_subprocesses(args, None, None, None)
            existing_launcher.assert_called_once()
            self.resolve.assert_called_once()
        finally:
            engine.restore_context(previous)


if __name__ == "__main__":
    unittest.main()
