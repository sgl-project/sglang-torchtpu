"""Both input hostname orders must produce the same physical TPU launch plan.

Run on every host of a dedicated two-host slice. This exercises real coordinate
discovery and startup normalization, not compilation or artifact portability.
"""

import json
import os
import unittest
from dataclasses import replace
from datetime import timedelta
from unittest.mock import patch

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=120,
    suite="tpu-integration",
    nightly=True,
    labels=["multihost", "infra", "normalization"],
)


class TestNodeNormalization(unittest.TestCase):
    def test_reversed_hostnames_produce_the_same_launch_plan(self):
        import torch.distributed as dist

        from sglang.srt.arg_groups.overrides import resolving_view
        from sglang.srt.server_args import ServerArgs
        from sglang_tpu import node_normalization as normalization

        hosts = tuple(
            filter(None, os.environ.get("TPU_WORKER_HOSTNAMES", "").split(","))
        )
        self.assertEqual(len(hosts), 2, "requires a dedicated two-host TPU slice")
        host_index = int(os.environ["TPU_WORKER_ID"])
        self.assertIn(host_index, range(len(hosts)))
        hostname = hosts[host_index]
        config = normalization.TopologyConfig(
            hosts,
            normalization.dimensions(os.environ["TORCH_TPU_TOPOLOGY"]),
            devices_per_host=int(os.environ.get("TORCH_TPU_DEVICES_PER_HOST", "8")),
        )
        # Keep the control store independent of either input order's leader.
        store = dist.TCPStore(
            hosts[0],
            29591,
            len(hosts),
            host_index == 0,
            timeout=timedelta(seconds=360),
        )
        plans = []
        for phase, input_hosts in enumerate((hosts, hosts[::-1])):
            record = {"plan": None, "error": None}
            try:
                input_index = input_hosts.index(hostname)
                input_config = replace(config, hosts=input_hosts)
                args = ServerArgs(
                    model_path="dummy",
                    device="tpu",
                    nnodes=len(hosts),
                    node_rank=input_index,
                    tp_size=len(hosts) * config.devices_per_host,
                    dist_init_addr=f"{input_hosts[0]}:{config.master_port}",
                    tpu_node_normalization=True,
                )
                with patch.dict(os.environ, input_config.environment(input_index)):
                    # Do not mock discovery: this is the same preparation step
                    # that Engine startup calls.
                    normalization.prepare_tpu_environment(args)
                    cfg = resolving_view(args)
                    ordered_hosts = tuple(cfg.tpu_worker_hostnames.split(","))
                    self.assertCountEqual(ordered_hosts, hosts)
                    self.assertEqual(cfg.node_rank, ordered_hosts.index(hostname))
                    self.assertEqual(
                        cfg.dist_init_addr, f"{ordered_hosts[0]}:{config.master_port}"
                    )
                    normalized = replace(config, hosts=ordered_hosts)
                    self.assertEqual(
                        cfg.torch_tpu_slicebuilder_addresses,
                        normalized.slicebuilder_addresses,
                    )
                    environment = normalized.environment(cfg.node_rank)
                    self.assertEqual(
                        {key: os.environ[key] for key in environment}, environment
                    )
                    record["plan"] = {
                        "host": hostname,
                        "node_rank": cfg.node_rank,
                        "dist_init_addr": cfg.dist_init_addr,
                        "environment": environment,
                    }
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            store.set(f"{phase}/result/{host_index}", json.dumps(record))
            records = [
                json.loads(store.get(f"{phase}/result/{index}"))
                for index in range(len(hosts))
            ]
            # A failed assertion must not close the leader's store before peers
            # have read the same failure; acknowledge both success and failure.
            store.set(f"{phase}/done/{host_index}", "1")
            if host_index == 0:
                store.wait([f"{phase}/done/{index}" for index in range(len(hosts))])
            self.assertEqual([r["error"] for r in records], [None] * len(hosts))
            plan = [r["plan"] for r in records]
            self.assertCountEqual([p["node_rank"] for p in plan], range(len(hosts)))
            self.assertEqual(
                plan[0]["environment"]["TPU_WORKER_HOSTNAMES"],
                plan[1]["environment"]["TPU_WORKER_HOSTNAMES"],
            )
            print(json.dumps({"input_hosts": input_hosts, "plan": plan}), flush=True)
            plans.append(plan)
        self.assertEqual(plans[0], plans[1])


if __name__ == "__main__":
    unittest.main()
