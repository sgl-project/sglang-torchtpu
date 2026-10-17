"""Exercise graph and physical-layout rejection through the public recipe runner."""

import json
import os
import shlex
import sys
import tempfile
import traceback
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
from torch_tpu._internal.compile import tpu_torch_compile as tc
from torch_tpu._internal.device_utils.annotations import LayoutContext, TpuLayout
from torch_tpu._internal.distributed import tpu_distributed  # noqa: F401
from torch_tpu._internal.utils import hardware

from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.compilation.direct_compile import direct_compile
from sglang_tpu.compilation.warmup_queue import collect_warmups
from sglang_tpu.testing.spmd_safe import run_recipes, verify_startup

register_tpu_ci(
    est_time=180,
    suite="stage-a-test-tpu-unit",
    chips=4,
    labels=["singlehost", "infra", "compile-tracer", "tp"],
)

_KEY = 'test/model.forward:["TARGET_VERIFY","NULL",1,8,false]'


class TestSPMDSafe(CustomTestCase):
    def test_verification_accepts_matching_graphs_and_rejects_differences(self):
        world_size = hardware.get_tpu_device_count()
        self.assertGreaterEqual(world_size, 2)
        python = shlex.quote(sys.executable)
        worker = shlex.quote(str(Path(__file__).resolve()))
        with tempfile.TemporaryDirectory() as temporary:
            for mode in ("matching", "mlir", "layout"):
                with self.subTest(mode=mode):
                    output = Path(temporary) / mode

                    def launch_recipe(recipe, directory):
                        command = (
                            "unset RANK LOCAL_RANK LOCAL_WORLD_SIZE WORLD_SIZE; "
                            f"eval $({python} -m torch.tpu.distributed); "
                            "export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE; "
                            f'{python} -m torch.distributed.run --standalone --nproc_per_node="$WORLD_SIZE" '
                            f"{worker} --spmd-safe-worker {shlex.quote(recipe)}"
                        )
                        verify_startup(
                            ["bash", "-lc", command],
                            world_size=world_size,
                            local_world_size=world_size,
                            timeout=600,
                            dump_dir=directory,
                            env={
                                "JAX_PLATFORMS": "cpu",
                                "XLA_FLAGS": f"--xla_force_host_platform_device_count={world_size}",
                            },
                        )

                    status = run_recipes([mode], launch_recipe, output=output)
                    self.assertEqual(status, int(mode != "matching"))
                    workers = [
                        json.loads(path.read_text())
                        for path in output.glob("*/spmd-*/worker-*.json")
                    ]
                    self.assertEqual(
                        sorted(w["rank"] for w in workers), list(range(world_size))
                    )
                    for record in workers:
                        self.assertEqual(record["keys"], [_KEY])
                        self.assertEqual(record["passed"], mode == "matching")
                        self.assertEqual(
                            set(record["differences"]),
                            {_KEY} if mode == "mlir" else set(),
                        )
                        self.assertEqual(
                            set(record["layout_differences"]),
                            {_KEY} if mode == "layout" else set(),
                        )
                    if mode == "layout":
                        self.assertEqual(
                            len({w["mlir_hashes"][_KEY] for w in workers}), 1
                        )
                        self.assertEqual(
                            workers[0]["layout_differences"][_KEY][1], [[0, 1]]
                        )
                    result_files = list(output.glob("*/spmd-*/result.json"))
                    self.assertEqual(len(result_files), 1)
                    result = json.loads(result_files[0].read_text())
                    self.assertEqual(result["passed"], mode == "matching")
                    self.assertEqual(
                        result["launcher_exit_code"], int(mode != "matching")
                    )
                    self.assertFalse(
                        any(
                            error.startswith("Verification startup exceeded")
                            for error in result["errors"]
                        )
                    )
                    if mode != "matching":
                        self.assertIn(json.dumps(_KEY), " ".join(result["errors"]))


def _run_worker(mode: str) -> None:
    dist.init_process_group(backend="tpu_dist")
    try:
        group = dist.new_group(backend="gloo")
        rank = dist.get_rank()
        order = [0, 1] if mode == "layout" and rank == 1 else [1, 0]
        with LayoutContext(TpuLayout(minor_to_major=order)):
            value = torch.ones((128, 128)).to("tpu")
        observed = tc.get_device_layout_if_materialized(value)
        assert observed is not None and observed[0] == order, observed
        constant = 2 if mode == "mlir" and rank == 1 else 1

        def forward(value):
            return value + constant

        compiled = direct_compile(
            forward,
            specialization_key=lambda value: ("TARGET_VERIFY", "NULL", 1, 8, False),
            name="test/model.forward",
            spmd_safe=True,
            lift_rank=False,
        )
        with get_context().override_server_args(use_spmd_compile=True):
            with collect_warmups(group=group) as queue:
                queue.calls.append(compiled.warmup_async(value))
        raise AssertionError("Verification must exit before warmup execution")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if "--spmd-safe-worker" in sys.argv:
        status = 1
        try:
            _run_worker(sys.argv[-1])
        except SystemExit as error:
            status = 0 if error.code in (None, 0) else 1
        except BaseException:
            traceback.print_exc()
        finally:
            # As in test_collective_compile_cache, skip libtpu's atexit
            # StopSession: it can wedge the next test case's TPU initialization.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(status)
    else:
        unittest.main()
