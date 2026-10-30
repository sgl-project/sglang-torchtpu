# SPDX-License-Identifier: Apache-2.0

"""Fresh-process persistence test for exported Pallas programs."""

import os
import shlex
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=85, suite="stage-a-test-tpu-unit", chips=4)

_CACHE_LOGGING = (
    "compilation_cache=2,tier2_compilation_cache=2,tier3_compilation_cache=2"
)
_DISTRIBUTED_ENV_KEYS = {
    "TORCH_TPU_TOPOLOGY",
    "TORCH_TPU_SLICEBUILDER_ADDRESSES",
    "WORLD_SIZE",
}
_WORKER = textwrap.dedent("""
    import sys

    import torch
    import torch.distributed as dist
    from torch_tpu._internal.distributed import tpu_distributed  # noqa: F401

    from sglang_tpu.compilation.direct_compile import direct_compile

    torch.manual_seed(0)
    torch.set_num_threads(4)
    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()

    model = torch.nn.Linear(128, 128, bias=True).eval()
    with torch.no_grad():
        model.weight.copy_(torch.eye(128))
        model.bias.zero_()
    model = model.to("tpu")
    torch.accelerator.synchronize()

    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    def add_kernel(value_ref, output_ref):
        output_ref[...] = value_ref[...] + KERNEL_INCREMENT

    def linear(
        value: jax.Array, weight: jax.Array, bias: jax.Array
    ) -> jax.Array:
        result = jnp.maximum(value @ weight.T + bias, 0)
        return pl.pallas_call(
            add_kernel,
            out_shape=jax.ShapeDtypeStruct(result.shape, result.dtype),
        )(result)

    pallas_op = cpu_export_jax_op("pallas_tier3_test::linear", linear)

    def forward(value):
        return pallas_op(value, model.weight, model.bias)

    compiled = direct_compile(
        forward,
        specialization_key=lambda value: tuple(value.shape),
        model=model,
    )
    value = torch.ones((32, 128), dtype=torch.float32, device="tpu")
    torch.accelerator.synchronize()
    result = compiled(value).cpu()
    expected = torch.full((32, 128), 1 + int(sys.argv[2]), dtype=torch.float32)
    if not torch.equal(result, expected):
        raise AssertionError(f"rank {rank} produced {result}, expected {expected}")

    print(f"PALLAS_TIER3_CASE_OK stage={sys.argv[1]} rank={rank}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    """)


def _singlehost_environment() -> dict[str, str]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.tpu.distributed",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"single-host setup failed:\n{completed.stderr}")

    environment = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in _DISTRIBUTED_ENV_KEYS:
            environment[key] = shlex.split(value)[0]
    if environment.keys() != _DISTRIBUTED_ENV_KEYS:
        raise RuntimeError(f"incomplete single-host environment: {environment}")
    return environment


def _cache_files(root: Path) -> dict[str, int]:
    return {
        str(path.relative_to(root)): path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
    }


class TestPallasTier3Cache(CustomTestCase):
    def test_fresh_process_reuses_cache_and_invalidates_changed_kernel(self):
        distributed_environment = _singlehost_environment()
        world_size = int(distributed_environment["WORLD_SIZE"])
        tier2_root = Path("/dev/shm/torch_tpu_cache")
        tier2_root.mkdir(parents=True, exist_ok=True)

        with (
            tempfile.TemporaryDirectory(prefix="pallas-tier3-") as work_directory,
            tempfile.TemporaryDirectory(
                prefix="pallas-tier3-", dir=tier2_root
            ) as tier2_directory,
        ):
            work_root = Path(work_directory)
            tier3_root = work_root / "cache"
            tier3_root.mkdir()
            outputs = {}
            cache_snapshots = {}
            for stage, increment in (
                ("seed", 1),
                ("restore", 1),
                ("changed", 2),
                ("changed_restore", 2),
            ):
                environment = os.environ.copy()
                environment.update(distributed_environment)
                environment.update(
                    {
                        "OMP_NUM_THREADS": "4",
                        "PYTHONHASHSEED": "0",
                        "TF_CPP_VMODULE": _CACHE_LOGGING,
                        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK": "1",
                        "TORCH_TPU_TIER2_COMPILATION_CACHE": f"{Path(tier2_directory).name}/{stage}",
                        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": str(tier3_root),
                    }
                )
                log_directory = work_root / f"{stage}-logs"
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--no_python",
                        "--nproc_per_node",
                        str(world_size),
                        "--log_dir",
                        str(log_directory),
                        "--redirects",
                        "3",
                        "--tee",
                        "3",
                        sys.executable,
                        "-c",
                        _WORKER.replace("KERNEL_INCREMENT", str(increment)),
                        stage,
                        str(increment),
                    ],
                    capture_output=True,
                    check=False,
                    text=True,
                    env=environment,
                    timeout=120,
                )
                worker_output = "".join(
                    path.read_text() for path in sorted(log_directory.rglob("*.log"))
                )
                launcher_output = completed.stdout + completed.stderr
                self.assertEqual(
                    completed.returncode,
                    0,
                    f"{stage} process failed:\n{launcher_output}{worker_output}",
                )
                self.assertEqual(
                    worker_output.count("PALLAS_TIER3_CASE_OK"), world_size
                )
                outputs[stage] = worker_output
                cache_snapshots[stage] = _cache_files(tier3_root)

            self.assertGreater(
                outputs["seed"].count("Tier-3 cache MISS"),
                0,
                outputs["seed"],
            )
            self.assertGreater(
                outputs["restore"].count("Tier-3 cache HIT"),
                0,
                outputs["restore"],
            )
            self.assertEqual(
                outputs["restore"].count("Tier-3 cache MISS"),
                0,
                outputs["restore"],
            )
            self.assertTrue(cache_snapshots["seed"])
            self.assertEqual(cache_snapshots["restore"], cache_snapshots["seed"])
            self.assertGreater(
                outputs["changed"].count("Tier-3 cache MISS"),
                0,
                outputs["changed"],
            )
            self.assertGreater(
                cache_snapshots["changed"].keys(), cache_snapshots["seed"].keys()
            )
            self.assertGreater(
                outputs["changed_restore"].count("Tier-3 cache HIT"),
                0,
                outputs["changed_restore"],
            )
            self.assertEqual(
                outputs["changed_restore"].count("Tier-3 cache MISS"),
                0,
                outputs["changed_restore"],
            )
            self.assertEqual(
                cache_snapshots["changed_restore"], cache_snapshots["changed"]
            )


if __name__ == "__main__":
    unittest.main()
