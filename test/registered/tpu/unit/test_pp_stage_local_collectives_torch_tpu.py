"""Guard pipeline stages with in-stage parallelism against torch_tpu changes.

With PP x TP or PP x PCP every pipeline stage owns a group of devices and runs
its own collectives over that group: all-reduces and all-gathers from c10d, and
Pallas kernels that exchange data over a mesh of the stage's devices (the
context-parallel attention ring). Stages launch these at different rates by
construction: stage 0 runs a micro-batch while the later stages are idle or on
another one, and the stages meet only in the hand-off between them. So a
stage's program must run on the stage's devices alone.

torch_tpu#4227 (on release/2026-09-22 as #4242, in torch-tpu
0.1.1.dev20260924085621 and later) broke that: it pads the device assignment of
every subgroup program to all devices of the slice, so every device must launch
a stage's program, and the first launch another stage does not mirror hangs the
slice. Nothing else in CI runs stage-local collectives at uneven rates, so a
torch-tpu bump that does this passes every other check.

Each case runs in its own torch.distributed.run and gives the stages different
launch counts. A rank that outlives _RANK_DEADLINE dumps its stack and exits,
so a hang fails the case instead of the job.
"""

import importlib.util
import os
import shlex
import signal
import subprocess
import sys
import unittest
from pathlib import Path

_WORKER_ARG = "--pp-stage-collectives-worker"

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(
    est_time=240,
    suite="stage-a-test-tpu-unit",
    disabled="needs a torch_tpu with standalone process subgroups (torch_tpu #4387 / #5092)",
    chips=4,
)

_PASS_MARKER = "PP_STAGE_COLLECTIVES_TEST_PASS"
# A healthy case finishes in well under a minute. A rank still running after
# this long waits for a launch that ranks outside its stage never make.
_RANK_DEADLINE = 180
_CASE_TIMEOUT = _RANK_DEADLINE + 120
# Rounds of the stage-mesh kernel inside one launch.
_RING_ROUNDS = 4
_HANG_HINT = (
    "A rank that exits with code 1 after the deadline dumps its stack above: it "
    "waits for a stage program that ranks outside its stage never launch, i.e. "
    "torch_tpu compiled the stage's program for devices outside the stage "
    "(the regression of torch_tpu#4227 / #4242)."
)
# torch_tpu launchers that print the TORCH_TPU_* launch env for a single host,
# newest first. singlehost_wrapper was removed from torch_tpu on 2026-09-24 and
# environment was renamed to distributed in 0.2.1.
_LAUNCH_ENV_MODULES = (
    "torch_tpu._internal.distributed.distributed",
    "torch_tpu._internal.distributed.environment",
    "torch_tpu._internal.distributed.launchers.singlehost_wrapper",
)
_LAUNCH_ENV_KEYS = {
    "TORCH_TPU_TOPOLOGY",
    "TORCH_TPU_SLICEBUILDER_ADDRESSES",
    "WORLD_SIZE",
}


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _device_count() -> int:
    try:
        if not _has_module("torch_tpu"):
            return 0
        from torch_tpu._internal.utils import hardware

        return hardware.get_tpu_device_count()
    except Exception:
        return 0


def _launch_env(env: dict[str, str]) -> dict[str, str]:
    module = next((m for m in _LAUNCH_ENV_MODULES if _has_module(m)), None)
    if module is None:
        raise AssertionError(f"torch_tpu has none of {_LAUNCH_ENV_MODULES}")
    completed = subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True,
        check=False,
        env=env,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise AssertionError(f"{module} failed:\n{completed.stderr}")
    # torch_tpu also prints banners on stdout, so keep only the keys we need.
    launch_env = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.removeprefix("export ").partition("=")
        if separator and key in _LAUNCH_ENV_KEYS:
            launch_env[key] = shlex.split(value)[0]
    if launch_env.keys() != _LAUNCH_ENV_KEYS:
        raise AssertionError(f"incomplete launch env from {module}: {launch_env}")
    return launch_env


@unittest.skipIf(_device_count() < 8, "Requires 8 TPU devices")
class TestPPStageLocalCollectivesTorchTpu(CustomTestCase):
    """Each case launches stage-local programs at a different rate per stage.
    A subprocess owns the distributed runtime of each case, so a hang in one
    case is reported on its own."""

    def _run_case(self, case: str, stage_size: int) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        env = os.environ.copy()
        # JOB_COMPLETION_INDEX makes the launcher take its multi-host path.
        for name in (
            "RANK",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "WORLD_SIZE",
            "JOB_COMPLETION_INDEX",
        ):
            env.pop(name, None)
        launch_env = _launch_env(env)
        env.update(launch_env)

        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node",
                launch_env["WORLD_SIZE"],
                str(Path(__file__).resolve()),
                _WORKER_ARG,
                case,
                str(stage_size),
            ],
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # Crashing ranks can write partial UTF-8 sequences.
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=_CASE_TIMEOUT)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            self.fail(
                f"{case} (stages of {stage_size}) timed out after "
                f"{_CASE_TIMEOUT}s. {_HANG_HINT}\n{output}"
            )

        self.assertEqual(
            process.returncode,
            0,
            f"{case} (stages of {stage_size}) failed. {_HANG_HINT}\n{output}",
        )
        self.assertIn(f"{_PASS_MARKER} {case}", output)

    def test_uneven_stage_all_reduce_pairs(self):
        """Stages of 2 (PP4 x TP2): stage s all-reduces over its group s + 1
        times, then every rank all-reduces over the world."""
        self._run_case("uneven_all_reduce", 2)

    def test_uneven_stage_all_reduce_quads(self):
        """Stages of 4 (PP2 x TP4). The second stage, on devices [6, 7, 2, 3],
        is off the slice origin."""
        self._run_case("uneven_all_reduce", 4)

    def test_only_first_stage_all_reduce(self):
        """Stage 0 all-reduces over its group while the other stages wait on
        the host, as when one micro-batch is in flight."""
        self._run_case("first_stage_only", 2)

    def test_uneven_compiled_stage_forward(self):
        """A compiled stage forward that all-gathers over its group, launched
        s + 1 times by stage s, as a PP x TP forward is."""
        self._run_case("uneven_compiled_forward", 2)

    def test_uneven_stage_mesh_kernel(self):
        """A Pallas kernel exchanging data over a mesh of the stage's devices
        (the context-parallel attention ring), launched s + 1 times by stage s."""
        self._run_case("uneven_stage_mesh_kernel", 2)


def _stage_mesh_ring(stage: int, stage_size: int, group):
    """jax_op of a Pallas kernel over the stage's devices: each round, every
    device sends its block to the next device and adds one to what it got."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from torch_tpu._internal import pallas as tpu_pallas

    spec = jax.sharding.PartitionSpec("stage")
    mesh = tpu_pallas.get_pallas_mesh(axis_names=("stage",), process_group=group)

    def body(x_ref, o_ref, buf, send_sem, recv_sem, sync_sem, free_sem):
        me = lax.axis_index("stage")
        right = lax.rem(me + 1, stage_size)
        left = lax.rem(me + stage_size - 1, stage_size)
        buf[0] = x_ref[0]

        def one_round(_, carry):
            copy = pltpu.make_async_remote_copy(
                src_ref=buf.at[0],
                dst_ref=buf.at[1],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(right,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copy.wait()
            # Tell the right neighbour its buffer holds this round's block.
            pl.semaphore_signal(
                sync_sem, 1, device_id=(right,), device_id_type=pl.DeviceIdType.MESH
            )
            pl.semaphore_wait(sync_sem, 1)
            buf[0] = buf[1] + 1
            # Tell the left neighbour it may overwrite this device's buffer.
            pl.semaphore_signal(
                free_sem, 1, device_id=(left,), device_id_type=pl.DeviceIdType.MESH
            )
            pl.semaphore_wait(free_sem, 1)
            return carry

        lax.fori_loop(0, _RING_ROUNDS, one_round, 0)
        o_ref[0] = buf[0]

    call = pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((stage_size, 8, 128), jnp.int32),
        scratch_shapes=[
            pltpu.VMEM((2, 8, 128), jnp.int32),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.REGULAR,
            pltpu.SemaphoreType.REGULAR,
        ],
    )

    def fn(x: jax.Array) -> jax.Array:
        return jax.shard_map(
            call, mesh=mesh, in_specs=(spec,), out_specs=spec, check_vma=False
        )(x)

    return tpu_pallas.jax_op(
        f"pp_stage_guard::ring_s{stage}",
        fn,
        mesh=mesh,
        input_partition_specs=[spec],
        output_partition_specs=[spec],
    )


def _run_distributed_worker(case: str, stage_size: int) -> None:
    import faulthandler

    import torch
    import torch.distributed as dist
    from torch_tpu._internal.distributed import spmd_util

    # On expiry faulthandler dumps every thread's stack, then exits with 1.
    faulthandler.dump_traceback_later(_RANK_DEADLINE, exit=True)
    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    cpu_group = dist.new_group(backend="gloo")
    # Every rank creates every group, in the same order, as c10d requires.
    stage_groups = [
        dist.new_group(ranks=list(range(s * stage_size, (s + 1) * stage_size)))
        for s in range(world_size // stage_size)
    ]
    stage = rank // stage_size
    group = stage_groups[stage]
    if hasattr(spmd_util, "make_standalone"):
        spmd_util.make_standalone(group)
    stage_ranks = range(stage * stage_size, (stage + 1) * stage_size)
    device = torch.device("tpu")
    launches = stage + 1

    # Mismatches are collected rather than raised: a stage that stopped
    # launching would hang the others instead of reporting.
    failures: list[str] = []

    def check(what: str, got: float, want: float) -> None:
        if got != want:
            failures.append(f"{what}: got {got}, expected {want}")

    if case == "uneven_all_reduce":
        for i in range(launches):
            x = torch.full((8, 128), float(rank + i), device=device)
            dist.all_reduce(x, group=group)
            want = float(sum(r + i for r in stage_ranks))
            # Reading the result back is what waits for the program to run.
            check(f"stage all_reduce {i}", float(x[0, 0].cpu()), want)
        x = torch.full((8, 128), float(rank), device=device)
        dist.all_reduce(x)
        check("world all_reduce", float(x[0, 0].cpu()), float(sum(range(world_size))))
    elif case == "first_stage_only":
        if stage == 0:
            x = torch.full((8, 128), float(rank + 1), device=device)
            dist.all_reduce(x, group=group)
            want = float(sum(r + 1 for r in stage_ranks))
            check("stage 0 all_reduce", float(x[0, 0].cpu()), want)
    elif case == "uneven_compiled_forward":

        def forward(x: torch.Tensor) -> torch.Tensor:
            y = x * 2 + 1
            # Dynamo traces the gather into a tensor of y's rank, so gather
            # along dim 0 and split the stage dim out afterwards.
            gathered = torch.empty(
                (stage_size * y.shape[0],) + tuple(y.shape[1:]),
                dtype=y.dtype,
                device=y.device,
            )
            dist.all_gather_into_tensor(gathered, y, group=group)
            return gathered.view(stage_size, *y.shape).sum(0)

        compiled = torch.compile(
            spmd_util.spmd_safe(forward), backend="tpu", fullgraph=True
        )
        for i in range(launches):
            x = torch.full((8, 128), float(rank + i), device=device)
            got = float(compiled(x)[0, 0].cpu())
            want = float(sum(2 * (r + i) + 1 for r in stage_ranks))
            check(f"compiled forward {i}", got, want)
    elif case == "uneven_stage_mesh_kernel":
        ring = _stage_mesh_ring(stage, stage_size, group)
        position = rank - stage_ranks[0]
        for i in range(launches):
            # Device p ends with the block of device p - _RING_ROUNDS, plus one
            # per round.
            source = stage_ranks[(position - _RING_ROUNDS) % stage_size]
            x = torch.full((1, 8, 128), rank + 10 * i, dtype=torch.int32, device=device)
            want = source + 10 * i + _RING_ROUNDS
            check(f"stage mesh ring {i}", int(ring(x)[0, 0, 0].cpu()), want)
    else:
        raise ValueError(f"unknown case {case!r}")

    torch.accelerator.synchronize()
    reports: list = [None] * world_size
    dist.all_gather_object(reports, (rank, failures), group=cpu_group)
    faulthandler.cancel_dump_traceback_later()
    if rank == 0:
        errors = [f"rank {r}: {f}" for r, fs in reports for f in fs]
        if errors:
            raise AssertionError(f"{case} checks failed:\n" + "\n".join(errors))
        print(f"{_PASS_MARKER} {case}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    if _WORKER_ARG in sys.argv:
        index = sys.argv.index(_WORKER_ARG)
        _run_distributed_worker(sys.argv[index + 1], int(sys.argv[index + 2]))
    else:
        unittest.main()
