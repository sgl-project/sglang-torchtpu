"""Fresh-process AOT cache hits must preserve logical all-gather ordering."""

import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.evals.eval_support import RecipeLauncher

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=4)

# Each launch is a fresh process with one rank assignment; its compiles run in
# order, with Dynamo reset between them so each one looks up the disk cache.
# The swap launch must miss what base stored, and base-again proves base's entry
# hits again from a fresh process.
_LAUNCHES = (
    ("base", ("cold", "warm")),
    ("swap", ("cold", "warm")),
    ("base-again", ("warm",)),
)
# TPU runtime start plus distributed init. Healthy starts take ~12 s and the
# slowest recovered start seen in CI took ~45 s; a stalled one never returns.
_INIT_DEADLINE = 150
_LAUNCH_TIMEOUT = 300
# A launch that stalls before every rank finished init never reached the code
# under test, so it is relaunched once. A launch past init is never relaunched.
_INIT_ATTEMPTS = 2


def worker(launch, source, init_dir):
    import faulthandler

    import torch
    import torch.distributed as dist
    import torch_tpu  # noqa: F401
    from torch._functorch._aot_autograd import autograd_cache

    # On expiry faulthandler dumps every thread's stack, then exits with 1.
    faulthandler.dump_traceback_later(_INIT_DEADLINE, exit=True)
    # Bind PJRT to the launcher's physical rank before permuting logical ranks.
    torch.ones((1,), device="tpu").cpu()
    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"]) ^ (1 if launch == "swap" else 0)
    dist.init_process_group(
        backend="cpu:gloo,tpu:tpu_dist",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
    faulthandler.cancel_dump_traceback_later()
    Path(init_dir, os.environ["RANK"]).write_text(str(world))

    spec = importlib.util.spec_from_file_location("cache_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.add_tpu_rank_assignment_to_compile_cache_key()

    @torch.library.custom_op("cache_regression::gather", mutates_args=())
    def gather(x: torch.Tensor) -> torch.Tensor:
        result = torch.empty((world * x.shape[0],), dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(result, x)
        return result

    @gather.register_fake
    def fake_gather(x):
        return torch.empty((world * x.shape[0],), dtype=x.dtype, device=x.device)

    def forward(x):
        return gather(x) + 0

    original_lookup = autograd_cache.AOTAutogradCache._lookup
    expected = torch.arange(world, dtype=torch.float32).repeat_interleave(4)
    for step in dict(_LAUNCHES)[launch]:
        hits = []

        def lookup(*args, **kwargs):
            result = original_lookup(*args, **kwargs)
            hits.append(result is not None)
            return result

        torch._dynamo.reset()
        with patch.object(
            autograd_cache.AOTAutogradCache, "_lookup", side_effect=lookup
        ):
            compiled = torch.compile(
                forward, backend="tpu", fullgraph=True, dynamic=False
            )
            actual = compiled(torch.full((4,), float(rank), device="tpu")).cpu()
        assert torch.equal(actual, expected), (launch, step, rank, actual, expected)
        assert hits == [step == "warm"], (launch, step, rank, hits)
        print(
            json.dumps({"launch": launch, "step": step, "rank": rank, "hits": hits}),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


def _all_ranks_initialized(init_dir: Path) -> bool:
    markers = list(init_dir.iterdir())
    return bool(markers) and len(markers) == int(markers[0].read_text())


class TestCollectiveCompileCache(unittest.TestCase):
    def test_cache_survives_rank_permutation(self):
        if importlib.util.find_spec("torch_tpu") is None:
            self.skipTest("requires TorchTPU and a dedicated TPU host")
        from torch_tpu._internal.utils import hardware

        if hardware.get_tpu_device_count() < 2:
            self.skipTest("requires at least two TPU devices")

        source = Path(__file__).resolve().parents[5] / "sglang_tpu/compile_cache.py"
        with tempfile.TemporaryDirectory(prefix="tpu-collective-cache-") as scratch:
            env = {
                **os.environ,
                "TORCHINDUCTOR_CACHE_DIR": os.path.join(scratch, "cache"),
                "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS": "false",
                "OMP_NUM_THREADS": "1",
            }
            for name in ("RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "WORLD_SIZE"):
                env.pop(name, None)
            for launch, _ in _LAUNCHES:
                self._run_launch(launch, source, env, Path(scratch))

    def _run_launch(self, launch: str, source: Path, env: dict, scratch: Path):
        python = shlex.quote(sys.executable)
        for attempt in range(1, _INIT_ATTEMPTS + 1):
            init_dir = scratch / f"init-{launch}-{attempt}"
            init_dir.mkdir()
            command = "\n".join(
                (
                    "set -euo pipefail",
                    f"eval $({python} -m torch.tpu.distributed)",
                    "export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE",
                    f'{python} -m torch.distributed.run --standalone --nproc-per-node="$WORLD_SIZE" '
                    f"{shlex.quote(__file__)} --worker {launch} "
                    f"{shlex.quote(str(source))} {shlex.quote(str(init_dir))}",
                )
            )
            processes = RecipeLauncher(
                command=["bash", "-c", command],
                start_new_session=True,
                terminate_timeout=10,
                tag=f"{launch}#{attempt}",
            )
            with patch.dict(os.environ, env, clear=True):
                try:
                    processes.start()
                    result = processes.wait(timeout=_LAUNCH_TIMEOUT)
                except subprocess.TimeoutExpired:
                    result = None
                finally:
                    processes.shutdown()
            if result == 0:
                return
            if _all_ranks_initialized(init_dir):
                self.fail(f"{launch} workers failed after init (exit code {result})")
            print(
                f"[{launch}#{attempt}] ranks did not finish TPU runtime and "
                f"distributed init (exit code {result}); this is not a cache failure",
                flush=True,
            )
        self.fail(
            f"{launch}: TPU runtime/distributed init failed in all "
            f"{_INIT_ATTEMPTS} launches, before any compile ran"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3], sys.argv[4])
        # Skip libtpu's atexit vBAR StopSession, which can hang forever on GKE
        # and wedge the next launch's PJRT init. This test covers cache keys,
        # not native teardown; the next launch reuses the chips like a restart.
        sys.stderr.flush()
        os._exit(0)
    else:
        unittest.main()
