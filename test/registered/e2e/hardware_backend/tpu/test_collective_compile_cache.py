"""Fresh-process AOT cache hits must preserve logical all-gather ordering."""

import importlib.util
import json
import os
import shlex
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.evals.eval_support import RecipeLauncher

register_tpu_ci(est_time=180, suite="stage-a-test-tpu-unit", chips=4)


def worker(phase, source):
    import torch
    import torch.distributed as dist
    import torch_tpu  # noqa: F401
    from torch._functorch._aot_autograd import autograd_cache

    # Bind PJRT to the launcher's physical rank before permuting logical ranks.
    torch.ones((1,), device="tpu").cpu()
    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"]) ^ (1 if phase.startswith("swap") else 0)
    dist.init_process_group(
        backend="cpu:gloo,tpu:tpu_dist",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
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

    hits = []
    original_lookup = autograd_cache.AOTAutogradCache._lookup

    def lookup(*args, **kwargs):
        result = original_lookup(*args, **kwargs)
        hits.append(result is not None)
        return result

    with patch.object(autograd_cache.AOTAutogradCache, "_lookup", side_effect=lookup):
        compiled = torch.compile(forward, backend="tpu", fullgraph=True, dynamic=False)
        actual = compiled(torch.full((4,), float(rank), device="tpu")).cpu()
    expected = torch.arange(world, dtype=torch.float32).repeat_interleave(4)
    assert torch.equal(actual, expected), (phase, rank, actual, expected)
    assert hits == ["warm" in phase], (phase, rank, hits)
    print(json.dumps({"phase": phase, "rank": rank, "hits": hits}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


class TestCollectiveCompileCache(unittest.TestCase):
    def test_cache_survives_rank_permutation(self):
        if importlib.util.find_spec("torch_tpu") is None:
            self.skipTest("requires TorchTPU and a dedicated TPU host")
        from torch_tpu._internal.utils import hardware

        if hardware.get_tpu_device_count() < 2:
            self.skipTest("requires at least two TPU devices")
        source = Path(__file__).resolve().parents[5] / "sglang_tpu/compile_cache.py"
        python = shlex.quote(sys.executable)
        wrapper = "torch_tpu._internal.distributed.launchers.singlehost_wrapper"
        with tempfile.TemporaryDirectory(prefix="tpu-collective-cache-") as cache:
            env = {
                **os.environ,
                "TORCHINDUCTOR_CACHE_DIR": cache,
                "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS": "false",
                "OMP_NUM_THREADS": "1",
            }
            for name in ("RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "WORLD_SIZE"):
                env.pop(name, None)
            for phase in (
                "base-cold",
                "base-warm",
                "swap-cold",
                "swap-warm",
                "base-warm-again",
            ):
                command = "\n".join(
                    (
                        "set -euo pipefail",
                        f"eval $({python} -m {wrapper})",
                        "export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE",
                        f'{python} -m torch.distributed.run --standalone --nproc-per-node="$WORLD_SIZE" '
                        f"{shlex.quote(__file__)} --worker {phase} {shlex.quote(str(source))}",
                    )
                )
                processes = RecipeLauncher(
                    command=["bash", "-c", command],
                    start_new_session=True,
                    terminate_timeout=10,
                    tag=phase,
                )
                with patch.dict(os.environ, env, clear=True):
                    try:
                        processes.start()
                        result = processes.wait(timeout=300)
                    finally:
                        processes.shutdown()
                self.assertEqual(result, 0, f"{phase} workers failed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
        # Skip libtpu's atexit vBAR StopSession, which can hang forever on GKE
        # and wedge the next phase's PJRT init. This test covers cache keys,
        # not native teardown; the next phase reuses the chips like a restart.
        sys.stderr.flush()
        os._exit(0)
    else:
        unittest.main()
