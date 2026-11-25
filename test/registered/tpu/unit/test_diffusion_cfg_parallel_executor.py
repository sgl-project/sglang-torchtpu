# SPDX-License-Identifier: Apache-2.0
"""Rank-0 stages of the diffusion ParallelExecutor under CFG parallel."""

import contextlib
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

# CPU ranks throughout: on the CPU platform the distributed bootstrap opens no
# TPU chip.
os.environ["SGLANG_DIFFUSION_PLATFORM_OVERRIDE"] = "cpu"

import torch
import torch.multiprocessing as mp

# The spawned ranks run without torch_tpu (its import takes the libtpu lock);
# srt modules look up its compile backend at import.
if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0":
    torch._dynamo.register_backend(lambda gm, example_inputs: gm.forward, "tpu")

from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.pipelines_core.executors.parallel_executor import (
    ParallelExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit", chips=1)


class _Executor(ParallelExecutor):
    """The stage dispatch alone, without residency, caching or profiling."""

    def __init__(self):
        pass

    def _should_use_stage_nvtx(self, payload, server_args):
        return False

    @contextlib.contextmanager
    def _component_residency_request(self, stages, payload, server_args):
        yield

    def _run_stage_with_executor_hooks(
        self, stage, index, payload, server_args, run_stage, use_nvtx
    ):
        return run_stage(stage, payload)


def _cfg2_sp2_rank(rank, world_size, init_file):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size))
    parallel_state.maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=2,
        cfg_degree=2,
        ulysses_degree=2,
        distributed_init_method=f"file://{init_file}",
    )
    stage = SimpleNamespace(
        parallelism_type=StageParallelismType.MAIN_RANK_ONLY_AND_SEND_TO_OTHERS
    )
    # Each running rank's result has its own size (as two decodes' metadata
    # can); world rank 0's is the longest.
    batch = _Executor()._execute_stages(
        [stage],
        "request",
        SimpleNamespace(enable_cfg_parallel=True),
        lambda stage, batch: "x" * (world_size - rank),
    )
    if parallel_state.get_classifier_free_guidance_rank() == 0:
        assert batch == "x" * (world_size - rank), batch  # ran the stage
    else:
        assert batch == "x" * world_size, batch  # world rank 0's result
    parallel_state.cleanup_dist_env_and_memory()


class TestCfgParallelRankZeroStages(CustomTestCase):
    def test_world_rank_zero_sends_to_every_rank(self):
        with (
            tempfile.TemporaryDirectory() as rendezvous,
            # Ranks without torch_tpu.
            patch.dict(os.environ, {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}),
        ):
            mp.spawn(_cfg2_sp2_rank, args=(4, f"{rendezvous}/store"), nprocs=4)


if __name__ == "__main__":
    unittest.main()
