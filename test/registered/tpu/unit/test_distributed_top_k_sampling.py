"""Validate exact distributed TPU TopK and report its TP8 batch crossover.
The physical speed gate covers the B64+ shapes claimed by this PR."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import unittest
from pathlib import Path

_WORKER_ARG = "--distributed-top-k-worker"
if _WORKER_ARG in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault(
        "XLA_FLAGS",
        f"--xla_force_host_platform_device_count={os.environ.get('WORLD_SIZE', '1')}",
    )

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=300, suite="stage-a-test-tpu-unit")

_BATCH_SIZES = (1, 2, 4, 8, 16, 64, 128, 256)
_SPEED_GATED_BATCH_SIZES = (64, 128, 256)
_VOCAB_SIZE = 202_048
_TOP_K = 50
_WARMUPS = 5
_ITERATIONS = 20
_TRIALS = 5
_MIN_GATED_SHAPE_SPEEDUP = 1.10
_MIN_GATED_MEDIAN_SPEEDUP = 1.50


class TestTpuVocabShardFullgraphCompile(CustomTestCase):
    """Guard: TpuVocabShard is constructed inside the torch.compile'd forward
    (fullgraph=True, so no graph break is possible), so it must be a container
    Dynamo can instantiate in-graph. The original msgspec.Struct crashed every
    tp>1 engine precompile with "object.__new__(TpuVocabShard) is not safe"."""

    def test_constructible_under_fullgraph_dynamo(self):
        import torch

        from sglang.srt.hardware_backend.tpu.top_k import TpuVocabShard

        def build_and_read(values: torch.Tensor) -> torch.Tensor:
            shard = TpuVocabShard(
                width=values.shape[-1],
                global_vocab_size=4 * values.shape[-1],
                org_vocab_start=0,
                org_vocab_count=values.shape[-1] - 2,
                org_vocab_padded_count=values.shape[-1] - 1,
                added_vocab_start=3 * values.shape[-1],
                added_vocab_count=1,
            )
            return values[..., : shard.org_vocab_count] + shard.added_vocab_count

        values = torch.randn(2, 16)
        # backend="eager": the guarded property is Dynamo's TRACE of the
        # construction, which is backend-independent; the default backend on a
        # TPU host would instead demand TPU-resident tensors.
        compiled = torch.compile(build_and_read, fullgraph=True, backend="eager")
        torch.testing.assert_close(compiled(values), build_and_read(values))


class TestDistributedTopKSampling(CustomTestCase):
    """Run exactness and paired latency checks on a physical TPU v7 TP8.
    A subprocess owns the complete distributed runtime for test isolation."""

    def test_tp8_exactness_and_speed_regression(self):
        repo_root = Path(__file__).resolve().parents[4]
        worker = Path(__file__).resolve()
        wrapper_module = "torch_tpu._internal.distributed.launchers.singlehost_wrapper"
        python = shlex.quote(sys.executable)
        command = "\n".join(
            (
                f"eval $({python} -m {wrapper_module})",
                "export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE",
                f'{python} -m torch.distributed.run --nproc_per_node="$WORLD_SIZE" '
                f"{shlex.quote(str(worker))} {_WORKER_ARG}",
            )
        )
        env = os.environ.copy()
        for name in ("RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "WORLD_SIZE"):
            env.pop(name, None)

        try:
            result = subprocess.run(
                ["bash", "-lc", command],
                cwd=repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=300,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            self.fail(f"Distributed TPU test timed out:\n{error.stdout or ''}")

        self.assertEqual(
            result.returncode,
            0,
            f"Distributed TPU test failed:\n{result.stdout}",
        )
        self.assertIn("DISTRIBUTED_TOP_K_TEST_PASS", result.stdout)


def _run_distributed_worker() -> None:
    """Exercise production candidate reduction under real TP8 collectives.
    Exact references use a full sort; latency uses paired, alternating trials."""

    import statistics
    import time

    import torch
    import torch.distributed as dist
    from torch_tpu._internal.distributed import tpu_distributed  # noqa: F401

    class _DistGroup:
        """Expose SGLang's concatenating all-gather contract over torch.distributed.
        The adapter keeps the test independent of global SGLang group state."""

        def __init__(self, world_size: int):
            self.world_size = world_size

        def all_gather(self, value: torch.Tensor, dim: int = -1) -> torch.Tensor:
            if dim < 0:
                dim += value.dim()
            output = torch.empty(
                (self.world_size * value.shape[0], *value.shape[1:]),
                dtype=value.dtype,
                device=value.device,
            )
            dist.all_gather_into_tensor(output, value)
            output = output.reshape(self.world_size, *value.shape).movedim(0, dim)
            shape = list(value.shape)
            shape[dim] *= self.world_size
            return output.reshape(shape)

    def _make_local_logits(batch_size: int, local_vocab: int, rank: int):
        """Create deterministic BF16 shards with 64 unique global maxima.
        Large exactly representable peaks remove cutoff-tie ambiguity."""

        generator = torch.Generator().manual_seed(17_000 + rank * 97 + batch_size)
        logits = torch.randn(
            batch_size,
            local_vocab,
            dtype=torch.bfloat16,
            generator=generator,
        )
        rows = torch.arange(batch_size, dtype=torch.int64)
        for slot in range(64):
            global_ids = (slot * 307 + rows * 17) % _VOCAB_SIZE
            owned = (global_ids // local_vocab) == rank
            if owned.any():
                logits[rows[owned], global_ids[owned] % local_vocab] = (
                    32.0 - slot * 0.25
                )
        return logits.to("tpu")

    def _time_ms(fn) -> float:
        """Return synchronized mean device latency over the fixed iteration count.
        Synchronization excludes asynchronous dispatch from the measurement."""

        torch.tpu.synchronize()
        start = time.perf_counter()
        for _ in range(_ITERATIONS):
            fn()
        torch.tpu.synchronize()
        return (time.perf_counter() - start) * 1_000 / _ITERATIONS

    def _measure_pair(post_gather, pre_gather, batch_size: int) -> tuple[float, float]:
        """Measure both paths in alternating order and return trial medians.
        Pairing controls for compilation warmup and device-frequency drift."""

        for warmup in range(_WARMUPS):
            functions = (post_gather, pre_gather)
            if (batch_size + warmup) % 2:
                functions = tuple(reversed(functions))
            for fn in functions:
                fn()
        torch.tpu.synchronize()

        post_times = []
        pre_times = []
        for trial in range(_TRIALS):
            functions = (post_gather, pre_gather)
            if (batch_size + trial) % 2:
                functions = tuple(reversed(functions))
            for fn in functions:
                elapsed = _time_ms(fn)
                (post_times if fn is post_gather else pre_times).append(elapsed)
        return statistics.median(post_times), statistics.median(pre_times)

    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 8:
        raise AssertionError(
            f"registered speed thresholds require TP8, got TP{world_size}"
        )
    if _VOCAB_SIZE % world_size:
        raise AssertionError("test vocabulary must divide evenly across TP ranks")

    from sglang.srt.hardware_backend.tpu import top_k as top_k_module
    from sglang.srt.hardware_backend.tpu.sampler import (
        sample_from_global_top_k_logits,
    )

    group = _DistGroup(world_size)
    local_vocab = _VOCAB_SIZE // world_size
    shard = top_k_module.TpuVocabShard(
        width=local_vocab,
        global_vocab_size=_VOCAB_SIZE,
        org_vocab_start=rank * local_vocab,
        org_vocab_count=local_vocab,
        org_vocab_padded_count=local_vocab,
        added_vocab_start=_VOCAB_SIZE,
        added_vocab_count=0,
    )

    # Compare boundary and representative k values against an independent full
    # sort. Unique maxima make both values and token IDs strictly comparable.
    correctness_logits = _make_local_logits(4, local_vocab, rank)
    gathered_logits = group.all_gather(correctness_logits, dim=-1).float()
    sorted_values, sorted_ids = gathered_logits.sort(dim=-1, descending=True)
    for top_k in (2, 50, 64):
        actual_values, actual_ids = top_k_module.distributed_tpu_top_k(
            correctness_logits,
            shard,
            top_k,
            group=group,
        )
        torch.testing.assert_close(
            actual_values,
            sorted_values[:, :top_k],
            rtol=0,
            atol=0,
        )
        if not torch.equal(
            actual_ids.cpu(), sorted_ids[:, :top_k].to(torch.int32).cpu()
        ):
            raise AssertionError(
                f"candidate token IDs differ from full sort for k={top_k}"
            )

    # Use a layout that can be produced by VocabParallelEmbedding: base padding
    # lives on the last rank and one added token lives at the first added slot.
    org_padded_width = local_vocab - 2
    global_org_vocab = org_padded_width * world_size - 2
    org_vocab_start = min(rank * org_padded_width, global_org_vocab)
    org_vocab_end = min((rank + 1) * org_padded_width, global_org_vocab)
    org_vocab_count = org_vocab_end - org_vocab_start
    global_vocab_size = global_org_vocab + 1
    added_vocab_start = min(global_org_vocab + rank * 2, global_vocab_size)
    added_vocab_end = min(global_org_vocab + (rank + 1) * 2, global_vocab_size)
    added_vocab_count = added_vocab_end - added_vocab_start
    padded_shard = top_k_module.TpuVocabShard(
        width=local_vocab,
        global_vocab_size=global_vocab_size,
        org_vocab_start=org_vocab_start,
        org_vocab_count=org_vocab_count,
        org_vocab_padded_count=org_padded_width,
        added_vocab_start=added_vocab_start,
        added_vocab_count=added_vocab_count,
    )
    padded_logits = torch.zeros(1, local_vocab, dtype=torch.bfloat16)
    padded_logits[:, org_vocab_count:org_padded_width] = 99
    if added_vocab_count:
        padded_logits[:, org_padded_width] = 10
    padded_logits[:, org_padded_width + added_vocab_count :] = 98
    padded_values, padded_ids = top_k_module.distributed_tpu_top_k(
        padded_logits.to("tpu"),
        padded_shard,
        1,
        group=group,
    )
    if padded_values.cpu().item() != 10:
        raise AssertionError("a padded vocabulary column entered the candidate set")
    if padded_ids.cpu().item() != global_org_vocab:
        raise AssertionError("the added-vocabulary candidate has the wrong global ID")

    # A 65-way tie may choose any 64 maxima, but every rank must produce the
    # same valid, duplicate-free subset or subsequent decoding can diverge.
    local_token_ids = (
        torch.arange(local_vocab, dtype=torch.int32, device="tpu") + rank * local_vocab
    )
    tie_logits = torch.zeros(1, local_vocab, dtype=torch.bfloat16, device="tpu")
    tie_logits = torch.where(
        (local_token_ids < 65).unsqueeze(0),
        torch.ones_like(tie_logits),
        tie_logits,
    )
    tie_values, tie_ids = top_k_module.distributed_tpu_top_k(
        tie_logits,
        shard,
        64,
        group=group,
    )
    if not torch.all(tie_values == 1).item():
        raise AssertionError("the cutoff tie returned a non-maximum value")
    tie_ids_cpu = tie_ids.cpu()
    if len(torch.unique(tie_ids_cpu)) != 64 or not torch.all(tie_ids_cpu < 65):
        raise AssertionError("the cutoff tie returned an invalid candidate subset")
    replica_ids = group.all_gather(tie_ids, dim=0).cpu()
    if not all(torch.equal(replica_ids[0], row) for row in replica_ids[1:]):
        raise AssertionError("the cutoff-tie subset differs across TP ranks")

    if rank == 0:
        print(f"TP={world_size} vocab={_VOCAB_SIZE} top_k={_TOP_K}")
        print("batch | post-gather TopK | pre-gather TopK | speedup")

    speedups_by_batch_size = {}
    for batch_size in _BATCH_SIZES:
        local_logits = _make_local_logits(batch_size, local_vocab, rank)
        temperatures = torch.ones(batch_size, 1, dtype=torch.float32, device="tpu")

        def post_gather_top_k() -> torch.Tensor:
            full_logits = group.all_gather(local_logits, dim=-1).float()
            probs = torch.softmax(full_logits, dim=-1)
            global_top_k_probs, global_top_k_token_ids = top_k_module.tpu_top_k(
                probs, _TOP_K
            )
            sampled = torch.multinomial(global_top_k_probs, num_samples=1)
            return torch.gather(global_top_k_token_ids, 1, sampled.to(torch.int64))

        def pre_gather_top_k() -> torch.Tensor:
            global_top_k_logits, global_top_k_token_ids = (
                top_k_module.distributed_tpu_top_k(
                    local_logits,
                    shard,
                    _TOP_K,
                    group=group,
                )
            )
            sampled = sample_from_global_top_k_logits(global_top_k_logits, temperatures)
            return torch.gather(
                global_top_k_token_ids,
                1,
                sampled.to(torch.int64).unsqueeze(-1),
            )

        post_ms, pre_ms = _measure_pair(post_gather_top_k, pre_gather_top_k, batch_size)
        speedup = post_ms / pre_ms
        speedups_by_batch_size[batch_size] = speedup
        if rank == 0:
            print(
                f"{batch_size:5d} | {post_ms:16.4f} ms | {pre_ms:15.4f} ms | "
                f"{speedup:7.3f}x"
            )

    # Keep the small shapes in every physical run so the TP8 crossover remains
    # visible. Gate only the B64+ shapes for which this PR claims a speedup.
    gated_speedups = {
        batch_size: speedups_by_batch_size[batch_size]
        for batch_size in _SPEED_GATED_BATCH_SIZES
    }
    median_speedup = statistics.median(gated_speedups.values())
    if min(gated_speedups.values()) < _MIN_GATED_SHAPE_SPEEDUP:
        raise AssertionError(
            f"pre-gather TopK regressed on rank {rank}: "
            f"gated_speedups={gated_speedups}, "
            f"required_each={_MIN_GATED_SHAPE_SPEEDUP:.2f}x"
        )
    if median_speedup < _MIN_GATED_MEDIAN_SPEEDUP:
        raise AssertionError(
            f"pre-gather TopK missed the median speed gate on rank {rank}: "
            f"gated_speedups={gated_speedups}, "
            f"required_median={_MIN_GATED_MEDIAN_SPEEDUP:.2f}x"
        )
    if rank == 0:
        print(f"B64+ median speedup: {median_speedup:.3f}x")
        print("DISTRIBUTED_TOP_K_TEST_PASS")
    dist.destroy_process_group()


if __name__ == "__main__":
    if _WORKER_ARG in sys.argv:
        _run_distributed_worker()
    else:
        unittest.main()
