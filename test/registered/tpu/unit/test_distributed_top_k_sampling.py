"""Validate the exact Pallas TopK ring and report its TP8/TP16 speedup.
The registered physical speed gate covers TP8; the worker also supports TP16."""

from __future__ import annotations

import importlib.util
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
from sglang.test.ci.gate_matrix import full_matrix
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=180, suite="stage-a-test-tpu-unit", chips=4)
# The latency benchmark below runs only on the nightly full-matrix run, which
# is where the speed gate lives; see the guard in _run_distributed_worker.
register_tpu_ci(est_time=900, suite="nightly-tpu-4-v7", nightly=True)

_BATCH_SIZES = (64, 128, 256)
_SPEED_GATED_BATCH_SIZES = (64, 128, 256)
_VOCAB_SIZE = 202_048
_TOP_K = 50
_WARMUPS = 5
_ITERATIONS = 20
_TRIALS = 5
_MIN_GATED_SHAPE_SPEEDUP = 1.05
_MIN_GATED_MEDIAN_SPEEDUP_BY_WORLD_SIZE = {8: 1.30, 16: 1.30}


class TestTpuVocabShardFullgraphCompile(CustomTestCase):
    """Guard: TpuVocabShard is constructed inside the torch.compile'd forward
    (fullgraph=True, so no graph break is possible), so it must be a container
    Dynamo can instantiate in-graph. The original msgspec.Struct crashed every
    tp>1 engine precompile with "object.__new__(TpuVocabShard) is not safe"."""

    def test_constructible_under_fullgraph_dynamo(self):
        import torch

        from sglang_tpu.layers.top_k import TpuVocabShard

        def build_and_read(values: torch.Tensor) -> torch.Tensor:
            shard = TpuVocabShard(
                width=values.shape[-1],
                global_vocab_size=4 * values.shape[-1],
                org_vocab_size=3 * values.shape[-1],
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


def _skip_if_not_tp8_or_tp16() -> bool:
    try:
        if importlib.util.find_spec("torch_tpu") is None:
            return True
        from torch_tpu._internal.utils import hardware

        return hardware.get_tpu_device_count() not in (8, 16)
    except Exception:
        return True


@unittest.skipIf(_skip_if_not_tp8_or_tp16(), "Requires 8 or 16 TPU devices")
class TestDistributedTopKSampling(CustomTestCase):
    """Run exactness and paired latency checks on a physical TPU v7 slice.
    A subprocess owns the complete distributed runtime for test isolation."""

    def test_exactness_and_speed_regression(self):
        repo_root = Path(__file__).resolve().parents[4]
        worker = Path(__file__).resolve()
        wrapper_module = "torch.tpu.distributed"
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
                timeout=1800,
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
    """Exercise production candidate reduction under real TP8/TP16 collectives.
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
            self.rank_in_group = dist.get_rank()

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

    def _make_local_logits(
        batch_size: int,
        local_vocab: int,
        rank: int,
        dtype=torch.bfloat16,
    ):
        """Create deterministic shards with 64 unique global maxima.
        Large exactly representable peaks remove cutoff-tie ambiguity."""

        generator = torch.Generator().manual_seed(17_000 + rank * 97 + batch_size)
        logits = torch.randn(
            batch_size,
            local_vocab,
            dtype=dtype,
            generator=generator,
        )
        rows = torch.arange(batch_size, dtype=torch.int64)
        for slot in range(64):
            owner = slot % world_size
            if owner == rank:
                local_ids = ((slot // world_size) * 307 + rows * 17) % local_vocab
                logits[rows, local_ids] = 32.0 - slot * 0.25
        return logits.to("tpu")

    def _time_ms(fn) -> float:
        """Return synchronized mean device latency over the fixed iteration count.
        Synchronization excludes asynchronous dispatch from the measurement."""

        torch.accelerator.synchronize()
        start = time.perf_counter()
        for _ in range(_ITERATIONS):
            fn()
        torch.accelerator.synchronize()
        return (time.perf_counter() - start) * 1_000 / _ITERATIONS

    def _measure_pair(reference, pallas, batch_size: int) -> tuple[float, float]:
        """Measure both paths in alternating order and return trial medians.
        Pairing controls for compilation warmup and device-frequency drift."""

        for warmup in range(_WARMUPS):
            functions = (reference, pallas)
            if (batch_size + warmup) % 2:
                functions = tuple(reversed(functions))
            for fn in functions:
                fn()
        torch.accelerator.synchronize()

        reference_times = []
        pallas_times = []
        for trial in range(_TRIALS):
            functions = (reference, pallas)
            if (batch_size + trial) % 2:
                functions = tuple(reversed(functions))
            for fn in functions:
                elapsed = _time_ms(fn)
                (reference_times if fn is reference else pallas_times).append(elapsed)
        return statistics.median(reference_times), statistics.median(pallas_times)

    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size not in (8, 16):
        raise AssertionError(
            f"Pallas physical validation requires TP8 or TP16, got TP{world_size}"
        )
    if _VOCAB_SIZE % world_size:
        raise AssertionError("test vocabulary must divide evenly across TP ranks")
    physical_ids = [None] * world_size
    dist.all_gather_object(physical_ids, int(tpu_distributed.global_device_id()))
    non_identity_mapping = tuple(physical_ids) != tuple(range(world_size))
    if rank == 0:
        print(
            "PALLAS_TOPK_PHYSICAL_MAPPING "
            f"non_identity={non_identity_mapping} ids={physical_ids}",
            flush=True,
        )
    if (
        os.environ.get("SGLANG_TPU_REQUIRE_NON_IDENTITY_TOPK_TEST") == "1"
        and not non_identity_mapping
    ):
        raise AssertionError(
            "the dedicated topology gate requires a non-identity "
            f"logical-to-physical mapping, got {physical_ids}"
        )

    from sglang_tpu.layers import top_k as top_k_module
    from sglang_tpu.layers.sampler import (
        sample_from_global_top_k_logits,
    )

    group = _DistGroup(world_size)
    local_vocab = _VOCAB_SIZE // world_size
    shard = top_k_module.TpuVocabShard(
        width=local_vocab,
        global_vocab_size=_VOCAB_SIZE,
        org_vocab_size=_VOCAB_SIZE,
        org_vocab_start=rank * local_vocab,
        org_vocab_count=local_vocab,
        org_vocab_padded_count=local_vocab,
        added_vocab_start=_VOCAB_SIZE,
        added_vocab_count=0,
    )
    from sglang_tpu.layers.distributed_topk_pallas import (
        register_pallas_distributed_top_k_op,
    )

    top_k_module.register_tpu_top_k_ops()
    register_pallas_distributed_top_k_op(world_size)

    empty_values, empty_ids = top_k_module.distributed_tpu_top_k(
        torch.empty((0, local_vocab), dtype=torch.bfloat16, device="tpu"),
        shard,
        64,
        group=group,
    )
    if empty_values.shape != (0, 64) or empty_values.dtype != torch.float32:
        raise AssertionError("empty TopK values have the wrong contract")
    if empty_ids.shape != (0, 64) or empty_ids.dtype != torch.int32:
        raise AssertionError("empty TopK IDs have the wrong contract")

    def _candidate_all_gather_reference(local_logits, shard, k):
        """Test-only oracle for the production direct-HBM reducer."""

        positions = torch.arange(
            shard.width,
            dtype=torch.int32,
            device=local_logits.device,
        )
        added_positions = positions - shard.org_vocab_padded_count
        valid = (positions < shard.org_vocab_count) | (
            (added_positions >= 0) & (added_positions < shard.added_vocab_count)
        )
        masked = torch.where(valid.unsqueeze(0), local_logits, -torch.inf)
        local_values, local_positions = top_k_module.tpu_top_k(masked, k)
        local_ids = torch.where(
            local_positions < shard.org_vocab_count,
            shard.org_vocab_start + local_positions,
            shard.added_vocab_start + local_positions - shard.org_vocab_padded_count,
        ).to(torch.int32)
        packed = torch.stack((local_values.float(), local_ids.float()), dim=-1)
        gathered = group.all_gather(packed, dim=-2)
        gathered_values = gathered[..., 0]
        gathered_ids = gathered[..., 1].to(torch.int32)
        values, selected = top_k_module.tpu_top_k(gathered_values, k)
        ids = torch.gather(gathered_ids, 1, selected.to(torch.int64))
        return values.float(), ids

    # Compare boundary and representative k values against an independent full
    # sort. Unique maxima make both values and token IDs strictly comparable.
    for dtype in (torch.bfloat16, torch.float32):
        correctness_logits = _make_local_logits(33, local_vocab, rank, dtype)
        gathered_logits = group.all_gather(correctness_logits, dim=-1).float()
        sorted_values, sorted_ids = gathered_logits.sort(dim=-1, descending=True)
        top_ks = (1, 2, 50, 64) if dtype == torch.bfloat16 else (1, 64)
        for top_k in top_ks:
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
                    "candidate token IDs differ from full sort for "
                    f"dtype={dtype}, k={top_k}"
                )

    # Exercise the production dispatch under the same full-graph compilation
    # mode used by the TPU engine. This catches graph breaks in validation,
    # environment reads, rank materialization, and custom-op registration.
    compile_logits = _make_local_logits(33, local_vocab, rank)
    compile_reference = group.all_gather(compile_logits, dim=-1).float()
    compile_reference_values, compile_reference_ids = compile_reference.sort(
        dim=-1, descending=True
    )

    def _compiled_distributed_top_k(logits):
        return top_k_module.distributed_tpu_top_k(
            logits,
            shard,
            50,
            group=group,
        )

    compiled_distributed_top_k = torch.compile(
        _compiled_distributed_top_k,
        backend="tpu",
        fullgraph=True,
        dynamic=False,
    )
    compiled_values, compiled_ids = compiled_distributed_top_k(compile_logits)
    torch.testing.assert_close(
        compiled_values,
        compile_reference_values[:, :50],
        rtol=0,
        atol=0,
    )
    if not torch.equal(
        compiled_ids.cpu(), compile_reference_ids[:, :50].to(torch.int32).cpu()
    ):
        raise AssertionError("compiled Pallas TopK returned incorrect token IDs")

    # Exercise skewed entry and rapid shape reuse. A peer may enter the direct-
    # HBM region before rank zero, and consecutive invocations must not consume
    # stale semaphore state from a prior shape.
    delayed_logits = _make_local_logits(1, local_vocab, rank)
    delayed_reference = group.all_gather(delayed_logits, dim=-1).float()
    delayed_values, delayed_ids = delayed_reference.sort(dim=-1, descending=True)
    if rank == 0:
        time.sleep(0.25)
    actual_values, actual_ids = top_k_module.distributed_tpu_top_k(
        delayed_logits,
        shard,
        64,
        group=group,
    )
    torch.testing.assert_close(actual_values, delayed_values[:, :64], rtol=0, atol=0)
    if not torch.equal(actual_ids.cpu(), delayed_ids[:, :64].to(torch.int32).cpu()):
        raise AssertionError("delayed-rank Pallas TopK returned incorrect token IDs")

    for replay_batch in (64, 33, 128, 1, 64):
        replay_logits = _make_local_logits(replay_batch, local_vocab, rank)
        replay_values, replay_ids = top_k_module.distributed_tpu_top_k(
            replay_logits,
            shard,
            1,
            group=group,
        )
        reference = group.all_gather(replay_logits, dim=-1).float()
        reference_values, reference_ids = reference.max(dim=-1, keepdim=True)
        torch.testing.assert_close(replay_values, reference_values, rtol=0, atol=0)
        if not torch.equal(replay_ids.cpu(), reference_ids.to(torch.int32).cpu()):
            raise AssertionError(
                f"back-to-back B{replay_batch} Pallas TopK returned wrong IDs"
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
        org_vocab_size=global_org_vocab,
        org_vocab_start=org_vocab_start,
        org_vocab_count=org_vocab_count,
        org_vocab_padded_count=org_padded_width,
        added_vocab_start=added_vocab_start,
        added_vocab_count=added_vocab_count,
    )
    padded_logits = torch.full(
        (1, local_vocab),
        -100,
        dtype=torch.bfloat16,
    )
    padded_logits[:, org_vocab_count:org_padded_width] = 200
    for slot in range(64):
        if slot % world_size == rank:
            padded_logits[:, slot // world_size] = 64 - slot * 0.25
    if added_vocab_count:
        padded_logits[:, org_padded_width] = 100
    padded_logits[:, org_padded_width + added_vocab_count :] = 199
    padded_values, padded_ids = top_k_module.distributed_tpu_top_k(
        padded_logits.to("tpu"),
        padded_shard,
        1,
        group=group,
    )
    if padded_values.cpu().item() != 100:
        raise AssertionError("a padded vocabulary column entered the candidate set")
    if padded_ids.cpu().item() != global_org_vocab:
        raise AssertionError("the added-vocabulary candidate has the wrong global ID")

    padded_reference_values, padded_reference_ids = _candidate_all_gather_reference(
        padded_logits.to("tpu"),
        padded_shard,
        64,
    )
    padded_top64_values, padded_top64_ids = top_k_module.distributed_tpu_top_k(
        padded_logits.to("tpu"),
        padded_shard,
        64,
        group=group,
    )
    torch.testing.assert_close(
        padded_top64_values,
        padded_reference_values,
        rtol=0,
        atol=0,
    )
    if not torch.equal(padded_top64_ids.cpu(), padded_reference_ids.cpu()):
        raise AssertionError("padded-vocabulary Top64 returned incorrect token IDs")

    # A 65-way tie may choose any 64 maxima, but every rank must produce the
    # same valid, duplicate-free subset or subsequent decoding can diverge.
    tie_logits = torch.zeros(1, local_vocab, dtype=torch.bfloat16, device="tpu")
    local_tie_count = 65 // world_size + int(rank < 65 % world_size)
    tie_logits[:, :local_tie_count] = 1
    tie_values, tie_ids = top_k_module.distributed_tpu_top_k(
        tie_logits,
        shard,
        64,
        group=group,
    )
    if not torch.all(tie_values == 1).item():
        raise AssertionError("the cutoff tie returned a non-maximum value")
    tie_ids_cpu = tie_ids.cpu()
    allowed_tie_ids = torch.tensor(
        [
            source_rank * local_vocab + local_position
            for source_rank in range(world_size)
            for local_position in range(
                65 // world_size + int(source_rank < 65 % world_size)
            )
        ],
        dtype=torch.int32,
    )
    if (
        len(torch.unique(tie_ids_cpu)) != 64
        or not torch.isin(tie_ids_cpu, allowed_tie_ids).all()
    ):
        raise AssertionError("the cutoff tie returned an invalid candidate subset")
    replica_ids = group.all_gather(tie_ids, dim=0).cpu()
    if not all(torch.equal(replica_ids[0], row) for row in replica_ids[1:]):
        raise AssertionError("the cutoff-tie subset differs across TP ranks")

    # Everything above is correctness: the empty-input contract, exactness
    # against a full sort over two dtypes and four k values, and the cutoff tie.
    # What follows is a paired latency benchmark -- 3 batch sizes x (5 warmups +
    # 5 trials x 2 paths x 20 iterations) -- and a 1.3x speedup assertion. That
    # is ~85% of this file's 754s and it is a performance gate, which the PR gate
    # is the wrong place for: the slice is shared, so the measurement is noisy
    # exactly when the gate is busiest. The nightly runs it.
    if not full_matrix():
        if rank == 0:
            print("speed gate deferred to the nightly full-matrix run")
            print("DISTRIBUTED_TOP_K_TEST_PASS")
        dist.destroy_process_group()
        return

    if rank == 0:
        print(f"TP={world_size} vocab={_VOCAB_SIZE} top_k={_TOP_K}")
        print("batch | candidate-gather reference | Pallas direct HBM | speedup")

    speedups_by_batch_size = {}
    for batch_size in _BATCH_SIZES:
        local_logits = _make_local_logits(batch_size, local_vocab, rank)
        temperatures = torch.ones(batch_size, 1, dtype=torch.float32, device="tpu")

        def candidate_all_gather_top_k(
            local_logits=local_logits,
            temperatures=temperatures,
        ) -> torch.Tensor:
            global_top_k_logits, global_top_k_token_ids = (
                _candidate_all_gather_reference(local_logits, shard, _TOP_K)
            )
            sampled = sample_from_global_top_k_logits(global_top_k_logits, temperatures)
            return torch.gather(
                global_top_k_token_ids,
                1,
                sampled.to(torch.int64).unsqueeze(-1),
            )

        def pallas_top_k(
            local_logits=local_logits,
            temperatures=temperatures,
        ) -> torch.Tensor:
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

        reference_ms, pallas_ms = _measure_pair(
            candidate_all_gather_top_k, pallas_top_k, batch_size
        )
        speedup = reference_ms / pallas_ms
        speedups_by_batch_size[batch_size] = speedup
        if rank == 0:
            print(
                f"{batch_size:5d} | {reference_ms:26.4f} ms | "
                f"{pallas_ms:17.4f} ms | "
                f"{speedup:7.3f}x"
            )

    # Gate only the B64+ production shapes. The floor intentionally leaves
    # headroom for shared-runner noise; exact measurements belong in the PR.
    gated_speedups = {
        batch_size: speedups_by_batch_size[batch_size]
        for batch_size in _SPEED_GATED_BATCH_SIZES
    }
    median_speedup = statistics.median(gated_speedups.values())
    if min(gated_speedups.values()) < _MIN_GATED_SHAPE_SPEEDUP:
        raise AssertionError(
            f"Pallas TopK regressed on rank {rank}: "
            f"gated_speedups={gated_speedups}, "
            f"required_each={_MIN_GATED_SHAPE_SPEEDUP:.2f}x"
        )
    required_median_speedup = _MIN_GATED_MEDIAN_SPEEDUP_BY_WORLD_SIZE[world_size]
    if median_speedup < required_median_speedup:
        raise AssertionError(
            f"Pallas TopK missed the median speed gate on rank {rank}: "
            f"gated_speedups={gated_speedups}, "
            f"required_median={required_median_speedup:.2f}x"
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
