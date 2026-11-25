# SPDX-License-Identifier: Apache-2.0
"""Multi-device TPU test for DP-encoder padded all-gather and broadcast isolation.

Exercises ProcessGroupTpu::_allgather_base and ProcessGroupTpu::broadcast via
GroupCoordinator across full, partially filled, empty, slice-view,
inference_mode reshape-view, aten._unsafe_view, squeeze_(), and dirty-stream
ranks with TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false in both
DEFER_NEVER and DEFER_AND_FUSE eager modes.
"""

import importlib.util
import json
import os
import shlex
import sys
import unittest
from datetime import timedelta
from unittest.mock import patch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tpu.evals.eval_support import RecipeLauncher

register_tpu_ci(est_time=90, suite="stage-a-test-tpu-unit", chips=4)

_TARGET = "sglang.srt.multimodal.mm_utils.gather_dp_encoder_rank_embeddings"


def _verify_standalone_mlir_on_tpu(device) -> None:
    import torch
    from torch_tpu._internal import execution_mode
    from torch_tpu._internal.sync.sync import (
        computation_mlir,
        is_materialized,
        is_materializing,
    )
    from torch_tpu._internal.sync.sync import synchronize as tpu_sync

    from sglang_tpu.hooks import dp_encoder_gather as hook_module

    max_len = 8
    ref_full = torch.arange(max_len * 8, dtype=torch.float32).to(torch.bfloat16)
    ref_full = ref_full.reshape(max_len, 8)
    ref_partial = ref_full[:5]

    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
        empty_src = torch.empty((0, 8), dtype=torch.bfloat16, device=device)
        partial_src = ref_partial.to(device)

        slice_backing = torch.zeros(
            (max_len * 2, 8), dtype=torch.bfloat16, device=device
        )
        slice_backing[:max_len].copy_(ref_full.to(device))
        slice_src = slice_backing[:max_len]

        with torch.inference_mode():
            inf_backing = ref_full.reshape(1, max_len, 8).to(device)
            inf_reshape_src = inf_backing.reshape(max_len, 8)

        unsafe_backing = ref_full.reshape(1, max_len, 8).to(device)
        unsafe_src = torch.ops.aten._unsafe_view(unsafe_backing, [max_len, 8])

        squeeze_src = ref_full.reshape(1, max_len, 8).to(device)
        squeeze_src.squeeze_(0)

        cases = [
            ("empty", empty_src, torch.zeros((max_len, 8), dtype=torch.bfloat16)),
            (
                "partial",
                partial_src,
                torch.cat(
                    [ref_partial, torch.zeros((max_len - 5, 8), dtype=torch.bfloat16)],
                    dim=0,
                ),
            ),
            ("slice_view", slice_src, ref_full),
            ("inference_mode_reshape", inf_reshape_src, ref_full),
            ("unsafe_view", unsafe_src, ref_full),
            ("squeeze_inplace", squeeze_src, ref_full),
        ]
        for name, src, want_cpu in cases:
            padded = hook_module._pad_dp_encoder_embeddings_tpu(src, max_len)
            tpu_sync(padded, wait=False)
            assert is_materializing(padded) or is_materialized(padded), name
            probe = padded + 1.0
            mlir = computation_mlir(probe)
            for forbidden in (
                "stablehlo.reshape",
                "stablehlo.slice",
                "dynamic_update_slice",
            ):
                assert forbidden not in mlir, (name, forbidden, mlir)
            assert f"tensor<{max_len}x8xbf16>" in mlir, (name, mlir)
            assert torch.equal(probe.cpu(), want_cpu + 1.0), name


def _wrap_rank_local_view(local_tpu, rank: int, length: int, tail_shape: tuple, device):
    import torch

    if length <= 0:
        return local_tpu
    variant = rank % 4
    if variant == 0:
        backing = torch.randn(
            (length * 2, *tail_shape), dtype=torch.bfloat16, device=device
        )
        backing[:length].copy_(local_tpu)
        return backing[:length]
    if variant == 1:
        with torch.inference_mode():
            reshaped_src = local_tpu.reshape(1, length, *tail_shape)
            return reshaped_src.reshape(length, *tail_shape)
    if variant == 2:
        reshaped_src = local_tpu.reshape(1, length, *tail_shape)
        return torch.ops.aten._unsafe_view(reshaped_src, [length, *tail_shape])
    squeezed = local_tpu.reshape(1, length, *tail_shape).clone()
    squeezed.squeeze_(0)
    return squeezed


def _worker() -> None:
    import torch
    import torch.distributed as dist
    import torch_tpu  # noqa: F401
    from torch_tpu._internal import execution_mode

    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.multimodal import mm_utils
    from sglang.srt.plugins import load_plugins
    from sglang.srt.plugins.hook_registry import HookRegistry
    from sglang.srt.runtime_context import get_parallel
    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.hooks import dp_encoder_gather as hook_module
    from sglang_tpu.models import muse_spark_omni

    torch.ones((1,), device="tpu").cpu()
    load_plugins()
    assert _TARGET in HookRegistry._patched

    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    dist.init_process_group(
        backend="cpu:gloo,tpu:tpu_dist",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )

    group = GroupCoordinator(
        group_ranks=[list(range(world))],
        local_rank=rank,
        torch_distributed_backend="cpu:gloo,tpu:tpu_dist",
        use_pynccl=False,
        use_mscclpp=False,
        use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        group_name="dp_encoder_attn_tp",
    )

    device = torch.device("tpu")
    _verify_standalone_mlir_on_tpu(device)

    shapes_and_lengths = [
        # Full + partial ranks (reproduces j-ed56c5cb-3af0-4de3-83f4 shape family).
        ([512 if r < max(1, world // 4) else 256 for r in range(world)], (16,)),
        # Full + empty ranks (reproduces j-65017e8f-a853-4cb2-946a shape family).
        ([256 if r < world // 2 else 0 for r in range(world)], (16,)),
        # Mix of full, partial, and empty ranks with empty rank 0.
        (
            [0 if r % 3 == 0 else (384 if r == 1 else 96 * r) for r in range(world)],
            (8,),
        ),
        # 3-D rope_2d layout with uneven lengths.
        ([16 * ((r * 3) % (world + 1)) for r in range(world)], (4, 8)),
        # Single non-empty rank at world - 1 (single-owner broadcast fast path).
        ([256 if r == world - 1 else 0 for r in range(world)], (16,)),
        # All-empty fast path.
        ([0 for _ in range(world)], (8,)),
    ]

    for eager_mode in (
        execution_mode.EagerMode.DEFER_NEVER,
        execution_mode.EagerMode.DEFER_AND_FUSE,
    ):
        with (
            execution_mode.set_eager_mode(eager_mode),
            get_parallel().override(
                attn_tp_size=world, attn_tp_rank=rank, attn_tp_group=group
            ),
        ):
            for gather_mode in ("all_gather", "broadcast"):
                with tpu_envs.SGLANG_TPU_DP_ENCODER_GATHER_MODE.override(gather_mode):
                    for lengths, tail_shape in shapes_and_lengths:
                        gen = torch.Generator().manual_seed(
                            sum(lengths) * 17 + len(tail_shape)
                        )
                        cpu_blocks = [
                            torch.randn(
                                (n, *tail_shape), generator=gen, dtype=torch.bfloat16
                            )
                            for n in lengths
                        ]
                        local_tpu = _wrap_rank_local_view(
                            cpu_blocks[rank].to(device),
                            rank,
                            lengths[rank],
                            tail_shape,
                            device,
                        )
                        keep_alive = None
                        if rank % 2 == 1:
                            keep_alive = (
                                torch.ones((8, 8), dtype=torch.bfloat16, device=device)
                                + rank
                            )[:, :4]

                        got = hook_module.gather_dp_encoder_rank_embeddings_tpu(
                            local_tpu, lengths
                        )
                        assert len(got) == world
                        for r, want_cpu in enumerate(cpu_blocks):
                            have_cpu = got[r].cpu()
                            assert have_cpu.shape == want_cpu.shape, (
                                eager_mode,
                                gather_mode,
                                r,
                                have_cpu.shape,
                                want_cpu.shape,
                            )
                            assert torch.equal(have_cpu, want_cpu), (
                                eager_mode,
                                gather_mode,
                                lengths,
                                r,
                            )
                        del keep_alive

            # End-to-end call sites through load_plugins() hook and muse_spark_omni.
            class _TpuVisionTower:
                spatial_merge_size = 2
                out_hidden_size = 8
                dtype = torch.bfloat16

                def __init__(self, dev):
                    self.device = dev

                def __call__(self, pixel_values, grid_thw):
                    rows = pixel_values.shape[0] // 4
                    with torch.inference_mode():
                        summed = pixel_values.reshape(rows, 4, 8).sum(dim=1)
                        return summed.reshape(1, rows, 8).reshape(rows, 8)

            tower = _TpuVisionTower(device)
            grids = [[1, 4, 4], [1, 2, 4], [1, 2, 2], [1, 4, 2], [1, 2, 2]]
            total_patches = sum(t * h * w for t, h, w in grids)
            gen = torch.Generator().manual_seed(2026)
            pixels_cpu = torch.randn(
                (total_patches, 8), generator=gen, dtype=torch.bfloat16
            )
            pixels_tpu = pixels_cpu.to(device)
            expected_sharded = tower(pixels_cpu, grids)

            actual_sharded = mm_utils.run_dp_sharded_mrope_vision_model(
                tower, pixels_tpu, grids, rope_type="rope_3d"
            ).cpu()
            assert torch.equal(actual_sharded, expected_sharded)

            # Single-image sharded call site (owner=0 and owner=world-1) with
            # pending stream ops and inference_mode reshape view from tower.
            single_grid = [[1, 4, 4]]
            single_pixels_cpu = pixels_cpu[:16]
            expected_single = tower(single_pixels_cpu, single_grid)
            for owner in (0, world - 1):
                dirty_single = None
                if rank == owner:
                    dirty_single = (
                        torch.ones((8, 8), dtype=torch.bfloat16, device=device) + 5.0
                    )[:, :4]
                sample_counts_single = [1 if r == owner else 0 for r in range(world)]
                loads_single = [16 if r == owner else 0 for r in range(world)]
                with patch.object(
                    mm_utils,
                    "get_dp_encoder_lb_assignment",
                    return_value=([0], sample_counts_single, loads_single),
                ):
                    actual_single = mm_utils.run_dp_sharded_mrope_vision_model(
                        tower,
                        single_pixels_cpu.to(device),
                        single_grid,
                        rope_type="rope_3d",
                    ).cpu()
                assert torch.equal(actual_single, expected_single), (
                    eager_mode,
                    owner,
                )
                del dirty_single

            # Presharded video call site: only ranks 0 and 1 have units.
            counts = [2, 1] + [0] * (world - 2)
            unit_grids = [[1, 2, 2], [1, 2, 2], [1, 2, 2]]
            presharded_cpu = pixels_cpu[:12]
            if rank == 0:
                local_pv = presharded_cpu[:8].to(device)
                local_grids = [[1, 2, 2], [1, 2, 2]]
            elif rank == 1:
                local_pv = presharded_cpu[8:12].to(device)
                local_grids = [[1, 2, 2]]
            else:
                local_pv = torch.empty((0, 8), dtype=torch.bfloat16, device=device)
                local_grids = []
            actual_presharded = mm_utils.run_dp_presharded_mrope_vision_model(
                tower, local_pv, local_grids, unit_grids, counts
            ).cpu()
            assert torch.equal(actual_presharded, tower(presharded_cpu, unit_grids))

            def _muse_encode_view(xs):
                joined = torch.cat(xs, dim=0)
                with torch.inference_mode():
                    return joined.reshape(1, joined.shape[0], 8).reshape(
                        joined.shape[0], 8
                    )

            # Muse Spark DP-sharded encode multi-item call site.
            muse_items_cpu = [
                pixels_cpu[:12],
                pixels_cpu[12:20],
                pixels_cpu[20:24],
            ]
            actual_muse = muse_spark_omni._run_dp_sharded_encode(
                features=[x.to(device) for x in muse_items_cpu],
                encode_cost_per_item=[12, 8, 4],
                tokens_per_item=[12, 8, 4],
                encode_fn=_muse_encode_view,
                out_hidden_size=8,
                device=device,
                dtype=torch.bfloat16,
            ).cpu()
            assert torch.equal(actual_muse, torch.cat(muse_items_cpu, dim=0))

            # Muse Spark single-item call site (owner=0 and owner=world-1)
            # with pending stream ops on sender rank.
            single_item_cpu = pixels_cpu[:12]
            for owner in (0, world - 1):
                dirty = None
                if rank == owner:
                    dirty = (
                        torch.ones((8, 8), dtype=torch.bfloat16, device=device) + 3.0
                    )[:, :4]
                sample_counts_1 = [1 if r == owner else 0 for r in range(world)]
                loads_1 = [12 if r == owner else 0 for r in range(world)]
                with patch.object(
                    muse_spark_omni,
                    "get_dp_encoder_lb_assignment",
                    return_value=([0], sample_counts_1, loads_1),
                ):
                    actual_muse_single = muse_spark_omni._run_dp_sharded_encode(
                        features=[single_item_cpu.to(device)],
                        encode_cost_per_item=[12],
                        tokens_per_item=[12],
                        encode_fn=_muse_encode_view,
                        out_hidden_size=8,
                        device=device,
                        dtype=torch.bfloat16,
                    ).cpu()
                assert torch.equal(actual_muse_single, single_item_cpu), (
                    eager_mode,
                    owner,
                )
                del dirty

    print(json.dumps({"rank": rank, "world": world, "status": "ok"}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


_SERVING_LIBTPU_INIT_ARGS = " ".join(
    (
        "--xla_tpu_enable_async_collective_fusion=true",
        "--xla_tpu_overlap_compute_collective_tc=true",
        "--xla_tpu_enable_async_collective_fusion_multiple_steps=true",
        "--xla_tpu_enable_latency_hiding_scheduler=true",
        "--xla_latency_hiding_scheduler_rerun=10",
        "--xla_tpu_all_gather_collective_matmul_mode=post_spmd_conservative",
        "--xla_tpu_reduce_scatter_collective_matmul_mode=post_spmd_conservative",
        "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false",
        "--xla_tpu_offload_gather_to_sparsecore=false",
        "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false",
    )
)


class TestDpEncoderPaddedAllGather(CustomTestCase):
    def test_padded_all_gather_completes_and_matches_across_ranks(self):
        if importlib.util.find_spec("torch_tpu") is None:
            self.skipTest("requires TorchTPU and a dedicated TPU host")
        import torch
        import torch_tpu  # noqa: F401

        if torch.accelerator.device_count() < 2:
            self.skipTest("requires at least two TPU devices")

        python = shlex.quote(sys.executable)
        command = "\n".join(
            (
                "set -euo pipefail",
                f"eval $({python} -m torch.tpu.distributed)",
                "export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE",
                f"{python} -m torch.distributed.run --standalone "
                f'--nproc-per-node="$WORLD_SIZE" {shlex.quote(__file__)} --worker',
            )
        )
        env = {
            **os.environ,
            "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS": "false",
            "LIBTPU_INIT_ARGS": (
                os.environ.get("LIBTPU_INIT_ARGS") or _SERVING_LIBTPU_INIT_ARGS
            ),
            "OMP_NUM_THREADS": "1",
        }
        for name in ("RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "WORLD_SIZE"):
            env.pop(name, None)

        processes = RecipeLauncher(
            command=["bash", "-c", command],
            start_new_session=True,
            terminate_timeout=10,
            tag="dp-encoder-padded-all-gather",
        )
        with patch.dict(os.environ, env, clear=True):
            try:
                processes.start()
                result = processes.wait(timeout=240)
            finally:
                processes.shutdown()
        self.assertEqual(result, 0, "TPU DP-encoder padded all-gather workers failed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _worker()
        sys.stderr.flush()
        os._exit(0)
    else:
        unittest.main()
