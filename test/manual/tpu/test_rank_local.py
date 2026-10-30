"""Run with torchrun on a full TPU mesh; check runtime rank selection numerically."""

import logging
from datetime import timedelta

import torch
import torch.distributed as dist

from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_parallel
from sglang_tpu.compilation.direct_compile import direct_compile
from sglang_tpu.moe.ep_rank import mask_tpu_moe_non_local_experts, traced_moe_ep_rank
from sglang_tpu.moe.pallas.fused_moe_op import _resolve_expert_group_offset
from sglang_tpu.rank_local import rank_slice


def compile_local(fn, args):
    compiled = direct_compile(fn, specialization_key=lambda *args: (), lift_rank=True)
    with torch.no_grad():
        return compiled.warmup_async(*args).wait()


def check_rank_slices(cpu_group, dtype):
    rank, size = dist.get_rank(), dist.get_world_size()
    group_size = size // 2
    rank_stride = min(4, size)
    # Reverse core order to exercise physical ownership independently of TP rank.
    rank_to_device = tuple(reversed(range(size)))
    physical_lane = rank_to_device[rank] % 2
    value_cpu = torch.arange(32 * 128, dtype=dtype).reshape(32, 128) + rank * 10000
    if dtype == torch.int64:
        value_cpu += 1 << 33
    value = value_cpu.to("tpu")
    value.cpu()

    def fn(x):
        return (
            current_platform.get_traced_rank(rank),
            current_platform.get_traced_rank(rank % group_size, group_size=group_size),
            current_platform.slice_tensor_for_rank(x, 0, size, rank),
            current_platform.slice_tensor_for_rank(x, 1, size, rank),
            current_platform.slice_tensor_for_rank(x, 0, 2, rank % 2),
            rank_slice(
                x, 0, size // rank_stride, rank // rank_stride, rank_stride=rank_stride
            ),
            rank_slice(x, 0, 2, physical_lane, rank_to_device=rank_to_device),
        )

    actual = compile_local(fn, (value,))
    actual = [tensor.cpu() for tensor in actual]
    rows, columns = 32 // size, 128 // size
    block_rows, block = rows * rank_stride, rank // rank_stride
    expected = [
        torch.tensor([rank], dtype=torch.int32),
        torch.tensor([rank % group_size], dtype=torch.int32),
        value_cpu[rank * rows : (rank + 1) * rows],
        value_cpu[:, rank * columns : (rank + 1) * columns],
        value_cpu[(rank % 2) * 16 : (rank % 2 + 1) * 16],
        value_cpu[block * block_rows : (block + 1) * block_rows],
        value_cpu[physical_lane * 16 : (physical_lane + 1) * 16],
    ]
    assert all(
        torch.equal(got, want) for got, want in zip(actual, expected, strict=True)
    )
    print(f"rank-input slices: rank={rank} dtype={dtype} PASS", flush=True)
    dist.barrier(group=cpu_group)


def check_expert_mask(cpu_group):
    rank, size = dist.get_rank(), dist.get_world_size()
    ids_cpu = torch.arange(128, dtype=torch.int32).reshape(32, 4) % (size * 4 + 8) - 4
    weights_cpu = (
        torch.arange(128, dtype=torch.float32).reshape(32, 4).to(torch.bfloat16) / 128
    )
    ids, weights = ids_cpu.to("tpu"), weights_cpu.to("tpu")
    ids.cpu()
    weights.cpu()

    def fn(weights, ids):
        # EP spans the world here, so the EP rank is the traced global rank.
        return mask_tpu_moe_non_local_experts(
            weights,
            ids,
            num_local_experts=4,
            moe_ep_rank=rank,
            traced_ep_rank=current_platform.get_traced_rank(rank),
        )

    actual_weights, actual_ids = compile_local(fn, (weights, ids))
    local_ids = ids_cpu - rank * 4
    valid = (local_ids >= 0) & (local_ids < 4)
    assert torch.equal(actual_weights.cpu(), weights_cpu * valid.to(weights_cpu.dtype))
    assert torch.equal(actual_ids.cpu(), torch.where(valid, local_ids, -1))
    print(f"rank-input expert mask: rank={rank} PASS", flush=True)
    dist.barrier(group=cpu_group)


def check_moe_ep_rank_below_tp(cpu_group):
    """The traced TP rank maps to (rank // moe_tp) % ep for EP == TP and EP < TP."""
    rank, size = dist.get_rank(), dist.get_world_size()
    num_local = 4
    ids_cpu = torch.arange(128, dtype=torch.int32).reshape(32, 4) % (size * 4 + 8) - 4
    weights_cpu = (
        torch.arange(128, dtype=torch.float32).reshape(32, 4).to(torch.bfloat16) / 128
    )
    ids, weights = ids_cpu.to("tpu"), weights_cpu.to("tpu")
    ids.cpu()
    weights.cpu()
    for moe_tp_size in (1, 2):
        ep_size = size // moe_tp_size
        ep_rank = rank // moe_tp_size % ep_size
        layout = dict(
            tp_rank=rank,
            tp_size=size,
            moe_dp_size=1,
            moe_ep_size=ep_size,
            moe_tp_size=moe_tp_size,
            moe_ep_rank=ep_rank,
        )

        def fn(weights, ids):
            traced = traced_moe_ep_rank()
            w, i = mask_tpu_moe_non_local_experts(
                weights, ids, num_local, ep_rank, traced_ep_rank=traced
            )
            # The FP8 and W4A16 paths lift their expert window offset instead.
            static, offset = _resolve_expert_group_offset(
                ep_rank * num_local, ep_size * num_local, num_local
            )
            assert static == 0
            return traced, w, i, offset

        with get_parallel().override(**layout):
            traced, actual_weights, actual_ids, offset = compile_local(
                fn, (weights, ids)
            )
        assert torch.equal(traced.cpu(), torch.tensor([ep_rank], dtype=torch.int32))
        assert torch.equal(
            offset.cpu(), torch.tensor([ep_rank * num_local], dtype=torch.int32)
        )
        local_ids = ids_cpu - ep_rank * num_local
        valid = (local_ids >= 0) & (local_ids < num_local)
        assert torch.equal(
            actual_weights.cpu(), weights_cpu * valid.to(weights_cpu.dtype)
        )
        assert torch.equal(actual_ids.cpu(), torch.where(valid, local_ids, -1))
        print(
            f"rank-input moe ep rank: rank={rank} moe_tp={moe_tp_size} "
            f"ep_rank={ep_rank} PASS",
            flush=True,
        )
    dist.barrier(group=cpu_group)


def main():
    logging.basicConfig(level=logging.INFO)
    dist.init_process_group("tpu_dist", timeout=timedelta(seconds=240))
    cpu_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=240))
    assert dist.get_world_size() in (2, 4, 8, 16)
    for dtype in (torch.int32, torch.int64):
        check_rank_slices(cpu_group, dtype)
    check_expert_mask(cpu_group)
    check_moe_ep_rank_below_tp(cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
