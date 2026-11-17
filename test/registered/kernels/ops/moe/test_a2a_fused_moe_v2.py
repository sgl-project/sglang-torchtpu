# SPDX-License-Identifier: Apache-2.0
"""Selected-route GLM EP kernel through the real TorchTPU export bridge.

Run with the candidate's absolute interpreter. The launcher uses that same
interpreter for all workers; their runtime evidence is recorded in this process,
not inferred from a separate probe. Full-model E2E is a separate acceptance.
"""

import os
import shlex
import subprocess
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(
    est_time=1200,
    suite="nightly-tpu-4-v7",
    nightly=True,
    chips=4,
    labels=["singlehost", "glm-5.2", "dpa"],
)


def full_shape_cases(ep, rank):
    """Real checkpoint geometry, with an analytic same-weight reference.

    Each expert has distinct FP8 constant slabs and nonuniform channel scales.
    This exercises the full H/I/E traffic and routing without materializing all
    E weights on every host process just to compute the reference.
    """
    import json
    import time

    import numpy as np
    import torch

    from sglang_tpu.moe.pallas.a2a_fused_moe_v2.op import get_op

    h, inter, e, k = 6144, 2048, 256, 8
    assert e % ep == 0
    local_e = e // ep
    ownership_rank = (rank * 3 + 1) % ep
    constants = []
    for expert in range(e):
        values = torch.tensor(
            [
                0.01 * (1 + expert % 7 / 16),
                0.015 * (1 + expert % 5 / 16),
                0.01 * ((expert % 3) - 1),
            ]
        )
        constants.append(values.to(torch.float8_e4m3fn).float())
    constants = torch.stack(constants)
    w13 = torch.empty((2, local_e, h, inter), dtype=torch.float8_e4m3fn)
    w2 = torch.empty((local_e, inter, h), dtype=torch.float8_e4m3fn)
    for local in range(local_e):
        values = constants[ownership_rank * local_e + local]
        for branch in range(2):
            w13[branch, local].copy_(
                torch.full((h, inter), float(values[branch])).to(torch.float8_e4m3fn)
            )
        w2[local].copy_(
            torch.full((inter, h), float(values[2])).to(torch.float8_e4m3fn)
        )
    s13 = (
        torch.linspace(0.7, 1.3, 2 * inter)
        .reshape(2, 1, inter)
        .expand(2, local_e, inter)
        .bfloat16()
        .contiguous()
    )
    s2 = (
        torch.linspace(0.8, 1.2, h)
        .reshape(1, 1, 1, h)
        .expand(local_e, 1, 1, h)
        .bfloat16()
        .contiguous()
    )
    gate_scale, up_scale = s13[:, 0].float()
    down_scale = s2[0, 0, 0].float()
    operands = [a.to("tpu") for a in (w13, w2, s13, s2)]
    del w13, w2, s13, s2
    shared_gate = torch.tensor(0.012).to(torch.float8_e4m3fn).float()
    shared_up = torch.tensor(0.016).to(torch.float8_e4m3fn).float()
    shared_down = torch.tensor(0.008).to(torch.float8_e4m3fn).float()
    sw13 = torch.stack(
        [
            torch.full((h, inter), float(c)).to(torch.float8_e4m3fn)
            for c in (shared_gate, shared_up)
        ]
    )
    sw2 = torch.full((inter, h), float(shared_down)).to(torch.float8_e4m3fn)
    ss13 = torch.linspace(0.6, 1.4, 2 * inter).reshape(2, inter)
    ss2 = torch.linspace(0.7, 1.3, h)
    shared = [a.to("tpu") for a in (sw13, sw2, ss13, ss2)]
    del sw13, sw2
    rank_arg = torch.tensor([ownership_rank], dtype=torch.int32, device="tpu")
    stub = torch.empty(0, device="tpu")
    generator = torch.Generator().manual_seed(61442048)
    reports = []
    for local_rows, has_shared in [(6, False), (24, True), (64, True), (2048, True)]:
        rows = ep * local_rows
        x = (torch.randn((rows, h), generator=generator) * 0.2 + 0.02).bfloat16()
        ids = torch.stack(
            [torch.randperm(e, generator=generator)[:k] for _ in range(rows)]
        ).int()
        weights = torch.rand((rows, k), generator=generator)
        weights /= weights.sum(-1, keepdim=True)
        sums = x.float().sum(-1)
        expected = torch.zeros((rows, h))
        for slot in range(k):
            c = constants[ids[:, slot].long()]
            gate = sums[:, None] * c[:, 0, None] * gate_scale
            up = sums[:, None] * c[:, 1, None] * up_scale
            scalar = (torch.nn.functional.silu(gate) * up).sum(-1) * c[:, 2]
            expected += scalar[:, None] * down_scale * weights[:, slot, None]
        if has_shared:
            gate = sums[:, None] * shared_gate * ss13[0]
            up = sums[:, None] * shared_up * ss13[1]
            expected += (
                (torch.nn.functional.silu(gate) * up).sum(-1)[:, None]
                * shared_down
                * ss2
            )
        args = [
            x.to("tpu"),
            ids.to("tpu"),
            weights.to("tpu"),
            *operands,
            rank_arg,
            *(shared if has_shared else (stub,) * 4),
        ]
        op = get_op(ep, has_shared)
        got = op(*args)
        torch.accelerator.synchronize()
        elapsed = []
        for _ in range(5):
            start = time.perf_counter()
            got = op(*args)
            torch.accelerator.synchronize()
            elapsed.append((time.perf_counter() - start) * 1000)
        got = got.cpu().float()
        error = float(
            torch.linalg.vector_norm(got - expected)
            / torch.linalg.vector_norm(expected).clamp_min(1e-8)
        )
        assert torch.isfinite(got).all() and error < 0.15, (
            "full_shape",
            local_rows,
            error,
        )
        report = dict(
            event="full_shape_pass",
            rank=rank,
            h=h,
            inter=inter,
            experts=e,
            topk=k,
            local_rows=local_rows,
            shared=has_shared,
            relative_l2=error,
            median_ms=float(np.median(elapsed)),
            range_ms=[min(elapsed), max(elapsed)],
        )
        reports.append(report)
        print(json.dumps(report), flush=True)
    return reports


def idle_bucket_cases(
    ep, rank, operands, rank_tensor, w13, w2, s13, s2, sw13, sw2, ss13, ss2, shared
):
    """Match the TPU FFN caller: consume a nonempty fixed DP slot while idle.

    The real compiled-forward shim sets every local DP buffer to pad_t, and
    speculative idle normalization keeps outputs padded. It does not crop the
    MoE result to a zero-row local tensor inside the compiled graph. Exercise
    the same nonempty index_select boundary as live DPs drain and refill.
    """
    import json
    import time

    import torch

    from sglang_tpu.moe.pallas.a2a_fused_moe_v2.op import get_op

    h = w2.shape[-1]
    e = w2.shape[0]
    k, bucket = 4, 64
    generator = torch.Generator().manual_seed(640064)
    x = (torch.randn((ep * bucket, h), generator=generator) * 0.2).bfloat16()
    ids = torch.stack(
        [torch.randperm(e, generator=generator)[:k] for _ in range(ep * bucket)]
    ).int()
    weights = torch.rand((ep * bucket, k), generator=generator)
    weights /= weights.sum(-1, keepdim=True)
    indices = (torch.arange(bucket) + rank * bucket).to(device="tpu", dtype=torch.int64)
    op = get_op(ep, True)

    def ffn_slot(
        x_arg,
        ids_arg,
        coefficients,
        gate_up,
        down,
        gate_scale,
        down_scale,
        logical_rank,
        shared_gate_up,
        shared_down,
        shared_gate_scale,
        shared_down_scale,
        slot,
    ):
        return op(
            x_arg,
            ids_arg,
            coefficients,
            gate_up,
            down,
            gate_scale,
            down_scale,
            logical_rank,
            shared_gate_up,
            shared_down,
            shared_gate_scale,
            shared_down_scale,
        ).index_select(0, slot)

    import torch_tpu._internal.compile  # noqa: F401

    compiled = torch.compile(ffn_slot, backend="tpu", fullgraph=True, dynamic=False)
    reports = []
    for live in (ep, ep // 2, 1, ep):
        mx, mi, mw = x.clone(), ids.clone(), weights.clone()
        mx[live * bucket :] = 0
        mi[live * bucket :] = -1
        mw[live * bucket :] = 0
        lx, li, lw = (a[rank * bucket : (rank + 1) * bucket] for a in (mx, mi, mw))
        expected = torch.zeros((bucket, h))
        for expert in range(e):
            token, column = torch.where(li == expert)
            if token.numel() == 0:
                continue
            gate, up = (
                lx[token].float() @ (w13[expert].float() * s13[expert, 0, 0])
            ).chunk(2, -1)
            out = (torch.nn.functional.silu(gate) * up) @ (
                w2[expert].float() * s2[expert, 0, 0]
            )
            expected.index_add_(0, token, out * lw[token, column, None])
        gate = lx.float() @ (sw13[0].float() * ss13[0])
        up = lx.float() @ (sw13[1].float() * ss13[1])
        expected += (torch.nn.functional.silu(gate) * up) @ (sw2.float() * ss2)
        args = [
            mx.to("tpu"),
            mi.to("tpu"),
            mw.to("tpu"),
            *operands,
            rank_tensor,
            *shared,
            indices,
        ]
        # First warm call also resolves the diagnostic wrapper outside Dynamo.
        ffn_slot(*args)
        torch.accelerator.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            got = compiled(*args)
        torch.accelerator.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        got = got.cpu().float()
        error = float(
            torch.linalg.vector_norm(got - expected)
            / torch.linalg.vector_norm(expected).clamp_min(1e-8)
        )
        assert torch.isfinite(got).all() and error < 0.15, (
            "idle_bucket",
            live,
            rank,
            error,
        )
        if rank >= live:
            assert torch.count_nonzero(got) == 0
        report = dict(
            event="idle_bucket_pass",
            rank=rank,
            live_dp_ranks=live,
            padded_rows=bucket,
            physical_result_rows=got.shape[0],
            relative_l2=error,
            first_compiled_or_replay_ms=elapsed,
        )
        reports.append(report)
        print(json.dumps(report), flush=True)
    return reports


def binder_loading_cases(ep, rank):
    """Run the real post-load storage and shared-gather operations on TPU."""
    import json
    from types import SimpleNamespace

    import torch
    from torch.nn import Parameter

    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang_tpu.moe.glm_a2a_fused_moe_v2 import _prepare_shared, _replace_data
    from sglang_tpu.moe.pallas.a2a_fused_moe_v2.op import (
        pack_gate_up,
        pack_gate_up_scale,
    )

    cpu = ((torch.arange(2 * 128 * 256).reshape(2, 128, 256) % 31).float() * 0.001).to(
        torch.float8_e4m3fn
    )
    parameter = Parameter(cpu.to("tpu"), requires_grad=False)
    marker = object()
    parameter.loader_marker = marker
    old = parameter.data
    old_storage = old.untyped_storage()
    packed = pack_gate_up(old)
    old_pointer, new_pointer = (
        old_storage.data_ptr(),
        packed.untyped_storage().data_ptr(),
    )
    _replace_data(parameter, packed)
    torch.accelerator.synchronize()
    assert parameter.loader_marker is marker
    assert torch.equal(
        parameter.cpu().view(torch.uint8), pack_gate_up(cpu).view(torch.uint8)
    )
    if old_pointer != new_pointer:
        assert old_storage.nbytes() == 0
    scale_cpu = torch.linspace(0.7, 1.3, 2 * 256).reshape(2, 1, 1, 256).bfloat16()
    scale = Parameter(scale_cpu.to("tpu"), requires_grad=False)
    _replace_data(scale, pack_gate_up_scale(scale.data))
    torch.accelerator.synchronize()
    assert torch.equal(scale.cpu(), pack_gate_up_scale(scale_cpu))

    # Create the real coordinator, not a mocked collective. It invokes the same
    # registered TPU all_gather_into_tensor path used by model preparation.
    group = GroupCoordinator(
        group_ranks=[list(range(ep))],
        local_rank=int(os.environ["LOCAL_RANK"]),
        torch_distributed_backend="tpu_dist",
        use_pynccl=False,
        use_mscclpp=False,
        use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        group_name="a2a-fused-moe-v2-loading",
    )
    try:
        h, inter = 256, 2048
        gate = (
            ((torch.arange(h * inter).reshape(h, inter) % 23).float() - 11) * 0.002
        ).to(torch.float8_e4m3fn)
        up = (
            ((torch.arange(h * inter).reshape(h, inter) % 19).float() - 9) * 0.003
        ).to(torch.float8_e4m3fn)
        down = (
            ((torch.arange(inter * h).reshape(inter, h) % 17).float() - 8) * 0.004
        ).to(torch.float8_e4m3fn)
        gate_scale = torch.linspace(0.6, 1.2, inter)
        up_scale = torch.linspace(0.8, 1.4, inter)
        down_scale = torch.linspace(0.7, 1.3, h)
        width = inter // ep
        shard = slice(rank * width, (rank + 1) * width)
        fixture_moe = torch.nn.Module()
        fixture_moe._shared_expert_tp1 = False
        fixture_moe.shared_experts = torch.nn.Module()
        gate_up = fixture_moe.shared_experts.gate_up_proj = torch.nn.Module()
        gate_up.weight = Parameter(
            torch.cat((gate[:, shard], up[:, shard]), 1).contiguous().to("tpu"),
            requires_grad=False,
        )
        gate_up.weight_scale = Parameter(
            torch.cat((gate_scale[shard], up_scale[shard])).to("tpu"),
            requires_grad=False,
        )
        gate_up.weight_block_size = None
        projection = fixture_moe.shared_experts.down_proj = torch.nn.Module()
        projection.weight = Parameter(
            down[shard].contiguous().to("tpu"), requires_grad=False
        )
        projection.weight_scale = Parameter(down_scale.to("tpu"), requires_grad=False)
        projection.weight_block_size = None
        assert _prepare_shared(fixture_moe, SimpleNamespace(tp_group=group, tp_size=ep))
        torch.accelerator.synchronize()
        assert torch.equal(
            fixture_moe._a2a_moe_v2_sw13.cpu().view(torch.uint8),
            torch.stack((gate, up)).view(torch.uint8),
        )
        assert torch.equal(
            fixture_moe._a2a_moe_v2_sw2.cpu().view(torch.uint8), down.view(torch.uint8)
        )
        assert torch.equal(
            fixture_moe._a2a_moe_v2_ss13.cpu(), torch.stack((gate_scale, up_scale))
        )
        assert torch.equal(fixture_moe._a2a_moe_v2_ss2.cpu(), down_scale)
        print(
            json.dumps(
                dict(
                    event="binder_loading_pass",
                    rank=rank,
                    ep=ep,
                    packed_storage_released=old_pointer != new_pointer,
                    loader_attributes_preserved=True,
                    shared_fp8_gather_bit_exact=True,
                    shared_geometry=[h, inter],
                )
            ),
            flush=True,
        )
    finally:
        group.destroy()


def worker():
    import json
    import time
    from importlib.metadata import version

    import jax
    import jaxlib
    import libtpu
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch_tpu

    from sglang_tpu.moe.glm_a2a_fused_moe_v2 import _replace_data
    from sglang_tpu.moe.pallas.a2a_fused_moe_v2.op import (
        get_op,
        pack_gate_up,
        pack_gate_up_scale,
    )

    dist.init_process_group(backend="tpu_dist")
    rank, ep = dist.get_rank(), dist.get_world_size()
    print(
        json.dumps(
            {
                "event": "worker_runtime",
                "rank": rank,
                "executable": sys.executable,
                "torch_file": torch.__file__,
                "torch_tpu_file": torch_tpu.__file__,
                "jax_file": jax.__file__,
                "jaxlib_file": jaxlib.__file__,
                "libtpu_file": libtpu.__file__,
                "jax_module_version": jax.__version__,
                "jaxlib_module_version": jaxlib.__version__,
                "versions": {
                    p: version(p)
                    for p in ("torch", "torch-tpu", "jax", "jaxlib", "libtpu")
                },
                "compiler": {
                    k: os.environ.get(k)
                    for k in ("JAX_PLATFORMS", "XLA_FLAGS", "LIBTPU_INIT_ARGS")
                },
            }
        ),
        flush=True,
    )
    torch.set_num_threads(1)
    binder_loading_cases(ep, rank)
    generator = torch.Generator().manual_seed(2198)
    # Small weights permit a complete independent float32 CPU reference on
    # every worker, including adversarial routing and decode padding shapes.
    h, inter, e, k = 512, 512, ep * 2, 4
    w13 = (torch.randn((e, h, 2 * inter), generator=generator) * 0.02).to(
        torch.float8_e4m3fn
    )
    w2 = (torch.randn((e, inter, h), generator=generator) * 0.02).to(
        torch.float8_e4m3fn
    )
    s13 = torch.linspace(0.7, 1.3, e * 2 * inter).reshape(e, 1, 1, 2 * inter).bfloat16()
    s2 = torch.linspace(0.8, 1.2, e * h).reshape(e, 1, 1, h).bfloat16()
    ownership_rank = (rank * 3 + 1) % ep
    sl = slice(ownership_rank * 2, (ownership_rank + 1) * 2)
    # These are the real post-load Parameter operations, consumed directly by
    # the subsequent bridge calls. Do not only check pack via a CPU copy and
    # then use unrelated CPU-packed weights for all numerical cases.
    packed_parameter = torch.nn.Parameter(w13[sl].to("tpu"), requires_grad=False)
    scale_parameter = torch.nn.Parameter(s13[sl].to("tpu"), requires_grad=False)
    _replace_data(packed_parameter, pack_gate_up(packed_parameter.data))
    _replace_data(scale_parameter, pack_gate_up_scale(scale_parameter.data))
    device_weights = [
        packed_parameter.data,
        w2[sl].to("tpu"),
        scale_parameter.data,
        s2[sl].to("tpu"),
    ]
    rank_tensor = torch.tensor([ownership_rank], dtype=torch.int32, device="tpu")
    stub = torch.empty(0, device="tpu")
    sw13 = (torch.randn((2, h, inter), generator=generator) * 0.02).to(
        torch.float8_e4m3fn
    )
    sw2 = (torch.randn((inter, h), generator=generator) * 0.02).to(torch.float8_e4m3fn)
    ss13 = torch.linspace(0.6, 1.4, 2 * inter).reshape(2, inter)
    ss2 = torch.linspace(0.7, 1.3, h)
    device_shared = [a.to("tpu") for a in (sw13, sw2, ss13, ss2)]
    reports = []
    for local_rows, routing, has_shared in [
        (6, "random", False),
        (12, "hot", False),
        (18, "zeros", False),
        (24, "padded", False),
        (24, "partial", False),
        (65, "random", False),
        (6, "random", True),
        (24, "padded", True),
        (24, "partial", True),
        (65, "hot", True),
    ]:
        rows = local_rows * ep - (3 if routing == "padded" else 0)
        x = (torch.randn((rows, h), generator=generator) * 0.2).to(torch.bfloat16)
        ids = torch.stack(
            [torch.randperm(e, generator=generator)[:k] for _ in range(rows)]
        ).int()
        weights = torch.rand((rows, k), generator=generator)
        weights /= weights.sum(-1, keepdim=True)
        if routing == "hot":
            ids[:] = torch.arange(k)
        if routing == "zeros":
            weights.zero_()
        if routing == "partial":
            ids[::3, 0] = -1
            ids[1::3, 1:3] = -1
            weights[ids < 0] = 0
        if routing == "padded":
            ids[-7:] = -1
            weights[-7:] = 0
            x[-7:] = 0
        expected = torch.zeros(rows, h)
        for expert in range(e):
            tokens, columns = torch.where(ids == expert)
            if tokens.numel() == 0:
                continue
            gate_up = x[tokens].float() @ (w13[expert].float() * s13[expert, 0, 0])
            gate, up = gate_up.chunk(2, -1)
            y = (torch.nn.functional.silu(gate) * up) @ (
                w2[expert].float() * s2[expert, 0, 0]
            )
            expected.index_add_(0, tokens, y * weights[tokens, columns, None])
        if has_shared:
            gate = x.float() @ (sw13[0].float() * ss13[0])
            up = x.float() @ (sw13[1].float() * ss13[1])
            expected += (torch.nn.functional.silu(gate) * up) @ (sw2.float() * ss2)
        args = [
            x.to("tpu"),
            ids.to("tpu"),
            weights.to("tpu"),
            *device_weights,
            rank_tensor,
            *(device_shared if has_shared else (stub, stub, stub, stub)),
        ]
        op = get_op(ep, has_shared)
        got = op(*args)
        torch.accelerator.synchronize()
        times = []
        for _ in range(5):
            started = time.perf_counter()
            got = op(*args)
            torch.accelerator.synchronize()
            times.append((time.perf_counter() - started) * 1e3)
        got = got.cpu().float()
        relative_l2 = float(
            torch.linalg.vector_norm(got - expected)
            / torch.linalg.vector_norm(expected).clamp_min(1e-8)
        )
        assert torch.isfinite(got).all(), (local_rows, routing)
        assert relative_l2 < 0.15, (local_rows, routing, relative_l2)
        if routing in ("zeros", "padded"):
            checked = got if routing == "zeros" else got[-7:]
            assert torch.count_nonzero(checked) == 0, (routing, checked.abs().max())
        report = {
            "local_rows": local_rows,
            "global_rows": rows,
            "routing": routing,
            "shared": has_shared,
            "relative_l2": relative_l2,
            "median_ms": float(np.median(times)),
            "range_ms": [min(times), max(times)],
        }
        reports.append(report)
        print(json.dumps({"event": "case_pass", "rank": rank, **report}), flush=True)
    reports.extend(
        idle_bucket_cases(
            ep,
            rank,
            device_weights,
            rank_tensor,
            w13,
            w2,
            s13,
            s2,
            sw13,
            sw2,
            ss13,
            ss2,
            device_shared,
        )
    )
    reports.extend(full_shape_cases(ep, rank))
    maps = Path("/proc/self/maps").read_text()
    libraries = sorted(
        {line.split()[-1] for line in maps.splitlines() if "libtpu" in line}
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "rank": rank,
                "cases": reports,
                "mapped_libtpu": libraries,
            }
        ),
        flush=True,
    )
    dist.destroy_process_group()


class TestA2AFusedMoEV2(CustomTestCase):
    def test_selected_route_torch_bridge(self):
        if "LOCAL_RANK" in os.environ:
            worker()
            return
        python = shlex.quote(sys.executable)
        script = shlex.quote(str(Path(__file__).resolve()))
        command = (
            "unset RANK LOCAL_RANK LOCAL_WORLD_SIZE WORLD_SIZE; "
            f"eval $({python} -m torch.tpu.distributed); "
            "export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE; "
            'export JAX_PLATFORMS=cpu; export XLA_FLAGS="--xla_force_host_platform_device_count=$WORLD_SIZE"; '
            f'{python} -m torch.distributed.run --standalone --nproc_per_node="$WORLD_SIZE" {script}'
        )
        subprocess.run(["bash", "-lc", command], check=True, timeout=1800)


if __name__ == "__main__":
    unittest.main()
