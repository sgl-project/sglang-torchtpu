"""EP32 GLM decode comparison through the real TorchTPU compiler."""

import importlib.metadata as metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time


def main():
    from sglang_tpu.patches import _prepare_tpu_env

    _prepare_tpu_env(world_size=32)
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    import torch_tpu
    import jax
    import jaxlib
    import libtpu

    from adapters import baseline_op
    from sglang_tpu.fp8_linear import apply_tpu_fp8_linear
    from sglang_tpu.moe.pallas.fused_moe_v2.selected_op import get_selected_op
    from sglang_tpu.moe.pallas.glm_moe_v2.op import (
        get_op,
        pack_gate_up,
        pack_gate_up_scale,
    )

    torch.set_num_threads(1)
    dist.init_process_group(backend="tpu_dist")
    rank, ep = dist.get_rank(), dist.get_world_size()
    assert ep == 32
    h, inter, experts, topk = 6144, 2048, 256, 8
    local_experts = experts // ep
    local_inter = inter // ep
    out = Path(os.environ["ARTIFACT_LOCAL_DIR"]) / f"rank-{rank}" / "benchmark"
    out.mkdir(parents=True, exist_ok=True)

    def emit(record):
        record["rank"] = rank
        with (out / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    versions = {
        name: metadata.version(name)
        for name in ("torch", "torch_tpu", "jax", "jaxlib", "libtpu")
    }
    assert versions["jax"] == versions["jaxlib"] == "0.11.2"
    assert versions["libtpu"] == "0.0.50"
    runtime = dict(
        executable=sys.executable,
        versions=versions,
        source=os.environ["SOURCE_COMMIT"],
        imports={
            m.__name__: m.__file__ for m in (torch, torch_tpu, jax, jaxlib, libtpu)
        },
        flags={
            k: os.environ.get(k)
            for k in ("XLA_FLAGS", "LIBTPU_INIT_ARGS", "JAX_PLATFORMS")
        },
    )
    (out / "runtime.json").write_text(json.dumps(runtime, indent=2))
    emit(dict(kind="runtime", **runtime))

    # Constant expert slabs retain the full weight traffic and support an
    # independent analytic reference without dense CPU matmuls over all experts.
    constants = torch.stack(
        [
            torch.tensor(
                [
                    0.01 * (1 + e % 7 / 16),
                    0.015 * (1 + e % 5 / 16),
                    0.01 * ((e % 3) - 1),
                ]
            )
            .to(torch.float8_e4m3fn)
            .float()
            for e in range(experts)
        ]
    )
    w13_cpu = torch.empty((local_experts, h, 2 * inter), dtype=torch.float8_e4m3fn)
    w2_cpu = torch.empty((local_experts, inter, h), dtype=torch.float8_e4m3fn)
    for local in range(local_experts):
        values = constants[rank * local_experts + local]
        w13_cpu[local, :, :inter].fill_(float(values[0]))
        w13_cpu[local, :, inter:].fill_(float(values[1]))
        w2_cpu[local].fill_(float(values[2]))
    s13_cpu = (
        torch.linspace(0.7, 1.3, 2 * inter)
        .reshape(1, 1, 1, -1)
        .expand(local_experts, 1, 1, -1)
        .bfloat16()
        .contiguous()
    )
    s2_cpu = (
        torch.linspace(0.8, 1.2, h)
        .reshape(1, 1, 1, -1)
        .expand(local_experts, 1, 1, -1)
        .bfloat16()
        .contiguous()
    )
    w13, w2, s13, s2 = [t.to("tpu") for t in (w13_cpu, w2_cpu, s13_cpu, s2_cpu)]
    packed13 = pack_gate_up(w13)
    packed_scale = pack_gate_up_scale(s13)
    del w13_cpu, w2_cpu

    shared_values = torch.tensor([0.012, 0.016, 0.008]).to(torch.float8_e4m3fn).float()
    shared_scales = torch.linspace(0.6, 1.4, 2 * inter).reshape(2, inter)
    shared_down_scale = torch.linspace(0.7, 1.3, h)
    sw13_cpu = torch.stack(
        [
            torch.full((h, inter), float(c)).to(torch.float8_e4m3fn)
            for c in shared_values[:2]
        ]
    )
    sw2_cpu = torch.full((inter, h), float(shared_values[2])).to(torch.float8_e4m3fn)
    sw13, sw2, ss13, ss2 = [
        t.to("tpu") for t in (sw13_cpu, sw2_cpu, shared_scales, shared_down_scale)
    ]
    first = rank * local_inter
    shared_w13 = (
        torch.cat(
            [
                sw13_cpu[0, :, first : first + local_inter],
                sw13_cpu[1, :, first : first + local_inter],
            ],
            dim=1,
        )
        .contiguous()
        .to("tpu")
    )
    shared_s13 = (
        torch.cat(
            [
                shared_scales[0, first : first + local_inter],
                shared_scales[1, first : first + local_inter],
            ]
        )
        .contiguous()
        .to("tpu")
    )
    shared_w2 = sw2_cpu[first : first + local_inter].contiguous().to("tpu")
    del sw13_cpu, sw2_cpu
    logical_rank = torch.tensor([rank], dtype=torch.int32, device="tpu")
    baseline = baseline_op()
    jax_glm = get_op(ep, True)
    existing = {
        capacity: get_selected_op(ep, capacity) for capacity in (32, 64, 128)
    }
    runtime["mapped_libtpu"] = sorted(
        set(
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "libtpu.so" in line
        )
    )
    assert runtime["mapped_libtpu"], "actual worker did not map libtpu"
    (out / "runtime.json").write_text(json.dumps(runtime, indent=2))

    def shared_partial(x):
        gate_up = apply_tpu_fp8_linear(x, shared_w13, shared_s13)
        gate, up = gate_up.chunk(2, dim=-1)
        return apply_tpu_fp8_linear(F.silu(gate) * up, shared_w2, ss2)

    def reduce_to_dp(partial):
        output = torch.empty(
            (partial.shape[0] // ep, h), dtype=partial.dtype, device=partial.device
        )
        dist.reduce_scatter_tensor(output, partial.contiguous())
        return output

    def baseline_full(x, ids, coefficients):
        routed = baseline(x, ids, coefficients, w13, w2, s13, s2, logical_rank)
        return reduce_to_dp(routed + shared_partial(x))

    def jax_full(x, ids, coefficients):
        result = jax_glm(
            x,
            ids,
            coefficients,
            packed13,
            w2,
            packed_scale,
            s2,
            logical_rank,
            sw13,
            sw2,
            ss13,
            ss2,
        )
        rows = x.shape[0] // ep
        return result[rank * rows : (rank + 1) * rows]

    def existing_full(capacity, x, ids, coefficients):
        result = existing[capacity](
            x, ids, coefficients, w13, w2, s13, s2, logical_rank
        )
        rows = x.shape[0] // ep
        local_shared = reduce_to_dp(shared_partial(x))
        return result[rank * rows : (rank + 1) * rows] + local_shared

    from functools import partial

    functions = dict(gmm_baseline=baseline_full, jax_glm_v2=jax_full)
    functions.update(
        {
            f"existing_v2_c{capacity}": partial(existing_full, capacity)
            for capacity in existing
        }
    )
    for local_rows in (4, 6, 12, 18, 24):
        rows = local_rows * ep
        generator = torch.Generator().manual_seed(61442048 + local_rows)
        x_cpu = (torch.randn((rows, h), generator=generator) * 0.2).bfloat16()
        # Continuous random scores give uniform distinct expert sets without
        # constructing one randperm per token.
        ids_cpu = (
            torch.rand((rows, experts), generator=generator)
            .topk(topk, dim=1)
            .indices.int()
        )
        coefficients_cpu = torch.rand((rows, topk), generator=generator)
        coefficients_cpu /= coefficients_cpu.sum(dim=-1, keepdim=True)
        coefficients_cpu = coefficients_cpu.bfloat16().float()
        inputs = (x_cpu.to("tpu"), ids_cpu.to("tpu"), coefficients_cpu.to("tpu"))

        start_row, end_row = rank * local_rows, (rank + 1) * local_rows
        sums = x_cpu[start_row:end_row].float().sum(dim=-1)
        reference = torch.zeros((local_rows, h))
        gate_scale, up_scale = s13_cpu[0, 0, 0].float().reshape(2, inter)
        for slot in range(topk):
            c = constants[ids_cpu[start_row:end_row, slot].long()]
            gate = sums[:, None] * c[:, 0, None] * gate_scale
            up = sums[:, None] * c[:, 1, None] * up_scale
            scalar = (F.silu(gate) * up).sum(dim=-1) * c[:, 2]
            reference += (
                scalar[:, None]
                * s2_cpu[0, 0, 0].float()
                * coefficients_cpu[start_row:end_row, slot, None]
            )
        shared_gate = sums[:, None] * shared_values[0] * shared_scales[0]
        shared_up = sums[:, None] * shared_values[1] * shared_scales[1]
        reference += (
            (F.silu(shared_gate) * shared_up).sum(dim=-1)[:, None]
            * shared_values[2]
            * shared_down_scale
        )
        del x_cpu, ids_cpu, coefficients_cpu

        compiled = {}
        for name, fn in functions.items():
            emit(
                dict(
                    kind="compile_begin",
                    variant=name,
                    local_rows=local_rows,
                    global_rows=rows,
                )
            )
            started = time.perf_counter()
            with torch.no_grad():
                fn(*inputs)  # Resolve each custom op before Dynamo captures it.
                torch.accelerator.synchronize()
                compiled[name] = torch.compile(
                    fn, backend="tpu", fullgraph=True, dynamic=False
                )
                result = compiled[name](*inputs)
                torch.accelerator.synchronize()
            emit(
                dict(
                    kind="compile",
                    variant=name,
                    local_rows=local_rows,
                    compile_first_s=time.perf_counter() - started,
                )
            )
            actual = result.cpu().float()
            relative_l2 = float(
                torch.linalg.vector_norm(actual - reference)
                / torch.linalg.vector_norm(reference).clamp_min(1e-8)
            )
            assert torch.isfinite(actual).all() and relative_l2 < 0.15, (
                name,
                rank,
                local_rows,
                relative_l2,
            )
            emit(
                dict(
                    kind="correctness",
                    variant=name,
                    local_rows=local_rows,
                    relative_l2=relative_l2,
                    all_finite=True,
                )
            )
            with torch.no_grad():
                for _ in range(3):
                    compiled[name](*inputs)
                    torch.accelerator.synchronize()
            dist.barrier()

        samples = {name: [] for name in compiled}
        names = list(compiled)
        with torch.no_grad():
            for iteration in range(20):
                order = names if iteration % 2 == 0 else names[::-1]
                for name in order:
                    dist.barrier()
                    torch.accelerator.synchronize()
                    started = time.perf_counter_ns()
                    result = compiled[name](*inputs)
                    torch.accelerator.synchronize()
                    samples[name].append((time.perf_counter_ns() - started) / 1e6)
        for name, values in samples.items():
            emit(
                dict(
                    kind="bench",
                    variant=name,
                    local_rows=local_rows,
                    global_rows=rows,
                    h=h,
                    intermediate=inter,
                    experts=experts,
                    topk=topk,
                    ep=ep,
                    shared=True,
                    latency_ms=statistics.median(values),
                    samples_ms=values,
                    boundary="replicated FFN rows + selected routes -> DP-local MoE output; shared expert and required collectives included",
                    input_distribution="synthetic uniform routing; expert-specific constant FP8 slabs and nonuniform channel scales",
                    timing="unprofiled torch.compile call plus accelerator synchronization; barriers outside timer",
                )
            )
        del compiled, result, inputs, reference
        dist.barrier()

    mapped = sorted(
        set(
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "libtpu.so" in line
        )
    )
    (out / "mapped-libtpu.json").write_text(json.dumps(mapped, indent=2))
    emit(dict(kind="pass", marker="GLM52_MOE_DECODE_EP32_PASS"))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
