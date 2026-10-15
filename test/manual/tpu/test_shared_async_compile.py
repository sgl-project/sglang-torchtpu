"""Run with torchrun on a full TPU mesh; validate owners and local state binding."""

import logging
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch_tpu._internal.compile import tpu_torch_compile as tc

from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_context
from sglang_tpu.compilation.direct_compile import direct_compile
from sglang_tpu.compilation.graph_identity import canonical
from sglang_tpu.compilation.warmup_queue import collect_warmups


class Pool:
    def __init__(self):
        self.value = torch.zeros((8, 128), dtype=torch.int32, device="tpu")

    def get_state(self):
        return [self.value]

    def set_state(self, state):
        self.value = state[0]


def check_owner_only_tracing(cpu_group):
    rank, size = dist.get_rank(), dist.get_world_size()
    batches = (1, 2, 4, 8, 16, 32, 64)
    names = ("target/model.forward", "target/vision.forward", "draft/forward")
    traces = []
    model = torch.nn.Module()
    model.register_buffer(
        "weight", torch.full((128,), rank + 1, device="tpu", dtype=torch.int32)
    )
    pool = Pool()
    original_pool = pool.value
    values = [
        torch.full((bs, 128), rank + bs, device="tpu", dtype=torch.int32)
        for bs in batches
    ]
    for tensor in (model.weight, pool.value, *values):
        tensor.cpu()

    def forward_for(policy):
        def forward(inputs):
            value = inputs["value"]
            traces.append((policy, value.shape[0]))
            local = (
                value * model.weight
                + current_platform.get_traced_rank(rank)
                + policy * 100
            )
            pool.value.add_(local.sum(dim=0))
            gathered = torch.empty(
                (size * value.shape[0], 128), device="tpu", dtype=torch.int32
            )
            dist.all_gather_into_tensor(gathered, local)
            return {"local": local, "gathered": gathered}

        return forward

    results = []
    keys = []
    with (
        torch.inference_mode(),
        get_context().override_server_args(allow_spmd_safe=True),
        patch.object(tc, "build_mlir", wraps=tc.build_mlir) as lower,
        patch.object(
            tc, "async_traverse_and_compile", wraps=tc.async_traverse_and_compile
        ) as native,
    ):
        with collect_warmups(group=cpu_group) as warmups:
            shared_compile = warmups.shared_compile
            assert shared_compile is not None
            inputs = {}
            for policy, name in enumerate(names):
                compiled = direct_compile(
                    forward_for(policy),
                    name=name,
                    spmd_safe=True,
                    lift_rank=True,
                    specialization_key=lambda inputs: tuple(inputs["value"].shape),
                    model=model,
                    kv_pool=pool,
                    prune_unused_inputs=False,
                )
                for value in values:
                    inputs["value"] = value
                    keys.append(f"{name}:{canonical(tuple(value.shape))}")
                    warmups.calls.append(
                        compiled.warmup_async(inputs).then(results.append)
                    )
                # A duplicate specialization must neither advance the cursor nor compile again.
                warmups.calls.append(
                    compiled.precompile({"value": values[0]}, async_compile=True)
                )
            owners = [shared_compile.submissions[key].owner for key in keys]
            assert owners == [owner % size for owner in range(len(keys))]
            assert len(shared_compile.submissions) == len(keys)
        assert warmups.shared_compile is None
        expected_traces = [
            (policy, bs)
            for policy in range(len(names))
            for index, bs in enumerate(batches)
            if (policy * len(batches) + index) % size == rank
        ]
        assert traces == expected_traces, (rank, traces, expected_traces)
        assert native.call_count == len(expected_traces)
        lower.assert_not_called()
        expected_kv = torch.zeros((8, 128), dtype=torch.int32)
        for (policy, bs), result in zip(
            ((policy, bs) for policy in range(len(names)) for bs in batches),
            results,
            strict=True,
        ):
            parts = [
                torch.full(
                    (bs, 128),
                    (peer + bs) * (peer + 1) + peer + policy * 100,
                    dtype=torch.int32,
                )
                for peer in range(size)
            ]
            assert torch.equal(result["local"].cpu(), parts[rank])
            assert torch.equal(result["gathered"].cpu(), torch.cat(parts))
            expected_kv += parts[rank].sum(dim=0)
        assert pool.value is original_pool
        assert torch.equal(pool.value.cpu(), expected_kv)
    print(
        f"shared compilation: rank={rank} traces={len(traces)} graphs={len(keys)} PASS",
        flush=True,
    )
    dist.barrier(group=cpu_group)


def main():
    logging.basicConfig(level=logging.INFO)
    dist.init_process_group("tpu_dist", timeout=timedelta(seconds=240))
    cpu_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=240))
    check_owner_only_tracing(cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
