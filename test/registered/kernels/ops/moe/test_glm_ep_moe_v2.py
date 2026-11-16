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


def worker():
    import json
    import time

    import numpy as np
    import torch
    import torch.distributed as dist
    import torch_tpu
    from importlib.metadata import version

    from sglang_tpu.moe.pallas.glm_moe_v2.op import (
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
    s13 = torch.linspace(0.7, 1.3, e * 2 * inter).reshape(e, 1, 1, 2 * inter)
    s2 = torch.linspace(0.8, 1.2, e * h).reshape(e, 1, 1, h)
    sl = slice(rank * 2, (rank + 1) * 2)
    device_weights = [
        pack_gate_up(w13[sl]).to("tpu"),
        w2[sl].to("tpu"),
        pack_gate_up_scale(s13[sl]).to("tpu"),
        s2[sl].to("tpu"),
    ]
    rank_tensor = torch.tensor([rank], dtype=torch.int32, device="tpu")
    stub = torch.empty(0, device="tpu")
    reports = []
    for local_rows, routing in [
        (6, "random"),
        (12, "hot"),
        (18, "zeros"),
        (24, "padded"),
        (65, "random"),
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
        if routing == "padded":
            ids[-7:] = -1
            weights[-7:] = 0
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
        args = [
            x.to("tpu"),
            ids.to("tpu"),
            weights.to("tpu"),
            *device_weights,
            rank_tensor,
            stub,
            stub,
            stub,
            stub,
        ]
        op = get_op(ep, False)
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
            "relative_l2": relative_l2,
            "median_ms": float(np.median(times)),
            "range_ms": [min(times), max(times)],
        }
        reports.append(report)
        print(json.dumps({"event": "case_pass", "rank": rank, **report}), flush=True)
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


class TestGLMEPMoEV2(CustomTestCase):
    def test_selected_route_torch_bridge(self):
        python = shlex.quote(sys.executable)
        script = shlex.quote(str(Path(__file__).resolve()))
        command = (
            "unset RANK LOCAL_RANK LOCAL_WORLD_SIZE WORLD_SIZE; "
            f"eval $({python} -m torch.tpu.distributed); "
            "export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE; "
            'export JAX_PLATFORMS=cpu; export XLA_FLAGS="--xla_force_host_platform_device_count=$WORLD_SIZE"; '
            f'{python} -m torch.distributed.run --standalone --nproc_per_node="$WORLD_SIZE" {script} --worker'
        )
        subprocess.run(["bash", "-lc", command], check=True, timeout=1800)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    else:
        unittest.main()
