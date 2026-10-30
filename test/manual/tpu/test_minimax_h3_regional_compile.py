# SPDX-License-Identifier: Apache-2.0
"""Manual four-host H3 block compilation test with FSDP2 and Ulysses.

Run this file on all four v6e-16 hosts together. This uses a small instance of
the native H3 block; complete Full/base video acceptance is a separate test.
"""

import argparse
import json
import os
import traceback
from pathlib import Path


def _worker(local_rank, args):
    rank = args.node_rank * 4 + local_rank
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(local_rank),
        WORLD_SIZE="16",
        MASTER_ADDR=args.hosts[0],
        MASTER_PORT=str(args.port),
    )
    receipt = {"rank": rank, "status": "RUNNING"}
    path = Path(args.receipts) / f"block-rank{rank}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        path.write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    try:
        import torch
        import torch.distributed as dist
        import torch_tpu._internal.compile  # noqa: F401
        from torch._dynamo.backends.registry import lookup_backend
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
            MiniMaxH3DiTArchConfig,
        )
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        from sglang.multimodal_gen.runtime.layers.attention.selector import (
            component_attn_backend_context_manager,
        )
        from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
            MiniMaxH3DiTBlock,
        )
        from sglang.multimodal_gen.runtime.platforms import (
            AttentionBackendEnum,
            current_platform,
        )

        torch.set_float32_matmul_precision("highest")
        current_platform.set_device(current_platform.get_device(local_rank))
        device = current_platform.get_local_torch_device()
        assert device.type == "tpu"
        init_distributed_environment(
            world_size=16,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method=f"tcp://{args.hosts[0]}:{args.port}",
            backend="tpu_dist",
            timeout=600,
        )
        initialize_model_parallel(
            data_parallel_size=2,
            sequence_parallel_degree=8,
            ulysses_degree=8,
            ring_degree=1,
            tensor_parallel_degree=1,
        )
        arch = MiniMaxH3DiTArchConfig(
            num_layers=1,
            token_refiner_num_layers=1,
            hidden_size=256,
            num_attention_heads=8,
            attention_head_dim=32,
            ffn_hidden_size=512,
            time_embed_dim=64,
            adaln_out_features=18 * 256,
            final_adaln_out_features=2 * 256,
        )
        with component_attn_backend_context_manager(
            AttentionBackendEnum.TORCH_SDPA, require_backend_selection=False
        ):
            block = MiniMaxH3DiTBlock(arch, None, prefix="blocks.0").eval()
        generator = torch.Generator().manual_seed(42)
        with torch.no_grad():
            for name, parameter in block.named_parameters():
                value = (
                    torch.ones(parameter.shape)
                    if "norm" in name and name.endswith("weight")
                    else torch.randn(parameter.shape, generator=generator) * 0.02
                )
                parameter.copy_(value.to(parameter.dtype))
        block = block.to(device)
        x = torch.randn(8, 256, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        adaln = torch.randn(2, 64, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        rope = torch.cat((torch.ones(8, 16), torch.zeros(8, 16)), dim=-1).to(
            device=device, dtype=torch.bfloat16
        )
        bounds = (0, 23, 23, 64)
        kwargs = {
            "combined_indices": (torch.arange(8) % 6).to(device),
            "rope_cache": (rope, torch.arange(8, device=device)),
            "cu_seqlens": torch.tensor(bounds, dtype=torch.int32, device=device),
            "cu_seqlens_host": bounds,
            "max_seqlen": 41,
            "ulysses_active": True,
            "ring_active": False,
        }
        with torch.no_grad():
            inputs = [adaln, adaln + 0.1, adaln + 0.2]
            reference = [
                block(x.clone(), adaln_input=t, **kwargs).cpu() for t in inputs
            ]
            mesh = init_device_mesh(
                "tpu", (2, 8), mesh_dim_names=("replicate", "shard")
            )
            fully_shard(
                block,
                mesh=mesh,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=None, reduce_dtype=torch.float32
                ),
                reshard_after_forward=True,
            )
            graphs = []
            native_backend = lookup_backend("tpu")

            def backend(graph_module, example_inputs, **options):
                graphs.append(
                    [
                        str(node.target)
                        for node in graph_module.graph.nodes
                        if node.op == "call_function"
                    ]
                )
                return native_backend(graph_module, example_inputs, **options)

            block.compile(backend=backend, dynamic=False, fullgraph=False)
            counts, errors = [], []
            for index, timestep in enumerate(inputs):
                actual = block(x.clone(), adaln_input=timestep, **kwargs).cpu()
                assert torch.isfinite(actual).all()
                errors.append(
                    (actual.float() - reference[index].float()).abs().max().item()
                )
                torch.testing.assert_close(
                    actual, reference[index], rtol=0.03, atol=0.03
                )
                counts.append(len(graphs))
            assert counts[-1] == counts[-2], counts
            captured = "\n".join(op for graph in graphs for op in graph)
            for operation in (
                "linear",
                "rms_norm",
                "scaled_dot_product_attention",
                "all_to_all_single",
                "silu",
            ):
                assert operation in captured, f"Not captured: {operation}"
            receipt.update(
                status="PASS",
                graph_counts=counts,
                max_absolute_errors=errors,
                graphs=graphs,
                rtol=0.03,
                atol=0.03,
            )
            save()
        dist.barrier()
        dist.destroy_process_group()
    except BaseException:
        receipt.update(status="FAIL", error=traceback.format_exc())
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", nargs=4, required=True)
    parser.add_argument("--node-rank", type=int, choices=range(4), required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--port", type=int, default=10320)
    parser.add_argument("--receipts", required=True)
    args = parser.parse_args()
    os.environ.update(
        TORCH_TPU_DEVICES_PER_HOST="4",
        TPU_WORKER_HOSTNAMES=",".join(args.hosts),
        TPU_WORKER_ID=str(args.node_rank),
        JAX_PLATFORMS="cpu",
        TPU_TOPOLOGY=args.topology,
    )
    import torch.multiprocessing as mp

    mp.spawn(_worker, args=(args,), nprocs=4, join=True)


if __name__ == "__main__":
    main()
