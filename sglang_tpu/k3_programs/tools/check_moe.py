# SPDX-License-Identifier: Apache-2.0
"""``check moe``: the fused EP8/TP4 prefill MoE
(``KernelConfig.prefill_fused_moe_ep8``) against the unfused EP8 path on the
same inputs and the prefill program's own weights.

Every rank draws its own rows of normalized hidden states (unit Gaussian) and
residual rows; both layers run each ``--windows`` window at each ``--layers``
MoE slot, once with every row valid and once with the window's last rows
padding. Reported per case over all ranks: the relative L2 distance of the MoE
outputs (layer output less the residual), their largest absolute difference,
the share of rows whose own relative distance exceeds ``--row-tolerance``
(routing near-ties) and that padding rows stay exactly the residual. The two
paths add each token's experts and the 32 ranks' partials in other FP32
orders, so small distances are expected; the check fails above
``--tolerance``.
"""

from __future__ import annotations

import numpy as np

from sglang_tpu.k3_programs.tools import harness


def add_arguments(sub) -> None:
    parser = sub.add_parser(
        "moe", help="the fused EP8 prefill MoE against the unfused EP8 path"
    )
    parser.add_argument("--windows", default="512,1024,2048,4096,8192")
    parser.add_argument("--layers", default="0,45,91", help="MoE slots (0..91)")
    parser.add_argument("--tolerance", type=float, default=2e-2)
    parser.add_argument("--row-tolerance", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=0)


def run_moe(args, records: harness.Records) -> None:
    import torch
    import torch.distributed as dist
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.k3_programs import bridge
    from sglang_tpu.k3_programs.kimi_k3 import config
    from sglang_tpu.k3_programs.kimi_k3 import program as k3_program
    from sglang_tpu.k3_programs.kimi_k3.kernels import prefill_megakernel as k
    from sglang_tpu.k3_programs.kimi_k3.kernels.prefill_moe import fused
    from sglang_tpu.k3_programs.kimi_k3.prefill import AXIS
    from sglang_tpu.k3_programs.tools import k3

    cfg = config.active()
    if cfg.prefill_expert_parallel != 8 or cfg.moe_reduce_dtype != "float32":
        harness.fail("check moe needs prefill_expert_parallel=8, FP32 MoE reduction")
    windows = [int(w) for w in args.windows.split(",") if w]
    prefiller = k3_program.build(
        k3.engine_config(
            "prefill",
            max_num_seqs=1,
            max_model_len=k3.logical(),
            max_num_batched_tokens=max(windows),
        ),
        args.presharded,
        tp_rank=dist.get_rank(),
    )
    prefiller.load_weights()
    program = prefiller.program
    names = tuple(program.names["moe"])
    families = program._flat("moe")
    mesh = program.mesh
    sharded, replicated = P(AXIS), P()
    generator = torch.Generator().manual_seed(args.seed * 1000 + harness.rank())
    failed = False
    for window in windows:
        rows = window // 32
        ops = {}
        for tag, make in (
            (
                "unfused",
                lambda w: k._make_indexed_moe_sp_ring_reference(mesh, seq_len=w),
            ),
            (
                "fused",
                lambda w: fused.make_indexed_moe_sp_ring(
                    mesh, seq_len=w, expert_parallel=8
                ),
            ),
        ):
            layer = make(window)

            def body(hidden, prefix, length, slot, *weights, layer=layer):
                return layer(hidden, prefix, dict(zip(names, weights)), length, slot)

            ops[tag] = bridge.register_global_op(
                f"pallas::k3_check_moe_{tag}_w{window}",
                mesh,
                body,
                (sharded, sharded, replicated, replicated) + (sharded,) * len(names),
                sharded,
            )
        for slot in [int(s) for s in args.layers.split(",") if s]:
            for length in (window, window - window // 8 - 3):
                hidden = torch.randn((1, rows, 7168), generator=generator)
                prefix = 0.25 * torch.randn((1, rows, 7168), generator=generator)
                hidden = hidden.to(torch.bfloat16).to("tpu")
                prefix = prefix.to(torch.bfloat16).to("tpu")
                scalars = (
                    torch.tensor([length], dtype=torch.int32).to("tpu"),
                    torch.tensor([slot], dtype=torch.int32).to("tpu"),
                )
                out = {}
                for tag, op in ops.items():
                    # A registered op returns its outputs as a tuple.
                    (result,) = op(hidden, prefix, *scalars, *families)
                    out[tag] = result.float()
                base = prefix.float()
                ref, new = out["unfused"] - base, out["fused"] - base
                diff = new - ref
                # This rank's rows: valid when their window position < length.
                position = torch.as_tensor(
                    k.sp_token_index(
                        program.rank, np.arange(rows), k.window_stripe(rows)
                    )
                ).to("tpu")
                valid = (position < length).reshape(1, rows, 1)
                row_err = diff.norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-30)
                stats = torch.stack(
                    (
                        (diff * valid).square().sum(),
                        (ref * valid).square().sum(),
                        ((row_err > args.row_tolerance) & valid[..., 0]).sum().float(),
                        valid.sum().float(),
                        (out["fused"] - base).abs().mul(~valid).max(),
                        (out["unfused"] - base).abs().mul(~valid).max(),
                    )
                )
                peak = (diff.abs() * valid).max().reshape(1)
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                dist.all_reduce(peak, op=dist.ReduceOp.MAX)
                stats = stats.cpu().numpy()
                rel = float(np.sqrt(stats[0] / max(stats[1], 1e-30)))
                flips = float(stats[2] / max(stats[3], 1))
                padding_ok = stats[4] == 0 and stats[5] == 0
                ok = rel <= args.tolerance and padding_ok
                failed |= not ok
                harness.log(
                    f"moe w{window} slot {slot} length {length}: rel L2 {rel:.2e}, "
                    f"max |diff| {float(peak.cpu()[0]):.3e}, rows over "
                    f"{args.row_tolerance:g}: {100 * flips:.3f}%, padding rows "
                    f"{'exact' if padding_ok else 'CHANGED'} "
                    f"{'ok' if ok else 'FAIL'}"
                )
                records.add(
                    check="moe",
                    window=window,
                    slot=slot,
                    length=length,
                    rel_l2=rel,
                    max_abs=float(peak.cpu()[0]),
                    row_flips=flips,
                    padding_exact=bool(padding_ok),
                )
    if failed:
        harness.fail("check moe: the fused EP8 MoE departs from the unfused path")
