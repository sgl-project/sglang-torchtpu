# SPDX-License-Identifier: Apache-2.0
"""Model options and weight loading of the pure-JAX megakernel programs (the standalone server;
the manual benchmark harness under test/manual/k3_megakernel/bench reuses them)."""

import argparse
import ast
import inspect
import time

import jax

ANCHOR = (1008, 18985, 387)  # anchor-protocol prompt


def log(msg):
    print(
        f"[k3mk p{jax.process_index()} {time.strftime('%H:%M:%S')}] {msg}", flush=True
    )


def add_model_args(p: argparse.ArgumentParser) -> None:
    """Checkpoint, topology, numerics and multi-process options."""
    p.add_argument("--ckpt", default="/weights/Kimi-K3")
    p.add_argument(
        "--layers",
        type=int,
        default=13,
        help="first N decoder layers (93 = full model)",
    )
    p.add_argument(
        "--expert-load",
        choices=("cache", "stream"),
        default="",
        help="routed experts from a per-host /dev/shm memmap cache, or streamed layer by layer from "
        "the checkpoint to the devices (nothing kept on the host); default: cache up to 13 layers",
    )
    p.add_argument(
        "--expert-load-procs",
        type=int,
        default=48,
        help="stream: reblocking processes per host",
    )
    p.add_argument(
        "--expert-cache",
        default="",
        help="default /dev/shm/k3mk_experts_<L>l_tp<world>",
    )
    p.add_argument(
        "--topology",
        default="",
        metavar="HOSTSxLANES",
        help="kernel topology, default <processes>x8; e.g. 2x4 on one host treats its 8 cores as two "
        "fake hosts (cross-host code paths on a single host)",
    )
    p.add_argument("--embed", choices=("full", "sharded"), default="full")
    p.add_argument(
        "--relax", default="", help="SGLANG_TPU_OPT_K3_MEGAKERNEL_RELAX value"
    )
    p.add_argument(
        "--kernel-opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra StackKernel keyword (VALUE is a Python literal); repeatable",
    )
    p.add_argument("--coordinator", default="")
    p.add_argument("--num-processes", type=int, default=1)
    p.add_argument("--process-id", type=int, default=0)


def kernel_kwargs(args, extra: dict | None = None) -> dict:
    """StackKernel keywords from ``--kernel-opt`` (plus ``extra``); a keyword the kernel does not
    take is an error."""
    from sglang_tpu.k3_megakernel.kernel.stack import StackKernel

    kw = dict(extra or {})
    for item in args.kernel_opt:
        key, _, value = item.partition("=")
        try:
            kw[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            kw[key] = value
    accepted = inspect.signature(StackKernel.__init__).parameters
    unknown = [k for k in kw if k not in accepted]
    if unknown:
        raise SystemExit(
            f"StackKernel does not take {unknown}; it takes {list(accepted)[1:]}"
        )
    return kw


def stack_spec(args, mesh):
    from sglang_tpu.k3_megakernel import model_io
    from sglang_tpu.k3_megakernel.config import Relax, Topology
    from sglang_tpu.k3_megakernel.kernel.stack import StackSpec

    world = mesh.devices.size
    hosts, lanes = (
        (int(x) for x in args.topology.split("x")) if args.topology else (world // 8, 8)
    )
    if hosts * lanes != world:
        raise SystemExit(f"--topology {hosts}x{lanes} does not cover {world} devices")
    return StackSpec(
        Topology(hosts=hosts, lanes=lanes),
        args.layers,
        model_io.mla_layers(args.layers, args.ckpt),
        relax=Relax.parse(args.relax.split(",")),
    )


def expert_load(args) -> str:
    return args.expert_load or ("cache" if args.layers <= 13 else "stream")


def expert_cache_dir(args, world: int) -> str:
    topo = f"_{args.topology}" if args.topology else ""
    return args.expert_cache or f"/dev/shm/k3mk_experts_{args.layers}l_tp{world}{topo}"


def load_weights(args, mesh, experts=None, log_fn=log):
    """``(stack_w, vocab_w, load_s)``; ``experts`` (routed-expert families already on device)
    skips the expert load, so a resident worker can reload only the dense weights."""
    from sglang_tpu.k3_megakernel import model_io

    spec = stack_spec(args, mesh)
    cache = (
        expert_cache_dir(args, mesh.devices.size)
        if expert_load(args) == "cache"
        else ""
    )
    log_fn(f"loading {spec}; experts {'kept' if experts else cache or 'streamed'}")
    t0 = time.time()
    stack_w = model_io.load_stack_weights(
        mesh,
        spec,
        args.ckpt,
        cache,
        log_fn,
        expert_procs=args.expert_load_procs,
        with_experts=experts is None,
    )
    stack_w.update(experts or {})
    vocab_w = model_io.load_vocab_weights(
        mesh, args.ckpt, embed_mode=args.embed, log=log_fn
    )
    return stack_w, vocab_w, time.time() - t0
