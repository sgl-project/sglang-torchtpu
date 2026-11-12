# SPDX-License-Identifier: Apache-2.0
"""Model options and weight loading of the pure-JAX megakernel programs (the standalone server;
the manual benchmark harness under test/manual/k3_megakernel/bench reuses them)."""

import argparse
import ast
import inspect
import time

import jax
import numpy as np

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
    p.add_argument(
        "--expert-blocks",
        default="auto",
        metavar="auto|none|B0,B1,..",
        help="checkpoint expert block served by each kernel host (MoeLayout.ckpt_blocks): 'auto' detects "
        "EP-local checkpoints (one expert block per host, possibly not in kernel host order) and "
        "gathers every host's block; 'none' = identity; or an explicit list (tests: e.g. 1,0 with --topology 2x4)",
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
    relax = Relax.parse(args.relax.split(","))
    topo = Topology(hosts=hosts, lanes=lanes)
    return StackSpec(
        topo,
        args.layers,
        model_io.mla_layers(args.layers, args.ckpt),
        relax=relax,
        dsa_layers=model_io.dsa_layers(args.layers, args.ckpt),
        expert_blocks=expert_blocks(args, mesh, topo, relax),
    )


_BLOCKS_CACHE: dict = {}


def expert_blocks(args, mesh, topo, relax) -> tuple[int, ...]:
    """``MoeLayout.ckpt_blocks`` from ``--expert-blocks`` (see ``add_model_args``); () = identity."""
    from sglang_tpu.k3_megakernel import model_io
    from sglang_tpu.k3_megakernel.config import K3Dims

    mode = args.expert_blocks or "none"
    if mode == "none":
        return ()
    if mode != "auto":
        blocks = tuple(int(x) for x in mode.split(","))
        if len(blocks) != topo.hosts:
            raise SystemExit(f"--expert-blocks {mode}: need {topo.hosts} entries")
        return () if blocks == tuple(range(topo.hosts)) else blocks
    key = (args.ckpt, topo.hosts, topo.lanes)
    if key in _BLOCKS_CACHE:
        return _BLOCKS_CACHE[key]
    eh = K3Dims().num_experts // topo.hosts
    mine = model_io.local_expert_block(args.ckpt, eh, K3Dims().num_experts)
    from sglang_tpu.k3_megakernel.mesh import local_ranks

    ranks = local_ranks(mesh)
    host = ranks[0] // topo.lanes
    if any(r // topo.lanes != host for r in ranks) and mine is not None:
        raise SystemExit(
            "EP-local checkpoint, but this process spans several kernel hosts"
        )
    pair = np.array([host, -1 if mine is None else mine], np.int32)
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        pairs = np.asarray(multihost_utils.process_allgather(pair)).reshape(-1, 2)
    else:
        pairs = pair[None]
    if (pairs[:, 1] < 0).all():
        out = ()
    elif (pairs[:, 1] < 0).any():
        raise SystemExit(
            f"mixed EP-local / full checkpoints over the hosts: {pairs.tolist()}"
        )
    else:
        got = {int(h): int(b) for h, b in pairs}
        if sorted(got) != list(range(topo.hosts)) or sorted(got.values()) != list(
            range(topo.hosts)
        ):
            raise SystemExit(f"EP-local expert blocks do not cover the hosts: {got}")
        out = tuple(got[h] for h in range(topo.hosts))
        if relax.expert_replicas and topo.hosts > 1:
            raise SystemExit(
                "expert_replicas needs host h-1's experts, which an EP-local checkpoint does not hold"
            )
        if out == tuple(range(topo.hosts)):
            out = ()
    log(f"expert blocks: {out or 'identity'} (process blocks {pairs.tolist()})")
    _BLOCKS_CACHE[key] = out
    return out


def expert_load(args) -> str:
    return args.expert_load or ("cache" if args.layers <= 13 else "stream")


def model_tag(ckpt: str) -> str:
    """ "k31" for a K3.1 (DSA) checkpoint, "" for Kimi-K3: keeps the default expert caches apart."""
    from sglang_tpu.k3_megakernel import model_io

    return "k31" if model_io.text_config(ckpt).get("dsa_config") else ""


def expert_cache_dir(args, world: int, blocks: tuple = ()) -> str:
    """Default ``/dev/shm/k3mk_experts[_k31]_<L>l_tp<world>[_<topology>][_b<blocks>]`` (the K3 name is
    unchanged; K3.1 and remapped expert layouts get their own)."""
    if args.expert_cache:
        return args.expert_cache
    topo = f"_{args.topology}" if args.topology else ""
    tag = model_tag(args.ckpt)
    tag = f"_{tag}" if tag else ""
    perm = "_b" + "".join(map(str, blocks)) if blocks else ""
    return f"/dev/shm/k3mk_experts{tag}_{args.layers}l_tp{world}{topo}{perm}"


def load_weights(args, mesh, experts=None, log_fn=log):
    """``(stack_w, vocab_w, load_s)``; ``experts`` (routed-expert families already on device)
    skips the expert load, so a resident worker can reload only the dense weights."""
    from sglang_tpu.k3_megakernel import model_io

    spec = stack_spec(args, mesh)
    cache = (
        expert_cache_dir(args, mesh.devices.size, spec.expert_blocks)
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
