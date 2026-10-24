# SPDX-License-Identifier: Apache-2.0
"""Per-process weight loading and request state for the anchor benchmark.

Every global array has a leading rank axis sharded over the mesh (one row per
device). A process only reads and uploads the rows of its own host's ranks.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sglang_tpu.k3_megakernel.kernel.stages.mla import BLOCK_PAGES, PAGE
from sglang_tpu.k3_megakernel.mesh import AXIS, local_ranks
from sglang_tpu.k3_megakernel.weights import (
    Checkpoint,
    load_stack,
    stream_expert_layers,
    tensor_ranges,
)


def mla_layers(num_layers: int, ckpt_path: str) -> tuple[int, ...]:
    """0-based MLA layer ids among the first ``num_layers`` (config lists them 1-based)."""
    import json
    import os

    with open(os.path.join(ckpt_path, "config.json")) as f:
        cfg = json.load(f)
    cfg = cfg.get("text_config", cfg)
    return tuple(
        i - 1
        for i in cfg["linear_attn_config"]["full_attn_layers"]
        if i - 1 < num_layers
    )


EXPERT_FAMILIES = ("expert_gu", "expert_gu_scale", "expert_down", "expert_down_scale")


def put_rows(mesh, rows_by_rank: dict, shape_tail: tuple, dtype) -> jax.Array:
    """Global ``[world, *shape_tail]`` array from this process's per-rank numpy rows."""
    devs = list(mesh.devices.flat)
    shards = [
        jax.device_put(
            np.ascontiguousarray(rows_by_rank[r]).reshape((1,) + tuple(shape_tail)),
            devs[r],
        )
        for r in local_ranks(mesh)
    ]
    sharding = NamedSharding(mesh, P(AXIS))
    return jax.make_array_from_single_device_arrays(
        (len(devs),) + tuple(shape_tail), sharding, shards
    )


def put_replicated(mesh, value: np.ndarray) -> jax.Array:
    """``value`` repeated on every rank (leading rank axis)."""
    return put_rows(
        mesh, {r: value for r in local_ranks(mesh)}, value.shape, value.dtype
    )


def prefetch_checkpoint(
    ckpt_path: str, num_layers: int, log, threads: int = 64, chunk: int = 32 << 20
) -> None:
    """Pull the dense tensors into the page cache with parallel reads; the loaders then
    memmap them. Single-threaded memmap faults over NFS ran at ~25 MB/s."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.time()
    jobs = [
        (p, o, min(chunk, off + size - o))
        for p, off, size in tensor_ranges(ckpt_path, num_layers)
        for o in range(off, off + size, chunk)
    ]

    def read(job):
        path, offset, size = job
        fd = os.open(path, os.O_RDONLY)
        try:
            return len(os.pread(fd, size, offset))
        finally:
            os.close(fd)

    with ThreadPoolExecutor(threads) as pool:
        total = sum(pool.map(read, jobs))
    dt = time.time() - t0
    log(
        f"prefetched {total / 2**30:.1f} GiB of checkpoint in {dt:.0f}s ({total / 2**20 / max(dt, 1e-3):.0f} MiB/s)"
    )


def load_stack_weights(
    mesh,
    spec,
    ckpt_path: str,
    expert_cache: str,
    log,
    expert_procs: int = 48,
    with_experts: bool = True,
) -> dict:
    """Kernel weights (``load_stack``) for this host's ranks, uploaded family by family. Routed
    experts come from the memmapped ``expert_cache`` or, when it is "", are streamed layer by
    layer from the checkpoint straight to the devices (no host copy of the stack persists);
    ``with_experts=False`` loads only the other families."""
    ranks = local_ranks(mesh)
    prefetch_checkpoint(ckpt_path, spec.num_layers, log)
    t0 = time.time()
    host = load_stack(
        Checkpoint(ckpt_path),
        spec,
        ranks,
        expert_cache=expert_cache,
        routed_experts=with_experts and bool(expert_cache),
    )
    log(f"load_stack: {len(host)} families in {time.time() - t0:.0f}s")
    out = {}
    for name in sorted(host):
        a = host.pop(name)
        out[name] = put_rows(
            mesh, {r: a[i] for i, r in enumerate(ranks)}, a.shape[1:], a.dtype
        )
        del a
    jax.block_until_ready(out)
    log(f"stack weights on device in {time.time() - t0:.0f}s")
    if with_experts and not expert_cache:
        out.update(stream_experts(mesh, spec, ckpt_path, log, procs=expert_procs))
    return out


def stream_experts(mesh, spec, ckpt_path: str, log, procs: int) -> dict:
    """Routed-expert families ``[world, moe_layers, experts_per_host, ...]``: one device buffer
    per local rank, filled in place (donated update) one MoE layer at a time from
    ``stream_expert_layers`` while its process pool reblocks the next layer. A consumed layer's
    shards are dropped from the page cache (host memory budget: cache + /dev/shm + processes)."""
    from jax.sharding import SingleDeviceSharding

    from sglang_tpu.k3_megakernel.config import MoeLayout

    devs = list(mesh.devices.flat)
    layers = range(1, spec.num_layers)
    n = len(layers)
    put_layer = jax.jit(lambda buf, x, i: buf.at[0, i].set(x), donate_argnums=0)
    shards = _expert_shards(ckpt_path, layers)
    layout = MoeLayout(spec.dims, spec.topo, replicas=spec.relax.expert_replicas)
    # One stream per (possibly fake, see anchor --topology) host among this process's ranks.
    groups: dict[int, list[int]] = {}
    for r in local_ranks(mesh):
        groups.setdefault(r // spec.topo.lanes, []).append(r)
    bufs: dict[str, dict[int, jax.Array]] = {}
    t0 = time.time()
    for ranks in groups.values():
        for i, arrays in stream_expert_layers(
            ckpt_path, layout, layers, ranks, procs=procs
        ):
            for name, a in arrays.items():
                per = bufs.setdefault(name, {})
                for k, r in enumerate(ranks):
                    if r not in per:
                        per[r] = jax.jit(
                            lambda s=a.shape[1:], d=a.dtype: jnp.zeros((1, n) + s, d),
                            out_shardings=SingleDeviceSharding(devs[r]),
                        )()
                    per[r] = put_layer(per[r], jax.device_put(a[k], devs[r]), i)
            # One layer in flight: queued uploads would otherwise pin their host copies (~150 GiB at TP8).
            jax.block_until_ready(bufs)
            _drop_page_cache(shards[layers[i]])
            if (i + 1) % 10 == 0 or i + 1 == n:
                log(
                    f"experts of {i + 1}/{n} MoE layers on device, ranks {ranks[0]}-{ranks[-1]} ({time.time() - t0:.0f}s)"
                )
    sharding = NamedSharding(mesh, P(AXIS))
    out = {}
    for name, per in bufs.items():
        per_rank = [per[r] for r in sorted(per)]
        shape = (len(devs),) + tuple(per_rank[0].shape[1:])
        out[name] = jax.make_array_from_single_device_arrays(shape, sharding, per_rank)
    jax.block_until_ready(out)
    log(f"routed experts streamed in {time.time() - t0:.0f}s")
    return out


def _expert_shards(ckpt_path: str, layers) -> dict[int, list[str]]:
    """Shard files holding each layer's routed experts."""
    import json
    import os

    with open(os.path.join(ckpt_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    out = {layer: set() for layer in layers}
    for name, shard in weight_map.items():
        if ".experts." in name and ".layers." in name:
            layer = int(name.split(".layers.")[1].split(".")[0])
            if layer in out:
                out[layer].add(os.path.join(ckpt_path, shard))
    return {layer: sorted(v) for layer, v in out.items()}


def _drop_page_cache(paths) -> None:
    import os

    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def load_vocab_weights(mesh, ckpt_path: str, *, embed_mode: str, log) -> dict:
    """``lm_head`` rows ``[V/world, H]`` per rank; the embedding either whole on every
    rank (``full``) or vocab-sharded like the head (``sharded``)."""
    t0 = time.time()
    ckpt = Checkpoint(ckpt_path, prefix="language_model.")
    world = mesh.devices.size
    ranks = local_ranks(mesh)
    head = ckpt.get("lm_head.weight")
    emb = ckpt.get("model.embed_tokens.weight")
    vocab = head.shape[0]
    if vocab % world:
        raise ValueError(f"vocab {vocab} not divisible by {world} ranks")
    vs = vocab // world
    out = {
        "lm_head": put_rows(
            mesh,
            {r: head[r * vs : (r + 1) * vs] for r in ranks},
            (vs, head.shape[1]),
            head.dtype,
        )
    }
    if embed_mode == "full":
        table = np.ascontiguousarray(emb)
        out["embed"] = put_rows(
            mesh, {r: table for r in ranks}, table.shape, table.dtype
        )
        del table
    elif embed_mode == "sharded":
        out["embed"] = put_rows(
            mesh,
            {r: emb[r * vs : (r + 1) * vs] for r in ranks},
            (vs, emb.shape[1]),
            emb.dtype,
        )
    else:
        raise ValueError(f"embed mode {embed_mode!r}")
    jax.block_until_ready(out)
    log(f"embedding ({embed_mode}) + lm_head on device in {time.time() - t0:.0f}s")
    return out


def page_count(capacity: int) -> int:
    return -(-capacity // PAGE)


def page_table(capacity: int) -> np.ndarray:
    """Identity page table over ``capacity`` tokens, padded to whole kernel blocks."""
    n = page_count(capacity)
    return np.concatenate(
        [np.arange(n, dtype=np.int32), np.zeros((-n) % BLOCK_PAGES, np.int32)]
    )


def state_shapes(kernel, spec, capacity: int, kda_rows: int = 1) -> dict:
    """Per-rank (shape, dtype) of every state input: one request slot (``kda_rows`` KDA rows: the
    verify checkpoints), ``capacity`` tokens of MLA pages."""
    from sglang_tpu.k3_megakernel.kernel.stack import state_inputs

    kd = kernel.kda
    shapes = {}
    for name in state_inputs(spec):
        if name.startswith("kda_conv_"):
            shapes[name] = ((kda_rows, 3, 1, 3 * kd.hd), jnp.float32)
        elif name.startswith("kda_rec_"):
            shapes[name] = ((kda_rows, kd.heads, 128, 128), jnp.float32)
        elif name.startswith("mla_cache_"):
            shapes[name] = ((page_count(capacity), PAGE // 2, 2, 640), jnp.bfloat16)
        else:
            raise ValueError(f"unknown state input {name}")
    return shapes


def make_fresh_states(mesh, shapes: dict):
    """A jitted ``() -> {name: zeros [world, ...]}`` for a new request."""
    world = mesh.devices.size
    sharding = NamedSharding(mesh, P(AXIS))
    out_sh = {n: sharding for n in shapes}

    def fresh():
        return {n: jnp.zeros((world,) + tuple(s), d) for n, (s, d) in shapes.items()}

    return jax.jit(fresh, out_shardings=out_sh)


def embed(table, token, *, mode: str, vocab_shard: int, rank):
    """[1, H] bf16 row of ``token``; the sharded table needs one psum (a single nonzero row, exact)."""
    if mode == "full":
        return jax.lax.dynamic_slice_in_dim(table, token, 1, axis=0)
    local = token - rank * vocab_shard
    mine = (local >= 0) & (local < vocab_shard)
    row = jax.lax.dynamic_slice_in_dim(
        table, jnp.clip(local, 0, vocab_shard - 1), 1, axis=0
    )
    row = jnp.where(mine, row.astype(jnp.float32), 0.0)
    return jax.lax.psum(row, AXIS).astype(jnp.bfloat16)


def greedy(final, head, *, vocab_shard: int, rank):
    """Global argmax over the vocab-sharded logits; ties go to the lowest id."""
    logits = jax.lax.dot_general(
        final, head, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )[0]
    local_id = jnp.argmax(logits).astype(jnp.int32)
    # Ids < 2**24 are exact in f32, so one f32 pair per rank carries both.
    packed = jnp.stack(
        [logits[local_id], (local_id + rank * vocab_shard).astype(jnp.float32)]
    )
    every = jax.lax.all_gather(packed, AXIS)  # [world, 2]
    return every[jnp.argmax(every[:, 0]), 1].astype(jnp.int32)
