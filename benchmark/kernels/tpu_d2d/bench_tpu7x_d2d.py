# SPDX-License-Identifier: Apache-2.0
"""Remote-DMA micro-bench between the two TensorCores of one tpu7x chip.

A tpu7x chip has two TensorCores. Each is its own JAX device with its own HBM;
a die-to-die (D2D) link joins them. This times Pallas remote DMAs between the
two cores of every chip of one host. The payload is one routed-expert slab of
the Kimi-K3.1 decode megakernel: one expert's per-rank MXFP4 gate/up, its
scales, down and its scales, 4,386,816 bytes in four arrays
(``load.selected_local_shapes``: expert_gu / expert_gus / expert_down /
expert_ds).

Variants (core 0 of each chip is side A, core 1 side B, unless noted):

  push_dN          A pushes slabs from its HBM into B's HBM, N slabs in flight
  push_dN_chunk4   the same with the two large arrays split into four DMAs each
  push_d2_flat     one contiguous 4.39 MB DMA per slab
  push_d2_rev      B pushes to A
  push_d2_onechip  only chip 0 pushes (the other chips idle)
  bidir_d2         A and B push to each other at once
  push_d2_vmem     A pushes from its HBM into B's VMEM
  local_vmem_d2    both cores copy their own slabs HBM -> VMEM (the expert stream)
  local_hbm_d2     A copies its own slabs HBM -> HBM
  push_d2+a_local  A pushes to B and streams its own slabs HBM -> VMEM at once
  push_d2+b_local  A pushes to B while B streams its own slabs HBM -> VMEM
  slab_latency     A pushes one slab, B rings A back after it lands (per slab)
  burst_K          the same with K slabs issued at once (K = 2, 4)
  pp_<mem>_<size>  64 KB / 4 KB ping-pong (A -> B, B -> A) between VMEM or HBM
  pp_sem           remote semaphore ping-pong (no data)
  ici_*            the same between core 0 of two x-neighbour chips (ICI)

Every kernel loops a runtime iteration count on the device. A variant runs at
two counts (seven runs each, best of each); the per-iteration time is the
difference of the two run times over the difference of the counts, so launch
and sync costs cancel. Push variants check before the timing that each
receiver's buffers hold its sibling's slabs.

    python3 bench_tpu7x_d2d.py                  # on a tpu7x host (8 devices)
    D2D_INTERPRET=1 python3 bench_tpu7x_d2d.py  # CPU interpret check (4 devices)
    D2D_AOT=1 python3 bench_tpu7x_d2d.py        # compile-only for tpu7x:2x2x1

``D2D_ONLY=name,name`` runs a subset. Results go to stdout and, as JSON lines,
to ``$D2D_OUT/d2d_bench.jsonl``.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import threading
import time

MODE = (
    "interpret"
    if os.environ.get("D2D_INTERPRET") == "1"
    else "aot"
    if os.environ.get("D2D_AOT") == "1"
    else "tpu"
)
if MODE == "interpret":
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    )
    os.environ["JAX_PLATFORMS"] = "cpu"

# One routed-expert slab of one rank (gu, gu scales, down, down scales).
SLAB_PARTS = (
    ((448, 1536), "uint32"),
    ((112, 1536), "uint8"),
    ((96, 3584), "uint32"),
    ((24, 3584), "uint8"),
)
FLAT_PARTS = (((1072, 1024), "uint32"),)
# Interpret mode: the same structure at a size the CPU interpreter runs quickly.
SMALL_SLAB_PARTS = (
    ((32, 256), "uint32"),
    ((32, 256), "uint8"),
    ((32, 384), "uint32"),
    ((32, 384), "uint8"),
)
SMALL_FLAT_PARTS = (((32, 512), "uint32"),)

SRC_SLABS = 8  # distinct source slabs per device
DST_SLOTS = 4  # HBM staging slots per device
COLLECTIVE_ID = 7

VARIANTS = (
    dict(name="push_d2", kind="push", depth=2),
    dict(name="push_d4", kind="push", depth=4),
    dict(name="push_d1", kind="push", depth=1),
    dict(name="slab_latency", kind="slablat"),
    dict(name="burst_2", kind="slablat", burst=2),
    dict(name="burst_4", kind="slablat", burst=4),
    dict(name="pp_vmem_64k", kind="pingpong", nbytes=65536, space="vmem"),
    dict(name="pp_hbm_64k", kind="pingpong", nbytes=65536, space="hbm"),
    dict(name="pp_vmem_4k", kind="pingpong", nbytes=4096, space="vmem"),
    dict(name="pp_sem", kind="sempp"),
    dict(name="push_d2_chunk4", kind="push", depth=2, chunks=4),
    dict(name="push_d4_chunk4", kind="push", depth=4, chunks=4),
    dict(name="push_d2_flat", kind="push", depth=2, flat=True),
    dict(name="push_d2_rev", kind="push", depth=2, reverse=True),
    dict(name="push_d2_onechip", kind="push", depth=2, one_chip=True),
    dict(name="bidir_d2", kind="push", depth=2, b_push=True),
    dict(name="push_d2_vmem", kind="push", depth=2, dst="vmem"),
    dict(
        name="local_vmem_d2",
        kind="push",
        depth=2,
        a_push=False,
        a_local=True,
        b_local=True,
    ),
    dict(
        name="local_hbm_d2",
        kind="push",
        depth=2,
        a_push=False,
        a_local=True,
        local_dst="hbm",
    ),
    dict(name="push_d2+a_local", kind="push", depth=2, a_local=True),
    dict(name="push_d2+b_local", kind="push", depth=2, b_local=True),
    dict(name="ici_push_d2", kind="push", depth=2, table="ici"),
    dict(
        name="ici_pp_vmem_64k", kind="pingpong", nbytes=65536, space="vmem", table="ici"
    ),
)


def _watchdog_main():
    """Run the bench in a child process; if it is silent for too long (a hung
    kernel), kill it and resume with the variants it has not finished."""
    done, hung = set(), set()
    names = [v["name"] for v in VARIANTS]
    only = [n for n in os.environ.get("D2D_ONLY", "").split(",") if n]
    wanted = only or names
    silence = float(os.environ.get("D2D_SILENCE_S", "420"))
    for attempt in range(4):
        todo = [n for n in wanted if n not in done and n not in hung]
        if not todo:
            break
        env = dict(os.environ, D2D_CHILD="1", D2D_ONLY=",".join(todo))
        child = subprocess.Popen(
            [sys.executable, "-u", os.path.abspath(__file__)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        last = [time.time()]
        current = [None]

        def pump():
            for line in child.stdout:
                last[0] = time.time()
                sys.stdout.write(line)
                sys.stdout.flush()
                if line.startswith("@@start "):
                    current[0] = line.split()[1]
                elif line.startswith("@@done "):
                    done.add(line.split()[1])
                    current[0] = None

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        while child.poll() is None:
            time.sleep(5)
            if time.time() - last[0] > silence:
                print(
                    f"@@watchdog: no output for {silence:.0f} s in {current[0]}; killing",
                    flush=True,
                )
                child.kill()
                child.wait()
                if current[0]:
                    hung.add(current[0])
                break
        reader.join(timeout=30)
        if child.returncode == 0 and not current[0]:
            break
        if current[0] and current[0] not in hung:
            hung.add(current[0])  # the child died inside this variant
        print(
            f"@@watchdog: attempt {attempt} rc={child.returncode}; hung {sorted(hung)}",
            flush=True,
        )
    missing = [n for n in wanted if n not in done]
    print(
        f"@@watchdog: finished {len(done)}/{len(wanted)}; not finished {missing}",
        flush=True,
    )
    return 0 if not missing else 1


if MODE == "tpu" and os.environ.get("D2D_CHILD") != "1" and __name__ == "__main__":
    sys.exit(_watchdog_main())

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

try:
    from jax import shard_map
except ImportError:  # older jax
    from jax.experimental.shard_map import shard_map

AXIS = "x"
MESH_ID = pl.DeviceIdType.MESH


def _devices():
    if MODE == "aot":
        from jax.experimental import topologies

        devices = topologies.get_topology_desc("tpu7x:2x2x1", "tpu").devices
    else:
        devices = jax.devices()
    # Ascending device id: the order the serving kernel mesh uses.
    return sorted(devices, key=lambda d: int(d.id))


def _chip_and_core(device, position):
    coords = getattr(device, "coords", None)
    core = getattr(device, "core_on_chip", None)
    if coords is None or core is None or MODE == "interpret":
        return (position // 2, 0, 0), position % 2  # CPU: pairs (2k, 2k + 1)
    return tuple(int(c) for c in coords), int(core)


def _tables(devices):
    """Peer and role per mesh position for the sibling (D2D) and ICI pairings.
    Roles: 0 side A, 1 side B, 2 idle."""
    chips = [_chip_and_core(d, i) for i, d in enumerate(devices)]
    n = len(devices)
    sibling = []
    for i, (chip, core) in enumerate(chips):
        peers = [j for j, (c2, k2) in enumerate(chips) if c2 == chip and k2 != core]
        if len(peers) != 1:
            raise RuntimeError(
                f"device {i} ({chip}, core {core}) has no single sibling core"
            )
        sibling.append(peers[0])
    ici_peer, ici_role = list(range(n)), [2] * n
    for i, (chip, core) in enumerate(chips):
        if core != 0:
            continue
        target = (chip[0] ^ 1, *chip[1:])
        peers = [j for j, (c2, k2) in enumerate(chips) if c2 == target and k2 == 0]
        if len(peers) == 1:
            ici_peer[i], ici_role[i] = peers[0], 0 if chip[0] % 2 == 0 else 1
    for i in range(n):
        assert sibling[sibling[i]] == i and ici_peer[ici_peer[i]] == i
    return chips, sibling, ici_peer, ici_role


def _assignment(variant, chips, sibling, ici_peer, ici_role):
    if variant.get("table") == "ici":
        return ici_peer, ici_role
    roles = []
    first_chip = chips[0][0]
    for chip, core in chips:
        role = core ^ 1 if variant.get("reverse") else core
        if variant.get("one_chip") and chip != first_chip:
            role = 2
        roles.append(role)
    return sibling, roles


def _parts(variant):
    small = MODE == "interpret"
    if variant.get("flat"):
        parts = SMALL_FLAT_PARTS if small else FLAT_PARTS
    else:
        parts = SMALL_SLAB_PARTS if small else SLAB_PARTS
    return tuple((shape, np.dtype(dtype)) for shape, dtype in parts)


def _part_bytes(parts):
    return sum(int(np.prod(shape)) * dtype.itemsize for shape, dtype in parts)


def _pieces(ref, part, chunks):
    """Row chunks of a 32-bit part (whole 8-row tiles); other parts stay whole."""
    (rows, _), dtype = part
    if chunks > 1 and dtype.itemsize == 4 and rows % (8 * chunks) == 0:
        step = rows // chunks
        return [ref.at[pl.ds(k * step, step)] for k in range(chunks)]
    return [ref]


def _when(condition, function):
    if condition is not None:
        pl.when(condition)(function)


def _pair_barrier(peer, active):
    @pl.when(active)
    def _():
        barrier = pltpu.get_barrier_semaphore()
        pl.semaphore_signal(barrier, 1, device_id=(peer,), device_id_type=MESH_ID)
        pl.semaphore_wait(barrier, 1)


def _push_kernel(variant, parts):
    """A and/or B push slabs to the peer; A and/or B stream local slabs."""
    depth = variant["depth"]
    chunks = variant.get("chunks", 1)
    dst_space = variant.get("dst", "hbm")
    a_push, b_push = variant.get("a_push", True), variant.get("b_push", False)
    a_local, b_local = variant.get("a_local", False), variant.get("b_local", False)
    local_dst = variant.get("local_dst", "vmem")
    pushes, streams_any = a_push or b_push, a_local or b_local
    count = len(parts)

    def kernel(params, *refs):
        refs = list(refs)
        take = lambda k: [refs.pop(0) for _ in range(k)]  # noqa: E731
        src = take(count)
        push_hbm = take(count)  # outputs (the pushed slabs land here)
        local_hbm = take(count) if streams_any and local_dst == "hbm" else None
        push_send, push_receive, local_semaphores = take(3)
        push_vmem = take(count) if pushes and dst_space == "vmem" else None
        ring = take(count) if streams_any and local_dst == "vmem" else None
        assert not refs

        iterations, peer, role = params[0], params[1], params[2]
        is_a, is_b = role == 0, role == 1

        def either(on_a, on_b):
            condition = None
            for flag, side in ((on_a, is_a), (on_b, is_b)):
                if flag:
                    condition = side if condition is None else condition | side
            return condition

        sends, receives = either(a_push, b_push), either(b_push, a_push)
        streams = either(a_local, b_local)

        def push_copies(i):
            slab, slot, semaphore = i % SRC_SLABS, i % DST_SLOTS, i % depth
            copies = []
            for p in range(count):
                destination = (
                    push_hbm[p].at[slot]
                    if dst_space == "hbm"
                    else push_vmem[p].at[semaphore]
                )
                for source, target in zip(
                    _pieces(src[p].at[slab], parts[p], chunks),
                    _pieces(destination, parts[p], chunks),
                    strict=True,
                ):
                    copies.append(
                        pltpu.make_async_remote_copy(
                            source,
                            target,
                            push_send.at[semaphore],
                            push_receive.at[semaphore],
                            device_id=(peer,),
                            device_id_type=MESH_ID,
                        )
                    )
            return copies

        def local_copies(i):
            slab, slot = (i + SRC_SLABS // 2) % SRC_SLABS, i % depth
            copies = []
            for p in range(count):
                destination = (
                    ring[p].at[slot]
                    if local_dst == "vmem"
                    else local_hbm[p].at[i % DST_SLOTS]
                )
                for source, target in zip(
                    _pieces(src[p].at[slab], parts[p], chunks),
                    _pieces(destination, parts[p], chunks),
                    strict=True,
                ):
                    copies.append(
                        pltpu.make_async_copy(source, target, local_semaphores.at[slot])
                    )
            return copies

        def retire(j):
            def wait_sends():
                for copy in push_copies(j):
                    copy.wait_send()

            def wait_receives():
                for copy in push_copies(j):
                    copy.wait_recv()

            def wait_local():
                for copy in local_copies(j):
                    copy.wait()

            _when(sends, wait_sends)
            _when(receives, wait_receives)
            _when(streams, wait_local)

        _pair_barrier(peer, role < 2)

        @pl.loop(0, iterations)
        def _(i):
            pl.when(i >= depth)(lambda: retire(i - depth))

            def start_push():
                for copy in push_copies(i):
                    copy.start()

            def start_local():
                for copy in local_copies(i):
                    copy.start()

            _when(sends, start_push)
            _when(streams, start_local)

        @pl.loop(jnp.maximum(iterations - depth, 0), iterations)
        def _(j):
            retire(j)

    scratch = [
        pltpu.SemaphoreType.DMA((depth,)),
        pltpu.SemaphoreType.DMA((depth,)),
        pltpu.SemaphoreType.DMA((depth,)),
    ]
    if pushes and dst_space == "vmem":
        scratch += [pltpu.VMEM((depth, *shape), dtype) for shape, dtype in parts]
    if streams_any and local_dst == "vmem":
        scratch += [pltpu.VMEM((depth, *shape), dtype) for shape, dtype in parts]
    outputs = [((DST_SLOTS, *shape), dtype) for shape, dtype in parts]
    if streams_any and local_dst == "hbm":
        outputs += [((DST_SLOTS, *shape), dtype) for shape, dtype in parts]
    inputs = [((SRC_SLABS, *shape), dtype) for shape, dtype in parts]
    return kernel, inputs, outputs, scratch


def _slab_latency_kernel(variant, parts):
    """A pushes ``burst`` slabs at once and waits for B's doorbell; B rings
    after all of them land (one receive semaphore per slab)."""
    count = len(parts)
    burst = variant.get("burst", 1)

    def kernel(params, *refs):
        src, dst = refs[:count], refs[count : 2 * count]
        send, receive = refs[2 * count :]
        iterations, peer, role = params[0], params[1], params[2]

        def copies(i):
            return [
                pltpu.make_async_remote_copy(
                    src[p].at[(i * burst + b) % SRC_SLABS],
                    dst[p].at[(i * burst + b) % DST_SLOTS],
                    send.at[0],
                    receive.at[b],
                    device_id=(peer,),
                    device_id_type=MESH_ID,
                )
                for b in range(burst)
                for p in range(count)
            ]

        _pair_barrier(peer, role < 2)

        @pl.when(role == 0)
        def _():
            @pl.loop(0, iterations)
            def _(i):
                started = copies(i)
                for copy in started:
                    copy.start()
                for copy in started:
                    copy.wait_send()
                pl.semaphore_wait(pltpu.get_barrier_semaphore(), 1)

        @pl.when(role == 1)
        def _():
            @pl.loop(0, iterations)
            def _(i):
                for copy in copies(i):
                    copy.wait_recv()
                pl.semaphore_signal(
                    pltpu.get_barrier_semaphore(),
                    1,
                    device_id=(peer,),
                    device_id_type=MESH_ID,
                )

    scratch = [pltpu.SemaphoreType.DMA((1,)), pltpu.SemaphoreType.DMA((burst,))]
    inputs = [((SRC_SLABS, *shape), dtype) for shape, dtype in parts]
    outputs = [((DST_SLOTS, *shape), dtype) for shape, dtype in parts]
    return kernel, inputs, outputs, scratch


def _pingpong_kernel(variant):
    """A -> B into B's buffer 0, then B -> A into A's buffer 1, repeated."""
    rows, space = variant["nbytes"] // 512, variant["space"]

    def kernel(params, *refs):
        if space == "hbm":
            source, buffers, send, receive = refs
        else:
            unused_out, send, receive, source, buffers = refs
            unused_out[...] = jnp.zeros(unused_out.shape, unused_out.dtype)
        iterations, peer, role = params[0], params[1], params[2]

        def copy(direction):
            return pltpu.make_async_remote_copy(
                source,
                buffers.at[direction],
                send.at[direction],
                receive.at[direction],
                device_id=(peer,),
                device_id_type=MESH_ID,
            )

        _pair_barrier(peer, role < 2)

        @pl.when(role == 0)
        def _():
            @pl.loop(0, iterations)
            def _(i):
                outgoing = copy(0)
                outgoing.start()
                outgoing.wait_send()
                copy(1).wait_recv()

        @pl.when(role == 1)
        def _():
            @pl.loop(0, iterations)
            def _(i):
                copy(0).wait_recv()
                outgoing = copy(1)
                outgoing.start()
                outgoing.wait_send()

    f32 = np.dtype("float32")
    scratch = [pltpu.SemaphoreType.DMA((2,)), pltpu.SemaphoreType.DMA((2,))]
    if space == "hbm":
        return kernel, [((rows, 128), f32)], [((2, rows, 128), f32)], scratch
    scratch += [pltpu.VMEM((rows, 128), f32), pltpu.VMEM((2, rows, 128), f32)]
    return kernel, [], [((8, 128), f32)], scratch


def _semaphore_pingpong_kernel(variant):
    def kernel(params, out_ref):
        out_ref[...] = jnp.zeros(out_ref.shape, out_ref.dtype)
        iterations, peer, role = params[0], params[1], params[2]
        barrier = pltpu.get_barrier_semaphore()
        _pair_barrier(peer, role < 2)

        @pl.when(role == 0)
        def _():
            @pl.loop(0, iterations)
            def _(i):
                pl.semaphore_signal(
                    barrier, 1, device_id=(peer,), device_id_type=MESH_ID
                )
                pl.semaphore_wait(barrier, 1)

        @pl.when(role == 1)
        def _():
            @pl.loop(0, iterations)
            def _(i):
                pl.semaphore_wait(barrier, 1)
                pl.semaphore_signal(
                    barrier, 1, device_id=(peer,), device_id_type=MESH_ID
                )

    return kernel, [], [((8, 128), np.dtype("float32"))], []


def build(variant, mesh):
    kind = variant["kind"]
    parts = _parts(variant)
    if kind == "push":
        kernel, inputs, outputs, scratch = _push_kernel(variant, parts)
    elif kind == "slablat":
        kernel, inputs, outputs, scratch = _slab_latency_kernel(variant, parts)
    elif kind == "pingpong":
        kernel, inputs, outputs, scratch = _pingpong_kernel(variant)
    else:
        kernel, inputs, outputs, scratch = _semaphore_pingpong_kernel(variant)
    vmem_outputs = kind == "sempp" or (
        kind == "pingpong" and variant["space"] == "vmem"
    )
    interpret = (
        pltpu.InterpretParams(detect_races=True, dma_execution_mode="on_wait")
        if MODE == "interpret"
        else False
    )
    call = pl.pallas_call(
        kernel,
        out_shape=[jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in outputs],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[pl.BlockSpec(memory_space=pl.ANY) for _ in inputs],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM if vmem_outputs else pl.ANY)
                for _ in outputs
            ],
            scratch_shapes=scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID, vmem_limit_bytes=56 * 1024 * 1024
        ),
        interpret=interpret,
        name="d2d_" + variant["name"].replace("+", "_"),
    )

    def body(params, *arrays):
        return tuple(call(params, *arrays))

    n = len(inputs)
    fn = jax.jit(
        shard_map(
            body,
            mesh=mesh,
            in_specs=(P(AXIS),) * (n + 1),
            out_specs=(P(AXIS),) * len(outputs),
            check_vma=False,
        )
    )
    return fn, parts, inputs, outputs


def _source_arrays(inputs, devices_count):
    """Distinct values everywhere: global element index hashed per part."""
    arrays = []
    for k, (shape, dtype) in enumerate(inputs):
        total = devices_count * int(np.prod(shape))
        values = (np.arange(total, dtype=np.uint64) * 2654435761 + 97 * (k + 1)) % (
            2**32 if dtype.itemsize == 4 and dtype.kind == "u" else 251
        )
        if dtype.kind == "f":
            arrays.append(
                (values % 4093)
                .astype(dtype)
                .reshape(devices_count * shape[0], *shape[1:])
            )
        else:
            arrays.append(
                values.astype(dtype).reshape(devices_count * shape[0], *shape[1:])
            )
    return arrays


def _params(devices_count, iterations, peers, roles):
    table = np.zeros((devices_count, 4), np.int32)
    table[:, 0] = iterations
    table[:, 1] = peers
    table[:, 2] = roles
    return table.reshape(-1)


def _check_push(
    variant, host_sources, results, iterations, peers, roles, devices_count
):
    """Each receiver's staging slots hold its peer's slabs of the last
    ``iterations`` copies (push variants with HBM staging; burst copies count
    one each)."""
    if variant["kind"] not in ("push", "slablat") or variant.get("dst", "hbm") != "hbm":
        return "n/a"
    if variant["kind"] == "push" and not (
        variant.get("a_push", True) or variant.get("b_push")
    ):
        return "n/a"
    receivers = (
        [r for r in range(devices_count) if roles[r] == 1]
        if not variant.get("b_push")
        else [r for r in range(devices_count) if roles[r] in (0, 1)]
    )
    checked = 0
    for r in receivers:
        q = peers[r]
        for slot in range(DST_SLOTS):
            last = [i for i in range(iterations) if i % DST_SLOTS == slot]
            if not last:
                continue
            slab = last[-1] % SRC_SLABS
            for p, (source, result) in enumerate(zip(host_sources, results)):
                expected = source[q * SRC_SLABS + slab]
                got = result[r * DST_SLOTS + slot]
                if not np.array_equal(expected, got):
                    return f"MISMATCH receiver {r} slot {slot} part {p}"
                checked += 1
    return f"ok ({checked} arrays from the peer)"


def _check_pingpong(variant, host_sources, results, peers, roles, devices_count):
    if variant["kind"] != "pingpong" or variant["space"] != "hbm":
        return "n/a"
    rows = variant["nbytes"] // 512
    source, buffers = host_sources[0], results[0]
    for r in range(devices_count):
        if roles[r] not in (0, 1):
            continue
        q, direction = peers[r], 1 if roles[r] == 0 else 0
        got = buffers[r * 2 + direction]
        expected = source[q * rows : (q + 1) * rows]
        if not np.array_equal(got, expected):
            return f"MISMATCH device {r}"
    return "ok"


def _iteration_counts(variant):
    if MODE == "interpret":
        # D2D_INTERPRET_ITERS=low,high (default 1,2: no staging slot is rewritten,
        # so a race report is a protocol error rather than the bench's reuse).
        low, high = os.environ.get("D2D_INTERPRET_ITERS", "1,2").split(",")
        return int(low), int(high)
    if variant["kind"] in ("pingpong", "sempp"):
        return 200, 4200
    if variant["kind"] == "slablat":
        return 32, 704
    return 64, 1536


def _bytes_per_iteration(variant, parts):
    if variant["kind"] in ("push", "slablat"):
        return variant.get("burst", 1) * _part_bytes(parts)
    if variant["kind"] == "pingpong":
        return variant["nbytes"]
    return 0


_SOURCES = {}  # source arrays by input signature, shared by the variants


def run_variant(variant, mesh, devices, tables, out_path):
    name = variant["name"]
    chips, sibling, ici_peer, ici_role = tables
    peers, roles = _assignment(variant, chips, sibling, ici_peer, ici_role)
    count = len(devices)
    fn, parts, inputs, outputs = build(variant, mesh)
    sharding = NamedSharding(mesh, P(AXIS))
    if MODE == "aot":
        args = [jax.ShapeDtypeStruct((count * 4,), jnp.int32, sharding=sharding)] + [
            jax.ShapeDtypeStruct(
                (count * shape[0], *shape[1:]), dtype, sharding=sharding
            )
            for shape, dtype in inputs
        ]
        started = time.time()
        fn.lower(*args).compile()
        print(f"{name:18s} compiled OK ({time.time() - started:.1f} s)", flush=True)
        return
    key = tuple((tuple(shape), str(dtype)) for shape, dtype in inputs)
    if key not in _SOURCES:
        host = _source_arrays(inputs, count)
        _SOURCES[key] = host, [jax.device_put(a, sharding) for a in host]
    host_sources, sources = _SOURCES[key]
    n_low, n_high = _iteration_counts(variant)
    burst = variant.get("burst", 1)
    n_check = max(1, DST_SLOTS // burst)  # each staging slot written at most once
    params = {
        n: jax.device_put(_params(count, n, peers, roles), sharding)
        for n in (n_low, n_high, n_check)
    }

    def run(n):
        started = time.perf_counter()
        out = jax.block_until_ready(fn(params[n], *sources))
        return time.perf_counter() - started, out

    started = time.time()
    run(n_low)
    compile_s = time.time() - started
    print(f"  {name}: compiled and ran once ({compile_s:.1f} s)", flush=True)
    run(n_high)
    # Data check after a run that writes each staging slot once: the receivers
    # hold the peer's slabs.
    checked = (
        variant["kind"] in ("push", "slablat") and variant.get("dst", "hbm") == "hbm"
    )
    checked |= variant["kind"] == "pingpong" and variant["space"] == "hbm"
    check = "n/a"
    if checked:
        _, out = run(n_check)
        host_out = [np.asarray(x) for x in out]
        check = _check_push(
            variant, host_sources, host_out, n_check * burst, peers, roles, count
        )
        if check == "n/a":
            check = _check_pingpong(
                variant, host_sources, host_out, peers, roles, count
            )
    if MODE == "interpret" and os.environ.get("D2D_INTERPRET_TIMING") != "1":
        print(f"{name:18s} interpret OK, data check: {check}", flush=True)
        return check
    # The xprof profiler is not used: jax.profiler.start_trace segfaults on
    # TPU in the sglang-torchtpu:20261017 image (PLUGIN_Profiler_Api size
    # mismatch). The kernels loop on the device; differencing two counts
    # cancels launch and sync costs.
    reps = 7
    wall_low, wall_high = [], []
    for _ in range(reps):
        wall_low.append(run(n_low)[0])
        wall_high.append(run(n_high)[0])
    wall_ns = (min(wall_high) - min(wall_low)) / (n_high - n_low) * 1e9
    median_ns = (
        (statistics.median(wall_high) - statistics.median(wall_low))
        / (n_high - n_low)
        * 1e9
    )
    nbytes = _bytes_per_iteration(variant, parts)
    rate = f"{nbytes / wall_ns:7.3f} TB/s" if nbytes else "       -    "
    print(
        f"{name:18s} {wall_ns / 1e3:9.3f} us/iter  {rate}  (medians {median_ns / 1e3:.3f} "
        f"us/iter; runs {min(wall_low) * 1e3:.3f} ms @{n_low}, {min(wall_high) * 1e3:.3f} ms "
        f"@{n_high}; {nbytes} B/iter; check {check}; compile {compile_s:.1f} s)",
        flush=True,
    )
    record = dict(
        name=name,
        variant=variant,
        bytes_per_iteration=nbytes,
        us_per_iteration=wall_ns / 1e3,
        median_us_per_iteration=median_ns / 1e3,
        tb_per_s=(nbytes / wall_ns) if nbytes else None,
        iterations=[n_low, n_high],
        wall_low_s=wall_low,
        wall_high_s=wall_high,
        check=check,
        peers=list(map(int, peers)),
        roles=list(map(int, roles)),
    )
    try:
        with open(out_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:  # stdout still has the result
        print(f"  {name}: could not write {out_path}: {exc}", flush=True)
    return record


def main():
    devices = _devices()
    print(
        f"mode {MODE}; jax {jax.__version__}; {len(devices)} devices "
        f"({getattr(devices[0], 'device_kind', '?')})",
        flush=True,
    )
    tables = _tables(devices)
    chips = tables[0]
    for i, d in enumerate(devices):
        print(
            f"  mesh {i}: id {d.id} chip {chips[i][0]} core {chips[i][1]} "
            f"sibling {tables[1][i]} ici-peer {tables[2][i]} ici-role {tables[3][i]}",
            flush=True,
        )
    mesh = Mesh(np.asarray(devices), (AXIS,))
    out_dir = os.environ.get("D2D_OUT") or (
        "/mnt/kimik3-ssd/k31scratch-steal/d2d"
        if os.path.isdir("/mnt/kimik3-ssd")
        else "/tmp/d2d"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "d2d_bench.jsonl")
    only = [n for n in os.environ.get("D2D_ONLY", "").split(",") if n]
    failures = 0
    for variant in VARIANTS:
        if only and variant["name"] not in only:
            continue
        print(f"@@start {variant['name']}", flush=True)
        try:
            result = run_variant(variant, mesh, devices, tables, out_path)
            if isinstance(result, str) and result.startswith("MISMATCH"):
                failures += 1
            elif isinstance(result, dict) and str(result["check"]).startswith(
                "MISMATCH"
            ):
                failures += 1
        except Exception as exc:  # one variant failing must not lose the others
            failures += 1
            print(
                f"{variant['name']:18s} FAILED {type(exc).__name__}: {str(exc)[:600]}",
                flush=True,
            )
        print(f"@@done {variant['name']}", flush=True)
    print(f"results: {out_path}; failures {failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
