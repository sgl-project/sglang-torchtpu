"""The sibling expert steal protocol (kernels/expert_steal.py) on two
interpret-mode devices (one chip's cores): per layer both cores plan from the
same routes, the heavier core pushes its highest-id experts' slabs into the
sibling's staging, the sibling runs them after its own, and a pair reduction
sums the partials. Integer weights make every sum exact, so each layer's total
must equal the reference bit for bit. A perturbed plan on one core (it plans
from fewer rows) must neither hang nor change the totals: the receiver follows
the sender's packet, and the mismatch is counted."""

import os
import subprocess
import sys
import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.ci.gate_matrix import full_matrix
from sglang.test.test_utils import CustomTestCase

# Whole kernels in Pallas interpret mode on host devices, minutes a case: the
# TPU nightly runs it (the CPU runners do not install jax).
register_tpu_ci(
    est_time=480,
    suite="nightly-tpu-4-v7",
    nightly=True,
    chips=1,
    labels=["singlehost", "kimi-k3", "k3-programs"],
)

GROUP_EXPERTS = 16  # local experts per group
ROWS, TOP_K, LAYERS, SLOTS = 8, 6, 8, 2
PARTS = ((16, 128), (8, 128))


def _child(mode):
    """Run the harness in interpret mode, or (``aot``) compile it for one
    tpu7x chip's two cores without a TPU."""
    aot = mode == "aot"
    if not aot:
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
        os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.k3_programs.kimi_k3.kernels import expert_steal as es

    rng = np.random.default_rng(7)
    experts = 2 * GROUP_EXPERTS
    # Per layer, rows draw from a skewed split of the two groups (so both
    # directions, ties, and gives past the slot cap all occur).
    winners = np.zeros((LAYERS, ROWS, TOP_K), np.int32)
    for layer in range(LAYERS):
        heavy = layer % 2
        weight = [0.5, 0.75, 0.9, 1.0][layer // 2]
        for row in range(ROWS):
            p = np.where(
                np.arange(experts) // GROUP_EXPERTS == heavy, weight, 1 - weight
            )
            p = np.maximum(p, 1e-3)
            winners[layer, row] = rng.choice(
                experts, TOP_K, replace=False, p=p / p.sum()
            )
    weights = [
        rng.integers(-8, 9, (2, LAYERS, GROUP_EXPERTS, *shape)).astype(np.float32)
        for shape in PARTS
    ]
    perturbed = mode == "perturbed"
    # The plan's live rows: core 1 drops rows 0-3 in odd layers when perturbed.
    plan_live = np.ones((2, LAYERS, ROWS), np.int32)
    if perturbed:
        plan_live[1, 1::2, :4] = 0

    def contribution(part0, part1):
        return jnp.sum(part0, axis=0, keepdims=True) + 2 * jnp.sum(
            part1, axis=0, keepdims=True
        )

    def kernel(
        winners_ref,
        plan_live_ref,
        part0_ref,
        part1_ref,
        totals_ref,
        info_ref,
        staging0_ref,
        staging1_ref,
        packets,
        packet_tables,
        packet_send,
        packet_receive,
        slab_send,
        table_copies,
        slab_receives,
        errors,
        buffer0,
        buffer1,
        partial,
        other,
        local_semaphores,
        pair_semaphores,
    ):
        rank = jax.lax.axis_index("x")
        sibling = rank ^ 1
        group, sibling_group = rank, rank ^ 1
        errors[0] = 0
        barrier = pltpu.get_barrier_semaphore()
        pl.semaphore_signal(
            barrier, 1, device_id=(sibling,), device_id_type=pl.DeviceIdType.MESH
        )
        pl.semaphore_wait(barrier, 1)

        def run_expert(part0_source, part1_source, global_id, enabled, routes):
            copies = [
                pltpu.make_async_copy(part0_source, buffer0, local_semaphores.at[0]),
                pltpu.make_async_copy(part1_source, buffer1, local_semaphores.at[1]),
            ]
            for copy in copies:
                copy.start()
            for copy in copies:
                copy.wait()
            routed = jnp.any(routes == global_id, axis=1, keepdims=True) & enabled
            partial[...] += jnp.where(
                routed, contribution(buffer0[...], buffer1[...]), 0.0
            )

        for layer in range(LAYERS):
            routes = winners_ref[layer]
            # Live rows as the decode stack builds them: stacked SMEM scalars.
            live = jnp.stack([plan_live_ref[layer, b] for b in range(ROWS)]) != 0
            plan = es.steal_plan(routes, live, group, GROUP_EXPERTS, SLOTS)
            steal = es.ExpertSteal(
                sibling,
                (part0_ref.at[layer], part1_ref.at[layer]),
                (staging0_ref, staging1_ref),
                packets,
                packet_tables,
                (packet_send, packet_receive, slab_send, table_copies, slab_receives),
                errors,
            )
            steal.begin(plan, layer)
            partial[...] = jnp.zeros(partial.shape, jnp.float32)
            for expert in range(GROUP_EXPERTS):
                # Every own expert a row routes, except the ones given away.
                keep = ~plan.gives(expert)
                run_expert(
                    part0_ref.at[layer, expert],
                    part1_ref.at[layer, expert],
                    group * GROUP_EXPERTS + expert,
                    keep,
                    routes,
                )
            expected = es.packet(plan.take, layer, plan.ids, SLOTS)
            count, table = steal.receive(expected)

            @pl.loop(0, count)
            def _(slot):
                steal.wait_slab(slot)
                run_expert(
                    staging0_ref.at[slot],
                    staging1_ref.at[slot],
                    sibling_group * GROUP_EXPERTS + table[es.IDS_ROW, slot],
                    True,
                    routes,
                )

            steal.finish()
            info_ref[layer, 0] = plan.give
            info_ref[layer, 1] = plan.take
            info_ref[layer, 2] = count
            info_ref[layer, 3] = plan.d_mine
            info_ref[layer, 4] = plan.d_theirs
            info_ref[layer, 5] = errors[0]
            # The layer's reduction: it also orders the cores before the next
            # layer reuses the staging slots and packets.
            exchange = pltpu.make_async_remote_copy(
                partial,
                other,
                pair_semaphores.at[0],
                pair_semaphores.at[1],
                device_id=(sibling,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            exchange.start()
            exchange.wait()
            totals_ref[layer] = jnp.where(
                rank == 0, partial[...] + other[...], other[...] + partial[...]
            )

    interpret = (
        False
        if aot
        else pltpu.InterpretParams(
            detect_races=True,
            dma_execution_mode=mode.split("-")[-1] if "-" in mode else "on_wait",
        )
    )
    call = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((LAYERS, ROWS, 128), jnp.float32),
            jax.ShapeDtypeStruct((LAYERS, 8), jnp.int32),
            jax.ShapeDtypeStruct((SLOTS, *PARTS[0]), jnp.float32),
            jax.ShapeDtypeStruct((SLOTS, *PARTS[1]), jnp.float32),
        ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=(
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ),
        scratch_shapes=[
            *es.scratch_shapes(SLOTS),
            pltpu.VMEM(PARTS[0], jnp.float32),
            pltpu.VMEM(PARTS[1], jnp.float32),
            pltpu.VMEM((ROWS, 128), jnp.float32),
            pltpu.VMEM((ROWS, 128), jnp.float32),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
        ],
        compiler_params=pltpu.CompilerParams(collective_id=9),
        interpret=interpret,
    )

    def body(winners, plan_live, part0, part1):
        totals, info, _, _ = call(winners, plan_live[0], part0[0], part1[0])
        return totals[None], info[None]

    if aot:
        from jax.experimental import topologies

        devices = topologies.get_topology_desc("tpu7x:2x2x1", "tpu").devices
        pair = sorted(devices, key=lambda d: d.id)[:2]
        assert tuple(pair[0].coords) == tuple(pair[1].coords), "not one chip"
    else:
        pair = jax.devices()
    mesh = Mesh(np.asarray(pair), ("x",))
    specs = (P(), P("x"), P("x"), P("x"))
    f = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=specs,
            out_specs=(P("x"), P("x")),
            check_vma=False,
        )
    )
    if aot:
        args = [
            jax.ShapeDtypeStruct(
                np.shape(a), np.asarray(a).dtype, sharding=NamedSharding(mesh, spec)
            )
            for a, spec in zip((winners, plan_live, *weights), specs)
        ]
        f.lower(*args).compile()
        print("compiled for tpu7x\nOK", flush=True)
        return 0
    totals, info = (np.asarray(x) for x in f(winners, plan_live, *weights))

    # Reference: every routed expert once, from its group's weights.
    contributions = np.zeros((experts, LAYERS, 128), np.float32)
    for e in range(experts):
        g, k = divmod(e, GROUP_EXPERTS)
        contributions[e] = weights[0][g, :, k].sum(1) + 2 * weights[1][g, :, k].sum(1)
    reference = np.zeros((LAYERS, ROWS, 128), np.float32)
    for layer in range(LAYERS):
        for row in range(ROWS):
            for e in winners[layer, row]:
                reference[layer, row] += contributions[e, layer]
    ok = True
    for layer in range(LAYERS):
        d = [
            len(
                {
                    int(e) % GROUP_EXPERTS
                    for e in winners[layer].ravel()
                    if e // GROUP_EXPERTS == g
                }
            )
            for g in (0, 1)
        ]
        give = [min(max(d[g] - d[g ^ 1], 0) // 2, SLOTS) for g in (0, 1)]
        exact = all(np.array_equal(totals[c, layer], reference[layer]) for c in (0, 1))
        line = (
            f"layer {layer} D {d} give {give} core0 {info[0, layer, :6].tolist()} "
            f"core1 {info[1, layer, :6].tolist()} exact {exact}"
        )
        print(line, flush=True)
        ok &= exact
        if not perturbed:
            for c in (0, 1):
                ok &= info[c, layer, 0] == give[c] and info[c, layer, 2] == give[c ^ 1]
                ok &= info[c, layer, 3] == d[c] and info[c, layer, 4] == d[c ^ 1]
    errors = info[:, -1, 5].tolist()
    ok &= errors == [0, 0] if not perturbed else (errors[0] > 0 or errors[1] > 0)
    races = interpret_pallas_call.races
    raced = bool(races is not None and races.races_found)
    print(f"mismatches {errors} races {raced}", flush=True)
    ok &= not raced
    print("OK" if ok else "FAIL", flush=True)
    return 0 if ok else 1


class TestK3ExpertSteal(CustomTestCase):
    def _run(self, mode):
        try:
            from jax.experimental.pallas import tpu as pltpu
        except ImportError:
            pltpu = None
        if not hasattr(pltpu, "InterpretParams"):
            # The TPU nightly runs this file: there it fails instead of skipping.
            self.assertFalse(full_matrix(), "no Pallas TPU interpret mode")
            self.skipTest("no Pallas TPU interpret mode")
        result = subprocess.run(
            [sys.executable, os.path.abspath(__file__), mode],
            capture_output=True,
            text=True,
            timeout=900,
        )
        output = result.stdout + result.stderr[-4000:]
        self.assertIn("OK", result.stdout.splitlines()[-1:] or [""], output)

    def test_steal_exact_on_wait(self):
        self._run("exact-on_wait")

    def test_steal_exact_eager(self):
        self._run("exact-eager")

    def test_perturbed_plan_neither_hangs_nor_changes_totals(self):
        self._run("perturbed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        sys.exit(_child(sys.argv[1]))
    unittest.main()
