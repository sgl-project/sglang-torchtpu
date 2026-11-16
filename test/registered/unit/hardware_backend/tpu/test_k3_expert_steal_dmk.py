"""The sibling expert steal inside the decode megakernel's MoE (variant B,
KernelConfig.expert_steal) on two interpret-mode devices (one chip's cores).

Each layer, both cores take every row's routes, plan, send their packets and
push the given MXFP4 slabs (decode_megakernel._ExpertStealStep), start the
initial expert copies and the expert schedule, receive the sibling's packet,
run the real _run_routed_expert_stream (the stolen experts after the own
ones, copied in from staging an iteration early when the slab poll says the
slab has landed, which the harness answers by layer parity) and combine (the receiver returns the stolen
experts' FP32 outputs; the giver adds them after its own in ascending id
order). A semaphore handshake then orders the cores, as the routed reduction
does.

The routed outputs must be bit-identical to the steal layout alone (each core
running its own group's experts) at B8 and B16, on random MXFP4 experts and
routings that give in both directions, up to the slot cap, with stolen
experts on every stream path; also with eager DMAs. Perturbed plans (each
core plans every other layer from fewer rows, so a receiver expects more or
fewer or other experts than its sibling sends) must not hang: the receivers
follow the senders' packets (counted as mismatches), every expert still runs
once, and no DMA semaphore is left nonzero.

A stolen expert runs on the receiver in another pairing and path (pair or
tail, single-row or full) than it would on the giver, so this also checks
that a row's expert output does not depend on them. XLA:CPU runs without FMA
contraction (AVX at most): the TPU's vector unit multiplies and adds
separately (the decoder's bundles hold no fused multiply-add), while a
contracted ``acc += out * weight`` in the stream's finishers would round once
where the giver's add of a returned ``out * weight`` rounds twice."""

import os
import subprocess
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=600, suite="base-a-test-cpu")

GROUP_EXPERTS, TOP_K = 112, 16
K, W, OUT = 512, 256, 128
LAYERS = 8


def _routing(batch, seed):
    """Per layer: every row's 16 distinct global experts (top-k order), their
    probabilities and totals, the live rows and the latents. Groups 0 and 1
    (cores 0 and 1) draw from per-layer pools of varied size and skew, the
    other experts from groups 2-7."""
    import numpy as np

    rng = np.random.default_rng(seed)
    winners = np.zeros((LAYERS, batch, TOP_K), np.int32)
    probabilities = np.zeros((LAYERS, batch, TOP_K), np.float32)
    live = np.ones((LAYERS, batch), np.int32)
    # (pool size, most routes a row) for groups 0 and 1, per layer.
    shapes = {
        8: [((9, 4), (2, 1)), ((1, 1), (10, 5)), ((12, 6), (12, 6)), ((14, 3), (3, 3)),
            ((2, 2), (13, 5)), ((6, 2), (5, 2)), ((11, 5), (0, 0)), ((3, 1), (12, 4))],
        16: [((14, 5), (3, 2)), ((2, 1), (16, 6)), ((20, 4), (6, 6)), ((9, 9), (2, 2)),
             ((5, 2), (22, 5)), ((12, 3), (11, 3)), ((24, 6), (1, 1)), ((4, 4), (18, 3))],
    }[batch]
    for layer, pools in enumerate(shapes):
        if batch == 16 and layer % 3 == 1:
            live[layer, 13:] = 0  # null serving slots
        # B16 layer 2: group 0's highest expert is routed by every row (a
        # stolen expert past a compact tile).
        hot = batch == 16 and layer == 2
        ids = []
        for group, (size, _) in enumerate(pools):
            pool = rng.choice(GROUP_EXPERTS - 1, size, replace=False)
            if size and (layer % 2 or (hot and group == 0)):
                pool[0] = GROUP_EXPERTS - 1  # the highest local id
            ids.append(group * GROUP_EXPERTS + pool)
        for row in range(batch):
            picks = []
            for group, (size, most) in enumerate(pools):
                n = int(rng.integers(0 if row else 1, most + 1)) if size else 0
                if hot and group == 0:
                    rest = rng.choice(ids[group][1:], min(n, size - 1), replace=False)
                    picks += [ids[group][0], *rest]
                    continue
                picks += list(rng.choice(ids[group], min(n, size), replace=False))
            rest = rng.choice(
                np.arange(2 * GROUP_EXPERTS, 896), TOP_K - len(picks), replace=False
            )
            row_picks = np.array(picks + list(rest), np.int32)
            winners[layer, row] = row_picks[rng.permutation(TOP_K)]
            probabilities[layer, row] = rng.uniform(0.01, 0.2, TOP_K)
    totals = np.where(live > 0, probabilities.sum(2), 1.0).astype(np.float32)
    latents = rng.standard_normal((LAYERS, batch, K)).astype(np.float32)
    return winners, probabilities, totals, live, latents


def _plan(winners, live, group, slots):
    """The host plan of one core: (distinct own, distinct sibling, give, take,
    given ids ascending)."""
    routed = winners[live > 0].ravel()
    used = [
        sorted({int(e) % GROUP_EXPERTS for e in routed if e // GROUP_EXPERTS == g})
        for g in (group, group ^ 1)
    ]
    d_mine, d_theirs = len(used[0]), len(used[1])
    give = min(max(d_mine - d_theirs, 0) // 2, slots)
    take = min(max(d_theirs - d_mine, 0) // 2, slots)
    return d_mine, d_theirs, give, take, used[0][len(used[0]) - give :]


def _child(mode, batch):
    os.environ["XLA_FLAGS"] = (
        "--xla_force_host_platform_device_count=2 --xla_cpu_max_isa=AVX "
        "--xla_cpu_enable_platform_dependent_math=false"
    )
    os.environ["JAX_PLATFORMS"] = "cpu"
    from sglang_tpu.k3_programs.kimi_k3 import config

    config.install(
        config.KernelConfig(
            expert_steal=True,
            expert_schedule=True,
            b8_singleton_select=True,
            b16_one_hot_staging=True,
        )
    )
    import jax
    import jax.numpy as jnp
    import ml_dtypes
    import numpy as np
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk
    from sglang_tpu.k3_programs.kimi_k3.kernels import expert_steal as es

    slots = es.steal_slots(batch)
    perturbed = mode == "perturbed"
    winners, probabilities, totals, live, latents = _routing(batch, 7 + batch)
    rng = np.random.default_rng(11)

    def mxfp4(rows, columns):
        """[2 groups, experts, rows / 8, columns] random packed MXFP4 codes."""
        shape = (2, GROUP_EXPERTS, rows // 8, columns)
        return rng.integers(0, 2**32, shape, dtype=np.uint64).astype(np.uint32)

    def exponents(rows, columns):
        shape = (2, GROUP_EXPERTS, rows // 32, columns)
        return rng.integers(119, 123, shape).astype(np.uint8)

    weights = (mxfp4(K, 2 * W), exponents(K, 2 * W), mxfp4(W, OUT), exponents(W, OUT))
    rows = dmk._expert_accumulator_rows(batch, jnp.uint32, group_pending_routes=True)
    compact = dmk._compact_rows(batch)
    sources = (
        (pltpu.VMEM((compact, K), jnp.float32),) * 2
        if compact
        else (pltpu.VMEM((1, K), jnp.bfloat16),) * 2
    )

    def run(steal):
        def kernel(winners_ref, probabilities_ref, totals_ref, live_ref, latents_ref, *refs):
            hbm, refs = refs[:4], refs[4:]
            out_ref, routes_ref, info_ref, *refs = refs
            staging, refs = refs[:4], refs[4:]
            slot_vmem, refs = refs[:4], refs[4:]
            (expert_semaphores, gate_up_acc, down_acc, processed, first_compact,
             second_compact, table, table_semaphore, experts_vmem, probabilities_vmem,
             counts_vmem, source, accumulators, handshake, *protocol) = refs
            (packets, packet_tables, packet_send, packet_receive, slab_send, table_copies,
             slab_receives, errors, ready, returns, return_send, return_receive) = protocol
            rank = jax.lax.axis_index("x")
            sibling = rank ^ 1
            group = es.expert_group(rank)
            errors[0] = 0
            barrier = pltpu.get_barrier_semaphore()
            pl.semaphore_signal(
                barrier, 1, device_id=(sibling,), device_id_type=pl.DeviceIdType.MESH)
            pl.semaphore_wait(barrier, 1)
            moe_weights = dmk.MixtureOfExpertsWeights(
                *(None,) * 11, hbm[0], slot_vmem[0], hbm[1], slot_vmem[1], hbm[2],
                slot_vmem[2], hbm[3], slot_vmem[3], expert_semaphores)
            row_ids = jnp.arange(batch)

            def store_routes(selected, selected_probabilities, counts):
                experts_vmem[...] = jnp.pad(selected, ((0, 0), (0, 128 - TOP_K)))
                probabilities_vmem[...] = jnp.pad(
                    selected_probabilities, ((0, 0), (0, 128 - TOP_K)))
                counts_vmem[...] = jnp.broadcast_to(counts[:, None], counts_vmem.shape)

            @pl.loop(0, LAYERS)
            def layer_step(layer):
                layer_winners = winners_ref[layer]
                layer_probabilities = probabilities_ref[layer]
                rows_live = jnp.stack([live_ref[layer, b] for b in range(batch)]) != 0
                source[...] = latents_ref[layer]
                schedule = dmk._expert_schedule(processed.at[:8], table, table_semaphore)
                if steal:
                    step = dmk._ExpertStealStep(
                        es.ExpertSteal(
                            sibling, hbm, staging, packets, packet_tables,
                            (packet_send, packet_receive, slab_send, table_copies,
                             slab_receives),
                            errors, ready=ready, returns=returns,
                            return_semaphores=(return_send, return_receive),
                            # Interpret mode has no semaphore reads: the slab
                            # poll says "landed" in even layers (the next
                            # pair's stolen experts start an iteration early)
                            # and "in flight" in odd ones.
                            landed=lambda slot: layer % 2 == 0),
                        group=group, local_expert_count=GROUP_EXPERTS,
                        live_rows=rows_live, accumulators=accumulators,
                        weights=moe_weights)
                    plan = None
                    if perturbed:
                        # Each core plans every other layer without rows 0-3
                        # (core 0 the even ones, core 1 the odd ones).
                        dropped = (layer % 2 == rank) & (row_ids < 4)
                        plan = es.steal_plan(
                            layer_winners, rows_live & ~dropped, group, GROUP_EXPERTS, slots)
                    # The router feeds the planner one selection at a time.
                    planner = step.planner(batch, TOP_K)
                    for selection in range(TOP_K):
                        planner.add(selection, layer_winners[:, selection],
                                    layer_probabilities[:, selection])
                    selected, selected_probabilities, counts = step.begin(
                        planner, layer, plan=plan)
                    copy_expert = dmk._steal_copy_expert(moe_weights, GROUP_EXPERTS)
                else:
                    nothing = jnp.zeros((1, 128), jnp.int32)
                    selected, selected_probabilities, counts = es.steal_routes(
                        layer_winners, layer_probabilities, rows_live, group,
                        GROUP_EXPERTS, nothing != 0, nothing, 0)
                    copy_expert = moe_weights.make_expert_copies
                store_routes(selected, selected_probabilities, counts)
                dmk._start_initial_expert_weight_copies(
                    selected, counts, copy_expert, slot_count=4,
                    already_started=jnp.int32(0), group_pending_routes=True,
                    first_staged_expert=GROUP_EXPERTS if steal else None)
                dmk._start_expert_schedule(schedule, selected, counts)
                if steal:
                    def replace_routes(routes):
                        store_routes(*routes)
                        schedule.copy.wait()
                        dmk._start_expert_schedule(schedule, routes[0], routes[2])

                    step.receive(replace_routes)
                routes_ref[layer, :, :128] = experts_vmem[...]
                routes_ref[layer, :, 128:] = counts_vmem[...]
                dmk._run_routed_expert_stream(
                    source, experts_vmem[:, :TOP_K], probabilities_vmem[:, :TOP_K],
                    counts_vmem[:, 0], totals_ref[layer][:, 0], *slot_vmem,
                    accumulators.at[0], gate_up_acc, down_acc, processed, first_compact,
                    second_compact, copy_expert, expert_input_size=K, expert_width=W,
                    expert_output_size=OUT, group_pending_routes=True,
                    expert_schedule=schedule,
                    stolen=step.stream_hooks() if steal else None)
                if steal:
                    step.combine(accumulators.at[0], OUT)
                    info_ref[layer, 0] = packet_tables[0, es.HEADER_ROW, es.GIVE_LANE]
                    info_ref[layer, 1] = packet_tables[1, es.HEADER_ROW, es.GIVE_LANE]
                    info_ref[layer, 2] = errors[0]
                out_ref[layer] = accumulators[0]
                # The routed reduction orders the cores before the next layer.
                pl.semaphore_signal(
                    handshake, 1, device_id=(sibling,), device_id_type=pl.DeviceIdType.MESH)
                pl.semaphore_wait(handshake, 1)

        vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
        smem = pl.BlockSpec(memory_space=pltpu.SMEM)
        anywhere = pl.BlockSpec(memory_space=pl.ANY)
        call = pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct((LAYERS, batch, OUT), jnp.float32),
                jax.ShapeDtypeStruct((LAYERS, batch, 256), jnp.int32),
                jax.ShapeDtypeStruct((LAYERS, 8), jnp.int32),
                *(jax.ShapeDtypeStruct((slots, *w.shape[2:]), w.dtype) for w in weights),
            ),
            in_specs=[vmem, vmem, vmem, smem, vmem] + [anywhere] * 4,
            out_specs=(vmem, vmem, smem, *(anywhere,) * 4),
            scratch_shapes=[
                *(pltpu.VMEM((4, *w.shape[2:]), w.dtype) for w in weights),
                pltpu.SemaphoreType.DMA((4, 4)),
                pltpu.VMEM((rows[0], 4 * W), jnp.float32),
                pltpu.VMEM((rows[1], OUT), jnp.float32),
                pltpu.VMEM((batch, 128), jnp.int32),
                *sources,
                pltpu.SMEM((8, 128), jnp.int32),
                pltpu.SemaphoreType.DMA,
                pltpu.VMEM((batch, 128), jnp.int32),
                pltpu.VMEM((batch, 128), jnp.float32),
                pltpu.VMEM((batch, 128), jnp.int32),
                pltpu.VMEM((batch, K), jnp.bfloat16),
                pltpu.VMEM((1 + slots if steal else 1, batch, OUT), jnp.float32),
                pltpu.SemaphoreType.REGULAR,
                *es.scratch_shapes(slots),
                *es.return_scratch_shapes(slots, batch, OUT),
            ],
            compiler_params=pltpu.CompilerParams(collective_id=9),
            interpret=pltpu.InterpretParams(
                detect_races=True,
                dma_execution_mode="eager" if mode == "eager" else "on_wait",
            ),
        )

        def body(winners, probabilities, totals, live, latents, *weights):
            out, routes, info, *_ = call(
                winners, probabilities, totals, live, latents, *(w[0] for w in weights))
            return out[None], routes[None], info[None]

        mesh = Mesh(np.asarray(jax.devices()), ("x",))
        f = jax.jit(jax.shard_map(
            body, mesh=mesh, in_specs=(P(),) * 5 + (P("x"),) * 4,
            out_specs=(P("x"),) * 3, check_vma=False))
        wide_totals = np.broadcast_to(totals[..., None], (*totals.shape, 128)).copy()
        return [np.asarray(x) for x in f(
            winners, probabilities, wide_totals, live,
            latents.astype(ml_dtypes.bfloat16), *weights)]

    reference, reference_routes, _ = run(False)
    stolen, routes, info = run(True)
    races = interpret_pallas_call.races
    raced = bool(races is not None and races.races_found)

    ok = True
    # Without a steal the routes are the router's ownership pass: own group,
    # live rows, selection order.
    for core in (0, 1):
        for layer in range(LAYERS):
            expect = np.zeros((batch, 128), np.int32)
            counts = np.zeros(batch, np.int32)
            for row in range(batch):
                for e in winners[layer, row]:
                    if e // GROUP_EXPERTS == core and live[layer, row]:
                        expect[row, counts[row]] = e % GROUP_EXPERTS
                        counts[row] += 1
            ok &= np.array_equal(reference_routes[core, layer, :, :128], expect)
            ok &= np.array_equal(reference_routes[core, layer, :, 128], counts)
    print(f"reference routes match the router's ownership: {ok}", flush=True)
    coverage = set()
    for layer in range(LAYERS):
        plans = [_plan(winners[layer], live[layer], core, slots) for core in (0, 1)]
        gives = [int(info[core, layer, 0]) for core in (0, 1)]
        received = [int(info[core, layer, 1]) for core in (0, 1)]
        exact = all(
            np.array_equal(stolen[c, layer].view(np.uint32), reference[c, layer].view(np.uint32))
            for c in (0, 1))
        close = np.allclose(stolen[:, layer], reference[:, layer], rtol=1e-5, atol=1e-6)
        print(f"B{batch} layer {layer}: distinct experts {[p[0] for p in plans]} "
              f"give {gives} received {received} "
              f"mismatches {[int(info[c, layer, 2]) for c in (0, 1)]} "
              f"exact {exact} close {close}", flush=True)
        if perturbed:
            ok &= close
            continue
        ok &= exact
        for core in (0, 1):
            d_mine, d_theirs, give, take, given = plans[core]
            ok &= gives[core] == give and received[core] == take
            if not give:
                coverage.add("no give" if abs(d_mine - d_theirs) < 2 else "")
                continue
            light = core ^ 1
            coverage.add(f"give from core {core}")
            coverage.add("capped" if d_mine - d_theirs >= 2 * slots + 2 else "under the cap")
            coverage.add(
                "first stolen in the initial window"
                if plans[light][0] < 4
                else "first stolen in the prefetch window")
            for e in given:
                n = sum(
                    bool(live[layer, r]) and core * GROUP_EXPERTS + e in winners[layer, r]
                    for r in range(batch))
                coverage.add(
                    "stolen single-row" if n == 1
                    else "stolen multi-row" if n <= 8 else "stolen past a compact tile")
            # The receiver's routes name the stolen experts 112 + slot.
            light_routes = routes[light, layer, :, :128]
            stolen_ids = (light_routes >= GROUP_EXPERTS) & (light_routes != es.UNROUTED)
            taken = set(light_routes[stolen_ids].tolist())
            ok &= taken == set(range(GROUP_EXPERTS, GROUP_EXPERTS + give))
    coverage.discard("")
    print(f"coverage {sorted(coverage)}", flush=True)
    errors = info[:, -1, 2].tolist()
    if perturbed:
        ok &= errors[0] > 0 or errors[1] > 0
    else:
        ok &= errors == [0, 0]
        wanted = {"give from core 0", "give from core 1", "capped", "under the cap",
                  "first stolen in the initial window",
                  "first stolen in the prefetch window",
                  "stolen single-row", "stolen multi-row"}
        if batch == 16:
            wanted.add("stolen past a compact tile")
        ok &= wanted <= coverage
    # Interpret mode prints any DMA semaphore left nonzero at kernel exit; the
    # test fails on it.
    print(f"mismatches {errors} races {raced}", flush=True)
    ok &= not raced
    print("OK" if ok else "FAIL", flush=True)
    return 0 if ok else 1


class TestK3ExpertStealMegakernel(CustomTestCase):
    def _run(self, mode, batch):
        try:
            from jax.experimental.pallas import tpu as pltpu
        except ImportError:
            self.skipTest("no Pallas TPU")
        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("no Pallas TPU interpret mode")
        result = subprocess.run(
            [sys.executable, os.path.abspath(__file__), mode, str(batch)],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        output = result.stdout + result.stderr[-4000:]
        self.assertIn("OK", result.stdout.splitlines()[-1:] or [""], output)
        self.assertNotIn("non-zero count", result.stdout, output)

    def test_b8_bit_identical_to_the_steal_layout(self):
        self._run("exact", 8)

    def test_b16_bit_identical_to_the_steal_layout(self):
        self._run("exact", 16)

    def test_b8_eager_dmas(self):
        self._run("eager", 8)

    def test_perturbed_plan_neither_hangs_nor_drops_experts(self):
        self._run("perturbed", 16)


if __name__ == "__main__":
    if len(sys.argv) > 2 and not sys.argv[1].startswith("-"):
        sys.exit(_child(sys.argv[1], int(sys.argv[2])))
    unittest.main()
