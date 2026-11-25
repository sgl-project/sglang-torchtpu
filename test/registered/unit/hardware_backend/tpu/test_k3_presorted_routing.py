"""``KernelConfig.decode_presorted_routing``: the decode router's presorted
top-16 (decode_megakernel._presorted_route_selections) and the steal plan in
vectors (expert_steal.StealPlanner.vector_plan).

The seven-input network must sort every input (the 0-1 principle: all 128
binary inputs). On four interpret-mode devices, ``_route_hidden_states`` with
``presorted`` must give the argmax loop's routes bit for bit -- with the steal
planner (its codes, probabilities, used-expert lanes and totals, and the
vector plan against ``plan``) and without (the ownership pass's experts,
probabilities, counts and totals) -- at 8 and 16 rows (null rows at 16), on
adversarial router logits: bf16 logit ties (with zero or duplicated biases),
saturated sigmoids, ties within a lane, across lanes (the lower lane holding
the higher id) and at the 16th/17th boundary (also each alone in a batch),
sixteen picks from two lanes,
neighbours one ulp apart, NaN, +inf and -inf biases (fewer than 16 finite
scores; a lone NaN or +inf among distinct scores), subnormal biases and
probabilities. The presorted selections alone must flag every batch whose
winners differ from a first-index argmax and none of the tie-free ones
(distinct biases, one-ulp neighbours), so those run the presorted path.
(XLA:CPU flushes subnormals as the TPU does, so they reach the selection as
zeros; the subnormal flag guards a backend whose adds keep them.)"""

import os
import subprocess
import sys
import time
import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.ci.gate_matrix import full_matrix
from sglang.test.test_utils import CustomTestCase

# Whole kernels in Pallas interpret mode on host devices, minutes a case: the
# TPU nightly runs it (the CPU runners do not install jax).
register_tpu_ci(
    est_time=1400,
    suite="nightly-tpu-4-v7",
    nightly=True,
    chips=1,
    labels=["singlehost", "kimi-k3", "k3-programs"],
)

H, E, K, LOCAL = 256, 896, 16, 112
# Batches with one kind of planted tie a row: (experts, at the 16th/17th
# scores). Across lanes (lanes 72 and 44, the lower lane holding the higher
# id), and within lane 5, where the network puts expert 773 ahead of 133.
PLANTED = {
    "tie_cross": ((200, 300), False),
    "tie_lane": ((133, 773), False),
    "tie_cross16": ((200, 300), True),
    "tie_lane16": ((133, 773), True),
}
# Tie-free batches: the presorted path must decide them (no rerun); batches
# with a NaN score or a planted tie must rerun.
DECIDED = ("rand", "near", "infrand")
FLAGGED = ("nan", "nanrand", *PLANTED)


def _child(kinds, batches, modes):
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    os.environ["JAX_PLATFORMS"] = "cpu"
    global np, jax, jnp, ml_dtypes, pl, pltpu, P, dmk, es, mesh
    import jax
    import jax.numpy as jnp
    import ml_dtypes
    import numpy as np
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.k3_programs.kimi_k3 import config as kc

    kc.install(kc.KernelConfig(expert_steal=True))
    from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk
    from sglang_tpu.k3_programs.kimi_k3.kernels import expert_steal as es

    mesh = Mesh(np.asarray(jax.devices()), ("tp",))
    failures = 0
    for B in batches:
        for kind in kinds:
            hidden, router, bias, live, names = make_batch(
                B, kind, seed=17 * B + len(kind)
            )
            for mode in modes:
                steal = mode == "steal"
                t0 = time.time()
                ref = run(hidden, router, bias, live, presorted=False, steal=steal)
                new = run(hidden, router, bias, live, presorted=True, steal=steal)
                bad = [
                    i
                    for i, (a, b) in enumerate(zip(ref, new))
                    if a.tobytes() != b.tobytes()
                ]
                failures += bool(bad)
                status = "OK" if not bad else f"MISMATCH outputs {bad}"
                print(
                    f"B{B} bias={kind} {mode}: {status} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
            raw = hidden.astype(np.float32) @ router.astype(np.float32)
            probs = np.asarray(jax.nn.sigmoid(jnp.asarray(raw)), np.float32)
            scores = (probs + bias).astype(np.float32)
            winners, flag = presorted_flags(scores)
            reference_winners, _ = reference(scores, probs)
            ambiguous = bool(flag[0, 0])
            differing = [
                r
                for r in range(B)
                if winners[r].tobytes() != reference_winners[r].tobytes()
            ]
            print(
                f"   presorted alone: ambiguous={ambiguous} rows differing from a "
                f"first-index argmax: {[(r, names[r]) for r in differing]}",
                flush=True,
            )
            if differing and not ambiguous:
                print("   ERROR: differing rows without the ambiguity flag", flush=True)
                failures += 1
            if kind in DECIDED and ambiguous:
                print("   ERROR: a tie-free batch was flagged", flush=True)
                failures += 1
            if kind in FLAGGED and not ambiguous:
                print(
                    "   ERROR: a batch with a NaN or a planted tie was not flagged",
                    flush=True,
                )
                failures += 1
    print("FAILURES", failures, flush=True)
    return 0 if failures == 0 else 1


def run(hidden, router, bias, live, presorted, steal):
    B = hidden.shape[0]

    def kernel(live_ref, hidden_ref, bias_ref, router_ref, *refs):
        outs, (scores_vmem, sends, receives) = refs[:-3], refs[-3:]
        live_rows = jnp.stack([live_ref[b] for b in range(B)]) != 0
        rank = jax.lax.axis_index("tp")
        common = dict(
            selected_expert_count=K,
            local_expert_count=LOCAL,
            expert_tensor_parallel=4,
            on_local_selection=lambda *unused: None,
            after_selection=lambda unused: None,
            live_rows=live_rows,
            wait_for_projection_half=lambda slot: None,
            presorted=presorted,
        )
        if steal:
            planner = es.StealPlanner(
                B,
                K,
                live_rows,
                es.expert_group(rank),
                LOCAL,
                es.steal_slots(B),
                dmk.STEAL_TIMING[8 if B <= 8 else 16],
            )
            totals = dmk._route_hidden_states(
                hidden_ref[...],
                router_ref,
                bias_ref,
                scores_vmem,
                sends.at[0],
                receives.at[0],
                steal_planner=planner,
                **common,
            )
            if presorted:
                # The probabilities follow the plan; the plan stays in vectors.
                plan = planner.vector_plan()
                totals = totals()
            else:
                plan = planner.plan()
            lanes = jax.lax.broadcasted_iota(jnp.int32, (8, 128), 1)
            table = jnp.zeros((8, 128), jnp.int32)
            for k, v in enumerate(
                (plan.d_mine, plan.d_theirs, plan.give, plan.take, *plan.ids)
            ):
                table = jnp.where(lanes == k, v, table)
            codes_out, probs_out, mine_out, theirs_out, totals_out, plan_out = outs
            codes_out[...] = planner.codes
            probs_out[...] = planner.probabilities
            mine_out[...] = planner.mine.astype(jnp.int32)
            theirs_out[...] = planner.theirs.astype(jnp.int32)
            totals_out[...] = jnp.broadcast_to(totals[:, None], totals_out.shape)
            plan_out[...] = table
        else:
            experts, probabilities, counts, totals = dmk._route_hidden_states(
                hidden_ref[...],
                router_ref,
                bias_ref,
                scores_vmem,
                sends.at[0],
                receives.at[0],
                **common,
            )
            e_out, p_out, c_out, t_out = outs
            e_out[...] = experts
            p_out[...] = probabilities
            c_out[...] = jnp.broadcast_to(counts[:, None], c_out.shape)
            t_out[...] = jnp.broadcast_to(totals[:, None], t_out.shape)

    vmem = pltpu.VMEM
    if steal:
        shapes = (
            (B, K, jnp.int32),
            (B, K, jnp.float32),
            (B, 128, jnp.int32),
            (B, 128, jnp.int32),
            (B, 128, jnp.float32),
            (8, 128, jnp.int32),
        )
    else:
        shapes = (
            (B, K, jnp.int32),
            (B, K, jnp.float32),
            (B, 128, jnp.int32),
            (B, 128, jnp.float32),
        )

    def body(live_rows, hidden, bias, router):
        lane = jax.lax.axis_index("tp") % 2
        halves = jnp.pad(router, ((0, 0), (0, 1024 - E)))
        halves = halves.reshape(H, 2, 512).transpose(1, 0, 2)
        own = jnp.where(lane == 0, halves[0], halves[1])
        peer = jnp.where(lane == 0, halves[1], halves[0])
        outputs = pl.pallas_call(
            kernel,
            out_shape=tuple(jax.ShapeDtypeStruct(s[:2], s[2]) for s in shapes),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=[pl.BlockSpec(memory_space=vmem)] * 3,
                out_specs=[pl.BlockSpec(memory_space=vmem)] * len(shapes),
                scratch_shapes=[
                    vmem((B, 1024), jnp.float32),
                    pltpu.SemaphoreType.DMA((1,)),
                    pltpu.SemaphoreType.DMA((1,)),
                ],
            ),
            interpret=True,
        )(live_rows, hidden, bias, jnp.stack([own, peer]))
        return tuple(x[None] for x in outputs)

    f = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P(),) * 4,
            out_specs=(P("tp"),) * len(shapes),
            check_vma=False,
        )
    )
    return [np.asarray(x) for x in f(live.astype(np.int32), hidden, bias, router)]


def presorted_flags(scores_np):
    """_presorted_route_selections alone: the winners and the ambiguity flag."""
    B = scores_np.shape[0]

    def kernel(s_ref, w_out, a_out):
        w, amb = dmk._presorted_route_selections(
            s_ref[...], after_selection=lambda unused: None
        )
        slots = jax.lax.broadcasted_iota(jnp.int32, (B, K), 1)
        wv = jnp.zeros((B, K), jnp.int32)
        for s in range(K):
            wv = jnp.where(slots == s, w[s][:, None], wv)
        w_out[...] = wv
        a_out[...] = jnp.broadcast_to(jnp.where(amb, 1, 0), a_out.shape)

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    out = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((B, K), jnp.int32),
            jax.ShapeDtypeStruct((8, 128), jnp.int32),
        ),
        in_specs=[vmem],
        out_specs=[vmem] * 2,
        interpret=True,
    )(scores_np)
    return [np.asarray(x) for x in out]


def reference(scores, probs):
    """numpy top-16 with argmax-first-index (XLA:CPU argmax) semantics."""
    B = scores.shape[0]
    s = scores.copy()
    W = np.zeros((B, K), np.int32)
    Pr = np.zeros((B, K), np.float32)
    for k in range(K):
        w = np.argmax(s, axis=1)
        W[:, k] = w
        Pr[:, k] = probs[np.arange(B), w]
        s[np.arange(B), w] = -np.inf
    return W, Pr


def bf16(x):
    return np.asarray(x, np.float32).astype(ml_dtypes.bfloat16)


def make_batch(B, bias_kind, seed):
    """One-hot hidden rows pick per-row logit vectors (router rows); the bias
    is the batch's. Returns hidden, router, bias, live, row names."""
    rng = np.random.default_rng(seed)
    logits = []
    names = []

    def base(scale=1.0, cap=3.0):
        return np.clip(rng.standard_normal(E) * scale, -cap, cap)

    lanes = lambda e: e % 128
    for row in range(B):
        kind = row % 12
        x = base()
        if kind == 0:
            name = "random"
            x = rng.standard_normal(E) * 2
        elif kind == 1:
            name = "flat"
            x = np.zeros(E)
        elif kind == 2:
            name = "saturate"
            x[rng.choice(E, 40, replace=False)] = 40.0
        elif kind == 3:  # within-lane tie at the top: experts l + 128 j1, l + 128 j2
            j1, j2 = rng.choice(7, 2, replace=False)
            l = int(rng.integers(128))
            x[l + 128 * j1] = x[l + 128 * j2] = 8.0
            name = f"lane_tie {l + 128 * j1},{l + 128 * j2}"
        elif kind == 4:  # cross-lane tie, the lower lane holding the higher id
            a = 2 * 128 + 44
            b = 128 + 72
            x[a] = x[b] = 8.0
            name = "cross_tie 300,200"
        elif kind == 5:  # 16th == 17th within a lane
            top = rng.choice(E, 15, replace=False)
            x[top] = 9.0 + 0.25 * np.arange(15)
            free = [e for e in range(E) if e not in set(top)]
            l = int(rng.choice([lanes(e) for e in free]))
            js = [j for j in range(7) if (l + 128 * j) not in set(top)]
            j1, j2 = rng.choice(js, 2, replace=False)
            x[l + 128 * j1] = x[l + 128 * j2] = 6.0
            name = f"boundary_lane {l + 128 * j1},{l + 128 * j2}"
        elif kind == 6:  # 16th == 17th across lanes
            top = rng.choice(E, 15, replace=False)
            x[top] = 9.0 + 0.25 * np.arange(15)
            free = [e for e in range(E) if e not in set(top)]
            a, b = rng.choice(free, 2, replace=False)
            while lanes(a) == lanes(b):
                a, b = rng.choice(free, 2, replace=False)
            x[a] = x[b] = 6.0
            name = f"boundary_cross {a},{b}"
        elif kind == 7:  # all 16 picks from one lane and its neighbours
            l = int(rng.integers(128))
            x[l + 128 * np.arange(7)] = 7.0 + 0.5 * np.arange(7)[::-1]
            x[(l + 1) % 128 + 128 * np.arange(7)] = 6.75 - 0.5 * np.arange(7)
            name = f"one_lane {l}"
        elif bias_kind == "near" and kind in (1, 8, 11):
            name = "near_pairs"
            x = np.clip(rng.standard_normal(E), -3, 3)
            x[:128] = 8.0
        elif kind == 8 and bias_kind in ("rand", "ulp", "nanrand", "infrand"):
            name = "random3"
            x = rng.standard_normal(E) * 3
        elif kind == 8:  # subnormal probabilities among the top
            x[rng.choice(E, 20, replace=False)] = -95.0
            name = "subnormal_prob"
        elif kind == 9:  # 1-ulp neighbours (bf16 logits one ulp apart)
            a, b = rng.choice(E, 2, replace=False)
            x[a] = 8.0
            x[b] = 8.0625
            name = f"near {a},{b}"
        elif kind == 10:
            name = "random2"
            x = rng.standard_normal(E) * 0.5
        else:
            name = "tie_pairs"
            for _ in range(8):
                a, b = rng.choice(E, 2, replace=False)
                v = float(bf16(rng.uniform(4, 6)))
                x[a] = x[b] = v
        if bias_kind in PLANTED:
            # One planted tie a row (the batch's only kind): experts PLANTED[0]
            # and [1] share a bias and a logit, at the top or as the 16th and
            # 17th scores (fifteen higher logits elsewhere); tiny distinct biases
            # keep every other score distinct.
            pair, at_boundary = PLANTED[bias_kind]
            x = rng.standard_normal(E)
            if at_boundary:
                others = [e for e in range(E) if e not in pair]
                x[rng.choice(others, 15, replace=False)] = 9.0 + 0.25 * np.arange(15)
            x[list(pair)] = 6.0 if at_boundary else 8.0
            name = f"{bias_kind} {pair[0]},{pair[1]}"
        logits.append(x)
        names.append(name)
    logits = bf16(np.stack(logits))
    hidden = np.zeros((B, H), ml_dtypes.bfloat16)
    router = bf16(rng.standard_normal((H, E)) * 0.05)
    for row in range(B):
        hidden[row, row] = 1
        router[row] = logits[row]
    if bias_kind in PLANTED:
        bias = rng.standard_normal(E) * 1e-4
        pair = PLANTED[bias_kind][0]
        bias[pair[1]] = bias[pair[0]]
    elif bias_kind == "rand":
        bias = rng.standard_normal(E) * 0.1
    elif bias_kind == "zero":
        bias = np.zeros(E)
    elif bias_kind == "dup":
        bias = 0.001 * rng.integers(-3, 4, E)
    elif bias_kind == "nan":
        bias = np.zeros(E)
        bias[777] = np.nan
    elif bias_kind == "nanrand":  # one NaN score, otherwise tie-free
        bias = rng.standard_normal(E) * 0.1
        bias[777] = np.nan
    elif bias_kind == "infrand":  # one +inf score, otherwise tie-free
        bias = rng.standard_normal(E) * 0.1
        bias[300] = np.inf
    elif bias_kind == "posinf":
        bias = np.zeros(E)
        bias[[5, 600]] = np.inf
    elif bias_kind == "neginf":
        bias = np.full(E, -np.inf)
        bias[rng.choice(E, 11, replace=False)] = 0.0
    elif bias_kind == "subnormal":
        bias = np.zeros(E)
        bias[rng.choice(E, 30, replace=False)] = 1e-40
    elif bias_kind == "near":
        # Distinct biases; experts 2k and 2k + 1 (k < 64) score one ulp apart
        # at logit 8 (rows of kind 'near_pairs' give them all logit 8).
        bias = (rng.standard_normal(E) * 0.1).astype(np.float32)
        s8 = np.float32(1) / (np.float32(1) + np.exp(np.float32(-8.0)))
        for k in range(64):
            a, b = 2 * k, 2 * k + 1
            target = np.nextafter(np.float32(s8 + bias[a]), np.float32(np.inf))
            cand = np.float32(target - s8)
            for _ in range(8):
                got = np.float32(s8 + cand)
                if got == target:
                    break
                cand = np.nextafter(
                    cand, np.float32(np.inf) if got < target else np.float32(-np.inf)
                )
            assert np.float32(s8 + cand) == target and np.float32(s8 + bias[a]) < target
            bias[b] = cand
    elif bias_kind == "ulp":
        bias = np.zeros(E)
        s8 = np.float32(1) / (np.float32(1) + np.exp(np.float32(-8.0)))
        bias[rng.choice(E, 200, replace=False)] = np.float32(np.spacing(s8))
    else:
        raise ValueError(bias_kind)
    bias = np.asarray(bias, np.float32)[None, :]
    live = np.ones(B, bool)
    if B == 16:
        live[[3, 9, 14]] = False
    return hidden, router, bias, live, names


class TestSortingNetwork(CustomTestCase):
    def test_sorts_every_binary_input(self):
        from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk

        for word in range(128):
            values = [(word >> bit) & 1 for bit in range(7)]
            for first, second in dmk.SORT_SEVEN:
                if values[second] > values[first]:
                    values[first], values[second] = values[second], values[first]
            self.assertEqual(values, sorted(values, reverse=True), word)


class TestPresortedRouting(CustomTestCase):
    def _run(self, kinds, batch, modes):
        try:
            from jax.experimental.pallas import tpu as pltpu
        except ImportError:
            pltpu = None
        if not hasattr(pltpu, "InterpretParams"):
            # The TPU nightly runs this file: there it fails instead of skipping.
            self.assertFalse(full_matrix(), "no Pallas TPU interpret mode")
            self.skipTest("no Pallas TPU interpret mode")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, os.path.abspath(__file__), kinds, str(batch), modes],
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,
        )
        output = result.stdout[-6000:] + result.stderr[-3000:]
        self.assertIn("FAILURES 0", result.stdout.splitlines()[-1:] or [""], output)

    def test_b8_matches_the_argmax_loop(self):
        self._run(
            "rand,near,zero,dup,nan,nanrand,infrand,neginf,subnormal,"
            "tie_cross,tie_lane,tie_cross16,tie_lane16",
            8,
            "steal,stock",
        )

    def test_b16_matches_the_argmax_loop(self):
        self._run("rand,near,zero,posinf,ulp", 16, "steal,stock")


if __name__ == "__main__":
    if len(sys.argv) > 3 and not sys.argv[1].startswith("-"):
        sys.exit(
            _child(
                sys.argv[1].split(","),
                [int(b) for b in sys.argv[2].split(",")],
                sys.argv[3].split(","),
            )
        )
    unittest.main()
