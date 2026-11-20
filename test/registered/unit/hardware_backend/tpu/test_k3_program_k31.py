"""Kimi-K3.1 on the K3 programs: the DSA indexer in the model config, its
weight families in the kernels' per-rank layout, and the pre-shard shapes."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.k3_programs.kimi_k3.kernels import Config, load

register_cpu_ci(est_time=13000, suite="base-a-test-cpu")

FULL = [4 * i for i in range(1, 24)] + [93]  # 1-based, as the HF config
DSA = {
    "dsa_layers": FULL,
    "index_topk": 2048,
    "index_head_dim": 128,
    "index_num_heads": 32,
    "index_rope_dim": 0,
    "index_k_norm_type": "layernorm",
    "index_topk_freq": 1,
    "index_skip_topk_offset": 1,
}


def _text_config(dsa=None):
    c = {
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "latent_moe_use_norm": True,
        "moe_renormalize": True,
        "moe_router_activation_func": "sigmoid",
        "hidden_act": "situ",
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "num_expert_group": 1,
        "routed_scaling_factor": 1.0,
        "hidden_size": 7168,
        "routed_expert_hidden_size": 3584,
        "moe_intermediate_size": 3072,
        "num_shared_experts": 2,
        "intermediate_size": 33792,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_hidden_layers": 93,
        "num_attention_heads": 96,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "vocab_size": 163840,
        "attn_res_block_size": 12,
        "rms_norm_eps": 1e-5,
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "linear_attn_config": {
            "full_attn_layers": FULL,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "use_full_rank_gate": True,
        },
    }
    if dsa is not None:
        c["dsa_config"] = dsa
    return {"text_config": c}


def _config(dsa=None):
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "config.json").write_text(json.dumps(_text_config(dsa)))
        return Config.from_checkpoint(directory)


class TestK31Config(CustomTestCase):
    def test_k3_has_no_indexer_and_k31_has_its_dsa(self):
        k3 = _config()
        self.assertEqual(k3, Config())
        self.assertFalse(k3.dsa)
        k31 = _config(DSA)
        self.assertTrue(k31.dsa)
        self.assertEqual(
            (k31.index_topk, k31.index_heads, k31.index_dim), (2048, 32, 128)
        )
        self.assertEqual(k31.without_dsa(), Config())

    def test_unsupported_dsa_settings_are_refused(self):
        for name, value in (
            ("index_rope_dim", 64),
            ("index_topk_freq", 2),
            ("dsa_layers", FULL[:-1]),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                _config(dict(DSA, **{name: value}))


class _Checkpoint(load.Checkpoint):
    """``load.Checkpoint`` over in-memory tensors (no shard files)."""

    def __init__(self, tensors, config):
        self.tensors, self.config = tensors, config

    def read(self, key, selection=None):
        value = self.tensors[key]
        return value if selection is None else value[selection]


class TestIndexerFamilies(CustomTestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        base = f"{load.PREFIX}layers.3.self_attn.indexer."
        self.raw = {
            "wq_b": rng.standard_normal((32 * 128, 1536)).astype(np.float32),
            "wk": rng.standard_normal((128, 7168)).astype(np.float32),
            "weights_proj": rng.standard_normal((32, 7168)).astype(np.float32),
            "k_norm.weight": rng.standard_normal(128).astype(np.float32),
            "k_norm.bias": rng.standard_normal(128).astype(np.float32),
        }
        tensors = {
            base + (k if k.startswith("k_norm") else k + ".weight"): v
            for k, v in self.raw.items()
        }
        self.ckpt = _Checkpoint(tensors, _config(DSA))

    def test_rank_layouts(self):
        for rank in (0, 5, 31):
            with self.subTest(rank=rank):
                wqb = self.ckpt.dense("d_wqb", 3, rank)
                np.testing.assert_array_equal(
                    wqb, self.raw["wq_b"][rank * 128 : (rank + 1) * 128].T
                )
                np.testing.assert_array_equal(
                    self.ckpt.dense("d_wk", 3, rank), self.raw["wk"].T
                )
                proj = self.ckpt.dense("d_wproj", 3, rank)
                self.assertEqual(proj.shape, (7168, 128))
                np.testing.assert_array_equal(proj[:, :32], self.raw["weights_proj"].T)
                self.assertFalse(proj[:, 32:].any())
                norm = self.ckpt.dense("d_knorm", 3, rank).astype(np.float32)
                np.testing.assert_allclose(
                    norm[0], self.raw["k_norm.weight"], rtol=1e-2
                )
                np.testing.assert_allclose(norm[1], self.raw["k_norm.bias"], rtol=1e-2)

    def test_shapes_and_layers(self):
        shapes = load.selected_local_shapes(93, 7168, dsa=True)
        self.assertEqual(shapes["d_wqb"], (24, 1536, 128))
        self.assertEqual(shapes["d_knorm"], (24, 2, 128))
        self.assertNotIn("d_wqb", load.selected_local_shapes(93, 7168))
        layers = load._family_layers(self.ckpt.config, "d_wk", 93)
        self.assertEqual(layers, list(self.ckpt.config.full_attention))
        # Every pre-shard family comes from the checkpoint (the decode's d_wks /
        # d_wpt are derived at load: rank_loader.decode_indexer_families).
        names = [n for n in shapes if n.startswith("d_")]
        self.assertEqual(sorted(names), ["d_knorm", "d_wk", "d_wproj", "d_wqb"])
        for name in names:
            value = self.ckpt.dense(name, 3, 0)
            self.assertEqual(tuple(value.shape), shapes[name][1:])


class TestTokenParallelLatentUp(CustomTestCase):
    """``KernelConfig.moe_token_parallel_up``: the ``latent_up_tp`` family is
    latent_up's block 2 * host + core, and the token-parallel MoE tail
    (``_TokenParallelUp``) in Pallas interpret mode on 32 host devices
    (TestAttentionHostBf16's harness, stale NaN bytes wherever a round lands,
    race detection) matches an op-for-op model of the lane-block tail bit for
    bit at 8 and 16 rows, with the shared reduce-scatter before or after the
    routed send: every rank's routed host sums (its rank pair's tokens) and
    owned output rows. A quarter of the partials spread over 2^+-12 and some
    carry cancelling large terms, so the lane and host sums round in FP32 and
    a change of their order shows. With ``odd_host_round``
    (``KernelConfig.moe_host_round_odd_cores``: only the core-1 ranks send the
    routed host round, to both cores of their chip on each other host) every
    case gives the same bits; so do three tails back to back on fresh partials
    (eager DMAs, the ranks of one core lagging before the host round), each
    tail's core-1 copies landing in host slots the previous tail's readers
    were done with, also with each tail's output round finished in the next
    tail's first code (``KernelConfig.moe_output_round_next_layer``: the
    round's buffer and semaphores alone, no barrier between tails)."""

    # The tail kernel and the lane-block model (after TestAttentionHostBf16's
    # PRELUDE).
    DEFS = r"""
from sglang_tpu.k3_programs.kimi_k3.kernels import load

F32, EPS = jnp.float32, 1e-5
bfv = lambda x: x.astype(jnp.bfloat16).astype(F32)
LAG = int(os.environ.get("LAG", "100"))  # loop trips a pair index


def tail(B, E, Sh, norm, blocks, late, races, odd=False, lag_core=None, deferred=False):
    # E [32, n, B, 4096] and Sh [32, n, B, H]: n tails back to back, each on
    # fresh partials. Before the next tail each rank waits until its seven
    # local peers have read their local rounds' slots (the decoder's attention
    # rounds between two MoE tails order more). With ``lag_core`` the ranks of
    # that core spin (p + 1) * LAG loop trips (pair p) before the host round.
    # With ``deferred`` a tail only starts its output round; the next tail's
    # first code (after the last tail, the kernel's) finishes it, the tail's
    # local buffers holding NaN.
    T, n = B // 4, E.shape[1]

    def kernel(e_ref, sh_ref, norm_ref, up_hbm, out_ref, tokens_out, sl, el, shh, eh,
               ssend, srecv, send, recv, outv, osend, orecv, routed, up_vmem, up_sem,
               shared, owned, spin, consumed):
        rank = jax.lax.axis_index("tp")
        for ref in (sl, el, shh, eh):
            ref[...] = jnp.full(ref.shape, jnp.nan, ref.dtype)
        outv[...] = jnp.full(outv.shape, jnp.nan, outv.dtype)
        spin[...] = jnp.zeros(spin.shape, F32)
        barrier = pltpu.get_barrier_semaphore()
        for peer in range(1, 32):
            pl.semaphore_signal(barrier, 1, device_id=(rank ^ peer,),
                                device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(barrier, 31)
        tp = dmk._TokenParallelUp(B, HALF, rank, sl, shh, el, eh, ssend, srecv, send,
                                  recv, outv, osend, orecv, odd_host_round=odd)

        def lag(i, carry):
            spin[...] = spin[...] + 1.0
            return carry

        def finish_round(i):  # the stack's finish_output_round
            for ref in (routed, up_vmem, shared, owned):
                ref[...] = jnp.full(ref.shape, jnp.nan, ref.dtype)
            dmk.finish_token_parallel_outputs(outv, osend, orecv, owned)
            out_ref[i] = owned[...]

        for i in range(n):
            if i and deferred:
                finish_round(i - 1)
            if i:
                pl.semaphore_wait(consumed, 7)
            copy = pltpu.make_async_copy(up_hbm, up_vmem, up_sem)
            copy.start()
            routed[...] = e_ref[i]
            shared[...] = sh_ref[i]
            if not late:
                tp.begin_shared(shared)
            tp.routed_send(routed[...])
            if late:
                tp.begin_shared(shared)
            else:
                tp.finish_shared_local()
            tp.routed_local_reduce()
            if lag_core is not None:
                trips = jnp.where(rank % 2 == lag_core, (rank % 8) // 2 + 1, 0) * LAG
                jax.lax.fori_loop(0, trips, lag, 0)
            tp.start_routed_host()
            if late:
                tp.finish_shared_local()
            tp.routed_host_reduce(routed)
            if i + 1 < n:  # this rank is done with its local rounds' slots
                for offset in range(1, 8):
                    pl.semaphore_signal(consumed, 1, device_id=(rank ^ offset,),
                                        device_id_type=pl.DeviceIdType.MESH)
            tokens_out[i] = routed[:T, :HALF]
            up = tp.project(routed, norm_ref, up_vmem, copy, epsilon=EPS,
                            latent_feature_count=HALF)
            tp.finish_routed_host_sends()
            tp.finish_shared_host(routed)
            if deferred:
                tp.start_outputs(up, routed)
                continue
            tp.send_outputs(up, routed, owned)
            out_ref[i] = owned[...]
        if deferred:
            finish_round(n - 1)

    vmem, dma = pltpu.VMEM, pltpu.SemaphoreType.DMA
    scratch = [*dmk.communication_scratch(B)[:4], dma((10,)), dma((10,)), dma((16,)),
               dma((16,)), vmem((12, T * 8, 128), jnp.bfloat16), dma((7,)), dma((7,)),
               vmem((B, 4096), F32), vmem((HALF, 1024), jnp.bfloat16), dma(()),
               vmem((B, H), F32), vmem((T, H), jnp.bfloat16), vmem((8, 128), F32),
               pltpu.SemaphoreType.REGULAR]

    def body(e, sh, nm, up):
        out = pl.pallas_call(
            kernel,
            out_shape=(jax.ShapeDtypeStruct((n, T, H), jnp.bfloat16),
                       jax.ShapeDtypeStruct((n, T, HALF), F32)),
            in_specs=[pl.BlockSpec(memory_space=vmem)] * 3 + [pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=[pl.BlockSpec(memory_space=vmem)] * 2,
            scratch_shapes=scratch,
            compiler_params=pltpu.CompilerParams(has_side_effects=True, collective_id=7),
            interpret=pltpu.InterpretParams(detect_races=True, dma_execution_mode=races),
        )(e[0], sh[0], nm, up[0])
        return tuple(x[None] for x in out)

    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("tp"), P("tp"), P(), P("tp")),
                              out_specs=(P("tp"),) * 2, check_vma=False))
    result = [np.asarray(x) for x in f(E, Sh, norm, blocks)]
    assert not ipc.races.races_found, (B, late, races, odd, lag_core, deferred)  # this call's
    return result


@jax.jit
def dot16(x, w):  # dmk._dot: whole 16-row tiles
    x = jnp.pad(x, ((0, -x.shape[0] % 16), (0, 0)))
    return jax.lax.dot_general(x, w, (((1,), (0,)), ((), ())),
                               preferred_element_type=F32)[:x.shape[0]]


def lane_block_tail(B, E, Sh, norm, U):
    # Routed reduce-scatter (eight lanes, BF16), host all-reduce, BF16 gather,
    # RMS norm on [B, 4096], 1024-column latent-up on every lane, shared
    # reduce-scatter to lane blocks (eight lanes, BF16, four hosts), BF16.
    # The lane-block reductions' shapes: lane l's token shard (B / 8 tokens)
    # over the eight lanes [8, 28 B / 8, 128], then over the four hosts.
    k = B // 8
    shards = []
    for lane in range(8):
        S = [bfv(jnp.sum(bfv(E[8 * h:8 * h + 8, lane * k:(lane + 1) * k, :HALF]).reshape(8, 28 * k, 128), axis=0))
             for h in range(4)]
        shards.append(jnp.sum(jnp.stack(S), axis=0).reshape(k, HALF))
    tokens = jnp.concatenate(shards, axis=0)
    routed = bfv(jnp.zeros((B, 4096), F32).at[:, :HALF].set(bfv(tokens)))
    normalized = (routed * jax.lax.rsqrt(jnp.sum(routed * routed, axis=1, keepdims=True)
                                         / HALF + EPS) * norm).astype(jnp.bfloat16)
    out = np.zeros((B, H), BF16)
    for L in range(7):
        cols = slice(1024 * L, 1024 * L + 1024)
        SS = [bfv(jnp.sum(bfv(Sh[8 * h:8 * h + 8, :, cols]).reshape(8, B * 8, 128), axis=0))
              for h in range(4)]
        shared = jnp.sum(jnp.stack(SS), axis=0).reshape(B, 1024)
        up = dot16(normalized[:, :HALF], U[:, cols])[:B]
        out[:, cols] = np.asarray((up + shared).astype(jnp.bfloat16))
    return out, np.asarray(tokens)


rng = np.random.default_rng(0)
U = (rng.standard_normal((HALF, H)) * HALF ** -0.5).astype(BF16)
U_pad = np.concatenate([U, np.zeros((HALF, 1024), BF16)], axis=1)
norm = np.zeros((1, 4096), np.float32)
norm[0, :HALF] = 1 + 0.1 * rng.standard_normal(HALF)
blocks = np.stack([U_pad[:, 1024 * load.token_parallel_up_block(r):][:, :1024]
                   for r in range(32)])
spread = lambda shape: np.where(rng.random(shape) < 0.25, 2.0 ** rng.integers(-12, 13, shape), 1.0)


def partials(B):
    E = np.zeros((32, B, 4096), np.float32)
    E[:, :, :HALF] = (rng.standard_normal((32, B, HALF)) * 0.05 * spread((32, B, HALF))
                      * rng.choice([1.0, 30.0], (1, B, 1), p=[0.75, 0.25]))
    Sh = (rng.standard_normal((32, B, H)) * 0.02 * spread((32, B, H))).astype(np.float32)
    # Cancelling large terms on lanes 1 and 6 of every host: the lane order
    # decides which small terms they absorb.
    for X, width in ((E, HALF), (Sh, H)):
        hit = rng.random((B, width)) < 0.05
        for h in range(4):
            X[8 * h + 1, :, :width] += np.where(hit, 1024.0, 0.0)
            X[8 * h + 6, :, :width] -= np.where(hit, 1024.0, 0.0)
        # And across hosts (lane 2 of hosts 0 and 3): the host order decides.
        hit = rng.random((B, width)) < 0.05
        X[2, :, :width] += np.where(hit, 2048.0, 0.0)
        X[26, :, :width] -= np.where(hit, 2048.0, 0.0)
    return E, Sh


def check(B, out, tokens, E, Sh, what):
    # Every rank's routed host sums and owned rows: its pair's tokens.
    want, want_tokens = lane_block_tail(B, jnp.asarray(E), jnp.asarray(Sh), norm, jnp.asarray(U))
    T = B // 4
    for r in range(32):
        pair = r % 8 // 2
        assert tokens[r].tobytes() == want_tokens[pair * T:(pair + 1) * T].tobytes(), (what, r)
        assert out[r].tobytes() == want[pair * T:(pair + 1) * T].tobytes(), (what, r)
"""

    MAIN = (
        DEFS
        + r"""
for B, late, races in ((8, True, "on_wait"), (16, True, "on_wait"), (8, False, "eager")):
    E, Sh = partials(B)
    runs = []
    for odd in (False, True):
        out, tokens = tail(B, E[:, None], Sh[:, None], norm, blocks, late, races, odd)
        check(B, out[:, 0], tokens[:, 0], E, Sh, (B, late, races, odd))
        runs.append(out.tobytes() + tokens.tobytes())
        print(f"B{B} late={late} {races} odd_host_round={odd}: bit-exact", flush=True)
    assert runs[0] == runs[1], (B, late, races)
print("OK")
"""
    )

    # Three tails back to back (core-0 ranks lagging, then core-1 ranks; then,
    # core-0 ranks lagging, with the output rounds finished in the next tail).
    LOOP = (
        DEFS
        + r"""
for odd, lag_core, deferred in ((True, 0, False), (True, 1, False), (False, 0, False), (True, 0, True)):
    E, Sh = (np.stack(x, axis=1) for x in zip(*(partials(8) for _ in range(3))))
    out, tokens = tail(8, E, Sh, norm, blocks, True, "eager", odd, lag_core, deferred)
    for i in range(3):
        check(8, out[:, i], tokens[:, i], E[:, i], Sh[:, i], (odd, lag_core, deferred, i))
    print(f"odd_host_round={odd} lagging core {lag_core} deferred round {deferred}: three tails bit-exact",
          flush=True)
print("OK")
"""
    )

    def test_family_is_the_host_core_block(self):
        from unittest import mock

        from sglang_tpu.k3_programs.kimi_k3.kernels import rank_loader

        blocks = [load.token_parallel_up_block(r) for r in range(32)]
        self.assertEqual(sorted(blocks), sorted(list(range(8)) * 4))
        for r in range(32):  # the rank pair on every host: blocks 0..7 once
            self.assertEqual(
                sorted(blocks[r ^ o] for o in (0, 1, 8, 9, 16, 17, 24, 25)),
                list(range(8)),
            )
        rng = np.random.default_rng(0)
        up = rng.standard_normal((7168, 3584)).astype(np.float32)
        key = f"{load.PREFIX}layers.2.block_sparse_moe.routed_expert_up_proj.weight"
        ckpt = _Checkpoint({key: up}, _config(DSA))
        for rank in (0, 3, 9, 14, 30, 31):
            with self.subTest(rank=rank):
                block = load.token_parallel_up_block(rank)
                np.testing.assert_array_equal(
                    ckpt.dense_selected("latent_up_tp", 2, rank),
                    ckpt.dense_selected("latent_up", 2, block),
                )
        self.assertFalse(ckpt.dense_selected("latent_up_tp", 2, 31).any())
        lanes = {b: np.full((2, 3584, 1024), b, np.float32) for b in range(8)}
        fake = lambda root, rank, layers: mock.Mock(read_dense=lambda n: lanes[rank])
        with mock.patch.object(rank_loader, "RankDir", fake):
            for rank in (6, 13, 27):
                family = rank_loader.token_parallel_up_family("x", rank, 3)
                self.assertTrue((family == load.token_parallel_up_block(rank)).all())

    def _run(self, script):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), NPROC="128")
        result = subprocess.run(
            [sys.executable, "-c", TestAttentionHostBf16.PRELUDE + script],
            capture_output=True,
            text=True,
            env=env,
            timeout=3600,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )
        self.assertNotIn("non-zero count", result.stdout + result.stderr)
        self.assertNotIn("RACE DETECTED", result.stdout + result.stderr)

    def test_tail_matches_the_lane_block_tail(self):
        self._run(self.MAIN)

    def test_odd_host_round_tails_back_to_back(self):
        self._run(self.LOOP)


class TestOutputRoundNextLayer(CustomTestCase):
    """``KernelConfig.moe_output_round_next_layer``: the token-parallel MoE
    tail (decode_late_shared_down's order) over three layers of a loop, each
    layer with its own routed partials and shared outputs, as in the
    decoder: every layer starts its output round
    (``_TokenParallelUp.start_outputs``) and the next layer's first code (the
    last layer's: the code after the loop) runs
    ``finish_token_parallel_outputs`` from the round's transport and
    semaphores alone, after NaN has been written over every other buffer the
    tail uses and over the owned rows. A barrier after it stands in for the
    next layer's attention rounds, and each layer's round lands on stale NaN
    bytes. Pallas interpret mode on 32 host devices (TestAttentionHostBf16's
    harness) with race detection, DMAs executed on wait and eagerly: every
    rank's owned rows of every layer equal those of ``send_outputs`` (the
    round finished in its own layer) bit for bit, and no DMA semaphore is
    left nonzero."""

    MAIN = r"""
import sys

from sglang_tpu.k3_programs.common import collectives32
from sglang_tpu.k3_programs.kimi_k3.kernels import load

F32, EPS, LAYERS = jnp.float32, 1e-5, 3
DMA_MODE, BATCHES = sys.argv[1], tuple(int(b) for b in sys.argv[2].split(","))


def tails(B, E, Sh, norm, blocks, deferred):
    T = B // 4
    nan = lambda ref: jnp.full(ref.shape, jnp.nan, ref.dtype)

    def kernel(e_ref, sh_ref, norm_ref, up_hbm, out_ref, sl, el, shh, eh, ssend, srecv,
               send, recv, outv, osend, orecv, routed, up_vmem, up_sem, rows):
        rank = jax.lax.axis_index("tp")
        for ref in (sl, el, shh, eh, outv, rows):
            ref[...] = nan(ref)
        tp = dmk._TokenParallelUp(B, HALF, rank, sl, shh, el, eh, ssend, srecv, send,
                                  recv, outv, osend, orecv)

        def finish(layer):
            if deferred:
                # Only the round's transport and semaphores are still live.
                for ref in (sl, el, shh, eh, routed, up_vmem, rows):
                    ref[...] = nan(ref)
                dmk.finish_token_parallel_outputs(outv, osend, orecv, rows)
            out_ref[layer] = rows[...]

        @pl.loop(0, LAYERS)
        def layer(index):
            if deferred:
                pl.when(index > 0)(lambda: finish(index - 1))
            outv[...] = nan(outv)
            collectives32.barrier()
            copy = pltpu.make_async_copy(up_hbm, up_vmem, up_sem)
            copy.start()
            routed[...] = e_ref[index]
            tp.routed_send(routed[...])
            tp.begin_shared(sh_ref.at[index])
            tp.routed_local_reduce()
            tp.start_routed_host()
            tp.finish_shared_local()
            tp.routed_host_reduce(routed)
            up = tp.project(routed, norm_ref, up_vmem, copy, epsilon=EPS,
                            latent_feature_count=HALF)
            tp.finish_shared_host(routed)
            if deferred:
                tp.start_outputs(up, routed)
            else:
                tp.send_outputs(up, routed, rows)
                finish(index)

        if deferred:
            finish(LAYERS - 1)

    vmem, dma = pltpu.VMEM, pltpu.SemaphoreType.DMA
    scratch = [*dmk.communication_scratch(B)[:4], dma((10,)), dma((10,)), dma((16,)),
               dma((16,)), vmem((12, T * 8, 128), jnp.bfloat16), dma((7,)), dma((7,)),
               vmem((B, 4096), F32), vmem((HALF, 1024), jnp.bfloat16), dma(()),
               vmem((T, H), jnp.bfloat16)]

    def body(e, sh, nm, up):
        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((LAYERS, T, H), jnp.bfloat16),
            in_specs=[pl.BlockSpec(memory_space=vmem)] * 3 + [pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=vmem),
            scratch_shapes=scratch,
            compiler_params=pltpu.CompilerParams(has_side_effects=True, collective_id=7),
            interpret=pltpu.InterpretParams(detect_races=True, dma_execution_mode=DMA_MODE),
        )(e[0], sh[0], nm, up[0])
        return out[None]

    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("tp"), P("tp"), P(), P("tp")),
                              out_specs=P("tp"), check_vma=False))
    return np.asarray(f(E, Sh, norm, blocks))


rng = np.random.default_rng(3)
U = (rng.standard_normal((HALF, H)) * HALF ** -0.5).astype(BF16)
U_pad = np.concatenate([U, np.zeros((HALF, 1024), BF16)], axis=1)
norm = np.zeros((1, 4096), np.float32)
norm[0, :HALF] = 1 + 0.1 * rng.standard_normal(HALF)
blocks = np.stack([U_pad[:, 1024 * load.token_parallel_up_block(r):][:, :1024]
                   for r in range(32)])
spread = lambda shape: np.where(rng.random(shape) < 0.25, 2.0 ** rng.integers(-12, 13, shape), 1.0)
for B in BATCHES:
    E = np.zeros((32, LAYERS, B, 4096), np.float32)
    E[..., :HALF] = rng.standard_normal((32, LAYERS, B, HALF)) * 0.05 * spread((32, LAYERS, B, HALF))
    Sh = (rng.standard_normal((32, LAYERS, B, H)) * 0.02 * spread((32, LAYERS, B, H))).astype(np.float32)
    inline = tails(B, E, Sh, norm, blocks, False)
    deferred = tails(B, E, Sh, norm, blocks, True)
    assert not np.isnan(inline.astype(np.float32)).any(), B
    for layer in range(1, LAYERS):  # each layer its own rows
        assert not np.array_equal(inline[:, layer], inline[:, layer - 1]), (B, layer)
    for r in range(32):  # the pair's ranks on every host hold the same rows
        assert inline[r].tobytes() == inline[r % 8 // 2 * 2].tobytes(), (B, r)
    assert deferred.tobytes() == inline.tobytes(), (B, DMA_MODE)
    print(f"B{B} {DMA_MODE}: deferred rounds bit-exact over {LAYERS} layers", flush=True)
assert not ipc.races.races_found
print("OK")
"""

    def _run(self, dma_mode, batches):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), NPROC="128")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                TestAttentionHostBf16.PRELUDE + self.MAIN,
                dma_mode,
                batches,
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,
        )
        output = result.stdout[-2000:] + result.stderr[-3000:]
        self.assertIn("OK", result.stdout, output)
        self.assertNotIn("non-zero count", result.stdout + result.stderr, output)

    def test_deferred_round_matches_the_inline_round(self):
        self._run("on_wait", "8,16")

    def test_deferred_round_with_eager_dmas(self):
        self._run("eager", "8")


class TestAuxCaptureInFlight(CustomTestCase):
    """``KernelConfig.aux_capture_in_flight``: the aux hidden capture of
    rank-row steps (both ranks of a pair hold its rows) over a loop of seven
    layers with captures after layers 1, 2 and 5 (aux ids 6, 2, 3: slots out
    of layer order), each layer's rows different on every host and pair, in
    Pallas interpret mode on 32 host devices (TestAttentionHostBf16's
    harness) with race detection, DMAs executed on wait and eagerly.
    ``capture_aux_rows`` after every layer and ``wait_aux_rows`` after the
    loop (with NaN over the pair's rows after each capture, as the next
    layer's mixture overwrites them, a barrier standing in for its rounds,
    and the aux output starting as NaN) give every rank's aux output bit for
    bit as the gather-and-copy capture (_gather_local_pair_rows with the
    crossed peers, then the batch's rows to the slot) and as the host
    reference (slot s: the four pairs' rows of the rank's host at the slot's
    layer), at 8 rows (the steps the stack takes it at) and 16, and no DMA
    semaphore is left nonzero."""

    MAIN = r"""
import sys

from sglang_tpu.k3_programs.common import collectives32

DMA_MODE, BATCHES = sys.argv[1], tuple(int(b) for b in sys.argv[2].split(","))
LAYERS, AUX = 7, (6, 2, 3)  # captures after layers 5, 1, 2
CAPTURES = sorted(a - 1 for a in AUX)


def captures(B, X, in_flight):
    R = B // 4
    nan = lambda ref: jnp.full(ref.shape, jnp.nan, ref.dtype)

    def kernel(x_ref, aux_in, aux_out, owned, prefix, mixture, sends, recvs, aux_sem, remote):
        del aux_in  # aliased to aux_out (NaN)
        for ref in (owned, prefix, mixture):
            ref[...] = nan(ref)
        stage = mixture.at[pl.ds(0, R)]

        @pl.loop(0, LAYERS)
        def layer(index):
            collectives32.barrier()  # the layer's rounds
            owned[...] = x_ref[index]  # q0: the pair's finished rows
            if in_flight:
                dmk.capture_aux_rows(index, AUX, CAPTURES[0], owned, stage, aux_out,
                                     aux_sem.at[0], remote, owners=4, cross_cores=True)
            else:
                for slot, aux_layer in enumerate(AUX):
                    @pl.when(index == aux_layer - 1)
                    def capture(slot=slot):
                        dmk._gather_local_pair_rows(owned, prefix, sends, recvs, None, owners=4,
                                                    cross_cores=True)
                        copy = pltpu.make_async_copy(prefix, aux_out.at[slot], aux_sem.at[0])
                        copy.start()
                        copy.wait()
            owned[...] = nan(owned)  # the next layer's mixture overwrites the rows

        if in_flight:
            dmk.wait_aux_rows(stage, aux_out, aux_sem.at[0], remote, owners=4, cross_cores=True)

    vmem, dma = pltpu.VMEM, pltpu.SemaphoreType.DMA

    def body(x, aux):
        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((len(AUX), B, H), jnp.bfloat16),
            in_specs=[pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            scratch_shapes=[vmem((R, H), jnp.bfloat16), vmem((B, H), jnp.bfloat16),
                            vmem((B, H), jnp.bfloat16), dma((16,)), dma((16,)), dma((1,)), dma((2,))],
            input_output_aliases={1: 0},
            compiler_params=pltpu.CompilerParams(has_side_effects=True, collective_id=5),
            interpret=pltpu.InterpretParams(detect_races=True, dma_execution_mode=DMA_MODE),
        )(x[0], aux[0])
        return out[None]

    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("tp"), P("tp")), out_specs=P("tp"),
                              check_vma=False))
    aux = np.full((32, len(AUX), B, H), np.nan, BF16)
    return np.asarray(f(X, aux))


rng = np.random.default_rng(5)
for B in BATCHES:
    R = B // 4
    # rows[host, layer, pair]: the pair's finished rows, held by both of its ranks
    rows = (rng.standard_normal((4, LAYERS, 4, R, H)) * 2.0 ** rng.integers(-8, 9, (4, LAYERS, 4, R, 1))).astype(BF16)
    X = np.stack([rows[r // 8, :, r % 8 // 2] for r in range(32)])
    want = np.stack([np.stack([rows[r // 8, a - 1].reshape(B, H) for a in AUX]) for r in range(32)])
    gathered = captures(B, X, False)
    in_flight = captures(B, X, True)
    assert gathered.tobytes() == want.tobytes(), (B, DMA_MODE, "gather")
    assert in_flight.tobytes() == want.tobytes(), (B, DMA_MODE)
    print(f"B{B} {DMA_MODE}: in-flight captures bit-exact", flush=True)
assert not ipc.races.races_found
print("OK")
"""

    def _run(self, dma_mode, batches):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), NPROC="128")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                TestAttentionHostBf16.PRELUDE + self.MAIN,
                dma_mode,
                batches,
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,
        )
        output = result.stdout[-2000:] + result.stderr[-3000:]
        self.assertIn("OK", result.stdout, output)
        self.assertNotIn("non-zero count", result.stdout + result.stderr, output)

    def test_in_flight_captures_match_the_gathered_captures(self):
        self._run("on_wait", "8,16")

    def test_in_flight_captures_with_eager_dmas(self):
        self._run("eager", "8,16")


class TestDsaReference(CustomTestCase):
    """The host reference of the DSA selection, including the distributed
    exact top-k the DCP kernels implement."""

    def test_quantization_and_scores(self):
        from sglang_tpu.k3_programs.kimi_k3 import dsa

        rng = np.random.default_rng(1)
        x = rng.standard_normal((4, 128)).astype(np.float32) * 3
        q8, exp = dsa.quantize_fp8(x)
        self.assertTrue(np.all(np.abs(q8) <= 448))
        np.testing.assert_allclose(q8 * np.exp2(exp), x, rtol=0.07, atol=1e-6)
        q, w = dsa.index_query(rng.standard_normal(32 * 128), rng.standard_normal(32))
        keys, kexp = dsa.quantize_fp8(rng.standard_normal((50, 128)))
        s = dsa.scores(q, w, keys, kexp)
        ref = np.array(
            [
                sum(max(float(keys[n] @ q[h]), 0.0) * w[h] for h in range(32))
                * 2.0 ** kexp[n, 0]
                for n in range(50)
            ]
        )
        np.testing.assert_allclose(s, ref, rtol=1e-4)

    def test_distributed_topk_matches_the_global_selection(self):
        from sglang_tpu.k3_programs.kimi_k3 import dsa

        rng = np.random.default_rng(2)
        cases = [
            ("random", lambda n: rng.standard_normal(n)),
            ("zeros", lambda n: np.where(rng.random(n) < 0.7, 0.0, rng.random(n))),
            ("negative", lambda n: -np.abs(rng.standard_normal(n))),
            ("ties", lambda n: rng.integers(0, 5, n).astype(np.float64)),
        ]
        for name, make in cases:
            for total, topk in ((1500, 2048), (9000, 2048), (40000, 2048), (3000, 64)):
                with self.subTest(case=name, total=total, topk=topk):
                    s = make(total).astype(np.float32)
                    keys = dsa.monotone_key(s)
                    positions = rng.permutation(total)
                    want = dsa.topk_mask(keys, positions, topk)
                    owner = positions // 512 % 32  # DCP32: 512-token chunks round robin
                    parts = [np.nonzero(owner == r)[0] for r in range(32)]
                    masks, rounds = dsa.distributed_topk(
                        [keys[i] for i in parts], [positions[i] for i in parts], topk
                    )
                    got = np.zeros(total, bool)
                    for idx, m in zip(parts, masks):
                        got[idx] = m
                    np.testing.assert_array_equal(got, want)
                    self.assertEqual(int(got.sum()), min(total, topk))
                    self.assertLessEqual(rounds, 7)  # 4 key digits + 3 position digits


class TestIndexPools(CustomTestCase):
    """Kimi-K3.1's DSA index-key pools beside the unified pools: one per MLA
    layer, rows aligned with the latent blocks, and in every list the P/D
    transfer and the host tier read."""

    def test_pools_transfer_and_layer_ids(self):
        from types import SimpleNamespace
        from unittest import mock

        import torch

        from sglang_tpu.hooks import k3_program as hooks
        from sglang_tpu.k3_programs import sglang_pools
        from sglang_tpu.k3_programs.kimi_k3.layout import KDA, MLA, NUM_LAYERS

        model = mock.patch(
            "sglang.srt.runtime_context.get_model",
            return_value=SimpleNamespace(kv_cache_dtype="fp8_e4m3"),
        )
        with model:
            state = sglang_pools.K3StatePool(size=4, mamba_layer_ids=KDA, device="cpu")
            kv = sglang_pools.K3UnifiedKVPool(
                size=6 * 16384,
                page_size=16384,
                dtype=torch.float8_e4m3fn,
                layer_num=len(MLA),
                device="cpu",
                index_head_dim=128,
            )
        self.assertEqual(len(kv.index_pools), len(MLA))
        self.assertEqual(
            tuple(kv.index_pools[0].shape), (kv.plan.num_blocks, 4 * 33, 4, 128)
        )
        self.assertEqual(kv.index_pools[0].dtype, torch.uint8)
        self.assertEqual(len(kv.raw_pools()), len(MLA) + 5 + len(MLA))
        ptrs, lens, items = kv.get_contiguous_buf_infos()
        ids = kv.get_kv_layer_ids()
        self.assertEqual(len(ptrs), len(ids))
        self.assertEqual(set(items[-len(MLA) :]), {4 * 33 * 4 * 128})
        self.assertEqual(
            ids[-len(MLA) :], list(range(NUM_LAYERS + 5, NUM_LAYERS + 5 + len(MLA)))
        )
        self.assertEqual(len(set(ids)), len(ids))
        # Kimi-K3 builds none.
        with model:
            sglang_pools.K3StatePool(size=4, mamba_layer_ids=KDA, device="cpu")
            k3 = sglang_pools.K3UnifiedKVPool(
                size=6 * 16384,
                page_size=16384,
                dtype=torch.float8_e4m3fn,
                layer_num=len(MLA),
                device="cpu",
            )
        self.assertEqual(k3.index_pools, [])
        self.assertEqual(len(k3.get_kv_layer_ids()), len(MLA) + 5)
        del hooks, state


class TestIndexCommit(CustomTestCase):
    """New tokens' index keys land on their DCP owner, in the block their
    latents go to, and read back as the reference quantization."""

    def test_records_round_trip_on_their_owners(self):
        from unittest import mock

        import jax.numpy as jnp

        from sglang_tpu.k3_programs.kimi_k3 import dsa
        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp, mla_dcp

        rng = np.random.default_rng(3)
        tokens, group, blocks, layers = 512, 32, 6, 2
        positions = np.array([5, 511, 512 * 33 + 130, 512 * 40 + 511], np.int32)
        keys = rng.standard_normal((len(positions), layers, 128)).astype(np.float32)
        records = dsa_dcp.key_records(jnp.asarray(keys))
        # Local tables: entry e covers global pages e*32 .. e*32+31; block ids 1.. per row.
        tables = np.zeros((len(positions), 4), np.int32)
        for b in range(len(positions)):
            tables[b] = np.arange(1, 5) + 4 * b
        for member in range(group):
            pools = tuple(
                jnp.zeros((blocks * 4, 4 * 33, 4, 128), jnp.uint8)
                for _ in range(layers)
            )
            with mock.patch.object(mla_dcp, "owner_index", return_value=member):
                pools = dsa_dcp.commit_index_rows(
                    pools,
                    records,
                    jnp.asarray(tables),
                    jnp.asarray(positions),
                    group=group,
                )
            for b, p in enumerate(positions):
                page = p // tokens
                block = tables[b, page // group]
                for m in range(layers):
                    got, gexp = dsa_dcp.read_records(pools[m][block], tokens)
                    t = p % tokens
                    if page % group == member:
                        want, wexp = dsa.quantize_fp8(keys[b, m][None])
                        np.testing.assert_array_equal(np.asarray(got[t]), want[0])
                        self.assertEqual(int(gexp[t, 0]), int(wexp[0, 0]))
                    else:
                        self.assertFalse(np.asarray(pools[m][block]).any())


class TestFusedDsaCommit(CustomTestCase):
    """The fused K3.1 epilogue commit (``paged._commit_dsa_rows_fused``, Pallas
    interpret mode) writes the bytes of the latent commit kernel and the XLA
    index scatters: runs inside a sub-page and across sub-pages and pages,
    other ranks' rows, two sequences, full waves, out-of-range and negative
    blocks; repeated null-block targets land in row order."""

    def _commit(self, positions, tables, members, layers=3, blocks=6, cache="fp8"):
        import functools
        from unittest import mock

        import jax
        import jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp, mla_dcp, paged
        from sglang_tpu.k3_programs.kimi_k3.layout import (
            cache_layout,
            index_block_shape,
        )

        layout = cache_layout(cache)
        rng = np.random.default_rng(len(positions) + int(np.sum(positions)) % 997)
        batch, packing = len(positions), layout.block_shape[1]
        page_tokens = layout.tokens_per_block
        dtype, bits = (
            (jnp.bfloat16, np.uint16)
            if layout.dtype == "bfloat16"
            else (jnp.float8_e4m3fn, np.uint8)
        )
        pool_map = tuple(int(i) for i in rng.permutation(layers + 1)[:layers])
        # Random bytes; finite BF16 values (interpret-mode BF16 selects
        # canonicalize NaN payloads, which the TPU's do not).
        lat = [
            rng.integers(0, 256, (blocks, *layout.block_shape), dtype=np.uint8)
            if bits is np.uint8
            else np.asarray(
                jnp.asarray(
                    rng.standard_normal((blocks, *layout.block_shape)), jnp.bfloat16
                )
            ).view(np.uint16)
            for _ in range(layers + 1)
        ]
        idx = [
            rng.integers(0, 256, (blocks, *index_block_shape(layout)), dtype=np.uint8)
            for _ in range(layers)
        ]
        latents = jnp.asarray(rng.standard_normal((batch, layers, 640)), jnp.bfloat16)
        records = rng.integers(0, 256, (batch, layers, 2, 128)).astype(np.int32)
        cast = lambda x: jax.lax.bitcast_convert_type(jnp.asarray(x), dtype)
        raw = lambda x: np.asarray(
            jax.lax.bitcast_convert_type(x, bits) if x.dtype == dtype else x
        )
        row_bytes = raw(latents.astype(dtype))
        tables_, positions_ = jnp.asarray(tables, jnp.int32), jnp.asarray(positions)
        interpret = functools.partial(
            pl.pallas_call, interpret=pltpu.InterpretParams()
        )
        for member in members:
            want_lat = [x.copy() for x in lat]
            want_idx = [x.copy() for x in idx]
            for b, p in enumerate(positions):  # row order
                page, t = p // page_tokens, p % page_tokens
                block = tables[b][page // 32]
                block = block + blocks if block < 0 else block
                if page % 32 != member or not 0 <= block < blocks:
                    continue
                (key_row, sub), (scale_row, lane) = dsa_dcp.record_place(t)
                for m in range(layers):
                    want_lat[pool_map[m]][block, t // packing, t % packing] = (
                        row_bytes[b, m]
                    )
                    want_idx[m][block, key_row, sub] = records[b, m, 0]
                    want_idx[m][block, scale_row, 0, lane] = records[b, m, 1, 0]
            with (
                mock.patch.object(mla_dcp, "owner_index", return_value=member),
                mock.patch.object(paged.pl, "pallas_call", interpret),
            ):
                pools = tuple(cast(x) for x in lat)
                index_pools = tuple(jnp.asarray(x) for x in idx)
                fused = paged._commit_dsa_rows_fused(
                    pools,
                    index_pools,
                    latents,
                    jnp.asarray(records),
                    tables_,
                    positions_,
                    pool_map,
                )
                old = paged.commit_latent_rows(
                    pools, latents, tables_, positions_, pool_map
                ) + dsa_dcp.commit_index_rows(
                    index_pools,
                    jnp.concatenate(
                        (records[:, :, 0], records[:, :, 1, :1]), axis=-1
                    ).astype(jnp.uint8),
                    tables_,
                    positions_,
                    group=32,
                )
            for got, base, want in zip(fused, old, want_lat + want_idx, strict=True):
                np.testing.assert_array_equal(raw(got), want)
                np.testing.assert_array_equal(raw(got), raw(base))

    @staticmethod
    def _interpretable():
        try:
            from jax.experimental.pallas import tpu as pltpu

            pltpu.InterpretParams()
        except (ImportError, AttributeError):
            return False
        return True

    def test_verify_runs(self):
        """One sequence's eight verify rows: inside a sub-page, across a
        sub-page, across a page (the next page's owner commits the rest)."""
        if not self._interpretable():
            self.skipTest("Pallas TPU interpret mode unavailable")
        tables = np.tile(np.array([[3, 1, 4, 2]], np.int32), (8, 1))
        for start in (0, 3, 29, 116, 125, 508):
            with self.subTest(start=start):
                base = 512 * 37 + start  # page 37: member 5's local entry 1
                positions = base + np.arange(8, dtype=np.int32)
                members = (5, 6, 7) if start == 508 else (5,)
                self._commit(positions, tables, members=members)

    def test_two_sequences_and_full_waves(self):
        """B2x7 (two sequences at one context: one owner, two blocks), then
        sixteen rows on sixteen blocks of one owner (two full waves)."""
        if not self._interpretable():
            self.skipTest("Pallas TPU interpret mode unavailable")
        positions = np.tile(512 * 70 + 250 + np.arange(8, dtype=np.int32), 2)
        tables = np.repeat(np.array([[0, 0, 2, 0], [0, 0, 5, 0]], np.int32), 8, axis=0)
        self._commit(positions, tables, members=(6, 7))
        positions = (512 * 32 * np.arange(16) + 512 * 3 + 7).astype(np.int32)
        tables = np.zeros((16, 16), np.int32)
        tables[np.arange(16), np.arange(16)] = np.arange(16)
        self._commit(positions, tables, members=(3,), blocks=17)

    def test_plain_rows_padding_and_bounds(self):
        """Plain decode rows: rows sharing blocks out of order, padding rows
        on null block 0, an out-of-range block, a negative block next to its
        normalized twin; then the BF16 layout's runs."""
        if not self._interpretable():
            self.skipTest("Pallas TPU interpret mode unavailable")
        positions = np.array(
            [512 * 33 + 5, 0, 512 * 33 + 9, 0, 512 * 65 + 130, 512 + 511]
            + [512 * 65 + 131, 512 * 33 + 6],
            np.int32,
        )
        tables = np.zeros((8, 4), np.int32)
        tables[[0, 2, 7], 1] = 4
        tables[[4, 6], 2] = -1, 5
        tables[5, 0] = 9
        self._commit(positions, tables, members=(0, 1))
        tables = np.tile(np.array([[3, 1, 4, 2]], np.int32), (8, 1))
        for start in (2, 124, 252):
            with self.subTest(cache="bf16", start=start):
                positions = 256 * 37 + start + np.arange(8, dtype=np.int32)
                self._commit(positions, tables, members=(5,), cache="bfloat16")

    def test_mla_layers_and_dispatch(self):
        """All 24 MLA layers; the decode step takes the fused commit only
        with ``dsa_fused_commit``."""
        if not self._interpretable():
            self.skipTest("Pallas TPU interpret mode unavailable")
        from unittest import mock

        import jax
        import jax.numpy as jnp

        from sglang_tpu.k3_programs.kimi_k3.config import KernelConfig
        from sglang_tpu.k3_programs.kimi_k3.kernels import paged

        positions = 512 * 40 + 61 + np.arange(8, dtype=np.int32)
        tables = np.tile(np.array([[0, 2, 0]], np.int32), (8, 1))
        self._commit(positions, tables, members=(8,), layers=24, blocks=3)
        pools = (jax.ShapeDtypeStruct((4, 128, 4, 640), jnp.float8_e4m3fn),) * 24
        index_pools = (jax.ShapeDtypeStruct((4, 132, 4, 128), jnp.uint8),) * 24
        rows = jax.ShapeDtypeStruct((8, 24, 640), jnp.bfloat16)
        records = jax.ShapeDtypeStruct((8, 24, 2, 128), jnp.int32)
        tables = jax.ShapeDtypeStruct((8, 64), jnp.int32)
        for enabled in (False, True):
            with mock.patch.object(
                paged, "_config", return_value=KernelConfig(dsa_fused_commit=enabled)
            ):
                self.assertEqual(
                    paged._dsa_commit_fusable(
                        pools, index_pools, rows, records, tables, tuple(range(24))
                    ),
                    enabled,
                )


class TestVerifyFusedTail(CustomTestCase):
    """``paged.commit_verify_tail`` (KernelConfig.verify_fused_tail, Pallas
    interpret mode): from the decoder's BF16 new-key handoff it writes the
    pools, index pools and KDA banks of the knob-off commits -- the XLA
    latent rows and FP8 convert into ``_commit_dsa_rows_fused``, and the
    selected post-states copied in row order (``_commit_kda_states``) --
    for a verify's runs on their owners and on other ranks, two sequences
    and repeated null-slot targets."""

    def _tail(self, positions, tables, members, targets, sources, heads=3, layers=3):
        import functools
        from unittest import mock

        import jax
        import jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        from sglang_tpu.k3_programs.kimi_k3.kernels import mla_dcp, paged
        from sglang_tpu.k3_programs.kimi_k3.layout import (
            cache_layout,
            index_block_shape,
        )

        layout = cache_layout("fp8")
        blocks, slots = 6, 5
        batch = len(positions)
        rng = np.random.default_rng(batch + int(np.sum(positions)) % 991)
        pool_map = tuple(int(i) for i in rng.permutation(layers + 1)[:layers])
        lat = [
            rng.integers(0, 256, (blocks, *layout.block_shape), dtype=np.uint8)
            for _ in range(layers + 1)
        ]
        idx = [
            rng.integers(0, 256, (blocks, *index_block_shape(layout)), dtype=np.uint8)
            for _ in range(layers)
        ]
        # The handoff's three latent rows (finite, past FP8's range too) and
        # stale padding rows.
        handoff = jnp.asarray(
            rng.standard_normal((batch, layers, heads, 256)) * 100, jnp.bfloat16
        )
        records = jnp.asarray(rng.integers(0, 256, (batch, layers, 2, 128)), jnp.int32)
        conv_banks = jnp.asarray(
            rng.standard_normal((slots, 3, 2, 4, 128)), jnp.bfloat16
        )
        state_banks = jnp.asarray(rng.standard_normal((slots, 3, 2, 128)), jnp.float32)
        new_conv = jnp.asarray(rng.standard_normal((batch, 3, 2, 4, 128)), jnp.bfloat16)
        new_state = jnp.asarray(rng.standard_normal((batch, 3, 2, 128)), jnp.float32)
        targets_ = jnp.asarray(targets, jnp.int32)
        sources_ = jnp.asarray(sources, jnp.int32)
        f8 = jnp.float8_e4m3fn
        cast = lambda x: jax.lax.bitcast_convert_type(jnp.asarray(x), f8)
        raw = lambda x: np.asarray(
            jax.lax.bitcast_convert_type(x, np.uint8) if x.dtype == f8 else x
        )
        bits = lambda x: np.asarray(x).view(
            {2: np.uint16, 4: np.uint32}[np.asarray(x).dtype.itemsize]
        )
        tables_, positions_ = jnp.asarray(tables, jnp.int32), jnp.asarray(positions)
        interpret = functools.partial(pl.pallas_call, interpret=pltpu.InterpretParams())
        want_conv, want_state = np.array(conv_banks), np.array(state_banks)
        for t, src in zip(targets, sources):  # row order
            want_conv[t], want_state[t] = (
                np.asarray(new_conv[src]),
                np.asarray(new_state[src]),
            )
        for member in members:
            with (
                mock.patch.object(mla_dcp, "owner_index", return_value=member),
                mock.patch.object(paged.pl, "pallas_call", interpret),
            ):
                pools = tuple(cast(x) for x in lat)
                index_pools = tuple(jnp.asarray(x) for x in idx)
                conv, state, *got = paged.commit_verify_tail(
                    conv_banks,
                    state_banks,
                    new_conv,
                    new_state,
                    targets_,
                    sources_,
                    pools,
                    index_pools,
                    handoff,
                    records,
                    tables_,
                    positions_,
                    pool_map,
                )
                old = paged._commit_dsa_rows_fused(
                    pools,
                    index_pools,
                    paged.latent_rows_from_handoff(handoff),
                    records,
                    tables_,
                    positions_,
                    pool_map,
                )
            self.assertEqual(len(got), len(old))
            changed = 0
            for g, o, base in zip(got, old, (*lat, *idx), strict=True):
                np.testing.assert_array_equal(raw(g), raw(o))
                changed += int(not np.array_equal(raw(o), base))
            np.testing.assert_array_equal(bits(conv), bits(want_conv))
            np.testing.assert_array_equal(bits(state), bits(want_state))
            yield member, changed

    @staticmethod
    def _interpretable():
        try:
            from jax.experimental.pallas import tpu as pltpu

            pltpu.InterpretParams()
        except (ImportError, AttributeError):
            return False
        return True

    def test_verify_runs_and_other_ranks(self):
        """One sequence's eight rows inside a sub-page and across a page (the
        next page's owner commits the rest); a rank owning none of them still
        commits the KDA row; repeated null-slot targets land in row order."""
        if not self._interpretable():
            self.skipTest("Pallas TPU interpret mode unavailable")
        tables = np.tile(np.array([[3, 1, 4, 2]], np.int32), (8, 1))
        for start, members, layers in (
            (29, (5,), 24),
            (29, (5, 9), 3),
            (508, (5, 6), 3),
        ):
            with self.subTest(start=start, layers=layers):
                positions = 512 * 37 + start + np.arange(8, dtype=np.int32)
                changed = dict(
                    self._tail(positions, tables, members, [3], [6], layers=layers)
                )
                self.assertGreater(changed[5], 0)  # the owner commits rows
                if 9 in changed:
                    self.assertEqual(changed[9], 0)  # another rank: none
        positions = np.tile(512 * 70 + 250 + np.arange(8, dtype=np.int32), 2)
        tables = np.repeat(np.array([[0, 0, 2, 0], [0, 0, 5, 0]], np.int32), 8, axis=0)
        list(self._tail(positions, tables, (6,), targets=[0, 0], sources=[7, 12]))
        list(
            self._tail(positions, tables, (6,), targets=[2, 4], sources=[3, 9], heads=8)
        )


class TestVerifyFusedTailOps(CustomTestCase):
    """The verify ops' plumbing for KernelConfig.verify_fused_tail (Pallas
    interpret mode, three MLA layers): ``decode_paged``'s deferred latent
    commit goes to ``DecodeOps._finish_verify``, which commits it with the
    accepted KDA row into the op's pools (window centres untouched) -- the
    bytes of the knob-off commits."""

    def test_finish_verify_commits_the_pending_rows(self):
        import functools
        from unittest import mock

        import jax
        import jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        from sglang_tpu.k3_programs.kimi_k3 import config as kernel_config
        from sglang_tpu.k3_programs.kimi_k3 import decode_ops, sampling
        from sglang_tpu.k3_programs.kimi_k3.kernels import mla_dcp, paged
        from sglang_tpu.k3_programs.kimi_k3.layout import (
            cache_layout,
            index_block_shape,
        )

        layout, layers, blocks, rows = cache_layout("fp8"), 3, 4, 8
        rng = np.random.default_rng(7)
        pool_map = (2, 0, 3)
        f8 = jnp.float8_e4m3fn
        lat = tuple(
            jax.lax.bitcast_convert_type(
                jnp.asarray(
                    rng.integers(0, 256, (blocks, *layout.block_shape)), jnp.uint8
                ),
                f8,
            )
            for _ in range(layers + 1)
        )
        idx = tuple(
            jnp.asarray(
                rng.integers(0, 256, (blocks, *index_block_shape(layout))), jnp.uint8
            )
            for _ in range(layers)
        )
        centres = jnp.asarray(rng.integers(0, 99, (1, 5, layers, rows)), jnp.int32)
        handoff = jnp.asarray(
            rng.standard_normal((rows, layers, 3, 256)) * 9, jnp.bfloat16
        )
        records = jnp.asarray(rng.integers(0, 256, (rows, layers, 2, 128)), jnp.int32)
        positions = jnp.asarray(512 * 41 + 100 + np.arange(rows), jnp.int32)
        tables = jnp.tile(jnp.asarray([[0, 3, 0]], jnp.int32), (rows, 1))
        slot = 2
        slots = jnp.full((rows,), slot, jnp.int32)
        conv = jnp.asarray(rng.standard_normal((4, 3, 2, 4, 128)), jnp.bfloat16)
        state = jnp.asarray(rng.standard_normal((4, 3, 2, 128)), jnp.float32)
        new_conv = jnp.asarray(rng.standard_normal((rows, 3, 2, 4, 128)), jnp.bfloat16)
        new_state = jnp.asarray(rng.standard_normal((rows, 3, 2, 128)), jnp.float32)
        decisions = jnp.asarray([[5, 77]], jnp.int32)  # 5 accepted: post-state row 5

        ops = object.__new__(decode_ops.DecodeOps)
        ops.paged, ops.attention_dp, ops.dsa_pools = paged, 1, layers
        ops.unified_pools = [None] * (2 * layers + 1)
        ops.pool_map, ops.owner_table = pool_map, None
        for enabled in (False, True):
            with mock.patch.object(
                kernel_config,
                "active",
                return_value=kernel_config.KernelConfig(verify_fused_tail=enabled),
            ):
                self.assertEqual(
                    ops._fused_tail_options(),
                    {"defer_latent_commit": True} if enabled else {},
                )
        outputs = ("logits", "aux", "conv", "state", ("pools",), "pending")
        self.assertEqual(
            ops._pending_commit(outputs, {"defer_latent_commit": True}),
            (outputs[:-1], "pending"),
        )
        self.assertEqual(ops._pending_commit(outputs[:-1], {}), (outputs[:-1], None))

        interpret = functools.partial(pl.pallas_call, interpret=pltpu.InterpretParams())
        control = mock.Mock()
        raw = lambda x: np.asarray(
            jax.lax.bitcast_convert_type(x, np.uint8) if x.dtype == f8 else x
        )
        with (
            mock.patch.object(mla_dcp, "owner_index", return_value=41 % 32),
            mock.patch.object(paged.pl, "pallas_call", interpret),
            mock.patch.object(sampling, "verify_acceptance", return_value=decisions),
        ):
            got = ops._finish_verify(
                control,
                conv,
                state,
                new_conv,
                new_state,
                None,
                slots,
                positions,
                slots,
                None,
                jnp.zeros((1,), jnp.int32),
                None,
                True,
                pool=(*lat, *idx, centres),
                pending=(handoff, records, tables, positions),
            )
            want_pools = paged._commit_dsa_rows_fused(
                lat,
                idx,
                paged.latent_rows_from_handoff(handoff),
                records,
                tables,
                positions,
                pool_map,
            )
        out_decisions, out_conv, out_state, out_pool = got
        np.testing.assert_array_equal(np.asarray(out_decisions), np.asarray(decisions))
        self.assertEqual(len(out_pool), 2 * layers + 2)
        for g, w in zip(out_pool[:-1], want_pools, strict=True):
            np.testing.assert_array_equal(raw(g), raw(w))
        self.assertTrue(
            any(
                not np.array_equal(raw(g), raw(b))
                for g, b in zip(out_pool, (*lat, *idx))
            )
        )
        np.testing.assert_array_equal(np.asarray(out_pool[-1]), np.asarray(centres))
        want_conv, want_state = np.array(conv), np.array(state)
        want_conv[slot], want_state[slot] = (
            np.asarray(new_conv[5]),
            np.asarray(new_state[5]),
        )
        np.testing.assert_array_equal(
            np.asarray(out_conv).view(np.uint16), want_conv.view(np.uint16)
        )
        np.testing.assert_array_equal(
            np.asarray(out_state).view(np.uint32), want_state.view(np.uint32)
        )


class TestIndexSelect(CustomTestCase):
    """The decode selection kernel (``dsa_dcp.select_call``) in Pallas
    interpret mode on a 4-rank DCP group of host devices: every rank's keys
    match the reference scores, the thresholds select exactly the top-k
    (lowest positions among tied keys) and agree on every rank, in-flight
    tokens are scored from their records (the pool holds garbage there), and
    the own-token flags follow the selection."""

    SCRIPT = """
        import os, sys
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import ml_dtypes, numpy as np, jax, jax.numpy as jnp
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3 import dsa
        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp

        G, T = 4, 512
        def case(firsts, S, topk, ties, seed):
            rng = np.random.default_rng(seed)
            B = len(firsts) * S
            positions = np.array([p0 + r for p0 in firsts for r in range(S)], np.int32)
            pages = max(-(-((p + T) // T) // G) for p in positions) + 1
            pools = rng.integers(0, 256, (G, 1 + len(firsts) * pages, 132, 4, 128), dtype=np.uint8)
            tables = np.zeros((G, B, pages), np.int32)
            new_k8 = np.zeros((B, 128), np.float32); new_exp = np.zeros(B, np.int32)
            seq_keys = []
            for s, p0 in enumerate(firsts):
                L = p0 + S
                raw = rng.standard_normal((L, 128)).astype(np.float32)
                if ties:  # five distinct keys: ties at the threshold
                    raw = rng.standard_normal((5, 128)).astype(np.float32)[rng.integers(0, 5, L)]
                k8, kexp = dsa.quantize_fp8(raw)
                seq_keys.append((k8, kexp))
                rec = np.asarray(dsa_dcp.key_records(jnp.asarray(raw)))
                for r in range(S):
                    tables[:, s * S + r] = 1 + s * pages + np.arange(pages)
                    new_k8[s * S + r] = k8[p0 + r]; new_exp[s * S + r] = rec[p0 + r, 128]
                for t in range(p0):
                    page = t // T
                    blk = tables[page % G, s * S, page // G]
                    (row, sub), (srow, lane) = dsa_dcp.record_place(t % T)
                    pools[page % G, blk, row, sub] = rec[t, :128]
                    pools[page % G, blk, srow, 0, lane] = rec[t, 128]
            q8 = np.zeros((B, 32, 128), np.float32); w = np.zeros((B, 32), np.float32)
            for b in range(B):
                q8[b], w[b] = dsa.index_query(rng.standard_normal(4096).astype(np.float32),
                                              rng.standard_normal(32).astype(np.float32))
            w_t = np.zeros((32, 128), np.float32); w_t[:, :B] = w.T
            mesh = Mesh(np.asarray(jax.devices()), ("tp",))
            def body(q8, w_t, pool, tables, positions, new_k8, new_exp):
                out = dsa_dcp.select_call(q8, w_t, new_k8, pool[0], tables[0], positions, new_exp,
                                          pages=pages, topk=topk, group=G, max_local_pages=pages,
                                          sequence_rows=S, interpret=pltpu.InterpretParams())
                return tuple(x[None] for x in out)
            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P(), P("tp"), P("tp"), P(), P(), P()),
                                      out_specs=(P("tp"),) * 4, check_vma=False))
            tau, ptie, own, keys = (np.asarray(x) for x in f(
                q8.astype(ml_dtypes.float8_e4m3fn), w_t, pools, tables.reshape(G, -1), positions, new_k8, new_exp))
            tied = 0
            for b in range(B):
                L = positions[b] + 1
                k8, kexp = seq_keys[b // S]
                t = np.arange(L)
                member = (t // T) % G
                kk = keys[member, b, (t // T) // G * (T // 128) + (t % T) // 128, t % 128].astype(np.int64)
                ref = dsa.monotone_key(dsa.scores(q8[b], w[b], k8[:L], kexp[:L]).astype(np.float32))
                unkey = lambda k: (k ^ ((k >> 31) & 0x7FFFFFFF)).astype(np.int32).view(np.float32)
                err = np.abs(unkey(kk) - unkey(ref)).max() / np.abs(unkey(ref)).max()
                assert err < 1e-5, (b, err)
                for name, v in (("tau", tau), ("p_tie", ptie), ("own", own)):
                    assert len(set(v[:, b])) == 1, (name, b, v[:, b])
                got = dsa_dcp.selected(kk, t, tau[0, b], ptie[0, b])
                np.testing.assert_array_equal(got, dsa.topk_mask(kk, t, topk))
                assert bool(own[0, b]) == bool(got[-1]), b
                tied += int(L > topk and np.sum(kk == tau[0, b]) > 1)
            return tied

        case([300, 2500], 1, 256, False, 0)
        assert case([1500, 2600], 2, 300, True, 1) > 0  # position tie-breaks exercised
        print("OK")
    """

    def test_selection_matches_the_reference_on_a_dcp_group(self):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])


class TestIndexSelectBins(CustomTestCase):
    """Bin selection (``KernelConfig.dsa_select_rounds`` < 8): the kernel's
    threshold is the lower bound of the ``4 * rounds``-bit key prefix bin that
    holds the ``topk``-th key (every key at or above it selected), on every
    rank alike, whatever digit widths its exchange rounds use
    (``dsa_dcp.digit_widths``)."""

    SCRIPT = """
        import os, sys
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        os.environ["JAX_PLATFORMS"] = "cpu"
        ROUNDS = int(os.environ["ROUNDS"])
        from sglang_tpu.k3_programs.kimi_k3 import config as kc
        kc.install(kc.KernelConfig(dsa_select_rounds=ROUNDS))
        import ml_dtypes, numpy as np, jax, jax.numpy as jnp
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3 import dsa
        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp

        G, T, INT_MIN = 4, 512, -(2**31)
        def case(firsts, S, topk, seed):
            rng = np.random.default_rng(seed)
            B = len(firsts) * S
            print("widths", dsa_dcp.digit_widths(B, ROUNDS))
            positions = np.array([p0 + r for p0 in firsts for r in range(S)], np.int32)
            pages = max(-(-((p + T) // T) // G) for p in positions) + 1
            pools = rng.integers(0, 256, (G, 1 + len(firsts) * pages, 132, 4, 128), dtype=np.uint8)
            tables = np.zeros((G, B, pages), np.int32)
            new_k8 = np.zeros((B, 128), np.float32); new_exp = np.zeros(B, np.int32)
            for s, p0 in enumerate(firsts):
                L = p0 + S
                raw = rng.standard_normal((L, 128)).astype(np.float32)
                k8, kexp = dsa.quantize_fp8(raw)
                rec = np.asarray(dsa_dcp.key_records(jnp.asarray(raw)))
                for r in range(S):
                    tables[:, s * S + r] = 1 + s * pages + np.arange(pages)
                    new_k8[s * S + r] = k8[p0 + r]; new_exp[s * S + r] = rec[p0 + r, 128]
                for t in range(p0):
                    page = t // T
                    blk = tables[page % G, s * S, page // G]
                    (row, sub), (srow, lane) = dsa_dcp.record_place(t % T)
                    pools[page % G, blk, row, sub] = rec[t, :128]
                    pools[page % G, blk, srow, 0, lane] = rec[t, 128]
            q8 = np.zeros((B, 32, 128), np.float32); w = np.zeros((B, 32), np.float32)
            for b in range(B):
                q8[b], w[b] = dsa.index_query(rng.standard_normal(4096).astype(np.float32),
                                              rng.standard_normal(32).astype(np.float32))
            w_t = np.zeros((32, 128), np.float32); w_t[:, :B] = w.T
            mesh = Mesh(np.asarray(jax.devices()), ("tp",))
            def body(q8, w_t, pool, tables, positions, new_k8, new_exp):
                out = dsa_dcp.select_call(q8, w_t, new_k8, pool[0], tables[0], positions, new_exp,
                                          pages=pages, topk=topk, group=G, max_local_pages=pages,
                                          sequence_rows=S, interpret=pltpu.InterpretParams())
                return tuple(x[None] for x in out)
            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P(), P("tp"), P("tp"), P(), P(), P()),
                                      out_specs=(P("tp"),) * 4, check_vma=False))
            tau, ptie, own, keys = (np.asarray(x) for x in f(
                q8.astype(ml_dtypes.float8_e4m3fn), w_t, pools, tables.reshape(G, -1), positions, new_k8, new_exp))
            extra = []
            for b in range(B):
                L = positions[b] + 1
                t = np.arange(L)
                kk = keys[(t // T) % G, b, (t // T) // G * (T // 128) + (t % T) // 128, t % 128].astype(np.int64)
                for name, v in (("tau", tau), ("p_tie", ptie), ("own", own)):
                    assert len(set(v[:, b])) == 1, (name, b, v[:, b])
                if L <= topk:
                    want_tau = INT_MIN
                    want = kk > INT_MIN
                else:
                    kth = np.sort(kk)[::-1][topk - 1]
                    drop = 32 - 4 * ROUNDS
                    unsigned = (int(kth) ^ INT_MIN) & 0xFFFFFFFF
                    want_tau = ((unsigned >> drop) << drop) ^ (1 << 31)
                    want_tau = want_tau - (1 << 32) if want_tau >= (1 << 31) else want_tau
                    want = kk >= want_tau
                    extra.append(int(want.sum()) - topk)
                assert tau[0, b] == want_tau, (b, tau[0, b], want_tau)
                got = dsa_dcp.selected(kk, t, tau[0, b], ptie[0, b])
                np.testing.assert_array_equal(got, want)
                assert np.all(got[dsa.topk_mask(kk, t, topk)]), b  # a superset of the top-k
                assert bool(own[0, b]) == bool(got[-1]), b
            print("superset extra tokens", extra)
            return tau, ptie, own, keys

        case([300, 2500], 1, 256, 0)
        case([1500, 2600], 2, 300, 1)
        case([3100], 8, 400, 2)
        print("OK")
    """

    def _run(self, rounds):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), ROUNDS=str(rounds))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=1200,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )

    def test_five_rounds(self):
        self._run(5)

    def test_four_rounds(self):
        self._run(4)


class TestIndexSelectGatherFinish(CustomTestCase):
    """``KernelConfig.dsa_gather_finish``: once every row's threshold bin is
    small the ranks gather its keys and finish the rounds locally -- the same
    thresholds as the distributed rounds, with and without position ties."""

    @staticmethod
    def script(knobs=""):
        import textwrap

        script = textwrap.dedent(TestIndexSelect.SCRIPT)
        script = script.replace(
            "from sglang_tpu.k3_programs.kimi_k3 import dsa\n",
            "from sglang_tpu.k3_programs.kimi_k3 import config as kc\n"
            f"kc.install(kc.KernelConfig(dsa_gather_finish=True{knobs}))\n"
            "from sglang_tpu.k3_programs.kimi_k3 import dsa\n",
        )
        script = script.replace(
            "from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp\n",
            "from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp\n"
            "from jax.experimental import pallas as pl\n"
            "_finish = dsa_dcp.IndexSelect._gather_finish\n"
            "def _traced(self, first_round):\n"
            "    pl.debug_print('GATHER at round {}', first_round)\n"
            "    return _finish(self, first_round)\n"
            "dsa_dcp.IndexSelect._gather_finish = _traced\n",
        )
        script = script.replace(
            "def case(firsts, S, topk, ties, seed):",
            "def case(firsts, S, topk, ties, seed, distinct=5):",
        )
        script = script.replace(
            "rng.standard_normal((5, 128)).astype(np.float32)[rng.integers(0, 5, L)]",
            "rng.standard_normal((distinct, 128)).astype(np.float32)[rng.integers(0, distinct, L)]",
        )
        tail = script[script.index("case([300, 2500], 1, 256, False, 0)") :]
        return script.replace(
            tail,
            "\n".join(
                [
                    "case([300, 2500], 1, 256, False, 0)",
                    "assert case([2600, 3100], 2, 256, True, 2, distinct=350) > 0  # ties inside the gathered bin",
                    "case([3500], 4, 512, False, 3)",
                    'print("OK")',
                    "",
                ]
            ),
        )

    def test_gather_finish_matches_the_reference(self):
        self._run()

    def test_gather_finish_with_odd_stride(self):
        # KernelConfig.dsa_odd_stride: the resident scan reads the padded rows.
        self._run(", dsa_odd_stride=True")

    def _run(self, knobs=""):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", self.script(knobs)],
            capture_output=True,
            text=True,
            env=env,
            timeout=1500,
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])
        self.assertGreaterEqual(
            result.stdout.count("GATHER at round"), 3 * 4, result.stdout[-2000:]
        )


class TestIndexSelectScoringKnobs(CustomTestCase):
    """The scoring loop's KernelConfig knobs (``dsa_stage_slots``,
    ``dsa_page_slots``, ``dsa_resident_keys``, ``dsa_patch_loop``) leave the
    selection bit for bit unchanged: ``select_call`` in interpret mode on a
    4-rank group, once with the defaults and once per knob set, must give the
    same thresholds, own-token flags and keys of every row's tokens (each run
    also checks the reference). The cases cover in-flight tokens across a
    page (and rank) boundary, eight one-row sequences, two two-row sequences,
    keys over both resident chunks (the second one clamped and not), and
    exact rounds with position ties."""

    SCRIPT = """
        import os, sys
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        os.environ["JAX_PLATFORMS"] = "cpu"
        from sglang_tpu.k3_programs.kimi_k3 import config as kc
        knobs = dict(kv.split("=") for kv in os.environ["KNOBS"].split(",") if kv)
        kc.install(kc.KernelConfig().replace(**knobs))
        import ml_dtypes, numpy as np, jax, jax.numpy as jnp
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3 import dsa
        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp

        G, T, INT_MIN = 4, 512, -(2**31)
        ROUNDS = kc.active().dsa_select_rounds
        outputs = {}

        def case(name, firsts, S, topk, seed, chunk=None, ties=False):
            if chunk:  # narrower count chunks: the keys span both resident chunks
                dsa_dcp.WIDE_CHUNK_ROWS = chunk
            rng = np.random.default_rng(seed)
            B = len(firsts) * S
            positions = np.array([p0 + r for p0 in firsts for r in range(S)], np.int32)
            pages = max(-(-((p + T) // T) // G) for p in positions) + 1
            print(name, "resident", dsa_dcp.resident_keys(B, dsa_dcp.key_rows_for(pages, T), S))
            pools = rng.integers(0, 256, (G, 1 + len(firsts) * pages, 132, 4, 128), dtype=np.uint8)
            tables = np.zeros((G, B, pages), np.int32)
            new_k8 = np.zeros((B, 128), np.float32); new_exp = np.zeros(B, np.int32)
            for s, p0 in enumerate(firsts):
                L = p0 + S
                raw = rng.standard_normal((L, 128)).astype(np.float32)
                if ties:  # five distinct keys: ties at the threshold
                    raw = rng.standard_normal((5, 128)).astype(np.float32)[rng.integers(0, 5, L)]
                k8, kexp = dsa.quantize_fp8(raw)
                rec = np.asarray(dsa_dcp.key_records(jnp.asarray(raw)))
                for r in range(S):
                    tables[:, s * S + r] = 1 + s * pages + np.arange(pages)
                    new_k8[s * S + r] = k8[p0 + r]; new_exp[s * S + r] = rec[p0 + r, 128]
                for t in range(p0):
                    page = t // T
                    blk = tables[page % G, s * S, page // G]
                    (row, sub), (srow, lane) = dsa_dcp.record_place(t % T)
                    pools[page % G, blk, row, sub] = rec[t, :128]
                    pools[page % G, blk, srow, 0, lane] = rec[t, 128]
            q8 = np.zeros((B, 32, 128), np.float32); w = np.zeros((B, 32), np.float32)
            for b in range(B):
                q8[b], w[b] = dsa.index_query(rng.standard_normal(4096).astype(np.float32),
                                              rng.standard_normal(32).astype(np.float32))
            w_t = np.zeros((32, 128), np.float32); w_t[:, :B] = w.T
            mesh = Mesh(np.asarray(jax.devices()), ("tp",))
            def body(q8, w_t, pool, tables, positions, new_k8, new_exp):
                out = dsa_dcp.select_call(q8, w_t, new_k8, pool[0], tables[0], positions, new_exp,
                                          pages=pages, topk=topk, group=G, max_local_pages=pages,
                                          sequence_rows=S, interpret=pltpu.InterpretParams())
                return tuple(x[None] for x in out)
            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P(), P("tp"), P("tp"), P(), P(), P()),
                                      out_specs=(P("tp"),) * 4, check_vma=False))
            tau, ptie, own, keys = (np.asarray(x) for x in f(
                q8.astype(ml_dtypes.float8_e4m3fn), w_t, pools, tables.reshape(G, -1), positions, new_k8, new_exp))
            row_keys = []
            for b in range(B):
                L = positions[b] + 1
                t = np.arange(L)
                kk = keys[(t // T) % G, b, (t // T) // G * (T // 128) + (t % T) // 128, t % 128].astype(np.int64)
                row_keys.append(kk)
                for v in (tau, ptie, own):
                    assert len(set(v[:, b])) == 1, (name, b)
                if ROUNDS == 8:
                    want = dsa.topk_mask(kk, t, topk)
                elif L <= topk:
                    want = kk > INT_MIN
                else:
                    kth = np.sort(kk)[::-1][topk - 1]
                    drop = 32 - 4 * ROUNDS
                    unsigned = (int(kth) ^ INT_MIN) & 0xFFFFFFFF
                    bound = ((unsigned >> drop) << drop) ^ (1 << 31)
                    want = kk >= (bound - (1 << 32) if bound >= (1 << 31) else bound)
                np.testing.assert_array_equal(dsa_dcp.selected(kk, t, tau[0, b], ptie[0, b]), want)
                assert bool(own[0, b]) == bool(want[-1]), (name, b)
            for key, v in (("tau", tau), ("ptie", ptie), ("own", own), ("keys", np.concatenate(row_keys))):
                outputs[f"{name}_{key}"] = v

        if ROUNDS == 8:
            case("ties", [1500, 2600], 2, 300, 1, ties=True)
            case("ties8", [3069], 8, 400, 6, ties=True)
        else:
            case("b8s8", [6 * T - 3], 8, 400, 2)  # in-flight tokens on two pages (ranks)
            case("b8s1", [300, 2500, 1100, 3600, 700, 2049, 1500, 3000], 1, 256, 3)
            case("b4s2", [1500, 2600], 2, 300, 1)
            case("two24", [9500], 8, 300, 4, chunk=16)  # the second chunk clamped
            case("two32", [13000], 8, 300, 5, chunk=16)
        np.savez(os.environ["OUT"], **outputs)
        print("OK")
    """

    # Rings of other depths than the defaults (not powers of two), the patch
    # loop; then the resident keys (no stage ring) with an eight-page ring.
    KNOBS = "dsa_stage_slots=3,dsa_page_slots=6,dsa_patch_loop=1"
    RESIDENT = "dsa_page_slots=8,dsa_resident_keys=1,dsa_patch_loop=1"
    # dsa_odd_stride: the count buffer's rows an odd number of tiles apart
    # (these cases' 16-row chunks: 24 rows), copied in and resident.
    ODD = "dsa_odd_stride=1"

    def _runs(self, rounds, knob_sets):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            for i, knobs in enumerate(("", *knob_sets)):
                env = dict(
                    os.environ,
                    PYTHONPATH=os.pathsep.join(sys.path),
                    KNOBS=",".join(filter(None, (f"dsa_select_rounds={rounds}", knobs))),
                    OUT=f"{tmp}/run{i}.npz",
                )
                result = subprocess.run(
                    [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=1500,
                )
                self.assertIn(
                    "OK",
                    result.stdout,
                    f"{knobs}: {result.stdout[-2000:]}{result.stderr[-3000:]}",
                )
            base = np.load(f"{tmp}/run0.npz")
            for i, knobs in enumerate(knob_sets, start=1):
                other = np.load(f"{tmp}/run{i}.npz")
                self.assertEqual(sorted(base.files), sorted(other.files))
                for name in base.files:
                    np.testing.assert_array_equal(
                        base[name], other[name], err_msg=f"{knobs}: {name}"
                    )

    def test_bin_selection_is_unchanged(self):
        self._runs(
            5,
            (
                self.KNOBS,
                self.RESIDENT,
                f"{self.KNOBS},{self.ODD}",
                f"{self.RESIDENT},{self.ODD}",
            ),
        )

    def test_exact_selection_with_ties_is_unchanged(self):
        self._runs(8, (self.RESIDENT, f"{self.RESIDENT},{self.ODD}"))


class TestSelectedAttention(CustomTestCase):
    """The DCP attention with a DSA selection (``mla_dcp.DcpSelection``) in
    Pallas interpret mode on a 4-rank group: each row attends exactly its
    selected tokens (pool tokens, cached verify tokens on their owners, its
    own token through the merge), at the FP8 attention's accuracy, and far
    from dense attention. Interpret mode has no ref bitcasts, so the FP8 page
    staging (``_pages``, ``_clean_step``, ``_cache_verify_row``) is replaced
    by value-level equivalents (packed ``[rows, 4, 640]`` or native
    ``[tokens, 640]`` page slots), unless ``K31_INTERPRET_VIEWS=1`` runs the
    kernel's own staging on an emulation of ref views (``TestDsaNativePages``).
    A native page DMA's source reshape (``start_native_copies``) reads the
    indexed pool block, reshaped row-major."""

    SCRIPT = """
        import os
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import ml_dtypes, numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3 import dsa
        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp, mla_dcp

        G, T, H, FP8 = 4, 512, 3, jnp.float8_e4m3fn
        mla_dcp.GROUPS = (G, 8, 16, 32)

        # Interpret mode reads and writes only indexed refs. Native page DMAs read a reshaped pool block,
        # and VIEWS (the kernel's own FP8 staging) loads and stores through bitcast and reshape views.
        VIEWS = os.environ.get("K31_INTERPRET_VIEWS") == "1"
        try:
            from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc
            from jax._src.pallas.mosaic.interpret import utils as interpret_utils
            from jax._src.state.types import BitcastTransform, ReshapeTransform
            ipc.get, ipc.swap, ipc.DMA.execute_read, interpret_utils.to_range
            INTERNALS = True
        except (ImportError, AttributeError):
            INTERNALS = False
        assert INTERNALS or not VIEWS, "interpret-mode views need the interpret internals"

        if INTERNALS:
            _to_range, _read = interpret_utils.to_range, ipc.DMA.execute_read

            def to_range(transforms):  # a DMA source's trailing reshape reads the indexed block
                transforms = list(transforms)
                while transforms and isinstance(transforms[-1], ReshapeTransform):
                    transforms.pop()
                return _to_range(transforms)

            def execute_read(self):  # ... reshaped row-major
                _read(self)
                with self.lock:
                    shapes = [t.shape for t in self.src_transforms if isinstance(t, ReshapeTransform)]
                    if shapes and self.data is not None:
                        self.data = self.data.reshape(shapes[-1])

            interpret_utils.to_range = to_range
            ipc.DMA.execute_read = execute_read

        def bitcast_value(x, dtype):  # pltpu.bitcast: a wider element packs consecutive second-minor rows
            dtype = np.dtype(dtype)
            x = np.ascontiguousarray(x)
            if x.dtype.itemsize == dtype.itemsize:
                return x.view(dtype)
            *lead, m, n = x.shape
            if x.dtype.itemsize < dtype.itemsize:
                k = dtype.itemsize // x.dtype.itemsize
                x = np.ascontiguousarray(np.swapaxes(x.reshape(*lead, m // k, k, n), -1, -2))
                return x.view(dtype).reshape(*lead, m // k, n)
            k = x.dtype.itemsize // dtype.itemsize
            x = x.view(dtype).reshape(*lead, m, n, k)
            return np.ascontiguousarray(np.swapaxes(x, -1, -2)).reshape(*lead, m * k, n)

        def view(x, t):  # the value of view t of array x
            if isinstance(t, BitcastTransform):
                return bitcast_value(x, t.dtype)
            if isinstance(t, ReshapeTransform):
                return x.reshape(t.shape)
            return x[interpret_utils.to_range([t])]

        def unview(x, t, y):  # array x with its view t replaced by y
            if isinstance(t, BitcastTransform):
                return bitcast_value(y, x.dtype)
            if isinstance(t, ReshapeTransform):
                return y.reshape(x.shape)
            x = x.copy()
            x[interpret_utils.to_range([t])] = y
            return x

        def split_views(transforms):  # (the indexers before the first view, the rest)
            transforms = tuple(transforms)
            for i, t in enumerate(transforms):
                if isinstance(t, (BitcastTransform, ReshapeTransform)):
                    return transforms[:i], jax.tree.map(int, transforms[i:])
            return transforms, ()

        if VIEWS:
            _get, _swap = ipc.get, ipc.swap

            def get(token, device, core, space, buffer, transforms, *args, **kwargs):
                lead, views = split_views(transforms)
                token, x = _get(token, device, core, space, buffer, lead if views else transforms, *args, **kwargs)
                for t in views:
                    x = view(x, t)
                return token, x

            def swap(token, device, core, space, buffer, transforms, val, mask, **kwargs):
                lead, views = split_views(transforms)
                if not views:
                    return _swap(token, device, core, space, buffer, transforms, val, mask, **kwargs)
                token, x = _get(token, device, core, space, buffer, lead, **kwargs)
                stages = [x]
                for t in views[:-1]:
                    stages.append(view(stages[-1], t))
                old = view(stages[-1], views[-1])
                new = np.asarray(val) if mask is None else np.where(np.asarray(mask), np.asarray(val), old)
                y = unview(stages[-1], views[-1], new)
                for t, x in zip(reversed(views[:-1]), reversed(stages[:-1])):
                    y = unview(x, t, y)
                token, _ = _swap(token, device, core, space, buffer, lead, y, None, **kwargs)
                return token, old

            get.__wrapped__ = _get.__wrapped__  # DMAs read indexed refs as before
            ipc.get, ipc.swap = get, swap

        def tokens(x):  # each element's token: packed [rows, 4, lanes] (row 4 r + k), native [tokens, lanes]
            if x.ndim == 3:
                return (jax.lax.broadcasted_iota(jnp.int32, x.shape, 0) * x.shape[1]
                        + jax.lax.broadcasted_iota(jnp.int32, x.shape, 1))
            return jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)

        _native_pages = mla_dcp.DcpAttention._pages

        def _pages(self, slot):
            if self.native:  # plain ref loads
                return _native_pages(self, slot)

            def load(lanes, tokens=None):
                x = self.ws.pages[slot]
                return x.reshape(x.shape[0] * x.shape[1], x.shape[2])[:, lanes]
            return load

        def _clean_step(self, slot, row, local_step, tokens_=None, end=None, pad=True):
            x = self.ws.pages[slot]
            cut = 0
            for k in range(self.span):
                first = ((local_step * self.span + k) * self.group + self.owner) * self.page_tokens
                cut = cut + jnp.clip(self._limit(row) - first, 0, self.page_tokens)
            lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, x.ndim - 1)
            keep = tokens(x) < cut
            if pad:  # compact steps (pad=False) leave the stale lanes 576:640 to their gathered tiles
                keep = keep & (lane < 576)
            self.ws.pages[slot] = jnp.where(keep, x, jnp.zeros((), x.dtype))

        def _cache_verify_row(self, slot, local_step, row):
            position = self.position_ref[row]
            page = position // self.page_tokens
            local_page = page // self.group - local_step * self.span

            @pl.when((page % self.group == self.owner) & (local_page >= 0) & (local_page < self.span))
            def write():
                if mla_dcp.chunked(self.batch):
                    chunk = mla_dcp.chunk_layout(self.batch, self.heads)[0]
                    latent = self.ws.new_rows[row // chunk, self.slots * (row % chunk)]
                else:
                    latent = self.ws.new_rows[self.slots * row]
                x = self.ws.pages[slot]
                offset = local_page * self.page_tokens + position % self.page_tokens
                latent = latent.astype(x.dtype).reshape((1,) * (x.ndim - 1) + latent.shape)
                self.ws.pages[slot] = jnp.where(tokens(x) == offset, latent, x)

        def _zero_step_pad_lanes(self, slot):
            x = self.ws.pages[slot]
            lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, x.ndim - 1)
            self.ws.pages[slot] = jnp.where(lane < 576, x, jnp.zeros((), x.dtype))

        def _zero_slot(self, slot):
            x = self.ws.pages[slot]
            self.ws.pages[slot] = jnp.zeros(x.shape, x.dtype)

        if not VIEWS:
            mla_dcp.DcpAttention._pages = _pages
            mla_dcp.DcpAttention._clean_step = _clean_step
            mla_dcp.DcpAttention._zero_slot = _zero_slot
            mla_dcp.DcpAttention._zero_step_pad_lanes = _zero_step_pad_lanes
            mla_dcp.DcpAttention._cache_verify_row = _cache_verify_row

        # Every step activity entry the attention reads, recorded (the reads of
        # KernelConfig.dsa_live_activity's entries: dsa_dcp.live_activity_steps).
        READS = []
        _active = mla_dcp.DcpAttention._active

        def _recorded_active(self, row, local_step):
            if READS and self.selection is not None and self.selection.active is not None:
                READS[0][row * self.selection.steps + local_step] = jnp.int32(1)
            return _active(self, row, local_step)

        mla_dcp.DcpAttention._active = _recorded_active

        import dataclasses
        base_config = mla_dcp._config()

        def case(firsts, S, topk, seed, grouped=False, activity="exact", poison=False, compact=0, expect=None,
                 window=0, wide=0, knobs=(), sent=False, native=False, stale=False):
            # knobs: KernelConfig flags on; sent: the queries leave early (send_queries) and, with
            # dsa_prologue_in_select, another attention stages them as the decode's selection does;
            # native: native page buffers (KernelConfig.dsa_native_pages), each page DMA retiling a pool block;
            # stale: the pool's stale bytes (lanes 576:640, tokens no request wrote) are NaN, as vLLM's can be.
            config = dataclasses.replace(base_config, dsa_compact_tokens=compact, dsa_compact_window=window,
                                         **{knob: True for knob in knobs})
            mla_dcp._config = lambda: config
            mla_dcp._grouped_spec = ((lambda b, g, h, rows: rows in (4, 8) and b % 4 == 0) if grouped
                                     else (lambda *a: False))
            rng = np.random.default_rng(seed)
            positions = np.array([p0 + r for p0 in firsts for r in range(S)], np.int32)
            B = len(positions)
            persistent = mla_dcp.persistent_scratch(B, G, cache_dtype=FP8, sequence_rows=S)
            assert INTERNALS or not native, "native page DMAs need the interpret internals"
            scoped = mla_dcp.scoped_scratch(B, group=G, cache_dtype=FP8, page_tokens=T, sequence_rows=S,
                                            native=True if native else None)
            selection = mla_dcp.selection_scratch(B, page_tokens=T, sequence_rows=S)
            span = selection[0].shape[2] * 128 // T
            pages = -(-(max(-(-(int(p) + T) // T) // G for p in positions) + 1) // span) * span
            pages += wide * span  # a table wider than the context (serving's 1M-context tables)
            key_rows = dsa_dcp.key_rows_for(pages, T, span)
            lat = (rng.standard_normal((len(firsts), pages * G * T, 640)) * 0.5).astype(np.float32)
            lat[..., 576:] = 0
            lat = lat.astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
            pool = rng.standard_normal((G, 1 + len(firsts) * pages, 128, 4, 640)).astype(ml_dtypes.float8_e4m3fn)
            if stale:  # e4m3fn 0x7f
                pool.view(np.uint8)[...] = 0x7F
            tables = np.zeros((G, B, pages), np.int32)
            keys = np.full((G, B, key_rows, 128), -2**31, np.int32)
            tok = rng.integers(-2**31 + 1, 2**31 - 1, (B, pages * G * T), dtype=np.int64)
            for b in range(B):  # a sequence's new tokens rank first: its cached rows and own tokens matter
                tok[b, firsts[b // S] : firsts[b // S] + S] = 2**31 - 2 - np.arange(S)
            for b in range(B):
                s, p0 = b // S, firsts[b // S]
                tables[:, b] = 1 + s * pages + np.arange(pages)
                for t in range(positions[b]):
                    page = t // T; g, j = page % G, page // G
                    if t < p0:
                        pool[g, tables[g, b, j], (t % T) // 4, t % 4] = lat[s, t].astype(ml_dtypes.float8_e4m3fn)
                    keys[g, b, j * (T // 128) + (t % T) // 128, t % 128] = tok[b, t]
                for t in range(positions[b], p0 + S - 1):  # later rows' cached tokens pass: the causal bound
                    page = t // T; g, j = page % G, page // G
                    keys[g, b, j * (T // 128) + (t % T) // 128, t % 128] = tok[b, t]
            if stale:
                pool.view(np.uint8)[..., 576:] = 0x7F
            tau = np.zeros(B, np.int32); ptie = np.zeros(B, np.int32); own = np.zeros(B, np.int32); masks = []
            for b in range(B):
                L = positions[b] + 1
                kk = tok[b, :L]
                m = dsa.topk_mask(kk, np.arange(L), topk)
                order = np.lexsort((np.arange(L), -kk))
                tau[b], ptie[b] = (-2**31, -1) if L <= topk else (kk[order[topk - 1]], 2**31 - 1)
                own[b] = int(m[-1]); masks.append(m)
            step_rows = selection[0].shape[2]
            steps = key_rows // step_rows
            active = np.zeros((G, B * steps), np.int32)
            for b in range(B):
                first = b - b % S
                for t in np.flatnonzero(masks[b]):
                    page = t // T; g, j = page % G, page // G
                    active[g, first * steps + (j * (T // 128) + (t % T) // 128) // step_rows] = 1
            if activity == "none":
                active[:] = 0
            if expect is not None:  # which compact steps the case exercises (per sequence, rank, step)
                counts = []
                for s in range(len(firsts)):
                    union = np.zeros(pages * G * T, bool)
                    for r in range(S):
                        union[:positions[s * S + r] + 1] |= masks[s * S + r]
                        union[positions[s * S + r]] = False  # its own token goes through the merge
                    for g in range(G):
                        for j in range(pages // span):
                            toks = [((j * span + k) * G + g) * T + t for k in range(span) for t in range(T)]
                            counts.append(int(union[toks].sum()))
                kinds = {"compact" if 0 < c <= compact else "dense" if c > compact else "empty" for c in counts}
                assert set(expect) <= kinds, (expect, sorted(counts))
            q = (rng.standard_normal((G, B, H, 640)) * 0.05).astype(np.float32)
            q[..., 576:] = 0
            new = np.broadcast_to(lat[np.arange(B) // S, positions], (G, B, 640)).astype(np.float32)
            n_p, n_s = len(persistent), len(scoped)

            def kernel(pos, table, tau_r, ptie_r, own_r, act_r, q_r, new_r, pool_r, keys_r, out_r, reads_r,
                       *scratch):
                for i in range(B * steps):
                    reads_r[i] = jnp.int32(0)
                READS[:] = [reads_r]
                cache = mla_dcp.DcpCache((pool_r,), table, pages, 0, group=G, global_pages=False)
                ws = mla_dcp.DcpWorkspace(*scratch[:n_p], *scratch[n_p:n_p + n_s])
                sel = mla_dcp.DcpSelection(keys_r, tau_r, ptie_r, own_r, *scratch[n_p + n_s:],
                                           active=act_r, steps=steps)
                att = mla_dcp.DcpAttention(cache, ws, pos, B, selection=sel, sequence_rows=S,
                                           cache_verify_rows=S > 1)
                assert att.native == native
                if poison:  # another scoped region's bytes: NaN where no step loads a page
                    ws.pages[...] = jnp.full(ws.pages.shape, jnp.nan, ws.pages.dtype)
                att.prime()
                if not sent:
                    out_r[...] = att.attend(q_r[...], new_r[...], 0)
                else:
                    att.send_queries(q_r[...], new_r[...], 0)
                    staged = "dsa_prologue_in_select" in knobs
                    if staged:
                        mla_dcp.DcpAttention(cache, ws, pos, B, sequence_rows=S,
                                             cache_verify_rows=S > 1).stage_queries(0)
                    out_r[...] = att.attend(None, new_r[...], 0, queries_sent=True, queries_staged=staged)
                READS[:] = []

            def body(pos, table, tau_, ptie_, own_, act_, q_, new_, pool_, keys_):
                vmem = pltpu.VMEM
                out, reads = pl.pallas_call(kernel, out_shape=(jax.ShapeDtypeStruct((B, H, 512), jnp.float32),
                                                               jax.ShapeDtypeStruct((B * steps,), jnp.int32)),
                    grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=6,
                        in_specs=[pl.BlockSpec(memory_space=vmem)] * 2 + [pl.BlockSpec(memory_space=pl.ANY)] * 2,
                        out_specs=(pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=pltpu.SMEM)),
                        scratch_shapes=[*persistent, *scoped, *selection]),
                    compiler_params=pltpu.CompilerParams(has_side_effects=True),
                    interpret=pltpu.InterpretParams())(pos, table[0], tau_, ptie_, own_, act_[0], q_[0], new_[0],
                                                       pool_[0], keys_[0])
                return out[None], reads[None]

            mesh = Mesh(np.asarray(jax.devices()), ("tp",))
            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P("tp"), P(), P(), P()) + (P("tp"),) * 5,
                                      out_specs=(P("tp"), P("tp")), check_vma=False))
            out, reads = (np.asarray(x) for x in f(positions, tables.reshape(G, -1), tau, ptie, own, active, q, new,
                                                   pool, keys))
            # The attention reads only each sequence's first row's entries below
            # dsa_dcp.live_activity_steps (the selection's longest row's steps and one more).
            for g in range(G):
                longest = max(max(-(-(int(p) + 1) // T) - g + G - 1, 0) // G for p in positions)
                live = min(-(-longest * (T // 128) // step_rows) + 1, steps)
                domain = np.zeros(B * steps, bool)
                for first in range(0, B, S):
                    domain[first * steps:first * steps + live] = True
                assert not reads[g][~domain].any(), (g, np.flatnonzero(reads[g]), live, steps)
            if activity == "none":  # no step is read: only the rows' own tokens (where selected) remain
                L = positions[-1] + 1
                vals = lat[(B - 1) // S, :L]
                s_ = vals @ q[0, B - 1, 0]
                p = np.exp(np.where(masks[-1], s_, -np.inf) - s_[masks[-1]].max()); p /= p.sum()
                ref = p @ vals[:, :512]
                assert np.abs(out[0, B - 1, 0] - ref).max() / np.abs(ref).max() > 0.3
                return out
            for b in range(B):
                L = positions[b] + 1
                vals = lat[b // S, :L]
                for d in range(G):
                    for j in range(H):
                        s_ = vals @ q[d, b, j]
                        errs = []
                        for m in (masks[b], np.ones(L, bool)):
                            p = np.exp(np.where(m, s_, -np.inf) - s_[m].max()); p /= p.sum()
                            ref = p @ vals[:, :512]
                            errs.append(np.abs(out[d, b, j] - ref).max() / np.abs(ref).max())
                        assert errs[0] < 6e-2, (b, d, j, errs)
                        assert L <= topk or errs[1] > 0.3, (b, d, j, errs)  # the mask matters
            return out

        case([700, 2000], 1, 64, 0)
        case([1500, 3000], 4, 200, 1, grouped=True)
        case([2047, 600, 64, 3000], 2, 128, 2)
        case([700, 5000], 1, 64, 3, activity="none")
        case([700, 9000], 1, 8, 4, poison=True)  # skipped steps on never-loaded slots stay finite
        # Compact verify steps (KernelConfig.dsa_compact_tokens): few selected tokens a step attend
        # only those, gathered; busier steps fall back to the dense masked step.
        case([1500, 3000], 4, 24, 5, grouped=True, compact=64, expect=("compact",))
        case([1500, 3000], 4, 200, 1, grouped=True, compact=128, expect=("compact", "dense"))
        case([900, 2600], 8, 40, 6, grouped=True, compact=64, expect=("compact",))
        # Windowed compact steps (KernelConfig.dsa_compact_window): gathered tokens of several steps fold
        # once; a small window flushes when the next step's slots would not fit.
        case([1500, 3000], 4, 200, 1, grouped=True, compact=128, expect=("compact", "dense"), window=512)
        case([900, 2600], 8, 40, 6, grouped=True, compact=64, expect=("compact",), window=64)
        case([700, 9000], 4, 8, 4, grouped=True, compact=64, poison=True, window=256)
        # Tables wider than the contexts: the reads stay within the live steps.
        case([700, 2000], 1, 64, 0, wide=3)
        case([1500, 3000], 4, 200, 1, grouped=True, compact=128, expect=("compact", "dense"), window=512,
             wide=3)
        case([3069], 8, 400, 7, grouped=True, compact=128, window=512, wide=3)
        print("OK")
    """

    def test_rows_attend_their_selected_tokens(self):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])


class TestPairCombine(CustomTestCase):
    """``KernelConfig.dcp_pair_combine`` (the two-stage DCP combine) in Pallas
    interpret mode on a 4-rank group whose chips are ranks 0/1 and 2/3, with
    a DSA selection: ``attend`` and ``send_queries`` + ``attend(queries_sent=
    True)`` agree bitwise, and every output is within one BF16 rounding of the
    one-stage combine's: |pair - one-stage| <= 2^-8 sum_i softmax(lse)_i |o_i|
    over the owner's received partials. A rank whose mate holds no selected
    tokens passes its partial through unchanged (the skewed case is bitwise
    the one-stage result)."""

    SCRIPT = """
        import dataclasses, os
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import ml_dtypes, numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3 import dsa
        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp, mla_dcp

        G, T, H, FP8 = 4, 512, 3, jnp.float8_e4m3fn
        mla_dcp.GROUPS = (G, 8, 16, 32)

        def tokens(x):
            return (jax.lax.broadcasted_iota(jnp.int32, x.shape, 0) * x.shape[1]
                    + jax.lax.broadcasted_iota(jnp.int32, x.shape, 1))

        def _pages(self, slot):  # interpret mode has no ref bitcasts (see TestSelectedAttention)
            def load(lanes, tokens=None):
                x = self.ws.pages[slot]
                return x.reshape(x.shape[0] * x.shape[1], x.shape[2])[:, lanes]
            return load

        def _clean_step(self, slot, row, local_step, tokens_=None, end=None, pad=True):
            x = self.ws.pages[slot]
            cut = 0
            for k in range(self.span):
                first = ((local_step * self.span + k) * self.group + self.owner) * self.page_tokens
                cut = cut + jnp.clip(self._limit(row) - first, 0, self.page_tokens)
            lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 2)
            keep = (tokens(x) < cut) & ((lane < 576) if pad else True)
            self.ws.pages[slot] = jnp.where(keep, x, jnp.zeros((), x.dtype))

        def _cache_verify_row(self, slot, local_step, row):
            position = self.position_ref[row]
            page = position // self.page_tokens
            local_page = page // self.group - local_step * self.span

            @pl.when((page % self.group == self.owner) & (local_page >= 0) & (local_page < self.span))
            def write():
                latent = self.ws.new_rows[self.slots * row]
                x = self.ws.pages[slot]
                offset = local_page * self.page_tokens + position % self.page_tokens
                self.ws.pages[slot] = jnp.where(tokens(x) == offset, latent.astype(x.dtype)[None, None, :], x)

        def _zero_step_pad_lanes(self, slot):
            x = self.ws.pages[slot]
            lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 2)
            self.ws.pages[slot] = jnp.where(lane < 576, x, jnp.zeros((), x.dtype))

        def _zero_slot(self, slot):
            self.ws.pages[slot] = jnp.zeros(self.ws.pages.shape[1:], self.ws.pages.dtype)

        mla_dcp.DcpAttention._pages = _pages
        mla_dcp.DcpAttention._clean_step = _clean_step
        mla_dcp.DcpAttention._zero_slot = _zero_slot
        mla_dcp.DcpAttention._zero_step_pad_lanes = _zero_step_pad_lanes
        mla_dcp.DcpAttention._cache_verify_row = _cache_verify_row
        base_config = mla_dcp._config()

        def run(firsts, S, topk, seed, pair, early, compact=0, window=0, favor=None):
            config = dataclasses.replace(base_config, dsa_compact_tokens=compact, dsa_compact_window=window,
                                         dcp_pair_combine=pair)
            mla_dcp._config = lambda: config
            mla_dcp._grouped_spec = lambda b, g, h, rows: rows in (4, 8) and b % 4 == 0
            rng = np.random.default_rng(seed)
            positions = np.array([p0 + r for p0 in firsts for r in range(S)], np.int32)
            B = len(positions)
            persistent = mla_dcp.persistent_scratch(B, G, cache_dtype=FP8, sequence_rows=S)
            scoped = mla_dcp.scoped_scratch(B, group=G, cache_dtype=FP8, page_tokens=T, sequence_rows=S)
            selection = mla_dcp.selection_scratch(B, page_tokens=T, sequence_rows=S)
            span = selection[0].shape[2] * 128 // T
            pages = -(-(max(-(-(int(p) + T) // T) // G for p in positions) + 1) // span) * span
            key_rows = dsa_dcp.key_rows_for(pages, T, span)
            lat = (rng.standard_normal((len(firsts), pages * G * T, 640)) * 0.5).astype(np.float32)
            lat[..., 576:] = 0
            lat = lat.astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
            pool = rng.standard_normal((G, 1 + len(firsts) * pages, 128, 4, 640)).astype(ml_dtypes.float8_e4m3fn)
            tables = np.zeros((G, B, pages), np.int32)
            keys = np.full((G, B, key_rows, 128), -2**31, np.int32)
            tok = rng.integers(-2**31 + 1, 2**31 - 1, (B, pages * G * T), dtype=np.int64)
            if favor is not None:  # rank `favor`'s tokens hold the top keys: only it attends pages
                hot = (np.arange(pages * G * T) // T) % G == favor
                tok[:, hot] = 2**31 - 2 - rng.integers(0, 2**20, (B, int(hot.sum())))
            for b in range(B):
                s, p0 = b // S, firsts[b // S]
                tables[:, b] = 1 + s * pages + np.arange(pages)
                for t in range(positions[b]):
                    page = t // T; g, j = page % G, page // G
                    if t < p0:
                        pool[g, tables[g, b, j], (t % T) // 4, t % 4] = lat[s, t].astype(ml_dtypes.float8_e4m3fn)
                    keys[g, b, j * (T // 128) + (t % T) // 128, t % 128] = tok[b, t]
            tau = np.zeros(B, np.int32); ptie = np.zeros(B, np.int32); own = np.zeros(B, np.int32)
            steps = key_rows // selection[0].shape[2]
            active = np.zeros((G, B * steps), np.int32)
            for b in range(B):
                L = positions[b] + 1
                kk = tok[b, :L]
                m = dsa.topk_mask(kk, np.arange(L), topk)
                order = np.lexsort((np.arange(L), -kk))
                tau[b], ptie[b] = (-2**31, -1) if L <= topk else (kk[order[topk - 1]], 2**31 - 1)
                own[b] = int(m[-1])
                for t in np.flatnonzero(m):
                    page = t // T; g, j = page % G, page // G
                    active[g, (b - b % S) * steps + (j * (T // 128) + (t % T) // 128) // selection[0].shape[2]] = 1
            q = (rng.standard_normal((G, B, H, 640)) * 0.05).astype(np.float32)
            q[..., 576:] = 0
            new = np.broadcast_to(lat[np.arange(B) // S, positions], (G, B, 640)).astype(np.float32)
            n_p, n_s, rows = len(persistent), len(scoped), persistent[2].shape[2]

            def kernel(pos, table, tau_r, ptie_r, own_r, act_r, q_r, new_r, pool_r, keys_r, out_r, recv_r, *scratch):
                cache = mla_dcp.DcpCache((pool_r,), table, pages, 0, group=G, global_pages=False)
                ws = mla_dcp.DcpWorkspace(*scratch[:n_p], *scratch[n_p:n_p + n_s])
                sel = mla_dcp.DcpSelection(keys_r, tau_r, ptie_r, own_r, *scratch[n_p + n_s:],
                                           active=act_r, steps=steps)
                att = mla_dcp.DcpAttention(cache, ws, pos, B, selection=sel, sequence_rows=S,
                                           cache_verify_rows=S > 1)
                if early:  # the queries leave first (as under the indexer); work fills the combine
                    mla_dcp.DcpAttention(cache, ws, pos, B, sequence_rows=S, cache_verify_rows=S > 1
                                         ).send_queries(q_r[...], new_r[...], 0)
                    att.prime()

                    def while_combining():
                        out_r[...] = jnp.full(out_r.shape, 3.0, jnp.float32)

                    out_r[...] = att.attend(None, new_r[...], 0, queries_sent=True,
                                            while_combining=while_combining)
                else:
                    att.prime()
                    out_r[...] = att.attend(q_r[...], new_r[...], 0)
                recv_r[...] = ws.receive[0].astype(jnp.float32)

            def body(pos, table, tau_, ptie_, own_, act_, q_, new_, pool_, keys_):
                vmem = pltpu.VMEM
                out, recv = pl.pallas_call(kernel,
                    out_shape=(jax.ShapeDtypeStruct((B, H, 512), jnp.float32),
                               jax.ShapeDtypeStruct((G, rows, 640), jnp.float32)),
                    grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=6,
                        in_specs=[pl.BlockSpec(memory_space=vmem)] * 2 + [pl.BlockSpec(memory_space=pl.ANY)] * 2,
                        out_specs=(pl.BlockSpec(memory_space=vmem),) * 2,
                        scratch_shapes=[*persistent, *scoped, *selection]),
                    compiler_params=pltpu.CompilerParams(has_side_effects=True),
                    interpret=pltpu.InterpretParams())(pos, table[0], tau_, ptie_, own_, act_[0], q_[0], new_[0],
                                                       pool_[0], keys_[0])
                return out[None], recv[None]

            mesh = Mesh(np.asarray(jax.devices()), ("tp",))
            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P("tp"), P(), P(), P()) + (P("tp"),) * 5,
                                      out_specs=(P("tp"), P("tp")), check_vma=False))
            out, recv = f(positions, tables.reshape(G, -1), tau, ptie, own, active, q, new, pool, keys)
            return np.asarray(out), np.asarray(recv), mla_dcp.packet_layout(B)[0]

        def case(*args, exact=False, **kwargs):
            one, recv, slots = run(*args, pair=False, early=False, **kwargs)
            two, _, _ = run(*args, pair=True, early=False, **kwargs)
            two_early, _, _ = run(*args, pair=True, early=True, **kwargs)
            assert np.array_equal(two, two_early), "queries_sent changes the pair combine's outputs"
            lse = recv[..., 512] + recv[..., 513]  # [owner, source, packet row]
            w = np.exp(lse - lse.max(axis=1, keepdims=True))
            w /= w.sum(axis=1, keepdims=True)
            bound = np.einsum("dir,dire->dre", w, np.abs(recv[..., :512]))
            bound = np.stack([bound[:, slots * b:slots * b + H] for b in range(one.shape[1])], axis=1)
            diff = np.abs(two.astype(np.float64) - one)
            assert np.isfinite(two).all()
            assert (diff <= 2.0**-8 * bound * (1 + 2.0**-6) + 1e-7 * bound.max()).all(), (
                args, (diff / np.maximum(2.0**-8 * bound, 1e-30)).max())
            assert not exact or np.array_equal(one, two), args
            print(args, "max |pair - one-stage| / (2^-8 bound):", (diff / np.maximum(2.0**-8 * bound, 1e-30)).max())

        # One 8-row verify (the B1 x7 shape) on compact steps with a 512-slot window; two 4-row
        # sequences dense; plain decode; ranks skewed so that only rank 1 attends pages (its mate and
        # the other chip reach the combine first; the mates' partials pass through).
        case([3000], 8, 200, 11, compact=128, window=512)
        case([1500, 3000], 4, 200, 1)
        case([2047, 600, 64, 3000], 1, 128, 2)
        case([3000], 8, 200, 13, favor=1, exact=True)
        print("OK")
    """

    def test_pair_combine_within_one_bf16_rounding(self):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])


class TestPairLseRows(CustomTestCase):
    """``KernelConfig.dcp_pair_lse_rows`` (the pair combine's stage two as
    the latent lanes plus two lse rows) must not change a bit. The
    attention: ``TestSelectedAttention``'s harness (interpret mode, a 4-rank
    group of chips 0/1 and 2/3, scratch NaN until written) with
    ``dcp_pair_combine``, the knob off and on, outputs compared bit for bit:
    B1 x7 (one 8-row sequence, compact steps, a 512-slot window) with and
    without early queries, two 4-row sequences (dense and compact steps),
    poisoned page slots, plain B4 and B1 rows (12- and 3-row packets) and a
    step activity that attends nothing (every partial empty). The lse rows'
    helpers: the same bits as the 640-lane packets' for any values."""

    TAIL = """
        PAIR = ("dcp_pair_combine",)
        for args, kwargs, modes in (
            (([2600], 8, 40, 7), dict(grouped=True, compact=128, window=512), (False, True)),
            (([1500, 3000], 4, 200, 1), dict(grouped=True, compact=128, window=512, expect=("compact", "dense")),
             (False,)),
            (([700, 9000], 4, 8, 4), dict(grouped=True, compact=64, poison=True, window=256), (True,)),
            (([2047, 600, 64, 3000], 1, 128, 2), dict(), (False,)),
            (([3000], 1, 64, 9), dict(wide=3), (True,)),
            (([700, 5000, 1200, 2600], 1, 64, 3), dict(activity="none"), (False,)),
        ):
            for sent in modes:
                off = case(*args, **kwargs, sent=sent, knobs=PAIR)
                on = case(*args, **kwargs, sent=sent, knobs=PAIR + ("dcp_pair_lse_rows",))
                assert np.isfinite(on).all(), (args, sent)
                assert np.array_equal(off.view(np.uint32), on.view(np.uint32)), (args, sent, np.abs(off - on).max())
                print(args, sent, "bit-identical", flush=True)
        print("OK")
    """

    def test_lse_rows_hold_the_packets_bits(self):
        """The helpers alone (XLA on CPU), at 24, 12 and 3 packet rows:
        ``_merge_pair_lse_rows`` gives ``_merge_pair``'s latent lanes and the
        BF16 values of its lse lanes 512 and 513 lane-major (zeros past the
        rows), and ``_lse_of_rows`` the FP32 sums lanes 512 + 513 give: bit
        for bit for nonzero values and infinities, NaNs as NaNs and zeros as
        zeros of either sign (see ``mla_dcp._lse_of_rows``)."""
        import jax
        import jax.numpy as jnp
        import ml_dtypes

        from sglang_tpu.k3_programs.kimi_k3.kernels import mla_dcp

        bf16 = ml_dtypes.bfloat16
        rng = np.random.default_rng(7)

        def bits(x):
            x = np.asarray(x)
            return x.view(np.uint16 if x.dtype == bf16 else np.uint32)

        def packet(lse):
            out = rng.standard_normal((len(lse), 640)).astype(np.float32)
            high = lse.astype(bf16).astype(np.float32)
            out[:, 512], out[:, 513], out[:, 514:] = high, lse - high, 0
            return out.astype(bf16)

        merge = jax.jit(mla_dcp._merge_pair)
        merge_rows = jax.jit(mla_dcp._merge_pair_lse_rows)
        lse_of_rows = jax.jit(mla_dcp._lse_of_rows, static_argnums=1)
        f32 = jnp.float32
        add = jax.jit(lambda x: x[:, 0].astype(f32) + x[:, 1].astype(f32))
        inf, nan = np.inf, np.nan
        high = [-0.0, -0.0, 1e-40, 0.0, inf, -inf, nan, 3.5, -1e-39, 1.0, 1e-40]
        low = [-0.0, 0.0, 1e-40, -1e-40, 1.0, 0.0, 0.0, 2.0**-10, 0.0, -0.0, -1e-40]
        for rows in (24, 12, 3):
            for scale in (1.0, 40.0):
                first = packet(rng.standard_normal(rows).astype(np.float32) * scale)
                second = packet(rng.standard_normal(rows).astype(np.float32) * scale)
                merged = bits(merge(first, second))
                outputs, lse_rows = (bits(x) for x in merge_rows(first, second))
                np.testing.assert_array_equal(outputs, merged[:, :512])
                np.testing.assert_array_equal(lse_rows[:, :rows], merged[:, 512:514].T)
                self.assertFalse(lse_rows[:, rows:].any())
            parts = rng.standard_normal((16, 2, 128)).astype(np.float32)
            parts[:, 0, : min(rows, len(high))] = high[:rows]
            parts[:, 1, : min(rows, len(low))] = low[:rows]
            parts = parts.astype(bf16)
            got = np.asarray(lse_of_rows(parts, rows))
            want = np.asarray(add(parts))[:, :rows]
            self.assertEqual(got.shape, (16, rows, 1))
            got = got[:, :, 0]
            nans, zeros = np.isnan(want), want == 0
            np.testing.assert_array_equal(np.isnan(got), nans)
            np.testing.assert_array_equal(got[zeros], 0.0)  # either sign
            other = ~nans & ~zeros
            np.testing.assert_array_equal(bits(got)[other], bits(want)[other])

    def test_lse_rows_leave_the_outputs_bit_identical(self):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        harness = textwrap.dedent(TestSelectedAttention.SCRIPT)
        harness = harness[: harness.index("\ncase([700, 2000], 1, 64, 0)\n") + 1]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", harness + textwrap.dedent(self.TAIL)],
            capture_output=True,
            text=True,
            env=env,
            timeout=2400,
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])


class TestDraftPairCombine(CustomTestCase):
    """``KernelConfig.draft_pair_combine``: the MLA draft's attention
    (explicit keys, its lse in the padding row, FP8 pages) through the host
    combine, its last chunk with the pair stage, in Pallas interpret mode at
    DCP16 with ``HOST_LANES`` 4 (four hosts of two chips; the draft runs four
    of four), one- and two-chunk calls: finite, an earlier chunk's rows the
    knob-off bits, the last chunk's within the combine's BF16 roundings of
    them -- |on - off| <= 3 2^-8 sum_i softmax(lse)_i |o_i| over every rank's
    partial of the row (the pair merge's rounding, plus the host merge's
    re-rounding of a value that moved)."""

    SCRIPT = """
        import os, time
        G = 16
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={G}"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import ml_dtypes, numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3.kernels import mla_dcp

        # The MLA draft's attention (explicit keys, lse in the padding row, FP8 pages) through the host combine,
        # with and without its pair stage. HOST_LANES 4 at DCP16: 4 hosts of 2 chips (the draft: 4 hosts of 4 chips).
        T, H, S, FP8 = 512, 3, 8, jnp.float8_e4m3fn
        mla_dcp.HOST_LANES = G // 4
        mla_dcp.GROUPS = (G, 8, 16, 32)
        LANES = mla_dcp.LANES

        def tokens(x):
            return (jax.lax.broadcasted_iota(jnp.int32, x.shape, 0) * x.shape[1]
                    + jax.lax.broadcasted_iota(jnp.int32, x.shape, 1))

        def _pages(self, slot):  # interpret mode has no ref bitcasts (see TestSelectedAttention)
            def load(lanes, tokens=None):
                x = self.ws.pages[slot]
                x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
                if hasattr(lanes, "size"):  # pl.ds
                    return jax.lax.dynamic_slice_in_dim(x, lanes.start, lanes.size, 1)
                return x[:, lanes]
            return load

        def _clean_step(self, slot, row, local_step, tokens_=None, end=None, pad=True):
            x = self.ws.pages[slot]
            cut = 0
            for k in range(self.span):
                first = ((local_step * self.span + k) * self.group + self.owner) * self.page_tokens
                cut = cut + jnp.clip(self._limit(row) - first, 0, self.page_tokens)
            lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 2)
            keep = (tokens(x) < cut) & ((lane < 576) if pad else True)
            self.ws.pages[slot] = jnp.where(keep, x, jnp.zeros((), x.dtype))

        def _zero_step_pad_lanes(self, slot):
            x = self.ws.pages[slot]
            lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 2)
            self.ws.pages[slot] = jnp.where(lane < 576, x, jnp.zeros((), x.dtype))

        def _zero_slot(self, slot):
            self.ws.pages[slot] = jnp.zeros(self.ws.pages.shape[1:], self.ws.pages.dtype)

        _pack = mla_dcp.DcpAttention._pack_partials

        def _pack_partials(self, first, end, base=0):  # also keep each chunk's packets (every owner's partials)
            _pack(self, first, end, base)
            capture, counter = self._capture
            capture[counter[0]] = self.ws.send[...].astype(jnp.float32)
            counter[0] += 1

        mla_dcp.DcpAttention._pack_partials = _pack_partials
        mla_dcp.DcpAttention._pages = _pages
        mla_dcp.DcpAttention._clean_step = _clean_step
        mla_dcp.DcpAttention._zero_slot = _zero_slot
        mla_dcp.DcpAttention._zero_step_pad_lanes = _zero_step_pad_lanes

        def run(firsts, counts, seed, pairs):
            # Outputs [G, B, H, 512] and every rank's packets of each chunk [G, chunks, G, H S, 640].
            rng = np.random.default_rng(seed)
            B = len(firsts) * S
            positions = np.array([p0 + r for p0 in firsts for r in range(S)], np.int32)
            plan = mla_dcp.page_plan(B, H, S, FP8, force_chunks=True)
            persistent = mla_dcp.persistent_scratch(B, G, H, cache_dtype=FP8, sequence_rows=S, fp8_wire=True,
                                                    force_chunks=True, chunk_rows=S)
            scoped = mla_dcp.scoped_scratch(B, plan["span"], plan["buffers"], heads=H, group=G, cache_dtype=FP8,
                                            page_tokens=T, sequence_rows=S, force_chunks=True, chunk_rows=S)
            ring, hosts = persistent[0].shape[0], G // mla_dcp.HOST_LANES
            stage2 = (pltpu.VMEM((ring, hosts, H * S, LANES), jnp.bfloat16),
                      pltpu.VMEM((hosts, H * S, LANES), jnp.bfloat16), pltpu.SemaphoreType.DMA((2, ring)))
            extra = (pltpu.SemaphoreType.DMA((2, ring)),) if pairs else ()
            pages = max(-(-(int(p0) + T) // T) // G for p0 in firsts) + 1
            lat = (rng.standard_normal((len(firsts), pages * G * T, LANES)) * 0.5).astype(np.float32)
            lat[..., 576:] = 0
            pool = rng.standard_normal((G, 1 + len(firsts) * pages, 128, 4, LANES)).astype(ml_dtypes.float8_e4m3fn)
            tables = np.zeros((G, B, pages), np.int32)
            for b in range(B):
                s_, p0 = b // S, firsts[b // S]
                tables[:, b] = 1 + s_ * pages + np.arange(pages)
                for t in range(p0):
                    page = t // T; g, j = page % G, page // G
                    pool[g, tables[g, b, j], (t % T) // 4, t % 4] = lat[s_, t].astype(ml_dtypes.float8_e4m3fn)
            q = (rng.standard_normal((G, B, H, LANES)) * 0.08).astype(np.float32)
            q[..., 576:] = 0
            keys = (rng.standard_normal((len(firsts), 2 * S, LANES)) * 0.5).astype(np.float32)
            keys[..., 576:] = 0
            lane = np.arange(2 * S)
            valid = np.stack([((lane < S) & (lane <= c)) | ((lane >= S) & (lane < 2 * S - 1)) for c in counts])
            valid = valid[:, None, :].astype(np.int32)
            n_p, n_s = len(persistent), len(scoped)
            rows = persistent[1].shape[1]

            def kernel(pos, table, q_r, keys_r, valid_r, pool_r, out_r, send_r, *scratch):
                counter = scratch[-1]
                counter[0] = 0
                scratch = scratch[:-1]
                cache = mla_dcp.DcpCache((pool_r,), table, pages, 0, group=G, global_pages=False)
                ws = mla_dcp.DcpWorkspace(*scratch[:n_p], *scratch[n_p:n_p + n_s])
                att = mla_dcp.DcpAttention(cache, ws, pos, B, sequence_rows=S, heads=H, chunk_rows=S)
                att.lse_in_padding = True
                att.group_fold = True
                att.force_chunked = True
                att.host_combine = tuple(scratch[n_p + n_s : n_p + n_s + 3])
                att.host_pairs = scratch[n_p + n_s + 3] if pairs else None
                att._capture = (send_r, counter)
                att.prime()
                out_r[...] = att.attend(q_r[...], jnp.zeros((B, LANES), jnp.float32), 0, keys=keys_r, key_valid=valid_r)

            def body(pos, table, q_, keys_, valid_, pool_):
                vmem = pltpu.VMEM
                out, send = pl.pallas_call(kernel,
                    out_shape=(jax.ShapeDtypeStruct((B, H, 512), jnp.float32),
                               jax.ShapeDtypeStruct((B // S, G, rows, LANES), jnp.float32)),
                    grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=2,
                        in_specs=[pl.BlockSpec(memory_space=vmem)] * 3 + [pl.BlockSpec(memory_space=pl.ANY)],
                        out_specs=(pl.BlockSpec(memory_space=vmem),) * 2,
                        scratch_shapes=[*persistent, *scoped, *stage2, *extra, pltpu.SMEM((1,), jnp.int32)]),
                    compiler_params=pltpu.CompilerParams(has_side_effects=True),
                    interpret=pltpu.InterpretParams())(pos, table[0], q_[0], keys_, valid_, pool_[0])
                return out[None], send[None]

            mesh = Mesh(np.asarray(jax.devices()), ("tp",))
            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P("tp"), P("tp"), P(), P(), P("tp")),
                                      out_specs=(P("tp"), P("tp")), check_vma=False))
            out, send = f(positions, tables.reshape(G, -1), q, keys, valid, pool)
            return np.asarray(out), np.asarray(send)

        def bound(send):
            # 2^-8 sum_i softmax(lse)_i |o_i| over every rank's partial of each owner's real rows ([G, S - 1, H, 512];
            # row S - 1 is the padding row, whose packet rows carry the lse's).
            hl = mla_dcp.HOST_LANES
            hosts = G // hl
            out = np.zeros((G, S, H, 512))
            for m in range(G):
                idx = (m % hl) * hosts + m // hl  # the owner's packet in the lane-major send buffer
                part = send[:, idx]  # [G ranks, H S, 640]
                lse = part[..., 512] + part[..., 513]  # [G, rows]
                w = np.exp(lse - lse.max(axis=0, keepdims=True)); w /= w.sum(axis=0, keepdims=True)
                b = np.einsum("gr,gre->re", w, np.abs(part[..., :512]))  # [rows, 512]
                out[m] = b.reshape(S, H, 512)
            return 2.0 ** -8 * out[:, : S - 1]

        t0 = time.time()
        ok = True
        for firsts, counts, seed in (([3000], [5], 1), ([5200], [7], 2), ([1500, 4100], [3, 7], 3)):
            off, send = run(firsts, counts, seed, False)
            on, _ = run(firsts, counts, seed, True)
            B = len(firsts) * S
            diff = np.abs(on.astype(np.float64) - off)  # [G, B, H, 512]
            finite = bool(np.isfinite(on).all())
            good, ratio, rel = finite, 0.0, 0.0
            # Only the last chunk takes the pair stage: the earlier ones' outputs are the knob-off bits.
            good &= bool(np.array_equal(on[:, : B - S], off[:, : B - S]))
            for c in range(len(firsts)):  # each chunk's real rows against its partials' bound
                bnd = bound(send[:, c])
                d = diff[:, c * S : c * S + S - 1]
                good &= bool((d <= 3 * bnd * (1 + 2.0**-6) + 1e-7 * bnd.max()).all())
                ratio = max(ratio, float((d / np.maximum(bnd, 1e-30)).max()))
                rel = max(rel, float(d.max() / np.abs(off[:, c * S : c * S + S - 1]).max()))
            print(f"firsts {firsts} counts {counts}: identical {np.array_equal(on, off)} max |on - off| / (2^-8 bound) "
                  f"{ratio:.3f} max rel {rel:.2e} finite {finite} -> {'ok' if good else 'FAIL'} ({time.time() - t0:.0f}s)",
                  flush=True)
            ok &= good
        print("OK" if ok else "FAIL")
    """

    def test_pair_stage_within_its_roundings(self):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        # 16 interpret devices block in callbacks: give the CPU client enough
        # threads (XLA reads NPROC for its pool size).
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), NPROC="128")
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=2400,
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])


class TestK31DraftNativePages(CustomTestCase):
    """``KernelConfig.draft_native_pages`` and ``draft_step_trims``: the MLA
    draft's attention (explicit keys, its lse in the padding row, FP8 pages,
    the host combine) in Pallas interpret mode at DCP16 with ``HOST_LANES``
    4, its pages staged native (each page DMA retiles a pool block) and / or
    its step loop trimmed (interior steps unmasked, the last peeled, the PV's
    1/256 applied once), gives the knob-off outputs and every rank's partials
    bit for bit: one eight-row sequence (a chunk) with two to three steps a
    rank (an odd page count re-reads its last page in the last step, which
    also holds the partial last page), two chunks with the pair stage (ranks
    without pages), four-row sequences, and attention DP2's draft (six
    heads, one unchunked sequence). Stale pool bytes are NaN, so a step that
    skips a zeroing it needs is not finite.

    Interpret mode reads only indexed refs: a DMA source's trailing reshape
    (``start_native_copies``) is applied to the data read, row-major as a ref
    reshape reads it, and the bitcast staging methods are replaced by
    value-level equivalents (packed pages; native pages load as plain refs).
    The CPU runs without fused multiply-add (``--xla_cpu_max_isa=AVX``): the
    TPU kernel rounds every product and sum (its bundles have no FMA), while
    XLA:CPU contracts a product into the next add only where no select sits
    between them, which the unmasked steps remove."""

    SCRIPT = """
        import os, time
        G = 16
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={G} --xla_cpu_max_isa=AVX"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import ml_dtypes, numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3.kernels import mla_dcp
        try:
            from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc
            from jax._src.pallas.mosaic.interpret import utils as interpret_utils
            from jax._src.state.types import ReshapeTransform
            ipc.DMA.execute_read, interpret_utils.to_range
        except (ImportError, AttributeError) as e:
            print("SKIP: interpret-mode DMA internals moved:", e)
            raise SystemExit(0)

        T, FP8 = 512, jnp.float8_e4m3fn
        mla_dcp.HOST_LANES = G // 4
        mla_dcp.GROUPS = (G, 8, 16, 32)
        LANES = mla_dcp.LANES

        _to_range = interpret_utils.to_range

        def to_range(transforms):  # a trailing reshape reads the indexed block
            transforms = list(transforms)
            while transforms and isinstance(transforms[-1], ReshapeTransform):
                transforms.pop()
            return _to_range(transforms)

        _read = ipc.DMA.execute_read

        def execute_read(self):  # ... reshaped row-major
            _read(self)
            with self.lock:
                shapes = [t.shape for t in self.src_transforms if isinstance(t, ReshapeTransform)]
                if shapes and self.data is not None:
                    self.data = self.data.reshape(shapes[-1])

        interpret_utils.to_range = to_range
        ipc.DMA.execute_read = execute_read

        def tokens(x):  # each element's token: packed [rows, 4, lanes] (row 4 r + k), native [tokens, lanes]
            if x.ndim == 3:
                return (jax.lax.broadcasted_iota(jnp.int32, x.shape, 0) * x.shape[1]
                        + jax.lax.broadcasted_iota(jnp.int32, x.shape, 1))
            return jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)

        _native_pages = mla_dcp.DcpAttention._pages

        def _pages(self, slot):
            if self.native:
                return _native_pages(self, slot)

            def load(lanes, tokens=None):
                x = self.ws.pages[slot]
                x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
                if hasattr(lanes, "size"):  # pl.ds
                    return jax.lax.dynamic_slice_in_dim(x, lanes.start, lanes.size, 1)
                return x[:, lanes]
            return load

        def _clean_step(self, slot, row, local_step, tokens_=None, end=None, pad=True):
            x = self.ws.pages[slot]
            cut = 0
            for k in range(self.span):
                first = ((local_step * self.span + k) * self.group + self.owner) * self.page_tokens
                cut = cut + jnp.clip(self._limit(row) - first, 0, self.page_tokens)
            lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, x.ndim - 1)
            keep = (tokens(x) < cut) & ((lane < 576) if pad else True)
            self.ws.pages[slot] = jnp.where(keep, x, jnp.zeros((), x.dtype))

        def _zero_step_pad_lanes(self, slot):
            x = self.ws.pages[slot]
            lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, x.ndim - 1)
            self.ws.pages[slot] = jnp.where(lane < 576, x, jnp.zeros((), x.dtype))

        def _zero_slot(self, slot):
            self.ws.pages[slot] = jnp.zeros(self.ws.pages.shape[1:], self.ws.pages.dtype)

        _pack = mla_dcp.DcpAttention._pack_partials

        def _pack_partials(self, first, end, base=0):  # also keep each chunk's packets (every owner's partials)
            _pack(self, first, end, base)
            capture, counter = self._capture
            capture[counter[0]] = self.ws.send[...].astype(jnp.float32)
            counter[0] += 1

        mla_dcp.DcpAttention._pack_partials = _pack_partials
        mla_dcp.DcpAttention._pages = _pages
        mla_dcp.DcpAttention._clean_step = _clean_step
        mla_dcp.DcpAttention._zero_slot = _zero_slot
        mla_dcp.DcpAttention._zero_step_pad_lanes = _zero_step_pad_lanes

        def run(firsts, counts, seed, S, pairs, H, native, trims):
            # Outputs [G, B, H, 512] and every rank's packets of each chunk [G, chunks, G, H S, 640]. Six heads:
            # attention DP2's draft (one unchunked sequence of the DCP group, no host combine).
            rng = np.random.default_rng(seed)
            B = len(firsts) * S
            hier = H == 3
            positions = np.array([p0 + r for p0 in firsts for r in range(S)], np.int32)
            plan = mla_dcp.page_plan(B, H, S, FP8, force_chunks=hier)
            persistent = mla_dcp.persistent_scratch(B, G, H, cache_dtype=FP8, sequence_rows=S, fp8_wire=True,
                                                    force_chunks=hier, chunk_rows=S)
            scoped = mla_dcp.scoped_scratch(B, plan["span"], plan["buffers"], heads=H, group=G, cache_dtype=FP8,
                                            page_tokens=T, sequence_rows=S, force_chunks=hier, chunk_rows=S,
                                            native=True if native else None)
            assert (scoped[1].shape[1:] == (plan["span"] * T, LANES)) == native
            ring, hosts = persistent[0].shape[0], G // mla_dcp.HOST_LANES
            stage2 = (pltpu.VMEM((ring, hosts, H * S, LANES), jnp.bfloat16),
                      pltpu.VMEM((hosts, H * S, LANES), jnp.bfloat16), pltpu.SemaphoreType.DMA((2, ring)))
            stage2 = stage2 if hier else ()
            extra = (pltpu.SemaphoreType.DMA((2, ring)),) if pairs else ()
            pages = max(-(-(int(p0) + T) // T) // G for p0 in firsts) + 1
            # Stale bytes are NaN (e4m3fn 0x7f): tokens past a request's and lanes 576:640.
            pool = np.full((G, 1 + len(firsts) * pages, 128, 4, LANES), 0x7F, np.uint8)
            pool = pool.view(ml_dtypes.float8_e4m3fn)
            tables = np.zeros((G, B, pages), np.int32)
            for b in range(B):
                s_, p0 = b // S, firsts[b // S]
                tables[:, b] = 1 + s_ * pages + np.arange(pages)
                if b % S == 0:
                    t = np.arange(p0)
                    g, j = (t // T) % G, (t // T) // G
                    lat = (rng.standard_normal((p0, 576)) * 0.5).astype(ml_dtypes.float8_e4m3fn)
                    pool[g, tables[g, b, j], (t % T) // 4, t % 4, :576] = lat
            q = (rng.standard_normal((G, B, H, LANES)) * 0.08).astype(np.float32)
            q[..., 576:] = 0
            keys = (rng.standard_normal((len(firsts), 2 * S, LANES)) * 0.5).astype(np.float32)
            keys[..., 576:] = 0
            lane = np.arange(2 * S)
            valid = np.stack([((lane < S) & (lane <= c)) | ((lane >= S) & (lane < 2 * S - 1)) for c in counts])
            valid = valid[:, None, :].astype(np.int32)
            n_p, n_s = len(persistent), len(scoped)
            rows = persistent[1].shape[1]

            def kernel(pos, table, q_r, keys_r, valid_r, pool_r, out_r, send_r, *scratch):
                counter = scratch[-1]
                counter[0] = 0
                scratch = scratch[:-1]
                cache = mla_dcp.DcpCache((pool_r,), table, pages, 0, group=G, global_pages=False)
                ws = mla_dcp.DcpWorkspace(*scratch[:n_p], *scratch[n_p:n_p + n_s])
                att = mla_dcp.DcpAttention(cache, ws, pos, B, sequence_rows=S, heads=H, chunk_rows=S)
                att.lse_in_padding = True
                att.group_fold = True
                att.step_trims = trims
                if hier:
                    att.force_chunked = True
                    att.host_combine = tuple(scratch[n_p + n_s : n_p + n_s + 3])
                    att.host_pairs = scratch[n_p + n_s + 3] if pairs else None
                assert att.native == native and att._trimmed_steps() == trims
                att._capture = (send_r, counter)
                att.prime()
                out_r[...] = att.attend(q_r[...], jnp.zeros((B, LANES), jnp.float32), 0, keys=keys_r,
                                        key_valid=valid_r)

            def body(pos, table, q_, keys_, valid_, pool_):
                vmem = pltpu.VMEM
                out, send = pl.pallas_call(kernel,
                    out_shape=(jax.ShapeDtypeStruct((B, H, 512), jnp.float32),
                               jax.ShapeDtypeStruct((B // S if hier else 1, G, rows, LANES), jnp.float32)),
                    grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=2,
                        in_specs=[pl.BlockSpec(memory_space=vmem)] * 3 + [pl.BlockSpec(memory_space=pl.ANY)],
                        out_specs=(pl.BlockSpec(memory_space=vmem),) * 2,
                        scratch_shapes=[*persistent, *scoped, *stage2, *extra, pltpu.SMEM((1,), jnp.int32)]),
                    compiler_params=pltpu.CompilerParams(has_side_effects=True),
                    interpret=pltpu.InterpretParams())(pos, table[0], q_[0], keys_, valid_, pool_[0])
                return out[None], send[None]

            mesh = Mesh(np.asarray(jax.devices()), ("tp",))
            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P("tp"), P("tp"), P(), P(), P("tp")),
                                      out_specs=(P("tp"), P("tp")), check_vma=False))
            out, send = f(positions, tables.reshape(G, -1), q, keys, valid, pool)
            return np.asarray(out), np.asarray(send)

        t0 = time.time()
        ok = True
        for firsts, counts, seed, S, pairs, H, variants in (
            # 79 pages: 5 on ranks 0-14 (3 steps, the last re-reading its page), 4 on rank 15; page 78 holds 64
            ([40000], [7], 1, 8, False, 3, ((True, True), (False, True), (True, False))),
            # 51 pages (3-4 a rank), then 6 (one step on ranks 0-5, the sixth partial; none on 6-15)
            ([26000, 3000], [3, 6], 2, 8, True, 3, ((True, True),)),
            ([33000, 12000], [2, 3], 3, 4, False, 3, ((True, True),)),
            ([40000], [7], 4, 8, False, 6, ((True, True),)),
        ):
            off, off_send = run(firsts, counts, seed, S, pairs, H, False, False)
            good = bool(np.isfinite(off).all() and np.isfinite(off_send).all())
            for native, trims in variants:
                on, on_send = run(firsts, counts, seed, S, pairs, H, native, trims)
                same = (np.array_equal(on.view(np.uint32), off.view(np.uint32)),
                        np.array_equal(on_send.view(np.uint32), off_send.view(np.uint32)))
                print(f"firsts {firsts} rows {S} heads {H} pairs {pairs} native {native} trims {trims}: outputs, "
                      f"partials identical {same} ({time.time() - t0:.0f}s)", flush=True)
                good &= all(same)
            ok &= good
        print("OK" if ok else "FAIL")
    """

    def test_native_pages_and_step_trims_are_bit_identical(self):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        # 16 interpret devices block in callbacks: give the CPU client enough
        # threads (XLA reads NPROC for its pool size).
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), NPROC="128")
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=3600,
        )
        if "SKIP:" in result.stdout:
            self.skipTest(result.stdout.strip().splitlines()[-1])
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )


class TestPrefillIndexRows(CustomTestCase):
    """A prefill window's index keys laid out as pool rows match the decode's
    per-token commit of the same keys, byte for byte."""

    def test_rows_match_the_token_commit(self):
        from unittest import mock

        import jax.numpy as jnp

        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp, mla_dcp

        rng = np.random.default_rng(5)
        tokens, pages = 512, 3
        keys = rng.standard_normal((tokens * pages, 128)).astype(np.float32)
        rows = np.asarray(
            dsa_dcp.index_rows(jnp.asarray(keys).astype(jnp.bfloat16), pages)
        )
        self.assertEqual(rows.shape, (pages, 4 * 33, 4, 128))
        # The same keys, token by token, into blocks 1..3 of one rank's pool
        # (a table whose entry e is global page e: every page on this rank).
        records = dsa_dcp.key_records(jnp.asarray(keys).astype(jnp.bfloat16))[
            :, None, :
        ]
        positions = np.arange(tokens * pages, dtype=np.int32)
        tables = np.tile(np.arange(1, pages + 1, dtype=np.int32), (len(positions), 1))
        pool = (jnp.zeros((pages + 1, 4 * 33, 4, 128), jnp.uint8),)
        with mock.patch.object(mla_dcp, "owner_index", return_value=0):
            (pool,) = dsa_dcp.commit_index_rows(
                pool,
                records,
                jnp.asarray(tables),
                jnp.asarray(positions),
                group=1,
                global_pages=True,
            )
        np.testing.assert_array_equal(np.asarray(pool)[1:], rows)


class TestDecodeIndexer(CustomTestCase):
    """The decode megakernel's indexer (``dsa_decode.DecodeIndexer``) in
    Pallas interpret mode on a 4-rank DCP group (8 index heads a rank): the
    rows' new key records, every rank's keys, the selection and the
    attention steps holding selected tokens match a numpy reference of the
    whole indexer (projections with sglang's native rounding, FP8 queries and
    head weights, scores, top-k), plain and verify rows."""

    SCRIPT = 'import os, sys, time\nG = 4\nos.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={G}"\nos.environ["JAX_PLATFORMS"] = "cpu"\nimport ml_dtypes, numpy as np, jax, jax.numpy as jnp\nfrom jax.experimental import pallas as pl\nfrom jax.experimental.pallas import tpu as pltpu\nfrom jax.sharding import Mesh, PartitionSpec as P\nif os.environ.get("GATHER") == "1":\n    from sglang_tpu.k3_programs.kimi_k3 import config as kc\n    kc.install(kc.KernelConfig(dsa_gather_finish=True))\nelif os.environ.get("ROUNDS"):\n    from sglang_tpu.k3_programs.kimi_k3 import config as kc\n    knobs = dict(kv.split("=") for kv in os.environ.get("KNOBS", "").split(",") if kv)\n    kc.install(kc.KernelConfig(dsa_select_rounds=int(os.environ["ROUNDS"])).replace(**knobs))\nfrom sglang_tpu.k3_programs.kimi_k3 import dsa\nfrom sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp, dsa_decode, mla_dcp\nif os.environ.get("GATHER") == "1":\n    _finish = dsa_dcp.IndexSelect._gather_finish\n    def _traced(self, first_round):\n        pl.debug_print("GATHER at round {}", first_round)\n        return _finish(self, first_round)\n    dsa_dcp.IndexSelect._gather_finish = _traced\nmla_dcp.GROUPS = (G, 8, 16, 32)\ndsa_dcp.ACTIVITY_MIN_STEPS = int(os.environ.get("ACTIVITY_MIN_STEPS", "0"))\nBF = ml_dtypes.bfloat16\nbf = lambda x: np.asarray(x, np.float32).astype(BF).astype(np.float32)\n\nS = int(os.environ["S"]); TOPK = int(os.environ["TOPK"])\nfirsts = [int(x) for x in os.environ["POS"].split(",")]\nH, QL, T, PER = 1024, 1536, 512, 32 // G\npositions = np.array([p0 + r for p0 in firsts for r in range(S)], np.int32)\nB = len(positions)\nrng = np.random.default_rng(0)\npages = max(-(-((p + T) // T) // G) for p in positions) + 1\npages += int(os.environ.get("WIDE", "0"))  # a table wider than the context (as the 1M-context serving tables)\nkey_rows = dsa_dcp.key_rows_for(pages, T)\nhidden = bf(rng.standard_normal((B, H)))\nqlat = bf(rng.standard_normal((B, QL)))\nwqb = bf(rng.standard_normal((QL, 32 * 128)) * QL**-0.5)\nwk = bf(rng.standard_normal((H, 128)) * H**-0.5)\nwproj = np.zeros((H, 128), np.float32); wproj[:, :32] = bf(rng.standard_normal((H, 32)) * H**-0.5)\nknorm = bf(np.stack([1 + 0.1 * rng.standard_normal(128), 0.1 * rng.standard_normal(128)]))\n\n# Reference projections (native rounding).\ndef layer_norm(x, w, b, eps=1e-5):\n    mean = x.mean(-1, keepdims=True); var = (x * x).mean(-1, keepdims=True) - mean * mean\n    return (x - mean) / np.sqrt(var + eps) * w + b\nkey = bf(layer_norm(bf(hidden @ wk), knorm[0], knorm[1]))\nnew_k8, new_e = dsa.quantize_fp8(key)\ngate = bf(bf(bf(hidden @ wproj)[:, :32]) * float(np.float32(BF(32**-0.5))))\nq_full = bf(qlat @ wqb)  # [B, 4096]\nq8 = np.zeros((B, 32, 128), np.float32); w = np.zeros((B, 32), np.float32)\nfor b in range(B):\n    q8[b], e = dsa.quantize_fp8(q_full[b].reshape(32, 128))\n    w[b] = gate[b] * np.exp2(e[:, 0].astype(np.float32)) * 128**-0.5\n\n# Pool: committed tokens\' records (random keys), garbage past them.\npool = rng.integers(0, 256, (G, 1 + len(firsts) * pages, 132, 4, 128), dtype=np.uint8)\ntables = np.zeros((G, B, pages), np.int32)\nseq_k8, seq_e = [], []\nfor s_, p0 in enumerate(firsts):\n    raw = rng.standard_normal((p0, 128)).astype(np.float32)\n    k8, ke = dsa.quantize_fp8(raw)\n    seq_k8.append(k8); seq_e.append(ke)\n    rec = np.asarray(dsa_dcp.key_records(jnp.asarray(raw)))\n    for r in range(S):\n        tables[:, s_ * S + r] = 1 + s_ * pages + np.arange(pages)\n    for t in range(p0):\n        page = t // T; g = page % G\n        blk = tables[g, s_ * S, page // G]\n        (row, sub), (srow, lane) = dsa_dcp.record_place(t % T)\n        pool[g, blk, row, sub] = rec[t, :128]; pool[g, blk, srow, 0, lane] = rec[t, 128]\n\nwqb_ranks = np.stack([wqb[:, g * PER * 128:(g + 1) * PER * 128] for g in range(G)]).astype(BF)\nwks_ranks = np.stack([dsa_decode.key_weight(wk, g, G) for g in range(G)])\nwpt = dsa_decode.gate_weight(wproj)\npersistent = dsa_decode.persistent_scratch(B, G)\nscoped = dsa_decode.scoped_scratch(B, hidden=H, index_block_shape=(132, 4, 128), key_rows=key_rows,\n                                   page_tokens=T, sequence_rows=S,\n                                   page_plan=mla_dcp.page_plan(B, 3, S, jnp.float8_e4m3fn, dsa=True))\nnames = list(scoped)\nSTEP_ROWS = scoped["attend_keys"].shape[2]\nSTEPS = key_rows // STEP_ROWS\n\ndef kernel(pos_ref, table_ref, hidden_ref, qlat_ref, wqb_ref, wks_ref, wpt_ref, knorm_ref, pool_ref, lat_ref,\n           tau_out, ptie_out, own_out, act_out, rec_out, keys_out, *scratch):\n    cache = mla_dcp.DcpCache((lat_ref,), table_ref, pages, 0, group=G, global_pages=False)\n    sc = dict(zip(names, scratch[len(persistent):]))\n    indexer = dsa_decode.DecodeIndexer(\n        weights={"d_wqb": wqb_ref, "d_wks": wks_ref, "d_wpt": wpt_ref, "d_knorm": knorm_ref},\n        pools=(pool_ref,), layer=0, keys=keys_out, records_out=rec_out, persistent=scratch[:len(persistent)],\n        scoped=sc, topk=TOPK, sequence_rows=S)\n    indexer.start()\n    sel = indexer.selection(cache, pos_ref, hidden_ref[...], qlat_ref[...], 0)\n    for b in range(B):\n        tau_out[b] = sel.tau[b]; ptie_out[b] = sel.p_tie[b]; own_out[b] = sel.own[b]\n    for i in range(B * STEPS):\n        act_out[i] = sel.active[i]\n\ndef body(pos, table, hid, ql, wqb_r, wks_r, wpt_r, kn_r, pool_r, lat_r):\n    vmem, smem = pltpu.VMEM, pltpu.SMEM\n    out = pl.pallas_call(kernel,\n        out_shape=(jax.ShapeDtypeStruct((B,), jnp.int32),) * 3 + (jax.ShapeDtypeStruct((B * STEPS,), jnp.int32),\n            \n            jax.ShapeDtypeStruct((B, 24, 2, 128), jnp.int32), jax.ShapeDtypeStruct((B, key_rows, 128), jnp.int32)),\n        grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=2,\n            in_specs=[pl.BlockSpec(memory_space=vmem)] * 2 + [pl.BlockSpec(memory_space=pl.ANY)] * 6,\n            out_specs=[pl.BlockSpec(memory_space=smem)] * 4 + [pl.BlockSpec(memory_space=pl.ANY)] * 2,\n            scratch_shapes=[*persistent, *scoped.values()]),\n        compiler_params=pltpu.CompilerParams(has_side_effects=True),\n        interpret=pltpu.InterpretParams())(pos, table[0], hid, ql, wqb_r[0], wks_r[0], wpt_r, kn_r, pool_r[0], lat_r)\n    return tuple(x[None] for x in out)\n\nmesh = Mesh(np.asarray(jax.devices()), ("tp",))\nf = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P("tp"), P(), P(), P("tp"), P("tp"), P(), P(), P("tp"), P()),\n                          out_specs=(P("tp"),) * 6, check_vma=False))\nt0 = time.time()\ntau, ptie, own, act, rec, keys = (np.asarray(x) for x in f(\n    positions, tables.reshape(G, -1), hidden.astype(BF), qlat.astype(BF), wqb_ranks, wks_ranks,\n    wpt, knorm.astype(BF), pool, jnp.zeros((1, 128, 4, 640), jnp.float8_e4m3fn)))\n# Records: the rows\' new keys (every rank the same).\nwant_rec = np.asarray(dsa_dcp.key_records(jnp.asarray(key)))\ngot_rec = np.concatenate([rec[0, :, 0, 0, :], rec[0, :, 0, 1, :1]], axis=1)\nrec_match = np.mean(got_rec == want_rec)\nprint(f"records match {rec_match:.4f}; identical on ranks {all(np.array_equal(rec[0], rec[g]) for g in range(G))}")\nok = rec_match > 0.99\nunkey = lambda k: (k ^ ((k >> 31) & 0x7FFFFFFF)).astype(np.int32).view(np.float32)\nfor b in range(B):\n    s_, p0, L = b // S, firsts[b // S], positions[b] + 1\n    t = np.arange(L)\n    kk = keys[(t // T) % G, b, (t // T) // G * (T // 128) + (t % T) // 128, t % 128].astype(np.int64)\n    k8 = np.concatenate([seq_k8[s_], new_k8[s_ * S:s_ * S + (b - s_ * S) + 1]])\n    ke = np.concatenate([seq_e[s_], new_e[s_ * S:s_ * S + (b - s_ * S) + 1]])\n    ref = dsa.monotone_key(dsa.scores(q8[b], w[b], k8, ke).astype(np.float32))\n    err = np.abs(unkey(kk) - unkey(ref)).max() / np.abs(unkey(ref)).max()\n    got = dsa_dcp.selected(kk, t, tau[0, b], ptie[0, b])\n    want = dsa.topk_mask(kk, t, TOPK)\n    if os.environ.get("ROUNDS") and L > TOPK:  # bin selection: keys at or above the prefix bin\n        kth = np.sort(kk)[::-1][TOPK - 1]; drop = 32 - 4 * int(os.environ["ROUNDS"])\n        u = (int(kth) ^ -(2**31)) & 0xFFFFFFFF; tb = ((u >> drop) << drop) ^ (1 << 31)\n        want = kk >= (tb - (1 << 32) if tb >= (1 << 31) else tb)\n    ref_sel = dsa.topk_mask(ref, t, TOPK)\n    agree = np.mean(got == ref_sel)\n    uniform = all(len(set(v[:, b])) == 1 for v in (tau, ptie, own))\n    print(f"row {b} L={L} key err {err:.1e} exact {np.array_equal(got, want)} vs-ref-selection {agree:.4f} "\n          f"own {own[0, b]}/{bool(own[0, b]) == bool(got[-1])} uniform {uniform}")\n    ok &= bool(err < 1e-2 and np.array_equal(got, want) and uniform and bool(own[0, b]) == bool(got[-1]) and agree > 0.99)\n# Step activity: per rank, row (its sequence\'s first row), step: any selected token of the sequence\'s rows.\nwant_act = np.zeros_like(act)\nfor b in range(B):\n    L, first = positions[b] + 1, b - b % S\n    t = np.arange(L)\n    kk = keys[(t // T) % G, b, (t // T) // G * (T // 128) + (t % T) // 128, t % 128].astype(np.int64)\n    for i in np.flatnonzero(dsa_dcp.selected(kk, t, tau[0, b], ptie[0, b])):\n        g, j = (i // T) % G, (i // T) // G\n        want_act[g, first * STEPS + (j * (T // 128) + (i % T) // 128) // STEP_ROWS] = 1\nif dsa_dcp.ACTIVITY_MIN_STEPS:\n    want_act = np.ones_like(act)\n# KernelConfig.dsa_live_activity: only the entries the attention reads are written (per rank, each\n# sequence first row: the steps of the longest row and one more); the rest are stale.\ndomain = np.ones_like(act, bool)\nif "dsa_live_activity=1" in os.environ.get("KNOBS", ""):\n    domain[:] = False\n    for g in range(G):\n        longest = max(max(-(-(int(p) + 1) // T) - g + G - 1, 0) // G for p in positions)\n        live = min(-(-longest * (T // 128) // STEP_ROWS) + 1, STEPS)\n        for f in range(0, B, S):\n            domain[g, f * STEPS:f * STEPS + live] = True\nact_ok = np.array_equal(act[domain], want_act[domain])\nprint(f"activity exact {act_ok} active steps {int(act[domain].sum())} of {int(domain.sum())}")\nok &= bool(act_ok)\nprint("OK" if ok else "FAIL")\n'

    def test_indexer_matches_the_reference(self):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        # The scoring knobs (TestIndexSelectScoringKnobs) inside the decode
        # indexer: resident keys (the step activity scans the HBM keys their
        # copies fill), the early plan and page copies under the query gather.
        SCORING = (
            "dsa_stage_slots=4,dsa_page_slots=8,dsa_resident_keys=1,"
            "dsa_patch_loop=1,dsa_early_plan=1"
        )
        LIVE = SCORING + ",dsa_live_activity=1,dsa_odd_stride=1"

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        for case in (
            {"S": "1", "TOPK": "128", "POS": "700,2100"},
            {"S": "4", "TOPK": "200", "POS": "1500,3000"},
            {"S": "4", "TOPK": "200", "POS": "1500,3000", "ACTIVITY_MIN_STEPS": "16"},
            {"S": "4", "TOPK": "200", "POS": "1500,3000", "GATHER": "1"},
            {"S": "4", "TOPK": "200", "POS": "1500,3000", "ROUNDS": "5"},
            {"S": "4", "TOPK": "200", "POS": "1500,3000", "ROUNDS": "5", "KNOBS": SCORING},
            {"S": "8", "TOPK": "400", "POS": "3069", "ROUNDS": "5", "KNOBS": SCORING},
            # KernelConfig.dsa_live_activity: the entries the attention reads,
            # scanned and dense, verify sequences and plain rows, on tables
            # wider than the contexts; with dsa_odd_stride (resident chunks of
            # 48 rows at 56 rows apart, and copied ones).
            {
                "S": "4",
                "TOPK": "200",
                "POS": "1500,3000",
                "ROUNDS": "5",
                "KNOBS": LIVE,
                "WIDE": "8",
            },
            {
                "S": "4",
                "TOPK": "200",
                "POS": "1500,3000",
                "ROUNDS": "5",
                "KNOBS": LIVE,
                "ACTIVITY_MIN_STEPS": "16",
                "WIDE": "8",
            },
            {
                "S": "8",
                "TOPK": "400",
                "POS": "3069",
                "ROUNDS": "5",
                "KNOBS": LIVE,
                "WIDE": "8",
            },
            {
                "S": "1",
                "TOPK": "128",
                "POS": "700,2100",
                "ROUNDS": "5",
                "KNOBS": "dsa_live_activity=1,dsa_odd_stride=1",
                "WIDE": "8",
            },
        ):
            with self.subTest(**case):
                env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), **case)
                result = subprocess.run(
                    [sys.executable, "-c", self.SCRIPT],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=900,
                )
                self.assertIn(
                    "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
                )
                if case.get("GATHER"):  # the gather finish ran (on every rank)
                    self.assertGreaterEqual(
                        result.stdout.count("GATHER at round"), 4, result.stdout[-2000:]
                    )


class TestRouterBothHalves(CustomTestCase):
    """The decode MoE router with both router halves on every rank
    (``KernelConfig.decode_router_both_halves``) in Pallas interpret mode on a
    rank pair of host devices: on both lanes the raw score rows assembled from
    the own and peer half projections, and the final score rows, selected
    experts, probabilities, route counts and probability totals, are
    bit-identical to the pair's score-half exchange, with stale NaN bytes in
    the router's padding columns."""

    SCRIPT = """
        import dataclasses, os
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import ml_dtypes, numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk

        H, E, K = 7168, 896, 16
        FIELDS = [f.name for f in dataclasses.fields(dmk.MixtureOfExpertsWeights)]
        mesh = Mesh(np.asarray(jax.devices()), ("tp",))

        def run(slots, hidden, router, bias, live):
            B = hidden.shape[0]

            def kernel(live_ref, hidden_ref, bias_ref, router_ref, raw_out, scores_out,
                       experts_out, probabilities_out, counts_out, totals_out,
                       router_vmem, router_semaphores, scores_vmem, sends, receives):
                weights = dmk.MixtureOfExpertsWeights(**dict(
                    dict.fromkeys(FIELDS), routing_projection=router_ref,
                    routing_projection_vmem=router_vmem,
                    routing_copy_semaphores=router_semaphores, routing_bias=bias_ref))
                rank = jax.lax.axis_index("tp")
                wait_half = weights.wait_for_routing_projection_half if slots == 2 else None

                def load_router():
                    # Stale arena bytes (NaN) under every router column, then the
                    # arena request's copies; one slot is waited for up front.
                    router_vmem[...] = jnp.full(router_vmem.shape, jnp.nan, jnp.bfloat16)
                    scores_vmem[...] = jnp.full(scores_vmem.shape, jnp.nan, jnp.float32)
                    dmk._router_copy(router_ref, router_vmem, router_semaphores, False,
                                     initialize_padding=False)
                    if slots == 1:
                        weights.wait_for_routing_projection_copy()

                load_router()
                dmk._project_routing_scores(rank, hidden_ref[...], router_vmem, scores_vmem,
                                            sends.at[0], receives.at[0],
                                            wait_for_projection_half=wait_half)
                raw_out[...] = scores_vmem[...]
                load_router()
                live_rows = (jnp.stack([live_ref[b] for b in range(B)]) != 0) if live else None
                experts, probabilities, counts, totals = dmk._route_hidden_states(
                    hidden_ref[...], router_vmem, bias_ref, scores_vmem, sends.at[0],
                    receives.at[0], selected_expert_count=K, local_expert_count=112,
                    expert_tensor_parallel=1, on_local_selection=lambda *unused: None,
                    after_selection=lambda unused: None, live_rows=live_rows,
                    wait_for_projection_half=wait_half)
                scores_out[...] = scores_vmem[...]
                experts_out[...] = experts
                probabilities_out[...] = probabilities
                counts_out[...] = jnp.broadcast_to(counts[:, None], counts_out.shape)
                totals_out[...] = jnp.broadcast_to(totals[:, None], totals_out.shape)

            vmem = pltpu.VMEM
            shapes = ((B, 1024, jnp.float32), (B, 1024, jnp.float32), (B, K, jnp.int32),
                      (B, K, jnp.float32), (B, 128, jnp.int32), (B, 128, jnp.float32))

            def body(live_rows, hidden, bias, router):
                outputs = pl.pallas_call(
                    kernel,
                    out_shape=tuple(jax.ShapeDtypeStruct(s[:2], s[2]) for s in shapes),
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=1,
                        in_specs=[pl.BlockSpec(memory_space=vmem)] * 2
                        + [pl.BlockSpec(memory_space=pl.ANY)],
                        out_specs=[pl.BlockSpec(memory_space=vmem)] * len(shapes),
                        scratch_shapes=[vmem((slots, H, 512), jnp.bfloat16),
                                        pltpu.SemaphoreType.DMA((2,)),
                                        vmem((B, 1024), jnp.float32),
                                        pltpu.SemaphoreType.DMA((1,)),
                                        pltpu.SemaphoreType.DMA((1,))]),
                    compiler_params=pltpu.CompilerParams(has_side_effects=True),
                    interpret=pltpu.InterpretParams(),
                )(live_rows, hidden, bias, router)
                return tuple(x[None] for x in outputs)

            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(),) * 4,
                                      out_specs=(P("tp"),) * len(shapes), check_vma=False))
            live_rows = (np.arange(B) % 3 != 1) if live else np.ones(B, bool)
            return [np.asarray(x) for x in f(live_rows.astype(np.int32), hidden, bias, router)]

        rng = np.random.default_rng(0)
        for B, live in ((1, False), (8, False), (16, True)):
            hidden = rng.standard_normal((B, H)).astype(ml_dtypes.bfloat16)
            router = (rng.standard_normal((H, E)) * H**-0.5).astype(ml_dtypes.bfloat16)
            bias = (0.1 * rng.standard_normal((1, E))).astype(np.float32)
            exchanged = run(1, hidden, router, bias, live)
            projected = run(2, hidden, router, bias, live)
            names = ("raw scores", "final scores", "experts", "probabilities", "counts",
                     "totals")
            for name, old, new in zip(names, exchanged, projected):
                assert old.tobytes() == new.tobytes(), (B, name)
            raw = exchanged[0]
            assert np.array_equal(raw[0], raw[1]), B  # both lanes hold the same rows
            assert not np.isnan(raw).any() and not raw[..., 896:].any(), B
            want = hidden.astype(np.float32) @ router.astype(np.float32)
            assert np.abs(raw[0, :, :896] - want).max() < 1e-3 * np.abs(want).max(), B
            assert exchanged[4][..., 0].sum(axis=1).all(), B  # both lanes own routes
        print("OK")
    """

    def test_both_halves_match_the_score_exchange(self):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )


class TestManyOperandOps(CustomTestCase):
    """Kimi-K3.1's decode steps take more operands than a torch op schema holds (64):
    neighbouring operands with one spec and donation become one ``Tensor[]`` argument."""

    SCRIPT = r"""
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import jax, numpy as np, torch
from jax.sharding import Mesh, PartitionSpec as P
from sglang_tpu.k3_programs import bridge

mesh = Mesh(np.array(jax.devices("cpu")[:4]), ("x",))
W, POOLS = 40, 50
def body(*args):
    acc = sum(a.sum() for a in args[:W])
    return (args[-1] + acc, *(p + 1 for p in args[W:W + POOLS]))
in_specs = (P("x"),) * (W + POOLS) + (P(),)
donated = tuple(range(W, W + POOLS))
op = bridge.register_op("pallas::k31_runs_check", mesh, body, in_specs,
                        (P(),) + (P("x"),) * POOLS, donate_argnums=donated)
assert op.runs == [(0, W), (W, W + POOLS), (W + POOLS, W + POOLS + 1)], op.runs
assert str(op.op._schema).startswith("(Tensor[] a0, Tensor[] a1, Tensor a2)"), op.op._schema
init = op._init_fn
assert init.donate_argnums in ([1], (1,)), init.donate_argnums
flat = [torch.zeros(2, 8)] * W + [torch.zeros(3, 8)] * POOLS + [torch.zeros(5)]
grouped = [flat[s] if e - s == 1 else list(flat[s:e]) for s, e in op.runs]
args, kwargs = init._get_jax_placeholders(*init._normalize_invocation_args(grouped, {}))
with jax._src.config.export_ignore_forward_compatibility(True):
    lowered = init.exported(*args, **kwargs)
assert len(lowered.in_avals) == W + POOLS + 1 and lowered.in_avals[0].shape == (8, 8)
assert len(lowered.out_avals) == POOLS + 1 and lowered.out_avals[1].shape == (12, 8)
small = bridge.register_op("pallas::k31_runs_small", mesh, lambda a, b: (a + b,),
                           (P("x"), P("x")), (P("x"),))
assert not isinstance(small, bridge._RunOp)
print("OK")
"""

    def test_operand_runs(self):
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.k3_programs import bridge

        specs = (P("x"),) * 3 + (P(),)
        self.assertEqual(
            bridge._operand_runs(specs, (1,)), [(0, 1), (1, 2), (2, 3), (3, 4)]
        )
        many = (P("x"),) * 70 + (P(),)
        self.assertEqual(
            bridge._operand_runs(many, tuple(range(40, 70))),
            [(0, 40), (40, 70), (70, 71)],
        )

    def test_registers_and_traces_with_flat_operands(self):
        import os
        import subprocess
        import sys

        env = dict(
            os.environ, PYTHONPATH=os.pathsep.join(sys.path), JAX_PLATFORMS="cpu"
        )
        result = subprocess.run(
            [sys.executable, "-c", self.SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )


class TestK31Draft(CustomTestCase):
    """Kimi-K3.1's DSpark draft (``Inferact/Kimi-K3.1-Pro-DSpark-preview``)
    in the MLA draft's config: three heads a rank, causal block rows, YaRN's
    attention factor (the rotary scale is then exactly one), its aux layers;
    the reference attention's block rows see only earlier block rows."""

    CONFIG = {
        "model_type": "k3_dspark",
        "hidden_size": 7168,
        "num_hidden_layers": 5,
        "num_attention_heads": 96,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "intermediate_size": 14336,
        "vocab_size": 163840,
        "rms_norm_eps": 1e-5,
        "target_layer_ids": [75, 79, 83, 87, 91],
        "mask_token_id": 163839,
        "markov_rank": 512,
        "block_size": 7,
        "rope_interleave": True,
        "full_attention_causal": True,
        "dflash_config": {"causal": True},
        "rope_parameters": {
            "rope_type": "yarn",
            "rope_theta": 1000000.0,
            "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 0.0,
            "attn_factor": 0.7426255848312643,
        },
    }

    def test_config_and_causal_reference(self):
        import dataclasses

        import jax.numpy as jnp

        from sglang_tpu.k3_programs.kimi_k3.kernels import dspark_mla as mla

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text(json.dumps(self.CONFIG))
            c = mla.MlaDraftConfig.from_checkpoint(Path(d))
        self.assertEqual(
            (c.heads_per_rank, c.causal, c.markov_rank, c.mask_token),
            (3, True, 512, 163839),
        )
        self.assertEqual(c.aux_layers, (76, 80, 84, 88, 92))
        self.assertAlmostEqual(mla.rope_mscale(c), 1.0, places=6)
        self.assertAlmostEqual(c.softmax_scale, 192**-0.5, places=9)
        small = dataclasses.replace(
            c,
            hidden=256,
            heads=4,
            q_lora=128,
            kv_lora=64,
            nope=32,
            rope=16,
            v_head=32,
            intermediate=128,
            vocab=512,
            layers=1,
        )
        rng = np.random.default_rng(0)
        w = {
            k: jnp.asarray(rng.standard_normal(s).astype(np.float32) * 0.05)
            for k, s in mla.weight_shapes(small).items()
        }
        h = jnp.asarray(rng.standard_normal((4, small.hidden)).astype(np.float32))
        positions, context = (
            jnp.arange(10, 14),
            jnp.asarray(rng.standard_normal((6, small.latent)).astype(np.float32)),
        )
        valid = jnp.ones((6,), bool)
        causal = mla.reference_attention(w, small, 0, h, positions, context, valid)
        full = mla.reference_attention(
            w, dataclasses.replace(small, causal=False), 0, h, positions, context, valid
        )
        alone = mla.reference_attention(
            w, small, 0, h[:1], positions[:1], context, valid
        )
        np.testing.assert_allclose(
            causal[-1], full[-1], atol=1e-5
        )  # the last row sees every block row
        np.testing.assert_allclose(
            causal[0], alone[0], atol=1e-5
        )  # the first sees only itself
        self.assertFalse(np.allclose(causal[0], full[0], atol=1e-5))


class TestExpertSchedule(CustomTestCase):
    """The grouped routed-expert stream (``_run_routed_expert_stream``) at B8
    and B16 in Pallas interpret mode, on random MXFP4 experts and routings
    (single-row and multi-row experts, more rows than a compact tile, odd
    expert counts, empty rows, expert 0, no experts): reading each pair from
    the SMEM schedule (``KernelConfig.expert_schedule``) gives bit-identical
    outputs to reducing the pending routes every iteration."""

    SCRIPT = """
        import os
        os.environ["JAX_PLATFORMS"] = "cpu"
        from types import SimpleNamespace
        import numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk

        E, K, IN, WIDTH, OUT = 112, 16, 512, 768, 128

        def stream(B, scheduled):
            rows = dmk._expert_accumulator_rows(B, jnp.uint32, group_pending_routes=True)
            compact = dmk._compact_rows(B)

            def kernel(sel_ref, prob_ref, count_ref, total_ref, source_ref, gu_hbm, gus_hbm,
                       down_hbm, ds_hbm, out_ref, gu, gus, down, ds, semaphores, gate_up_acc,
                       down_acc, processed, first_compact, second_compact, table, table_semaphore):
                selected, counts = sel_ref[:, :K], count_ref[:, 0]

                def copy_expert(expert, slot, *, gate_up=True, down=True):
                    def copy(hbm, vmem, kind, wanted):
                        return (pltpu.make_async_copy(hbm.at[expert], vmem.at[slot],
                                                      semaphores.at[slot, kind]) if wanted else None)
                    return SimpleNamespace(
                        gate_up=copy(gu_hbm, gu, 0, gate_up), gate_up_scales=copy(gus_hbm, gus, 1, gate_up),
                        down=copy(down_hbm, down_slots, 2, down), down_scales=copy(ds_hbm, ds, 3, down))

                down_slots = down
                dmk._start_initial_expert_weight_copies(
                    selected, counts, copy_expert, slot_count=4, already_started=jnp.int32(0),
                    group_pending_routes=True)
                schedule = None
                if scheduled:
                    schedule = dmk._expert_schedule(processed.at[:8], table, table_semaphore)
                    dmk._start_expert_schedule(schedule, selected, counts)
                dmk._run_routed_expert_stream(
                    source_ref, selected, prob_ref[:, :K], counts, total_ref[:, 0], gu, gus, down_slots,
                    ds, out_ref, gate_up_acc, down_acc, processed, first_compact, second_compact,
                    copy_expert, expert_input_size=IN, expert_width=WIDTH, expert_output_size=OUT,
                    group_pending_routes=True, expert_schedule=schedule)

            sources = ((pltpu.VMEM((compact, IN), jnp.float32),) * 2 if compact
                       else (pltpu.VMEM((1, IN), jnp.bfloat16),) * 2)
            return jax.jit(pl.pallas_call(
                kernel, out_shape=jax.ShapeDtypeStruct((B, OUT), jnp.float32),
                in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 5
                + [pl.BlockSpec(memory_space=pl.ANY)] * 4,
                out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                scratch_shapes=[
                    pltpu.VMEM((4, IN // 8, 2 * WIDTH), jnp.uint32),
                    pltpu.VMEM((4, IN // 32, 2 * WIDTH), jnp.uint8),
                    pltpu.VMEM((4, WIDTH // 8, OUT), jnp.uint32),
                    pltpu.VMEM((4, WIDTH // 32, OUT), jnp.uint8),
                    pltpu.SemaphoreType.DMA((4, 4)),
                    pltpu.VMEM((rows[0], 4 * WIDTH), jnp.float32),
                    pltpu.VMEM((rows[1], OUT), jnp.float32),
                    pltpu.VMEM((B, 128), jnp.int32), *sources,
                    pltpu.SMEM((8, 128), jnp.int32), pltpu.SemaphoreType.DMA],
                interpret=pltpu.InterpretParams()))

        rng = np.random.default_rng(0)
        weights = (rng.integers(0, 2**32, (E, IN // 8, 2 * WIDTH), dtype=np.uint32),
                   rng.integers(122, 128, (E, IN // 32, 2 * WIDTH), dtype=np.uint8),
                   rng.integers(0, 2**32, (E, WIDTH // 8, OUT), dtype=np.uint32),
                   rng.integers(122, 128, (E, WIDTH // 32, OUT), dtype=np.uint8))

        def routing(B, seed, pool, most):
            # Rows pick up to `most` distinct experts of a pool holding expert 0
            # (row 0 the most, others 0..most); slots past a row's count hold
            # expert 0, as after routing.
            r = np.random.default_rng(seed)
            experts = r.choice(E, pool, replace=False)
            experts[0] = 0
            sel = np.zeros((B, 128), np.int32)
            prob = np.zeros((B, 128), np.float32)
            cnt = np.zeros((B, 128), np.int32)
            for b in range(B):
                n = int(r.integers(0, min(most, pool) + 1)) if b else min(most, pool)
                sel[b, :n] = r.choice(experts, n, replace=False)
                prob[b, :n] = r.uniform(0.01, 1.0, n)
                cnt[b] = n
            total = np.broadcast_to(prob.sum(1, keepdims=True) + r.uniform(0.5, 2.0, (B, 1)), (B, 128))
            source = r.standard_normal((B, IN)).astype(jnp.bfloat16)
            return sel, prob, cnt, np.ascontiguousarray(total, np.float32), source

        CASES = {8: [(1, 112, 4), (3, 9, 5), (4, 1, 1), (8, 5, 0)],
                 16: [(11, 112, 3), (12, 12, 12), (13, 20, 6)]}
        for B, cases in CASES.items():
            old, new = stream(B, False), stream(B, True)
            for seed, pool, most in cases:
                sel, prob, cnt, total, source = routing(B, seed, pool, most)
                a = np.asarray(old(sel, prob, cnt, total, source, *weights))
                b = np.asarray(new(sel, prob, cnt, total, source, *weights))
                used = {int(sel[r, s]) for r in range(B) for s in range(cnt[r, 0])}
                rows_per_expert = [sum(e in sel[r, :cnt[r, 0]] for r in range(B)) for e in sorted(used)]
                assert np.isfinite(a).all() and (np.abs(a).sum(1) > 0).sum() == (cnt[:, 0] > 0).sum()
                assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), (B, seed)
                print(f"B{B} seed {seed}: rows per expert {rows_per_expert} bit-identical")
        print("OK")
    """

    def test_schedule_matches_the_pending_route_loop(self):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])


class TestK31DraftWholeWeights(CustomTestCase):
    """``KernelConfig.draft_whole_weights``: Kimi-K3.1's draft (three heads a
    rank) holds o whole at up to 8 rows and down at up to 16 instead of
    streaming them one 896-column chunk at a time (other drafts and shapes as
    before), and the chunks ``all_reduce`` multiplies
    (``dspark_mla_kernel._ChunkedWeight``) give bit-identical products either
    way (Pallas interpret mode)."""

    SCRIPT = """
        import os
        os.environ["JAX_PLATFORMS"] = "cpu"
        import numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from sglang_tpu.k3_programs.kimi_k3.kernels import dspark_mla as mla
        from sglang_tpu.k3_programs.kimi_k3.kernels import dspark_mla_kernel as kern

        for heads in (64, 96):
            c = mla.MlaDraftConfig(heads=heads)
            for rows in (4, 8, 12, 16, 24, 32, 48, 64):
                for dp in (1, 2):
                    old = rows > 32 or heads > 64  # before the knob: down; o also under DP
                    assert kern.streamed_weights(c, rows, dp) == (old or dp > 1, old)
                    if heads > 64 and dp == 1:
                        want = (rows > 8, rows > 16)
                    else:
                        want = (old or dp > 1, old)
                    assert kern.streamed_weights(c, rows, dp, whole=True) == want, (heads, rows, dp)

        LAYERS, CHUNK = 2, kern.CHUNK
        def products(lhs, weight, streamed):
            T, K = lhs.shape
            def kernel(lhs_ref, weight_hbm, out_ref, buffer, sems):
                copy = lambda src, dst, sem: pltpu.make_async_copy(src, dst, sems.at[sem])
                chunked = kern._ChunkedWeight(weight_hbm, buffer, (0, 1), copy)
                assert chunked.streamed == streamed
                for layer in range(LAYERS):
                    chunked.load(layer).wait()
                    for k, operand in enumerate(chunked.chunks(layer)):
                        out_ref[layer, k] = kern._dot(lhs_ref[...], operand)
            buffer = (2, K, CHUNK) if streamed else (K, 8 * CHUNK)
            return np.asarray(pl.pallas_call(
                kernel,
                out_shape=jax.ShapeDtypeStruct((LAYERS, 8, T, CHUNK), jnp.float32),
                in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM), pl.BlockSpec(memory_space=pl.ANY)],
                out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                scratch_shapes=[pltpu.VMEM(buffer, jnp.bfloat16), pltpu.SemaphoreType.DMA((2,))],
                interpret=pltpu.InterpretParams(),
            )(lhs, weight))

        rng = np.random.default_rng(0)
        for T, K in ((8, 384), (16, 512)):  # o (three heads) at 8 rows, down at 16
            lhs = jnp.asarray(rng.standard_normal((T, K)), jnp.bfloat16)
            weight = jnp.asarray(rng.standard_normal((LAYERS, K, 8 * CHUNK)), jnp.bfloat16)
            streamed, whole = products(lhs, weight, True), products(lhs, weight, False)
            assert np.array_equal(streamed.view(np.uint32), whole.view(np.uint32)), (T, K)
            ref = np.asarray(lhs, np.float32) @ np.asarray(weight, np.float32)
            ref = ref.reshape(LAYERS, T, 8, CHUNK).transpose(0, 2, 1, 3)
            np.testing.assert_allclose(whole, ref, rtol=1e-4, atol=1e-3)
        print("OK")
    """

    def test_whole_and_streamed_chunks_match(self):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])


class TestK31DraftFusedEpilogue(CustomTestCase):
    """``KernelConfig.draft_fused_epilogue`` and ``draft_lm_prefetch`` in
    Pallas interpret mode, against the XLA code they replace, bitwise:

    * ``paged.commit_draft_rows`` (one tile read / write per touched tile)
      lands the bytes of ``commit_latent_rows``'s scatter on a 4-rank group:
      rows within one tile, across tiles, across a page (two owners), into
      the next table entry, a negative block, a dropped block and entries
      past the table.
    * ``dspark_mla_kernel.single_epilogue`` gives ``_build_single``'s emitted
      tokens, next ids and control state, slot indices that wrap or drop
      included, and (``draft_verify_packet``) the packing op's next verify
      operand, bank writes that land in the plan row included.
    * ``dspark_mla_kernel.stream_chunks``: the LM head through six slots
      (four started early) gives the two-slot ring's logits, race-free.
    * ``MlaDsparkDraft._build_single``'s op with the knob (the fused kernel
      stubbed) returns the knob-off op's vector, pools, emitted tokens, next
      ids and control state; with ``draft_verify_packet`` (either epilogue)
      also ``DecodeOps.pack_verify_metadata`` of its next ids, control state
      and verify mode 1."""

    COMMIT = """
        import os
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import dataclasses
        import numpy as np, ml_dtypes, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3.kernels import paged

        G, ROWS, PAGES, L, BLOCKS, PT = 4, 8, 6, 5, 7, 512
        base_config = paged._config()
        scatter_config = dataclasses.replace(base_config, fused_cache_commit=False)
        stock = pl.pallas_call
        paged.pl.pallas_call = lambda *a, **k: stock(*a, **{"interpret": pltpu.InterpretParams(), **k})
        mesh = Mesh(np.asarray(jax.devices()), ("tp",))

        def new(pools, latents, ids):
            return paged.commit_draft_rows(tuple(pools), latents.astype(jnp.float8_e4m3fn), ids, ROWS, PAGES,
                                           tuple(range(L)), group=G, global_pages=False)

        def old(pools, latents, ids):
            paged._config = lambda: scatter_config
            try:
                table = ids[3 * ROWS + 1 : 3 * ROWS + 1 + PAGES]
                positions = ids[ROWS] + jnp.arange(ROWS, dtype=jnp.int32)
                return paged.commit_latent_rows(tuple(pools), latents.transpose(1, 0, 2),
                                                jnp.broadcast_to(table[None], (ROWS, PAGES)), positions,
                                                tuple(range(L)), group=G, global_pages=False)
            finally:
                paged._config = lambda: base_config

        def run(base, table, seed):
            rng = np.random.default_rng(seed)
            pools = (rng.integers(0, 256, (L, G * BLOCKS, 128, 4, 640), dtype=np.uint8) & 0x7E).view(
                ml_dtypes.float8_e4m3fn)
            latents = (rng.standard_normal((L, ROWS, 640)) * 2).astype(ml_dtypes.bfloat16)
            ids = rng.integers(0, 100, 3 * ROWS + 1 + ROWS * PAGES).astype(np.int32)
            ids[ROWS] = base
            ids[3 * ROWS + 1 : 3 * ROWS + 1 + PAGES] = table
            outs = []
            for fn in (new, old):
                f = jax.jit(jax.shard_map(lambda *a, fn=fn: fn(a[:L], *a[L:]), mesh=mesh,
                                          in_specs=(P("tp"),) * L + (P(), P()), out_specs=(P("tp"),) * L,
                                          check_vma=False))
                outs.append([np.asarray(x).view(np.uint8) for x in f(*map(jnp.asarray, pools),
                                                                    jnp.asarray(latents), jnp.asarray(ids))])
            assert all(np.array_equal(a, b) for a, b in zip(*outs)), (base, table)
            return sum(int((a != p.view(np.uint8)).sum()) for a, p in zip(outs[1], pools))

        written = [
            run(5, [1, 2, 3, 4, 5, 6], 0),  # one tile
            run(28, [3, 1, 2, 4, 5, 6], 1),  # two tiles
            run(PT - 3, [2, 5, 1, 1, 1, 1], 2),  # two pages, two owners
            run(4 * PT - 4, [1, 6, 2, 3, 4, 5], 3),  # the next table entry
            run(3 * PT + 100, [-1, 2, 3, 4, 5, 6], 4),  # a negative block
            run(30 * PT + 500, [1, 2, 3, 4, 5, 6], 6),  # entries past the table (clamped)
        ]
        assert min(written) > 0
        assert run(2 * PT + 7, [G * BLOCKS + 3, 2, 3, 1, 1, 1], 5) == 0  # out of range: dropped
        print("OK")
    """

    EPILOGUE = """
        import itertools, os
        os.environ["JAX_PLATFORMS"] = "cpu"
        import numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from sglang_tpu.k3_programs.kimi_k3 import config as kc
        kc.install(kc.KernelConfig(draft_epilogue_unroll=int(os.environ.get("UNROLL", "0"))))
        from sglang_tpu.k3_programs.kimi_k3.kernels import dspark_mla_kernel as kern
        from sglang_tpu.k3_programs.kimi_k3.kernels.dspark import OUT_BONUS, OUT_COUNT, OUT_DRAFTS

        ROWS, FIELDS = 8, 8

        def reference(vector, ids, state, slots, reset):  # MlaDsparkDraft._build_single's XLA epilogue
            rows = ROWS
            tokens, base, slot = ids[:rows], ids[rows], ids[2 * rows]
            accepted, bonus = vector[0, OUT_COUNT], vector[0, OUT_BONUS]
            row = jnp.arange(rows, dtype=jnp.int32)
            emitted = jnp.where(row < accepted, tokens[jnp.minimum(row + 1, rows - 1)],
                                jnp.where(row == accepted, bonus, -1)).astype(jnp.int32)
            following = jnp.concatenate((bonus[None], vector[0, OUT_DRAFTS : OUT_DRAFTS + rows - 1]))
            following_ids = jnp.concatenate((following, base + accepted + 1 + row, ids[2 * rows : 3 * rows],
                                             accepted[None], ids[3 * rows + 1 :])).astype(jnp.int32)
            state = state.at[slot].set(bonus)
            state = state.at[slots + slot].set(base + accepted + 1)
            state = state.at[2 * slots + slot].set(accepted)
            state = state.at[reset].set(jnp.where(slot != 0, 0, state[reset]))
            # The packing op's next verify operand (verify mode 1).
            packet = jnp.concatenate((following_ids, state[3 * slots : 3 * slots + FIELDS], jnp.ones(1, jnp.int32)))
            return emitted[None], following_ids, state, packet

        def kernel_outputs(spec, vector, ids, state):
            def kernel(ids_ref, state_ref, vector_ref, *outs):
                kern.single_epilogue(spec, ROWS, ids_ref, state_ref, vector_ref, *outs)
            sizes = [spec.ids_size, spec.state_size] + [spec.ids_size + spec.fields + 1] * (spec.fields > 0)
            return pl.pallas_call(
                kernel,
                out_shape=(jax.ShapeDtypeStruct((1, ROWS), jnp.int32),
                           *(jax.ShapeDtypeStruct((size,), jnp.int32) for size in sizes)),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=3, in_specs=[],
                    out_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)] * (1 + len(sizes))),
                interpret=pltpu.InterpretParams(),
            )(ids, state, vector)

        rng = np.random.default_rng(0)
        for (slots, plan_rows, pages), fields in itertools.product(((3, 1, 26), (9, 4, 7)), (0, FIELDS)):
            size = 3 * slots + plan_rows * (8 + pages)
            spec = kern.SingleEpilogue(slots=slots, reset=3 * slots + 1, state_size=size,
                                       ids_size=3 * ROWS + 1 + ROWS * pages, plan=3 * slots, fields=fields)
            for accepted in (-1, 0, 3, 7, 8):
                # slots + 1 / + 2: the row bank write lands in plan row 0 (the packet carries it)
                for slot in (0, 1, -1, slots + 1, slots + 2, size + 5):
                    vector = rng.integers(-2**31, 2**31 - 1, (8, 128), dtype=np.int64).astype(np.int32)
                    vector[0, OUT_COUNT] = accepted
                    ids = rng.integers(0, 200000, spec.ids_size).astype(np.int32)
                    ids[2 * ROWS] = slot
                    state = rng.integers(-1000, 1000, size).astype(np.int32)
                    got = kernel_outputs(spec, *map(jnp.asarray, (vector, ids, state)))
                    want = reference(jnp.asarray(vector[:1]), jnp.asarray(ids), jnp.asarray(state), slots,
                                     spec.reset)
                    assert len(got) == (4 if fields else 3)
                    for g, w in zip(got, want):
                        assert np.array_equal(np.asarray(g), np.asarray(w)), (slots, accepted, slot, fields)
        print("OK")
    """

    LM_RING = """
        import os
        os.environ["JAX_PLATFORMS"] = "cpu"
        import numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter
        from sglang_tpu.k3_programs.kimi_k3.kernels import dspark_mla_kernel as kern

        H, CH, N, T = 512, kern.LM_CHUNK, 9, 8

        def logits(hidden, lm_head, early):
            def kernel(hidden_ref, lm_hbm, out_ref, gate, up, a, b, sems):
                def copy(k, slot, slots, sem_of):
                    return pltpu.make_async_copy(lm_hbm.at[:, pl.ds(k * CH, CH)], slots[slot],
                                                 sems.at[sem_of[slot]])
                if early:  # as the kernel: four gate / up halves started first, then the two scoped slots
                    slots = [buf.at[:, pl.ds(h * CH, CH)] for buf in (gate, up) for h in range(2)] + [a, b]
                    sem_of = (0, 1, 2, 3, 4, 5)
                    for k in range(4):
                        copy(k, k, slots, sem_of).start()
                else:
                    slots, sem_of = [a, b], (4, 5)
                chunks = kern.stream_chunks(lambda k, s: copy(k, s, slots, sem_of), slots, N,
                                            started=4 if early else 0)
                for k, slot in chunks:
                    out_ref[:, pl.ds(k * CH, CH)] = kern._f32(kern._bf16(kern._dot(hidden_ref[...], slot[...])))
            return np.asarray(pl.pallas_call(
                kernel,
                out_shape=jax.ShapeDtypeStruct((T, N * CH), jnp.float32),
                in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM), pl.BlockSpec(memory_space=pl.ANY)],
                out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                scratch_shapes=[pltpu.VMEM((H, 2 * CH), jnp.bfloat16), pltpu.VMEM((H, 2 * CH), jnp.bfloat16),
                                pltpu.VMEM((H, CH), jnp.bfloat16), pltpu.VMEM((H, CH), jnp.bfloat16),
                                pltpu.SemaphoreType.DMA((6,))],
                interpret=pltpu.InterpretParams(detect_races=True),
            )(hidden, lm_head))

        rng = np.random.default_rng(0)
        hidden = jnp.asarray(rng.standard_normal((T, H)), jnp.bfloat16)
        lm_head = jnp.asarray(rng.standard_normal((H, N * CH)), jnp.bfloat16)
        two, six = logits(hidden, lm_head, False), logits(hidden, lm_head, True)
        assert not interpreter.races.races_found
        assert np.array_equal(two.view(np.uint32), six.view(np.uint32))
        print("OK")
    """

    OP = """
        import os
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import contextlib, dataclasses, sys, types
        import numpy as np, ml_dtypes, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P

        # The op bodies as MlaDsparkDraft._build_single registers them (bridge stubbed), the fused kernel a stub
        # returning one vector and latents: knob on and off must return the same vector, pools, tokens, ids, state.
        captured = []
        bridge = types.ModuleType("sglang_tpu.k3_programs.bridge")
        bridge.register_op = lambda name, mesh, body, ins, outs, donate_argnums=(), export_key=None: (
            captured.append((body, ins, outs)) or body)
        for name in ("hbm", "kernel_mesh", "materialize", "upload", "report_load_stats", "stream_to_device"):
            setattr(bridge, name, lambda *a, **k: None)
        sys.modules["sglang_tpu.k3_programs.bridge"] = bridge
        from sglang_tpu.k3_programs.kimi_k3 import config as kconfig, draft_mla
        from sglang_tpu.k3_programs.kimi_k3.decode_ops import Control, DecodeOps
        from sglang_tpu.k3_programs.kimi_k3.kernels import dspark, dspark_mla_kernel as kern, mla_dcp, paged

        G, ROWS, PAGES, L, BLOCKS, PT, SLOTS = 4, 8, 6, 5, 7, 512, 3
        mla_dcp.GROUPS = (G, 8, 16, 32)
        stock = pl.pallas_call
        paged.pl.pallas_call = lambda *a, **k: stock(*a, **{"interpret": pltpu.InterpretParams(), **k})
        base_config = kconfig.active()
        for fn in (paged.commit_latent_rows, paged.commit_draft_rows):
            fn.__kwdefaults__["group"] = G

        @dataclasses.dataclass(frozen=True)
        class DraftConfig:
            layers: int = L

        def run(knob, vector, latents, pools, ids, state, decisions, packet=False):
            # The scatter commit as the reference (interpret mode has no aliased HBM refs).
            config = dataclasses.replace(base_config, draft_fused_epilogue=knob, draft_verify_packet=packet,
                                         fused_cache_commit=False)
            kconfig.active = paged._config = lambda: config
            stub = types.SimpleNamespace(**{n: getattr(kern, n) for n in (
                "WEIGHT_NAMES", "LEADING_NAMES", "SingleEpilogue", "single_epilogue", "copy_words")})
            stub.fused_draft_local = lambda *a, **k: (lambda *args: (jnp.asarray(vector), jnp.asarray(latents)))
            d = object.__new__(draft_mla.MlaDsparkDraft)
            d.dspark, d.kernel, d.paged, d.config = dspark, stub, paged, DraftConfig()
            d.attention_dp, d.max_position, d.owner_table, d._ops, d.mesh = 1, 1 << 20, None, {}, None
            d.names = ("ln1", "ln2", "q_norm", "kv_norm", "rot")
            captured.clear()
            d._build_single(Control(SLOTS, 1, PAGES))
            body, ins, outs = captured[0]
            weights = [np.zeros((G, L, 1, 64), ml_dtypes.bfloat16)] * 4 + [np.zeros((G, 64, 64), ml_dtypes.bfloat16)]
            rest = (np.zeros((3, ROWS, 64), ml_dtypes.bfloat16), np.zeros((G, ROWS, 128), np.float32),
                    np.zeros((1, 128), np.int32), ids, state, decisions)
            mesh = Mesh(np.asarray(jax.devices()), ("tp",))
            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=ins, out_specs=outs, check_vma=False))
            out = f(*map(jnp.asarray, (*weights, *pools, *rest)))
            return [np.asarray(x).view(np.uint8) if x.dtype == jnp.float8_e4m3fn else np.asarray(x) for x in out]

        rng = np.random.default_rng(0)
        for base, accepted, slot in ((5, 3, 1), (PT - 3, 7, 2), (28, 0, 0), (4 * PT - 4, 5, -1), (40, 2, SLOTS + 2)):
            vector = rng.integers(0, 160000, (1, 128)).astype(np.int32)
            vector[0, dspark.OUT_COUNT] = accepted
            latents = (rng.standard_normal((L, ROWS, 640)) * 2).astype(ml_dtypes.bfloat16)
            pools = [(rng.integers(0, 256, (G * BLOCKS, 128, 4, 640), dtype=np.uint8) & 0x7E).view(
                ml_dtypes.float8_e4m3fn) for _ in range(L)]
            ids = rng.integers(1, BLOCKS, 3 * ROWS + 1 + ROWS * PAGES).astype(np.int32)
            ids[ROWS], ids[2 * ROWS] = base, slot
            state = rng.integers(-50, 50, 3 * SLOTS + 8 + PAGES).astype(np.int32)
            decisions = np.array([[-1, -1]], np.int32)
            args = (vector, latents, pools, ids, state, decisions)
            old = run(False, *args)
            new = run(True, *args)
            assert len(old) == len(new) == 1 + L + 3
            for k, (a, b) in enumerate(zip(old, new)):
                assert a.shape == b.shape and np.array_equal(a, b), (base, k)
            assert any((o != p.view(np.uint8)).any() for o, p in zip(old[1 : 1 + L], pools))
            # KernelConfig.draft_verify_packet: the same outputs, then the packing op's next operand.
            want = DecodeOps.pack_verify_metadata(Control(SLOTS, 1, PAGES), 1, jnp.asarray(old[-2]),
                                                  jnp.asarray(old[-1]), jnp.ones(1, jnp.int32))
            for knob in (False, True):
                *outs, packet = run(knob, *args, packet=True)
                assert all(np.array_equal(a, b) for a, b in zip(old, outs, strict=True)), (base, knob)
                assert packet.dtype == np.int32 and np.array_equal(packet, np.asarray(want)), (base, knob)
        print("OK")
    """

    def _run(self, script, **env_vars):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), **env_vars)
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])

    def test_single_op_matches_the_xla_op(self):
        self._run(self.OP)

    def test_commit_draft_rows_matches_the_scatter(self):
        self._run(self.COMMIT)

    def test_single_epilogue_matches_the_xla_epilogue(self):
        self._run(self.EPILOGUE)

    def test_single_epilogue_with_unrolled_copies(self):
        # KernelConfig.draft_epilogue_unroll: blocks of 8 and 3 words, then the
        # rest (these sizes leave 0-7 words past the last block).
        for unroll in ("8", "3"):
            with self.subTest(unroll=unroll):
                self._run(self.EPILOGUE, UNROLL=unroll)

    def test_lm_ring_matches_two_slots(self):
        self._run(self.LM_RING)


class TestK31DraftVerifyPacket(CustomTestCase):
    """``KernelConfig.draft_verify_packet`` in the decoder (the packets
    themselves: ``TestK31DraftFusedEpilogue``): ``KimiDecoder._speculate``
    verifies from the draft op's packet exactly when it chains (same
    request, unchanged plan) at verify mode 1, and ``_enter_batched`` /
    ``_verify_and_draft`` hand the verify the packet the packing op
    (``DecodeOps.pack_verify_metadata``) builds from the same control and
    modes."""

    HOST = """
        import os
        os.environ["JAX_PLATFORMS"] = "cpu"
        import sys, types
        import numpy as np, jax, jax.numpy as jnp, torch

        bridge = types.ModuleType("sglang_tpu.k3_programs.bridge")
        bridge.register_op = bridge.register_global_op = (
            lambda name, mesh, fn, ins, outs, donate_argnums=(), export_key=None: fn)
        for name in ("hbm", "kernel_mesh", "materialize", "upload", "report_load_stats", "stream_to_device"):
            setattr(bridge, name, lambda *a, **k: None)
        sys.modules["sglang_tpu.k3_programs.bridge"] = bridge
        from sglang_tpu.k3_programs.kimi_k3 import config as kconfig
        kconfig.install(kconfig.KernelConfig().replace(draft_verify_packet=True))
        from sglang_tpu.k3_programs.kimi_k3 import decode
        from sglang_tpu.k3_programs.kimi_k3.decode_ops import Control, DecodeOps

        class Ops:
            mesh = None
            def __init__(self):
                self.calls = []
            def cached_op(self, key, build):
                return build()
            def verify_packed(self, ids, aux_layers, control, state, modes, rates=None, packet=None):
                self.calls.append((ids, int(modes[0]), packet))
                return "sharded", "aux", "decisions"
            def verify_batched(self, ids, aux_layers, requests, rows_per, control, state, modes, rates=None,
                               packet=None):
                self.calls.append((ids, modes, packet))
                return "sharded", "aux", "decisions"

        class Draft:  # step_single's outputs tagged with the step number
            aux_layers = (1,)
            def __init__(self, packets):
                self.packets, self.steps = packets, 0
            def _synthetic_rates(self, drafts):
                return None
            def step_single(self, state, aux, logits, ids, control_state, decisions, control):
                self.steps += 1
                out = (state, "emitted", ("ids", self.steps), ("control", self.steps))
                return out + ((("packet", self.steps),) if self.packets else ())
            def step_batched(self, states, aux, logits, ids, control_state, decisions, control, rows):
                return states, "emitted", control_state

        def decoder(packets):
            d = object.__new__(decode.KimiDecoder)
            d.ops, d.draft = Ops(), Draft(packets)
            d.layout, d.control, d.device_plan = Control(5, 4, 3), ("control", 0), None
            d.spec_req = d.draft_state = d.next_ids = d.next_packet = None
            d._drafts, d._fresh, d._offered, d._group, d.tracked = {}, {}, {}, {}, set()
            d.slot_map = types.SimpleNamespace(release=lambda finished: None)
            d._modes = {0: torch.zeros(1, dtype=torch.int32), 1: torch.ones(1, dtype=torch.int32)}
            d.stats = dict.fromkeys(("plans", "enters", "chained", "packets", "plain", "batched"), 0)
            d._take_state = lambda req_id: types.SimpleNamespace(vector=("vector", req_id))
            d._enter = lambda: (lambda control, vector, keep: (("entered", int(keep[0])),))
            return d

        def step(d, req_id, reset, scheduled, changed):
            d._sync_plan = lambda entries, tables: changed
            request = types.SimpleNamespace(req_id=req_id, scheduled=scheduled)
            d._speculate(request, [(1, reset, 0, 0, None)], [None], False)
            return d.ops.calls[-1]

        d = decoder(True)
        assert step(d, "a", True, 8, True) == (("entered", 0), 0, None)  # a new request: placeholders
        assert step(d, "a", False, 8, False) == (("ids", 1), 1, ("packet", 1))  # chained: the draft op's packet
        assert step(d, "a", False, 5, False) == (("ids", 2), 0, None)  # chained, fewer rows: mode 0, packed
        assert step(d, "a", False, 8, True) == (("entered", 1), 1, None)  # a new plan: the operand rebuilt
        assert step(d, "a", False, 8, False) == (("ids", 4), 1, ("packet", 4))
        assert step(d, "b", False, 8, False) == (("entered", 0), 0, None)  # another request
        assert d.next_packet == ("packet", 6)
        d.forget(["b"])
        assert d.next_ids is None and d.next_packet is None
        assert step(d, "b", False, 8, False) == (("entered", 0), 0, None)
        d._park_single()
        assert d.next_packet is None
        assert d.stats["packets"] == 2 and d.stats["chained"] == 3 and d.stats["enters"] == 4, d.stats
        d = decoder(False)  # a draft without packets (or the knob off): always packed
        step(d, "a", True, 8, True)
        assert step(d, "a", False, 8, False) == (("ids", 1), 1, None) and d.stats["packets"] == 0

        # The batched enter op's packet is the packing op's, from the same control and modes.
        rng = np.random.default_rng(3)
        d = decoder(True)
        layout = d.layout
        for requests, rows_per in ((2, 4), (4, 8)):
            control = rng.integers(0, 200, layout.size).astype(np.int32)
            fields = control[layout.plan_start : layout.tables_start].reshape(layout.rows, layout.FIELDS)
            fields[:, layout.SLOT] = rng.integers(0, layout.slots, layout.rows)
            fields[:, layout.RESET] = rng.integers(0, 2, layout.rows)
            control[layout.plan_start : layout.tables_start] = fields.reshape(-1)
            control = jnp.asarray(control)
            vectors = [jnp.asarray(rng.integers(0, 160000, (1, 128)).astype(np.int32)) for _ in range(requests)]
            modes = jnp.asarray(rng.integers(0, 2, requests).astype(np.int32))
            (ids,) = d._enter_batched(requests, rows_per)(control, *vectors, modes)
            ids2, packet = d._enter_batched(requests, rows_per, packet=True)(control, *vectors, modes)
            want = DecodeOps.pack_verify_metadata(layout, requests, ids, control, modes)
            assert np.array_equal(np.asarray(ids), np.asarray(ids2))
            assert np.asarray(packet).dtype == np.int32 and np.array_equal(np.asarray(packet), np.asarray(want))
            d.control = control
            states = [types.SimpleNamespace(vector=v) for v in vectors]
            d._verify_and_draft(requests, rows_per, states, modes)
            got_ids, _, got_packet = d.ops.calls[-1]
            assert np.array_equal(np.asarray(got_ids), np.asarray(ids))
            assert np.array_equal(np.asarray(got_packet), np.asarray(want))
        print("OK")
    """

    def _run(self, script):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])

    def test_decoder_verifies_from_the_packet(self):
        self._run(self.HOST)


class TestOwnerEmbedding(CustomTestCase):
    """``KernelConfig.decode_embedding_broadcast`` in Pallas interpret mode on
    four host devices (race detection on): ``decode_megakernel.decode_local``
    up to its fused stack (stubbed to record the embedding it is given) with
    the knob on runs ``token_kernels.owner_embedding`` and hands the stack the
    rows the gather + ``tp_sum`` gives (batched and per row), bit for bit, on
    every rank: tables with -0.0 (+0.0 out), +-inf and the smallest normal,
    ids at shard and 16-row block edges, one owner of every row, duplicates,
    ids outside the vocabulary (zero rows), 1 to 32 rows, 512 and 7168 wide."""

    SCRIPT = """
        import os
        N = 4
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N}"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import dataclasses
        import ml_dtypes, numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk, token_kernels

        called, captured = [], []
        stock = pl.pallas_call

        def interpret(*a, **k):
            called.append(k.get("name"))
            return stock(*a, **{"interpret": pltpu.InterpretParams(detect_races=True), **k})

        token_kernels.pl.pallas_call = interpret

        def stack(prefix, embedding, weights, states, positions, **kwargs):
            captured.append(embedding)
            return (prefix, None, states[0], states[1], None, None, None)

        dmk.stack = stack
        base = dmk._config()
        mesh = Mesh(np.asarray(jax.devices()), ("tp",))

        def embed(tokens, table, knob, batched):
            config = dataclasses.replace(base, decode_embedding_broadcast=knob)
            dmk._config = lambda: config
            try:
                def body(tokens, table):
                    captured.clear()
                    weights = {"embedding": table[0], "lm_head": jnp.zeros((table.shape[-1], 8), jnp.bfloat16)}
                    dmk.decode_local(weights, (jnp.zeros(1), jnp.zeros(1)), tokens, jnp.zeros_like(tokens),
                                     batch_size=tokens.shape[0], sequence_parallel_pre_attention=True,
                                     batched_decoder_boundaries=batched)
                    return captured[0][None]
                f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P("tp")), out_specs=P("tp"),
                                          check_vma=False))
                return np.asarray(f(jnp.asarray(tokens, jnp.int32), table)).view(np.uint16)
            finally:
                dmk._config = lambda: base

        rng = np.random.default_rng(0)
        for W, H in ((64, 512), (32, 7168)):
            V = N * W
            table = rng.standard_normal((N, W, H)).astype(ml_dtypes.bfloat16)
            bits = table.view(np.uint16)
            bits[rng.random(bits.shape) < 0.05] = 0x8000  # -0.0
            bits[rng.random(bits.shape) < 0.05] = 0x0000
            bits[:, :, :3] = [0x7F80, 0xFF80, 0x0080]  # +inf, -inf, the smallest normal
            flat = table.reshape(V, H).view(np.uint16)
            assert (flat == 0x8000).any()
            table = jnp.asarray(table)
            edges = [r * W + d for r in range(N) for d in (0, 15, 16, W - 1)]
            cases = [
                [5],
                [V - 1, 0, -1, V, W, 2 * W - 1, -2**31, 2**31 - 1],  # shard edges, outside
                edges[:16],
                [3, 3, 4, 19, 3, 2, 1, 0],  # one owner, duplicates, one 16-row block
                list(rng.integers(-W, V + W, 32)),
                [-5] * 4,  # nothing owned: zero rows
            ]
            for tokens in cases:
                want = embed(tokens, table, False, True)
                assert np.array_equal(want, embed(tokens, table, False, False))  # both XLA paths
                assert all(np.array_equal(want[0], want[r]) for r in range(N))
                called.clear()
                got = embed(tokens, table, True, len(tokens) % 2 == 0)
                assert called == ["kimi_owner_embedding"], called
                assert np.array_equal(got, want), (W, H, tokens)
                for b, t in enumerate(tokens):  # in range: the row, -0.0 as +0.0; else zeros
                    row = flat[t] if 0 <= t < V else np.zeros(H, np.uint16)
                    assert np.array_equal(got[0, b], np.where(row == 0x8000, 0, row)), (b, t)
        assert not interpreter.races.races_found
        print("OK")
    """

    def test_owner_rows_match_the_tp_sum(self):
        _run_script(self, self.SCRIPT)


class TestLocalEmbedding(CustomTestCase):
    """``KernelConfig.decode_embedding_replicated`` on four host devices.
    ``DecodeOps.load_weights`` (bridge stubbed: ops are jitted shard_maps) adds
    ``embedding_full``, every rank's copy the shards in mesh order bit for bit
    (NaN payloads included), only with the knob on. ``decode_local`` up to its
    stubbed stack then runs ``token_kernels.local_embedding`` alone (Pallas
    interpret mode: race detection, NaN-filled scratch, eager and on-wait
    DMAs) and hands the stack the rows of the knob-off paths (the batched and
    per-row gather + ``tp_sum``, ``owner_embedding``) on every rank, bit for
    bit (a NaN as a NaN of the same sign: XLA:CPU's FP32 -> BF16 convert keeps
    or drops payloads with the shape), with decode_embedding_broadcast off and
    on: tables with -0.0 (+0.0 out), +-inf, the smallest normal, subnormals
    and NaN payloads, ids at shard and 16-row block edges, duplicates, ids
    outside the vocabulary (zero rows), 1 to 32 rows, 512 and 7168 wide."""

    SCRIPT = """
        import os
        N = 4
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N}"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import dataclasses, sys, types
        import ml_dtypes, numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter
        from jax.sharding import Mesh, PartitionSpec as P

        mesh = Mesh(np.asarray(jax.devices()), ("tp",))
        registered, materialized = [], []
        bridge = types.ModuleType("sglang_tpu.k3_programs.bridge")

        def register_op(name, mesh_, body, ins, outs, donate_argnums=(), export_key=None):
            assert export_key is not None, name
            registered.append(name)
            return jax.jit(jax.shard_map(body, mesh=mesh, in_specs=ins, out_specs=outs, check_vma=False))

        def stream_to_device(iterator, store):
            for group, name, array in iterator:
                store(group, name, array)
            return {"wall": 0.0, "gib": 0.0}

        bridge.register_op, bridge.stream_to_device, bridge.materialize = register_op, stream_to_device, materialized.append
        for name in ("hbm", "kernel_mesh", "report_load_stats", "register_global_op", "upload"):
            setattr(bridge, name, lambda *a, **k: None)
        sys.modules["sglang_tpu.k3_programs.bridge"] = bridge
        from sglang_tpu.k3_programs.kimi_k3 import config as kconfig, decode_ops
        from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk, token_kernels

        class Host:  # torch.zeros(...).to("tpu") of the KDA slots
            def __init__(self, array):
                self.array = array

            def to(self, *args):
                return self.array

        decode_ops.torch = types.SimpleNamespace(bfloat16=jnp.bfloat16,
                                                 zeros=lambda shape, dtype=None: Host(jnp.zeros(shape, dtype)))
        base = kconfig.active()

        def load(families, knob):
            # DecodeOps.load_weights over a stand-in rank loader yielding families (global [N, ...] arrays).
            config = dataclasses.replace(base, decode_embedding_replicated=knob)
            kconfig.active = lambda: config
            try:
                ops = object.__new__(decode_ops.DecodeOps)
                ops.presharded, ops.rank, ops.mesh, ops.attention_dp, ops.num_slots = None, 0, mesh, 1, 3
                ops.recurrent_dtype, ops.weights, ops.names = jnp.float32, {}, ()
                ops.rank_loader = types.SimpleNamespace(
                    iter_decode_rank=lambda *a, **k: (("decode", n, v) for n, v in families.items()))
                registered.clear(), materialized.clear()
                ops.load_weights()
                return ops
            finally:
                kconfig.active = lambda: base

        called, captured, mode = [], [], ["eager"]
        stock = pl.pallas_call

        def interpret(*a, **k):
            called.append(k.get("name"))
            params = pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan", dma_execution_mode=mode[0])
            return stock(*a, **{"interpret": params, **k})

        token_kernels.pl.pallas_call = interpret

        def stack(prefix, embedding, weights, states, positions, **kwargs):
            captured.append(embedding)
            return (prefix, None, states[0], states[1], None, None, None)

        dmk.stack = stack

        def embed(tokens, weights, knobs, batched):
            config = dataclasses.replace(base, **knobs)
            dmk._config = lambda: config
            try:
                def body(tokens, shard, full):
                    captured.clear()
                    local = {"embedding": shard[0], "embedding_full": full[0],
                             "lm_head": jnp.zeros((shard.shape[-1], 8), jnp.bfloat16)}
                    dmk.decode_local(local, (jnp.zeros(1), jnp.zeros(1)), tokens, jnp.zeros_like(tokens),
                                     batch_size=tokens.shape[0], sequence_parallel_pre_attention=True,
                                     batched_decoder_boundaries=batched)
                    return captured[0][None]
                called.clear()
                f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P("tp"), P("tp")), out_specs=P("tp"),
                                          check_vma=False))
                out = f(jnp.asarray(tokens, jnp.int32), weights["embedding"], weights["embedding_full"])
                return np.asarray(out).view(np.uint16), list(called)
            finally:
                dmk._config = lambda: base

        def same(a, b):
            # Bit for bit, but a NaN's payload: XLA:CPU's f32 -> BF16 convert keeps or drops it depending on the
            # operand's shape (at 7168 lanes one path keeps it, another drops it), so NaNs compare by sign.
            nan = lambda x: (x & 0x7F80 == 0x7F80) & (x & 0x7F != 0)
            return np.array_equal(np.where(nan(a), a & 0x8000, a), np.where(nan(b), b & 0x8000, b)) and (
                np.array_equal(nan(a), nan(b)))

        rng = np.random.default_rng(0)
        SPECIAL = [0x8000, 0x7F80, 0xFF80, 0x0080, 0x0001, 0x8001, 0x007F, 0x7FC1, 0xFFC1, 0x7F81]
        for W, H in ((64, 512), (32, 7168)):
            V = N * W
            table = rng.standard_normal((N, W, H)).astype(ml_dtypes.bfloat16)
            bits = table.view(np.uint16)
            for value in SPECIAL:  # -0.0, +-inf, the smallest normal, subnormals, NaN payloads (quiet, signalling)
                bits[rng.random(bits.shape) < 0.01] = value
            bits[rng.random(bits.shape) < 0.04] = 0x0000
            bits[:, :, : len(SPECIAL)] = SPECIAL
            flat = table.reshape(V, H).view(np.uint16)
            families = {"lm_head": jnp.zeros((N, 8, 8), jnp.bfloat16), "embedding": jnp.asarray(table)}

            # The load op: only with the knob; every rank holds the shards in mesh order, bit for bit.
            decode_ops.TP_SIZE, decode_ops.VOCAB = N, V
            off = load(families, False)
            assert off.names == ("embedding", "lm_head") and not registered, (off.names, registered)
            ops = load(families, True)
            assert ops.names == ("embedding", "embedding_full", "lm_head"), ops.names
            assert registered == ["pallas::k3_decode_embedding_full"], registered
            full = ops.weights["embedding_full"]
            assert any(t is full for t in materialized)
            assert full.shape == (N, V, H) and full.dtype == jnp.bfloat16, (full.shape, full.dtype)
            gathered = np.asarray(full).view(np.uint16)
            assert all(np.array_equal(gathered[r], flat) for r in range(N))
            try:
                load({"embedding": jnp.asarray(table[:, :16])}, True)
                raise AssertionError("a table that is not V / n rows a rank was taken")
            except RuntimeError:
                pass

            edges = [r * W + d for r in range(N) for d in (0, 15, 16, W - 1)]
            cases = [
                [5],
                [V - 1, 0, -1, V, W, 2 * W - 1, -2**31, 2**31 - 1],  # shard edges, outside
                edges,  # every shard's and 16-row block's edges
                [3, 3, 4, 19, 3, 2, 1, 0],  # one owner, duplicates, one 16-row block
                list(rng.integers(-W, V + W, 32)),
                [-5] * 4,  # nothing owned: zero rows
                [V, -1, 2**31 - 1, -2**31, V + 7],  # every row outside the vocabulary
                [17, 17, V - 16],
                list(rng.integers(0, V, 2)),
                list(rng.integers(0, V, 7)),
                list(rng.integers(0, V, 31)),
            ][: 7 if H > 512 else None]  # (the 7168-wide table: the first seven)
            for tokens in cases:
                batched = len(tokens) % 2 == 0
                want, _ = embed(tokens, ops.weights, {}, True)
                assert same(want, embed(tokens, ops.weights, {}, False)[0])  # both XLA paths
                assert all(np.array_equal(want[0], want[r]) for r in range(N))
                for b, t in enumerate(tokens):
                    # In range: the row with -0.0 as +0.0, NaNs as NaNs and subnormals as themselves or +0.0 (the
                    # platform's f32 flush; XLA:CPU flushes); else zeros.
                    row = flat[t] if 0 <= t < V else np.zeros(H, np.uint16)
                    nan = (row & 0x7F80 == 0x7F80) & (row & 0x7F != 0)
                    subnormal = (row & 0x7F80 == 0) & (row & 0x7F != 0)
                    out = want[0, b]
                    assert ((out == np.where(row == 0x8000, 0, row)) | (subnormal & (out == 0)) | nan).all(), (b, t)
                    assert (out[nan] & 0x7F80 == 0x7F80).all() and (out[nan] & 0x7F != 0).all(), (b, t)
                mode[0] = "eager"
                owner, ran = embed(tokens, ops.weights, dict(decode_embedding_broadcast=True), batched)
                assert ran == ["kimi_owner_embedding"] and same(owner, want), (W, H, tokens, ran)
                for mode[0], broadcast in (("eager", False), ("on_wait", True)):  # the knob wins over the broadcast
                    knobs = dict(decode_embedding_replicated=True, decode_embedding_broadcast=broadcast)
                    got, ran = embed(tokens, ops.weights, knobs, batched)
                    assert ran == ["kimi_local_embedding"], ran
                    assert all(np.array_equal(got[0], got[r]) for r in range(N))
                    assert same(got, want) and same(got, owner), (W, H, tokens, mode[0], broadcast)
            print("width", H, "cases", len(cases), flush=True)
        assert not interpreter.races.races_found
        print("OK")
    """

    def test_local_rows_match_the_knob_off_rows(self):
        _run_script(self, self.SCRIPT)


class TestVerifyGreedyKernel(CustomTestCase):
    """``KernelConfig.verify_greedy_kernel`` in Pallas interpret mode on four
    host devices (race detection on): ``sampling.verify_acceptance`` with the
    knob on runs ``token_kernels.greedy_decisions`` and returns the knob-off
    decisions (``jnp.argmax`` + ``all_gather``) bit for bit on every rank --
    greedy steps through ``lax.cond``, verify mode 0, synthetic rates and
    mixed sampled / greedy requests, 1x8, 2x8 and 3x4 rows -- on logits with
    ties of the maximum within and across shards, maxima of -0.0 and +0.0,
    rows of -inf, +inf at shard edges and NaNs (the maximum; the first one
    wins), and drafts matching their targets for 0 to all rows; the greedy
    rows' decisions are also numpy's first maxima. The kernel alone gives the
    XLA expression's ``take_along_axis`` fill for forced counts past the rows."""

    SCRIPT = """
        import os
        N = 4
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N}"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import dataclasses
        import numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3 import config as kconfig, sampling
        from sglang_tpu.k3_programs.kimi_k3.kernels import token_kernels

        called = []
        stock = pl.pallas_call

        def interpret(*a, **k):
            called.append(k.get("name"))
            return stock(*a, **{"interpret": pltpu.InterpretParams(detect_races=True), **k})

        token_kernels.pl.pallas_call = interpret
        base = kconfig.active()
        mesh = Mesh(np.asarray(jax.devices()), ("tp",))
        W = 128
        V = N * W
        NAN, INF, ZERO = np.float32(np.nan), np.float32(np.inf), np.float32(0.0)
        compiled = {}

        def decide(logits, tokens, modes, temperature, rates, knob):
            R, rows = tokens.shape
            config = dataclasses.replace(base, verify_greedy_kernel=knob)
            kconfig.active = lambda: config  # read when the op traces (its first call)
            try:
                traced = (R, rows, rates, knob) not in compiled
                if traced:
                    def body(logits, tokens, positions, modes, t, p, k, s):
                        return sampling.verify_acceptance(logits[0], tokens, positions, modes, (t, p, k, s),
                                                          synthetic_rates=rates, single=R == 1)[None]
                    compiled[R, rows, rates, knob] = jax.jit(jax.shard_map(
                        body, mesh=mesh, in_specs=(P("tp"),) + (P(),) * 7, out_specs=P("tp"), check_vma=False))
                    called.clear()
                shards = np.ascontiguousarray(logits.reshape(R * rows, N, W).transpose(1, 0, 2))
                args = (shards, tokens, np.arange(R, dtype=np.int32) * 97 + 5, modes, temperature,
                        np.full(R, 0.9, np.float32), np.zeros(R, np.int32), np.arange(R, dtype=np.int32) + 11)
                out = np.asarray(compiled[R, rows, rates, knob](*map(jnp.asarray, args)))
                if traced:  # the knob-on op runs the kernel, the knob-off op does not
                    assert ("kimi_greedy_decisions" in called) == knob, (knob, called)
                return out
            finally:
                kconfig.active = lambda: base

        def first_max(row):  # NaN is the maximum and the first one wins, as jnp.argmax
            nan = np.flatnonzero(np.isnan(row))
            return int(nan[0]) if nan.size else int(np.argmax(row))

        rng = np.random.default_rng(0)
        ids = np.arange(V)

        def adversarial(T, ordered=False):
            x = rng.integers(-4, 3, (T, V)).astype(np.float32)  # ties of the maximum
            low = lambda r: -np.abs(r) - 1
            kinds = [
                lambda r: r,
                lambda r: np.where(r == r.max(), r, r - 9),
                lambda r: np.full_like(r, -INF),  # all -inf: id 0
                lambda r: np.where(ids % W == W - 1, -ZERO, low(r)),  # -0.0 maxima on every rank
                lambda r: np.where(ids == 2 * W + 5, ZERO, np.where(ids == W + 9, -ZERO, low(r))),
                lambda r: np.where(ids == 2 * W + 5, -ZERO, np.where(ids == W + 9, ZERO, low(r))),
                lambda r: np.where(ids == W + 3, -ZERO, np.where(ids == W + 7, ZERO, low(r))),
                lambda r: np.where(np.isin(ids, [W - 1, W, 3 * W]), INF, r),  # +inf at shard edges
                lambda r: np.where(ids == V - 1, NAN, r),  # one NaN, the last id
                lambda r: np.where(np.isin(ids, [2 * W + 7, W + 100, V - 1]), NAN, np.where(ids == 0, INF, r)),
                lambda r: np.where(ids % 3 == 1, NAN, r),
                lambda r: np.where(ids == W, NAN, r),  # a NaN at a shard's first id
                lambda r: rng.standard_normal(V).astype(np.float32) * 10,
            ]
            pick = (lambda t: t) if ordered else (lambda t: t + int(rng.integers(len(kinds))))
            return np.stack([kinds[pick(t) % len(kinds)](x[t]) for t in range(T)]).astype(np.float32)

        for R, rows in ((1, 8), (2, 8), (3, 4)):
            T = R * rows
            rates = (0.9, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1)[: rows - 1]
            for trial in range(3):
                logits = adversarial(T)
                targets = np.array([first_max(logits[t]) for t in range(T)]).reshape(R, rows)
                tokens = rng.integers(0, V, (R, rows)).astype(np.int32)
                for r in range(R):  # request r's drafts match their targets for k leading rows
                    k = (r + trial * 3) % rows
                    tokens[r, 1 : k + 1] = targets[r, :k]
                    if k + 1 < rows and tokens[r, k + 1] == targets[r, k]:
                        tokens[r, k + 1] = (targets[r, k] + 1) % V
                for modes, temperature, synthetic in (
                    (np.ones(R, np.int32), np.zeros(R, np.float32), None),  # greedy: lax.cond
                    ((np.arange(R) % 2).astype(np.int32), np.zeros(R, np.float32), None),  # mode 0
                    (np.ones(R, np.int32), np.zeros(R, np.float32), rates),  # synthetic: no cond
                    (np.ones(R, np.int32), np.where(np.arange(R) % 2, 0.8, 0.0).astype(np.float32), None),
                ):
                    if temperature.any() and np.isnan(logits).any():
                        continue  # sampled rows of NaN logits
                    want = decide(logits, tokens, modes, temperature, synthetic, False)
                    got = decide(logits, tokens, modes, temperature, synthetic, True)
                    assert all(np.array_equal(got[0], got[r]) for r in range(N))
                    assert np.array_equal(got, want), (R, rows, trial, modes, temperature, got[0], want[0])
                    for r in np.flatnonzero((temperature == 0) & (modes == 1) & (synthetic is None)):
                        miss = np.flatnonzero(tokens[r, 1:] != targets[r, :-1])
                        count = int(miss[0]) if miss.size else rows - 1
                        assert got[0][r].tolist() == [count, targets[r, count]], (r, got[0][r], count)

        def xla(logits, tokens, forced):  # verify_acceptance's greedy expression
            R, rows = tokens.shape
            targets = sampling.greedy(logits).reshape(R, rows)
            match = tokens[:, 1:] == targets[:, :-1]
            first = jnp.min(jnp.where(match, rows - 1, jnp.arange(rows - 1)), axis=1)
            count = jnp.where(forced >= 0, forced, first)
            return jnp.stack((count, jnp.take_along_axis(targets, count[:, None], axis=1)[:, 0]), axis=1)

        # Every row's target (every kind of row) through the bonus of a forced count, and
        # counts past the rows.
        R, rows = 4, 8
        logits = adversarial(R * rows, ordered=True)
        shards = jnp.asarray(np.ascontiguousarray(logits.reshape(R * rows, N, W).transpose(1, 0, 2)))
        tokens = jnp.asarray(rng.integers(0, V, (R, rows)).astype(np.int32))
        ops = [jax.jit(jax.shard_map(lambda l, t, c, fn=fn: fn(l[0], t, c)[None], mesh=mesh,
                                     in_specs=(P("tp"), P(), P()), out_specs=P("tp"), check_vma=False))
               for fn in (xla, token_kernels.greedy_decisions)]
        for k in range(rows + 1):
            forced = jnp.asarray(np.array([k, k, k + rows, -1], np.int32))
            want, got = (np.asarray(op(shards, tokens, forced)) for op in ops)
            assert np.array_equal(want, got), (k, want[0], got[0])
            if k < rows:
                assert got[0, 0, 1] == first_max(logits[k]) and got[0, 1, 1] == first_max(logits[rows + k])
            assert (got[:, 2, 1] == -2**31).all() and (k < rows or (got[:, :2, 1] == -2**31).all())
        assert not interpreter.races.races_found
        print("OK")
    """

    def test_kernel_decisions_match_the_xla_decisions(self):
        _run_script(self, self.SCRIPT)


def _run_script(case, script):
    import os
    import subprocess
    import sys
    import textwrap

    from jax.experimental.pallas import tpu as pltpu

    if not hasattr(pltpu, "InterpretParams"):
        case.skipTest("Pallas TPU interpret mode unavailable")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=1800,
    )
    case.assertIn("OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:])


class TestK31DraftWithVerify(CustomTestCase):
    """``KernelConfig.draft_with_verify`` through the real op bodies on a
    4-device CPU mesh, with stand-ins for the decoder and draft megakernels
    that read every operand (each op deletes its donated operands).
    ``KimiDecoder._speculate`` over chains of single-request steps (entering,
    chained at verify mode 1 and 0, a table that grows, a second request,
    the first one back from its parked state, a reset, sampled rows), with
    Kimi-K3.1 index pools and window centres (``dsa_seeded_window``,
    ``verify_fused_tail`` off and on) or without: at every step the one op
    gives the emitted tokens, next ids, control, draft vector, KDA banks,
    pools, centres and draft pools of the verify op then the draft op,
    bitwise, alone after the enter and plan ops. ``KimiDecoder._warm_steps``
    then serving steps: the precompiled step programs are those that run,
    and serving runs the one op on warmed operand histories only."""

    SCRIPT = """
        import os
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import dataclasses, hashlib, sys, types
        import numpy as np, ml_dtypes, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P

        # KimiDecoder over chains of single-request steps, with the real op bodies (bridge stubbed: every op a jitted
        # shard_map that deletes its donated operands) and stand-ins for the decoder and draft megakernels: the verify op then
        # the draft op (draft_verify_packet off / on) against KernelConfig.draft_with_verify's one op, both draft epilogues,
        # with Kimi-K3.1 index pools and window centres (dsa_seeded_window; verify_fused_tail off / on) or without them.
        G, ROWS, PAGES, L, BLOCKS, SLOTS, V, H, IDX = 4, 8, 3, 5, 7, 5, 1024, 16, 2
        mesh = Mesh(np.asarray(jax.devices()), ("tp",))
        launched, source, kept, kinds = [], {}, [], {"host": "host"}
        bridge = types.ModuleType("sglang_tpu.k3_programs.bridge")


        def record(name, args, outputs):
            # The launch with its operands' producers ("op:output", else "host"); its outputs recorded as the op's.
            launched.append((name, tuple(source.get(id(a), "host") for a in args)))
            for i, out in enumerate(outputs):
                source[id(out)] = f"{name}:{i}"
                kept.append(out)  # ids stay unique


        def register_op(name, mesh_, body, ins, outs, donate_argnums=(), export_key=None):
            assert export_key is not None, name
            kinds[name] = "sharded"
            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=ins, out_specs=outs, check_vma=False),
                        donate_argnums=tuple(donate_argnums))

            def call(*args):
                assert len(args) == len(ins), (name, len(args), len(ins))
                outputs = f(*args)
                record(name, args, outputs)
                return outputs
            call.specs, call.op_name, call.export_key = (ins, outs, tuple(donate_argnums)), name, export_key
            return call


        def register_global_op(name, mesh_, fn, ins, outs, donate_argnums=(), export_key=None):
            kinds[name] = "global"
            f = jax.jit(fn, donate_argnums=tuple(donate_argnums))

            def call(*args):
                outputs = f(*args)
                record(name, args, outputs)
                return outputs
            call.op_name = name
            return call


        bridge.register_op, bridge.register_global_op = register_op, register_global_op
        for name in ("hbm", "kernel_mesh", "materialize", "upload", "report_load_stats", "stream_to_device"):
            setattr(bridge, name, lambda *a, **k: None)
        sys.modules["sglang_tpu.k3_programs.bridge"] = bridge
        from sglang_tpu.k3_programs.kimi_k3 import config as kconfig, decode, decode_ops, draft_mla
        from sglang_tpu.k3_programs.kimi_k3.decode_ops import Control, DecodeOps
        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp, dspark, dspark_mla_kernel as kern, mla_dcp, paged

        mla_dcp.GROUPS = (G, 8, 16, 32)
        stock = pl.pallas_call
        paged.pl.pallas_call = lambda *a, **k: stock(*a, **{"interpret": pltpu.InterpretParams(), **k})
        for fn in (paged.commit_latent_rows, paged.commit_draft_rows):
            fn.__kwdefaults__["group"] = G
        draft_mla._settled = lambda state: state
        base_config = kconfig.active()
        VL = V // G


        class Host:  # a host tensor the decoder builds: .to(device) gives the device array
            def __init__(self, array):
                self.array = array

            def to(self, *args, **kwargs):
                return self.array


        decode.torch = types.SimpleNamespace(
            int32=jnp.int32, float32=jnp.float32, Tensor=object, zeros=lambda shape, dtype=None: Host(jnp.zeros(shape, dtype)),
            ones=lambda shape, dtype=None: Host(jnp.ones(shape, dtype)), from_numpy=lambda value: Host(jnp.asarray(value)),
            tensor=lambda value, dtype=None: Host(jnp.asarray(value, dtype)),
            full=lambda shape, value, dtype=None, device=None: jnp.full(shape, value, dtype))


        def ints(x):  # an exact integer digest of an array (any order of summation)
            x = jnp.asarray(x)
            if x.dtype in (jnp.bfloat16, jnp.float8_e4m3fn):
                x = jax.lax.bitcast_convert_type(x, jnp.int16 if x.dtype == jnp.bfloat16 else jnp.int8)
            elif x.dtype == jnp.float32:
                x = jax.lax.bitcast_convert_type(x, jnp.int32)
            return jnp.sum((x.astype(jnp.int32) & 0xFFF) * 3 + 1) % 9973


        def target_of(tokens, positions):  # the stand-in decoder's greedy token of a row
            return (tokens * 5 + positions) % V


        def owned_rows(tables, positions):  # each row's latent row on its owner (else past the pool)
            rank = jax.lax.axis_index("tp")
            row = jnp.arange(positions.shape[0], dtype=jnp.int32)
            block = tables[row, jnp.minimum((positions // 16) // G, tables.shape[1] - 1)]
            return jnp.where((positions // 16) % G == rank, block, G * BLOCKS), positions % 16


        def decode_paged(weights, conv, recurrent, pool, tokens, positions, tables, slots, *, batch_size, dcp_pool_map,
                         dcp_owner_table, dsa_pools, layers, state_row=None, dsa_centres=None, defer_kda_commit=False,
                         defer_latent_commit=False, **options):
            # Stand-in decoder: logits whose argmax is target_of (values carrying the input state, pools, centres and
            # weights), aux rows (verifiers), per-row post-states (verifiers) or committed slots, each row's latent and index
            # key committed on its owner (verify_fused_tail: left to commit_verify_tail), and the rows' next window centres
            # scattered into their slots (paged.decode_paged's gather and scatter).
            verify = state_row is not None
            assert defer_kda_commit == verify and dsa_pools == len(pool) - 2, (defer_kda_commit, dsa_pools, len(pool))
            assert (dsa_centres is not None) == (dsa_pools and kconfig.active().dsa_seeded_window)
            assert defer_latent_commit == (verify and dsa_pools > 0 and kconfig.active().verify_fused_tail)
            rows = options.get("sequence_rows", 1)
            state = ints(conv[slots[0]]) + ints(recurrent[slots[0]]) + (7 * state_row if verify else 0)
            if dsa_centres is not None:
                state = state + 13 * ints(dsa_dcp.gather_centres(dsa_centres, slots, rows))
            # Replicated results (aux) agree on every rank, as the decoder's: the per-rank digests summed.
            mix = jax.lax.psum((state + ints(pool[0]) + 3 * ints(pool[1]) + 5 * ints(weights["w0"])
                                + 7 * ints(weights["lm_head"]) + 11 * ints(weights["embedding"])
                                + sum((17 + j) * ints(p) for j, p in enumerate(pool[2:]))) % 251, "tp") % 251
            vocab = jax.lax.axis_index("tp") * VL + jnp.arange(VL, dtype=jnp.int32)
            t = target_of(tokens, positions)
            logits = -jnp.abs(vocab[None, :] - t[:, None]).astype(jnp.float32) + ((vocab[None, :] * 7 + mix) % 3).astype(
                jnp.float32) * 0.125
            row = jnp.arange(batch_size, dtype=jnp.int32)
            new_conv = conv[slots] + ((tokens + row)[:, None, None] % 5).astype(conv.dtype)
            new_recurrent = recurrent[slots] + ((positions * 3 + mix)[:, None, None] % 7).astype(recurrent.dtype)
            if not verify:
                new_conv, new_recurrent = conv.at[slots].set(new_conv), recurrent.at[slots].set(new_recurrent)
            latents = (tokens + mix).astype(pool[0].dtype)[:, None]
            keys = (tokens * 3 + positions + mix).astype(pool[0].dtype)[:, None]
            pool, pending = list(pool), None
            if defer_latent_commit:
                pending = (latents, keys, tables, positions)
            else:
                first, offset = owned_rows(tables, positions)
                pool[0] = pool[0].at[first, offset].set(latents, mode="drop")
                if dsa_pools:
                    pool[2] = pool[2].at[first, offset].set(keys, mode="drop")
            extra = ()
            if dsa_centres is not None:
                new = (tokens[None, :] * 7 + positions[None, :] + mix + 31 * jnp.arange(dsa_pools)[:, None]) % 100003
                extra = (dsa_dcp.scatter_centres(dsa_centres, slots, rows, new.reshape(-1).astype(jnp.int32)),)
            if defer_latent_commit:
                extra += (pending,)
            if verify:
                aux = ((tokens[None, :, None] + positions[None, :, None] * 3 + mix + jnp.arange(3)[:, None, None] * 11
                        + jnp.arange(H)[None, None, :]) % 64).astype(jnp.bfloat16)
                assert options["aux_hidden_layers"] == (1, 2, 3)
                return logits, aux, new_conv, new_recurrent, tuple(pool), *extra
            return logits, new_conv, new_recurrent, tuple(pool), *extra


        def commit_verify_tail(conv, recurrent, new_conv, new_recurrent, targets, sources, latent_pools, index_pools, latents,
                               keys, tables, positions, pool_map, owner_table=None):
            # Stand-in for the fused tail: the accepted KDA rows (paged._commit_kda_states), the latents and index keys.
            conv, recurrent = paged._commit_kda_states(conv, recurrent, new_conv, new_recurrent, targets, source_rows=sources)
            first, offset = owned_rows(tables, positions)
            latent_pools, index_pools = list(latent_pools), list(index_pools)
            latent_pools[0] = latent_pools[0].at[first, offset].set(latents, mode="drop")
            index_pools[0] = index_pools[0].at[first, offset].set(keys, mode="drop")
            return conv, recurrent, *latent_pools, *index_pools


        def fused_draft_local(config, *, requests, pages, owner_table, single):
            # Stand-in draft kernel: the forced decisions' count and bonus, drafts that follow the decoder's greedy
            # tokens where a digest of every operand says so, and latents from the aux rows.
            def fused(weights, pools, aux, logits, previous, *rest):
                if len(rest) == 2:  # draft_fused_epilogue: the packed ids and the decisions
                    ids, decisions = rest
                    base, count, bonus = ids[ROWS], decisions[0, 0], decisions[0, 1]
                else:
                    bases, counts, bonuses, _ = rest
                    base, count, bonus = bases[0], counts[0], bonuses[0]
                digest = (ints(aux) + 3 * ints(logits) + 5 * ints(previous)
                          + sum((7 + j) * ints(p) for j, p in enumerate(pools))
                          + sum((11 + j) * ints(weights[n]) for j, n in enumerate(sorted(weights)))) % 9973
                digest = jax.lax.psum(digest, "tp") % 9973  # the vector is replicated, as the kernel's
                start = base + count + 1
                drafts, token = [], bonus
                for k in range(ROWS - 1):
                    follow = (digest + 13 * k + start) % 5 != 0
                    token = jnp.where(follow, target_of(token, start + k), (token * 3 + digest + k) % V)
                    drafts.append(token)
                vector = jnp.full((1, 128), -1, jnp.int32)
                vector = vector.at[0, dspark.OUT_DRAFTS : dspark.OUT_DRAFTS + ROWS - 1].set(jnp.stack(drafts))
                vector = vector.at[0, dspark.OUT_COUNT].set(count).at[0, dspark.OUT_BONUS].set(bonus)
                vector = vector.at[0, dspark.OUT_TARGETS].set(digest)
                latents = ((aux[0][None, :, :1].astype(jnp.float32) + digest % 17 + jnp.arange(L)[:, None, None])
                           * jnp.ones((L, ROWS, 640))) % 32
                return vector, latents.astype(jnp.bfloat16)
            return fused


        @dataclasses.dataclass(frozen=True)
        class DraftConfig:
            layers: int = L
            aux_layers: tuple = (1, 2, 3)
            hidden: int = H
            vocab: int = V


        def build(knobs, rates, index=False):
            # A decoder over the stand-ins; index: Kimi-K3.1 index pools (with window centres under dsa_seeded_window).
            config = dataclasses.replace(base_config, fused_cache_commit=False, **knobs)
            kconfig.active = paged._config = kern._config = lambda: config
            rng = np.random.default_rng(7)
            ops = object.__new__(DecodeOps)
            ops.mesh, ops.rank, ops.attention_dp, ops.batch_sizes, ops.num_slots = mesh, 0, 1, (1, 2), SLOTS
            ops.names = ("embedding", "lm_head", "w0")
            ops.weights = {
                "embedding": jnp.asarray(rng.standard_normal((G, VL, H)), jnp.bfloat16),
                "lm_head": jnp.asarray(rng.standard_normal((G, H, VL)), jnp.bfloat16),
                "w0": jnp.asarray(rng.standard_normal((G, 4, 4)), jnp.bfloat16),
            }
            ops.slots = [jnp.asarray(rng.integers(0, 4, (G, SLOTS, 2, 8)), jnp.bfloat16),
                         jnp.asarray(rng.integers(0, 4, (G, SLOTS, 3, 8)), jnp.float32)]
            pools = 2 + (IDX if index else 0)
            ops.unified_pools = [jnp.asarray(rng.integers(0, 9, (G * BLOCKS, 16, 1)), jnp.float32) for _ in range(pools)]
            ops.dsa_pools = IDX if index else 0
            ops.dsa_centres = None
            if index and config.dsa_seeded_window:  # the same array on every rank (DecodeOps.use_unified_pools' layout)
                ops.dsa_centres = jnp.tile(jnp.asarray(rng.integers(-99, 99, (1, SLOTS, IDX, ROWS)), jnp.int32), (G, 1, 1, 1))
            ops.pool_map, ops.owner_table, ops._ops, ops._shared = (0, 1), None, {}, {}
            ops.paged = types.SimpleNamespace(decode_paged=decode_paged, _commit_kda_states=paged._commit_kda_states,
                                              commit_verify_tail=commit_verify_tail)
            ops._run = lambda op, rows, *args: op(*args)

            def adopt(conv, recurrent, pools):  # DecodeOps._adopt, the pools a list (no torch storage swap)
                pools = list(pools)
                if ops.dsa_centres is not None:
                    *pools, ops.dsa_centres = pools
                assert len(pools) == len(ops.unified_pools)
                ops.slots, ops.unified_pools = [conv, recurrent], pools
            ops._adopt = adopt
            d = object.__new__(draft_mla.MlaDsparkDraft)
            stub = types.SimpleNamespace(**{n: getattr(kern, n) for n in (
                "WEIGHT_NAMES", "LEADING_NAMES", "SingleEpilogue", "single_epilogue", "copy_words")})
            stub.fused_draft_local = fused_draft_local
            d.dspark, d.kernel, d.paged, d.config, d.mesh = dspark, stub, paged, DraftConfig(), mesh
            d.attention_dp, d.max_position, d.owner_table, d._ops = 1, 1 << 20, None, {}
            d.synthetic_rates, d.synthetic_rates_by_width, d._aux_layers = rates, None, (1, 2, 3)
            d.names = tuple(kern.WEIGHT_NAMES)
            d.weights = {n: jnp.asarray(rng.standard_normal((G, L, 1, 4) if n in kern.LEADING_NAMES else (G, 2, 4)),
                                        jnp.bfloat16) for n in d.names}
            d.weights["lm_head"], d.weights["embed"] = ops.weights["lm_head"], ops.weights["embedding"]
            d.pools = [(jnp.asarray(rng.integers(0, 256, (G * BLOCKS, 128, 4, 640), dtype=np.uint8) & 0x7E)
                        .view(jnp.float8_e4m3fn)) for _ in range(L)]

            def draft_adopt(pools):
                d.pools = list(pools)
            d._adopt = draft_adopt
            d.new_state = lambda anchor: draft_mla.MlaDraftState(
                d._new_state_op()(jnp.asarray(anchor, jnp.int32).reshape(1))[0])
            k = object.__new__(decode.KimiDecoder)
            k.ops, k.draft, k.layout, k.sub_base = ops, d, Control(SLOTS, 2, PAGES), SLOTS
            k.control = jnp.asarray(rng.integers(0, 50, k.layout.size), jnp.int32)
            k.device_plan = k.spec_req = k.draft_state = k.next_ids = k.next_packet = None
            k._drafts, k._fresh, k._offered, k._group, k.tracked = {}, {}, {}, {}, set()
            k.slot_map = types.SimpleNamespace(release=lambda finished: None)
            k._zero = jnp.zeros(1, jnp.int32)
            k._modes = {0: jnp.zeros(1, jnp.int32), 1: jnp.ones(1, jnp.int32)}
            k.stats = dict.fromkeys(("plans", "enters", "chained", "packets", "plain", "batched", "predrafts"), 0)
            return k


        def digest(arrays):  # the arrays' dtypes, shapes and bytes, hashed
            return tuple((str(a.dtype), a.shape, hashlib.sha256(np.asarray(a).tobytes()).hexdigest())
                         for a in map(np.asarray, arrays))


        def same(a, b):
            a, b = np.asarray(a), np.asarray(b)
            if a.dtype != b.dtype or a.shape != b.shape:
                return False
            if a.dtype.itemsize == 1:
                return np.array_equal(a.view(np.uint8), b.view(np.uint8))
            return np.array_equal(a.view(f"u{a.dtype.itemsize}"), b.view(f"u{b.dtype.itemsize}"))


        PART = os.environ.get("PART", "chain")
        if PART == "chain":
            # (request, reset, scheduled, table, sampling): entering, chained (mode 1 / fewer rows: mode 0), a table that
            # grew (the operand rebuilt), another request, the first one back (its parked state), a reset, sampled rows.
            GREEDY, SAMPLED = (0.0, 1.0, 0, 0), (0.7, 0.9, 20, 3)
            CHAIN = [("a", True, 8, [1, 2], GREEDY)] + [("a", False, 8, [1, 2], GREEDY)] * 3 + [
                ("a", False, 5, [1, 2], GREEDY), ("a", False, 8, [1, 2], GREEDY), ("a", False, 8, [1, 2, 3], GREEDY),
                ("a", False, 8, [1, 2, 3], GREEDY), ("b", True, 8, [4, 5, 6], GREEDY), ("b", False, 8, [4, 5, 6], GREEDY),
                ("a", False, 8, [1, 2, 3], GREEDY), ("a", False, 8, [1, 2, 3], GREEDY), ("a", True, 8, [3, 1, 2], GREEDY),
                ("a", False, 8, [3, 1, 2], SAMPLED), ("a", False, 8, [3, 1, 2], SAMPLED), ("a", False, 8, [3, 1, 2], GREEDY),
            ]


            def run(knobs, rates, index):
                source.clear(), kept.clear()  # (one run's outputs held at a time)
                k = build(knobs, rates, index)
                slot_of, position = {"a": 1, "b": 3}, {"a": 100, "b": 1500}
                trace = []
                for req, reset, scheduled, table, sampling in CHAIN:
                    launched.clear()
                    request = types.SimpleNamespace(req_id=req, scheduled=scheduled)
                    entry = (slot_of[req], reset, 11 + len(trace), position[req], sampling)
                    emitted = k._speculate(request, [entry], [np.asarray(table, np.int32)], sampling[0] > 0)
                    accepted = int(np.sum(np.asarray(emitted) >= 0))
                    position[req] += accepted
                    for value in (emitted, k.next_ids, k.control, k.draft_state.vector):  # replicated on every rank
                        shards = [np.asarray(s.data) for s in value.addressable_shards]
                        assert len(shards) == G and all(np.array_equal(shards[0], s) for s in shards), (len(trace), shards)
                    centres = k.ops.dsa_centres
                    if centres is not None:  # the same array on every rank
                        assert all(np.array_equal(np.asarray(centres)[0], c) for c in np.asarray(centres)), len(trace)
                    trace.append(dict(
                        emitted=np.asarray(emitted), ids=np.asarray(k.next_ids), control=np.asarray(k.control),
                        vector=np.asarray(k.draft_state.vector), slots=digest(k.ops.slots), pools=digest(k.ops.unified_pools),
                        centres=None if centres is None else digest([centres]), draft_pools=digest(k.draft.pools),
                        ops=tuple(name for name, _ in launched)))
                return trace, k


            # (rates, draft_fused_epilogue, index pools, decoder knobs)
            SEEDED, TAIL = dict(dsa_seeded_window=True), dict(dsa_seeded_window=True, verify_fused_tail=True)
            RATES = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
            COMBOS = [(None, False, False, {}), (None, True, False, {}), (RATES, False, False, {}), (RATES, True, False, {}),
                      (None, True, True, SEEDED), (RATES, False, True, TAIL), (None, True, True, TAIL),
                      (RATES, True, True, dict(TAIL, draft_epilogue_unroll=8)), (None, False, True, {})]
            ARMS = (("separate", {}), ("packet", dict(draft_verify_packet=True)), ("one", dict(draft_with_verify=True)),
                    ("one+packet", dict(draft_with_verify=True, draft_verify_packet=True)))
            for combo in os.environ.get("COMBOS", "0,1,2,3,4,5,6,7,8").split(","):
                rates, epilogue, index, extra = COMBOS[int(combo)]
                arms = {}
                for name, knobs in ARMS:
                    if name not in os.environ.get("ARMS", "separate,packet,one,one+packet").split(","):
                        continue
                    arms[name] = run(dict(knobs, draft_fused_epilogue=epilogue, **extra), rates, index)
                    print("ran", rates is not None, epilogue, index, extra, name, flush=True)
                reference, _ = arms["separate"]
                counts = [int(np.sum(np.asarray(s["emitted"]) >= 0)) - 1 for s in reference]
                assert len(set(counts)) >= 3, counts  # accepted counts vary along the chain
                for name, (trace, k) in arms.items():
                    for i, (want, got) in enumerate(zip(reference, trace, strict=True)):
                        for field in ("emitted", "ids", "control", "vector"):
                            assert same(want[field], got[field]), (rates, epilogue, index, name, i, field)
                        for field in ("slots", "pools", "centres", "draft_pools"):
                            assert want[field] == got[field], (rates, epilogue, index, name, i, field)
                    programs = [s["ops"] for s in trace]
                    if name.startswith("one"):
                        # One step program: the verify-and-draft op (after the splice / enter ops when they run).
                        assert all(p[-1] == "pallas::k3_decode_verify8_draft" for p in programs), programs
                        assert all("pallas::k3_decode_verify8" not in p and "pallas::k3_dspark_mla_draft_single" not in p
                                   and "pallas::k3_verify_metadata_1" not in p for p in programs), programs
                        assert k.stats["packets"] == 0, k.stats
                    else:
                        assert all(p[-2:] == ("pallas::k3_decode_verify8", "pallas::k3_dspark_mla_draft_single")
                                   for p in programs), programs
                    print(rates is not None, epilogue, index, extra, name, "steps", len(trace), "counts", counts, "stats",
                          k.stats, "programs", sum(len(p) for p in programs), flush=True)
                # The pools, states and centres moved along the chain (the comparison is not vacuous).
                for field in ("slots", "pools", "draft_pools") + (("centres",) if reference[0]["centres"] else ()):
                    assert len({s[field] for s in reference}) > len(CHAIN) // 2, field
                assert (reference[0]["centres"] is not None) == bool(index and extra.get("dsa_seeded_window")), extra

            # The one op's compilation call (KimiDecoder._warm_steps'; PART=warm checks which programs compile): operands as
            # the op takes them (the window centres after the pools), the verify's compiler options, the KDA banks, pools,
            # centres and draft pools donated (the separate verify's the same); it is the op verify_and_draft runs.
            for index in (False, True):
                k = build(dict(draft_with_verify=True, draft_fused_epilogue=True, **(SEEDED if index else {})), None, index)
                (call,) = [c for c in k.ops.compilation_calls(k.layout, k.sub_base, (1, 2, 3), True, (), None, draft=k.draft)
                           if c.op.op_name == "pallas::k3_decode_verify8_draft"]
                ins, outs, donated = call.op.specs
                first, persistent = len(k.ops.names), 2 + len(k.ops.unified_pools) + len(k.ops.centre_operands())
                first_pool = first + persistent + len(k.draft.names) - len(k.draft.target_weights)
                assert persistent == 4 + (IDX + 1 if index else 0) and call.options == DecodeOps._compiler_options(8)
                assert call.op is k.ops._build_verify_draft(k.draft, (1, 2, 3), k.layout, None), call.op
                assert len(call.operands) == len(ins), (len(call.operands), len(ins))
                assert call.operands[first + persistent - 1] is (k.ops.dsa_centres if index else k.ops.unified_pools[-1])
                assert all(call.operands[first_pool + j] is p for j, p in enumerate(k.draft.pools))
                assert [tuple(t.shape) for t in call.operands[-4:]] == [
                    (1, 128), (3 * ROWS + 1 + ROWS * PAGES,), (k.layout.size,), (1,)], call.operands[-4:]
                assert donated == tuple(range(first, first + persistent)) + tuple(range(first_pool, first_pool + L)), donated
                assert len(outs) == persistent + 1 + L + 3 and ("'centres'" in call.op.export_key) == index
                assert k.ops._build_verify((1, 2, 3), k.layout, None).specs[2] == donated[:persistent]
            print("OK")
        if PART == "warm":
            # KimiDecoder._warm_steps (one request, seven drafts; first-step drafts off / on) with the knob off and on, then a
            # serving chain: the precompiled programs, the programs the warm-up runs and every launch's operand history (each
            # operand's producing op output, else the host; torch_tpu compiles an op per operand history) against serving's.
            from sglang_tpu.k3_programs import compilation
            compiled = []
            compilation.compile_ops = compiled.extend
            export_cache = types.ModuleType("sglang_tpu.k3_programs.export_cache")
            export_cache.prepare_distributed = lambda calls: None
            sys.modules["sglang_tpu.k3_programs.export_cache"] = export_cache
            sync = types.ModuleType("torch_tpu._internal.sync")
            sync.synchronize = lambda *a, **k: None
            sys.modules["torch_tpu._internal.sync"] = sync

            decode.VOCAB = decode_ops.VOCAB = 32 * VL  # grammar masks over the stand-in vocabulary (32 shards' words)
            TAIL = dict(dsa_seeded_window=True, verify_fused_tail=True)
            ONE, SINGLE, DRAFT = ("pallas::k3_decode_verify8_draft", "pallas::k3_decode_verify8",
                                  "pallas::k3_dspark_mla_draft_single")
            prefill_aux = register_op(  # stands in for the prefill's first-step aux export (another op's output)
                "pallas::k3_prefill_first_aux", mesh,
                lambda x: (((x[0] + jnp.arange(3 * ROWS * H).reshape(3, ROWS, H)) % 64).astype(jnp.bfloat16),),
                (P(),), (P(),), export_key="first_aux")

            def warmed(knobs, first_aux):
                compiled.clear(), launched.clear(), source.clear(), kept.clear()
                k = build(dict(knobs, draft_fused_epilogue=True, **TAIL), None, index=True)
                k.layout = k.control = k._zero = None
                k._modes, k._mode_rows, k._pages, k._ring, k._compiled_layout, k.rank = {}, {}, PAGES, False, None, 1
                k.spec_alone, k.spec_batches, k.spec_shapes, k.groups = True, (), (), 1
                k.first_aux = {} if first_aux else None
                k._predraft_logits = jnp.zeros((G, ROWS, VL), jnp.float32)  # (the decoder's, with the mesh's leading axis)
                k._warm_steps()
                k._warm_steps()  # warmup() runs the steps twice (around a KDA import)
                return k, list(launched), [c.op.op_name for c in compiled]

            def serve(k):
                # Serving after the warm-up: new requests (first-step drafts when on), chained steps (mode 1 / mode 0),
                # a table that grows, a second request, the first back from its parked state, a reset, sampled rows.
                launched.clear()
                slot_of, position, token = {"a": 1, "b": 3}, {"a": 100, "b": 1500}, 11
                greedy, sampled = (0.0, 1.0, 0, 0), (0.7, 0.9, 20, 3)
                chain = [("a", True, 8, [1, 2], greedy), ("a", False, 8, [1, 2], greedy), ("a", False, 5, [1, 2], greedy),
                         ("a", False, 8, [1, 2, 3], greedy), ("a", False, 8, [1, 2, 3], greedy),
                         ("b", True, 8, [4, 5, 6], greedy), ("b", False, 8, [4, 5, 6], sampled),
                         ("a", False, 8, [1, 2, 3], greedy), ("a", False, 8, [1, 2, 3], greedy),
                         ("a", True, 8, [3, 1, 2], greedy), ("a", False, 8, [3, 1, 2], greedy)]
                for req, reset, scheduled, table, sampling in chain:
                    if reset and k.first_aux is not None:
                        k.first_aux[req] = prefill_aux(jnp.full((1,), token, jnp.int32))[0]
                    request = types.SimpleNamespace(req_id=req, scheduled=scheduled, computed=position[req],
                                                    num_prompt_tokens=position[req], token_ids=np.arange(position[req]) % V)
                    entry = (slot_of[req], reset, token, position[req], sampling)
                    emitted = np.asarray(k._speculate(request, [entry], [np.asarray(table, np.int32)], sampling[0] > 0))
                    position[req] += int(np.sum(emitted >= 0))
                    token = int(emitted[0, np.sum(emitted >= 0) - 1])
                return list(launched)

            CONFIGS = {"separate": ({}, False), "one": (dict(draft_with_verify=True), False), "separate+first": ({}, True),
                       "one+first": (dict(draft_with_verify=True, draft_verify_packet=True), True)}
            for name in os.environ.get("WARM", "separate,one,separate+first,one+first").split(","):
                knobs, first = CONFIGS[name]
                k, warm, precompiled = warmed(knobs, first)
                served = serve(k)
                assert (k.stats["predrafts"] >= 2) == first, k.stats
                # Histories: each operand's producing op output (strict) or producer kind (the host, a global or a sharded
                # op; hardware recompiled on these: KimiDecoder._warm_steps' control from the splice or the draft op).
                report, kind = {}, lambda h: tuple(kinds[x.rsplit(":", 1)[0]] for x in h)
                for op in (ONE, SINGLE, DRAFT):
                    warm_h, serve_h = {h for n, h in warm if n == op}, {h for n, h in served if n == op}
                    if warm_h or serve_h:
                        report[op] = dict(precompiled=precompiled.count(op), warm_launches=sum(n == op for n, _ in warm),
                                          serve_launches=sum(n == op for n, _ in served), warm_histories=len(warm_h),
                                          serve_histories=len(serve_h), unwarmed=len(serve_h - warm_h),
                                          warm_kinds=len(set(map(kind, warm_h))),
                                          unwarmed_kinds=len(set(map(kind, serve_h)) - set(map(kind, warm_h))))
                print(name, "precompiled", len(precompiled), "warm launches", len(warm), "served", len(served), k.stats)
                for op, row in report.items():
                    print("   ", op, row, flush=True)
                # The step programs precompiled are those the warm-up or serving runs, once each.
                assert all(row["precompiled"] == 1 for row in report.values()), (name, report)
                assert set(precompiled) & {ONE, SINGLE, DRAFT} == set(report), (name, precompiled)
                # Serving runs the step programs on warmed operand histories (producer kinds) only.
                assert all(row["unwarmed_kinds"] == 0 for row in report.values()), (name, report)
                if name.startswith("one"):
                    # The one op runs on histories (producing ops) the warm-up ran it on; no separate verify runs; the
                    # single draft op runs alone for first-step drafts only.
                    assert report[ONE]["unwarmed"] == 0 and SINGLE not in report and (DRAFT in report) == first, report
                    assert not first or report[DRAFT]["serve_launches"] == k.stats["predrafts"], (report, k.stats)
                    assert k.stats["packets"] == 0, k.stats  # the one op packs its own operand
                else:
                    assert ONE not in report, report
            print("OK")
    """

    def _run(self, **env_vars):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), **env_vars)
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=2400,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )

    def test_one_op_matches_the_verify_then_the_draft(self):
        self._run(PART="chain", COMBOS="2,6", ARMS="separate,one,one+packet")

    def test_warm_up_compiles_what_serving_runs(self):
        self._run(PART="warm", WARM="one,one+first")


class TestB16OneHotStaging(CustomTestCase):
    """KernelConfig.b16_one_hot_staging: the B16 routed-expert stream stages
    its compact rows with a one-hot product instead of a scalar row scan. In
    Pallas interpret mode both give bitwise-identical outputs over routes that
    take every pair and tail path, with NaN/inf latents in the rows without
    routes (null slots; the one-hot path zeroes them) and signed zeros in
    routed rows."""

    SCRIPT = r"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
from types import SimpleNamespace
import jax, jax.numpy as jnp, ml_dtypes, numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk

B, C, K, W, OUT, SLOTS = 16, 8, 512, 256, 128, 4
NAMES = ("gate_up", "gate_up_scales", "down", "down_scales")


def stream(one_hot, latents, experts, probabilities, counts, totals, weights):
    dmk.B16_ONE_HOT_STAGING = one_hot  # read when the stream is traced
    k = experts.shape[1]

    def kernel(latent_ref, expert_ref, probability_ref, count_ref, total_ref, *refs):
        hbm, (out_ref, source_ref), vmem = refs[:4], refs[4:6], refs[6:10]
        gate_up_acc, down_acc, processed, first, second, semaphores = refs[10:]
        source_ref[...] = latent_ref[...]

        def copy_expert(expert, slot, *, gate_up=True, down=True):
            wanted = (gate_up, gate_up, down, down)
            return SimpleNamespace(**{
                name: pltpu.make_async_copy(hbm[i].at[expert], vmem[i].at[slot],
                                            semaphores.at[slot, i]) if wanted[i] else None
                for i, name in enumerate(NAMES)})

        selected, route_counts = expert_ref[:, :k], count_ref[:, 0]
        dmk._start_initial_expert_weight_copies(
            selected, route_counts, copy_expert, slot_count=SLOTS, already_started=0,
            group_pending_routes=True)
        dmk._run_routed_expert_stream(
            source_ref, selected, probability_ref[:, :k], route_counts, total_ref[:, 0], *vmem,
            out_ref, gate_up_acc, down_acc, processed, first, second, copy_expert,
            expert_input_size=K, expert_width=W, expert_output_size=OUT,
            group_pending_routes=True)

    vmem, hbm = pl.BlockSpec(memory_space=pltpu.VMEM), pl.BlockSpec(memory_space=pl.ANY)
    wide = lambda x: np.pad(x.reshape(B, -1), ((0, 0), (0, 128 - x.size // B)), mode="edge")
    out, source = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((B, OUT), jnp.float32),
                   jax.ShapeDtypeStruct(latents.shape, jnp.bfloat16)),
        in_specs=[vmem] * 5 + [hbm] * 4,
        out_specs=(vmem, vmem),
        scratch_shapes=[*(pltpu.VMEM((SLOTS, *w.shape[1:]), w.dtype) for w in weights),
                        pltpu.VMEM((2 * C, 4 * W), jnp.float32), pltpu.VMEM((C, OUT), jnp.float32),
                        pltpu.VMEM((B, 128), jnp.int32), pltpu.VMEM((C, K), jnp.float32),
                        pltpu.VMEM((C, K), jnp.float32), pltpu.SemaphoreType.DMA((SLOTS, 4))],
        interpret=pltpu.InterpretParams(),
    )(latents, wide(experts), wide(probabilities), wide(counts), wide(totals), *weights)
    return np.asarray(out), np.asarray(source)


def case(seed, live, local_experts, k):
    rng = np.random.default_rng(seed)
    latents = rng.standard_normal((B, K + 128)).astype(np.float32)
    rows = rng.choice(B, live, replace=False)
    dead = np.setdiff1d(np.arange(B), rows)
    if dead.size:
        latents[dead[:, None], rng.integers(0, K, 3)[None, :]] = (np.nan, np.inf, -np.inf)
        latents[dead[0]] = np.nan
    for r in rows[:3]:
        latents[r, rng.integers(0, K, 6)] = (0.0, -0.0, -0.0, 1e-30, -1e-30, 3e4)
    popularity = rng.dirichlet(np.full(local_experts, 0.6))
    experts, probabilities = np.zeros((B, k), np.int32), np.zeros((B, k), np.float32)
    counts = np.zeros(B, np.int32)
    for r in rows:
        n = counts[r] = rng.integers(1, k + 1)
        experts[r, :n] = rng.choice(local_experts, n, replace=False, p=popularity)
        probabilities[r, :n] = rng.uniform(0.02, 0.2, n)
    totals = np.where(counts > 0, probabilities.sum(1) + rng.uniform(0.5, 1, B), 1)
    bits = lambda *shape: rng.integers(0, 2**32, shape, dtype=np.uint64).astype(np.uint32)
    exponents = lambda *shape: rng.integers(119, 123, shape).astype(np.uint8)
    weights = (bits(local_experts, K // 8, 2 * W), exponents(local_experts, K // 32, 2 * W),
               bits(local_experts, W // 8, OUT), exponents(local_experts, W // 32, OUT))
    return (latents.astype(ml_dtypes.bfloat16), experts, probabilities, counts,
            totals.astype(np.float32), weights)


def paths(experts, counts):
    # Distinct experts run in pairs in id order; their row counts pick the path.
    routed = (np.arange(experts.shape[1]) < counts[:, None])
    sizes = [n for e in range(experts.max() + 1) if (n := int(np.sum(routed & (experts == e))))]
    kinds = set()
    for i in range(0, len(sizes), 2):
        pair = sizes[i:i + 2]
        size = "pass" if max(pair) > C else "single" if max(pair) == 1 else "multi"
        kinds.add(f"{size}_{'pair' if len(pair) == 2 else 'tail'}")
    return kinds


covered = set()
for seed, live, local_experts, k in ((11, 13, 9, 4), (10, 14, 5, 3), (1, 3, 5, 3)):
    latents, experts, probabilities, counts, totals, weights = case(seed, live, local_experts, k)
    covered |= paths(experts, counts)
    old, old_source = stream(False, latents, experts, probabilities, counts, totals, weights)
    new, new_source = stream(True, latents, experts, probabilities, counts, totals, weights)
    routed = counts > 0
    assert np.array_equal(old.view(np.uint32), new.view(np.uint32)), seed
    assert np.all(np.isfinite(new)) and np.all(np.any(old[routed] != 0, axis=1)), seed
    assert np.array_equal(old_source.view(np.uint16), latents.view(np.uint16)), seed
    assert np.array_equal(new_source[routed].view(np.uint16), latents[routed].view(np.uint16))
    assert np.all(new_source[~routed, :K].astype(np.float32) == 0), seed
assert covered == {f"{s}_{p}" for s in ("single", "multi", "pass") for p in ("pair", "tail")}, covered
print("OK")
"""

    def test_matches_the_row_scan(self):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", self.SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )


class TestSharedExpertPasses(CustomTestCase):
    """KernelConfig.shared_expert_passes: the routed-expert stream with compact
    tiles (B16, B32) runs an iteration's noncompact pair or tail (experts with
    more rows than a compact tile) from one copy of the eight-row passes, a
    loop over its one or two experts, instead of three unrolled copies. In
    Pallas interpret mode both give bitwise-identical outputs over routes that
    take every pair and tail path, with and without the SMEM expert schedule
    and B16 one-hot staging (NaN/inf latents in rows without routes, signed
    zeros in routed rows)."""

    SCRIPT = r"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
from types import SimpleNamespace
import jax, jax.numpy as jnp, ml_dtypes, numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk

C, K, W, OUT, SLOTS = 8, 512, 256, 128, 4
NAMES = ("gate_up", "gate_up_scales", "down", "down_scales")


def stream(B, shared, scheduled, one_hot, latents, experts, probabilities, counts, totals, weights):
    dmk.SHARED_EXPERT_PASSES = shared  # read when the stream is traced
    dmk.B16_ONE_HOT_STAGING = one_hot
    k = experts.shape[1]
    rows = dmk._expert_accumulator_rows(B, jnp.uint32, group_pending_routes=True)

    def kernel(latent_ref, expert_ref, probability_ref, count_ref, total_ref, *refs):
        hbm, (out_ref, source_ref), vmem = refs[:4], refs[4:6], refs[6:10]
        gate_up_acc, down_acc, processed, first, second, semaphores, table, table_semaphore = refs[10:]
        source_ref[...] = latent_ref[...]

        def copy_expert(expert, slot, *, gate_up=True, down=True):
            wanted = (gate_up, gate_up, down, down)
            return SimpleNamespace(**{
                name: pltpu.make_async_copy(hbm[i].at[expert], vmem[i].at[slot],
                                            semaphores.at[slot, i]) if wanted[i] else None
                for i, name in enumerate(NAMES)})

        selected, route_counts = expert_ref[:, :k], count_ref[:, 0]
        dmk._start_initial_expert_weight_copies(
            selected, route_counts, copy_expert, slot_count=SLOTS, already_started=0,
            group_pending_routes=True)
        schedule = None
        if scheduled:
            schedule = dmk._expert_schedule(processed.at[:8], table, table_semaphore)
            dmk._start_expert_schedule(schedule, selected, route_counts)
        dmk._run_routed_expert_stream(
            source_ref, selected, probability_ref[:, :k], route_counts, total_ref[:, 0], *vmem,
            out_ref, gate_up_acc, down_acc, processed, first, second, copy_expert,
            expert_input_size=K, expert_width=W, expert_output_size=OUT,
            group_pending_routes=True, expert_schedule=schedule)

    vmem, hbm = pl.BlockSpec(memory_space=pltpu.VMEM), pl.BlockSpec(memory_space=pl.ANY)
    wide = lambda x: np.pad(x.reshape(B, -1), ((0, 0), (0, 128 - x.size // B)), mode="edge")
    out, source = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((B, OUT), jnp.float32),
                   jax.ShapeDtypeStruct(latents.shape, jnp.bfloat16)),
        in_specs=[vmem] * 5 + [hbm] * 4,
        out_specs=(vmem, vmem),
        scratch_shapes=[*(pltpu.VMEM((SLOTS, *w.shape[1:]), w.dtype) for w in weights),
                        pltpu.VMEM((rows[0], 4 * W), jnp.float32),
                        pltpu.VMEM((rows[1], OUT), jnp.float32),
                        pltpu.VMEM((B, 128), jnp.int32), pltpu.VMEM((C, K), jnp.float32),
                        pltpu.VMEM((C, K), jnp.float32), pltpu.SemaphoreType.DMA((SLOTS, 4)),
                        pltpu.SMEM((8, 128), jnp.int32), pltpu.SemaphoreType.DMA],
        interpret=pltpu.InterpretParams(),
    )(latents, wide(experts), wide(probabilities), wide(counts), wide(totals), *weights)
    return np.asarray(out), np.asarray(source)


def case(B, seed, live, local_experts, k):
    rng = np.random.default_rng(seed)
    latents = rng.standard_normal((B, K + 128)).astype(np.float32)
    rows = rng.choice(B, live, replace=False)
    dead = np.setdiff1d(np.arange(B), rows)
    if dead.size:
        latents[dead[:, None], rng.integers(0, K, 3)[None, :]] = (np.nan, np.inf, -np.inf)
        latents[dead[0]] = np.nan
    for r in rows[:3]:
        latents[r, rng.integers(0, K, 6)] = (0.0, -0.0, -0.0, 1e-30, -1e-30, 3e4)
    popularity = rng.dirichlet(np.full(local_experts, 0.6))
    experts, probabilities = np.zeros((B, k), np.int32), np.zeros((B, k), np.float32)
    counts = np.zeros(B, np.int32)
    for r in rows:
        n = counts[r] = rng.integers(1, k + 1)
        experts[r, :n] = rng.choice(local_experts, n, replace=False, p=popularity)
        probabilities[r, :n] = rng.uniform(0.02, 0.2, n)
    totals = np.where(counts > 0, probabilities.sum(1) + rng.uniform(0.5, 1, B), 1)
    bits = lambda *shape: rng.integers(0, 2**32, shape, dtype=np.uint64).astype(np.uint32)
    exponents = lambda *shape: rng.integers(119, 123, shape).astype(np.uint8)
    weights = (bits(local_experts, K // 8, 2 * W), exponents(local_experts, K // 32, 2 * W),
               bits(local_experts, W // 8, OUT), exponents(local_experts, W // 32, OUT))
    return (latents.astype(ml_dtypes.bfloat16), experts, probabilities, counts,
            totals.astype(np.float32), weights)


def paths(experts, counts):
    # Distinct experts run in pairs in id order; their row counts pick the path.
    routed = (np.arange(experts.shape[1]) < counts[:, None])
    sizes = [n for e in range(experts.max() + 1) if (n := int(np.sum(routed & (experts == e))))]
    kinds = set()
    for i in range(0, len(sizes), 2):
        pair = sizes[i:i + 2]
        size = "pass" if max(pair) > C else "single" if max(pair) == 1 else "multi"
        kinds.add(f"{size}_{'pair' if len(pair) == 2 else 'tail'}")
    return kinds


# (batch, schedule, one-hot staging): B16 as before and as served; B32 (no schedule).
CASES = {(16, False, False): ((11, 13, 9, 4), (10, 14, 5, 3), (1, 3, 5, 3), (22, 16, 4, 4)),
         (16, True, True): ((11, 13, 9, 4), (10, 14, 5, 3), (1, 3, 5, 3), (22, 16, 4, 4)),
         (32, False, False): ((37, 20, 7, 3), (32, 28, 6, 4))}
for (B, scheduled, one_hot), cases in CASES.items():
    covered = set()
    for seed, live, local_experts, k in cases:
        latents, experts, probabilities, counts, totals, weights = case(B, seed, live, local_experts, k)
        covered |= paths(experts, counts)
        inputs = (latents, experts, probabilities, counts, totals, weights)
        old, old_source = stream(B, False, scheduled, one_hot, *inputs)
        new, new_source = stream(B, True, scheduled, one_hot, *inputs)
        routed = counts > 0
        assert np.array_equal(old.view(np.uint32), new.view(np.uint32)), (B, seed)
        assert np.array_equal(old_source.view(np.uint16), new_source.view(np.uint16)), (B, seed)
        assert np.all(np.isfinite(new)) and np.all(np.any(old[routed] != 0, axis=1)), (B, seed)
    want = {"pass_pair", "pass_tail", "multi_pair"}
    if B == 16:
        want = {f"{s}_{p}" for s in ("single", "multi", "pass") for p in ("pair", "tail")}
    assert want <= covered, (B, covered)
print("OK")
"""

    def test_one_copy_matches_the_unrolled_passes(self):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", self.SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )


class TestB8CompactExperts(CustomTestCase):
    """KernelConfig.b8_compact_experts: the B8 routed-expert stream runs an
    expert pair of at most 4 rows an expert (not two singletons) on compact
    tiles (gate/up narrowed to 4 rows a tile) instead of the 8-row pair. In
    Pallas interpret mode (XLA:CPU
    without FMA contraction, as the TPU's vector unit) both give bitwise-
    identical outputs on random MXFP4 experts for every pair mix of 1-4 rows,
    pairs with 5-8 rows next to 1-8, tails of 1-8 rows, odd and even expert
    counts and no experts, with and without the SMEM expert schedule and
    b8_singleton_select. Latents and scales span a wide range, so the FP32
    sums round. Rows without routes hold NaN/inf latents in half of the cases
    (null slots; the compact path zeroes them, so their outputs are zero where
    the eight-row pairs leave NaN) and finite values in the rest (every row is
    bitwise equal); routed rows hold signed zeros and subnormals."""

    SCRIPT = r"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_cpu_max_isa=AVX --xla_cpu_enable_platform_dependent_math=false"
from types import SimpleNamespace
import jax, jax.numpy as jnp, ml_dtypes, numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk

B, K, W, OUT, SLOTS, E, TOP = 8, 1024, 768, 128, 4, 24, 16
NAMES = ("gate_up", "gate_up_scales", "down", "down_scales")


def stream(compact, scheduled, singleton_select):
    dmk.B8_COMPACT_EXPERTS = compact  # read when the stream is traced
    dmk.B8_SINGLETON_SELECT = singleton_select
    rows = dmk._expert_accumulator_rows(B, jnp.uint32, group_pending_routes=True)
    capacity = dmk._compact_rows(B)
    assert (capacity, rows) == ((8, (16, 8)) if compact else (0, (8, 8))), (capacity, rows)
    sources = ((pltpu.VMEM((capacity, K), jnp.float32),) * 2 if capacity
               else (pltpu.VMEM((1, K), jnp.bfloat16),) * 2)

    def kernel(latent_ref, expert_ref, probability_ref, count_ref, total_ref, *refs):
        hbm, (out_ref, source_ref), vmem = refs[:4], refs[4:6], refs[6:10]
        gate_up_acc, down_acc, processed, first, second, semaphores, table, table_semaphore = refs[10:]
        source_ref[...] = latent_ref[...]

        def copy_expert(expert, slot, *, gate_up=True, down=True):
            wanted = (gate_up, gate_up, down, down)
            return SimpleNamespace(**{
                name: pltpu.make_async_copy(hbm[i].at[expert], vmem[i].at[slot],
                                            semaphores.at[slot, i]) if wanted[i] else None
                for i, name in enumerate(NAMES)})

        selected, route_counts = expert_ref[:, :TOP], count_ref[:, 0]
        dmk._start_initial_expert_weight_copies(
            selected, route_counts, copy_expert, slot_count=SLOTS, already_started=0,
            group_pending_routes=True)
        schedule = None
        if scheduled:
            schedule = dmk._expert_schedule(processed.at[:8], table, table_semaphore)
            dmk._start_expert_schedule(schedule, selected, route_counts)
        dmk._run_routed_expert_stream(
            source_ref, selected, probability_ref[:, :TOP], route_counts, total_ref[:, 0], *vmem,
            out_ref, gate_up_acc, down_acc, processed, first, second, copy_expert,
            expert_input_size=K, expert_width=W, expert_output_size=OUT,
            group_pending_routes=True, expert_schedule=schedule)

    vmem, hbm = pl.BlockSpec(memory_space=pltpu.VMEM), pl.BlockSpec(memory_space=pl.ANY)
    return jax.jit(pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((B, OUT), jnp.float32),
                   jax.ShapeDtypeStruct((B, K + 128), jnp.bfloat16)),
        in_specs=[vmem] * 5 + [hbm] * 4,
        out_specs=(vmem, vmem),
        scratch_shapes=[pltpu.VMEM((SLOTS, K // 8, 2 * W), jnp.uint32),
                        pltpu.VMEM((SLOTS, K // 32, 2 * W), jnp.uint8),
                        pltpu.VMEM((SLOTS, W // 8, OUT), jnp.uint32),
                        pltpu.VMEM((SLOTS, W // 32, OUT), jnp.uint8),
                        pltpu.VMEM((rows[0], 4 * W), jnp.float32),
                        pltpu.VMEM((rows[1], OUT), jnp.float32),
                        pltpu.VMEM((B, 128), jnp.int32), *sources,
                        pltpu.SemaphoreType.DMA((SLOTS, 4)),
                        pltpu.SMEM((8, 128), jnp.int32), pltpu.SemaphoreType.DMA],
        interpret=pltpu.InterpretParams()))


def case(seed, sizes, null_rows, nonfinite):
    # Experts with the given row counts (rows picked among the others than
    # null_rows, fewest routes first); each row's slots in random order.
    rng = np.random.default_rng(seed)
    live = np.setdiff1d(np.arange(B), np.asarray(null_rows, int))
    experts = np.sort(rng.choice(E, len(sizes), replace=False))
    routes = [[] for _ in range(B)]
    for e, n in zip(experts, rng.permutation(np.asarray(sizes, int))):
        for r in sorted(rng.permutation(live), key=lambda r: len(routes[r]))[:n]:
            routes[r].append(e)
    sel, prob, cnt = (np.zeros((B, 128), t) for t in (np.int32, np.float32, np.int32))
    for r in range(B):
        picks = rng.permutation(np.asarray(routes[r], np.int32))
        sel[r, :picks.size], cnt[r] = picks, picks.size
        prob[r, :picks.size] = rng.uniform(0.01, 1.0, picks.size)
    total = np.broadcast_to(prob.sum(1, keepdims=True) + rng.uniform(0.5, 2.0, (B, 1)), (B, 128))
    # A wide dynamic range (latents and scales): FP32 sums round, so their
    # order shows.
    latents = rng.standard_normal((B, K + 128)) * 2.0 ** rng.integers(-12, 13, (B, K + 128))
    latents = latents.astype(np.float32)
    routed = cnt[:, 0] > 0
    dead = np.flatnonzero(~routed)
    if dead.size and nonfinite:
        latents[dead[:, None], rng.integers(0, K, 3)[None, :]] = (np.nan, np.inf, -np.inf)
        latents[dead[0]] = np.nan
    for r in np.flatnonzero(routed)[:3]:
        latents[r, rng.integers(0, K, 6)] = (0.0, -0.0, -0.0, 1e-30, -1e-30, 3e4)
    bits = lambda *shape: rng.integers(0, 2**32, shape, dtype=np.uint64).astype(np.uint32)
    exponents = lambda *shape: rng.integers(112, 132, shape).astype(np.uint8)
    weights = (bits(E, K // 8, 2 * W), exponents(E, K // 32, 2 * W),
               bits(E, W // 8, OUT), exponents(E, W // 32, OUT))
    return (latents.astype(ml_dtypes.bfloat16), sel, prob, cnt,
            np.ascontiguousarray(total, np.float32), weights)


def paths(sel, cnt):
    # Distinct experts run in pairs in id order; their row counts pick the path.
    routed = np.arange(128)[None, :] < cnt[:, :1]
    sizes = [n for e in range(E) if (n := int(np.sum(routed & (sel == e))))]
    kinds = set()
    for i in range(0, len(sizes), 2):
        pair = sizes[i:i + 2]
        size = "single" if max(pair) == 1 else "compact" if max(pair) <= 4 else "full"
        kinds.add(f"{size}_{'pair' if len(pair) == 2 else 'tail'}")
        kinds |= {f"compact_pair_{min(pair)}_{max(pair)}"} if size == "compact" and len(pair) == 2 else set()
    return kinds


# Row counts of the used experts (each list is shuffled over its experts).
SIZES = [[2, 3], [1, 4], [4, 4], [1, 2, 3, 4, 2, 2, 3], [5, 1], [8, 4, 6, 2, 7], [1, 1, 3],
         [3, 1, 1, 2, 4, 4, 1, 1, 2], [8], [3], [1], [2, 8, 1, 5, 4, 3, 1, 6, 2, 2, 4],
         [4, 1, 2, 2, 1, 4, 3, 3, 1, 2, 4, 2], [7, 3, 2, 4, 1, 1, 4, 4, 2, 8, 3], [],
         [1, 2, 1, 3, 1, 4, 2, 1, 3, 1, 4, 1, 2, 3, 2], [3, 3], [4, 3, 4, 4, 3, 4]]
NULLS = [(), (5,), (0, 7), (), (3,), (), (1, 2), (), (), (6,), (), (4,), (), (2,), (0,), (),
         (7,), ()]
covered = set()
for scheduled, singleton_select in ((True, True), (False, False)):
    old, new = stream(False, scheduled, singleton_select), stream(True, scheduled, singleton_select)
    for index, (sizes, nulls) in enumerate(zip(SIZES, NULLS)):
        nonfinite = index % 2 == 0
        latents, sel, prob, cnt, total, weights = case(1000 + index, sizes, nulls, nonfinite)
        covered |= paths(sel, cnt)
        a, a_source = (np.asarray(x) for x in old(latents, sel, prob, cnt, total, *weights))
        b, b_source = (np.asarray(x) for x in new(latents, sel, prob, cnt, total, *weights))
        routed = cnt[:, 0] > 0
        label = (scheduled, singleton_select, index)
        assert np.array_equal(a[routed].view(np.uint32), b[routed].view(np.uint32)), label
        assert nonfinite or np.array_equal(a.view(np.uint32), b.view(np.uint32)), label
        assert np.all(b[~routed].view(np.uint32) == 0), label
        assert np.all(np.isfinite(b)) and np.all(np.any(b[routed] != 0, axis=1)), label
        assert np.array_equal(a_source.view(np.uint16), latents.view(np.uint16)), label
        assert np.array_equal(b_source[routed].view(np.uint16), latents[routed].view(np.uint16))
        assert np.all(b_source[~routed, :K].astype(np.float32) == 0), label
want = {f"{s}_{p}" for s in ("single", "compact", "full") for p in ("pair", "tail")} - {"compact_tail"}
want |= {f"compact_pair_{i}_{j}" for j in range(2, 5) for i in range(1, j + 1)}
assert want <= covered, sorted(want - covered)
print("OK")
"""

    ROWS_SCRIPT = r"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_cpu_max_isa=AVX --xla_cpu_enable_platform_dependent_math=false"
import jax, jax.numpy as jnp, ml_dtypes, numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk

B, C, ACTIVE, K, W, OUT = 8, 8, 4, 1024, 768, 256


def kernel(source_ref, first_ref, second_ref, middle_ref, compact_middle_ref, gu_ref, gus_ref,
           down_ref, ds_ref, full_gu, compact_gu, full_down, compact_down):
    dmk._paired_expert_gate_up(source_ref, gu_ref, gus_ref, full_gu, 0, 1, K)
    dmk._paired_expert_gate_up(first_ref, gu_ref, gus_ref, compact_gu, 0, 1, K,
                               second_source=second_ref, active_capacity=ACTIVE)
    for middle, out in ((middle_ref, full_down), (compact_middle_ref, compact_down)):
        dmk._quantized_expert_dot(middle, down_ref.at[0], ds_ref.at[0], out, W, OUT,
                                  block_major=True, dot_tile=512)


call = jax.jit(pl.pallas_call(
    kernel,
    out_shape=(jax.ShapeDtypeStruct((B, 4 * W), jnp.float32),
               jax.ShapeDtypeStruct((2 * C, 4 * W), jnp.float32),
               jax.ShapeDtypeStruct((B, OUT), jnp.float32),
               jax.ShapeDtypeStruct((C, OUT), jnp.float32)),
    interpret=pltpu.InterpretParams()))
view = lambda x: x.view(np.uint32)
for seed in range(8):
    rng = np.random.default_rng(seed)
    wide = lambda *shape: rng.standard_normal(shape) * 2.0 ** rng.integers(-12, 13, shape)
    source = wide(B, K).astype(ml_dtypes.bfloat16)
    rows = [np.sort(rng.choice(B, rng.integers(1, ACTIVE + 1), replace=False)) for _ in range(2)]
    tiles = np.zeros((2, C, K), np.float32)
    for tile, picked in zip(tiles, rows):
        tile[:picked.size] = source[picked].astype(np.float32)
    bits = lambda *shape: rng.integers(0, 2**32, shape, dtype=np.uint64).astype(np.uint32)
    exponents = lambda *shape: rng.integers(112, 132, shape).astype(np.uint8)
    gu, gus = bits(2, K // 8, 2 * W), exponents(2, K // 32, 2 * W)
    down, ds = bits(1, W // 8, OUT), exponents(1, W // 32, OUT)
    middle = wide(B, W).astype(ml_dtypes.bfloat16).astype(np.float32)
    compact_middle = np.zeros((C, W), np.float32)
    compact_middle[:rows[0].size] = middle[rows[0]]
    full_gu, compact_gu, full_down, compact_down = (np.asarray(x) for x in call(
        source, tiles[0], tiles[1], middle, compact_middle, gu, gus, down, ds))
    first, second = rows
    assert np.array_equal(view(compact_gu[:first.size, :2 * W]), view(full_gu[first, :2 * W])), seed
    assert np.array_equal(view(compact_gu[C:C + second.size, 2 * W:]), view(full_gu[second, 2 * W:]))
    assert not np.any(compact_gu[:C, 2 * W:]) and not np.any(compact_gu[C:, :2 * W]), seed
    assert not np.any(compact_gu[ACTIVE:C]) and not np.any(compact_gu[C + ACTIVE:]), seed
    assert np.array_equal(view(compact_down[:first.size]), view(full_down[first])), seed
    assert np.all(np.isfinite(full_gu)) and np.all(np.isfinite(full_down)), seed
print("OK")
"""

    def test_compact_products_match_the_eight_row_rows(self):
        """The compact pair's products row by row in FP32 (before any BF16
        rounding), on wide-range latents and scales: the paired gate/up of
        two compact tiles (second_source, register operands narrowed to 4
        rows a tile) against the 8-row paired gate/up, and the block-major
        down of the compact rows against the 8-row one."""
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", self.ROWS_SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )

    def test_compact_pairs_match_the_eight_row_pairs(self):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", self.SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )


class TestSingletonRowStaging(CustomTestCase):
    """A singleton route's row staging (``_stage_singleton_row``) in Pallas
    interpret mode: at 8 rows, ``KernelConfig.b8_singleton_select`` (one select
    over the row tile, no branch) stages the bits of the per-row copies for
    every active row and every BF16 bit pattern (signed zeros, subnormals,
    infinities, NaN payloads); other row counts keep the per-row copies."""

    def test_select_matches_the_row_copies(self):
        import jax
        import jax.numpy as jnp
        import ml_dtypes
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        width = 3584  # expert input columns of the [rows, 4096] latent tile
        # -0, +0, -/+ smallest subnormal, -/+ largest finite, -/+ infinity,
        # NaNs (quiet, signalling, payloads, negative), +/-1.
        special = np.array(
            [0x8000, 0x0000, 0x8001, 0x0001, 0xFF7F, 0x7F7F, 0xFF80, 0x7F80]
            + [0x7FC0, 0x7F81, 0x7FC1, 0xFFC0, 0xFFFF, 0x7FFF, 0x3F80, 0xBF80],
            np.uint16,
        )

        def staging(rows, row_select):
            def kernel(routes_ref, source_ref, out_ref):
                for case in range(rows):
                    dmk._stage_singleton_row(
                        source_ref,
                        jnp.any(routes_ref[case] != 0, axis=1),
                        out_ref.at[case],
                        expert_input_size=width,
                        block_select=False,
                        row_select=row_select,
                    )

            return pl.pallas_call(
                kernel,
                out_shape=jax.ShapeDtypeStruct((rows, 1, width), jnp.bfloat16),
                interpret=pltpu.InterpretParams(),
            )

        def inputs(rows):
            bits = rng.integers(0, 1 << 16, (rows, 4096), dtype=np.uint16)
            # Each row holds every special pattern; in each of those columns
            # the other rows hold other ones.
            for column in range(0, width, 448):
                for row in range(rows):
                    bits[row, column : column + 16] = np.roll(special, row)
            # Case r: row r is the only row with a route to the expert.
            routes = np.zeros((rows, rows, 16), np.int32)
            for row in range(rows):
                routes[row, row, rng.integers(0, 16)] = 1 + rng.integers(0, 128)
            return bits, routes

        def branches(call, *args):
            return str(jax.make_jaxpr(call)(*args)).count("cond[")

        rng = np.random.default_rng(0)
        bits, routes = inputs(8)
        source = bits.view(ml_dtypes.bfloat16)
        copy, select = staging(8, False), staging(8, True)
        self.assertEqual(branches(copy, routes, source), 8 * 8)
        self.assertEqual(branches(select, routes, source), 0)
        for call in (copy, select):
            staged = np.asarray(call(routes, source)).view(np.uint16)
            np.testing.assert_array_equal(staged[:, 0], bits[:, :width])
        # Other row counts keep the per-row copies.
        bits, routes = inputs(12)
        source = bits.view(ml_dtypes.bfloat16)
        self.assertEqual(branches(staging(12, True), routes, source), 12 * 12)


class TestAttentionHostBf16(CustomTestCase):
    """The token-scattered attention output projection (``_project_attention_2d``)
    at B8, B16 and B24 in Pallas interpret mode on 32 host devices (4 hosts x 4 rank
    pairs x 2 lanes), with stale NaN bytes wherever a round lands. With FP32
    host sums and with BF16 host sums (``KernelConfig.attention_host_bf16``)
    every rank's owned rows equal, bit for bit, the reduction's rounding points
    applied to the ranks' own FP32 partials: BF16 pair partials, FP32 sums of
    each host's four pairs in pair order, the host sums kept in FP32 or rounded
    to BF16, an FP32 sum of the four hosts in host order, a BF16 output. The four
    hosts' copies of every row are bitwise equal."""

    # The interpret harness (shared with TestAttentionRankRows): bitcast and
    # reshape views, ``project`` and the rounding-point reference.
    PRELUDE = r"""
import math, os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
os.environ["JAX_PLATFORMS"] = "cpu"
import jax, jax.numpy as jnp, ml_dtypes, numpy as np
from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc
from jax._src.pallas.mosaic.interpret import utils as interpret_utils
from jax._src.state import indexing, types as state_types
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, PartitionSpec as P
from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk

BF16 = ml_dtypes.bfloat16
H, HALF, W = 7168, 3584, 384


# TPU interpret mode reads and writes NDIndexer views only; give its loads,
# stores and DMAs the kernel's bitcast and reshape views too (TPU semantics: a
# 32-bit row r is 16-bit rows 2r and 2r + 1; dmk._PaddedBf16Rows is the leading
# rows of its backing).
def view_index(content, transforms):
    units = content.dtype.itemsize // 2
    index = np.arange(content.size * units).reshape(*content.shape, units)
    dtype = content.dtype
    for t in transforms:
        if isinstance(t, indexing.NDIndexer):
            index = index[interpret_utils.to_range((t,))]
        elif isinstance(t, state_types.BitcastTransform):
            dtype = np.dtype(t.dtype)
            new, (*lead, rows, cols, old) = dtype.itemsize // 2, index.shape
            if new < old:
                index = index.swapaxes(-1, -2).reshape(*lead, rows * old, cols, 1)
            elif new > old:
                index = index.reshape(*lead, rows // new, new, cols).swapaxes(-1, -2)
        elif isinstance(t, state_types.ReshapeTransform):
            shape = tuple(t.shape)
            if math.prod(shape) == math.prod(index.shape[:-1]):
                index = index.reshape(*shape, index.shape[-1])
            else:
                index = index.reshape(-1, shape[-1], index.shape[-1])[: shape[0]]
        else:
            raise NotImplementedError(t)
    return index, dtype


def access(device_id, local_core_id, memory_space, buffer_id, transforms, val=None):
    memory = ipc._get_shared_memory()
    space = ipc.TPU_MEMORY_SPACE_NAMES[int(memory_space)]
    core = ipc._local_core_id_or_zero_if_hbm(int(local_core_id), space)
    content = memory.mem[(space, int(buffer_id), int(device_id), core)].content
    with memory.lock:
        index, dtype = view_index(content, transforms)
        flat = content.reshape(-1).view(np.uint16)
        old = np.ascontiguousarray(flat[index]).view(dtype)[..., 0]
        if val is not None:
            new = np.ascontiguousarray(np.asarray(val, dtype)[..., None]).view(np.uint16)
            flat[index] = new.reshape(index.shape)
    return old


ORIGINAL = {name: getattr(ipc, name).__wrapped__ for name in ("get", "store", "swap")}
plain = lambda transforms: all(isinstance(t, indexing.NDIndexer) for t in transforms)


@ipc.fail_on_exception
def get(token, device_id, core, space, buffer_id, transforms, *args, **kwargs):
    transforms = jax.tree.map(int, transforms)
    if plain(transforms):
        return ORIGINAL["get"](token, device_id, core, space, buffer_id, transforms, *args, **kwargs)
    return token, access(device_id, core, space, buffer_id, transforms)


@ipc.fail_on_exception
def store(token, device_id, core, space, buffer_id, transforms, val, *args, **kwargs):
    transforms = jax.tree.map(int, transforms)
    if plain(transforms):
        return ORIGINAL["store"](token, device_id, core, space, buffer_id, transforms, val, *args, **kwargs)
    access(device_id, core, space, buffer_id, transforms, val)
    return token


@ipc.fail_on_exception
def swap(token, device_id, core, space, buffer_id, transforms, val, mask, **kwargs):
    transforms = jax.tree.map(int, transforms)
    if plain(transforms) or mask is not None:
        return ORIGINAL["swap"](token, device_id, core, space, buffer_id, transforms, val, mask, **kwargs)
    return token, access(device_id, core, space, buffer_id, transforms, val)


ipc.get, ipc.store, ipc.swap = get, store, swap
mesh = Mesh(np.asarray(jax.devices()), ("tp",))


def project(attended, weight, host_bf16, rank_rows=False):
    # Every rank's owned rows (native 16-row backing) and FP32 partial.
    B = attended.shape[1]
    dmk.ATTENTION_HOST_BF16 = host_bf16  # read when the kernel is traced
    scratch = dmk.communication_scratch(B)
    backing = max(16, -(-B // 16) * 16)

    def kernel(attended_ref, weight_ref, owned_out, partial_out, out, comm, local, host,
               features, outputs, sends, recvs, phase):
        phase[0] = 0
        local[...] = jnp.full(local.shape, jnp.nan, jnp.float32)
        host[...] = jnp.full(host.shape, jnp.nan, jnp.float32)
        features[...] = jnp.full(features.shape, jnp.nan, jnp.bfloat16)
        outputs[...] = jnp.full(outputs.shape, jnp.nan, jnp.bfloat16)
        dmk._project_attention_2d(
            attended_ref[...], weight_ref, out, comm, local, host, features, outputs,
            sends, recvs, phase, dmk._dot, token_scatter_output=True,
            rank_rows=rank_rows)
        owned_out[...] = outputs[...]
        partial_out[...] = comm[...]

    vmem = pltpu.VMEM

    def body(attended, weight):
        owned, partial = pl.pallas_call(
            kernel,
            out_shape=(jax.ShapeDtypeStruct((backing, H), jnp.bfloat16),
                       jax.ShapeDtypeStruct((B, 8192), jnp.float32)),
            in_specs=[pl.BlockSpec(memory_space=vmem)] * 2,
            out_specs=[pl.BlockSpec(memory_space=vmem)] * 2,
            scratch_shapes=[vmem((B, H), jnp.bfloat16), vmem((B, 8192), jnp.float32),
                            scratch[0], scratch[2], vmem((backing, 1024), jnp.bfloat16),
                            vmem((backing, H), jnp.bfloat16), pltpu.SemaphoreType.DMA((16,)),
                            pltpu.SemaphoreType.DMA((16,)), pltpu.SMEM((1,), jnp.int32)],
            compiler_params=pltpu.CompilerParams(has_side_effects=True),
            interpret=pltpu.InterpretParams(),
        )(attended[0], weight[0])
        return owned[None], partial[None]

    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("tp"), P("tp")),
                              out_specs=(P("tp"), P("tp")), check_vma=False))
    return [np.asarray(x) for x in f(attended, weight)]


f32 = lambda x: np.asarray(x, np.float32)
bf = lambda x: f32(x).astype(BF16)


def ordered_sum(parts):
    total = parts[0]
    for part in parts[1:]:
        total = f32(total + part)
    return total


def reference(partial, B, host_bf16):
    tpp = B // 4
    out = np.zeros((32, tpp, H), BF16)
    for t in range(4):
        rows = slice(t * tpp, (t + 1) * tpp)
        halves = []
        for q in range(2):
            hosts = [ordered_sum([f32(bf(partial[8 * h + 2 * p + q, rows, :HALF]))
                                  for p in range(4)]) for h in range(4)]
            if host_bf16:
                hosts = [f32(bf(s)) for s in hosts]
            halves.append(bf(ordered_sum(hosts)))
        for h in range(4):
            out[8 * h + 2 * t: 8 * h + 2 * t + 2] = np.concatenate(halves, axis=1)
    return out
"""

    SCRIPT = PRELUDE + r"""
rng = np.random.default_rng(0)
for B in (8, 16, 24):
    tpp = B // 4
    attended = (rng.standard_normal((32, B, W)) * rng.uniform(0.2, 2.0, (32, 1, W))
                / (1 + np.exp(-rng.standard_normal((32, B, W))))).astype(BF16)
    full = (rng.standard_normal((32 * W, H)) * 0.01).astype(BF16)
    # Rank r's shard: its pair's 768 features x its lane's output half.
    weight = np.stack([full[768 * (r // 2):768 * (r // 2 + 1), (r % 2) * HALF:(r % 2 + 1) * HALF]
                       for r in range(32)])
    rows = {}
    for host_bf16 in (False, True):
        owned, partial = project(attended, weight, host_bf16)
        got = owned[:, :tpp]
        want = reference(partial, B, host_bf16)
        assert np.array_equal(got.view(np.uint16), want.view(np.uint16)), (B, host_bf16)
        for r in range(32):  # the four hosts' copies
            for offset in (8, 16, 24):
                assert np.array_equal(got[r].view(np.uint16), got[r ^ offset].view(np.uint16))
        rows[host_bf16] = np.concatenate([got[2 * t] for t in range(4)])
        print(f"B{B} host_bf16={host_bf16}: bit-exact")
    changed = np.mean(rows[False] != rows[True])
    assert 0.2 < changed < 0.6, changed  # the extra rounding moves a share of outputs by an ulp
print("OK")
"""

    def test_host_sums_match_their_rounding_points(self):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        # 32 interpret devices block in callbacks: give the CPU client enough
        # threads (XLA reads NPROC for its pool size).
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), NPROC="128")
        result = subprocess.run(
            [sys.executable, "-c", self.SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            timeout=1200,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )


class TestAttentionHostRoundAfterJoin(CustomTestCase):
    """``KernelConfig.attention_host_round_after_join``: rank-row output
    projections (B8 and B16, FP32 and BF16 host sums) that return with their
    cross-host round in flight inside one branch of a two-branch ``cond``
    (each branch its own inputs, as the decoder's KDA and MLA branches), then,
    after the join and NaN written over the projection's other storage (the
    reduction buffer, the local-round and feature buffers, the output rows),
    ``finish_rank_rows_host_round``. Pallas interpret mode on 32 host devices
    (TestAttentionHostBf16's harness), with DMAs executed on wait and eagerly;
    barriers follow the NaN fill and each projection, standing in for the
    decoder's start and its cross-host rounds between two layers'
    projections (no rank starts its next host round before its host peers
    have read this one). Each deferred projection's rows equal, bit for bit,
    the same branch's projection finished in line and the reduction's
    rounding points (in one branch two hosts' sums cancel, so the rows
    show the hosts' order); the phase advances once a projection and no DMA
    semaphore is left nonzero."""

    SCRIPT = TestAttentionHostBf16.PRELUDE + r"""
import sys

from sglang_tpu.k3_programs.common import collectives32

DMA_MODE = sys.argv[1]


def project_after_join(attended, weight, host_bf16):
    # attended [32, 2, B, W] and weight [32, 2, 768, HALF]: two branches'
    # inputs. One kernel runs each branch's projection in line, then each
    # branch through the cond with the host round deferred (select [0, 1]).
    B = attended.shape[2]
    dmk.ATTENTION_HOST_BF16 = host_bf16  # read when the kernel is traced
    scratch = dmk.communication_scratch(B)
    nan = lambda ref: jnp.full(ref.shape, jnp.nan, ref.dtype)

    def kernel(select_ref, attended_ref, weight_ref, rows_out, partial_out, phase_out,
               out, comm, local, host, features, outputs, sends, recvs, phase):
        phase[0] = 0
        for ref in (local, host, features, outputs):
            ref[...] = nan(ref)
        collectives32.barrier()  # eager DMAs: no peer writes before these stores

        def projection(index, defer):
            dmk._project_attention_2d(
                attended_ref[index], weight_ref.at[index], out, comm, local, host,
                features, outputs, sends, recvs, phase, dmk._dot,
                token_scatter_output=True, rank_rows=True, defer_host_round=defer)

        for index in range(2):
            projection(index, False)
            rows_out[index] = outputs[...]
            partial_out[index] = comm[...]
            collectives32.barrier()
        for slot in range(2):
            jax.lax.cond(select_ref[slot] == 0,
                         lambda: projection(0, True), lambda: projection(1, True))
            for ref in (comm, local, features, outputs):
                ref[...] = nan(ref)
            dmk.finish_rank_rows_host_round(host, outputs, sends, recvs, phase,
                                            batch_size=B)
            rows_out[2 + slot] = outputs[...]
            collectives32.barrier()
        phase_out[0] = phase[0]

    vmem, smem = pltpu.VMEM, pltpu.SMEM

    def body(select, attended, weight):
        rows, partial, phase = pl.pallas_call(
            kernel,
            out_shape=(jax.ShapeDtypeStruct((4, 16, H), jnp.bfloat16),
                       jax.ShapeDtypeStruct((2, B, 8192), jnp.float32),
                       jax.ShapeDtypeStruct((1,), jnp.int32)),
            in_specs=[pl.BlockSpec(memory_space=smem)] + [pl.BlockSpec(memory_space=vmem)] * 2,
            out_specs=[pl.BlockSpec(memory_space=vmem)] * 2 + [pl.BlockSpec(memory_space=smem)],
            scratch_shapes=[vmem((B, H), jnp.bfloat16), vmem((B, 8192), jnp.float32),
                            scratch[0], scratch[2], vmem((16, 1024), jnp.bfloat16),
                            vmem((16, H), jnp.bfloat16), pltpu.SemaphoreType.DMA((16,)),
                            pltpu.SemaphoreType.DMA((16,)), pltpu.SMEM((1,), jnp.int32)],
            compiler_params=pltpu.CompilerParams(has_side_effects=True, collective_id=0),
            interpret=pltpu.InterpretParams(dma_execution_mode=DMA_MODE),
        )(select[0], attended[0], weight[0])
        return rows[None], partial[None], phase[None]

    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("tp"), P("tp"), P("tp")),
                              out_specs=(P("tp"), P("tp"), P("tp")), check_vma=False))
    select = np.tile(np.arange(2, dtype=np.int32), (32, 1))
    return [np.asarray(x) for x in f(select, attended, weight)]


rng = np.random.default_rng(2)
for B in (8, 16):
    rows = B // 8
    attended = (rng.standard_normal((32, 2, B, W)) * rng.uniform(0.2, 2.0, (32, 2, 1, W))
                / (1 + np.exp(-rng.standard_normal((32, 2, B, W))))).astype(BF16)
    fulls = [(rng.standard_normal((32 * W, H)) * 0.01).astype(BF16) for _ in range(2)]
    # Branch 1: host 1's ranks repeat host 0's features (2^18 times hosts 2
    # and 3's) on negated weights, so host 1's sums negate host 0's and the
    # rows are hosts 2 and 3's sum only when the hosts add in order.
    attended[:8, 1] *= 2.0 ** 18
    attended[8:16, 1] = attended[:8, 1]
    fulls[1][4 * 768:8 * 768] = -fulls[1][:4 * 768]
    weight = np.stack([
        np.stack([full[768 * (r // 2):768 * (r // 2 + 1), (r % 2) * HALF:(r % 2 + 1) * HALF]
                  for r in range(32)])
        for full in fulls
    ], axis=1)
    for host_bf16 in (False, True):
        got, partial, phase = project_after_join(attended, weight, host_bf16)
        assert (phase == 4).all(), phase
        for branch in range(2):
            want = reference(partial[:, branch], B, host_bf16)
            inline = got[:, branch, :rows].view(np.uint16)
            deferred = got[:, 2 + branch, :rows].view(np.uint16)
            for r in range(32):
                mine = slice((r % 2) * rows, (r % 2 + 1) * rows)
                assert np.array_equal(inline[r], want[r, mine].view(np.uint16)), (B, branch, r)
            assert np.array_equal(deferred, inline), (B, host_bf16, branch)
        assert not np.array_equal(got[:, 0, :rows], got[:, 1, :rows])
        print(f"B{B} host_bf16={host_bf16} dma={DMA_MODE}: deferred rows bit-exact")
print("OK")
"""

    def _run(self, dma_mode):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), NPROC="128")
        result = subprocess.run(
            [sys.executable, "-c", self.SCRIPT, dma_mode],
            capture_output=True,
            text=True,
            env=env,
            timeout=1500,
        )
        output = result.stdout[-2000:] + result.stderr[-3000:]
        self.assertIn("OK", result.stdout, output)
        self.assertNotIn("non-zero count", result.stdout, output)

    def test_deferred_rows_match_the_inline_rows(self):
        self._run("on_wait")

    def test_deferred_rows_with_eager_dmas(self):
        self._run("eager")


class TestAttentionRankRows(CustomTestCase):
    """``KernelConfig.attention_rank_rows`` in the token-scattered attention
    output projection (``_project_attention_2d(rank_rows=True)``) at B8 and
    B16 in Pallas interpret mode on 32 host devices (TestAttentionHostBf16's
    harness, stale NaN bytes wherever a round lands). With FP32 and with BF16
    host sums, rank 2p + q's rows (one at B8, two at B16) equal, bit for bit,
    rows q r .. q r + r - 1 of the rows its pair holds without the knob (the
    same inputs) and the reduction's rounding points; the hosts' copies are
    bitwise equal. The MoE input gather of those rows (eight local owners on
    one semaphore, one-row wires in private storage) delivers every local
    rank's rows in owner order."""

    SCRIPT = TestAttentionHostBf16.PRELUDE + r"""
rng = np.random.default_rng(1)
for B in (8, 16):
    tpp, rows = B // 4, B // 8
    attended = (rng.standard_normal((32, B, W)) * rng.uniform(0.2, 2.0, (32, 1, W))
                / (1 + np.exp(-rng.standard_normal((32, B, W))))).astype(BF16)
    full = (rng.standard_normal((32 * W, H)) * 0.01).astype(BF16)
    weight = np.stack([full[768 * (r // 2):768 * (r // 2 + 1), (r % 2) * HALF:(r % 2 + 1) * HALF]
                       for r in range(32)])
    for host_bf16 in (False, True):
        pair_owned, partial = project(attended, weight, host_bf16)
        rank_owned, rank_partial = project(attended, weight, host_bf16, rank_rows=True)
        assert np.array_equal(partial.view(np.uint32), rank_partial.view(np.uint32))
        want = reference(partial, B, host_bf16)
        for r in range(32):
            mine = slice((r % 2) * rows, (r % 2 + 1) * rows)
            got = rank_owned[r, :rows].view(np.uint16)
            assert np.array_equal(got, want[r, mine].view(np.uint16)), (B, host_bf16, r)
            assert np.array_equal(got, pair_owned[r, mine].view(np.uint16)), (B, host_bf16, r)
            for offset in (8, 16, 24):
                assert np.array_equal(got, rank_owned[r ^ offset, :rows].view(np.uint16))
        print(f"B{B} host_bf16={host_bf16}: rank rows bit-exact")


def gather(rows_in):
    # The MoE input gather of rank rows: eight local owners, one semaphore,
    # one-row wires in private storage (B8) or in-place copies (B16).
    rows = rows_in.shape[1]

    def kernel(rows_ref, out_ref, wires, dest, sends, recvs):
        wires[...] = jnp.full(wires.shape, jnp.nan, jnp.float32)
        dest[...] = jnp.full(dest.shape, jnp.nan, jnp.bfloat16)
        dmk._gather_local_pair_rows(
            rows_ref, dest, sends, recvs, wires, owners=8,
            semaphore=dmk.RANK_ROWS_GATHER_SEMAPHORE, transpose_wires=True)
        out_ref[...] = dest[...]

    vmem = pltpu.VMEM

    def body(x):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((8 * rows, H), jnp.bfloat16),
            in_specs=[pl.BlockSpec(memory_space=vmem)],
            out_specs=pl.BlockSpec(memory_space=vmem),
            scratch_shapes=[vmem((8, 32, 128), jnp.float32), vmem((8 * rows, H), jnp.bfloat16),
                            pltpu.SemaphoreType.DMA((16,)), pltpu.SemaphoreType.DMA((16,))],
            compiler_params=pltpu.CompilerParams(has_side_effects=True),
            interpret=pltpu.InterpretParams(),
        )(x[0])[None]

    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=P("tp"), out_specs=P("tp"),
                              check_vma=False))
    return np.asarray(f(rows_in))


for rows in (1, 2):
    rows_in = rng.standard_normal((32, rows, H)).astype(BF16)
    got = gather(rows_in)
    for r in range(32):
        want = rows_in[8 * (r // 8):8 * (r // 8) + 8].reshape(8 * rows, H)
        assert np.array_equal(got[r].view(np.uint16), want.view(np.uint16)), (rows, r)
    print(f"gather of {rows}-row owners bit-exact")
print("OK")
"""

    def test_rank_rows_match_the_pair_rows(self):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), NPROC="128")
        result = subprocess.run(
            [sys.executable, "-c", self.SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            timeout=1500,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )


class TestLatentDownPrefetch(CustomTestCase):
    """``KernelConfig.decode_latent_down_prefetch``: the decode MoE
    (``decode_megakernel.moe``) at 8 rows with the presorted router, the
    expert schedule, the late shared down projection and the token-parallel
    latent-up, in Pallas interpret mode on 32 host devices (TestAttentionHostBf16's
    views), race detection on. With the latent-down copy started before the
    MoE (``latent_input_prefetch``: the latent product and its 31 sends then
    run before the router products) every rank's output rows and gathered
    latents equal the plain MoE's bit for bit, with eager DMAs and with DMAs
    executed at their waits; the latent-down storage holds NaNs until a copy
    lands, so a product that does not wait for it shows."""

    SETUP = r"""
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
os.environ["JAX_PLATFORMS"] = "cpu"
from sglang_tpu.k3_programs.kimi_k3 import config as kc
kc.install(kc.KernelConfig(decode_presorted_routing=True, decode_router_both_halves=True,
                           expert_schedule=True, moe_token_parallel_up=True,
                           decode_late_shared_down=True))
"""

    MAIN = r"""
assert dmk.PRESORTED_ROUTING and dmk.TOKEN_PARALLEL_UP and dmk.EXPERT_SCHEDULE
F32 = jnp.float32
# EP32 routed experts (28 a rank) with a small MXFP4 shape; the latent and
# shared shapes are the decode's.
B, E, LOC, K, WD, OUT, LAT, SW = 8, 896, 28, 512, 256, 3584, 4096, 192
rng = np.random.default_rng(0)
bf = lambda x: np.asarray(x, np.float32).astype(BF16)
hidden = bf(rng.standard_normal((B, H)))
router = np.broadcast_to(bf(rng.standard_normal((H, E)) * H**-0.5), (32, H, E))
bias = (0.01 * rng.standard_normal((1, E))).astype(np.float32)
norm = np.zeros((1, LAT), np.float32)
norm[0, :OUT] = 1 + 0.1 * rng.standard_normal(OUT)
ld = bf(rng.standard_normal((32, H, 128)) * H**-0.5)
lu = bf(rng.standard_normal((32, OUT, 1024)) * OUT**-0.5)
sgu = bf(rng.standard_normal((32, H, 2 * SW)) * H**-0.5)
sd = bf(rng.standard_normal((32, SW, H)) * SW**-0.5)
egu = rng.integers(0, 2**32, (32, LOC, K // 8, 2 * WD), dtype=np.uint32)
egus = rng.integers(118, 124, (32, LOC, K // 32, 2 * WD), dtype=np.uint8)
edn = rng.integers(0, 2**32, (32, LOC, WD // 8, OUT), dtype=np.uint32)
eds = rng.integers(118, 124, (32, LOC, WD // 32, OUT), dtype=np.uint8)

acc_rows = dmk._expert_accumulator_rows(B, jnp.uint32, group_pending_routes=True)
crows = dmk._compact_rows(B, group_pending_routes=True)
vmem, dma, smem = pltpu.VMEM, pltpu.SemaphoreType.DMA, pltpu.SMEM
WS = (vmem((B, 128), jnp.int32), vmem((B, 128), F32), vmem((B, 128), jnp.int32),
      vmem((B, 128), F32), vmem((B, 1024), F32), vmem((B, H), F32), vmem((B, LAT), F32),
      vmem((acc_rows[0], 4 * WD), F32), vmem((acc_rows[1], OUT), F32),
      vmem((B, 128), jnp.int32),
      *((vmem((crows, K), F32),) * 2 if crows else (vmem((1, K), jnp.bfloat16),) * 2),
      dma((10,)), dma((10,)), vmem((12, B // 4 * 8, 128), jnp.bfloat16), dma((7,)),
      dma((7,)), dma((2,)), dma((2,)))
OWN = (vmem((2, H, 512), jnp.bfloat16), dma((2,)), vmem((OUT, 1024), jnp.bfloat16), dma(()),
       vmem((4, K // 8, 2 * WD), jnp.uint32), vmem((4, K // 32, 2 * WD), jnp.uint8),
       vmem((4, WD // 8, OUT), jnp.uint32), vmem((4, WD // 32, OUT), jnp.uint8), dma((4, 4)),
       vmem((H, 128), jnp.bfloat16), vmem((H, 2 * SW), jnp.bfloat16),
       vmem((SW, H), jnp.bfloat16), dma(()), smem((8, 128), jnp.int32), dma(()))
COMM = dmk.communication_scratch(B)


def run(prefetch, mode):
    def kernel(hidden_ref, bias_ref, norm_ref, router_hbm, ld_hbm, lu_hbm, sgu_hbm, sd_hbm,
               egu_hbm, egus_hbm, ed_hbm, eds_hbm, out_ref, latent_out, *scratch):
        own, ws, comm = scratch[:len(OWN)], scratch[len(OWN):-len(COMM)], scratch[-len(COMM):]
        (router_v, router_sems, lu_v, lu_sem, egu_v, egus_v, ed_v, eds_v, e_sems, ld_v, sgu_v,
         sd_v, ld_sem, schedule, schedule_sem) = own
        comm_ws = dmk.AttentionCommunicationWorkspace(*comm)
        comm_ws.reduction_phase[0] = 0
        ld_v[...] = jnp.full(ld_v.shape, jnp.nan, ld_v.dtype)
        weights = dmk.MixtureOfExpertsWeights(
            router_hbm, router_v, router_sems, bias_ref, ld_hbm, lu_hbm, lu_v, lu_sem, norm_ref,
            sgu_hbm, sd_hbm, egu_hbm, egu_v, egus_hbm, egus_v, ed_hbm, ed_v, eds_hbm, eds_v,
            e_sems)
        copy = None
        if prefetch:  # as the decoder starts it during the attention
            copy = pltpu.make_async_copy(ld_hbm, ld_v, ld_sem)
            copy.start()
        dmk.moe(hidden_ref, out_ref, weights=weights,
                workspace=dmk.MixtureOfExpertsWorkspace(
                    *ws, expert_schedule_smem=schedule, expert_schedule_semaphore=schedule_sem),
                communication_workspace=comm_ws, group_pending_expert_routes=True,
                token_scatter_output=True, routed_reduce_scatter=True, token_parallel_up=True,
                prepare_weights_vmem=(ld_v, sgu_v, sd_v), latent_input_prefetch=copy)
        latent_out[...] = comm_ws.latent_gather_vmem[...]

    def body(h, b, n, *hbm):
        out = pl.pallas_call(
            kernel,
            out_shape=(jax.ShapeDtypeStruct((B // 4, H), jnp.bfloat16),
                       jax.ShapeDtypeStruct((B, LAT), jnp.bfloat16)),
            in_specs=[pl.BlockSpec(memory_space=vmem)] * 3 + [pl.BlockSpec(memory_space=pl.ANY)] * 9,
            out_specs=[pl.BlockSpec(memory_space=vmem)] * 2,
            scratch_shapes=[*OWN, *WS, *COMM],
            compiler_params=pltpu.CompilerParams(has_side_effects=True, collective_id=7),
            interpret=pltpu.InterpretParams(detect_races=True, dma_execution_mode=mode),
        )(h, b, n, *(x[0] for x in hbm))
        return tuple(x[None] for x in out)

    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P(), P(), *(P("tp"),) * 9),
                              out_specs=(P("tp"),) * 2, check_vma=False))
    return [np.asarray(x) for x in f(hidden, bias, norm, router, ld, lu, sgu, sd, egu, egus,
                                     edn, eds)]


for mode in ("eager", "on_wait"):
    want = run(False, mode)
    assert not ipc.races.races_found, mode  # reset by every interpreted call
    got = run(True, mode)
    assert not ipc.races.races_found, mode
    out = want[0].astype(np.float32)
    assert np.isfinite(out).all() and np.abs(out).max() > 0, mode
    assert np.isfinite(want[1].astype(np.float32)).all(), mode
    for name, a, b in zip(("output rows", "gathered latents"), want, got):
        assert a.tobytes() == b.tobytes(), (mode, name)
    print(f"{mode}: bit-exact (rms {np.sqrt(np.mean(out ** 2)):.3f})", flush=True)
print("OK")
"""

    def test_prefetched_latent_down_moe_is_bitwise_unchanged(self):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        prelude = TestAttentionHostBf16.PRELUDE
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), NPROC="128")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                self.SETUP + prelude[: prelude.index("def project(")] + self.MAIN,
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )
        self.assertNotIn("non-zero count", result.stdout + result.stderr)


class TestFirstPairGateUpCopies(CustomTestCase):
    """The initial routed-expert copies with ``gate_up_first_pair``
    (``KernelConfig.decode_late_shared_down``) in Pallas interpret mode: on
    random grouped and per-slot routings at B8 and B16 (with prefetched slots
    and rows without routes), every slot copies the same expert's same weight
    families as the slot-by-slot order, and the copies run in the order
    first two slots' gate/up and scales, their down and scales, then the
    remaining slots."""

    SCRIPT = """
        import os
        os.environ["JAX_PLATFORMS"] = "cpu"
        from types import SimpleNamespace
        import numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk

        FAMILIES = ("gate_up", "gate_up_scales", "down", "down_scales")

        class Copy:
            def __init__(self, experts, order, counter, slot, family, expert):
                self.refs = experts, order, counter
                self.at = slot, family
                self.expert = expert

            def start(self):
                experts, order, counter = self.refs
                slot, family = self.at
                experts[slot, family] = self.expert + 1
                order[slot, family] = counter[0]
                counter[0] = counter[0] + 1

        def run(experts_in, counts_in, grouped, started, first_pair):
            def kernel(experts_ref, counts_ref, experts_out, order_out, counter):
                experts_out[...] = jnp.zeros(experts_out.shape, jnp.int32)
                order_out[...] = jnp.full(order_out.shape, -1, jnp.int32)
                counter[0] = 0

                def copy_expert(expert, slot, *, gate_up=True, down=True):
                    wanted = (gate_up, gate_up, down, down)
                    return SimpleNamespace(**{
                        name: Copy(experts_out, order_out, counter, slot, family, expert)
                        if wanted[family] else None
                        for family, name in enumerate(FAMILIES)})

                dmk._start_initial_expert_weight_copies(
                    experts_ref[...], counts_ref[...][:, 0], copy_expert, slot_count=4,
                    already_started=started, group_pending_routes=grouped,
                    gate_up_first_pair=first_pair)

            smem = pltpu.SMEM
            return [np.asarray(x) for x in pl.pallas_call(
                kernel,
                out_shape=(jax.ShapeDtypeStruct((4, 4), jnp.int32),
                           jax.ShapeDtypeStruct((4, 4), jnp.int32)),
                in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 2,
                out_specs=[pl.BlockSpec(memory_space=smem)] * 2,
                scratch_shapes=[smem((1,), jnp.int32)],
                interpret=pltpu.InterpretParams(),
            )(experts_in, counts_in)]

        rng = np.random.default_rng(0)
        cases = 0
        for B in (8, 16):
            for trial in range(6):
                pool = rng.choice(112, size=rng.integers(2, 12), replace=False)
                experts = rng.choice(pool, size=(B, 16)).astype(np.int32)
                counts = rng.integers(0, 4, size=B).astype(np.int32)
                counts[rng.integers(B)] = 0
                counts = np.broadcast_to(counts[:, None], (B, 128)).copy()
                for grouped in (True, False):
                    for started in (0, 1):
                        plain = run(experts, counts, grouped, started, False)
                        paired = run(experts, counts, grouped, started, True)
                        assert np.array_equal(plain[0], paired[0]), (B, trial, grouped)
                        live = paired[1] >= 0
                        assert np.array_equal(live, plain[1] >= 0)
                        # Each slot copies all four families or none.
                        assert all(row.all() or not row.any() for row in live)
                        slots = [s for s in range(4) if live[s].all()]
                        want = [(s, f) for s in slots[:2] if s < 2 for f in (0, 1)]
                        want += [(s, f) for s in slots[:2] if s < 2 for f in (2, 3)]
                        want += [(s, f) for s in slots if s >= 2 for f in range(4)]
                        got = sorted(want, key=lambda sf: paired[1][sf])
                        assert got == want, (got, want)
                        assert sorted(paired[1][live]) == list(range(len(want)))
                        cases += 1
        print("cases", cases)
        print("OK")
    """

    def test_same_copies_first_pair_gate_up_first(self):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )


class TestKdaFp8Families(CustomTestCase):
    """KernelConfig.kda_fp8_projections at load: each column of the decode's
    KDA input and output projections becomes FP8 e4m3 of the column divided by
    its absolute maximum / 448 (rounded to nearest), with that FP32 scale."""

    def test_columns_scale_to_448_and_round_to_nearest(self):
        import ml_dtypes

        from sglang_tpu.k3_programs.kimi_k3.kernels import rank_loader

        rng = np.random.default_rng(0)
        weight = (rng.standard_normal((256, 384)) * rng.uniform(1e-3, 1, 384)).astype(
            ml_dtypes.bfloat16
        )
        weight[:, 5] = 0
        values, scale = rank_loader.quantize_columns_fp8(weight)
        self.assertEqual(values.dtype, ml_dtypes.float8_e4m3fn)
        self.assertEqual((values.shape, scale.shape), ((256, 384), (1, 384)))
        self.assertEqual(scale.dtype, np.float32)
        exact = weight.astype(np.float32)
        peak = np.abs(exact).max(axis=0)
        np.testing.assert_array_equal(scale[0, 5], 1)
        np.testing.assert_array_equal(values[:, 5].astype(np.float32), 0)
        live = peak > 0
        np.testing.assert_array_equal(scale[0, live], peak[live] / np.float32(448))
        # The column maximum lands on 448; every value is the nearest FP8.
        decoded = values.astype(np.float32)
        np.testing.assert_array_equal(np.abs(decoded[:, live]).max(axis=0), 448)
        np.testing.assert_array_equal(
            decoded, (exact / scale).astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
        )
        # Normal-range values are within half an FP8 ulp (2^-4 relative).
        normal = np.abs(exact / scale) >= 2.0**-6
        error = np.abs(decoded * scale - exact)[normal] / np.abs(exact)[normal]
        self.assertLessEqual(error.max(), 2.0**-4 + 1e-6)

    def test_families_replace_both_projections_per_layer(self):
        import ml_dtypes

        from sglang_tpu.k3_programs.kimi_k3.kernels import rank_loader

        rng = np.random.default_rng(1)
        families = {
            "k_projection": (rng.standard_normal((3, 64, 256)) * 0.02).astype(
                ml_dtypes.bfloat16
            ),
            "k_o": (rng.standard_normal((3, 96, 128)) * 0.02).astype(
                ml_dtypes.bfloat16
            ),
            "k_gate": np.ones((3, 64, 128), ml_dtypes.bfloat16),
        }
        sources = {name: families[name].copy() for name in ("k_projection", "k_o")}
        converted = rank_loader.kda_fp8_families(families)
        self.assertEqual(sorted(families), ["k_gate"])  # BF16 sources popped
        self.assertEqual(
            sorted(converted),
            ["k_o", "k_o_scale", "k_projection", "k_projection_scale"],
        )
        for name, source in sources.items():
            values, scale = converted[name], converted[f"{name}_scale"]
            self.assertEqual(values.dtype, ml_dtypes.float8_e4m3fn)
            self.assertEqual(scale.shape, (3, 1, source.shape[-1]))
            for layer in range(3):
                want_values, want_scale = rank_loader.quantize_columns_fp8(
                    source[layer]
                )
                np.testing.assert_array_equal(
                    values[layer].view(np.uint8), want_values.view(np.uint8)
                )
                np.testing.assert_array_equal(scale[layer], want_scale)

    def test_fp8_families_upload_as_torch_fp8(self):
        import ml_dtypes
        import torch

        from sglang_tpu.k3_programs import bridge

        values = np.linspace(-448, 448, 64, dtype=np.float32).astype(
            ml_dtypes.float8_e4m3fn
        )
        tensor = bridge.host_tensor(values.reshape(8, 8))
        self.assertEqual(tensor.dtype, torch.float8_e4m3fn)
        np.testing.assert_array_equal(
            tensor.view(torch.uint8).numpy().reshape(-1), values.view(np.uint8)
        )
        halves = np.ones(4, ml_dtypes.bfloat16)
        self.assertEqual(bridge.host_tensor(halves).dtype, torch.bfloat16)

    def test_colocated_prefill_keeps_its_bf16_input_projection(self):
        from types import SimpleNamespace
        from unittest import mock

        from sglang_tpu.k3_programs import sglang_colocated as colocated

        for fp8 in (False, True):
            config = SimpleNamespace(attention_dp=1, kda_fp8_projections=fp8)
            with mock.patch(
                "sglang_tpu.k3_programs.kimi_k3.config.active", return_value=config
            ):
                kda = colocated.shared_dense()["kda"]
            self.assertEqual("k_projection" in kda, not fp8)
            self.assertIn("k_gate", kda)


class TestKdaFp8OutputProjection(CustomTestCase):
    """The token-scattered attention output projection
    (``_project_attention_2d``) with FP8 weights and FP32 column scales
    (KernelConfig.kda_fp8_projections, rank_loader.quantize_columns_fp8) at B8
    and B16 in Pallas interpret mode on 32 host devices: every rank's FP32
    partial is the FP64 product of the dequantized weights (own and peer
    halves, columns scaled ~10x apart) to FP32 accuracy, and its owned rows
    are, bit for bit, the reduction's rounding points applied to those
    partials (as TestAttentionHostBf16 checks for BF16 weights)."""

    MAIN = r"""

def project_scaled(attended, weight, scale):
    B = attended.shape[1]
    dmk.ATTENTION_HOST_BF16 = False
    scratch = dmk.communication_scratch(B)
    backing = max(16, -(-B // 16) * 16)

    def kernel(attended_ref, weight_ref, scale_ref, owned_out, partial_out, out, comm,
               local, host, features, outputs, sends, recvs, phase):
        phase[0] = 0
        local[...] = jnp.full(local.shape, jnp.nan, jnp.float32)
        host[...] = jnp.full(host.shape, jnp.nan, jnp.float32)
        features[...] = jnp.full(features.shape, jnp.nan, jnp.bfloat16)
        outputs[...] = jnp.full(outputs.shape, jnp.nan, jnp.bfloat16)
        dmk._project_attention_2d(
            attended_ref[...], weight_ref, out, comm, local, host, features, outputs,
            sends, recvs, phase, dmk._dot, token_scatter_output=True,
            weight_scale=scale_ref)
        owned_out[...] = outputs[...]
        partial_out[...] = comm[...]

    vmem = pltpu.VMEM

    def body(attended, weight, scale):
        owned, partial = pl.pallas_call(
            kernel,
            out_shape=(jax.ShapeDtypeStruct((backing, H), jnp.bfloat16),
                       jax.ShapeDtypeStruct((B, 8192), jnp.float32)),
            in_specs=[pl.BlockSpec(memory_space=vmem)] * 3,
            out_specs=[pl.BlockSpec(memory_space=vmem)] * 2,
            scratch_shapes=[vmem((B, H), jnp.bfloat16), vmem((B, 8192), jnp.float32),
                            scratch[0], scratch[2], vmem((backing, 1024), jnp.bfloat16),
                            vmem((backing, H), jnp.bfloat16), pltpu.SemaphoreType.DMA((16,)),
                            pltpu.SemaphoreType.DMA((16,)), pltpu.SMEM((1,), jnp.int32)],
            compiler_params=pltpu.CompilerParams(has_side_effects=True),
            interpret=pltpu.InterpretParams(),
        )(attended[0], weight[0], scale[0])
        return owned[None], partial[None]

    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("tp"),) * 3,
                              out_specs=(P("tp"), P("tp")), check_vma=False))
    return [np.asarray(x) for x in f(attended, weight, scale)]


F8 = ml_dtypes.float8_e4m3fn
rng = np.random.default_rng(0)
for B in (8, 16):
    tpp = B // 4
    attended = (rng.standard_normal((32, B, W)) * rng.uniform(0.2, 2.0, (32, 1, W))
                / (1 + np.exp(-rng.standard_normal((32, B, W))))).astype(BF16)
    full = (rng.standard_normal((32 * W, H)) * 0.01 * rng.uniform(0.3, 3, H)).astype(BF16)
    # Rank r's shard: its pair's 768 features x its lane's output half.
    weight = np.stack([full[768 * (r // 2):768 * (r // 2 + 1), (r % 2) * HALF:(r % 2 + 1) * HALF]
                       for r in range(32)]).astype(np.float32)
    peak = np.abs(weight).max(axis=1, keepdims=True)
    scale = (peak / np.float32(448)).astype(np.float32)
    values = (weight / scale).astype(F8)
    owned, partial = project_scaled(attended, values, scale)
    a = attended.astype(np.float64)
    for r in range(32):
        w = values[r].astype(np.float64) * scale[r].astype(np.float64)
        own, peer = (r % 2) * W, (r % 2 ^ 1) * W
        want = a[r] @ w[own:own + W] + a[r ^ 1] @ w[peer:peer + W]
        err = np.abs(partial[r, :, :HALF] - want).max() / np.abs(want).max()
        assert err < 1e-5, (B, r, err)
    got = owned[:, :tpp]
    assert np.array_equal(got.view(np.uint16), reference(partial, B, False).view(np.uint16)), B
    for r in range(32):  # the four hosts' copies
        for offset in (8, 16, 24):
            assert np.array_equal(got[r].view(np.uint16), got[r ^ offset].view(np.uint16))
    print(f"B{B}: FP64 partials, bit-exact rows")
print("OK")
"""

    def test_fp8_partials_and_rows_match_the_dequantized_weights(self):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        harness = TestAttentionHostBf16.SCRIPT.split("rng = np.random.default_rng(0)")[0]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), NPROC="128")
        result = subprocess.run(
            [sys.executable, "-c", harness + self.MAIN],
            capture_output=True,
            text=True,
            env=env,
            timeout=1200,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )



class TestKdaFp8Layer(CustomTestCase):
    """A KDA decode layer (``kda``: gate, input projection, short conv,
    delta-rule recurrence, gated norm, token-scattered output projection) at
    B8, as one 8-row verify sequence and as 8 plain rows, in Pallas interpret
    mode on 32 host devices, with FP8 input and output projections and
    power-of-two column scales (KernelConfig.kda_fp8_projections): every
    rank's owned output rows and post-states equal, bit for bit, the layer
    with FP32 copies of the dequantized weights (the interpret mode's
    mixed-dtype products run in FP32 for both, and each scaled product is
    exact)."""

    MAIN = r"""
from sglang_tpu.k3_programs.kimi_k3.layout import CONV_HEAD_ROWS

F8 = ml_dtypes.float8_e4m3fn
B = 8


def layer(hidden, gate, inputs, fb, output, aux, history, state, scales, sequence_rows):
    comm = dmk.communication_scratch(B)
    in_dtype, out_dtype = inputs.dtype, output.dtype

    def kernel(hidden_ref, gate_ref, input_ref, fb_ref, output_ref, conv_ref, decay_ref,
               bias_ref, norm_ref, history_ref, state_ref, input_scale_ref, output_scale_ref,
               history_out, state_out, owned_out, out, gate_vmem, input_vmem, fb_vmem,
               output_vmem, copy_sems, *workspace):
        ws = dmk.AttentionCommunicationWorkspace(*workspace)
        ws.reduction_phase[0] = 0
        for buffer in workspace[:4]:
            buffer[...] = jnp.full(buffer.shape, jnp.nan, jnp.float32)
        ws.attention_features_vmem[...] = jnp.full(ws.attention_features_vmem.shape, jnp.nan, jnp.bfloat16)
        ws.attention_output_vmem[...] = jnp.full(ws.attention_output_vmem.shape, jnp.nan, jnp.bfloat16)
        dmk.kda(
            hidden_ref, out, history_out, state_out,
            weights=dmk.KimiDeltaAttentionWeights(
                gate_ref, gate_vmem, input_ref, input_vmem, fb_ref, fb_vmem, output_ref,
                output_vmem, conv_ref, decay_ref, bias_ref, norm_ref, copy_sems,
                input_projection_scale=input_scale_ref if scales else None,
                output_projection_scale=output_scale_ref if scales else None),
            convolution_history_ref=history_ref, recurrent_state_ref=state_ref,
            communication_workspace=ws, synchronize_devices=False,
            weights_are_prefetched=False, token_scatter_output=True,
            sequence_rows=sequence_rows)
        owned_out[...] = ws.attention_output_vmem[...]

    vmem, anywhere = pltpu.VMEM, pl.ANY
    spaces = [vmem, anywhere, anywhere, anywhere, anywhere] + [vmem] * 8

    def body(*args):
        outs = pl.pallas_call(
            kernel,
            out_shape=(jax.ShapeDtypeStruct(history.shape[1:], jnp.bfloat16),
                       jax.ShapeDtypeStruct(state.shape[1:], jnp.float32),
                       jax.ShapeDtypeStruct((16, H), jnp.bfloat16)),
            in_specs=[pl.BlockSpec(memory_space=s) for s in spaces],
            out_specs=[pl.BlockSpec(memory_space=vmem)] * 3,
            scratch_shapes=[vmem((B, H), jnp.bfloat16), vmem((H, 128), jnp.bfloat16),
                            vmem((H, 1664), in_dtype), vmem((128, 384), jnp.bfloat16),
                            vmem((768, HALF), out_dtype), pltpu.SemaphoreType.DMA((4,)), *comm],
            compiler_params=pltpu.CompilerParams(has_side_effects=True),
            interpret=pltpu.InterpretParams(),
        )(*(a[0] for a in args))
        return tuple(o[None] for o in outs)

    operands = (hidden, gate, inputs, fb, output, *aux, history, state, *scales_or_ones(scales))
    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("tp"),) * len(operands),
                              out_specs=(P("tp"),) * 3, check_vma=False))
    return [np.asarray(x) for x in f(*operands)]


def scales_or_ones(scales):
    if scales:
        return scales
    return (np.ones((32, 1, 1664), np.float32), np.ones((32, 1, HALF), np.float32))


def pow2_fp8(weight):
    # FP8 values and power-of-two column scales; values x scales is exact in FP32.
    exponent = np.floor(np.log2(np.abs(weight).max(axis=-2, keepdims=True) / 448)) + 1
    scale = np.exp2(exponent).astype(np.float32)
    values = (weight / scale).astype(F8)
    exact = values.astype(np.float32) * scale
    assert np.array_equal(exact.astype(np.float64), values.astype(np.float64) * scale)
    return values, scale, exact


rng = np.random.default_rng(0)
rep = lambda x: np.broadcast_to(x, (32, *x.shape)).copy()
hidden = rep((rng.standard_normal((B, H)) * rng.uniform(0.3, 3, H)).astype(BF16))
gate = (rng.standard_normal((32, H, 128)) * 0.02).astype(BF16)
inputs = (rng.standard_normal((32, H, 1664)) * 0.02 * rng.uniform(0.2, 5, (32, 1, 1664))).astype(np.float32)
fb = (rng.standard_normal((32, 128, 384)) * 0.1).astype(BF16)
output = (rng.standard_normal((32, 768, HALF)) * 0.02 * rng.uniform(0.2, 5, (32, 1, HALF))).astype(np.float32)
aux = ((rng.standard_normal((32, 4, 3, 3, 128)) * 0.5).astype(BF16),
       rng.uniform(-1, 1, (32, 3, 1)).astype(np.float32),
       rng.uniform(-2, 2, (32, 3, 128)).astype(np.float32),
       (1 + 0.1 * rng.standard_normal((32, 8, 128))).astype(BF16))
history = (rng.standard_normal((32, B, 3, 3, CONV_HEAD_ROWS, 128))).astype(BF16)
state = (rng.standard_normal((32, B, 3, 128, 128)) * 0.1).astype(np.float32)
in8, in_scale, in32 = pow2_fp8(inputs)
out8, out_scale, out32 = pow2_fp8(output)
for sequence_rows in (8, 1):
    got = layer(hidden, gate, in8, fb, out8, aux, history, state, (in_scale, out_scale), sequence_rows)
    want = layer(hidden, gate, in32, fb, out32, aux, history, state, None, sequence_rows)
    for g, w, name in zip(got, want, ("history", "state", "rows")):
        if name == "history":  # the three real heads of the four head rows
            g, w = g[..., :3, :], w[..., :3, :]
        if name == "rows":
            g, w = g[:, :B // 4], w[:, :B // 4]
        assert np.isfinite(g.astype(np.float32)).all(), name
        bits = np.uint16 if g.dtype.itemsize == 2 else np.uint32
        assert np.array_equal(g.view(bits), w.view(bits)), (sequence_rows, name)
    # Without the scales the same FP8 values give a different layer.
    unscaled = layer(hidden, gate, in8, fb, out8, aux, history, state, None, sequence_rows)
    assert not np.array_equal(unscaled[2][:, :B // 4].view(np.uint16), want[2][:, :B // 4].view(np.uint16))
    print(f"sequence_rows={sequence_rows}: bit-exact")
print("OK")
"""

    def test_fp8_layer_matches_the_dequantized_weights(self):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        harness = TestAttentionHostBf16.SCRIPT.split("rng = np.random.default_rng(0)")[0]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), NPROC="128")
        result = subprocess.run(
            [sys.executable, "-c", harness + self.MAIN],
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )


class TestCopyFreeWeights(CustomTestCase):
    """KernelConfig.decode_copy_free_weights: ``rank_loader.copy_free_families``
    holds the norm stacks' BF16 values in FP32 and the KDA filter, decay rate
    and time-step bias as FP32 [layers, 1, width] rows; in Pallas interpret
    mode the kernel's reads of those rows (``_kda_filter``, ``_kda_heads``)
    and its FP32 norm products give the knob-off values bit for bit."""

    def _families(self, rng):
        import ml_dtypes

        bf16 = ml_dtypes.bfloat16
        return {
            "attn_norm": rng.standard_normal((5, 1, 7168)).astype(bf16),
            "ffn_norm": rng.standard_normal((5, 1, 7168)).astype(bf16),
            "m_qnorm": rng.standard_normal((2, 1, 1536)).astype(bf16),
            "m_knorm": rng.standard_normal((2, 1, 512)).astype(bf16),
            "final_norm": rng.standard_normal((1, 7168)).astype(bf16),
            "k_conv": rng.standard_normal((3, 4, 3, 3, 128)).astype(np.float32),
            "k_a_log": rng.uniform(-1, 1, (3, 3, 1)).astype(np.float32),
            "k_dt": rng.uniform(-2, 2, (3, 3, 128)).astype(np.float32),
            "k_norm": rng.standard_normal((3, 8, 128)).astype(bf16),
        }

    def test_families_hold_the_same_values(self):
        from sglang_tpu.k3_programs.kimi_k3.kernels import rank_loader

        families = self._families(np.random.default_rng(0))
        got = rank_loader.copy_free_families(families)
        self.assertEqual(
            set(got), set(rank_loader.COPY_FREE_NORMS + rank_loader.COPY_FREE_KDA)
        )
        for name in rank_loader.COPY_FREE_NORMS:
            self.assertEqual(got[name].dtype, np.float32)
            self.assertEqual(got[name].shape, families[name].shape)
            np.testing.assert_array_equal(got[name], families[name].astype(np.float32))
        self.assertEqual(got["k_conv"].shape, (3, 1, 4608))
        self.assertEqual(got["k_dt"].shape, (3, 1, 384))
        self.assertEqual(got["k_a_log"].shape, (3, 1, 384))
        for layer in range(3):
            for tap, part, head in np.ndindex(4, 3, 3):
                lanes = 128 * (9 * tap + 3 * part + head)
                np.testing.assert_array_equal(
                    got["k_conv"][layer, 0, lanes : lanes + 128],
                    families["k_conv"][layer, tap, part, head],
                )
            for head in range(3):
                lanes = slice(128 * head, 128 * (head + 1))
                np.testing.assert_array_equal(
                    got["k_dt"][layer, 0, lanes], families["k_dt"][layer, head]
                )
                np.testing.assert_array_equal(
                    got["k_a_log"][layer, 0, lanes],
                    np.full(128, families["k_a_log"][layer, head, 0]),
                )

    def test_kernel_reads_match_bit_for_bit(self):
        """The KDA layer's filter product and decay (as ``kda`` computes
        them), and the norms' FP32 products (``normalize`` and the MLA q/kv
        norms), from both representations."""
        import functools

        import jax
        import jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk
        from sglang_tpu.k3_programs.kimi_k3.kernels import rank_loader

        rng = np.random.default_rng(1)
        families = self._families(rng)
        flat = rank_loader.copy_free_families(families)
        rows = 8
        window = jnp.asarray(
            rng.standard_normal((rows, 4, 3, 3, 128)) * 2, jnp.bfloat16
        )
        gate = jnp.asarray(rng.standard_normal((rows, 384)) * 3, jnp.bfloat16)
        hidden = jnp.asarray(rng.standard_normal((rows, 7168)) * 2, jnp.bfloat16)
        latent = jnp.asarray(rng.standard_normal((rows, 1536)), jnp.float32)

        def kernel(
            window_ref,
            gate_ref,
            hidden_ref,
            latent_ref,
            conv_ref,
            decay_ref,
            bias_ref,
            norm_ref,
            qnorm_ref,
            filtered_out,
            decay_out,
            filter_out,
            normalized_out,
            query_out,
        ):
            convolution_filter = dmk._kda_filter(conv_ref).astype(jnp.float32)
            filter_out[...] = convolution_filter
            filtered_out[...] = jax.nn.silu(
                jnp.sum(
                    window_ref[...].astype(jnp.float32) * convolution_filter[None],
                    axis=1,
                )
            ).astype(jnp.bfloat16)
            decay_out[...] = -5.0 * jax.nn.sigmoid(
                jnp.exp(dmk._kda_heads(decay_ref))[None]
                * (
                    gate_ref[...].reshape(rows, 3, 128).astype(jnp.float32)
                    + dmk._kda_heads(bias_ref)[None]
                )
            )
            value = hidden_ref[...].astype(jnp.float32)
            normalized_out[...] = (
                value
                * jax.lax.rsqrt(jnp.mean(value * value, axis=1, keepdims=True) + 1e-5)
                * norm_ref[...]
            ).astype(jnp.bfloat16)
            query = latent_ref[...]
            query_out[...] = (
                query
                * jax.lax.rsqrt(jnp.mean(query * query, axis=1, keepdims=True) + 1e-5)
                * qnorm_ref[...]
            ).astype(jnp.bfloat16)

        call = functools.partial(
            pl.pallas_call,
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct((rows, 3, 3, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((rows, 3, 128), jnp.float32),
                jax.ShapeDtypeStruct((4, 3, 3, 128), jnp.float32),
                jax.ShapeDtypeStruct((rows, 7168), jnp.bfloat16),
                jax.ShapeDtypeStruct((rows, 1536), jnp.bfloat16),
            ),
            interpret=pltpu.InterpretParams(),
        )

        def run(source, layer):
            return [
                np.asarray(x)
                for x in call()(
                    window,
                    gate,
                    hidden,
                    latent,
                    jnp.asarray(source["k_conv"][layer]),
                    jnp.asarray(source["k_a_log"][layer]),
                    jnp.asarray(source["k_dt"][layer]),
                    jnp.asarray(source["attn_norm"][layer]),
                    jnp.asarray(source["m_qnorm"][layer % 2]),
                )
            ]

        for layer in (0, 2):
            want, got = run(families, layer), run(flat, layer)
            np.testing.assert_array_equal(
                want[2], families["k_conv"][layer].astype(np.float32)
            )
            for w, g in zip(want, got, strict=True):
                bits = {2: np.uint16, 4: np.uint32}[w.dtype.itemsize]
                np.testing.assert_array_equal(g.view(bits), w.view(bits))


class TestSeededWindowMath(CustomTestCase):
    """``KernelConfig.dsa_seeded_window``'s arithmetic against brute force:
    lane ``j`` of ``dsa_dcp.window_thresholds`` holds the ``(j + 1)``-th
    smallest of the 63 coarse and 64 fine thresholds (centres at the int32
    ends, garbage, a coarse threshold on, beside and between fine ones), a
    key's ``window_digit`` counts the thresholds at or below it, and the
    kernel's search (the seeded round's bin, one or two 7-bit rounds from
    its lower bound) finds the 20-bit prefix of the topk-th key whatever the
    centre (ties, int32 extremes, tiny and large selections). Also the decode
    step's centre gather and scatter (``gather_centres`` / ``scatter_centres``)."""

    INT_MIN = -(2**31)

    def _centres(self, rng, randoms):
        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp

        INT_MIN = self.INT_MIN
        centres = [INT_MIN, INT_MIN + 1, -1, 0, 1, 2**31 - 1, 2**31 - 4096, 12345]
        centres += [dsa_dcp.CENTRE_LOW + d for d in (-4096, -1, 0, 1)]
        centres += [dsa_dcp.CENTRE_HIGH + d for d in (-1, 0, 1, 4096)]
        for c in range(1, 64):  # a coarse threshold inside, at and beside the window
            for off in (
                -(2**24),
                -(2**24) + 4096,
                -(2**24) - 4096,
                0,
                2**24 - 4096,
                -(2**19),
                2**19,
                -(2**25),
                2**25,
                5 * 2**19 + 4096,
            ):
                v = (c << 26) + INT_MIN + off
                if INT_MIN <= v < 2**31:
                    centres.append(v)
        centres += list(rng.integers(INT_MIN, 2**31, randoms, dtype=np.int64))
        return np.array(centres, np.int64).astype(np.int32)

    def _union(self, start):
        """The coarse and fine thresholds, sorted (unsigned bits, int64)."""
        first = int(start) + (1 << 19) - self.INT_MIN
        return np.array(
            sorted(
                [c << 26 for c in range(1, 64)] + [first + (k << 19) for k in range(64)]
            ),
            np.int64,
        )

    def test_thresholds_and_digits(self):
        import jax.numpy as jnp

        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp

        rng = np.random.default_rng(0)
        centres = self._centres(rng, 300)
        starts = np.asarray(dsa_dcp.window_start(jnp.asarray(centres)))
        lane = jnp.arange(128, dtype=jnp.int32)
        table = np.asarray(
            dsa_dcp.window_thresholds(jnp.asarray(starts)[:, None], lane[None])
        )
        for i, start in enumerate(starts):
            first = int(start) + (1 << 19) - self.INT_MIN
            self.assertTrue((1 << 19) <= first <= (1 << 32) - (1 << 25), centres[i])
            self.assertEqual(first % 4096, 0)
            union = self._union(start)
            got = table[i, :127].astype(np.int64) - self.INT_MIN
            np.testing.assert_array_equal(got, union, err_msg=str(centres[i]))
            self.assertEqual(table[i, 127], self.INT_MIN)  # 2^32
            if i % 5:
                continue
            keys = np.concatenate(
                [
                    union - 1,
                    union,
                    union + 1,
                    [0, 1, 2**32 - 2, 2**32 - 1],
                    rng.integers(0, 2**32, 300),
                ]
            )
            keys = np.clip(keys, 0, 2**32 - 1)
            digit = dsa_dcp.window_digit(
                jnp.asarray((keys + self.INT_MIN).astype(np.int32)), jnp.int32(start)
            )
            np.testing.assert_array_equal(
                np.asarray(digit),
                np.searchsorted(union, keys, side="right"),
                err_msg=str(centres[i]),
            )

    def test_search_finds_the_prefix_for_any_centre(self):
        import jax.numpy as jnp

        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp

        INT_MIN, lane = self.INT_MIN, jnp.arange(128, dtype=jnp.int32)
        rng = np.random.default_rng(1)

        def search(keys, topk, centre):
            """IndexSelect._window_update and _seeded_rounds, in numpy."""
            start = jnp.int32(np.asarray(dsa_dcp.window_start(jnp.int32(centre))))
            digit = np.asarray(dsa_dcp.window_digit(jnp.asarray(keys), start))
            counts = np.bincount(digit, minlength=128)[::-1].cumsum()[::-1][1:]
            d = int((counts >= topk).sum())
            table = np.asarray(dsa_dcp.window_thresholds(start, lane)).astype(np.int64)
            lo = INT_MIN if d == 0 else int(table[d - 1])
            fine = (int(table[d]) - lo) % 2**32 <= 2**19  # int32 differences wrap
            prefix, u = lo - INT_MIN, keys.astype(np.int64) - INT_MIN
            for shift in [12] if fine else [19, 12]:
                above = np.minimum((u[u >= prefix] - prefix) >> shift, 127)
                count = np.bincount(above, minlength=128)[::-1].cumsum()[::-1][1:]
                prefix += int((count >= topk).sum()) << shift
            return prefix, fine

        hits = misses = 0
        for trial in range(150):
            n = int(rng.integers(2, 5000))
            topk = int(rng.integers(1, n + 1)) if trial % 3 else min(n, 2048)
            kind = trial % 5
            if kind == 0:  # scores over a few binades
                keys = rng.lognormal(0, 1.5, n).astype(np.float32).view(np.int32)
            elif kind == 1:  # five distinct keys: ties
                keys = rng.choice(rng.integers(INT_MIN, 2**31, 5, dtype=np.int64), n)
            elif kind == 2:
                keys = rng.integers(INT_MIN, 2**31, n, dtype=np.int64)
            elif kind == 3:  # around a coarse threshold
                c = int(rng.integers(1, 64))
                keys = (c << 26) + INT_MIN + rng.integers(-(2**21), 2**21, n)
            else:  # int32 extremes
                keys = rng.choice(np.array([INT_MIN, INT_MIN + 1, -1, 0, 2**31 - 1]), n)
            keys = np.asarray(keys).astype(np.int32)
            kth = int(np.sort(keys.astype(np.int64) - INT_MIN)[::-1][topk - 1])
            want = (kth >> 12) << 12
            for centre in (
                want + INT_MIN,
                want + INT_MIN + (int(rng.integers(-40, 40)) << 19),
                int(rng.integers(INT_MIN, 2**31)),
                INT_MIN,
                2**31 - 1,
            ):
                got, fine = search(keys, topk, int(np.clip(centre, INT_MIN, 2**31 - 1)))
                self.assertEqual(got, want, (trial, kind, n, topk, centre))
                hits, misses = hits + fine, misses + (not fine)
        self.assertGreater(hits, 100)
        self.assertGreater(misses, 100)

    def test_centre_slots(self):
        import jax.numpy as jnp

        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp

        centres = np.arange(5 * 3 * 4, dtype=np.int32).reshape(
            5, 3, 4
        )  # [slots, layers, rows]
        slots = np.array(
            [2, 2, 2, 2, 0, 0, 3, 3], np.int32
        )  # rows of 4: padding on slot 0
        got = np.asarray(
            dsa_dcp.gather_centres(jnp.asarray(centres), jnp.asarray(slots), 4)
        )
        want = [centres[slots[b], m, b % 4] for m in range(3) for b in range(8)]
        np.testing.assert_array_equal(got, want)
        new = 1000 + np.arange(24, dtype=np.int32)
        out = np.asarray(
            dsa_dcp.scatter_centres(
                jnp.asarray(centres), jnp.asarray(slots), 4, jnp.asarray(new)
            )
        )
        expect = centres.copy()
        for m in range(3):
            for b in range(8):
                if slots[b]:
                    expect[slots[b], m, b % 4] = new[m * 8 + b]
        np.testing.assert_array_equal(out, expect)  # slot 0 unchanged


class TestIndexSelectSeededWindow(CustomTestCase):
    """``KernelConfig.dsa_seeded_window``: ``select_call`` in interpret mode
    on a 4-rank group over multi-step streams, each step's centres the last
    step's output: 8 rows of one growing sequence (B1 x7), 8 single-row
    sequences of mixed lengths (inactive rows, a row crossing topk, rows
    taking other requests) and two 8-row sequences (16 rows, two groups),
    with score scale jumps, garbage and edge-of-window centres, new requests
    with ties. Every step matches the reference (the topk-th key's 20-bit
    prefix bin) and the knob-off run bit for bit (thresholds, tie positions,
    own flags, keys); both window hits and misses occur, active rows' next
    centres are their thresholds (bit 0: the hit), inactive rows keep
    theirs, and every rank runs two count exchanges a hit and three a miss."""

    SCRIPT = """
        import os, sys
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        os.environ["JAX_PLATFORMS"] = "cpu"
        from sglang_tpu.k3_programs.kimi_k3 import config as kc
        knobs = dict(kv.split("=") for kv in os.environ["KNOBS"].split(",") if kv)
        kc.install(kc.KernelConfig().replace(**knobs))
        import ml_dtypes, numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3 import dsa
        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp

        SEEDED = kc.active().dsa_seeded_window
        _exchange = dsa_dcp.IndexSelect._exchange_wide
        def _counted(self, *args, **kwargs):
            pl.debug_print("ROUND_EXCHANGE")
            return _exchange(self, *args, **kwargs)
        dsa_dcp.IndexSelect._exchange_wide = _counted

        G, T, INT_MIN, TOPK = 4, 512, -(2**31), 300
        mesh = Mesh(np.asarray(jax.devices()), ("tp",))
        outputs, flags, steps = {}, [], [0]

        def stream(name, S, plan, seed, max_len):
            # plan: per step, (lengths of each sequence's first in-flight position, w scale
            # exponent per sequence, centre override or None); sequences keep their keys
            # (append-only) unless a step gives a sequence a new seed (a new request).
            rng = np.random.default_rng(seed)
            nseq = len(plan[0][0])
            B = nseq * S
            pages = -(-(max_len + T) // T) // G + 2
            blocks = 1 + nseq * pages
            calls = {}
            def kernel(centred):
                if centred in calls:
                    return calls[centred]
                def body(q8, w_t, pool, tables, positions, new_k8, new_exp, centres):
                    out = dsa_dcp.select_call(q8, w_t, new_k8, pool[0], tables[0], positions, new_exp,
                                              pages=pages, topk=TOPK, group=G, max_local_pages=pages,
                                              sequence_rows=S, interpret=pltpu.InterpretParams(),
                                              centres=centres if centred else None)
                    return tuple(x[None] for x in out)
                n_out = 5 if centred else 4
                f = jax.jit(jax.shard_map(body, mesh=mesh,
                                          in_specs=(P(), P(), P("tp"), P("tp"), P(), P(), P(), P()),
                                          out_specs=(P("tp"),) * n_out, check_vma=False))
                calls[centred] = f
                return f
            seqs = []
            def new_seq(s, kind, step):
                r = np.random.default_rng(seed * 1000 + s * 17 + step)
                if kind == "ties":  # few distinct keys: ties at the threshold
                    raw = r.standard_normal((7, 128)).astype(np.float32)[r.integers(0, 7, max_len + S)]
                else:
                    raw = r.standard_normal((max_len + S, 128)).astype(np.float32)
                q = [dsa.index_query(r.standard_normal(4096).astype(np.float32),
                                     r.standard_normal(32).astype(np.float32)) for _ in range(S)]
                return dict(raw=raw, k8=dsa.quantize_fp8(raw), rec=np.asarray(dsa_dcp.key_records(jnp.asarray(raw))), q=q)
            centres = np.zeros(B, np.int32)
            for step, (firsts, scales, kinds, override) in enumerate(plan):
                for s, kind in enumerate(kinds):
                    if step == 0 or kind is not None:
                        if len(seqs) <= s:
                            seqs.append(None)
                        seqs[s] = new_seq(s, kind or "plain", step)
                positions = np.array([p0 + r for p0 in firsts for r in range(S)], np.int32)
                pools = rng.integers(0, 256, (G, blocks, 132, 4, 128), dtype=np.uint8)
                tables = np.zeros((G, B, pages), np.int32)
                new_k8 = np.zeros((B, 128), np.float32); new_exp = np.zeros(B, np.int32)
                q8 = np.zeros((B, 32, 128), np.float32); w = np.zeros((B, 32), np.float32)
                for s, p0 in enumerate(firsts):
                    sq = seqs[s]
                    k8, _ = sq["k8"]
                    rec = sq["rec"]
                    for r in range(S):
                        b = s * S + r
                        tables[:, b] = 1 + s * pages + np.arange(pages)
                        new_k8[b] = k8[p0 + r]; new_exp[b] = rec[p0 + r, 128]
                        q8[b], w[b] = sq["q"][r]
                        w[b] *= np.float32(2.0) ** scales[s]
                    t = np.arange(p0)
                    page = t // T
                    blk = tables[page % G, s * S, page // G]
                    (row, sub), (srow, lane) = dsa_dcp.record_place(t % T)
                    pools[page % G, blk, row, sub] = rec[t, :128]
                    pools[page % G, blk, srow, 0, lane] = rec[t, 128]
                w_t = np.zeros((32, 128), np.float32); w_t[:, :B] = w.T
                if override is not None:
                    centres = np.asarray(override(centres), np.int32)
                args = (q8.astype(ml_dtypes.float8_e4m3fn), w_t, pools, tables.reshape(G, -1), positions,
                        new_k8, new_exp, centres)
                out = [np.asarray(x) for x in kernel(SEEDED)(*args)]
                tau, ptie, own, keys = out[:4]
                for b in range(B):
                    L = positions[b] + 1
                    t = np.arange(L)
                    kk = keys[(t // T) % G, b, (t // T) // G * (T // 128) + (t % T) // 128, t % 128].astype(np.int64)
                    for v in out[:3] + out[4:]:
                        assert len(set(v[:, b])) == 1, (name, step, b)
                    if L <= TOPK:
                        want_tau, want = INT_MIN, kk > INT_MIN
                    else:
                        kth = np.sort(kk)[::-1][TOPK - 1]
                        unsigned = (int(kth) ^ INT_MIN) & 0xFFFFFFFF
                        bound = ((unsigned >> 12) << 12) ^ (1 << 31)
                        want_tau = bound - (1 << 32) if bound >= (1 << 31) else bound
                        want = kk >= want_tau
                    assert tau[0, b] == want_tau, (name, step, b, tau[0, b], want_tau)
                    np.testing.assert_array_equal(dsa_dcp.selected(kk, t, tau[0, b], ptie[0, b]), want)
                    assert bool(own[0, b]) == bool(want[-1]), (name, step, b)
                    outputs[f"{name}_{step}_keys_{b}"] = kk
                for key, v in (("tau", tau), ("ptie", ptie), ("own", own)):
                    outputs[f"{name}_{step}_{key}"] = v[0]
                if SEEDED:
                    new = out[4][0]
                    active = positions + 1 > TOPK
                    # Active rows: tau | the window round's hit; others keep their centre.
                    hit = int(new[np.argmax(active)] & 1) if active.any() else 0
                    np.testing.assert_array_equal(new[active], tau[0][active] | hit)
                    np.testing.assert_array_equal(new[~active], centres[~active])
                    flags.append((name, step, int(active.any()), hit))
                    centres = new
                steps[0] += 1
                print(name, step, "taus", tau[0][:4], flush=True)

        # 8 rows of one sequence (B1 x7): the stream grows; the head weights scale by
        # 2^6 at step 3 and back at step 4 (misses); garbage centres at step 5; a new
        # request (ties) at step 6 keeping the old centres.
        garbage = lambda c: np.array([INT_MIN, 2**31 - 1, -1, 0, 1, dsa_dcp.CENTRE_HIGH + 1, 12345, -98765], np.int64).astype(np.int32)
        edge = lambda c: c + (np.array([31, 32, -32, -33, 30, -31, 33, 0]) << 19)
        plain = [None]
        plan8 = [
            ([2800], [0], plain, None),
            ([2803], [0], [None], None),
            ([2808], [0], [None], None),
            ([2811], [6], [None], None),
            ([2815], [0], [None], None),
            ([2820], [0], [None], garbage),
            ([2560], [0], ["ties"], None),
            ([2563], [0], [None], edge),
            ([2568], [-1], [None], None),
        ]
        stream("b8s8", 8, plan8, 0, 2900)
        # 8 single-row sequences, mixed lengths (some at most topk: inactive), a row
        # crossing topk, rows exchanging requests (centres stay with the rows).
        lens = [100, 299, 300, 700, 1500, 2100, 2600, 3000]
        plan1 = [
            (lens, [0] * 8, [None] * 8, None),
            ([l + 1 for l in lens], [0] * 8, [None] * 8, None),
            ([l + 2 for l in lens], [0] * 8, [None] * 8, None),
            ([l + 3 for l in lens], [0, 0, 0, 0, 0, 0, 0, 3], [None] * 8, None),
            ([lens[(i + 3) % 8] + 4 for i in range(8)], [0] * 8, [None] * 8, None),
            ([l + 5 for l in lens], [0] * 8, [None, None, "ties", None, None, None, None, "plain"], edge),
            ([l + 6 for l in lens], [0] * 8, [None] * 8, None),
        ]
        stream("b8s1", 1, plan1, 1, 3200)
        # Two 8-row sequences (B2 x7: 16 rows, two groups): one hits, one misses.
        plan16 = [
            ([1200, 2600], [0, 0], [None, None], None),
            ([1205, 2604], [0, 0], [None, None], None),
            ([1208, 2610], [0, -5], [None, None], None),
            ([1211, 2612], [0, -5], [None, None], None),
        ]
        stream("b16s8", 8, plan16, 2, 2700)
        np.savez(os.environ["OUT"], **outputs)
        # Exchanges a rank runs: three a selection, seeded two when the window hit
        # (or no row is active).
        expect = 3 * steps[0]
        if SEEDED:
            live = [f for f in flags if f[2]]
            print("flags", flags)
            print("WINDOW hits", sum(f[3] for f in live), "misses", sum(1 - f[3] for f in live))
            expect = sum(2 if hit or not active else 3 for _, _, active, hit in flags)
        print("EXPECT", expect)
        print("OK")
    """

    # The production scoring knobs (16 rows: no resident keys).
    KNOBS = (
        "dsa_select_rounds=5,dsa_page_slots=8,dsa_resident_keys=1,"
        "dsa_patch_loop=1,dsa_stage_slots=4,dsa_early_plan=1"
    )

    def test_seeded_window_matches_three_rounds_bit_for_bit(self):
        import os
        import re
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        stdout = {}
        with tempfile.TemporaryDirectory() as tmp:
            for name, knobs in (
                ("base", self.KNOBS),
                ("seeded", self.KNOBS + ",dsa_seeded_window=1"),
            ):
                env = dict(
                    os.environ,
                    PYTHONPATH=os.pathsep.join(sys.path),
                    KNOBS=knobs,
                    OUT=f"{tmp}/{name}.npz",
                )
                result = subprocess.run(
                    [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=2400,
                )
                self.assertIn(
                    "OK",
                    result.stdout,
                    f"{name}: {result.stdout[-2000:]}{result.stderr[-3000:]}",
                )
                expect = int(re.search(r"^EXPECT (\d+)$", result.stdout, re.M)[1])
                self.assertEqual(
                    result.stdout.count("ROUND_EXCHANGE"), 4 * expect, name
                )
                stdout[name] = result.stdout
            hits, misses = map(
                int,
                re.search(r"WINDOW hits (\d+) misses (\d+)", stdout["seeded"]).groups(),
            )
            self.assertGreaterEqual(hits, 5, stdout["seeded"][-1500:])
            self.assertGreaterEqual(misses, 5, stdout["seeded"][-1500:])
            base, seeded = np.load(f"{tmp}/base.npz"), np.load(f"{tmp}/seeded.npz")
            self.assertEqual(sorted(base.files), sorted(seeded.files))
            for name in base.files:
                np.testing.assert_array_equal(base[name], seeded[name], err_msg=name)


class TestCrossedPairGathers(CustomTestCase):
    """``KernelConfig.crossed_pair_gathers``: the four-owner gathers of the
    pairs' rows (``_gather_local_pair_rows(cross_cores=True)``) in Pallas
    interpret mode on 32 host devices (TestAttentionHostBf16's harness, stale
    NaN bytes in the destinations and the transport wires), at 2, 3 and 4 rows
    a pair (B8 and B16 copy in place, B12 through flat wires). Each rank's
    slot of every other pair holds the rows of that pair's rank of the other
    parity (rank ^ 2k ^ 1) and its own slot its own rows (without the knob:
    the same parity, rank ^ 2k). With both ranks of each pair holding the
    same rows (the decoder's rows before attention) the gather equals the
    same-parity gather bit for bit. Then the decoder's order on rank-row
    steps: the aux capture's gather of a layer's rows and the next layer's
    input gather back to back on the same semaphores, each read as it returns,
    with the core-0 ranks starting late (eager DMAs: each sender's copies land
    in issue order). Both crossed, each read sees its own round's rows
    (crossing only one of the two fails here: one round's copy completes the
    other round's wait)."""

    SCRIPT = TestAttentionHostBf16.PRELUDE + r"""
rng = np.random.default_rng(2)
vmem = pltpu.VMEM
LAG = int(os.environ.get("LAG", "100"))  # core-0 lag: loop trips a pair index


def run(kernel, rows_in, outputs, **interpret):
    rows = rows_in.shape[1]

    def body(x):
        return pl.pallas_call(
            kernel,
            out_shape=[jax.ShapeDtypeStruct((4 * rows, H), jnp.bfloat16)] * outputs,
            in_specs=[pl.BlockSpec(memory_space=vmem)],
            out_specs=[pl.BlockSpec(memory_space=vmem)] * outputs,
            scratch_shapes=[vmem((8, 224, 128), jnp.float32), vmem((8, 128), jnp.float32),
                            pltpu.SemaphoreType.DMA((16,)), pltpu.SemaphoreType.DMA((16,)),
                            pltpu.SemaphoreType.REGULAR],
            compiler_params=pltpu.CompilerParams(has_side_effects=True),
            interpret=pltpu.InterpretParams(**interpret),
        )(x[0])

    f = jax.jit(jax.shard_map(lambda x: [y[None] for y in body(x)], mesh=mesh,
                              in_specs=P("tp"), out_specs=[P("tp")] * outputs,
                              check_vma=False))
    return [np.asarray(y) for y in f(rows_in)]


def handshake(ready):
    # Every local rank has filled its buffers before any copy lands (the
    # decoder's previous all-local round).
    rank = jax.lax.axis_index("tp")
    for offset in range(1, 8):
        pltpu.semaphore_signal(ready, 1, device_id=(rank ^ offset,),
                               device_id_type=pl.DeviceIdType.MESH)
    pltpu.semaphore_wait(ready, 7)


def gather(rows_in, cross):
    def kernel(rows_ref, out_ref, wires, spin, sends, recvs, ready):
        wires[...] = jnp.full(wires.shape, jnp.nan, jnp.float32)
        out_ref[...] = jnp.full(out_ref.shape, jnp.nan, jnp.bfloat16)
        handshake(ready)
        dmk._gather_local_pair_rows(rows_ref, out_ref, sends, recvs, wires, owners=4,
                                    cross_cores=cross)

    return run(kernel, rows_in, 1)[0]


def expected(rows_in, cross):
    # Rank r's slot of pair p: rank 8 h + 2 p + (r % 2), or its other core.
    rows = rows_in.shape[1]
    out = np.empty((32, 4 * rows, H), rows_in.dtype)
    for r in range(32):
        for p in range(4):
            source = 8 * (r // 8) + 2 * p + (r % 2)
            if cross and p != (r % 8) // 2:
                source ^= 1
            out[r, p * rows:(p + 1) * rows] = rows_in[source]
    return out


bits = lambda x: x.view(np.uint16)
for rows in (2, 3, 4):
    distinct = rng.standard_normal((32, rows, H)).astype(BF16)
    assert not np.array_equal(bits(expected(distinct, True)), bits(expected(distinct, False)))
    for cross in (False, True):
        got = gather(distinct, cross)
        assert np.array_equal(bits(got), bits(expected(distinct, cross))), (rows, cross)
    print(f"{rows} rows a pair: same-parity and crossed sources", flush=True)
for rows in (2, 4):
    paired = np.repeat(rng.standard_normal((16, rows, H)).astype(BF16), 2, axis=0)
    same, crossed = gather(paired, False), gather(paired, True)
    assert np.array_equal(bits(same), bits(expected(paired, False))), rows
    assert np.array_equal(bits(crossed), bits(same)), rows
    print(f"{rows} rows a pair: crossed gather bit-exact", flush=True)


def back_to_back(rows_in, aux_cross=True, input_cross=True):
    # The aux capture's gather of the rows, then the next layer's input gather
    # of new rows (here negated) on semaphores 0..2, no round between them;
    # each gather's rows are read as soon as it returns (the capture's copy,
    # the attention). After the handshake core 0 of pair p starts
    # (p + 1) * LAG loop trips late.
    def kernel(rows_ref, aux_out, input_out, wires, spin, sends, recvs, ready):
        def gathered(aux, inputs):
            for rows in (aux, inputs):
                rows[...] = jnp.full(rows.shape, jnp.nan, jnp.bfloat16)
            wires[...] = jnp.full(wires.shape, jnp.nan, jnp.float32)
            handshake(ready)
            rank = jax.lax.axis_index("tp")
            spin[...] = jnp.zeros(spin.shape, jnp.float32)

            def lag(i, carry):
                spin[0:1, :] = spin[0:1, :] + 1.0
                return carry

            trips = jnp.where(rank % 2 == 0, (rank % 8) // 2 + 1, 0) * LAG
            jax.lax.fori_loop(0, trips, lag, 0)
            dmk._gather_local_pair_rows(rows_ref, aux, sends, recvs, wires, owners=4,
                                        cross_cores=aux_cross)
            aux_out[...] = aux[...]
            rows_ref[...] = -rows_ref[...]
            dmk._gather_local_pair_rows(rows_ref, inputs, sends, recvs, wires, owners=4,
                                        cross_cores=input_cross)
            input_out[...] = inputs[...]

        pl.run_scoped(gathered, vmem(aux_out.shape, jnp.bfloat16),
                      vmem(input_out.shape, jnp.bfloat16))

    return run(kernel, rows_in, 2, dma_execution_mode="eager")


for rows in (2, 4):
    paired = np.repeat(rng.standard_normal((16, rows, H)).astype(BF16), 2, axis=0)
    aux, inputs = back_to_back(paired)
    assert np.array_equal(bits(aux), bits(expected(paired, False))), rows
    assert np.array_equal(bits(inputs), bits(expected(-paired, False))), rows
    print(f"{rows} rows a pair: back-to-back crossed gathers bit-exact", flush=True)
print("OK")
"""

    def test_crossed_gathers_match_the_same_parity_gathers(self):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), NPROC="128")
        result = subprocess.run(
            [sys.executable, "-c", self.SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            timeout=1500,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )


class TestAttentionInputGatherInBranches(CustomTestCase):
    """``KernelConfig.attention_input_gather_in_branches``: the rank-row
    steps' gather of the attention input rows
    (``_gather_local_pair_rows(defer_waits=True)``: the four pairs' rows
    copied in place, crossed and same-parity) started before a two-branch
    ``cond`` and waited for in each branch through the returned function, in
    Pallas interpret mode on 32 host devices (TestAttentionHostBf16's
    harness, stale NaN bytes in the rows and the transport wires). One
    branch waits first (the decoder's KDA branch); the other first starts
    and waits a local copy, holds the core-0 ranks back and waits inside a
    loop of one trip (the MLA branch's weight copies, overlay handler and
    prefix region). At 2 and 4 rows a pair (B8, B16), in two rounds with
    different rows and both branches taken in each, the rows read in either
    branch equal the in-line gather's bit for bit; each rank then writes NaN
    over its source (free once the waits return: a copy still reading it
    would deliver NaN). Then the decoder's order on these steps: the aux
    capture's in-line gather and the next layer's input gather waited in the
    branch, back to back on semaphores 0..2 with the core-0 ranks starting
    late (eager DMAs). DMAs on wait and eager; no DMA semaphore is left
    nonzero; with happens-before race detection no read of the rows races
    a copy."""

    SCRIPT = (
        TestAttentionHostBf16.PRELUDE
        + r"""
import sys

DMA_MODE = sys.argv[1]
# Happens-before race detection (slower): the crossed 2-row cases only.
RACES = len(sys.argv) > 2 and sys.argv[2] == "races"
rng = np.random.default_rng(3)
vmem = pltpu.VMEM
LAG = int(os.environ.get("LAG", "100"))  # core-0 lag: loop trips a pair index
bits = lambda x: x.view(np.uint16)
nan = lambda ref: jnp.full(ref.shape, jnp.nan, ref.dtype)


def run(kernel, inputs, out_shapes, scratch, mode=None):
    def body(*xs):
        outs = pl.pallas_call(
            kernel,
            out_shape=out_shapes,
            in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)]
            + [pl.BlockSpec(memory_space=vmem)] * (len(xs) - 1),
            out_specs=[pl.BlockSpec(memory_space=vmem)] * len(out_shapes),
            scratch_shapes=scratch,
            compiler_params=pltpu.CompilerParams(has_side_effects=True),
            interpret=pltpu.InterpretParams(dma_execution_mode=mode or DMA_MODE,
                                            detect_races=RACES),
        )(*(x[0] for x in xs))
        return [y[None] for y in outs]

    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("tp"),) * len(inputs),
                              out_specs=[P("tp")] * len(out_shapes), check_vma=False))
    return [np.asarray(y) for y in f(*inputs)]


def handshake(ready):
    # Every local rank has filled its buffers before any copy lands. One
    # semaphore per handshake, so a fast rank's next handshake cannot count
    # towards this one.
    rank = jax.lax.axis_index("tp")
    for offset in range(1, 8):
        pltpu.semaphore_signal(ready, 1, device_id=(rank ^ offset,),
                               device_id_type=pl.DeviceIdType.MESH)
    pltpu.semaphore_wait(ready, 7)


def lag(spin, trips):
    spin[...] = jnp.zeros(spin.shape, jnp.float32)

    def trip(i, carry):
        spin[0:1, :] = spin[0:1, :] + 1.0
        return carry

    jax.lax.fori_loop(0, trips, trip, 0)


def expected(rows_in, cross):
    # Rank r's slot of pair p: rank 8 h + 2 p + (r % 2), or its other core.
    rows = rows_in.shape[1]
    out = np.empty((32, 4 * rows, H), rows_in.dtype)
    for r in range(32):
        for p in range(4):
            source = 8 * (r // 8) + 2 * p + (r % 2)
            if cross and p != (r % 8) // 2:
                source ^= 1
            out[r, p * rows:(p + 1) * rows] = rows_in[source]
    return out


def in_branches(rows_in, select, cross):
    # rows_in [32, 2, r, H]: two rounds' rows; select [32, 2]: each round's
    # branch. Each round gathers its rows in line, then again with the waits
    # deferred into a two-branch cond: branch 0 waits first (the KDA
    # branch); branch 1 first starts and waits a local copy and spins the
    # core-0 ranks (the MLA branch's weight copies and overlay handler) and
    # waits for the rows inside a loop of one trip (the MLA prefix region).
    # After the waits each rank reads the rows, then writes NaN over its
    # source (free once the waits have returned) and the rows.
    rows = rows_in.shape[2]

    def kernel(select_ref, rows_ref, inline_out, deferred_out, source, rows_vmem, wires,
               spin, other, sends, recvs, local_sems, ready):
        rank = jax.lax.axis_index("tp")
        for rnd in range(2):
            source[...] = rows_ref[rnd]
            rows_vmem[...] = nan(rows_vmem)
            wires[...] = nan(wires)
            handshake(ready.at[3 * rnd])
            dmk._gather_local_pair_rows(source, rows_vmem, sends, recvs, wires, owners=4,
                                        cross_cores=cross)
            inline_out[rnd] = rows_vmem[...]
            rows_vmem[...] = nan(rows_vmem)
            handshake(ready.at[3 * rnd + 1])
            wait_for_rows = dmk._gather_local_pair_rows(
                source, rows_vmem, sends, recvs, wires, owners=4, cross_cores=cross,
                defer_waits=True)

            def wait_first():
                wait_for_rows()
                deferred_out[rnd] = rows_vmem[...]

            def wait_later():
                copy = pltpu.make_async_copy(rows_ref.at[1 - rnd], other, local_sems.at[0])
                copy.start()
                lag(spin, jnp.where(rank % 2 == 0, (rank % 8) // 2 + 1, 0) * LAG)
                copy.wait()

                def one_trip(_, carry):
                    wait_for_rows()
                    deferred_out[rnd] = rows_vmem[...]
                    return carry

                jax.lax.fori_loop(0, jnp.minimum(select_ref[rnd], 1), one_trip, 0)

            jax.lax.cond(select_ref[rnd] == 0, wait_first, wait_later)
            source[...] = nan(source)
            rows_vmem[...] = nan(rows_vmem)
            handshake(ready.at[3 * rnd + 2])

    out = [jax.ShapeDtypeStruct((2, 4 * rows, H), jnp.bfloat16)] * 2
    scratch = [vmem((rows, H), jnp.bfloat16), vmem((4 * rows, H), jnp.bfloat16),
               vmem((8, 224, 128), jnp.float32), vmem((8, 128), jnp.float32),
               vmem((rows, H), jnp.bfloat16), pltpu.SemaphoreType.DMA((16,)),
               pltpu.SemaphoreType.DMA((16,)), pltpu.SemaphoreType.DMA((1,)),
               pltpu.SemaphoreType.REGULAR((6,))]
    return run(kernel, (select, rows_in), out, scratch)


for rows in (2,) if RACES else (2, 4):  # B8 and B16 rank-row steps
    distinct = rng.standard_normal((32, 2, rows, H)).astype(BF16)
    distinct[:, 1] = -distinct[:, 1]
    # Every pair of rounds through both branches, on different ranks.
    select = np.stack([np.array([r % 2, 1 - r % 2]) for r in range(32)]).astype(np.int32)
    select[8:16] = select[8:16, ::-1]
    for cross in (True,) if RACES else (False, True):
        inline, deferred = in_branches(distinct, select, cross)
        for rnd in range(2):
            want = expected(distinct[:, rnd], cross)
            assert np.array_equal(bits(inline[:, rnd]), bits(want)), (rows, cross, rnd)
            assert np.array_equal(bits(deferred[:, rnd]), bits(inline[:, rnd])), (rows, cross, rnd)
        print(f"{rows} rows a pair, crossed={cross}, dma={DMA_MODE}: rows waited in either "
              "branch bit-exact", flush=True)


def back_to_back(rows_in, select):
    # The decoder's order on rank-row steps: the aux capture's gather of a
    # layer's rows (in line), then the next layer's input gather of new rows
    # (here negated) with its waits in the attention branch, both crossed, on
    # semaphores 0..2 with no round between them; each read as soon as its
    # waits return. After the handshake core 0 of pair p starts
    # (p + 1) * LAG loop trips late. Eager DMAs: each sender's copies land in
    # issue order (on wait, the interpreter may run a semaphore's later copy
    # first).
    rows = rows_in.shape[1]

    def kernel(select_ref, rows_ref, aux_out, input_out, source, aux, inputs, wires, spin,
               sends, recvs, ready):
        for ref in (aux, inputs):
            ref[...] = nan(ref)
        wires[...] = nan(wires)
        source[...] = rows_ref[...]
        handshake(ready.at[0])
        rank = jax.lax.axis_index("tp")
        lag(spin, jnp.where(rank % 2 == 0, (rank % 8) // 2 + 1, 0) * LAG)
        dmk._gather_local_pair_rows(source, aux, sends, recvs, wires, owners=4,
                                    cross_cores=True)
        aux_out[...] = aux[...]
        source[...] = -source[...]
        wait_for_rows = dmk._gather_local_pair_rows(
            source, inputs, sends, recvs, wires, owners=4, cross_cores=True,
            defer_waits=True)

        def branch():
            wait_for_rows()
            input_out[...] = inputs[...]

        jax.lax.cond(select_ref[0] == 0, branch, branch)

    out = [jax.ShapeDtypeStruct((4 * rows, H), jnp.bfloat16)] * 2
    scratch = [vmem((rows, H), jnp.bfloat16), vmem((4 * rows, H), jnp.bfloat16),
               vmem((4 * rows, H), jnp.bfloat16), vmem((8, 224, 128), jnp.float32),
               vmem((8, 128), jnp.float32), pltpu.SemaphoreType.DMA((16,)),
               pltpu.SemaphoreType.DMA((16,)), pltpu.SemaphoreType.REGULAR((1,))]
    return run(kernel, (select, rows_in), out, scratch, mode="eager")


for rows in (2,) if RACES else (2, 4):
    paired = np.repeat(rng.standard_normal((16, rows, H)).astype(BF16), 2, axis=0)
    select = (np.arange(32, dtype=np.int32) % 2)[:, None]
    aux, inputs = back_to_back(paired, select)
    assert np.array_equal(bits(aux), bits(expected(paired, False))), rows
    assert np.array_equal(bits(inputs), bits(expected(-paired, False))), rows
    print(f"{rows} rows a pair: aux capture then input gather waited in the branch "
          "bit-exact", flush=True)
print("OK")
"""
    )

    def _run(self, *args):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), NPROC="128")
        result = subprocess.run(
            [sys.executable, "-c", self.SCRIPT, *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=1500,
        )
        output = result.stdout[-2000:] + result.stderr[-3000:]
        self.assertIn("OK", result.stdout, output)
        self.assertNotIn("non-zero count", result.stdout, output)
        self.assertNotIn("RACE DETECTED", result.stdout, output)

    def test_rows_waited_in_either_branch(self):
        self._run("on_wait")

    def test_rows_waited_in_either_branch_with_eager_dmas(self):
        self._run("eager")

    def test_rows_waited_in_either_branch_race_free(self):
        self._run("eager", "races")


class TestSelectExchangeHook(CustomTestCase):
    """``IndexSelect.select(in_exchange, keys_ready)`` (the decode's
    ``dsa_prologue_in_select`` runs the attention's prologue in the first and
    its first key copies in the second): ``in_exchange`` runs once, in the
    second exchange round after its sends and before its wait; ``keys_ready``
    once, after the last round and the resident keys' copies out; work in
    them leaves the thresholds, own flags and keys bit for bit unchanged
    (``TestIndexSelectBins``' harness with the recipe's scoring knobs: an
    8-row sequence and two one-row sequences with five bin rounds, the
    8-row sequence with the eight exact rounds)."""

    TAIL = """
        from jax.experimental import pallas as pl
        events = []
        real_remote, real_copy = pltpu.make_async_remote_copy, pltpu.make_async_copy
        real_wide, real_round = dsa_dcp.IndexSelect._key_round_wide, dsa_dcp.IndexSelect._key_round

        def remote(*args, **kwargs):
            events.append("send")
            return real_remote(*args, **kwargs)

        def copy(*args, **kwargs):
            events.append("copy")
            return real_copy(*args, **kwargs)

        def logged(real):
            def round_(self, *args, **kwargs):
                events.append("round")
                return real(self, *args, **kwargs)
            return round_

        def work(name):  # independent work: a scoped buffer written
            def run():
                events.append(name)
                def body(buf):
                    buf[...] = jnp.full(buf.shape, 1.0, jnp.float32)
                pl.run_scoped(body, pltpu.VMEM((8, 128), jnp.float32))
            return run

        pltpu.make_async_remote_copy, pltpu.make_async_copy = remote, copy
        dsa_dcp.IndexSelect._key_round_wide = logged(real_wide)
        dsa_dcp.IndexSelect._key_round = logged(real_round)
        real_select = dsa_dcp.IndexSelect.select
        cases = (([3100], 8, 400, 2), ([300, 2500], 1, 256, 0))
        for firsts, S, topk, seed in cases[: int(os.environ["HOOK_CASES"])]:
            dsa_dcp.IndexSelect.select = real_select
            plain = case(firsts, S, topk, seed)
            events.clear()
            dsa_dcp.IndexSelect.select = lambda self: real_select(
                self, in_exchange=work("hook"), keys_ready=work("ready"))
            hooked = case(firsts, S, topk, seed)
            for a, b in zip(plain, hooked):
                np.testing.assert_array_equal(a, b)
            assert events.count("hook") == 1, events.count("hook")
            rounds = [i for i, e in enumerate(events) if e == "round"]
            second = events[rounds[1] : rounds[2] if len(rounds) > 2 else len(events)]
            at = second.index("hook")
            assert "send" in second[:at] and "send" not in second[at:], second
            assert "copy" in second[at:], second  # the round's wait follows
            ready = events.index("ready")  # after every DMA wait (the keys' copies out too)
            assert events.count("ready") == 1 and ready > rounds[-1], events
            assert not {"send", "copy"} & set(events[ready:]), events[ready:]
        print("OK")
    """

    def _run(self, rounds, cases):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        harness = textwrap.dedent(TestIndexSelectBins.SCRIPT)
        harness = harness[: harness.index("\ncase([300, 2500], 1, 256, 0)\n") + 1]
        install = "kc.install(kc.KernelConfig(dsa_select_rounds=ROUNDS))"
        assert install in harness
        harness = harness.replace(  # resident keys: their copies out precede keys_ready
            install,
            "kc.install(kc.KernelConfig(dsa_select_rounds=ROUNDS, dsa_resident_keys=True, "
            "dsa_stage_slots=4, dsa_page_slots=8, dsa_patch_loop=True))",
        )
        env = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join(sys.path),
            ROUNDS=str(rounds),
            HOOK_CASES=str(cases),
        )
        result = subprocess.run(
            [sys.executable, "-c", harness + textwrap.dedent(self.TAIL)],
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )

    def test_hook_in_the_second_bin_round(self):
        self._run(5, cases=2)

    def test_hook_in_the_second_exact_round(self):
        self._run(8, cases=1)  # the 8-row sequence (the exact rounds compile slowly)


class TestIndexSelectFillsAndPatch(TestIndexSelectScoringKnobs):
    """``KernelConfig.dsa_exchange_fill`` (own keys, p_tie and the resident
    keys' copy wait in the first count exchange's wait),
    ``dsa_vector_patch`` and ``dsa_skip_empty_subpages`` leave the selection
    bit for bit unchanged: ``select_call`` in interpret mode on a 4-rank
    group, with the defaults and with the knobs over the resident chunks and
    over the stage ring, must give the same thresholds, own flags and every
    key of each row's pages (the masked tail of the last page too). The
    in-flight tokens sit at a page's start (three empty subpages), across a
    subpage and a page (rank) boundary, at gapped and repeated positions
    (the patch's row order), in one-row and four-row sequences and in the
    resident keys' second chunk; one sequence holds at most ``topk`` tokens
    (p_tie -1); exact rounds keep p_tie from the tie rounds."""

    SCRIPT = """
        import os, sys
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        os.environ["JAX_PLATFORMS"] = "cpu"
        from sglang_tpu.k3_programs.kimi_k3 import config as kc
        knobs = dict(kv.split("=") for kv in os.environ["KNOBS"].split(",") if kv)
        kc.install(kc.KernelConfig().replace(**knobs))
        import ml_dtypes, numpy as np, jax, jax.numpy as jnp
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3 import dsa
        from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp

        G, T, INT_MIN = 4, 512, -(2**31)
        ROUNDS = kc.active().dsa_select_rounds
        outputs = {}

        def case(name, firsts, S, topk, seed, chunk=128, ties=False, offsets=None):
            dsa_dcp.WIDE_CHUNK_ROWS = chunk
            rng = np.random.default_rng(seed)
            B = len(firsts) * S
            offsets = list(range(S)) if offsets is None else offsets  # row r at p0 + offsets[r]
            positions = np.array([p0 + offsets[r] for p0 in firsts for r in range(S)], np.int32)
            pages = max(-(-((p + T) // T) // G) for p in positions) + 1
            pools = rng.integers(0, 256, (G, 1 + len(firsts) * pages, 132, 4, 128), dtype=np.uint8)
            tables = np.zeros((G, B, pages), np.int32)
            new_k8 = np.zeros((B, 128), np.float32); new_exp = np.zeros(B, np.int32)
            for s, p0 in enumerate(firsts):
                L = p0 + S
                raw = rng.standard_normal((L, 128)).astype(np.float32)
                if ties:  # five distinct keys: ties at the threshold
                    raw = rng.standard_normal((5, 128)).astype(np.float32)[rng.integers(0, 5, L)]
                k8, kexp = dsa.quantize_fp8(raw)
                rec = np.asarray(dsa_dcp.key_records(jnp.asarray(raw)))
                for r in range(S):
                    tables[:, s * S + r] = 1 + s * pages + np.arange(pages)
                    new_k8[s * S + r] = k8[p0 + r]; new_exp[s * S + r] = rec[p0 + r, 128]
                for t in range(p0):
                    page = t // T
                    blk = tables[page % G, s * S, page // G]
                    (row, sub), (srow, lane) = dsa_dcp.record_place(t % T)
                    pools[page % G, blk, row, sub] = rec[t, :128]
                    pools[page % G, blk, srow, 0, lane] = rec[t, 128]
            q8 = np.zeros((B, 32, 128), np.float32); w = np.zeros((B, 32), np.float32)
            for b in range(B):
                q8[b], w[b] = dsa.index_query(rng.standard_normal(4096).astype(np.float32),
                                              rng.standard_normal(32).astype(np.float32))
            w_t = np.zeros((32, 128), np.float32); w_t[:, :B] = w.T
            mesh = Mesh(np.asarray(jax.devices()), ("tp",))
            def body(q8, w_t, pool, tables, positions, new_k8, new_exp):
                out = dsa_dcp.select_call(q8, w_t, new_k8, pool[0], tables[0], positions, new_exp,
                                          pages=pages, topk=topk, group=G, max_local_pages=pages,
                                          sequence_rows=S, interpret=pltpu.InterpretParams())
                return tuple(x[None] for x in out)
            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P(), P("tp"), P("tp"), P(), P(), P()),
                                      out_specs=(P("tp"),) * 4, check_vma=False))
            tau, ptie, own, keys = (np.asarray(x) for x in f(
                q8.astype(ml_dtypes.float8_e4m3fn), w_t, pools, tables.reshape(G, -1), positions, new_k8, new_exp))
            row_keys = []
            for b in range(B):
                L, first = positions[b] + 1, b - b % S
                span = -(-(positions[first:first + S].max() + 1) // T) * T  # the sequence's pages
                t = np.arange(span)
                kk = keys[(t // T) % G, b, (t // T) // G * (T // 128) + (t % T) // 128, t % 128].astype(np.int64)
                row_keys.append(kk)
                for v in (tau, ptie, own):
                    assert len(set(v[:, b])) == 1, (name, b)
                if offsets != list(range(S)):
                    continue  # gapped positions: compared across the knob sets only
                assert np.all(kk[L:] == INT_MIN), (name, b)
                kk, t = kk[:L], t[:L]
                if ROUNDS == 8:
                    want = dsa.topk_mask(kk, t, topk)
                elif L <= topk:
                    want = kk > INT_MIN
                else:
                    kth = np.sort(kk)[::-1][topk - 1]
                    drop = 32 - 4 * ROUNDS
                    unsigned = (int(kth) ^ INT_MIN) & 0xFFFFFFFF
                    bound = ((unsigned >> drop) << drop) ^ (1 << 31)
                    want = kk >= (bound - (1 << 32) if bound >= (1 << 31) else bound)
                np.testing.assert_array_equal(dsa_dcp.selected(kk, t, tau[0, b], ptie[0, b]), want)
                assert bool(own[0, b]) == bool(want[-1]), (name, b)
            for key, v in (("tau", tau), ("ptie", ptie), ("own", own), ("keys", np.concatenate(row_keys))):
                outputs[f"{name}_{key}"] = v

        if ROUNDS == 8:
            case("ties", [1500, 2600], 2, 300, 1, ties=True)
            case("ties8", [3069], 8, 400, 6, ties=True)
        else:
            case("b8s8", [6 * T - 3], 8, 400, 2)  # in-flight tokens on two pages (ranks)
            case("b8s8p", [4 * T], 8, 400, 7)  # a page's first tokens: three empty subpages
            case("b8s8m", [5 * T + 250], 8, 400, 8)  # across a subpage boundary
            case("b8s8d", [5 * T + 250], 8, 400, 9, offsets=[0, 1, 1, 3, 3, 4, 6, 7])
            case("b8s1", [300, 2500, 1100, 3600, 700, 2049, 1500, 3000], 1, 256, 3)
            case("b8s4", [2 * T + 125, 3 * T + 380], 4, 300, 10)  # two-subpage products
            case("short", [100, 2600], 4, 300, 11)  # at most topk tokens: p_tie -1
            case("two24", [9500], 8, 300, 4, chunk=16)  # the second chunk, clamped
        np.savez(os.environ["OUT"], **outputs)
        print("OK")
    """

    FILLS = "dsa_exchange_fill=1,dsa_vector_patch=1,dsa_skip_empty_subpages=1"
    # Over the stage ring (depths not powers of two), then the resident keys.
    KNOBS = "dsa_stage_slots=3,dsa_page_slots=6," + FILLS
    RESIDENT = "dsa_page_slots=8,dsa_resident_keys=1,dsa_patch_loop=1," + FILLS

    def test_bin_selection_is_unchanged(self):
        # dsa_odd_stride: the vector patch's strided key rows at put's stride.
        self._runs(5, (self.KNOBS, self.RESIDENT, f"{self.RESIDENT},{self.ODD}"))

    def test_exact_selection_with_ties_is_unchanged(self):
        self._runs(8, (self.RESIDENT,))


class TestAttentionExactKnobs(CustomTestCase):
    """Default-off rewrites of the DSA verify attention that must not change
    a bit (``TestSelectedAttention``'s harness, interpret mode, 4-rank group):
    ``dsa_row_block_masks`` (row-block score masks in compact, window and
    dense folds), ``dcp_shift_scalars`` (shift/mask step scalars, the cached
    rows' step found before the loop), ``dcp_query_tail_unpack`` (FP8 query
    packets widen only the last lane tile) and ``dsa_prologue_in_select``
    (the gather wait, query unpack and state reset staged by another
    attention after ``send_queries``, as the decode's selection runs them).
    Every case runs with all of them off and on and compares the outputs
    bit for bit: B1 x7 (one 8-row sequence), two 4-row sequences with dense
    and compact steps, a small window with flushes, poisoned page slots,
    the chunked 16-row verifies of two 8-row sequences (B2 x7) and of four
    4-row sequences (B4 x3, two a chunk) and plain rows in two row groups;
    early-query cases with and without staging."""

    TAIL = """
        KNOBS = ("dsa_row_block_masks", "dsa_prologue_in_select", "dcp_shift_scalars",
                 "dcp_query_tail_unpack")
        for args, kwargs, modes in (
            (([2600], 8, 40, 7), dict(grouped=True, compact=128, window=512), (False, True)),
            (([1500, 3000], 4, 200, 1), dict(grouped=True, compact=128, window=512, expect=("compact", "dense")),
             (False, True)),
            (([900, 2600], 8, 40, 6), dict(grouped=True, compact=64, window=64), (False,)),
            (([700, 9000], 4, 8, 4), dict(grouped=True, compact=64, poison=True, window=256), (True,)),
            (([1500, 3000], 4, 24, 5), dict(grouped=True, compact=64, expect=("compact",)), (True,)),
            (([700, 2000], 1, 64, 0), dict(), (True,)),
            (([700, 1500, 2600, 3000], 4, 24, 8), dict(grouped=True, compact=64, window=128), (False,)),
        ):
            for sent in modes:
                off = case(*args, **kwargs, sent=sent)
                on = case(*args, **kwargs, sent=sent, knobs=KNOBS)
                assert np.array_equal(off.view(np.uint32), on.view(np.uint32)), (args, sent, np.abs(off - on).max())
        print("OK")
    """

    def test_knobs_leave_the_outputs_bit_identical(self):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        harness = textwrap.dedent(TestSelectedAttention.SCRIPT)
        harness = harness[: harness.index("\ncase([700, 2000], 1, 64, 0)\n") + 1]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", harness + textwrap.dedent(self.TAIL)],
            capture_output=True,
            text=True,
            env=env,
            timeout=2400,
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])


class TestDsaNativePages(CustomTestCase):
    """``KernelConfig.dsa_native_pages``: the DSA verify attention with its
    pages staged native (``[span * 512, 640]`` FP8 buffers, each page DMA
    retiling a pool block in ``DcpCache.start_native_copies``) gives the
    packed pages' outputs bit for bit (``TestSelectedAttention``'s harness
    and reference checks, interpret mode, 4-rank group). The kernel's own
    page staging runs on both layouts -- the stale-token and pad-lane
    zeroing, the cached verify rows and the slot zeroing store through
    32-bit word views -- on an emulation of ref views
    (``K31_INTERPRET_VIEWS=1``: a bitcast packs consecutive rows into each
    word, as ``pltpu.bitcast`` does), and the pool's stale bytes (lanes
    576:640, tokens no request wrote) are NaN, so a zeroing that either
    layout skips is not finite. Cases: B1 x7 with compact and dense steps
    (early queries, the staged prologue and the other exact knobs on; and
    all off), the chunked B2 x7 with a small window, poisoned page slots
    (compact steps, and two-row sequences on the ungrouped fold, whose
    skipped steps read their zeroed slots), two 4-row sequences, B4 x3,
    two-row sequences and plain rows. The knob stages native pages for the
    8- and 16-row verifies of 8-row sequences only."""

    TAIL = """
        if not INTERNALS:
            print("SKIP: interpret-mode DMA internals moved")
            raise SystemExit(0)
        assert VIEWS
        from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel

        on = dataclasses.replace(base_config, dsa_native_pages=True)
        pages_at = len(mla_dcp.SCOPED_NAMES) - mla_dcp.SCOPED_NAMES.index("dcp latent pages")
        for config, shapes in ((base_config, ()), (on, ((8, 8), (16, 8)))):
            mla_dcp._config = lambda config=config: config
            for b, s in ((8, 8), (16, 8), (16, 4), (8, 4), (32, 8), (8, 1)):
                assert mla_dcp.native_pages(b, H, s, FP8, dsa=True) == ((b, s) in shapes), (b, s)
                assert not mla_dcp.native_pages(b, H, s, FP8), (b, s)  # K3's verifies keep packed pages
                for indexer in (False, True):  # the decode's MLA scratch (DSA: the K3.1 decode)
                    pages = decode_megakernel._mla_scoped_scratch(
                        None, batch_size=b, sequence_rows=s, dcp_dtype=FP8, dcp_page_tokens=T, dsa=indexer
                    )[-pages_at]
                    assert (len(pages.shape) == 3) == (indexer and (b, s) in shapes), (b, s, indexer, pages.shape)
            assert not mla_dcp.native_pages(8, H, 8, jnp.bfloat16, dsa=True)
        KNOBS = ("dsa_row_block_masks", "dsa_prologue_in_select", "dcp_shift_scalars",
                 "dcp_query_tail_unpack")
        for args, kwargs in (
            (([2600], 8, 40, 7), dict(grouped=True, compact=128, window=512, knobs=KNOBS, sent=True)),
            (([2600], 8, 40, 7), dict(grouped=True, compact=128, window=512)),
            (([3069], 8, 400, 7), dict(grouped=True, compact=128, window=512, wide=3, knobs=KNOBS, sent=True)),
            (([900, 2600], 8, 40, 6), dict(grouped=True, compact=64, window=64, knobs=KNOBS)),
            (([700, 9000], 4, 8, 4), dict(grouped=True, compact=64, poison=True, window=256)),
            (([1500, 3000], 4, 200, 1), dict(grouped=True, compact=128, window=512, expect=("compact", "dense"),
                                             knobs=KNOBS)),
            (([700, 1500, 2600, 3000], 4, 24, 8), dict(grouped=True, compact=64, window=128, knobs=KNOBS)),
            (([2047, 600, 64, 3000], 2, 128, 2), dict()),
            (([700, 9000], 2, 8, 4), dict(poison=True)),  # skipped steps read their zeroed slots
            (([700, 2000], 1, 64, 0), dict()),
        ):
            packed = case(*args, **kwargs, stale=True)
            native = case(*args, **kwargs, stale=True, native=True)
            assert np.isfinite(packed).all(), args
            assert np.array_equal(packed.view(np.uint32), native.view(np.uint32)), (args, np.abs(packed - native).max())
        print("OK")
    """

    def test_native_pages_are_bit_identical(self):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        harness = textwrap.dedent(TestSelectedAttention.SCRIPT)
        harness = harness[: harness.index("\ncase([700, 2000], 1, 64, 0)\n") + 1]
        env = dict(
            os.environ, PYTHONPATH=os.pathsep.join(sys.path), K31_INTERPRET_VIEWS="1"
        )
        result = subprocess.run(
            [sys.executable, "-c", harness + textwrap.dedent(self.TAIL)],
            capture_output=True,
            text=True,
            env=env,
            timeout=2400,
        )
        if "SKIP:" in result.stdout:
            self.skipTest(result.stdout.strip().splitlines()[-1])
        self.assertIn("OK", result.stdout, result.stderr[-3000:])


class TestDecodeIndexerExchangeFill(CustomTestCase):
    """``KernelConfig.dsa_exchange_fill`` inside the decode indexer: the
    dense regime's step activity marked in the first count exchange's wait
    (with the own keys, p_tie and the resident keys' copy wait), and the
    scan regime after the rounds from the keys that wait completed, give bit
    for bit the indexer's outputs without it (records, keys, thresholds, own
    flags, step activity; each run also checks ``TestDecodeIndexer``'s
    reference)."""

    SCORING = (
        "dsa_stage_slots=4,dsa_page_slots=8,dsa_resident_keys=1,"
        "dsa_patch_loop=1,dsa_early_plan=1"
    )

    def test_same_outputs_with_the_fills(self):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        script = TestDecodeIndexer.SCRIPT.replace(
            'print("OK" if ok else "FAIL")',
            'np.savez(os.environ["OUT"], tau=tau, ptie=ptie, own=own, act=act, rec=rec, keys=keys)\n'
            'print("OK" if ok else "FAIL")',
        )
        fills = "dsa_exchange_fill=1,dsa_vector_patch=1,dsa_skip_empty_subpages=1"
        bins = {"S": "8", "TOPK": "400", "POS": "3069", "ROUNDS": "5"}
        exact = {"S": "4", "TOPK": "200", "POS": "1500,3000", "ROUNDS": "8"}
        cases = (
            {**bins, "ACTIVITY_MIN_STEPS": "16"},  # dense: marked in the wait
            {**bins, "ACTIVITY_MIN_STEPS": "0"},  # scanned after the rounds
            {**exact, "ACTIVITY_MIN_STEPS": "16"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            for i, case in enumerate(cases):
                runs = []
                for knobs in (self.SCORING, f"{self.SCORING},{fills}"):
                    out = f"{tmp}/case{i}_{len(runs)}.npz"
                    env = dict(
                        os.environ,
                        PYTHONPATH=os.pathsep.join(sys.path),
                        KNOBS=knobs,
                        OUT=out,
                        **case,
                    )
                    result = subprocess.run(
                        [sys.executable, "-c", script],
                        capture_output=True,
                        text=True,
                        env=env,
                        timeout=900,
                    )
                    self.assertIn(
                        "OK",
                        result.stdout,
                        f"{case} {knobs}: {result.stdout[-2000:]}{result.stderr[-3000:]}",
                    )
                    runs.append(np.load(out))
                base, other = runs
                self.assertEqual(sorted(base.files), sorted(other.files))
                for name in base.files:
                    np.testing.assert_array_equal(
                        base[name], other[name], err_msg=f"{case}: {name}"
                    )


class TestSelectHooksSeededFills(CustomTestCase):
    """The selection's hooks under the seeded window and the first wait's
    fill together: ``select_call`` in interpret mode on a 4-rank group over
    ``TestIndexSelectSeededWindow``'s streams (8 rows of one sequence for 5
    steps, two 8-row sequences for 4; window hits and misses), with
    ``select`` given the decode's three hooks (``in_exchange``: the
    attention's prologue under ``dsa_prologue_in_select``; ``keys_ready``:
    its first key copies; ``fill``: the dense step activity under
    ``dsa_exchange_fill``), each a scoped buffer written. The exchanges'
    sends and the hooks count themselves in an SMEM counter (interpret mode
    runs loads and stores in program order; its prints may move) and print
    their place. Every rank runs each hook once a selection, on a hit (two
    exchanges), a miss (three) and the unseeded rounds alike:
    ``in_exchange`` after the second exchange's sends (before its wait);
    with the fill, ``fill`` and then ``keys_ready`` after the first
    exchange's sends, else ``keys_ready`` after the last exchange. The
    thresholds, tie positions, own flags and keys equal the run with every
    knob off bit for bit (the seeded window, the hooks and the fill in four
    combinations, with the vector patch and the odd count stride)."""

    PATCH = """
        HOOKS = os.environ.get("HOOKS") == "1"
        _inside, _counter = [], []

        def _event(self, name, exchange=False):
            # [exchanges so far, events so far] in SMEM, printed with the event.
            counter = _counter[-1]
            if exchange:
                counter[0] = counter[0] + 1
            counter[1] = counter[1] + 1
            pl.debug_print("EVENT {} " + name + " {} {}", self.member, counter[1], counter[0])

        def _exchanging(real):
            def exchange(self, *args, **kwargs):
                _inside.append(1)
                try:
                    return real(self, *args, **kwargs)
                finally:
                    _inside.pop()
            return exchange

        dsa_dcp.IndexSelect._exchange_wide = _exchanging(dsa_dcp.IndexSelect._exchange_wide)
        dsa_dcp.IndexSelect._all_reduce = _exchanging(dsa_dcp.IndexSelect._all_reduce)
        _fill_wait = dsa_dcp.IndexSelect._fill_wait

        def _marked_fill_wait(self):
            if _inside:  # an exchange's sends are out (its queued work and hook, then its wait)
                _event(self, "XCHG", exchange=True)
            return _fill_wait(self)

        dsa_dcp.IndexSelect._fill_wait = _marked_fill_wait
        _select = dsa_dcp.IndexSelect.select

        def _hooked(self):
            def work(name):
                def run():
                    _event(self, name)
                    def body(buf):
                        buf[...] = jnp.full(buf.shape, 1.0, jnp.float32)
                    pl.run_scoped(body, pltpu.VMEM((8, 128), jnp.float32))
                return run

            def selection(counter):
                counter[0] = jnp.int32(0)
                counter[1] = jnp.int32(0)
                _counter.append(counter)
                if HOOKS:
                    fill = work("FILL") if kc.active().dsa_exchange_fill else None
                    _select(self, work("IN_EXCHANGE"), work("KEYS_READY"), fill)
                else:
                    _select(self)
                _counter.pop()

            pl.run_scoped(selection, pltpu.SMEM((2,), jnp.int32))

        dsa_dcp.IndexSelect.select = _hooked
    """

    @staticmethod
    def runs():
        """``{name: (knobs, hooks)}``: every knob off, then the seeded window,
        the hooks and the fill in combination (the other knobs on)."""
        base = TestIndexSelectSeededWindow.KNOBS
        on = base + ",dsa_vector_patch=1,dsa_odd_stride=1"
        seeded, fill = ",dsa_seeded_window=1", ",dsa_exchange_fill=1"
        return {
            "off": (base, False),
            "all": (on + seeded + fill, True),
            "seeded_hooks": (on + seeded, True),
            "seeded_fill": (on + seeded + fill, False),
            "hooks_fill": (on + fill, True),
        }

    @classmethod
    def script(cls):
        import textwrap

        script = textwrap.dedent(TestIndexSelectSeededWindow.SCRIPT)
        anchor = "dsa_dcp.IndexSelect._exchange_wide = _counted\n"
        streams = (
            'stream("b8s8", 8, plan8, 0, 2900)\n',
            'stream("b8s1", 1, plan1, 1, 3200)\n',
        )
        assert script.count(anchor) == 1 and all(script.count(s) == 1 for s in streams)
        script = script.replace(anchor, anchor + textwrap.dedent(cls.PATCH))
        script = script.replace(streams[0], 'stream("b8s8", 8, plan8[:5], 0, 2900)\n')
        return script.replace(streams[1], "")

    @staticmethod
    def events(stdout):
        """``[(stream, step, {rank: [event, ...]})]`` from a run's output, each
        rank's events in program order (their counts, not the print order)."""
        import re

        steps, current = [], {}
        for line in stdout.splitlines():
            event = re.search(r"EVENT (\d+) (\w+) (\d+) (\d+)", line)
            if event:
                rank, name, seq, sent = event.groups()
                current.setdefault(int(rank), []).append((int(seq), name, int(sent)))
                continue
            done = re.match(r"(\w+) (\d+) taus", line)
            if done:
                ranks = {}
                for rank, got in current.items():
                    got.sort()
                    seqs = [seq for seq, _, _ in got]
                    assert seqs == list(range(1, len(got) + 1)), got
                    # Each event's exchange count matches its place.
                    count = 0
                    for _, name, sent in got:
                        count += name == "XCHG"
                        assert sent == count, got
                    ranks[rank] = [name for _, name, _ in got]
                steps.append((done[1], int(done[2]), ranks))
                current = {}
        return steps

    def _check_events(self, name, ranks, exchanges, hooks, filled):
        """One selection's events on every rank (``exchanges``: its count)."""
        for rank, got in ranks.items():
            where = f"{name} rank {rank}: {got}"
            self.assertEqual(got.count("XCHG"), exchanges, where)
            hook_events = [e for e in got if e != "XCHG"]
            if not hooks:
                self.assertEqual(hook_events, [], where)
                continue
            if filled:
                order = ["FILL", "KEYS_READY", "IN_EXCHANGE"]
            else:
                order = ["IN_EXCHANGE", "KEYS_READY"]
            self.assertEqual(hook_events, order, where)
            # The exchanges whose sends precede each hook.
            sent = {e: got[: got.index(e)].count("XCHG") for e in hook_events}
            self.assertEqual(sent["IN_EXCHANGE"], 2, where)
            if filled:
                self.assertEqual((sent["FILL"], sent["KEYS_READY"]), (1, 1), where)
            else:
                self.assertEqual(sent["KEYS_READY"], exchanges, where)
                self.assertEqual(got[-1], "KEYS_READY", where)

    def test_hooks_run_once_on_every_path_bit_for_bit(self):
        import ast
        import os
        import re
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        script = self.script()
        with tempfile.TemporaryDirectory() as tmp:
            outputs, exchanges = {}, {}
            for name, (knobs, hooks) in self.runs().items():
                env = dict(
                    os.environ,
                    PYTHONPATH=os.pathsep.join(sys.path),
                    KNOBS=knobs,
                    HOOKS="1" if hooks else "0",
                    OUT=f"{tmp}/{name}.npz",
                )
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=2400,
                )
                self.assertIn(
                    "OK",
                    result.stdout,
                    f"{name}: {result.stdout[-2000:]}{result.stderr[-3000:]}",
                )
                outputs[name] = np.load(f"{tmp}/{name}.npz")
                seeded = "seeded_window=1" in knobs
                filled = "exchange_fill=1" in knobs
                # Seeded: two exchanges on a hit (or with no active row), else three.
                flags = {}
                if seeded:
                    line = re.search(r"^flags (.*)$", result.stdout, re.M)[1]
                    for stream, step, active, hit in ast.literal_eval(line):
                        flags[stream, step] = 2 if hit or not active else 3
                steps = self.events(result.stdout)
                self.assertEqual(len(steps), 9, name)
                exchanges[name] = []
                for stream, step, ranks in steps:
                    where = f"{name} {stream} step {step}"
                    self.assertEqual(sorted(ranks), [0, 1, 2, 3], where)
                    count = flags.get((stream, step), 3)
                    exchanges[name].append(count)
                    self._check_events(where, ranks, count, hooks, filled)
            for name in ("all", "seeded_hooks", "seeded_fill"):  # hits and misses
                self.assertIn(2, exchanges[name], name)
                self.assertIn(3, exchanges[name], name)
            base = outputs["off"]
            for name, other in outputs.items():
                self.assertEqual(sorted(base.files), sorted(other.files), name)
                for key in base.files:
                    np.testing.assert_array_equal(
                        base[key], other[key], err_msg=f"{name}: {key}"
                    )


class TestMlaQ6Knobs(CustomTestCase):
    """The decode MLA layer with the DSA indexer (``decode_megakernel.mla``)
    in Pallas interpret mode on a 4-rank DCP group, with the serving
    selection, attention and prologue knobs, under its q6 knobs:
    ``KernelConfig.decode_mla_early_handout`` (the new rows' hand-out copies
    start once q6 has written them and are waited after the layer, as q8
    does) and ``dsa_gather_overlap`` (the index query gather leaves before
    kv-down, q-up, the new rows' hand-out and the attention's query packet,
    which run under it; the packet's remote copies leave once it has landed),
    also inside the one-trip prefix region (``decode_mla_prefix_region``).
    The layer's output, handed-out rows, index keys and records and the
    selection equal the knobs-off layer's bit for bit, with eager and with
    on-wait DMA execution (a copy started before its source is written, or a
    source written after its copy started, reads different bytes in the two
    modes). Interpret mode sums every product in program order, so this
    checks the moved operations and the buffers they share, not the compiled
    program's MXU accumulation order. The output projection's TP reduction is
    a local product in every run.

    ``after_query_up`` (``KernelConfig.decode_latent_down_prefetch``, whose
    latent-down copy overwrites the q-up weight's pooled storage there): a
    hook that copies NaNs over the q-up weight runs exactly once and leaves
    the layer bit-identical, so nothing reads that weight after it."""

    SCRIPT = r"""
import dataclasses, os
G = 4
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={G}"
os.environ["JAX_PLATFORMS"] = "cpu"
from types import SimpleNamespace
import ml_dtypes, numpy as np, jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, PartitionSpec as P
from sglang_tpu.k3_programs.kimi_k3 import config as kc
kc.install(kc.KernelConfig(dsa_select_rounds=5, dsa_compact_tokens=128, dsa_compact_window=512,
                           dcp_pair_combine=True, dsa_stage_slots=4, dsa_page_slots=8, dsa_resident_keys=True,
                           dsa_patch_loop=True, dsa_early_plan=True, dsa_live_activity=True, dsa_odd_stride=True,
                           dsa_prologue_in_select=True, dsa_row_block_masks=True, dcp_shift_scalars=True,
                           dcp_query_tail_unpack=True, dsa_exchange_fill=True, dsa_vector_patch=True))
from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk, dsa_dcp, dsa_decode, mla_dcp
mla_dcp.GROUPS = (G, 8, 16, 32)
mla_dcp._grouped_spec = lambda b, g, h, rows: rows in (4, 8) and b % 4 == 0
dsa_dcp.ACTIVITY_MIN_STEPS = int(os.environ["ACTIVITY"])

def tokens(x):
    return (jax.lax.broadcasted_iota(jnp.int32, x.shape, 0) * x.shape[1]
            + jax.lax.broadcasted_iota(jnp.int32, x.shape, 1))

def _pages(self, slot):  # interpret mode has no ref bitcasts (see TestSelectedAttention)
    def load(lanes, tokens=None):
        x = self.ws.pages[slot]
        return x.reshape(x.shape[0] * x.shape[1], x.shape[2])[:, lanes]
    return load

def _clean_step(self, slot, row, local_step, tokens_=None, end=None, pad=True):
    x = self.ws.pages[slot]
    cut = 0
    for k in range(self.span):
        first = ((local_step * self.span + k) * self.group + self.owner) * self.page_tokens
        cut = cut + jnp.clip(self._limit(row) - first, 0, self.page_tokens)
    lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 2)
    keep = (tokens(x) < cut) & ((lane < 576) if pad else True)
    self.ws.pages[slot] = jnp.where(keep, x, jnp.zeros((), x.dtype))

def _cache_verify_row(self, slot, local_step, row):
    position = self.position_ref[row]
    page = position // self.page_tokens
    local_page = page // self.group - local_step * self.span

    @pl.when((page % self.group == self.owner) & (local_page >= 0) & (local_page < self.span))
    def write():
        if mla_dcp.chunked(self.batch):
            chunk = mla_dcp.chunk_layout(self.batch, self.heads)[0]
            latent = self.ws.new_rows[row // chunk, self.slots * (row % chunk)]
        else:
            latent = self.ws.new_rows[self.slots * row]
        x = self.ws.pages[slot]
        offset = local_page * self.page_tokens + position % self.page_tokens
        self.ws.pages[slot] = jnp.where(tokens(x) == offset, latent.astype(x.dtype)[None, None, :], x)

def _zero_step_pad_lanes(self, slot):
    x = self.ws.pages[slot]
    lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 2)
    self.ws.pages[slot] = jnp.where(lane < 576, x, jnp.zeros((), x.dtype))

def _zero_slot(self, slot):
    self.ws.pages[slot] = jnp.zeros(self.ws.pages.shape[1:], self.ws.pages.dtype)

mla_dcp.DcpAttention._pages = _pages
mla_dcp.DcpAttention._clean_step = _clean_step
mla_dcp.DcpAttention._zero_slot = _zero_slot
mla_dcp.DcpAttention._zero_step_pad_lanes = _zero_step_pad_lanes
mla_dcp.DcpAttention._cache_verify_row = _cache_verify_row

def _project_local(attended, weight, out, *args, **kwargs):
    out[...] = jnp.dot(attended, weight[...], preferred_element_type=jnp.float32)

dmk._project_attention_2d = _project_local
BF, FP8 = ml_dtypes.bfloat16, jnp.float8_e4m3fn
S, TOPK = int(os.environ["S"]), int(os.environ["TOPK"])
firsts = [int(x) for x in os.environ["POS"].split(",")]
T, H, QL, PER, PAD = 512, 1024, 1536, 32 // G, 8
positions = np.array([p0 + r for p0 in firsts for r in range(S)], np.int32)
B = len(positions)
rng = np.random.default_rng(0)
bf = lambda x: np.asarray(x, np.float32).astype(BF)
plan = mla_dcp.page_plan(B, 3, S, FP8, dsa=True)
key_buf = mla_dcp.selection_scratch(B, page_tokens=T, sequence_rows=S, cache_dtype=FP8, **plan)[0]
span = key_buf.shape[2] * 128 // T
pages = -(-(max(-(-(int(p) + T) // T) // G for p in positions) + 1) // span) * span
key_rows = dsa_dcp.key_rows_for(pages, T, span)

hidden = bf(rng.standard_normal((B, H)))
mla_w = [bf(rng.standard_normal((G, i, o)) * i**-0.5)
         for i, o in ((H, QL), (H, 640), (QL, 768), (512, 768), (H, 384), (384, H))]
qnorm, knorm_mla = bf(1 + 0.1 * rng.standard_normal((1, QL))), bf(1 + 0.1 * rng.standard_normal((1, 512)))
wqb = bf(rng.standard_normal((QL, 32 * 128)) * QL**-0.5)
wk = bf(rng.standard_normal((H, 128)) * H**-0.5)
wproj = np.zeros((H, 128), np.float32)
wproj[:, :32] = rng.standard_normal((H, 32)) * H**-0.5
knorm = bf(np.stack([1 + 0.1 * rng.standard_normal(128), 0.1 * rng.standard_normal(128)]))
wqb_ranks = np.stack([wqb[:, g * PER * 128:(g + 1) * PER * 128] for g in range(G)])
wks_ranks = np.stack([dsa_decode.key_weight(wk, g, G) for g in range(G)])
wpt = dsa_decode.gate_weight(wproj)
ipool = rng.integers(0, 256, (G, 1 + len(firsts) * pages, 132, 4, 128), dtype=np.uint8)
lpool = (rng.standard_normal((G, 1 + len(firsts) * pages, 128, 4, 640)) * 0.5).astype(ml_dtypes.float8_e4m3fn)
tables = np.zeros((G, B, pages), np.int32)
for s_, p0 in enumerate(firsts):
    rec = np.asarray(dsa_dcp.key_records(jnp.asarray(rng.standard_normal((p0, 128)).astype(np.float32))))
    for r in range(S):
        tables[:, s_ * S + r] = 1 + s_ * pages + np.arange(pages)
    for t in range(p0):
        page = t // T; g = page % G
        blk = tables[g, s_ * S, page // G]
        (row, sub), (srow, lane) = dsa_dcp.record_place(t % T)
        ipool[g, blk, row, sub] = rec[t, :128]; ipool[g, blk, srow, 0, lane] = rec[t, 128]

n_dsa_p = len(dsa_decode.PERSISTENT_NAMES)
dsa_scoped = dsa_decode.scoped_scratch(B, hidden=H, index_block_shape=(132, 4, 128), key_rows=key_rows,
                                       page_tokens=T, sequence_rows=S, page_plan=plan)
vmem, smem = pltpu.VMEM, pltpu.SMEM
groups = [
    (*dsa_decode.persistent_scratch(B, G), *mla_dcp.persistent_scratch(B, G, cache_dtype=FP8, sequence_rows=S)),
    tuple(dsa_scoped.values()),
    mla_dcp.scoped_scratch(B, group=G, cache_dtype=FP8, page_tokens=T, sequence_rows=S, **plan),
    (vmem((8, 128), jnp.bfloat16), vmem((8, 128), jnp.bfloat16), pltpu.SemaphoreType.DMA((1,)),
     vmem((B, 3, 256), jnp.bfloat16), vmem((B, 384), jnp.bfloat16), vmem((B, 3, 256), jnp.bfloat16),
     vmem((B, 3, 128), jnp.bfloat16), vmem((B, 3, 1), jnp.float32), vmem((B, 3, 1), jnp.float32),
     vmem((B, 3, 128), jnp.float32)),
    (*(vmem(w.shape[1:], jnp.bfloat16) for w in mla_w), pltpu.SemaphoreType.DMA((6,))),
    (smem((1,), jnp.int32), vmem((B, PAD, 256), jnp.bfloat16), vmem((B, PAD, 128), jnp.bfloat16),
     pltpu.SemaphoreType.DMA((2,))),
    (vmem((QL, 768), jnp.bfloat16), pltpu.SemaphoreType.DMA(())),
]
bounds = np.cumsum([0] + [len(x) for x in groups])
STEPS = key_rows // dsa_scoped["attend_keys"].shape[2]
installed = kc.active()

def run(knobs, params):
    clobber = knobs.pop("qup_clobber", None) is not None
    config = installed.replace(**knobs)
    handout = config.decode_mla_early_handout

    def kernel(pos_ref, table_ref, hidden_ref, qnorm_ref, knorm_ref, qa, ka, qb, kb, gate, o, wqb_ref, wks_ref,
               wpt_ref, dknorm_ref, ipool_ref, lpool_ref, out_ref, tau_out, ptie_out, own_out, act_out, key_out,
               value_out, rec_out, keys_out, clobbered_out, *scratch):
        persist, dscoped, dcps, mws, wv, (phase, new_key, new_value, sems), (nans, nan_sem) = (
            scratch[bounds[i]:bounds[i + 1]] for i in range(len(groups)))
        phase[0] = 0
        sc = dict(zip(dsa_scoped, dscoped))
        indexer = dsa_decode.DecodeIndexer(
            weights={"d_wqb": wqb_ref, "d_wks": wks_ref, "d_wpt": wpt_ref, "d_knorm": dknorm_ref},
            pools=(ipool_ref,), layer=0, keys=keys_out, records_out=rec_out, persistent=persist[:n_dsa_p],
            scoped=sc, topk=TOPK, sequence_rows=S)
        weights = dmk.MultiHeadLatentAttentionWeights(qa, wv[0], ka, wv[1], qb, wv[2], kb, wv[3], gate, wv[4],
                                                      o, wv[5], qnorm_ref, knorm_ref, wv[6])
        comm = SimpleNamespace(**dict.fromkeys(f.name for f in dataclasses.fields(dmk.AttentionCommunicationWorkspace)))
        comm.reduction_phase = phase
        copies = lambda: (pltpu.make_async_copy(new_key, key_out, sems.at[0]),
                          pltpu.make_async_copy(new_value, value_out, sems.at[1]))

        def start_hand_out():
            for copy in copies():
                copy.start()

        calls = []
        nan_copy = pltpu.make_async_copy(nans, wv[2], nan_sem)

        def clobber_query_up():
            calls.append(1)
            nans[...] = jnp.full(nans.shape, jnp.nan, nans.dtype)
            nan_copy.start()

        dmk.mla(pos_ref, hidden_ref, out_ref, new_key, new_value, weights=weights,
                workspace=dmk.MultiHeadLatentAttentionWorkspace(*mws), communication_workspace=comm,
                dcp=(mla_dcp.DcpCache((lpool_ref,), table_ref, pages, 0, group=G, global_pages=False),
                     mla_dcp.DcpWorkspace(*persist[n_dsa_p:], *dcps)),
                synchronize_devices=False, output_projection_after_query_down=True, sequence_rows=S, dsa=indexer,
                hand_out=start_hand_out if handout else None,
                after_query_up=clobber_query_up if clobber else None)
        clobbered_out[0] = 0
        if clobber:
            assert len(calls) == 1, calls  # traced once, on every q6 path
            nan_copy.wait()
            clobbered_out[0] = jnp.isnan(wv[2][...].astype(jnp.float32)).all().astype(jnp.int32)
        if not handout:  # q8 as compute() runs it
            start_hand_out()
        for copy in copies():
            copy.wait()
        for b in range(B):
            tau_out[b] = sc["tau"][b]; ptie_out[b] = sc["p_tie"][b]; own_out[b] = sc["own"][b]
        for i in range(B * STEPS):
            act_out[i] = sc["active"][i]

    def body(pos, table, hid, qn, kn, *rest):
        w6, (wqb_, wks_, wpt_, dkn, ip, lp) = rest[:6], rest[6:]
        out = pl.pallas_call(kernel,
            out_shape=(jax.ShapeDtypeStruct((B, H), jnp.float32), *(jax.ShapeDtypeStruct((B,), jnp.int32),) * 3,
                       jax.ShapeDtypeStruct((B * STEPS,), jnp.int32), jax.ShapeDtypeStruct((B, PAD, 256), jnp.bfloat16),
                       jax.ShapeDtypeStruct((B, PAD, 128), jnp.bfloat16),
                       jax.ShapeDtypeStruct((B, 24, 2, 128), jnp.int32),
                       jax.ShapeDtypeStruct((B, key_rows, 128), jnp.int32),
                       jax.ShapeDtypeStruct((1,), jnp.int32)),
            grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=2,
                in_specs=[pl.BlockSpec(memory_space=vmem)] * 3 + [pl.BlockSpec(memory_space=pl.ANY)] * 12,
                out_specs=[pl.BlockSpec(memory_space=vmem)] + [pl.BlockSpec(memory_space=smem)] * 4
                          + [pl.BlockSpec(memory_space=pl.ANY)] * 4 + [pl.BlockSpec(memory_space=smem)],
                scratch_shapes=[x for grp in groups for x in grp]),
            compiler_params=pltpu.CompilerParams(has_side_effects=True),
            interpret=params)(pos, table[0], hid, qn, kn, *(w[0] for w in w6), wqb_[0], wks_[0], wpt_, dkn,
                              ip[0], lp[0])
        return tuple(x[None] for x in out)

    sharded = (P("tp"),) + (P(),) * 3 + (P("tp"),) * 8 + (P(), P(), P("tp"), P("tp"))
    f = jax.jit(jax.shard_map(body, mesh=Mesh(np.asarray(jax.devices()), ("tp",)), in_specs=(P(), *sharded),
                              out_specs=(P("tp"),) * 10, check_vma=False))
    dmk._config = lambda: config  # the knobs mla() reads while tracing
    try:
        return [np.asarray(x) for x in f(positions, tables.reshape(G, -1), hidden, qnorm, knorm_mla, *mla_w,
                                         wqb_ranks, wks_ranks, wpt, knorm, ipool, lpool)]
    finally:
        dmk._config = kc.active

bits = lambda x: x.view(np.uint16 if x.dtype.itemsize == 2 else np.uint32)
names = ("output", "tau", "p_tie", "own", "active", "key rows", "value rows", "records", "keys")
ok, want = True, None
for variant in os.environ["VARIANTS"].split(";"):
    knobs = dict(kv.split("=") for kv in variant.split(",") if kv)
    clobber = "qup_clobber" in knobs
    got = run(knobs, pltpu.InterpretParams(dma_execution_mode=os.environ["MODE"]))
    # The hook's copy landed: every rank's q-up weight holds NaNs after the layer.
    ok &= bool((got.pop()[:, 0] == int(clobber)).all())
    if want is None:
        want = got
        out, rows = got[0], got[5][:, :, :3].astype(np.float32)
        print(f"rms output {np.sqrt(np.mean(out ** 2)):.3f} rows {np.sqrt(np.mean(rows ** 2)):.3f}; "
              f"tau {got[1][0].tolist()}; active {int(got[4].sum())} of {got[4].size}")
        # Every row has more than TOPK tokens: a real threshold, a live layer.
        ok &= bool(np.isfinite(out).all() and np.isfinite(rows).all() and np.abs(out).max() > 0)
        ok &= bool((got[1][0] != -2**31).all())
        continue
    same = [np.array_equal(bits(a), bits(b)) for a, b in zip(want, got)]
    print(f"{variant}: bitwise {all(same)}", [n for n, s in zip(names, same) if not s])
    ok &= all(same)
print("OK" if ok else "FAIL")
"""

    def test_layer_is_bitwise_unchanged(self):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        handout = "decode_mla_early_handout=1"
        prefix = "decode_mla_prefix_region=1"
        overlap = "dsa_gather_overlap=1"
        for case in (
            # The B1 x7 verify shape, dense regime (every step active).
            {"S": "8", "POS": "3000", "TOPK": "200", "ACTIVITY": "16", "MODE": "eager",
             "VARIANTS": f";{handout};{prefix},{handout};{overlap};{overlap},{prefix},{handout};"
                         f"{overlap},{handout}"},
            # Selection-driven step activity, copies executed at their waits.
            {"S": "8", "POS": "3000", "TOPK": "200", "ACTIVITY": "0", "MODE": "on_wait",
             "VARIANTS": f";{prefix},{handout};{overlap},{prefix},{handout}"},
            # Two 8-row sequences (B2 x7): chunked attention; the gather
            # overlap stays off (8-row verify calls only).
            {"S": "8", "POS": "1500,3000", "TOPK": "200", "ACTIVITY": "16", "MODE": "eager",
             "VARIANTS": f";{overlap},{prefix},{handout}"},
        ):  # fmt: skip
            with self.subTest(**case):
                env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), **case)
                result = subprocess.run(
                    [sys.executable, "-c", self.SCRIPT],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=1800,
                )
                self.assertIn(
                    "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
                )

    def test_query_up_weight_is_dead_after_its_hook(self):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        clobber = "qup_clobber=1"
        serving = (
            "dsa_gather_overlap=1,decode_mla_prefix_region=1,decode_mla_early_handout=1"
        )
        for case in (
            # B1 x7 with eager copies: the NaNs land as the hook runs.
            {"S": "8", "POS": "3000", "TOPK": "200", "ACTIVITY": "16", "MODE": "eager",
             "VARIANTS": f";{clobber};{serving},{clobber}"},
            # Copies executed at their waits (after the layer).
            {"S": "8", "POS": "3000", "TOPK": "200", "ACTIVITY": "0", "MODE": "on_wait",
             "VARIANTS": f";{serving},{clobber}"},
            # B2 x7 (chunked attention; no gather overlap at 16 rows).
            {"S": "8", "POS": "1500,3000", "TOPK": "200", "ACTIVITY": "16", "MODE": "eager",
             "VARIANTS": f";{serving},{clobber}"},
        ):  # fmt: skip
            with self.subTest(**case):
                env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), **case)
                result = subprocess.run(
                    [sys.executable, "-c", self.SCRIPT],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=1800,
                )
                self.assertIn(
                    "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
                )


class TestK31DraftPrologueOverlap(CustomTestCase):
    """``KernelConfig.draft_prologue_overlap`` and ``draft_markov_rows``: the
    fused MLA draft kernel in Kimi-K3.1's shape (three heads a rank, causal
    block rows; shrunk to hidden 1024, 512 vocabulary rows a rank, two layers
    and Markov rank and MLP lanes 256, so that w2t fits the down buffer as
    Kimi-K3.1's 512 do) on 32 devices in Pallas interpret mode, with
    draft_whole_weights and draft_lm_prefetch, returns the knob-off kernel's
    output vector and context latents on every rank, bit for bit:

    - the single-request op (B1: the overlap with w2t in the down buffer, the
      w1 rows in the o buffer) with forced decisions (the serving path), and
      with NaNs in the anchor's and token 0's w1 rows (no Markov slot has a
      winner: each drafts token 0, and the next slot reads its row from HBM
      as knob off does), its knob-on runs under the race detector with every
      DMA executed at its start;
    - two requests through the scalar operands, both forced with anchors one
      rank owns, or the second's decisions unforced (the acceptance first);
      16 rows: o streams, so the Markov rows stay off.

    Three requests at 8 rows stream down_proj and keep the sequential
    prologue (VMEM). The cached pages hold O(1) latents (FP8 of N(0, 1/4)),
    so the attention output, the MLP's and the new keys all reach the drafts
    (a layer keeping layer 0's block keys, or a zeroed MLP output, changes
    them)."""

    SCRIPT = """
        import os, sys
        G = 32
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={G}"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import ml_dtypes, numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3 import config as kconfig
        from sglang_tpu.k3_programs.kimi_k3.kernels import dspark_mla as mla, dspark_mla_kernel as kern, mla_dcp
        from sglang_tpu.k3_programs.kimi_k3.kernels.dspark import OUT_DRAFTS

        L, H, V, MLP, MARKOV, ROWS, PAGES, BLOCKS, LANES = 2, 1024, 512, 256, 256, 8, 2, 3, mla_dcp.LANES
        kern.CHUNK, kern.FC_IN, kern.ROW_PACKED = H // 8, 5 * H // 4, H // 256
        kern.LM_CHUNK, kern.LM_CHUNKS, kern.MLP_PADDED = 128, V // 128, MLP
        c = mla.MlaDraftConfig(hidden=H, layers=L, heads=96, vocab=G * V, markov_rank=MARKOV, causal=True)
        BASE = kconfig.KernelConfig(draft_whole_weights=True, draft_lm_prefetch=True)
        KNOBS = {"draft_prologue_overlap": True, "draft_markov_rows": True}
        kern._config = lambda: BASE.replace(**KNOBS)
        # Where the knobs apply: the overlap with down_proj whole (up to 16 rows), the w1 rows with o whole (8).
        assert c.heads_per_rank == 3
        assert [kern.prologue_overlap(c, T) for T in (8, 16, 24, 32)] == [True, True, False, False]
        assert [kern.markov_row_windows(c, T) for T in (8, 16)] == [2, 0]
        kern._config = lambda: BASE
        assert not kern.prologue_overlap(c, 8) and kern.markov_row_windows(c, 8) == 0
        stock = pl.pallas_call
        interpret = [None]
        pl.pallas_call = lambda *a, **k: stock(*a, **{"interpret": interpret[0], **k})

        def tokens(x):  # interpret mode has no ref bitcasts (see TestDraftPairCombine)
            return (jax.lax.broadcasted_iota(jnp.int32, x.shape, 0) * x.shape[1]
                    + jax.lax.broadcasted_iota(jnp.int32, x.shape, 1))

        def _pages(self, slot):
            def load(lanes, tokens=None):
                x = self.ws.pages[slot]
                x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
                if hasattr(lanes, "size"):
                    return jax.lax.dynamic_slice_in_dim(x, lanes.start, lanes.size, 1)
                return x[:, lanes]
            return load

        def _clean_step(self, slot, row, local_step, tokens_=None, end=None, pad=True):
            x = self.ws.pages[slot]
            cut = 0
            for k in range(self.span):
                first = ((local_step * self.span + k) * self.group + self.owner) * self.page_tokens
                cut = cut + jnp.clip(self._limit(row) - first, 0, self.page_tokens)
            lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 2)
            keep = (tokens(x) < cut) & ((lane < 576) if pad else True)
            self.ws.pages[slot] = jnp.where(keep, x, jnp.zeros((), x.dtype))

        def _zero_step_pad_lanes(self, slot):
            x = self.ws.pages[slot]
            lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 2)
            self.ws.pages[slot] = jnp.where(lane < 576, x, jnp.zeros((), x.dtype))

        def _zero_slot(self, slot):
            self.ws.pages[slot] = jnp.zeros(self.ws.pages.shape[1:], self.ws.pages.dtype)

        mla_dcp.DcpAttention._pages = _pages
        mla_dcp.DcpAttention._clean_step = _clean_step
        mla_dcp.DcpAttention._zero_slot = _zero_slot
        mla_dcp.DcpAttention._zero_step_pad_lanes = _zero_step_pad_lanes

        rng = np.random.default_rng(0)
        bf = ml_dtypes.bfloat16
        per = c.heads_per_rank
        SHAPES = {"ln1": (L, 1, H), "ln2": (L, 1, H), "q_norm": (L, 1, c.q_lora), "kv_norm": (L, 1, LANES),
                  "context_norm": (1, H), "final_norm": (1, H), "mask_embed": (1, H), "rot": (c.rope, c.rope),
                  "qkv_tile": (L, H, 128), "context_tile": (H, 128), "q_b": (L, c.q_lora, per * 256),
                  "w_uk": (L, per, 128, 512), "w_uv": (L, per, 512, 128), "o": (L, per * 128, H),
                  "gate": (L, H, MLP), "up": (L, H, MLP), "down": (L, MLP, H), "fc": (kern.FC_IN, kern.CHUNK),
                  "w2t": (MARKOV, V), "lm_head": (H, V), "embed": (V, H)}
        weights = {}
        for name in kern.WEIGHT_NAMES:
            if name == "w1":
                w1 = rng.standard_normal((c.vocab, MARKOV)).astype(bf)
                w1[rng.random(w1.shape) < 0.01] = -0.0  # signed zeros through the packets
                weights[name] = w1  # w1 and rope: replicated
            elif name == "rope":
                weights[name] = rng.uniform(-1, 1, (4096, 128)).astype(np.float32)
            else:
                shape = SHAPES[name]
                scale = 1.0 if len(shape) < 3 or name in ("ln1", "ln2", "q_norm", "kv_norm") else shape[-2] ** -0.5
                if name in kern.VMEM_NAMES:  # the norms, mask row and rotation: the same on every rank
                    weights[name] = np.broadcast_to((rng.standard_normal(shape) * scale).astype(bf), (G, *shape))
                else:
                    weights[name] = (rng.standard_normal((G, *shape)) * scale).astype(bf)

        def inputs(R, single, firsts, decisions, no_winner=False):
            r = np.random.default_rng(R)
            pools = [(r.standard_normal((G * BLOCKS, 128, 4, LANES), np.float32) * 0.5).astype(
                ml_dtypes.float8_e4m3fn) for _ in range(L)]
            aux = r.standard_normal((5, R * ROWS, H)).astype(bf)
            logits = r.standard_normal((G, R, ROWS, V)).astype(np.float32)
            target = logits.transpose(1, 2, 0, 3).reshape(R, ROWS, G * V).argmax(-1)
            previous = np.full((R, 128), -1, np.int32)
            previous[:, OUT_DRAFTS : OUT_DRAFTS + 3] = target[:, :3]  # an unforced count of 3
            table = 1 + np.arange(PAGES, dtype=np.int32)
            if single:
                ids = np.zeros(3 * ROWS + 1 + ROWS * PAGES, np.int32)
                ids[ROWS : 2 * ROWS] = firsts[0] + np.arange(ROWS)
                ids[3 * ROWS + 1 : 3 * ROWS + 1 + PAGES] = table
                operands = (ids, np.asarray(decisions, np.int32))
            else:
                d = np.asarray(decisions, np.int32)
                operands = (np.asarray(firsts, np.int32), d[:, 0], d[:, 1], np.tile(table, (R, 1)))
            w1 = weights["w1"]
            if no_winner:  # a NaN in the anchor's and token 0's w1 rows: no Markov slot has a winner
                w1 = w1.copy()
                w1[[decisions[0][1], 0], 7] = np.nan
            return w1, (*pools, aux, logits, previous, *operands)

        def build(knobs, R, single, races=False):
            # The config and interpret mode are read when the kernel traces (the first call).
            config = BASE.replace(**knobs)
            kern._config = lambda: config
            interpret[0] = (pltpu.InterpretParams(detect_races=True, dma_execution_mode="eager") if races
                            else pltpu.InterpretParams())
            ids_size = 3 * ROWS + 1 + ROWS * PAGES
            spec = kern.SingleEpilogue(slots=3, reset=10, state_size=40, ids_size=ids_size) if single else None
            fused = kern.fused_draft_local(c, requests=R, pages=PAGES, rows=ROWS, single=spec)
            n = len(kern.WEIGHT_NAMES)

            def body(*a):
                w = {name: a[i] if name in ("w1", "rope") else a[i][0] for i, name in enumerate(kern.WEIGHT_NAMES)}
                for name in kern.LEADING_NAMES if single else ():
                    w[name] = w[name].reshape(L, -1)
                vector, latents = fused(w, a[n : n + L], *a[n + L :])
                return vector[None], latents[None]

            operands = 2 if single else 4
            specs = (*(P() if name in ("w1", "rope") else P("tp") for name in kern.WEIGHT_NAMES), *(P("tp"),) * L,
                     P(), P("tp"), P(), *(P(),) * operands)
            f = jax.jit(jax.shard_map(body, mesh=Mesh(np.asarray(jax.devices()), ("tp",)), in_specs=specs,
                                      out_specs=(P("tp"), P("tp")), check_vma=False))

            def run(data):
                w1, rest = data
                out = f(*(w1 if name == "w1" else weights[name] for name in kern.WEIGHT_NAMES), *rest)
                return [np.asarray(x).view(np.uint32 if x.dtype == jnp.int32 else np.uint16) for x in out]

            return run

        CASES = {"single": (1, True, [1400], {"forced": [[5, 7 * V + 33]], "no_winner": [[5, 7 * V + 33]]}),
                 "two": (2, False, [1500, 2600], {"forced": [[2, 3 * V + 401], [6, 3 * V + 7]],  # one owner
                                                  "mixed": [[2, 3 * V + 401], [-1, -1]]})}  # (unforced count 3)
        R, single, firsts, decisions = CASES[os.environ["CASE"]]
        data = {name: inputs(R, single, firsts, d, name == "no_winner") for name, d in decisions.items()}
        off_run = build({}, R, single)
        off = {name: off_run(x) for name, x in data.items()}
        for name, out in off.items():
            drafts = out[0].view(np.int32)[0, :, OUT_DRAFTS : OUT_DRAFTS + ROWS - 1]
            assert all(np.array_equal(out[k][0], out[k][g]) for k in range(2) for g in range(G))
            if name == "no_winner":  # knob off drafts token 0 at every slot (and so reads its row)
                assert not drafts.any(), drafts
            else:
                assert all(len(set(row.tolist())) > 1 for row in drafts), drafts
            assert np.isfinite(out[1].view(bf).astype(np.float32)).all()
        on_run = build(KNOBS, R, single, races=single)
        for name, x in data.items():
            on = on_run(x)
            same = [np.array_equal(a, b) for a, b in zip(on, off[name])]
            print(name, "vector, latents identical:", same, flush=True)
            assert all(same), (on[0].view(np.int32)[0, 0, :32], off[name][0].view(np.int32)[0, 0, :32])
            # (the detector's state is the last kernel run's: each run starts a new one)
            assert not (single and interpreter.races.races_found), name
        print("OK")
    """

    def _run(self, case):
        import os
        import subprocess
        import sys
        import textwrap

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        # 32 interpret devices block in callbacks: give the CPU client enough
        # threads (XLA reads NPROC for its pool size).
        env = dict(
            os.environ, PYTHONPATH=os.pathsep.join(sys.path), NPROC="128", CASE=case
        )
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=3600,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )

    def test_single_request_matches_knob_off_without_races(self):
        self._run("single")

    def test_two_requests_match_knob_off(self):
        self._run("two")


class TestDecodeKnobsTogether(CustomTestCase):
    """Decode layer pieces that several knobs change at once, in Pallas
    interpret mode, bit for bit against the same pieces with those knobs off.

    ``moe`` (TestLatentDownPrefetch's 8-row harness: presorted router, expert
    schedule, late shared down, token-parallel latent-up; 32 host devices,
    race detection) for three consecutive layers, the reduction phase carried
    from layer to layer and a barrier between layers (the attention's place).
    ``KernelConfig.decode_latent_down_prefetch`` (each layer's own latent-down
    weight copied before its MoE on the semaphore of the layer's parity, as
    the attention starts it) and ``moe_routed_send_in_expert_region`` (the
    token-parallel tail's stream and routed send in one region) run together,
    with eager DMAs and with DMAs executed at their waits (each run in its own
    process). Between layers the latent-down, shared-expert and latent-up
    storage holds NaNs (region B holds other weights then), and so does every
    scratch buffer at the start.

    ``mla`` (TestMlaQ6Knobs's harness, B1 x7 and B2 x7) with its input rows
    still in flight at entry (``KernelConfig.attention_input_gather_in_branches``:
    a copy into NaN rows, waited once by ``wait_for_input``) and the q-up hook
    of ``decode_latent_down_prefetch`` (NaNs copied over the q-up weight) in
    the same layer, with the serving q6 knobs and ``dcp_pair_lse_rows``.

    The layer chain (``moe``'s harness at B1 x7's 8 rows, layers 1..4 of a
    loop) runs each layer's pieces in the decoder's order: the pair's rows
    from its prefix rows, the crossed pre-attention gather, an attention
    stand-in by layer (the delta branch: the gather's wait, its state reads,
    the latent-down copy; the latent branch: its q-up copy into the
    latent-down storage, the wait, q-up, the latent-down copy), a TP32 sum
    into the pair's prefix rows, the MoE, the residual rows and the aux
    captures after layers 1 and 3. With ``decode_latent_down_prefetch``,
    ``moe_routed_send_in_expert_region``, ``attention_input_gather_in_branches``,
    ``moe_output_round_next_layer`` (the round and the residual rows finished
    in the next layer's first code), ``aux_capture_in_flight`` and
    ``moe_host_round_odd_cores`` (the routed host round from the core-1
    ranks) on, eager and with DMAs at their waits, every layer's finished and
    gathered rows, the gathered latents, the aux slots and the last rows are
    those of the run with them off, byte for byte. Each MoE's scoped buffers
    and region B are poisoned after it (with the deferred round, the owned
    rows too)."""

    MOE_SETUP = r"""
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
os.environ["JAX_PLATFORMS"] = "cpu"
from sglang_tpu.k3_programs.kimi_k3 import config as kc
kc.install(kc.KernelConfig(decode_presorted_routing=True, decode_router_both_halves=True,
                           expert_schedule=True, moe_token_parallel_up=True,
                           decode_late_shared_down=True))
"""

    MOE_MAIN = r"""
assert dmk.PRESORTED_ROUTING and dmk.TOKEN_PARALLEL_UP and dmk.EXPERT_SCHEDULE
F32 = jnp.float32
LAYERS = 3
B, E, LOC, K, WD, OUT, LAT, SW = 8, 896, 28, 512, 256, 3584, 4096, 192
rng = np.random.default_rng(1)
bf = lambda x: np.asarray(x, np.float32).astype(BF16)
hidden = bf(rng.standard_normal((LAYERS, B, H)))
router = np.broadcast_to(bf(rng.standard_normal((H, E)) * H**-0.5), (32, H, E))
bias = (0.01 * rng.standard_normal((1, E))).astype(np.float32)
norm = np.zeros((1, LAT), np.float32)
norm[0, :OUT] = 1 + 0.1 * rng.standard_normal(OUT)
# Each layer's own latent-down weight.
ld = bf(rng.standard_normal((32, LAYERS, H, 128)) * H**-0.5)
lu = bf(rng.standard_normal((32, OUT, 1024)) * OUT**-0.5)
sgu = bf(rng.standard_normal((32, H, 2 * SW)) * H**-0.5)
sd = bf(rng.standard_normal((32, SW, H)) * SW**-0.5)
egu = rng.integers(0, 2**32, (32, LOC, K // 8, 2 * WD), dtype=np.uint32)
egus = rng.integers(118, 124, (32, LOC, K // 32, 2 * WD), dtype=np.uint8)
edown = rng.integers(0, 2**32, (32, LOC, WD // 8, OUT), dtype=np.uint32)
eds = rng.integers(118, 124, (32, LOC, WD // 32, OUT), dtype=np.uint8)

acc_rows = dmk._expert_accumulator_rows(B, jnp.uint32, group_pending_routes=True)
crows = dmk._compact_rows(B, group_pending_routes=True)
vmem, dma, smem = pltpu.VMEM, pltpu.SemaphoreType.DMA, pltpu.SMEM
WS = (vmem((B, 128), jnp.int32), vmem((B, 128), F32), vmem((B, 128), jnp.int32),
      vmem((B, 128), F32), vmem((B, 1024), F32), vmem((B, H), F32), vmem((B, LAT), F32),
      vmem((acc_rows[0], 4 * WD), F32), vmem((acc_rows[1], OUT), F32),
      vmem((B, 128), jnp.int32),
      *((vmem((crows, K), F32),) * 2 if crows else (vmem((1, K), jnp.bfloat16),) * 2),
      dma((10,)), dma((10,)), vmem((12, B // 4 * 8, 128), jnp.bfloat16), dma((7,)),
      dma((7,)), dma((2,)), dma((2,)))
OWN = (vmem((2, H, 512), jnp.bfloat16), dma((2,)), vmem((OUT, 1024), jnp.bfloat16), dma(()),
       vmem((4, K // 8, 2 * WD), jnp.uint32), vmem((4, K // 32, 2 * WD), jnp.uint8),
       vmem((4, WD // 8, OUT), jnp.uint32), vmem((4, WD // 32, OUT), jnp.uint8), dma((4, 4)),
       vmem((H, 128), jnp.bfloat16), vmem((H, 2 * SW), jnp.bfloat16),
       vmem((SW, H), jnp.bfloat16), dma((2,)), smem((8, 128), jnp.int32), dma(()),
       vmem((B, H), jnp.bfloat16))
COMM = dmk.communication_scratch(B)


def run(prefetch, region, mode):
    dmk.ROUTED_SEND_IN_EXPERT_REGION = region  # read when moe() is traced

    def kernel(hidden_ref, bias_ref, norm_ref, router_hbm, ld_hbm, lu_hbm, sgu_hbm, sd_hbm,
               egu_hbm, egus_hbm, ed_hbm, eds_hbm, out_ref, latent_out, *scratch):
        own, ws, comm = scratch[:len(OWN)], scratch[len(OWN):-len(COMM)], scratch[-len(COMM):]
        (router_v, router_sems, lu_v, lu_sem, egu_v, egus_v, ed_v, eds_v, e_sems, ld_v, sgu_v,
         sd_v, ld_sems, schedule, schedule_sem, rows_v) = own
        comm_ws = dmk.AttentionCommunicationWorkspace(*comm)
        comm_ws.reduction_phase[0] = 0
        for layer in range(LAYERS):
            if layer:
                # The attention's place: no rank starts the next layer's MoE
                # before every rank has finished this one.
                dmk.collectives32.barrier()
            # Region B between layers: other weights' bytes until a copy lands.
            for view in (ld_v, sgu_v, sd_v, lu_v):
                view[...] = jnp.full(view.shape, jnp.nan, view.dtype)
            rows_v[...] = hidden_ref[layer]
            weights = dmk.MixtureOfExpertsWeights(
                router_hbm, router_v, router_sems, bias_ref, ld_hbm.at[layer], lu_hbm, lu_v,
                lu_sem, norm_ref, sgu_hbm, sd_hbm, egu_hbm, egu_v, egus_hbm, egus_v, ed_hbm, ed_v,
                eds_hbm, eds_v, e_sems)
            copy = None
            if prefetch:  # as the attention starts it, on its layer parity's semaphore
                copy = pltpu.make_async_copy(ld_hbm.at[layer], ld_v, ld_sems.at[layer % 2])
                copy.start()
            dmk.moe(rows_v, out_ref.at[layer], weights=weights,
                    workspace=dmk.MixtureOfExpertsWorkspace(
                        *ws, expert_schedule_smem=schedule, expert_schedule_semaphore=schedule_sem),
                    communication_workspace=comm_ws, synchronize_devices=layer == 0,
                    group_pending_expert_routes=True, token_scatter_output=True,
                    routed_reduce_scatter=True, token_parallel_up=True,
                    prepare_weights_vmem=(ld_v, sgu_v, sd_v), latent_input_prefetch=copy)
            latent_out[layer] = comm_ws.latent_gather_vmem[...]

    def body(h, b, n, *hbm):
        out = pl.pallas_call(
            kernel,
            out_shape=(jax.ShapeDtypeStruct((LAYERS, B // 4, H), jnp.bfloat16),
                       jax.ShapeDtypeStruct((LAYERS, B, LAT), jnp.bfloat16)),
            in_specs=[pl.BlockSpec(memory_space=vmem)] * 3 + [pl.BlockSpec(memory_space=pl.ANY)] * 9,
            out_specs=[pl.BlockSpec(memory_space=vmem)] * 2,
            scratch_shapes=[*OWN, *WS, *COMM],
            compiler_params=pltpu.CompilerParams(has_side_effects=True, collective_id=7),
            interpret=pltpu.InterpretParams(detect_races=True, dma_execution_mode=mode,
                                            uninitialized_memory="nan"),
        )(h, b, n, *(x[0] for x in hbm))
        return tuple(x[None] for x in out)

    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P(), P(), *(P("tp"),) * 9),
                              out_specs=(P("tp"),) * 2, check_vma=False))
    return [np.asarray(x) for x in f(hidden, bias, norm, router, ld, lu, sgu, sd, egu, egus,
                                     edown, eds)]


prefetch, region, mode = os.environ["RUN"].split(",")
rows, latents = run(prefetch == "1", region == "1", mode)
assert not ipc.races.races_found  # reset by every interpreted call
out = rows.astype(np.float32)
assert np.isfinite(out).all() and (np.abs(out).max(axis=(0, 2, 3)) > 0).all()
assert np.isfinite(latents.astype(np.float32)).all()
# The layers differ (their rows and latent-down weights do).
assert not np.array_equal(rows[:, 0], rows[:, 1])
for name, x in (("rows", rows), ("latents", latents)):
    np.save(os.path.join(os.environ["OUT"], name + ".npy"), x.view(np.uint16))
print("OK")
"""

    CHAIN_MAIN = r"""# After MOE_MAIN's setup with LAYERS = 5: layers 1..4 run in the loop.
from sglang_tpu.k3_programs.common import collectives32

T, AUX = 2, (4, 2)  # the pair's rows; captures after layers 3 and 1 (slots out of layer order)
CAPTURES = sorted(a - 1 for a in AUX)
x0 = hidden[0]
qup = bf(rng.standard_normal((32, H, 128)) * H**-0.5)  # the "MLA" layers' q-up (in region B)
state = (rng.standard_normal((32, LAYERS, B, H)) * 0.05).astype(np.float32)  # "KDA" state reads
noise = (rng.standard_normal((32, LAYERS, B, H)) * 0.01).astype(np.float32)
ROUND = 14  # WS index of the output round's transport (then its two semaphore arrays)
# owned, prefix, mixture (AUX staging), attention input, "KDA" state, state / q-up semaphores,
# aux semaphore, AUX's remote pair, the attention stand-in's reduction buffers and semaphores.
MINE = (vmem((T, H), jnp.bfloat16), vmem((B, H), jnp.bfloat16), vmem((B, H), jnp.bfloat16),
        vmem((B, H), F32), vmem((B, H), F32), dma(()), dma(()), dma((1,)), dma((2,)),
        COMM[0], COMM[2], dma((16,)), dma((16,)))


def run(knobs, mode, args):
    on = knobs == "on"
    prefetch = region = ovb = aux_flight = input_wait = on
    dmk.ROUTED_SEND_IN_EXPERT_REGION = region  # read when moe() is traced
    dmk.OUTPUT_ROUND_NEXT_LAYER = ovb
    dmk.TOKEN_PARALLEL_ODD_HOST_ROUND = on

    def kernel(x0_ref, bias_ref, norm_ref, noise_ref, router_hbm, ld_hbm, qup_hbm, lu_hbm, sgu_hbm,
               sd_hbm, egu_hbm, egus_hbm, ed_hbm, eds_hbm, state_hbm, aux_in, owned_out,
               gathered_out, latent_out, final_out, aux_out, phase_out, *scratch):
        del aux_in  # aliased to aux_out (NaN)
        own = scratch[:len(OWN)]
        ws = scratch[len(OWN):len(OWN) + len(WS)]
        comm = scratch[len(OWN) + len(WS):len(OWN) + len(WS) + len(COMM)]
        (owned, prefix, mixture, att_in, kst_v, kst_sem, qup_sem, aux_sem, aux_remote, att_local,
         att_host, att_send, att_recv) = scratch[len(OWN) + len(WS) + len(COMM):]
        (router_v, router_sems, lu_v, lu_sem, egu_v, egus_v, ed_v, eds_v, e_sems, ld_v, sgu_v,
         sd_v, ld_sems, schedule, schedule_sem, rows_v) = own
        comm_ws = dmk.AttentionCommunicationWorkspace(*comm)
        workspace = dmk.MixtureOfExpertsWorkspace(
            *ws, expert_schedule_smem=schedule, expert_schedule_semaphore=schedule_sem)
        outv, osend, orecv = ws[ROUND:ROUND + 3]
        phase = comm_ws.reduction_phase
        phase[0] = 0
        rank = jax.lax.axis_index("tp")
        pair = (rank % 8) // 2
        def poison(ref):  # NaN, or -1 in integer buffers
            bad = jnp.nan if jnp.issubdtype(ref.dtype, jnp.floating) else -1
            ref[...] = jnp.full(ref.shape, bad, ref.dtype)
        stage = mixture.at[pl.ds(0, T)]
        prefix[...] = x0_ref[...]
        collectives32.barrier()

        def latent_down_copy(index):  # as the stack's: the layer parity's semaphore
            return pltpu.make_async_copy(ld_hbm.at[index], ld_v, ld_sems.at[index % 2])

        def start_latent_down_copy(index):  # the stack's region on a never-false condition
            pl.when(phase[0] >= 0)(lambda: latent_down_copy(index).start())

        def finish(index, capture=True):  # q0 (prefix rows + MoE rows), then the aux capture
            for o in range(4):
                @pl.when(pair == o)
                def q0(o=o):
                    rows = pl.ds(o * T, T)
                    finished = (prefix[rows].astype(F32) + owned[...].astype(F32)).astype(jnp.bfloat16)
                    owned[...] = finished
                    prefix[rows] = finished
            owned_out[index] = owned[...]
            if not capture:
                return
            if aux_flight:
                dmk.capture_aux_rows(index, AUX, CAPTURES[0], owned, stage, aux_out, aux_sem.at[0],
                                     aux_remote, owners=4, cross_cores=True)
            else:
                for slot, aux_layer in enumerate(AUX):
                    @pl.when(index == aux_layer - 1)
                    def gather_capture(slot=slot):
                        dmk._gather_local_pair_rows(
                            owned, prefix, comm_ws.send_semaphores, comm_ws.receive_semaphores,
                            comm_ws.shared_local_reduction_vmem, owners=4, cross_cores=True)
                        copy = pltpu.make_async_copy(prefix, aux_out.at[slot], aux_sem.at[0])
                        copy.start()
                        copy.wait()

        @pl.loop(1, LAYERS)
        def layer(index):
            if ovb:
                @pl.when(index > 1)
                def finish_previous_layer():  # the stack's first code with the knob
                    dmk.finish_token_parallel_outputs(outv, osend, orecv, owned)
                    phase[0] += 3
                    finish(index - 1)

            for o in range(4):  # q1: the pair's attention rows from its prefix rows
                @pl.when(pair == o)
                def q1(o=o):
                    owned[...] = (prefix[pl.ds(o * T, T)].astype(F32) * 0.75 + 0.25).astype(jnp.bfloat16)
            # q2: the crossed pre-attention gather (communication semaphores 0..2).
            wait_input = dmk._gather_local_pair_rows(
                owned, rows_v, comm_ws.send_semaphores, comm_ws.receive_semaphores,
                comm_ws.shared_local_reduction_vmem, owners=4, cross_cores=True,
                defer_waits=input_wait)

            def wait_for_attention_input():
                if wait_input is not None:
                    wait_input()

            def delta_attention():  # waits first, then the state reads and the latent-down copy
                wait_for_attention_input()
                state_copy = pltpu.make_async_copy(state_hbm.at[index], kst_v, kst_sem)
                state_copy.start()
                if prefetch:
                    start_latent_down_copy(index)
                state_copy.wait()
                gathered_out[index] = rows_v[...]
                att_in[...] = rows_v[...].astype(F32) * 0.5 + kst_v[...]

            def latent_attention():  # its q-up copy into region B, the wait, q-up, the copy after it
                qup_copy = pltpu.make_async_copy(qup_hbm, ld_v, qup_sem)
                qup_copy.start()
                wait_for_attention_input()
                qup_copy.wait()
                rows = rows_v[...]
                query = jnp.dot(rows, ld_v[...], preferred_element_type=F32)
                if prefetch:
                    start_latent_down_copy(index)
                gathered_out[index] = rows
                att_in[...] = rows.astype(F32) * 0.5 + jnp.tile(query, (1, H // 128))

            jax.lax.cond(index % 2 == 1, latent_attention, delta_attention)
            att = dmk._all_reduce_rows(noise_ref[index] + att_in[...], att_local, att_host,
                                       att_send, att_recv)
            for o in range(4):  # q9: the attention rows into the pair's prefix rows
                @pl.when(pair == o)
                def q9(o=o):
                    rows = pl.ds(o * T, T)
                    prefix[rows] = (prefix[rows].astype(F32) + att[o * T:(o + 1) * T]).astype(jnp.bfloat16)
            rows_v[...] = (rows_v[...].astype(F32) + att).astype(jnp.bfloat16)  # the MoE input
            collectives32.barrier()  # the MoE input gather (t10) and the routing rounds
            weights = dmk.MixtureOfExpertsWeights(
                router_hbm, router_v, router_sems, bias_ref, ld_hbm.at[index], lu_hbm, lu_v,
                lu_sem, norm_ref, sgu_hbm, sd_hbm, egu_hbm, egu_v, egus_hbm, egus_v, ed_hbm, ed_v,
                eds_hbm, eds_v, e_sems)
            dmk.moe(rows_v, owned, weights=weights, workspace=workspace,
                    communication_workspace=comm_ws, synchronize_devices=False,
                    group_pending_expert_routes=True, token_scatter_output=True,
                    routed_reduce_scatter=True, token_parallel_up=True,
                    prepare_weights_vmem=(ld_v, sgu_v, sd_v),
                    latent_input_prefetch=latent_down_copy(index) if prefetch else None,
                    output_round_next_layer=ovb)
            latent_out[index] = comm_ws.latent_gather_vmem[...]
            # The MoE's scoped buffers and region B hold other data from here on (the round's
            # transport is persistent); with the deferred round, so do the owned rows.
            for ref in (*ws[:12], ld_v, sgu_v, sd_v, lu_v, *((owned,) if ovb else ())):
                poison(ref)
            if not ovb:
                finish(index)

        if ovb:  # the last layer's round and finish (no capture)
            dmk.finish_token_parallel_outputs(outv, osend, orecv, owned)
            phase[0] += 3
            finish(jnp.int32(LAYERS - 1), capture=False)
        if aux_flight:  # the last capture's copies
            dmk.wait_aux_rows(stage, aux_out, aux_sem.at[0], aux_remote, owners=4, cross_cores=True)
        owned[...] = (owned[...].astype(F32) * 0.5).astype(jnp.bfloat16)
        dmk._gather_local_pair_rows(owned, rows_v, comm_ws.send_semaphores, comm_ws.receive_semaphores,
                                    comm_ws.shared_local_reduction_vmem, owners=4, cross_cores=True)
        final_out[...] = rows_v[...]
        phase_out[0] = phase[0]

    def body(x, b, n, z, *hbm):
        out = pl.pallas_call(
            kernel,
            out_shape=(jax.ShapeDtypeStruct((LAYERS, T, H), jnp.bfloat16),
                       jax.ShapeDtypeStruct((LAYERS, B, H), jnp.bfloat16),
                       jax.ShapeDtypeStruct((LAYERS, B, LAT), jnp.bfloat16),
                       jax.ShapeDtypeStruct((B, H), jnp.bfloat16),
                       jax.ShapeDtypeStruct((len(AUX), B, H), jnp.bfloat16),
                       jax.ShapeDtypeStruct((1,), jnp.int32)),
            in_specs=[pl.BlockSpec(memory_space=vmem)] * 4 + [pl.BlockSpec(memory_space=pl.ANY)] * 12,
            out_specs=[pl.BlockSpec(memory_space=vmem)] * 4 + [pl.BlockSpec(memory_space=pl.ANY),
                                                               pl.BlockSpec(memory_space=smem)],
            scratch_shapes=[*OWN, *WS, *COMM, *MINE],
            input_output_aliases={15: 4},
            compiler_params=pltpu.CompilerParams(has_side_effects=True, collective_id=7),
            interpret=pltpu.InterpretParams(detect_races=True, dma_execution_mode=mode,
                                            uninitialized_memory="nan"),
        )(x, b, n, z[0], *(h[0] for h in hbm))
        return tuple(o[None] for o in out)

    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P(), P(), *(P("tp"),) * 13),
                              out_specs=(P("tp"),) * 6, check_vma=False))
    return [np.asarray(x) for x in f(*args)]


# The operands on their devices; the host copies go (32 interpret devices peak at about 9 GB).
put = lambda x, spec: jax.device_put(x, jax.sharding.NamedSharding(mesh, spec))
args = [put(x0, P()), put(bias, P()), put(norm, P())]
for name in ("noise", "router", "ld", "qup", "lu", "sgu", "sd", "egu", "egus", "edown", "eds", "state"):
    args.append(put(globals().pop(name), P("tp")))
args.append(put(np.full((32, len(AUX), B, H), np.nan, BF16), P("tp")))
del hidden
knobs, mode = os.environ["RUN"].split(",")
owned_rows, gathered, latents, final, aux, phase = run(knobs, mode, args)
assert not ipc.races.races_found  # reset by every interpreted call
for name, x in (("owned", owned_rows), ("gathered", gathered), ("latents", latents), ("final", final),
                ("aux", aux)):
    assert np.isfinite(x[:, 1:].astype(np.float32) if name in ("owned", "gathered", "latents")
                       else x.astype(np.float32)).all(), name
assert len(set(phase[:, 0].tolist())) == 1, phase[:, 0]
for r in range(32):  # both ranks of a pair and every host hold the pair's rows
    assert owned_rows[r].tobytes() == owned_rows[r % 8 // 2 * 2].tobytes(), r
# Each aux slot: the four pairs' finished rows of its layer, on every rank.
want = np.stack([np.concatenate([owned_rows[2 * o, a - 1] for o in range(4)]) for a in AUX])
assert all(aux[r].tobytes() == want.tobytes() for r in range(32))
for layer in range(2, LAYERS):  # the layers differ
    assert owned_rows[0, layer].tobytes() != owned_rows[0, layer - 1].tobytes(), layer
for name, x in (("owned", owned_rows), ("gathered", gathered), ("latents", latents), ("final", final),
                ("aux", aux), ("phase", phase)):
    np.save(os.path.join(os.environ["OUT"], name + ".npy"), x.view(np.int32 if name == "phase" else np.uint16))
print(f"{knobs} {mode}: phase {phase[0, 0]}")
print("OK")
"""

    def test_moe_latent_down_with_the_expert_region(self):
        import os
        import subprocess
        import sys
        import tempfile

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        prelude = TestAttentionHostBf16.PRELUDE
        script = (
            self.MOE_SETUP + prelude[: prelude.index("def project(")] + self.MOE_MAIN
        )
        knobs = {"off": "0,0", "both": "1,1"}
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ("eager", "on_wait"):
                outputs = {}
                for name in knobs:
                    # One process a run: 32 interpret devices with race
                    # detection peak at about 9 GB.
                    out = os.path.join(tmp, f"{mode}-{name}")
                    os.makedirs(out)
                    env = dict(
                        os.environ,
                        PYTHONPATH=os.pathsep.join(sys.path),
                        NPROC="128",
                        RUN=f"{knobs[name]},{mode}",
                        OUT=out,
                    )
                    result = subprocess.run(
                        [sys.executable, "-c", script],
                        capture_output=True,
                        text=True,
                        env=env,
                        timeout=3600,
                    )
                    self.assertIn(
                        "OK",
                        result.stdout,
                        f"{mode} {name}: {result.returncode} "
                        + result.stdout[-2000:]
                        + result.stderr[-3000:],
                    )
                    self.assertNotIn("non-zero count", result.stdout + result.stderr)
                    outputs[name] = [
                        np.load(os.path.join(out, f"{what}.npy"))
                        for what in ("rows", "latents")
                    ]
                for what, want, got in zip(
                    ("output rows", "gathered latents"),
                    outputs["off"],
                    outputs["both"],
                ):
                    self.assertEqual(want.tobytes(), got.tobytes(), (mode, what))

    def test_layer_chain_with_the_knobs_together(self):
        import os
        import subprocess
        import sys
        import tempfile

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        prelude, setup = TestAttentionHostBf16.PRELUDE, self.MOE_MAIN
        setup = setup[: setup.index("def run(")].replace("LAYERS = 3\n", "LAYERS = 5\n")
        script = (
            self.MOE_SETUP
            + prelude[: prelude.index("def project(")]
            + setup
            + self.CHAIN_MAIN
        )
        names = ("owned", "gathered", "latents", "final", "aux", "phase")
        outputs = {}
        with tempfile.TemporaryDirectory() as tmp:
            # One process a run (32 interpret devices peak at about 9 GB).
            for run in ("off,eager", "on,eager", "on,on_wait"):
                out = os.path.join(tmp, run.replace(",", "-"))
                os.makedirs(out)
                env = dict(
                    os.environ,
                    PYTHONPATH=os.pathsep.join(sys.path),
                    NPROC="128",
                    RUN=run,
                    OUT=out,
                )
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=3600,
                )
                self.assertIn(
                    "OK",
                    result.stdout,
                    f"{run}: {result.returncode} "
                    + result.stdout[-2000:]
                    + result.stderr[-3000:],
                )
                self.assertNotIn("non-zero count", result.stdout + result.stderr)
                outputs[run] = [
                    np.load(os.path.join(out, f"{name}.npy")) for name in names
                ]
        for run in ("on,eager", "on,on_wait"):
            for name, want, got in zip(names, outputs["off,eager"], outputs[run]):
                self.assertEqual(want.tobytes(), got.tobytes(), (run, name))

    @staticmethod
    def _mla_script():
        """TestMlaQ6Knobs's harness with two more variant keys: ``input_wait``
        (mla()'s input rows copied into NaN rows before the layer and waited
        by ``wait_for_input``) and the knobs reaching ``mla_dcp`` too; race
        detection with RACES=1."""
        script = TestMlaQ6Knobs.SCRIPT
        edits = (
            (
                "from jax.sharding import Mesh, PartitionSpec as P\n",
                "from jax.sharding import Mesh, PartitionSpec as P\n"
                "from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc\n"
                'RACES = os.environ.get("RACES") == "1"\n',
            ),
            (
                "    (vmem((QL, 768), jnp.bfloat16), pltpu.SemaphoreType.DMA(())),\n]\n",
                "    (vmem((QL, 768), jnp.bfloat16), pltpu.SemaphoreType.DMA(())),\n"
                "    (vmem((B, H), jnp.bfloat16), pltpu.SemaphoreType.DMA(())),\n]\n",
            ),
            (
                '    clobber = knobs.pop("qup_clobber", None) is not None\n',
                '    clobber = knobs.pop("qup_clobber", None) is not None\n'
                '    input_wait = knobs.pop("input_wait", None) is not None\n',
            ),
            (
                "(phase, new_key, new_value, sems), (nans, nan_sem) = (\n",
                "(phase, new_key, new_value, sems), (nans, nan_sem), (rows_v, rows_sem) = (\n",
            ),
            (
                "        dmk.mla(pos_ref, hidden_ref, out_ref,",
                "        waits = []\n"
                "        rows_copy = pltpu.make_async_copy(hidden_ref, rows_v, rows_sem)\n"
                "        if input_wait:  # the rows still in flight at entry, into stale bytes\n"
                "            rows_v[...] = jnp.full(rows_v.shape, jnp.nan, rows_v.dtype)\n"
                "            rows_copy.start()\n"
                "\n"
                "        def wait_rows():\n"
                "            waits.append(1)\n"
                "            rows_copy.wait()\n"
                "\n"
                "        dmk.mla(pos_ref, rows_v if input_wait else hidden_ref, out_ref,",
            ),
            (
                "                after_query_up=clobber_query_up if clobber else None)\n",
                "                after_query_up=clobber_query_up if clobber else None,\n"
                "                wait_for_input=wait_rows if input_wait else None)\n"
                "        assert len(waits) == int(input_wait), waits  # traced once\n",
            ),
            (
                "    dmk._config = lambda: config  # the knobs mla() reads while tracing\n",
                "    dmk._config = lambda: config  # the knobs mla() reads while tracing\n"
                "    mla_dcp._config = lambda: config\n",
            ),
            (
                "        dmk._config = kc.active\n",
                "        dmk._config = kc.active\n        mla_dcp._config = kc.active\n",
            ),
            (
                '    got = run(knobs, pltpu.InterpretParams(dma_execution_mode=os.environ["MODE"]))\n',
                '    lse_rows = "dcp_pair_lse_rows" in knobs\n'
                "    LSE_CALLS.clear()\n"
                '    got = run(knobs, pltpu.InterpretParams(dma_execution_mode=os.environ["MODE"],\n'
                '                                           detect_races=RACES, uninitialized_memory="nan"))\n'
                "    ok &= not ipc.races.races_found\n"
                "    # The knob's stage two was traced (8-row calls; 16 rows run chunked,\n"
                "    # without the pair stage).\n"
                "    ok &= bool(LSE_CALLS) == (lse_rows and B <= mla_dcp.MAX_ROWS)\n",
            ),
            (
                "bits = lambda x:",
                "LSE_CALLS = []\n"
                "_lse_rows = mla_dcp._merge_pair_lse_rows\n"
                "\n"
                "def _counted_lse_rows(*args):\n"
                "    LSE_CALLS.append(1)\n"
                "    return _lse_rows(*args)\n"
                "\n"
                "mla_dcp._merge_pair_lse_rows = _counted_lse_rows\n"
                "bits = lambda x:",
            ),
        )
        for old, new in edits:
            assert script.count(old) == 1, old
            script = script.replace(old, new)
        return script

    def test_mla_input_wait_with_the_query_up_hook(self):
        import os
        import subprocess
        import sys

        from jax.experimental.pallas import tpu as pltpu

        if not hasattr(pltpu, "InterpretParams"):
            self.skipTest("Pallas TPU interpret mode unavailable")
        script = self._mla_script()
        hooks = "qup_clobber=1,input_wait=1,dcp_pair_lse_rows=1"
        serving = (
            "dsa_gather_overlap=1,decode_mla_prefix_region=1,decode_mla_early_handout=1"
        )
        for case in (
            # B1 x7, eager copies, race detection.
            {"S": "8", "POS": "3000", "TOPK": "200", "ACTIVITY": "16", "MODE": "eager",
             "RACES": "1", "VARIANTS": f";{serving},{hooks}"},
            # Copies executed at their waits: a read of the rows before
            # wait_for_input sees NaNs.
            {"S": "8", "POS": "3000", "TOPK": "200", "ACTIVITY": "0", "MODE": "on_wait",
             "RACES": "0", "VARIANTS": f";input_wait=1;{serving},{hooks}"},
            # B2 x7 (chunked attention; no gather overlap at 16 rows).
            {"S": "8", "POS": "1500,3000", "TOPK": "200", "ACTIVITY": "16", "MODE": "on_wait",
             "RACES": "0", "VARIANTS": f";decode_mla_prefix_region=1,decode_mla_early_handout=1,{hooks}"},
        ):  # fmt: skip
            with self.subTest(**case):
                env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), **case)
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=3600,
                )
                self.assertIn(
                    "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
                )


if __name__ == "__main__":
    unittest.main()
