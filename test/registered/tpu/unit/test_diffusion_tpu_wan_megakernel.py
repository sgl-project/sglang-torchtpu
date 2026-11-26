# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Wan2.2 megakernel (sglang_tpu.multimodal_gen.wan_megakernel).

The kernel runs in Pallas TPU interpret mode (DMAs, semaphores and remote copies
simulated, races detected) against ``reference`` at a small size; ``pack`` is
checked against the reference's layout. Skipped where JAX is unavailable.
"""

import dataclasses
import os
import subprocess
import sys
import textwrap
import types
import unittest
import unittest.mock

# Host-CPU tests: in the TPU image the diffusion platform and JAX would
# otherwise select the TPU.
os.environ.setdefault("SGLANG_DIFFUSION_PLATFORM_OVERRIDE", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(
    est_time=200,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "diffusion"],
)

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

try:
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    from sglang_tpu.multimodal_gen.wan_megakernel import kernel as mk
    from sglang_tpu.multimodal_gen.wan_megakernel import reference as ref
    from sglang_tpu.multimodal_gen.wan_megakernel.tiling import Tiling

    _HAS_JAX = True
except Exception:  # pragma: no cover - depends on JAX availability
    _HAS_JAX = False

_SKIP = "Requires JAX with Pallas TPU interpret mode"


def _case(
    layers=1,
    heads=4,
    rows=256,
    devices=1,
    valid_pad=6,
    qk_scale=1.0,
    seed=1,
    ffn=1024,
    text_len=64,
    text_real=None,
    craft=None,
    rope=True,
    e_rows=None,
):
    """Random weights and inputs at a test size, plus the bf16 and f32 references.
    ``text_real``: the context's rows from it on are zero (sglang's T5 padding).
    ``craft(layer, cfg)`` edits each ``reference.pack_layer`` dict in place; ``rope``
    False rotates by nothing; ``e_rows`` sets modulation rows ``{row: value}``."""
    cfg = ref.small_config(layers=layers, heads=heads, ffn=ffn, text_len=text_len)
    rng = np.random.default_rng(seed)
    layers_ = [ref.pack_layer(ref.random_layer(cfg, rng), cfg) for _ in range(layers)]
    for one in layers_:
        one["qn"] = one["qn"] * qk_scale
        one["kn"] = one["kn"] * qk_scale
        if craft is not None:
            craft(one, cfg)
    st = ref.stack_layers(layers_)
    for k in ("wqkv", "wo", "wq2", "wo2", "w1", "w2", "wk2", "wv2"):
        st[k] = np.asarray(jnp.asarray(st[k], jnp.bfloat16).astype(jnp.float32))
    w = {k: jnp.asarray(v) for k, v in st.items()}
    S = rows * devices
    valid = S - valid_pad
    x = (
        jnp.asarray(rng.standard_normal((S, cfg.hidden)), jnp.bfloat16)
        .at[valid:]
        .set(0)
    )
    e = jnp.asarray(
        st["table"] + rng.standard_normal((6, cfg.hidden))[None] * 0.3, jnp.float32
    )
    for row, value in (e_rows or {}).items():
        e = e.at[:, row].set(value)
    ctx = jnp.asarray(
        rng.standard_normal((cfg.text_len, cfg.hidden)), jnp.bfloat16
    ).astype(jnp.float32)
    if text_real is not None:
        ctx = ctx.at[text_real:].set(0)
    kv = [
        ref.cross_kv(ctx, {k: v[i] for k, v in w.items()}, cfg) for i in range(layers)
    ]
    k2 = jnp.stack([a for a, _ in kv])
    v2 = jnp.stack([b for _, b in kv])
    cos, sin = ref.rope_tables(
        ref.rope_angles(S // 128 + 1, 8, 16, cfg)[: valid if rope else 0], rows=S
    )
    cos, sin = jnp.asarray(cos), jnp.asarray(sin)
    want = ref.stack(
        x.astype(jnp.float32), e, k2, v2, cos, sin, w, cfg, keys=valid, mode="bf16"
    )
    gold = ref.stack(
        x.astype(jnp.float32), e, k2, v2, cos, sin, w, cfg, keys=valid, mode="f32"
    )
    return (
        cfg,
        st,
        x,
        e,
        k2,
        v2,
        cos,
        sin,
        valid,
        np.asarray(want.astype(jnp.float32)),
        np.asarray(gold),
    )


def _tiling(mlp_mode="twopass"):
    return Tiling(
        qkv_rows=128,
        mlp_rows=128,
        cols=256,
        block_q=128,
        block_kv=256,
        sub_kv=128,
        chunk_rows=64,
        mlp_mode=mlp_mode,
        down_cols=256,
    )


def _races_found():
    """The race detector's verdict on the last interpreted kernel."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc

    return bool(ipc.races is not None and ipc.races.races_found)


def _rel(a, b, valid):
    a, b = np.asarray(a, np.float64)[:valid], np.asarray(b, np.float64)[:valid]
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _cross_keys(k2h, v2h, counts):
    """``pack.cross_keys`` (torch) on heads-major bf16 K/V, as JAX arrays."""
    import torch

    from sglang_tpu.multimodal_gen.wan_megakernel import pack

    def bf16(a):
        return torch.from_numpy(np.array(a.astype(jnp.float32))).to(torch.bfloat16)

    keys, weights = pack.cross_keys(bf16(k2h), bf16(v2h), counts)
    return jnp.asarray(keys.numpy()), jnp.asarray(weights.numpy())


def _interpret(case, tiling, softmax="auto", dma=None):
    """One interpreted kernel call on a ``_case`` (every text row a key): its
    output, every row, as float32. ``dma``: the DMA execution mode (else the
    interpreter's default)."""
    cfg, st, x, e, k2, v2, cos, sin, valid, _, _ = case
    e8 = jnp.concatenate(
        [e, jnp.zeros((cfg.layers, 2, cfg.hidden), jnp.float32)], axis=1
    )
    k2h, v2h = (
        a.reshape(cfg.layers, cfg.text_len, cfg.heads, cfg.head_dim)
        .transpose(0, 2, 1, 3)
        .astype(jnp.bfloat16)
        for a in (k2, v2)
    )
    got = mk.stack(
        x,
        e8,
        k2h,
        v2h,
        cos,
        sin,
        mk.pack_kernel_weights(st, cfg, tiling),
        valid,
        cfg=cfg,
        tiling=tiling,
        interpret=pltpu.InterpretParams(
            detect_races=True, **({"dma_execution_mode": dma} if dma else {})
        ),
        softmax=softmax,
    )
    if _races_found():
        raise AssertionError(f"a DMA race in the interpreted kernel ({softmax})")
    return np.asarray(got.astype(jnp.float32))


def envs_mod():
    from sglang_tpu.envs import envs

    return envs


def _norm_weight(layer, cfg, score):
    """q and k norm weights g = sqrt(score / (sqrt(D) log2(e))): unit-RMS q and k rows
    pointing the same way score ``score`` (log2 units) in every head."""
    g = np.sqrt(score / (np.sqrt(cfg.head_dim) * np.log2(np.e)))
    layer["qn"] = np.full_like(layer["qn"], g)
    layer["kn"] = np.full_like(layer["kn"], g)


def _all_rows_far(layer, cfg):
    """Every query against every key at -102 to -105 (bound ~107: the fixed path),
    values near 0.01: a row's fixed sum is ~2^-95 and most products p v 2^-17 fall
    under 2^-126 (flushed to zero, on TPU and by XLA on CPU)."""
    h, w, b = cfg.hidden, layer["wqkv"], layer["bqkv"]
    w[:, : 2 * h] *= 0.03  # q and k: one shared direction plus a little of the row
    b[:, :h], b[:, h : 2 * h] = 1.0, -1.0
    w[:, 2 * h :] *= 0.01
    b[:, 2 * h :] *= 0.5
    _norm_weight(layer, cfg, 104.0)


def _padding_rows_far(layer, cfg):
    """Keys share one direction. A padding row (zeros, so n1 = shift = 0 and its q is
    the bias's direction) has a head-0 query twice as long as a real row's, pointing
    against the keys: score ~ -98, bound ~ 100 (the fixed path), sum ~2^-90. The real
    rows' queries are random: scores within about +-14."""
    h, w, b = cfg.hidden, layer["wqkv"], layer["bqkv"]
    b[:, :h] = 0.0
    b[:, : cfg.head_dim] = -0.05
    w[:, h : 2 * h] *= 0.02
    b[:, h : 2 * h] = 1.0
    _norm_weight(layer, cfg, 49.0)


def _padding_rows_rowfix_far(layer, cfg):
    """``_padding_rows_far`` at twice the scores: a padding row's head-0 score is
    ~-200 and its bound ~204, so the item takes rowfix (bound > FIXED_BOUND) with
    that row offset by ~124: its weights underflow, its sum is 0 and the rowfix
    floor reruns the item online."""
    _padding_rows_far(layer, cfg)
    _norm_weight(layer, cfg, 98.0)


@unittest.skipUnless(_HAS_JAX, _SKIP)
class TestReference(CustomTestCase):
    def test_qk_order_keeps_scores(self):
        """The even-then-odd head-dim reorder with split RoPE equals interleaved RoPE."""
        cfg = ref.small_config(heads=2)
        rng = np.random.default_rng(0)
        q = rng.standard_normal((5, cfg.heads, cfg.head_dim))
        k = rng.standard_normal((5, cfg.heads, cfg.head_dim))
        ang = rng.standard_normal((5, cfg.head_dim // 2))

        def rope_pairs(x):
            x1, x2 = x[..., 0::2], x[..., 1::2]
            c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
            out = np.empty_like(x)
            out[..., 0::2] = x1 * c - x2 * s
            out[..., 1::2] = x1 * s + x2 * c
            return out

        want = np.einsum("shd,thd->hst", rope_pairs(q), rope_pairs(k))
        order = ref.qk_order(cfg).reshape(cfg.heads, cfg.head_dim)[0]
        cos, sin = ref.rope_tables(ang)
        qr = np.asarray(
            ref.rope_split(
                jnp.asarray(q[..., order]), jnp.asarray(cos), jnp.asarray(sin)
            )
        )
        kr = np.asarray(
            ref.rope_split(
                jnp.asarray(k[..., order]), jnp.asarray(cos), jnp.asarray(sin)
            )
        )
        got = np.einsum("shd,thd->hst", qr, kr)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


@unittest.skipUnless(_HAS_JAX, _SKIP)
class TestKernelInterpret(CustomTestCase):
    def _run(
        self,
        softmax="auto",
        qk_scale=1.0,
        mlp_mode="twopass",
        layers=1,
        tiling=None,
        dma="on_wait",
        rows=256,
        err_factor=1.1,
        want_tol=1e-2,
        heads=4,
        text_len=64,
        text_real=None,
        unique=False,
    ):
        """The kernel against the references; ``unique``: cross-attention over the
        keys of ``pack.cross_keys`` (else every text row). Returns the real rows'
        output and the keys attended to."""
        cfg, st, x, e, k2, v2, cos, sin, valid, want, gold = _case(
            layers=layers,
            qk_scale=qk_scale,
            rows=rows,
            heads=heads,
            text_len=text_len,
            text_real=text_real,
        )
        tiling = tiling or _tiling(mlp_mode)
        kw = mk.pack_kernel_weights(st, cfg, tiling)
        e8 = jnp.concatenate(
            [e, jnp.zeros((cfg.layers, 2, cfg.hidden), jnp.float32)], axis=1
        )

        def heads_major(t):
            return (
                t.reshape(cfg.layers, cfg.text_len, cfg.heads, cfg.head_dim)
                .transpose(0, 2, 1, 3)
                .astype(jnp.bfloat16)
            )

        k2h, v2h = heads_major(k2), heads_major(v2)
        keys, key_weights = (
            _cross_keys(k2h, v2h, tiling.cross_keys) if unique else (None, None)
        )
        interp = pltpu.InterpretParams(detect_races=True, dma_execution_mode=dma)
        got = mk.stack(
            x,
            e8,
            k2h,
            v2h,
            cos,
            sin,
            kw,
            valid,
            None if keys is None else keys[0],
            key_weights,
            cfg=cfg,
            tiling=tiling,
            interpret=interp,
            softmax=softmax,
        )
        got = np.asarray(got.astype(jnp.float32))
        self.assertFalse(_races_found())
        # Padding rows too: they are the next layer's masked keys.
        self.assertTrue(np.isfinite(got).all())
        err, ref_err = _rel(got, gold, valid), _rel(want, gold, valid)
        # The kernel's error against the f32 model is the bf16 reference's own.
        self.assertLess(err, err_factor * ref_err + 1e-4, (err, ref_err))
        self.assertLess(_rel(got, want, valid), want_tol)
        return got[:valid], None if keys is None else int(keys[0])

    def test_auto_softmax(self):
        self._run()

    def test_each_softmax_variant(self):
        for variant in ("fixed", "rowfix", "online"):
            with self.subTest(variant=variant):
                self._run(softmax=variant, qk_scale=0.25)

    def test_large_scores_fall_back(self):
        self._run(qk_scale=3.0)

    def test_floor_guard(self):
        """The fixed path's sums are floored like rowfix's: a real row under ROW_FLOOR
        reruns its item with a running max (the output is the online path's, bit for
        bit), padding rows do not count, and normal sums leave the fixed output as is.
        On the rowfix path every row counts: a padding row whose sum underflows to 0
        reruns its item instead of leaving 0 / 0 for the next layer's keys."""
        tiling = _tiling()
        with self.subTest("normal sums: the fixed output"):
            case = _case(qk_scale=0.25)
            np.testing.assert_array_equal(
                _interpret(case, tiling), _interpret(case, tiling, "fixed")
            )
        with self.subTest("tiny sums in padding rows only: the fixed output"):
            case = _case(craft=_padding_rows_far, rope=False, e_rows={ref.SHIFT: 0.0})
            np.testing.assert_array_equal(
                _interpret(case, tiling), _interpret(case, tiling, "fixed")
            )
        with self.subTest("tiny sums in real rows: the online output"):
            # gate 50: the attention term dominates the block's output.
            case = _case(
                rows=512, craft=_all_rows_far, rope=False, e_rows={ref.GATE: 50.0}
            )
            valid, want, gold = case[-3:]
            got = _interpret(case, tiling)
            np.testing.assert_array_equal(got, _interpret(case, tiling, "online"))
            ref_err = _rel(want, gold, valid)
            self.assertLess(_rel(got, gold, valid), 1.1 * ref_err + 1e-4)
            # What the floor prevents: the fixed path loses most of its products.
            fixed = _interpret(case, tiling, "fixed")
            self.assertGreater(_rel(fixed, gold, valid), 10 * ref_err)
        with self.subTest("a zero rowfix sum in a padding row: the online rerun"):
            case = _case(
                craft=_padding_rows_rowfix_far, rope=False, e_rows={ref.SHIFT: 0.0}
            )
            self.assertTrue(np.isfinite(_interpret(case, tiling)).all())
            self.assertFalse(np.isfinite(_interpret(case, tiling, "rowfix")).all())

    def test_layout_options(self):
        """Tiling.pipelined_epilogues lays the same arithmetic out differently: on
        every softmax path the output is the default layout's bit for bit.
        128-column weight tiles: four tiles a phase C projection and eight MLP up
        tiles, so the pipelined streams reach their steady state."""
        base = dataclasses.replace(_tiling(), cols=128)
        options = (
            {"pipelined_epilogues": True},
            # (interpret mode always checks bounds: dma_bounds_checks rides along)
            {"pipelined_epilogues": True, "dma_bounds_checks": False},
        )
        cases = {
            "fixed": _case(qk_scale=0.25),
            "rowfix": _case(qk_scale=3.0),
            "online rerun": _case(
                rows=512, craft=_all_rows_far, rope=False, e_rows={ref.GATE: 50.0}
            ),
        }
        for name, case in cases.items():
            want = _interpret(case, base)
            self.assertTrue(np.isfinite(want).all())
            for opts in options:
                with self.subTest(name, **opts):
                    got = _interpret(case, dataclasses.replace(base, **opts))
                    np.testing.assert_array_equal(got, want)

    def test_more_layout_options(self):
        """Tiling.mask_scores, zero_ahead and semaphore_checks lay the same
        arithmetic out differently: on every softmax path the output is the
        default layout's bit for bit, alone, together and beside
        pipelined_epilogues. 512 rows: a full key tile before the masked one; 256
        rows: the masked key tile alone."""
        options = (
            {"mask_scores": True},
            {"zero_ahead": True},
            # (interpret mode keeps its checks: semaphore_checks rides along)
            {"semaphore_checks": False},
            {
                "mask_scores": True,
                "zero_ahead": True,
                "pipelined_epilogues": True,
            },
        )
        cases = {
            "fixed": _case(qk_scale=0.25),
            "fixed, two key tiles": _case(qk_scale=0.25, rows=512),
            "rowfix": _case(qk_scale=3.0),
            "online rerun": _case(
                rows=512, craft=_all_rows_far, rope=False, e_rows={ref.GATE: 50.0}
            ),
        }
        base = dataclasses.replace(_tiling(), cols=128)
        for name, case in cases.items():
            want = _interpret(case, base)
            self.assertTrue(np.isfinite(want).all())
            for opts in options:
                with self.subTest(name, **opts):
                    got = _interpret(case, dataclasses.replace(base, **opts))
                    np.testing.assert_array_equal(got, want)

    def test_masked_scores_each_variant(self):
        """mask_scores with every softmax variant forced: the fixed path's masked
        tile subtracts its column, rowfix and online keep the select; each gives
        the default layout's output bit for bit (rowfix and online subtract an
        offset after the scale, where a column would change the rounding)."""
        case = _case(qk_scale=3.0)
        base = dataclasses.replace(_tiling(), cols=128)
        ms = dataclasses.replace(base, mask_scores=True)
        for variant in ("fixed", "rowfix", "online"):
            with self.subTest(variant=variant):
                np.testing.assert_array_equal(
                    _interpret(case, ms, variant), _interpret(case, base, variant)
                )

    def test_pipelined_epilogues_dma_orders(self):
        """The pipelined streams with one and with several tiles a stream (cols 512:
        one phase C projection tile, two MLP up tiles; cols 128: four and eight), in
        both DMA orders: no race, and the default layout's output bit for bit."""
        case = _case()
        for cols in (512, 128):
            base = dataclasses.replace(_tiling(), cols=cols)
            pipe = dataclasses.replace(base, pipelined_epilogues=True)
            want = _interpret(case, base)
            for dma in ("eager", "on_wait"):
                with self.subTest(cols=cols, dma=dma):
                    np.testing.assert_array_equal(_interpret(case, pipe, dma=dma), want)
        with self.subTest("acc MLP: the phase C projections only"):
            base = _tiling("acc")
            pipe = dataclasses.replace(base, pipelined_epilogues=True)
            np.testing.assert_array_equal(
                _interpret(case, pipe), _interpret(case, base)
            )

    def test_cross_unique_keys(self):
        """A prompt zero-padded to the text length (sglang's T5 padding) attends to
        its distinct keys (pack.cross_keys) and matches the references: 40 of 256
        rows -> 64 keys, the copies weighted by the binary digits of 216 (the
        all-rows kernel's output up to f32 rounding of the sums); 62 -> 64 keys in
        equal shares; with six heads (three K2/V2 groups of two, so the slots
        alternate, each in steps of one head) and eager DMAs, 100 -> 128 keys."""
        full, keys = self._run(text_len=256, text_real=40)
        self.assertIsNone(keys)
        got, keys = self._run(text_len=256, text_real=40, unique=True)
        self.assertEqual(keys, 64)
        self.assertLess(_rel(got, full, len(got)), 1e-3)
        _, keys = self._run(text_len=256, text_real=62, unique=True)
        self.assertEqual(keys, 64)
        _, keys = self._run(
            heads=6,
            text_len=256,
            text_real=100,
            dma="eager",
            unique=True,
            tiling=dataclasses.replace(_tiling(), cross_step=1),
        )
        self.assertEqual(keys, 128)

    def test_cross_keys_arguments(self):
        """keys and key_weights go together, with the [T, 1] f32 weights."""
        cfg, st, x, e, k2, v2, cos, sin, valid, _, _ = _case()
        tiling = _tiling()
        kw = mk.pack_kernel_weights(st, cfg, tiling)
        T = cfg.text_len
        k2, v2 = (
            t.reshape(1, T, cfg.heads, cfg.head_dim)
            .transpose(0, 2, 1, 3)
            .astype(jnp.bfloat16)
            for t in (k2, v2)
        )
        good = jnp.ones((T, 1), jnp.float32)
        for keys, weights in (
            (32, None),
            (None, good),
            (32, jnp.ones((T,), jnp.float32)),
            (32, jnp.ones((T, 1), jnp.bfloat16)),
            (32.0, good),
        ):
            with self.subTest(
                keys=keys, weights=None if weights is None else weights.shape
            ):
                with self.assertRaises(ValueError):
                    mk.stack(
                        x,
                        e,
                        k2,
                        v2,
                        cos,
                        sin,
                        kw,
                        valid,
                        keys,
                        weights,
                        cfg=cfg,
                        tiling=tiling,
                        interpret=True,
                    )

    def test_accumulator_mlp(self):
        self._run(mlp_mode="acc", layers=2)

    def test_odd_tile_counts(self):
        """Three QKV weight tiles (the ring's parity runs on across row tiles), one
        tile a phase-C projection (each stream's tile 0 goes into the slot the
        previous stream just read), four phase-C row tiles (x slots and prefetched
        inputs reused) and four MLP output tiles (an x slot is refilled only after
        its write-back). Eager DMAs land at start, deferred ones at their wait: each
        order exposes one hazard."""
        tiling = Tiling(
            qkv_rows=128,
            mlp_rows=64,
            cols=512,
            block_q=128,
            block_kv=256,
            sub_kv=128,
            chunk_rows=64,
            mlp_mode="twopass",
            down_cols=128,
        )
        for dma in ("eager", "on_wait"):
            with self.subTest(dma=dma):
                self._run(tiling=tiling, dma=dma)

    def test_v_staging_reuse(self):
        """Four phase-A tiles a projection (cols 128), so the v staging slots are
        reused inside the loop: their old sends are waited before the slot is
        written, in both DMA orders."""
        tiling = dataclasses.replace(_tiling(), cols=128)
        for dma in ("eager", "on_wait"):
            with self.subTest(dma=dma):
                self._run(tiling=tiling, dma=dma)


@unittest.skipUnless(_HAS_JAX, _SKIP)
class TestKernelInterpretMultiDevice(CustomTestCase):
    def test_two_devices_bulk_exchange(self):
        """Two devices under shard_map (Ulysses exchange, bulk head streaming, races).
        A zero-padded prompt, with each device's own cross keys: 64 on device 0, 128
        on device 1 (one kernel, a code path a device), its heads in steps of two."""
        self._two_devices({}, valid_pad=6, err_factor=1.1)

    def _two_devices(self, tiling_over, valid_pad, err_factor):
        """Two layers over two devices of 256 rows, ``_tiling()`` (cross_step 2) with
        ``tiling_over``."""
        script = textwrap.dedent(f"""
            import os, sys
            os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
            os.environ["JAX_PLATFORMS"] = "cpu"
            sys.path.insert(0, {_REPO_ROOT!r})
            sys.path.insert(0, {os.path.dirname(__file__)!r})
            import types, unittest
            try:
                import sglang.test.ci.ci_register, sglang.test.test_utils  # noqa: F401
            except Exception:  # only the helpers below are needed here
                for name in ("sglang", "sglang.test", "sglang.test.ci"):
                    sys.modules.setdefault(name, types.ModuleType(name))
                ci = types.ModuleType("sglang.test.ci.ci_register"); ci.register_tpu_ci = lambda **k: None
                tu = types.ModuleType("sglang.test.test_utils"); tu.CustomTestCase = unittest.TestCase
                sys.modules["sglang.test.ci.ci_register"], sys.modules["sglang.test.test_utils"] = ci, tu
            import numpy as np, jax, jax.numpy as jnp
            from jax.experimental.pallas import tpu as pltpu
            from jax.sharding import Mesh, PartitionSpec as P
            import test_diffusion_tpu_wan_megakernel as t
            from sglang_tpu.multimodal_gen.wan_megakernel import kernel as mk
            cfg, st, x, e, k2, v2, cos, sin, valid, want, gold = t._case(
                layers=2, devices=2, text_len=256, text_real=40, valid_pad={valid_pad!r})
            tiling = t.dataclasses.replace(t._tiling(), cross_step=2, **{tiling_over!r})
            kw = mk.pack_kernel_weights(st, cfg, tiling)
            e8 = jnp.concatenate([e, jnp.zeros((cfg.layers, 2, cfg.hidden), jnp.float32)], axis=1)
            hm = lambda a: a.reshape(cfg.layers, cfg.text_len, cfg.heads, cfg.head_dim).transpose(0, 2, 1, 3).astype(jnp.bfloat16)
            k2h, v2h = hm(k2), hm(v2)
            per = [t._cross_keys(k2h, v2h, (c,)) for c in (64, 128)]
            keys, kws = (jnp.concatenate(a) for a in zip(*per))
            interp = pltpu.InterpretParams(detect_races=True, dma_execution_mode="on_wait")
            mesh = Mesh(np.asarray(jax.devices()[:2]), ("sp",))
            def body(x_, c_, s_, e_, k_, v_, keys_, kw_, *w):
                return mk.stack(x_, e_, k_, v_, c_, s_, dict(zip(mk.KERNEL_WEIGHTS, w)), valid, keys_[0], kw_,
                                cfg=cfg, tiling=tiling, axis="sp", devices=2, interpret=interp)
            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("sp"),) * 3 + (P(),) * 3 + (P("sp"),) * 2 + (P(),) * 7,
                                      out_specs=P("sp"), check_vma=False))
            got = np.asarray(f(x, cos, sin, e8, k2h, v2h, keys, kws, *(kw[n] for n in mk.KERNEL_WEIGHTS)).astype(jnp.float32))
            from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc
            print("RESULT", t._rel(got, gold, valid), t._rel(want, gold, valid), ipc.races.races_found,
                  ",".join(str(int(k)) for k in keys), bool(np.isfinite(got).all()))
        """)
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=900
        )
        line = next(
            (ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")), None
        )
        self.assertIsNotNone(line, out.stderr[-3000:])
        err, ref_err, races, keys, finite = line.split()[1:]
        self.assertEqual(races, "False")
        self.assertEqual(keys, "64,128")
        self.assertEqual(finite, "True")
        self.assertLess(float(err), err_factor * float(ref_err) + 1e-4)


@unittest.skipUnless(_HAS_JAX, _SKIP)
class TestPack(CustomTestCase):
    def test_pack_matches_reference_layout(self):
        """pack.pack_blocks (torch, sglang module attributes) == reference + kernel layout."""
        import torch

        from sglang_tpu.multimodal_gen.wan_megakernel import pack

        cfg = ref.small_config(layers=2, heads=4, ffn=1024, text_len=64)
        rng = np.random.default_rng(3)
        raws = [ref.random_layer(cfg, rng) for _ in range(cfg.layers)]

        def lin(w, b):
            return types.SimpleNamespace(
                weight=torch.from_numpy(w).to(torch.bfloat16),
                bias=torch.from_numpy(b).to(torch.bfloat16),
            )

        def norm(w, b=None):
            n = types.SimpleNamespace(
                weight=torch.from_numpy(w).to(torch.bfloat16), variance_epsilon=1e-6
            )
            if b is not None:
                n.bias = torch.from_numpy(b).to(torch.bfloat16)
            return n

        def block(r):
            b = types.SimpleNamespace()
            b.to_q, b.to_k = (
                lin(r["attn1.to_q.weight"], r["attn1.to_q.bias"]),
                lin(r["attn1.to_k.weight"], r["attn1.to_k.bias"]),
            )
            b.to_v, b.to_out = (
                lin(r["attn1.to_v.weight"], r["attn1.to_v.bias"]),
                lin(r["attn1.to_out.0.weight"], r["attn1.to_out.0.bias"]),
            )
            b.norm_q, b.norm_k = (
                norm(r["attn1.norm_q.weight"]),
                norm(r["attn1.norm_k.weight"]),
            )
            b.self_attn_residual_norm = types.SimpleNamespace(
                norm=norm(r["norm2.weight"], r["norm2.bias"])
            )
            b.attn2 = types.SimpleNamespace(
                to_q=lin(r["attn2.to_q.weight"], r["attn2.to_q.bias"]),
                to_out=lin(r["attn2.to_out.0.weight"], r["attn2.to_out.0.bias"]),
                norm_q=norm(r["attn2.norm_q.weight"]),
            )
            b.ffn = types.SimpleNamespace(
                fc_in=lin(r["ffn.net.0.proj.weight"], r["ffn.net.0.proj.bias"]),
                fc_out=lin(r["ffn.net.2.weight"], r["ffn.net.2.bias"]),
            )
            b.scale_shift_table = torch.from_numpy(r["scale_shift_table"]).float()
            b.to = lambda *_a, **_k: b
            return b

        model = types.SimpleNamespace(
            blocks=[block(r) for r in raws],
            num_attention_heads=cfg.heads,
            text_len=cfg.text_len,
        )

        def bf(v):
            return np.asarray(
                torch.from_numpy(np.asarray(v, np.float32)).to(torch.bfloat16).float()
            )

        raws16 = [
            {k: (bf(v) if k != "scale_shift_table" else v) for k, v in r.items()}
            for r in raws
        ]
        st = ref.stack_layers([ref.pack_layer(r, cfg) for r in raws16])
        for mode in ("acc", "twopass"):
            tiling = _tiling(mode)
            got = pack.pack_blocks(
                model,
                cfg,
                tiling.cols,
                free_originals=False,
                stage_on_host=False,
                down_cols=tiling.down_cols if mode == "twopass" else None,
            )
            want = mk.pack_kernel_weights(st, cfg, tiling)
            for name in pack.WEIGHT_ORDER:
                with self.subTest(mode=mode, name=name):
                    a = got[name].float().numpy()
                    b = np.asarray(jnp.asarray(want[name]).astype(jnp.float32))
                    self.assertEqual(a.shape, b.shape)
                    np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(
            pack.modulation_tables(model).numpy(), st["table"]
        )


@unittest.skipUnless(_HAS_JAX, _SKIP)
class TestCrossKeys(CustomTestCase):
    """pack.cross_keys: the keys a zero-padded prompt's cross-attention attends to."""

    def kv(self, n, T=512):
        """Random ``[L, heads, T, D]`` K/V whose rows from ``n`` on repeat the last."""
        import torch

        g = torch.Generator().manual_seed(n)
        k, v = (torch.randn(2, 3, T, 16, generator=g).bfloat16() for _ in range(2))
        k[:, :, n:], v[:, :, n:] = k[:, :, -1:], v[:, :, -1:]
        return k, v

    def test_counts_and_weights(self):
        import torch

        from sglang_tpu.multimodal_gen.wan_megakernel import pack

        cases = {  # distinct rows -> keys, the copies' leading weights
            40: (64, [8, 16, 64, 128, 256, 0]),  # 472's binary digits, then 0
            1: (64, [1, 2, 4, 8, 16, 32, 64, 128, 256, 0]),
            62: (64, [225, 225]),  # 450 has 4 digits for 2 copies: equal shares
            98: (128, [2, 4, 8, 16, 128, 256, 0]),
            300: (512, [1, 1]),  # above every count: every row, weights 1
        }
        for n, (want, lead) in cases.items():
            with self.subTest(n=n):
                k, v = self.kv(n)
                keys, w = pack.cross_keys(k, v, (64, 128, 256))
                self.assertEqual(keys.dtype, torch.int32)
                self.assertEqual((w.dtype, tuple(w.shape)), (torch.float32, (512, 1)))
                self.assertEqual(int(keys[0]), want)
                w = w[:want, 0].double()
                self.assertEqual(w[:n].tolist(), [1.0] * n)
                self.assertEqual(w[n : n + len(lead)].tolist(), lead)
                self.assertEqual(float(w.sum()), 512.0)
                # The weighted softmax over the first keys rows is the all-rows one.
                q = torch.randn(3, 5, 16, dtype=torch.float64)

                def attend(k_, v_, w_):
                    s = torch.einsum("hqd,hkd->hqk", q, k_.double())
                    p = torch.exp(s - s.amax(-1, keepdim=True)) * w_
                    return (
                        torch.einsum("hqk,hkd->hqd", p, v_.double())
                        / p.sum(-1)[..., None]
                    )

                torch.testing.assert_close(
                    attend(k[0, :, :want], v[0, :, :want], w),
                    attend(k[0], v[0], torch.ones(512, dtype=torch.float64)),
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_only_equal_rows_collapse(self):
        """Rows are compared exactly: a differing padding row ends the repeated run."""
        from sglang_tpu.multimodal_gen.wan_megakernel import pack

        k, v = self.kv(40)
        v[1, 2, 100, 5] += 1  # rows 101.. still repeat: 128 keys
        self.assertEqual(int(pack.cross_keys(k, v, (64, 128))[0][0]), 128)
        k[0, 0, 511, 0] += 1  # the last row is unique: every row
        keys, w = pack.cross_keys(k, v, (64, 128))
        self.assertEqual(int(keys[0]), 512)
        self.assertEqual(w.unique().tolist(), [1.0])

    def test_nonblocking_upload(self):
        """nonblocking: the same keys and weights, and no blocking upload
        (``torch.tensor(..., device=...)``): the counts go up by a non-blocking
        copy of a host tensor."""
        import torch

        from sglang_tpu.multimodal_gen.wan_megakernel import pack

        real = torch.tensor
        for n in (40, 98, 300):
            k, v = self.kv(n)
            want = pack.cross_keys(k, v, (64, 128, 256))
            calls = []

            def counted(*args, **kwargs):
                calls.append("device" in kwargs)
                return real(*args, **kwargs)

            with unittest.mock.patch.object(torch, "tensor", counted):
                got = pack.cross_keys(k, v, (64, 128, 256), nonblocking=True)
            with self.subTest(n=n):
                self.assertEqual(calls, [False])
                for a, b in zip(got, want):
                    self.assertEqual((a.dtype, a.shape), (b.dtype, b.shape))
                    self.assertTrue(torch.equal(a, b))


@unittest.skipUnless(_HAS_JAX, _SKIP)
class TestTiling(CustomTestCase):
    def test_production_shape(self):
        Tiling().check(8192, 4, 5120, 13824, 40)
        opts = Tiling(pipelined_epilogues=True)
        for devices in (4, 2, 1):  # 4 / 2 / 1 chips (the staging tile fits VMEM)
            opts.check(32768 // devices, devices, 5120, 13824, 40)
        self.assertEqual(opts.opt_suffix(), "_pe")
        self.assertEqual(Tiling(dma_bounds_checks=False).opt_suffix(), "_nb")
        self.assertEqual(
            Tiling(pipelined_epilogues=True, dma_bounds_checks=False).opt_suffix(),
            "_pe_nb",
        )
        more = Tiling(
            pipelined_epilogues=True,
            dma_bounds_checks=False,
            semaphore_checks=False,
            mask_scores=True,
            zero_ahead=True,
        )
        self.assertEqual(more.opt_suffix(), "_pe_nb_ns_ms_za")
        for devices in (4, 2, 1):
            more.check(32768 // devices, devices, 5120, 13824, 40)

    def test_for_tokens(self):
        """Rows a device and attention tiles a token count runs with."""
        from sglang_tpu.multimodal_gen.wan_megakernel.tiling import for_tokens

        cases = {  # (tokens, devices) -> rows a device, block_q, block_kv, sub_kv
            (21 * 30 * 52, 4): (8192, 8192, 1024, 512),  # 832x480x81
            (21 * 30 * 52, 8): (4096, 4096, 2048, 1024),  # CFG 1 x Ulysses 8
            (21 * 45 * 80, 4): (19456, 9728, 1024, 512),  # 1280x720x81: 2.9% padding
            (21 * 45 * 80, 8): (10240, 10240, 1024, 512),
            (5 * 30 * 52, 4): (2048, 2048, 2048, 1024),  # 832x480x17
            (4 * 23 * 1024 - 100, 4): (
                24576,
                8192,
                1024,
                512,
            ),  # 23 x 1024 splits 1024s
        }
        for (tokens, devices), want in cases.items():
            with self.subTest(tokens=tokens, devices=devices):
                tiling, rows = for_tokens(
                    tokens, devices, Tiling(pipelined_epilogues=True)
                )
                got = (rows, tiling.block_q, tiling.block_kv, tiling.sub_kv)
                self.assertEqual(got, want)
                self.assertTrue(tiling.pipelined_epilogues)
                self.assertGreaterEqual(rows * devices, tokens)
                tiling.check(rows, devices, 5120, 13824, 40)

    def test_rejects_misfit(self):
        with self.assertRaises(ValueError):
            Tiling().check(1024, 4, 5120, 13824, 40)  # rows below the query tile
        big = Tiling(block_q=19456, block_kv=512, sub_kv=512)
        with self.assertRaises(
            ValueError
        ):  # phase B over VMEM (fails to compile on v7x)
            big.check(19456, 4, 5120, 13824, 40)
        Tiling(block_q=19456, block_kv=512, sub_kv=256).check(19456, 4, 5120, 13824, 40)
        with self.assertRaises(ValueError):  # phases C and C2 over VMEM
            Tiling(mlp_rows=2048).check(8192, 4, 5120, 13824, 40)
        for counts in ((64, 40), (128, 64), (64, 72)):  # ascending multiples of 16
            with self.assertRaises(ValueError):
                Tiling(cross_keys=counts).check(8192, 4, 5120, 13824, 40)
        with self.assertRaises(ValueError):
            Tiling(cross_step=3).check(8192, 4, 5120, 13824, 40)


if __name__ == "__main__":
    unittest.main()
