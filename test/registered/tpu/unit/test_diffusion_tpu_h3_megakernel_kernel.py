# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the vendored MiniMax-H3 megakernel
(sglang_tpu.multimodal_gen.h3_megakernel: kernel, reference, tiling).

The kernel runs in Pallas TPU interpret mode, which simulates its DMAs,
semaphores and remote copies, against ``reference`` at a small size, in the
configuration the sglang hook calls: a traced live length, the fixed-max
softmax only on the layers the norm weights prove, bf16. Multi-device cases run
in a fresh process with 2 or 4 CPU devices and the race detector; libtpu's
compile-only client checks that Mosaic lowers it for v7x at H3's widths.
Skipped where JAX is unavailable.
"""

import dataclasses
import json
import os
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

# Host-CPU tests: in the TPU image the diffusion platform and JAX would
# otherwise select the TPU.
os.environ.setdefault("SGLANG_DIFFUSION_PLATFORM_OVERRIDE", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(
    est_time=1000,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "diffusion"],
)

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from sglang_tpu.multimodal_gen.h3_megakernel import bounds  # noqa: E402

try:
    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.multimodal_gen.h3_megakernel import kernel as mk
    from sglang_tpu.multimodal_gen.h3_megakernel import reference as ref
    from sglang_tpu.multimodal_gen.h3_megakernel import tiling as tl

    _HAS_JAX = True
except Exception:  # pragma: no cover - depends on JAX availability
    _HAS_JAX = False

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

_SKIP = "Requires JAX with Pallas TPU interpret mode"

if _HAS_JAX:
    # Two heads a device on up to two devices, the real head and RoPE widths.
    _CONFIG = ref.Config(hidden=256, heads=4, head_dim=128, ffn=512, layers=2)
    # Packing of the tiny config: two heads a QKV tile, 128-column tiles.
    _PACK = {"qkv_heads": 2, "out_cols": 128, "ffn_cols": 128, "down_cols": 128}
    # Two query tiles a device at 256 rows, two compute steps a key tile.
    _TILING = tl.Tiling(
        qkv_rows=64,
        mlp_rows=64,
        block_q=128,
        block_kv=64,
        block_kv_compute=32,
        sub_rows=32,
        n_chunk=128,
        chunk_rows=16,
        epilogue_rows=32,
        stage_slots=3,
    )
    # static_structure cases: 256-row query tiles, so a 512-row device can
    # end its real rows in either one; a cut key tile of two fixed-max and
    # four online sub-blocks.
    _STRUCTURE_TILING = dataclasses.replace(
        _TILING, block_q=256, block_kv_compute_online=16
    )
# Relative L2 and max errors a correct kernel stays under: bf16 rounding of a
# few layers, while a swapped weight or a stale buffer is ~10x more.
_L2, _MAX = 5e-3, 2e-2
# What the sglang hook passes besides the operands: the kernel derives the
# fixed-max layers from the norm weights (nothing vouched), no peaks, bf16, no
# sparse, windowed or cached attention.
_LOSSLESS = dict(
    fixed=None,
    fixed_heads=None,
    return_peaks=False,
    fp8=False,
    scaled_v=False,
    fast_phase_c=False,
)
# The released FL2VA transformer/config.json.
_RELEASED = {
    "hidden_size": 5376,
    "num_layers": 50,
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "ffn_hidden_size": 14336,
    "rope_inv_freq_len": 16,
    "norm_eps": 1e-05,
    "qk_norm_eps": 1e-05,
}


# ------------------------------------------------------------------ fixtures


def _blocks(config, seed=0):
    """Checkpoint-layout blocks with weights scaled like a trained DiT's."""
    rng = np.random.default_rng(seed)
    hidden, width, ffn, dim = config.hidden, config.width, config.ffn, config.head_dim

    def normal(*shape, scale):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    return [
        {
            "attn.qkv_proj.weight": normal(3 * width, hidden, scale=0.06),
            "attn.out_proj.weight": normal(hidden, width, scale=0.04),
            "mlp.fc1.weight": normal(2 * ffn, hidden, scale=0.06),
            "mlp.fc2.weight": normal(hidden, ffn, scale=0.04),
            "norm1.weight": 1 + normal(hidden, scale=0.1),
            "norm2.weight": 1 + normal(hidden, scale=0.1),
            # A score sees the q and k norms almost only as their product;
            # RoPE's cross terms, w_q[i] w_k[i + 48], tell them apart once
            # each dim's weight differs, as a log-normal spread does.
            "attn.q_norm.weight": np.exp(normal(dim, scale=0.6)),
            "attn.k_norm.weight": np.exp(normal(dim, scale=0.6)),
        }
        for _ in range(config.layers)
    ]


def _request(config, tokens, seed=1):
    """A packed request: rows, AdaLN tables, one-hot, RoPE tables."""
    rng = np.random.default_rng(seed)
    steps, time_dim = 2, 64
    adaln_input = rng.standard_normal((steps, time_dim)).astype(np.float32)
    weights = rng.standard_normal((config.layers, 18 * config.hidden, time_dim))
    biases = rng.standard_normal((config.layers, 18 * config.hidden))
    rows = rng.integers(0, 3 * steps, size=tokens)
    # Long runs of one row, as [text | keyframes | audio | video] packs.
    rows[: tokens // 3] = 1
    rows[tokens // 3 : tokens // 2] = 3
    positions = np.stack(
        (np.arange(tokens) % 17, np.arange(tokens) % 5, np.arange(tokens) % 7),
        axis=-1,
    ).astype(np.float32)
    inv_freq = 10000.0 ** (-np.arange(0, 32, 2) / 32)
    cos, sin = ref.rope_tables(
        jnp.asarray(positions), jnp.asarray(inv_freq, jnp.float32), config
    )
    return {
        "x": jnp.asarray(rng.standard_normal((tokens, config.hidden)), jnp.bfloat16),
        "tables": ref.adaln_tables(
            jnp.asarray(adaln_input),
            jnp.asarray(weights * 0.05, jnp.float32),
            jnp.asarray(biases * 0.05, jnp.float32),
        ),
        "rows": rows,
        "onehot": ref.table_onehot(jnp.asarray(rows)),
        "cos": cos,
        "sin": sin,
        "positions": positions,
        "inv_freq": inv_freq,
    }


def _relative_error(got, want):
    return float(np.abs(got - want).max() / np.abs(want).max())


def _errors(got, want):
    """(relative L2, max relative) error of ``got``'s real rows, both
    infinite if any of them is not finite."""
    got = np.asarray(got, np.float32)[: want.shape[0]]
    if not np.isfinite(got).all():
        return float("inf"), float("inf")
    return (
        float(np.linalg.norm(got - want) / np.linalg.norm(want)),
        _relative_error(got, want),
    )


def _weights(config, norm_scale=None):
    """Packed blocks; ``norm_scale``: per layer, a factor on the q and k norm
    weights (small ones put a layer's softmax at a fixed max)."""
    weights = dict(ref.pack_weights(_blocks(config), config, **_PACK))
    if norm_scale is not None:
        factor = np.asarray(norm_scale, np.float32)[:, None, None]
        for name in ("q_norm", "k_norm"):
            weights[name] = (weights[name].astype(np.float32) * factor).astype(
                weights[name].dtype
            )
    return weights


def _run(
    sp=1,
    *,
    rows=256,
    used=None,
    layers=2,
    heads=None,
    traced=True,
    races=False,
    norm_scale=None,
    tiling=None,
    fixture_rows=None,
    **kwargs,
):
    """The kernel on ``sp`` devices of ``rows`` rows, as the hook calls it.

    Args:
      used: Real rows of the sequence; default all but the last 37.
      traced: Pass ``used`` as a traced ``[1]`` int32 operand (the hook's
        ``cu_seqlens[1:2]``), else as a Python int.
      races: Run the interpreter's race detector, with eager DMAs.
      fixture_rows: Cut the operands from a request of this many rows (the
        same real rows at another row count).
      kwargs: More ``stack`` arguments, over ``_LOSSLESS``.

    Returns:
      ``[sp rows, H]`` bf16, every row the kernel returns (the padding too),
      ``[used, H]`` f32 from the reference, and whether a race was found.
    """
    total = rows * sp
    used = total - 37 if used is None else used
    config = dataclasses.replace(_CONFIG, layers=layers, heads=heads or _CONFIG.heads)
    weights = _weights(config, norm_scale)
    fixture = _request(config, max(total, fixture_rows or 0))
    operands = [fixture[name][:total] for name in ("x", "onehot", "cos", "sin")]
    interpret = pltpu.InterpretParams(
        detect_races=races, dma_execution_mode="eager" if races else "on_wait"
    )
    options = dict(_LOSSLESS, **kwargs)

    def local(x, onehot, cos, sin, weights, tables, live):
        return mk.stack(
            x,
            weights,
            tables,
            onehot,
            cos,
            sin,
            live[0] if traced else used,
            config=config,
            tiling=tiling or _TILING,
            axis="sp" if sp > 1 else None,
            interpret=interpret,
            **options,
        )

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:sp]), ("sp",))
    shard = P("sp")
    got = jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(shard, shard, shard, shard, P(), P(), P()),
            out_specs=shard,
            check_vma=False,
        )
    )(*operands, weights, fixture["tables"], jnp.asarray([used], jnp.int32))
    got = np.asarray(got)
    want = ref.reference_stack(
        *operands[:1], weights, fixture["tables"], *operands[1:], used, config
    )
    return got, np.asarray(want, np.float32)[:used], _races_found()


def _races_found():
    """The race detector's verdict on the last interpreted kernel."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc

    return bool(ipc.races is not None and ipc.races.races_found)


def _same_bits(a, b):
    return bool(
        np.array_equal(np.asarray(a).view(np.uint16), np.asarray(b).view(np.uint16))
    )


def _in_fresh_process(devices, code, timeout):
    """Runs ``code`` (this module imported as ``t``) with ``devices`` CPU
    devices; returns its stdout."""
    prelude = textwrap.dedent(
        f"""
        import os, sys
        sys.path.append({_REPO_ROOT!r})
        sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
        import types, unittest
        try:
            import sglang.test.ci.ci_register, sglang.test.test_utils  # noqa: F401
        except Exception:  # only the helpers of this module are needed here
            for name in ("sglang", "sglang.test", "sglang.test.ci"):
                sys.modules.setdefault(name, types.ModuleType(name))
            ci = types.ModuleType("sglang.test.ci.ci_register")
            ci.register_tpu_ci = lambda *a, **k: None
            tu = types.ModuleType("sglang.test.test_utils")
            tu.CustomTestCase = unittest.TestCase
            sys.modules["sglang.test.ci.ci_register"] = ci
            sys.modules["sglang.test.test_utils"] = tu
        import json, jax
        jax.config.update("jax_platforms", "cpu")
        import test_diffusion_tpu_h3_megakernel_kernel as t
        """
    )
    env = dict(
        os.environ,
        XLA_FLAGS=f"--xla_force_host_platform_device_count={devices}",
        JAX_PLATFORMS="cpu",
        # A compile-only libtpu client beside other processes using libtpu.
        ALLOW_MULTIPLE_LIBTPU_LOAD="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode:
        raise AssertionError(result.stdout[-2000:] + result.stderr[-4000:])
    return result.stdout


# --------------------------------------------------------------------- tests


@unittest.skipUnless(_HAS_JAX, _SKIP)
class TestReference(CustomTestCase):
    """The reference and its operands are H3's block on the checkpoint."""

    @unittest.skipUnless(_HAS_TORCH, "needs torch")
    def test_reference_is_the_checkpoint_block(self):
        layers = _blocks(_CONFIG)
        weights = ref.pack_weights(layers, _CONFIG, **_PACK)
        tokens, used = 128, 91
        fixture = _request(_CONFIG, tokens)
        want = torch.tensor(np.asarray(fixture["x"], np.float32)).bfloat16()
        rows = torch.tensor(fixture["rows"])
        # MiniMaxH3Rope + _rope_cos_sin_cache: [S, 96] cos | sin halves.
        angles = torch.tensor(fixture["positions"])[:, :, None] * torch.tensor(
            fixture["inv_freq"], dtype=torch.float32
        )
        half = angles.reshape(tokens, -1)
        cache = torch.cat((half.cos(), half.sin()), dim=-1).bfloat16()
        x = fixture["x"]
        for index, block in enumerate(layers):
            layer = {name: value[index] for name, value in weights.items()}
            tables = fixture["tables"][index]
            x = ref.reference_layer(
                x,
                layer,
                tables,
                fixture["onehot"],
                fixture["cos"],
                fixture["sin"],
                used,
                _CONFIG,
            )
            params = [
                torch.tensor(np.asarray(tables[k], np.float32)).bfloat16()
                for k in range(6)
            ]
            want = _torch_block(want, block, params, rows, cache, used, _CONFIG)
        got = np.asarray(x, np.float32)[:used]
        self.assertLess(_relative_error(got, want.float().numpy()[:used]), 1.5e-2)

    @unittest.skipUnless(_HAS_TORCH, "needs torch")
    def test_adaln_tables_read_the_projection_as_view_and_chunk(self):
        rng = np.random.default_rng(2)
        steps, time_dim, hidden = 3, 32, 128
        adaln_input = rng.standard_normal((steps, time_dim)).astype(np.float32)
        weight = rng.standard_normal((1, 18 * hidden, time_dim)).astype(np.float32)
        bias = rng.standard_normal((1, 18 * hidden)).astype(np.float32)
        got = ref.adaln_tables(
            jnp.asarray(adaln_input), jnp.asarray(weight), jnp.asarray(bias)
        )
        # MiniMaxH3AdalnProj: linear -> view(3 M, 6 H) -> chunk(6).
        out = torch.addmm(
            torch.tensor(bias[0]).bfloat16(),
            torch.tensor(adaln_input).bfloat16(),
            torch.tensor(weight[0]).bfloat16().T,
        )
        want = out.view(3 * steps, 6 * hidden).chunk(6, dim=-1)
        self.assertEqual(got.shape, (1, 6, 16, hidden))
        for k in range(6):
            np.testing.assert_allclose(
                np.asarray(got[0, k, : 3 * steps], np.float32),
                want[k].float().numpy(),
                rtol=2e-2,
                atol=2e-2,
            )
        self.assertFalse(np.asarray(got[0, :, 3 * steps :], np.float32).any())

    def test_config_reads_the_released_transformer_config(self):
        config = ref.Config.from_transformer_config(_RELEASED)
        self.assertEqual(config, ref.Config())
        self.assertEqual(config.width, 7168)

    def test_pack_weights_refuses_tiles_off_the_lanes(self):
        for widths in ({"out_cols": 64}, {"ffn_cols": 96}, {"qkv_heads": 3}):
            with self.subTest(**widths):
                with self.assertRaisesRegex(ValueError, "cannot tile"):
                    ref.pack_weights(_blocks(_CONFIG), _CONFIG, **{**_PACK, **widths})


@unittest.skipUnless(_HAS_JAX, _SKIP)
class TestTiling(CustomTestCase):
    """Bounds cut down to divide a shape; padded rows that tile well."""

    def test_refuses_shapes_it_cannot_tile(self):
        tiling = tl.Tiling(qkv_rows=64, mlp_rows=64, block_q=128, sub_rows=32)
        with self.assertRaisesRegex(ValueError, "multiple of 128 rows"):
            tiling.fit(_CONFIG, 96, 1)
        with self.assertRaisesRegex(ValueError, "split 3 ways"):
            tiling.fit(_CONFIG, 128, 3)
        # 56 heads split 1, 2, 4 and 8 ways, not 16.
        with self.assertRaisesRegex(ValueError, "split 16 ways"):
            tl.Tiling().fit(ref.Config(), 2048, 16)

    def test_padded_rows_fit_the_tiling(self):
        tiling = tl.Tiling(qkv_rows=64, mlp_rows=64, block_q=128, sub_rows=32)
        self.assertEqual(tl.padded_rows(1, 1, tiling), 128)
        self.assertEqual(tl.padded_rows(129, 2, tiling), 256)
        # 5 s at 768x1344 and at 512x896 on 8 devices: 3% and 7% padding,
        # row tiles of 512/512 and 640/480 rows.
        released = tl.Tiling()
        for tokens, rows, qkv, mlp in (
            (31696, 32768, 512, 512),
            (14394, 15360, 640, 480),
        ):
            with self.subTest(tokens=tokens):
                self.assertEqual(tl.padded_rows(tokens, 8, released), rows)
                fitted = released.fit(ref.Config(), rows // 8, 8)
                self.assertEqual((fitted.qkv_rows, fitted.mlp_rows), (qkv, mlp))
        # A bound off its granule still ends; one no row count meets takes the
        # fewest rows.
        self.assertEqual(tl.padded_rows(31696, 8, tl.Tiling(block_q=192)), 32768)
        self.assertEqual(tl.padded_rows(1000, 1, tl.Tiling(mlp_rows=96)), 1024)

    def test_key_tiles_take_whole_mxu_passes(self):
        # 10 s, 4 s and 5 s at 768x1344 and 5 s at 512x896 on 8 devices: the
        # largest multiple of 256 that divides the rows, not 1920 or 2112.
        released = tl.Tiling()
        for rows, block_kv in (
            (76800, 2560),
            (33792, 1536),
            (32768, 2048),
            (15360, 2560),
        ):
            with self.subTest(rows=rows):
                self.assertEqual(
                    released.fit(ref.Config(), rows // 8, 8).block_kv, block_kv
                )
        # Rows off 256, or a 16-row divisor over twice the 256 one, keep the
        # 16-row rule.
        self.assertEqual(tl.kv_divisor(128, 2560), 128)
        self.assertEqual(tl.kv_divisor(16 * 2047, 2560), 1424)
        self.assertEqual(tl.kv_divisor(256 * 292, 2560), 2336)

    def test_1344x768x124_rows_a_core(self):
        # 1344x768, 124 frames: 37,746-37,757 rows. The tiling the hook runs
        # (the padding bounds plus the tuned mlp_rows and chunk_rows) at
        # the per-core row counts it pads to; 18,944 at 2 cores (the kernel's
        # padded_rows) takes 512-row query tiles.
        tiling = tl.Tiling(
            sub_rows=640,
            stage_slots=24,
            block_kv_compute_online=1024,
            mlp_rows=640,
            chunk_rows=80,
        )
        for devices, rows, want in (
            (8, 4992, (624, 624, 1664, 1536, 48)),
            (8, 4864, (608, 608, 2432, 2048, 32)),
            (4, 9600, (640, 640, 3200, 2560, 80)),
            (2, 19456, (608, 608, 2432, 2048, 32)),
            (2, 18944, (592, 592, 512, 2368, 16)),
        ):
            with self.subTest(devices=devices, rows=rows):
                fitted = tiling.fit(ref.Config(), rows, devices)
                got = (
                    fitted.qkv_rows,
                    fitted.mlp_rows,
                    fitted.block_q,
                    fitted.block_kv,
                    fitted.chunk_rows,
                )
                self.assertEqual(got, want)
                self.assertGreaterEqual(rows * devices, 37757)

    def test_query_tiles_across_devices_off_128_rows(self):
        # split_q_tiles at 4,800 rows a device (no 128-row tile divides it):
        # 3,200-row query tiles, halves on 1,600-row (16-row) boundaries, the
        # 4 x 9,600 key tiles and 480-row phase tiles. Off 128 rows without
        # split_q_tiles, or with no tile whose half divides them: refused.
        tiling = dataclasses.replace(
            tl.Tiling(
                sub_rows=640,
                stage_slots=24,
                block_kv_compute_online=1024,
                mlp_rows=640,
                chunk_rows=80,
            ),
            qkv_rows=704,
        )
        split = dataclasses.replace(tiling, split_q_tiles=True)
        fitted = split.fit(ref.Config(), 4800, 8)
        self.assertEqual(
            (fitted.block_q, fitted.block_kv, fitted.qkv_rows, fitted.chunk_rows),
            (3200, 2560, 480, 80),
        )
        with self.assertRaisesRegex(ValueError, "multiple of 128 rows"):
            tiling.fit(ref.Config(), 4800, 8)
        # 4,816 rows: no 128-row multiple of all 38,528 has a half dividing
        # them.
        with self.assertRaisesRegex(ValueError, "no query tile"):
            split.fit(ref.Config(), 4816, 8)
        # 128-row multiples keep 256-row steps: 3,840 at 9,600 rows a device.
        self.assertEqual(split.fit(ref.Config(), 9600, 4).block_q, 3840)


@unittest.skipUnless(_HAS_JAX, _SKIP)
class TestStackInterpret(CustomTestCase):
    """One device: the hook's call computes the reference's layers."""

    def assert_close(self, errors, slack=1.0):
        l2, worst = errors
        self.assertLess(l2, slack * _L2)
        self.assertLess(worst, slack * _MAX)

    def test_traced_live_length_matches_the_reference(self):
        # Layer 0's norms prove its bound (fixed max), layer 1's do not
        # (online). 219 real rows of 256: the second query tile straddles the
        # padding; 100: the second is all padding; 256: none.
        weights = _weights(_CONFIG, (0.3, 1.5))
        self.assertEqual(
            bounds.fixed_max_layers(
                weights["q_norm"], weights["k_norm"]
            ).layers.tolist(),
            [True, False],
        )
        for used, races in ((219, True), (100, False), (256, False)):
            with self.subTest(used=used):
                got, want, raced = _run(used=used, races=races, norm_scale=(0.3, 1.5))
                self.assertFalse(raced)
                self.assertTrue(np.isfinite(np.asarray(got, np.float32)).all())
                self.assert_close(_errors(got, want))

    def test_traced_live_length_changes_no_bit(self):
        # A traced live length runs the generic walk: the same kernel as a
        # static one without static_items (every row, the padding too), and
        # the same real rows as the static walk.
        # Passing the norm-proven layers and heads as fixed / fixed_heads
        # (bounds.fixed_max_layers) adds nothing to the kernel's own set.
        kwargs = dict(used=219, norm_scale=(0.3, 1.5))
        traced, want, _ = _run(**kwargs)
        generic, _, _ = _run(traced=False, static_items=False, **kwargs)
        static, _, _ = _run(traced=False, static_items=True, **kwargs)
        weights = _weights(_CONFIG, (0.3, 1.5))
        proof = bounds.fixed_max_layers(weights["q_norm"], weights["k_norm"])
        flagged, _, _ = _run(
            fixed=jnp.asarray(proof.layers),
            fixed_heads=jnp.asarray(proof.heads(_CONFIG.heads)),
            **kwargs,
        )
        self.assert_close(_errors(traced, want))
        self.assertTrue(_same_bits(traced, generic))
        self.assertTrue(_same_bits(traced[:219], static[:219]))
        self.assertTrue(_same_bits(traced, flagged))


@unittest.skipUnless(_HAS_JAX, _SKIP)
class TestStackInterpretMultiDevice(CustomTestCase):
    """The Ulysses exchange by remote DMA, race detector on, in a fresh
    process with 2 or 4 CPU devices."""

    def assert_runs(self, stdout, cases, slack=1.0):
        self.assertNotIn("RACE DETECTED", stdout)
        self.assertNotIn("non-zero count", stdout)
        results = json.loads(stdout.splitlines()[-1])
        self.assertEqual(len(results), cases)
        for l2, worst, same in results:
            self.assertLess(l2, slack * _L2)
            self.assertLess(worst, slack * _MAX)
            self.assertTrue(same, "the traced live length changed bits")

    def test_two_devices(self):
        # Two heads a device, 256 rows each, 400 real: the padding starts on
        # device 1, whose last query tile is all padding.
        code = """
            kw = dict(used=400, norm_scale=(0.3, 1.5))
            got, want, raced = t._run(2, races=True, **kw)
            static, _, _ = t._run(2, traced=False, static_items=False, **kw)
            print(json.dumps([[*t._errors(got, want), t._same_bits(got, static) and not raced]]))
        """
        self.assert_runs(_in_fresh_process(2, code, timeout=1800), 1)

    def test_four_devices_new_levers_change_no_bit(self):
        # step_tables (phase A's step scalars, the head groups rotated per
        # device and each head's owner from SMEM tables) and cast_first
        # together, on four devices with three heads a device (the
        # bulk exchange's sends and kept rows) and two layers (fixed max, then
        # online): every row bit for bit against the plain call; the race
        # detector on.
        code = """
            kw = dict(rows=128, heads=12, used=350, norm_scale=(0.3, 1.5))
            base, want, _ = t._run(4, **kw)
            got, _, raced = t._run(4, races=True, step_tables=True,
                                   cast_first=True, **kw)
            print(json.dumps([[*t._errors(got, want),
                               t._same_bits(got, base) and not raced]]))
        """
        self.assert_runs(_in_fresh_process(4, code, timeout=3600), 1, slack=2)

    def test_four_devices_bulk_exchange(self):
        # Three heads a device (bulk_exchange streams the second and third
        # during phase B), 128 rows each, 350 real (the padding starts on
        # device 2; device 3 holds padding only), three layers: fixed max,
        # online, then fixed max again on layer 0's buffers.
        code = """
            kw = dict(rows=128, heads=12, used=350, layers=3, norm_scale=(0.3, 1.5, 0.3))
            got, want, raced = t._run(4, races=True, **kw)
            static, _, _ = t._run(4, traced=False, static_items=False, **kw)
            print(json.dumps([[*t._errors(got, want), t._same_bits(got, static) and not raced]]))
        """
        # Twelve heads (a 1536-wide attention output into 256 hidden) round
        # to about twice the bound.
        self.assert_runs(_in_fresh_process(4, code, timeout=3600), 1, slack=2)

    def test_four_devices_query_tiles_across_devices(self):
        # 192 rows a device (no 128-row tile divides it): 128-row query tiles
        # whose halves sit on two devices (split_q_tiles), the walk rotated
        # past each device's rows; 700 live rows of 768, the dense walk with
        # its static structure against the static live length, bit for bit,
        # and the reference's rows. Two heads a device (bulk exchange).
        code = """
            tiling = t.dataclasses.replace(t._TILING, split_q_tiles=True,
                                           block_kv_compute_online=16)
            config = t.dataclasses.replace(t._CONFIG, heads=8)
            fitted = tiling.fit(config, 192, 4)
            assert (fitted.block_q, fitted.block_kv) == (128, 64), fitted
            kw = dict(rows=192, heads=8, used=700, norm_scale=(0.3, 1.5),
                      tiling=tiling, narrow_tail=True)
            first = 700
            while (t.mk.loop_structure(first - 1, 192, 4, config, tiling)
                   == t.mk.loop_structure(700, 192, 4, config, tiling)):
                first -= 1
            got, want, raced = t._run(4, races=True, static_structure=first,
                                      split_items=True, **kw)
            static, _, _ = t._run(4, traced=False, split_items=True, **kw)
            print(json.dumps([[*t._errors(got, want),
                               t._same_bits(got, static) and not raced]]))
        """
        self.assert_runs(_in_fresh_process(4, code, timeout=3600), 1, slack=2)

    def test_four_devices_static_structure(self):
        # A traced live length of 857 rows shaped by static_structure=850
        # (both: last query tile 3 with 89 and 82 real rows, last key tile 13
        # cut inside its first fixed-max and second online sub-block; no
        # query tile of padding) is the static 857's program on every device:
        # the same rows, padding included. Two heads a device (bulk exchange).
        code = """
            kw = dict(rows=256, heads=8, used=857, norm_scale=(0.3, 1.5),
                      tiling=t._STRUCTURE_TILING, narrow_tail=True)
            config = t.dataclasses.replace(t._CONFIG, heads=8)
            same = [t.mk.loop_structure(u, 256, 4, config, t._STRUCTURE_TILING)
                    for u in (850, 857)]
            got, want, raced = t._run(4, races=True, static_structure=850, **kw)
            static, _, _ = t._run(4, traced=False, **kw)
            print(json.dumps([[*t._errors(got, want),
                               same[0] == same[1] and t._same_bits(got, static) and not raced]]))
        """
        self.assert_runs(_in_fresh_process(4, code, timeout=3600), 1)


# ----------------------------------------------------- static loop structure


def _kernel_arguments(used, *, traced=True, structure=None, rows=512, **options):
    """The static arguments ``stack`` hands its kernel on one device for a
    live length ``used``, traced (shaped by ``structure``) or a Python int,
    with ``options`` (more ``stack`` arguments). ``pallas_call`` is stubbed:
    nothing is lowered or run."""
    config = dataclasses.replace(_CONFIG, layers=2)
    weights = _weights(config, (0.3, 1.5))
    fixture = _request(config, rows)
    captured = {}

    def pallas_call(kernel, *, out_shape, **unused):
        captured.update(kernel.keywords)
        return lambda *operands: tuple(
            jnp.zeros(shape.shape, shape.dtype) for shape in out_shape
        )

    def local(live):
        extra = {} if structure is None else {"static_structure": structure}
        return mk.stack(
            fixture["x"],
            weights,
            fixture["tables"],
            fixture["onehot"],
            fixture["cos"],
            fixture["sin"],
            live[0] if traced else used,
            config=config,
            tiling=_STRUCTURE_TILING,
            narrow_tail=True,
            **_LOSSLESS,
            **extra,
            **options,
        )

    with patch.object(mk.pl, "pallas_call", pallas_call):
        jax.eval_shape(local, jax.ShapeDtypeStruct((1,), jnp.int32))
    return captured


def _loop_shapes(arguments):
    """What a kernel's loops take from its static ``used``: the narrowed tail
    and cut key tile, and under the dense walk (``static_used``) its last key
    tile, whether ``used`` cuts it and the tail's query tile (``static_item``).
    The masks read the live length from the scalar operand."""
    shapes = {name: arguments[name] for name in ("tail_rows", "cut_subs", "cut_full")}
    walked = arguments["static_used"]
    tiling = arguments["tiling"]
    if walked:
        shapes.update(
            last=(walked + tiling.block_kv - 1) // tiling.block_kv - 1,
            cut=walked % tiling.block_kv != 0,
            tail=(walked - 1) // tiling.block_q,
        )
    others = {name: value for name, value in arguments.items() if name not in shapes}
    others.pop("static_used")
    return shapes, others


@unittest.skipUnless(_HAS_JAX, _SKIP)
class TestStaticStructure(CustomTestCase):
    """``stack(static_structure=S)`` with a traced live length: the loops of
    a static ``used`` that shares S's ``loop_structure``, the masks of the
    live one. One device here; four in TestStackInterpretMultiDevice."""

    def assert_close(self, errors):
        l2, worst = errors
        self.assertLess(l2, _L2)
        self.assertLess(worst, _MAX)

    def test_a_structure_shapes_the_loops_of_a_static_used(self):
        # 512 rows, 256-row query tiles, 64-row key tiles. 289-303 end in the
        # second query tile (no tile of padding: the dense walk), 97-111 in
        # the first (the second is padding: the generic walk); both narrow
        # the tail to 128 rows and cut key tile 4 / 1 inside its second
        # fixed-max and third online sub-block.
        config = dataclasses.replace(_CONFIG, layers=2)

        def structure(used):
            return mk.loop_structure(used, 512, 1, config, _STRUCTURE_TILING)

        for shape, used in ((300, 289), (300, 303), (100, 97), (100, 111)):
            with self.subTest(structure=shape, used=used):
                self.assertEqual(structure(shape), structure(used))
                self.assertEqual(
                    _loop_shapes(_kernel_arguments(used, structure=shape)),
                    _loop_shapes(_kernel_arguments(used, traced=False)),
                )
        dense, _ = _loop_shapes(_kernel_arguments(289, structure=300))
        self.assertEqual(dense["tail_rows"], 128)
        self.assertEqual((dense["last"], dense["cut"], dense["tail"]), (4, True, 1))
        generic, _ = _loop_shapes(_kernel_arguments(97, structure=100))
        self.assertNotIn("last", generic)
        # Off the structure: 288's keys end with key tile 4's first fixed-max
        # sub-block, 305's reach its fourth online one, 321's key tile 5.
        for other in (288, 305, 321):
            self.assertNotEqual(structure(other), structure(300))
        # Without a structure a traced live length takes the generic walk as
        # before: no dense walk, narrowed tail or cut tile.
        plain, _ = _loop_shapes(_kernel_arguments(300))
        self.assertEqual(
            plain, {"tail_rows": 0, "cut_subs": (0, 0), "cut_full": (0, 0)}
        )
        # A static live length ignores a structure.
        self.assertEqual(
            _kernel_arguments(300, traced=False, structure=100),
            _kernel_arguments(300, traced=False),
        )

    def test_refuses_a_structure_off_the_rows(self):
        for shape in (0, 513):
            with self.subTest(structure=shape):
                with self.assertRaisesRegex(ValueError, "static_structure takes"):
                    _kernel_arguments(300, structure=shape)

    def test_traced_live_length_with_a_structure_is_the_static_program(self):
        # Interpret mode, every row (the padding too) bit for bit against the
        # static live length; the race detector on the first case.
        for shape, used, races in (
            (300, 289, True),
            (300, 303, False),
            (100, 111, False),
        ):
            with self.subTest(structure=shape, used=used):
                kwargs = dict(
                    used=used,
                    rows=512,
                    norm_scale=(0.3, 1.5),
                    tiling=_STRUCTURE_TILING,
                    narrow_tail=True,
                )
                got, want, raced = _run(races=races, static_structure=shape, **kwargs)
                static, _, _ = _run(traced=False, **kwargs)
                self.assertFalse(raced)
                self.assert_close(_errors(got, want))
                self.assertTrue(_same_bits(got, static))


def _lever_case(
    cut: int,
    rows: int = 512,
    used: int = 280,
    heads: int = 4,
    devices: int = 1,
    tiling=None,
):
    """Kernel lever cases: the dense walk of a traced live length shaped by
    its first static structure (the hook's call), 256-row query tiles (the
    last narrowed to 128 rows), 64-row key tiles of one fixed-max sub-block
    each (or ``tiling``); layer 0 proven, layer 1 online with local head 1 of
    each device tried at a fixed max (both item loops of a layer run)."""
    tiling = tiling or dataclasses.replace(
        _TILING, block_q=256, block_kv_compute=64, block_kv_compute_online=16
    )
    config = dataclasses.replace(_CONFIG, layers=2, heads=heads)
    extra = {"cut_kv_compute": cut} if cut else {}

    def structure(value):
        return mk.loop_structure(value, rows, devices, config, tiling, **extra)

    first = used
    while structure(first - 1) == structure(used):
        first -= 1
    fixed = np.zeros((2, heads), bool)
    fixed[1, 1 :: heads // devices] = True
    return dict(
        used=used,
        rows=rows,
        heads=heads,
        norm_scale=(0.3, 1.5),
        tiling=tiling,
        narrow_tail=True,
        static_structure=first,
        fixed_heads=jnp.asarray(fixed),
        **extra,
    )


@unittest.skipUnless(_HAS_JAX, _SKIP)
class TestKernelLevers(CustomTestCase):
    """stack's cut_kv_compute and split_items: the same bits as without."""

    def test_cut_keys_halve_while_the_real_keys_fit(self):
        # 2,048-key sub-blocks: the 1344x768x124 structure's 769-896 real keys
        # run 1,024; past 1,024 the sub-block stays whole, under the floor
        # it stops halving.
        self.assertEqual(mk._cut_keys(769, 2048, 256), 1024)
        self.assertEqual(mk._cut_keys(1024, 2048, 256), 1024)
        self.assertEqual(mk._cut_keys(1025, 2048, 256), 0)
        self.assertEqual(mk._cut_keys(200, 2048, 256), 256)
        self.assertEqual(mk._cut_keys(200, 2048, 0), 0)
        # 1,280-key sub-blocks (9,600 rows a device): the second holds the
        # cut, 513-640 real keys of it run 640 keys.
        self.assertEqual(mk._cut_keys(1793, 1280, 256), 640)
        self.assertEqual(mk._cut_keys(1920, 1280, 256), 640)
        self.assertEqual(mk._cut_keys(1921, 1280, 256), 0)
        # One 1,536-key sub-block (4,992 rows a device), 886 real keys; a
        # half off whole 16-row tiles.
        self.assertEqual(mk._cut_keys(886, 1536, 256), 0)
        self.assertEqual(mk._cut_keys(10, 48, 16), 0)
        # The halving can split a plain structure: with whole-tile online
        # sub-blocks 270 and 280 live rows (14 and 24 real keys in the cut
        # tile) share the plain structure, not the lean one.
        config = dataclasses.replace(_CONFIG, layers=2)
        tiling = dataclasses.replace(
            _lever_case(16)["tiling"], block_kv_compute_online=64
        )

        def structure(used, cut=0):
            extra = {"cut_kv_compute": cut} if cut else {}
            return mk.loop_structure(used, 512, 1, config, tiling, **extra)

        self.assertEqual(structure(270), structure(280))
        self.assertEqual((structure(270, 16)[-1], structure(280, 16)[-1]), (16, 32))

    def test_lean_cut_and_split_items_change_no_bit(self):
        # One device, 280 live rows of 512: 24 real keys in cut key tile 4,
        # so lean_cut runs 32 of its 64 keys. Every row, the padding too, bit
        # for bit; the race detector on the levers.
        base, want, _ = _run(**_lever_case(0))
        lean, _, raced = _run(races=True, **_lever_case(16))
        split, _, split_raced = _run(races=True, split_items=True, **_lever_case(16))
        l2, worst = _errors(base, want)
        self.assertLess(l2, _L2)
        self.assertLess(worst, _MAX)
        self.assertFalse(raced or split_raced)
        self.assertTrue(_same_bits(lean, base))
        self.assertTrue(_same_bits(split, base))

    def test_modulate_pieces_change_no_bit(self):
        # Four head groups and 64-row phase A tiles of four 16-row chunks:
        # the next row tile's modulate in one piece, two (the default) or
        # four (a chunk a step). At 512 rows row tiles 1 and 3 share one
        # AdaLN row, so the steps of tiles 0 and 2 modulate them; layer 0 is
        # proven, layer 1 takes the peaks. Every row bit for bit; the race
        # detector on.
        case = dict(rows=512, heads=8, norm_scale=(0.3, 1.5))
        base, want, _ = _run(**case)
        l2, worst = _errors(base, want)
        self.assertLess(l2, _L2)
        self.assertLess(worst, _MAX)
        for pieces in (1, 4):
            with self.subTest(pieces=pieces):
                got, _, raced = _run(races=True, modulate_pieces=pieces, **case)
                self.assertFalse(raced)
                self.assertTrue(_same_bits(got, base))
        # Three pieces do not divide four chunks; four exceed two head groups.
        for heads, pieces in ((8, 3), (4, 4)):
            with self.subTest(heads=heads, pieces=pieces):
                with self.assertRaisesRegex(ValueError, "modulate_pieces"):
                    _run(rows=512, heads=heads, modulate_pieces=pieces)

    def test_epilogue_chunk_changes_no_bit(self):
        # Phase A's pipelined epilogue over 128-row tiles in chunks of 64, 96
        # (and a shorter last one of 32) and 128 rows instead of the tiling's
        # 32, and the peaks layers' epilogue alone in other chunks; layer 0
        # proven, layer 1 takes the peaks. Every row bit for bit; the race
        # detector on. (Chunks under 32 rows change XLA:CPU's own row-sum
        # order in the interpreter, hence none here.)
        case = dict(
            rows=512,
            heads=8,
            norm_scale=(0.3, 1.5),
            tiling=dataclasses.replace(_TILING, qkv_rows=128),
        )
        base, want, _ = _run(**case)
        l2, worst = _errors(base, want)
        self.assertLess(l2, _L2)
        self.assertLess(worst, _MAX)
        for chunks in (
            dict(epilogue_chunk=64),
            dict(epilogue_chunk=96),
            dict(epilogue_chunk=128),
            dict(epilogue_chunk_peaks=96),
            dict(epilogue_chunk=64, epilogue_chunk_peaks=128),
        ):
            with self.subTest(**chunks):
                got, _, raced = _run(races=True, **chunks, **case)
                self.assertFalse(raced)
                self.assertTrue(_same_bits(got, base))
        with self.assertRaisesRegex(ValueError, "epilogue_chunk_peaks"):
            _run(epilogue_chunk_peaks=40, **case)

    def test_zero_ahead_changes_no_bit(self):
        # The dense walk of 280 live rows of 512 (the last query tile
        # narrowed to its real rows: its padding sums one), with and without
        # split items: each item's state zeroed in the item before's
        # epilogue. Every row bit for bit; the race detector on.
        for split in (False, True):
            with self.subTest(split=split):
                case = _lever_case(16)
                base, want, _ = _run(split_items=split, **case)
                l2, worst = _errors(base, want)
                self.assertLess(l2, _L2)
                self.assertLess(worst, _MAX)
                got, _, raced = _run(
                    races=True, split_items=split, zero_ahead=True, **case
                )
                self.assertFalse(raced)
                self.assertTrue(_same_bits(got, base))

    def test_mask_scores_changes_no_bit(self):
        # Masked sub-blocks subtract a per-key column (the fixed max, or 0
        # online, at a visible key; +inf at a masked one) instead of
        # selecting probabilities: the dense walk's cut key tile (280 live
        # rows of 512, with lean_cut; and 1,100 of 1,280 in key groups whose
        # second tile holds the cut), and the generic walk, whose padding
        # query tiles see only the padding keys. Layer 0 proven, layer 1
        # online (each device's head 1 tried at a fixed max in the lever
        # cases). Every row bit for bit; the race detector on.
        grouped = dataclasses.replace(
            _TILING,
            block_q=256,
            block_kv=256,
            block_kv_compute=64,
            block_kv_compute_online=32,
        )
        cases = {
            "cut": dict(split_items=True, **_lever_case(16)),
            "groups": dict(
                split_items=True,
                kv_group=2,
                **_lever_case(16, rows=1280, used=1100, tiling=grouped),
            ),
            "generic": dict(rows=512, used=300, norm_scale=(0.3, 1.5)),
        }
        for name, case in cases.items():
            with self.subTest(case=name):
                base, want, _ = _run(**case)
                l2, worst = _errors(base, want)
                self.assertLess(l2, _L2)
                self.assertLess(worst, _MAX)
                got, _, raced = _run(races=True, mask_scores=True, **case)
                self.assertFalse(raced)
                self.assertTrue(_same_bits(got, base))

    def test_uniform_adaln_changes_no_bit(self):
        # Phase C's 64-row tiles at 512 rows: tiles 0, 1 and 3 read one AdaLN
        # row each (uniform: the MLP input's shift and scale from one lookup),
        # the others a row a lookup; layer 0 proven, layer 1 online. Every
        # row bit for bit; the race detector on. Not with fast_phase_c.
        fixture = _request(dataclasses.replace(_CONFIG, layers=2), 512)
        uniform = np.asarray(mk.uniform_row_tiles(fixture["onehot"], _TILING.mlp_rows))
        self.assertTrue(uniform.any() and not uniform.all())
        case = dict(rows=512, norm_scale=(0.3, 1.5))
        base, want, _ = _run(**case)
        l2, worst = _errors(base, want)
        self.assertLess(l2, _L2)
        self.assertLess(worst, _MAX)
        got, _, raced = _run(races=True, uniform_adaln=True, **case)
        self.assertFalse(raced)
        self.assertTrue(_same_bits(got, base))
        with self.assertRaisesRegex(ValueError, "uniform_adaln"):
            _run(rows=512, uniform_adaln=True, fast_phase_c=True)

    def test_cast_first_step_tables_and_slab_modulate_change_no_bit(self):
        # cast_first: the item epilogue rounds to bf16 before its transpose;
        # step_tables: phase A's step scalars from SMEM tables. 512 rows of
        # 64-row phase A tiles, four head groups (rotated by device on two),
        # layer 0 proven and layer 1 with the peaks. Every row bit for bit;
        # the race detector on.
        case = dict(rows=512, heads=8, norm_scale=(0.3, 1.5))
        base, want, _ = _run(**case)
        l2, worst = _errors(base, want)
        self.assertLess(l2, _L2)
        self.assertLess(worst, _MAX)
        for option in ("cast_first", "step_tables"):
            with self.subTest(option=option):
                got, _, raced = _run(races=True, **{option: True}, **case)
                self.assertFalse(raced)
                self.assertTrue(_same_bits(got, base))
        # slab_modulate: phase C's uniform tiles (tiles 0, 1 and 3 of the
        # kernel tests' 512-row fixture) modulate by modulate_slabs.
        adaln, _, _ = _run(uniform_adaln=True, **case)
        got, _, raced = _run(races=True, uniform_adaln=True, slab_modulate=True, **case)
        self.assertFalse(raced)
        self.assertTrue(_same_bits(got, adaln))
        self.assertTrue(_same_bits(adaln, base))

    def test_key_groups_change_no_bit(self):
        # 256-key tiles of four fixed-max and eight online sub-blocks, walked
        # two at a time (kv_group), with lean_cut and split_items. 800 live
        # rows of 1,024: the last group's second key tile holds the cut;
        # 1,100 of 1,280: its first, and the last group's DMA reaches 256
        # rows past the 1,280 (into the buffer's extra rows: interpret mode
        # fills unwritten memory with NaN and raises on reads out of
        # bounds). Every row bit for bit; the race detector on.
        tiling = dataclasses.replace(
            _TILING,
            block_q=256,
            block_kv=256,
            block_kv_compute=64,
            block_kv_compute_online=32,
        )
        for rows, used, group in (
            (1024, 800, 2),
            (1280, 1100, 2),
            (1280, 1100, (1, 2)),
        ):
            with self.subTest(rows=rows, used=used, group=group):
                case = _lever_case(16, rows=rows, used=used, tiling=tiling)
                base, want, _ = _run(split_items=True, **case)
                grouped, _, raced = _run(
                    races=True, split_items=True, kv_group=group, **case
                )
                l2, worst = _errors(base, want)
                self.assertLess(l2, _L2)
                self.assertLess(worst, _MAX)
                self.assertFalse(raced)
                self.assertTrue(_same_bits(grouped, base))

    def test_two_devices_key_groups_keep_the_bulk_exchange(self):
        # Three heads a device (bulk_exchange), split_items, groups of three
        # 256-key tiles: 1,900 live rows of 2 x 1,024 end 364 keys into the
        # third group, whose DMA reaches 256 rows past the 2,048 (the
        # buffer's extra rows, also the peers' remote-write target).
        code = """
            tiling = t.dataclasses.replace(t._TILING, block_q=256, block_kv=256,
                                           block_kv_compute=64,
                                           block_kv_compute_online=32)
            case = dict(rows=1024, used=1900, heads=6, devices=2, tiling=tiling)
            base, want, _ = t._run(2, split_items=True, **t._lever_case(16, **case))
            grouped, _, raced = t._run(2, races=True, split_items=True, kv_group=3,
                                       **t._lever_case(16, **case))
            print(json.dumps([[*t._errors(base, want),
                               t._same_bits(grouped, base) and not raced]]))
        """
        TestStackInterpretMultiDevice.assert_runs(
            self, _in_fresh_process(2, code, timeout=2400), 1, slack=2
        )

    def test_a_key_tile_off_the_rows(self):
        # 2 x 19,200 against 2 x 19,456 in small: 600 live rows of a 768-row
        # request at 768 rows (96-row phase tiles, 256-row query tiles, the
        # last narrowed) and at 640 (80-row phase tiles, 128-row query
        # tiles), both with 256-key tiles: at 640 rows key_tile keeps them
        # (2.5 a device; kv_divisor would cut 128) over a q, k, v buffer of
        # 768 rows. The reference's rows either way (the other tiles round
        # differently), and two key tiles a step change no bit there.
        tiling = dataclasses.replace(
            _TILING,
            qkv_rows=96,
            mlp_rows=96,
            block_q=256,
            block_kv=256,
            block_kv_compute=64,
            block_kv_compute_online=32,
        )
        base, want, _ = _run(
            split_items=True, **_lever_case(16, rows=768, used=600, tiling=tiling)
        )
        lean = _lever_case(
            16, rows=640, used=600, tiling=dataclasses.replace(tiling, key_tile=256)
        )
        fitted = lean["tiling"].fit(_CONFIG, 640, 1)
        self.assertEqual((fitted.block_q, fitted.block_kv), (128, 256))
        one, lean_want, _ = _run(split_items=True, fixture_rows=768, **lean)
        two, _, raced = _run(
            races=True, split_items=True, kv_group=2, fixture_rows=768, **lean
        )
        for got, ref_rows in ((base, want), (one, lean_want)):
            l2, worst = _errors(got, ref_rows)
            self.assertLess(l2, _L2)
            self.assertLess(worst, _MAX)
        self.assertFalse(raced)
        self.assertTrue(_same_bits(two, one))
        # A key tile off the rows needs the dense walk: 97 live rows of 384
        # leave query tiles of padding.
        with self.assertRaisesRegex(ValueError, "takes the dense walk"):
            _run(
                used=97,
                rows=384,
                layers=1,
                traced=False,
                tiling=dataclasses.replace(_TILING, key_tile=256),
            )

    def test_key_groups_take_only_the_dense_walk(self):
        # The kernel gets the group and the group's cut under the dense walk;
        # the generic walk (a query tile of padding: 100 live rows of 512)
        # keeps one key tile a step and its q, k, v buffer.
        def arguments(used, structure, group):
            return _kernel_arguments(used, structure=structure, kv_group=group)

        dense = arguments(289, 300, 2)
        self.assertEqual(dense["kv_group"], (2, 2))
        # The structure's 300 rows end 44 keys into key group 2 (two 64-key
        # tiles): in its second fixed-max (32-key) and third online (16-key)
        # sub-block.
        self.assertEqual((dense["cut_subs"], dense["cut_full"]), ((2, 3), (1, 2)))
        self.assertEqual(arguments(289, 300, 1)["kv_group"], (1, 1))
        generic = arguments(97, 100, 2)
        self.assertEqual(generic["kv_group"], (1, 1))
        # The online softmax alone: 340 rows end 20 keys into fixed-max key
        # tile 5 (64 keys) and 84 into online key group 2 (128 keys).
        online = arguments(340, 340, (1, 2))
        self.assertEqual(online["kv_group"], (1, 2))
        self.assertEqual((online["cut_subs"], online["cut_full"]), ((1, 6), (0, 5)))
        for bad in (0, (2,), (1, 0)):
            with self.assertRaisesRegex(ValueError, "kv_group takes"):
                arguments(289, 300, bad)

    def test_two_devices_split_items_keep_the_bulk_exchange(self):
        # Three heads a device: bulk_exchange sends heads 1 and 2 during
        # phase B, from the full-width loop under split_items. 280 live rows
        # of 2 x 256: device 1's query tile is the narrowed tail, which the
        # devices' rotated walks place differently.
        code = """
            case = dict(rows=256, used=280, heads=6, devices=2)
            base, want, _ = t._run(2, **t._lever_case(0, **case))
            split, _, raced = t._run(2, races=True, split_items=True,
                                     **t._lever_case(16, **case))
            print(json.dumps([[*t._errors(base, want),
                               t._same_bits(split, base) and not raced]]))
        """
        TestStackInterpretMultiDevice.assert_runs(
            self, _in_fresh_process(2, code, timeout=1800), 1, slack=2
        )


_COMPILE = """
    from jax.experimental import topologies
    from jax.sharding import NamedSharding, PartitionSpec as P
    import jax.numpy as jnp
    import numpy as np
    from sglang_tpu.multimodal_gen.h3_megakernel import kernel as mk
    from sglang_tpu.multimodal_gen.h3_megakernel import reference as ref
    from sglang_tpu.multimodal_gen.h3_megakernel import tiling as tl
    try:
        devices = topologies.get_topology_desc("tpu7x:1x1x1", "tpu").devices
    except Exception as error:
        print("SKIP", type(error).__name__)
        raise SystemExit(0)
    mesh = jax.sharding.Mesh(np.asarray(devices), ("sp",))
    config = ref.Config(layers=1)
    rows = 2 * 2048
    tiling = tl.Tiling(sub_rows=640, stage_slots=24, block_kv_compute_online=1024,
                       mlp_rows=640, chunk_rows=80)
    packed = {"norm1": (1, 1, config.hidden), "norm2": (1, 1, config.hidden),
              "q_norm": (1, 1, 128), "k_norm": (1, 1, 128),
              "wqkv": (1, 28, config.hidden, 768), "wo": (1, 7, config.width, 768),
              "w13": (1, 56, config.hidden, 512), "w2": (1, 21, config.ffn, 256)}
    def spec(shape, partition, dtype=jnp.bfloat16):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, partition))
    def local(x, onehot, cos, sin, tables, used, *weights):
        return mk.stack(x, dict(zip(mk._WEIGHT_ORDER, weights)), tables, onehot, cos, sin,
                        used[0], config=config, tiling=tiling, axis="sp", **t._LOSSLESS)
    shard = P("sp")
    args = [spec((rows, config.hidden), shard), spec((rows, 128), shard),
            spec((rows, 128), shard), spec((rows, 128), shard),
            spec((1, 6, 16, config.hidden), P()), spec((1,), P(), jnp.int32)]
    args += [spec(packed[name], P()) for name in mk._WEIGHT_ORDER]
    jax.jit(jax.shard_map(local, mesh=mesh, check_vma=False,
                          in_specs=(shard,) * 4 + (P(),) * 10,
                          out_specs=shard)).lower(*args).compile()
    print("COMPILED")
"""


@unittest.skipUnless(_HAS_JAX, _SKIP)
class TestCompiledForTpu(CustomTestCase):
    """Mosaic lowers the hook's call, remote DMAs included, for two v7x
    cores at H3's widths (libtpu's compile-only client)."""

    def test_compiles_for_v7x_at_h3_widths(self):
        stdout = _in_fresh_process(1, _COMPILE, timeout=1800)
        if "SKIP" in stdout:
            self.skipTest(f"no compile-only v7x client: {stdout.strip()}")
        self.assertIn("COMPILED", stdout)

    def test_a_structure_lowers_to_the_static_program(self):
        # The Mosaic kernel (without source locations) of a traced live
        # length shaped by static_structure 3,000 or 2,950 is the static
        # 3,010's: 4,096 rows on two v7x cores, 2,048-row query and key
        # tiles, all three ending in query tile 1 with 1,024 tail rows and
        # cutting key tile 1. Without the structure it is another program.
        stdout = _in_fresh_process(1, _LOWER_STRUCTURE, timeout=1800)
        if "SKIP" in stdout:
            self.skipTest(f"no compile-only v7x client: {stdout.strip()}")
        result = json.loads(stdout.splitlines()[-1])
        self.assertEqual(
            len(set(map(tuple, result["structures"]))), 1, result["structures"]
        )
        self.assertTrue(result["shaped_is_static"])
        self.assertTrue(result["one_program_per_structure"])
        self.assertFalse(result["traced_is_static"])


_LOWER_STRUCTURE = r"""
    import base64, re
    from jax._src.interpreters import mlir as jax_mlir
    from jax._src.lib import tpu
    from jax._src.lib.mlir import ir
    from jax.experimental import topologies
    from jax.sharding import NamedSharding, PartitionSpec as P
    import jax.numpy as jnp
    import numpy as np
    from sglang_tpu.multimodal_gen.h3_megakernel import kernel as mk
    from sglang_tpu.multimodal_gen.h3_megakernel import reference as ref
    from sglang_tpu.multimodal_gen.h3_megakernel import tiling as tl
    try:
        devices = topologies.get_topology_desc("tpu7x:1x1x1", "tpu").devices
    except Exception as error:
        print("SKIP", type(error).__name__)
        raise SystemExit(0)
    mesh = jax.sharding.Mesh(np.asarray(devices), ("sp",))
    config = ref.Config(layers=1)
    rows = 2048
    tiling = tl.Tiling(sub_rows=640, stage_slots=24, block_kv_compute_online=1024,
                       mlp_rows=640, chunk_rows=80)
    packed = {"norm1": (1, 1, config.hidden), "norm2": (1, 1, config.hidden),
              "q_norm": (1, 1, 128), "k_norm": (1, 1, 128),
              "wqkv": (1, 28, config.hidden, 768), "wo": (1, 7, config.width, 768),
              "w13": (1, 56, config.hidden, 512), "w2": (1, 21, config.ffn, 256)}
    def spec(shape, partition, dtype=jnp.bfloat16):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, partition))
    shard = P("sp")
    total = rows * len(devices)
    args = [spec((total, config.hidden), shard), spec((total, 128), shard),
            spec((total, 128), shard), spec((total, 128), shard),
            spec((1, 6, 16, config.hidden), P()), spec((1,), P(), jnp.int32)]
    args += [spec(packed[name], P()) for name in mk._WEIGHT_ORDER]
    def kernel_body(static_used=None, structure=None):
        def local(x, onehot, cos, sin, tables, live, *weights):
            extra = {} if structure is None else {"static_structure": structure}
            return mk.stack(x, dict(zip(mk._WEIGHT_ORDER, weights)), tables, onehot, cos, sin,
                            live[0] if static_used is None else static_used, config=config,
                            tiling=tiling, axis="sp", narrow_tail=True, **t._LOSSLESS, **extra)
        text = jax.jit(jax.shard_map(local, mesh=mesh, check_vma=False,
                                     in_specs=(shard,) * 4 + (P(),) * 10,
                                     out_specs=shard)).lower(*args).as_text()
        (body,) = set(re.findall(
            r'(?:\\22|\\"|")body(?:\\22|\\"|"):\s*(?:\\22|\\"|")([A-Za-z0-9+/=]+)', text))
        with jax_mlir.make_ir_context() as context:
            context.allow_unregistered_dialects = True
            tpu.register_dialect(context)
            module = ir.Module.parse(base64.b64decode(body))
            return module.operation.get_asm(enable_debug_info=False)
    static = kernel_body(static_used=3010)
    shaped = kernel_body(structure=3000)
    print(json.dumps({
        "structures": [list(mk.loop_structure(u, rows, len(devices), config, tiling))
                       for u in (2950, 3000, 3010)],
        "shaped_is_static": shaped == static,
        "one_program_per_structure": kernel_body(structure=2950) == shaped,
        "traced_is_static": kernel_body() == static,
    }))
"""


@torch.no_grad() if _HAS_TORCH else (lambda f: f)
def _torch_block(x, block, params, rows, cache, used, config):
    """sglang's MiniMaxH3DiTBlock, eager (non-CUDA) path, bf16 on CPU."""
    bf16 = torch.bfloat16
    heads, dim, ffn = config.heads, config.head_dim, config.ffn
    weight = {k: torch.tensor(v).to(bf16) for k, v in block.items()}

    def rms_norm(value, name, eps):
        norm = torch.nn.RMSNorm(value.shape[-1], eps=eps, dtype=bf16)
        norm.weight.data.copy_(weight[name])
        return norm(value)

    def modulate(value, shift, scale):
        return (value * (1.0 + scale.index_select(0, rows))) + shift.index_select(
            0, rows
        )

    def gate(value, gate_, other):
        return value + gate_.index_select(0, rows) * other

    shift1, scale1, gate1, shift2, scale2, gate2 = params
    # _reorder_grouped_qkv_to_qkv: per-head [q, k, v] rows -> [q; k; v].
    grouped = weight["attn.qkv_proj.weight"].reshape(heads, 3, dim, -1)
    qkv_weight = grouped.transpose(0, 1).reshape(3 * heads * dim, -1)
    h = modulate(rms_norm(x, "norm1.weight", config.eps), shift1, scale1)
    q, k, v = (h @ qkv_weight.T).chunk(3, dim=-1)
    q = rms_norm(q.view(-1, heads, dim), "attn.q_norm.weight", config.qk_eps)
    k = rms_norm(k.view(-1, heads, dim), "attn.k_norm.weight", config.qk_eps)
    half = cache.shape[-1] // 2
    cos = torch.cat((cache[:, :half], cache[:, :half]), dim=-1).unsqueeze(1)
    sin = torch.cat((cache[:, half:], cache[:, half:]), dim=-1).unsqueeze(1)

    def rope(value):
        rot, rest = value[..., : 2 * half], value[..., 2 * half :]
        first, second = rot.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return torch.cat((rot * cos + rotated * sin, rest), dim=-1)

    q, k = rope(q), rope(k)
    # Varlen attention over the documents [0, used) and [used, T).
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * dim**-0.5
    padding = torch.arange(x.shape[0]) >= used
    visible = padding[:, None] == padding[None, :]
    scores = scores.masked_fill(~visible, float("-inf"))
    attn = torch.einsum(
        "hqk,khd->qhd", scores.softmax(-1), v.view(-1, heads, dim).float()
    )
    out = attn.to(bf16).reshape(x.shape[0], -1) @ weight["attn.out_proj.weight"].T
    x = gate(x, gate1, out)
    h = modulate(rms_norm(x, "norm2.weight", config.eps), shift2, scale2)
    gate_up = h @ weight["mlp.fc1.weight"].T
    act = torch.nn.functional.silu(gate_up[:, :ffn]) * gate_up[:, ffn:]
    return gate(x, gate2, act @ weight["mlp.fc2.weight"].T)


if __name__ == "__main__":
    unittest.main()
