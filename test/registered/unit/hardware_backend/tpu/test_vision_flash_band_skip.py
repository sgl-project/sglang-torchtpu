"""SGLANG_TPU_VISION_FLASH_BAND_SKIP and ``band_skip_flash_attention``.

The first three groups run on the JAX CPU device, with Pallas in interpret mode. No
TPU or torch_tpu is needed: the torch_tpu op registration is stubbed so the vision
flash op's JAX body can be called directly.

* Dispatch (no kernel runs): with the flag unset, the op has main's name and calls
  the dense ``flash_attention`` with main's arguments for every shape. With the flag
  set, ``band_skip_flash_attention`` runs exactly at the shapes that have a config
  (q tile 256 at kv_len 3072-12288 needs SGLANG_TPU_VISION_FLASH_BLOCK_Q=256; tile
  128 at 16384), the rest keep the dense call, and the op name gains ``_bs``.
* Op numerics: with the flag set, kv_len 3072 at BLOCK_Q=256 runs band-skip and
  matches a float64 segment-masked softmax reference and the flag-unset (dense)
  output. Shapes without a config give the flag-unset output bit for bit. With the
  flag unset, the output is bit for bit ``flash_attention`` called as main calls it.
* Kernel numerics (float32): kv_len 3072 at q tile 256 (static band widths and the
  pipelined dense body) and kv_len 16384 at tile 128 (the band loop) against the
  reference and the dense kernel (single-step blocks, as main calls it there).
* On a TPU-backed JAX (the TPU CI): the kernel compiled for the TPU, not
  interpreted, at every config in ``_CONFIGS``, bit for bit against the dense kernel
  on the same TPU, in bf16, on layouts that reach every body of each config.

Byte identity with the dense kernel is a TPU claim (it relies on the TPU's
reduction order). On CPU the two kernels differ by float rounding, so the CPU groups
check the masking, the band logic and the dispatch; the TPU group checks the claim.
"""

from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci

register_cpu_ci(est_time=90, suite="base-a-test-cpu")
register_tpu_ci(est_time=150, suite="stage-a-test-tpu-unit", chips=1)

import contextlib
import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

from sglang.test.test_utils import CustomTestCase

_HAS_JAX = importlib.util.find_spec("jax") is not None
_SCALE = 1.0 / 128**0.5
_FLAG = "SGLANG_TPU_VISION_FLASH_BAND_SKIP"


def _has_jax_tpu() -> bool:
    if not _HAS_JAX:
        return False
    try:
        import jax

        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:  # pragma: no cover - no usable JAX backend
        return False


_HAS_JAX_TPU = _has_jax_tpu()


@contextlib.contextmanager
def _flags(block_q=0, max_seq=4096, band_skip=None):
    """Set the vision flash settings; ``band_skip=None`` leaves the flag unset."""
    env = {
        "SGLANG_TPU_VISION_FLASH_BLOCK_Q": str(block_q),
        "SGLANG_TPU_VISION_FLASH_BLOCK_Q_MAX_SEQ": str(max_seq),
    }
    with mock.patch.dict(os.environ, env):
        os.environ.pop(_FLAG, None)
        if band_skip is not None:
            os.environ[_FLAG] = "1" if band_skip else "0"
        yield


class _Op:
    """What the stubbed ``pallas.jax_op`` returns: the op name and its JAX body."""

    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    def register_fake(self, fake):
        self.fake = fake


@contextlib.contextmanager
def _stub_torch_tpu():
    internal = types.ModuleType("torch_tpu._internal")
    internal.pallas = types.SimpleNamespace(jax_op=_Op)
    root = types.ModuleType("torch_tpu")
    root._internal = internal
    stubs = {"torch_tpu": root, "torch_tpu._internal": internal}
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _get_op(block_q=0, max_seq=4096, band_skip=None, fresh=True):
    """``_get_vision_flash_op`` under these settings, from an empty op cache."""
    from sglang_tpu import attention_pallas as ap

    cache = mock.patch.dict(ap._VISION_FLASH_OPS, clear=True)
    with _flags(block_q, max_seq, band_skip), _stub_torch_tpu():
        with cache if fresh else contextlib.nullcontext():
            return ap._get_vision_flash_op(_SCALE)


def _mains_op_name(block_q, max_seq):
    """The op name upstream main gives the vision flash op."""
    name = f"sglang_tpu::vision_flash_s{float(_SCALE)!r}"
    if block_q:
        name += f"_bq{block_q}_max{max_seq}"
    return name.replace(".", "_").replace("-", "m")


@contextlib.contextmanager
def _interpret_on_cpu():
    """Run Pallas kernels in interpret mode on the JAX CPU device."""
    import jax
    from jax.experimental import pallas as pl

    real = pl.pallas_call

    def interpret(*args, **kwargs):
        kwargs["interpret"] = True
        return real(*args, **kwargs)

    with mock.patch.object(pl, "pallas_call", interpret):
        with jax.default_device(jax.devices("cpu")[0]):
            yield


def _segments(kv_len, lens, ids=None):
    """Segment ids: one run per length (ids in order unless given), then a pad
    tail in its own segment."""
    ids = range(len(lens)) if ids is None else ids
    runs = [np.full(n, i, np.int32) for i, n in zip(ids, lens)]
    runs.append(np.full(kv_len - sum(lens), len(lens), np.int32))
    return np.concatenate(runs)


def _cpu(x):
    """``x`` committed to the JAX CPU device, so kernels run there on a TPU host too."""
    import jax

    return jax.device_put(x, jax.devices("cpu")[0])


def _tpu(x):
    """``x`` committed to the first TPU device."""
    import jax

    return jax.device_put(x, next(d for d in jax.devices() if d.platform == "tpu"))


def _qkv(heads, kv_len, dtype, seed, put=_cpu):
    rng = np.random.default_rng(seed)
    shape = (1, heads, kv_len, 128)
    return tuple(
        put(rng.standard_normal(shape, np.float32).astype(dtype)) for _ in range(3)
    )


def _reference(q, k, v, seg):
    """Segment-masked softmax attention in float64, one segment at a time."""
    q, k, v = (np.asarray(x.astype("float32"), np.float64) for x in (q, k, v))
    out = np.zeros_like(q)
    for s in np.unique(seg):
        rows = np.nonzero(seg == s)[0]
        scores = q[:, :, rows] @ k[:, :, rows].swapaxes(-1, -2) * _SCALE
        p = np.exp(scores - scores.max(-1, keepdims=True))
        out[:, :, rows] = (p / p.sum(-1, keepdims=True)) @ v[:, :, rows]
    return out


def _f64(x):
    return np.asarray(x.astype("float32"), np.float64)


def _band_widths(seg, block_q):
    """Chunks in each q block's band (``band_tables``, the kernel's reference)."""
    from sglang_tpu.kernels.band_skip_flash_attention import band_tables

    ids = _cpu(seg[None])
    return set(np.asarray(band_tables(ids, ids, block_q)[1]).tolist())


# One bf16 ulp for |x| in [1, 2).
_BF16_ULP = 2.0**-7

# Layouts (segment ids, expected band widths) per (kv_len, block_q), chosen with
# ``band_tables`` to reach each body of the kernel.
_LAYOUTS = {
    # 6 chunks of 512 keys: bands of 1-4 chunks take the static-width bodies,
    # bands of 5-6 the pipelined dense body.
    (3072, 256): {
        "narrow segments": (_segments(3072, [400] * 7), {1, 2, 3}),
        "one image": (_segments(3072, [2772]), {1, 6}),
        "mixed": (_segments(3072, [1000, 300, 1500]), {1, 2, 3, 4, 5}),
        # Ids in reverse order: a block's [min, max] id range meets chunks
        # that share no id with it, so bands widen (and must stay exact).
        "ids out of order": (
            _segments(3072, [400] * 7, ids=range(6, -1, -1)),
            {1, 3, 4, 5, 6},
        ),
    },
    # Static widths 1-6, pipelined dense body 7-8.
    (4096, 256): {
        "mixed": (_segments(4096, [1000, 300, 1500, 1000]), {1, 2, 3, 4, 5, 6}),
        "ids out of order": (
            _segments(4096, [500] * 8, ids=range(7, -1, -1)),
            {2, 3, 4, 5, 6, 7, 8},
        ),
    },
    # Static widths 1-9, pipelined dense body 10-12.
    (6144, 256): {
        "narrow segments": (_segments(6144, [400] * 15), {1, 2, 3}),
        "mixed": (_segments(6144, [2000, 500, 3000]), {2, 4, 5, 7, 8}),
        "ids out of order": (
            _segments(6144, [700] * 8, ids=range(7, -1, -1)),
            {2, 3, 5, 6, 7, 9, 10, 11, 12},
        ),
    },
    # Band loop (4 chunks per step, then 0-3 leftover chunks) up to 12 chunks,
    # pipelined dense body (no ring) above.
    (8192, 256): {
        "wide bands": (_segments(8192, [3000, 3000, 1500]), {2, 4, 5, 6, 7, 10, 12}),
        "mixed": (_segments(8192, [1500, 2100, 3700, 800]), {2, 3, 6, 8, 9, 13}),
    },
    # Band loop up to 18 chunks, dense body above.
    (12288, 256): {
        "wide bands": (_segments(12288, [3000] * 4), {1, 6, 7, 12, 13}),
        "mixed": (
            _segments(12288, [1700, 2900, 4500, 3000]),
            {4, 6, 7, 9, 10, 15, 16},
        ),
        "large image": (_segments(12288, [9500, 2600]), {6, 19, 24}),
    },
    # The only config the default BLOCK_Q (0, tile 128) reaches. 32 chunks: bands
    # of up to 24 run the band loop, wider ones the dense body.
    (16384, 128): {
        # Bands of 1-2 chunks: no loop step, leftover chunks only.
        "narrow segments": (_segments(16384, [1024] * 15 + [700]), {1, 2}),
        # Bands of 3-13 chunks: 0-3 loop steps, and 0-3 leftover chunks.
        "wide bands": (_segments(16384, [3000] * 5), {3, 6, 7, 9, 12, 13}),
        # Segment 0 split over the first and last chunk: its 8 blocks have
        # 32-chunk bands (dense body), the 1024-token segments between have 2.
        "split segment": (
            np.concatenate(
                [
                    np.zeros(512, np.int32),
                    np.repeat(np.arange(1, 16, dtype=np.int32), 1024),
                    np.zeros(512, np.int32),
                ]
            ),
            {2, 32},
        ),
    },
}


@unittest.skipUnless(_HAS_JAX, "requires JAX")
class TestVisionFlashBandSkipDispatch(CustomTestCase):
    """Which kernel the op's JAX body calls, with stand-in kernels (nothing runs)."""

    HEADS = (16, 3, 2)
    SEQS = (1024, 3072, 4096, 6144, 8192, 12288, 16384, 20480)
    SETTINGS = ((0, 4096), (256, 4096), (256, 12288), (512, 4096))
    # (heads, kv_len) that run band-skip with the flag set, per (BLOCK_Q, MAX_SEQ).
    # Tile 256 at 3072-8192 takes an even head count; 12288 and 16384 take any.
    BAND_SKIP = {
        (0, 4096): {(h, 16384) for h in HEADS},
        (256, 4096): {(16, 3072), (2, 3072), (16, 4096), (2, 4096)}
        | {(h, 16384) for h in HEADS},
        (256, 12288): {(h, s) for h in (16, 2) for s in (3072, 4096, 6144, 8192)}
        | {(h, s) for h in HEADS for s in (12288, 16384)},
        (512, 4096): {(h, 16384) for h in HEADS},
    }

    def _call(self, op, heads, kv_len):
        from sglang_tpu import attention_pallas as ap

        q = types.SimpleNamespace(shape=(1, heads, kv_len, 128))
        with (
            mock.patch.object(ap, "flash_attention", lambda *a, **kw: ("dense", a, kw)),
            mock.patch.object(
                ap,
                "band_skip_flash_attention",
                lambda *a, **kw: ("band_skip", a, kw),
                create=True,
            ),
        ):
            return q, op.fn(q, q, q, "seg")

    def _dense_call(self, q, block_q, max_seq):
        """The dense call as main makes it."""
        from sglang_tpu.attention_pallas import _vision_flash_block_sizes
        from sglang_tpu.kernels.flash_attention import SegmentIds

        kv_len = q.shape[2]
        return (
            "dense",
            (q, q, q),
            dict(
                segment_ids=SegmentIds(q="seg", kv="seg"),
                causal=False,
                sm_scale=_SCALE,
                block_sizes=_vision_flash_block_sizes(block_q, max_seq, kv_len, kv_len),
                vmem_limit_bytes=None,
            ),
        )

    def test_flag_unset_is_mains_dense_call(self):
        for block_q, max_seq in self.SETTINGS:
            for band_skip in (None, False):
                op = _get_op(block_q, max_seq, band_skip)
                self.assertEqual(op.name, _mains_op_name(block_q, max_seq))
                for heads in self.HEADS:
                    for kv_len in self.SEQS:
                        with self.subTest(
                            block_q=block_q,
                            max_seq=max_seq,
                            flag=band_skip,
                            heads=heads,
                            kv_len=kv_len,
                        ):
                            q, got = self._call(op, heads, kv_len)
                            self.assertEqual(got, self._dense_call(q, block_q, max_seq))

    def test_flag_set_runs_band_skip_only_where_configured(self):
        from sglang_tpu.attention_pallas import _vision_flash_block_sizes
        from sglang_tpu.kernels.flash_attention import SegmentIds

        for (block_q, max_seq), expected in self.BAND_SKIP.items():
            op = _get_op(block_q, max_seq, band_skip=True)
            self.assertEqual(op.name, _mains_op_name(block_q, max_seq) + "_bs")
            used = set()
            for heads in self.HEADS:
                for kv_len in self.SEQS:
                    with self.subTest(
                        block_q=block_q, max_seq=max_seq, heads=heads, kv_len=kv_len
                    ):
                        q, got = self._call(op, heads, kv_len)
                        if got[0] == "dense":
                            self.assertEqual(got, self._dense_call(q, block_q, max_seq))
                            continue
                        used.add((heads, kv_len))
                        blocks = _vision_flash_block_sizes(
                            block_q, max_seq, kv_len, kv_len
                        )
                        tile = 128 if blocks is None else blocks.block_q
                        self.assertEqual(
                            got,
                            (
                                "band_skip",
                                (q, q, q, SegmentIds(q="seg", kv="seg")),
                                dict(sm_scale=_SCALE, block_q=tile),
                            ),
                        )
            self.assertEqual(used, expected, (block_q, max_seq))

    def test_flag_set_and_unset_are_separate_cached_ops(self):
        from sglang_tpu import attention_pallas as ap

        with mock.patch.dict(ap._VISION_FLASH_OPS, clear=True):
            off = _get_op(256, 4096, band_skip=None, fresh=False)
            on = _get_op(256, 4096, band_skip=True, fresh=False)
            self.assertIsNot(on, off)
            self.assertEqual(on.name, off.name + "_bs")
            self.assertIs(_get_op(256, 4096, band_skip=False, fresh=False), off)
            self.assertIs(_get_op(256, 4096, band_skip=True, fresh=False), on)
            self.assertEqual(len(ap._VISION_FLASH_OPS), 2)


@unittest.skipUnless(_HAS_JAX, "requires JAX")
class TestVisionFlashBandSkipOpNumerics(CustomTestCase):
    """The op's JAX body, run in Pallas interpret mode on the JAX CPU device."""

    def _run(self, op, q, k, v, seg):
        from sglang_tpu import attention_pallas as ap

        calls = []
        real = getattr(ap, "band_skip_flash_attention", None)

        def spy(*args, **kwargs):
            calls.append(kwargs["block_q"])
            return real(*args, **kwargs)

        with _interpret_on_cpu():
            with mock.patch.object(ap, "band_skip_flash_attention", spy, create=True):
                out = op.fn(q, k, v, _cpu(seg[None]))
            return np.asarray(out), calls

    def test_flag_set_runs_band_skip_and_matches_dense(self):
        import jax.numpy as jnp

        # 2 heads, kv_len 3072, BLOCK_Q=256: the smallest band-skip config.
        seg = _segments(3072, [400] * 7)
        q, k, v = _qkv(2, 3072, jnp.bfloat16, seed=0)
        on, calls = self._run(_get_op(256, 4096, band_skip=True), q, k, v, seg)
        self.assertEqual(calls, [256])
        off, calls = self._run(_get_op(256, 4096), q, k, v, seg)
        self.assertEqual(calls, [])
        ref = _reference(q, k, v, seg)
        np.testing.assert_allclose(_f64(on), ref, atol=1e-2, rtol=0)
        np.testing.assert_allclose(_f64(on), _f64(off), atol=_BF16_ULP, rtol=0)

    def test_flag_set_without_a_config_is_bitwise_dense(self):
        import jax.numpy as jnp

        # (heads, kv_len, BLOCK_Q): no config for kv_len 1024 at tile 256, for
        # kv_len 3072 at the default tile 128, or for an odd head count at 3072.
        for heads, kv_len, block_q in ((2, 1024, 256), (2, 3072, 0), (3, 3072, 256)):
            with self.subTest(heads=heads, kv_len=kv_len, block_q=block_q):
                seg = _segments(kv_len, [kv_len // 3, kv_len // 4])
                q, k, v = _qkv(heads, kv_len, jnp.bfloat16, seed=kv_len + heads)
                on, calls = self._run(_get_op(block_q, 4096, True), q, k, v, seg)
                self.assertEqual(calls, [])
                off, _ = self._run(_get_op(block_q, 4096), q, k, v, seg)
                self.assertTrue(np.array_equal(on, off))

    def test_flag_unset_is_bitwise_mains_dense_kernel(self):
        import jax.numpy as jnp

        from sglang_tpu.attention_pallas import _vision_flash_block_sizes
        from sglang_tpu.kernels.flash_attention import SegmentIds, flash_attention

        for kv_len, block_q in ((1024, 0), (1024, 256), (3072, 0), (3072, 256)):
            with self.subTest(kv_len=kv_len, block_q=block_q):
                seg = _segments(kv_len, [400] * (kv_len // 400))
                q, k, v = _qkv(2, kv_len, jnp.bfloat16, seed=kv_len + block_q)
                out, calls = self._run(_get_op(block_q, 4096), q, k, v, seg)
                self.assertEqual(calls, [])
                ids = _cpu(seg[None])
                with _interpret_on_cpu():
                    main = flash_attention(
                        q,
                        k,
                        v,
                        segment_ids=SegmentIds(q=ids, kv=ids),
                        causal=False,
                        sm_scale=_SCALE,
                        block_sizes=_vision_flash_block_sizes(
                            block_q, 4096, kv_len, kv_len
                        ),
                        vmem_limit_bytes=None,
                    )
                self.assertTrue(np.array_equal(out, np.asarray(main)))


@unittest.skipUnless(_HAS_JAX, "requires JAX")
class TestBandSkipKernelNumerics(CustomTestCase):
    """``band_skip_flash_attention`` in float32 against the reference and the dense
    kernel, on layouts chosen to reach each of its bodies."""

    def _check(self, heads, kv_len, block_q, seg):
        from sglang_tpu.kernels.band_skip_flash_attention import (
            band_skip_flash_attention,
        )
        from sglang_tpu.kernels.flash_attention import (
            BlockSizes,
            SegmentIds,
            flash_attention,
        )

        q, k, v = _qkv(heads, kv_len, np.float32, seed=kv_len)
        ids = _cpu(seg[None])
        segment_ids = SegmentIds(q=ids, kv=ids)
        with _interpret_on_cpu():
            out = _f64(
                band_skip_flash_attention(
                    q, k, v, segment_ids, sm_scale=_SCALE, block_q=block_q
                )
            )
            np.testing.assert_allclose(out, _reference(q, k, v, seg), atol=1e-5, rtol=0)
            # The dense single-step path, which band-skip is meant to match.
            expected = flash_attention(
                q,
                k,
                v,
                segment_ids=segment_ids,
                causal=False,
                sm_scale=_SCALE,
                block_sizes=BlockSizes(block_q, kv_len, kv_len, 1),
                vmem_limit_bytes=None,
            )
            np.testing.assert_allclose(out, _f64(expected), atol=1e-5, rtol=0)

    def test_kv3072_tile256_layouts(self):
        # Static band widths and the pipelined dense body.
        for name, (seg, widths) in _LAYOUTS[(3072, 256)].items():
            with self.subTest(layout=name):
                self.assertEqual(_band_widths(seg, 256), widths)
                self._check(2, 3072, 256, seg)

    def test_kv16384_tile128_layouts(self):
        # The band loop and the dense body.
        for name, (seg, widths) in _LAYOUTS[(16384, 128)].items():
            with self.subTest(layout=name):
                self.assertEqual(_band_widths(seg, 128), widths)
                self._check(1, 16384, 128, seg)


@unittest.skipUnless(_HAS_JAX_TPU, "requires a TPU-backed JAX")
class TestBandSkipKernelOnTpu(CustomTestCase):
    """``band_skip_flash_attention`` compiled for the TPU (a Mosaic custom call, not
    interpret mode), bit for bit against the dense kernel on the same TPU, in bf16,
    at every config in ``_CONFIGS``. The layouts reach every body of each config:
    every static band width and the pipelined dense body (3072-6144); the band loop
    with 0-3 loop steps and 0-3 leftover chunks, and the pipelined dense body without
    the ring (8192); the band loop and the dense body (12288, 16384). Each call has
    two head groups, so the second reads back the bands the first computed."""

    def test_every_config_is_bitwise_dense(self):
        import jax.numpy as jnp

        from sglang_tpu.kernels.band_skip_flash_attention import (
            _CONFIGS,
            band_skip_flash_attention,
        )
        from sglang_tpu.kernels.flash_attention import (
            BlockSizes,
            SegmentIds,
            flash_attention,
        )

        self.assertEqual(set(_LAYOUTS), set(_CONFIGS))
        for (kv_len, block_q), layouts in _LAYOUTS.items():
            heads = 2 * _CONFIGS[(kv_len, block_q)].heads_per_step
            q, k, v = _qkv(heads, kv_len, jnp.bfloat16, seed=kv_len + 1, put=_tpu)
            kwargs = dict(sm_scale=_SCALE, block_q=block_q)
            with self.subTest(kv_len=kv_len, block_q=block_q, check="lowering"):
                ids = _tpu(np.zeros((1, kv_len), np.int32))
                lowered = band_skip_flash_attention.lower(
                    q, k, v, SegmentIds(q=ids, kv=ids), **kwargs
                )
                self.assertIn("tpu_custom_call", lowered.as_text())
            for name, (seg, widths) in layouts.items():
                with self.subTest(kv_len=kv_len, block_q=block_q, layout=name):
                    self.assertEqual(_band_widths(seg, block_q), widths)
                    ids = _tpu(seg[None])
                    segment_ids = SegmentIds(q=ids, kv=ids)
                    out = np.asarray(
                        band_skip_flash_attention(q, k, v, segment_ids, **kwargs)
                    )
                    dense = np.asarray(
                        flash_attention(
                            q,
                            k,
                            v,
                            segment_ids=segment_ids,
                            causal=False,
                            sm_scale=_SCALE,
                            block_sizes=BlockSizes(block_q, kv_len, kv_len, 1),
                            vmem_limit_bytes=None,
                        )
                    )
                    diff = np.abs(_f64(out) - _f64(dense)).max()
                    self.assertTrue(
                        np.array_equal(out.view(np.uint16), dense.view(np.uint16)),
                        f"not bitwise; max |band-skip - dense| = {diff}",
                    )


if __name__ == "__main__":
    unittest.main()
