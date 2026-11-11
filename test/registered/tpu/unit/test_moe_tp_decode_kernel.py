# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""moe_tp_decode (MoE-TP fused decode MoE): admission at GLM-5.3-Flash TP8 shapes,
and the kernel fed by SGLang's top-k output on 8 devices.

The 8-device checks run this file as a script in subprocesses: the CPU worker
because the host device count (XLA_FLAGS) must be set before JAX starts, the
TPU worker so that this process never holds the TPU. The CPU worker uses TPU
interpret mode with the race detector (DMA and semaphore protocols, in-kernel
all-reduce, arithmetic on SGLang top-k output); the TPU worker compiles the
kernel through Mosaic on one tpu7x 2x2x1 host and checks the arithmetic at
GLM-5.3-Flash TP8 shapes, both against an f32 reference with the SwiGLU clamp.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=30,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "glm-5.3-flash", "tp"],
)
# The gate runs the admission checks only (no device, one chip slot); the
# nightly runs the kernel on 8 devices and the interpret-mode race check.
register_tpu_ci(
    est_time=600,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "glm-5.3-flash", "tp"],
)

import contextlib
import glob
import io
import json
import math
import os
import subprocess
import sys
import time
import unittest
from unittest import mock

from sglang.test.ci.gate_matrix import full_matrix
from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental.pallas import tpu as pltpu

    _HAS_JAX = True
except Exception:  # pragma: no cover - environments without JAX
    _HAS_JAX = False

N_RANKS = 8
# One rank's slice of GLM-5.3-Flash at TP8 with MoE-TP (ep_size=1).
GLM53_TP8 = dict(h=4096, e=288, i=256, fs=256)
# Interpret-mode config: h / 8 = 512 gives four lane groups per chunk, so the
# all-reduce fold path exists; 8 experts still refill the 6-slot weight ring.
SMALL = dict(h=4096, e=8, i=128, fs=128, topk=4)
ROUTED_SCALE = 2.5
SWIGLU_LIMIT = 10.0
# (tokens, padded TopK rows among them). Covers padding up to each row bucket
# (5 -> 16, 20 -> 32, 33 -> 64), TopK's -1 rows, and full buckets.
CASES = ((5, 0), (16, 0), (20, 4), (33, 0), (64, 64))
# x scaled so gate/up exceed SWIGLU_LIMIT: the clamp must change the output.
CLAMP_CASE = (16, 0)
CLAMP_X_SCALE = 64.0
# Kernel (FP8 activations, bf16 roundings) vs an f32 reference with dequantized
# weights; vllm-torchtpu#1763 measured rel-L2 0.044 to 0.046 on its inputs.
TOLERANCE = dict(rel_l2=0.06, cos=0.998)
# CLAMP_CASE's x is 64x larger, so the FP8 activation error is too: measured
# rel-L2 0.088 to 0.089, cos 0.996 on TPU and in interpret mode, while the
# unclamped reference is ~43 away, so this still separates clamp from no clamp.
CLAMP_TOLERANCE = dict(rel_l2=0.12, cos=0.995)


def _tpu7x():
    return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


def _shapes(m, h, e, i, fs, *, w_dtype=None, table_experts=None):
    sds = jax.ShapeDtypeStruct
    f32, w = jnp.float32, w_dtype or jnp.float8_e4m3fn
    return (
        sds((m, h), jnp.bfloat16),
        sds((m, table_experts or e), f32),
        sds((e, h, 2 * i), w),
        sds((e, i, h), w),
        sds((e, 1, 1, 2 * i), f32),
        sds((e, 1, 1, h), f32),
        sds((h, 2 * fs), w),
        sds((2 * fs,), f32),
        sds((fs, h), w),
        sds((h,), f32),
    )


def _glm53(rows, **kwargs):
    c = GLM53_TP8
    return _shapes(rows, c["h"], c["e"], c["i"], c["fs"], **kwargs)


@unittest.skipUnless(_HAS_JAX, "requires JAX")
class TestMoeTpDecodeAdmission(CustomTestCase):
    def _can_use(self, info, shapes, n_ranks=N_RANKS):
        from sglang_tpu.moe.pallas.moe_tp_decode import kernel

        with mock.patch.object(pltpu, "get_tpu_info", info):
            return kernel.can_use_moe_tp_decode(*shapes, n_ranks=n_ranks)

    def test_admits_decode_buckets_at_glm53_tp8_shapes(self):
        for rows, admitted in (
            (16, True),
            (32, True),
            (64, True),
            (8, False),
            (48, False),
            (128, False),
        ):
            with self.subTest(rows=rows):
                self.assertIs(self._can_use(_tpu7x, _glm53(rows)), admitted)

    def test_refuses(self):
        v6e = lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1)
        cases = {
            "no-fp8-mxu": (v6e, _glm53(16), N_RANKS),
            "bf16-weights": (_tpu7x, _glm53(16, w_dtype=jnp.bfloat16), N_RANKS),
            "not-every-expert": (_tpu7x, _glm53(16, table_experts=128), N_RANKS),
            "one-rank": (_tpu7x, _glm53(16), 1),
        }
        for name, (info, shapes, n_ranks) in cases.items():
            with self.subTest(name):
                self.assertFalse(self._can_use(info, shapes, n_ranks))


def _has_tpu():
    # Device files, not jax.devices(): initializing JAX here would claim the TPU
    # that the worker subprocess needs.
    return bool(glob.glob("/dev/vfio/[0-9]*") or glob.glob("/dev/accel*"))


def _run_worker(mode="worker"):
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(os.environ)
    if mode == "worker":
        env.update(
            JAX_PLATFORMS="cpu",
            # bf16 rounded after every op in the interpreted kernel too, as on TPU.
            XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={N_RANKS}"
            " --xla_allow_excess_precision=false",
        )
    proc = subprocess.run(
        [sys.executable, __file__, mode],
        env=env,
        capture_output=True,
        text=True,
        timeout=1500,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


def _assert_case(test, case, tolerance):
    test.assertTrue(case["ranks_identical"], case)
    test.assertTrue(case["padding_rows_positive_zero"], case)
    test.assertFalse(case["races_found"], case)
    test.assertEqual(case["nan"], 0, case)
    if "rel_l2" in case:
        test.assertLess(case["rel_l2"], tolerance["rel_l2"], case)
        test.assertGreaterEqual(case["cos"], tolerance["cos"], case)
    if "unclamped_vs_clamped_rel_l2" in case:
        # The clamp changes the result on these inputs, so the tolerance check
        # above would fail if the kernel ignored it.
        test.assertGreater(case["unclamped_vs_clamped_rel_l2"], 0.1, case)


def _tolerance(name):
    return CLAMP_TOLERANCE if name.endswith("clamp") else TOLERANCE


@unittest.skipUnless(_HAS_JAX, "requires JAX")
@unittest.skipUnless(full_matrix(), "nightly only: interpret mode is slow on the gate")
class TestMoeTpDecodeOn8Devices(CustomTestCase):
    result = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.result = _run_worker()

    def test_race_detector_is_live(self):
        self.assertEqual(self.result["race_control"], {"racy": True, "clean": False})

    def test_kernel_with_topk_output(self):
        for name, case in self.result["cases"].items():
            with self.subTest(name):
                _assert_case(self, case, _tolerance(name))


@unittest.skipUnless(_HAS_JAX and _has_tpu(), "requires a TPU host")
@unittest.skipUnless(full_matrix(), "nightly only: needs all 8 devices of the host")
class TestMoeTpDecodeOnTpu(CustomTestCase):
    result = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.result = _run_worker("worker-tpu")

    def test_kernel_on_tpu(self):
        for name, case in self.result["cases"].items():
            with self.subTest(name):
                _assert_case(self, case, _tolerance(name))


def _fp8_columns(w):
    """Per-output-channel FP8 of [..., K, N]: (fp8, f32 scale [..., N])."""
    f8 = jnp.float8_e4m3fn
    s = jnp.max(jnp.abs(w), axis=-2) / float(jnp.finfo(f8).max)
    s = jnp.where(s == 0, 1.0, s)
    return (w / s[..., None, :]).astype(f8), s


def _rank_weights(key, c):
    h, e, i, fs = c["h"], c["e"], c["i"], c["fs"]
    k13, k2, ks13, ks2 = jax.random.split(key, 4)
    f32 = jnp.float32
    w13, s13 = _fp8_columns(jax.random.normal(k13, (e, h, 2 * i), f32) / math.sqrt(h))
    w2, s2 = _fp8_columns(jax.random.normal(k2, (e, i, h), f32) / math.sqrt(16 * i))
    sw13, ss13 = _fp8_columns(jax.random.normal(ks13, (h, 2 * fs), f32) / math.sqrt(h))
    sw2, ss2 = _fp8_columns(jax.random.normal(ks2, (fs, h), f32) / math.sqrt(16 * fs))
    return (
        w13,
        w2,
        s13.reshape(e, 1, 1, 2 * i),
        s2.reshape(e, 1, 1, h),
        sw13,
        ss13,
        sw2,
        ss2,
    )


def _swiglu_ref(gate, up, limit):
    if limit is not None:
        gate = jnp.minimum(gate, limit)
        up = jnp.clip(up, -limit, limit)
    return jax.nn.silu(gate) * up


def _reference(x, ids, w, ranks, c, limit):
    """f32 sum over ranks of shared + scale * routed (dequantized weights);
    rows with no live id are 0."""
    f32 = jnp.float32
    x = x.astype(f32)
    valid = ids >= 0
    w = jnp.where(valid, w, 0.0)
    coef = jnp.einsum("nk,nke->ne", w, jax.nn.one_hot(ids, c["e"], dtype=f32))
    out = jnp.zeros(x.shape, f32)
    for w13, w2, s13, s2, sw13, ss13, sw2, ss2 in ranks:
        gu = jnp.einsum("nh,ehk->enk", x, w13.astype(f32) * s13[:, 0])
        act = _swiglu_ref(gu[..., : c["i"]], gu[..., c["i"] :], limit)
        y = jnp.einsum("eni,eih->enh", act, w2.astype(f32) * s2[:, 0])
        routed = jnp.einsum("ne,enh->nh", coef, y)
        sgu = x @ (sw13.astype(f32) * ss13)
        sact = _swiglu_ref(sgu[:, : c["fs"]], sgu[:, c["fs"] :], limit)
        out = out + sact @ (sw2.astype(f32) * ss2) + ROUTED_SCALE * routed
    return np.asarray(jnp.where(jnp.any(valid, axis=1, keepdims=True), out, 0.0))


def _rel_l2(got, want):
    got, want = got.astype(np.float64), want.astype(np.float64)
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


def _interpret(fn, *args):
    """fn(*args) with fresh interpreter state; (out, whether races were found)."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    pltpu.reset_tpu_interpret_mode_state()
    with contextlib.redirect_stdout(io.StringIO()):
        out = np.asarray(jax.block_until_ready(fn(*args)))
    return out, bool(interpret_pallas_call.races.races_found)


def _race_control():
    """The detector on a kernel that reads a DMA target before its wait."""
    from jax.experimental import pallas as pl

    def make(read_early):
        def body(x_hbm, o_ref, buf, sem):
            copy = pltpu.make_async_copy(x_hbm, buf, sem)
            copy.start()
            if read_early:
                o_ref[...] = buf[...]
            copy.wait()
            if not read_early:
                o_ref[...] = buf[...]

        return pl.pallas_call(
            body,
            out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                pltpu.VMEM((8, 128), jnp.float32),
                pltpu.SemaphoreType.DMA(()),
            ],
        )

    x = jnp.ones((8, 128), jnp.float32)
    return {
        "racy": _interpret(make(True), x)[1],
        "clean": _interpret(make(False), x)[1],
    }


# x, top-k ids and weights are the same on every rank (replicated).
def _specs():
    from jax.sharding import PartitionSpec as P

    return (P(), P(), P(), P("d")) + (P("d"),) * 8


def _build_run(mesh, c):
    """The kernel fed by SGLang top-k output, per rank under shard_map."""
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.moe.pallas.moe_tp_decode import kernel
    from sglang_tpu.moe.pallas.moe_tp_decode.routing import kernel_inputs

    def body(x, ids, w, rank, *weights):
        m = x.shape[0]
        xp, wtab, n_valid = kernel_inputs(x, ids, w, c["e"])
        out = kernel.moe_tp_decode(
            xp,
            wtab,
            n_valid,
            rank.reshape(()),
            *weights,
            routed_scale=ROUTED_SCALE,
            swiglu_limit=SWIGLU_LIMIT,
            n_ranks=N_RANKS,
        )
        return out[:m]

    return jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=_specs(), out_specs=P("d"), check_vma=False
        )
    )


def _build_layer(mesh, c):
    """Per-rank FP8 weights (and their mesh-sharded stack) plus the router."""
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    keys = jax.random.split(jax.random.key(0), N_RANKS + 3)
    ranks = [_rank_weights(keys[r], c) for r in range(N_RANKS)]
    stacked = [
        jax.device_put(jnp.concatenate(ws, axis=0), NamedSharding(mesh, P("d")))
        for ws in zip(*ranks, strict=True)
    ]
    router = dict(
        gate=jax.random.normal(keys[-1], (c["e"], c["h"])) / math.sqrt(c["h"]),
        bias=jax.random.uniform(keys[-2], (c["e"],), jnp.float32, -0.05, 0.05),
        x_key=keys[-3],
    )
    return ranks, stacked, router


def _topk_output(router, c, tokens, padded, x_scale=1.0):
    """SGLang-style top-k output; TopK's padded rows have id -1 and NaN weights
    (which must not reach the routing table)."""
    x = (jax.random.normal(router["x_key"], (tokens, c["h"])) * x_scale).astype(
        jnp.bfloat16
    )
    scores = jax.nn.sigmoid(x.astype(jnp.float32) @ router["gate"].T)
    _, ids = jax.lax.top_k(scores + router["bias"], c["topk"])
    w = jnp.take_along_axis(scores, ids, axis=-1)
    w = w / jnp.sum(w, axis=-1, keepdims=True)
    if padded:
        real = tokens - padded
        ids = ids.at[real:].set(-1)
        w = w.at[real:].set(jnp.nan)
    return x, ids.astype(jnp.int32), w.astype(jnp.float32)


def _evaluate(run, mesh, layer, c, tokens, padded, on_tpu, x_scale=1.0):
    from jax.sharding import NamedSharding

    ranks, stacked, router = layer
    x, ids, w = _topk_output(router, c, tokens, padded, x_scale)
    rank_ids = jnp.arange(N_RANKS, dtype=jnp.int32).reshape(N_RANKS, 1)
    small = [
        jax.device_put(a, NamedSharding(mesh, s))
        for a, s in zip((x, ids, w, rank_ids), _specs()[:4], strict=True)
    ]
    start = time.monotonic()
    if on_tpu:
        out, races = np.asarray(jax.block_until_ready(run(*small, *stacked))), False
    else:
        out, races = _interpret(run, *small, *stacked)
    out = out.reshape(N_RANKS, tokens, c["h"]).astype(np.float32)
    bits, real = out.view(np.uint32), tokens - padded
    pad = out[:, real:]
    case = dict(
        seconds=round(time.monotonic() - start, 1),
        ranks_identical=bool((bits == bits[0]).all()),
        padding_rows_positive_zero=bool(np.all((pad == 0) & ~np.signbit(pad))),
        races_found=races,
        nan=int(np.isnan(out).sum()),
    )
    if real:
        got = out[0, :real].astype(np.float64)
        want = _reference(x, ids, w, ranks, c, SWIGLU_LIMIT)[:real]
        cos = np.vdot(got, want) / (np.linalg.norm(got) * np.linalg.norm(want))
        case.update(rel_l2=_rel_l2(got, want), cos=float(cos))
        if x_scale != 1.0:
            unclamped = _reference(x, ids, w, ranks, c, None)[:real]
            case["unclamped_vs_clamped_rel_l2"] = _rel_l2(unclamped, want)
    return case


def _run_config(mesh, c, on_tpu):
    run, layer = _build_run(mesh, c), _build_layer(mesh, c)
    cases = {}
    for tokens, padded in CASES:
        name = f"{tokens}:{padded}"
        cases[name] = _evaluate(run, mesh, layer, c, tokens, padded, on_tpu)
        print(f"CASE {name} {json.dumps(cases[name])}", flush=True)
    cases["clamp"] = _evaluate(
        run, mesh, layer, c, *CLAMP_CASE, on_tpu, x_scale=CLAMP_X_SCALE
    )
    print(f"CASE clamp {json.dumps(cases['clamp'])}", flush=True)
    return cases


def _worker(on_tpu=False):
    from jax.sharding import Mesh

    if on_tpu:
        devices, c = jax.devices(), dict(GLM53_TP8, topk=8)
    else:
        from jax._src.pallas.mosaic import tpu_info

        pltpu.get_tpu_info = _tpu7x
        tpu_info.get_tpu_info = _tpu7x
        devices, c = jax.devices("cpu"), dict(SMALL)
    devices = sorted(devices, key=lambda d: d.id)
    assert len(devices) == N_RANKS, devices
    mesh = Mesh(np.asarray(devices), ("d",))
    result = {"cases": _run_config(mesh, c, on_tpu)}
    if not on_tpu:
        result["race_control"] = _race_control()
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["worker"]:
        with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
            _worker()
    elif sys.argv[1:] == ["worker-tpu"]:
        _worker(on_tpu=True)
    else:
        unittest.main()
