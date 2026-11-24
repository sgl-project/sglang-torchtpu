# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the MiniMax-H3 megakernel's vouched heads
(``op.local_vouched_stack``, ``SGLANG_TPU_OPT_H3_MEGAKERNEL_VOUCH``).

A step tries at a fixed max the heads its request's previous step vouched
for, checks its own peaks on device and, when a tried head left the
fixed-max bound, recomputes itself with the proven layers only. The kernel
runs in Pallas TPU interpret mode on the kernel tests' small model with three
layers: layer 0's q/k norm weights prove its bound; layer 1's do not, but its
peaks keep every score inside it; layer 2's scores reach ~170 log2 units,
where ``exp2(s - 17)`` overflows f32.

- A step that tries layer 1's heads returns the kernel's result with those
  heads at a fixed max, bit for bit, and vouches for them again.
- A step that also tries layer 2's (whose unguarded kernel result is not
  finite) returns the proven-only result, bit for bit.
- No heads to try is the plain op's call, bit for bit.
- The same on two devices with the race detector, the recomputation too.
- The call differs from the plain op's only in ``fixed_heads`` and
  ``return_peaks``; lowered for v7x, both ops hold the same Mosaic kernel,
  once (libtpu's compile-only client).
- With the static-structure option too (``op.stack_kwargs``), each branch
  is the structured kernel's call bit for bit (no heads to try: the
  structured op's), on one device and on two with the race detector, and
  the vouched program holds the structured op's Mosaic kernel, once.

Skipped where JAX is unavailable.
"""

import dataclasses
import functools
import json
import os
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1400, suite="base-a-test-cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, *[os.pardir] * 3))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import jax
    import jax.numpy as jnp

    # The kernel tests' small model, request and tiling.
    import test_tpu_h3_megakernel_kernel as kt
    from jax import lax
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.multimodal_gen.h3_megakernel import kernel as mk
    from sglang_tpu.multimodal_gen.h3_megakernel import op
    from sglang_tpu.multimodal_gen.h3_megakernel import reference as ref

    _HAS_JAX = kt._HAS_JAX
except Exception:  # pragma: no cover - depends on JAX availability
    _HAS_JAX = False

_SKIP = "Requires JAX with Pallas TPU interpret mode"

# Per layer, the factor on the q/k norm weights: 0 is proven by them (bound
# 21 log2 units), 1 is not (151) though its peaks bound its scores by ~90,
# 2 is not (1,572) and its scores reach ~170.
_SCALES = (0.3, 1.25, 3.0)
_MARGIN = 1.1


@functools.lru_cache(maxsize=None)
def _case(sp: int):
    """The small model on ``sp`` devices, 256 rows each, all but 37 real:
    (config, mesh, operands, weights, used, reference rows)."""
    config = dataclasses.replace(kt._CONFIG, layers=len(_SCALES))
    weights = kt._weights(config, _SCALES)
    total = 256 * sp
    used = total - 37
    fixture = kt._request(config, total)
    operands = tuple(fixture[name] for name in ("x", "onehot", "cos", "sin"))
    operands += (fixture["tables"], jnp.asarray([used], jnp.int32))
    reference = np.asarray(
        ref.reference_stack(
            fixture["x"],
            weights,
            fixture["tables"],
            *operands[1:4],
            used,
            config,
        ),
        np.float32,
    )[:used]
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:sp]), ("sp",))
    return config, mesh, operands, weights, used, reference


def _interpret(races: bool):
    return pltpu.InterpretParams(
        detect_races=races, dma_execution_mode="eager" if races else "on_wait"
    )


def _weight_list(weights):
    return tuple(weights[name] for name in op.pack.WEIGHT_ORDER)


def _structure(sp: int) -> int:
    """The static structure the hook passes for ``_case(sp)``'s live length:
    the first live length of its loop structure at the tests' tiling (193 for
    219 live rows on one device, 449 for 475 on two)."""
    config, _, _, _, used, _ = _case(sp)

    def shape(value):
        return mk.loop_structure(value, 256, sp, config, kt._TILING)

    first = used
    while first > 1 and shape(first - 1) == shape(used):
        first -= 1
    return first


def _options(sp: int, structure=None) -> dict:
    """The op's kernel options: the plain op's, or with ``structure`` the
    structured op's (``op.stack_kwargs``)."""
    return op.STACK_KWARGS if structure is None else op.stack_kwargs(sp, structure)


def _heads(config, layers=()) -> np.ndarray:
    """``[L, heads]`` int32: every head of ``layers``."""
    heads = np.zeros((config.layers, config.heads), np.int32)
    heads[list(layers)] = 1
    return heads


@functools.lru_cache(maxsize=None)
def _vouched_fn(sp, margin, races, structure=None):
    """The vouched op's JAX function, jitted once a configuration;
    ``structure``: with the structured op's kernel options."""
    config, mesh = _case(sp)[:2]
    return jax.jit(
        op.sharded_vouched_stack(
            mesh,
            config,
            kt._TILING,
            margin,
            _interpret(races),
            options=_options(sp, structure),
        )
    )


def _vouched(sp, vouch, tally=(0, 0), margin=_MARGIN, races=False, structure=None):
    """The vouched op's JAX function: (rows, next heads, tally) as numpy."""
    _, _, operands, weights, _, _ = _case(sp)
    out, heads, counts = _vouched_fn(sp, margin, races, structure)(
        *operands,
        jnp.asarray(vouch, jnp.int32),
        jnp.asarray(tally, jnp.int32),
        *_weight_list(weights),
    )
    return np.asarray(out), np.asarray(heads), np.asarray(counts)


@functools.lru_cache(maxsize=None)
def _plain_fn(sp, races, structure=None):
    """The plain op's kernel call with ``fixed_heads`` as an operand (all
    false: the plain op's own call, bit for bit) and the peaks maxed over
    devices, jitted once a configuration; ``structure``: the structured
    op's call."""
    config, mesh = _case(sp)[:2]

    def local(x, onehot, cos, sin, tables, used, fixed, *flat):
        out, peaks = mk.stack(
            x,
            dict(zip(op.pack.WEIGHT_ORDER, flat)),
            tables,
            onehot,
            cos,
            sin,
            used[0],
            config=config,
            tiling=kt._TILING,
            axis="sp",
            interpret=_interpret(races),
            **dict(_options(sp, structure), fixed_heads=fixed, return_peaks=True),
        )
        return out, lax.pmax(peaks, "sp")

    shard = P("sp")
    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(shard,) * 4 + (P(),) * (3 + len(op.pack.WEIGHT_ORDER)),
            out_specs=(shard, P()),
            check_vma=False,
        )
    )


def _plain(sp, fixed_heads=None, races=False, structure=None):
    """The kernel's result with ``fixed_heads`` at a fixed max besides the
    proven layers, and its peaks: (rows, peaks) as numpy."""
    config, _, operands, weights, _, _ = _case(sp)
    if fixed_heads is None:
        fixed_heads = _heads(config)
    out, peaks = _plain_fn(sp, races, structure)(
        *operands, jnp.asarray(fixed_heads, jnp.bool_), *_weight_list(weights)
    )
    return np.asarray(out), np.asarray(peaks)


def _plain_op(sp, structure=None):
    """The plain op's JAX function (``op.sharded_stack``), as served;
    ``structure``: the structured op's for that static structure."""
    config, mesh, operands, weights, _, _ = _case(sp)
    fn = op.sharded_stack(
        mesh, config, kt._TILING, _interpret(False), options=_options(sp, structure)
    )
    return np.asarray(jax.jit(fn)(*operands, *_weight_list(weights)))


def _vouched_by(sp, peaks, margin=_MARGIN) -> np.ndarray:
    """What a step whose accepted peaks are ``peaks`` vouches for next."""
    config, _, _, weights, _, _ = _case(sp)
    unproven = ~np.asarray(mk.fixed_max_layers(weights, config))[:, None]
    safe = np.asarray(mk.safe_heads(jnp.asarray(peaks), margin))[:, : config.heads]
    return (safe & unproven).astype(np.int32)


def _vouch_checks(sp: int, races: bool, structure=None) -> dict:
    """Both branches of a step on ``sp`` devices, as JSON-able results;
    ``structure``: both ops with the static-structure option's options."""
    config, _, _, _, used, reference = _case(sp)
    run = dict(races=races, structure=structure)
    proven, proven_peaks = _plain(sp, **run)
    inside, outside = _heads(config, (1,)), _heads(config, (1, 2))
    out, heads, tally = _vouched(sp, inside, **run)
    want, want_peaks = _plain(sp, inside, **run)
    redo, redo_heads, redo_tally = _vouched(sp, outside, tally=(2, 5), **run)
    return {
        "vouched_by_proven": _vouched_by(sp, proven_peaks).tolist(),
        "inside_same": kt._same_bits(out, want),
        "inside_differs_from_proven": not kt._same_bits(want, proven),
        "inside_next": heads.tolist(),
        "inside_next_want": _vouched_by(sp, want_peaks).tolist(),
        "inside_tally": tally.tolist(),
        "inside_errors": list(kt._errors(out[:used], reference)),
        "proven_errors": list(kt._errors(proven[:used], reference)),
        "redo_same": kt._same_bits(redo, proven),
        "redo_next": redo_heads.tolist(),
        "redo_next_want": _vouched_by(sp, proven_peaks).tolist(),
        "redo_tally": redo_tally.tolist(),
        "races": kt._races_found(),
    }


def _in_fresh_process(devices: int, code: str, timeout: int) -> str:
    """Runs ``code`` (this module imported as ``t``) with ``devices`` CPU
    devices; returns its stdout."""
    prelude = textwrap.dedent(
        f"""
        import os, sys
        sys.path.append({_REPO_ROOT!r})
        sys.path.insert(0, {_HERE!r})
        import types, unittest
        try:
            import sglang.test.ci.ci_register, sglang.test.test_utils  # noqa: F401
        except Exception:  # only this module's helpers are needed here
            for name in ("sglang", "sglang.test", "sglang.test.ci"):
                sys.modules.setdefault(name, types.ModuleType(name))
            ci = types.ModuleType("sglang.test.ci.ci_register")
            ci.register_cpu_ci = lambda *a, **k: None
            tu = types.ModuleType("sglang.test.test_utils")
            tu.CustomTestCase = unittest.TestCase
            sys.modules["sglang.test.ci.ci_register"] = ci
            sys.modules["sglang.test.test_utils"] = tu
        import json, jax
        jax.config.update("jax_platforms", "cpu")
        import test_tpu_h3_megakernel_vouch as t
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


def _assert_checks(test, checks: dict, config) -> None:
    """``_vouch_checks``'s results, as each branch must give them."""
    inside = _heads(config, (1,)).tolist()
    # The proven-only step's peaks vouch for layer 1's heads, not 2's.
    test.assertEqual(checks["vouched_by_proven"], inside)
    # Tried heads inside the bound: the kernel with them at a fixed max (a
    # different rounding from the proven-only call), vouched for again by
    # this step's own peaks.
    test.assertTrue(checks["inside_same"])
    test.assertTrue(checks["inside_differs_from_proven"])
    test.assertEqual(checks["inside_next"], checks["inside_next_want"])
    test.assertEqual(checks["inside_next"], inside)
    test.assertEqual(checks["inside_tally"], [0, config.heads])
    # As close to the f32 reference as the proven-only kernel (the E7
    # tests' 1.25x floor ratio): layer 2's ~170-unit scores make its
    # attention near one-hot, which magnifies any kernel's rounding past the
    # kernel tests' absolute bounds.
    (l2, worst), (proven_l2, proven_worst) = (
        checks["inside_errors"],
        checks["proven_errors"],
    )
    test.assertLess(l2, 1.25 * proven_l2)
    test.assertLess(worst, 1.25 * proven_worst)
    test.assertLess(proven_l2, 5 * kt._L2)
    # A tried head outside it: the proven-only call, bit for bit, one
    # recomputation counted, the next heads from the proven-only peaks.
    test.assertTrue(checks["redo_same"])
    test.assertEqual(checks["redo_next"], checks["redo_next_want"])
    test.assertEqual(checks["redo_tally"], [3, 5])
    test.assertFalse(checks["races"])


# --------------------------------------------------------------------- tests


@unittest.skipUnless(_HAS_JAX, _SKIP)
class TestVouchedStep(CustomTestCase):
    """One device: each branch of a step is a kernel call bit for bit."""

    def test_each_branch_is_a_kernel_call_bit_for_bit(self):
        _assert_checks(self, _vouch_checks(1, races=False), _case(1)[0])

    def test_the_heads_outside_the_bound_break_an_unguarded_fixed_max(self):
        # The negative control for the recomputation: with layer 2's heads
        # at a fixed max the kernel overflows (or lands far off).
        config, _, _, _, used, reference = _case(1)
        unguarded, peaks = _plain(1, _heads(config, (1, 2)))
        bounds = np.asarray(mk.layer_bounds(jnp.asarray(peaks)))[:, : config.heads]
        self.assertTrue((bounds[2] * mk._BOUND_SLACK > mk._FIXED_BOUND).all())
        l2, _ = kt._errors(unguarded[:used], reference)
        self.assertGreater(l2, 10 * kt._L2)

    def test_no_heads_to_try_is_the_plain_op(self):
        config = _case(1)[0]
        none = _heads(config)
        out, heads, tally = _vouched(1, none)
        self.assertTrue(kt._same_bits(out, _plain_op(1)))
        self.assertEqual(heads.tolist(), _heads(config, (1,)).tolist())
        self.assertEqual(tally.tolist(), [0, 0])
        # Margin 0 vouches for every head of the layers the norm weights
        # leave online (the recomputation's check); norm-proven ones never.
        _, every, _ = _vouched(1, none, margin=0.0)
        self.assertEqual(every.tolist(), _heads(config, (1, 2)).tolist())
        # Heads of norm-proven layers are ignored: nothing to try.
        out, _, tally = _vouched(1, _heads(config, (0,)))
        self.assertTrue(kt._same_bits(out, _plain_op(1)))
        self.assertEqual(tally.tolist(), [0, 0])

    def test_the_vouched_call_is_pinned(self):
        # The kernel call of each attempt is the plain op's (STACK_KWARGS)
        # with the heads to try and the peaks: nothing else changes.
        config, _, operands, weights, _, _ = _case(1)
        calls = []

        def stack(x, named, tables, onehot, cos, sin, used, **kwargs):
            calls.append((list(named), kwargs))
            return x, jnp.zeros((config.layers, 2, 8, mk.TABLE_LANES), jnp.float32)

        fn = functools.partial(
            op.local_vouched_stack,
            config=config,
            tiling=kt._TILING,
            axis=None,
            margin=_MARGIN,
        )
        with patch.object(op.kernel, "stack", stack):
            jax.make_jaxpr(fn)(
                *operands,
                jnp.zeros((config.layers, config.heads), jnp.int32),
                jnp.zeros(2, jnp.int32),
                *_weight_list(weights),
            )
        ((names, kwargs),) = calls  # one kernel call site, in the loop body
        self.assertEqual(names, list(op.pack.WEIGHT_ORDER))
        fixed = kwargs.pop("fixed_heads")
        self.assertEqual((fixed.shape, fixed.dtype), ((3, 4), jnp.bool_))
        want = dict(op.STACK_KWARGS, return_peaks=True)
        del want["fixed_heads"]
        want.update(config=config, tiling=kt._TILING, axis=None, interpret=False)
        self.assertEqual(kwargs, want)


@unittest.skipUnless(_HAS_JAX, _SKIP)
class TestVouchedStructuredStep(CustomTestCase):
    """Both options, one device: the vouched op's kernel call takes the
    structured op's options, so each branch of a step is the structured
    kernel's call bit for bit. The static structure (193) shapes the loops
    of the traced live length (219): the dense walk, the last key tile cut
    in its first fixed-max sub-block."""

    def test_each_branch_is_the_structured_kernel_call_bit_for_bit(self):
        checks = _vouch_checks(1, races=False, structure=_structure(1))
        _assert_checks(self, checks, _case(1)[0])

    def test_no_heads_to_try_is_the_structured_op(self):
        config, _, _, _, used, _ = _case(1)
        structure = _structure(1)
        self.assertEqual((structure, used), (193, 219))
        out, heads, tally = _vouched(1, _heads(config), structure=structure)
        self.assertTrue(kt._same_bits(out, _plain_op(1, structure)))
        self.assertEqual(heads.tolist(), _heads(config, (1,)).tolist())
        self.assertEqual(tally.tolist(), [0, 0])

    def test_the_vouched_call_takes_the_structured_options(self):
        # Each attempt's kernel call is the structured op's (stack_kwargs; at
        # four devices', without the down-weight prefetch) with the heads to
        # try and the peaks.
        config, _, operands, weights, _, _ = _case(1)
        calls = []

        def stack(x, named, tables, onehot, cos, sin, used, **kwargs):
            calls.append(kwargs)
            return x, jnp.zeros((config.layers, 2, 8, mk.TABLE_LANES), jnp.float32)

        options = op.stack_kwargs(4, _structure(1))
        fn = functools.partial(
            op.local_vouched_stack,
            config=config,
            tiling=kt._TILING,
            axis=None,
            margin=_MARGIN,
            options=options,
        )
        with patch.object(op.kernel, "stack", stack):
            jax.make_jaxpr(fn)(
                *operands,
                jnp.zeros((config.layers, config.heads), jnp.int32),
                jnp.zeros(2, jnp.int32),
                *_weight_list(weights),
            )
        (kwargs,) = calls
        fixed = kwargs.pop("fixed_heads")
        self.assertEqual((fixed.shape, fixed.dtype), ((3, 4), jnp.bool_))
        want = dict(options, return_peaks=True)
        del want["fixed_heads"]
        want.update(config=config, tiling=kt._TILING, axis=None, interpret=False)
        self.assertEqual(kwargs, want)
        self.assertEqual(
            (
                kwargs["static_structure"],
                kwargs["narrow_tail"],
                kwargs["prefetch_down"],
            ),
            (193, True, False),
        )


@unittest.skipUnless(_HAS_JAX, _SKIP)
class TestVouchedStepMultiDevice(CustomTestCase):
    """Two devices, two heads each, race detector on: the loop runs the
    kernel's remote copies and barrier once or twice, the peaks and the heads
    to try agree across devices."""

    def test_two_devices(self):
        code = """
            print(json.dumps(t._vouch_checks(2, races=True)))
        """
        stdout = _in_fresh_process(2, code, timeout=5400)
        self.assertNotIn("RACE DETECTED", stdout)
        _assert_checks(self, json.loads(stdout.splitlines()[-1]), _case(1)[0])

    def test_two_devices_structured(self):
        # Both options: the static structure 449 shapes the loops of the
        # traced 475 on every device.
        code = """
            print(json.dumps(t._vouch_checks(2, races=True, structure=t._structure(2))))
        """
        stdout = _in_fresh_process(2, code, timeout=5400)
        self.assertNotIn("RACE DETECTED", stdout)
        _assert_checks(self, json.loads(stdout.splitlines()[-1]), _case(1)[0])


_LOWER = r"""
    import hashlib, re
    from jax.experimental import topologies
    from jax.sharding import NamedSharding, PartitionSpec as P
    import jax.numpy as jnp
    import numpy as np
    from sglang_tpu.multimodal_gen.h3_megakernel import op
    from sglang_tpu.multimodal_gen.h3_megakernel import reference as ref
    try:
        devices = topologies.get_topology_desc("tpu7x:1x1x1", "tpu").devices
    except Exception as error:
        print("SKIP", type(error).__name__)
        raise SystemExit(0)
    mesh = jax.sharding.Mesh(np.asarray(devices), ("sp",))
    config = ref.Config(layers=2)
    rows = 2 * 2048
    packed = {"norm1": (2, 1, config.hidden), "norm2": (2, 1, config.hidden),
              "q_norm": (2, 1, 128), "k_norm": (2, 1, 128),
              "wqkv": (2, 28, config.hidden, 768), "wo": (2, 7, config.width, 768),
              "w13": (2, 56, config.hidden, 512), "w2": (2, 21, config.ffn, 256)}
    def spec(shape, partition, dtype=jnp.bfloat16):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, partition))
    shard = P("sp")
    head = [spec((rows, config.hidden), shard), spec((rows, 128), shard),
            spec((rows, 128), shard), spec((rows, 128), shard),
            spec((2, 6, 16, config.hidden), P()), spec((1,), P(), jnp.int32)]
    weights = [spec(packed[name], P()) for name in op.pack.WEIGHT_ORDER]
    extra = [spec((2, config.heads), P(), jnp.int32), spec((2,), P(), jnp.int32)]
    # The static-structure option's options for a live length 37 rows short
    # of the rows (no query tile of padding: the dense walk).
    options = op.stack_kwargs(2, rows - 37)
    texts = {
        "plain": jax.jit(op.sharded_stack(mesh, config, op.STACK_TILING)).lower(
            *head, *weights).as_text(),
        "vouched": jax.jit(op.sharded_vouched_stack(mesh, config, op.STACK_TILING, 1.1)).lower(
            *head, *extra, *weights).as_text(),
        "structured": jax.jit(op.sharded_stack(
            mesh, config, op.STACK_TILING, options=options)).lower(*head, *weights).as_text(),
        "vouched_structured": jax.jit(op.sharded_vouched_stack(
            mesh, config, op.STACK_TILING, 1.1, options=options)).lower(
            *head, *extra, *weights).as_text(),
    }
    body = re.compile(r'(?:\\22|\\"|")body(?:\\22|\\"|"):\s*(?:\\22|\\"|")([A-Za-z0-9+/=]+)')
    out = {}
    for name, text in texts.items():
        # Each Mosaic kernel body's digest: equal digests, equal bytes.
        bodies = [hashlib.sha256(b.encode()).hexdigest() for b in body.findall(text)]
        out[name] = {"calls": text.count("tpu_custom_call"), "bodies": bodies,
                     "while": text.count("stablehlo.while")}
    print("LOWERED " + json.dumps(out))
"""


@unittest.skipUnless(_HAS_JAX, _SKIP)
class TestLoweredForTpu(CustomTestCase):
    """Lowered for two v7x cores at H3's widths: the vouched op's program
    holds the plain op's Mosaic kernel, byte for byte, once (in its loop):
    the heads to try reach the kernel as data. With the static-structure
    option's options, it holds the structured op's kernel."""

    def test_same_kernel_once(self):
        stdout = _in_fresh_process(2, _LOWER, timeout=1800)
        if "SKIP" in stdout:
            self.skipTest(f"no compile-only v7x client: {stdout.strip()}")
        line = [line for line in stdout.splitlines() if line.startswith("LOWERED ")]
        lowered = json.loads(line[-1][len("LOWERED ") :])
        plain, vouched = lowered["plain"], lowered["vouched"]
        self.assertEqual((plain["calls"], vouched["calls"]), (1, 1))
        self.assertEqual(len(plain["bodies"]), 1)
        self.assertEqual(vouched["bodies"], plain["bodies"])
        self.assertEqual((plain["while"], vouched["while"]), (0, 1))
        # Both options: the structured op's kernel, once; the structure gives
        # another kernel than the plain op's.
        structured, both = lowered["structured"], lowered["vouched_structured"]
        self.assertEqual((structured["calls"], both["calls"]), (1, 1))
        self.assertEqual(len(structured["bodies"]), 1)
        self.assertEqual(both["bodies"], structured["bodies"])
        self.assertNotEqual(structured["bodies"], plain["bodies"])
        self.assertEqual((structured["while"], both["while"]), (0, 1))


if __name__ == "__main__":
    unittest.main()
