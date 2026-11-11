# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted for SGLang-TorchTPU from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: tests/kernels/test_fused_shared_expert.py
# Revision: ef2599d3d4342f41d2ad223b266e126299862b53
"""Fused FP8 shared expert: dispatch envelope and bitwise parity with the
unfused ops (two quantized matmuls around the clamped SiLU-and-mul).

vLLM-TorchTPU checks parity against its torch path; SGLang has no torch op for
it, so the TPU cases compare against this module's own unfused JAX ops,
compiled without excess precision as torch_tpu compiles the served program.
"""

import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sglang_tpu import fp8_swiglu_mlp as se

HIDDEN, INTER, LIMIT = 5120, 2304, 10.0


def _abstract(tokens=288, hidden=HIDDEN, inter=INTER):
    return (
        jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16),
        jax.ShapeDtypeStruct((hidden, 2 * inter), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((2 * inter,), jnp.float32),
        jax.ShapeDtypeStruct((inter, hidden), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((hidden,), jnp.float32),
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("tokens,expected", [(288, True), (240, False), (2048, False)])
@pytest.mark.parametrize("generation", [-1, 6, 7])
def test_envelope(monkeypatch, tokens, expected, generation):
    monkeypatch.setattr(se, "get_tpu_version", lambda: generation)
    assert se.is_supported(*_abstract(tokens)) is (expected and generation == 7)


@pytest.mark.cpu_test
def test_envelope_rejects_other_dims(monkeypatch):
    monkeypatch.setattr(se, "get_tpu_version", lambda: 7)
    assert not se.is_supported(*_abstract(inter=2560))
    x, w13, s13, w2, s2 = _abstract()
    blockwise = jax.ShapeDtypeStruct((1, 40, 1, 2 * INTER), jnp.float32)
    assert not se.is_supported(x, w13, blockwise, w2, s2)


def _weights(seed):
    rng = np.random.default_rng(seed)
    w13 = np.clip(rng.standard_normal((HIDDEN, 2 * INTER)) * 64, -448, 448)
    w2 = np.clip(rng.standard_normal((INTER, HIDDEN)) * 64, -448, 448)
    s13 = (rng.random(2 * INTER) + 0.5) * 1e-3
    s2 = (rng.random(HIDDEN) + 0.5) * 1e-3
    fp8 = jnp.float8_e4m3fn
    return (
        jnp.asarray(w13, jnp.float32).astype(fp8),
        jnp.asarray(s13, jnp.float32),
        jnp.asarray(w2, jnp.float32).astype(fp8),
        jnp.asarray(s2, jnp.float32),
    )


def _unfused(x, w13, s13, w2, s2):
    gate, up = jnp.split(se.xla_quantized_matmul(x, w13, s13), 2, axis=-1)
    return se.xla_quantized_matmul(se.silu_and_mul_with_clamp(gate, up, LIMIT), w2, s2)


_AS_SERVED = {"xla_allow_excess_precision": False}


@pytest.mark.parametrize(
    "tokens,pattern,seed",
    [
        (288, "random", 0),
        (288, "random", 1),
        (288, "mixed", 2),
        (288, "zero", 3),
        (256, "random", 4),
    ],
)
def test_fused_shared_expert_is_bitwise_equal(tokens, pattern, seed):
    if jax.default_backend() != "tpu" or se.get_tpu_version() != 7:
        pytest.skip("the fused shared expert is measured on TPU v7 only")
    graph = str(
        jax.make_jaxpr(lambda *a: se.fp8_swiglu_mlp(*a, limit=LIMIT))(
            *_abstract(tokens)
        )
    )
    assert ("fused_shared_expert_fp8" in graph) is (tokens == 288)
    rng = np.random.default_rng(100 + seed)
    x = rng.standard_normal((tokens, HIDDEN)).astype(np.float32)
    if pattern == "zero":
        x[:] = 0
    elif pattern == "mixed":
        x *= np.exp(rng.random((tokens, 1)) * 8 - 4)
        x[:16] = 0
    x = jnp.asarray(x).astype(jnp.bfloat16)
    weights = _weights(seed)
    want = jax.jit(_unfused, compiler_options=_AS_SERVED)(x, *weights)
    got = jax.jit(
        lambda *a: se.fp8_swiglu_mlp(*a, limit=LIMIT), compiler_options=_AS_SERVED
    )(x, *weights)
    got = np.asarray(got)
    assert np.isfinite(got.astype(np.float32)).all()
    np.testing.assert_array_equal(got.view(np.uint16), np.asarray(want).view(np.uint16))
