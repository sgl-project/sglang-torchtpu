# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 CSA2 torch ops against a dense float reference.

The reference attends, per token, the causal 128-position SWA window (from the
request's minimum visible position) plus the token's selected compressed
records, with an attention sink, then inverse-rotates the last 64 features as
``conj(e^{i theta})``. The TPU cases run the ops through TorchTPU for prefill
(ragged requests, an empty request, padding rows, replay minimum, an odd SWA
partner row) and decode, at compress ratios 0, 1 and 2. The CPU cases check the
JAX bridges' traced shapes against the registered fakes and the addressing
math, and need neither TorchTPU nor a TPU.

Usage:
    python test/registered/tpu/unit/test_dsv41_csa2_ops.py
"""

import functools
import importlib.util
import os
import sys

# TorchTPU owns the device; JAX exports the custom calls on the CPU.
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
import torch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.models.deepseek_v41.kernels.csa2.cache import (
    encode_csa2_kv,
)
from sglang_tpu.models.deepseek_v41.ops import csa2 as ops

register_tpu_ci(
    est_time=600,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)

HAS_TORCH_TPU = importlib.util.find_spec("torch_tpu") is not None
requires_tpu = pytest.mark.skipif(not HAS_TORCH_TPU, reason="requires TorchTPU")

MAX_POSITION = 2048
HEADS = 64
SCALE = 512**-0.5


# =============================================================================
# Shared helpers
# =============================================================================


def _theta(max_position=MAX_POSITION):
    # Interleaved-pair RoPE angles [max_position, 32], as freqs_cis = e^{i theta}.
    freqs = 10000.0 ** (-np.arange(0, ops.ROPE_DIM, 2) / ops.ROPE_DIM)
    return np.arange(max_position)[:, None] * freqs[None, :]


def _cos_sin(theta):
    """``get_or_build_cos_sin_cache(freqs_cis)``: ``[cos | sin]`` FP32."""
    freqs_cis = torch.polar(
        torch.ones(theta.shape, dtype=torch.float64), torch.from_numpy(theta)
    ).to(torch.complex64)
    real_imag = torch.view_as_real(freqs_cis)
    return torch.cat([real_imag[..., 0], real_imag[..., 1]], dim=-1).float()


def _inverse_rope_reference(output, theta_rows):
    """``output`` [..., H, 512]; rotate pairs of the last 64 by conj(e^{i theta})."""
    result = output.astype(np.float64)
    tail = result[..., 512 - ops.ROPE_DIM :].reshape(*result.shape[:-1], -1, 2)
    z = (tail[..., 0] + 1j * tail[..., 1]) * np.exp(-1j * theta_rows)[..., None, :]
    tail = np.stack((z.real, z.imag), axis=-1).reshape(*result.shape[:-1], -1)
    result[..., 512 - ops.ROPE_DIM :] = tail
    return result


def _decode_main(words, scales):
    """Packed FP8 words [R,128] + scales [R] -> float [R,512]."""
    bits = words.view(np.uint32)
    planes = np.stack([((bits >> (8 * g)) & 255).astype(np.uint8) for g in range(4)])
    values = planes.transpose(1, 0, 2).reshape(-1, 512)
    return values.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scales[:, None]


# =============================================================================
# TPU: ops against the dense reference
# =============================================================================

# (prefix tokens, new tokens, minimum visible position) per request. Prefill:
# a fresh request, an odd prefix whose SWA word row is shared with history and
# a replay minimum hiding part of the window, an empty request, a long prefix.
_PREFILL = ((0, 70, 0), (301, 45, 200), (50, 0, 0), (900, 20, 0))
_DECODE = ((5, 1, 0), (130, 1, 0), (700, 1, 0), (1500, 1, 0))
_PADDED_TOKENS = {_PREFILL: 144, _DECODE: 8}


class _Case:
    """Pools, step inputs and the dense reference for one batch."""

    def __init__(self, requests, ratio, seed=0, tokens=None):
        rng = np.random.default_rng(seed)
        self.ratio = ratio
        self.theta = _theta()
        self.requests = requests
        lengths = [p + n for p, n, _ in requests]
        tokens = _PADDED_TOKENS[requests] if tokens is None else tokens
        counts = [n for _, n, _ in requests]
        self.starts = np.r_[0, np.cumsum(counts)].astype(np.int32)
        self.positions = np.zeros(tokens, np.int32)
        for b, (prefix, new, _) in enumerate(requests):
            self.positions[self.starts[b] : self.starts[b + 1]] = prefix + np.arange(
                new
            )
        self.minimum = np.array([m for _, _, m in requests], np.int32)
        self.kv = [
            rng.normal(size=(length, 512)).astype(ml_dtypes.bfloat16)
            for length in lengths
        ]
        self.q = rng.normal(size=(tokens, HEADS, 512)).astype(ml_dtypes.bfloat16)
        self.sink = rng.uniform(-1, 1, HEADS).astype(np.float32)
        self._build_swa(rng, lengths, tokens)
        if ratio:
            self._build_main(rng, lengths, tokens)

    def _build_swa(self, rng, lengths, tokens):
        columns = max(-(-length // 64) for length in lengths) + 1
        pages = sum(-(-length // 64) for length in lengths) + 4
        ids = iter(rng.permutation(np.arange(1, pages)))
        self.swa_table = np.zeros((len(lengths), columns), np.int32)
        self.swa_pages = np.zeros((pages, 64, 512), ml_dtypes.bfloat16)
        for b, ((prefix, _, _), length) in enumerate(zip(self.requests, lengths)):
            # Pages wholly before the first query's window are evicted (null).
            evicted = max(prefix - 127, 0) // 64
            for column in range(evicted, -(-length // 64)):
                self.swa_table[b, column] = next(ids)
            for position in range(evicted * 64, prefix):
                page = self.swa_table[b, position // 64]
                self.swa_pages[page, position % 64] = self.kv[b][position]
        self.new_kv = np.zeros((tokens, 512), ml_dtypes.bfloat16)
        for t, b, p in self._rows():
            self.new_kv[t] = self.kv[b][p]

    def _build_main(self, rng, lengths, tokens):
        r = self.ratio
        self.page_records = 256 // r
        columns = max(-(-length // 256) for length in lengths)
        pages = len(lengths) * columns + 2
        self.main_pages = rng.permutation(np.arange(1, pages))[: len(lengths) * columns]
        self.main_pages = self.main_pages.reshape(len(lengths), columns).astype(
            np.int32
        )
        self.records = pages * self.page_records
        self.latents = [
            rng.normal(size=(length // r, 512)).astype(ml_dtypes.bfloat16)
            for length in lengths
        ]
        records, latents = [], []
        for b, latent in enumerate(self.latents):
            ids = np.arange(len(latent))
            records.append(
                self.main_pages[b, ids // self.page_records] * self.page_records
                + ids % self.page_records
            )
            latents.append(latent)
        # Skipped rows: -1 records with junk latents must leave the pool intact.
        records.append(np.full(3, -1))
        latents.append(np.full((3, 512), 7, ml_dtypes.bfloat16))
        self.publish_records = np.concatenate(records).astype(np.int32)
        self.publish_latents = np.concatenate(latents)
        self.selected = np.full((tokens, 512), -1, np.int32)
        for t, _, p in self._rows():
            count = (p + 1) // r
            ids = rng.permutation(count)[:512]
            self.selected[t, : len(ids)] = ids

    def _rows(self):
        for b in range(len(self.requests)):
            for t in range(self.starts[b], self.starts[b + 1]):
                yield t, b, int(self.positions[t])

    def reference(self, words=None, scales=None):
        """Dense attention with sink; ``words``/``scales`` are the published pool."""
        main = _decode_main(words, scales) if self.ratio else None
        out = np.zeros(self.q.shape, np.float64)
        for t, b, p in self._rows():
            low = max(p - ops.SWA_WINDOW + 1, int(self.minimum[b]), 0)
            keys = [self.kv[b][low : p + 1].astype(np.float32)]
            if self.ratio:
                ids = self.selected[t][self.selected[t] >= 0]
                records = (
                    self.main_pages[b, ids // self.page_records] * self.page_records
                    + ids % self.page_records
                )
                keys.insert(0, main[records])
            keys = np.concatenate(keys).astype(np.float64)
            logits = self.q[t].astype(np.float64) @ keys.T * SCALE
            maximum = np.maximum(logits.max(axis=1), self.sink)
            weights = np.exp(logits - maximum[:, None])
            weights /= (weights.sum(axis=1) + np.exp(self.sink - maximum))[:, None]
            out[t] = weights @ keys
        rotated = _inverse_rope_reference(out, self.theta[self.positions])
        rotated[self.starts[-1] :] = 0
        return rotated


def _device(x):
    if isinstance(x, np.ndarray) and x.dtype == ml_dtypes.bfloat16:
        x = torch.from_numpy(x.view(np.uint16).astype(np.int16)).view(torch.bfloat16)
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(x))
    return x.contiguous().to("tpu")


def _host(x):
    x = x.cpu()
    if x.dtype == torch.bfloat16:
        return x.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return x.numpy()


def _step(case, inputs):
    """One layer step: SWA metadata, SWA and main publication, attention."""
    meta, bounds, swa_records = ops.swa_metadata(
        inputs["swa_table"], inputs["starts"], inputs["positions"], inputs["minimum"]
    )
    inputs["swa_pages"].copy_(
        ops.publish_swa(inputs["swa_pages"], swa_records, inputs["new_kv"])
    )
    main = None
    if case.ratio:
        words, scales = ops.publish_main(
            inputs["words"],
            inputs["scales"],
            inputs["publish_records"],
            inputs["publish_latents"],
        )
        inputs["words"].copy_(words)
        inputs["scales"].copy_(scales)
        main = (
            inputs["words"],
            inputs["scales"],
            inputs["selected"],
            inputs["main_pages"],
            case.page_records,
        )
    out = ops.attention(
        inputs["q"],
        inputs["swa_pages"],
        meta,
        bounds,
        inputs["starts"],
        inputs["positions"],
        inputs["sink"],
        inputs["cos_sin"],
        case.ratio,
        SCALE,
        main=main,
    )
    return out, swa_records


def _inputs(case):
    inputs = {
        "swa_table": case.swa_table,
        "starts": case.starts,
        "positions": case.positions,
        "minimum": case.minimum,
        "swa_pages": case.swa_pages,
        "new_kv": case.new_kv,
        "q": case.q,
        "sink": case.sink,
        "cos_sin": _cos_sin(case.theta),
    }
    if case.ratio:
        inputs.update(
            words=np.zeros((case.records, 128), np.int32),
            scales=np.zeros(case.records, np.float32),
            publish_records=case.publish_records,
            publish_latents=case.publish_latents,
            selected=case.selected,
            main_pages=case.main_pages,
        )
    return {name: _device(value) for name, value in inputs.items()}


def _check(case, inputs, out, swa_records):
    rows = case.starts[-1]
    records = _host(swa_records)
    pages = _host(inputs["swa_pages"]).reshape(-1, 512)
    expected_records = np.array(
        [case.swa_table[b, p // 64] * 64 + p % 64 for _, b, p in case._rows()]
    )
    np.testing.assert_array_equal(records[:rows], expected_records)
    np.testing.assert_array_equal(records[rows:], -1)
    # New rows landed; history rows, including odd-partner halves, survived.
    np.testing.assert_array_equal(pages[expected_records], case.new_kv[:rows])
    np.testing.assert_array_equal(
        pages.reshape(case.swa_pages.shape),
        np.where(
            np.isin(np.arange(pages.shape[0]), expected_records)[:, None],
            pages,
            case.swa_pages.reshape(-1, 512),
        ).reshape(case.swa_pages.shape),
    )
    words = scales = None
    if case.ratio:
        words, scales = _host(inputs["words"]), _host(inputs["scales"])
        valid = case.publish_records >= 0
        encoded, encoded_scales = map(
            np.asarray, encode_csa2_kv(jnp.asarray(case.publish_latents[valid]))
        )
        np.testing.assert_array_equal(words[case.publish_records[valid]], encoded)
        np.testing.assert_array_equal(
            scales[case.publish_records[valid]], encoded_scales
        )
        untouched = np.setdiff1d(np.arange(case.records), case.publish_records)
        assert not words[untouched].any() and not scales[untouched].any()
    actual = _host(out).astype(np.float64)
    expected = case.reference(words, scales)
    assert actual.shape == case.q.shape
    np.testing.assert_array_equal(actual[rows:], 0)
    np.testing.assert_allclose(actual, expected, atol=2e-2, rtol=2e-2)
    error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert error < 1e-2, error


@pytest.fixture(scope="module")
def tpu_ops():
    import torch_tpu  # noqa: F401

    ops.register_csa2_ops()
    return ops


@requires_tpu
@pytest.mark.parametrize("ratio", [0, 1, 2])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_attention_matches_dense_reference(tpu_ops, phase, ratio):
    case = _Case(_PREFILL if phase == "prefill" else _DECODE, ratio, seed=ratio)
    inputs = _inputs(case)
    out, swa_records = _step(case, inputs)
    _check(case, inputs, out, swa_records)


# One decode request on its own: at tokens == 1 the kernel's bq is 1, so
# ``batch == 1 and tokens % bq == 0`` takes CSA2's fixed-tile path. The served
# --tpu-compile-bs 1 decode graph (one request per DP rank) produced NaN from
# the second token on; vllm-torchtpu pads to MIN_NUM_SEQS = 8 requests and never
# reaches this path. The two padded layouts run the general tile builder on the
# same request: 8 rows (8 % 6 != 0), and the bs-8 bucket with 7 empty requests.
_ONE = ((700, 1, 0),)
_SINGLE_DECODE = {
    "one_token_fixed_tiles": (_ONE, 1),
    "padded_rows": (_ONE, 8),
    "padded_requests": (_ONE + ((50, 0, 0),) * 7, 8),
}


@requires_tpu
@pytest.mark.parametrize("ratio", [0, 1, 2])
@pytest.mark.parametrize("layout", sorted(_SINGLE_DECODE))
def test_single_request_decode_matches_dense_reference(tpu_ops, layout, ratio):
    requests, tokens = _SINGLE_DECODE[layout]
    case = _Case(requests, ratio, seed=20 + ratio, tokens=tokens)
    inputs = _inputs(case)
    out, swa_records = _step(case, inputs)
    _check(case, inputs, out, swa_records)


@requires_tpu
@pytest.mark.parametrize("ratio", [0, 2])
def test_compiled_step_updates_caller_pools(tpu_ops, ratio):
    case = _Case(_PREFILL, ratio, seed=10 + ratio)
    inputs = _inputs(case)

    @functools.partial(torch.compile, backend="tpu", fullgraph=True, dynamic=False)
    def step(inputs):
        return _step(case, inputs)

    out, swa_records = step(inputs)
    assert out.shape == case.q.shape and out.dtype == torch.bfloat16
    assert swa_records.shape == case.positions.shape
    _check(case, inputs, out, swa_records)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
