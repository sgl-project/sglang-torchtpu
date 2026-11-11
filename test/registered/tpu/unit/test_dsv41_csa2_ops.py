# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 CSA2 torch ops against a dense float reference.

The reference attends, per token, the causal 128-position SWA window (from the
request's minimum visible position) plus the token's selected compressed
records, with an attention sink, then inverse-rotates the last 64 features as
``conj(e^{i theta})``. The TPU cases run the ops through TorchTPU for prefill
(ragged requests, an empty request, padding rows, replay minimum, an odd SWA
partner row) and decode, at compress ratios 0, 1 and 2. CPU bridge-shape and
addressing checks live in the CPU-registered CSA2 bridge test module.

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


def _pack_swa(tokens):
    """BF16 [P,64,512] -> flat INT32 [P*128,128] owner.

    Word feature f holds token 2k's BF16 bits in low16 and token 2k+1's
    bits in high16; each pair's 512 features span four consecutive word rows.
    This is not a contiguous BF16-to-INT32 view (which would pair features).
    """
    bits = tokens.view(np.uint16).reshape(-1, 2, 512).astype(np.uint32)
    return (bits[:, 0] | (bits[:, 1] << 16)).view(np.int32).reshape(-1, 128)


def _unpack_swa(words):
    """Flat INT32 word owner -> logical BF16 [P*64,512] token rows."""
    bits = words.view(np.uint32).reshape(-1, 512)
    halves = np.stack((bits & 65535, bits >> 16), axis=1).astype(np.uint16)
    return halves.view(ml_dtypes.bfloat16).reshape(-1, 512)


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

    def __init__(self, requests, ratio, seed=0):
        rng = np.random.default_rng(seed)
        self.ratio = ratio
        self.theta = _theta()
        self.requests = requests
        lengths = [p + n for p, n, _ in requests]
        tokens = _PADDED_TOKENS[requests]
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
        "swa_pages": _pack_swa(case.swa_pages),
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
    owner = _host(inputs["swa_pages"])
    assert owner.shape == (case.swa_pages.shape[0] * 128, 128)
    assert owner.dtype == np.int32
    pages = _unpack_swa(owner)
    expected_records = np.array(
        [case.swa_table[b, p // 64] * 64 + p % 64 for _, b, p in case._rows()]
    )
    np.testing.assert_array_equal(records[:rows], expected_records)
    np.testing.assert_array_equal(records[rows:], -1)
    # New rows landed; history rows, including odd-partner halves, survived.
    np.testing.assert_array_equal(pages[expected_records], case.new_kv[:rows])
    expected_pages = case.swa_pages.copy()
    expected_pages.reshape(-1, 512)[expected_records] = case.new_kv[:rows]
    np.testing.assert_array_equal(owner, _pack_swa(expected_pages))
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


@requires_tpu
def test_swa_publication_preserves_paired_neighbors(tpu_ops):
    """Updating either token half must preserve its nonzero adjacent neighbor.

    Unsorted records also update both halves of one pair in the same call;
    null-page, negative and out-of-capacity records must not alter any word.
    """
    rng = np.random.default_rng(21)
    before = rng.normal(size=(3, 64, 512)).astype(ml_dtypes.bfloat16)
    records = np.array([65, 68, 67, 64, -1, 0, 63, 192], np.int32)
    values = rng.normal(size=(len(records), 512)).astype(ml_dtypes.bfloat16)
    expected = before.copy()
    # 64/65 share a pair; 67 and 68 leave opposite-parity neighbors 66/69.
    expected.reshape(-1, 512)[records[:4]] = values[:4]
    owner = _device(_pack_swa(before))
    owner.copy_(ops.publish_swa(owner, _device(records), _device(values)))
    actual = _host(owner)
    assert actual.shape == (3 * 128, 128) and actual.dtype == np.int32
    np.testing.assert_array_equal(actual, _pack_swa(expected))
    np.testing.assert_array_equal(_unpack_swa(actual), expected.reshape(-1, 512))


@requires_tpu
def test_compiled_encoded_publication_updates_both_pool_owners(tpu_ops):
    """Already encoded bytes/scales survive donation, copy-back and replay."""
    words = _device(np.full((512, 128), 17, np.int32))
    scales = _device(np.full(512, 0.125, np.float32))
    records = np.array([257, -1, 130, 511], np.int32)
    ids = _device(records)
    expected_words = np.full((512, 128), 17, np.int32)
    expected_scales = np.full(512, 0.125, np.float32)

    @functools.partial(torch.compile, backend="tpu", fullgraph=True, dynamic=False)
    def publish(words, scales, ids, encoded, encoded_scales):
        new_words, new_scales = ops.publish_main_encoded(
            words, scales, ids, encoded, encoded_scales
        )
        words.copy_(new_words)
        scales.copy_(new_scales)
        # A real consumer reads both owners after copy-back.
        return words[130].clone(), scales[130].clone()

    for replay in range(2):
        payload = np.arange(4 * 128, dtype=np.int32).reshape(4, 128) + replay * 1000
        payload_scales = (np.arange(1, 5, dtype=np.float32) + replay)[:, None] / 32
        read_words, read_scale = publish(
            words, scales, ids, _device(payload), _device(payload_scales)
        )
        live = records >= 0
        expected_words[records[live]] = payload[live]
        expected_scales[records[live]] = payload_scales[live, 0]
        np.testing.assert_array_equal(_host(words), expected_words)
        np.testing.assert_array_equal(_host(scales), expected_scales)
        np.testing.assert_array_equal(_host(read_words), payload[2])
        np.testing.assert_array_equal(_host(read_scale), payload_scales[2, 0])


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


@requires_tpu
def test_fake_shapes_under_fake_tensor_mode(tpu_ops):
    from torch._subclasses.fake_tensor import FakeTensorMode

    case = _Case(_DECODE, 2)
    with FakeTensorMode(allow_non_fake_inputs=False) as mode:
        inputs = {
            name: mode.from_tensor(value) for name, value in _inputs(case).items()
        }
        out, swa_records = _step(case, inputs)
    assert out.shape == case.q.shape and out.dtype == torch.bfloat16
    assert swa_records.shape == case.positions.shape
    assert swa_records.dtype == torch.int32


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
