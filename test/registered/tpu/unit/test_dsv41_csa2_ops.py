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
import inspect
import os
import sys
from unittest import mock

# TorchTPU owns the device; JAX exports the custom calls on the CPU.
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
import torch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.models.deepseek_v41.kernels.csa2.cache import (
    encode_csa2_kv,
    prepare_main_records,
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


def _tpu_info_patch():
    from sglang_tpu import pallas_export

    return mock.patch.object(
        pallas_export, "get_pallas_tpu_device_kind", return_value="TPU7x"
    )


# =============================================================================
# CPU: bridges, fakes, addressing
# =============================================================================

_BATCH, _TOKENS, _COLUMNS = 3, 40, 8
_RATIO = 2
_RECORDS = 256 // _RATIO


def _bridge_examples():
    """name -> (array inputs, static inputs) for each JAX bridge."""
    i32 = np.int32
    table = ((_BATCH, _COLUMNS), i32)
    starts = ((_BATCH + 1,), i32)
    positions = ((_TOKENS,), i32)
    pages = ((20 * 128, 128), i32)
    q = ((_TOKENS, 16, 512), ml_dtypes.bfloat16)
    common = (starts, positions, ((16,), np.float32), ((MAX_POSITION, 64), np.float32))
    meta = ((_BATCH, 2 + ops.swa_metadata_columns(_TOKENS)), i32)
    examples = {
        "swa_metadata": ((table, starts, positions, ((_BATCH,), i32)), ()),
        "publish_swa": ((pages, positions, ((_TOKENS, 512), ml_dtypes.bfloat16)), ()),
        "publish_main": (
            (
                ((4 * _RECORDS, 128), i32),
                ((4 * _RECORDS,), np.float32),
                ((24,), i32),
                ((24, 512), ml_dtypes.bfloat16),
            ),
            (),
        ),
        "attend_swa": ((q, pages, *common, meta), (SCALE,)),
        "attend_main": (
            (
                q,
                ((4 * _RECORDS, 128), i32),
                ((4 * _RECORDS,), np.float32),
                ((_TOKENS, 512), i32),
                ((_BATCH, 4), i32),
                pages,
                *common,
                meta,
            ),
            (_RECORDS, _RATIO, SCALE),
        ),
        "wo_a": (
            (
                ((_TOKENS, 64, 512), ml_dtypes.bfloat16),
                ((4096, 8192), ml_dtypes.float8_e4m3fn),
                ((8192,), np.float32),
            ),
            (),
        ),
    }
    examples["gather_attend"] = examples["attend_main"]
    examples["attend_prepared"] = (
        (
            q,
            ((_TOKENS, 512, 128), i32),
            ((_TOKENS, 512), np.float32),
            ((_TOKENS,), i32),
            pages,
            *common,
            meta,
        ),
        (_RATIO, SCALE),
    )
    return examples


def _torch_dtype(dtype):
    return {
        np.dtype(np.int32): torch.int32,
        np.dtype(np.float32): torch.float32,
        np.dtype(ml_dtypes.bfloat16): torch.bfloat16,
        np.dtype(ml_dtypes.float8_e4m3fn): torch.float8_e4m3fn,
    }[np.dtype(dtype)]


@pytest.mark.parametrize("name", sorted(ops._DEFINITIONS))
def test_bridge_traces_match_fakes(name):
    function, fake, donated = ops._DEFINITIONS[name]
    arrays, static_args = _bridge_examples()[name]
    # cpu_export_jax_op makes every non-jax.Array parameter static.
    parameters = list(inspect.signature(function).parameters.values())
    tensors = [p for p in parameters if p.annotation is jax.Array]
    assert [p.annotation for p in parameters[len(tensors) :]] == [
        type(s) for s in static_args
    ]
    assert len(tensors) == len(arrays)
    assert all(i < len(tensors) for i in donated)

    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    specs = [jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in arrays]
    with _tpu_info_patch(), tpu_info_for_cpu_export():
        traced = jax.eval_shape(lambda *xs: function(*xs, *static_args), *specs)
    fakes = fake(
        *(torch.empty(s, dtype=_torch_dtype(d), device="meta") for s, d in arrays),
        *static_args,
    )
    traced, fakes = jax.tree.leaves(traced), jax.tree.leaves(fakes)
    assert [(tuple(t.shape), _torch_dtype(t.dtype)) for t in traced] == [
        (tuple(f.shape), f.dtype) for f in fakes
    ]
    for i in donated:
        # A donated pool returns in its own shape so XLA can alias it.
        assert tuple(traced[donated.index(i)].shape) == arrays[i][0]


def test_cos_sin_table_matches_backend_builder():
    theta = _theta(64)
    freqs_cis = torch.polar(torch.ones(theta.shape), torch.from_numpy(theta).float())
    try:
        from sglang_tpu.dsv4.dsv4_memory_pool import get_or_build_cos_sin_cache
    except ImportError as error:
        pytest.skip(f"dsv4 memory pool unavailable: {error}")
    torch.testing.assert_close(
        # FP32 polar against an FP64 reference; the check is the [cos | sin] order.
        get_or_build_cos_sin_cache(freqs_cis),
        _cos_sin(theta),
        atol=1e-5,
        rtol=0,
    )


def test_inverse_rope_table_undoes_rope():
    rng = np.random.default_rng(0)
    theta = _theta()
    positions = rng.integers(0, MAX_POSITION, 17).astype(np.int32)
    table = np.asarray(
        ops._inverse_rope(jnp.asarray(_cos_sin(theta).numpy()), jnp.asarray(positions))
    )
    assert table.shape == (17, 2, 128) and table.dtype == np.float32
    output = rng.normal(size=(17, 4, 512)).astype(np.float32)
    tail = output[..., 384:]
    # The kernel's finish(): rotated = tail * rope[0] + tail[lane ^ 1] * rope[1].
    rotated = (
        tail * table[:, None, 0] + tail[..., np.arange(128) ^ 1] * table[:, None, 1]
    )
    expected = _inverse_rope_reference(output, theta[positions])[..., 384:]
    np.testing.assert_allclose(rotated, expected, atol=1e-5)


def test_swa_metadata_records_and_padding():
    rng = np.random.default_rng(1)
    lengths = [5, 0, 70]
    positions = np.zeros(_TOKENS, np.int32)
    positions[:5] = np.arange(60, 65)
    positions[5 : 75 - 40 + 5] = 0
    lengths = [5, 0, 30]
    positions[5:35] = np.arange(100, 130)
    starts = np.array([0, 5, 5, 35], np.int32)
    table = rng.permutation(np.arange(1, 1 + _BATCH * _COLUMNS)).astype(np.int32)
    table = table.reshape(_BATCH, _COLUMNS)
    meta, bounds, records = map(
        np.asarray,
        ops._swa_metadata(
            jnp.asarray(table),
            jnp.asarray(starts),
            jnp.asarray(positions),
            jnp.zeros(_BATCH, jnp.int32),
        ),
    )
    assert meta.shape == (_BATCH, 2 + ops.swa_metadata_columns(_TOKENS))
    requests = np.repeat(np.arange(_BATCH), lengths)
    real = positions[: starts[-1]]
    expected = table[requests, real // 64] * 64 + real % 64
    np.testing.assert_array_equal(records[: starts[-1]], expected)
    np.testing.assert_array_equal(records[starts[-1] :], -1)


@pytest.mark.parametrize("ratio", [1, 2])
def test_main_records_are_full_loc_over_ratio(ratio):
    """Read addressing (full page id as main page id) equals the publish record
    ``full_loc // r`` of the record's last token."""
    rng = np.random.default_rng(2 + ratio)
    page_records = 256 // ratio
    lengths = np.array([700, 1, 300])
    columns = 4
    full_pages = rng.permutation(np.arange(1, 1 + len(lengths) * columns))
    full_pages = full_pages.reshape(len(lengths), columns).astype(np.int32)
    tokens = len(lengths)
    positions = (lengths - 1).astype(np.int32)
    starts = np.arange(tokens + 1, dtype=np.int32)
    selected = np.full((tokens, 512), -1, np.int32)
    for t, p in enumerate(positions):
        count = (p + 1) // ratio
        ids = rng.permutation(count)[:512]
        selected[t, : len(ids)] = ids
    records, counts = map(
        np.asarray,
        prepare_main_records(
            jnp.asarray(selected),
            jnp.asarray(full_pages),
            jnp.asarray(starts),
            page_size=page_records,
        ),
    )
    for t in range(tokens):
        ids = selected[t, : counts[t]]
        last = ids * ratio + ratio - 1
        full_loc = full_pages[t, last // 256] * 256 + last % 256
        np.testing.assert_array_equal(records[t, : counts[t]], full_loc // ratio)
        np.testing.assert_array_equal(records[t, counts[t] :], -1)
        assert counts[t] == min(512, (positions[t] + 1) // ratio)


def test_bf16_wo_a_matches_grouped_matmul():
    torch.manual_seed(0)
    groups, rank, heads = 8, 64, 16
    depth = heads * 512 // groups
    o = torch.randn(5, heads, 512, dtype=torch.bfloat16)
    weight = torch.randn(groups * rank, depth, dtype=torch.bfloat16) / depth**0.5
    expected = torch.cat(
        [
            o.reshape(5, groups, depth)[:, g].float()
            @ weight[g * rank : (g + 1) * rank].float().T
            for g in range(groups)
        ],
        dim=1,
    )
    actual = ops.wo_a(o, weight)
    assert actual.shape == (5, groups * rank) and actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected, atol=3e-2, rtol=2e-2)


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
@pytest.mark.parametrize("ratio", [0, 2])
def test_compiled_step_updates_caller_pools(tpu_ops, ratio):
    case = _Case(_PREFILL, ratio, seed=10 + ratio)
    inputs = _inputs(case)

    @functools.partial(torch.compile, backend="tpu", fullgraph=True, dynamic=False)
    def step(inputs):
        return _step(case, inputs)

    out, swa_records = step(inputs)
    _check(case, inputs, out, swa_records)


@requires_tpu
def test_fp8_wo_a_projection(tpu_ops):
    torch.manual_seed(0)
    tokens, heads, groups, rank = 16, 64, 8, 1024
    depth = heads * 512 // groups
    o = torch.randn(tokens, heads, 512, dtype=torch.bfloat16)
    weight = (torch.randn(depth, groups * rank) * 0.5).to(torch.float8_e4m3fn)
    scale = torch.rand(groups * rank) * 0.01 + 0.005
    expected = (
        torch.cat(
            [
                o.reshape(tokens, groups, depth)[:, g].float()
                @ weight[:, g * rank : (g + 1) * rank].float()
                for g in range(groups)
            ],
            dim=1,
        )
        * scale
    )
    actual = ops.wo_a(o.to("tpu"), weight.to("tpu"), scale.to("tpu")).cpu().float()
    assert actual.shape == (tokens, groups * rank)
    error = (actual - expected).norm() / expected.norm()
    # Activations are quantized to FP8 per row inside the kernel.
    assert error < 5e-2, error


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
