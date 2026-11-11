# SPDX-License-Identifier: Apache-2.0
"""Torch ops over the DeepSeek-V4.1 CSA2 JAX kernels (SWA, main KV, attention).

Single bank, one device per process (no mesh). Each op is a
``cpu_export_jax_op``: the JAX function is exported for TPU on the CPU and
called through TorchTPU. Pool-updating ops donate the pool and return its new
value; the caller writes it back with ``pool.copy_(result)``.

Layouts (``memory_pool.DSV41TPUTokenToKVPool``):

- SWA pages: BF16 ``[pages, 64, 512]``, page 0 null. The SWA table
  ``[B, C]`` holds, in column ``j``, the 64-token SWA page of request
  positions ``[64 j, 64 j + 64)`` (``swa_loc // 64``).
- Main KV: INT32 words ``[R, 128]`` (packed FP8, byte ``g`` of lane ``l`` is
  feature ``128 g + l``) plus FP32 per-record scales ``[R]``. Record
  ``full_loc // r``; one main page is ``256 // r`` records and its id is the
  full page id.
- ``cos_sin``: the layer's FP32 ``[max_position, 64]`` ``[cos | sin]`` table,
  ``sglang_tpu.dsv4.dsv4_memory_pool.get_or_build_cos_sin_cache(
  layer.freqs_cis)``. Pass the attention layer's own table: compressed layers
  use the compressed YaRN RoPE, ratio-0 layers the main RoPE. RoPE pairs are
  interleaved (``2 i``, ``2 i + 1``) over the last 64 of the 512 features.

The main-KV path takes logical compressed ids per token, resolves them to
records with ``prepare_main_records`` and runs ``csa2_pipeline``. The vllm
envs flags of the epic bridge keep their defaults: one-hot request lookup on,
blocked RoPE gather off (a plain gather), deep SparseCore gather on.
"""

# PEP 563 must stay off so cpu_export_jax_op sees real jax.Array annotations.

from threading import Lock

import jax
import jax.numpy as jnp
import torch

from sglang_tpu.models.deepseek_v41.kernels.csa2.cache import (
    encode_csa2_kv,
    prepare_main_records,
    prepare_swa_metadata,
    publish_csa2_kv,
)
from sglang_tpu.models.deepseek_v41.kernels.csa2.csa2 import csa2
from sglang_tpu.models.deepseek_v41.kernels.csa2.pipeline import (
    csa2_gather_chunk_size,
    csa2_pipeline,
)
from sglang_tpu.models.deepseek_v41.kernels.paged_swa import write_swa_pages

SWA_PAGE_SIZE = 64
SWA_WINDOW = 128
HEAD_DIM = 512
ROPE_DIM = 64
MAX_SELECTED = 512

_OP_PREFIX = "sglang_tpu_pallas::dsv41_csa2_"
_OPS: dict = {}
_OPS_LOCK = Lock()


# =============================================================================
# JAX bridges (traced by cpu_export_jax_op; non-array arguments are static)
# =============================================================================


def _swa_metadata(
    table: jax.Array, starts: jax.Array, positions: jax.Array, minimum: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return prepare_swa_metadata(
        table,
        starts,
        positions,
        minimum,
        page_size=SWA_PAGE_SIZE,
        page_stride=SWA_PAGE_SIZE,
        row_offset=0,
        non_causal=False,
        onehot_requests=True,
    )


def _publish_swa(pages: jax.Array, records: jax.Array, kv: jax.Array) -> jax.Array:
    return write_swa_pages(pages, records, kv)


def _publish_main(
    words: jax.Array, scales: jax.Array, records: jax.Array, latent: jax.Array
) -> tuple[jax.Array, jax.Array]:
    encoded, encoded_scales = encode_csa2_kv(latent)
    return publish_csa2_kv(words, scales, records, encoded, encoded_scales)


def _inverse_rope(cos_sin: jax.Array, positions: jax.Array) -> jax.Array:
    """CSA2's ``[T, 2, 128]`` rotation of the output's last 128 features.

    Q and K carry RoPE; out = conj(e^{i theta}) * o on each interleaved pair
    of the last 64 features. The first 64 of the 128 pass through.
    """
    rows = jnp.take(
        cos_sin, jnp.clip(positions, 0, cos_sin.shape[0] - 1), axis=0
    ).astype(jnp.float32)
    half = ROPE_DIM // 2
    tokens = positions.shape[0]
    cos = jnp.concatenate(
        (jnp.ones((tokens, 128 - ROPE_DIM)), jnp.repeat(rows[:, :half], 2, axis=1)),
        axis=1,
    )
    # Kernel: out[l] = o[l] * cos[l] + o[l ^ 1] * sin[l]; inverse rotation is
    # (re, im) -> (re c + im s, im c - re s).
    sin = jnp.repeat(rows[:, half:], 2, axis=1)
    sin *= jnp.where(jnp.arange(ROPE_DIM) % 2 == 0, 1.0, -1.0)
    sin = jnp.concatenate((jnp.zeros((tokens, 128 - ROPE_DIM)), sin), axis=1)
    return jnp.stack((cos, sin), axis=1).astype(jnp.float32)


def _bq(tokens: int) -> int:
    return 6 if tokens <= 384 else 16


def _attend_swa(
    q: jax.Array,
    swa: jax.Array,
    starts: jax.Array,
    positions: jax.Array,
    sink: jax.Array,
    cos_sin: jax.Array,
    metadata: jax.Array,
    scale: float,
) -> jax.Array:
    """Ratio-0 layers: attention over the SWA window alone."""
    return csa2(
        q,
        None,
        None,
        swa,
        starts,
        positions,
        sink.astype(jnp.float32),
        _inverse_rope(cos_sin, positions),
        swa_metadata=metadata,
        scale=scale,
        bq=_bq(q.shape[0]),
    )


def _attend_main(
    q: jax.Array,
    words: jax.Array,
    scales: jax.Array,
    selected: jax.Array,
    main_pages: jax.Array,
    swa: jax.Array,
    starts: jax.Array,
    positions: jax.Array,
    sink: jax.Array,
    cos_sin: jax.Array,
    metadata: jax.Array,
    main_page_records: int,
    compress_ratio: int,
    scale: float,
) -> jax.Array:
    """SWA window plus the selected compressed records of each token."""
    # As the epic caller: ids at or past the token's compressed length are
    # invalid. Selections must still form a valid prefix per row.
    lengths = (positions + 1) // compress_ratio
    selected = jnp.where(selected < lengths[:, None], selected, -1)
    records, counts = prepare_main_records(
        selected,
        main_pages,
        starts,
        page_size=main_page_records,
        onehot_requests=True,
    )
    output, _, _ = csa2_pipeline(
        q,
        words,
        scales,
        records,
        swa,
        starts,
        positions,
        sink.astype(jnp.float32),
        _inverse_rope(cos_sin, positions),
        metadata,
        None,
        counts,
        chunk_size=csa2_gather_chunk_size(q.shape[0]),
        bq=_bq(q.shape[0]),
        scale=scale,
        compress_ratio=compress_ratio,
        deep_gather=True,
    )
    return output


# =============================================================================
# Fakes (shape functions for torch.compile / FakeTensorMode)
# =============================================================================


def swa_metadata_columns(tokens: int) -> int:
    """Physical-page columns of ``prepare_swa_metadata`` for ``tokens`` rows."""
    return (tokens + SWA_WINDOW + SWA_PAGE_SIZE - 1) // SWA_PAGE_SIZE + 1


def _fake_swa_metadata(table, starts, positions, minimum):
    batch = table.shape[0]
    return (
        table.new_empty((batch, 2 + swa_metadata_columns(positions.shape[0]))),
        table.new_empty((batch, 2)),
        positions.new_empty(positions.shape),
    )


def _fake_publish_swa(pages, records, kv):
    return torch.empty_like(pages)


def _fake_publish_main(words, scales, records, latent):
    return torch.empty_like(words), torch.empty_like(scales)


def _fake_attend(q, *args):
    return torch.empty_like(q)


# name -> (JAX function, fake, donated argument positions)
_DEFINITIONS = {
    "swa_metadata": (_swa_metadata, _fake_swa_metadata, ()),
    "publish_swa": (_publish_swa, _fake_publish_swa, (0,)),
    "publish_main": (_publish_main, _fake_publish_main, (0, 1)),
    "attend_swa": (_attend_swa, _fake_attend, ()),
    "attend_main": (_attend_main, _fake_attend, ()),
}


def _op(name: str):
    op = _OPS.get(name)
    if op is not None:
        return op
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    with _OPS_LOCK:
        op = _OPS.get(name)
        if op is None:
            function, fake, donated = _DEFINITIONS[name]
            # Donation carries pool ownership through the Torch boundary;
            # without it publication copies the whole pool.
            op = cpu_export_jax_op(_OP_PREFIX + name, function, donate_argnums=donated)
            op.register_fake(fake)
            _OPS[name] = op
    return op


def register_csa2_ops() -> None:
    """Register every op; call before torch.compile traces the model."""
    for name in _DEFINITIONS:
        _op(name)


# =============================================================================
# Public interface
# =============================================================================


def _i32(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.int32 else x.to(torch.int32)


def swa_metadata(
    swa_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    minimum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-step SWA metadata, shared by every layer.

    ``swa_table`` [B, C] i32 (64-token SWA page ids), ``query_start_loc``
    [B+1] i32, ``positions`` [T] i32 (T >= query_start_loc[-1], rows past it
    are padding), ``minimum`` [B] i32 first visible position (0 unless
    replay/window state says otherwise).

    Returns ``meta`` [B, 2 + swa_metadata_columns(T)] i32, ``bounds`` [B, 2]
    i32 (noncausal only, unused here) and ``swa_write_records`` [T] i32
    (``swa_loc`` of each token; -1 on padding rows).
    """
    return _op("swa_metadata")(
        _i32(swa_table), _i32(query_start_loc), _i32(positions), _i32(minimum)
    )


def publish_swa(
    pages: torch.Tensor, records: torch.Tensor, kv: torch.Tensor
) -> torch.Tensor:
    """Write normed, RoPE'd BF16 ``kv`` [T, 512] at ``records`` [T] i32.

    Records < 64 (null page 0, padding -1) are dropped; valid records must be
    unique. Returns the new ``pages``; the caller does ``pages.copy_(...)``.
    """
    return _op("publish_swa")(pages, _i32(records), kv.reshape(-1, HEAD_DIM))


def publish_main(
    words: torch.Tensor,
    scales: torch.Tensor,
    records: torch.Tensor,
    latent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode post-RoPE BF16 ``latent`` [N, 512] to FP8 and publish it.

    ``records`` [N] i32 is ``full_loc // r`` per compressed record, -1 skips;
    valid records must be unique. Returns the new ``(words, scales)``.
    """
    return _op("publish_main")(words, scales, _i32(records), latent)


def attention(
    q: torch.Tensor,
    swa_pages: torch.Tensor,
    swa_meta: torch.Tensor,
    swa_bounds: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    sink: torch.Tensor,
    cos_sin: torch.Tensor,
    compress_ratio: int,
    scale: float,
    main: (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int] | None
    ) = None,
) -> torch.Tensor:
    """CSA2 attention; returns ``[T, H, 512]`` BF16 with inverse RoPE applied.

    ``q`` [T, H, 512] BF16 RoPE'd (H a multiple of 8 in [8, 128]); ``sink``
    [H] FP32; ``cos_sin`` see the module docstring. Visibility is causal: the
    last 128 positions of the request (from ``swa_meta``'s minimum) plus, with
    ``main``, the selected compressed records. ``swa_bounds`` is accepted for
    the noncausal (DSpark) contract and ignored. Padding rows return zeros.

    ``main`` = ``(words, scales, selected, main_pages, main_page_records)``:
    ``selected`` [T, <=512] i32 logical compressed ids, -1 padded, valid
    prefix, ids < (position + 1) // compress_ratio; ``main_pages`` [B, C'] i32
    full page ids; ``main_page_records`` = 256 // compress_ratio. ``None``
    (ratio 0) attends the SWA window alone.
    """
    del swa_bounds
    if not q.shape[0]:
        return torch.empty_like(q)
    if cos_sin.dim() != 2 or cos_sin.shape[1] != ROPE_DIM:
        raise ValueError(f"cos_sin must be [max_position, {ROPE_DIM}]")
    cos_sin = cos_sin.float()
    common = (_i32(query_start_loc), _i32(positions), sink.float(), cos_sin)
    if main is None:
        return _op("attend_swa")(q, swa_pages, *common, _i32(swa_meta), float(scale))
    if not compress_ratio:
        raise ValueError("main KV requires a positive compress_ratio")
    words, scales, selected, main_pages, main_page_records = main
    if selected.shape[1] > MAX_SELECTED:
        raise ValueError(f"CSA2 attends at most {MAX_SELECTED} compressed records")
    return _op("attend_main")(
        q,
        words,
        scales,
        _i32(selected),
        _i32(main_pages),
        swa_pages,
        *common,
        _i32(swa_meta),
        int(main_page_records),
        int(compress_ratio),
        float(scale),
    )
