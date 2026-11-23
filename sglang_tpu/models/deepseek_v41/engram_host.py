# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 Engram lookup from one host table shared by the node's ranks.

Every rank maps the same memfd copy of each layer's table, so under attention
DP a rank looks up its own tokens only: no id all-gather, no value all-reduce,
no table in HBM and no gather of a 100 GB table inside the compiled graph.

On the fast serving path (decode, extend with or without folded encoder SWA
bounded replay, speculative extend, mixed, and idle without beam/CP/PP):
1. ``HostEngramCoordinator.prepare_if_supported`` snapshots each scheduled
   chunk (including folded ``<= 128`` encoder SWA replay prefixes) and its
   ``n - 1`` predecessors on CPU without waiting for step N-1's in-flight
   sampled token (marking it as a hole in ``DeferredEngramInputs``).
2. Right before the compiled forward, ``stage_all`` flushes any deferred
   speculative verify history commits, resolves any hole (waiting for
   ``copy_done``), hashes on CPU with ``hash_on_host``, packs each layer's FP8
   rows and aligned E8M0 scale bytes into one flat ``int32`` buffer using
   single-threaded NumPy (``pack_rows``), and uploads via double-buffered async
   raw DMA (``PreparedEngramUpload``).
3. On speculative ``TARGET_VERIFY``, ``stage_verify`` reads host
   ``req_pool_indices_cpu`` and ``spec_info.live_seq_lens_cpu`` (skipping the
   ``positions`` D2H whenever all live requests already have ``>= n - 1``
   committed tokens) plus the drafted verify tokens, builds predecessor
   windows from ``hasher.history`` in single-threaded NumPy
   (``verify_inputs_from_host``), hashes via ``hash_on_host``, and stages
   through the same double-buffered async DMA slots, while
   ``defer_engram_commit`` defers post-verify history updates to the next host
   staging point so the host never stalls on verify acceptance.
4. After the compiled forward is dispatched, ``finish_forward`` prefetches the
   next same-size chunked-prefill chunk into the inactive double-buffer slots.
5. Inside the compiled graph, ``EngramHasher.forward`` returns cached
   ``local_row_ids`` (token indices ``0..T-1``) and ``staged_lookup`` unpacks
   and dequantizes the staged ``int32`` payload (``unpack_rows``), slicing rows
   first if a late-layer tail selected a subset of tokens.

Unsupported or warmup paths (non-verify warmup/precompile, non-folded two-pass
encoder SWA replay hits, beam search, PP/CP/DCP > 1) fall back cleanly to
``host_hash_ids`` + ``stage_from_hashes``, producing the exact same packed
``int32`` graph input signature with zero recompilation.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import weakref
from dataclasses import dataclass, is_dataclass
from types import SimpleNamespace
from typing import Any

import msgspec
import numpy as np
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

logger = logging.getLogger(__name__)

# Per-call inputs the host stage sets; None outside a staged V4.1 forward.
ForwardBatch.tpu_engram_local_ids = None
ForwardBatch.tpu_engram_staged = None

_INVALID_SCALE = 255
# The SGLang checkpoint layout uses one scale per 32 FP8 values.
_FP8_BLOCK_SIZE = 32
# Multimodal pad token IDs sit at or above MM_PAD_SHIFT_VALUE in SGLang.
_MM_PAD_SHIFT_VALUE = 1_000_000

# (num_tokens, num_layers, num_cols, device_str) -> cached [T, L, C] int32 ids.
_LOCAL_ROW_IDS: dict[tuple[int, int, int, str], torch.Tensor] = {}


def signed_sqrt(dot: torch.Tensor, clamp_value: float) -> torch.Tensor:
    """FP32 signed square root, including the sign of negative zero."""
    magnitude = dot.abs().clamp_min(clamp_value).sqrt()
    # Integer comparison reads the sign bit; a floating comparison loses -0.
    return torch.where(dot.view(torch.int32) < 0, -magnitude, magnitude)


@dataclass(frozen=True)
class HostEngramInputs:
    # Current token first, then predecessors. Negative positions are padding.
    tokens: np.ndarray
    positions: np.ndarray


class PendingTokens:
    """One sampled result, bound to request objects and logical positions.

    The scheduler owns result-copy submission. Waiting here observes its
    completion event; it does not advance request output_ids or token history.
    A request-pool slot alone cannot identify a request after cancellation.
    """

    def __init__(self, requests, positions, result):
        self.rows = {
            id(req): (req, int(position), row)
            for row, (req, position) in enumerate(zip(requests, positions, strict=True))
        }
        self.result = result
        self._tokens = None

    def read(self, req, position: int) -> int:
        entry = self.rows.get(id(req))
        if entry is None or entry[0] is not req or entry[1] != position:
            raise RuntimeError(
                f"Engram input token at position {position} is not available on host"
            )
        if self._tokens is None:
            event = self.result.copy_done
            if event is not None:
                # TPUAsyncCopyEvent waits for both worker submission and transfer.
                event.synchronize()
            tokens = self.result.next_token_ids
            if tokens is None:
                raise RuntimeError("Engram needs a completed sampled-token result")
            if isinstance(tokens, torch.Tensor):
                if event is not None and tokens.device.type != "cpu":
                    raise RuntimeError(
                        "Engram result copy completed without CPU tokens"
                    )
                # Non-overlap scheduling can leave the result tensor on device.
                tokens = tokens.detach().to("cpu").reshape(-1).tolist()
            self._tokens = tokens
        return int(self._tokens[entry[2]])


class DeferredEngramInputs(msgspec.Struct, frozen=True):
    """HostEngramInputs windows with holes for tokens whose copies are in flight.

    Each hole is ``(request, position, flat cell indices into tokens)``.
    """

    tokens: np.ndarray
    positions: np.ndarray
    holes: list

    def resolve(self, pending: PendingTokens | None) -> HostEngramInputs:
        """Fill the holes from the previous step's result; waits for its copy."""
        if self.holes:
            if pending is None:
                raise RuntimeError("Engram input token is not available on host")
            flat = self.tokens.reshape(-1)
            for req, position, cells in self.holes:
                flat[cells] = pending.read(req, position)
        return HostEngramInputs(tokens=self.tokens, positions=self.positions)


def capture_inputs(
    requests,
    starts: list[int],
    lengths: list[int],
    ngram: int,
    pending: PendingTokens | None = None,
) -> DeferredEngramInputs:
    """Snapshot only each scheduled chunk and its n-1 predecessors.

    ``starts`` must come from the scheduled batch (prefix_lens for extend,
    seq_lens_cpu - 1 for decode). Request output_ids can lag by one under
    overlap; a hole marks that missing token.
    Capture never waits for the previous step's sampled-token copy.
    """
    windows, all_positions, holes = [], [], []
    shifts = np.arange(ngram, dtype=np.int64)
    row_base = 0
    for req, start, length in zip(requests, starts, lengths, strict=True):
        if start < 0 or length < 0:
            raise ValueError("Engram chunk start and length must be non-negative")
        if not length:
            continue
        stop = start + length
        lo = max(0, start - ngram + 1)
        prompt, output = req.origin_input_ids, req.output_ids
        prompt_len = len(prompt)
        known_end = prompt_len + len(output)
        # Do not copy a whole million-token context on every decode step.
        source = list(prompt[lo : min(stop, prompt_len)])
        if stop > prompt_len:
            source.extend(output[max(0, lo - prompt_len) : stop - prompt_len])
        missing = []
        for position in range(max(lo, known_end), stop):
            if pending is None:
                raise RuntimeError(
                    f"Engram input token at position {position} is not available on host"
                )
            missing.append((position, len(source)))
            source.append(-1)
        positions = np.arange(start, stop, dtype=np.int64)
        past = positions[:, None] - shifts
        indices = np.maximum(past - lo, 0)
        windows.append(np.asarray(source, dtype=np.int64)[indices])
        all_positions.append(positions)
        for position, slot in missing:
            rows, cols = np.nonzero(indices == slot)
            holes.append((req, position, (row_base + rows) * ngram + cols))
        row_base += length
    return DeferredEngramInputs(
        tokens=(
            np.concatenate(windows) if windows else np.empty((0, ngram), dtype=np.int64)
        ),
        positions=(
            np.concatenate(all_positions)
            if all_positions
            else np.empty(0, dtype=np.int64)
        ),
        holes=holes,
    )


def _verify_positions_from_host(
    live_seq_lens_cpu: torch.Tensor | None,
    slots_np: np.ndarray,
    *,
    bs: int,
    block: int,
    ngram: int,
) -> np.ndarray | None:
    """Build verify positions on CPU without a ``positions`` D2H copy when safe.

    ``hash_on_host`` only checks ``positions < ngram - 1`` to block lookbacks
    before sequence start. Under 1-step overlap, ``live_seq_lens_cpu`` holds the
    host committed base (``<= device seq_lens``) for the ``bs_live`` live rows,
    while padded rows (slot 0) have ``seq_lens == 0``. Whenever every live row
    already has ``live_seq_lens_cpu >= ngram - 1``, ``device_pos >= ngram - 1``
    holds for every live verify token, so host-built positions yield identical
    lookback masks with zero D2H on ``positions``; shorter prompts fall back to
    the device ``positions`` D2H so 1-step overlap lag stays exact.
    """
    if (
        live_seq_lens_cpu is None
        or not isinstance(live_seq_lens_cpu, torch.Tensor)
        or live_seq_lens_cpu.device.type != "cpu"
        or live_seq_lens_cpu.ndim != 1
    ):
        return None
    bs_live = int(live_seq_lens_cpu.shape[0])
    if bs_live > bs or np.any(slots_np[bs_live:] != 0):
        return None
    live_np = np.asarray(live_seq_lens_cpu, dtype=np.int64)
    if np.any(live_np < ngram - 1):
        return None
    base_pos = np.zeros(bs, dtype=np.int64)
    base_pos[:bs_live] = live_np
    return (base_pos[:, None] + np.arange(block, dtype=np.int64)[None, :]).reshape(
        bs * block
    )


def verify_inputs_from_host(
    history: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    req_slots: torch.Tensor,
    *,
    block: int,
    ngram: int,
    live_seq_lens_cpu: torch.Tensor | None = None,
) -> HostEngramInputs:
    """Build ``[bs * block, ngram]`` verify windows on CPU in single-threaded NumPy."""
    if block <= 0 or ngram <= 0:
        raise ValueError("Verify block width and ngram size must be positive")
    bs = int(req_slots.shape[0])
    num_tokens = int(input_ids.shape[0])
    if num_tokens != bs * block or int(positions.shape[0]) != num_tokens:
        raise ValueError("Verify batch shapes must equal [bs * block]")
    if num_tokens == 0:
        return HostEngramInputs(
            tokens=np.empty((0, ngram), dtype=np.int64),
            positions=np.empty(0, dtype=np.int64),
        )
    hist_np = np.asarray(history.detach().cpu(), dtype=np.int64)
    if hist_np.ndim != 2 or hist_np.shape[1] != ngram - 1:
        raise ValueError("EngramHasher.history shape does not match ngram - 1")
    slots_np = np.asarray(req_slots.detach().cpu(), dtype=np.int64).reshape(bs)
    if np.any((slots_np < 0) | (slots_np >= hist_np.shape[0])):
        raise IndexError("Verify request slot is outside EngramHasher.history")
    pos_np = _verify_positions_from_host(
        live_seq_lens_cpu, slots_np, bs=bs, block=block, ngram=ngram
    )
    if pos_np is None:
        pos_np = np.asarray(positions.detach().cpu(), dtype=np.int64).reshape(
            num_tokens
        )
    ver_np = np.asarray(input_ids.detach().cpu(), dtype=np.int64).reshape(bs, block)
    window = np.concatenate([hist_np[slots_np], ver_np], axis=1)
    cols = (
        (ngram - 1)
        + np.arange(block, dtype=np.int64)[:, None]
        - np.arange(ngram, dtype=np.int64)[None, :]
    )
    tokens = window[:, cols].reshape(num_tokens, ngram)
    return HostEngramInputs(tokens=tokens, positions=pos_np)


def _resolve_extend_ranges(
    batch,
    *,
    enable_encoder_swa_bounded_replay: bool = False,
    folds_encoder_swa_replay: bool = False,
    has_request_window: bool = False,
) -> tuple[list[int], list[int], bool] | None:
    """Return ``(starts, lengths, any_folded)`` for an extend batch, or ``None``.

    Mirrors ``encoder_swa_replay.fold_encoder_swa_replay`` and
    ``run_encoder_swa_replay`` so folded and cache-miss extends under
    ``--enable-encoder-swa-bounded-replay`` stay on the zero-D2H host fast path.
    """
    reqs = batch.reqs
    prefix_lens = batch.prefix_lens
    extend_lens = batch.extend_lens
    if (
        reqs is None
        or prefix_lens is None
        or extend_lens is None
        or len(prefix_lens) != len(reqs)
        or len(extend_lens) != len(reqs)
    ):
        return None
    starts = [int(p) for p in prefix_lens]
    lengths = [int(n) for n in extend_lens]
    if not enable_encoder_swa_bounded_replay:
        return starts, lengths, False
    if not folds_encoder_swa_replay and not has_request_window:
        return None
    if not has_request_window:
        return starts, lengths, False
    resets = getattr(batch, "encoder_swa_reset", None)
    if resets is None or len(resets) != len(reqs):
        return None
    if not folds_encoder_swa_replay:
        # Separate replay forward runs iff a reset request has cached prefix tokens.
        if any(bool(rst) and p > 0 for rst, p in zip(resets, starts, strict=True)):
            return None
        return starts, lengths, False
    from sglang.srt.model_executor.encoder_swa_replay import replay_len

    if any(bool(rst) and (p % 2 != 0) for rst, p in zip(resets, starts, strict=True)):
        # Let core fold_encoder_swa_replay raise its odd-boundary ValueError.
        return None
    replays = [
        replay_len(p) if (bool(rst) and p > 0) else 0
        for rst, p in zip(resets, starts, strict=True)
    ]
    folded_starts = [p - r for p, r in zip(starts, replays, strict=True)]
    folded_lengths = [n + r for n, r in zip(lengths, replays, strict=True)]
    return folded_starts, folded_lengths, any(r > 0 for r in replays)


def hash_on_host(
    inputs: HostEngramInputs,
    *,
    token_map: np.ndarray,
    multipliers: np.ndarray,
    primes: np.ndarray,
    offsets: np.ndarray,
    pad_id: int,
    image_token_id: int | None = None,
    mm_pad_shift: int = _MM_PAD_SHIFT_VALUE,
) -> torch.Tensor:
    """The V4.1 int64 multiply/XOR/modulo hash, with compressed PAD ids."""
    tokens = inputs.tokens
    ngram = multipliers.shape[1]
    if tokens.shape != (len(inputs.positions), ngram):
        raise ValueError("Engram token windows do not match the hash layout")
    sentinel = (tokens < 0) | (tokens >= mm_pad_shift)
    if image_token_id is not None:
        sentinel = sentinel | (tokens == image_token_id)
    # Once a lookback window hits sequence start or an image/pad sentinel,
    # that shift and every older predecessor in the window map to PAD.
    blocked = np.maximum.accumulate(
        (inputs.positions[:, None] < np.arange(ngram)) | sentinel,
        axis=1,
    )
    if np.any((~blocked) & ((tokens < 0) | (tokens >= len(token_map)))):
        raise ValueError("Engram token is outside the compressed token map")
    safe_tokens = np.where(blocked, 0, tokens)
    source = np.where(blocked, pad_id, token_map[safe_tokens])
    terms = source[:, None, :] * multipliers[None, :, :]
    rolled = np.bitwise_xor.accumulate(terms, axis=2)[:, :, 1:]
    hashed = (rolled[:, :, :, None] % primes[None, :, :, :]).reshape(
        len(tokens), multipliers.shape[0], offsets.shape[1]
    ) + offsets[None, :, :]
    return torch.from_numpy(np.ascontiguousarray(hashed))


def packed_row_words(
    heads: int,
    dim: int,
    padded_tokens: int,
    block_size: int = _FP8_BLOCK_SIZE,
) -> int:
    if padded_tokens < 0 or dim % block_size or dim % 4:
        raise ValueError("Invalid Engram staging shape")
    blocks = dim // block_size
    raw_bytes = heads * dim
    scale_bytes = heads * blocks
    row_bytes = raw_bytes + ((scale_bytes + 3) // 4) * 4
    return padded_tokens * (row_bytes // 4)


def pack_rows(
    weight: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    row_start: int,
    padded_tokens: int,
    block_size: int = _FP8_BLOCK_SIZE,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather CPU bytes into one flat int32 upload, with zero-output padding.

    Single-threaded NumPy, not torch: torch's intra-op thread pool made this
    gather cost milliseconds on a host shared by 8 scheduler processes.
    """
    if sys.byteorder != "little":
        raise RuntimeError("Packed Engram staging requires a little-endian host")
    if any(t.device.type != "cpu" for t in (weight, scale, indices)):
        raise ValueError("Engram row gathering requires CPU tensors")
    tokens, heads = indices.shape
    rows, dim = weight.shape
    if padded_tokens < tokens or dim % block_size or dim % 4:
        raise ValueError("Invalid Engram staging shape")
    blocks = dim // block_size
    if scale.shape != (rows, blocks):
        raise ValueError("Engram scale shape does not match the FP8 table")
    raw_bytes = heads * dim
    scale_bytes = heads * blocks
    # Interleave payload and scales per token, so graph padding has whole rows.
    row_bytes = raw_bytes + ((scale_bytes + 3) // 4) * 4
    words = padded_tokens * (row_bytes // 4)
    if out is None:
        packed = np.zeros((padded_tokens, row_bytes), dtype=np.uint8)
        result = torch.from_numpy(packed).view(torch.int32).reshape(-1)
    else:
        if (
            out.device.type != "cpu"
            or out.dtype != torch.int32
            or out.shape != (words,)
        ):
            raise ValueError("Engram output buffer does not match packed shape")
        packed = out.view(torch.uint8).numpy().reshape(padded_tokens, row_bytes)
        if tokens < padded_tokens or not rows:
            packed[tokens if rows else 0 :, :raw_bytes] = 0
        result = out
    packed[:, raw_bytes:] = _INVALID_SCALE
    if tokens and rows:
        local = indices.detach().numpy().astype(np.int64, copy=False) - row_start
        owned = (local >= 0) & (local < rows)
        local = np.clip(local, 0, rows - 1).reshape(-1)
        raw = np.take(weight.detach().view(torch.uint8).numpy(), local, axis=0)
        scales = np.take(scale.detach().view(torch.uint8).numpy(), local, axis=0)
        scales[~owned.reshape(-1)] = _INVALID_SCALE
        packed[:tokens, :raw_bytes] = raw.reshape(tokens, raw_bytes)
        packed[:tokens, raw_bytes : raw_bytes + scale_bytes] = scales.reshape(
            tokens, scale_bytes
        )
    return result


def unpack_rows(
    packed: torch.Tensor,
    *,
    tokens: int,
    heads: int,
    dim: int,
    block_size: int = _FP8_BLOCK_SIZE,
) -> torch.Tensor:
    """Device-side inverse of ``pack_rows``: dequantize FP8 + E8M0 bytes to bf16.

    Integer e8m0 dequant: the power of two is ``(byte << 23)`` bitcast to fp32,
    exact for bytes 1..254 (byte 0 gives 0 instead of 2**-127; byte 255 marks
    padded/unowned rows and yields exact 0.0 even if raw FP8 bytes are NaN).
    """
    raw_bytes = heads * dim
    blocks = dim // block_size
    scale_bytes = heads * blocks
    shifts = torch.arange(0, 32, 8, dtype=torch.int32, device=packed.device)
    unpacked = ((packed.unsqueeze(-1) >> shifts) & 0xFF).to(torch.uint8)
    row_bytes = raw_bytes + ((scale_bytes + 3) // 4) * 4
    unpacked = unpacked.reshape(tokens, row_bytes)
    raw = unpacked[:, :raw_bytes].contiguous().view(tokens, heads, dim)
    scale = unpacked[:, raw_bytes : raw_bytes + scale_bytes].contiguous()
    scale = scale.view(tokens, heads, blocks)
    values = raw.view(torch.float8_e4m3fn).float().unflatten(-1, (blocks, block_size))
    power = (scale.to(torch.int32) << 23).view(torch.float32)
    invalid = (scale == _INVALID_SCALE).unsqueeze(-1)
    scaled = torch.where(invalid, 0.0, values * power.unsqueeze(-1))
    return scaled.flatten(-2).to(torch.bfloat16)


def _physical_words(tensor: torch.Tensor, layout_api: Any = None) -> int:
    if layout_api is None:
        from torch_tpu._internal.compile import tpu_torch_compile as layout_api

    layout = layout_api.get_device_layout_if_materialized(tensor)
    if layout is None:
        raise RuntimeError("Engram raw DMA requires a materialized int32 buffer")
    order, tiles, element_bits = layout
    if list(order) != [0] or element_bits not in (0, 32):
        raise NotImplementedError(f"Unsupported Engram raw int32 layout: {layout}")
    words = tensor.numel()
    for tile in tiles:
        if len(tile) != 1 or tile[0] <= 0:
            raise NotImplementedError(f"Unsupported Engram raw 1-D tile: {layout}")
        words = -(-words // tile[0]) * tile[0]
    return words


class _CpuTransfer:
    """In-memory async transfer double for CPU execution and unit tests."""

    def __init__(self, device_words: torch.Tensor, host_words: torch.Tensor) -> None:
        self.device_words = device_words
        self.host_words = host_words

    def d2h(self) -> None:
        self.host_words.copy_(self.device_words)

    def h2d_async(self) -> Any:
        def wait() -> None:
            self.device_words.copy_(self.host_words)

        return SimpleNamespace(wait=wait)


class _BatchTransfer:
    """Raw DMA adapter backed by ``torch_tpu._internal.batch_transfer``.

    Uses TorchTPU's in-tree ``PJRT_RawBuffer`` bindings directly instead of
    ``tpu_sync.api.torch.raw_transfer`` so host<->device raw DMA stays
    ABI-compatible with ``torch_tpu``'s XLA runtime across wheel upgrades.
    """

    def __init__(
        self,
        batch_transfer_mod: Any,
        device_tensor: torch.Tensor,
        host_buffer: torch.Tensor,
    ) -> None:
        self._mod = batch_transfer_mod
        self._device = [device_tensor]
        self._host = [host_buffer]

    def d2h(self) -> None:
        self._mod.batch_transfer_d2h_sync(self._device, self._host)

    def h2d_async(self) -> Any:
        return self._mod.batch_transfer_h2d(self._host, self._device)


class PreparedEngramUpload:
    """One immutable device allocation, its host storage and consumer lifetime."""

    def __init__(
        self,
        num_words: int,
        device: torch.device | str,
        *,
        batch_transfer_mod: Any = None,
        layout_mod: Any = None,
        sync_fn: Any = None,
    ) -> None:
        dev = torch.device(device)
        self._sync_fn = sync_fn
        self.pending: Any = None
        self.consumer_event: Any = None
        self.consumers: list[torch.Tensor] = []
        self.needs_consumer = False
        if num_words == 0 or (dev.type == "cpu" and batch_transfer_mod is None):
            self.device = torch.zeros(num_words, dtype=torch.int32, device=dev)
            self.host_buffer = torch.zeros(num_words, dtype=torch.int32, device="cpu")
            self.host_words = self.host_buffer
            self.transfer = _CpuTransfer(self.device, self.host_words)
            return
        if batch_transfer_mod is None:
            from torch_tpu._internal import batch_transfer as batch_transfer_mod
        if self._sync_fn is None:
            from torch_tpu._internal.sync.sync import synchronize as _tpu_sync

            self._sync_fn = _tpu_sync
        probe = torch.arange(num_words, dtype=torch.int32, device="cpu")
        probe.bitwise_xor_(0x13579BDF)
        self.device = probe.to(dev)
        phys_words = _physical_words(self.device, layout_mod)
        # Allocate pinned host memory from PJRT's HostMemoryAllocator on TPU so
        # PJRT_RawBuffer_CopyRawHostToDevice performs direct zero-copy DMA.
        self.host_buffer = torch.zeros(
            phys_words,
            dtype=torch.int32,
            device="cpu",
            pin_memory=(dev.type != "cpu"),
        )
        self.host_words = self.host_buffer[:num_words]
        self.transfer = _BatchTransfer(
            batch_transfer_mod, self.device, self.host_buffer
        )
        self.transfer.d2h()
        if not torch.equal(self.host_words, probe):
            raise RuntimeError("Engram raw layout validation failed")

    def acquire(self) -> torch.Tensor:
        self.wait()
        return self.host_words

    def release_consumers(self) -> None:
        if self.consumer_event is not None:
            with torch.profiler.record_function("engram::slot_consumer_wait"):
                self.consumer_event.synchronize()
            self.consumer_event = None
        if self.consumers:
            if self._sync_fn is not None:
                with torch.profiler.record_function("engram::slot_consumer_wait"):
                    self._sync_fn(self.consumers, wait=True)
            self.consumers = []

    def start(self) -> None:
        if self.needs_consumer:
            raise RuntimeError("Record the Engram consumer before overwriting its slot")
        self.release_consumers()
        with torch.profiler.record_function("engram::h2d_submit"):
            self.pending = self.transfer.h2d_async()

    def wait(self) -> None:
        if self.pending is not None:
            with torch.profiler.record_function("engram::h2d_wait"):
                self.pending.wait()
            self.pending = None

    def activate(self) -> None:
        self.wait()
        self.needs_consumer = True

    def record_consumer_event(self, event: Any) -> None:
        if self.needs_consumer:
            if event is None:
                raise ValueError("Engram slot reuse requires a consumer event")
            self.consumer_event = event
            self.consumers = []
            self.needs_consumer = False

    def record_consumers(self, tensors: list[torch.Tensor]) -> None:
        if self.needs_consumer:
            if not tensors:
                raise ValueError("Engram slot reuse requires a consumer output")
            self.consumer_event = None
            self.consumers = tensors
            self.needs_consumer = False

    def copy_from(self, words: torch.Tensor) -> None:
        if words.shape != self.host_words.shape:
            raise ValueError("Engram upload must match the slot's token bucket")
        self.acquire().copy_(words)
        self.start()
        self.wait()


@dataclass
class _EngramPrefetch:
    requests: tuple[Any, ...]
    starts: list[int]
    lengths: list[int]
    inputs: HostEngramInputs
    hashes: torch.Tensor
    padded_tokens: int
    uploads: dict[int, PreparedEngramUpload]

    def matches(
        self,
        requests: list[Any],
        starts: list[int],
        lengths: list[int],
        ngram: int,
    ) -> bool:
        if (
            len(requests) != len(self.requests)
            or starts != self.starts
            or lengths != self.lengths
            or any(a is not b for a, b in zip(requests, self.requests, strict=True))
        ):
            return False
        if any(
            start + length > len(req.origin_input_ids)
            for req, start, length in zip(requests, starts, lengths, strict=True)
        ):
            return False
        actual = capture_inputs(requests, starts, lengths, ngram)
        return (
            not actual.holes
            and np.array_equal(actual.tokens, self.inputs.tokens)
            and np.array_equal(actual.positions, self.inputs.positions)
        )

    def discard(self) -> None:
        for upload in self.uploads.values():
            upload.wait()
        self.uploads.clear()


class HostEngramCoordinator:
    """Overlap host Engram input capture, NumPy row packing, and async raw DMA."""

    def __init__(
        self,
        hasher,
        embeddings: list[tuple[int, Any]],
        device: torch.device | str,
        *,
        batch_transfer_mod: Any = None,
        layout_mod: Any = None,
        sync_fn: Any = None,
        event_cls: Any = None,
    ):
        self._hasher = hasher
        self.ngram = hasher.max_ngram_size
        self.tables = {
            "token_map": np.asarray(hasher.token_map.detach().cpu(), dtype=np.int64),
            "multipliers": np.asarray(
                hasher.multipliers.detach().cpu(), dtype=np.int64
            ),
            "primes": np.asarray(hasher.primes.detach().cpu(), dtype=np.int64),
            "offsets": np.asarray(hasher.offsets.detach().cpu(), dtype=np.int64),
        }
        self.pad_id = hasher.pad_id
        self.image_token_id = hasher.image_token_id
        self.embeddings = list(embeddings)
        self.by_layer = dict(embeddings)
        self.device = torch.device(device)
        self._batch_transfer_mod = batch_transfer_mod
        self._layout_mod = layout_mod
        if event_cls is None and sync_fn is None and self.device.type == "tpu":
            event_cls = torch.get_device_module("tpu").Event
        self._event_cls = event_cls
        if sync_fn is None and self.device.type != "cpu":
            from torch_tpu._internal.sync.sync import synchronize as sync_fn
        self._sync_fn = sync_fn
        self.pending: PendingTokens | None = None
        self.hashes: torch.Tensor | None = None
        self.started = False
        # This step's windows while the previous step's sampled token is in flight.
        self.deferred: DeferredEngramInputs | None = None
        self._step_inputs: HostEngramInputs | None = None
        self._history_commit: list[tuple[int, int]] = []
        # Double-buffered upload slots per (layer_index, padded_tokens) bucket.
        self._uploads_by_layer: dict[int, dict[int, list[PreparedEngramUpload]]] = {
            layer_index: {} for layer_index, _ in self.embeddings
        }
        self._active_uploads: dict[int, PreparedEngramUpload] = {}
        self._last_bucket_upload: dict[tuple[int, int], PreparedEngramUpload] = {}
        self._staged_uploads: dict[int, PreparedEngramUpload] = {}
        self._staged_padded_tokens: int | None = None
        self._next_prompt_spec: tuple[tuple[Any, ...], list[int], list[int]] | None = (
            None
        )
        self._prepared_extend_ranges: tuple[list[int], list[int]] | None = None
        self._prefetch: _EngramPrefetch | None = None

    def _next_prompt_prediction(
        self,
        requests: list[Any],
        starts: list[int],
        lengths: list[int],
    ) -> tuple[tuple[Any, ...], list[int], list[int]] | None:
        if not requests:
            return None
        next_starts = []
        for req, start, length in zip(requests, starts, lengths, strict=True):
            if length <= 0 or req.output_ids:
                return None
            next_start = start + length
            if next_start + length > len(req.origin_input_ids):
                return None
            next_starts.append(next_start)
        return (tuple(requests), next_starts, list(lengths))

    def _reserve_upload(
        self, layer_index: int, padded_tokens: int
    ) -> PreparedEngramUpload:
        by_tokens = self._uploads_by_layer[layer_index]
        slots = by_tokens.get(padded_tokens)
        if slots is None:
            embedding = self.by_layer[layer_index]
            heads = self.tables["offsets"].shape[1]
            words = packed_row_words(heads, embedding.weight.shape[1], padded_tokens)
            slots = [
                PreparedEngramUpload(
                    words,
                    self.device,
                    batch_transfer_mod=self._batch_transfer_mod,
                    layout_mod=self._layout_mod,
                    sync_fn=self._sync_fn,
                )
                for _ in range(2)
            ]
            by_tokens[padded_tokens] = slots
        last = self._last_bucket_upload.get((layer_index, padded_tokens))
        return next(slot for slot in slots if slot is not last)

    def _start_layer_upload(
        self,
        layer_index: int,
        padded_tokens: int,
        hashes: torch.Tensor,
    ) -> PreparedEngramUpload:
        embedding = self.by_layer[layer_index]
        upload = self._reserve_upload(layer_index, padded_tokens)
        pack_rows(
            embedding.weight,
            embedding.scale,
            hashes[:, layer_index],
            row_start=0,
            padded_tokens=padded_tokens,
            out=upload.acquire(),
        )
        upload.start()
        return upload

    def _discard_staged_uploads(self) -> None:
        for upload in self._staged_uploads.values():
            upload.wait()
        self._staged_uploads.clear()
        self._staged_padded_tokens = None

    def _discard_prefetch(self) -> None:
        if self._prefetch is not None:
            self._prefetch.discard()
            self._prefetch = None

    def _reset_step_inputs(self) -> None:
        self._discard_staged_uploads()
        self._discard_prefetch()
        self.hashes = None
        self.deferred = None
        self._step_inputs = None
        self._history_commit = []
        self._next_prompt_spec = None
        self._prepared_extend_ranges = None
        self.started = False

    @staticmethod
    def _batch_req_slots(batch, requests: list[Any]) -> list[int | None]:
        raw = batch.req_pool_indices_cpu
        if raw is not None:
            values = raw.tolist() if isinstance(raw, torch.Tensor) else list(raw)
            if len(values) >= len(requests):
                return [int(v) for v in values[: len(requests)]]
        return [
            None if req.kv.req_pool_idx is None else int(req.kv.req_pool_idx)
            for req in requests
        ]

    def _commit_history(self) -> None:
        """Mirror fast-path n-1 token tails into ``hasher.history`` on CPU.

        Keeps ``EngramHasher.history`` current when a later decode or
        speculative verify step reads ``hasher.history`` on CPU. Any queued
        verify history commits are flushed first so a freed slot reused by an
        extend batch is never overwritten out of order.
        """
        flush_engram_commits(self._hasher)
        if not self._history_commit or self._step_inputs is None:
            return
        history = self._hasher.history
        if history is None:
            return
        n1 = self.ngram - 1
        if n1 <= 0 or history.shape[1] != n1:
            return
        num_slots = history.shape[0]
        valid = [
            (slot, row) for slot, row in self._history_commit if 0 <= slot < num_slots
        ]
        if not valid:
            return
        slots = [slot for slot, _ in valid]
        rows = np.asarray([row for _, row in valid], dtype=np.int64)
        shifts = np.arange(n1, dtype=np.int64)
        tail = np.where(
            self._step_inputs.positions[rows, None] >= shifts,
            np.minimum(self._step_inputs.tokens[rows, :n1], _MM_PAD_SHIFT_VALUE),
            0,
        )[:, ::-1]
        tail_tensor = torch.as_tensor(
            np.ascontiguousarray(tail), dtype=history.dtype, device=history.device
        )
        slot_tensor = torch.tensor(slots, dtype=torch.int64, device=history.device)
        history[slot_tensor] = tail_tensor
        self._history_commit = []

    def prepare(
        self,
        batch,
        *,
        starts: list[int] | None = None,
        lengths: list[int] | None = None,
        any_folded: bool = False,
    ) -> tuple[list, list[int]]:
        """Capture this batch's host Engram windows (raising if unsupported)."""
        spec_active = (
            batch.spec_algorithm is not None and not batch.spec_algorithm.is_none()
        )
        if any(req.beam_group is not None for req in batch.reqs):
            raise NotImplementedError("Host Engram does not yet support beam search")
        mode = batch.forward_mode
        is_plain_extend = (
            mode.is_extend_without_speculative()
            and not getattr(mode, "is_dllm_extend", lambda: False)()
            and not getattr(mode, "is_split_prefill", lambda: False)()
        )
        if mode.is_decode():
            if spec_active:
                raise NotImplementedError(
                    "Host Engram decode capture does not serve speculative decode"
                )
            requests = list(batch.reqs)
            starts = [int(n) - 1 for n in batch.seq_lens_cpu.tolist()]
            lengths = [1] * len(requests)
        elif is_plain_extend:
            if spec_active and getattr(mode, "is_mixed", lambda: False)():
                raise NotImplementedError(
                    "Host Engram does not support mixed speculative extend"
                )
            requests = list(batch.reqs)
            if starts is None or lengths is None:
                starts = [int(p) for p in batch.prefix_lens]
                lengths = [int(n) for n in batch.extend_lens]
            if spec_active and any(
                s + n > len(req.origin_input_ids) + len(req.output_ids)
                for req, s, n in zip(requests, starts, lengths, strict=True)
            ):
                raise NotImplementedError(
                    "Speculative extend requires host-resident chunk tokens"
                )
        elif mode.is_idle():
            requests, starts, lengths = [], [], []
        else:
            raise NotImplementedError(f"Host Engram does not support {mode}")
        self._discard_staged_uploads()
        self._next_prompt_spec = (
            self._next_prompt_prediction(requests, starts, lengths)
            if is_plain_extend and not any_folded
            else None
        )
        self._prepared_extend_ranges = (
            (list(starts), list(lengths)) if is_plain_extend else None
        )
        row_end = 0
        commit: list[tuple[int, int]] = []
        for slot, length in zip(
            self._batch_req_slots(batch, requests), lengths, strict=True
        ):
            if length <= 0:
                continue
            row_end += length
            if slot is not None:
                commit.append((slot, row_end - 1))
        self._history_commit = commit
        prefetched = self._prefetch
        self._prefetch = None
        if (
            prefetched is not None
            and is_plain_extend
            and prefetched.matches(requests, starts, lengths, self.ngram)
        ):
            self.hashes = prefetched.hashes
            self._step_inputs = prefetched.inputs
            self.deferred = None
            self._staged_uploads = dict(prefetched.uploads)
            self._staged_padded_tokens = prefetched.padded_tokens
        else:
            if prefetched is not None:
                prefetched.discard()
            self.hashes = None
            self._step_inputs = None
            self.deferred = capture_inputs(
                requests, starts, lengths, self.ngram, self.pending
            )
            if not self.deferred.holes:
                self.resolve()
        self.started = True
        return requests, [s + n for s, n in zip(starts, lengths, strict=True)]

    def prepare_if_supported(
        self,
        batch,
        *,
        parallel: Any = None,
        enable_encoder_swa_bounded_replay: bool = False,
        folds_encoder_swa_replay: bool = False,
        has_request_window: bool = False,
        is_dllm: bool = False,
    ) -> tuple[list, list[int]] | None:
        """Capture fast-path inputs when supported; otherwise reset for fallback."""
        if batch is None or is_dllm:
            self._reset_step_inputs()
            return None
        if parallel is not None and (
            parallel.pp_size != 1
            or parallel.attn_cp_size != 1
            or parallel.attn_dcp_size != 1
        ):
            self._reset_step_inputs()
            return None
        if batch.reqs is None or any(req.beam_group is not None for req in batch.reqs):
            self._reset_step_inputs()
            return None
        spec_active = (
            batch.spec_algorithm is not None and not batch.spec_algorithm.is_none()
        )
        mode = batch.forward_mode
        is_plain_extend = (
            mode.is_extend_without_speculative()
            and not getattr(mode, "is_dllm_extend", lambda: False)()
            and not getattr(mode, "is_split_prefill", lambda: False)()
        )
        if mode.is_decode():
            if (
                spec_active
                or batch.seq_lens_cpu is None
                or len(batch.seq_lens_cpu) != len(batch.reqs)
            ):
                self._reset_step_inputs()
                return None
            return self.prepare(batch)
        if is_plain_extend:
            if spec_active and getattr(mode, "is_mixed", lambda: False)():
                self._reset_step_inputs()
                return None
            resolved = _resolve_extend_ranges(
                batch,
                enable_encoder_swa_bounded_replay=enable_encoder_swa_bounded_replay,
                folds_encoder_swa_replay=folds_encoder_swa_replay,
                has_request_window=has_request_window,
            )
            if resolved is None:
                self._reset_step_inputs()
                return None
            starts, lengths, any_folded = resolved
            if spec_active and any(
                s + n > len(req.origin_input_ids) + len(req.output_ids)
                for req, s, n in zip(batch.reqs, starts, lengths, strict=True)
            ):
                self._reset_step_inputs()
                return None
            return self.prepare(
                batch, starts=starts, lengths=lengths, any_folded=any_folded
            )
        if mode.is_idle():
            return self.prepare(batch)
        self._reset_step_inputs()
        return None

    def resolve(self) -> torch.Tensor | None:
        """Fill overlap holes (waiting for copy_done) and hash on host.

        Later calls in the same step return the same hashes without waiting.
        """
        if self.deferred is not None:
            inputs = self.deferred.resolve(self.pending)
            self._step_inputs = inputs
            self.hashes = hash_on_host(
                inputs,
                **self.tables,
                pad_id=self.pad_id,
                image_token_id=self.image_token_id,
            )
            self.deferred = None
        return self.hashes

    def _matches_forward_extend_ranges(self, forward_batch: Any) -> bool:
        """Verify predicted extend ranges match the actually-folded ForwardBatch."""
        if forward_batch is None or self._prepared_extend_ranges is None:
            return True
        if getattr(forward_batch, "encoder_swa_replay", False):
            return False
        expected_starts, expected_lengths = self._prepared_extend_ranges
        n_reqs = len(expected_lengths)
        for attr, expected in (
            ("extend_seq_lens_cpu", expected_lengths),
            ("extend_prefix_lens_cpu", expected_starts),
        ):
            actual = getattr(forward_batch, attr, None)
            if actual is None:
                continue
            if isinstance(actual, torch.Tensor):
                if actual.device.type != "cpu":
                    continue
                actual = actual.tolist()
            if len(actual) < n_reqs:
                return False
            if [int(x) for x in actual[:n_reqs]] != expected:
                return False
            if any(int(x) != 0 for x in actual[n_reqs:]):
                return False
        return True

    def _stage_layers(
        self, hashes: torch.Tensor, padded_tokens: int, *, use_prestaged: bool
    ) -> tuple[tuple[torch.Tensor, ...], list[PreparedEngramUpload], int]:
        if hashes.shape[0] > padded_tokens:
            raise ValueError("Host Engram rows exceed the padded local token count")
        if not use_prestaged:
            # Verify and fallback staging bypass prepare(), so any pending
            # next-chunk prefetch must be dropped before _reserve_upload can
            # reuse its slot on the same padded token bucket.
            self._discard_prefetch()
        if not use_prestaged or (
            self._staged_uploads
            and self._staged_padded_tokens is not None
            and self._staged_padded_tokens != padded_tokens
        ):
            self._discard_staged_uploads()
        # Pass 1: start any un-staged layers immediately so layer i's H2D DMA
        # overlaps with layer i+1's single-threaded NumPy pack_rows.
        uploads: list[tuple[int, PreparedEngramUpload]] = []
        for layer_index, _ in self.embeddings:
            upload = (
                self._staged_uploads.pop(layer_index, None) if use_prestaged else None
            )
            if upload is None:
                upload = self._start_layer_upload(layer_index, padded_tokens, hashes)
            uploads.append((layer_index, upload))
        self._staged_padded_tokens = None
        # Pass 2: wait for every layer's H2D DMA and mark slots active.
        ordered: list[PreparedEngramUpload] = []
        for layer_index, upload in uploads:
            upload.activate()
            self._active_uploads[layer_index] = upload
            self._last_bucket_upload[(layer_index, padded_tokens)] = upload
            ordered.append(upload)
        num_cols = int(hashes.shape[2])
        return tuple(upload.device for upload in ordered), ordered, num_cols

    def stage_all(
        self, padded_tokens: int, *, forward_batch: Any = None
    ) -> tuple[tuple[torch.Tensor, ...], list[PreparedEngramUpload], int] | None:
        """Resolve deferred overlap tokens (if any) and stage every Engram layer."""
        if not self._matches_forward_extend_ranges(forward_batch):
            logger.warning(
                "Host Engram predicted extend ranges %s did not match ForwardBatch; "
                "falling back to host_hash_ids.",
                self._prepared_extend_ranges,
            )
            self._reset_step_inputs()
            return None
        hashes = self.resolve()
        if hashes is None:
            if self.started:
                raise RuntimeError("Host Engram forward has no prepared request inputs")
            hashes = torch.empty(
                (0, *self.tables["offsets"].shape), dtype=torch.int64, device="cpu"
            )
        self._commit_history()
        return self._stage_layers(hashes, padded_tokens, use_prestaged=True)

    def stage_verify(
        self, input_ids: torch.Tensor, forward_batch
    ) -> tuple[tuple[torch.Tensor, ...], list[PreparedEngramUpload], int] | None:
        """Hash and stage a ``TARGET_VERIFY`` step on CPU in single-threaded NumPy."""
        self._discard_prefetch()
        flush_engram_commits(self._hasher)
        if not forward_batch.forward_mode.is_target_verify():
            return None
        history = self._hasher.history
        spec_info = getattr(forward_batch, "spec_info", None)
        # Upstream keeps the ragged layout on the forward batch, not spec_info.
        assert getattr(forward_batch, "ragged_verify_layout", None) is None, (
            "Host Engram stage_verify assumes uniform draft_token_num per request "
            "and does not yet support ragged verify"
        )
        block = int(getattr(spec_info, "draft_token_num", 0) or 0)
        if history is None or block <= 0 or history.shape[1] != self.ngram - 1:
            return None
        req_slots = getattr(forward_batch, "req_pool_indices_cpu", None)
        if (
            req_slots is None
            or not isinstance(req_slots, torch.Tensor)
            or req_slots.device.type != "cpu"
            or req_slots.shape != forward_batch.req_pool_indices.shape
        ):
            req_slots = _cpu(forward_batch.req_pool_indices)
        padded_tokens = int(input_ids.shape[0])
        bs = int(req_slots.shape[0])
        assert padded_tokens == bs * block, (
            "Host Engram stage_verify expects one equal block per request "
            f"(ragged verify not yet supported), got {padded_tokens} tokens for "
            f"{bs} requests of {block}"
        )
        if int(forward_batch.positions.shape[0]) != padded_tokens:
            return None
        try:
            inputs = verify_inputs_from_host(
                history,
                input_ids,
                forward_batch.positions,
                req_slots,
                block=block,
                ngram=self.ngram,
                live_seq_lens_cpu=getattr(spec_info, "live_seq_lens_cpu", None),
            )
        except (ValueError, IndexError):
            return None
        hashes = hash_on_host(
            inputs,
            **self.tables,
            pad_id=self.pad_id,
            image_token_id=self.image_token_id,
        )
        return self._stage_layers(hashes, padded_tokens, use_prestaged=False)

    def stage_from_hashes(
        self, hashes: torch.Tensor, padded_tokens: int
    ) -> tuple[tuple[torch.Tensor, ...], list[PreparedEngramUpload], int]:
        """Stage precomputed CPU hash ids through the same packed double buffers."""
        return self._stage_layers(hashes, padded_tokens, use_prestaged=False)

    def make_consumer_event(self) -> Any:
        """Record a stream completion event on TPU without retaining output tensors.

        Holding output ``next_token_logits`` (e.g. ``[64, 129280]`` float32 =
        33 MiB per slot) across inactive bucket slots pins hundreds of MiB of
        HBM; a recorded TPU ``Event`` tracks the PJRT execution future directly
        with zero HBM buffer retention and flushes the stream for prefetching.
        """
        if self._event_cls is None:
            return None
        event = self._event_cls()
        event.record()
        return event

    def record_consumer_event(
        self,
        uploads: list[PreparedEngramUpload],
        event: Any,
    ) -> None:
        """Bind a stream completion event to active slots without pinning outputs."""
        for upload in uploads:
            if upload.needs_consumer:
                upload.record_consumer_event(event)

    def bind_consumer_event(
        self,
        uploads: list[PreparedEngramUpload],
        event: Any,
    ) -> None:
        """Bind a deferred warmup call's stream completion event to its slots."""
        for upload in uploads:
            upload.consumer_event = event
            upload.consumers = []
            upload.needs_consumer = False

    def record_consumers(
        self,
        uploads: list[PreparedEngramUpload],
        consumers: list[torch.Tensor],
    ) -> None:
        """Bind compiled forward outputs to the active slots for safe reuse."""
        for upload in uploads:
            if upload.needs_consumer:
                upload.record_consumers(consumers if consumers else [upload.device])

    def flush_consumers(self, consumers: list[torch.Tensor]) -> None:
        """Flush compiled forward to device before submitting prefetch DMA."""
        if consumers and self._sync_fn is not None:
            self._sync_fn(consumers, wait=False)

    def has_next_prompt_prefetch(self) -> bool:
        return self._next_prompt_spec is not None and bool(self.embeddings)

    def finish_forward(self, padded_tokens: int) -> None:
        """Prefetch the next same-size prompt chunk into inactive upload slots."""
        self._discard_prefetch()
        spec = self._next_prompt_spec
        self._next_prompt_spec = None
        if spec is None or not self.embeddings:
            return
        requests, next_starts, next_lengths = spec
        if any(
            start + length > len(req.origin_input_ids)
            for req, start, length in zip(
                requests, next_starts, next_lengths, strict=True
            )
        ):
            return
        deferred = capture_inputs(list(requests), next_starts, next_lengths, self.ngram)
        if deferred.holes:
            return
        inputs = deferred.resolve(None)
        hashes = hash_on_host(
            inputs,
            **self.tables,
            pad_id=self.pad_id,
            image_token_id=self.image_token_id,
        )
        if hashes.shape[0] > padded_tokens:
            return
        uploads = {
            layer_index: self._start_layer_upload(layer_index, padded_tokens, hashes)
            for layer_index, _ in self.embeddings
        }
        self._prefetch = _EngramPrefetch(
            requests=requests,
            starts=next_starts,
            lengths=next_lengths,
            inputs=inputs,
            hashes=hashes,
            padded_tokens=padded_tokens,
            uploads=uploads,
        )

    def remember(self, requests, positions, result) -> None:
        self.pending = PendingTokens(requests, positions, result)

    def remember_fallback_if_applicable(self, batch, result) -> None:
        """Keep PendingTokens valid across fallback decode and extend steps."""
        self.pending = None
        if batch is None or result is None:
            return
        if batch.spec_algorithm is not None and not batch.spec_algorithm.is_none():
            return
        reqs = batch.reqs
        if reqs is None:
            return
        mode = batch.forward_mode
        if mode.is_decode():
            seq_lens_cpu = batch.seq_lens_cpu
            if seq_lens_cpu is None:
                return
            seq_lens = (
                seq_lens_cpu.tolist()
                if isinstance(seq_lens_cpu, torch.Tensor)
                else list(seq_lens_cpu)
            )
            # Beam tails append member rows after the reqs-aligned base rows.
            if len(seq_lens) < len(reqs):
                return
            positions = [int(n) for n in seq_lens[: len(reqs)]]
        elif mode.is_extend_without_speculative():
            prefix_lens = batch.prefix_lens
            extend_lens = batch.extend_lens
            if (
                prefix_lens is None
                or extend_lens is None
                or len(reqs) != len(prefix_lens)
                or len(reqs) != len(extend_lens)
            ):
                return
            positions = [s + n for s, n in zip(prefix_lens, extend_lens, strict=True)]
        else:
            return
        self.remember(list(reqs), positions, result)

    def clear_step(self) -> None:
        self._discard_staged_uploads()
        for upload in self._active_uploads.values():
            if upload.needs_consumer:
                upload.record_consumers([upload.device])
        self.hashes = None
        self.deferred = None
        self._step_inputs = None
        self._history_commit = []
        self._next_prompt_spec = None
        self._prepared_extend_ranges = None
        self.started = False


def shared_host_embeddings(model) -> list[tuple[int, Any]]:
    """``(hash_index, EngramEmbedding)`` for every Engram layer, sorted by hash
    index; empty unless every Engram table is a shared host table. Tags each
    embedding with ``tpu_hash_index`` for ``staged_lookup``."""
    embeddings = {}
    for layer in model.model.layers:
        engram = layer.engram
        if engram is None:
            continue
        embed = engram.embed
        if not embed._shared:
            return []
        embed.tpu_hash_index = engram.layer_hash_index
        embeddings[engram.layer_hash_index] = embed
    return [(i, embeddings[i]) for i in sorted(embeddings)]


def shared_host_tables(model) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """(weight, scale) of each Engram layer's shared host table, by hash index;
    empty unless every Engram table is a shared host table."""
    return tuple(
        (embed.weight, embed.scale) for _, embed in shared_host_embeddings(model)
    )


def get_or_create_coordinator(
    model, hasher, device: torch.device | str
) -> HostEngramCoordinator | None:
    """Return the model's ``HostEngramCoordinator``, creating it lazily."""
    coordinator = getattr(model, "_tpu_host_engram_coordinator", None)
    if coordinator is not None:
        return coordinator
    embeddings = shared_host_embeddings(model)
    if not embeddings:
        return None
    move_hasher_to_host(hasher)
    coordinator = HostEngramCoordinator(hasher, embeddings, device)
    model._tpu_host_engram_coordinator = coordinator
    return coordinator


def local_row_ids(
    num_tokens: int, num_layers: int, num_cols: int, device
) -> torch.Tensor:
    """Cached [T, L, C] int32 token indices (token t -> row t in packed buffer)."""
    dev = torch.device(device)
    key = (num_tokens, num_layers, num_cols, str(dev))
    cached = _LOCAL_ROW_IDS.get(key)
    if cached is None:
        rows = torch.arange(num_tokens, dtype=torch.int32).view(num_tokens, 1, 1)
        cached = rows.expand(num_tokens, num_layers, num_cols).contiguous().to(dev)
        _LOCAL_ROW_IDS[key] = cached
    return cached


def staged_inputs(forward_batch):
    """(local row ids, staged rows) for this forward, or (None, None).

    The V4 compile policy stages them on the batch. The direct tracer rebuilds ForwardBatch
    from its dataclass fields only, so inside its trace they come from the
    attention metadata, which carries them as graph inputs."""
    if forward_batch is not None and forward_batch.tpu_engram_local_ids is not None:
        return forward_batch.tpu_engram_local_ids, forward_batch.tpu_engram_staged
    from sglang.srt.model_executor.forward_context import (
        get_forward_context,
        has_forward_context,
    )

    if not has_forward_context():
        return None, None
    metadata = get_forward_context().attn_backend.forward_metadata
    carried = getattr(metadata, "engram", None)
    return carried if carried is not None else (None, None)


def staged_lookup(embedding, indices, staged) -> torch.Tensor:
    """EngramEmbedding.forward from the packed int32 rows staged for this batch."""
    packed = staged[embedding.tpu_hash_index]
    tokens, heads = indices.shape
    dim = embedding.dim
    row_words = packed_row_words(heads, dim, 1)
    if packed.numel() != tokens * row_words:
        # A late-layer tail (or other row selection) sliced dim 0 of local_row_ids.
        packed = packed.view(-1, row_words)[indices[:, 0]].reshape(-1)
    return unpack_rows(packed, tokens=tokens, heads=heads, dim=dim)


def collect_output_tensors(
    obj: Any, device: torch.device | str | None = None
) -> list[torch.Tensor]:
    """Collect non-empty output tensors (matching ``device.type`` if given)."""
    target_type = torch.device(device).type if device is not None else None
    tensors: list[torch.Tensor] = []
    seen: set[int] = set()

    def _visit(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            if (
                value.numel() > 0
                and (target_type is None or value.device.type == target_type)
                and id(value) not in seen
            ):
                seen.add(id(value))
                tensors.append(value)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                _visit(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                _visit(item)
            return
        if is_dataclass(value) and not isinstance(value, type):
            for field in value.__dataclass_fields__:
                _visit(getattr(value, field))
            return
        if isinstance(value, PPProxyTensors):
            _visit(value.tensors)

    _visit(obj)
    return tensors


def move_hasher_to_host(hasher) -> None:
    """Hash on the host from now on: tables and history go to CPU. The compiled
    forward reads the staged local ids instead of calling the hasher."""
    if hasher.token_map.device.type != "cpu":
        hasher.to("cpu")
    if hasher.history is not None and hasher.history.device.type != "cpu":
        hasher.history = hasher.history.to("cpu")


def _cpu(t):
    return t.detach().to("cpu") if isinstance(t, torch.Tensor) else t


def host_hash_ids(hasher, input_ids: torch.Tensor, forward_batch) -> torch.Tensor:
    """[T, L, C] int64 hash ids of the padded batch, history committed on CPU.

    Fallback path for warmup and modes not handled by ``HostEngramCoordinator``
    (non-verify warmup, two-pass encoder SWA replay hits, beam search, PP/CP/DCP > 1).
    """
    flush_engram_commits(hasher)
    num_tokens = input_ids.shape[0]
    num_layers, num_cols = hasher.offsets.shape
    mode = forward_batch.forward_mode
    serves = mode.is_decode() or mode.is_target_verify() or mode.is_extend()
    if num_tokens == 0 or not serves:
        return torch.zeros(num_tokens, num_layers, num_cols, dtype=torch.int64)
    batch = SimpleNamespace(
        forward_mode=mode,
        spec_info=forward_batch.spec_info,
        positions=_cpu(forward_batch.positions),
        req_pool_indices=_cpu(forward_batch.req_pool_indices),
        out_cache_loc=_cpu(forward_batch.out_cache_loc),
        extend_seq_lens=_cpu(forward_batch.extend_seq_lens),
        extend_start_loc=_cpu(forward_batch.extend_start_loc),
        # Bucket padding lengthens the device lengths; the host list would not.
        extend_seq_lens_cpu=None,
        engram_history=_padded_history(
            _cpu(forward_batch.engram_history), forward_batch.req_pool_indices
        ),
        tpu_engram_local_ids=None,
        # Engram.forward reads it in target-verify mode; None is equal blocks.
        ragged_verify_layout=getattr(forward_batch, "ragged_verify_layout", None),
    )
    return hasher(_cpu(input_ids), batch)


def _padded_history(history, req_pool_indices):
    """Scheduler history covers the real requests only, but the batch is padded
    to its bucket's request count, so the hasher indexes rows past it. Padding
    requests commit to the spare history row, so zero rows serve them."""
    if history is None or history.shape[0] >= req_pool_indices.shape[0]:
        return history
    missing = req_pool_indices.shape[0] - history.shape[0]
    return torch.cat([history, history.new_zeros(missing, history.shape[1])])


@contextlib.contextmanager
def without_cuda_host_register():
    """_HostTable pins its mapping with cudaHostRegister for GPU reads, and
    EngramEmbedding asks the current GPU whether to read rows in address order
    (sglang#45879); the TPU lookup gathers on the host, so neither applies here."""
    from sglang.srt.layers import engram

    stub = SimpleNamespace(cudaHostRegister=lambda *args: 0)
    original = (torch.cuda.cudart, torch.cuda.current_device, engram._sorted_host_reads)
    torch.cuda.cudart = lambda: stub
    torch.cuda.current_device = lambda: 0
    engram._sorted_host_reads = lambda device: False
    try:
        yield
    finally:
        torch.cuda.cudart, torch.cuda.current_device, engram._sorted_host_reads = (
            original
        )


# hasher -> verify commits not yet applied to its host history, oldest first.
_PENDING_COMMITS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def commit_engram_history(
    hasher,
    *,
    verify_ids_2d: torch.Tensor,
    req_pool_indices: torch.Tensor,
    commit_lens: torch.Tensor,
) -> None:
    """Core EngramHasher.commit_after_verify on the host history.

    Each request's history becomes the last n - 1 of history + its first
    commit_lens verify tokens, oldest first. Padded rows (commit_lens 0) write
    the hasher's pad row: core writes them into req slot 0, a live slot.
    """
    history = hasher.history
    if history is None:
        return
    # The history lives on the host; the verify tensors come from the device.
    device = history.device
    verify_ids_2d, req_pool_indices, commit_lens = (
        t.to(device) for t in (verify_ids_2d, req_pool_indices, commit_lens)
    )
    n1 = hasher.max_ngram_size - 1
    if n1 <= 0 or req_pool_indices.numel() == 0:
        return
    req = req_pool_indices.to(torch.int64)
    clamped_ids = verify_ids_2d.clamp_max(_MM_PAD_SHIFT_VALUE).to(history.dtype)
    window = torch.cat([history[req], clamped_ids], dim=1)
    cols = commit_lens.clamp_min(0).to(torch.int64).unsqueeze(-1) + torch.arange(
        n1, device=window.device
    )
    values = window.gather(1, cols)
    rows = torch.where(commit_lens > 0, req, torch.full_like(req, hasher.pad_row))
    history.copy_(history.index_put((rows,), values))


def defer_engram_commit(
    hasher,
    *,
    verify_ids_2d: torch.Tensor,
    req_pool_indices: torch.Tensor,
    commit_lens: torch.Tensor,
) -> None:
    """Queue a verify's Engram commit for flush_engram_commits.

    Committing in the verify step copies the verify outputs to the host and
    blocks it until the verify finishes; the next reader of the history runs
    after the next draft is dispatched.
    """
    commit = (verify_ids_2d, req_pool_indices, commit_lens)
    _PENDING_COMMITS.setdefault(hasher, []).append(commit)


def has_pending_engram_commits(hasher) -> bool:
    # Only DSpark verify queues; without a queue the hasher is never weak-referenced.
    return bool(
        hasher is not None and _PENDING_COMMITS and _PENDING_COMMITS.get(hasher)
    )


def flush_engram_commits(hasher) -> None:
    """Apply the queued verify commits in order; call before reading or writing history."""
    if hasher is None or not _PENDING_COMMITS:
        return
    for verify_ids_2d, req_pool_indices, commit_lens in _PENDING_COMMITS.pop(
        hasher, ()
    ):
        commit_engram_history(
            hasher,
            verify_ids_2d=verify_ids_2d,
            req_pool_indices=req_pool_indices,
            commit_lens=commit_lens,
        )
