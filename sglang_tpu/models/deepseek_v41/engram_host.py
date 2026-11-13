# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 Engram lookup from one host table shared by the node's ranks.

Every rank maps the same memfd copy of each layer's table, so under attention
DP a rank looks up its own tokens only: no id all-gather, no value all-reduce,
no table in HBM and no gather of a 100 GB table inside the compiled graph.

On the fast serving path (decode, extend, mixed, idle without spec/beam/CP/PP):
1. ``HostEngramCoordinator.prepare_if_supported`` snapshots each scheduled
   chunk and its ``n - 1`` predecessors on CPU without waiting for step N-1's
   in-flight sampled token (marking it as a hole in ``DeferredEngramInputs``).
2. Right before the compiled forward, ``stage_all`` resolves any hole (waiting
   for ``copy_done``), hashes on CPU with ``hash_on_host``, packs each layer's
   FP8 rows and aligned E8M0 scale bytes into one flat ``int32`` buffer using
   single-threaded NumPy (``pack_rows``), and uploads via double-buffered async
   raw DMA (``PreparedEngramUpload``).
3. After the compiled forward is dispatched, ``finish_forward`` prefetches the
   next same-size chunked-prefill chunk into the inactive double-buffer slots.
4. Inside the compiled graph, ``EngramHasher.forward`` returns cached
   ``local_row_ids`` (token indices ``0..T-1``) and ``staged_lookup`` unpacks
   and dequantizes the staged ``int32`` payload (``unpack_rows``), slicing rows
   first if a late-layer tail selected a subset of tokens.

Unsupported or warmup paths (warmup/precompile, speculative target-verify,
folded encoder SWA replay, beam search, PP/CP/DCP > 1) fall back cleanly to
``host_hash_ids`` + ``stage_from_hashes``, producing the exact same packed
``int32`` graph input signature with zero recompilation.
"""

from __future__ import annotations

import contextlib
import ctypes
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import msgspec
import numpy as np
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch

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


def _host_words(buffer: Any, num_words: int) -> torch.Tensor:
    if num_words == 0:
        return torch.empty(0, dtype=torch.int32, device="cpu")
    owner = (ctypes.c_uint8 * buffer.size_bytes).from_address(buffer.data_ptr)
    return torch.frombuffer(owner, dtype=torch.uint8)[: num_words * 4].view(torch.int32)


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


class PreparedEngramUpload:
    """One immutable device allocation, its host storage and consumer lifetime."""

    def __init__(
        self,
        num_words: int,
        device: torch.device | str,
        *,
        raw_transfer_mod: Any = None,
        layout_mod: Any = None,
        sync_fn: Any = None,
    ) -> None:
        dev = torch.device(device)
        self._sync_fn = sync_fn
        self.pending: Any = None
        self.consumers: list[torch.Tensor] = []
        self.needs_consumer = False
        if num_words == 0 or (dev.type == "cpu" and raw_transfer_mod is None):
            self.device = torch.zeros(num_words, dtype=torch.int32, device=dev)
            self.host_words = torch.zeros(num_words, dtype=torch.int32, device="cpu")
            self.host_buffer = SimpleNamespace(size_bytes=num_words * 4)
            self.transfer = _CpuTransfer(self.device, self.host_words)
            return
        if raw_transfer_mod is None:
            from tpu_sync.api.torch import raw_transfer as raw_transfer_mod
        if self._sync_fn is None:
            from torch_tpu._internal.sync.sync import synchronize as _tpu_sync

            self._sync_fn = _tpu_sync
        probe = torch.arange(num_words, dtype=torch.int32, device="cpu")
        probe.bitwise_xor_(0x13579BDF)
        self.device = probe.to(dev)
        phys_words = _physical_words(self.device, layout_mod)
        self.host_buffer = raw_transfer_mod.RawHostBuffer(phys_words * 4)
        self.transfer = raw_transfer_mod.PreparedTorchRawTransfer(
            self.device, self.host_buffer, unsafe_skip_buffer_lock=False
        )
        self.host_words = _host_words(self.host_buffer, num_words)
        self.transfer.d2h()
        if not torch.equal(self.host_words, probe):
            raise RuntimeError("Engram raw layout validation failed")

    def acquire(self) -> torch.Tensor:
        self.wait()
        return self.host_words

    def release_consumers(self) -> None:
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

    def record_consumers(self, tensors: list[torch.Tensor]) -> None:
        if self.needs_consumer:
            if not tensors:
                raise ValueError("Engram slot reuse requires a consumer output")
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
        raw_transfer_mod: Any = None,
        layout_mod: Any = None,
        sync_fn: Any = None,
    ):
        self.ngram = hasher.max_ngram_size
        self.tables = {
            name: np.asarray(getattr(hasher, name).detach().cpu(), dtype=np.int64)
            for name in ("token_map", "multipliers", "primes", "offsets")
        }
        self.pad_id = hasher.pad_id
        self.image_token_id = getattr(hasher, "image_token_id", None)
        self.embeddings = list(embeddings)
        self.by_layer = dict(embeddings)
        self.device = torch.device(device)
        self._raw_transfer_mod = raw_transfer_mod
        self._layout_mod = layout_mod
        if sync_fn is None and self.device.type != "cpu":
            from torch_tpu._internal.sync.sync import synchronize as sync_fn
        self._sync_fn = sync_fn
        self.pending: PendingTokens | None = None
        self.hashes: torch.Tensor | None = None
        self.started = False
        # This step's windows while the previous step's sampled token is in flight.
        self.deferred: DeferredEngramInputs | None = None
        # Double-buffered upload slots per (layer_index, padded_tokens) bucket.
        self._uploads_by_layer: dict[int, dict[int, list[PreparedEngramUpload]]] = {
            layer_index: {} for layer_index, _ in self.embeddings
        }
        self._active_uploads: dict[int, PreparedEngramUpload] = {}
        self._staged_uploads: dict[int, PreparedEngramUpload] = {}
        self._staged_padded_tokens: int | None = None
        self._next_prompt_spec: tuple[tuple[Any, ...], list[int], list[int]] | None = (
            None
        )
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
        for tokens, slots in by_tokens.items():
            if tokens != padded_tokens:
                for slot in slots:
                    slot.release_consumers()
        slots = by_tokens.get(padded_tokens)
        if slots is None:
            embedding = self.by_layer[layer_index]
            heads = self.tables["offsets"].shape[1]
            words = packed_row_words(heads, embedding.weight.shape[1], padded_tokens)
            slots = [
                PreparedEngramUpload(
                    words,
                    self.device,
                    raw_transfer_mod=self._raw_transfer_mod,
                    layout_mod=self._layout_mod,
                    sync_fn=self._sync_fn,
                )
                for _ in range(2)
            ]
            by_tokens[padded_tokens] = slots
        active = self._active_uploads.get(layer_index)
        return next(slot for slot in slots if slot is not active)

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
        self._next_prompt_spec = None
        self.started = False

    def prepare(self, batch) -> tuple[list, list[int]]:
        """Capture this batch's host Engram windows (raising if unsupported)."""
        if not batch.spec_algorithm.is_none():
            raise NotImplementedError(
                "Host Engram does not yet support speculative decode"
            )
        if any(getattr(req, "beam_group", None) is not None for req in batch.reqs):
            raise NotImplementedError("Host Engram does not yet support beam search")
        mode = batch.forward_mode
        if mode.is_decode():
            requests = list(batch.reqs)
            starts = [int(n) - 1 for n in batch.seq_lens_cpu.tolist()]
            lengths = [1] * len(requests)
        elif mode.is_extend_without_speculative():
            requests = list(batch.reqs)
            starts, lengths = list(batch.prefix_lens), list(batch.extend_lens)
        elif mode.is_idle():
            requests, starts, lengths = [], [], []
        else:
            raise NotImplementedError(f"Host Engram does not support {mode}")
        self._discard_staged_uploads()
        self._next_prompt_spec = (
            self._next_prompt_prediction(requests, starts, lengths)
            if mode.is_extend_without_speculative()
            else None
        )
        prefetched = self._prefetch
        self._prefetch = None
        if (
            prefetched is not None
            and mode.is_extend_without_speculative()
            and prefetched.matches(requests, starts, lengths, self.ngram)
        ):
            self.hashes = prefetched.hashes
            self.deferred = None
            self._staged_uploads = dict(prefetched.uploads)
            self._staged_padded_tokens = prefetched.padded_tokens
        else:
            if prefetched is not None:
                prefetched.discard()
            self.hashes = None
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
    ) -> tuple[list, list[int]] | None:
        """Capture fast-path inputs when supported; otherwise reset for fallback."""
        if batch is None:
            self._reset_step_inputs()
            return None
        if parallel is not None and any(
            getattr(parallel, name, 1) != 1
            for name in ("pp_size", "attn_cp_size", "attn_dcp_size")
        ):
            self._reset_step_inputs()
            return None
        spec_algorithm = getattr(batch, "spec_algorithm", None)
        if spec_algorithm is not None and not spec_algorithm.is_none():
            self._reset_step_inputs()
            return None
        reqs = getattr(batch, "reqs", None)
        if reqs is None or any(
            getattr(req, "beam_group", None) is not None for req in reqs
        ):
            self._reset_step_inputs()
            return None
        mode = batch.forward_mode
        if mode.is_decode():
            if getattr(batch, "seq_lens_cpu", None) is None:
                self._reset_step_inputs()
                return None
        elif mode.is_extend_without_speculative():
            if (
                enable_encoder_swa_bounded_replay
                or getattr(batch, "prefix_lens", None) is None
                or getattr(batch, "extend_lens", None) is None
            ):
                self._reset_step_inputs()
                return None
        elif not mode.is_idle():
            self._reset_step_inputs()
            return None
        return self.prepare(batch)

    def resolve(self) -> torch.Tensor | None:
        """Fill overlap holes (waiting for copy_done) and hash on host.

        Later calls in the same step return the same hashes without waiting.
        """
        if self.deferred is not None:
            inputs = self.deferred.resolve(self.pending)
            self.hashes = hash_on_host(
                inputs,
                **self.tables,
                pad_id=self.pad_id,
                image_token_id=self.image_token_id,
            )
            self.deferred = None
        return self.hashes

    def _stage_layers(
        self, hashes: torch.Tensor, padded_tokens: int, *, use_prestaged: bool
    ) -> tuple[tuple[torch.Tensor, ...], list[PreparedEngramUpload], int]:
        if hashes.shape[0] > padded_tokens:
            raise ValueError("Host Engram rows exceed the padded local token count")
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
            ordered.append(upload)
        num_cols = int(hashes.shape[2])
        return tuple(upload.device for upload in ordered), ordered, num_cols

    def stage_all(
        self, padded_tokens: int
    ) -> tuple[tuple[torch.Tensor, ...], list[PreparedEngramUpload], int]:
        """Resolve deferred overlap tokens (if any) and stage every Engram layer."""
        hashes = self.resolve()
        if hashes is None:
            if self.started:
                raise RuntimeError("Host Engram forward has no prepared request inputs")
            hashes = torch.empty(
                (0, *self.tables["offsets"].shape), dtype=torch.int64, device="cpu"
            )
        return self._stage_layers(hashes, padded_tokens, use_prestaged=True)

    def stage_from_hashes(
        self, hashes: torch.Tensor, padded_tokens: int
    ) -> tuple[tuple[torch.Tensor, ...], list[PreparedEngramUpload], int]:
        """Stage precomputed CPU hash ids through the same packed double buffers."""
        return self._stage_layers(hashes, padded_tokens, use_prestaged=False)

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
        """Keep PendingTokens valid across a fallback extend (e.g. folded CED)."""
        if batch is None:
            return
        spec_algorithm = getattr(batch, "spec_algorithm", None)
        if spec_algorithm is not None and not spec_algorithm.is_none():
            return
        mode = batch.forward_mode
        if not mode.is_extend_without_speculative():
            return
        reqs = getattr(batch, "reqs", None)
        prefix_lens = getattr(batch, "prefix_lens", None)
        extend_lens = getattr(batch, "extend_lens", None)
        if (
            reqs is None
            or prefix_lens is None
            or extend_lens is None
            or len(reqs) != len(prefix_lens)
            or len(reqs) != len(extend_lens)
        ):
            return
        positions = [s + n for s, n in zip(prefix_lens, extend_lens, strict=True)]
        self.remember(list(reqs), positions, result)

    def clear_step(self) -> None:
        self._discard_staged_uploads()
        for upload in self._active_uploads.values():
            if upload.needs_consumer:
                upload.record_consumers([upload.device])
        self.hashes = None
        self.deferred = None
        self._next_prompt_spec = None
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


def staged_lookup(embedding, indices, forward_batch) -> torch.Tensor:
    """EngramEmbedding.forward from the packed int32 rows staged for this batch."""
    packed = forward_batch.tpu_engram_staged[embedding.tpu_hash_index]
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
        if hasattr(value, "__dataclass_fields__"):
            for field in value.__dataclass_fields__:
                _visit(getattr(value, field, None))
            return
        inner = getattr(value, "tensors", None)
        if isinstance(inner, dict):
            _visit(inner)

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
    (speculative verify, folded encoder replay, beam search, PP/CP/DCP > 1).
    """
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
    """_HostTable pins its mapping with cudaHostRegister for GPU reads; the TPU
    lookup gathers on the host, so the pin is a no-op here."""
    stub = SimpleNamespace(cudaHostRegister=lambda *args: 0)
    original = torch.cuda.cudart
    torch.cuda.cudart = lambda: stub
    try:
        yield
    finally:
        torch.cuda.cudart = original
