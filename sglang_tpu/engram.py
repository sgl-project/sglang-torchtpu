# SPDX-License-Identifier: Apache-2.0
"""CPU Engram hashing and packed row staging, outside the model graph.

The hash and FP8 staging follow vllm-torchtpu's DeepSeek V4.1 CPU-offload path
at bdb38700dd25959c029c7d77f00cee91ccd3a0bc (common/engram.py and
v2/host_engram.py). SGLang retains its checkpoint layout and shared host table.
Only selected FP8 bytes and E8M0 scales cross to the device. The graph
dequantizes them before the existing Engram projection and gate.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import msgspec
import numpy as np
import torch

_INVALID_SCALE = 255
# The SGLang checkpoint layout uses one scale per 32 FP8 values.
_FP8_BLOCK_SIZE = 32
# Multimodal pad token IDs sit at or above MM_PAD_SHIFT_VALUE in SGLang.
_MM_PAD_SHIFT_VALUE = 1_000_000


def signed_sqrt(dot, clamp_value):
    """FP32 signed square root, including the sign of negative zero."""
    magnitude = dot.abs().clamp_min(clamp_value).sqrt()
    # Integer comparison reads the sign bit; a floating comparison loses -0.
    return torch.where(dot.view(torch.int32) < 0, -magnitude, magnitude)


def engram_gate(x, kv, q_weight, k_weight, eps, clamp_value):
    """Core Engram gate with a bitcast/select form of FP32 copysign."""
    hc_mult, dim = x.shape[-2:]
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc_mult, dim))
    weight = q_weight.float() * k_weight.float()
    h = x.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(signed_sqrt(dot, clamp_value))
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


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
    Each hole is (request, position, flat cell indices into tokens).
    """

    tokens: np.ndarray
    positions: np.ndarray
    holes: list

    def resolve(self, pending: PendingTokens | None) -> HostEngramInputs:
        """Fill the holes from the previous step's result; waits for its copy."""
        if self.holes:
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
        local = indices.numpy().astype(np.int64, copy=False) - row_start
        owned = (local >= 0) & (local < rows)
        local = np.clip(local, 0, rows - 1).reshape(-1)
        raw = np.take(weight.view(torch.uint8).numpy(), local, axis=0)
        scales = np.take(scale.view(torch.uint8).numpy(), local, axis=0)
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
    """Device-side inverse of pack_rows, matching the vLLM FP8 dequantizer."""
    raw_bytes = heads * dim
    scale_bytes = heads * (dim // block_size)
    shifts = torch.arange(0, 32, 8, dtype=torch.int32, device=packed.device)
    unpacked = ((packed.unsqueeze(-1) >> shifts) & 0xFF).to(torch.uint8)
    row_bytes = raw_bytes + ((scale_bytes + 3) // 4) * 4
    unpacked = unpacked.reshape(tokens, row_bytes)
    raw = unpacked[:, :raw_bytes].contiguous().view(tokens, heads, dim)
    scale = unpacked[:, raw_bytes : raw_bytes + scale_bytes].contiguous()
    scale = scale.view(tokens, heads, dim // block_size)
    values = raw.view(torch.float8_e4m3fn).float()
    power = (scale.to(torch.int32) << 23).view(torch.float32)
    power = power.repeat_interleave(block_size, -1)
    invalid = (scale == _INVALID_SCALE).repeat_interleave(block_size, -1)
    return torch.where(invalid, 0.0, values * power).to(torch.bfloat16)


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
    """Hash once, then gather each Engram layer at its segment boundary."""

    def __init__(
        self,
        hasher,
        embeddings: list[tuple[int, Any]],
        device,
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
        self.embeddings = embeddings
        self.by_layer = dict(embeddings)
        self.padded_tokens = None
        self.device = device
        self._raw_transfer_mod = raw_transfer_mod
        self._layout_mod = layout_mod
        self._sync_fn = sync_fn
        self.pending: PendingTokens | None = None
        self.hashes: torch.Tensor | None = None
        self.started = False
        # This step's windows while the previous step's sampled token is in flight.
        self.deferred: DeferredEngramInputs | None = None
        # Double-buffered upload slots per (layer_index, padded_tokens) bucket.
        self._uploads_by_layer: dict[int, dict[int, list[PreparedEngramUpload]]] = {
            layer_index: {} for layer_index, _ in embeddings
        }
        self._active_uploads: dict[int, PreparedEngramUpload] = {}
        self._staged_uploads: dict[int, PreparedEngramUpload] = {}
        self._staged_padded_tokens: int | None = None
        layer_order = [layer_index for layer_index, _ in embeddings]
        self._next_layer: dict[int, int] = dict(
            zip(layer_order, layer_order[1:], strict=False)
        )
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

    def prepare(self, batch) -> tuple[list, list[int]]:
        if not batch.spec_algorithm.is_none():
            raise NotImplementedError(
                "Host Engram does not yet support speculative decode"
            )
        if any(getattr(req, "beam_group", None) is not None for req in batch.reqs):
            raise NotImplementedError("Host Engram does not yet support beam search")
        mode = batch.forward_mode
        if mode.is_decode():
            starts = [int(n) - 1 for n in batch.seq_lens_cpu.tolist()]
            lengths = [1] * len(batch.reqs)
        elif mode.is_extend_without_speculative():
            starts, lengths = list(batch.prefix_lens), list(batch.extend_lens)
        elif mode.is_idle():
            starts, lengths = [], []
        else:
            raise NotImplementedError(f"Host Engram does not support {mode}")
        requests = list(batch.reqs)
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

    def resolve(self) -> torch.Tensor | None:
        """Fill the holes from the previous step's result, waiting for its copy, then hash.

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

    def begin_step(self, padded_tokens: int) -> None:
        pending = self.hashes if self.deferred is None else self.deferred.tokens
        if pending is not None and pending.shape[0] > padded_tokens:
            raise ValueError("Host Engram rows exceed the padded local token count")
        self.padded_tokens = padded_tokens
        if (
            self._staged_uploads
            and self._staged_padded_tokens is not None
            and self._staged_padded_tokens != padded_tokens
        ):
            self._discard_staged_uploads()
        if (
            self.started
            and self.hashes is not None
            and self.embeddings
            and not self._staged_uploads
        ):
            first_layer = self.embeddings[0][0]
            self._staged_uploads[first_layer] = self._start_layer_upload(
                first_layer, padded_tokens, self.hashes
            )
            self._staged_padded_tokens = padded_tokens

    def stage_layer(self, layer_index: int, padded_tokens: int) -> None:
        """Activate one layer's double-buffered upload before its segment runs."""
        hashes = self.resolve()
        if hashes is None:
            if self.started:
                raise RuntimeError("Host Engram forward has no prepared request inputs")
            hashes = torch.empty(
                (0, *self.tables["offsets"].shape), dtype=torch.int64, device="cpu"
            )
        upload = self._staged_uploads.pop(layer_index, None)
        if upload is None or self._staged_padded_tokens != padded_tokens:
            if upload is not None:
                upload.wait()
            upload = self._start_layer_upload(layer_index, padded_tokens, hashes)
        upload.activate()
        self._active_uploads[layer_index] = upload
        self.by_layer[layer_index]._tpu_engram_packed = upload.device

    def finish_segment(
        self,
        layer_index: int,
        padded_tokens: int,
        consumers: list[torch.Tensor],
    ) -> None:
        """Record segment outputs for slot reuse and pipeline the next layer's upload."""
        upload = self._active_uploads.get(layer_index)
        if upload is not None:
            upload.record_consumers(consumers)
        next_layer = self._next_layer.get(layer_index)
        if (
            next_layer is not None
            and next_layer not in self._staged_uploads
            and self.hashes is not None
        ):
            self._staged_uploads[next_layer] = self._start_layer_upload(
                next_layer, padded_tokens, self.hashes
            )
            self._staged_padded_tokens = padded_tokens

    def finish_forward(self, padded_tokens: int) -> None:
        """Prefetch the next same-size prompt chunk after the final segment submits."""
        if self._prefetch is not None:
            self._prefetch.discard()
            self._prefetch = None
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

    def stage(self, padded_tokens: int) -> None:
        """Compatibility helper for direct leaf tests; serving stages per layer."""
        self.begin_step(padded_tokens)
        for layer_index, embedding in self.embeddings:
            self.stage_layer(layer_index, padded_tokens)
            self.finish_segment(
                layer_index, padded_tokens, [embedding._tpu_engram_packed]
            )
        self.finish_forward(padded_tokens)

    def remember(self, requests, positions, result) -> None:
        self.pending = PendingTokens(requests, positions, result)

    def clear_step(self) -> None:
        self._discard_staged_uploads()
        for upload in self._active_uploads.values():
            if upload.needs_consumer:
                upload.record_consumers([upload.device])
        self.hashes = None
        self.deferred = None
        self._next_prompt_spec = None
        self.started = False
