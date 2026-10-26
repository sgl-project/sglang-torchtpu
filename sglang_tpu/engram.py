# SPDX-License-Identifier: Apache-2.0
"""CPU Engram hashing and packed row staging, outside the model graph.

The hash and FP8 staging follow vllm-torchtpu's DeepSeek V4.1 CPU-offload path
at bdb38700dd25959c029c7d77f00cee91ccd3a0bc (common/engram.py and
v2/host_engram.py). SGLang retains its checkpoint layout and shared host table.
Only selected FP8 bytes and E8M0 scales cross to the device. The graph
dequantizes them before the existing Engram projection and gate.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

_INVALID_SCALE = 255
# The SGLang checkpoint layout uses one scale per 32 FP8 values.
_FP8_BLOCK_SIZE = 32


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


def capture_inputs(
    requests,
    starts: list[int],
    lengths: list[int],
    ngram: int,
    pending: PendingTokens | None = None,
) -> HostEngramInputs:
    """Snapshot only each scheduled chunk and its n-1 predecessors.

    ``starts`` must come from the scheduled batch (prefix_lens for extend,
    seq_lens_cpu - 1 for decode). Request output_ids can lag by one under
    overlap; only that missing token is resolved from the previous result.
    """
    windows, all_positions = [], []
    shifts = np.arange(ngram, dtype=np.int64)
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
        for position in range(max(lo, known_end), stop):
            if pending is None:
                raise RuntimeError(
                    f"Engram input token at position {position} is not available on host"
                )
            source.append(pending.read(req, position))
        positions = np.arange(start, stop, dtype=np.int64)
        past = positions[:, None] - shifts
        indices = np.maximum(past - lo, 0)
        windows.append(np.asarray(source, dtype=np.int64)[indices])
        all_positions.append(positions)
    return HostEngramInputs(
        tokens=(
            np.concatenate(windows) if windows else np.empty((0, ngram), dtype=np.int64)
        ),
        positions=(
            np.concatenate(all_positions)
            if all_positions
            else np.empty(0, dtype=np.int64)
        ),
    )


def hash_on_host(
    inputs: HostEngramInputs,
    *,
    token_map: np.ndarray,
    multipliers: np.ndarray,
    primes: np.ndarray,
    offsets: np.ndarray,
    pad_id: int,
) -> torch.Tensor:
    """The V4.1 int64 multiply/XOR/modulo hash, with compressed PAD ids."""
    tokens = inputs.tokens
    ngram = multipliers.shape[1]
    if tokens.shape != (len(inputs.positions), ngram):
        raise ValueError("Engram token windows do not match the hash layout")
    blocked = inputs.positions[:, None] < np.arange(ngram)
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


def pack_rows(
    weight: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    row_start: int,
    padded_tokens: int,
    block_size: int = _FP8_BLOCK_SIZE,
) -> torch.Tensor:
    """Gather CPU bytes into one flat int32 upload, with zero-output padding."""
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
    packed = torch.zeros((padded_tokens, row_bytes), dtype=torch.uint8, device="cpu")
    packed[:, raw_bytes:].fill_(_INVALID_SCALE)
    if tokens and rows:
        local = indices.to(torch.int64) - row_start
        owned = (local >= 0) & (local < rows)
        local = local.clamp(0, rows - 1).reshape(-1)
        raw = weight.view(torch.uint8).index_select(0, local).view(tokens, -1)
        scales = scale.view(torch.uint8).index_select(0, local)
        scales = scales.view(tokens, heads, blocks)
        scales.masked_fill_(~owned.unsqueeze(-1), _INVALID_SCALE)
        packed[:tokens, :raw_bytes].copy_(raw)
        packed[:tokens, raw_bytes : raw_bytes + scale_bytes].copy_(
            scales.reshape(tokens, -1)
        )
    return packed.view(torch.int32).reshape(-1)


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


class HostEngramCoordinator:
    """Hash once, then gather each Engram layer at its segment boundary."""

    def __init__(self, hasher, embeddings: list[tuple[int, Any]], device):
        self.ngram = hasher.max_ngram_size
        self.tables = {
            name: np.asarray(getattr(hasher, name).detach().cpu(), dtype=np.int64)
            for name in ("token_map", "multipliers", "primes", "offsets")
        }
        self.pad_id = hasher.pad_id
        self.embeddings = embeddings
        self.by_layer = dict(embeddings)
        self.padded_tokens = None
        self.device = device
        self.pending: PendingTokens | None = None
        self.hashes: torch.Tensor | None = None
        self.started = False

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
        inputs = capture_inputs(requests, starts, lengths, self.ngram, self.pending)
        self.hashes = hash_on_host(inputs, **self.tables, pad_id=self.pad_id)
        self.started = True
        return requests, [s + n for s, n in zip(starts, lengths, strict=True)]

    def begin_step(self, padded_tokens: int) -> None:
        if self.hashes is not None and self.hashes.shape[0] > padded_tokens:
            raise ValueError("Host Engram rows exceed the padded local token count")
        self.padded_tokens = padded_tokens

    def stage_layer(self, layer_index: int, padded_tokens: int) -> None:
        """Upload exactly one layer's local rows after the preceding submission.

        Each upload owns a fresh device tensor. An outstanding segment retains
        its previous tensor through the runtime, so staging cannot overwrite an
        in-flight reader. The blocking copy completes before the next dispatch.
        """
        hashes = self.hashes
        if hashes is None:
            if self.started:
                raise RuntimeError("Host Engram forward has no prepared request inputs")
            hashes = torch.empty(
                (0, *self.tables["offsets"].shape), dtype=torch.int64, device="cpu"
            )
        embedding = self.by_layer[layer_index]
        packed = pack_rows(
            embedding.weight,
            embedding.scale,
            hashes[:, layer_index],
            row_start=0,
            padded_tokens=padded_tokens,
        )
        embedding._tpu_engram_packed = packed.to(self.device, non_blocking=False)

    def stage(self, padded_tokens: int) -> None:
        """Compatibility helper for direct leaf tests; serving stages per layer."""
        self.begin_step(padded_tokens)
        for layer_index, _ in self.embeddings:
            self.stage_layer(layer_index, padded_tokens)

    def remember(self, requests, positions, result) -> None:
        self.pending = PendingTokens(requests, positions, result)

    def clear_step(self) -> None:
        self.hashes = None
