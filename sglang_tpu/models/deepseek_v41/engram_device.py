# SPDX-License-Identifier: Apache-2.0
# NOTE: do *not* add `from __future__ import annotations` here: cpu_export_jax_op
# inspects `param.annotation is jax.Array`.
"""On-device TPU Engram verify hashing, history commit, and table gather.

When the Engram FP8 + E8M0 tables fit comfortably in TPU HBM (e.g.
DeepSeek-V4.1-Flash's ~211 MiB total across 2 layers), running TARGET_VERIFY
Engram hashing and gathering directly on TPU eliminates the per-step blocking
D2H read of (input_ids, positions, verify_ids_2d), CPU hashing/packing, and
8-way contending PCIe H2D transfers (~2.8 ms/step host idle bubble).

EXTEND (prefill) keeps the existing host-staged pipeline unchanged; the tiny
[num_slots + 1, ngram - 1] int32 history table (~3 KiB) is synced lazily only
at EXTEND <-> VERIFY transitions.
"""

from functools import lru_cache
from typing import Any, Sequence

import numpy as np
import torch

_MM_PAD_SHIFT_VALUE = 1 << 30
_FP8_BLOCK_SIZE = 32
_MAX_DEVICE_TABLE_BYTES = 512 * 1024 * 1024  # 512 MiB cap (DSv4.1-Flash is ~211 MiB)


def _mul_u17_u47_to_u32_pair(s_u32, m0: int, m1: int, m2: int):
    """Exact 64-bit product s * m in uint32 arithmetic for 0 <= s < 2^17, 0 <= m < 2^47.

    m = m0 + m1 * 2^16 + m2 * 2^32 with 0 <= m_k < 2^16.
    s = s_lo + s_hi * 2^16 with 0 <= s_lo < 2^16, s_hi in {0, 1}.
    Every intermediate fits strictly within uint32 without overflow.
    Returns (hi_u32, lo_u32) such that value = hi * 2^32 + lo.
    """
    import jax.numpy as jnp

    s_lo = s_u32 & jnp.uint32(0xFFFF)
    s_hi = s_u32 >> jnp.uint32(16)

    q0 = s_lo * jnp.uint32(m0)
    q1 = s_lo * jnp.uint32(m1)
    q2 = s_lo * jnp.uint32(m2)

    w0 = q0 & jnp.uint32(0xFFFF)
    t1 = (q0 >> jnp.uint32(16)) + (q1 & jnp.uint32(0xFFFF)) + s_hi * jnp.uint32(m0)
    w1 = t1 & jnp.uint32(0xFFFF)
    t2 = (
        (t1 >> jnp.uint32(16))
        + (q1 >> jnp.uint32(16))
        + (q2 & jnp.uint32(0xFFFF))
        + s_hi * jnp.uint32(m1)
    )
    w2 = t2 & jnp.uint32(0xFFFF)
    t3 = (t2 >> jnp.uint32(16)) + (q2 >> jnp.uint32(16)) + s_hi * jnp.uint32(m2)
    w3 = t3 & jnp.uint32(0xFFFF)

    lo = w0 | (w1 << jnp.uint32(16))
    hi = w2 | (w3 << jnp.uint32(16))
    return hi, lo


def _hash_verify_jax(
    input_ids,
    positions,
    req_slots,
    history,
    token_map,
    *,
    block: int,
    ngram: int,
    pad_id: int,
    image_token_id: int,
    mult_limbs: tuple[tuple[tuple[int, int, int], ...], ...],
    primes_tuple: tuple[tuple[tuple[int, ...], ...], ...],
    offsets_tuple: tuple[tuple[int, ...], ...],
):
    """Exact TPU-native equivalent of verify_inputs_from_host + hash_on_host.

    Returns tuple of n_layers int32 arrays, each [T, total_heads].
    """
    import jax.numpy as jnp

    bs = req_slots.shape[0]
    num_tokens = input_ids.shape[0]
    slots = jnp.clip(req_slots.astype(jnp.int32), 0, history.shape[0] - 1)
    ver_2d = input_ids.astype(jnp.int32).reshape(bs, block)
    window = jnp.concatenate([history[slots].astype(jnp.int32), ver_2d], axis=1)
    # Build [bs, block, ngram] lookback windows (col 0 = current, col s = s-th pred).
    cols = (
        (ngram - 1)
        + jnp.arange(block, dtype=jnp.int32)[:, None]
        - jnp.arange(ngram, dtype=jnp.int32)[None, :]
    )
    tokens = window[:, cols].reshape(num_tokens, ngram)
    pos = positions.astype(jnp.int32).reshape(num_tokens)

    sentinel = (tokens < 0) | (tokens >= _MM_PAD_SHIFT_VALUE)
    if image_token_id >= 0:
        sentinel = sentinel | (tokens == image_token_id)

    # Prefix-OR across the ngram dimension (ngram is small, e.g. 4).
    blocked_cols = []
    acc = jnp.zeros((num_tokens,), dtype=jnp.bool_)
    for k in range(ngram):
        acc = acc | (pos < k) | sentinel[:, k]
        blocked_cols.append(acc)
    blocked = jnp.stack(blocked_cols, axis=1)

    safe_tokens = jnp.where(
        blocked, jnp.int32(0), jnp.clip(tokens, 0, token_map.shape[0] - 1)
    )
    source = jnp.where(
        blocked, jnp.uint32(pad_id), token_map[safe_tokens].astype(jnp.uint32)
    )  # [T, ngram] uint32

    n_layers = len(mult_limbs)
    layer_hashes = []
    for li in range(n_layers):
        # Compute terms for each lookback k=0..ngram-1 and accumulate XOR.
        t_hi, t_lo = _mul_u17_u47_to_u32_pair(source[:, 0], *mult_limbs[li][0])
        r_hi, r_lo = t_hi, t_lo
        head_cols = []
        off_idx = 0
        for k in range(1, ngram):
            k_hi, k_lo = _mul_u17_u47_to_u32_pair(source[:, k], *mult_limbs[li][k])
            r_hi = r_hi ^ k_hi
            r_lo = r_lo ^ k_lo
            for p in primes_tuple[li][k - 1]:
                p_u = jnp.uint32(p)
                pow32_mod_p = jnp.uint32((1 << 32) % p)
                rem = ((r_hi % p_u) * pow32_mod_p + (r_lo % p_u)) % p_u
                off = jnp.int32(offsets_tuple[li][off_idx])
                head_cols.append(rem.astype(jnp.int32) + off)
                off_idx += 1
        layer_hashes.append(jnp.stack(head_cols, axis=1))  # [T, total_heads]
    return tuple(layer_hashes)


@lru_cache(maxsize=32)
def _get_verify_stage_op(
    block: int,
    ngram: int,
    pad_id: int,
    image_token_id: int,
    n_layers: int,
    mult_limbs: tuple[tuple[tuple[int, int, int], ...], ...],
    primes_tuple: tuple[tuple[tuple[int, ...], ...], ...],
    offsets_tuple: tuple[tuple[int, ...], ...],
    dim: int,
):
    import jax
    import jax.numpy as jnp

    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    total_heads = len(offsets_tuple[0])
    raw_words = total_heads * (dim // 4)
    scale_words = total_heads * ((dim // _FP8_BLOCK_SIZE) // 4)
    row_words = raw_words + scale_words

    # Build a fixed-arity function matching n_layers (DSv4.1 has n_layers=2).
    if n_layers == 2:
        def _fn(
            input_ids: jax.Array,
            positions: jax.Array,
            req_slots: jax.Array,
            history: jax.Array,
            token_map: jax.Array,
            w0: jax.Array,
            s0: jax.Array,
            w1: jax.Array,
            s1: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            hashes = _hash_verify_jax(
                input_ids,
                positions,
                req_slots,
                history,
                token_map,
                block=block,
                ngram=ngram,
                pad_id=pad_id,
                image_token_id=image_token_id,
                mult_limbs=mult_limbs,
                primes_tuple=primes_tuple,
                offsets_tuple=offsets_tuple,
            )
            T = input_ids.shape[0]
            outs = []
            for h_idx, (w, s) in zip(hashes, ((w0, s0), (w1, s1))):
                safe_idx = jnp.clip(h_idx, 0, w.shape[0] - 1)
                raw = w[safe_idx].reshape(T, raw_words)
                scl = s[safe_idx].reshape(T, scale_words)
                outs.append(jnp.concatenate([raw, scl], axis=1).reshape(-1))
            return outs[0], outs[1]

        op = cpu_export_jax_op(
            f"sglang_tpu_pallas::dsv41_dev_engram_verify_b{block}_n{ngram}_l2_d{dim}",
            _fn,
        )
        op.register_fake(
            lambda input_ids, *_: (
                input_ids.new_empty((input_ids.shape[0] * row_words,), dtype=torch.int32),
                input_ids.new_empty((input_ids.shape[0] * row_words,), dtype=torch.int32),
            )
        )
        return op

    raise NotImplementedError(f"DeviceEngramVerify supports n_layers=2, got {n_layers}")


@lru_cache(maxsize=16)
def _get_commit_history_op(n1: int, pad_row: int):
    import jax
    import jax.numpy as jnp

    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    def _commit(
        history: jax.Array,
        verify_ids_2d: jax.Array,
        req_pool_indices: jax.Array,
        commit_lens: jax.Array,
    ) -> jax.Array:
        req = jnp.clip(req_pool_indices.astype(jnp.int32), 0, history.shape[0] - 1)
        clamped = jnp.minimum(verify_ids_2d.astype(jnp.int32), jnp.int32(_MM_PAD_SHIFT_VALUE))
        window = jnp.concatenate([history[req].astype(jnp.int32), clamped], axis=1)
        clen = jnp.maximum(commit_lens.astype(jnp.int32), jnp.int32(0))
        cols = clen[:, None] + jnp.arange(n1, dtype=jnp.int32)[None, :]
        cols = jnp.clip(cols, 0, window.shape[1] - 1)
        values = jnp.take_along_axis(window, cols, axis=1)
        rows = jnp.where(commit_lens.astype(jnp.int32) > 0, req, jnp.int32(pad_row))
        return history.astype(jnp.int32).at[rows].set(values)

    op = cpu_export_jax_op(
        f"sglang_tpu_pallas::dsv41_dev_engram_commit_n{n1}_p{pad_row}",
        _commit,
    )
    op.register_fake(lambda history, *_: torch.empty_like(history, dtype=torch.int32))
    return op


class DeviceEngramVerify:
    """HBM-resident Engram tables + on-device verify hash/gather/commit for TPU."""

    def __init__(self, coordinator: Any, hasher: Any, device: torch.device) -> None:
        self._hasher = hasher
        self._device = device
        self.ngram = int(coordinator.ngram)
        self.pad_id = int(coordinator.pad_id)
        self.image_token_id = (
            -1 if coordinator.image_token_id is None else int(coordinator.image_token_id)
        )
        self.n_layers = len(coordinator.embeddings)
        self.dim = int(coordinator.embeddings[0][1].dim)
        self.num_cols = int(coordinator.tables["offsets"].shape[1])

        # Precompute 16-bit limbs of int64 multipliers for exact uint32 math.
        mults = np.asarray(coordinator.tables["multipliers"], dtype=np.int64)
        self._mult_limbs = tuple(
            tuple(
                (int(m) & 0xFFFF, (int(m) >> 16) & 0xFFFF, (int(m) >> 32) & 0xFFFF)
                for m in row
            )
            for row in mults
        )
        primes = np.asarray(coordinator.tables["primes"], dtype=np.int64)
        self._primes_tuple = tuple(
            tuple(tuple(int(p) for p in ng) for ng in layer) for layer in primes
        )
        offsets = np.asarray(coordinator.tables["offsets"], dtype=np.int64)
        self._offsets_tuple = tuple(tuple(int(o) for o in layer) for layer in offsets)

        self._token_map_dev = torch.as_tensor(
            np.ascontiguousarray(coordinator.tables["token_map"], dtype=np.int32),
            dtype=torch.int32,
        ).to(device)

        # Copy FP8 weights and E8M0 scales to TPU HBM as packed int32 rows.
        self._w_devs: list[torch.Tensor] = []
        self._s_devs: list[torch.Tensor] = []
        for _, emb in coordinator.embeddings:
            w_u8 = emb.weight.detach().view(torch.uint8).contiguous()
            s_u8 = emb.scale.detach().view(torch.uint8).contiguous()
            self._w_devs.append(w_u8.view(torch.int32).to(device))
            self._s_devs.append(s_u8.view(torch.int32).to(device))

        self.history_dev = hasher.history.detach().to(torch.int32).contiguous().to(device)
        self._dev_ahead = False
        self._cpu_ahead = False

    @staticmethod
    def can_enable(coordinator: Any, hasher: Any, device: torch.device) -> bool:
        if device.type != "tpu" or hasher.history is None:
            return False
        embs = coordinator.embeddings
        if len(embs) != 2:
            return False
        # Exact uint32 limb arithmetic bounds:
        # - (r_hi % p) * ((1<<32) % p) + (r_lo % p) fits in uint32 iff p < 2^16.
        # - _mul_u17_u47_to_u32_pair requires 0 <= source < 2^17 and 0 <= mult < 2^47.
        tables = getattr(coordinator, "tables", None)
        if not isinstance(tables, dict):
            return False
        primes = np.asarray(tables["primes"], dtype=np.int64)
        mults = np.asarray(tables["multipliers"], dtype=np.int64)
        token_map = np.asarray(tables["token_map"], dtype=np.int64)
        pad_id = int(coordinator.pad_id)
        if (
            primes.size == 0
            or int(primes.min()) <= 0
            or int(primes.max()) >= (1 << 16)
            or mults.size == 0
            or int(mults.min()) < 0
            or int(mults.max()) >= (1 << 47)
            or token_map.size == 0
            or int(token_map.min()) < 0
            or int(token_map.max()) >= (1 << 17)
            or pad_id < 0
            or pad_id >= (1 << 17)
        ):
            return False
        total_bytes = 0
        dim0 = int(embs[0][1].dim)
        if dim0 % 4 != 0 or (dim0 // _FP8_BLOCK_SIZE) % 4 != 0:
            return False
        for _, emb in embs:
            if not getattr(emb, "_shared", False) or getattr(emb, "mxfp4", False):
                return False
            if int(emb.dim) != dim0:
                return False
            total_bytes += emb.weight.numel() * emb.weight.element_size()
            total_bytes += emb.scale.numel() * emb.scale.element_size()
        return total_bytes <= _MAX_DEVICE_TABLE_BYTES

    @property
    def has_pending_commits(self) -> bool:
        return self._dev_ahead

    def mark_cpu_history_updated(self) -> None:
        self._cpu_ahead = True

    def sync_device_to_cpu_if_needed(self) -> None:
        """Before CPU reads/writes hasher.history (on EXTEND), pull latest TPU verify commits."""
        if not self._dev_ahead:
            return
        cpu_hist = self.history_dev.cpu().to(self._hasher.history.dtype)
        self._hasher.history.copy_(cpu_hist)
        self._dev_ahead = False

    def sync_cpu_to_device_if_needed(self) -> None:
        """Before TPU VERIFY reads history_dev, push latest CPU EXTEND commits."""
        if not self._cpu_ahead:
            return
        self.history_dev = (
            self._hasher.history.detach().to(torch.int32).contiguous().to(self._device)
        )
        self._cpu_ahead = False

    def commit_after_verify(
        self,
        *,
        verify_ids_2d: torch.Tensor,
        req_pool_indices: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        n1 = self.ngram - 1
        if n1 <= 0 or req_pool_indices.numel() == 0:
            return
        self.sync_cpu_to_device_if_needed()
        op = _get_commit_history_op(n1, int(self._hasher.pad_row))
        self.history_dev = op(
            self.history_dev,
            verify_ids_2d.to(torch.int32).contiguous(),
            req_pool_indices.to(torch.int32).contiguous(),
            commit_lens.to(torch.int32).contiguous(),
        )
        self._dev_ahead = True

    def stage_verify(
        self, input_ids: torch.Tensor, forward_batch: Any
    ) -> tuple[tuple[torch.Tensor, ...], int] | None:
        if not forward_batch.forward_mode.is_target_verify():
            return None
        if getattr(forward_batch, "ragged_verify_layout", None) is not None:
            return None
        spec_info = getattr(forward_batch, "spec_info", None)
        block = int(getattr(spec_info, "draft_token_num", 0) or 0)
        if block <= 0:
            return None
        req_slots = forward_batch.req_pool_indices
        if int(input_ids.shape[0]) != int(req_slots.shape[0]) * block:
            return None
        self.sync_cpu_to_device_if_needed()
        # Reuse int32 positions from attention metadata if already built on TPU.
        positions = forward_batch.positions
        if positions.dtype != torch.int32:
            attn_backend = getattr(forward_batch, "attn_backend", None)
            meta = getattr(attn_backend, "forward_metadata", None)
            meta_pos = getattr(meta, "positions", None)
            if (
                isinstance(meta_pos, torch.Tensor)
                and meta_pos.dtype == torch.int32
                and meta_pos.shape == positions.shape
                and meta_pos.device == positions.device
            ):
                positions = meta_pos
            else:
                positions = positions.to(torch.int32)
        op = _get_verify_stage_op(
            block,
            self.ngram,
            self.pad_id,
            self.image_token_id,
            self.n_layers,
            self._mult_limbs,
            self._primes_tuple,
            self._offsets_tuple,
            self.dim,
        )
        staged = op(
            input_ids.to(torch.int32).contiguous(),
            positions.contiguous(),
            req_slots.to(torch.int32).contiguous(),
            self.history_dev,
            self._token_map_dev,
            self._w_devs[0],
            self._s_devs[0],
            self._w_devs[1],
            self._s_devs[1],
        )
        return staged, self.num_cols


@lru_cache(maxsize=16)
def _get_safe_embed_op(fallback_id: int, hc_mult: int):
    import jax
    import jax.numpy as jnp

    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    def _safe_embed(ids: jax.Array, weight: jax.Array) -> jax.Array:
        ids_i32 = ids.astype(jnp.int32)
        oob = (ids_i32 < 0) | (ids_i32 >= _MM_PAD_SHIFT_VALUE)
        safe = jnp.where(oob, jnp.int32(fallback_id), ids_i32)
        safe = jnp.clip(safe, 0, weight.shape[0] - 1)
        out = weight[safe]
        if hc_mult > 1:
            out = jnp.broadcast_to(
                out[:, None, :], (out.shape[0], hc_mult, out.shape[1])
            )
        return out

    op = cpu_export_jax_op(
        f"sglang_tpu_pallas::dsv41_safe_embed_fb{fallback_id}_hc{hc_mult}",
        _safe_embed,
    )
    if hc_mult > 1:
        op.register_fake(
            lambda ids, weight: weight.new_empty(
                (ids.shape[0], hc_mult, weight.shape[1])
            )
        )
    else:
        op.register_fake(
            lambda ids, weight: weight.new_empty((ids.shape[0], weight.shape[1]))
        )
    return op


def fused_safe_embed_tpu(
    ids: torch.Tensor, weight: torch.Tensor, *, fallback_id: int, hc_mult: int = 1
) -> torch.Tensor:
    """Single-op TPU bounds-checked embedding gather (replaces 8 eager ops)."""
    op = _get_safe_embed_op(int(fallback_id), int(hc_mult))
    return op(ids.to(torch.int32).contiguous(), weight)
