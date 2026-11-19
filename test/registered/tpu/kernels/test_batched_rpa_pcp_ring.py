# SPDX-License-Identifier: Apache-2.0
"""Pure-JAX multi-device TPU tests for the fused PCP ring attention kernel
(experimental ``sglang_tpu.attention.pallas.batched_rpa_pcp`` package).

Ported from google3 ``correctness_test_cp.RaggedPagedAttentionPCPTest``
(cl/985434096) and adapted to SGLang's batched RPA conventions (page size 256,
SGLang ``distribution`` order, ``update_kv_cache=False`` by default with a
companion per-rank stripe write, plus parity checks for ``update_kv_cache=True``
with ``new_kv_write_rows``).

Requires JAX on >= 2 TPU devices in one process (an 8-device single host covers
every case, including PCP=4, PCP=8, a 2-D ``(pcp, model)`` mesh, and 128k long
context); skipped when fewer devices are visible. No torch_tpu, no server.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(
    est_time=480,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "infra", "pcp-ring"],
)

# Standalone raw-JAX kernel test, not a TorchTPU runtime process: select the
# physical TPU backend before the first JAX import.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"


def _tpu_device_count() -> int:
    try:
        import jax
    except Exception:
        return 0
    try:
        devs = jax.devices()
    except Exception:
        return 0
    if not devs or not any("TPU" in d.device_kind for d in devs):
        return 0
    return len(devs)


PAGE_SIZE = 256
TOLERANCE = 0.05


def _cdiv(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _round_through(x: np.ndarray, dtype) -> np.ndarray:
    import jax.numpy as jnp

    j = jnp.asarray(x, dtype=jnp.float32).astype(dtype).astype(jnp.float32)
    # np.array (not asarray): host views of JAX buffers are read-only.
    return np.array(j, dtype=np.float32)


def _dense_causal_attention(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    q_positions: np.ndarray,
    *,
    sm_scale: float,
    k_scale: float | None = None,
    v_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Float32 reference causal attention over ``keys``/``values``; returns ``(out, lse)``."""
    num_q, num_q_heads, head_dim = queries.shape
    num_k, num_kv_heads, _ = keys.shape
    group = num_q_heads // num_kv_heads
    k = np.repeat(keys.astype(np.float32), group, axis=1)
    v = np.repeat(values.astype(np.float32), group, axis=1)
    if k_scale is not None:
        k = k * float(k_scale)
    if v_scale is not None:
        v = v * float(v_scale)
    q = queries.astype(np.float32)
    logits = np.einsum("qhd,khd->hqk", q, k) * float(sm_scale)
    k_pos = np.arange(num_k, dtype=np.int64)[None, :]
    mask = k_pos <= q_positions[:, None]
    logits = np.where(mask[None, :, :], logits, -np.inf)
    m = np.max(logits, axis=-1, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    exp = np.exp(logits - m_safe)
    exp = np.where(mask[None, :, :], exp, 0.0)
    denom = np.sum(exp, axis=-1, keepdims=True)
    out = np.einsum("hqk,khd->qhd", exp / np.maximum(denom, 1e-30), v)
    lse = np.swapaxes(
        np.squeeze(m_safe + np.log(np.maximum(denom, 1e-30)), axis=-1), 0, 1
    )
    return out.astype(np.float32), lse.astype(np.float32)


def _merge_by_lse(
    partials: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    lses = np.stack([lse for _, lse in partials], axis=0)
    outs = np.stack([out for out, _ in partials], axis=0)
    m = np.max(lses, axis=0)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    weights = np.where(np.isfinite(lses), np.exp(lses - m_safe[None]), 0.0)
    denom = np.sum(weights, axis=0)
    out = np.sum(outs * weights[..., None], axis=0) / np.maximum(
        denom[..., None], 1e-30
    )
    lse = np.where(np.isfinite(m), m_safe + np.log(np.maximum(denom, 1e-30)), -np.inf)
    return out, lse


def _to_has_cache(fused_kv: np.ndarray, kv_dtype) -> np.ndarray:
    """Pack ``[pages, page, 2 * nkv, hd]`` float32 values into ``HEAD_ALONG_SUBLANE``."""
    import jax
    import jax.numpy as jnp

    from sglang_tpu.attention.pallas.batched_rpa_pcp import utils, wrapper

    pages, page, nkv2, hd = fused_kv.shape
    nkv = nkv2 // 2
    shape = wrapper.get_kv_cache_shape(pages, page, nkv, hd, kv_dtype)
    packing = utils.get_dtype_packing(kv_dtype)
    aligned_nkv2 = shape[2] * packing
    aligned_hd = shape[4]
    arr = jnp.asarray(fused_kv, dtype=jnp.float32).astype(kv_dtype)
    arr = jnp.pad(
        arr,
        ((0, 0), (0, 0), (0, aligned_nkv2 - nkv2), (0, aligned_hd - hd)),
        constant_values=0,
    ).reshape(shape)
    return np.array(jax.device_get(arr))


def _from_has_cache(cache, nkv: int, hd: int) -> tuple[np.ndarray, np.ndarray]:
    """Unpack ``HEAD_ALONG_SUBLANE`` cache to float32 ``(k, v)`` of shape ``[pages, page, nkv, hd]``."""
    import jax.numpy as jnp

    c = np.array(jnp.asarray(cache).astype(jnp.float32))
    pages, page = c.shape[:2]
    c = c.reshape(pages, page, -1, c.shape[-1])[:, :, : 2 * nkv, :hd]
    return c[:, :, 0::2], c[:, :, 1::2]


class _PcpChunkLayout:
    """Host-side rank-uniform layout matching SGLang's ring-mode ``cp_layout``."""

    def __init__(
        self,
        cp: int,
        page: int,
        cached: list[int],
        new_lens: list[int],
        *,
        num_req_slots: int | None = None,
        t_local: int | None = None,
        pages_per_req: int | None = None,
    ):
        self.cp = cp
        self.page = page
        self.cached = [int(c) for c in cached]
        self.new_lens = [int(n) for n in new_lens]
        self.n_req = len(cached)
        self.block = [
            max(page, _cdiv(_cdiv(max(n, 1), 2 * cp), page) * page)
            for n in self.new_lens
        ]
        self.slot_off = np.concatenate(
            [[0], np.cumsum([2 * b for b in self.block])]
        ).astype(int)
        used = int(self.slot_off[-1])
        self.t_local = used if t_local is None else t_local
        assert self.t_local >= used and self.t_local % page == 0
        self.s_req = self.n_req if num_req_slots is None else num_req_slots
        total = [c + n for c, n in zip(self.cached, self.new_lens)]
        need = max([1] + [_cdiv(_cdiv(t, page), cp) for t in total])
        self.pages_per_req = need if pages_per_req is None else pages_per_req
        assert self.pages_per_req >= need
        self.vpp = cp * self.t_local // page

    def rank_of_block(self, c: int) -> int:
        return c if c < self.cp else 2 * self.cp - 1 - c

    def block_span(self, i: int, c: int) -> tuple[int, int]:
        start = c * self.block[i]
        return start, int(np.clip(self.new_lens[i] - start, 0, self.block[i]))

    def local_offset(self, i: int, c: int) -> int:
        return int(self.slot_off[i]) + (self.block[i] if c >= self.cp else 0)

    def rank_tokens(self, r: int) -> list[tuple[int, int, int, int]]:
        out = []
        for i in range(self.n_req):
            for c in (r, 2 * self.cp - 1 - r):
                start, n = self.block_span(i, c)
                if n:
                    out.append((i, start, n, self.local_offset(i, c)))
        return out

    def cache_rows(self) -> dict[str, np.ndarray]:
        s = self.s_req
        cu = np.full(s + 1, int(self.slot_off[-1]), np.int32)
        cu[: self.n_req + 1] = self.slot_off
        kv = np.zeros(s, np.int32)
        for i in range(self.n_req):
            kv[i] = self.cached[i] + 2 * self.block[i]
        dist = np.full(3, self.n_req, np.int32)
        return dict(cache_cu_q_lens=cu, cache_kv_lens=kv, cache_distribution=dist)

    def new_rows(self, r: int) -> dict[str, np.ndarray]:
        s = 2 * self.s_req
        cu = np.full(s + 1, int(self.slot_off[-1]), np.int32)
        nkl = np.zeros(s, np.int32)
        kcl = np.zeros(s, np.int32)
        qpos = np.zeros(s, np.int32)
        wr = np.zeros(s, np.int32)
        nkpi = np.zeros((s, self.vpp), np.int32)
        for i in range(self.n_req):
            b = self.block[i]
            for half, c in enumerate((r, 2 * self.cp - 1 - r)):
                row = 2 * i + half
                cu[row] = self.slot_off[i] + half * b
                nkl[row] = self.cached[i] + self.new_lens[i]
                kcl[row] = self.cached[i]
                qpos[row] = self.cached[i] + c * b
                wr[row] = half
                for j in range(_cdiv(self.new_lens[i], self.page)):
                    blk = j * self.page // b
                    within = j - blk * b // self.page
                    nkpi[row, j] = (
                        self.rank_of_block(blk) * self.t_local
                        + self.local_offset(i, blk)
                    ) // self.page + within
        cu[2 * self.n_req :] = self.slot_off[self.n_req]
        dist = np.full(3, 2 * self.n_req, np.int32)
        return dict(
            new_cu_q_lens=cu,
            new_kv_lens=nkl,
            kv_cache_lens=kcl,
            q_positions=qpos,
            new_kv_write_rows=wr,
            new_kv_page_indices=nkpi.reshape(-1),
            new_distribution=dist,
        )

    def stripe_slots(self, i: int, r: int, table: np.ndarray, start: int, stop: int):
        pos = np.arange(start, stop, dtype=np.int64)
        gpage = pos // self.page
        own = gpage % self.cp == r
        pos, gpage = pos[own], gpage[own]
        return pos, table[i, gpage // self.cp], pos % self.page

    def stripe_len(self, total: int, r: int) -> int:
        sp = self.cp * self.page
        return (total // sp) * self.page + int(
            np.clip(total % sp - r * self.page, 0, self.page)
        )


class _PcpRingHarness:
    """Multi-chunk PCP ring driver with per-rank stripe verification and DCP decode."""

    def __init__(
        self,
        cp: int,
        *,
        page: int = PAGE_SIZE,
        nq: int = 16,
        nkv: int = 2,
        hd: int = 128,
        kv_dtype=None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        seed: int = 0,
        mesh_shape: tuple[int, ...] | None = None,
        pages_per_req: int = 8,
        num_req_slots: int | None = None,
        n_req: int = 1,
        start_lens: list[int] | None = None,
    ):
        import jax
        import jax.numpy as jnp
        from jax.sharding import Mesh

        self.cp = cp
        self.page = page
        self.nq, self.nkv, self.hd = nq, nkv, hd
        self.kv_dtype = jnp.dtype(jnp.bfloat16 if kv_dtype is None else kv_dtype)
        self.k_scale, self.v_scale = k_scale, v_scale
        self.rng = np.random.default_rng(seed)
        self.n_req = n_req
        self.s_req = num_req_slots or max(n_req, 2)
        self.pages_per_req = pages_per_req
        self.sm_scale = 1.0 / math.sqrt(hd)
        if mesh_shape is None:
            mesh_shape = (cp,)
            self.axis_names = ("model",)
            self.ring_axis = "model"
        else:
            self.axis_names = ("pcp", "model")
            self.ring_axis = "pcp"
        devs = np.array(jax.devices()[: int(np.prod(mesh_shape))]).reshape(mesh_shape)
        self.mesh = Mesh(devs, self.axis_names)

        self.hist_k = [np.zeros((0, nkv, hd), np.float32) for _ in range(n_req)]
        self.hist_v = [np.zeros((0, nkv, hd), np.float32) for _ in range(n_req)]
        self.num_pages = self.s_req * pages_per_req
        self.table = (
            self.rng.permutation(self.num_pages)
            .astype(np.int32)
            .reshape(self.s_req, pages_per_req)
        )
        # Initialize unused cache slots with large finite stale values so any
        # out-of-bounds read immediately corrupts the output.
        shape = (cp, self.num_pages, page, nkv, hd)
        self.pool_k = self._rand_kv(shape, scale=10.0)
        self.pool_v = self._rand_kv(shape, scale=10.0)
        if start_lens:
            for i, n in enumerate(start_lens):
                if n > 0:
                    self._write_stripe_host(
                        i, self._rand_kv((n, nkv, hd)), self._rand_kv((n, nkv, hd))
                    )

    def _rand_kv(self, shape, *, scale: float = 1.0) -> np.ndarray:
        x = scale * self.rng.standard_normal(shape, dtype=np.float32)
        if self.k_scale is not None:
            x = np.clip(x, -20.0, 20.0)
        return _round_through(x, self.kv_dtype)

    def _write_stripe_host(self, i: int, k: np.ndarray, v: np.ndarray) -> None:
        start = len(self.hist_k[i])
        self.hist_k[i] = np.concatenate([self.hist_k[i], k], axis=0)
        self.hist_v[i] = np.concatenate([self.hist_v[i], v], axis=0)
        lay = _PcpChunkLayout(
            self.cp, self.page, [0], [1], pages_per_req=self.pages_per_req
        )
        for r in range(self.cp):
            pos, pages, rows = lay.stripe_slots(
                0, r, self.table[i : i + 1], start, start + len(k)
            )
            self.pool_k[r, pages, rows] = k[pos - start]
            self.pool_v[r, pages, rows] = v[pos - start]

    def _expected_pools(
        self, new_k: list[np.ndarray], new_v: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        ek, ev = self.pool_k.copy(), self.pool_v.copy()
        lay = _PcpChunkLayout(
            self.cp, self.page, [0], [1], pages_per_req=self.pages_per_req
        )
        for i in range(self.n_req):
            start = len(self.hist_k[i])
            for r in range(self.cp):
                pos, pages, rows = lay.stripe_slots(
                    0, r, self.table[i : i + 1], start, start + len(new_k[i])
                )
                ek[r, pages, rows] = new_k[i][pos - start]
                ev[r, pages, rows] = new_v[i][pos - start]
        return ek, ev

    def _pack_rank_caches(self) -> list[np.ndarray]:
        out = []
        for r in range(self.cp):
            fused = np.stack([self.pool_k[r], self.pool_v[r]], axis=3).reshape(
                self.num_pages, self.page, 2 * self.nkv, self.hd
            )
            out.append(_to_has_cache(fused, self.kv_dtype))
        return out

    def run_chunk(
        self,
        tc: CustomTestCase,
        label: str,
        new_lens: list[int],
        *,
        update_kv_cache: bool = False,
        token_major_q: bool = False,
        t_local: int | None = None,
        cache_lens_cp_page_aligned: bool = False,
    ) -> None:
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.attention.pallas.batched_rpa_pcp import wrapper

        cp, page = self.cp, self.page
        cached = [len(k) for k in self.hist_k]
        lay = _PcpChunkLayout(
            cp,
            page,
            cached,
            new_lens,
            num_req_slots=self.s_req,
            t_local=t_local,
            pages_per_req=self.pages_per_req,
        )
        t_local = lay.t_local
        wrapper.check_pcp_fused_schedule(
            num_q_heads=self.nq,
            num_kv_heads=self.nkv,
            head_dim=self.hd,
            dtype_q=jnp.bfloat16,
            dtype_kv=self.kv_dtype,
            page_size=page,
            cp_group_size=cp,
            total_q_tokens=t_local,
            total_kv_tokens=cp * t_local,
            cache_cu_q_lens=lay.cache_rows()["cache_cu_q_lens"],
            cache_kv_lens=lay.cache_rows()["cache_kv_lens"],
            cache_pages_per_row=self.pages_per_req,
            new_cu_q_lens=lay.new_rows(0)["new_cu_q_lens"],
            new_kv_lens=lay.new_rows(0)["new_kv_lens"],
            kv_cache_lens=lay.new_rows(0)["kv_cache_lens"],
            q_positions=lay.new_rows(0)["q_positions"],
            new_pages_per_row=self.pages_per_req,
            new_kv_pages_per_row=lay.vpp,
            num_cache_rows_live=self.n_req,
            num_new_rows_live=2 * self.n_req,
            new_kv_write_rows=lay.new_rows(0)["new_kv_write_rows"]
            if update_kv_cache
            else None,
            update_kv_cache=update_kv_cache,
            cache_lens_cp_page_aligned=cache_lens_cp_page_aligned,
        )

        new_q = [
            _round_through(
                self.rng.standard_normal((n, self.nq, self.hd), np.float32),
                jnp.bfloat16,
            )
            for n in new_lens
        ]
        new_k = [self._rand_kv((n, self.nkv, self.hd)) for n in new_lens]
        new_v = [self._rand_kv((n, self.nkv, self.hd)) for n in new_lens]

        q_loc = np.zeros((cp, t_local, self.nq, self.hd), np.float32)
        # Garbage in padded K/V rows so reading beyond real tokens fails.
        k_loc = self._rand_kv((cp, t_local, self.nkv, self.hd), scale=10.0)
        v_loc = self._rand_kv((cp, t_local, self.nkv, self.hd), scale=10.0)
        for r in range(cp):
            for i, start, n, off in lay.rank_tokens(r):
                q_loc[r, off : off + n] = new_q[i][start : start + n]
                k_loc[r, off : off + n] = new_k[i][start : start + n]
                v_loc[r, off : off + n] = new_v[i][start : start + n]

        crow = lay.cache_rows()
        nrows = [lay.new_rows(r) for r in range(cp)]
        new_table = np.repeat(self.table, 2, axis=0)
        rank_caches = self._pack_rank_caches()

        # Replicate across any secondary ("model") mesh axis if 2-D mesh is used.
        sec = self.mesh.shape.get("model", 1) if self.ring_axis == "pcp" else 1
        ring_ax = self.ring_axis

        def _put_ring(arr, dtype):
            a = jnp.asarray(arr, dtype=dtype)
            if sec > 1:
                a = jnp.repeat(a, sec, axis=0)
                return jax.device_put(
                    a.reshape((cp * sec,) + a.shape[1:]),
                    NamedSharding(self.mesh, P((ring_ax, "model"))),
                )
            return jax.device_put(a, NamedSharding(self.mesh, P(ring_ax)))

        q_dev = _put_ring(q_loc.reshape(cp, t_local, self.nq, self.hd), jnp.bfloat16)
        k_dev = _put_ring(k_loc.reshape(cp, t_local, self.nkv, self.hd), self.kv_dtype)
        v_dev = _put_ring(v_loc.reshape(cp, t_local, self.nkv, self.hd), self.kv_dtype)
        cache_stacked = np.stack(rank_caches, axis=0)
        cache_dev = _put_ring(cache_stacked, self.kv_dtype)
        ncu_dev = _put_ring(np.stack([n["new_cu_q_lens"] for n in nrows]), jnp.int32)
        qpos_dev = _put_ring(np.stack([n["q_positions"] for n in nrows]), jnp.int32)
        wr_dev = _put_ring(np.stack([n["new_kv_write_rows"] for n in nrows]), jnp.int32)

        ckl = jnp.asarray(crow["cache_kv_lens"], jnp.int32)
        cpi = jnp.asarray(self.table.reshape(-1), jnp.int32)
        ccu = jnp.asarray(crow["cache_cu_q_lens"], jnp.int32)
        cdist = jnp.asarray(crow["cache_distribution"], jnp.int32)
        nkl = jnp.asarray(nrows[0]["new_kv_lens"], jnp.int32)
        npi = jnp.asarray(new_table.reshape(-1), jnp.int32)
        ndist = jnp.asarray(nrows[0]["new_distribution"], jnp.int32)
        nkpi = jnp.asarray(nrows[0]["new_kv_page_indices"], jnp.int32)
        kcl = jnp.asarray(nrows[0]["kv_cache_lens"], jnp.int32)

        part = P((ring_ax, "model")) if sec > 1 else P(ring_ax)

        def per_rank(q_b, k_b, v_b, cache_b, ncu_b, qpos_b, wr_b):
            q = q_b[0]
            k = k_b[0]
            v = v_b[0]
            cache = cache_b[0]
            cp_rank = jnp.reshape(jax.lax.axis_index(ring_ax), (1,)).astype(jnp.int32)
            k_all = jax.lax.all_gather(k, ring_ax, axis=0, tiled=True)
            v_all = jax.lax.all_gather(v, ring_ax, axis=0, tiled=True)
            res = wrapper.ragged_paged_attention_pcp_fused(
                q,
                k_all,
                v_all,
                cache,
                cache_kv_lens=ckl,
                cache_page_indices=cpi,
                cache_cu_q_lens=ccu,
                cache_distribution=cdist,
                new_kv_lens=nkl,
                new_page_indices=npi,
                new_cu_q_lens=ncu_b[0],
                new_distribution=ndist,
                new_kv_page_indices=nkpi,
                q_positions=qpos_b[0],
                kv_cache_lens=kcl,
                cp_rank=cp_rank,
                new_kv_write_rows=wr_b[0] if update_kv_cache else None,
                cp_group_size=cp,
                ring_axis_name=ring_ax,
                ring_mesh_axis_names=self.axis_names,
                sm_scale=self.sm_scale,
                k_scale=self.k_scale,
                v_scale=self.v_scale,
                token_major_q=token_major_q,
                update_kv_cache=update_kv_cache,
                cache_lens_cp_page_aligned=cache_lens_cp_page_aligned,
            )
            if update_kv_cache:
                o, c = res
                return o[None], c[None]
            return res[None], cache[None]

        fn = jax.jit(
            jax.shard_map(
                per_rank,
                mesh=self.mesh,
                in_specs=(part,) * 7,
                out_specs=(part, part),
                check_vma=False,
            )
        )
        out_dev, cache_out_dev = fn(
            q_dev, k_dev, v_dev, cache_dev, ncu_dev, qpos_dev, wr_dev
        )
        out_np = np.asarray(out_dev.astype(jnp.float32))
        cache_out_np = jax.device_get(cache_out_dev)
        if sec > 1:
            out_np = out_np.reshape(cp, sec, t_local, self.nq, self.hd)[:, 0]
            cache_out_np = cache_out_np.reshape((cp, sec) + cache_out_np.shape[1:])[
                :, 0
            ]

        tc.assertTrue(bool(np.isfinite(out_np).all()), f"{label}: non-finite output")

        for i in range(self.n_req):
            k_full = np.concatenate([self.hist_k[i], new_k[i]], axis=0)
            v_full = np.concatenate([self.hist_v[i], new_v[i]], axis=0)
            q_pos = cached[i] + np.arange(new_lens[i], dtype=np.int64)
            ref, _ = _dense_causal_attention(
                new_q[i],
                k_full,
                v_full,
                q_pos,
                sm_scale=self.sm_scale,
                k_scale=self.k_scale,
                v_scale=self.v_scale,
            )
            got = np.zeros_like(ref)
            for c in range(2 * cp):
                start, n = lay.block_span(i, c)
                if n:
                    off = lay.local_offset(i, c)
                    got[start : start + n] = out_np[lay.rank_of_block(c), off : off + n]
            diff = float(np.abs(got - ref).max())
            denom = max(float(np.abs(ref).max()), 1e-6)
            rel = diff / denom
            if diff > 5e-4:
                tc.assertLess(
                    rel,
                    TOLERANCE,
                    f"{label} req {i}: rel_err={rel:.4f} max_abs_err={diff:.4f}",
                )

        ek, ev = self._expected_pools(new_k, new_v)
        for r in range(cp):
            gk, gv = _from_has_cache(cache_out_np[r], self.nkv, self.hd)
            if update_kv_cache:
                np.testing.assert_array_equal(
                    gk, ek[r], f"{label} rank {r} written K stripe"
                )
                np.testing.assert_array_equal(
                    gv, ev[r], f"{label} rank {r} written V stripe"
                )
            else:
                # Read-only mode must leave the pool bit-identical; companion stripe write updates it.
                np.testing.assert_array_equal(
                    gk, self.pool_k[r], f"{label} rank {r} read-only K mutated"
                )
                np.testing.assert_array_equal(
                    gv, self.pool_v[r], f"{label} rank {r} read-only V mutated"
                )

        # Advance request history & per-rank stripe pools (companion local write in read-only mode).
        self.pool_k, self.pool_v = ek, ev
        for i in range(self.n_req):
            self.hist_k[i] = np.concatenate([self.hist_k[i], new_k[i]], axis=0)
            self.hist_v[i] = np.concatenate([self.hist_v[i], new_v[i]], axis=0)

    def run_dcp_decode_step(self, tc: CustomTestCase, label: str) -> None:
        """Run one DCP ``STRIPED`` decode step over the cache written by PCP ring prefill,
        merge partials across ranks by LSE, and verify both output and owner token write.

        Decode runs on the production ``batched_rpa`` kernel (as in serving: only
        the ring prefill uses the experimental ``batched_rpa_pcp`` package), so
        this also checks that the two packages agree on the striped cache.
        """
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import configs, wrapper

        cp, page = self.cp, self.page
        n_req = self.n_req
        q_step = _round_through(
            self.rng.standard_normal((n_req, self.nq, self.hd), np.float32),
            jnp.bfloat16,
        )
        k_step = self._rand_kv((n_req, self.nkv, self.hd))
        v_step = self._rand_kv((n_req, self.nkv, self.hd))

        seq_lens_after = [len(self.hist_k[i]) + 1 for i in range(n_req)]
        kv_lens_global = np.zeros(self.s_req, dtype=np.int32)
        kv_lens_global[:n_req] = seq_lens_after
        cu_q_lens = np.zeros(self.s_req + 1, dtype=np.int32)
        cu_q_lens[: n_req + 1] = np.arange(n_req + 1, dtype=np.int32)
        cu_q_lens[n_req + 1 :] = n_req
        # Padded token slots up to s_req
        q_pad = np.zeros((self.s_req, self.nq, self.hd), np.float32)
        k_pad = np.zeros((self.s_req, self.nkv, self.hd), np.float32)
        v_pad = np.zeros((self.s_req, self.nkv, self.hd), np.float32)
        q_pad[:n_req], k_pad[:n_req], v_pad[:n_req] = q_step, k_step, v_step

        lay = _PcpChunkLayout(
            cp, page, [0] * n_req, [1] * n_req, pages_per_req=self.pages_per_req
        )
        rank_caches = self._pack_rank_caches()
        partials = []
        updated_caches = []
        for r in range(cp):
            kv_lens_local = np.zeros(self.s_req, dtype=np.int32)
            for i in range(n_req):
                prev_len = seq_lens_after[i] - 1
                owns = ((prev_len // page) % cp) == r
                kv_lens_local[i] = lay.stripe_len(prev_len, r) + int(owns)
            out_r, cache_r, lse_r = wrapper.ragged_paged_attention(
                jnp.asarray(q_pad, dtype=jnp.bfloat16),
                jnp.asarray(k_pad, dtype=self.kv_dtype),
                jnp.asarray(v_pad, dtype=self.kv_dtype),
                jnp.asarray(rank_caches[r], dtype=self.kv_dtype),
                jnp.asarray(kv_lens_local, dtype=jnp.int32),
                jnp.asarray(self.table.reshape(-1), dtype=jnp.int32),
                jnp.asarray(cu_q_lens, dtype=jnp.int32),
                jnp.asarray([0, 0, n_req], dtype=jnp.int32),  # SGLang DECODE range
                sm_scale=self.sm_scale,
                k_scale=self.k_scale,
                v_scale=self.v_scale,
                kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
                return_lse=True,
                cp_group_size=cp,
                cp_rank=jnp.asarray([r], dtype=jnp.int32),
                kv_lens_global=jnp.asarray(kv_lens_global, dtype=jnp.int32),
                attention_scope=configs.AttentionScope.STRIPED,
                rpa_cases=(configs.RpaCase.DECODE,),
            )
            partials.append(
                (
                    np.asarray(out_r.astype(jnp.float32))[:n_req],
                    np.asarray(lse_r.astype(jnp.float32))[:n_req],
                )
            )
            updated_caches.append(cache_r)

        merged_out, _ = _merge_by_lse(partials)
        ek, ev = self._expected_pools(
            [k_step[i : i + 1] for i in range(n_req)],
            [v_step[i : i + 1] for i in range(n_req)],
        )
        for i in range(n_req):
            k_full = np.concatenate([self.hist_k[i], k_step[i : i + 1]], axis=0)
            v_full = np.concatenate([self.hist_v[i], v_step[i : i + 1]], axis=0)
            ref, _ = _dense_causal_attention(
                q_step[i : i + 1],
                k_full,
                v_full,
                np.asarray([seq_lens_after[i] - 1], dtype=np.int64),
                sm_scale=self.sm_scale,
                k_scale=self.k_scale,
                v_scale=self.v_scale,
            )
            diff = float(np.abs(merged_out[i] - ref[0]).max())
            rel = diff / max(float(np.abs(ref[0]).max()), 1e-6)
            if diff > 5e-4:
                tc.assertLess(
                    rel, TOLERANCE, f"{label} decode req {i}: rel_err={rel:.4f}"
                )

        for r in range(cp):
            gk, gv = _from_has_cache(updated_caches[r], self.nkv, self.hd)
            np.testing.assert_array_equal(
                gk, ek[r], f"{label} decode rank {r} K stripe"
            )
            np.testing.assert_array_equal(
                gv, ev[r], f"{label} decode rank {r} V stripe"
            )

        self.pool_k, self.pool_v = ek, ev
        for i in range(n_req):
            self.hist_k[i] = np.concatenate([self.hist_k[i], k_step[i : i + 1]], axis=0)
            self.hist_v[i] = np.concatenate([self.hist_v[i], v_step[i : i + 1]], axis=0)


@unittest.skipIf(_tpu_device_count() < 2, "Requires >= 2 TPU devices")
class TestBatchedRpaPcpRing(CustomTestCase):
    def test_pcp2_multichunk_read_only_and_dcp_decode(self):
        """Two chunks of PCP=2 prefill (read-only + companion stripe write)
        followed by two DCP ``STRIPED`` decode steps over the same striped cache."""
        h = _PcpRingHarness(2, nq=16, nkv=2, hd=128, n_req=2, pages_per_req=8, seed=11)
        h.run_chunk(self, "pcp2 chunk0", [1024, 600])
        h.run_chunk(self, "pcp2 chunk1", [1024, 900])
        h.run_dcp_decode_step(self, "pcp2->dcp2 step0")
        h.run_dcp_decode_step(self, "pcp2->dcp2 step1")

    def test_pcp2_in_kernel_writeback_parity(self):
        """Parity check for ``update_kv_cache=True`` with ``new_kv_write_rows``
        across multiple chunks, including a chunk wider than one K block."""
        h = _PcpRingHarness(2, nq=16, nkv=2, hd=128, n_req=1, pages_per_req=12, seed=22)
        h.run_chunk(self, "pcp2 wb chunk0", [2048], update_kv_cache=True)
        h.run_chunk(
            self, "pcp2 wb chunk1", [2048], update_kv_cache=True, token_major_q=True
        )

    @unittest.skipIf(_tpu_device_count() < 4, "Requires >= 4 TPU devices")
    def test_pcp4_unequal_stripes_and_short_requests(self):
        """Cached prefix not a multiple of ``page`` or ``P * page`` (unequal
        stripe lengths across ranks) and two requests of different lengths."""
        h = _PcpRingHarness(
            4,
            nq=16,
            nkv=2,
            hd=128,
            n_req=2,
            num_req_slots=4,
            pages_per_req=8,
            start_lens=[600, 0],
            seed=33,
        )
        h.run_chunk(self, "pcp4 unequal chunk0", [2500, 700])
        h.run_chunk(self, "pcp4 unequal chunk1", [900, 2048])
        h.run_dcp_decode_step(self, "pcp4->dcp4 decode")

    @unittest.skipIf(_tpu_device_count() < 8, "Requires >= 8 TPU devices")
    def test_pcp8_qwen3_30b_fp8_scales_and_dcp8_decode(self):
        """Target server shape: Qwen3-30B-A3B on 8 devices (PCP=8, 32 q heads,
        4 KV heads, head_dim 128, fp8 KV with static k_scale/v_scale, page 256),
        two prefill chunks + DCP8 ``STRIPED`` decode."""
        import jax.numpy as jnp

        h = _PcpRingHarness(
            8,
            nq=32,
            nkv=4,
            hd=128,
            kv_dtype=jnp.float8_e4m3fn,
            k_scale=0.5,
            v_scale=0.25,
            n_req=2,
            num_req_slots=4,
            pages_per_req=6,
            start_lens=[2300, 0],
            seed=44,
        )
        h.run_chunk(self, "pcp8 fp8 chunk0", [4096, 1500])
        h.run_chunk(self, "pcp8 fp8 chunk1", [4096, 4096])
        h.run_dcp_decode_step(self, "pcp8->dcp8 fp8 decode")

    @unittest.skipIf(_tpu_device_count() < 8, "Requires >= 8 TPU devices")
    def test_pcp4_x_tp2_2d_mesh(self):
        """2-D ``(pcp=4, model=2)`` mesh exercising multi-axis ring ``device_id``."""
        import jax.numpy as jnp

        h = _PcpRingHarness(
            4,
            mesh_shape=(4, 2),
            nq=16,
            nkv=2,
            hd=128,
            kv_dtype=jnp.float8_e4m3fn,
            k_scale=0.5,
            v_scale=0.5,
            n_req=1,
            pages_per_req=6,
            start_lens=[1500],
            seed=55,
        )
        h.run_chunk(self, "pcp4x2 chunk0", [2048])

    @unittest.skipIf(_tpu_device_count() < 8, "Requires >= 8 TPU devices")
    def test_pcp4_x_tp2_whole_stripe_prefix_skips_start_mask(self):
        """Cached prefixes in whole stripes (page x cp = 1024 tokens), as SGLang's
        ring mode guarantees: the new-token phase drops its start mask and must
        give the same attention, first chunk and cached-prefix chunk alike."""
        import jax.numpy as jnp

        h = _PcpRingHarness(
            4,
            mesh_shape=(4, 2),
            nq=16,
            nkv=2,
            hd=128,
            kv_dtype=jnp.float8_e4m3fn,
            k_scale=0.5,
            v_scale=0.5,
            n_req=2,
            pages_per_req=8,
            start_lens=[0, 2048],
            seed=77,
        )
        h.run_chunk(
            self, "aligned chunk0", [2048, 1024], cache_lens_cp_page_aligned=True
        )
        h.run_chunk(
            self, "aligned chunk1", [1024, 2048], cache_lens_cp_page_aligned=True
        )

    @unittest.skipIf(_tpu_device_count() < 8, "Requires >= 8 TPU devices")
    def test_pcp8_128k_long_context(self):
        """128k context (124k cached prefix + 4k new chunk) on PCP=8 with fp8 KV,
        compared against a streamed JAX float32 online-softmax reference."""
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.attention.pallas.batched_rpa_pcp import wrapper

        cp, page = 8, PAGE_SIZE
        nq, nkv, hd = 32, 4, 128
        grp = nq // nkv
        context_len = 131072
        req_chunk = 2 * cp * page  # 4096
        cached_len = context_len - req_chunk
        gpage = cp * page
        pages_per_req = context_len // gpage  # 64
        k_scale, v_scale = 0.5, 0.25
        sm_scale = 1.0 / math.sqrt(hd)
        fp8, bf16 = jnp.float8_e4m3fn, jnp.bfloat16

        mesh = jax.sharding.Mesh(np.array(jax.devices()[:cp]), ("model",))
        key = jax.random.PRNGKey(99)
        kq, kk_p, kv_p, kk_c, kv_c = jax.random.split(key, 5)
        q_full = jax.random.normal(kq, (req_chunk, nq, hd), dtype=bf16)
        k_prev = jax.random.normal(kk_p, (cached_len, nkv, hd), dtype=bf16).astype(fp8)
        v_prev = jax.random.normal(kv_p, (cached_len, nkv, hd), dtype=bf16).astype(fp8)
        k_curr = jax.random.normal(kk_c, (req_chunk, nkv, hd), dtype=bf16).astype(fp8)
        v_curr = jax.random.normal(kv_c, (req_chunk, nkv, hd), dtype=bf16).astype(fp8)

        # Pack striped cache on device: [pages_per_req, cp, page, nkv, hd] -> per-rank HAS cache.
        per_shape = wrapper.get_kv_cache_shape(pages_per_req, page, nkv, hd, fp8)
        # The pool holds whole pages: zero-fill past the cached prefix (the
        # kernel never reads those slots).
        pad = ((0, context_len - cached_len), (0, 0), (0, 0))
        kv_inter = jnp.stack(
            [jnp.pad(k_prev, pad), jnp.pad(v_prev, pad)], axis=2
        ).reshape(pages_per_req, cp, page, 2 * nkv, hd)
        kv_inter = jnp.transpose(kv_inter, (1, 0, 2, 3, 4)).reshape(
            cp, pages_per_req, page, per_shape[2], per_shape[3], per_shape[4]
        )
        cache_dev = jax.device_put(kv_inter, NamedSharding(mesh, P("model")))

        lay = _PcpChunkLayout(
            cp,
            page,
            [cached_len],
            [req_chunk],
            num_req_slots=2,
            pages_per_req=pages_per_req,
        )
        t_local = lay.t_local  # 512

        def _zig(arr):
            # [2*cp, page, ...] -> rank r holds (r, 2*cp - 1 - r)
            blks = arr.reshape((2 * cp, page) + arr.shape[1:])
            rows = [
                jnp.concatenate([blks[r], blks[2 * cp - 1 - r]], axis=0)
                for r in range(cp)
            ]
            return jax.device_put(
                jnp.stack(rows, axis=0).reshape((cp * t_local,) + arr.shape[1:]),
                NamedSharding(mesh, P("model")),
            )

        q_dev, k_dev, v_dev = _zig(q_full), _zig(k_curr), _zig(v_curr)
        crow = lay.cache_rows()
        nrows = [lay.new_rows(r) for r in range(cp)]
        table = np.zeros((2, pages_per_req), dtype=np.int32)
        table[0] = np.arange(pages_per_req, dtype=np.int32)
        new_table = np.repeat(table, 2, axis=0)

        ncu_dev = jax.device_put(
            jnp.asarray(np.stack([n["new_cu_q_lens"] for n in nrows])),
            NamedSharding(mesh, P("model")),
        )
        qpos_dev = jax.device_put(
            jnp.asarray(np.stack([n["q_positions"] for n in nrows])),
            NamedSharding(mesh, P("model")),
        )

        ckl = jnp.asarray(crow["cache_kv_lens"], jnp.int32)
        cpi = jnp.asarray(table.reshape(-1), jnp.int32)
        ccu = jnp.asarray(crow["cache_cu_q_lens"], jnp.int32)
        cdist = jnp.asarray(crow["cache_distribution"], jnp.int32)
        nkl = jnp.asarray(nrows[0]["new_kv_lens"], jnp.int32)
        npi = jnp.asarray(new_table.reshape(-1), jnp.int32)
        ndist = jnp.asarray(nrows[0]["new_distribution"], jnp.int32)
        nkpi = jnp.asarray(nrows[0]["new_kv_page_indices"], jnp.int32)
        kcl = jnp.asarray(nrows[0]["kv_cache_lens"], jnp.int32)

        def per_rank(q, k, v, cache_b, ncu_b, qpos_b):
            cp_rank = jnp.reshape(jax.lax.axis_index("model"), (1,)).astype(jnp.int32)
            k_all = jax.lax.all_gather(k, "model", axis=0, tiled=True)
            v_all = jax.lax.all_gather(v, "model", axis=0, tiled=True)
            return wrapper.ragged_paged_attention_pcp_fused(
                q,
                k_all,
                v_all,
                cache_b[0],
                cache_kv_lens=ckl,
                cache_page_indices=cpi,
                cache_cu_q_lens=ccu,
                cache_distribution=cdist,
                new_kv_lens=nkl,
                new_page_indices=npi,
                new_cu_q_lens=ncu_b[0],
                new_distribution=ndist,
                new_kv_page_indices=nkpi,
                q_positions=qpos_b[0],
                kv_cache_lens=kcl,
                cp_rank=cp_rank,
                cp_group_size=cp,
                ring_axis_name="model",
                ring_mesh_axis_names=("model",),
                sm_scale=sm_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                update_kv_cache=False,
            )

        out_dev = jax.jit(
            jax.shard_map(
                per_rank,
                mesh=mesh,
                in_specs=(P("model"),) * 6,
                out_specs=P("model"),
                check_vma=False,
            )
        )(q_dev, k_dev, v_dev, cache_dev, ncu_dev, qpos_dev)
        out_ranks = np.asarray(out_dev.astype(jnp.float32)).reshape(cp, 2, page, nq, hd)
        got = np.zeros((2 * cp, page, nq, hd), np.float32)
        for r in range(cp):
            got[r] = out_ranks[r, 0]
            got[2 * cp - 1 - r] = out_ranks[r, 1]
        got = got.reshape(req_chunk, nq, hd)

        @jax.jit
        def _ref_long(q_val, k_p, v_p, k_c, v_c):
            qf = q_val.astype(jnp.float32)
            kc = jnp.repeat(k_c.astype(jnp.float32) * k_scale, grp, axis=1)
            vc = jnp.repeat(v_c.astype(jnp.float32) * v_scale, grp, axis=1)
            sc_curr = (
                jnp.einsum("qhd,khd->hqk", qf, kc, preferred_element_type=jnp.float32)
                * sm_scale
            )
            cmask = jnp.tril(jnp.ones((req_chunk, req_chunk), dtype=bool))
            sc_curr = jnp.where(cmask[None, :, :], sc_curr, -1e9)
            cm = jnp.max(sc_curr, axis=-1, keepdims=True)
            cp_exp = jnp.exp(sc_curr - cm)
            cden = jnp.sum(cp_exp, axis=-1, keepdims=True)
            curr_lse = jnp.swapaxes(jnp.squeeze(cm + jnp.log(cden), axis=-1), 0, 1)
            curr_out = jnp.einsum("hqk,khd->qhd", cp_exp / cden, vc)

            chunk_sz = 512
            n_ch = cached_len // chunk_sz
            kp_ch = k_p.reshape(n_ch, chunk_sz, nkv, hd)
            vp_ch = v_p.reshape(n_ch, chunk_sz, nkv, hd)

            def scan_fn(carry, elem):
                m, l_exp, acc = carry
                k_ch, v_ch = elem
                k_ch = jnp.repeat(k_ch.astype(jnp.float32) * k_scale, grp, axis=1)
                v_ch = jnp.repeat(v_ch.astype(jnp.float32) * v_scale, grp, axis=1)
                sc = (
                    jnp.einsum(
                        "qhd,khd->hqk", qf, k_ch, preferred_element_type=jnp.float32
                    )
                    * sm_scale
                )
                ch_m = jnp.max(sc, axis=-1, keepdims=True)
                new_m = jnp.maximum(m, ch_m)
                alpha = jnp.exp(m - new_m)
                p = jnp.exp(sc - new_m)
                new_l = l_exp * alpha + jnp.sum(p, axis=-1, keepdims=True)
                new_acc = acc * jnp.swapaxes(alpha, 0, 1) + jnp.einsum(
                    "hqk,khd->qhd", p, v_ch
                )
                return (new_m, new_l, new_acc), None

            init = (
                jnp.full((nq, req_chunk, 1), -1e9, jnp.float32),
                jnp.zeros((nq, req_chunk, 1), jnp.float32),
                jnp.zeros((req_chunk, nq, hd), jnp.float32),
            )
            (fm, fl, facc), _ = jax.lax.scan(scan_fn, init, (kp_ch, vp_ch))
            ctx_out = facc / jnp.swapaxes(fl, 0, 1)
            ctx_lse = jnp.swapaxes(jnp.squeeze(fm + jnp.log(fl), axis=-1), 0, 1)
            m = jnp.maximum(ctx_lse, curr_lse)
            w0, w1 = jnp.exp(ctx_lse - m), jnp.exp(curr_lse - m)
            return (
                (ctx_out * w0[..., None] + curr_out * w1[..., None])
                / (w0 + w1)[..., None]
            ).astype(bf16)

        expected = np.asarray(
            _ref_long(q_full, k_prev, v_prev, k_curr, v_curr).astype(jnp.float32)
        )
        max_abs = float(np.max(np.abs(got - expected)))
        rel = max_abs / max(float(np.max(np.abs(expected))), 1e-6)
        if max_abs > 5e-4:
            self.assertLess(
                rel,
                TOLERANCE,
                f"128k PCP=8 mismatch: rel_err={rel:.4f} max_abs={max_abs:.4f}",
            )


if __name__ == "__main__":
    unittest.main()
