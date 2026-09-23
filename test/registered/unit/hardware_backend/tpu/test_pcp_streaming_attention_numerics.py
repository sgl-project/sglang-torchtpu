# SPDX-License-Identifier: Apache-2.0

"""Numerical parity for the PCP streaming attention path on real TPU devices.

Every other PCP test in this directory checks index arithmetic on the host.
This one runs the actual Pallas kernels over a ``pcp``-axis mesh of
``PCP_SIZE`` TPU devices and checks the two things that arithmetic cannot:

1. the attention output matches a naive causal-attention reference, and
2. the fused KV writeback lands the current chunk's K/V in exactly the cache
   slots the paged layout says it should.

(2) matters because the writeback no longer happens before the kernel call.
The current pass captures fresh KV off the ring into a dense staging buffer
keyed by *cache owner*, and a donated segmented copy moves it into the cache
between the current and history passes. A wrong capture offset produces no
error at all -- just wrong attention a few steps later, once the mis-placed
rows become history.

Coverage spans fresh prefill, chunked-prefill continuation (non-empty history),
multi-request batches, a decode-shaped batch, and the head_dim / ring-tile
geometries the served models actually run at.

Importing sglang configures the TorchTPU Pallas export environment, which pins
``JAX_PLATFORMS=cpu`` for the whole process -- so a test that imports sglang can
never see a TPU-backed ``jax.devices()``. The cases therefore run in a
subprocess that initializes JAX against the TPU backend *before* importing
sglang. ``python <this file>`` runs the cases directly, without pytest.

Usage::

    cd /path/to/sglang-torchtpu
    python -m pytest test/registered/unit/hardware_backend/tpu/test_pcp_streaming_attention_numerics.py
"""

import math
import os
import subprocess
import sys
import unittest

import msgspec
import numpy as np

if "--case" in sys.argv:
    # Subprocess entry. Bind JAX to the TPU backend before anything imports
    # sglang, which pins JAX_PLATFORMS=cpu process-wide for the Pallas export
    # path. Once JAX has initialized a backend that pin no longer applies.
    import jax

    try:
        jax.devices()
    except Exception:  # noqa: BLE001  (reported with a reason in _main)
        pass

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=300, suite="stage-a-test-tpu-unit")

PCP_SIZE = 4
# bf16 Q/K/V through an online-softmax ring accumulate. The reference is fp32,
# so the gap is dominated by input rounding, not by the kernel.
RTOL = 3e-2


class Geometry(msgspec.Struct, frozen=True):
    """Attention shape one case runs at.

    The ring tile (``q_block``) is what production derives per model from
    ``resolve_pcp_ring_tile_tokens(local_q_heads)``, so the cases below pin the
    tile each served head count actually gets rather than a single constant.
    """

    head_dim: int = 128
    kv_heads: int = 2
    q_per_kv: int = 4
    page_size: int = 128
    q_block: int = 128

    @property
    def q_heads(self):
        return self.kv_heads * self.q_per_kv


# Qwen3-Coder shapes: head_dim 128, 8 local Q heads.
BASE = Geometry()
# Qwen3.5-397B at attn_tp=1: 32 Q heads over 2 KV heads, head_dim 256. The
# head budget in resolve_pcp_ring_tile_tokens gives this a ring tile of 512.
Q35_397B = Geometry(head_dim=256, kv_heads=2, q_per_kv=16, page_size=256, q_block=512)
# Qwen3.5-35B: 16 Q heads, so the same budget allows a 1024 tile.
Q35_35B = Geometry(head_dim=256, kv_heads=2, q_per_kv=8, page_size=256, q_block=1024)


class Case(msgspec.Struct, frozen=True):
    """One shape to run. ``q_block`` None means the geometry's own ring tile."""

    q_lens: list
    kv_lens: list
    interleave_size: int
    geom: Geometry = BASE
    q_block: int | None = None


def _batch_flat_owner_starts(q_lens):
    starts = np.zeros(len(q_lens), dtype=np.int64)
    if len(q_lens) > 1:
        np.cumsum(np.asarray(q_lens[:-1], dtype=np.int64), out=starts[1:])
    return starts


def _naive_causal_attention(q, k_full, v_full, abs_starts, q_lens, sm_scale, geom):
    """Request-major [T, q_heads, head_dim] -> same, in fp32."""
    out = np.zeros_like(q, dtype=np.float32)
    q_off = 0
    for req, q_len in enumerate(q_lens):
        k_ctx = k_full[req].astype(np.float32)
        v_ctx = v_full[req].astype(np.float32)
        for j in range(q_len):
            pos = abs_starts[req] + j
            q_row = (
                q[q_off + j]
                .astype(np.float32)
                .reshape(geom.kv_heads, geom.q_per_kv, geom.head_dim)
            )
            scores = np.einsum("hqd,shd->hqs", q_row, k_ctx[: pos + 1]) * sm_scale
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            out[q_off + j] = np.einsum("hqs,shd->hqd", probs, v_ctx[: pos + 1]).reshape(
                geom.q_heads, geom.head_dim
            )
        q_off += q_len
    return out


def _run_case_in_subprocess(case):
    """Run one case name in a fresh process with the TPU JAX backend."""
    env = dict(os.environ)
    env.pop("JAX_PLATFORMS", None)
    env.pop("XLA_FLAGS", None)
    repo_python = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "..",
        "python",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.abspath(repo_python), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--case", case],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )


class TestPcpStreamingAttentionNumerics(CustomTestCase):
    """PCP attention + fused KV writeback against a naive reference."""

    def _check(self, case):
        result = _run_case_in_subprocess(case)
        if result.returncode == _EXIT_NO_DEVICES:
            raise unittest.SkipTest("PCP numerics need a TPU; none available.")
        self.assertNotEqual(
            result.returncode,
            _EXIT_TOO_FEW_DEVICES,
            f"PCP numerics need {PCP_SIZE} TPU devices on this runner:\n"
            f"{result.stdout[-2000:]}",
        )
        self.assertEqual(
            result.returncode,
            0,
            f"case {case!r} failed:\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}",
        )

    def _run_case(
        self, *, q_lens, kv_lens, interleave_size, geom=BASE, q_block=None, seed=0
    ):
        import jax
        import jax.numpy as jnp
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec as P

        from sglang.srt.layers.attention.pallas.batched_rpa import (
            wrapper as batched_rpa_wrapper,
        )
        from sglang.srt.layers.attention.pallas.pcp_streaming_rpa.pcp_layout import (
            build_pcp_rank_major_token_order,
        )
        from sglang.srt.layers.attention.pallas.pcp_streaming_rpa.wrapper import (
            PCP_AXIS_NAME,
            sharded_pcp_ragged_paged_attention,
        )

        q_block = geom.q_block if q_block is None else q_block
        dtype = jnp.bfloat16
        rng = np.random.default_rng(seed)
        num_reqs = len(q_lens)
        abs_starts = [kv_lens[i] - q_lens[i] for i in range(num_reqs)]
        total_q = sum(q_lens)
        sm_scale = 1.0 / math.sqrt(geom.head_dim)

        k_full = [
            rng.normal(size=(kv_lens[r], geom.kv_heads, geom.head_dim)).astype(
                np.float32
            )
            for r in range(num_reqs)
        ]
        v_full = [
            rng.normal(size=(kv_lens[r], geom.kv_heads, geom.head_dim)).astype(
                np.float32
            )
            for r in range(num_reqs)
        ]
        q_request_major = rng.normal(
            size=(total_q, geom.q_heads, geom.head_dim)
        ).astype(np.float32)
        # The K/V the kernel is handed are exactly the current chunk, i.e. the
        # tail of each request's history.
        k_current = np.concatenate(
            [k_full[r][abs_starts[r] :] for r in range(num_reqs)]
        )
        v_current = np.concatenate(
            [v_full[r][abs_starts[r] :] for r in range(num_reqs)]
        )

        kernel_bucket = PCP_SIZE * q_block
        padded = max(
            kernel_bucket,
            ((total_q + kernel_bucket - 1) // kernel_bucket) * kernel_bucket,
        )
        token_order, inverse_order = build_pcp_rank_major_token_order(
            q_lens,
            pcp_size=PCP_SIZE,
            interleave_size=interleave_size,
            padded_num_tokens=padded,
            token_owner_start_offsets_per_req=_batch_flat_owner_starts(q_lens),
        )

        def pack_rank_major(x):
            out = np.zeros((padded,) + x.shape[1:], dtype=x.dtype)
            valid = token_order >= 0
            out[valid] = x[token_order[valid]]
            return out

        # One block-table entry covers a virtual page of page_size * pcp_size
        # tokens; ownership rotates every interleave_size tokens inside it.
        virtual_page = geom.page_size * PCP_SIZE
        pages_per_req = max(
            1,
            max(
                (kv_lens[r] + virtual_page - 1) // virtual_page for r in range(num_reqs)
            ),
        )
        local_pages = pages_per_req * num_reqs
        block_tables = np.arange(local_pages, dtype=np.int32).reshape(
            num_reqs, pages_per_req
        )

        def pack_kv_row(k_row, v_row):
            _, packed = batched_rpa_wrapper.prepare_inputs(
                jnp.zeros((1, geom.q_heads, geom.head_dim), dtype),
                jnp.asarray(k_row[None].astype(np.float32), dtype),
                jnp.asarray(v_row[None].astype(np.float32), dtype),
                dtype,
                dtype,
            )
            return np.asarray(packed.astype(jnp.float32))[0]

        groups, packing = pack_kv_row(
            np.zeros((geom.kv_heads, geom.head_dim)),
            np.zeros((geom.kv_heads, geom.head_dim)),
        ).shape[:2]
        cache = np.zeros(
            (PCP_SIZE * local_pages, geom.page_size, groups, packing, geom.head_dim),
            dtype=np.float32,
        )

        def write_cache(dst, req, pos, k_row, v_row):
            vpage, voff = pos // virtual_page, pos % virtual_page
            rank = (voff // interleave_size) % PCP_SIZE
            offset = (voff // (PCP_SIZE * interleave_size)) * interleave_size + (
                voff % interleave_size
            )
            row = rank * local_pages + int(block_tables[req, vpage])
            dst[row, offset] = pack_kv_row(k_row, v_row)

        # Seed history only. The kernel is responsible for the current chunk.
        for req in range(num_reqs):
            for pos in range(abs_starts[req]):
                write_cache(cache, req, pos, k_full[req][pos], v_full[req][pos])
        expected_cache = cache.copy()
        for req in range(num_reqs):
            for pos in range(abs_starts[req], kv_lens[req]):
                write_cache(
                    expected_cache, req, pos, k_full[req][pos], v_full[req][pos]
                )

        mesh = Mesh(np.array(jax.devices()[:PCP_SIZE]), (PCP_AXIS_NAME,))
        cu_q_lens = np.zeros(num_reqs + 1, dtype=np.int32)
        cu_q_lens[1:] = np.cumsum(q_lens)
        decode_end = num_reqs if all(q == 1 for q in q_lens) else 0
        distribution = np.array([decode_end, num_reqs, num_reqs], dtype=np.int32)

        def shard(value, spec):
            return jax.device_put(value, jax.sharding.NamedSharding(mesh, spec))

        sharded = jax.jit(
            lambda *args: sharded_pcp_ragged_paged_attention(
                mesh,
                *args,
                attention_sink=None,
                sm_scale=sm_scale,
                cp_kv_cache_interleave_size=interleave_size,
                q_block_size=q_block,
            )
        )
        output, new_cache = sharded(
            shard(
                jnp.asarray(pack_rank_major(q_request_major), dtype), P(PCP_AXIS_NAME)
            ),
            shard(jnp.asarray(pack_rank_major(k_current), dtype), P(PCP_AXIS_NAME)),
            shard(jnp.asarray(pack_rank_major(v_current), dtype), P(PCP_AXIS_NAME)),
            shard(jnp.asarray(cache, dtype), P(PCP_AXIS_NAME)),
            shard(jnp.asarray(np.asarray(kv_lens, np.int32)), P()),
            shard(jnp.asarray(block_tables.reshape(-1)), P()),
            shard(jnp.asarray(cu_q_lens), P()),
            shard(jnp.asarray(distribution), P()),
        )

        got = np.asarray(output.astype(jnp.float32)).reshape(
            padded, geom.q_heads, geom.head_dim
        )[inverse_order]
        want = _naive_causal_attention(
            q_request_major, k_full, v_full, abs_starts, q_lens, sm_scale, geom
        )
        attn_rel = np.abs(got - want).max() / max(np.abs(want).max(), 1e-6)
        self.assertLess(
            attn_rel,
            RTOL,
            f"attention output mismatch: max relative error {attn_rel:.3e}",
        )

        got_cache = np.asarray(new_cache.astype(jnp.float32))
        cache_rel = np.abs(got_cache - expected_cache).max() / max(
            np.abs(expected_cache).max(), 1e-6
        )
        # The writeback moves already-quantized cache-dtype bytes, so it is
        # exact -- any nonzero error here is a mis-routed row, not rounding.
        self.assertLess(
            cache_rel,
            RTOL,
            f"fused KV writeback landed wrong bytes: max relative error {cache_rel:.3e}",
        )

    def test_fresh_prefill_single_request(self):
        self._check("fresh_prefill_single_request")

    def test_fresh_prefill_multi_request(self):
        self._check("fresh_prefill_multi_request")

    def test_chunked_prefill_continuation(self):
        self._check("chunked_prefill_continuation")

    def test_decode_batch(self):
        self._check("decode_batch")

    def test_large_ring_tile(self):
        self._check("large_ring_tile")

    def test_qwen35_head_dim_256(self):
        self._check("qwen35_head_dim_256")

    def test_qwen35_served_ring_tiles(self):
        self._check("qwen35_served_ring_tiles")


# name -> list of Case
CASES = {
    # Single request: the only shape the pre-batch-flat ownership scheme could
    # ever pack, kept as the baseline.
    "fresh_prefill_single_request": [
        Case([1024], [1024], 128),
        Case([2048], [2048], 128),
    ],
    # Every request starts at absolute 0, so request-absolute ownership would
    # pile all of them onto rank 0. Batch-flat keeps them balanced.
    "fresh_prefill_multi_request": [
        Case([768, 256], [768, 256], 128),
        Case([512, 384, 128], [512, 384, 128], 64),
    ],
    # Non-empty history: exercises the history no-causal pass and the boundary
    # page that straddles the current chunk's start.
    "chunked_prefill_continuation": [
        Case([1024], [2048], 128),
        Case([512, 512], [1536, 1024], 128),
    ],
    # One query row per request against a long history.
    "decode_batch": [
        Case([1] * 8, [1024 + 13 * i for i in range(8)], 1),
    ],
    # A ring tile larger than one Q compute block. The tile sets how many
    # local Q rows share a single trip around the ring, so this is the shape
    # resolve_pcp_ring_tile_tokens() picks for anything below ~128 Q heads,
    # and it exercises the multi-Q-block path inside one ring pass.
    "large_ring_tile": [
        Case([4096], [4096], 128, q_block=512),
        Case([2048, 2048], [2048, 2048], 128, q_block=512),
        Case([2048], [8192], 128, q_block=512),
    ],
    # head_dim 256 with a wide Q:KV ratio. Every shape above is head_dim 128;
    # the packed KV layout and the online-softmax accumulator are both sized
    # off head_dim, so 256 is a genuinely separate path.
    "qwen35_head_dim_256": [
        Case([1024], [1024], 256, geom=Q35_397B, q_block=256),
        Case([1024], [3072], 256, geom=Q35_397B, q_block=256),
        Case([512, 512], [512, 512], 256, geom=Q35_35B, q_block=256),
    ],
    # The ring tiles resolve_pcp_ring_tile_tokens() actually hands the served
    # Qwen3.5 head counts: 512 at 32 Q heads, 1024 at 16. Both sit exactly on
    # the tile x heads <= 16384 budget, so this is the case that fails first if
    # that budget is ever raised.
    "qwen35_served_ring_tiles": [
        Case([2048], [2048], 256, geom=Q35_397B),
        Case([4096], [4096], 256, geom=Q35_35B),
    ],
}

_EXIT_NO_DEVICES = 77
# A host that has TPUs but fewer than PCP_SIZE of them is a misconfigured
# runner, not an absent accelerator. Reported separately so it fails loudly:
# this file is the only end-to-end correctness gate the PCP path has, and
# silently skipping it is indistinguishable from passing.
_EXIT_TOO_FEW_DEVICES = 78


def _run_one_case(argv):
    """Subprocess entry behind the explicit ``--case`` flag."""
    case = argv[argv.index("--case") + 1]
    # JAX was bound to the TPU backend at import time, above.
    import jax

    try:
        num_devices = len(jax.devices())
    except Exception as exc:  # noqa: BLE001
        # libtpu is single-process. If the parent pytest process already holds
        # it (importing sglang starts the torch_tpu runtime), this subprocess
        # cannot attach. Report unavailable so the case skips instead of
        # reporting a numerics failure it never measured.
        print(f"TPU unavailable to this process: {type(exc).__name__}: {exc}")
        return _EXIT_NO_DEVICES
    if num_devices == 0:
        print("no JAX devices")
        return _EXIT_NO_DEVICES
    if num_devices < PCP_SIZE:
        print(f"only {num_devices} JAX devices; need {PCP_SIZE}")
        return _EXIT_TOO_FEW_DEVICES

    runner = TestPcpStreamingAttentionNumerics("test_fresh_prefill_single_request")
    for spec in CASES[case]:
        q_lens, kv_lens, interleave_size = (
            spec.q_lens,
            spec.kv_lens,
            spec.interleave_size,
        )
        q_block = spec.q_block if spec.q_block is not None else spec.geom.q_block
        runner._run_case(
            geom=spec.geom,
            q_lens=q_lens,
            kv_lens=kv_lens,
            interleave_size=interleave_size,
            q_block=q_block,
        )
        print(
            f"ok: {case} q_lens={q_lens} kv_lens={kv_lens} "
            f"ilv={interleave_size} q_block={q_block}"
        )
    return 0


if __name__ == "__main__":
    if "--case" in sys.argv:
        sys.exit(_run_one_case(sys.argv))
    unittest.main()
