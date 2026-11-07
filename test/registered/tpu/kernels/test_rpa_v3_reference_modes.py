"""RPA v3 kernel vs ``ref_ragged_paged_attention`` on production-only modes.

The unit file (test/registered/tpu/unit/test_ragged_paged_attention_v3.py)
covers head_dim 128, zero sinks and hand-aligned KV head counts. These cases
add what production runs and nothing else checks against the reference:
non-zero attention sinks, head_dim 64/256/512 (kernel-side lane padding), and
fp8 KV in the pool layout with padded KV-head slots, composed with sinks,
sliding window and soft cap on one batch that hits both DECODE and MIXED.

Every case builds the fused cache the way ``TpuFusedMHATokenToKVPool`` lays it
out: interleaved [K0, V0, K1, V1, ...] heads padded up to ``kv_packing`` and
head_dim padded to 128, on shuffled pages with the backend's rectangular
``cu_kv_lens``. The kernel writes the new K/V itself; every live slot is then
compared with the reference's pages, and pages outside the table must be
untouched.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=150,
    suite="stage-a-test-tpu-kernel",
    labels=["kernel", "attention", "rpa-v3"],
)

# This is a standalone raw-JAX kernel test, not a TorchTPU runtime process.
# SGLang platform discovery selects CPU-only JAX for export, so select the
# physical backend before the first JAX import.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"


def _has_jax_tpu() -> bool:
    try:
        import jax
    except Exception:
        return False
    try:
        return any("TPU" in d.device_kind for d in jax.devices())
    except Exception:
        return False


_SKIP_REASON = "Requires JAX with a TPU device"

# TPU platform default page size (GPT-OSS recipe).
PAGE_SIZE = 64
# Pages outside every sequence's table; a stray write lands here.
SPARE_PAGES = 4
TOLERANCE = 0.05
# Kernel feeds fp8 K/V to the MXU where the reference upcasts first.
FP8_TOLERANCE = 0.1
# Distinct powers of two: a k/v swap is caught and the reference's bf16
# dequant stays exact.
K_SCALE, V_SCALE = 0.5, 0.25


def _cdiv(a, b):
    return (a + b - 1) // b


def _align(x, a):
    return _cdiv(x, a) * a


def _quantize(t, scale, kv_dtype):
    """Mirror of the backend's ``_quantize_kv_to_fp8``; identity for bf16."""
    import jax.numpy as jnp

    if kv_dtype == jnp.bfloat16:
        return t
    finfo = jnp.finfo(kv_dtype)
    s = jnp.clip(t.astype(jnp.float32) / scale, float(finfo.min), float(finfo.max))
    return s.astype(kv_dtype)


def _build_case(
    seqs,
    *,
    num_q_heads,
    num_kv_heads,
    head_dim,
    kv_dtype,
    seed=0,
):
    """Kernel + reference inputs for ``seqs = [(kv_len, q_len), ...]``.

    ``kv_len`` includes the ``q_len`` new tokens the kernel appends.
    """
    import jax
    import jax.numpy as jnp

    from sglang_tpu.attention.pallas.util import get_dtype_packing

    is_fp8 = kv_dtype != jnp.bfloat16
    k_scale, v_scale = (K_SCALE, V_SCALE) if is_fp8 else (None, None)
    num_seqs = len(seqs)
    pages_per_seq = max(_cdiv(kv, PAGE_SIZE) for kv, _ in seqs)
    total_pages = num_seqs * pages_per_seq + SPARE_PAGES
    total_q = sum(q for _, q in seqs)
    max_num_tokens = _align(total_q, 8)

    rng = np.random.default_rng(seed)
    page_table = (
        rng.permutation(total_pages)[: num_seqs * pages_per_seq]
        .astype(np.int32)
        .reshape(num_seqs, pages_per_seq)
    )

    keys = jax.random.split(jax.random.PRNGKey(seed), 5)
    tok_shape = (max_num_tokens, num_kv_heads, head_dim)
    page_shape = (total_pages, PAGE_SIZE, num_kv_heads, head_dim)
    q = jax.random.normal(
        keys[0], (max_num_tokens, num_q_heads, head_dim), jnp.bfloat16
    )
    new_k = _quantize(
        jax.random.normal(keys[1], tok_shape, jnp.bfloat16), k_scale, kv_dtype
    )
    new_v = _quantize(
        jax.random.normal(keys[2], tok_shape, jnp.bfloat16), v_scale, kv_dtype
    )
    # Random pre-write content, so a skipped write cannot pass.
    k_pages = _quantize(
        jax.random.normal(keys[3], page_shape, jnp.bfloat16), k_scale, kv_dtype
    )
    v_pages = _quantize(
        jax.random.normal(keys[4], page_shape, jnp.bfloat16), v_scale, kv_dtype
    )

    ref_k, ref_v = k_pages, v_pages
    offset = 0
    for row, (kv_len, q_len) in enumerate(seqs):
        pos = np.arange(kv_len - q_len, kv_len)
        pages = page_table[row, pos // PAGE_SIZE]
        ref_k = ref_k.at[pages, pos % PAGE_SIZE].set(new_k[offset : offset + q_len])
        ref_v = ref_v.at[pages, pos % PAGE_SIZE].set(new_v[offset : offset + q_len])
        offset += q_len

    # Pool layout: [pages, page, align(2 * kv_heads, packing) / packing,
    # packing, align(head_dim, 128)], heads interleaved K0, V0, K1, V1, ...
    packing = get_dtype_packing(kv_dtype)
    heads_x2 = _align(2 * num_kv_heads, packing)
    padded_dim = _align(head_dim, 128)
    kv_fused = jnp.stack([k_pages, v_pages], axis=3).reshape(
        total_pages, PAGE_SIZE, 2 * num_kv_heads, head_dim
    )
    kv_fused = jnp.pad(
        kv_fused,
        ((0, 0), (0, 0), (0, heads_x2 - 2 * num_kv_heads), (0, padded_dim - head_dim)),
    ).reshape(total_pages, PAGE_SIZE, heads_x2 // packing, packing, padded_dim)

    return {
        "seqs": seqs,
        "q": q,
        "new_k": new_k,
        "new_v": new_v,
        "kv_fused": kv_fused,
        "kv_fused_before": np.asarray(kv_fused.astype(jnp.float32)),
        "ref_k": ref_k,
        "ref_v": ref_v,
        "page_table": page_table,
        "kv_lens": jnp.asarray([kv for kv, _ in seqs], jnp.int32),
        "cu_q_lens": jnp.asarray(np.cumsum([0] + [q for _, q in seqs]), jnp.int32),
        # Backend convention: cu_kv_lens[i] = i * pages_per_seq * page_size.
        "cu_kv_lens": jnp.asarray(
            np.arange(num_seqs + 1) * pages_per_seq * PAGE_SIZE, jnp.int32
        ),
        "total_q": total_q,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "heads_x2": heads_x2,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "tolerance": FP8_TOLERANCE if is_fp8 else TOLERANCE,
    }


def _sinks(num_q_heads, seed=0):
    """Distinct per-head sink logits; bf16 like the model parameter."""
    import jax.numpy as jnp

    rng = np.random.default_rng(seed + 1)
    # Large enough that dropping a sink or giving it to another head moves the
    # output by >= 0.3 on these batches (measured), well above tolerance.
    return jnp.asarray(rng.uniform(0.0, 6.0, num_q_heads), jnp.bfloat16)


def _reference(case, *, sinks=None, sliding_window=None, soft_cap=None):
    import jax.numpy as jnp

    from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
        ref_ragged_paged_attention,
    )

    out = ref_ragged_paged_attention(
        case["q"][: case["total_q"]],
        case["ref_k"],
        case["ref_v"],
        case["kv_lens"],
        jnp.asarray(case["page_table"]),
        case["cu_q_lens"],
        jnp.asarray([len(case["seqs"])], jnp.int32),
        sm_scale=case["head_dim"] ** -0.5,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        k_scale=case["k_scale"],
        v_scale=case["v_scale"],
        attention_sink=sinks,
    )
    return np.asarray(out.astype(jnp.float32))


@unittest.skipIf(not _has_jax_tpu(), _SKIP_REASON)
class TestRpaV3ReferenceModes(unittest.TestCase):
    def _run_and_compare(
        self,
        case,
        *,
        distribution,
        sinks=None,
        sliding_window=None,
        soft_cap=None,
        decode_query_size=1,
    ):
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
            ragged_paged_attention,
        )

        # Before the kernel call: q/k/v/cache are donated.
        expected = _reference(
            case, sinks=sinks, sliding_window=sliding_window, soft_cap=soft_cap
        )
        out, cache = ragged_paged_attention(
            case["q"],
            case["new_k"],
            case["new_v"],
            case["kv_fused"],
            case["kv_lens"],
            jnp.asarray(case["page_table"].reshape(-1)),
            case["cu_q_lens"],
            case["cu_kv_lens"],
            jnp.asarray(distribution, jnp.int32),
            None,  # custom_mask
            sinks,
            sm_scale=case["head_dim"] ** -0.5,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            k_scale=case["k_scale"],
            v_scale=case["v_scale"],
            decode_query_size=decode_query_size,
        )
        got = np.asarray(out[: case["total_q"]].astype(jnp.float32))
        self.assertEqual(got.shape, expected.shape)
        diff = float(np.abs(got - expected).max())
        self.assertLess(
            diff, case["tolerance"], f"max abs diff {diff} >= {case['tolerance']}"
        )
        self._assert_cache_written(case, cache)

    def _assert_cache_written(self, case, cache):
        """Live K/V slots hold the reference pages; spare pages are untouched."""
        import jax.numpy as jnp

        num_pages = case["kv_fused_before"].shape[0]
        shape = (num_pages, PAGE_SIZE, case["heads_x2"], -1)
        got = np.asarray(cache.astype(jnp.float32)).reshape(shape)
        ref_k = np.asarray(case["ref_k"].astype(jnp.float32))
        ref_v = np.asarray(case["ref_v"].astype(jnp.float32))
        heads_x2, head_dim = 2 * case["num_kv_heads"], case["head_dim"]
        for row, (kv_len, _) in enumerate(case["seqs"]):
            pos = np.arange(kv_len)
            pages = case["page_table"][row, pos // PAGE_SIZE]
            slots = pos % PAGE_SIZE
            np.testing.assert_array_equal(
                got[pages, slots, 0:heads_x2:2, :head_dim], ref_k[pages, slots]
            )
            np.testing.assert_array_equal(
                got[pages, slots, 1:heads_x2:2, :head_dim], ref_v[pages, slots]
            )
        spare = sorted(set(range(num_pages)) - set(case["page_table"].ravel()))
        before = case["kv_fused_before"].reshape(shape)
        np.testing.assert_array_equal(got[spare], before[spare])

    def test_attention_sinks(self):
        """Non-zero per-head sinks match the reference on every kernel route.

        Rows 0-1 run DECODE, rows 2-3 MIXED (one single-token, one 37-token
        extend spanning two q blocks, where sinks are re-tiled per block);
        the verify subcase runs static 4-row DECODE blocks. Two KV heads with
        distinct sinks catch a sink mapped to the wrong head.
        """
        import jax.numpy as jnp

        subcases = (
            (
                "decode_and_mixed",
                [(150, 1), (70, 1), (201, 1), (260, 37)],
                (2, 2, 4),
                1,
            ),
            ("spec_verify", [(90, 4), (131, 4)], (2, 2, 2), 4),
        )
        for name, seqs, distribution, width in subcases:
            with self.subTest(name):
                case = _build_case(
                    seqs,
                    num_q_heads=16,
                    num_kv_heads=2,
                    head_dim=128,
                    kv_dtype=jnp.bfloat16,
                )
                sinks = _sinks(16)
                # The sinks must move the answer, or this case checks nothing.
                self.assertGreater(
                    np.abs(_reference(case, sinks=sinks) - _reference(case)).max(),
                    4 * TOLERANCE,
                )
                self._run_and_compare(
                    case,
                    distribution=distribution,
                    sinks=sinks,
                    decode_query_size=width,
                )

    def test_head_dims(self):
        """head_dim 64 is padded to one lane tile inside the kernel and
        sliced back; 256 and 512 span several tiles and shrink the default
        blocks. Both KV dtypes: fp8 is the RPA default KV cache."""
        import jax.numpy as jnp

        for head_dim in (64, 256, 512):
            for kv_dtype in (jnp.bfloat16, jnp.float8_e4m3fn):
                with self.subTest(head_dim=head_dim, kv_dtype=kv_dtype):
                    case = _build_case(
                        [(150, 1), (201, 1), (260, 37)],
                        num_q_heads=8,
                        num_kv_heads=1,
                        head_dim=head_dim,
                        kv_dtype=kv_dtype,
                    )
                    self._run_and_compare(case, distribution=(1, 1, 3))

    def test_gpt_oss_production_config(self):
        """GPT-OSS-120B at tp8 per rank: 8 q heads over 1 KV head, head_dim
        64, fp8 e4m3 KV (one KV head pads to four fp8 head slots), page 64,
        per-head sinks; sliding-window layers (window 127 + 1 = 128 as the
        backend passes it) alternate with full layers.

        Row 0 (kv 390) lets the decode window skip whole KV blocks while the
        sink must still count; row 3 extends across the window edge.
        """
        import jax.numpy as jnp

        for sliding_window in (128, None):
            with self.subTest(sliding_window=sliding_window):
                case = _build_case(
                    [(390, 1), (100, 1), (77, 1), (300, 37)],
                    num_q_heads=8,
                    num_kv_heads=1,
                    head_dim=64,
                    kv_dtype=jnp.float8_e4m3fn,
                )
                self._run_and_compare(
                    case,
                    distribution=(2, 2, 4),
                    sinks=_sinks(8),
                    sliding_window=sliding_window,
                )

    def test_fp8_soft_cap_after_k_scale(self):
        """The cap applies to dequantized logits: k_scale must scale before
        tanh. A small cap makes tanh saturate on N(0, 1) logits, so moving
        the scale past the cap changes the output."""
        import jax.numpy as jnp

        case = _build_case(
            [(390, 1), (100, 1), (77, 1), (300, 37)],
            num_q_heads=8,
            num_kv_heads=1,
            head_dim=64,
            kv_dtype=jnp.float8_e4m3fn,
        )
        self._run_and_compare(
            case, distribution=(2, 2, 4), sliding_window=128, soft_cap=2.0
        )


if __name__ == "__main__":
    unittest.main()
