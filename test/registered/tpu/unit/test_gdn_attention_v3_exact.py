"""Bit-exactness of the GDN/KDA kernel's layout and fast-path variants.

The TPU state pool stores conv state in the kernel-native rank-4
``[slots, K-1, 1, dim]`` form. That only skips a relayout, so the kernel must
produce the same bits (output, conv state, recurrent state) as with the rank-3
pool, for GDN (Qwen3.5) and KDA (Kimi), on batches that read non-zero initial
state.

The decode/extend fast paths are held to the same bar against the default
two-pass call: single-pass ``batched_only`` (bf16, unpadded) and
``prefill_only``, packed KDA weights, and packed SMEM metadata. Decode padding
rows, including the only row of an idle DP-attention rank's bucket, stay zero.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)

import dataclasses
import unittest
from unittest import mock

from sglang.test.test_utils import CustomTestCase


def _has_jax_tpu() -> bool:
    try:
        import jax
    except Exception:
        return False
    try:
        return any("TPU" in d.device_kind for d in jax.devices())
    except Exception:
        return False


if _has_jax_tpu():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from sglang_tpu.layers.attention.linear.kda_layout import pack_kda_weights
    from sglang_tpu.layers.attention.pallas.gdn import fused_conv1d_gdn, memory_ref
    from sglang_tpu.layers.attention.pallas.gdn.config import AttentionMode

_K = 4
_D = 128
# (name, lengths, distribution): decode rows first, then prefill rows.
_BATCHES = [
    ("decode", [1] * 8, [8, 8, 8]),
    ("mixed", [1, 1, 1, 40], [3, 4, 4]),
    ("prefill", [24, 40], [0, 2, 2]),
]
# Decode batch sizes: one row, a partial decode tile, full.
_DECODE_SIZES = (1, 5, 8)


def _case(kda: bool, lengths, distribution, seed=0, kda_heads=4):
    """Numpy inputs for one call; every slot starts from non-zero state."""
    rng = np.random.default_rng(seed)
    n_kq, n_v = (kda_heads, kda_heads) if kda else (2, 8)
    dim = 2 * n_kq * _D + n_v * _D
    num_seqs, num_tokens = len(lengths), sum(lengths)

    def rn(*shape, scale=1.0):
        return (scale * rng.standard_normal(shape)).astype(np.float32)

    arrays = dict(
        qkv=rn(num_tokens, dim).astype(jnp.bfloat16),
        b=rn(num_tokens, n_v).astype(jnp.bfloat16),
        a=rn(num_tokens, n_v * _D if kda else n_v).astype(jnp.bfloat16),
        conv_state=rn(num_seqs + 1, _K - 1, dim),
        recurrent_state=rn(num_seqs + 1, n_v, _D, _D, scale=0.1),
        conv_weight=rn(dim, 1, _K),
        conv_bias=None if kda else rn(dim),
        a_log=rn(n_v),
        dt_bias=rn(n_v, _D) if kda else rn(n_v),
        query_start_loc=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32),
        state_indices=np.arange(1, num_seqs + 1, dtype=np.int32),
        distribution=np.array(distribution, np.int32),
        # 5 tokens of context each: has_initial_state is True everywhere.
        seq_lens=(np.array(lengths) + 5).astype(np.int32),
    )
    static = dict(n_kq=n_kq, n_v=n_v, d_k=_D, d_v=_D, kernel_size=_K)
    if kda:
        static.update(attention_mode=AttentionMode.KDA, gate_lower_bound=-5.0)
    return arrays, static


def _call(arrays, static, kernel=None, **overrides):
    """Run the kernel on fresh device copies (the states are donated)."""
    inputs = {**arrays, **overrides}
    inputs = {
        k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in inputs.items()
    }
    (conv, rec), out = (kernel or fused_conv1d_gdn)(**inputs, **static)
    return np.asarray(conv), np.asarray(rec), np.asarray(out.astype(jnp.float32))


@unittest.skipUnless(_has_jax_tpu(), "Requires JAX with a TPU device")
class TestGDNKernelLayouts(CustomTestCase):
    def assert_same_bits(self, got, want, conv_shape=None):
        conv, rec, out = got
        if conv_shape is not None:
            self.assertEqual(conv.shape, conv_shape)
        np.testing.assert_array_equal(conv.reshape(want[0].shape), want[0])
        np.testing.assert_array_equal(rec, want[1])
        np.testing.assert_array_equal(out, want[2])

    def test_rank4_conv_state_is_bit_identical(self):
        for kda in (False, True):
            for name, lengths, dist in _BATCHES:
                with self.subTest(kda=kda, batch=name):
                    arrays, static = _case(kda, lengths, dist)
                    ref = _call(arrays, static)
                    conv = arrays["conv_state"]
                    conv4 = conv.reshape(conv.shape[0], _K - 1, 1, -1)
                    # Rank-4 pool: the state comes back in the pool's shape.
                    self.assert_same_bits(
                        _call(arrays, static, conv_state=conv4), ref, conv4.shape
                    )


def _packed_kda_weights(arrays, static):
    """The weights packed the way Kimi-K3 packs them."""
    import torch

    t = lambda name: torch.from_numpy(np.asarray(arrays[name]))
    packed = pack_kda_weights(
        t("conv_weight"), t("dt_bias"), t("a_log"), static["n_v"], static["d_k"]
    )
    return packed.numpy()


@unittest.skipUnless(_has_jax_tpu(), "Requires JAX with a TPU device")
class TestGDNKernelFastPaths(CustomTestCase):
    assert_same_bits = TestGDNKernelLayouts.assert_same_bits

    def test_single_pass_variants_are_bit_identical(self):
        for kda in (False, True):
            for bs in _DECODE_SIZES:
                with self.subTest(kda=kda, decode=bs):
                    arrays, static = _case(kda, [1] * bs, [bs] * 3)
                    self.assert_same_bits(
                        _call(arrays, static, batched_only=True),
                        _call(arrays, static),
                    )
            with self.subTest(kda=kda, batch="prefill"):
                arrays, static = _case(kda, *_BATCHES[2][1:])
                self.assert_same_bits(
                    _call(arrays, static, prefill_only=True), _call(arrays, static)
                )

    def test_decode_padding_rows_are_zero(self):
        """Rows past distribution[0] are bucket padding the kernel never walks.
        They must still come back zero, as from the default call: residual,
        MoE and collectives read them. Includes a one-row bucket with no real
        row, which an idle DP-attention rank runs as decode."""
        for kda in (False, True):
            for bs, real in ((1, 0), (4, 0), (4, 1)):
                with self.subTest(kda=kda, bucket=bs, real=real):
                    arrays, static = _case(kda, [1] * bs, [real] * 3)
                    got = _call(arrays, static, batched_only=True)
                    np.testing.assert_array_equal(got[2][real:], 0.0)
                    self.assert_same_bits(got, _call(arrays, static))

    def test_mixed_dtype_decode_is_bit_identical(self):
        """f32 b / a next to bf16 qkv take the f32 path on decode, as on the
        default call, instead of failing the kernel's dtype check."""
        for kda in (False, True):
            with self.subTest(kda=kda):
                arrays, static = _case(kda, [1] * 5, [5] * 3)
                f32 = {k: arrays[k].astype(np.float32) for k in ("b", "a")}
                self.assert_same_bits(
                    _call(arrays, static, batched_only=True, **f32),
                    _call(arrays, static, **f32),
                )

    def test_pack_metadata_keeps_every_set_field(self):
        """Every set MetadataRef field round-trips through the packed operand,
        so a field added later cannot be dropped; unset fields stay None."""
        rng = np.random.default_rng(0)

        def ints(*shape):
            return jnp.asarray(rng.integers(0, 9, shape), jnp.int32)

        fields = {}
        for field in dataclasses.fields(memory_ref.MetadataRef):
            if field.name == "num_tiles":
                fields[field.name] = jnp.int32(3)
            elif field.name.startswith("p_id_"):
                fields[field.name] = memory_ref.SmemWrapper(ints(8), (1, 4))
            elif field.name == "s_idx_has_initial_state":
                fields[field.name] = ints(5) > 4
            elif field.name == "s_idx_to_ckpt_indices":
                fields[field.name] = ints(5, 3)
            else:
                fields[field.name] = ints(5)
        metadata = memory_ref.MetadataRef(**fields)
        packed, layout = memory_ref.pack_metadata(metadata)
        self.assertEqual([entry[0] for entry in layout], list(fields))

        view = memory_ref.unpack_metadata(packed, layout)
        for name, want in fields.items():
            got = getattr(view, name)
            if isinstance(want, memory_ref.SmemWrapper):
                indices = [(p, i) for p in range(2) for i in range(4)]
                want = want.data.reshape(2, 4)
            else:
                indices = list(np.ndindex(want.shape))
            for idx in indices:
                self.assertEqual(int(got[idx]), int(want[idx]), (name, idx))

        unset = dataclasses.replace(metadata, s_idx_to_ckpt_indices=None)
        view = memory_ref.unpack_metadata(*memory_ref.pack_metadata(unset))
        self.assertIsNone(view.s_idx_to_ckpt_indices)

    def test_packed_kda_weights_are_bit_identical(self):
        # 3 heads: Kimi-K3 at TP32, dt_bias / A_log rows off the 8-row tile.
        for heads in (3, 4):
            for name, lengths, dist in _BATCHES:
                with self.subTest(heads=heads, batch=name):
                    arrays, static = _case(True, lengths, dist, kda_heads=heads)
                    packed = _packed_kda_weights(arrays, static)
                    self.assert_same_bits(
                        _call(arrays, static, packed_weights=packed),
                        _call(arrays, static),
                    )

    def test_packed_metadata_is_bit_identical(self):
        def unpacked_kernel():
            # jax caches traces by the jitted function, not the jit object:
            # re-jitting fused_conv1d_gdn.__wrapped__ would reuse the packed
            # trace of an earlier test and never call the patched pack_metadata.
            def kernel(*args, **kwargs):
                return fused_conv1d_gdn.__wrapped__(*args, **kwargs)

            return jax.jit(
                kernel,
                static_argnames=(
                    "n_kq",
                    "n_v",
                    "d_k",
                    "d_v",
                    "kernel_size",
                    "attention_mode",
                    "gate_lower_bound",
                ),
            )

        for kda in (False, True):
            for name, lengths, dist in _BATCHES:
                with self.subTest(kda=kda, batch=name):
                    arrays, static = _case(kda, lengths, dist)
                    with mock.patch.object(
                        memory_ref, "pack_metadata", side_effect=lambda md: (md, None)
                    ) as unpacked:
                        ref = _call(arrays, static, kernel=unpacked_kernel())
                    unpacked.assert_called()
                    self.assert_same_bits(_call(arrays, static), ref)


if __name__ == "__main__":
    unittest.main()
