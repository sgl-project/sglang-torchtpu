"""TPU integration tests for expert-parallel (group_offset) fused MoE."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

import unittest

import torch  # noqa: F401

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:
    _HAS_TPU = False

_SKIP = "requires a JAX TPU backend"


def _make_inputs(key, m, hidden, intermediate, num_experts, topk, dtype=None):
    dtype = dtype or jnp.bfloat16
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    x = (jax.random.normal(k1, (m, hidden), jnp.float32) * 0.3).astype(dtype)
    w1 = (
        jax.random.normal(k2, (num_experts, hidden, 2 * intermediate), jnp.float32)
        * 0.05
    ).astype(dtype)
    w2 = (
        jax.random.normal(k3, (num_experts, intermediate, hidden), jnp.float32) * 0.05
    ).astype(dtype)
    topk_weights = jax.nn.softmax(
        jax.random.normal(k4, (m, topk), jnp.float32), axis=-1
    ).astype(dtype)
    topk_ids = jax.random.randint(k5, (m, topk), 0, num_experts).astype(jnp.int32)
    return x, w1, w2, topk_weights, topk_ids


@unittest.skipUnless(_HAS_TPU, _SKIP)
class TestFusedMoEExpertParallel(CustomTestCase):
    HIDDEN = 256
    INTERMEDIATE = 128
    GLOBAL_EXPERTS = 16
    EP_SIZE = 4
    TOPK = 4
    RTOL = 0.03

    def make_inputs(self, *, tokens, seed=0):
        return _make_inputs(
            jax.random.key(seed),
            tokens,
            self.HIDDEN,
            self.INTERMEDIATE,
            self.GLOBAL_EXPERTS,
            self.TOPK,
        )

    def run_kernel(
        self,
        *,
        x,
        w1,
        w2,
        topk_weights,
        topk_ids,
        use_sc_gather,
        no_combine=False,
        global_num_experts=0,
        expert_group_offset=0,
    ):
        from sglang_tpu.moe.pallas.fused_moe_op import _fused_moe_jax

        return _fused_moe_jax(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            w1_scale=None,
            w2_scale=None,
            w1_bias=None,
            w2_bias=None,
            topk=self.TOPK,
            activation_name="silu",
            no_combine=no_combine,
            global_num_experts=global_num_experts,
            expert_group_offset=expert_group_offset,
            use_sc_gather=use_sc_gather,
        )

    def run_full(self, inputs, *, use_sc_gather, no_combine=False):
        x, w1, w2, topk_weights, topk_ids = inputs
        return self.run_kernel(
            x=x,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            use_sc_gather=use_sc_gather,
            no_combine=no_combine,
        )

    def run_rank(self, rank, inputs, *, use_sc_gather, no_combine=False):
        x, w1, w2, topk_weights, topk_ids = inputs
        local = self.GLOBAL_EXPERTS // self.EP_SIZE
        offset = rank * local
        return self.run_kernel(
            x=x,
            w1=w1[offset : offset + local],
            w2=w2[offset : offset + local],
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            use_sc_gather=use_sc_gather,
            no_combine=no_combine,
            global_num_experts=self.GLOBAL_EXPERTS,
            expert_group_offset=offset,
        )

    def run_all_ranks(self, inputs, *, use_sc_gather):
        return [
            self.run_rank(rank, inputs, use_sc_gather=use_sc_gather)
            for rank in range(self.EP_SIZE)
        ]

    def assert_close(self, actual, expected, *, context=""):
        diff = jnp.max(
            jnp.abs(actual.astype(jnp.float32) - expected.astype(jnp.float32))
        )
        scale = jnp.maximum(jnp.max(jnp.abs(expected.astype(jnp.float32))), 1e-9)
        self.assertLess(float(diff / scale), self.RTOL, context)

    def test_ep_sum_matches_full_moe(self):
        """Per-rank partial outputs must sum to the full (non-EP) MoE."""
        cases = [
            # (name, tokens, use_sc_gather)
            # First/middle/last rank windows, SparseCore gathers.
            ("sparse_core_prefill", 512, 1),
            # Same invariant on the TensorCore fallback path.
            ("tensor_core_prefill", 512, 0),
            # Decode-sized batch takes the kernels' small-input fallbacks.
            ("sparse_core_decode", 16, 1),
            # Multi-block: exercise the full-range SC gather together with the
            # per-rank GMM and un-permute windows.
            ("sparse_core_multiblock", 4096, 1),
        ]
        for name, tokens, use_sc_gather in cases:
            with self.subTest(name=name):
                inputs = self.make_inputs(tokens=tokens)
                expected = self.run_full(inputs, use_sc_gather=use_sc_gather)
                partials = self.run_all_ranks(inputs, use_sc_gather=use_sc_gather)
                actual = jnp.stack(partials).sum(axis=0)
                self.assert_close(actual, expected, context=name)

    def test_sc_gather_matches_tensorcore_multiblock(self):
        """The full-range SparseCore gather must agree per-rank with the
        TensorCore gather.

        ``test_ep_sum_matches_full_moe`` compares a *sum* over ranks, which can
        hide a bug that moves contribution between ranks. This compares each
        rank directly at a token count spanning several SC blocks.

        Repeated because the GMM leaves non-local rows uninitialized and a fresh
        HBM buffer is often zero, so an un-permute masking bug can pass once and
        fail only after the allocator hands back a dirty buffer.
        """
        tokens = 4096  # out_size = tokens * TOPK = 16384 = 8 blocks of 2048
        local = self.GLOBAL_EXPERTS // self.EP_SIZE
        for attempt in range(5):
            inputs = self.make_inputs(tokens=tokens, seed=attempt)
            x, w1, w2, topk_weights, topk_ids = inputs
            for rank in range(self.EP_SIZE):
                offset = rank * local
                kw = dict(
                    x=x,
                    w1=w1[offset : offset + local],
                    w2=w2[offset : offset + local],
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    global_num_experts=self.GLOBAL_EXPERTS,
                    expert_group_offset=offset,
                )
                sparse_core = self.run_kernel(use_sc_gather=1, **kw)
                full_range = self.run_kernel(use_sc_gather=0, **kw)
                self.assert_close(
                    sparse_core,
                    full_range,
                    context=f"attempt={attempt} rank={rank}",
                )

    def test_rank_without_tokens(self):
        """A rank whose expert window receives no routed tokens contributes
        (near) zero and the partial sum still matches."""
        x, w1, w2, topk_weights, topk_ids = self.make_inputs(tokens=256)
        local = self.GLOBAL_EXPERTS // self.EP_SIZE
        # remap any id in rank 1's window [local, 2*local) to rank 0
        in_r1 = (topk_ids >= local) & (topk_ids < 2 * local)
        topk_ids = jnp.where(in_r1, topk_ids - local, topk_ids)
        inputs = (x, w1, w2, topk_weights, topk_ids)

        # The empty rank itself must contribute (near) zero.
        empty_rank_out = self.run_rank(1, inputs, use_sc_gather=1)
        self.assertLess(
            float(jnp.max(jnp.abs(empty_rank_out.astype(jnp.float32)))),
            1e-6,
            "rank with no routed tokens must output zeros",
        )

        expected = self.run_full(inputs, use_sc_gather=1)
        partials = self.run_all_ranks(inputs, use_sc_gather=1)
        actual = jnp.stack(partials).sum(axis=0)
        self.assert_close(actual, expected, context="rank_without_tokens")

    def test_no_combine_zeroes_non_local_slots(self):
        """no_combine output must be exactly zero for slots whose expert is
        not local — the caller's later weighted sum depends on it — while
        local slots must match the full (non-EP) per-slot output, so an
        all-zero implementation cannot pass."""
        inputs = self.make_inputs(tokens=128, seed=1)
        _, _, _, _, topk_ids = inputs
        local = self.GLOBAL_EXPERTS // self.EP_SIZE
        rank = 1
        out = self.run_rank(rank, inputs, use_sc_gather=1, no_combine=True)
        non_local = (topk_ids < rank * local) | (
            topk_ids >= (rank + 1) * local
        )  # [m, topk]
        masked = jnp.where(non_local[:, :, None], out.astype(jnp.float32), 0.0)
        self.assertEqual(float(jnp.max(jnp.abs(masked))), 0.0)

        # Local slots must carry the same per-slot expert outputs as the
        # full MoE run — guards against a trivially all-zero kernel.
        full = self.run_full(inputs, use_sc_gather=1, no_combine=True)
        local_mask = ~non_local[:, :, None]
        self.assert_close(
            jnp.where(local_mask, out.astype(jnp.float32), 0.0),
            jnp.where(local_mask, full.astype(jnp.float32), 0.0),
            context="no_combine local slots",
        )

    def test_no_combine_zeroes_weight_masked_slots_without_group_offset(self):
        """The TPU FP4 path marks non-local slots -1 (dropped) and zeros their
        kernel weights before calling the local Pallas kernel. no_combine must
        return zeros for those weight-masked slots because a model with a
        post-expert norm applies the original router weights later, when it
        finalizes the per-expert outputs.
        """
        x, w1, w2, topk_weights, topk_ids = self.make_inputs(tokens=128, seed=2)
        masked_weights = topk_weights.at[:, 1::2].set(0)
        local_ids = jnp.where(masked_weights == 0, 0, topk_ids % self.GLOBAL_EXPERTS)
        out = self.run_kernel(
            x=x,
            w1=w1,
            w2=w2,
            topk_weights=masked_weights,
            topk_ids=local_ids,
            use_sc_gather=1,
            no_combine=True,
        )
        masked = jnp.where(masked_weights[:, :, None] == 0, out.astype(jnp.float32), 0)
        self.assertEqual(float(jnp.max(jnp.abs(masked))), 0.0)

        # The old bug returned the remapped expert-0 placeholder output for
        # masked slots. Prove those placeholders are nonzero and would pollute
        # the caller's later combine with the original router weights.
        placeholder_out = self.run_kernel(
            x=x,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=local_ids,
            use_sc_gather=1,
            no_combine=True,
        ).astype(jnp.float32)
        masked_placeholder = jnp.where(
            masked_weights[:, :, None] == 0, placeholder_out, 0
        )
        self.assertGreater(float(jnp.max(jnp.abs(masked_placeholder))), 1e-6)

        original_weights = topk_weights[:, :, None].astype(jnp.float32)
        expected_final = (
            jnp.where(masked_weights[:, :, None] != 0, out.astype(jnp.float32), 0)
            * original_weights
        ).sum(axis=1)
        actual_final = (out.astype(jnp.float32) * original_weights).sum(axis=1)
        contaminated_final = (placeholder_out * original_weights).sum(axis=1)
        self.assertEqual(float(jnp.max(jnp.abs(actual_final - expected_final))), 0.0)
        self.assertGreater(
            float(jnp.max(jnp.abs(contaminated_final - expected_final))), 1e-6
        )


if __name__ == "__main__":
    unittest.main()
