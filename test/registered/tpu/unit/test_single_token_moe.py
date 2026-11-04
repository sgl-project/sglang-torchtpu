"""Single-token packed-MXFP4 MoE kernel vs the gmm_v2 path of _fused_moe_jax."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

import unittest
from unittest import mock

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

# Scaled-down Kimi-K3 shape: EP4, 16 local of 64 experts, top-8, SiTU. Not a
# validated MoE, so these cases force validated() to exercise the kernel math.
H, N, LOCAL, GLOBAL, K = 512, 256, 16, 64, 8
ACT = "situ:4.0:25.0"
# Kimi-K3's validated MoE: latent 3584, intermediate 3072 / 4 per rank (TP32),
# top-16 of 896. 16 local experts instead of EP8's 112 keep the test small; the
# gate and the expert offsets do not depend on the local count.
K3_H, K3_N, K3_LOCAL, K3_GLOBAL, K3_K = 3584, 768, 16, 896, 16


def _weights(h, n, local, seed=0):
    """Random packed-MXFP4 w13 [E, h/2, 2n], w2 [E, n/2, h] and their scales."""
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    u8 = lambda key, shape: jax.random.randint(key, shape, 0, 256, jnp.int32).astype(
        jnp.uint8
    )
    e8 = lambda key, shape: (
        2.0 ** jax.random.randint(key, shape, -9, -5).astype(jnp.float32)
    )
    return dict(
        w1=u8(keys[0], (local, h // 2, 2 * n)),
        w2=u8(keys[1], (local, n // 2, h)),
        w1_scale=e8(keys[2], (local, h // 32, 1, 2 * n)),
        w2_scale=e8(keys[3], (local, n // 32, 1, h)),
    )


def _x(h, scale=0.5, seed=1):
    return jnp.asarray(
        np.random.default_rng(seed).standard_normal((1, h)) * scale, jnp.bfloat16
    )


@unittest.skipUnless(_HAS_TPU, "requires a TPU backend")
class TestSingleTokenMoe(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights = _weights(H, N, LOCAL)
        cls.x = _x(H)

    def _run(
        self,
        ids,
        offset,
        kernel_on,
        weights=None,
        x=None,
        num_global=GLOBAL,
        threshold=512,
        force_validated=True,
    ):
        """_fused_moe_jax output, and whether the fused kernel ran."""
        from sglang_tpu.envs import envs as tpu_envs
        from sglang_tpu.layers.moe.pallas import single_token_moe
        from sglang_tpu.layers.moe.pallas.fused_moe_op import _fused_moe_jax

        weights = self.weights if weights is None else weights
        x = self.x if x is None else x
        k = len(ids)
        w = np.random.default_rng(k).random(k).astype(np.float32)
        w /= w.sum()
        with (
            tpu_envs.SGLANG_TPU_ENABLE_SINGLE_TOKEN_MOE_KERNEL.override(kernel_on),
            mock.patch.object(
                single_token_moe,
                "single_token_moe",
                wraps=single_token_moe.single_token_moe,
            ) as spy,
            mock.patch.object(
                single_token_moe,
                "validated",
                (lambda *a: True) if force_validated else single_token_moe.validated,
            ),
        ):
            out = jax.jit(
                lambda: _fused_moe_jax(
                    hidden_states=x,
                    **weights,
                    topk_weights=jnp.asarray(w[None], jnp.bfloat16),
                    topk_ids=jnp.asarray(np.asarray(ids)[None], jnp.int32),
                    w1_bias=None,
                    w2_bias=None,
                    topk=k,
                    activation_name=ACT,
                    global_num_experts=num_global,
                    expert_group_offset=offset,
                    permute_dense_threshold=threshold,
                )
            )()
        return np.asarray(out.astype(jnp.float32)), spy.called

    def _check(self, ids, offset, msg, **kw):
        ref, ref_ran = self._run(ids, offset, kernel_on=False, **kw)
        got, ran = self._run(ids, offset, kernel_on=True, **kw)
        self.assertFalse(ref_ran, msg)
        self.assertTrue(ran, f"kernel did not run: {msg}")
        self.assertEqual(got.shape, ref.shape)
        rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        self.assertLess(float(rel), 1e-2, msg)

    def test_matches_gmm_path(self):
        rng = np.random.default_rng(0)
        cases = [
            (rng.choice(GLOBAL, K, replace=False), 16),  # some local hits
            (np.arange(K) + 16, 16),  # every slot local
            (np.r_[np.arange(K - 1) + 40, -1], 32),  # a padded (-1) slot
        ]
        for ids, offset in cases:
            self._check(ids, offset, f"{ids=} {offset=}")

    def test_no_local_hit_is_zero(self):
        got, ran = self._run(np.arange(K) + 40, 0, kernel_on=True)
        self.assertTrue(ran)
        np.testing.assert_array_equal(got, np.zeros((1, H), np.float32))

    def test_independent_of_permute_dense_threshold(self):
        """The kernel reads topk_ids directly, so it also runs with the default
        threshold of 0 (argsort permute on the general path)."""
        self._check(np.arange(K) + 16, 16, "threshold=0", threshold=0)

    def test_unvalidated_moe_keeps_gmm_path(self):
        """Only VALIDATED_MOE takes the kernel; this scaled-down MoE does not."""
        _, ran = self._run(np.arange(K) + 16, 16, kernel_on=True, force_validated=False)
        self.assertFalse(ran)

    def test_kimi_k3_shape(self):
        """The validated Kimi-K3 MoE takes the kernel and matches gmm_v2."""
        offset = K3_LOCAL * 2  # this rank owns experts 32..47
        rng = np.random.default_rng(3)
        remote = np.setdiff1d(
            np.arange(K3_GLOBAL), np.arange(offset, offset + K3_LOCAL)
        )
        # 4 local hits among the 16, the rest on other ranks.
        ids = np.r_[
            offset + rng.choice(K3_LOCAL, 4, replace=False),
            rng.choice(remote, K3_K - 4, replace=False),
        ]
        self._check(
            rng.permutation(ids),
            offset,
            "Kimi-K3 TP32/EP8",
            weights=_weights(K3_H, K3_N, K3_LOCAL, seed=3),
            x=_x(K3_H),
            num_global=K3_GLOBAL,
            force_validated=False,
        )

    def test_validated_gate(self):
        from sglang_tpu.layers.moe.pallas.single_token_moe import validated

        for n in (3072, 1536, 768, 384):  # any TP split of the intermediate
            self.assertTrue(validated(3584, n, 896, 16, "situ:4.0:25.0"))
        self.assertFalse(validated(3584, 768, 384, 16, "situ:4.0:25.0"))  # experts
        self.assertFalse(validated(3584, 768, 896, 8, "situ:4.0:25.0"))  # top-k
        self.assertFalse(validated(3584, 768, 896, 16, "silu"))  # activation
        self.assertFalse(validated(7168, 768, 896, 16, "situ:4.0:25.0"))  # width
        self.assertFalse(validated(3584, 1000, 896, 16, "situ:4.0:25.0"))  # not a split

    def test_multiple_intermediate_tiles(self):
        # N=1536 -> two 768-wide tiles; N=640 -> five 128-wide tiles.
        for n in (1536, 640):
            with self.subTest(n=n):
                self._check(
                    np.arange(4),
                    0,
                    f"{n=}",
                    weights=_weights(H, n, 4, seed=n),
                    num_global=4,
                )

    def test_kimi_k3_shard_shape(self):
        """K=3584 latent, N=768 shard (TP32 / EP8), as Kimi-K3 runs it, with a
        larger activation so SiTU saturates."""
        self._check(
            np.arange(4),
            0,
            "K3 shard",
            weights=_weights(3584, 768, 4, seed=3),
            x=_x(3584, scale=4.0),
            num_global=4,
        )


if __name__ == "__main__":
    unittest.main()
