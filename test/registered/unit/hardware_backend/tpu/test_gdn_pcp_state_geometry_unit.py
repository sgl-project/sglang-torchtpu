# SPDX-License-Identifier: Apache-2.0

"""Geometry the context-parallel GDN state pool is built from.

Two independent pieces of arithmetic have to agree or the fused kernel reads
state from the wrong place: the allocator sizes each state index's window
(``derive_pool_block_tokens``) and the kernel's copy-plan places the conv and
recurrent regions inside that window (``derive_pooled_gdn_state_layout``). They
carry separate copies of the same conv-tile loop, so this pins them together.

The shard factor is here for the same reason: it multiplies into the state
shapes of every hybrid model on every platform, so it has to be exactly 1
whenever context parallelism is off.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

from sglang.srt.hardware_backend.tpu.mamba_pool_pooled import (
    POOL_KERNEL_BLOCK_TOKENS,
    POOL_TOKEN_BYTES,
    derive_pool_block_tokens,
)
from sglang.test.test_utils import CustomTestCase

# (n_v, conv_dim, d_k, d_v, kernel_size) per rank, for the head shards a
# Qwen3.5 model lands on at pcp 1/2/4/8. conv_dim is
# 2 * local_n_kq * d_k + local_n_v * d_v.
SHARDS = [
    (64, 12288, 128, 128, 4),  # 397B, no CP
    (32, 6144, 128, 128, 4),  # 397B at cp=2
    (16, 3072, 128, 128, 4),  # 397B at cp=4
    (8, 1536, 128, 128, 4),  # 397B at cp=8
    (32, 8192, 128, 128, 4),  # 35B, no CP
    (4, 1024, 128, 128, 4),  # 35B at cp=8
]


class TestPooledGDNStateGeometry(CustomTestCase):
    def test_window_is_whole_kernel_blocks(self):
        for n_v, conv_dim, d_k, d_v, k in SHARDS:
            with self.subTest(n_v=n_v, conv_dim=conv_dim):
                tokens = derive_pool_block_tokens(
                    n_v=n_v, conv_dim=conv_dim, d_k=d_k, d_v=d_v, kernel_size=k
                )
                self.assertEqual(tokens % POOL_KERNEL_BLOCK_TOKENS, 0)

    def test_window_holds_both_regions(self):
        """The window must fit the recurrent state and the conv taps.

        Undersizing does not fault: the conv region would start inside bytes
        the allocator never reserved, and the state would read back as another
        slot's.
        """
        for n_v, conv_dim, d_k, d_v, k in SHARDS:
            with self.subTest(n_v=n_v, conv_dim=conv_dim):
                tokens = derive_pool_block_tokens(
                    n_v=n_v, conv_dim=conv_dim, d_k=d_k, d_v=d_v, kernel_size=k
                )
                ssm_bytes = n_v * d_k * d_v * 4
                conv_bytes = (k - 1) * conv_dim * 2
                self.assertGreaterEqual(
                    tokens * POOL_TOKEN_BYTES, ssm_bytes + conv_bytes
                )

    def test_conv_region_starts_on_a_boundary_the_kernel_can_address(self):
        """The conv tile count must divide the recurrent token count.

        The kernel addresses the conv region with a single static offset past
        the recurrent blocks, which is only well-defined when the two divide.
        """
        for n_v, conv_dim, d_k, d_v, k in SHARDS:
            with self.subTest(n_v=n_v, conv_dim=conv_dim):
                ssm_tokens = (n_v * d_k * d_v * 4) // POOL_TOKEN_BYTES
                conv_tokens = 1
                while (
                    conv_tokens * POOL_TOKEN_BYTES < (k - 1) * conv_dim * 2
                    or ssm_tokens % conv_tokens
                ):
                    conv_tokens *= 2
                self.assertEqual(ssm_tokens % conv_tokens, 0)
                self.assertLessEqual(
                    ssm_tokens + conv_tokens,
                    derive_pool_block_tokens(
                        n_v=n_v, conv_dim=conv_dim, d_k=d_k, d_v=d_v, kernel_size=k
                    ),
                )

    def test_recurrent_state_must_fill_whole_pool_tokens(self):
        """A head shard whose state is not a pool-token multiple is rejected.

        Silently rounding would leave the tail of the recurrent region
        overlapping the conv rows.
        """
        with self.assertRaises(ValueError):
            derive_pool_block_tokens(n_v=1, conv_dim=1536, d_k=1, d_v=1, kernel_size=4)


class TestGdnStateCpShards(CustomTestCase):
    def test_no_extra_sharding_when_cp_is_off(self):
        """Must be exactly 1 off the PCP path.

        The factor multiplies into ``Mamba2StateShape.create``'s
        ``tp_world_size``, so anything else silently changes the recurrent and
        conv state shapes of every hybrid model, on every platform.
        """
        from sglang.srt.configs.qwen3_next import _gdn_state_cp_shards

        self.assertEqual(_gdn_state_cp_shards(), 1)


if __name__ == "__main__":
    unittest.main()
