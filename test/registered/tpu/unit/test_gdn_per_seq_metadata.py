"""The GDN/KDA per-sequence (prefill) tile metadata is sized by tiles.

The kernel reads one metadata entry per tile, so the arrays hold one entry per
tile the batch can hold (full tiles plus one partial tile per sequence), not one
per token. Token-sized arrays cost three token-wide gathers per call on every
linear-attention layer of every prefill chunk.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

import unittest

import jax.numpy as jnp
import numpy as np

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.pallas.gdn import config, metadata


def _cfg(batch_size: int, tile_size: int) -> config.GDNConfig:
    return config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        dtypes=config.Dtypes(
            act_in=jnp.float32.dtype,
            act_out=jnp.bfloat16.dtype,
            compute=jnp.float32.dtype,
            recurrent_state=jnp.float32.dtype,
            conv_state=jnp.float32.dtype,
        ),
        batch_size=batch_size,
        dim_size=384,
        kernel_size=4,
        tile_size=tile_size,
        num_kq_heads=1,
        num_v_heads=1,
        kq_head_dim=128,
        v_head_dim=128,
    )


class TestPerSeqMetadata(CustomTestCase):
    def test_sized_by_tiles_not_tokens(self):
        max_num_seqs, tile, batch_size = 8, 64, 8192
        lengths = np.array([100, 64, 4000, 1], np.int32)
        query_start = np.zeros(max_num_seqs + 1, np.int32)
        query_start[1 : lengths.size + 1] = np.cumsum(lengths)
        query_start[lengths.size + 1 :] = query_start[lengths.size]
        seq_lens = np.zeros(max_num_seqs, np.int32)
        seq_lens[: lengths.size] = lengths
        meta = metadata.compute_per_seq_metadata(
            cfg=_cfg(batch_size, tile),
            seq_lens=jnp.asarray(seq_lens),
            query_start_loc=jnp.asarray(query_start),
            state_indices=jnp.arange(max_num_seqs, dtype=jnp.int32),
            read_offsets=jnp.zeros(max_num_seqs, jnp.int32),
            start_seq=jnp.int32(0),
            end_seq=jnp.int32(lengths.size),
        )

        self.assertEqual(
            meta.p_id_to_s_idx.data.shape, (batch_size // tile + max_num_seqs,)
        )
        tiles = -(-lengths // tile)
        num_tiles = int(meta.num_tiles)
        self.assertEqual(num_tiles, tiles.sum())
        seq = np.repeat(np.arange(lengths.size), tiles)
        start = np.concatenate(([0], np.cumsum(tiles)[:-1]))
        tile_in_seq = np.arange(num_tiles) - start[seq]
        base = query_start[seq] + tile_in_seq * tile
        np.testing.assert_array_equal(meta.p_id_to_s_idx.data[:num_tiles], seq)
        np.testing.assert_array_equal(meta.p_id_to_r_base.data[:num_tiles], base)
        np.testing.assert_array_equal(
            meta.p_id_to_r_size.data[:num_tiles],
            np.minimum(query_start[seq + 1] - base, tile),
        )
        np.testing.assert_array_equal(
            meta.p_id_is_first_tile.data[:num_tiles], tile_in_seq == 0
        )
        np.testing.assert_array_equal(
            meta.p_id_is_last_tile.data[:num_tiles], tile_in_seq == tiles[seq] - 1
        )


if __name__ == "__main__":
    unittest.main()
