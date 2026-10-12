"""CPU unit test for the gpt-oss QKV sharding under decode context parallel.

Under DCP the ranks of a group replicate K/V and split only the queries, so the
KV projections shard across ``attn_tp // dcp`` groups instead of ``attn_tp``.
``GptOssAttention`` must build its ``QKVParallelLinear`` with that narrower
``kv_tp_size`` / ``kv_tp_rank`` pair, keep Q at full attention-TP sharding, and
end up with the KV head count the KV pool is sized for
(``ModelConfig.get_num_kv_heads(attn_tp, dcp_size)``).

Pure CPU construction: the parallel topology is published on the parallel bag
and every linear layer receives its rank/size explicitly, so no process group
is created.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt import runtime_context as rc
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=90, suite="stage-a-test-tpu-unit")

ATTN_TP_SIZE = 8
TOTAL_NUM_HEADS = 16
TOTAL_NUM_KV_HEADS = 8
HEAD_DIM = 8
HIDDEN_SIZE = TOTAL_NUM_HEADS * HEAD_DIM


def _model_config():
    """ModelConfig stand-in exposing only what get_num_kv_heads reads."""
    model_config = ModelConfig.__new__(ModelConfig)
    model_config.hf_config = SimpleNamespace(model_type="gpt_oss")
    model_config.hf_text_config = SimpleNamespace(
        num_key_value_heads=TOTAL_NUM_KV_HEADS
    )
    model_config.is_draft_model = False
    return model_config


def _build_attention(attn_tp_rank: int, dcp_size: int, layer_type="full_attention"):
    from sglang.srt.models.gpt_oss import GptOssAttention

    # The published config derives every width (attention, MoE, DCP) from
    # tp_size and dcp_size; only this process's ranks are stated by hand.
    with (
        get_context().override_server_args(
            attention_backend="pallas_tpu_rpa_batched",
            tp_size=ATTN_TP_SIZE,
            dcp_size=dcp_size,
        ),
        rc.get_parallel().override(
            tp_rank=attn_tp_rank,
            attn_tp_rank=attn_tp_rank,
            dcp_rank=attn_tp_rank % dcp_size,
            attn_dcp_rank=attn_tp_rank % dcp_size,
        ),
    ):
        return GptOssAttention(
            hidden_size=HIDDEN_SIZE,
            num_heads=TOTAL_NUM_HEADS,
            num_kv_heads=TOTAL_NUM_KV_HEADS,
            layer_id=0,
            max_position_embeddings=128,
            head_dim=HEAD_DIM,
            attention_bias=True,
            layer_type=layer_type,
            params_dtype=torch.float32,
        )


class TestGptOssDcpShard(CustomTestCase):
    """dcp=2 halves the KV-projection sharding width and doubles local KV heads."""

    DCP = 2

    def test_qkv_uses_the_narrowed_kv_tp_pair(self):
        for attn_tp_rank in range(ATTN_TP_SIZE):
            with self.subTest(attn_tp_rank=attn_tp_rank):
                attn = _build_attention(attn_tp_rank, self.DCP)
                self.assertEqual(attn.qkv_proj.kv_tp_size, ATTN_TP_SIZE // self.DCP)
                self.assertEqual(attn.qkv_proj.kv_tp_rank, attn_tp_rank // self.DCP)
                # Q keeps the full attention-TP split.
                self.assertEqual(attn.qkv_proj.tp_size, ATTN_TP_SIZE)
                self.assertEqual(attn.qkv_proj.tp_rank, attn_tp_rank)

    def test_ranks_of_one_dcp_group_hold_the_same_kv_shard(self):
        for group_start in range(0, ATTN_TP_SIZE, self.DCP):
            ranks = range(group_start, group_start + self.DCP)
            kv_ranks = {
                _build_attention(r, self.DCP).qkv_proj.kv_tp_rank for r in ranks
            }
            self.assertEqual(len(kv_ranks), 1, f"group at {group_start}: {kv_ranks}")

    def test_local_head_counts(self):
        attn = _build_attention(3, self.DCP)
        # Queries stay sharded by attn_tp; KV heads double relative to TP-only.
        self.assertEqual(attn.num_heads, TOTAL_NUM_HEADS // ATTN_TP_SIZE)
        self.assertEqual(
            attn.num_kv_heads, TOTAL_NUM_KV_HEADS // (ATTN_TP_SIZE // self.DCP)
        )
        self.assertEqual(attn.attn.tp_q_head_num, attn.num_heads)
        self.assertEqual(attn.attn.tp_k_head_num, attn.num_kv_heads)
        self.assertEqual(attn.attn.tp_v_head_num, attn.num_kv_heads)

    def test_head_count_agrees_with_the_kv_pool_sizing(self):
        model_config = _model_config()
        for dcp_size in (1, 2, 4):
            with self.subTest(dcp_size=dcp_size):
                attn = _build_attention(0, dcp_size)
                self.assertEqual(
                    attn.num_kv_heads,
                    model_config.get_num_kv_heads(ATTN_TP_SIZE, dcp_size=dcp_size),
                )

    def test_qkv_output_partition_matches_the_local_heads(self):
        attn = _build_attention(1, self.DCP)
        expected_rows = (attn.num_heads + 2 * attn.num_kv_heads) * HEAD_DIM
        self.assertEqual(attn.qkv_proj.weight.shape[0], expected_rows)
        # gpt-oss carries attention bias; it shards with the same split.
        self.assertEqual(attn.qkv_proj.bias.shape[0], expected_rows)
        self.assertEqual(attn.q_size, attn.num_heads * HEAD_DIM)
        self.assertEqual(attn.kv_size, attn.num_kv_heads * HEAD_DIM)

    def test_sinks_stay_per_query_head(self):
        attn = _build_attention(5, self.DCP)
        self.assertEqual(tuple(attn.sinks.shape), (attn.num_heads,))

    def test_dcp_one_keeps_todays_tp_sharding(self):
        for attn_tp_rank in (0, 7):
            with self.subTest(attn_tp_rank=attn_tp_rank):
                attn = _build_attention(attn_tp_rank, 1)
                self.assertEqual(attn.qkv_proj.kv_tp_size, ATTN_TP_SIZE)
                self.assertEqual(attn.qkv_proj.kv_tp_rank, attn_tp_rank)
                self.assertEqual(attn.num_kv_heads, TOTAL_NUM_KV_HEADS // ATTN_TP_SIZE)

    def test_sliding_window_layers_shard_the_same_way(self):
        attn = _build_attention(2, self.DCP, layer_type="sliding_attention")
        self.assertEqual(attn.qkv_proj.kv_tp_size, ATTN_TP_SIZE // self.DCP)
        self.assertEqual(attn.qkv_proj.kv_tp_rank, 1)


if __name__ == "__main__":
    unittest.main()
