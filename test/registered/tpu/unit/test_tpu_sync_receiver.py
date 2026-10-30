"""The tpu-sync receiver must slice and place every trainer weight where sglang's TP ranks hold it."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)

import unittest

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.weight_sync.tpu_sync_receiver import install_plan, tp_slice

# Qwen3-4B: hidden 2560, 32 query and 8 KV heads of 128, intermediate 9728.
_LAYER = "model.layers.0"


def _slice_rows(
    full: torch.Tensor, name: str, *, tp_size: int, tp_rank: int, num_kv_heads: int
) -> torch.Tensor:
    """What the TP rank receives: block ``tp_rank`` (or its KV head) of the mesh tpu-sync is given."""
    s = tp_slice(
        name,
        list(full.shape),
        tp_size=tp_size,
        tp_rank=tp_rank,
        num_kv_heads=num_kv_heads,
    )
    if s.sharding_spec == ["", "tp"]:
        return full.chunk(tp_size, dim=1)[tp_rank]
    if s.sharding_spec == ["tp", ""]:
        index = s.global_shard_indices[0] if s.global_shard_indices else tp_rank
        return full.chunk(s.mesh_shape[0], dim=0)[index]
    return full


class TestTpSlice(CustomTestCase):
    def test_slices_match_sglang_tp_layout(self):
        for name, shape, tp, expected in [
            (f"{_LAYER}.self_attn.q_proj.weight", [4096, 2560], 4, [1024, 2560]),
            (f"{_LAYER}.self_attn.k_proj.weight", [1024, 2560], 4, [256, 2560]),
            (f"{_LAYER}.self_attn.o_proj.weight", [2560, 4096], 4, [2560, 1024]),
            (f"{_LAYER}.mlp.down_proj.weight", [2560, 9728], 2, [2560, 4864]),
            (f"{_LAYER}.input_layernorm.weight", [2560], 4, [2560]),
            ("model.embed_tokens.weight", [151936, 2560], 4, [37984, 2560]),
            (f"{_LAYER}.self_attn.q_proj.weight", [4096, 2560], 1, [4096, 2560]),
        ]:
            with self.subTest(name=name, tp=tp):
                s = tp_slice(name, shape, tp_size=tp, tp_rank=0, num_kv_heads=8)
                self.assertEqual(s.local_shape, expected)

    def test_kv_heads_replicated_when_tp_exceeds_them(self):
        # 2 KV heads at TP 8: ranks 0-3 share head 0, ranks 4-7 head 1
        heads = [
            tp_slice(
                f"{_LAYER}.self_attn.v_proj.weight",
                [256, 2560],
                tp_size=8,
                tp_rank=r,
                num_kv_heads=2,
            )
            for r in range(8)
        ]
        self.assertEqual(
            [h.global_shard_indices[0] for h in heads], [0, 0, 0, 0, 1, 1, 1, 1]
        )
        self.assertEqual(heads[0].local_shape, [128, 2560])


class TestInstallPlan(CustomTestCase):
    def test_fuses_qkv_and_gate_up_and_ties_lm_head(self):
        hf = (
            [f"{_LAYER}.self_attn.{p}_proj.weight" for p in ("q", "k", "v", "o")]
            + [f"{_LAYER}.mlp.{p}_proj.weight" for p in ("gate", "up", "down")]
            + ["model.embed_tokens.weight"]
        )
        params = [
            f"{_LAYER}.self_attn.qkv_proj.weight",
            f"{_LAYER}.self_attn.o_proj.weight",
            f"{_LAYER}.mlp.gate_up_proj.weight",
            f"{_LAYER}.mlp.down_proj.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ]
        plan = install_plan(params, hf)
        self.assertEqual(
            plan[f"{_LAYER}.self_attn.qkv_proj.weight"],
            tuple(f"{_LAYER}.self_attn.{p}_proj.weight" for p in ("q", "k", "v")),
        )
        self.assertEqual(plan["lm_head.weight"], ("model.embed_tokens.weight",))

    def test_rejects_unmatched_names(self):
        with self.assertRaisesRegex(ValueError, "lack HF tensors"):
            install_plan(
                [f"{_LAYER}.mlp.gate_up_proj.weight"],
                [f"{_LAYER}.mlp.gate_proj.weight"],
            )
        with self.assertRaisesRegex(ValueError, "fill no parameter"):
            install_plan(
                ["model.norm.weight"], ["model.norm.weight", "model.extra.weight"]
            )

    def test_fused_slices_equal_sglang_shard(self):
        # sglang's TP-rank qkv shard is cat(q_r, k_r, v_r); the staged slices must rebuild exactly that
        torch.manual_seed(0)
        tp, kv_heads = 4, 8
        full = {
            p: torch.randn(rows, 16)
            for p, rows in (("q", 32 * 4), ("k", 8 * 4), ("v", 8 * 4))
        }
        for rank in range(tp):
            staged = [
                _slice_rows(
                    full[p],
                    f"{_LAYER}.self_attn.{p}_proj.weight",
                    tp_size=tp,
                    tp_rank=rank,
                    num_kv_heads=kv_heads,
                )
                for p in ("q", "k", "v")
            ]
            expected = torch.cat(
                [full[p].chunk(tp, dim=0)[rank] for p in ("q", "k", "v")]
            )
            self.assertTrue(torch.equal(torch.cat(staged), expected))


if __name__ == "__main__":
    unittest.main()
