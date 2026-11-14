"""Fetch-plan tests for EP-sharded Run:AI weight streaming."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import ep_weight_shard as shard

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_tpu_ci(est_time=2, suite="stage-a-test-tpu-unit", chips=1)


def _weight_map(*names):
    return {name: f"shard-{index}.safetensors" for index, name in enumerate(names)}


class TestPlanExpertFetch(CustomTestCase):
    def test_keeps_dense_and_local_experts_and_skips_the_rest(self):
        layer = "model.layers.0.mlp"
        names = (
            "model.embed_tokens.weight",
            *(f"{layer}.experts.{expert}.w1.weight" for expert in range(4)),
            *(f"{layer}.experts.{expert}.w1.scale" for expert in range(4)),
            "model.layers.0.mlp.shared_experts.gate_proj.weight",
        )
        plan = shard.plan_expert_fetch(
            _weight_map(*names), num_routed_experts=4, ep_size=2, ep_rank=1
        )
        self.assertEqual(
            set(plan.kept),
            {
                "model.embed_tokens.weight",
                "model.layers.0.mlp.shared_experts.gate_proj.weight",
                f"{layer}.experts.2.w1.weight",
                f"{layer}.experts.2.w1.scale",
                f"{layer}.experts.3.w1.weight",
                f"{layer}.experts.3.w1.scale",
            },
        )
        self.assertEqual(
            set(plan.skipped),
            {
                f"{layer}.experts.0.w1.weight",
                f"{layer}.experts.0.w1.scale",
                f"{layer}.experts.1.w1.weight",
                f"{layer}.experts.1.w1.scale",
            },
        )
        self.assertEqual(plan.local_experts, (2, 4))

    def test_mtp_experts_are_never_skipped(self):
        """The draft head's expert ids need not follow the target's EP slots."""
        layer = "model.layers.0.mlp"
        names = (
            *(f"{layer}.experts.{expert}.w1.weight" for expert in range(4)),
            "model.mtp.layers.0.mlp.experts.7.w1.weight",
            "mtp.layers.0.block_sparse_moe.experts.7.w3.weight",
        )
        plan = shard.plan_expert_fetch(
            _weight_map(*names), num_routed_experts=4, ep_size=2, ep_rank=0
        )
        self.assertEqual(
            set(plan.kept),
            {
                f"{layer}.experts.0.w1.weight",
                f"{layer}.experts.1.w1.weight",
                "model.mtp.layers.0.mlp.experts.7.w1.weight",
                "mtp.layers.0.block_sparse_moe.experts.7.w3.weight",
            },
        )
        self.assertEqual(
            set(plan.skipped),
            {
                f"{layer}.experts.2.w1.weight",
                f"{layer}.experts.3.w1.weight",
            },
        )

    def test_fused_expert_tensor_without_an_id_is_kept(self):
        names = ("model.layers.0.mlp.experts.gate_up_proj",)
        plan = shard.plan_expert_fetch(
            _weight_map(*names), num_routed_experts=8, ep_size=2, ep_rank=0
        )
        self.assertEqual(plan.kept, names)
        self.assertEqual(plan.skipped, ())

    def test_non_weight_expert_tensors_are_kept(self):
        """Biases and input scales are tiny and may be needed by every rank."""
        layer = "model.layers.0.mlp"
        names = (
            *(f"{layer}.experts.{expert}.w1.weight" for expert in range(4)),
            f"{layer}.experts.0.bias",
            f"{layer}.experts.3.input_scale",
        )
        plan = shard.plan_expert_fetch(
            _weight_map(*names), num_routed_experts=4, ep_size=2, ep_rank=1
        )
        self.assertIn(f"{layer}.experts.0.bias", plan.kept)
        self.assertIn(f"{layer}.experts.3.input_scale", plan.kept)

    def test_expert_id_outside_the_config_is_rejected(self):
        """A mismatch must fail to the stock loader, never drop a tensor."""
        with self.assertRaisesRegex(ValueError, "outside 8 routed experts"):
            shard.plan_expert_fetch(
                _weight_map("model.layers.0.mlp.experts.8.w1.weight"),
                num_routed_experts=8,
                ep_size=2,
                ep_rank=0,
            )

    def test_layer_missing_an_expert_id_is_rejected(self):
        """A different expert id space must not be sharded from the index."""
        names = tuple(
            f"model.layers.0.mlp.experts.{expert}.w1.weight" for expert in range(3)
        )
        with self.assertRaisesRegex(ValueError, "names 3 of 4 expert ids"):
            shard.plan_expert_fetch(
                _weight_map(*names), num_routed_experts=4, ep_size=2, ep_rank=0
            )

    def test_experts_with_different_tensor_names_are_rejected(self):
        """Per-expert tensor sets must be uniform or the plan is unsafe."""
        names = (
            "model.layers.0.mlp.experts.0.w1.weight",
            "model.layers.0.mlp.experts.1.w2.weight",
            "model.layers.0.mlp.experts.2.w2.weight",
            "model.layers.0.mlp.experts.3.w2.weight",
        )
        with self.assertRaisesRegex(ValueError, "different tensors"):
            shard.plan_expert_fetch(
                _weight_map(*names), num_routed_experts=4, ep_size=2, ep_rank=0
            )


class TestShardable(CustomTestCase):
    def test_every_unsafe_layout_keeps_the_stock_loader(self):
        supported = dict(
            env_enabled=True,
            model_type="qwen3_moe",
            is_draft_model=False,
            draft_model_idx=None,
            ep_size=2,
            num_routed_experts=8,
            enable_eplb=False,
            redundant_experts=0,
            init_expert_location="trivial",
            ep_join_mode=None,
        )
        self.assertTrue(shard.shardable(**supported))
        for name, override in {
            "env off": {"env_enabled": False},
            "draft model": {"is_draft_model": True},
            "draft index": {"draft_model_idx": 0},
            "deepseek_v41": {"model_type": "deepseek_v41"},
            "no EP": {"ep_size": 1},
            "unknown expert count": {"num_routed_experts": None},
            "not divisible": {"num_routed_experts": 7},
            "elastic EP": {"ep_join_mode": "scale"},
            "elastic EP recover": {"ep_join_mode": "recover"},
            "EPLB": {"enable_eplb": True},
            "redundant experts": {"redundant_experts": 2},
            "static placement": {"init_expert_location": "file"},
        }.items():
            with self.subTest(name):
                self.assertFalse(shard.shardable(**{**supported, **override}))


class TestLoaderFetchPlan(CustomTestCase):
    def _loader(self):
        load_config = SimpleNamespace(
            model_loader_extra_config={}, draft_model_idx=None, load_format=None
        )
        return shard.EPShardedRunaiModelStreamerLoader(
            load_config,
            model_config=None,
            num_routed_experts=2,
            ep_size=2,
            ep_rank=1,
        )

    def test_streams_only_the_shards_of_kept_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, shard.INDEX_NAME).write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.embed_tokens.weight": "b.safetensors",
                            "model.layers.0.mlp.experts.0.w1.weight": "a.safetensors",
                            "model.layers.0.mlp.experts.1.w1.weight": "b.safetensors",
                        }
                    }
                )
            )
            files = [str(Path(tmp, "a.safetensors")), str(Path(tmp, "b.safetensors"))]
            source = SimpleNamespace(
                model_or_path=tmp, revision=None, prefix="", model_config=None
            )
            loader = self._loader()
            with patch.object(loader, "_prepare_weights", return_value=(tmp, files)):
                prepared = loader._prepare_fetch(source)
        self.assertIsNotNone(prepared)
        stream_files, plan = prepared
        self.assertEqual(stream_files, [files[1]])
        self.assertEqual(
            plan.kept,
            ("model.embed_tokens.weight", "model.layers.0.mlp.experts.1.w1.weight"),
        )
        self.assertEqual(plan.skipped, ("model.layers.0.mlp.experts.0.w1.weight",))

    def test_extra_mtp_shard_outside_the_index_keeps_the_stock_stream(self):
        """Dropping an indexed-only plan would lose the shard's tensors."""
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, shard.INDEX_NAME).write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.layers.0.mlp.experts.0.w1.weight": "a.safetensors",
                            "model.layers.0.mlp.experts.1.w1.weight": "b.safetensors",
                        }
                    }
                )
            )
            files = [
                str(Path(tmp, "a.safetensors")),
                str(Path(tmp, "b.safetensors")),
                str(Path(tmp, "mtp.safetensors")),
            ]
            source = SimpleNamespace(
                model_or_path=tmp, revision=None, prefix="", model_config=None
            )
            loader = self._loader()
            with patch.object(loader, "_prepare_weights", return_value=(tmp, files)):
                prepared = loader._prepare_fetch(source)
        self.assertIsNone(prepared)


if __name__ == "__main__":
    unittest.main()
