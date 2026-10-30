import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.vaes import base as vae_base
from sglang.multimodal_gen.runtime.pipelines_core.stages import decoding
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    mova,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising_av import (
    LTX2RefinementStage,
)


class FakeVAE(nn.Module):
    def decode(self, latents):
        return latents


class TestDecodingStageParallelism(unittest.TestCase):
    def test_component_use_honors_decode_precision_override(self):
        stage = DecodingStage(FakeVAE())
        server_args = SimpleNamespace(
            component_precisions={},
            pipeline_config=SimpleNamespace(
                vae_precision="fp32",
                vae_decode_precision="bf16",
            ),
        )

        [component_use] = stage.component_uses(server_args)

        self.assertEqual(component_use.target_dtype, torch.bfloat16)

    def test_component_use_honors_exact_component_precision(self):
        stage = DecodingStage(FakeVAE())
        server_args = SimpleNamespace(
            component_precisions={"vae": "fp16"},
            pipeline_config=SimpleNamespace(
                vae_precision="fp32",
                vae_decode_precision="bf16",
            ),
        )

        [component_use] = stage.component_uses(server_args)

        self.assertEqual(component_use.target_dtype, torch.float16)

    def test_cfg_parallel_uses_replicated_decode_when_decode_group_has_multiple_ranks(
        self,
    ):
        stage = object.__new__(DecodingStage)
        stage.vae = SimpleNamespace(use_parallel_decode=True)

        with (
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding.get_global_server_args",
                return_value=SimpleNamespace(enable_cfg_parallel=True),
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding.model_parallel_is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding.get_decode_parallel_world_size",
                return_value=2,
            ),
        ):
            self.assertEqual(
                stage.parallelism_type,
                StageParallelismType.REPLICATED,
            )

    def test_torch_compile_decode_cache_is_replaced_for_new_vae_instance(self):
        vae = FakeVAE()
        stage = DecodingStage(vae)
        server_args = SimpleNamespace(enable_torch_compile=True)

        with patch(
            "torch.compile",
            side_effect=lambda fn, **_: fn,
        ) as compile_fn:
            compiled_vae_decode = stage._get_vae_decode_fn(vae, server_args)
            self.assertIs(
                stage._get_vae_decode_fn(vae, server_args), compiled_vae_decode
            )
            self.assertEqual(compile_fn.call_count, 1)

            new_vae = FakeVAE()
            new_compiled_vae_decode = stage._get_vae_decode_fn(new_vae, server_args)
            self.assertIsNot(new_compiled_vae_decode, compiled_vae_decode)
            self.assertEqual(compile_fn.call_count, 2)
            self.assertEqual(stage._compiled_vae_decode.target_id, id(new_vae))
            self.assertIs(
                stage._compiled_vae_decode.compiled_module, new_compiled_vae_decode
            )

    def test_cfg_parallel_keeps_main_rank_decode_without_parallel_decode(self):
        stage = object.__new__(DecodingStage)
        stage.vae = SimpleNamespace(use_parallel_decode=False)

        with (
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding.get_global_server_args",
                return_value=SimpleNamespace(enable_cfg_parallel=True),
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding.model_parallel_is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding.get_decode_parallel_world_size",
                return_value=2,
            ),
        ):
            self.assertEqual(
                stage.parallelism_type,
                StageParallelismType.MAIN_RANK_ONLY,
            )

    def test_cfg_parallel_keeps_main_rank_decode_when_decode_group_is_single_rank(
        self,
    ):
        stage = object.__new__(DecodingStage)
        stage.vae = SimpleNamespace(use_parallel_decode=True)

        with (
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding.get_global_server_args",
                return_value=SimpleNamespace(enable_cfg_parallel=True),
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding.model_parallel_is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding.get_decode_parallel_world_size",
                return_value=1,
            ),
        ):
            self.assertEqual(
                stage.parallelism_type,
                StageParallelismType.MAIN_RANK_ONLY,
            )

    def test_non_cfg_parallel_keeps_replicated_decode(self):
        stage = object.__new__(DecodingStage)
        stage.vae = SimpleNamespace(use_parallel_decode=True)

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding.get_global_server_args",
            return_value=SimpleNamespace(enable_cfg_parallel=False),
        ):
            self.assertEqual(
                stage.parallelism_type,
                StageParallelismType.REPLICATED,
            )

    def test_ltx2_refinement_broadcasts_when_decode_uses_all_ranks(self):
        stage = object.__new__(LTX2RefinementStage)
        stage.server_args = SimpleNamespace(enable_cfg_parallel=True)
        stage.vae = SimpleNamespace(use_parallel_decode=True)

        with (
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising_av.model_parallel_is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising_av.get_decode_parallel_world_size",
                return_value=2,
            ),
        ):
            self.assertEqual(
                stage.parallelism_type,
                StageParallelismType.MAIN_RANK_ONLY_AND_SEND_TO_OTHERS,
            )

    def test_ltx2_refinement_stays_main_rank_only_without_parallel_decode(self):
        stage = object.__new__(LTX2RefinementStage)
        stage.server_args = SimpleNamespace(enable_cfg_parallel=True)
        stage.vae = SimpleNamespace(use_parallel_decode=False)

        with (
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising_av.model_parallel_is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising_av.get_decode_parallel_world_size",
                return_value=2,
            ),
        ):
            self.assertEqual(
                stage.parallelism_type,
                StageParallelismType.MAIN_RANK_ONLY,
            )


class TestMovaDecodingStageParallelism(unittest.TestCase):
    def test_cfg_parallel_runs_a_parallel_decode_on_every_rank(self):
        # A spatial or tiled decode issues collectives over the whole decode
        # group, so a main-rank-only stage would hang the other CFG ranks.
        cases = (
            (True, True, 2, StageParallelismType.REPLICATED),
            (True, False, 2, StageParallelismType.MAIN_RANK_ONLY),
            (True, True, 1, StageParallelismType.MAIN_RANK_ONLY),
            (False, True, 2, StageParallelismType.REPLICATED),
            (False, False, 1, StageParallelismType.REPLICATED),
        )
        stage = object.__new__(mova.MOVADecodingStage)
        for cfg_parallel, parallel_decode, decode_world_size, expected in cases:
            stage.video_vae = SimpleNamespace(
                use_parallel_decode=parallel_decode,
                parallel_decode_mode="spatial_shard",
            )
            with (
                self.subTest(
                    cfg_parallel=cfg_parallel,
                    parallel_decode=parallel_decode,
                    decode_world_size=decode_world_size,
                ),
                patch.object(
                    mova,
                    "get_global_server_args",
                    return_value=SimpleNamespace(enable_cfg_parallel=cfg_parallel),
                ),
                patch.object(
                    decoding, "model_parallel_is_initialized", return_value=True
                ),
                patch.object(
                    decoding,
                    "get_decode_parallel_world_size",
                    return_value=decode_world_size,
                ),
            ):
                self.assertEqual(stage.parallelism_type, expected)


class TestDecodeUsesEveryRank(unittest.TestCase):
    def test_auto_decode_without_p2p_uses_every_rank_only_for_opted_in_vaes(self):
        def vae(mode, opted_in):
            return SimpleNamespace(
                use_parallel_decode=True,
                parallel_decode_mode=mode,
                auto_shards_without_p2p=opted_in,
            )

        cases = (
            (False, vae("auto", True), 2, True),
            (False, vae("auto", True), 32, False),
            (False, vae("auto", False), 2, False),
            (False, vae("spatial_shard", False), 2, True),
            (False, vae("tiled", False), 2, True),
            (True, vae("auto", False), 32, True),
        )
        for p2p, stub, world_size, expected in cases:
            platform = SimpleNamespace(supports_distributed_p2p=lambda p2p=p2p: p2p)
            with (
                self.subTest(
                    p2p=p2p, mode=stub.parallel_decode_mode, world_size=world_size
                ),
                # The policy helper lives with the VAE config policy.
                patch.object(vae_base, "current_platform", platform),
                patch.object(
                    decoding, "model_parallel_is_initialized", return_value=True
                ),
                patch.object(
                    decoding, "get_decode_parallel_world_size", return_value=world_size
                ),
            ):
                self.assertEqual(decoding.decode_uses_every_rank(stub), expected)

    def test_vae_policy_survives_an_arch_config_on_the_module(self):
        # AutoencoderKLQwenImage replaces .config with its arch config; the
        # policy comes from the VAEConfig given to ParallelTiledVAE.
        from sglang.multimodal_gen.configs.models.vaes.qwenimage import (
            QwenImageVAEConfig,
        )
        from sglang.multimodal_gen.configs.models.vaes.wanvae import WanVAEConfig
        from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE

        class _ArchConfigVAE(ParallelTiledVAE):
            def __init__(self, config):
                super().__init__(config)
                self.config = config.arch_config

            def _encode(self, x):
                return x

            def _decode(self, z):
                return z

        platform = SimpleNamespace(supports_distributed_p2p=lambda: False)
        with (
            patch.object(vae_base, "current_platform", platform),
            patch.object(decoding, "model_parallel_is_initialized", return_value=True),
            patch.object(decoding, "get_decode_parallel_world_size", return_value=2),
        ):
            for config, expected in (
                (QwenImageVAEConfig(), False),
                (WanVAEConfig(), True),
            ):
                with self.subTest(vae=type(config).__name__):
                    vae = _ArchConfigVAE(config)
                    self.assertEqual(decoding.decode_uses_every_rank(vae), expected)


if __name__ == "__main__":
    unittest.main()
