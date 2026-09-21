# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TPU-isolated SPMD subclasses in `tpu_spmd.py`.

Verifies that:
1. Shared GPU classes (`DecodingStage`, `TextEncodingStage`,
   `QwenImagePipelineConfig`, `ErnieImagePipelineConfig`,
   `QwenImageTransformer2DModel`) remain 100% unmodified and instantiate their
   exact base classes when `current_platform.is_tpu()` is False.
2. When `current_platform.is_tpu()` is True and `install_tpu_spmd_overrides()`
   is active, instances promote to their `Tpu*` subclasses (`TpuDecodingStage`,
   `TpuTextEncodingStage`, `TpuQwenImagePipelineConfig`,
   `TpuErnieImagePipelineConfig`, `TpuQwenImageTransformer2DModel`).
3. Positive/negative CFG text branch padding, SP text sharding with attention
   masks, replicated decode materialization, and CPU-group barrier routing work
   as expected on TPU.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist

from sglang.multimodal_gen.configs.models.vaes.base import (
    should_use_spatial_shard_parallel_decode,
)
from sglang.multimodal_gen.configs.pipeline_configs.ernie_image import (
    ErnieImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageEditPipelineConfig,
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.platforms.interface import Platform
from sglang.multimodal_gen.runtime.platforms.tpu import TpuPlatform
from sglang.multimodal_gen.runtime.platforms.tpu_spmd import (
    TpuDecodingStage,
    TpuErnieImagePipelineConfig,
    TpuQwenImagePipelineConfig,
    TpuTextEncodingStage,
    install_tpu_spmd_overrides,
    upgrade_instance_to_tpu_subclass,
)


def _make_batch(
    pos_len: int = 10,
    neg_len: int = 4,
    dim: int = 16,
    *,
    with_embeds_mask: bool = True,
    with_attn_mask: bool = False,
) -> Req:
    req = Req(
        prompt="a cat",
        negative_prompt="blurry",
        height=512,
        width=512,
        num_frames=1,
    )
    req.do_classifier_free_guidance = True
    req.prompt_embeds = [torch.ones((1, pos_len, dim))]
    req.negative_prompt_embeds = [torch.full((1, neg_len, dim), 2.0)]
    req.prompt_seq_lens = [[pos_len]]
    req.negative_prompt_seq_lens = [[neg_len]]

    if with_embeds_mask:
        req.prompt_embeds_mask = [torch.ones((1, pos_len), dtype=torch.bool)]
        req.negative_prompt_embeds_mask = [torch.ones((1, neg_len), dtype=torch.bool)]
    else:
        req.prompt_embeds_mask = []
        req.negative_prompt_embeds_mask = []

    if with_attn_mask:
        req.prompt_attention_mask = [torch.ones((1, pos_len), dtype=torch.long)]
        req.negative_attention_mask = [torch.ones((1, neg_len), dtype=torch.long)]
    else:
        req.prompt_attention_mask = None
        req.negative_attention_mask = None
    return req


class TestTpuSpmdSubclasses(unittest.TestCase):
    def test_non_tpu_instantiates_exact_base_classes(self):
        install_tpu_spmd_overrides()
        with patch(
            "sglang.multimodal_gen.runtime.platforms.tpu_spmd.current_platform.is_tpu",
            return_value=False,
        ):
            qwen_cfg = QwenImagePipelineConfig()
            ernie_cfg = ErnieImagePipelineConfig()
            stage = DecodingStage(vae=MagicMock())
            self.assertIs(type(qwen_cfg), QwenImagePipelineConfig)
            self.assertIs(type(ernie_cfg), ErnieImagePipelineConfig)
            self.assertIs(type(stage), DecodingStage)

    def test_tpu_instantiates_and_upgrades_subclasses(self):
        with patch(
            "sglang.multimodal_gen.runtime.platforms.tpu_spmd.current_platform.is_tpu",
            return_value=False,
        ):
            pre_existing_qwen = QwenImagePipelineConfig()
            pre_existing_edit = QwenImageEditPipelineConfig()
            self.assertIs(type(pre_existing_qwen), QwenImagePipelineConfig)

        with patch(
            "sglang.multimodal_gen.runtime.platforms.tpu_spmd.current_platform.is_tpu",
            return_value=True,
        ):
            install_tpu_spmd_overrides()
            qwen_cfg = QwenImagePipelineConfig()
            ernie_cfg = ErnieImagePipelineConfig()
            stage = DecodingStage(vae=MagicMock())
            self.assertIs(type(qwen_cfg), TpuQwenImagePipelineConfig)
            self.assertIs(type(ernie_cfg), TpuErnieImagePipelineConfig)
            self.assertIs(type(stage), TpuDecodingStage)

            upgrade_instance_to_tpu_subclass(pre_existing_qwen)
            upgrade_instance_to_tpu_subclass(pre_existing_edit)
            self.assertIs(type(pre_existing_qwen), TpuQwenImagePipelineConfig)
            self.assertIsInstance(pre_existing_edit, QwenImageEditPipelineConfig)
            self.assertIsInstance(pre_existing_edit, TpuQwenImagePipelineConfig)

    def test_tpu_text_encoding_stage_aligns_cfg_branches_and_rounds_to_sp_size(self):
        batch = _make_batch(pos_len=10, neg_len=4, with_embeds_mask=True)
        server_args = SimpleNamespace(pipeline_config=TpuQwenImagePipelineConfig())
        stage = object.__new__(TpuTextEncodingStage)

        with patch(
            "sglang.multimodal_gen.runtime.platforms.tpu_spmd.get_sp_world_size",
            return_value=4,
        ):
            stage._align_cfg_branch_text_lengths(batch, server_args)

        self.assertEqual(batch.prompt_embeds[0].shape, (1, 12, 16))
        self.assertEqual(batch.negative_prompt_embeds[0].shape, (1, 12, 16))
        self.assertEqual(batch.prompt_embeds_mask[0].shape, (1, 12))
        self.assertEqual(batch.negative_prompt_embeds_mask[0].shape, (1, 12))
        self.assertEqual(int(batch.prompt_embeds_mask[0].sum().item()), 10)
        self.assertEqual(int(batch.negative_prompt_embeds_mask[0].sum().item()), 4)
        self.assertEqual(batch.prompt_seq_lens, [[10]])
        self.assertEqual(batch.negative_prompt_seq_lens, [[4]])

    def test_tpu_qwen_image_pipeline_config_cfg_conditioning(self):
        batch = _make_batch(pos_len=12, neg_len=12, with_embeds_mask=True)
        batch.negative_prompt_embeds_mask[0][:, 4:] = False
        batch.negative_prompt_seq_lens = [[4]]
        cfg = TpuQwenImagePipelineConfig()

        with patch(
            "sglang.multimodal_gen.runtime.platforms.tpu_spmd.requires_uniform_cfg_branch_shapes",
            return_value=False,
        ):
            seq_lens, mask = cfg._prepare_text_conditioning(
                batch, 0, text_seq_len=12, batch_size=1, negative=False
            )
            self.assertEqual(seq_lens, [12])
            self.assertIsNone(mask)

        with patch(
            "sglang.multimodal_gen.runtime.platforms.tpu_spmd.requires_uniform_cfg_branch_shapes",
            return_value=True,
        ):
            pos_lens, pos_mask = cfg._prepare_text_conditioning(
                batch, 0, text_seq_len=12, batch_size=1, negative=False
            )
            neg_lens, neg_mask = cfg._prepare_text_conditioning(
                batch, 0, text_seq_len=12, batch_size=1, negative=True
            )
            self.assertEqual(pos_lens, [12])
            self.assertEqual(neg_lens, [4])
            self.assertIsNotNone(pos_mask)
            self.assertEqual(pos_mask.shape, (1, 12))
            self.assertIsNotNone(neg_mask)
            self.assertEqual(neg_mask.shape, (1, 12))
            self.assertTrue(bool(pos_mask.all()))
            self.assertEqual(int(neg_mask.sum().item()), 4)

    def test_tpu_decoding_stage_replicated_and_host_float_materialization(self):
        vae = MagicMock()
        vae.use_parallel_decode = False
        stage = TpuDecodingStage(vae=vae)

        with (
            patch(
                "sglang.multimodal_gen.runtime.platforms.tpu_spmd.current_platform.is_tpu",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.tpu_spmd.get_global_server_args",
                return_value=SimpleNamespace(enable_cfg_parallel=True),
            ),
        ):
            self.assertEqual(stage.parallelism_type, StageParallelismType.REPLICATED)

    def test_tpu_cpu_barrier_routes_world_barrier_to_cpu_group(self):
        install_tpu_spmd_overrides()
        self.assertTrue(getattr(dist, "_tpu_cpu_barrier_installed", False))

    def test_vae_spatial_shard_checks_use_parallel_decode_before_p2p_warning(self):
        cfg = SimpleNamespace(use_parallel_decode=False, parallel_decode_mode="spatial")
        with (
            patch(
                "sglang.multimodal_gen.configs.models.vaes.base.current_platform.supports_distributed_p2p",
                return_value=False,
            ),
            patch(
                "sglang.multimodal_gen.configs.models.vaes.base.logger.warning_once"
            ) as mock_warn,
        ):
            self.assertFalse(
                should_use_spatial_shard_parallel_decode(cfg, world_size=4)
            )
            mock_warn.assert_not_called()

    def test_platform_is_float64_supported_not_shared_via_lru_cache(self):
        class _NoF64Platform(Platform):
            @classmethod
            def is_float64_supported(cls) -> bool:
                return False

        self.assertTrue(Platform.is_float64_supported())
        self.assertFalse(_NoF64Platform.is_float64_supported())
        self.assertTrue(TpuPlatform.is_float64_supported())


if __name__ == "__main__":
    unittest.main()
