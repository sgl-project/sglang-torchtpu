# SPDX-License-Identifier: Apache-2.0
"""Tests for SANA-Video 720p and sglang_tpu's SANA-Video RoPE (sglang_tpu.multimodal_gen.sana_video)."""

import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen import registry
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs import sana_video as config
from sglang.multimodal_gen.configs.sample.sana_video import (
    SanaVideo720pSamplingParams,
    SanaVideoSamplingParams,
)
from sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae import AutoencoderKLLTX2Video
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
    scale_and_shift_latents,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.utils.common import parse_size
from sglang.multimodal_gen.runtime.utils.precision import resolve_decode_precision
from sglang.multimodal_gen.runtime.warmup_request_builder import (
    _resolve_warmup_num_frames,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=30, suite="nightly-tpu-4-v7", nightly=True)

HF_ID_720P = "Efficient-Large-Model/SANA-Video_2B_720p_diffusers"
# The latent layout keys of the 720p checkpoint's vae/config.json.
VAE_720P = {
    "latent_channels": 128,
    "patch_size": 4,
    "scaling_factor": 1.0,
    "spatial_compression_ratio": 32,
    "temporal_compression_ratio": 8,
    "decoder_upsample_type": None,
    "decoder_causal": False,
}
RECIPE = Path(__file__).parents[4] / "benchmark/tpu/diffusion/sana_video.sh"
# A one-rank tensor-parallel group: the DiT builds without a process group.


class TestSanaVideo720pConfig(CustomTestCase):
    def _config(self):
        cfg = config.SanaVideo720pPipelineConfig()
        # What the VAE and DiT loaders read from the checkpoint.
        cfg.vae_config.update_model_arch(dict(VAE_720P))
        cfg.dit_config.update_model_arch(
            {"in_channels": 128, "out_channels": 128, "patch_size": [1, 1, 1]}
        )
        return cfg

    def test_registry_resolves_the_720p_checkpoint(self):
        cache = f"/hf/hub/models--{HF_ID_720P.replace('/', '--')}/snapshots/0a1b2c"
        cases = (
            (
                HF_ID_720P,
                config.SanaVideo720pPipelineConfig,
                SanaVideo720pSamplingParams,
            ),
            (cache, config.SanaVideo720pPipelineConfig, SanaVideo720pSamplingParams),
            (
                "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
                config.SanaVideoPipelineConfig,
                SanaVideoSamplingParams,
            ),
        )
        index = {"_class_name": "SanaVideoPipeline"}
        with patch.object(registry, "maybe_download_model_index", lambda _: index):
            for path, pipeline_config_cls, sampling_cls in cases:
                registry.get_model_info.cache_clear()
                info = registry.get_model_info(path)
                self.assertIs(info.pipeline_config_cls, pipeline_config_cls, path)
                self.assertIs(info.sampling_param_cls, sampling_cls, path)
        registry.get_model_info.cache_clear()

    def test_latents_follow_the_ltx2_vae(self):
        cfg = self._config()
        sampling = SanaVideo720pSamplingParams()
        batch = SimpleNamespace(
            height=sampling.height, width=sampling.width, num_frames=81
        )
        stage = LatentPreparationStage.__new__(LatentPreparationStage)
        frames = stage.adjust_video_length(batch, SimpleNamespace(pipeline_config=cfg))
        self.assertEqual(cfg.adjust_num_frames(81), 81)
        self.assertEqual(
            cfg.prepare_latent_shape(batch, 1, frames), (1, 128, 11, 22, 40)
        )
        # The checkpoint's VAE config builds the LTX-2 decoder.
        with torch.device("meta"):
            vae = AutoencoderKLLTX2Video(cfg.vae_config)
        self.assertEqual(tuple(vae.latents_std.shape), (128,))

    def test_fp32_decode_denormalizes_as_diffusers(self):
        cfg = self._config()  # with --vae-precision fp32
        gen = torch.Generator().manual_seed(0)
        vae = SimpleNamespace(
            latents_mean=torch.randn(128, generator=gen),
            latents_std=torch.rand(128, generator=gen) + 0.5,
        )
        latents = torch.randn(1, 128, 2, 3, 4, generator=gen)
        got = scale_and_shift_latents(
            latents, SimpleNamespace(pipeline_config=cfg), vae
        )
        # Diffusers' SanaVideoPipeline for AutoencoderKLLTX2Video.
        std = 1.0 / vae.latents_std.view(1, -1, 1, 1, 1)
        want = latents / std + vae.latents_mean.view(1, -1, 1, 1, 1)
        self.assertTrue(torch.equal(got, want))

    def test_bf16_decode_rounds_the_float32_denormalization_once(self):
        cfg = self._config()
        gen = torch.Generator().manual_seed(0)
        # A bf16-loaded VAE holds its latent statistics in bf16.
        mean = torch.randn(128, generator=gen).bfloat16()
        std = (torch.rand(128, generator=gen) + 0.5).bfloat16()
        vae = SimpleNamespace(latents_mean=mean, latents_std=std)
        latents = torch.randn(1, 128, 4, 6, 8, generator=gen)
        # DecodingStage.decode: de-normalize the float32 latents, then cast.
        got = scale_and_shift_latents(
            latents, SimpleNamespace(pipeline_config=cfg), vae
        ).bfloat16()
        mean, std = (t.float().view(1, -1, 1, 1, 1) for t in (mean, std))
        self.assertTrue(torch.equal(got, (latents / (1.0 / std) + mean).bfloat16()))
        # Diffusers casts first and de-normalizes in bf16, so its VAE input
        # differs by bf16 roundings and lies farther from the exact value.
        bf16 = latents.bfloat16() / (1.0 / std.bfloat16()) + mean.bfloat16()
        exact = latents.double() * std.double() + mean.double()
        err, bf16_err = ((t.double() - exact).abs().max() for t in (got, bf16))
        self.assertFalse(torch.equal(got, bf16))
        self.assertLess(err, bf16_err)

    def test_requests_keep_their_frames_and_warmup_runs_the_served_shape(self):
        recipe = RECIPE.read_text()
        size = re.search(r"--warmup-resolutions (\S+)", recipe).group(1)
        frames = int(re.search(r"--warmup-num-frames (\d+)", recipe).group(1))
        for cfg, sampling_cls in (
            (self._config(), SanaVideo720pSamplingParams),
            (config.SanaVideoPipelineConfig(), SanaVideoSamplingParams),
        ):
            for num_gpus in (1, 2):  # 2: CFG parallel
                args = SimpleNamespace(
                    pipeline_config=cfg,
                    num_gpus=num_gpus,
                    output_path=None,
                    comfyui_mode=True,
                    warmup_num_frames=frames,
                )
                served = sampling_cls(num_frames=81)
                served._adjust(args)
                self.assertEqual(served.num_frames, 81, (sampling_cls, num_gpus))
                if sampling_cls is SanaVideo720pSamplingParams:
                    warmup = _resolve_warmup_num_frames(
                        args, sampling_cls(), server_based_warmup=True
                    )
                    self.assertEqual(
                        (parse_size(size), warmup),
                        ((served.width, served.height), served.num_frames),
                    )

    def test_sizes_off_the_32_pixel_latent_grid_are_rejected(self):
        cfg = self._config()
        SanaVideo720pSamplingParams()._validate_with_pipeline_config(cfg)
        for width, height in ((1280, 720), (1270, 704)):
            params = SanaVideo720pSamplingParams(width=width, height=height)
            with self.assertRaisesRegex(ValueError, "multiples of 32"):
                params._validate_with_pipeline_config(cfg)

    def test_vae_runs_bf16_and_fp32_is_one_flag_away(self):
        cfg = self._config()
        args = SimpleNamespace(component_precisions={}, pipeline_config=cfg)
        self.assertEqual(cfg.vae_precision, "bf16")
        self.assertIs(resolve_decode_precision(args, "vae"), torch.bfloat16)
        cfg.vae_precision = "fp32"  # --vae-precision fp32
        self.assertIs(resolve_decode_precision(args, "vae"), torch.float32)

    def test_480p_keeps_the_wan_vae(self):
        cfg = config.SanaVideoPipelineConfig()
        self.assertIsInstance(cfg.vae_config, WanVAEConfig)
        batch = SimpleNamespace(height=480, width=832, num_frames=81)
        stage = LatentPreparationStage.__new__(LatentPreparationStage)
        frames = stage.adjust_video_length(batch, SimpleNamespace(pipeline_config=cfg))
        self.assertEqual(
            cfg.prepare_latent_shape(batch, 1, frames), (1, 16, 21, 60, 104)
        )
        self.assertEqual(cfg.vae_decode_precision, "fp32")


if __name__ == "__main__":
    unittest.main()
