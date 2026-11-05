"""LTX-2 two-stage on torch_tpu: subgroup collectives run as slice-wide
programs, so every rank must issue the same collectives in the same order, and
there are no autocast conv kernels, so the bf16 VAE needs bf16 latents."""

import itertools
import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.decoding_av import (
    LTX2AVDecodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising_av import (
    LTX2RefinementStage,
)

_LTX2 = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2"
)
_LATENT_CHANNELS = 4


def _platform(on_tpu: bool):
    return SimpleNamespace(is_tpu=lambda: on_tpu)


class TestLtx2RefinementStageParallelism(unittest.TestCase):
    @staticmethod
    def _parallelism_type(*, on_tpu, cfg_parallel, decode_uses_every_rank):
        stage = object.__new__(LTX2RefinementStage)
        stage.server_args = SimpleNamespace(enable_cfg_parallel=cfg_parallel)
        stage.vae = SimpleNamespace(use_parallel_decode=True)
        with (
            patch(f"{_LTX2}.denoising_av.current_platform", _platform(on_tpu)),
            patch(
                f"{_LTX2}.denoising_av.decode_uses_every_rank",
                return_value=decode_uses_every_rank,
            ),
        ):
            return stage.parallelism_type

    def test_tpu_runs_stage2_on_every_cfg_rank(self):
        # Run on CFG rank 0 alone, stage 2's Ulysses all-to-alls would wait for
        # the CFG ranks parked in the barrier.
        for cfg_parallel, every_rank in itertools.product((True, False), repeat=2):
            with self.subTest(cfg_parallel=cfg_parallel, decode_every_rank=every_rank):
                self.assertEqual(
                    self._parallelism_type(
                        on_tpu=True,
                        cfg_parallel=cfg_parallel,
                        decode_uses_every_rank=every_rank,
                    ),
                    StageParallelismType.REPLICATED,
                )

    def test_other_platforms_keep_stage2_on_the_main_cfg_rank(self):
        cases = (
            (True, True, StageParallelismType.MAIN_RANK_ONLY_AND_SEND_TO_OTHERS),
            (True, False, StageParallelismType.MAIN_RANK_ONLY),
            (False, True, StageParallelismType.REPLICATED),
            (False, False, StageParallelismType.REPLICATED),
        )
        for cfg_parallel, every_rank, expected in cases:
            with self.subTest(cfg_parallel=cfg_parallel, decode_every_rank=every_rank):
                self.assertEqual(
                    self._parallelism_type(
                        on_tpu=False,
                        cfg_parallel=cfg_parallel,
                        decode_uses_every_rank=every_rank,
                    ),
                    expected,
                )


class _Bf16VAE(nn.Module):
    """A bf16 video VAE that records the latent dtype each decode receives."""

    def __init__(self):
        super().__init__()
        self.conv_out = nn.Conv3d(
            _LATENT_CHANNELS, 3, kernel_size=1, dtype=torch.bfloat16
        )
        self.register_buffer(
            "latents_mean", torch.zeros(_LATENT_CHANNELS, dtype=torch.bfloat16)
        )
        self.register_buffer(
            "latents_std", torch.ones(_LATENT_CHANNELS, dtype=torch.bfloat16)
        )
        self.decoded_dtypes = []

    def decode(self, latents):
        self.decoded_dtypes.append(latents.dtype)
        return self.conv_out(latents)


@contextmanager
def _resident_component(*, component_name, module=None):
    yield module


class TestLtx2DecodeCastsLatentsToVaeDtype(unittest.TestCase):
    @staticmethod
    def _decode(*, autocast, on_tpu=True):
        vae = _Bf16VAE()
        stage = object.__new__(LTX2AVDecodingStage)
        stage.vae = vae
        stage.diffusion_decoder = None
        stage.load_model = lambda: None
        stage.use_declared_component = _resident_component
        stage.video_processor = SimpleNamespace(
            postprocess_video=lambda video, output_type: video
        )
        server_args = SimpleNamespace(
            component_precisions={},
            disable_autocast=False,
            pipeline_config=SimpleNamespace(
                vae_decode_precision="bf16",
                vae_tiling=False,
                vae_config=SimpleNamespace(
                    arch_config=SimpleNamespace(video_decoder_variant="ltx_2")
                ),
                preprocess_decoding=lambda latents, server_args, vae: latents,
            ),
        )
        # Two-stage refinement hands fp32 latents to the decoder.
        batch = SimpleNamespace(
            latents=torch.ones(1, _LATENT_CHANNELS, 2, 2, 2),
            use_diffusion_decoder=False,
            audio_latents=None,
            trajectory_timesteps=None,
            trajectory_latents=None,
            metrics=None,
            rollout_trajectory_data=None,
        )
        with (
            patch(
                f"{_LTX2}.decoding_av.get_local_torch_device",
                return_value=torch.device("cpu"),
            ),
            patch(f"{_LTX2}.decoding_av.current_platform", _platform(on_tpu)),
            patch(f"{_LTX2}.decoding_av.autocast_enabled", return_value=autocast),
            # torch_tpu reports autocast enabled but has no autocast conv
            # kernels, so the VAE gets the latents in whatever dtype they carry.
            patch(
                f"{_LTX2}.decoding_av.autocast_context",
                side_effect=lambda *args, **kwargs: (
                    nullcontext()
                    if on_tpu
                    else torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast)
                ),
            ),
            torch.no_grad(),
        ):
            output = stage.forward(batch, server_args)
        return vae.decoded_dtypes, output.output

    def test_tpu_casts_fp32_latents_to_the_vae_dtype(self):
        # Uncast, the bf16 VAE fails as on TPU: "Input type (float) and bias
        # type (c10::BFloat16) should be the same".
        for autocast in (True, False):
            with self.subTest(autocast=autocast):
                decoded_dtypes, video = self._decode(autocast=autocast)
                self.assertEqual(decoded_dtypes, [torch.bfloat16])
                self.assertEqual(video.dtype, torch.bfloat16)

    def test_other_platforms_leave_casting_to_autocast(self):
        decoded_dtypes, _ = self._decode(autocast=True, on_tpu=False)
        self.assertEqual(decoded_dtypes, [torch.float32])
        decoded_dtypes, _ = self._decode(autocast=False, on_tpu=False)
        self.assertEqual(decoded_dtypes, [torch.bfloat16])


if __name__ == "__main__":
    unittest.main()
