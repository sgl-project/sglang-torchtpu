# SPDX-License-Identifier: Apache-2.0
"""Sampling defaults for SANA-Video 2B 480p and 720p."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class SanaVideoSamplingParams(SamplingParams):
    data_type: DataType = DataType.VIDEO
    num_frames: int = 81
    fps: int = 16
    guidance_scale: float = 6.0
    num_inference_steps: int = 50
    height: int = 480
    width: int = 832
    max_sequence_length: int | None = 300
    # The DiT does not shard the sequence, so the latent frame count stays as
    # requested under --num-gpus (CFG parallel runs one branch per rank).
    adjust_frames: bool = False
    negative_prompt: str = (
        "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, "
        "sudden disappearance, jump cuts, jerky movements, rapid shot changes, "
        "frames out of sync, inconsistent character shapes, temporal artifacts, "
        "jitter, and ghosting effects, creating a disorienting visual experience."
    )


@dataclass
class SanaVideo720pSamplingParams(SanaVideoSamplingParams):
    height: int = 704
    width: int = 1280

    def _validate_with_pipeline_config(self, pipeline_config):
        super()._validate_with_pipeline_config(pipeline_config)
        # Each latent row and column decodes to 32 pixels (LTX-2 VAE), so other
        # sizes would come back cropped; diffusers' check_inputs rejects them too.
        scale = pipeline_config.vae_config.arch_config.spatial_compression_ratio
        if self.height % scale or self.width % scale:
            raise ValueError(
                f"SANA-Video 720p height and width must be multiples of {scale}, "
                f"got {self.width}x{self.height} (e.g. 1280x704)"
            )
