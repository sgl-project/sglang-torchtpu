import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2DenoisingStage,
)

_CONFIG = "sglang.multimodal_gen.configs.pipeline_configs.ltx_2"


def _shard(fn, latents, *, rank, world_size=4, **batch_kwargs):
    batch = SimpleNamespace(**batch_kwargs)
    with (
        patch(f"{_CONFIG}.get_sp_world_size", return_value=world_size),
        patch(f"{_CONFIG}.get_sp_parallel_rank", return_value=rank),
    ):
        fn(batch, latents)
    return batch


def _video_self_attention_mask(batch, *, seq_len, batch_size):
    return LTX2DenoisingStage._build_ltx2_sp_self_attention_mask(
        batch,
        seq_len=seq_len,
        batch_size=batch_size,
        key="sp_video_valid_token_count",
        padded_key="sp_video_padded",
        device=torch.device("cpu"),
    )


class TestLtx2SpPaddingMask(unittest.TestCase):
    def test_video_sharding_records_global_padding_on_every_rank(self):
        config = LTX2PipelineConfig()
        tokens_per_frame = 4 * 4
        height = width = 4 * config.vae_scale_factor
        for frames, padded in ((16, False), (17, True)):
            latents = torch.zeros(1, frames * tokens_per_frame, 8)
            for rank in range(4):
                with self.subTest(frames=frames, rank=rank):
                    batch = _shard(
                        config.shard_latents_for_sp,
                        latents,
                        rank=rank,
                        height=height,
                        width=width,
                    )
                    self.assertIs(batch.sp_video_padded, padded)

    def test_audio_sharding_records_global_padding_on_every_rank(self):
        config = LTX2PipelineConfig()
        for frames, padded in ((128, False), (126, True)):
            for rank in range(4):
                with self.subTest(frames=frames, rank=rank):
                    batch = _shard(
                        config.shard_audio_latents_for_sp,
                        torch.zeros(1, frames, 8),
                        rank=rank,
                    )
                    self.assertIs(batch.sp_audio_padded, padded)

    def test_mixed_padding_masks_only_the_padded_stream(self):
        cpu = torch.device("cpu")
        for video_padded in (False, True):
            audio_padded = not video_padded
            with self.subTest(video_padded=video_padded):
                batch = SimpleNamespace(
                    sp_video_valid_token_count=6 if video_padded else 8,
                    sp_video_padded=video_padded,
                    sp_audio_valid_token_count=3 if audio_padded else 4,
                    sp_audio_padded=audio_padded,
                )
                video, audio, a2v, v2a = (
                    LTX2DenoisingStage._build_ltx2_sp_attention_masks(
                        batch,
                        video_seq_len=8,
                        audio_seq_len=4,
                        batch_size=2,
                        video_device=cpu,
                        audio_device=cpu,
                    )
                )
                padded, unpadded = (video, audio) if video_padded else (audio, video)
                self.assertIsNone(unpadded)
                self.assertIsNotNone(padded)
                self.assertFalse(bool(padded[:, -1].any()))
                # Cross-attention takes its key stream's mask.
                self.assertIs(a2v, audio)
                self.assertIs(v2a, video)

    def test_unpadded_sp_shards_skip_the_self_attention_mask(self):
        batch = SimpleNamespace(sp_video_valid_token_count=8, sp_video_padded=False)
        self.assertIsNone(_video_self_attention_mask(batch, seq_len=8, batch_size=2))

    def test_padded_sp_shards_keep_the_mask_on_full_ranks(self):
        # A full rank still builds its all-true mask when another rank is padded,
        # so every rank takes the masked path.
        full = SimpleNamespace(sp_video_valid_token_count=8, sp_video_padded=True)
        tail = SimpleNamespace(sp_video_valid_token_count=5, sp_video_padded=True)
        self.assertTrue(_video_self_attention_mask(full, seq_len=8, batch_size=1).all())
        self.assertEqual(
            _video_self_attention_mask(tail, seq_len=8, batch_size=1).tolist(),
            [[True] * 5 + [False] * 3],
        )

    def test_shards_without_the_padding_flag_keep_the_mask(self):
        batch = SimpleNamespace(sp_video_valid_token_count=8)
        mask = _video_self_attention_mask(batch, seq_len=8, batch_size=1)
        self.assertTrue(mask.all())


if __name__ == "__main__":
    unittest.main()
