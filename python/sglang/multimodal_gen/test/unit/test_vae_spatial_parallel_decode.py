import threading
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.models.vaes.base import (
    VAEConfig,
    is_spatial_shard_parallel_decode_mode,
    should_use_spatial_shard_parallel_decode,
)
from sglang.multimodal_gen.configs.models.vaes.ernie_image import ErnieImageVAEConfig
from sglang.multimodal_gen.configs.models.vaes.flux import Flux2VAEConfig, FluxVAEConfig
from sglang.multimodal_gen.configs.models.vaes.glmimage import GlmImageVAEConfig
from sglang.multimodal_gen.configs.models.vaes.hunyuan3d import Hunyuan3DVAEConfig
from sglang.multimodal_gen.configs.models.vaes.hunyuanvae import HunyuanVAEConfig
from sglang.multimodal_gen.configs.models.vaes.ltx_audio import LTXAudioVAEConfig
from sglang.multimodal_gen.configs.models.vaes.ltx_video import LTXVideoVAEConfig
from sglang.multimodal_gen.configs.models.vaes.qwenimage import QwenImageVAEConfig
from sglang.multimodal_gen.configs.models.vaes.sana import SanaVAEConfig
from sglang.multimodal_gen.configs.models.vaes.stablediffusion3 import (
    StableDiffusion3VAEConfig,
)
from sglang.multimodal_gen.configs.models.vaes.wanvae import WanVAEConfig
from sglang.multimodal_gen.configs.utils import update_config_from_args
from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.distributed.utils import RankGenerator
from sglang.multimodal_gen.runtime.layers import parallel_conv
from sglang.multimodal_gen.runtime.layers.parallel_conv import (
    SpatialParallelCausalConv3d,
    SpatialParallelConv2d,
    SpatialParallelConv3d,
    chunk_height_by_sizes,
    disable_spatial_parallel_decode,
    split_for_parallel_decode,
)
from sglang.multimodal_gen.runtime.models.vaes import wanvae
from sglang.multimodal_gen.runtime.models.vaes.autoencoder import AutoencoderKL
from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage import (
    QwenImageAttentionBlock,
    QwenImageDecoder3d,
)
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE
from sglang.multimodal_gen.runtime.models.vaes.hunyuanvae import (
    HunyuanVideoDecoder3D,
    HunyuanVideoMidBlock3D,
    HunyuanVideoResnetBlockCausal3D,
    _enable_hunyuan_decoder_spatial_parallel,
)
from sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae import (
    LTX2VideoCausalConv3d,
    LTX2VideoDecoder3d,
    _enable_ltx_decoder_spatial_parallel,
)
from sglang.multimodal_gen.runtime.models.vaes.wanvae import (
    WanCausalConv3d,
    WanDecoder3d,
    WanDistAttentionBlock,
    WanEncoder3d,
    feat_idx,
    first_chunk,
    forward_context,
    split_for_parallel_encode,
)
from sglang.multimodal_gen.runtime.utils.argparse import FlexibleArgumentParser


class _DispatchProbeVAE(ParallelTiledVAE):
    def __init__(self, config):
        super().__init__(config)
        self.used_decode = False
        self.used_parallel_tiled_decode = False

    def _encode(self, x):
        return x

    def _decode(self, z):
        self.used_decode = True
        return z

    def parallel_tiled_decode(self, z):
        self.used_parallel_tiled_decode = True
        raise AssertionError(
            "spatial_shard decode should not use parallel_tiled_decode"
        )


_thread_rank = threading.local()


class _ThreadedDecodeGroup:
    """Decode group stand-in that runs one thread per rank."""

    def __init__(self, world_size: int):
        self.world_size = world_size
        self.ranks = list(range(world_size))
        self.device_group = None
        # A rank-divergent collective sequence fails instead of hanging.
        self.barrier = threading.Barrier(world_size, timeout=60)
        self._slots = [None] * world_size

    def _exchange(self, tensor):
        self._slots[_thread_rank.value] = tensor.clone()
        self.barrier.wait()
        gathered = list(self._slots)
        self.barrier.wait()
        return gathered

    def all_gather(self, input_, dim=0, separate_tensors=False):
        gathered = self._exchange(input_)
        return gathered if separate_tensors else torch.cat(gathered, dim=dim)

    def gather_height_sizes(self, x):
        return [int(h) for h in self._exchange(torch.tensor(x.shape[-2]))]


@contextmanager
def _collectives_only_decode_group(world_size: int):
    """Patch the decode group with ``world_size`` thread ranks and no p2p."""
    group = _ThreadedDecodeGroup(world_size)
    with (
        patch.object(
            parallel_conv, "get_decode_parallel_group_coordinator", lambda: group
        ),
        patch.object(
            parallel_conv, "get_decode_parallel_rank", lambda: _thread_rank.value
        ),
        patch.object(
            parallel_conv, "get_decode_parallel_world_size", lambda: world_size
        ),
        patch.object(parallel_conv, "gather_height_sizes", group.gather_height_sizes),
        patch.object(
            parallel_conv.current_platform,
            "supports_distributed_p2p",
            return_value=False,
        ),
    ):
        yield group


def _run_on_thread_ranks(group: _ThreadedDecodeGroup, fn) -> list:
    outputs = [None] * group.world_size
    errors = []

    def run(rank):
        _thread_rank.value = rank
        try:
            outputs[rank] = fn(rank)
        except BaseException as e:
            errors.append(e)
            group.barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(group.world_size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return outputs


def _run_height_sharded_without_p2p(make_conv, x: torch.Tensor, world_size: int):
    """Run one conv per rank on height shards, as a collectives-only platform does."""
    shards = torch.tensor_split(x, world_size, dim=-2)
    with _collectives_only_decode_group(world_size) as group:
        _thread_rank.value = 0
        reference_conv = make_conv()
        with disable_spatial_parallel_decode():
            expected = reference_conv(x)
        convs = []
        for rank in range(world_size):
            _thread_rank.value = rank
            conv = make_conv()
            conv.load_state_dict(reference_conv.state_dict())
            convs.append(conv)
        outputs = _run_on_thread_ranks(group, lambda r: convs[r](shards[r]))
    return torch.cat(outputs, dim=-2), expected


def _encode_wan_video(encoder, x: torch.Tensor) -> torch.Tensor:
    num_convs = sum(
        isinstance(m, (WanCausalConv3d, SpatialParallelCausalConv3d))
        for m in encoder.modules()
    )
    outputs = []
    with forward_context(feat_cache_arg=[None] * num_convs, feat_idx_arg=0):
        for start, end in [(0, 1)] + [(s, s + 4) for s in range(1, x.shape[2], 4)]:
            feat_idx.set(0)
            outputs.append(encoder(x[:, :, start:end]))
    return torch.cat(outputs, dim=2)


def _decode_wan_latents(decoder, z: torch.Tensor) -> torch.Tensor:
    num_convs = sum(
        isinstance(m, (WanCausalConv3d, SpatialParallelCausalConv3d))
        for m in decoder.modules()
    )
    outputs = []
    with forward_context(feat_cache_arg=[None] * num_convs, feat_idx_arg=0):
        for i in range(z.shape[2]):
            feat_idx.set(0)
            first_chunk.set(i == 0)
            outputs.append(decoder(z[:, :, i : i + 1]))
    return torch.cat(outputs, dim=2)


class TestVAESpatialParallelDecode(unittest.TestCase):
    def test_image_video_vae_configs_default_to_auto_parallel_decode(self):
        configs = (
            ErnieImageVAEConfig(),
            FluxVAEConfig(),
            Flux2VAEConfig(),
            GlmImageVAEConfig(),
            HunyuanVAEConfig(),
            LTXVideoVAEConfig(),
            QwenImageVAEConfig(),
            SanaVAEConfig(),
            StableDiffusion3VAEConfig(),
            WanVAEConfig(),
        )

        for config in configs:
            with self.subTest(config=type(config).__name__):
                self.assertTrue(config.use_parallel_decode)
                self.assertEqual(config.parallel_decode_mode, "auto")

    @patch("sglang.multimodal_gen.configs.models.vaes.base.current_platform")
    def test_auto_parallel_decode_policy_is_conservative(self, platform):
        # With p2p halos; test_tpu_platform covers auto mode without p2p.
        platform.supports_distributed_p2p.return_value = True
        self.assertFalse(is_spatial_shard_parallel_decode_mode("auto"))
        self.assertFalse(should_use_spatial_shard_parallel_decode(VAEConfig()))
        self.assertFalse(should_use_spatial_shard_parallel_decode(QwenImageVAEConfig()))
        self.assertTrue(should_use_spatial_shard_parallel_decode(LTXVideoVAEConfig()))
        self.assertTrue(should_use_spatial_shard_parallel_decode(WanVAEConfig()))

        ltx23_config = LTXVideoVAEConfig()
        ltx23_config.arch_config.video_decoder_variant = "ltx_2_3"
        self.assertTrue(should_use_spatial_shard_parallel_decode(ltx23_config))
        ltx23_config.parallel_decode_mode = "spatial_shard"
        self.assertTrue(should_use_spatial_shard_parallel_decode(ltx23_config))

        config = QwenImageVAEConfig()
        self.assertFalse(
            should_use_spatial_shard_parallel_decode(
                config, torch.empty(1, 16, 1, 128, 128), 2
            )
        )
        config.parallel_decode_mode = "spatial_shard"
        self.assertTrue(should_use_spatial_shard_parallel_decode(config))

        hunyuan_config = HunyuanVAEConfig()
        self.assertTrue(should_use_spatial_shard_parallel_decode(hunyuan_config))
        self.assertFalse(
            should_use_spatial_shard_parallel_decode(
                hunyuan_config, torch.empty(1, 16, 9, 16, 16), 2
            )
        )
        self.assertTrue(
            should_use_spatial_shard_parallel_decode(
                hunyuan_config, torch.empty(1, 16, 9, 96, 96), 2
            )
        )

    def test_unsupported_vae_configs_opt_out_of_spatial_parallel_decode(self):
        configs = (Hunyuan3DVAEConfig(), LTXAudioVAEConfig())

        for config in configs:
            with self.subTest(config=type(config).__name__):
                self.assertFalse(config.use_parallel_decode)
                self.assertEqual(config.parallel_decode_mode, "tiled")

    def test_vae_nested_cli_defaults_do_not_override_model_defaults(self):
        parser = FlexibleArgumentParser()
        VAEConfig.add_cli_args(parser)
        parsed = vars(parser.parse_args([]))
        config = FluxVAEConfig()

        update_config_from_args(config, parsed, "vae_config")

        self.assertTrue(config.use_parallel_decode)
        self.assertEqual(config.parallel_decode_mode, "auto")

    def test_vae_nested_cli_explicit_args_override_model_defaults(self):
        parser = FlexibleArgumentParser()
        VAEConfig.add_cli_args(parser)
        parsed = vars(
            parser.parse_args(
                [
                    "--vae-config.use-parallel-decode",
                    "false",
                    "--vae-config.parallel-decode-mode",
                    "patch",
                ]
            )
        )
        config = FluxVAEConfig()

        update_config_from_args(config, parsed, "vae_config")

        self.assertFalse(config.use_parallel_decode)
        self.assertEqual(config.parallel_decode_mode, "patch")

    def test_base_decode_prefers_spatial_parallel_dispatch(self):
        config = VAEConfig()
        config.arch_config.temporal_compression_ratio = 1
        config.arch_config.spatial_compression_ratio = 1
        config.use_parallel_decode = True
        config.parallel_decode_mode = "spatial_shard"
        vae = _DispatchProbeVAE(config)
        z = torch.randn(1, 1, 1, 2, 2)

        with (
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.common.dist.is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.common.model_parallel_is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.common.get_decode_parallel_group_coordinator",
                return_value=SimpleNamespace(),
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.common.get_decode_parallel_world_size",
                return_value=2,
            ),
        ):
            out = vae.decode(z)

        self.assertTrue(vae.used_decode)
        self.assertFalse(vae.used_parallel_tiled_decode)
        torch.testing.assert_close(out, z)

    def test_auto_decode_does_not_fall_back_to_parallel_tiling(self):
        config = VAEConfig()
        config.arch_config.temporal_compression_ratio = 1
        config.arch_config.spatial_compression_ratio = 1
        config.use_parallel_decode = True
        config.parallel_decode_mode = "auto"
        vae = _DispatchProbeVAE(config)
        z = torch.randn(1, 1, 1, 2, 2)

        with patch(
            "sglang.multimodal_gen.runtime.models.vaes.common.get_sp_world_size",
            return_value=2,
        ):
            out = vae.decode(z)

        self.assertTrue(vae.used_decode)
        self.assertFalse(vae.used_parallel_tiled_decode)
        torch.testing.assert_close(out, z)

    def test_spatial_alias_still_uses_spatial_shard_dispatch(self):
        self.assertTrue(is_spatial_shard_parallel_decode_mode("spatial"))

    def test_decode_parallel_group_uses_dedicated_group(self):
        old_decode = parallel_state._VAE_DECODE
        decode_group = SimpleNamespace(world_size=4, rank_in_group=2)
        try:
            parallel_state._VAE_DECODE = decode_group
            self.assertIs(
                parallel_state.get_decode_parallel_group_coordinator(), decode_group
            )
            self.assertEqual(parallel_state.get_decode_parallel_world_size(), 4)
            self.assertEqual(parallel_state.get_decode_parallel_rank(), 2)
        finally:
            parallel_state._VAE_DECODE = old_decode

    def test_decode_rank_groups_cover_non_dp_parallel_axes(self):
        rank_generator = RankGenerator(
            tp=2,
            sp=2,
            pp=1,
            cfg=2,
            dp=2,
            order="tp-sp-pp-cfg-dp",
        )

        self.assertEqual(
            parallel_state._get_vae_decode_group_ranks(rank_generator),
            [list(range(0, 8)), list(range(8, 16))],
        )

    def test_spatial_split_keeps_uneven_height_lossless(self):
        x = torch.arange(5).view(1, 1, 1, 5, 1)

        rank0, expected_height = split_for_parallel_decode(
            x, upsample_count=1, world_size=2, rank=0
        )
        rank1, _ = split_for_parallel_decode(x, upsample_count=1, world_size=2, rank=1)

        self.assertEqual(expected_height, 10)
        self.assertEqual(rank0.shape[-2], 3)
        self.assertEqual(rank1.shape[-2], 2)
        torch.testing.assert_close(torch.cat([rank0, rank1], dim=-2), x)

    def test_chunk_height_by_sizes_keeps_original_partition(self):
        x = torch.arange(10).view(1, 1, 10, 1)

        with patch(
            "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_rank",
            return_value=0,
        ):
            rank0 = chunk_height_by_sizes(x, [6, 4])
        with patch(
            "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_rank",
            return_value=1,
        ):
            rank1 = chunk_height_by_sizes(x, [6, 4])

        self.assertEqual(rank0.shape[-2], 6)
        self.assertEqual(rank1.shape[-2], 4)
        torch.testing.assert_close(torch.cat([rank0, rank1], dim=-2), x)

    def test_collective_halo_exchange_matches_unsharded_conv(self):
        torch.manual_seed(0)
        cases = [
            (SpatialParallelConv2d, torch.randn(1, 4, 10, 7)),
            (SpatialParallelCausalConv3d, torch.randn(1, 4, 3, 10, 7)),
        ]
        for conv_cls, x in cases:
            make_conv = lambda: conv_cls(4, 5, 3, padding=1)
            for world_size in (2, 3, 4):
                with self.subTest(conv=conv_cls.__name__, world=world_size):
                    actual, expected = _run_height_sharded_without_p2p(
                        make_conv, x, world_size
                    )
                    torch.testing.assert_close(actual, expected)

    def test_wan_decoder_shards_height_without_p2p(self):
        # is_residual: the Wan2.2 (TI2V-5B) VAE's residual up blocks.
        for is_residual in (False, True):

            def build(use_parallel_decode, is_residual=is_residual):
                return WanDecoder3d(
                    dim=8,
                    z_dim=4,
                    dim_mult=(1, 2, 2),
                    num_res_blocks=1,
                    attn_scales=(),
                    temperal_upsample=(True, False),
                    out_channels=3,
                    is_residual=is_residual,
                    use_parallel_decode=use_parallel_decode,
                ).eval()

            with self.subTest(is_residual=is_residual):
                torch.manual_seed(0)
                world_size = 3
                reference = build(False)
                z = torch.randn(1, 4, 3, 10, 6)
                with torch.no_grad():
                    expected = _decode_wan_latents(reference, z)

                with (
                    _collectives_only_decode_group(world_size) as group,
                    patch.object(wanvae.dist, "is_initialized", return_value=True),
                    patch.object(
                        wanvae, "get_decode_parallel_world_size", lambda: world_size
                    ),
                    patch.object(
                        wanvae, "get_decode_parallel_rank", lambda: _thread_rank.value
                    ),
                ):
                    decoders = []
                    for rank in range(world_size):
                        _thread_rank.value = rank
                        decoder = build(True)
                        decoder.load_state_dict(reference.state_dict())
                        decoders.append(decoder)
                    with torch.no_grad():
                        outputs = _run_on_thread_ranks(
                            group, lambda r: _decode_wan_latents(decoders[r], z)
                        )

                for output in outputs:
                    torch.testing.assert_close(output, expected)

    def _encode_on_thread_ranks(self, is_residual, x, world_size=3):
        """The unsharded Wan encoder's output, and each thread rank's when the
        encoder shards over a collectives-only decode group."""

        def build(use_parallel_encode):
            return WanEncoder3d(
                dim=8,
                z_dim=4,
                dim_mult=(1, 2, 2),
                num_res_blocks=1,
                temperal_downsample=(True, False),
                is_residual=is_residual,
                use_parallel_encode=use_parallel_encode,
            ).eval()

        torch.manual_seed(0)
        reference = build(False)
        with torch.no_grad():
            expected = _encode_wan_video(reference, x)

        with (
            _collectives_only_decode_group(world_size) as group,
            patch.object(wanvae.dist, "is_initialized", return_value=True),
            patch.object(wanvae, "get_sp_world_size", lambda: world_size),
            patch.object(wanvae, "get_decode_parallel_world_size", lambda: world_size),
            patch.object(
                wanvae, "get_decode_parallel_rank", lambda: _thread_rank.value
            ),
        ):
            encoders = []
            for rank in range(world_size):
                _thread_rank.value = rank
                encoder = build(True)
                encoder.load_state_dict(reference.state_dict())
                encoders.append(encoder)
            with torch.no_grad():
                outputs = _run_on_thread_ranks(
                    group, lambda r: _encode_wan_video(encoders[r], x)
                )
        return expected, outputs

    def test_wan_encoder_shards_height_without_p2p(self):
        # is_residual: the Wan2.2 (TI2V-5B) VAE's residual down blocks.
        for is_residual in (False, True):
            with self.subTest(is_residual=is_residual):
                # 10 blocks of 4 rows split 4/3/3, so the shards are uneven.
                # 9 frames encode in chunks of 1, 4 and 4: the last resumes
                # the height-sharded conv caches, the strided temporal
                # downsample's included.
                x = torch.randn(
                    1, 3, 9, 40, 8, generator=torch.Generator().manual_seed(1)
                )
                expected, outputs = self._encode_on_thread_ranks(is_residual, x)
                for output in outputs:
                    torch.testing.assert_close(output, expected)

    def test_wan_encoder_encodes_unsplit_heights_unsharded(self):
        # A height that is not whole downsampling blocks (38 rows, blocks of
        # 4), or fewer blocks than ranks (8 rows on 3 ranks), stays unsplit:
        # every rank encodes the whole input. The residual blocks need an
        # even height at every level, so they take the second case.
        for is_residual, height in ((False, 38), (True, 8)):
            with self.subTest(is_residual=is_residual, height=height):
                x = torch.randn(
                    1, 3, 9, height, 8, generator=torch.Generator().manual_seed(2)
                )
                self.assertIsNone(split_for_parallel_encode(x, 2, 3, 0))
                expected, outputs = self._encode_on_thread_ranks(is_residual, x)
                for output in outputs:
                    torch.testing.assert_close(output, expected)

    def test_wan_encoder_shards_only_under_sequence_parallelism(self):
        def build(sp_world_size, decode_world_size):
            _thread_rank.value = 0
            with (
                _collectives_only_decode_group(decode_world_size),
                patch.object(wanvae.dist, "is_initialized", return_value=True),
                patch.object(wanvae, "get_sp_world_size", lambda: sp_world_size),
                patch.object(
                    wanvae, "get_decode_parallel_world_size", lambda: decode_world_size
                ),
                patch.object(wanvae, "get_decode_parallel_rank", lambda: 0),
            ):
                return WanEncoder3d(
                    dim=8,
                    z_dim=4,
                    dim_mult=(1, 2, 2),
                    num_res_blocks=1,
                    temperal_downsample=(True, False),
                    use_parallel_encode=True,
                )

        # CFG-parallel only (decode group 2, one SP rank): every rank encodes
        # the whole input, as before.
        encoder = build(sp_world_size=1, decode_world_size=2)
        self.assertEqual(encoder.world_size, 1)
        self.assertNotIsInstance(encoder.conv_in, SpatialParallelCausalConv3d)
        # SP 2 x CFG 2: the input splits over the 4-rank decode group.
        encoder = build(sp_world_size=2, decode_world_size=4)
        self.assertEqual(encoder.world_size, 4)
        self.assertIsInstance(encoder.conv_in, SpatialParallelCausalConv3d)

    def test_explicit_spatial_shard_needs_a_latent_row_per_rank(self):
        from sglang.multimodal_gen.runtime.models.vaes import common

        config = WanVAEConfig()
        config.parallel_decode_mode = "spatial_shard"
        with patch.object(common, "_decode_parallel_world_size", lambda: 4):
            self.assertFalse(
                common.should_run_spatial_shard_parallel_decode(
                    config, torch.empty(1, 4, 1, 3, 8)
                )
            )
            self.assertTrue(
                common.should_run_spatial_shard_parallel_decode(
                    config, torch.empty(1, 4, 1, 4, 8)
                )
            )

    def test_wan_encoder_returns_the_whole_input_when_sharding_is_disabled(self):
        encoder = WanEncoder3d(
            dim=8,
            z_dim=4,
            dim_mult=(1, 2, 2),
            num_res_blocks=1,
            temperal_downsample=(True, False),
            use_parallel_encode=True,
        ).eval()
        # As if built under a 3-rank group; the disabled context must win.
        encoder.world_size, encoder.rank = 3, 1
        x = torch.randn(1, 3, 1, 40, 8)
        with (
            patch.object(
                encoder, "_forward", side_effect=lambda t: t.shape[-2]
            ) as forward,
            disable_spatial_parallel_decode(),
        ):
            self.assertEqual(encoder(x), 40)
        forward.assert_called_once()

    def test_wan_encode_split_keeps_whole_downsampling_blocks(self):
        x = torch.arange(40).view(1, 1, 1, 40, 1)
        shards = [split_for_parallel_encode(x, 3, 3, rank) for rank in range(3)]
        self.assertEqual([shard.shape[-2] for shard, _ in shards], [16, 16, 8])
        self.assertEqual({height for _, height in shards}, {5})
        torch.testing.assert_close(torch.cat([s for s, _ in shards], dim=-2), x)
        # Heights that are not whole blocks, or too few blocks, stay unsplit.
        self.assertIsNone(split_for_parallel_encode(x[..., :36, :], 3, 3, 0))
        self.assertIsNone(split_for_parallel_encode(x[..., :16, :], 3, 3, 0))

    def test_qwen_decoder_uses_spatial_parallel_components(self):
        with (
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage.dist.is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage.model_parallel_is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage.get_decode_parallel_world_size",
                return_value=2,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage.get_decode_parallel_rank",
                return_value=0,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_world_size",
                return_value=2,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_rank",
                return_value=0,
            ),
        ):
            decoder = QwenImageDecoder3d(
                dim=4,
                z_dim=2,
                dim_mult=(1, 1),
                num_res_blocks=1,
                attn_scales=(),
                temperal_upsample=(False,),
                input_channels=3,
                use_parallel_decode=True,
            )

        self.assertTrue(
            any(isinstance(m, SpatialParallelCausalConv3d) for m in decoder.modules())
        )
        self.assertTrue(
            any(
                isinstance(m, QwenImageAttentionBlock) and m.spatial_parallel
                for m in decoder.modules()
            )
        )

    def test_wan_decoder_uses_spatial_parallel_components(self):
        with (
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.wanvae.dist.is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.wanvae.get_decode_parallel_world_size",
                return_value=2,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.wanvae.get_decode_parallel_rank",
                return_value=0,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_world_size",
                return_value=2,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_rank",
                return_value=0,
            ),
        ):
            decoder = WanDecoder3d(
                dim=4,
                z_dim=2,
                dim_mult=(1, 1),
                num_res_blocks=1,
                attn_scales=(),
                temperal_upsample=(False,),
                out_channels=3,
                use_parallel_decode=True,
            )

        self.assertTrue(
            any(isinstance(m, SpatialParallelCausalConv3d) for m in decoder.modules())
        )
        self.assertTrue(
            any(isinstance(m, WanDistAttentionBlock) for m in decoder.modules())
        )

    def test_diffusers_2d_decoder_uses_spatial_parallel_components(self):
        config = StableDiffusion3VAEConfig()
        config.use_parallel_decode = True
        config.parallel_decode_mode = "spatial_shard"
        config.arch_config.latent_channels = 2
        config.arch_config.block_out_channels = (4, 4)
        config.arch_config.down_block_types = (
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        )
        config.arch_config.up_block_types = (
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        )
        config.arch_config.layers_per_block = 1
        config.arch_config.norm_num_groups = 1
        config.arch_config.sample_size = 8

        with (
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.common.dist.is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.common.model_parallel_is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.common.get_decode_parallel_group_coordinator",
                return_value=SimpleNamespace(),
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.common.get_decode_parallel_world_size",
                return_value=2,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_world_size",
                return_value=2,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_rank",
                return_value=0,
            ),
        ):
            vae = AutoencoderKL(config)

        self.assertTrue(vae._spatial_parallel_decode_enabled)
        self.assertTrue(
            any(isinstance(m, SpatialParallelConv2d) for m in vae.decoder.modules())
        )

    def test_ltx_decoder_uses_spatial_parallel_conv3d(self):
        with (
            patch(
                "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_world_size",
                return_value=2,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_rank",
                return_value=0,
            ),
        ):
            decoder = LTX2VideoDecoder3d(
                in_channels=4,
                out_channels=3,
                block_out_channels=(8,),
                spatio_temporal_scaling=(False,),
                layers_per_block=(1, 1),
                patch_size=1,
                upsample_residual=(False,),
                upsample_factor=(1,),
                spatial_padding_mode="reflect",
            )
            _enable_ltx_decoder_spatial_parallel(decoder)

        causal_convs = [
            m for m in decoder.modules() if isinstance(m, LTX2VideoCausalConv3d)
        ]
        self.assertGreater(len(causal_convs), 0)
        self.assertTrue(
            all(isinstance(m.conv, SpatialParallelConv3d) for m in causal_convs)
        )

    def test_hunyuan_decoder_uses_spatial_parallel_components(self):
        with (
            patch(
                "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_world_size",
                return_value=2,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_rank",
                return_value=0,
            ),
        ):
            decoder = HunyuanVideoDecoder3D(
                in_channels=4,
                out_channels=3,
                up_block_types=("HunyuanVideoUpBlock3D", "HunyuanVideoUpBlock3D"),
                block_out_channels=(4, 8),
                layers_per_block=1,
                norm_num_groups=1,
                mid_block_add_attention=True,
                time_compression_ratio=4,
                spatial_compression_ratio=2,
            )
            _enable_hunyuan_decoder_spatial_parallel(decoder)

        self.assertTrue(decoder.spatial_parallel)
        self.assertTrue(
            any(
                isinstance(m, HunyuanVideoMidBlock3D) and m.spatial_parallel
                for m in decoder.modules()
            )
        )
        self.assertTrue(
            any(isinstance(m, SpatialParallelConv3d) for m in decoder.modules())
        )
        shortcut_convs = [
            m.conv_shortcut.conv
            for m in decoder.modules()
            if isinstance(m, HunyuanVideoResnetBlockCausal3D)
            and m.conv_shortcut is not None
        ]
        self.assertTrue(shortcut_convs)
        self.assertTrue(
            all(not isinstance(conv, SpatialParallelConv3d) for conv in shortcut_convs)
        )


if __name__ == "__main__":
    unittest.main()
