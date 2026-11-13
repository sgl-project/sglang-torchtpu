# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's MOVA patches (sglang_tpu.multimodal_gen.mova) and
the MOVA denoising stage they run under."""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.models.bridges.mova_dual_tower import (
    MOVADualTowerArchConfig,
    MOVADualTowerConfig,
)
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.distributed import parallel_state, sp_shard_utils
from sglang.multimodal_gen.runtime.distributed import utils as dist_utils
from sglang.multimodal_gen.runtime.layers.attention import layer as attention_layer
from sglang.multimodal_gen.runtime.layers.rotary_embedding import base as rope_base
from sglang.multimodal_gen.runtime.layers.rotary_embedding import utils as rope_utils
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.bridges import mova_dual_tower as bridges
from sglang.multimodal_gen.runtime.models.dits import mova_audio_dit as audio
from sglang.multimodal_gen.runtime.models.dits import mova_video_dit as video
from sglang.multimodal_gen.runtime.models.schedulers.flow_match_pair import (
    FlowMatchPairScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    mova as stage,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import mova, worker

# Nightly: the stage-a unit suite is at its time budget.
register_tpu_ci(est_time=15, suite="nightly-tpu-4-v7", nightly=True)


@contextlib.contextmanager
def _installed():
    with contextlib.ExitStack() as stack:
        for owner, name in (
            (rope_utils, "_apply_rotary_emb_complex"),
            (rope_base, "_apply_rotary_emb_complex"),
            (video.WanModel, "_init_freqs"),
            (audio.WanAudioModel, "_init_freqs"),
        ):
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        stack.enter_context(patch.object(mova, "_DEVICE_TYPES", ("cpu",)))
        mova.install()
        mova.install()  # idempotent
        yield


def _video_freqs(freqs, t, h, w):
    """The stage's per-token table: [t * h * w, 1, n] from the three axes."""
    return torch.cat(
        [
            freqs[0][:t].view(t, 1, 1, -1).expand(t, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(t, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(t, h, w, -1),
        ],
        dim=-1,
    ).reshape(t * h * w, 1, -1)


class TestMovaRope(CustomTestCase):
    def test_real_tables_rotate_as_the_complex_ones(self):
        head_dim, (t, h, w) = 128, (2, 3, 4)
        model = SimpleNamespace(freqs=None, dim=head_dim * 2, num_heads=2)
        complex_freqs = video.precompute_freqs_cis_3d(head_dim)
        gen = torch.Generator().manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            x = torch.randn(1, t * h * w, 2, head_dim, generator=gen).to(dtype)
            want = rope_utils._apply_rotary_emb_complex(
                x, _video_freqs(complex_freqs, t, h, w), dtype=torch.float64
            )
            with _installed():
                with patch.object(
                    video, "precompute_freqs_cis_3d", lambda d: complex_freqs
                ):
                    video.WanModel._init_freqs(model)
                self.assertFalse(any(f.is_complex() for f in model.freqs))
                got = rope_base._apply_rotary_emb_complex(
                    x, _video_freqs(model.freqs, t, h, w), dtype=torch.float64
                )
            model.freqs = None
            if dtype == torch.float32:
                torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
            else:
                # One rounding from float32 against the stock's from float64.
                torch.testing.assert_close(
                    got.float(), want.float(), atol=2e-2, rtol=1e-2
                )
                self.assertLess((got != want).float().mean().item(), 1e-3)

    def test_complex_tables_keep_the_stock_path(self):
        x = torch.randn(1, 4, 2, 8)
        freqs = video.precompute_freqs_cis(8, 4).unsqueeze(1)
        with _installed():
            got = rope_base._apply_rotary_emb_complex(x, freqs, dtype=torch.float64)
        want = mova._STOCK_COMPLEX_ROPE(x, freqs, dtype=torch.float64)
        self.assertTrue(torch.equal(got, want))


class TestMovaRegionalCompile(CustomTestCase):
    def test_regional_compile_compiles_the_blocks(self):
        class Dit(torch.nn.Module):
            _compile_conditions = video.WanModel._compile_conditions
            _compile_regions_spmd_safe = False

            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

        dit = Dit()
        denoise = stage.MOVADenoisingStage.__new__(stage.MOVADenoisingStage)
        args = SimpleNamespace(enable_torch_compile=True, regional_compile=True)
        denoise._maybe_enable_torch_compile(dit, args)
        self.assertIsNotNone(dit.blocks[0]._compiled_call_impl)
        self.assertIsNone(dit._compiled_call_impl)


class TestMovaDenoisingLoop(CustomTestCase):
    """forward combines the guidance branches and steps the scheduler."""

    SCALE = 5.0

    def _denoise(self, do_cfg, cfg_parallel=False, cfg_rank=0):
        gen = torch.Generator().manual_seed(0)
        video0 = torch.randn(1, 4, 2, 3, 3, generator=gen)
        audio0 = torch.randn(1, 4, 7, generator=gen)
        prompt, negative = torch.ones(1, 5, 8), torch.zeros(1, 5, 8)
        preds = {
            branch: (
                torch.randn(video0.shape, generator=gen),
                torch.randn(audio0.shape, generator=gen),
            )
            for branch in ("pos", "neg")
        }
        predicted, steps = [], []

        def predict(dit, latents, audio_latents, y, context, *args):
            predicted.append("pos" if context is prompt else "neg")
            return preds[predicted[-1]]

        def step_from_to(pred, t, next_t, latents):
            steps.append((t.item(), None if next_t is None else next_t.item()))
            return latents + pred

        def all_reduce(partial):
            # Adds the other CFG rank's share of the guided prediction.
            i = 0 if partial.shape == video0.shape else 1
            if cfg_rank == 0:
                return partial + (1 - self.SCALE) * preds["neg"][i]
            return partial + self.SCALE * preds["pos"][i]

        denoise = stage.MOVADenoisingStage.__new__(stage.MOVADenoisingStage)
        denoise.video_dit, denoise.video_dit_2 = SimpleNamespace(), None
        denoise._torch_compiled = True
        denoise._component_residency_manager = None
        denoise._predict = predict
        batch = SimpleNamespace(
            paired_timesteps=torch.tensor([[1000.0, 900.0], [500.0, 400.0]]),
            scheduler=SimpleNamespace(step_from_to=step_from_to),
            y=None,
            image_latent=None,
            is_warmup=True,
            latents=video0,
            audio_latents=audio0,
            prompt_embeds=[prompt],
            negative_prompt_embeds=[negative],
            fps=24,
            do_classifier_free_guidance=do_cfg,
            guidance_scale=self.SCALE,
            guidance_rescale=0.0,
        )
        server_args = SimpleNamespace(
            enable_torch_compile=False,
            enable_cfg_parallel=cfg_parallel,
            pipeline_config=SimpleNamespace(boundary_ratio=None),
        )
        with (
            patch.object(
                stage, "get_classifier_free_guidance_rank", return_value=cfg_rank
            ),
            patch.object(stage, "cfg_model_parallel_all_reduce", all_reduce),
            patch.object(stage, "get_local_torch_device", return_value="cpu"),
        ):
            denoise.forward(batch, server_args)

        self.assertEqual(
            steps, [(1000.0, 500.0), (900.0, 400.0), (500.0, None), (400.0, None)]
        )
        for i, (got, start) in enumerate(
            ((batch.latents, video0), (batch.audio_latents, audio0))
        ):
            pos, neg = preds["pos"][i], preds["neg"][i]
            pred = neg + self.SCALE * (pos - neg) if do_cfg else pos
            torch.testing.assert_close(got, start + 2 * pred)
        return predicted

    def test_without_cfg(self):
        self.assertEqual(self._denoise(do_cfg=False), ["pos", "pos"])

    def test_serial_cfg(self):
        self.assertEqual(self._denoise(do_cfg=True), ["pos", "neg"] * 2)

    def test_cfg_parallel(self):
        for cfg_rank, branch in ((0, "pos"), (1, "neg")):
            with self.subTest(cfg_rank=cfg_rank):
                predicted = self._denoise(True, cfg_parallel=True, cfg_rank=cfg_rank)
                self.assertEqual(predicted, [branch] * 2)

    def test_cfg_parallel_broadcast_keeps_each_ranks_scheduler(self):
        # The CFG-parallel executor pickles the batch minus these fields.
        batch = SimpleNamespace(num_inference_steps=4)
        timesteps = stage.MOVATimestepPreparationStage.__new__(
            stage.MOVATimestepPreparationStage
        )
        timesteps.scheduler = FlowMatchPairScheduler()
        timesteps.forward(batch, None)
        denoise = stage.MOVADenoisingStage.__new__(stage.MOVADenoisingStage)
        local = denoise.cfg_parallel_local_batch_fields(batch, None)
        for name in local:
            setattr(batch, name, None)
        dist_utils._dumps(batch)


class _Stop(Exception):
    pass


class _TimeEmbedding(torch.nn.Linear):
    fc_in = property(lambda self: self)


def _tiny_dit(freqs, patchify):
    """The parts of a MOVA DiT that inference_single_step reads before the blocks."""
    return SimpleNamespace(
        dim=8,
        freq_dim=8,
        freqs=freqs,
        _init_freqs=lambda: None,
        time_embedding=_TimeEmbedding(8, 8),
        time_projection=lambda t: (t.repeat(1, 6), None),
        text_embedding=lambda context: context,
        patchify=patchify,
    )


class TestMovaBlockInputsAreRankInvariant(CustomTestCase):
    def test_every_rank_feeds_blocks_offset_zero_shards(self):
        # 2 x 3 x 3 = 18 video tokens and 7 audio tokens at SP 4: 5 and 2 per
        # rank, the last rank's shards end in pad rows.
        world_size, (t, h, w), f = 4, (2, 3, 3), 7
        freqs = video.precompute_freqs_cis_3d(12)
        video_dit = _tiny_dit(
            freqs, lambda x: (x.flatten(2).transpose(1, 2), x.shape[2:])
        )
        audio_dit = _tiny_dit(freqs, lambda x, _: (x.transpose(1, 2), x.shape[2:]))
        latents, audio_latents = torch.randn(1, 4, t, h, w), torch.randn(1, 4, f)
        full = {
            "visual_x": latents.flatten(2).transpose(1, 2),
            "audio_x": audio_latents.transpose(1, 2),
            "visual_freqs": _video_freqs(freqs, t, h, w).unsqueeze(0),
            "audio_freqs": torch.cat([x[:f] for x in freqs], -1).view(1, f, 1, -1),
        }
        layouts = []
        for rank in range(world_size):
            denoise = stage.MOVADenoisingStage.__new__(stage.MOVADenoisingStage)
            denoise.audio_dit = audio_dit
            fed = {}

            def record(**kwargs):
                fed.update(kwargs)
                raise _Stop

            denoise.forward_dual_tower_dit = record
            with (
                patch.object(
                    sp_shard_utils, "get_sp_world_size", return_value=world_size
                ),
                patch.object(sp_shard_utils, "get_sp_parallel_rank", return_value=rank),
                self.assertRaises(_Stop),
            ):
                denoise.inference_single_step(
                    video_dit,
                    latents,
                    audio_latents,
                    None,
                    torch.randn(1, 5, 8),
                    torch.tensor([900.0]),
                    torch.tensor([800.0]),
                    24,
                )
            layout = []
            for name, tensor in full.items():
                # Sequence on dim 1, zero-padded at its tail.
                local_len = -(-tensor.shape[1] // world_size)
                pad = tensor.new_zeros(
                    1, local_len * world_size - tensor.shape[1], *tensor.shape[2:]
                )
                padded = torch.cat([tensor, pad], dim=1)
                want = padded[:, rank * local_len : (rank + 1) * local_len]
                got = fed[name] if name.endswith("_x") else fed[name].unsqueeze(0)
                msg = f"{name} on rank {rank}"
                self.assertTrue(torch.equal(got, want), msg)
                self.assertEqual(fed[name].storage_offset(), 0, msg)
                self.assertTrue(fed[name].is_contiguous(), msg)
                layout.append((fed[name].shape, fed[name].stride()))
            layouts.append(layout)
        for rank in range(world_size):
            self.assertEqual(layouts[rank], layouts[0], f"rank {rank}")


def _tiny_bridge():
    arch = MOVADualTowerArchConfig(
        visual_layers=2,
        audio_layers=2,
        visual_hidden_dim=64,
        audio_hidden_dim=32,
        head_dim=16,
    )
    bridge = bridges.DualTowerConditionalBridge(
        config=MOVADualTowerConfig(arch_config=arch), hf_config={}
    ).eval()
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in bridge.parameters():
            param.copy_(torch.randn(param.shape, generator=gen) * 0.3)
    return bridge


class TestMovaBridgeUnderSP(CustomTestCase):
    """Under SP the audio-to-video keys are every rank's real audio rows."""

    # 8 video tokens; 5 audio tokens, which end in a pad row at SP 2.
    GRID, AUDIO_LEN = (2, 2, 2), 5

    def setUp(self):
        self._previous_args = server_args_module._global_server_args
        server_args_module.set_global_server_args(_make_unit_server_args())
        if not parallel_state.model_parallel_is_initialized():
            ensure_distributed_env_defaults()
            with patch.object(worker, "install_runtime_patches"):
                parallel_state.maybe_init_distributed_environment_and_model_parallel(
                    tp_size=1, sp_size=1
                )

    def tearDown(self):
        server_args_module.set_global_server_args(self._previous_args)

    def test_replicated_audio_attends_without_sp(self):
        bridge = _tiny_bridge()
        gen = torch.Generator().manual_seed(1)
        visual = torch.randn(1, 8, 64, generator=gen)
        audio_x = torch.randn(1, self.AUDIO_LEN, 32, generator=gen)
        freqs = bridge.build_aligned_freqs(
            24, self.GRID, self.AUDIO_LEN, "cpu", torch.float32
        )
        stock_forward = attention_layer.USPAttention.forward
        skips = []

        def forward(attn, *args, skip_sequence_parallel_override=False, **kwargs):
            skips.append(skip_sequence_parallel_override)
            return stock_forward(
                attn,
                *args,
                skip_sequence_parallel_override=skip_sequence_parallel_override,
                **kwargs,
            )

        with set_forward_context(current_timestep=0, attn_metadata=None):
            with torch.no_grad():
                want, _ = bridge(0, visual, audio_x, x_freqs=freqs[0], y_freqs=freqs[1])
                with patch.object(attention_layer.USPAttention, "forward", forward):
                    # Each SP rank's video rows, with the whole real audio.
                    for rows in (slice(0, 4), slice(4, 8)):
                        got = bridge.apply_conditional_control(
                            0,
                            "a2v",
                            visual[:, rows],
                            audio_x,
                            x_freqs=tuple(f[:, rows] for f in freqs[0]),
                            y_freqs=freqs[1],
                            condition_replicated=True,
                        )
                        torch.testing.assert_close(got, want[:, rows])
        self.assertEqual(skips, [True, True])

    def test_stage_passes_every_ranks_real_audio(self):
        audio_x = torch.randn(1, self.AUDIO_LEN, 32)
        calls = []

        class Bridge:
            build_aligned_freqs = _tiny_bridge().build_aligned_freqs

            def should_interact(self, layer_idx, direction):
                return True

            def __call__(self, layer_idx, visual_x, audio_x, **kwargs):
                calls.append(kwargs)
                return visual_x, audio_x

        identity = [lambda x, *args, **kwargs: x] * 2
        denoise = stage.MOVADenoisingStage.__new__(stage.MOVADenoisingStage)
        denoise.dual_tower_bridge = Bridge()
        denoise.audio_dit = SimpleNamespace(blocks=identity)

        def run(visual_len):
            return denoise.forward_dual_tower_dit(
                visual_dit=SimpleNamespace(blocks=identity),
                visual_x=torch.randn(1, 4, 64),
                audio_x=audio_x[:, 3:],
                visual_context=None,
                audio_context=None,
                visual_t_mod=None,
                audio_t_mod=None,
                visual_freqs=None,
                audio_freqs=None,
                grid_size=self.GRID,
                video_fps=24,
                full_visual_seq_len=visual_len,
                full_audio_seq_len=self.AUDIO_LEN,
            )

        with (
            patch.object(stage, "get_sp_world_size", return_value=2),
            patch.object(sp_shard_utils, "get_sp_world_size", return_value=2),
            patch.object(sp_shard_utils, "get_sp_parallel_rank", return_value=1),
            # The gathered audio: every rank's rows less the pad row.
            patch.object(stage, "gather_seq", lambda x, n: audio_x[:, :n]),
        ):
            run(8)
            with self.assertRaises(NotImplementedError):
                run(7)
        full = Bridge.build_aligned_freqs(24, self.GRID, self.AUDIO_LEN)
        self.assertEqual(len(calls), 2)
        for kwargs in calls:
            self.assertTrue(torch.equal(kwargs["replicated_audio"], audio_x))
            for got, want in zip(kwargs["replicated_audio_freqs"], full[1]):
                self.assertTrue(torch.equal(got, want))
            # The sharded tables stay with the sharded rows (rank 1 of 2).
            self.assertTrue(torch.equal(kwargs["x_freqs"][0], full[0][0][:, 4:]))


if __name__ == "__main__":
    import unittest

    unittest.main()
