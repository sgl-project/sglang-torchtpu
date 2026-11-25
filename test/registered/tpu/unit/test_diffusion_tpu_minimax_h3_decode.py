# SPDX-License-Identifier: Apache-2.0
"""CPU tests for sglang_tpu's MiniMax H3 decode-stage patches (minimax_h3_decode).

Ranks run as threads joined by in-process collectives, each with its own copy
of the stage, as each worker process holds its own.
"""

import contextlib
import copy
import filecmp
import functools
import os
import re
import sys
import tempfile
import threading
import types
import unittest
import unittest.mock
from types import SimpleNamespace
from unittest.mock import patch

# Host-CPU tests: in the TPU image the diffusion platform and JAX would
# otherwise select the TPU.
os.environ.setdefault("SGLANG_DIFFUSION_PLATFORM_OVERRIDE", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.task_type import DataType
from sglang.multimodal_gen.runtime import distributed as sgl_distributed
from sglang.multimodal_gen.runtime.entrypoints import utils as entry_utils
from sglang.multimodal_gen.runtime.managers import gpu_worker
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency_strategies import (
    ComponentOffloadStrategy,
    ResidentStrategy,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3VideoVAE
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_audio_vae import (
    DacAudioVAE,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import klvae
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.klvae import (
    AutoencoderKLLegacy,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (
    decoding,
)
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    ActiveTargetCompiledCallable,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(
    est_time=400,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "diffusion"],
)

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)
# sglang_tpu from the checkout where it is not installed.
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from sglang_tpu.envs import envs as tpu_envs  # noqa: E402
from sglang_tpu.multimodal_gen import minimax_h3_decode as hd  # noqa: E402

hd.install()


def _tiny_video_vae(
    num_layers: int = 1, dim_head: int = 16, heads: int = 2
) -> MiniMaxH3VideoVAE:
    """The H3 video VAE's temporal/tiling recipe on a small ViT decoder
    (32 wide by default; dim_head=64 gives the real decoder's head width)."""
    torch.manual_seed(0)
    vae = AutoencoderKLLegacy(
        in_channels=3,
        out_ch=3,
        ch=32,
        embed_dim=24,
        z_channels=24,
        use_3d_conv=True,
        ch_mult=[1, 1, 1, 1, 1, 1],
        space_down=[2, 2, 2, 2, 1, 1],
        space_up=[1, 2, 2, 2, 2, 1],
        time_down=[1, 2, 2, 1, 1, 1],
        padding_mode="reflect",
        use_t_isolated_gn=True,
        causal_encoder=True,
        causal_decoder=False,
        use_vit_decoder=True,
        vit_decoder_kwargs=dict(
            dim_head=dim_head,
            ffn_activation_fn="silu",
            ffn_use_gated=True,
            heads=heads,
            norm_affine=True,
            norm_type="rms_norm",
            num_layers=num_layers,
            qk_norm_affine=False,
            qk_norm_type="rms_norm",
            rope_dim_ratio=0.75,
            rope_theta=100.0,
        ),
        clip_length=17,
        token_drop=3,
        encoder_tiling=True,
        decoder_tiling=True,
        parallel_tiling=True,
        tile_size=256,
        tile_overlap_min=64,
    ).eval()
    with torch.no_grad():
        for block in vae.decoder.transformer_blocks:
            block.scale1.fill_(0.3)
            block.scale2.fill_(0.3)
    # The decode stage's video VAE class (whose decode_temporal is patched).
    vae.__class__ = MiniMaxH3VideoVAE
    return vae


def _tiny_audio_vae() -> DacAudioVAE:
    torch.manual_seed(1)
    return DacAudioVAE(
        encoder_dim=8,
        encoder_rates=[2, 4, 4, 5, 5],
        latent_dim=64,
        decoder_dim=128,
        decoder_rates=[5, 5, 2, 2, 2, 2, 2],
        sample_rate=32000,
        vae_latent_channels=32,
        attn_proj=True,
        decoder_type="bigvgan",
    ).eval()


class _World:
    """Threads as ranks of one process group, with blocking collectives.

    A failed rank breaks only the exchanges that have not completed, so a rank
    that raises after the last collective leaves its peers' results intact.
    """

    def __init__(self, world: int):
        self.world = world
        self.local = threading.local()
        self.cond = threading.Condition()
        self.generation = 0
        self.arrived = 0
        self.failed = False
        self.slots: list = [None] * world
        self.results: dict[int, list] = {}
        self.group = _Group(self)

    @property
    def rank(self) -> int:
        return self.local.rank

    def exchange(self, value):
        with self.cond:
            if self.failed:
                raise RuntimeError("a peer rank failed")
            generation = self.generation
            self.slots[self.rank] = value
            self.arrived += 1
            if self.arrived == self.world:
                self.results[generation] = list(self.slots)
                self.slots = [None] * self.world
                self.arrived = 0
                self.generation += 1
                self.cond.notify_all()
            while self.generation == generation:
                if self.failed:
                    raise RuntimeError("a peer rank failed")
                self.cond.wait()
            return self.results[generation]

    def fail(self) -> None:
        with self.cond:
            self.failed = True
            self.cond.notify_all()

    def run(self, fn):
        outputs: list = [None] * self.world
        errors: list = [None] * self.world

        def body(rank):
            self.local.rank = rank
            torch.set_num_threads(1)
            try:
                outputs[rank] = fn(rank)
            except BaseException as exc:  # surfaced to the caller
                errors[rank] = exc
                self.fail()

        threads = [threading.Thread(target=body, args=(r,)) for r in range(self.world)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return outputs, errors

    @contextlib.contextmanager
    def installed(self):
        world = self
        with contextlib.ExitStack() as stack:
            for module, name, value in (
                (sgl_distributed, "model_parallel_is_initialized", lambda: True),
                (sgl_distributed, "get_replica_group", lambda: world.group),
                (
                    sgl_distributed,
                    "get_decode_parallel_group_coordinator",
                    lambda: world.group,
                ),
                (sgl_distributed, "get_decode_parallel_rank", lambda: world.rank),
                (decoding, "model_parallel_is_initialized", lambda: True),
                (decoding, "get_replica_group", lambda: world.group),
                (klvae, "get_tile_parallel_state", lambda: (world.rank, world.world)),
                (
                    klvae,
                    "get_decode_parallel_group_coordinator",
                    lambda: world.group,
                ),
            ):
                stack.enter_context(patch.object(module, name, value))
            yield self


class _Group:
    def __init__(self, world: _World):
        self._world = world
        self.ranks = list(range(world.world))
        self.world_size = world.world

    @property
    def rank_in_group(self) -> int:
        return self._world.rank

    def all_gather(self, tensor, dim=0, separate_tensors=False):
        parts = [part.clone() for part in self._world.exchange(tensor)]
        return parts if separate_tensors else torch.cat(parts, dim=dim)

    def broadcast_object(self, obj=None, src=0):
        return self._world.exchange(obj)[src]

    def broadcast_tensor_dict(self, tensor_dict=None, src=0):
        values = self._world.exchange(tensor_dict)[src]
        if self._world.rank == src:
            return tensor_dict
        return {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in values.items()
        }


def _server_args(video_precision: str = "bf16"):
    config = MiniMaxH3PipelineConfig()
    torch.manual_seed(2)
    config.vae_config.arch_config.latents_mean = (torch.randn(24) * 0.2).tolist()
    config.vae_config.arch_config.latents_std = (torch.rand(24) + 0.5).tolist()
    config.audio_vae_config.arch_config.latents_mean = (torch.randn(32) * 0.1).tolist()
    config.audio_vae_config.arch_config.latents_std = (torch.rand(32) + 0.5).tolist()
    return SimpleNamespace(
        pipeline_config=config,
        component_precisions={"video_vae": video_precision},
        disable_autocast=False,
        enable_torch_compile=False,
    )


def _stage(server_args, video_vae, audio_vae):
    stage = decoding.MiniMaxH3DecodingStage.__new__(decoding.MiniMaxH3DecodingStage)
    stage.server_args = server_args
    stage._component_residency_manager = None
    stage._registered_stage_name = None
    stage._profile_stage_name = None
    stage.vae = video_vae
    stage.video_vae = video_vae
    stage.audio_vae = audio_vae
    stage.component_name = "video_vae"
    stage.pipeline = None
    stage._compiled_vae_decode = ActiveTargetCompiledCallable()
    stage._compiled_audio_vae_decode = ActiveTargetCompiledCallable()
    return stage


class _Batch(SimpleNamespace):
    def output_file_path(self, num_outputs=1, output_idx=None):
        return self.path


def _batch(latents, audio_latents, path=None, **overrides):
    fields = dict(
        latents=latents,
        audio_latents=audio_latents,
        extra={},
        trajectory_timesteps=None,
        trajectory_latents=None,
        rollout_trajectory_data=None,
        metrics=None,
        request_id="req",
        is_warmup=False,
        save_output=True,
        return_file_paths_only=True,
        return_raw_frames=False,
        return_frames=False,
        data_type=DataType.VIDEO,
        enable_frame_interpolation=False,
        enable_upscaling=False,
        num_outputs_per_prompt=1,
        fps=24,
        output_compression=63,
        x264_preset="ultrafast",
        # The stage gates its VAE fast path on the request's quality tier.
        sampling_params=SimpleNamespace(quality="lossless"),
        path=path,
    )
    fields.update(overrides)
    return _Batch(**fields)


def _flags(**values):
    """Env overrides for a block, restored even if it raises (EnvField.override
    restores after its yield without a finally)."""
    stack = contextlib.ExitStack()
    for name, value in values.items():
        field = getattr(tpu_envs, name)
        stack.callback(
            _restore_flag,
            field,
            field.name in os.environ,
            os.environ.get(field.name),
            field._set_to_none,
        )
        stack.enter_context(field.override(value))
    return stack


def _restore_flag(field, present, value, set_to_none):
    if present:
        os.environ[field.name] = value
    else:
        os.environ.pop(field.name, None)
    field._set_to_none = set_to_none


_ALL_ON = dict(
    SGLANG_TPU_ENABLE_H3_VAE_FLAT_DECODE=True,
    SGLANG_TPU_ENABLE_H3_AUDIO_FIRST=True,
    SGLANG_TPU_ENABLE_H3_STREAM_SAVE=True,
    # 18 units over 4 ranks: an audio wave of 7 and two more, so clips stream across waves.
    SGLANG_TPU_OPT_H3_AUDIO_WAVE_UNITS=2,
)


def _non_main_rank():
    """The logger as on ranks 1-7: neither the main nor the local main process,
    whose info and warning lines the logger's defaults drop."""
    from sglang.multimodal_gen.runtime.utils import logging_utils

    stack = contextlib.ExitStack()
    for name in ("get_is_main_process", "get_is_local_main_process"):
        stack.enter_context(patch.object(logging_utils, name, return_value=False))
    return stack


def _messages(mock) -> list[str]:
    """The formatted messages of a patched logger method's calls."""
    return [call.args[0] % call.args[1:] for call in mock.call_args_list]


_STATUS_RE = re.compile(
    r"decode status, request (?P<request>\S+?)(?P<warmup> \(warmup\))?, decode rank "
    r"(?P<rank>\d+): (?P<units>\d+) units in decoder calls of \[(?P<calls>[^\]]*)\] "
    r"\(cap (?P<cap>\d+)\); free memory cut (?P<cuts>\d+) calls(?: \((?P<cut>[^)]*)\))?, "
    r"least free (?P<free>[^;]+); fallbacks: (?P<fallbacks>.*); peak (?P<peak>.*)$"
)


def _expand(runs: str) -> list[int]:
    sizes = []
    for part in filter(None, (p.strip() for p in runs.split(","))):
        size, _, count = part.partition("x")
        sizes += [int(size)] * int(count or 1)
    return sizes


def _status_rows(messages) -> dict[int, dict]:
    """Decode rank -> its status line's fields (one line a rank expected)."""
    rows = {}
    for message in messages:
        match = _STATUS_RE.search(message)
        if match is None:
            continue
        row = match.groupdict()
        rank = int(row["rank"])
        assert rank not in rows, f"two status lines for rank {rank}"
        row["calls"] = _expand(row["calls"])
        rows[rank] = row
    return rows


def _save_like_worker(batch, output_batch):
    """GPUWorker._save_output_paths' call, through the worker's save_outputs."""
    return gpu_worker.save_outputs(
        output_batch.output,
        batch.data_type,
        batch.fps,
        True,
        lambda idx: batch.output_file_path(len(output_batch.output), idx),
        audio=output_batch.audio,
        audio_sample_rate=output_batch.audio_sample_rate,
        output_compression=batch.output_compression,
        x264_preset=batch.x264_preset,
        enable_frame_interpolation=batch.enable_frame_interpolation,
        frame_interpolation_exp=1,
        frame_interpolation_scale=1.0,
        frame_interpolation_model_path=None,
        enable_upscaling=batch.enable_upscaling,
        upscaling_model_path=None,
        upscaling_scale=4,
    )


class TestPlanWaves(CustomTestCase):
    def _check_cover(self, waves, num_units, world):
        seen = []
        for wave in waves:
            owners = wave.owners()
            self.assertEqual(len(owners), wave.stop - wave.start)
            self.assertEqual(sum(wave.counts), wave.stop - wave.start)
            for rank in range(world):
                self.assertEqual(owners.count(rank), wave.counts[rank])
                self.assertEqual(
                    wave.units_of(rank),
                    [wave.start + i for i, owner in enumerate(owners) if owner == rank],
                )
            seen.extend(range(wave.start, wave.stop))
        self.assertEqual(seen, list(range(num_units)))

    def test_regular_waves_are_about_a_clip_dealt_in_turn(self):
        self.assertEqual(
            [(w.start, w.stop) for w in hd.plan_waves(168, 28, 8)],
            [(0, 32), (32, 64), (64, 96), (96, 128), (128, 160), (160, 168)],
        )
        # A tail shorter than one unit a rank joins the wave before it.
        self.assertEqual(
            [(w.start, w.stop) for w in hd.plan_waves(9, 3, 4)], [(0, 4), (4, 9)]
        )
        self.assertEqual(
            [(w.start, w.stop) for w in hd.plan_waves(14, 4, 4)],
            [(0, 4), (4, 8), (8, 14)],
        )
        for num_units, tiles, world in (
            (168, 28, 8),
            (196, 28, 8),
            (9, 3, 4),
            (6, 3, 2),
        ):
            waves = hd.plan_waves(num_units, tiles, world)
            self._check_cover(waves, num_units, world)
            for wave in waves:
                for rank in range(world):
                    self.assertEqual(
                        wave.units_of(rank),
                        list(range(wave.start + rank, wave.stop, world)),
                    )

    def test_1344x768x124_deals_25_units_at_most(self):
        # 1344x768x124: 7 clips x 28 tiles over 8 ranks (stock: 7 x 4 = 28).
        waves = hd.plan_waves(196, 28, 8)
        totals = [sum(w.counts[r] for w in waves) for r in range(8)]
        self.assertEqual(totals, [25, 25, 25, 25, 24, 24, 24, 24])
        self.assertEqual(len(waves), 6)

    def test_audio_wave(self):
        for audio_rank in (0, 3, 7):
            waves = hd.plan_waves(196, 28, 8, audio_rank=audio_rank, audio_wave_units=8)
            self._check_cover(waves, 196, 8)
            first = waves[0]
            self.assertEqual((first.start, first.stop), (0, 57))
            self.assertEqual(first.counts[audio_rank], 1)
            self.assertEqual(
                [c for r, c in enumerate(first.counts) if r != audio_rank], [8] * 7
            )
            # The audio rank's unit is dealt first, so it decodes before its audio.
            self.assertEqual(first.units_of(audio_rank), [audio_rank])
            totals = [sum(w.counts[r] for w in waves) for r in range(8)]
            self.assertEqual(sum(totals), 196)
            self.assertLess(
                totals[audio_rank],
                min(t for r, t in enumerate(totals) if r != audio_rank),
            )
        # A wave of at least a clip's share for the others; everything at once if larger.
        self.assertEqual(hd.plan_waves(196, 28, 8, audio_rank=0)[0].counts[1], 4)
        everything = hd.plan_waves(196, 28, 8, audio_rank=0, audio_wave_units=40)
        self.assertEqual(len(everything), 1)
        self.assertEqual(everything[0].counts, (1, 28, 28, 28, 28, 28, 28, 27))
        self._check_cover(everything, 196, 8)
        # Small: a regular tail after the audio wave keeps its own gather.
        small = hd.plan_waves(10, 3, 4, audio_rank=1, audio_wave_units=2)
        self._check_cover(small, 10, 4)
        self.assertEqual(small[0].counts, (2, 1, 2, 2))

    def test_rejects_what_cannot_be_dealt(self):
        with self.assertRaises(ValueError):
            hd.plan_waves(10, 3, 1)
        with self.assertRaises(ValueError):
            hd.plan_waves(3, 3, 4)
        with self.assertRaises(ValueError):
            hd.plan_waves(10, 3, 4, audio_rank=4)

    def test_audio_rank_units_balance_small_decode_groups(self):
        def totals(waves, world):
            return [sum(w.counts[r] for w in waves) for r in range(world)]

        # 1344x768x124 at 2 chips (4 ranks) and 1 chip (2 ranks): the audio wave
        # cannot go below a clip's share, so with one unit on the audio owner
        # another rank decodes 51 or 104 units instead of 49 or 98.
        for world, share, busiest_one in ((4, 7, 51), (2, 14, 104)):
            one = hd.plan_waves(196, 28, world, audio_rank=0, audio_wave_units=share)
            self.assertEqual(max(totals(one, world)), busiest_one)
            own = hd.plan_waves(
                196,
                28,
                world,
                audio_rank=0,
                audio_wave_units=share,
                audio_rank_units=share - 1,
            )
            self._check_cover(own, 196, world)
            self.assertEqual(totals(own, world), [196 // world] * world)
            self.assertEqual(own[0].counts[0], share - 1)
            self.assertEqual(own[0].units_of(0)[0], 0)
        # Default: the plans above; capped at the other ranks' count.
        self.assertEqual(
            hd.plan_waves(196, 28, 8, audio_rank=0, audio_wave_units=8),
            hd.plan_waves(
                196, 28, 8, audio_rank=0, audio_wave_units=8, audio_rank_units=1
            ),
        )
        capped = hd.plan_waves(
            196, 28, 8, audio_rank=2, audio_wave_units=4, audio_rank_units=9
        )
        self.assertEqual(capped[0].counts, (4,) * 8)
        self._check_cover(capped, 196, 8)
        # Few units: every other rank keeps one unit of the audio wave (the
        # audio owner asks for 3 of 5, gets 2).
        small = hd.plan_waves(
            5, 3, 4, audio_rank=1, audio_wave_units=3, audio_rank_units=3
        )
        self._check_cover(small, 5, 4)
        self.assertEqual(small[0].counts, (1, 2, 1, 1))
        self.assertTrue(all(c >= 1 for c in small[0].counts))


def _flat_decode(
    vae,
    z,
    world,
    *,
    autocast_dtype=None,
    audio_rank=None,
    audio_wave_units=0,
    hooks=None,
):
    """Every rank's output of run_flat_decode, ranks as threads."""
    plan = hd.plan_units(vae, z, world)
    assert plan is not None
    waves = hd.plan_waves(
        len(plan.units),
        plan.tiles_per_clip,
        world,
        audio_rank=audio_rank,
        audio_wave_units=audio_wave_units,
    )
    stock = hd._stock["decode_temporal"]
    copies = [copy.deepcopy(vae) for _ in range(world)]
    world_ = _World(world)

    def gather(tensor):
        return world_.group.all_gather(tensor, separate_tensors=True)

    def rank_decode(rank):
        with (
            torch.no_grad(),
            torch.autocast(
                "cpu",
                dtype=autocast_dtype or torch.bfloat16,
                enabled=bool(autocast_dtype),
            ),
        ):
            return hd.run_flat_decode(
                copies[rank],
                z,
                plan,
                waves,
                stock,
                rank,
                world,
                gather,
                after_first_unit=(hooks or {}).get(rank),
            )

    outputs, errors = world_.run(rank_decode)
    for error in errors:
        if error is not None:
            raise error
    return outputs


class TestFlatDecode(CustomTestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.vae = _tiny_video_vae()

    def _reference(self, z, autocast_dtype=None):
        with (
            torch.no_grad(),
            torch.autocast(
                "cpu",
                dtype=autocast_dtype or torch.bfloat16,
                enabled=bool(autocast_dtype),
            ),
        ):
            return hd._stock["decode_temporal"](self.vae, z)

    def test_bit_identical_to_stock_decode(self):
        torch.manual_seed(3)
        # 12 latent frames: 2 clips, 16x40: 1x3 tiles; 13: a padded third clip.
        for shape, worlds in (
            ((1, 24, 12, 16, 40), (2, 3, 4, 6)),
            ((1, 24, 13, 20, 24), (2, 4)),
        ):
            z = torch.randn(shape)
            reference = self._reference(z)
            for world in worlds:
                for rank, output in enumerate(_flat_decode(self.vae, z, world)):
                    self.assertTrue(
                        torch.equal(output, reference),
                        f"{shape} world={world} rank={rank}",
                    )

    def test_1344x768x124_geometry(self):
        # 1344x768x124: latent 37 x 48 x 84 gives 7 clips x 4x7 tiles = 196 units
        # of one shape (planned only: eight decoded copies do not fit a CPU test).
        z = torch.zeros(1, 24, 37, 48, 84)
        plan = hd.plan_units(self.vae, z, 8)
        self.assertEqual(
            (len(plan.clips), plan.tiles_per_clip, len(plan.units)), (7, 28, 196)
        )
        self.assertEqual(tuple(plan.units[0].shape), (1, 24, 7, 16, 16))
        self.assertEqual(plan.offsets, [28 * i for i in range(7)])
        _, _, frames = self.vae._decode_temporal_output_frame_plan(
            plan.padded, None, None, plan.num_chunks, plan.pad_tokens
        )
        self.assertEqual(frames, 124)

    def test_seven_clips_with_an_audio_wave(self):
        # The temporal structure of 124 frames (7 clips) on a 2-tile canvas.
        torch.manual_seed(4)
        z = torch.randn(1, 24, 37, 16, 24)
        reference = self._reference(z)
        self.assertEqual(reference.shape[2], 124)
        for world, audio_rank in ((4, 0), (5, 3)):
            calls = []
            outputs = _flat_decode(
                self.vae,
                z,
                world,
                audio_rank=audio_rank,
                audio_wave_units=2,
                hooks={audio_rank: lambda: calls.append("audio")},
            )
            self.assertEqual(calls, ["audio"])
            for rank, output in enumerate(outputs):
                self.assertTrue(
                    torch.equal(output, reference), f"world={world} rank={rank}"
                )
            del outputs

    def test_bit_identical_under_autocast_and_list_assembly(self):
        torch.manual_seed(5)
        z = torch.randn(1, 24, 12, 16, 40)
        for stream in ("1", "0"):
            with patch.dict(
                os.environ, {"MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT": stream}
            ):
                reference = self._reference(z, torch.bfloat16)
                outputs = _flat_decode(
                    self.vae,
                    z,
                    4,
                    autocast_dtype=torch.bfloat16,
                    audio_rank=2,
                    audio_wave_units=2,
                )
            for output in outputs:
                self.assertTrue(torch.equal(output, reference), stream)

    def test_audio_runs_after_exactly_one_unit_of_the_audio_rank(self):
        torch.manual_seed(6)
        z = torch.randn(1, 24, 12, 16, 40)
        plan = hd.plan_units(self.vae, z, 3)
        waves = hd.plan_waves(
            len(plan.units), plan.tiles_per_clip, 3, audio_rank=1, audio_wave_units=2
        )
        vae = copy.deepcopy(self.vae)
        events = []
        stock_decode = type(vae).decode

        def counting_decode(self_, unit):
            events.append("unit")
            return stock_decode(self_, unit)

        def gather(tensor):
            # Rank 1 of 3; the others echo it.
            return [tensor, tensor, tensor]

        with patch.object(type(vae), "decode", counting_decode), torch.no_grad():
            hd.run_flat_decode(
                vae,
                z,
                plan,
                waves,
                hd._stock["decode_temporal"],
                1,
                3,
                gather,
                after_first_unit=lambda: events.append("audio"),
            )
        self.assertEqual(events[:2], ["unit", "audio"])
        self.assertEqual(events.count("audio"), 1)
        self.assertNotIn("_adaptive_decode", vars(vae))

    def test_stock_path_where_the_flat_schedule_does_not_apply(self):
        z = torch.randn(1, 24, 12, 16, 40)
        self.assertIsNone(hd.plan_units(self.vae, z, 1))
        self.assertIsNone(hd.plan_units(self.vae, z, 7))  # 6 units < 7 ranks
        for attr in ("isolated_first_frame", "isolated_last_frame"):
            with patch.object(self.vae, attr, True):
                self.assertIsNone(hd.plan_units(self.vae, z, 2), attr)
        for caps in ((0, 1), (2, 1), (1, 2)):  # the stock decode's own batching
            with patch.object(self.vae, "_batch_caps", lambda caps=caps: caps):
                self.assertIsNone(hd.plan_units(self.vae, z, 2), caps)
        with patch.object(self.vae, "parallel_tiling", False):
            self.assertIsNone(hd.plan_units(self.vae, z, 2))

    def test_wrapper_runs_stock_decode_when_disabled_or_alone(self):
        z = torch.randn(1, 24, 12, 16, 40)
        reference = self._reference(z)
        with (
            patch.object(hd, "plan_units", side_effect=AssertionError("must not plan")),
            torch.no_grad(),
        ):
            self.assertTrue(torch.equal(self.vae.decode_temporal(z), reference))
        with _flags(SGLANG_TPU_ENABLE_H3_VAE_FLAT_DECODE=True), torch.no_grad():
            # world 1 (no process group): the stock path.
            self.assertTrue(torch.equal(self.vae.decode_temporal(z), reference))

    def test_on_frames_runs_the_stock_decode(self):
        z = torch.randn(1, 24, 12, 16, 40)
        frames = []
        with (
            _flags(SGLANG_TPU_ENABLE_H3_VAE_FLAT_DECODE=True),
            patch.object(hd, "plan_units", side_effect=AssertionError("must not plan")),
            torch.no_grad(),
        ):
            out = self.vae.decode_temporal(z, on_frames=frames.append)
        self.assertTrue(torch.equal(out, self._reference(z)))
        self.assertTrue(torch.equal(torch.cat(frames, dim=2), out))

    def test_stage_encode_while_decoding_stays_off(self):
        # The flat decode's streamed save or the worker's save writes the mp4.
        self.assertIsNone(decoding.streamed_video_output_path(None, None, 1))


class TestDecodeStage(CustomTestCase):
    """The patched stage forward against the stock one, 4 ranks as threads."""

    world = 4

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.server_args = _server_args()
        cls.video_vae = _tiny_video_vae()
        cls.audio_vae = _tiny_audio_vae()
        torch.manual_seed(7)
        # 2 clips x 3x3 tiles (32x40 latent, 512x640 px), 39 frames; 9 audio latent frames.
        cls.latents = torch.randn(1, 24, 12, 32, 40)
        cls.audio_latents = torch.randn(2, 32, 9)
        cls.tmp = tempfile.mkdtemp(prefix="h3dec_")
        cls.stock = cls._run_world(False, "stock.mp4")

    @classmethod
    def _run_world(
        cls,
        flags_on,
        name,
        *,
        batch_overrides=None,
        stages=None,
        after=None,
        flags=None,
        latents=None,
    ):
        path = os.path.join(cls.tmp, name)
        world = _World(cls.world)
        stages = stages or [
            _stage(
                cls.server_args,
                copy.deepcopy(cls.video_vae),
                copy.deepcopy(cls.audio_vae),
            )
            for _ in range(cls.world)
        ]
        batch_kwargs = dict(batch_overrides or {})
        latents = cls.latents if latents is None else latents

        def rank_forward(rank):
            batch = _batch(latents, cls.audio_latents, path, **batch_kwargs)
            output = stages[rank].forward(batch, cls.server_args)
            saved = None
            if rank == 0:
                saved = _save_like_worker(batch, output)
            if after is not None:
                after(rank, output)
            return output, saved

        with (
            world.installed(),
            _flags(**(flags if flags is not None else (_ALL_ON if flags_on else {}))),
            patch.object(
                entry_utils, "_save_video_ffmpeg", wraps=entry_utils._save_video_ffmpeg
            ) as encode,
        ):
            outputs, errors = world.run(rank_forward)
        return SimpleNamespace(
            path=path,
            outputs=outputs,
            errors=errors,
            stages=stages,
            encodes=encode.call_count,
        )

    def _assert_same_as_stock(self, run):
        for error in run.errors:
            if error is not None:
                raise error
        for rank in range(self.world):
            stock, ours = self.stock.outputs[rank][0], run.outputs[rank][0]
            self.assertTrue(
                torch.equal(stock.output, ours.output), f"frames rank {rank}"
            )
            self.assertTrue(torch.equal(stock.audio, ours.audio), f"audio rank {rank}")
            self.assertEqual(stock.audio_sample_rate, ours.audio_sample_rate)
        self.assertEqual(run.outputs[0][1], [run.path])

    def test_stock_run_saves_a_video_with_audio(self):
        self.assertTrue(all(e is None for e in self.stock.errors))
        self.assertGreater(os.path.getsize(self.stock.path), 1000)
        self.assertEqual(
            tuple(self.stock.outputs[0][0].output.shape), (1, 3, 39, 512, 640)
        )

    def test_flat_audio_first_and_streamed_mp4_are_identical(self):
        audio_decodes = []
        stock_decode = DacAudioVAE.decode

        def counting(vae, z):
            audio_decodes.append(torch.is_autocast_enabled("cpu"))
            return stock_decode(vae, z)

        with patch.object(DacAudioVAE, "decode", counting):
            run = self._run_world(True, "streamed.mp4")
        self._assert_same_as_stock(run)
        # Once, on the audio owner, without autocast (inside the video decode).
        self.assertEqual(audio_decodes, [False])
        # One encode (the writer thread's), none in the worker's save.
        self.assertEqual(run.encodes, 1)
        self.assertTrue(filecmp.cmp(self.stock.path, run.path, shallow=False))
        self.assertEqual(hd._SAVES, {})

    def test_flat_and_audio_first_without_streaming(self):
        run = self._run_world(
            True,
            "nostream.mp4",
            flags=dict(
                SGLANG_TPU_ENABLE_H3_VAE_FLAT_DECODE=True,
                SGLANG_TPU_ENABLE_H3_AUDIO_FIRST=True,
            ),
        )
        self._assert_same_as_stock(run)
        self.assertTrue(filecmp.cmp(self.stock.path, run.path, shallow=False))

    def test_flat_decode_alone(self):
        run = self._run_world(
            True, "flat.mp4", flags=dict(SGLANG_TPU_ENABLE_H3_VAE_FLAT_DECODE=True)
        )
        self._assert_same_as_stock(run)

    def test_cropped_canvas_streams_the_cropped_frames(self):
        # A target canvas smaller than the decoded one: the stage crops, so must the stream.
        state = {MINIMAX_H3_DENOISE_STATE_EXTRA_KEY: {"latent_h": 30, "latent_w": 36}}
        stock = self._run_world(
            False, "crop_stock.mp4", batch_overrides={"extra": state}
        )
        ours = self._run_world(
            True, "crop_streamed.mp4", batch_overrides={"extra": state}
        )
        for rank in range(self.world):
            self.assertTrue(
                torch.equal(stock.outputs[rank][0].output, ours.outputs[rank][0].output)
            )
        self.assertEqual(tuple(ours.outputs[0][0].output.shape[-2:]), (480, 576))
        self.assertEqual(ours.encodes, 1)
        self.assertTrue(filecmp.cmp(stock.path, ours.path, shallow=False))

    def test_padded_last_clip_streams_the_kept_frames(self):
        # 13 latent frames: a third clip of padded latents, its tail frames dropped.
        torch.manual_seed(10)
        latents = torch.randn(1, 24, 13, 32, 40)
        stock = self._run_world(False, "pad_stock.mp4", latents=latents)
        ours = self._run_world(True, "pad_streamed.mp4", latents=latents)
        for rank in range(self.world):
            self.assertTrue(
                torch.equal(stock.outputs[rank][0].output, ours.outputs[rank][0].output)
            )
        self.assertEqual(ours.encodes, 1)
        self.assertTrue(filecmp.cmp(stock.path, ours.path, shallow=False))

    def test_failed_stream_saves_again_from_the_frames(self):
        with patch.object(hd, "_pixels", side_effect=RuntimeError("planted")):
            run = self._run_world(True, "fallback.mp4")
        self._assert_same_as_stock(run)
        self.assertGreaterEqual(run.encodes, 2)  # the stream's, then the stock save
        self.assertTrue(filecmp.cmp(self.stock.path, run.path, shallow=False))
        self.assertEqual(hd._SAVES, {})

    def test_writer_error_saves_again(self):
        def broken(frames):
            raise RuntimeError("planted writer failure")

        with patch.object(hd._QueuedFrames, "__iter__", broken):
            run = self._run_world(True, "writer_fail.mp4")
        self._assert_same_as_stock(run)
        self.assertTrue(filecmp.cmp(self.stock.path, run.path, shallow=False))
        self.assertEqual(hd._SAVES, {})

    def test_audio_error_reaches_every_rank(self):
        def failing(stage, latent, server_args):
            raise ValueError("planted audio failure")

        with patch.dict(hd._stock, decode_audio=failing):
            run = self._run_world(True, "audio_fail.mp4")
        self.assertIsInstance(run.errors[0], ValueError)
        for rank in range(1, self.world):
            self.assertIsInstance(run.errors[rank], RuntimeError)
            self.assertIn("planted audio failure", str(run.errors[rank]))
        self.assertFalse(os.path.exists(run.path))
        self.assertEqual(hd._SAVES, {})

    def test_failure_after_the_stream_started_discards_it(self):
        stock_post = MiniMaxH3PipelineConfig.post_decoding

        # post_decoding runs in the stream (per part) and, on the whole video,
        # after the audio broadcast, when every frame has been streamed.
        def post(config, frames, server_args):
            if frames.shape[2] == 39:
                raise RuntimeError("planted late failure")
            return stock_post(config, frames, server_args)

        with patch.object(MiniMaxH3PipelineConfig, "post_decoding", post):
            run = self._run_world(True, "late_fail.mp4")
        for rank in range(self.world):
            self.assertIsNotNone(run.errors[rank])
        self.assertFalse(os.path.exists(run.path))
        self.assertEqual(hd._SAVES, {})

    def test_audio_stays_last_with_an_offloaded_vae(self):
        manager = SimpleNamespace(
            strategy_for=lambda name, module: (
                ComponentOffloadStrategy()
                if name == "audio_vae"
                else ResidentStrategy()
            )
        )
        stage = _stage(self.server_args, self.video_vae, self.audio_vae)
        stage._component_residency_manager = manager
        world = _World(self.world)
        with world.installed():
            outputs, _ = world.run(lambda rank: hd._audio_decode_rank(stage))
        self.assertEqual(outputs, [None] * self.world)
        stage._component_residency_manager = None
        with world.installed():
            outputs, _ = world.run(lambda rank: hd._audio_decode_rank(stage))
        self.assertEqual(outputs, [0] * self.world)

    def test_warmup_and_frame_returns_do_not_stream(self):
        world = _World(1)
        world.local.rank = 0
        with world.installed():
            base = _batch(
                self.latents, self.audio_latents, os.path.join(self.tmp, "x.mp4")
            )
            self.assertIsNotNone(hd._stream_path(base))
            for override in (
                {"is_warmup": True},
                {"save_output": False},
                {"return_file_paths_only": False},
                {"return_frames": True},
                {"enable_upscaling": True},
                {"num_outputs_per_prompt": 2},
                {"extra": {"dynamic_batch_output_paths": ["a.mp4"]}},
                {"latents": torch.zeros(2, 24, 12, 32, 40)},
                {"path": os.path.join(self.tmp, "x.png")},
            ):
                fields = {
                    "latents": self.latents,
                    "audio_latents": self.audio_latents,
                    "path": base.path,
                    **override,
                }
                batch = _batch(**fields)
                self.assertIsNone(hd._stream_path(batch), override)


class TestQueuedFrames(CustomTestCase):
    def test_first_frame_reads_ahead_and_iteration_yields_every_frame(self):
        import queue

        video = torch.arange(5 * 2 * 2 * 3, dtype=torch.uint8).reshape(5, 2, 2, 3)
        chunks = queue.Queue()
        frames = hd._QueuedFrames(chunks, 5)
        chunks.put(video[:3])
        chunks.put(video[3:])
        with self.assertRaises(IndexError):
            frames[1]
        self.assertTrue(torch.equal(torch.from_numpy(frames[0]), video[0]))
        got = torch.stack([torch.from_numpy(f) for f in frames])
        self.assertTrue(torch.equal(got, video))
        self.assertEqual(frames.yielded, 5)

    def test_streamed_save_writes_the_stock_bytes(self):
        tmp = tempfile.mkdtemp(prefix="h3dec_save_")
        torch.manual_seed(8)
        video = torch.rand(3, 30, 96, 128)
        audio = torch.rand(2, 40000) * 2 - 1
        stock_path, ours_path = os.path.join(tmp, "a.mp4"), os.path.join(tmp, "b.mp4")
        entry_utils.save_outputs(
            [video],
            DataType.VIDEO,
            24,
            True,
            lambda idx: stock_path,
            audio=audio.unsqueeze(0),
            audio_sample_rate=32000,
            output_compression=63,
            x264_preset="ultrafast",
        )
        save = hd.StreamedSave(
            ours_path,
            audio=audio,
            audio_sample_rate=32000,
            fps=24,
            num_frames=30,
            output_compression=63,
            x264_preset="ultrafast",
        )
        frames = (
            (video * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 3, 0).contiguous()
        )
        for start in range(0, 30, 7):
            save.put(frames[start : start + 7])
        self.assertTrue(save.wait(120))
        self.assertTrue(filecmp.cmp(stock_path, ours_path, shallow=False))

    def test_single_iteration_length_and_abort(self):
        chunks = __import__("queue").Queue()
        frames = hd._QueuedFrames(chunks, 3)
        self.assertEqual(len(frames), 3)
        self.assertIsInstance(frames, list)
        chunks.put(torch.zeros(2, 4, 4, 3, dtype=torch.uint8))
        chunks.put(hd._ABORT)
        iterator = iter(frames)
        next(iterator)
        next(iterator)
        with self.assertRaises(RuntimeError):
            next(iterator)
        with self.assertRaises(RuntimeError):
            iter(frames).__next__()


class TestBlockCompile(CustomTestCase):
    def setUp(self):
        torch.set_num_threads(1)
        hd._compiled_vaes.clear()

    def test_decoder_block_traces_as_one_graph(self):
        import torch._dynamo as dynamo

        vae = _tiny_video_vae(num_layers=2)
        block = vae.decoder.transformer_blocks[0]
        hidden = torch.randn(1, 64 + 5, 32)
        ids = torch.zeros(1, 69, 3)
        rope = vae.decoder.pos_embed(ids)
        for autocast in (False, True):
            dynamo.reset()
            with (
                torch.no_grad(),
                torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast),
            ):
                explained = dynamo.explain(block)(hidden, rope)
            self.assertEqual(explained.graph_count, 1, explained.break_reasons)
            self.assertEqual(explained.graph_break_count, 0)

    def test_compiled_blocks_decode_and_fall_back_to_eager(self):
        torch.manual_seed(9)
        z = torch.randn(1, 24, 7, 16, 16)
        vae = _tiny_video_vae(num_layers=2)
        with torch.no_grad():
            reference = vae.decode(z)
        count = hd.compile_decoder_blocks(
            vae, {"backend": "aot_eager", "dynamic": False, "fullgraph": True}
        )
        self.assertEqual(count, 2)
        self.assertEqual(hd.compile_decoder_blocks(vae), 0)  # once a VAE
        with torch.no_grad():
            self.assertTrue(torch.equal(vae.decode(z), reference))

        def broken_backend(gm, example_inputs):
            raise RuntimeError("planted compile failure")

        torch._dynamo.reset()
        broken = _tiny_video_vae(num_layers=2)
        hd.compile_decoder_blocks(
            broken, {"backend": broken_backend, "dynamic": False, "fullgraph": True}
        )
        with torch.no_grad():
            self.assertTrue(torch.equal(broken.decode(z), reference))
        self.assertTrue(
            all(
                b._compiled_call_impl is None for b in broken.decoder.transformer_blocks
            )
        )

    def test_layerwise_offloaded_vae_is_not_compiled(self):
        vae = _tiny_video_vae()
        with patch(
            "sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload.is_layerwise_offloaded_module",
            return_value=True,
        ):
            self.assertEqual(hd.compile_decoder_blocks(vae), 0)


def _stub_decode(vae, z):
    """vae.decode's output shape from elementwise mul/add only: batch-invariant
    (bit-identical in any batch), so batched decodes must match exactly."""
    x = z[:, :3] * 0.7 + z[:, 3:6] * z[:, 6:9]
    return (
        x.repeat_interleave(4, dim=2)
        .repeat_interleave(16, dim=3)
        .repeat_interleave(16, dim=4)
    )


class _CallLog:
    """vae.decode replaced by :func:`_stub_decode`, recording each call's units
    (in all, and by decode rank); ``wrapped`` keeps the plugin's vae.decode
    wrapper (which records calls in the rank's status) around the stub."""

    def __init__(self, fail_batched: bool = False, wrapped: bool = False):
        self.sizes: list[int] = []
        self.by_rank: dict[int, list[int]] = {}
        self.fail_batched = fail_batched
        self.wrapped = wrapped
        self.lock = threading.Lock()

    def __call__(self, vae, z):
        with self.lock:
            self.sizes.append(int(z.shape[0]))
            try:  # a _World rank, or 0 without a process group
                rank = sgl_distributed.get_decode_parallel_rank()
            except Exception:
                rank = 0
            if self.fail_batched and z.shape[0] > 1:
                raise RuntimeError("planted batched failure")
            self.by_rank.setdefault(rank, []).append(int(z.shape[0]))
        return _stub_decode(vae, z)

    def patch(self):
        def decode(vae, z):
            return self(vae, z)

        if self.wrapped:
            decode = hd._routed_decode(decode)
        return patch.object(MiniMaxH3VideoVAE, "decode", decode)


class _BatchCase(CustomTestCase):
    def setUp(self):
        torch.set_num_threads(1)
        hd._batch_failed.clear()
        hd._warned.clear()

    def tearDown(self):
        hd._batch_failed.clear()
        hd._warned.clear()


class TestUnitCalls(_BatchCase):
    def test_call_sizes_are_near_equal_and_capped(self):
        self.assertEqual(hd.call_sizes(14, 4), [4, 4, 3, 3])
        self.assertEqual(hd.call_sizes(7, 4), [4, 3])
        self.assertEqual(hd.call_sizes(16, 4), [4, 4, 4, 4])
        self.assertEqual(hd.call_sizes(3, 4), [3])
        self.assertEqual(hd.call_sizes(5, 1), [1] * 5)
        self.assertEqual(hd.call_sizes(0, 4), [])
        for count in range(1, 40):
            for cap in range(1, 9):
                sizes = hd.call_sizes(count, cap)
                self.assertEqual(sum(sizes), count)
                self.assertLessEqual(max(sizes), cap)
                self.assertLessEqual(max(sizes) - min(sizes), 1)
                self.assertEqual(len(sizes), -(-count // cap))

    def _calls(self, count, cap, **kwargs):
        vae = _tiny_video_vae()
        torch.manual_seed(11)
        latents = [torch.randn(1, 24, 7, 16, 16) for _ in range(count)]
        log = _CallLog(kwargs.pop("fail_batched", False))
        with log.patch(), torch.no_grad():
            calls = list(hd._unit_calls(vae, latents, cap, **kwargs))
            singles = [_stub_decode(vae, latent) for latent in latents]
        starts = [start for start, _ in calls]
        lengths = [int(out.shape[0]) for _, out in calls]
        outputs = torch.cat([out for _, out in calls], dim=0)
        self.assertEqual(starts, [sum(lengths[:i]) for i in range(len(starts))])
        self.assertTrue(torch.equal(outputs, torch.cat(singles, dim=0)))
        return vae, latents, log.sizes

    def test_units_decode_in_order_in_capped_calls(self):
        self.assertEqual(self._calls(14, 4)[2], [4, 4, 3, 3])
        self.assertEqual(self._calls(14, 4, first_alone=True)[2], [1, 4, 3, 3, 3])
        self.assertEqual(self._calls(3, 1)[2], [1, 1, 1])
        self.assertEqual(self._calls(1, 4, first_alone=True)[2], [1])

    def test_free_memory_caps_a_call(self):
        vae = _tiny_video_vae()
        latent = torch.zeros(1, 24, 7, 16, 16)
        per_unit = hd.unit_call_bytes(vae, latent)
        reserve = hd.DECODE_BATCH_RESERVE_BYTES
        for free, sizes in (
            (None, [4, 4, 3, 3]),
            (reserve + 4 * per_unit, [4, 4, 3, 3]),
            (reserve + 2 * per_unit + per_unit // 2, [2] * 7),
            (reserve + per_unit, [1] * 14),
            (0, [1] * 14),
        ):
            hd._warned.clear()
            with (
                patch.object(hd, "free_device_bytes", return_value=free),
                patch.object(hd.logger, "warning") as warning,
            ):
                self.assertEqual(self._calls(14, 4)[2], sizes, free)
            self.assertEqual(warning.call_count, int(max(sizes) < 4), free)

    def test_unit_bytes_at_the_1344x768_unit(self):
        # The released decoder: 36 blocks, 32 heads x 64, gated FFN 2 x 8192.
        decoder = SimpleNamespace(
            num_register_tokens=4,
            config=SimpleNamespace(out_channels=3, patch_size_t=4, patch_size=16),
            transformer_blocks=[
                SimpleNamespace(
                    attn=SimpleNamespace(heads=32, attn_inner_dim=2048),
                    ff=SimpleNamespace(w1=SimpleNamespace(out_features=16384)),
                )
            ],
        )
        tokens = 7 * 16 * 16 + 5
        out_bytes = 3 * 28 * 256 * 256 * 4
        expected = 2 * 32 * tokens**2 * 4 + tokens * (26 * 2048 + 3 * 16384)
        expected += 2 * out_bytes
        vae, latent = SimpleNamespace(decoder=decoder), torch.zeros(1, 24, 7, 16, 16)
        got = hd.unit_call_bytes(vae, latent)
        self.assertEqual(got, expected)
        self.assertAlmostEqual(got / 1e9, 1.055, places=2)
        # The flash ViT attention writes no scores.
        with (
            patch.object(hd, "_flash_attention", True),
            patch.dict(hd._flash_decoders, {id(vae): True}),
        ):
            got = hd.unit_call_bytes(vae, latent)
        self.assertEqual(got, expected - 2 * 32 * tokens**2 * 4)
        self.assertAlmostEqual(got / 1e9, 0.228, places=2)

    def test_free_bytes_unknown_off_device(self):
        self.assertIsNone(hd.free_device_bytes(torch.device("cpu")))
        for info in ((123, 456), {"free": 123, "total": 456}):
            with patch.object(torch.accelerator, "get_memory_info", return_value=info):
                self.assertEqual(hd.free_device_bytes(torch.device("meta")), 123)
        with patch.object(torch.accelerator, "get_memory_info", return_value=None):
            self.assertIsNone(hd.free_device_bytes(torch.device("meta")))
        with patch.object(
            torch.accelerator, "get_memory_info", side_effect=RuntimeError("no")
        ):
            self.assertIsNone(hd.free_device_bytes(torch.device("meta")))

    def test_failed_batched_call_redoes_units_and_ends_batching(self):
        with (
            _flags(SGLANG_TPU_OPT_H3_VAE_DECODE_BATCH=4),
            patch.object(hd.logger, "warning") as warning,
        ):
            self.assertEqual(hd.decode_batch_units(), 4)
            sizes = self._calls(7, 4, fail_batched=True)[2]
            # The failed call of 4, its 4 units, then unit by unit.
            self.assertEqual(sizes, [4, 1, 1, 1, 1, 1, 1, 1])
            self.assertEqual(hd.decode_batch_units(), 1)
        self.assertEqual(warning.call_count, 1)
        self.assertIn("unit by unit", warning.call_args.args[0])

    def test_flag_values(self):
        for value, units in ((0, 1), (1, 1), (4, 4), (-3, 1)):
            with _flags(SGLANG_TPU_OPT_H3_VAE_DECODE_BATCH=value):
                self.assertEqual(hd.decode_batch_units(), units)


class TestBatchedDecode(_BatchCase):
    """Batched unit decodes in the flat and the stock schedule."""

    def test_flat_schedule_matches_stock_with_a_batch_invariant_decoder(self):
        vae = _tiny_video_vae()
        torch.manual_seed(12)
        for shape, world, audio, own in (
            ((1, 24, 12, 16, 40), 2, None, 1),
            ((1, 24, 12, 32, 40), 3, None, 1),
            ((1, 24, 13, 20, 24), 2, None, 1),
            ((1, 24, 37, 16, 24), 4, 0, 1),
            ((1, 24, 37, 16, 24), 3, 1, 3),
        ):
            z = torch.randn(shape)
            log = _CallLog()
            events = []
            with log.patch(), torch.no_grad():
                reference = hd._stock["decode_temporal"](vae, z)
                del log.sizes[:]
                plan = hd.plan_units(vae, z, world)
                waves = hd.plan_waves(
                    len(plan.units),
                    plan.tiles_per_clip,
                    world,
                    audio_rank=audio,
                    audio_wave_units=2,
                    audio_rank_units=own,
                )
                copies = [copy.deepcopy(vae) for _ in range(world)]
                world_ = _World(world)

                def rank_decode(rank):
                    hook = None
                    if rank == audio:

                        def hook():
                            events.append(("audio", len(log.sizes)))

                    return hd.run_flat_decode(
                        copies[rank],
                        z,
                        plan,
                        waves,
                        hd._stock["decode_temporal"],
                        rank,
                        world,
                        lambda t: world_.group.all_gather(t, separate_tensors=True),
                        after_first_unit=hook,
                        batch=3,
                    )

                outputs, errors = world_.run(rank_decode)
            for error in errors:
                if error is not None:
                    raise error
            for rank, output in enumerate(outputs):
                self.assertTrue(torch.equal(output, reference), (shape, world, rank))
            self.assertLessEqual(max(log.sizes), 3)
            self.assertGreater(max(log.sizes), 1, (shape, world))
            self.assertEqual(sum(log.sizes), len(plan.units))
            if audio is not None:
                self.assertEqual(len(events), 1)

    def test_audio_owner_decodes_one_unit_alone_before_its_audio(self):
        vae = _tiny_video_vae()
        torch.manual_seed(13)
        z = torch.randn(1, 24, 12, 16, 40)
        plan = hd.plan_units(vae, z, 3)
        waves = hd.plan_waves(
            len(plan.units),
            plan.tiles_per_clip,
            3,
            audio_rank=1,
            audio_wave_units=2,
            audio_rank_units=2,
        )
        events = []
        log = _CallLog()

        def counting(vae_, unit):
            events.append(("decode", int(unit.shape[0])))
            return log(vae_, unit)

        with patch.object(MiniMaxH3VideoVAE, "decode", counting), torch.no_grad():
            hd.run_flat_decode(
                vae,
                z,
                plan,
                waves,
                hd._stock["decode_temporal"],
                1,
                3,
                lambda tensor: [tensor, tensor, tensor],
                after_first_unit=lambda: events.append(("audio", 0)),
                batch=4,
            )
        self.assertEqual(events[:3], [("decode", 1), ("audio", 0), ("decode", 1)])
        self.assertEqual(events.count(("audio", 0)), 1)

    def test_stock_schedule_matches_with_a_batch_invariant_decoder(self):
        torch.manual_seed(14)
        z = torch.randn(1, 24, 12, 32, 40)  # 2 clips x 3x3 tiles
        vae = _tiny_video_vae()
        with _CallLog().patch(), torch.no_grad():
            reference = hd._stock["decode_temporal"](vae, z)
        # No process group: every tile is local, 9 a clip in calls of 2.
        log = _CallLog()
        with (
            log.patch(),
            _flags(SGLANG_TPU_OPT_H3_VAE_DECODE_BATCH=2),
            torch.no_grad(),
        ):
            self.assertTrue(torch.equal(vae.decode_temporal(z), reference))
        self.assertEqual(log.sizes, [2, 2, 2, 2, 1] * 2)
        for world in (2, 4):
            log = _CallLog()
            world_ = _World(world)
            copies = [copy.deepcopy(vae) for _ in range(world)]
            with (
                world_.installed(),
                log.patch(),
                _flags(SGLANG_TPU_OPT_H3_VAE_DECODE_BATCH=2),
                torch.no_grad(),
            ):
                outputs, errors = world_.run(lambda r: copies[r].decode_temporal(z))
            for error in errors:
                if error is not None:
                    raise error
            for output in outputs:
                self.assertTrue(torch.equal(output, reference), world)
            # 9 tiles a clip: 5/4 (2 ranks) or 3/2/2/2 (4 ranks) per rank.
            self.assertLessEqual(max(log.sizes), 2)
            self.assertEqual(sum(log.sizes), 2 * 9)
            self.assertEqual(
                sorted(log.sizes),
                sorted({2: [2, 2, 1, 2, 2] * 2, 4: [2, 1, 2, 2, 2] * 2}[world]),
            )

    def test_encode_and_stacked_tiling_keep_the_stock_path(self):
        vae = _tiny_video_vae()
        tiles = [torch.randn(1, 24, 7, 16, 16) for _ in range(3)]
        with (
            _flags(SGLANG_TPU_OPT_H3_VAE_DECODE_BATCH=4),
            patch.object(hd, "_unit_calls", side_effect=AssertionError("batched")),
            torch.no_grad(),
        ):
            for args, tile_batch in (
                ((tiles, [0, 1, 2], vae.decode), 0),
                ((tiles, [0, 1, 2], lambda t: t), 1),
                ((tiles, [1], vae.decode), 1),
            ):
                out = vae._run_tile_tasks(*args, tile_batch=tile_batch)
                self.assertEqual(len(out), len(args[1]))

    def test_real_decoder_batched_matches_per_unit_within_rounding(self):
        vae = _tiny_video_vae(num_layers=2)
        torch.manual_seed(15)
        units = [torch.randn(1, 24, 7, 16, 16) for _ in range(3)]
        for dtype, tolerance in ((None, 1e-5), (torch.bfloat16, 0.1)):
            with (
                torch.no_grad(),
                torch.autocast(
                    "cpu", dtype=dtype or torch.bfloat16, enabled=dtype is not None
                ),
            ):
                singles = torch.cat([vae.decode(u) for u in units])
                batched = vae.decode(torch.cat(units))
            self.assertEqual(batched.shape, singles.shape)
            self.assertLess(float((batched - singles).abs().max()), tolerance)


class TestBlockCompileFallbacks(_BatchCase):
    def setUp(self):
        super().setUp()
        hd._compiled_vaes.clear()
        torch._dynamo.reset()

    def test_block_failure_that_also_fails_eagerly_keeps_the_block_compiles(self):
        """A batched call that runs out of memory both through the compiled
        blocks and eagerly (BLOCK_COMPILE with DECODE_BATCH): the unit-by-unit
        redo and every later call still run the compiled blocks."""
        from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
            base_module,
        )

        vae = _tiny_video_vae(num_layers=3)
        blocks = list(vae.decoder.transformer_blocks)
        impls, compiled_calls = [], []
        for block in blocks:

            def compiled(hidden, rope, _eager=block._call_impl):
                if hidden.shape[0] > 1:
                    raise RuntimeError("planted RESOURCE_EXHAUSTED (compiled block)")
                compiled_calls.append(int(hidden.shape[0]))
                return _eager(hidden, rope)

            block._compiled_call_impl = compiled
            impls.append(compiled)
        stock_forward = base_module.TransformerBlock.forward

        def forward(block, hidden, rope):
            if hidden.shape[0] > 1:
                raise RuntimeError("planted RESOURCE_EXHAUSTED (eager block)")
            return stock_forward(block, hidden, rope)

        torch.manual_seed(21)
        latents = [torch.randn(1, 24, 7, 16, 16) for _ in range(4)]
        status = hd.DecodeStatus("req")
        token = hd._STATUS.set(status)
        try:
            with (
                patch.object(base_module.TransformerBlock, "forward", forward),
                patch.object(hd.logger, "warning") as warning,
                torch.no_grad(),
            ):
                calls = list(hd._unit_calls(vae, latents, 4))
        finally:
            hd._STATUS.reset(token)
        self.assertEqual([int(out.shape[0]) for _, out in calls], [1, 1, 1, 1])
        self.assertEqual([b._compiled_call_impl for b in blocks], impls)
        # The four redone units ran through the three compiled blocks.
        self.assertEqual(compiled_calls, [1] * 12)
        self.assertEqual(status.calls, [1, 1, 1, 1])
        self.assertEqual(len(status.events), 2, status.events)
        self.assertIn("eager too", status.events[0])
        self.assertIn("batched call of 4 units failed", status.events[1])
        self.assertEqual(warning.call_count, 1)
        self.assertIn("unit by unit", warning.call_args.args[0])
        for block in blocks:
            block._compiled_call_impl = None

    def test_stage_compiles_the_blocks_with_the_flag(self):
        stage = SimpleNamespace(video_vae=object())
        for flags, blocks in (
            ({}, 0),
            ({"SGLANG_TPU_ENABLE_H3_VAE_BLOCK_COMPILE": True}, 1),
        ):
            with (
                _flags(**flags),
                patch.object(hd, "compile_decoder_blocks") as per_block,
                patch.object(hd, "_new_request", side_effect=RuntimeError("stop")),
            ):
                forward = hd._decode_stage_forward(lambda *a: None)
                with self.assertRaises(RuntimeError):
                    forward(stage, None, None)
            self.assertEqual(per_block.call_count, blocks)


class TestFp32Projections(CustomTestCase):
    """SGLANG_TPU_ENABLE_H3_VAE_FP32_PROJECTIONS: x_embedder and proj_out
    outside autocast on any device, as the released decoder runs them."""

    def setUp(self):
        torch.set_num_threads(1)

    def tearDown(self):
        hd.refresh_precision_flags()

    def _decode(self, vae, z, on, hooked=()):
        seen = {}

        def record(name):
            def hook(module, inputs, output):
                seen[name] = (inputs[0], output)

            return hook

        handles = [
            module.register_forward_hook(record(name)) for name, module in hooked
        ]
        try:
            with _flags(SGLANG_TPU_ENABLE_H3_VAE_FP32_PROJECTIONS=on):
                hd.refresh_precision_flags()
                with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
                    output = vae.decode(z)
        finally:
            for handle in handles:
                handle.remove()
            hd.refresh_precision_flags()
        return output, seen

    def test_projections_run_outside_autocast_on_any_device(self):
        vae = _tiny_video_vae(num_layers=2)
        vae.decoder.prepare_autocast_linear_weights(torch.float16)
        decoder = vae.decoder
        hooked = (
            ("x_embedder", decoder.x_embedder),
            ("proj_out", decoder.proj_out),
            ("to_qkv", decoder.transformer_blocks[0].attn.to_qkv),
        )
        torch.manual_seed(22)
        z = torch.randn(1, 24, 7, 16, 16)
        _, off = self._decode(vae, z, False, hooked)
        _, on = self._decode(vae, z, True, hooked)
        for name in ("x_embedder", "proj_out"):
            module = getattr(decoder, name)
            # Off: fp16 operands under the decode's autocast (today's TPU path).
            self.assertEqual(off[name][1].dtype, torch.float16, name)
            inputs, output = on[name]
            self.assertEqual(output.dtype, torch.float32, name)
            exact = F.linear(inputs.float(), module.weight, module.bias)
            self.assertTrue(torch.equal(output, exact), name)
        # The blocks keep the fp16 autocast.
        self.assertEqual(on["to_qkv"][1].dtype, torch.float16)
        self.assertEqual(off["to_qkv"][1].dtype, torch.float16)

    def test_decode_moves_toward_fp32_and_traces_as_one_graph(self):
        import torch._dynamo as dynamo

        torch.manual_seed(23)
        z = torch.randn(2, 24, 7, 16, 16)
        vae = _tiny_video_vae(num_layers=2)
        with torch.no_grad():
            reference = vae.decode(z)
        vae.decoder.prepare_autocast_linear_weights(torch.float16)
        outs = {on: self._decode(vae, z, on)[0] for on in (False, True)}
        error = {on: float((outs[on] - reference).pow(2).mean()) for on in outs}
        self.assertLess(error[True], error[False])
        with _flags(SGLANG_TPU_ENABLE_H3_VAE_FP32_PROJECTIONS=True):
            hd.refresh_precision_flags()
            dynamo.reset()
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
                explained = dynamo.explain(vae.decoder)(z)
        self.assertEqual(explained.graph_count, 1, explained.break_reasons)
        self.assertEqual(explained.graph_break_count, 0)
        dynamo.reset()

    def test_stage_reads_the_flag_each_call(self):
        stage = SimpleNamespace(video_vae=None)
        forward = hd._decode_stage_forward(lambda *a: None)
        for on in (True, False):
            with (
                _flags(SGLANG_TPU_ENABLE_H3_VAE_FP32_PROJECTIONS=on),
                patch.object(hd, "_new_request", side_effect=RuntimeError("stop")),
            ):
                with self.assertRaises(RuntimeError):
                    forward(stage, None, None)
            self.assertIs(hd._fp32_projections, on)


def _fp16_product(x, weight, bias):
    """An fp16 matmul with float32 accumulation: exact products, float32 sums,
    the fp16 bias added in float32, one rounding to fp16 (cuBLAS addmm)."""
    out = torch.matmul(x.float(), weight.float().t())
    return (out + bias.float()).to(torch.float16)


def _all_fp16() -> torch.Tensor:
    """Every fp16 bit pattern."""
    return (
        torch.arange(-(2**15), 2**15, dtype=torch.int32)
        .to(torch.int16)
        .view(torch.float16)
    )


def _probe_qkv(batch=1, tokens=1797, heads=4, seed=0):
    """fp16 q/k/v in the ViT attention's [batch, tokens, heads, 64] layout:
    seed 0 at the default shape draws the TPU precision probe's plain normal
    inputs (after its linear operands); other seeds draw decoder-like ones
    (q/k RMS-normalized per head as the H3 decoder's parameter-free RMSNorm
    leaves them, correlated across heads)."""
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator)

    if seed == 0:
        randn(512, 2048), randn(1024, 2048), randn(1024)
        return [randn(batch, tokens, heads, 64).half() for _ in range(3)]

    def rms(t):
        return (t / t.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()).half()

    shared = randn(1, tokens, 1, 64) * 2
    q = rms(randn(batch, tokens, heads, 64) * 3 + shared)
    k = rms(randn(batch, tokens, heads, 64) * 3 + shared)
    return q, k, (randn(batch, tokens, heads, 64) * 0.5).half()


def _exact_attention(q, k, v):
    """float64 softmax(q k^T / 8) v in the [batch, tokens, heads, 64] layout."""
    q, k, v = (t.transpose(1, 2).double() for t in (q, k, v))
    out = torch.softmax(q @ k.transpose(-1, -2) / 8.0, -1) @ v
    return out.transpose(1, 2)


def _tpu_fp16_attention(q, k, v):
    """torch_tpu's fp16 SDPA (math path) as its roundings: scores to fp16,
    s - max to fp16, float32 exp and sum, P to fp16, P V to fp16 (the exact
    sums of each rounded operand). Reproduces the TPU precision probe's
    6.63e-4 on _probe_qkv's seed-0 inputs."""

    def r16(t):
        return t.half().double()

    q, k, v = (t.transpose(1, 2).double() for t in (q, k, v))
    s = r16(q @ k.transpose(-1, -2) / 8.0)
    e = torch.exp(r16(s - s.amax(-1, keepdim=True)))
    out = r16(r16(e / e.sum(-1, keepdim=True)) @ v)
    return out.transpose(1, 2)


def _rel_l2(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def _interpret_flash(q, k, v):
    """The Pallas kernel (minimax_h3_vae_attention) in interpret mode on
    [batch, tokens, heads, 64] fp16 q/k/v."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from sglang_tpu.multimodal_gen import minimax_h3_vae_attention as fa

    batch, tokens, heads, dim = q.shape
    run = jax.jit(lambda *a: fa.flash_attention(*a, interpret=True))
    with jax.default_device(jax.devices("cpu")[0]):
        out = run(
            *[
                jnp.asarray(t.reshape(batch, tokens, heads * dim).numpy())
                for t in (q, k, v)
            ]
        )
        out = np.asarray(out.astype(jnp.float32))
    return torch.from_numpy(out).half().view(batch, tokens, heads, dim)


class TestFlashAttention(CustomTestCase):
    """SGLANG_TPU_OPT_H3_VAE_FLASH_ATTENTION: the ViT attention as an
    fp16-faithful flash kernel inside decodes of a decoder it holds for."""

    def setUp(self):
        torch.set_num_threads(4)
        hd._flash_decoders.clear()

    def tearDown(self):
        hd._flash_decoders.clear()
        hd.refresh_precision_flags()

    def _need_pallas(self):
        try:
            import jax  # noqa: F401
            from jax.experimental.pallas import tpu as pltpu  # noqa: F401
        except ImportError:
            self.skipTest("jax with Pallas TPU is not installed")

    def test_at_least_as_accurate_as_the_fp16_attention(self):
        for seed in (0, 1):
            q, k, v = _probe_qkv(seed=seed)
            exact = _exact_attention(q, k, v)
            with torch.no_grad():
                ours = hd.flash_attention(q, k, v)
                cpu_fp16 = hd._stock["sdpa"](q, k, v)
            self.assertEqual(ours.dtype, torch.float16)
            self.assertEqual(ours.shape, q.shape)
            error = _rel_l2(ours, exact)
            # One fp16 rounding of the exact result, up to float32 sums.
            self.assertLess(error, 1.002 * _rel_l2(exact.half(), exact), seed)
            # TPU's fp16 SDPA rounds scores and probabilities (6.63e-4 at seed 0).
            tpu = _rel_l2(_tpu_fp16_attention(q, k, v), exact)
            self.assertLess(error, 0.4 * tpu, seed)
            self.assertLessEqual(error, _rel_l2(cpu_fp16, exact), seed)
            if seed == 0:
                self.assertAlmostEqual(tpu, 6.63e-4, delta=0.01e-4)
            # Elementwise too: never worse than the TPU path's largest error.
            worst = float((ours.double() - exact).abs().max())
            self.assertLess(
                worst, float((_tpu_fp16_attention(q, k, v) - exact).abs().max())
            )

    def test_kernel_matches_the_torch_arithmetic(self):
        self._need_pallas()
        # Odd token counts: padded key rows and a ragged last query block.
        for batch, tokens, heads, seed in (
            (2, 300, 4, 3),
            (1, 1797, 4, 0),
            (1, 1797, 2, 4),
        ):
            q, k, v = _probe_qkv(batch, tokens, heads, seed)
            with torch.no_grad():
                kernel = _interpret_flash(q, k, v)
                torch_path = hd.flash_attention(q, k, v)
            self.assertTrue(bool(torch.isfinite(kernel).all()))
            same = float((kernel == torch_path).float().mean())
            self.assertGreater(same, 0.98, (batch, tokens, heads))
            # Differences are float32 summation order: one fp16 step at most,
            # or float32 noise (1e-5) next to zero.
            step = torch.finfo(torch.float16).eps * torch_path.float().abs()
            step = step.clamp_min(1e-5)
            self.assertTrue(
                bool(((kernel.float() - torch_path.float()).abs() <= step).all()),
                (batch, tokens, heads),
            )
            exact = _exact_attention(q, k, v)
            self.assertLess(
                _rel_l2(kernel, exact), 1.002 * _rel_l2(exact.half(), exact)
            )

    def test_kernel_ignores_what_padding_holds(self):
        self._need_pallas()
        import jax.numpy as jnp
        import numpy as np

        from sglang_tpu.multimodal_gen import minimax_h3_vae_attention as fa

        q, k, v = _probe_qkv(1, 200, 2, 5)
        parts = [
            t
            for x in (q, k, v)
            for t in fa.split_bf16(jnp.asarray(x.reshape(1, 200, 128).numpy()))
        ]
        # 200 keys in a 256-row block: NaN in the rows past the array.
        padded = [
            jnp.pad(
                p.astype(jnp.float32), ((0, 0), (0, 56), (0, 0)), constant_values=np.nan
            ).astype(jnp.bfloat16)
            for p in parts
        ]
        call = fa.flash_call(1, 200, 2, interpret=True)
        out = np.asarray(call(*parts))
        # The same kernel reading NaN padding (a 256-token array, 200 real).
        nan_call = fa.flash_call(1, 200, 2, interpret=True)
        out_nan = np.asarray(nan_call(*[p[:, :200] for p in padded]))
        self.assertTrue(np.isfinite(out).all())
        np.testing.assert_array_equal(out, out_nan)
        exact = _exact_attention(q, k, v).reshape(1, 200, 128).numpy()
        self.assertLess(np.abs(out - exact).max(), 2e-3)

    def test_split_is_exact_for_every_fp16_value(self):
        everything = _all_fp16()
        finite = torch.isfinite(everything)
        hi, lo = hd._bf16_parts(everything.float())
        self.assertTrue(torch.equal((hi + lo)[finite], everything.float()[finite]))
        self.assertTrue(torch.equal(hi.bfloat16().float()[finite], hi[finite]))
        self.assertTrue(torch.equal(lo.bfloat16().float()[finite], lo[finite]))
        infinite = torch.isinf(everything)
        self.assertTrue(torch.equal(hi[infinite], everything.float()[infinite]))
        self.assertTrue(bool((lo[infinite] == 0).all()))
        try:
            import jax.numpy as jnp
            import numpy as np

            from sglang_tpu.multimodal_gen import minimax_h3_vae_attention as fa
        except ImportError:
            return
        jhi, jlo = fa.split_bf16(jnp.asarray(everything.numpy()))
        jhi = np.asarray(jhi.astype(jnp.float32))
        jlo = np.asarray(jlo.astype(jnp.float32))
        # The kernel's split (XLA) and the torch path's agree bit for bit.
        np.testing.assert_array_equal(jhi, hi.numpy())
        np.testing.assert_array_equal(jlo, lo.numpy())

    def test_decoder_check(self):
        self.assertIsNone(hd.flash_decoder_reason(_tiny_video_vae(2, dim_head=64)))
        self.assertIn("head_dim 16", hd.flash_decoder_reason(_tiny_video_vae()))
        self.assertIn(
            "odd head count", hd.flash_decoder_reason(_tiny_video_vae(1, 64, 3))
        )
        vae = _tiny_video_vae(2, dim_head=64)
        vae.decoder.transformer_blocks[1].attn.norm_k = torch.nn.RMSNorm(64)
        self.assertIn("block 1: norm_k", hd.flash_decoder_reason(vae))
        vae.decoder.transformer_blocks[1].attn.norm_k = torch.nn.LayerNorm(
            64, elementwise_affine=False
        )
        self.assertIn("block 1: norm_k", hd.flash_decoder_reason(vae))
        self.assertIn("no ViT decoder", hd.flash_decoder_reason(SimpleNamespace()))

    def test_tpu_decode_registers_the_op_first(self):
        vae = _tiny_video_vae(2, dim_head=64)
        z = SimpleNamespace(device=torch.device("cpu"))
        tpu_z = unittest.mock.MagicMock(spec=torch.Tensor)
        tpu_z.device = SimpleNamespace(type="tpu")
        with (
            _flags(SGLANG_TPU_OPT_H3_VAE_FLASH_ATTENTION=True),
            patch.object(hd, "_flash_attention_op") as register,
            patch.object(hd, "_flash_op", None),
            patch.object(hd.logger, "info"),
        ):
            hd.refresh_precision_flags()
            self.assertFalse(hd._flash_wanted(vae, z))  # not a tensor
            self.assertTrue(hd._flash_wanted(vae, torch.zeros(1)))
            register.assert_not_called()  # a CPU decode
            self.assertTrue(hd._flash_wanted(vae, tpu_z))
            register.assert_called_once()
            with patch.object(hd, "_flash_op", object()):
                self.assertTrue(hd._flash_wanted(vae, tpu_z))
            register.assert_called_once()
            narrow = _tiny_video_vae(2)
            with patch.object(hd.logger, "warning"):
                self.assertFalse(hd._flash_wanted(narrow, tpu_z))
            register.assert_called_once()
        hd.refresh_precision_flags()

    def test_routing(self):
        from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
            attention,
        )

        q, k, v = _probe_qkv(1, 64, 4, 6)
        stock = hd._stock["sdpa"](q, k, v)
        flash = hd.flash_attention(q, k, v)
        self.assertFalse(torch.equal(stock, flash))
        self.assertTrue(torch.equal(attention._sdpa_attention(q, k, v), stock))
        hd._flash_active = True
        try:
            self.assertTrue(torch.equal(attention._sdpa_attention(q, k, v), flash))
            for other in (
                [t.bfloat16() for t in (q, k, v)],
                [t.float() for t in (q, k, v)],
                [t[..., :32].contiguous() for t in (q, k, v)],
                [t[:, :, :3].contiguous() for t in (q, k, v)],
            ):
                self.assertTrue(
                    torch.equal(
                        attention._sdpa_attention(*other), hd._stock["sdpa"](*other)
                    )
                )
            self.assertFalse(hd.flash_attention_fits(q, k, v.float()))
            self.assertFalse(hd.flash_attention_fits(q, k[:, :32], v[:, :32]))
        finally:
            hd._flash_active = False

    def _decode(self, vae, z, on, autocast=True, count=True):
        # The call count needs a mock, which compiled code cannot trace.
        counter = (
            patch.object(hd, "flash_attention", wraps=hd.flash_attention)
            if count
            else contextlib.nullcontext(None)
        )
        with _flags(SGLANG_TPU_OPT_H3_VAE_FLASH_ATTENTION=on):
            hd.refresh_precision_flags()
            with (
                counter as flash,
                torch.no_grad(),
                torch.autocast("cpu", dtype=torch.float16, enabled=autocast),
            ):
                output = vae.decode(z)
        hd.refresh_precision_flags()
        self.assertFalse(hd._flash_active)
        return output, None if flash is None else flash.call_count

    def test_decode_runs_it_only_with_the_flag_and_a_fitting_decoder(self):
        torch.manual_seed(47)
        z = torch.randn(1, 24, 7, 16, 16)
        vae = _tiny_video_vae(num_layers=2, dim_head=64)
        with torch.no_grad():
            reference = vae.decode(z)
        vae.decoder.prepare_autocast_linear_weights(torch.float16)
        with patch.object(hd.logger, "info") as info:
            on, calls = self._decode(vae, z, True)
        self.assertEqual(calls, 2)  # one a block
        self.assertIn("flash ViT attention on (2 blocks)", " ".join(_messages(info)))
        off, calls = self._decode(vae, z, False)
        self.assertEqual(calls, 0)
        self.assertFalse(torch.equal(on, off))
        # At least as close to the fp32 decode as the stock fp16 attention.
        error_on = float((on - reference).abs().mean())
        error_off = float((off - reference).abs().mean())
        self.assertLessEqual(error_on, 1.02 * error_off)
        self.assertLess(float((on - reference).abs().max()), 0.05)
        # A decoder the bound does not cover keeps the stock attention.
        narrow = _tiny_video_vae(num_layers=2)
        narrow.decoder.prepare_autocast_linear_weights(torch.float16)
        with patch.object(hd.logger, "warning") as warning:
            _, calls = self._decode(narrow, z, True)
        self.assertEqual(calls, 0)
        self.assertIn("flash ViT attention off", " ".join(_messages(warning)))
        # Outside a decode (the encoder, other modules): stock.
        self.assertFalse(hd._flash_active)

    def test_composes_with_block_compile_and_batches(self):
        import torch._dynamo as dynamo

        torch.manual_seed(48)
        z = torch.randn(2, 24, 7, 16, 16)
        vae = _tiny_video_vae(num_layers=2, dim_head=64)
        vae.decoder.prepare_autocast_linear_weights(torch.float16)
        eager, calls = self._decode(vae, z, True)
        self.assertEqual(calls, 2)
        with _flags(SGLANG_TPU_OPT_H3_VAE_FLASH_ATTENTION=True):
            hd.refresh_precision_flags()
            hd._flash_active = True
            try:
                dynamo.reset()
                with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
                    explained = dynamo.explain(vae.decoder)(z)
                self.assertEqual(explained.graph_count, 1, explained.break_reasons)
                self.assertEqual(explained.graph_break_count, 0)
            finally:
                hd._flash_active = False
        dynamo.reset()
        compile_kwargs = {"backend": "aot_eager", "dynamic": False, "fullgraph": True}
        stock, _ = self._decode(vae, z, False)
        hd._compiled_vaes.discard(id(vae))
        hd.compile_decoder_blocks(vae, compile_kwargs)
        traced = []
        inner = hd._flash_attention_torch

        def recording(*args):
            traced.append(args[0].shape)
            return inner(*args)

        with patch.object(hd, "_flash_attention_torch", recording):
            compiled, _ = self._decode(vae, z, True, count=False)
        # The compiled blocks traced the flash path (once a block).
        self.assertEqual(len(traced), 2)
        self._assert_rounding_close(compiled, eager)
        self.assertGreater(
            float((compiled == eager).float().mean()),
            float((compiled == stock).float().mean()),
        )
        for block in vae.decoder.transformer_blocks:
            self.assertIsNotNone(block._compiled_call_impl)
            block._compiled_call_impl = None
        dynamo.reset()
        # Batched decoder calls (SGLANG_TPU_OPT_H3_VAE_DECODE_BATCH).
        per_unit = torch.cat(
            [self._decode(vae, z[i : i + 1], True)[0] for i in range(2)]
        )
        self._assert_rounding_close(per_unit, eager)
        with _flags(SGLANG_TPU_OPT_H3_VAE_FLASH_ATTENTION=True):
            hd.refresh_precision_flags()
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
                calls = list(hd._unit_calls(vae, [z[0:1], z[1:2]], 2))
        self.assertEqual([int(out.shape[0]) for _, out in calls], [2])
        self._assert_rounding_close(calls[0][1], eager)

    def _assert_rounding_close(self, a, b):
        self.assertLess(float((a - b).abs().max()), 2e-3)
        self.assertGreater(float((a == b).float().mean()), 0.5)

    def test_stage_reads_the_flag_each_call(self):
        stage = SimpleNamespace(video_vae=None)
        forward = hd._decode_stage_forward(lambda *a: None)
        for on in (True, False):
            with (
                _flags(SGLANG_TPU_OPT_H3_VAE_FLASH_ATTENTION=on),
                patch.object(hd, "_new_request", side_effect=RuntimeError("stop")),
            ):
                with self.assertRaises(RuntimeError):
                    forward(stage, None, None)
            self.assertIs(hd._flash_attention, on)

    def test_stage_decodes_with_the_flash_attention(self):
        torch.manual_seed(49)
        latents = torch.randn(1, 24, 12, 16, 16)
        audio_latents = torch.randn(2, 32, 9)
        outputs = {}
        for on in (False, True):
            server_args = _server_args("fp16")
            vae = _tiny_video_vae(num_layers=2, dim_head=64)
            stage = _stage(server_args, vae, _tiny_audio_vae())
            batch = _batch(latents, audio_latents, None, save_output=False)
            with (
                _flags(SGLANG_TPU_OPT_H3_VAE_FLASH_ATTENTION=on),
                patch.object(hd, "flash_attention", wraps=hd.flash_attention) as flash,
                patch.object(hd.logger, "info"),
            ):
                outputs[on] = stage.forward(batch, server_args).output
            # 2 clips of one tile: 2 decoder calls of 2 blocks.
            self.assertEqual(flash.call_count, 4 if on else 0)
        self.assertFalse(torch.equal(outputs[True], outputs[False]))
        self.assertLess(float((outputs[True] - outputs[False]).abs().max()), 1e-2)


def _todays_fp16_linear(x, weight, bias):
    """Today's TPU fp16 linear as XLA's HLO computes it: hi*hi + hi*lo +
    lo*hi of the exact bf16 parts in float32, rounded to fp16, the fp16 bias
    added and rounded again (the device probe's 2.66e-4 on its inputs)."""
    xh, xl = hd._bf16_parts(x.float())
    wh, wl = hd._bf16_parts(weight.float())
    product = (xh @ wh.t() + xh @ wl.t() + xl @ wh.t()).half()
    return (product.float() + bias.float()).half()


def _interpret_linear(x, weight, bias):
    """minimax_h3_vae_linear's kernel in Pallas interpret mode."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from sglang_tpu.multimodal_gen import minimax_h3_vae_linear as vl

    run = jax.jit(lambda *a: vl.pallas_linear(*a, interpret=True))
    with jax.default_device(jax.devices("cpu")[0]):
        out = run(*(jnp.asarray(t.numpy()) for t in (x, weight, bias)))
        out = np.asarray(out.astype(jnp.float32))
    return torch.from_numpy(out).half()


class TestPallasLinear(CustomTestCase):
    """SGLANG_TPU_OPT_H3_VAE_PALLAS_LINEAR: the ViT block linears as a Pallas
    kernel over exact bf16 splits made in VMEM."""

    def setUp(self):
        torch.set_num_threads(4)

    def tearDown(self):
        hd.refresh_precision_flags()

    def _need_pallas(self):
        try:
            import jax  # noqa: F401
            from jax.experimental.pallas import tpu as pltpu  # noqa: F401
        except ImportError:
            self.skipTest("jax with Pallas TPU is not installed")

    @staticmethod
    def _operands(rows, inner, outer, seed):
        """Decoder-like fp16 operands; seed 0 the TPU precision probe's."""
        generator = torch.Generator().manual_seed(seed)

        def randn(*shape):
            return torch.randn(*shape, generator=generator)

        if seed == 0:
            x, weight = randn(512, 2048), randn(1024, 2048) * 0.02
            return x.half(), weight.half(), randn(1024).half()
        x = (randn(rows, inner) * 2).half()
        # Subnormal and extreme magnitudes, as fp16 activations can hold them.
        x[0, :8] = torch.tensor([2.0**-24, -(2.0**-20), 6e-5, 3e4, -3e4, 0, -0.0, 1])
        return x, (randn(outer, inner) * 0.02).half(), (randn(outer) * 0.1).half()

    def test_more_precise_than_todays_fp16_linear(self):
        for seed in (0, 1):
            x, weight, bias = self._operands(256, 2048, 512, seed)
            exact = x.double() @ weight.double().t() + bias.double()
            ours = hd.pallas_linear(x, weight, bias)
            self.assertEqual(ours.dtype, torch.float16)
            today = _rel_l2(_todays_fp16_linear(x, weight, bias), exact)
            if seed == 0:
                self.assertAlmostEqual(today, 2.66e-4, delta=0.01e-4)
            # One rounding where today's path rounds twice (0.78x and 0.87x
            # its error), within 1e-4 of an fp16 matmul with float32 sums.
            self.assertLess(_rel_l2(ours, exact), 0.9 * today, seed)
            self.assertLess(
                _rel_l2(ours, exact),
                (1 + 1e-4) * _rel_l2(_fp16_product(x, weight, bias), exact),
                seed,
            )
            # Leading dimensions pass through.
            stacked = hd.pallas_linear(x.view(2, -1, x.shape[1]), weight, bias)
            self.assertTrue(torch.equal(stacked.view_as(ours), ours))

    def test_codec_is_exact(self):
        self._need_pallas()
        import jax
        import jax.numpy as jnp
        import numpy as np

        from sglang_tpu.multimodal_gen import minimax_h3_vae_linear as vl

        everything = _all_fp16()
        bits = jnp.asarray(everything.view(torch.int16).numpy()).astype(jnp.int32)
        decoded = torch.from_numpy(np.asarray(jax.jit(vl.decode_fp16)(bits)))
        same = decoded.view(torch.int32) == everything.float().view(torch.int32)
        self.assertTrue(bool((same | (decoded.isnan() & everything.isnan())).all()))
        finite = everything[torch.isfinite(everything)].float().unique()
        mid = (finite[:-1].double() + finite[1:].double()).div(2).float()
        values = torch.cat(
            [
                finite,
                mid,
                torch.nextafter(mid, torch.tensor(float("inf"))),
                torch.nextafter(mid, torch.tensor(float("-inf"))),
                torch.tensor([65519.996, 65520.0, 1e30, float("inf"), 2.0**-25]),
                -torch.tensor([65520.0, float("inf"), 2.0**-25, 3 * 2.0**-26]),
            ]
        )
        encoded = np.asarray(jax.jit(vl.encode_fp16)(jnp.asarray(values.numpy())))
        expected = values.half().view(torch.int16).numpy().astype(np.int32) & 0xFFFF
        np.testing.assert_array_equal(encoded & 0xFFFF, expected)
        nan = np.asarray(jax.jit(vl.encode_fp16)(jnp.asarray([np.nan], jnp.float32)))
        self.assertEqual(int(nan[0]) & 0x7E00, 0x7E00)
        # Real call shapes: 2 row blocks a unit, 480 rows at k 8192.
        self.assertEqual(vl.tiles(7188, 2048, 16384), (912, 1024))
        self.assertEqual(vl.tiles(1797, 2048, 6144), (912, 1024))
        self.assertEqual(vl.tiles(7188, 8192, 2048), (480, 512))

    def test_kernel_matches_the_torch_arithmetic(self):
        self._need_pallas()
        # Ragged row blocks, a 1,797-row unit, 256-wide column blocks.
        for rows, inner, outer, seed in ((300, 512, 256, 2), (1797, 2048, 512, 3)):
            x, weight, bias = self._operands(rows, inner, outer, seed)
            kernel = _interpret_linear(x, weight, bias)
            torch_path = hd.pallas_linear(x, weight, bias)
            self.assertTrue(bool(torch.isfinite(kernel).all()))
            self.assertGreater(float((kernel == torch_path).float().mean()), 0.99)
            # Differences are float32 summation order: one fp16 step, or
            # float32 noise next to zero.
            step = (torch_path.float().abs() * 2.0**-10).clamp_min(1e-5)
            self.assertTrue(
                bool(((kernel.float() - torch_path.float()).abs() <= step).all())
            )
            exact = x.double() @ weight.double().t() + bias.double()
            self.assertLess(
                _rel_l2(kernel, exact), (1 + 1e-4) * _rel_l2(torch_path, exact)
            )
        # A non-finite input spoils its own row only.
        x[5, 7] = float("inf")
        x[9, 3] = float("nan")
        spoiled = _interpret_linear(x, weight, bias)
        self.assertFalse(bool(torch.isfinite(spoiled[[5, 9]]).any()))
        rows = [r for r in range(x.shape[0]) if r not in (5, 9)]
        self.assertTrue(torch.equal(spoiled[rows], kernel[rows]))

    def _decode(self, vae, z, flags, dtype=torch.float16):
        with (
            _flags(**flags),
            patch.object(hd, "pallas_linear", wraps=hd.pallas_linear) as pallas,
            patch.object(hd.logger, "info"),
        ):
            hd.refresh_precision_flags()
            with torch.no_grad(), torch.autocast("cpu", dtype=dtype):
                output = vae.decode(z)
        hd.refresh_precision_flags()
        self.assertFalse(hd._linear_active)
        return output, pallas.call_count

    def test_decode_routes_the_block_linears_under_fp16_autocast_only(self):
        torch.manual_seed(50)
        z = torch.randn(1, 24, 7, 16, 16)
        # 256 wide: every block linear has the kernel's granule.
        vae = _tiny_video_vae(num_layers=2, dim_head=64, heads=4)
        with torch.no_grad():
            reference = vae.decode(z)
        vae.decoder.prepare_autocast_linear_weights(torch.float16)
        on = dict(SGLANG_TPU_OPT_H3_VAE_PALLAS_LINEAR=True)
        stock, calls = self._decode(vae, z, {})
        self.assertEqual(calls, 0)
        ours, calls = self._decode(vae, z, on)
        self.assertEqual(calls, 8)  # 4 linears a block, 2 blocks
        self.assertFalse(torch.equal(ours, stock))
        self.assertLess(float((ours - stock).abs().max()), 1e-2)
        self.assertLessEqual(
            float((ours - reference).pow(2).mean()),
            1.05 * float((stock - reference).pow(2).mean()),
        )
        # bf16 autocast keeps the stock linears; so does a decoder whose
        # widths are not multiples of 256 (it runs, unrouted).
        _, calls = self._decode(vae, z, on, dtype=torch.bfloat16)
        self.assertEqual(calls, 0)
        narrow = _tiny_video_vae(num_layers=2)
        narrow.decoder.prepare_autocast_linear_weights(torch.float16)
        narrow_on, calls = self._decode(narrow, z, on)
        self.assertEqual(calls, 0)
        self.assertTrue(torch.equal(narrow_on, self._decode(narrow, z, {})[0]))
        with _flags(**on):
            hd.refresh_precision_flags()
            self.assertFalse(hd._linear_wanted(vae, z))  # autocast off
            with torch.autocast("cpu", dtype=torch.float16):
                self.assertTrue(hd._linear_wanted(vae, z))
                self.assertFalse(hd._linear_wanted(SimpleNamespace(), z))
        hd.refresh_precision_flags()

    def test_block_traces_as_one_graph(self):
        import torch._dynamo as dynamo

        vae = _tiny_video_vae(num_layers=1, dim_head=64, heads=4)
        vae.decoder.prepare_autocast_linear_weights(torch.float16)
        with _flags(SGLANG_TPU_OPT_H3_VAE_PALLAS_LINEAR=True):
            hd.refresh_precision_flags()
            with torch.autocast("cpu", dtype=torch.float16):
                self.assertTrue(hd._linear_wanted(vae, torch.zeros(1)))
            hd._linear_active = True
            try:
                dynamo.reset()
                with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
                    explained = dynamo.explain(vae.decoder)(
                        torch.randn(1, 24, 7, 16, 16)
                    )
            finally:
                hd._linear_active = False
                dynamo.reset()
        self.assertEqual(explained.graph_count, 1, explained.break_reasons)

    def test_tpu_decode_registers_the_op_first(self):
        vae = _tiny_video_vae(1, dim_head=64, heads=4)
        tpu_z = unittest.mock.MagicMock(spec=torch.Tensor)
        tpu_z.device = SimpleNamespace(type="tpu")
        with (
            _flags(SGLANG_TPU_OPT_H3_VAE_PALLAS_LINEAR=True),
            patch.object(hd, "_pallas_linear_op") as register,
            patch.object(hd, "_linear_op", None),
            patch.object(torch, "is_autocast_enabled", return_value=True),
            patch.object(torch, "get_autocast_dtype", return_value=torch.float16),
        ):
            hd.refresh_precision_flags()
            self.assertTrue(hd._linear_wanted(vae, torch.zeros(1)))
            register.assert_not_called()  # a CPU decode
            self.assertTrue(hd._linear_wanted(vae, tpu_z))
            register.assert_called_once()
            with patch.object(hd, "_linear_op", object()):
                self.assertTrue(hd._linear_wanted(vae, tpu_z))
            register.assert_called_once()
        hd.refresh_precision_flags()

    def test_stage_reads_the_flag_each_call(self):
        stage = SimpleNamespace(video_vae=None)
        forward = hd._decode_stage_forward(lambda *a: None)
        for on in (True, False):
            with (
                _flags(SGLANG_TPU_OPT_H3_VAE_PALLAS_LINEAR=on),
                patch.object(hd, "_new_request", side_effect=RuntimeError("stop")),
            ):
                with self.assertRaises(RuntimeError):
                    forward(stage, None, None)
            self.assertIs(hd._pallas_linear, on)


_FUSED_FLAGS = dict(
    SGLANG_TPU_OPT_H3_VAE_PALLAS_LINEAR=True,
    SGLANG_TPU_OPT_H3_VAE_FLASH_ATTENTION=True,
    SGLANG_TPU_OPT_H3_VAE_FUSED_LINEARS=True,
)


def _fused_block(tokens=300, batch=2, seed=60):
    """A 256-wide decoder block (4 heads of 64) with fp16 linears and held
    int16 operands, an f32 residual and NeoX tables (48 of 64 lanes)."""
    vae = _tiny_video_vae(1, dim_head=64, heads=4)
    vae.decoder.prepare_autocast_linear_weights(torch.float16)
    block = vae.decoder.transformer_blocks[0]
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for norm in (block.norm1, block.norm2):
            norm.weight.copy_(1 + 0.2 * torch.randn(256, generator=generator))
        for linear in hd._block_linears(vae.decoder):
            linear.bias.copy_(0.1 * torch.randn(linear.bias.shape, generator=generator))
    hd._hold_fused_operands(vae, None)  # off device: no latent needed
    angles = torch.randn(1, tokens, 1, 24, generator=generator) * 3
    rope = tuple(f(torch.cat([angles] * 2, -1)).half() for f in (torch.cos, torch.sin))
    return vae, block, torch.randn(batch, tokens, 256, generator=generator) * 3, rope


class TestFusedLinears(CustomTestCase):
    """SGLANG_TPU_OPT_H3_VAE_FUSED_LINEARS: a ViT block as five custom calls
    with its norms, RoPE, gating and residuals inside, one fp16 rounding."""

    def setUp(self):
        torch.set_num_threads(4)

    def tearDown(self):
        hd._fused_ops = hd._stage_room = None
        hd._flash_decoders.clear()
        hd.refresh_precision_flags()

    def test_operands_are_held_in_buffers_of_their_own(self):
        """The int16 operands are computed, not cloned (on TPU a clone of an
        fp16 tensor's int16 view stays a view, which every block program call
        copies again); an fp16 weight then views its buffer."""
        vae = _tiny_video_vae(num_layers=1, dim_head=64, heads=4)
        vae.decoder.prepare_autocast_linear_weights(torch.float16)
        linears = hd._block_linears(vae.decoder)
        before = {
            m: (m.weight.detach().clone(), m.bias.detach().clone()) for m in linears
        }
        with patch.object(torch.Tensor, "clone", side_effect=AssertionError("clone")):
            self.assertTrue(hd._hold_fused_operands(vae, None))
        qkv = vae.decoder.transformer_blocks[0].attn.to_qkv
        for m, (weight, bias) in before.items():
            bits, held = m.h3_bias_bits, m.h3_weight_bits
            if held.dim() == 2:  # w2 is held split off device
                self.assertTrue(torch.equal(held.view(torch.float16), weight))
                storages = (t.untyped_storage().data_ptr() for t in (held, m.weight))
                self.assertEqual(*storages)
            if m is qkv:  # rows q, k, v of each head's 3 x 64, then zeros
                bias = bias.view(4, 3, 64).transpose(0, 1).reshape(3, -1)
                bias = torch.cat([bias, bias.new_zeros(5, bias.shape[1])])
            rows = 8 if m is qkv else 1
            self.assertTrue(torch.equal(bits.view(torch.float16), bias.view(rows, -1)))

    def test_weight_split_is_exact_for_every_fp16_value(self):
        values = _all_fp16()
        hi, lo = hd.split_fp16(values)
        self.assertEqual((hi.dtype, lo.dtype), (torch.bfloat16, torch.bfloat16))
        finite = torch.isfinite(values)
        self.assertTrue(
            torch.equal((hi.double() + lo.double())[finite], values.double()[finite])
        )
        # hi is the bf16 rounding; lo carries the rest (subnormals included).
        self.assertTrue(torch.equal(hi[finite], values[finite].to(torch.bfloat16)))
        subnormal = finite & (values != 0) & (values.abs() < 2.0**-14)
        self.assertEqual(int(subnormal.sum()), 2 * 1023)
        # Subnormals with more than 8 significant bits need their low part too.
        self.assertTrue(
            torch.equal(
                (hi.double() + lo.double())[subnormal], values.double()[subnormal]
            )
        )
        self.assertGreater(int(lo[subnormal].count_nonzero()), 0)
        # Some normal values need a non-zero low part; none overflows.
        self.assertGreater(int(lo[finite].count_nonzero()), 50000)
        self.assertTrue(bool(torch.isfinite(hi[finite]).all()))
        inf = torch.isinf(values)
        self.assertTrue(torch.equal(hi[inf].float(), values[inf].float()))
        self.assertTrue(bool((lo[inf] == 0).all()))
        nan = torch.isnan(values)
        self.assertTrue(bool(torch.isnan(hi[nan]).all() & torch.isnan(lo[nan]).all()))
        with self.assertRaises(TypeError):
            hd.split_fp16(values.float())

    @torch.no_grad()
    def test_kernels_match_the_torch_arithmetic(self):
        try:
            import jax
            import jax.numpy as jnp
            import numpy as np

            from sglang_tpu.multimodal_gen import minimax_h3_vae_fused as vf
        except ImportError:
            self.skipTest("jax with Pallas TPU is not installed")
        _, block, h, rope = _fused_block()
        attn, ff, rows = block.attn, block.ff, hd._fused_rows(300)
        # Garbage in the padding rows reaches no real row.
        h = torch.cat([h, torch.full((2, rows - 300, 256), 1e4)], 1)

        def kernel(fn, *args):  # bf16 crosses as f32 (numpy has no bf16)
            args = [
                jnp.asarray(a.float().numpy()).astype(jnp.bfloat16)
                if isinstance(a, torch.Tensor) and a.dtype == torch.bfloat16
                else jnp.asarray(a.numpy())
                if isinstance(a, torch.Tensor)
                else a
                for a in args
            ]
            with jax.default_device(jax.devices("cpu")[0]):
                out = fn(*args, interpret=True)
            outs = [
                torch.from_numpy(
                    np.asarray(o.astype(jnp.float32) if o.dtype == jnp.bfloat16 else o)
                )
                for o in (out if isinstance(out, tuple) else (out,))
            ]
            return outs if isinstance(out, tuple) else outs[0]

        def close(got, want, tol):
            got, want = got[:, :300].double(), want[:, :300].double()
            self.assertTrue(bool(torch.isfinite(got).all()))
            self.assertLess(float((got - want).norm() / want.norm()), tol)

        tables = hd.fused_tables(rope, rows)
        qkv = (h, block.norm1.weight.view(1, -1), attn.to_qkv.h3_weight_bits)
        qkv += (attn.to_qkv.h3_bias_bits, tables, 1e-5, 1e-5, 24)
        parts = hd._fused_qkv_torch(*qkv)
        got = kernel(vf.fused_qkv, *qkv)
        # The split kernel is torch's exact split; held split weights give
        # the kernels' own results bit for bit.
        held = [
            kernel(vf.split_weight, m.h3_weight_bits).bfloat16()
            for m in (attn.to_qkv, ff.w1)
        ]
        self.assertTrue(torch.equal(held[0], hd._split_torch(qkv[2])))
        split_qkv = kernel(vf.fused_qkv, *qkv[:2], held[0], *qkv[3:])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(split_qkv, got)))
        for i in (0, 2, 4):  # q, k, v: f32 summation order and the rsqrt differ
            close(got[i] + got[i + 1], parts[i].float() + parts[i + 1].float(), 3e-6)

        flash = functools.partial(vf.attention.flash_call, 2, 300, 4, rows=rows)
        attention = kernel(lambda *p, interpret: flash(interpret=interpret)(*p), *parts)
        close(attention, hd._flash_parts_torch(*parts, seq=300), 1e-5)
        gate = (h, block.norm2.weight.view(1, -1), ff.w1.h3_weight_bits)
        gate += (ff.w1.h3_bias_bits, 1e-5)
        bits = kernel(vf.fused_gate, *gate)
        split_gate = kernel(vf.fused_gate, *gate[:2], held[1], *gate[3:])
        self.assertTrue(torch.equal(split_gate, bits))
        want = hd._fused_gate_torch(*gate).view(torch.float16)[:, :300].float()
        diff = (bits.view(torch.float16)[:, :300].float() - want).abs()
        # One fp16 step, plus the f32 summation-order difference near zero:
        # torch's CPU kernels sum in an order that depends on the vector width
        # (AVX2, AVX-512), which moves a sum near zero by about 1e-6.
        self.assertTrue(bool((diff <= want.abs() * 2.0**-10 + 4e-6).all()))  # one step
        w2 = (ff.w2.h3_weight_bits, ff.w2.h3_bias_bits, block.scale2.view(1, -1))
        out = kernel(vf.fused_residual, bits, *w2, h)
        close(out - h, hd._fused_residual_torch(bits, *w2, h) - h, 1e-5)

    def test_block_is_more_precise_than_todays_path(self):
        _, block, h, rope = _fused_block(tokens=304)
        reference = copy.deepcopy(block).double()
        for linear in (block.attn.to_qkv, block.attn.to_out, block.ff.w1, block.ff.w2):
            linear.forward = types.MethodType(hd._pallas_linear_forward, linear)
        with torch.no_grad():
            exact = reference(h.double(), tuple(t.double() for t in rope)) - h.double()
            hd._linear_active = hd._flash_active = True
            try:
                with torch.autocast("cpu", dtype=torch.float16):
                    today = block(h, rope) - h  # the Pallas linear and flash path
            finally:
                hd._linear_active = hd._flash_active = False
            fused = hd.fused_block(block, h, rope) - h
        self.assertEqual(fused.dtype, torch.float32)
        # The stock block rounds to fp16 13 times; the fused block once.
        self.assertLess(_rel_l2(fused, exact), 0.5 * _rel_l2(today, exact))
        worst = float((fused.double() - exact).abs().max())
        self.assertLess(worst, float((today.double() - exact).abs().max()))

    def _decode(self, vae, z, flags, count=True):
        counter = (
            patch.object(hd, "fused_block", wraps=hd.fused_block)
            if count
            else contextlib.nullcontext(None)
        )
        with _flags(**flags), patch.object(hd.logger, "info"):
            hd.refresh_precision_flags()
            with (
                counter as fused,
                torch.no_grad(),
                torch.autocast("cpu", dtype=torch.float16),
            ):
                output = vae.decode(z)
        hd.refresh_precision_flags()
        self.assertFalse(hd._fused_active)
        return output, None if fused is None else fused.call_count

    def test_decode_routes_compiles_and_holds_its_operands_once(self):
        import torch._dynamo as dynamo

        torch.manual_seed(61)
        z = torch.randn(2, 24, 7, 16, 16)
        vae = _tiny_video_vae(num_layers=2, dim_head=64, heads=4)
        with torch.no_grad():
            reference = vae.decode(z)
        vae.decoder.prepare_autocast_linear_weights(torch.float16)
        off = dict(_FUSED_FLAGS, SGLANG_TPU_OPT_H3_VAE_FUSED_LINEARS=False)
        today, calls = self._decode(vae, z, off)
        self.assertEqual(calls, 0)
        fused, calls = self._decode(vae, z, _FUSED_FLAGS)
        self.assertEqual(calls, 2)  # one a block
        self.assertLess(float((fused - today).abs().max()), 5e-3)
        error = float((fused - reference).pow(2).mean())
        self.assertLessEqual(error, float((today - reference).pow(2).mean()))
        # The fused blocks need the flash attention routed too.
        no_flash = dict(_FUSED_FLAGS, SGLANG_TPU_OPT_H3_VAE_FLASH_ATTENTION=False)
        self.assertEqual(self._decode(vae, z, no_flash)[1], 0)
        # Their int16 operands are made once, and again when a weight changes;
        # an fp16 weight shares its buffer (one copy), and the stock blocks
        # read it there.
        qkv = vae.decoder.transformer_blocks[0].attn.to_qkv
        held = qkv.h3_weight_bits

        def storage(t):
            return t.untyped_storage().data_ptr()

        self.assertEqual(storage(qkv.weight), storage(held))
        self.assertTrue(torch.equal(self._decode(vae, z, off)[0], today))
        self._decode(vae, z, _FUSED_FLAGS)
        self.assertIs(qkv.h3_weight_bits, held)
        with torch.no_grad():
            qkv.weight.mul_(2)
        self._decode(vae, z, _FUSED_FLAGS)
        self.assertTrue(torch.equal(qkv.h3_weight_bits.view(torch.float16), qkv.weight))
        with torch.no_grad():
            qkv.weight.div_(2)
        # Short device memory: fp16 weights need none; an int16 copy of a
        # non-fp16 weight does.
        short = (
            patch.object(hd, "free_device_bytes", return_value=1 << 30),
            patch.object(hd, "peak_headroom_bytes", return_value=1 << 30),
            patch.object(hd.logger, "warning"),
        )
        with contextlib.ExitStack() as stack:
            for context in short:
                stack.enter_context(context)
            qkv.weight.data = qkv.weight.data.clone()
            self.assertEqual(self._decode(vae, z, _FUSED_FLAGS)[1], 2)
            self.assertEqual(storage(qkv.weight), storage(qkv.h3_weight_bits))
            qkv.weight.data = qkv.weight.data.float()
            self.assertEqual(self._decode(vae, z, _FUSED_FLAGS)[1], 0)
            qkv.weight.data = qkv.weight.data.half()
        # Compiled blocks run the fused path and match it.
        eager, _ = self._decode(vae, z, _FUSED_FLAGS, count=False)
        self.assertTrue(torch.equal(eager, fused))
        dynamo.reset()
        hd._compiled_vaes.discard(id(vae))
        kwargs = {"backend": "aot_eager", "dynamic": False, "fullgraph": True}
        hd.compile_decoder_blocks(vae, kwargs)
        compiled, _ = self._decode(vae, z, _FUSED_FLAGS, count=False)
        for block in vae.decoder.transformer_blocks:
            self.assertIsNotNone(block._compiled_call_impl)
            block._compiled_call_impl = None
        dynamo.reset()
        self.assertTrue(torch.equal(compiled, eager))

    def test_w2_split_waits_for_room_under_a_stage_peak(self):
        """On a device w2 stays int16 (split in the kernel) until a decode
        stage's peak has left room for its bf16 hi and lo; same output."""
        torch.manual_seed(62)
        z = torch.randn(1, 24, 7, 16, 16)
        vae = _tiny_video_vae(num_layers=2, dim_head=64, heads=4)
        vae.decoder.prepare_autocast_linear_weights(torch.float16)
        w2 = vae.decoder.transformer_blocks[1].ff.w2
        extra = 4 * 2 * w2.weight.numel()  # hi and lo of two blocks' w2
        need = extra + hd.SPLIT_RESERVE_BYTES

        call = hd.unit_call_bytes(vae, z)  # the planned call, one unit
        with patch.object(hd, "peak_headroom_bytes", return_value=need + call):
            hd._hold_fused_operands(vae, z)  # no stage yet: room beside the call
            self.assertEqual(w2.h3_weight_bits.dim(), 3)
        for block in vae.decoder.transformer_blocks:
            block.ff.w2._h3_bits_key = None  # held again, from scratch
        with patch.object(hd, "peak_headroom_bytes", return_value=need):
            first = self._decode(vae, z, _FUSED_FLAGS)[0]
            self.assertEqual(w2.h3_weight_bits.dim(), 2)  # not beside the call
            hd._stage_room = need - 1
            self._decode(vae, z, _FUSED_FLAGS)
            self.assertEqual(w2.h3_weight_bits.dim(), 2)  # a stage left too little
            hd._stage_room = None
            hd._note_stage_room(vae)
            self.assertEqual(hd._stage_room, need)
            split = self._decode(vae, z, _FUSED_FLAGS)[0]
            self.assertEqual(w2.h3_weight_bits.dim(), 3)
        self.assertTrue(torch.equal(split, first))  # the split is exact
        # A stage measured with the split held counts it back.
        with patch.object(hd, "peak_headroom_bytes", return_value=need - extra):
            hd._note_stage_room(vae)
        self.assertEqual(hd._stage_room, need)

    def test_split_weights_held_or_split_per_block_give_the_same_output(self):
        torch.manual_seed(63)
        z = torch.randn(2, 24, 7, 16, 16)
        vae = _tiny_video_vae(num_layers=2, dim_head=64, heads=4)
        vae.decoder.prepare_autocast_linear_weights(torch.float16)
        fused = self._decode(vae, z, _FUSED_FLAGS)[0]
        flags = dict(_FUSED_FLAGS, SGLANG_TPU_OPT_H3_VAE_SPLIT_WEIGHTS=True)
        blocks = vae.decoder.transformer_blocks
        targets = [m for b in blocks for m in (b.attn.to_qkv, b.ff.w1, b.ff.w2)]

        def decode(headroom, free):
            for m in targets:
                m._h3_bits_key = None  # held again under these readings
            with (
                patch.object(hd, "peak_headroom_bytes", return_value=headroom),
                patch.object(hd, "free_device_bytes", return_value=free),
            ):
                output = self._decode(vae, z, flags)[0]
            held = {m.h3_weight_bits.dim() for m in targets}
            in_graph = {m._h3_split_in_graph for m in targets}
            return output, held, in_graph

        # Room under the stage peak: held split. Else split in each block
        # program, or (no free room for one block's copies) in the kernels.
        self.assertEqual(decode(None, None)[1:], ({3}, {False}))
        with patch.object(hd, "peak_headroom_bytes", return_value=1 << 30):
            hd._note_stage_room(vae)  # held splits counted back
        extra = sum(4 * m.weight.numel() for m in targets)
        self.assertEqual(hd._stage_room, (1 << 30) + extra)
        cases = ((1 << 30, 16 << 30, True), (1 << 30, 1 << 30, False))
        for headroom, free, in_graph in cases:
            output, held, split = decode(headroom, free)
            self.assertEqual((held, split), ({2}, {in_graph}))
            self.assertTrue(torch.equal(output, fused), (headroom, free))
        self.assertTrue(torch.equal(decode(None, None)[0], fused))

    def test_compiled_block_holds_only_the_kernels(self):
        """The fused calls take held int16 weights and each other's outputs: no
        dtype view, cast, copy or pad between them in a compiled block."""
        import torch._dynamo as dynamo

        _, block, h, rope = _fused_block(tokens=304)
        ops = tuple(
            dynamo.allow_in_graph(lambda *a, f=f: f(*a)) for f in hd._FUSED_TORCH
        )
        graphs = []

        def backend(gm, inputs):
            graphs.append(gm)
            return gm.forward

        hd._fused_ops, hd._fused_active = ops, True
        rope = (*rope, hd.fused_tables(rope, 304))  # made once a decoder call
        try:
            dynamo.reset()
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
                torch.compile(block, backend=backend, fullgraph=True)(h, rope)
        finally:
            hd._fused_active = False
            dynamo.reset()
        nodes = [n for n in graphs[0].graph.nodes if n.op.startswith("call_")]
        self.assertEqual(sum(n.target in ops for n in nodes), 5)
        # The linears' weights (fp16 views of the held buffers) are no input.
        inputs = [n.name for n in graphs[0].graph.nodes if n.op == "placeholder"]
        self.assertFalse(
            [n for n in inputs if "parameters_weight" in n and "norm" not in n]
        )
        others = [n for n in nodes if n.target not in ops]
        names = {getattr(n.target, "__name__", n.target) for n in others}
        self.assertLessEqual(names, {"getitem"}, names)  # parameters as they are
        for node in others:  # views of shape only, never of dtype
            self.assertFalse(any(isinstance(a, torch.dtype) for a in node.args))
        # int16 weights split in each block program: a split call each more.
        linears = (block.attn.to_qkv, block.ff.w1, block.ff.w2)
        linears = [m for m in linears if m.h3_weight_bits.dim() == 2]
        for m in linears:
            m._h3_split_in_graph = True
        hd._fused_ops, hd._fused_active = ops, True
        try:
            dynamo.reset()
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
                torch.compile(block, backend=backend, fullgraph=True)(h, rope)
        finally:
            hd._fused_active = False
            dynamo.reset()
        nodes = [n for n in graphs[-1].graph.nodes if n.op.startswith("call_")]
        self.assertEqual(sum(n.target is ops[4] for n in nodes), len(linears))
        self.assertEqual(sum(n.target in ops for n in nodes), 5 + len(linears))
        self.assertGreater(len(linears), 1)

    def test_ops_register_with_their_schemas(self):
        try:
            from sglang_tpu.multimodal_gen import minimax_h3_vae_fused as vf

            schemas = [str(op._opoverload._schema) for op in vf.ops()]
        except ImportError:
            self.skipTest("torch_tpu or jax is not installed")
        tensors = "(" + ", ".join(["Tensor"] * 6) + ")"
        self.assertTrue(schemas[0].endswith(f"float q, SymInt half) -> {tensors}"))
        self.assertTrue(schemas[1].endswith("Tensor scale, Tensor h) -> Tensor"))
        self.assertTrue(schemas[2].endswith("float eps) -> Tensor"))
        self.assertTrue(schemas[3].endswith("SymInt seq) -> Tensor"))


class TestFusedLaunches(CustomTestCase):
    """minimax_h3_vae_fused.Launch: every launch of the op bench's SWEEP gives
    the default launch's output bit for bit, and the env's launches reach
    the ops."""

    def setUp(self):
        torch.set_num_threads(4)
        try:
            import jax  # noqa: F401

            from sglang_tpu.multimodal_gen import minimax_h3_vae_fused as vf
        except ImportError:
            self.skipTest("jax with Pallas TPU is not installed")
        self.vf = vf

    def _operands(self):
        """Small shapes that every sweep launch takes: 2 units of 64 rows; gate
        and qkv k 512 (8 heads), gate n 2 x 1024, out n 1024, w2 k 1024."""
        import jax
        import jax.numpy as jnp

        keys = iter(jax.random.split(jax.random.PRNGKey(7), 64))

        def normal(*shape, scale=1.0):
            return scale * jax.random.normal(next(keys), shape, jnp.float32)

        def bits(*shape, scale=0.02):
            return jax.lax.bitcast_convert_type(
                normal(*shape, scale=scale).astype(jnp.float16), jnp.int16
            )

        x, h, nw = normal(2, 64, 512), normal(2, 64, 1024), 1 + normal(512, scale=0.1)
        split = functools.partial(self.vf.split_weight, interpret=True)
        tables = jnp.stack([jnp.cos(normal(64, 128)), jnp.sin(normal(64, 128))])
        parts = [normal(2, 64, 256).astype(jnp.bfloat16) for _ in range(6)]
        return {
            "gate": (x, nw, split(bits(2048, 512)), bits(1, 2048)),
            "out": (x, bits(1024, 512), bits(1, 1024), normal(1024), h),
            "w2": (bits(2, 64, 1024, scale=0.5), split(bits(512, 1024)), bits(1, 512))
            + (normal(512), normal(2, 64, 512)),
            "qkv": (x, nw, split(bits(1536, 512)), bits(8, 512), tables),
            "flash": tuple(parts),
        }

    def _call(self, kernel, launch, args):
        vf = self.vf
        if kernel == "gate":
            return vf.fused_gate(*args, 1e-6, launch=launch, interpret=True)
        if kernel in ("out", "w2"):
            return vf.fused_residual(*args, launch=launch, interpret=True)
        if kernel == "qkv":
            fn = vf.fused_qkv
            return fn(*args, 1e-6, 1e-6, 24, launch=launch, interpret=True)
        call = vf.attention.flash_call(2, 60, 4, rows=64, interpret=True, **launch)
        return call(*args)

    def test_every_sweep_launch_gives_the_default_bits(self):
        import jax
        import numpy as np

        def host_bits(out):
            leaves = jax.tree.leaves(out)
            return [
                np.asarray(o).view({2: np.uint16, 4: np.uint32}[o.dtype.itemsize])
                for o in leaves
            ]

        with jax.default_device(jax.devices("cpu")[0]):
            operands = self._operands()
            for kernel, launches in self.vf.SWEEP.items():
                want = host_bits(self._call(kernel, launches["base"], operands[kernel]))
                for name, launch in launches.items():
                    got = host_bits(self._call(kernel, launch, operands[kernel]))
                    same = all(np.array_equal(g, w) for g, w in zip(got, want))
                    self.assertTrue(same, f"{kernel} {name}")
            # The epilogue's two kernels: gate and up as the gate kernel has them.
            launch = self.vf.SWEEP["gate"]["epi"]
            g, u = self.vf.gate_sums(
                *operands["gate"], 1e-6, launch=launch, interpret=True
            )
            got = host_bits(self.vf.silu_product(g, u, interpret=True))
            base = self._call("gate", self.vf.DEFAULT, operands["gate"])
            self.assertTrue(np.array_equal(got[0], host_bits(base)[0]))

    def test_env_launches_reach_the_ops(self):
        vf = self.vf
        table = vf.parse_launches(
            "gate:blocks=3,tn=1024; w2@4:blocks=5,weight_buffers=3;"
            "out:bounds_checks=0,split_epilogue=0;flash:blocks=2,buffers=3"
        )
        self.assertEqual(table[("gate", None)], vf.Launch(blocks=3, tn=1024))
        self.assertEqual(table[("w2", 4)], vf.Launch(blocks=5, weight_buffers=3))
        self.assertEqual(table[("out", None)], vf.Launch(bounds_checks=False))
        self.assertEqual(table[("flash", None)], {"blocks": 2, "buffers": 3})
        for bad in ("gate:bogus=1", "proj:blocks=1"):
            with self.assertRaises(ValueError):
                vf.parse_launches(bad)
        self.assertEqual(vf.parse_launches(""), {})
        seen = []

        def capture(*args, launch=None, **kwargs):
            seen.append(launch)

        with (
            patch.object(vf, "_LAUNCHES", table),
            patch.object(vf, "fused_residual", side_effect=capture),
            patch.object(vf, "fused_gate", side_effect=capture),
        ):
            self.assertEqual(vf._launch("w2", 3), vf.DEFAULT)  # w2's is for 4 units
            self.assertEqual(vf._launch("flash", 1), {"blocks": 2, "buffers": 3})
            x16, x32 = torch.zeros(4, 8, 8, dtype=torch.int16), torch.zeros(4, 8, 8)
            for x in (x16.numpy(), x32.numpy()):
                vf.residual_jax(x, None, None, None, None)
            vf.gate_jax(x32.numpy(), None, None, None, 1e-6)
        want = [table[("w2", 4)], table[("out", None)], table[("gate", None)]]
        self.assertEqual(seen, want)  # w2 takes the gate's fp16 bits, out f32
        # The ops read the env once, when they register.
        with (
            _flags(SGLANG_TPU_OPT_H3_VAE_FUSED_LAUNCH="qkv:pieces=2"),
            patch.object(vf, "_OPS", None),
            patch.object(vf, "_LAUNCHES", {}),
            patch("sglang_tpu.pallas_jax_op.cpu_export_jax_op", return_value=None),
        ):
            vf.ops()
            self.assertEqual(vf._launch("qkv", 4), vf.Launch(pieces=2))


class TestBatchedDecodeStage(_BatchCase):
    """The stage forward with batched decodes against the stock one, 4 ranks,
    both on the batch-invariant stand-in decoder (so frames must match)."""

    world = 4

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.server_args = _server_args()
        cls.video_vae = _tiny_video_vae()
        cls.audio_vae = _tiny_audio_vae()
        torch.manual_seed(18)
        # 2 clips x 3x3 tiles, 39 frames, as TestDecodeStage.
        cls.latents = torch.randn(1, 24, 12, 32, 40)
        cls.audio_latents = torch.randn(2, 32, 9)
        cls.tmp = tempfile.mkdtemp(prefix="h3dec2_")
        cls.stock = cls._run(dict(), "stock.mp4")

    @classmethod
    def _run(cls, flags, name, log=None):
        log = log or _CallLog()
        with log.patch():
            run = TestDecodeStage._run_world.__func__(cls, False, name, flags=flags)
        run.sizes = log.sizes
        return run

    def _assert_same(self, run, *, streamed):
        for error in run.errors:
            if error is not None:
                raise error
        for rank in range(self.world):
            stock, ours = self.stock.outputs[rank][0], run.outputs[rank][0]
            self.assertTrue(torch.equal(stock.output, ours.output), rank)
            self.assertTrue(torch.equal(stock.audio, ours.audio), rank)
        self.assertTrue(filecmp.cmp(self.stock.path, run.path, shallow=False))
        self.assertEqual(run.encodes, 1 if streamed else self.stock.encodes)
        self.assertEqual(hd._SAVES, {})

    def test_batched_stock_order(self):
        run = self._run({"SGLANG_TPU_OPT_H3_VAE_DECODE_BATCH": 4}, "batch.mp4")
        self._assert_same(run, streamed=False)
        self.assertGreater(max(run.sizes), 1)

    def test_batched_with_flat_audio_first_and_stream(self):
        for own in (1, 2):
            run = self._run(
                {
                    **_ALL_ON,
                    "SGLANG_TPU_OPT_H3_VAE_DECODE_BATCH": 4,
                    "SGLANG_TPU_OPT_H3_AUDIO_RANK_WAVE_UNITS": own,
                },
                f"batch_h3s_{own}.mp4",
            )
            self._assert_same(run, streamed=True)
            self.assertGreater(max(run.sizes), 1)
            self.assertLessEqual(max(run.sizes), 4)

    def test_batched_failure_falls_back_with_the_same_video(self):
        log = _CallLog(fail_batched=True)
        with patch.object(hd.logger, "warning") as warning:
            run = self._run(
                {**_ALL_ON, "SGLANG_TPU_OPT_H3_VAE_DECODE_BATCH": 4},
                "batch_fail.mp4",
                log,
            )
        self._assert_same(run, streamed=True)
        self.assertTrue(
            any("unit by unit" in call.args[0] for call in warning.call_args_list)
        )

    def test_short_memory_decodes_unit_by_unit_with_the_same_video(self):
        with patch.object(hd, "free_device_bytes", return_value=0):
            run = self._run(
                {**_ALL_ON, "SGLANG_TPU_OPT_H3_VAE_DECODE_BATCH": 4}, "batch_mem.mp4"
            )
        self._assert_same(run, streamed=True)
        self.assertEqual(set(run.sizes), {1})

    def test_status_lines_hold_each_ranks_call_sizes(self):
        for name, flags, streamed in (
            ("status_stock.mp4", {"SGLANG_TPU_OPT_H3_VAE_DECODE_BATCH": 4}, False),
            (
                "status_h3s.mp4",
                {**_ALL_ON, "SGLANG_TPU_OPT_H3_VAE_DECODE_BATCH": 4},
                True,
            ),
        ):
            log = _CallLog(wrapped=True)
            with patch.object(hd.logger, "info") as info:
                run = self._run(flags, name, log)
            self._assert_same(run, streamed=streamed)
            rows = _status_rows(_messages(info))
            self.assertEqual(sorted(rows), list(range(self.world)), name)
            for rank, row in rows.items():
                self.assertEqual(row["calls"], log.by_rank[rank], (name, rank))
                self.assertEqual(int(row["units"]), sum(log.by_rank[rank]))
                self.assertEqual((row["cap"], row["cuts"]), ("4", "0"), name)
                self.assertEqual(row["fallbacks"], "none", name)
                self.assertEqual(row["peak"], "n/a")  # no device peak on CPU
            # 2 clips x 3x3 tiles, each decoded once over the ranks.
            self.assertEqual(sum(sum(r["calls"]) for r in rows.values()), 18)
            self.assertGreater(max(max(r["calls"]) for r in rows.values()), 1)

    def test_status_lines_show_calls_free_memory_cut(self):
        latent = torch.zeros(1, 24, 7, 16, 16)
        per_unit = hd.unit_call_bytes(self.video_vae, latent)
        free = hd.DECODE_BATCH_RESERVE_BYTES + 2 * per_unit + per_unit // 2
        log = _CallLog(wrapped=True)
        with (
            patch.object(hd, "free_device_bytes", return_value=free),
            patch.object(hd.logger, "info") as info,
        ):
            run = self._run({"SGLANG_TPU_OPT_H3_VAE_DECODE_BATCH": 4}, "cut.mp4", log)
        self._assert_same(run, streamed=False)
        rows = _status_rows(_messages(info))
        self.assertEqual(sorted(rows), list(range(self.world)))
        for rank, row in rows.items():
            self.assertEqual(row["calls"], log.by_rank[rank], rank)
            self.assertLessEqual(max(row["calls"]), 2)
            cuts = [
                tuple(map(int, c.split("->")))
                for c in (row["cut"] or "").split(", ")
                if c
            ]
            self.assertEqual(len(cuts), int(row["cuts"]))
            self.assertTrue(all(taken == 2 < planned for planned, taken in cuts), cuts)
            self.assertEqual(row["free"], f"{free / 2**30:.2f} GiB")
        # 9 tiles a clip over 4 ranks (3/2/2/2): rank 0's calls of 3 were cut.
        self.assertGreater(int(rows[0]["cuts"]), 0)

    def test_every_rank_logs_its_lines(self):
        log = _CallLog(fail_batched=True, wrapped=True)
        with (
            _non_main_rank(),
            self.assertLogs(hd.logger.name, level="INFO") as logs,
        ):
            hd.logger.info("control: the logger's default keeps main-process lines")
            run = self._run(
                {**_ALL_ON, "SGLANG_TPU_OPT_H3_VAE_DECODE_BATCH": 4},
                "ranks.mp4",
                log,
            )
        self._assert_same(run, streamed=True)
        messages = [record.getMessage() for record in logs.records]
        self.assertFalse(any(m.startswith("control:") for m in messages))
        rows = _status_rows(messages)
        self.assertEqual(sorted(rows), list(range(self.world)))
        self.assertTrue(any("first call of" in m for m in messages))
        self.assertTrue(any("decoding unit by unit" in m for m in messages))
        failed = [r for r in rows.values() if "batched call of" in r["fallbacks"]]
        self.assertTrue(failed, rows)
        for row in failed:
            self.assertEqual(row["cap"], "1")


class TestRankStatus(_BatchCase):
    """DecodeStatus: what the unit decode records, and when stages log it."""

    def _unit_calls(self, count, cap, *, free=None, fail_batched=False):
        vae = _tiny_video_vae()
        torch.manual_seed(25)
        latents = [torch.randn(1, 24, 7, 16, 16) for _ in range(count)]
        status = hd.DecodeStatus("req")
        token = hd._STATUS.set(status)
        try:
            with (
                _CallLog(fail_batched=fail_batched, wrapped=True).patch(),
                patch.object(hd, "free_device_bytes", return_value=free),
                patch.object(hd.logger, "warning"),
                torch.no_grad(),
            ):
                list(hd._unit_calls(vae, latents, cap))
        finally:
            hd._STATUS.reset(token)
        return vae, status

    def test_runs(self):
        self.assertEqual(hd.runs([]), "")
        self.assertEqual(hd.runs([1, 4, 4, 4, 3, 3, 1]), "1, 4x3, 3x2, 1")
        for sizes in ([4], [1, 1], [3, 4, 4, 2, 2, 2, 4]):
            self.assertEqual(_expand(hd.runs(sizes)), sizes)

    def test_calls_cuts_and_least_free_memory(self):
        vae, status = self._unit_calls(7, 4)
        self.assertEqual((status.calls, status.cuts), ([4, 3], []))
        self.assertIsNone(status.min_free)
        per_unit = hd.unit_call_bytes(vae, torch.zeros(1, 24, 7, 16, 16))
        free = hd.DECODE_BATCH_RESERVE_BYTES + 2 * per_unit + per_unit // 2
        _, status = self._unit_calls(7, 4, free=free)
        self.assertEqual(status.calls, [2, 2, 2, 1])
        self.assertEqual(status.cuts, [(4, 2), (3, 2), (3, 2)])
        self.assertEqual(status.min_free, free)
        self.assertEqual(status.events, [])

    def test_failed_batched_call_is_a_fallback_event(self):
        _, status = self._unit_calls(8, 4, fail_batched=True)
        self.assertEqual(status.calls, [1] * 8)
        self.assertEqual(len(status.events), 1)
        self.assertIn("batched call of 4 units failed", status.events[0])
        self.assertIn("planted batched failure", status.events[0])

    def test_status_line(self):
        status = hd.DecodeStatus("rid", warmup=True)
        status.calls[:] = [1, 4, 4, 3]
        status.cuts.append((4, 3))
        status.saw_free(5 << 30)
        status.saw_free(None)
        status.saw_free(7 << 30)
        status.events.append("compiled ViT decoder blocks dropped (planted)")
        with (
            _flags(SGLANG_TPU_OPT_H3_VAE_DECODE_BATCH=4),
            patch.object(hd.logger, "info") as info,
        ):
            status.log(3, (1024.0, 512.0))
        self.assertEqual(info.call_args.kwargs, hd._ALL_RANKS)
        row = _status_rows(_messages(info))[3]
        self.assertEqual(row["request"], "rid")
        self.assertTrue(row["warmup"])
        self.assertEqual((row["units"], row["calls"]), ("12", [1, 4, 4, 3]))
        self.assertEqual((row["cap"], row["cuts"], row["cut"]), ("4", "1", "4->3"))
        self.assertEqual(row["free"], "5.00 GiB")
        self.assertEqual(
            row["fallbacks"], "compiled ViT decoder blocks dropped (planted)"
        )
        self.assertEqual(row["peak"], "1024 MiB reserved, 512 MiB allocated")

    def test_stage_logs_a_status_line_only_with_decoder_flags(self):
        stage = SimpleNamespace(video_vae=None)
        batch = SimpleNamespace(
            latents=torch.zeros(1), request_id="rid", is_warmup=False
        )

        def forward(stage_, batch_, server_args):
            status = hd._STATUS.get()
            if status is not None:
                status.calls.append(2)
            if server_args == "raise":
                raise RuntimeError("planted stage failure")

        wrapped = hd._decode_stage_forward(forward)
        for flags, logged in (
            ({}, False),
            ({"SGLANG_TPU_ENABLE_H3_VAE_FLAT_DECODE": True}, False),
            ({"SGLANG_TPU_OPT_H3_VAE_DECODE_BATCH": 1}, False),
            ({"SGLANG_TPU_OPT_H3_VAE_DECODE_BATCH": 4}, True),
            ({"SGLANG_TPU_ENABLE_H3_VAE_BLOCK_COMPILE": True}, True),
            ({"SGLANG_TPU_ENABLE_H3_VAE_FP32_PROJECTIONS": True}, True),
            ({"SGLANG_TPU_OPT_H3_VAE_FLASH_ATTENTION": True}, True),
        ):
            with (
                _flags(**flags),
                patch.object(hd, "compile_decoder_blocks"),
                patch.object(hd.logger, "info") as info,
            ):
                self.assertEqual(hd._status_wanted(), logged, flags)
                wrapped(stage, batch, None)
                # A stage call that raises still logs its line.
                with self.assertRaises(RuntimeError):
                    wrapped(stage, batch, "raise")
            rows = [m for m in _messages(info) if "decode status" in m]
            self.assertEqual(len(rows), 2 if logged else 0, flags)
            if logged:
                self.assertIn("decode rank 0: 2 units in decoder calls of [2]", rows[0])
            self.assertIsNone(hd._STATUS.get())
        hd.refresh_precision_flags()

    def test_peak_off_device_and_on(self):
        self.assertIsNone(hd.peak_mib(torch.device("cpu")))
        module = SimpleNamespace(
            max_memory_reserved=lambda: 3 << 20, max_memory_allocated=lambda: 1 << 20
        )
        with patch.object(torch, "get_device_module", return_value=module):
            self.assertEqual(hd.peak_mib(torch.device("meta")), (3.0, 1.0))
        with patch.object(torch, "get_device_module", side_effect=RuntimeError("no")):
            self.assertIsNone(hd.peak_mib(torch.device("meta")))

    def test_one_off_lines_reach_every_rank(self):
        torch._dynamo.reset()
        hd._compiled_vaes.clear()

        def broken_backend(gm, example_inputs):
            raise RuntimeError("planted compile failure")

        broken = {"backend": broken_backend, "dynamic": False, "fullgraph": True}
        z = torch.randn(1, 24, 7, 16, 16)
        try:
            with (
                _non_main_rank(),
                self.assertLogs(hd.logger.name, level="INFO") as logs,
                torch.no_grad(),
            ):
                hd.logger.info("control: the logger's default keeps main-process lines")
                blocks = _tiny_video_vae()
                hd.compile_decoder_blocks(blocks, broken)
                blocks.decode(z)
                hd._warn_once("memory", "planted memory line")
        finally:
            torch._dynamo.reset()
        messages = [record.getMessage() for record in logs.records]
        self.assertFalse(any(m.startswith("control:") for m in messages))
        for expected in (
            "compiling 1 ViT decoder blocks",
            "compiled ViT decoder block failed",
            "planted memory line",
        ):
            self.assertTrue(any(expected in m for m in messages), expected)


class TestInstall(CustomTestCase):
    def test_requested_and_installed_once(self):
        self.assertFalse(hd.requested())
        for name, value in (
            ("SGLANG_TPU_ENABLE_H3_STREAM_SAVE", True),
            ("SGLANG_TPU_OPT_H3_VAE_DECODE_BATCH", 4),
            ("SGLANG_TPU_ENABLE_H3_VAE_FP32_PROJECTIONS", True),
            ("SGLANG_TPU_OPT_H3_VAE_FLASH_ATTENTION", True),
        ):
            with _flags(**{name: value}):
                self.assertTrue(hd.requested(), name)
        decode_temporal = MiniMaxH3VideoVAE.decode_temporal
        hd.install()
        self.assertIs(MiniMaxH3VideoVAE.decode_temporal, decode_temporal)
        self.assertTrue(getattr(gpu_worker.save_outputs, "__wrapped_for_tpu__", False))
        self.assertTrue(
            getattr(
                decoding.MiniMaxH3DecodingStage.forward, "__wrapped_for_tpu__", False
            )
        )

    def test_worker_installs_when_requested(self):
        from sglang_tpu.multimodal_gen import worker

        with (
            patch.object(hd, "install") as install,
            patch("sglang_tpu.multimodal_gen.rope.install"),
            patch("sglang_tpu.multimodal_gen.minimax_h3.install"),
            patch("sglang_tpu.patches._register_aten_fallbacks"),
            patch.object(torch, "set_float32_matmul_precision"),
        ):
            worker.install_runtime_patches()
            install.assert_not_called()
            with _flags(SGLANG_TPU_ENABLE_H3_VAE_FLAT_DECODE=True):
                worker.install_runtime_patches()
            install.assert_called_once()


if __name__ == "__main__":
    unittest.main()
