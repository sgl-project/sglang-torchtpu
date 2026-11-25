# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 decode stage on TPU workers: flat unit schedule, audio first, mp4 streaming.

Runtime patch, tracked debt:
- Why: after the denoise loop an H3 request still runs its decode stage as
  four serial parts on v7x-8. Each clip's 28 tiles split 4/3 over 8 ranks
  with an all-gather a clip, so the busiest rank decodes 28 of the 196
  (clip, tile) units at 1344x768x124 where 24.5 is the even share; the audio
  VAE runs on rank 0 only after the video, while the other ranks wait; the
  36-block ViT decoder dispatches ~2,800 eager ops a unit; and the mp4 encode
  (libx264 + AAC, about a second on the host) starts only once every frame is
  decoded and copied to the host.
- How: :func:`install` wraps, each behind its env flag (all off by default):
  (1) ``MiniMaxH3VideoVAE.decode_temporal``: every unit of a video decode is
      dealt over the decode group in one list (:func:`plan_waves`), the
      outputs are gathered in waves of about a clip, and the stock
      ``decode_temporal`` replays with ``_adaptive_decode`` answering each clip
      from the gathered units: the stock slicing, decode calls, assembly and
      blending, so the frames are bit-identical;
  (2) ``MiniMaxH3DecodingStage``: with resident VAEs, the audio owner (replica
      rank 0) decodes the audio inside (1), after its first unit, while each
      other rank decodes its larger share of the first wave; the stage's own
      ``_decode_audio`` call then returns that result (the same call on the
      same rank and inputs, autocast off as outside the video decode);
  (3) with (2), the output rank passes each clip's finished frames through the
      stage's pixel steps to the stock ``save_materialized_output`` (imageio,
      libx264, AAC) on a writer thread while later clips decode; the worker's
      ``save_outputs`` returns that file (the same call on the same bytes);
  (4) each ViT decoder block is torch.compiled (static shapes, fullgraph; the
      blocks run eagerly again if a compiled call raises). This moves rounding
      within the fp16-autocast decode, so it is not bit-identical;
  (5) each rank decodes its units of a clip (``_run_tile_tasks``) or of a
      wave (1) in near-equal ``vae.decode`` calls of up to N units stacked on
      the batch dimension instead of one call a unit (~1,240 device programs a
      unit in eager mode): the same tiles, assembly, blend and fp16 autocast,
      only the matmul shapes change, so rounding moves. A call takes only as
      many units as the device's free memory covers;
  (6) the ViT decoder's input and output projections (x_embedder, proj_out)
      run outside autocast on any device, as the released decoder runs them
      (``torch.autocast("cuda", enabled=False)``): sglang's
      ``_cuda_autocast_disabled`` turns autocast off for CUDA tensors only, so
      on TPU both run with fp16 operands. Moves the decode toward the released
      precision, so it is gated on a decode against fp32;
  (7) the ViT attention runs, inside a decode with fp16 q/k/v and a decoder
      whose q/k norms are parameter-free RMSNorms over 64-wide heads, as a
      Pallas flash kernel (minimax_h3_vae_attention: scores kept in VMEM,
      the three bf16 products of an fp16 dot left in float32, a float32
      softmax with the norm bound as its shift, all four products of P V, one
      fp16 rounding) instead of torch_tpu's fp16 SDPA (no flash path; it
      writes and rounds every score). Rounding moves, at least as precise as
      the fp16 SDPA; gated on a decode against fp32;
  (8) the ViT block linears run under the fp16 autocast as a Pallas kernel
      (minimax_h3_vae_linear: fp16 activations and weights split exactly
      into bf16 hi + lo in VMEM, XLA's three MXU products, the bias added in
      float32, one fp16 rounding) instead of XLA's fp16 dot, which streams
      the activations as unpacked fp32 rows and rounds before the bias add.
      Rounding moves, at least as precise as today's fp16 linear; gated on a
      decode against fp32;
  (9) with (7) and (8), each ViT block runs as five custom calls
      (minimax_h3_vae_fused and the flash kernel) with its norms, RoPE,
      gating and residual adds inside, over int16 copies of the block
      weights that the fp16 weights then view (w2 also as bf16 hi + lo,
      2.4 GB, where a decode stage leaves room) and a residual stream padded
      to 1,800 tokens a unit; one fp16 rounding a block where the stock
      block has 13; gated on a decode against fp32.
  With any of (4)-(9), each rank logs one status line a stage call: its
  decoder-call sizes, the calls free memory cut, fallbacks, and its peak.
  While installed, the stage's own encode-while-decoding (sglang#41819) stays
  off: the mp4 is written by (3), or after the decode by the worker's save.
  A ``decode_temporal`` call that still passes ``on_frames`` runs the stock
  decode.
- Upstream PR or issue: none.
- Remove when: the decode group schedules units across clips, the audio decode
  can overlap the video decode, VAEs have a platform compile path and batch
  their tiles within a memory budget, the mp4 writer takes frames as they are
  decoded, the H3 decoder disables autocast for its projections on every
  device, torch_tpu has an fp16 flash SDPA at least as precise as its fp16
  math path, XLA's fp16 dot runs at bf16 MXU speed with one rounding, and
  XLA fuses the ViT block's norms, RoPE and residual adds into its linears.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
import math
import os
import queue
import threading
import types
from collections.abc import Callable, Sequence
from typing import Any

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang_tpu.envs import envs as tpu_envs

logger = init_logger(__name__)

# Seconds the worker's save waits for the streamed mp4 before encoding again.
STREAM_SAVE_TIMEOUT_S = 600.0

_FLAGS = (
    "SGLANG_TPU_ENABLE_H3_VAE_FLAT_DECODE",
    "SGLANG_TPU_ENABLE_H3_AUDIO_FIRST",
    "SGLANG_TPU_ENABLE_H3_STREAM_SAVE",
    "SGLANG_TPU_ENABLE_H3_VAE_BLOCK_COMPILE",
    "SGLANG_TPU_OPT_H3_VAE_DECODE_BATCH",
    "SGLANG_TPU_ENABLE_H3_VAE_FP32_PROJECTIONS",
    "SGLANG_TPU_OPT_H3_VAE_FLASH_ATTENTION",
    "SGLANG_TPU_OPT_H3_VAE_PALLAS_LINEAR",
    "SGLANG_TPU_OPT_H3_VAE_FUSED_LINEARS",
    "SGLANG_TPU_OPT_H3_VAE_SPLIT_WEIGHTS",
)

# Keyword arguments that make the logger print on every rank: its info and
# warning defaults keep only the (local) main process's lines, so the other
# ranks' would never reach the server log.
_ALL_RANKS = {"main_process_only": False, "local_main_process_only": False}

_installed = False
_stock: dict[str, Callable] = {}


def requested() -> bool:
    return any(getattr(tpu_envs, name).get() for name in _FLAGS)


def _on(name: str) -> str:
    return "on" if getattr(tpu_envs, name).get() else "off"


def install() -> None:
    """Wrap the H3 decode stage, video VAE, ViT decoder and worker save once."""
    global _installed
    if _installed:
        return
    from sglang.multimodal_gen.runtime.managers import gpu_worker
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import (
        MiniMaxH3VideoVAE,
    )
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
        attention as vae_attention,
    )
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
        base_module,
        vae_vit,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (
        decoding,
    )

    ViT3DDecoder = vae_vit.ViT3DDecoder
    MiniMaxH3DecodingStage = decoding.MiniMaxH3DecodingStage
    _stock["sdpa"] = vae_attention._sdpa_attention
    _stock["decode_temporal"] = MiniMaxH3VideoVAE.decode_temporal
    _stock["run_tile_tasks"] = MiniMaxH3VideoVAE._run_tile_tasks
    _stock["vae_decode"] = MiniMaxH3VideoVAE.decode
    _stock["forward"] = MiniMaxH3DecodingStage.forward
    _stock["decode_audio"] = MiniMaxH3DecodingStage._decode_audio
    _stock["forward_transformer_blocks"] = ViT3DDecoder.forward_transformer_blocks
    _stock["block_forward"] = base_module.TransformerBlock.forward
    _stock["save_outputs"] = gpu_worker.save_outputs
    _stock["projection_autocast"] = vae_vit._cuda_autocast_disabled
    _stock["streamed_video_output_path"] = decoding.streamed_video_output_path

    MiniMaxH3VideoVAE.decode_temporal = _flat_decode_temporal(_stock["decode_temporal"])
    MiniMaxH3VideoVAE._run_tile_tasks = _batched_run_tile_tasks(
        _stock["run_tile_tasks"]
    )
    MiniMaxH3VideoVAE.decode = _routed_decode(_stock["vae_decode"])
    MiniMaxH3DecodingStage.forward = _decode_stage_forward(_stock["forward"])
    decoding.streamed_video_output_path = _no_stage_stream(
        _stock["streamed_video_output_path"]
    )
    MiniMaxH3DecodingStage._decode_audio = _decode_audio(_stock["decode_audio"])
    ViT3DDecoder.forward_transformer_blocks = _padded_for_fused_blocks(
        _eager_on_compile_failure(_stock["forward_transformer_blocks"])
    )
    base_module.TransformerBlock.forward = _fused_block_forward(_stock["block_forward"])
    gpu_worker.save_outputs = _save_outputs(_stock["save_outputs"])
    vae_attention._sdpa_attention = _flash_sdpa(_stock["sdpa"])
    vae_vit._cuda_autocast_disabled = _projection_autocast(
        _stock["projection_autocast"]
    )
    refresh_precision_flags()
    _installed = True
    logger.info(
        "MiniMax H3 decode: flat unit schedule %s, audio first %s (%d units a rank "
        "in the audio wave, %d on the audio owner), streamed mp4 save %s, ViT "
        "block compile %s, %d units a decoder call, fp32 ViT projections %s, "
        "flash ViT attention %s, Pallas ViT linears %s, fused ViT blocks %s, "
        "split ViT weights %s",
        _on("SGLANG_TPU_ENABLE_H3_VAE_FLAT_DECODE"),
        _on("SGLANG_TPU_ENABLE_H3_AUDIO_FIRST"),
        tpu_envs.SGLANG_TPU_OPT_H3_AUDIO_WAVE_UNITS.get(),
        tpu_envs.SGLANG_TPU_OPT_H3_AUDIO_RANK_WAVE_UNITS.get(),
        _on("SGLANG_TPU_ENABLE_H3_STREAM_SAVE"),
        _on("SGLANG_TPU_ENABLE_H3_VAE_BLOCK_COMPILE"),
        decode_batch_units(),
        "on" if _fp32_projections else "off",
        "on" if _flash_attention else "off",
        "on" if _pallas_linear else "off",
        "on" if _fused_linears else "off",
        "on" if _split_weights else "off",
        **_ALL_RANKS,
    )


# ---------------------------------------------------------------------------
# Flat (clip, tile) unit schedule
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class UnitPlan:
    """Every (clip, tile) unit of one decode_temporal call, in stock order."""

    clips: list[torch.Tensor]
    # Per clip: (rows, columns, y_overlap, x_overlap) as tiled_decode uses them.
    layouts: list[tuple[int, int, list[int], list[int]]]
    units: list[torch.Tensor]
    # First unit of each clip.
    offsets: list[int]
    # The padded latent and chunk count decode_temporal derives (frame plan).
    padded: torch.Tensor
    num_chunks: int
    pad_tokens: int

    @property
    def tiles_per_clip(self) -> int:
        rows, cols, _, _ = self.layouts[0]
        return rows * cols


def _clip_inputs(vae, z: torch.Tensor):
    """decode_temporal's padded latent, chunk count and clip latents."""
    chunk = vae.tokens_chunk_size
    pseudo_total_tokens = z.shape[2] + vae.token_drop
    pad_tokens = (-pseudo_total_tokens) % chunk
    if pad_tokens:
        pad_z = z[:, :, -1:, :, :].expand(-1, -1, pad_tokens, -1, -1)
        z = torch.cat([z, pad_z], dim=2)
    num_chunks = (pseudo_total_tokens + pad_tokens) // chunk - int(vae.token_drop > 0)
    clips = [
        z[:, :, i * chunk : i * chunk + chunk + vae.token_overlap, :, :]
        for i in range(num_chunks)
    ]
    return z, num_chunks, pad_tokens, clips


def _tile_inputs(vae, clip_z: torch.Tensor):
    """tiled_decode's tiles of one clip and its assembly layout."""
    ratio = vae.vae_ratio
    y_idx, y_len, y_overlap = vae.split_tiles(clip_z.shape[-2] * ratio, True)
    x_idx, x_len, x_overlap = vae.split_tiles(clip_z.shape[-1] * ratio, True)
    tiles = []
    for i_pos, i_len in zip(y_idx, y_len):
        i_pos, i_len = i_pos // ratio, i_len // ratio
        for j_pos, j_len in zip(x_idx, x_len):
            j_pos, j_len = j_pos // ratio, j_len // ratio
            tiles.append(clip_z[..., i_pos : i_pos + i_len, j_pos : j_pos + j_len])
    return tiles, (len(y_idx), len(x_idx), y_overlap, x_overlap)


def plan_units(vae, z: torch.Tensor, world: int) -> UnitPlan | None:
    """The units of ``z``'s decode, or None where the stock path must run."""
    if (
        world <= 1
        or vae.training
        or not vae.decoder_tiling
        or not vae.parallel_tiling
        or vae._batch_caps() != (1, 1)
        or vae.isolated_first_frame
        or vae.isolated_last_frame
    ):
        return None
    padded, num_chunks, pad_tokens, clips = _clip_inputs(vae, z)
    if num_chunks <= 0:
        return None
    layouts, units, offsets = [], [], []
    for clip_z in clips:
        tiles, layout = _tile_inputs(vae, clip_z)
        offsets.append(len(units))
        units.extend(tiles)
        layouts.append(layout)
    if len(units) < world or len({tuple(u.shape) for u in units}) != 1:
        return None
    if len({rows * cols for rows, cols, _, _ in layouts}) != 1:
        return None
    return UnitPlan(clips, layouts, units, offsets, padded, num_chunks, pad_tokens)


@dataclasses.dataclass(frozen=True)
class Wave:
    """Units [start, stop) gathered together; ``counts[r]`` of them are rank r's."""

    start: int
    stop: int
    counts: tuple[int, ...]

    def owners(self) -> list[int]:
        """The rank of each unit: dealt in turn to the ranks with units left."""
        left = list(self.counts)
        owners: list[int] = []
        rank = 0
        for _ in range(self.stop - self.start):
            while left[rank] == 0:
                rank = (rank + 1) % len(left)
            owners.append(rank)
            left[rank] -= 1
            rank = (rank + 1) % len(left)
        return owners

    def units_of(self, rank: int) -> list[int]:
        return [
            self.start + i for i, owner in enumerate(self.owners()) if owner == rank
        ]


def _deal(num: int, ranks: Sequence[int], world: int) -> list[int]:
    counts = [0] * world
    for k, rank in enumerate(ranks):
        counts[rank] = num // len(ranks) + int(k < num % len(ranks))
    return counts


def plan_waves(
    num_units: int,
    tiles_per_clip: int,
    world: int,
    *,
    audio_rank: int | None = None,
    audio_wave_units: int = 0,
    audio_rank_units: int = 1,
) -> list[Wave]:
    """The gather waves of a flat decode.

    Without an audio rank: waves of ``world * ceil(tiles_per_clip / world)``
    units (about a clip) dealt in turn over every rank, a tail shorter than one
    unit a rank joining the wave before it. With one: the first wave gives the
    audio rank ``audio_rank_units`` units (by default one; its first is decoded
    before its audio, which also makes its padding shape known) and every
    other rank ``audio_wave_units`` units (at least a clip's share), so the
    audio decodes while they do.
    """
    if world < 2 or num_units < world:
        raise ValueError(f"{num_units} units cannot be dealt over {world} ranks")
    per_rank = math.ceil(tiles_per_clip / world)
    waves: list[Wave] = []
    start = 0
    if audio_rank is not None:
        if not 0 <= audio_rank < world:
            raise ValueError(f"audio rank {audio_rank} outside {world} ranks")
        per_other = max(int(audio_wave_units), per_rank, 1)
        own = min(max(int(audio_rank_units), 1), per_other)
        size = min(num_units, (world - 1) * per_other + own)
        # Every other rank keeps at least one unit of the wave.
        own = max(1, min(own, size - (world - 1)))
        others = [rank for rank in range(world) if rank != audio_rank]
        counts = _deal(size - own, others, world)
        counts[audio_rank] = own
        waves.append(Wave(0, size, tuple(counts)))
        start = size
    step = world * per_rank
    bounds = list(range(start, num_units, step)) + [num_units]
    regular = list(zip(bounds[:-1], bounds[1:]))
    if len(regular) > 1 and regular[-1][1] - regular[-1][0] < world:
        tail = regular.pop()
        regular[-1] = (regular[-1][0], tail[1])
    for lo, hi in regular:
        waves.append(Wave(lo, hi, tuple(_deal(hi - lo, range(world), world))))
    return waves


def run_flat_decode(
    vae,
    z: torch.Tensor,
    plan: UnitPlan,
    waves: Sequence[Wave],
    decode_temporal: Callable[[object, torch.Tensor], torch.Tensor],
    rank: int,
    world: int,
    all_gather: Callable[[torch.Tensor], Sequence[torch.Tensor]],
    *,
    after_first_unit: Callable[[], None] | None = None,
    on_clip: Callable[[int, torch.Tensor, bool], None] | None = None,
    batch: int = 1,
) -> torch.Tensor:
    """Decode this rank's units wave by wave and replay the stock assembly.

    ``decode_temporal`` (the stock one) runs with ``_adaptive_decode``
    answering each clip from the gathered units, so slicing, blending code and
    order are the stock ones. With ``batch`` 1 each unit runs alone through
    ``vae.decode`` as in tiled_decode, and every rank's outputs come from
    equal-shape all-gathers, so the result is bit-identical to the stock path;
    a larger ``batch`` decodes a rank's units of a wave in calls of up to that
    many (:func:`_unit_calls`), which moves only rounding. ``after_first_unit``
    runs once, after this rank's first unit of the first wave (decoded alone);
    ``on_clip`` sees each assembled clip before decode_temporal does.
    """
    results: dict[int, torch.Tensor] = {}
    state = {"wave": 0, "clip": 0}
    template: list[torch.Tensor] = []
    # Batched decode: (shape, dtype, device) of one unit's output, to pad a
    # wave in which this rank has no unit.
    unit_meta: list[tuple] = []

    def decode_wave_batched(index: int, mine: list[int], width: int):
        hook = after_first_unit if index == 0 else None
        stacked = None
        for start, out in _unit_calls(
            vae,
            [plan.units[unit] for unit in mine],
            batch,
            first_alone=hook is not None,
        ):
            if start == 0 and hook is not None:
                hook()
            if stacked is None and out.shape[0] == width:
                # One call holds the rank's whole wave: gather its output as is.
                stacked = out.unsqueeze(1)
                continue
            if stacked is None:
                stacked = out.new_empty((width, 1, *out.shape[1:]))
            stacked[start : start + out.shape[0]].copy_(out.unsqueeze(1))
        if hook is not None and not mine:
            hook()
        if stacked is None:
            if not unit_meta:
                raise RuntimeError(
                    f"MiniMax H3 flat decode: rank {rank} has no unit to pad wave {index}"
                )
            shape, dtype, device = unit_meta[0]
            return torch.zeros((width, *shape), dtype=dtype, device=device)
        unit_meta[:] = [(tuple(stacked.shape[1:]), stacked.dtype, stacked.device)]
        if len(mine) < width:
            stacked[len(mine) :].zero_()
        return stacked

    def decode_wave(index: int, wave: Wave) -> None:
        owners = wave.owners()
        mine = [wave.start + i for i, owner in enumerate(owners) if owner == rank]
        width = max(wave.counts)
        audio = index == 0 and after_first_unit is not None
        if batch > 1:
            stacked = decode_wave_batched(index, mine, width)
            gathered = all_gather(stacked)
            del stacked
        else:
            local = []
            for k, unit in enumerate(mine):
                local.append(vae.decode(plan.units[unit]))
                if k == 0 and audio:
                    after_first_unit()
            if index == 0 and not mine and after_first_unit is not None:
                after_first_unit()
            if local:
                template[:] = local[:1]
            if not template:
                raise RuntimeError(
                    f"MiniMax H3 flat decode: rank {rank} has no unit to pad wave {index}"
                )
            if len(local) == width:
                stacked = torch.stack(local, dim=0)
            else:
                stacked = template[0].new_empty((width, *template[0].shape))
                if local:
                    torch.stack(local, dim=0, out=stacked[: len(local)])
                stacked[len(local) :].zero_()
            del local
            gathered = all_gather(stacked)
        for src, tensors in enumerate(gathered):
            src_units = [
                wave.start + i for i, owner in enumerate(owners) if owner == src
            ]
            for k, unit in enumerate(src_units):
                results[unit] = tensors[k]

    def gather_through(stop: int) -> None:
        while state["wave"] < len(waves) and waves[state["wave"]].start < stop:
            index = state["wave"]
            state["wave"] += 1
            decode_wave(index, waves[index])

    def replay(clip_z: torch.Tensor) -> torch.Tensor:
        clip = state["clip"]
        if clip >= len(plan.clips) or clip_z.shape != plan.clips[clip].shape:
            raise RuntimeError(
                f"MiniMax H3 flat decode: clip {clip} of shape {tuple(clip_z.shape)} "
                f"does not match the plan ({len(plan.clips)} clips)"
            )
        rows, cols, y_overlap, x_overlap = plan.layouts[clip]
        start = plan.offsets[clip]
        gather_through(start + rows * cols)
        tiles = [
            [results.pop(start + i * cols + j).to(clip_z.device) for j in range(cols)]
            for i in range(rows)
        ]
        state["clip"] += 1
        output = vae._assemble_tiles(tiles, y_overlap, x_overlap)
        if on_clip is not None:
            on_clip(clip, output, clip == len(plan.clips) - 1)
        return output

    previous = vars(vae).get("_adaptive_decode")
    vars(vae)["_adaptive_decode"] = replay
    try:
        output = decode_temporal(vae, z)
    finally:
        if previous is None:
            vars(vae).pop("_adaptive_decode", None)
        else:
            vars(vae)["_adaptive_decode"] = previous
    if state["clip"] != len(plan.clips) or results:
        raise RuntimeError(
            f"MiniMax H3 flat decode replayed {state['clip']} of {len(plan.clips)} clips"
        )
    return output


def _no_stage_stream(streamed_video_output_path):
    """The stage's encode-while-decoding output path: always none here."""

    @functools.wraps(streamed_video_output_path)
    def no_stage_stream(batch, server_args, batch_size):
        return None

    return no_stage_stream


def _flat_decode_temporal(decode_temporal):
    @functools.wraps(decode_temporal)
    def flat_decode_temporal(self, z, on_frames=None):
        if on_frames is not None:
            return decode_temporal(self, z, on_frames=on_frames)
        if not tpu_envs.SGLANG_TPU_ENABLE_H3_VAE_FLAT_DECODE.get():
            return decode_temporal(self, z)
        from sglang.multimodal_gen.runtime.distributed import (
            get_decode_parallel_group_coordinator,
        )
        from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.klvae import (
            get_tile_parallel_state,
        )

        rank, world = get_tile_parallel_state()
        plan = plan_units(self, z, world)
        if plan is None:
            return decode_temporal(self, z)
        request = _REQUEST.get()
        audio_rank = request.audio_rank if request is not None else None
        waves = plan_waves(
            len(plan.units),
            plan.tiles_per_clip,
            world,
            audio_rank=audio_rank,
            audio_wave_units=tpu_envs.SGLANG_TPU_OPT_H3_AUDIO_WAVE_UNITS.get(),
            audio_rank_units=tpu_envs.SGLANG_TPU_OPT_H3_AUDIO_RANK_WAVE_UNITS.get(),
        )
        _, _, output_frames = self._decode_temporal_output_frame_plan(
            plan.padded, None, None, plan.num_chunks, plan.pad_tokens
        )
        after_first_unit = on_clip = None
        if request is not None and audio_rank == rank:
            after_first_unit = functools.partial(
                _decode_audio_first, request, output_frames
            )
            on_clip = _ClipStream(self, request, output_frames)
        group = get_decode_parallel_group_coordinator()
        return run_flat_decode(
            self,
            z,
            plan,
            waves,
            decode_temporal,
            rank,
            world,
            lambda tensor: group.all_gather(tensor, separate_tensors=True),
            after_first_unit=after_first_unit,
            on_clip=on_clip,
            batch=decode_batch_units(),
        )

    flat_decode_temporal.__wrapped_for_tpu__ = True
    return flat_decode_temporal


# ---------------------------------------------------------------------------
# Batched unit decode
# ---------------------------------------------------------------------------

# Device memory a batched decoder call leaves free beyond its own estimate
# (allocator fragmentation, the estimate's slack, buffers the call does not see).
DECODE_BATCH_RESERVE_BYTES = 2 << 30

# A batched call raised (its error): the rest of the process decodes unit by
# unit. Only the message is kept, so no traceback holds the call's tensors.
_batch_failed: list[str] = []
_warned: set[str] = set()


@dataclasses.dataclass
class DecodeStatus:
    """What one rank's decode stage call did, for its status line."""

    request_id: str | None
    warmup: bool = False
    # Units of each video decoder call (vae.decode) that returned, in order.
    calls: list[int] = dataclasses.field(default_factory=list)
    # (planned, taken) units of each batched call the free memory cut.
    cuts: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    # The least free device memory read before a batched call.
    min_free: int | None = None
    # Fallbacks taken during the call.
    events: list[str] = dataclasses.field(default_factory=list)

    def saw_free(self, free: int | None) -> None:
        if free is not None and (self.min_free is None or free < self.min_free):
            self.min_free = free

    def log(self, rank: int, peak: tuple[float, float] | None) -> None:
        cuts = ", ".join(f"{planned}->{taken}" for planned, taken in self.cuts)
        logger.info(
            "MiniMax H3 decode status, request %s%s, decode rank %d: %d units in "
            "decoder calls of [%s] (cap %d); free memory cut %d calls%s, least "
            "free %s; fallbacks: %s; peak %s",
            self.request_id,
            " (warmup)" if self.warmup else "",
            rank,
            sum(self.calls),
            runs(self.calls),
            decode_batch_units(),
            len(self.cuts),
            f" ({cuts})" if cuts else "",
            "n/a" if self.min_free is None else f"{self.min_free / 2**30:.2f} GiB",
            "; ".join(self.events) or "none",
            "n/a"
            if peak is None
            else f"{peak[0]:.0f} MiB reserved, {peak[1]:.0f} MiB allocated",
            **_ALL_RANKS,
        )


_STATUS: contextvars.ContextVar[DecodeStatus | None] = contextvars.ContextVar(
    "minimax_h3_decode_status", default=None
)


def runs(sizes: Sequence[int]) -> str:
    """Call sizes in order, a run of equal ones as ``size x count``."""
    parts: list[str] = []
    k = 0
    while k < len(sizes):
        end = k
        while end < len(sizes) and sizes[end] == sizes[k]:
            end += 1
        parts.append(str(sizes[k]) if end - k == 1 else f"{sizes[k]}x{end - k}")
        k = end
    return ", ".join(parts)


def _event(message: str) -> None:
    status = _STATUS.get()
    if status is not None:
        status.events.append(message)


def _status_wanted() -> bool:
    """Whether stage calls log a status line on every rank (the flags that
    change the video decoder's calls, compile or precision)."""
    return (
        int(tpu_envs.SGLANG_TPU_OPT_H3_VAE_DECODE_BATCH.get() or 0) > 1
        or tpu_envs.SGLANG_TPU_ENABLE_H3_VAE_BLOCK_COMPILE.get()
        or tpu_envs.SGLANG_TPU_ENABLE_H3_VAE_FP32_PROJECTIONS.get()
        or tpu_envs.SGLANG_TPU_OPT_H3_VAE_FLASH_ATTENTION.get()
        or tpu_envs.SGLANG_TPU_OPT_H3_VAE_PALLAS_LINEAR.get()
        or tpu_envs.SGLANG_TPU_OPT_H3_VAE_FUSED_LINEARS.get()
        or tpu_envs.SGLANG_TPU_OPT_H3_VAE_SPLIT_WEIGHTS.get()
    )


def peak_mib(device: torch.device) -> tuple[float, float] | None:
    """This rank's allocator peaks (reserved, allocated MiB) since the worker
    reset them at the request's start (read as the worker's memory snapshot
    reads them); None off device."""
    if device.type == "cpu":
        return None
    try:
        module = torch.get_device_module()
        return (
            module.max_memory_reserved() / 2**20,
            module.max_memory_allocated() / 2**20,
        )
    except Exception:
        return None


def decode_batch_units() -> int:
    """Units a decoder call takes (SGLANG_TPU_OPT_H3_VAE_DECODE_BATCH, 1 = off)."""
    if _batch_failed:
        return 1
    return max(1, int(tpu_envs.SGLANG_TPU_OPT_H3_VAE_DECODE_BATCH.get() or 0))


def call_sizes(count: int, cap: int) -> list[int]:
    """Near-equal decoder-call sizes of at most ``cap`` units covering ``count``."""
    if count <= 0:
        return []
    cap = max(1, int(cap))
    calls = -(-count // cap)
    size, extra = divmod(count, calls)
    return [size + 1] * extra + [size] * (calls - extra)


def unit_call_bytes(vae, latent: torch.Tensor) -> int:
    """Device bytes one unit of ``latent``'s shape adds to a decoder call.

    Conservative for the fp16-autocast eager decode: two fp32 attention score
    buffers ([heads, tokens, tokens], the non-flash SDPA path; none where the
    flash ViT attention runs), the block's widest activations (fp32 residual,
    norm and q/k norm copies, fp16 qkv and gated-FFN tensors) and the fp32
    output with its unpatchify copy. About 1.05e9 bytes for a 1344x768 unit
    ([1, 24, 7, 16, 16] latent, 1,797 tokens), 0.23e9 with the flash kernel.
    """
    decoder = vae.decoder
    _, _, frames, height, width = latent.shape
    tokens = frames * height * width + int(decoder.num_register_tokens or 0) + 1
    config = decoder.config
    out = (
        int(config.out_channels)
        * frames
        * int(config.patch_size_t)
        * height
        * int(config.patch_size)
        * width
        * int(config.patch_size)
        * 4
    )
    blocks = decoder.transformer_blocks
    if len(blocks) == 0:
        return 2 * out
    attn = blocks[0].attn
    inner = attn.attn_inner_dim
    ffn = blocks[0].ff.w1.out_features
    flash = _flash_attention and _flash_decoders.get(id(vae))
    scores = 0 if flash else 2 * attn.heads * tokens * tokens * 4
    activations = tokens * (26 * inner + 3 * ffn)
    return scores + activations + 2 * out


def free_device_bytes(device: torch.device) -> int | None:
    """Free bytes of ``device`` (torch_tpu's allocator), None where unknown."""
    if device.type == "cpu":
        return None
    try:
        info = torch.accelerator.get_memory_info(device)
        # (free, total), or a mapping in some torch_tpu versions.
        return int(info["free"] if isinstance(info, dict) else info[0])
    except Exception:
        return None


def peak_headroom_bytes(device: torch.device) -> int | None:
    """Device bytes above this request's allocator peak (reserved; the worker
    resets the peaks at a request's start), None where unknown."""
    if device.type == "cpu":
        return None
    try:
        info = torch.accelerator.get_memory_info(device)
        total = int(info["total"] if isinstance(info, dict) else info[1])
        return total - int(torch.get_device_module().max_memory_reserved())
    except Exception:
        return None


def _warn_once(key: str, message: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message, *args, **_ALL_RANKS)


def _fitting_units(vae, latent: torch.Tensor, planned: int) -> int:
    """Up to ``planned`` units whose decoder call the free device memory covers."""
    if planned <= 1:
        return 1
    free = free_device_bytes(latent.device)
    status = _STATUS.get()
    if status is not None:
        status.saw_free(free)
    if free is None:
        return planned
    per_unit = unit_call_bytes(vae, latent)
    fits = max(1, (free - DECODE_BATCH_RESERVE_BYTES) // per_unit)
    if fits < planned:
        if status is not None:
            status.cuts.append((planned, int(fits)))
        _warn_once(
            "memory",
            "MiniMax H3 batched decode: %.1f GiB free covers %d of %d units "
            "(%.2f GiB a unit + %.0f GiB reserve); decoding %d a call",
            free / 2**30,
            fits,
            planned,
            per_unit / 2**30,
            DECODE_BATCH_RESERVE_BYTES / 2**30,
            min(fits, planned),
        )
    return int(min(fits, planned))


def _unit_calls(vae, latents: Sequence[torch.Tensor], cap: int, *, first_alone=False):
    """Decode ``latents`` (one unit each, ``[1, C, T, h, w]``) in calls of up to
    ``cap`` units; yield ``(index of the call's first unit, [n, ...] output)``.

    Calls are near-equal splits of the units left (:func:`call_sizes`), each
    capped by the free device memory (:func:`_fitting_units`). A one-unit call
    is the stock ``vae.decode(unit)``. ``first_alone`` decodes the first unit
    by itself (the audio owner's unit before its audio). A call of several
    units that raises is redone unit by unit, and later calls take one unit.
    """
    index = 0
    count = len(latents)
    if first_alone and count:
        yield 0, vae.decode(latents[0])
        index = 1
    while index < count:
        cap = 1 if _batch_failed else cap
        size = _fitting_units(vae, latents[index], call_sizes(count - index, cap)[0])
        if size <= 1:
            yield index, vae.decode(latents[index])
            index += 1
            continue
        if "first_call" not in _warned:
            _warned.add("first_call")
            free = free_device_bytes(latents[index].device)
            logger.info(
                "MiniMax H3 batched decode: first call of %d units (%s free before "
                "it, %.2f GiB a unit estimated)",
                size,
                "unknown" if free is None else f"{free / 2**30:.1f} GiB",
                unit_call_bytes(vae, latents[index]) / 2**30,
                **_ALL_RANKS,
            )
        error = None
        try:
            output = vae.decode(torch.cat(list(latents[index : index + size]), dim=0))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        if error is not None:
            # Outside the handler: the failed call's tensors are released first.
            _batch_failed.append(error)
            _event(f"batched call of {size} units failed ({error}), unit by unit after")
            logger.warning(
                "MiniMax H3 batched decode of %d units failed (%s); decoding unit "
                "by unit from here on",
                size,
                error,
                **_ALL_RANKS,
            )
            for k in range(size):
                yield index + k, vae.decode(latents[index + k])
            index += size
            continue
        yield index, output
        index += size


def _is_own_decode(vae, forward_fn) -> bool:
    return getattr(forward_fn, "__self__", None) is vae and getattr(
        forward_fn, "__func__", None
    ) is getattr(type(vae), "decode", None)


def _batched_run_tile_tasks(run_tile_tasks):
    """tiled_decode's local tiles through :func:`_unit_calls` (stock schedule)."""

    @functools.wraps(run_tile_tasks)
    def run_tile_tasks_batched(
        self, tiles, tile_indices, forward_fn, cls_agg=None, *, tile_batch=1
    ):
        cap = decode_batch_units()
        if (
            cap <= 1
            or tile_batch != 1
            or cls_agg is not None
            or len(tile_indices) < 2
            or not _is_own_decode(self, forward_fn)
        ):
            return run_tile_tasks(
                self, tiles, tile_indices, forward_fn, cls_agg, tile_batch=tile_batch
            )
        tasks = []
        for _, output in _unit_calls(self, [tiles[i] for i in tile_indices], cap):
            tasks.extend(output[k : k + 1] for k in range(output.shape[0]))
        return tasks

    run_tile_tasks_batched.__wrapped_for_tpu__ = True
    return run_tile_tasks_batched


# ---------------------------------------------------------------------------
# Audio first, on the audio owner
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Request:
    """One decode-stage call that decodes its audio inside the video decode."""

    stage: Any
    batch: Any
    server_args: Any
    audio_latent: torch.Tensor
    # Decode-group rank of the audio owner, the rank that decodes the audio.
    audio_rank: int
    # Output path when this rank streams the mp4.
    stream_path: str | None = None
    audio_ran: bool = False
    # The stage decoded the audio before the video (sglang#41819 ordering).
    audio_from_stage: bool = False
    audio_payload: dict | None = None
    audio_error: BaseException | None = None
    save: StreamedSave | None = None


_REQUEST: contextvars.ContextVar[_Request | None] = contextvars.ContextVar(
    "minimax_h3_decode_request", default=None
)


def _is_resident(stage, component_name: str, module) -> bool:
    from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency_strategies import (
        ResidentStrategy,
    )

    manager = stage._component_residency_manager
    if manager is None or module is None:
        return manager is None
    return type(manager.strategy_for(component_name, module)) is ResidentStrategy


def _audio_decode_rank(stage) -> int | None:
    """The decode-group rank of the audio owner, if the audio can go first."""
    from sglang.multimodal_gen.runtime.distributed import (
        get_decode_parallel_group_coordinator,
        get_replica_group,
        model_parallel_is_initialized,
    )

    if not model_parallel_is_initialized():
        return None
    decode_group = get_decode_parallel_group_coordinator()
    if decode_group.world_size < 2:
        return None
    owner = get_replica_group().ranks[0]
    if owner not in decode_group.ranks:
        return None
    # A use of the audio VAE inside the video VAE's use ends the latter in the
    # residency manager; only resident components make that a no-op.
    if not (
        _is_resident(stage, "video_vae", stage.video_vae)
        and _is_resident(stage, "audio_vae", stage.audio_vae)
    ):
        return None
    return list(decode_group.ranks).index(owner)


def _stream_path(batch) -> str | None:
    """The output path of a request whose save the output rank can stream."""
    from sglang.multimodal_gen.configs.sample.sampling_params import DataType
    from sglang.multimodal_gen.runtime.distributed import get_replica_group

    if get_replica_group().rank_in_group != 0:
        return None
    extra = getattr(batch, "extra", None) or {}
    if (
        batch.is_warmup
        or not batch.save_output
        or not batch.return_file_paths_only
        or getattr(batch, "return_raw_frames", False)
        or getattr(batch, "return_frames", False)
        or batch.data_type != DataType.VIDEO
        or batch.enable_frame_interpolation
        or batch.enable_upscaling
        or int(batch.num_outputs_per_prompt or 1) != 1
        or extra.get("dynamic_batch_output_paths") is not None
        or not isinstance(getattr(batch, "latents", None), torch.Tensor)
        or batch.latents.shape[0] != 1
    ):
        return None
    path = batch.output_file_path(1, 0)
    if not path or os.path.splitext(path)[1].lower() != ".mp4":
        return None
    return path


def _new_request(stage, batch, server_args) -> _Request | None:
    if not (
        tpu_envs.SGLANG_TPU_ENABLE_H3_VAE_FLAT_DECODE.get()
        and tpu_envs.SGLANG_TPU_ENABLE_H3_AUDIO_FIRST.get()
    ):
        return None
    audio_latent = getattr(batch, "audio_latents", None)
    if not isinstance(audio_latent, torch.Tensor) or audio_latent.ndim != 3:
        return None
    if stage.audio_vae is None or stage.video_vae is None:
        return None
    audio_rank = _audio_decode_rank(stage)
    if audio_rank is None:
        return None
    request = _Request(stage, batch, server_args, audio_latent, audio_rank)
    if tpu_envs.SGLANG_TPU_ENABLE_H3_STREAM_SAVE.get():
        request.stream_path = _stream_path(batch)
    return request


@contextlib.contextmanager
def _residency_bypassed(stage):
    """The stage without its residency manager (resident components only)."""
    manager = stage._component_residency_manager
    stage._component_residency_manager = None
    try:
        yield
    finally:
        stage._component_residency_manager = manager


def _decode_audio_first(request: _Request, num_frames: int) -> None:
    """The stage's audio decode, run by the audio owner inside the video decode.

    When the stage already decoded the audio before the video, only the
    streamed save starts here; the unit schedule stays the same.
    """
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if request.audio_from_stage:
        _start_stream(request, num_frames)
        return
    request.audio_ran = True
    try:
        # Outside the video decode the audio runs without autocast.
        with (
            _residency_bypassed(request.stage),
            torch.autocast(device_type=current_platform.device_type, enabled=False),
        ):
            request.audio_payload = _stock["decode_audio"](
                request.stage, request.audio_latent, request.server_args
            )
    except Exception as exc:
        request.audio_error = exc
        return
    _start_stream(request, num_frames)


def _start_stream(request: _Request, num_frames: int) -> None:
    if request.stream_path is not None:
        try:
            request.save = _start_save(request, num_frames)
        except Exception:
            logger.warning(
                "MiniMax H3 streamed mp4 save did not start; saving after decode",
                exc_info=True,
                **_ALL_RANKS,
            )
            request.save = None


def _decode_audio(decode_audio):
    @functools.wraps(decode_audio)
    def decode_audio_once(self, audio_latent, server_args):
        request = _REQUEST.get()
        ours = (
            request is not None
            and request.stage is self
            and request.audio_latent is audio_latent
        )
        if ours and not request.audio_ran and not request.audio_from_stage:
            # The stage decodes the audio before the video: keep it for the
            # streamed save, which the video decode starts.
            payload = _stock["decode_audio"](self, audio_latent, server_args)
            request.audio_payload = payload
            request.audio_from_stage = True
            return payload
        if ours and request.audio_ran:
            request.audio_ran = False
            error, request.audio_error = request.audio_error, None
            if error is not None:
                raise error
            payload, request.audio_payload = request.audio_payload, None
            return payload
        return decode_audio(self, audio_latent, server_args)

    decode_audio_once.__wrapped_for_tpu__ = True
    return decode_audio_once


def _decode_stage_forward(forward):
    @functools.wraps(forward)
    def decode_stage_forward(self, batch, server_args):
        refresh_precision_flags()
        if tpu_envs.SGLANG_TPU_ENABLE_H3_VAE_BLOCK_COMPILE.get():
            compile_decoder_blocks(self.video_vae)
        request = _new_request(self, batch, server_args)
        status = None
        if _status_wanted():
            status = DecodeStatus(
                getattr(batch, "request_id", None),
                bool(getattr(batch, "is_warmup", False)),
            )
        token = _REQUEST.set(request)
        status_token = _STATUS.set(status)
        try:
            output = forward(self, batch, server_args)
        except BaseException:
            if request is not None:
                _abort_save(request)
            raise
        finally:
            _STATUS.reset(status_token)
            _REQUEST.reset(token)
            if status is not None:
                _log_status(status, batch)
            if _fused_linears:
                _note_stage_room(self.video_vae)
        return output

    decode_stage_forward.__wrapped_for_tpu__ = True
    return decode_stage_forward


def _log_status(status: DecodeStatus, batch) -> None:
    """This rank's status line; never raises (it may run while an error unwinds)."""
    try:
        from sglang.multimodal_gen.runtime.distributed import (
            get_decode_parallel_rank,
            model_parallel_is_initialized,
        )

        rank = get_decode_parallel_rank() if model_parallel_is_initialized() else 0
        latents = getattr(batch, "latents", None)
        device = latents.device if isinstance(latents, torch.Tensor) else None
        status.log(rank, None if device is None else peak_mib(device))
    except Exception:
        logger.warning(
            "MiniMax H3 decode status line failed", exc_info=True, **_ALL_RANKS
        )


# ---------------------------------------------------------------------------
# Streamed mp4 on the output rank
# ---------------------------------------------------------------------------


class _ClipStream:
    """The frames the stock streaming decode_temporal writes, clip by clip.

    The parts each assembled clip contributes to the output (the first frames
    after the pre-padding, the first of them blended with the previous clip's
    overlap, the last overlap at the end, at most ``output_frames`` in all),
    taken with the same slicing and ``blend`` calls, then the stage's pixel
    steps and the save's uint8 conversion, handed to the request's save.
    """

    def __init__(self, vae, request: _Request, output_frames: int):
        self.vae = vae
        self.request = request
        self.output_frames = output_frames
        self.chunk_frames = vae.tokens_chunk_size * vae.vae_ratio_t
        self.split_count = int(vae.token_drop > 0) + 1
        self.overlap: torch.Tensor | None = None
        self.written = 0
        self.active = False

    def __call__(self, index: int, clip: torch.Tensor, last: bool) -> None:
        if index == 0:
            self.active = self.request.save is not None
        if not self.active:
            return
        try:
            self._clip(clip, last)
        except Exception:
            logger.warning(
                "MiniMax H3 streamed mp4 save stopped; saving after decode",
                exc_info=True,
                **_ALL_RANKS,
            )
            self.active = False
            _abort_save(self.request)

    def _clip(self, clip: torch.Tensor, last: bool) -> None:
        vae = self.vae
        # Clips keep the decoder's dtype, as the stock decode_temporal does.
        for j in range(self.split_count):
            start = j * self.chunk_frames
            stop = min(start + self.chunk_frames, clip.shape[2])
            part = clip[:, :, start:stop, :, :][:, :, vae.frame_pre_padding :, :, :]
            if j == 0:
                if self.overlap is not None:
                    part = vae.blend(self.overlap, part, vae.frame_overlap, dim=-3)
                    self.overlap = None
                self._write(part)
            else:
                self.overlap = part.contiguous()
        if last and self.overlap is not None:
            self._write(self.overlap)
            self.overlap = None
        if last and self.written != self.output_frames:
            raise RuntimeError(
                f"streamed {self.written} of {self.output_frames} frames"
            )

    def _write(self, part: torch.Tensor) -> None:
        take = min(int(part.shape[2]), self.output_frames - self.written)
        if take <= 0:
            return
        self.request.save.put(_pixels(self.vae, self.request, part[:, :, :take]))
        self.written += take


def _pixels(vae, request: _Request, part: torch.Tensor) -> torch.Tensor:
    """A decoded part as the saved frames: [frames, H, W, 3] uint8 on device.

    The decode stage's steps after the video decode (revert, canonical layout,
    crop, float32, post_decoding) and the save's conversion of sample 0, per
    element as on the whole video.
    """
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (
        decoding,
    )
    from sglang.multimodal_gen.runtime.platforms import current_platform

    frames = vae.processor.revert_tensor(part)
    frames = decoding._canonical_visual_video_frames(frames, batch_size=1)
    frames = decoding._crop_to_target_canvas(request.batch, frames)
    if frames.dtype != torch.float32 or not frames.is_contiguous():
        canonical = torch.empty_like(
            frames, dtype=torch.float32, memory_format=torch.contiguous_format
        )
        canonical.copy_(frames)
        frames = canonical
    with torch.autocast(device_type=current_platform.device_type, enabled=False):
        server_args = request.server_args
        frames = server_args.pipeline_config.post_decoding(frames, server_args)
        sample = frames[0]
        if sample.dim() == 3:
            sample = sample.unsqueeze(1)
        sample = (sample * 255).clamp(0, 255).to(torch.uint8)
        return sample.permute(1, 2, 3, 0).contiguous()


_ABORT = object()


class _QueuedFrames(list):
    """A video's frames for the stock save, read from device chunks as they come.

    A list to the save, with the video's length. ``frames[0]``, which the save
    reads for the video size before it encodes, waits for the first chunk;
    the single iteration copies each chunk to the host and yields its frames.
    """

    def __init__(self, chunks: queue.Queue, num_frames: int):
        super().__init__()
        self._chunks = chunks
        self._num_frames = num_frames
        self._iterated = False
        self._head: Any = None
        self.yielded = 0

    def __len__(self) -> int:
        return self._num_frames

    def __bool__(self) -> bool:
        return self._num_frames > 0

    def _next_chunk(self):
        chunk = self._chunks.get()
        if chunk is _ABORT:
            raise RuntimeError("MiniMax H3 streamed mp4 save aborted")
        return chunk.cpu().numpy()

    def __getitem__(self, index):
        if index != 0 or self._iterated:
            raise IndexError("MiniMax H3 streamed frames: only frames[0] reads ahead")
        if self._head is None:
            self._head = self._next_chunk()
        return self._head[0]

    def __iter__(self):
        if self._iterated:
            raise RuntimeError("MiniMax H3 streamed frames were consumed already")
        self._iterated = True
        head, self._head = self._head, None
        while self.yielded < self._num_frames:
            frames, head = (self._next_chunk() if head is None else head), None
            for frame in frames:
                if self.yielded == self._num_frames:
                    raise RuntimeError("MiniMax H3 streamed more frames than planned")
                self.yielded += 1
                yield frame


class StreamedSave:
    """The stock save of one video on a thread, fed frames while they decode."""

    def __init__(
        self,
        path: str,
        *,
        audio: torch.Tensor,
        audio_sample_rate: int,
        fps: int,
        num_frames: int,
        output_compression: int | None,
        x264_preset: str | None,
    ):
        self.path = path
        self._chunks: queue.Queue = queue.Queue()
        self.frames = _QueuedFrames(self._chunks, num_frames)
        self._audio = audio
        self._kwargs = dict(
            audio_sample_rate=audio_sample_rate,
            output_compression=output_compression,
            x264_preset=x264_preset,
        )
        self._fps = fps
        self.error: BaseException | None = None
        self._done = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="minimax-h3-mp4", daemon=True
        )
        self._thread.start()

    def put(self, chunk: torch.Tensor) -> None:
        self._chunks.put(chunk)

    def abort(self) -> None:
        self._chunks.put(_ABORT)

    def _run(self) -> None:
        from sglang.multimodal_gen.configs.sample.sampling_params import DataType
        from sglang.multimodal_gen.runtime.entrypoints.utils import (
            MaterializedOutput,
            save_materialized_output,
        )

        try:
            save_materialized_output(
                MaterializedOutput(
                    sample=None, frames=self.frames, audio=self._audio, fps=self._fps
                ),
                DataType.VIDEO,
                self.path,
                save_output=True,
                **self._kwargs,
            )
            if self.frames.yielded != len(self.frames):
                raise RuntimeError(
                    f"saved {self.frames.yielded} of {len(self.frames)} frames"
                )
        except BaseException as exc:
            self.error = exc
        finally:
            self._audio = None
            self._done.set()

    def wait(self, timeout: float | None = None) -> bool:
        """Whether the save ended (in time) and wrote the whole video."""
        return self._done.wait(timeout) and self.error is None


# Output path -> its streamed save, on the output rank.
_SAVES: dict[str, StreamedSave] = {}


def _start_save(request: _Request, num_frames: int) -> StreamedSave:
    from sglang.multimodal_gen.runtime.entrypoints.utils import select_output_audio
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (
        decoding,
    )

    # A save the worker never took belongs to a request that failed later.
    for stale in list(_SAVES.values()):
        _discard(stale, keep_complete=True)
    _SAVES.clear()
    payload = request.audio_payload
    # The audio the worker's save would take: the stage's output layout,
    # sample 0 of it.
    audio = select_output_audio(
        decoding._canonical_output_audio_waveform(payload["waveform"], batch_size=1), 0
    )
    batch = request.batch
    save = StreamedSave(
        request.stream_path,
        audio=audio,
        audio_sample_rate=int(payload["sample_rate"]),
        fps=batch.fps,
        num_frames=num_frames,
        output_compression=batch.output_compression,
        x264_preset=batch.x264_preset,
    )
    _SAVES[save.path] = save
    return save


def _discard(save: StreamedSave, *, keep_complete: bool = False) -> None:
    """Stop a streamed save and remove what it wrote (a whole file on request)."""
    save.abort()
    if save.wait(60.0) and keep_complete:
        return
    with contextlib.suppress(OSError):
        os.remove(save.path)


def _abort_save(request: _Request) -> None:
    save, request.save = request.save, None
    if save is None:
        return
    if _SAVES.get(save.path) is save:
        del _SAVES[save.path]
    _discard(save)


def _save_outputs(save_outputs):
    @functools.wraps(save_outputs)
    def save_streamed_outputs(
        outputs, data_type, fps, save_output, build_output_path, **kwargs
    ):
        from sglang.multimodal_gen.configs.sample.sampling_params import DataType

        if (
            _SAVES
            and save_output
            and data_type == DataType.VIDEO
            and len(outputs) == 1
            and kwargs.get("samples_out") is None
            and kwargs.get("audios_out") is None
            and kwargs.get("frames_out") is None
            and not kwargs.get("enable_frame_interpolation")
            and not kwargs.get("enable_upscaling")
        ):
            path = build_output_path(0)
            save = _SAVES.pop(path, None)
            if save is not None:
                if save.wait(STREAM_SAVE_TIMEOUT_S):
                    return [path]
                logger.warning(
                    "MiniMax H3 streamed mp4 save failed (%s); saving again",
                    save.error,
                    **_ALL_RANKS,
                )
                _discard(save)
        return save_outputs(
            outputs, data_type, fps, save_output, build_output_path, **kwargs
        )

    save_streamed_outputs.__wrapped_for_tpu__ = True
    return save_streamed_outputs


# ---------------------------------------------------------------------------
# ViT decoder block compile
# ---------------------------------------------------------------------------

_compiled_vaes: set[int] = set()


def compile_decoder_blocks(vae, compile_kwargs: dict | None = None) -> int:
    """torch.compile each ViT decoder block of ``vae`` once; the number compiled."""
    from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
        is_layerwise_offloaded_module,
    )
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if vae is None or id(vae) in _compiled_vaes:
        return 0
    _compiled_vaes.add(id(vae))
    # Layerwise offload swaps block weights under hooks a compiled block bakes in.
    if is_layerwise_offloaded_module(vae):
        return 0
    if compile_kwargs is None:
        compile_kwargs = {
            "backend": current_platform.get_compile_backend(),
            "dynamic": False,
            "fullgraph": True,
        }
    compiled = 0
    for block in vae.decoder.transformer_blocks:
        if getattr(block, "_compiled_call_impl", None) is None:
            block.compile(**compile_kwargs)
            compiled += 1
    logger.info(
        "MiniMax H3 video VAE: compiling %d ViT decoder blocks", compiled, **_ALL_RANKS
    )
    return compiled


def _eager_on_compile_failure(forward_transformer_blocks):
    """A failed call through compiled blocks reruns eagerly; the compiles are
    dropped only if that rerun works (an error the eager blocks raise too,
    such as running out of memory at one batch size, is not the compile's)."""

    @functools.wraps(forward_transformer_blocks)
    def guarded(self, hidden_states, rotary_pos_emb):
        blocks = list(self.transformer_blocks)
        compiled = [getattr(block, "_compiled_call_impl", None) for block in blocks]
        if all(impl is None for impl in compiled):
            return forward_transformer_blocks(self, hidden_states, rotary_pos_emb)
        error = None
        try:
            return forward_transformer_blocks(self, hidden_states, rotary_pos_emb)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        # Outside the handler: the failed call's tensors are released first.
        # Blocks do not write their inputs, so the eager rerun starts clean.
        for block in blocks:
            block._compiled_call_impl = None
        try:
            output = forward_transformer_blocks(self, hidden_states, rotary_pos_emb)
        except Exception:
            for block, impl in zip(blocks, compiled):
                block._compiled_call_impl = impl
            _event(f"compiled ViT decoder blocks failed ({error}), eager too")
            raise
        _event(f"compiled ViT decoder blocks dropped ({error})")
        logger.warning(
            "MiniMax H3 compiled ViT decoder block failed (%s); decoding eagerly",
            error,
            **_ALL_RANKS,
        )
        return output

    guarded.__wrapped_for_tpu__ = True
    return guarded


# ---------------------------------------------------------------------------
# Decoder calls: Pallas routing and the rank's status
# ---------------------------------------------------------------------------


def _routed_decode(decode):
    """``vae.decode``: routes the call's ViT attention, block linears and
    blocks to the Pallas paths its flags select, and records the call's units
    in the rank's status."""

    @functools.wraps(decode)
    def routed(self, z):
        global _flash_active, _linear_active, _fused_active
        linear = _linear_wanted(self, z)
        flash = _flash_wanted(self, z)
        fused = _fused_wanted(self, z, linear, flash)
        previous_flash, _flash_active = _flash_active, flash
        previous_linear, _linear_active = _linear_active, linear
        previous_fused, _fused_active = _fused_active, fused
        try:
            output = decode(self, z)
        finally:
            _flash_active = previous_flash
            _linear_active = previous_linear
            _fused_active = previous_fused
        status = _STATUS.get()
        if status is not None:
            status.calls.append(int(z.shape[0]))
        return output

    routed.__wrapped_for_tpu__ = True
    return routed


# ---------------------------------------------------------------------------
# Decoder precision: fp32 projections, flash ViT attention
# ---------------------------------------------------------------------------

# The precision flags as of the current decode stage call (module constants to
# compiled code, which reads no environment):
# SGLANG_TPU_ENABLE_H3_VAE_FP32_PROJECTIONS,
_fp32_projections = False
# SGLANG_TPU_OPT_H3_VAE_FLASH_ATTENTION,
_flash_attention = False
# SGLANG_TPU_OPT_H3_VAE_PALLAS_LINEAR,
_pallas_linear = False
# SGLANG_TPU_OPT_H3_VAE_FUSED_LINEARS,
_fused_linears = False
# SGLANG_TPU_OPT_H3_VAE_SPLIT_WEIGHTS.
_split_weights = False


def refresh_precision_flags() -> None:
    global _fp32_projections, _flash_attention, _pallas_linear, _fused_linears
    global _split_weights
    _fp32_projections = bool(tpu_envs.SGLANG_TPU_ENABLE_H3_VAE_FP32_PROJECTIONS.get())
    _flash_attention = bool(tpu_envs.SGLANG_TPU_OPT_H3_VAE_FLASH_ATTENTION.get())
    _pallas_linear = bool(tpu_envs.SGLANG_TPU_OPT_H3_VAE_PALLAS_LINEAR.get())
    _fused_linears = bool(tpu_envs.SGLANG_TPU_OPT_H3_VAE_FUSED_LINEARS.get())
    _split_weights = bool(tpu_envs.SGLANG_TPU_OPT_H3_VAE_SPLIT_WEIGHTS.get())


def _flash_sdpa(sdpa):
    """The ViT attention: the flash kernel inside a decode that routes it, on
    q/k/v it takes; the stock SDPA otherwise."""

    @functools.wraps(sdpa)
    def flash_sdpa(query, key, value):
        if _flash_active and flash_attention_fits(query, key, value):
            return flash_attention(query, key, value)
        return sdpa(query, key, value)

    flash_sdpa.__wrapped_for_tpu__ = True
    return flash_sdpa


def _projection_autocast(cuda_autocast_disabled):
    """vae_vit's ``_cuda_autocast_disabled`` around x_embedder and proj_out:
    with FP32_PROJECTIONS, autocast off for the tensor's device on any device
    (the released decoder's ``torch.autocast("cuda", enabled=False)``)."""

    @functools.wraps(cuda_autocast_disabled)
    def projection_autocast(tensor):
        if _fp32_projections and not tensor.is_cuda:
            return torch.autocast(device_type=tensor.device.type, enabled=False)
        return cuda_autocast_disabled(tensor)

    projection_autocast.__wrapped_for_tpu__ = True
    return projection_autocast


def _block_linears(decoder) -> list:
    blocks = getattr(decoder, "transformer_blocks", None) or []
    return [
        linear
        for block in blocks
        for linear in (block.attn.to_qkv, block.attn.to_out, block.ff.w1, block.ff.w2)
    ]


# ---------------------------------------------------------------------------
# Pallas fp16 linear for the ViT decoder's block linears
# ---------------------------------------------------------------------------

# True while a vae.decode runs under fp16 autocast with the flag on: the block
# linears read it at call time (a constant to compiled code, guarded).
_linear_active = False
# The TPU op (minimax_h3_vae_linear.op()) once registered.
_linear_op = None


def _linear_wanted(vae, z) -> bool:
    """Whether this vae.decode runs the block linears as the Pallas linear
    (flag on, fp16 autocast on the latent's device), routing them to it; for
    a decode on TPU the op is registered here, before any compiled region
    calls it."""
    if not _pallas_linear or not isinstance(z, torch.Tensor):
        return False
    device_type = z.device.type
    if not torch.is_autocast_enabled(device_type):
        return False
    if torch.get_autocast_dtype(device_type) != torch.float16:
        return False
    linears = _block_linears(getattr(vae, "decoder", None))
    for linear in linears:
        if "forward" not in linear.__dict__:
            linear.forward = types.MethodType(_pallas_linear_forward, linear)
    if linears and device_type == "tpu" and _linear_op is None:
        _pallas_linear_op()
    return bool(linears)


def _pallas_linear_forward(self, x):
    """A decoder block linear: the Pallas linear while a vae.decode runs
    under fp16 autocast with its flag on, the stock ``nn.Linear`` otherwise."""
    if _linear_active and pallas_linear_fits(x, self.weight):
        return pallas_linear(x, self.weight, self.bias)
    return torch.nn.Linear.forward(self, x)


def _pallas_linear_op():
    global _linear_op
    if _linear_op is None:
        from sglang_tpu.multimodal_gen import minimax_h3_vae_linear

        _linear_op = minimax_h3_vae_linear.op()
    return _linear_op


def pallas_linear_fits(x, weight) -> bool:
    """Whether the kernel takes this linear: both weight sides multiples of
    256 (an MXU pass), a non-empty input."""
    return (
        weight.dim() == 2
        and weight.shape[0] % 256 == 0
        and weight.shape[1] % 256 == 0
        and x.numel() > 0
    )


def pallas_linear(x, weight, bias):
    """``F.linear`` under fp16 autocast as minimax_h3_vae_linear computes it:
    the operands rounded to fp16 as autocast rounds them, hi*hi + hi*lo +
    lo*hi of their exact bf16 parts (XLA's fp16 products) in float32 with the
    bias, one rounding to fp16. On TPU the Pallas kernel; elsewhere
    :func:`_pallas_linear_torch`."""
    shape = x.shape
    x16 = x.reshape(-1, shape[-1]).to(torch.float16)
    # The op has no autograd formula: never hand it a parameter that tracks one.
    weight16 = weight.detach().to(torch.float16)
    if bias is None:
        bias16 = torch.zeros(weight.shape[0], dtype=torch.float16, device=x.device)
    else:
        bias16 = bias.detach().to(torch.float16)
    if x16.device.type == "tpu":
        op = _linear_op if _linear_op is not None else _pallas_linear_op()
        # The kernel takes fp16 bits: int16 views of the same storage (Mosaic
        # takes no fp16 operand, and XLA would copy for a bitcast).
        bits = [t.contiguous().view(torch.int16) for t in (x16, weight16, bias16)]
        out = op(bits[0], bits[1], bits[2].view(1, -1)).view(torch.float16)
    else:
        out = _pallas_linear_torch(x16, weight16, bias16)
    return out.reshape(*shape[:-1], out.shape[-1])


def _pallas_linear_torch(x16, weight16, bias16):
    """The kernel's arithmetic as float32 matmuls of the bf16 parts (each
    product exact in float32), off TPU; only the float32 summation order
    differs."""
    with torch.autocast(device_type=x16.device.type, enabled=False):
        acc = _three_products(x16.float(), weight16)
        return (acc + bias16.float()).to(torch.float16)


def _three_products(x, weight):
    """hi*hi + hi*lo + lo*hi of the bf16 parts of float32 x and of weight
    [n, k] (fp16, or its bits as int16) or its held split [2, n, k], as
    float32 matmuls (each product exact in float32)."""
    xh, xl = _bf16_parts(x)
    if weight.dim() == 3:
        wh, wl = weight.float().unbind(0)
    else:
        wh, wl = _bf16_parts(weight.view(torch.float16).float())
    return xh @ wh.t() + xh @ wl.t() + xl @ wh.t()


# ---------------------------------------------------------------------------
# Flash ViT attention (fp16-faithful Pallas kernel)
# ---------------------------------------------------------------------------

# True while a vae.decode of a decoder the kernel's softmax bound holds for
# runs with the flag on: the ViT attention reads it at call time (a constant
# to compiled code, guarded).
_flash_active = False
# The TPU op (minimax_h3_vae_attention.op()) once registered.
_flash_op = None
# id(vae) -> whether its decoder takes the kernel (checked once a VAE).
_flash_decoders: dict[int, bool] = {}
FLASH_HEAD_DIM = 64
# The kernel's key-count limit (minimax_h3_vae_attention.MAX_KEYS).
FLASH_MAX_TOKENS = 2048


def flash_decoder_reason(vae) -> str | None:
    """Why ``vae``'s ViT decoder cannot take the flash kernel, or None.

    The kernel's softmax uses the score bound sqrt(64) = 8 as its shift: it
    holds when every block's q and k are normalized per 64-wide head by an
    RMSNorm without weights (|q.k| / 8 <= |q| |k| / 8 <= 8)."""
    blocks = getattr(getattr(vae, "decoder", None), "transformer_blocks", None)
    if not blocks:
        return "no ViT decoder blocks"
    for index, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        if getattr(attn, "dim_head", None) != FLASH_HEAD_DIM:
            return f"block {index}: head_dim {getattr(attn, 'dim_head', None)}, not 64"
        if getattr(attn, "heads", 1) % 2:
            return f"block {index}: an odd head count ({attn.heads})"
        for name in ("norm_q", "norm_k"):
            norm = getattr(attn, name, None)
            if not isinstance(norm, torch.nn.RMSNorm) or norm.weight is not None:
                return f"block {index}: {name} is not a parameter-free RMSNorm"
            if tuple(norm.normalized_shape) != (FLASH_HEAD_DIM,):
                return f"block {index}: {name} does not normalize a 64-wide head"
    return None


def _flash_wanted(vae, z) -> bool:
    """Whether this vae.decode runs the flash ViT attention: the flag is on
    and the decoder takes the kernel (checked and logged once a VAE). For a
    decode on TPU the op is registered here, before any compiled region
    calls it."""
    if not _flash_attention or not isinstance(z, torch.Tensor):
        return False
    known = _flash_decoders.get(id(vae))
    if known is None:
        known = _check_flash_decoder(vae)
    if known and z.device.type == "tpu" and _flash_op is None:
        _flash_attention_op()
    return known


def _check_flash_decoder(vae) -> bool:
    reason = flash_decoder_reason(vae)
    _flash_decoders[id(vae)] = reason is None
    if reason is None:
        logger.info(
            "MiniMax H3 video VAE: flash ViT attention on (%d blocks)",
            len(vae.decoder.transformer_blocks),
            **_ALL_RANKS,
        )
    else:
        _warn_once(
            "flash",
            "MiniMax H3 video VAE: flash ViT attention off (%s); stock SDPA",
            reason,
        )
    return reason is None


def flash_attention_fits(query, key, value) -> bool:
    """Whether the kernel takes these q/k/v: fp16 [batch, tokens, heads, 64]
    of one shape, an even head count, at most FLASH_MAX_TOKENS tokens."""
    return (
        query.dtype == torch.float16
        and key.dtype == torch.float16
        and value.dtype == torch.float16
        and query.dim() == 4
        and query.shape == key.shape == value.shape
        and query.shape[-1] == FLASH_HEAD_DIM
        and query.shape[2] % 2 == 0
        and 0 < query.shape[1] <= FLASH_MAX_TOKENS
    )


def _flash_attention_op():
    """The TPU op, registered once; _flash_wanted registers it before any
    compiled region calls it (which then reads ``_flash_op``)."""
    global _flash_op
    if _flash_op is None:
        from sglang_tpu.multimodal_gen import minimax_h3_vae_attention

        _flash_op = minimax_h3_vae_attention.op()
    return _flash_op


def flash_attention(query, key, value):
    """softmax(q k^T / 8) v over the ViT attention's [batch, tokens, heads,
    64] fp16 q/k/v -> fp16, as minimax_h3_vae_attention computes it: on TPU
    its Pallas kernel, elsewhere :func:`_flash_attention_torch` (the same
    arithmetic as float32 matmuls of the bf16 parts)."""
    batch, tokens, heads, dim = query.shape
    flat = [t.reshape(batch, tokens, heads * dim) for t in (query, key, value)]
    if query.device.type == "tpu":
        op = _flash_op if _flash_op is not None else _flash_attention_op()
        out = op(*flat)
    else:
        out = _flash_attention_torch(*flat)
    return out.view(batch, tokens, heads, dim)


def _bf16_parts(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """float32 holding fp16 (or float32) values -> float32 holding bf16 hi and
    lo, hi rounded to nearest (an fp16 value is hi + lo exactly)."""
    hi = t.to(torch.bfloat16).float()
    lo = torch.where(torch.isinf(t), torch.zeros_like(t), t - hi)
    return hi, lo.to(torch.bfloat16).float()


def _flash_attention_torch(q, k, v):
    """The kernel's arithmetic off TPU over [batch, tokens, heads * 64] fp16:
    scores qh.kh + qh.kl + ql.kh minus the shift 64 (q.k units), exp2 of them
    times log2(e) / 8, P split into bf16 parts, P V and the row sums from all
    four exact products, float32 sums, one fp16 rounding. Each product of
    bf16 values is exact in float32, so only summation order differs."""
    parts = [p for t in (q, k, v) for p in _bf16_parts(t.float())]
    return _flash_parts_torch(*parts).to(torch.float16)


def _flash_parts_torch(qh, ql, kh, kl, vh, vl, seq=None):
    """:func:`_flash_attention_torch` before its rounding, on the bf16 parts
    [batch, rows, heads * 64] (keys from ``seq`` on are padding) -> f32."""
    batch, rows, width = qh.shape
    seq = rows if seq is None else seq
    heads = width // FLASH_HEAD_DIM
    with torch.autocast(device_type=qh.device.type, enabled=False):

        def split_heads(t, count):
            t = t[:, :count].float().reshape(batch, count, heads, FLASH_HEAD_DIM)
            return t.transpose(1, 2)

        qh, ql = (split_heads(t, rows) for t in (qh, ql))
        kh, kl, vh, vl = (split_heads(t, seq) for t in (kh, kl, vh, vl))
        scale = FLASH_HEAD_DIM**-0.5
        bound = FLASH_HEAD_DIM**0.5
        scores = qh @ kh.transpose(-1, -2)
        scores = scores + qh @ kl.transpose(-1, -2)
        scores = scores + ql @ kh.transpose(-1, -2)
        e = torch.exp2((scores - bound / scale) * (scale * math.log2(math.e)))
        eh, el = _bf16_parts(e)
        out = (eh @ vh + eh @ vl) + (el @ vh + el @ vl)
        out = out / (eh.sum(-1, keepdim=True) + el.sum(-1, keepdim=True))
    return out.transpose(1, 2).reshape(batch, rows, width)


# ---------------------------------------------------------------------------
# Fused ViT blocks (the linears with their norms, RoPE, gating and residuals)
# ---------------------------------------------------------------------------

# True while a vae.decode runs the fused blocks (FUSED_LINEARS with the Pallas
# linears and the flash attention routed): the blocks read it at call time (a
# constant to compiled code, guarded).
_fused_active = False
# minimax_h3_vae_fused.ops() once registered: (qkv, residual, gate, flash_parts).
_fused_ops = None
# The least headroom under a decode stage's reserved peak seen so far, with w2
# unsplit (:func:`_note_stage_room`); None until a stage ran on a device.
_stage_room = None
# Device bytes kept free beyond the weights held split when they are made.
SPLIT_RESERVE_BYTES = 4 << 30
# Token rows of the fused calls, a multiple of 8: the TPU's default layout of a
# [batch, rows, width] activation is then Mosaic's (1,797 tokens: 1,800 rows).
FUSED_ROW_MULTIPLE = 8


def _fused_wanted(vae, z, linear: bool, flash: bool) -> bool:
    """Whether this vae.decode runs the fused blocks: the flag on, the Pallas
    linears and the flash attention routed, the kernels' int16 operands held.
    Those and the TPU ops are made here, outside any compiled region."""
    if not (_fused_linears and linear and flash):
        return False
    if z.device.type == "tpu" and _fused_ops is None:
        _fused_op_table()
    return _hold_fused_operands(vae, z)


def _fused_op_table():
    global _fused_ops
    if _fused_ops is None:
        from sglang_tpu.multimodal_gen import minimax_h3_vae_fused

        _fused_ops = minimax_h3_vae_fused.ops()
    return _fused_ops


def _bits_key(linear) -> tuple:
    # Storage addresses: an fp16 weight is a dtype view of its held buffer.
    w, b = linear.weight, linear.bias
    w_at, b_at = (t.untyped_storage().data_ptr() + t.storage_offset() for t in (w, b))
    return (w_at, w._version, w.dtype, b_at, b._version)


def _hold_fused_operands(vae, z) -> bool:
    """Each block linear's fp16 weight and bias as int16 buffers, made once a
    weight version (the kernels take fp16 bits; a per-call view of an fp16
    tensor is a copy on TPU). An fp16 weight becomes a view of its buffer, so
    the two share one copy. w2's weight (with SPLIT_WEIGHTS also to_qkv's and
    w1's) is held as its bf16 hi and lo [2, n, k] instead, 4 bytes a weight
    more, where a decode stage has left room for them under its peak
    (:func:`_note_stage_room`; before one has, room beside the decoder call
    the batched decode plans), and always off device. With SPLIT_WEIGHTS the
    rest are split in each block program before its kernels (one block's
    copies at a time) where free memory takes that; else in the kernels. The
    split is exact, so every form gives the same output. to_qkv's bias is
    [8, heads * 64] (rows q, k, v, 0...). False if a bias is missing or an
    int16 copy of a non-fp16 weight does not fit."""
    from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
        is_layerwise_offloaded_module,
    )

    blocks = getattr(getattr(vae, "decoder", None), "transformer_blocks", None)
    linears = _block_linears(vae.decoder) if blocks else []
    if not linears or any(m.bias is None for m in linears):
        return False
    if is_layerwise_offloaded_module(vae):
        return False  # its weights move under hooks; held copies would not
    device = linears[0].weight.device
    targets = [block.ff.w2 for block in blocks]
    if _split_weights:
        targets += [m for b in blocks for m in (b.attn.to_qkv, b.ff.w1)]
    split_ids = {id(m) for m in targets}
    room = peak_headroom_bytes(device)
    stage = _stage_room
    if stage is None and room is not None:
        stage = room - decode_batch_units() * unit_call_bytes(vae, z)
    extra = sum(4 * m.weight.numel() for m in targets)
    split = room is None or min(stage, room) - extra >= SPLIT_RESERVE_BYTES
    stale = [
        m
        for m in linears
        if getattr(m, "_h3_bits_key", None) != _bits_key(m)
        or (split and id(m) in split_ids and m.h3_weight_bits.dim() == 2)
    ]
    copies = sum(2 * m.weight.numel() for m in stale if m.weight.dtype != torch.float16)
    free = [b for b in (free_device_bytes(device), room) if b is not None]
    if copies and free and min(free) < copies + SPLIT_RESERVE_BYTES:
        _warn_once(
            "fused-memory",
            "MiniMax H3 fused ViT blocks: %.2f GiB free, int16 copies of the "
            "non-fp16 block weights need %.2f GiB with the reserve; decoding "
            "without them",
            min(free) / 2**30,
            (copies + SPLIT_RESERVE_BYTES) / 2**30,
        )
        return False
    heads = {id(block.attn.to_qkv): block.attn.heads for block in blocks}
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        for m in stale:
            bias = _fp16(m.bias)
            if id(m) in heads:
                bias = bias.view(heads[id(m)], 3, -1).transpose(0, 1).reshape(3, -1)
                bias = torch.cat([bias, bias.new_zeros(5, bias.shape[1])])
            weight = _bits(m.weight)
            if split and id(m) in split_ids:
                weight = _split_held(weight)
            elif m.weight.dtype == torch.float16:
                m.weight.data = weight.view(torch.float16)  # one copy for both
            m.register_buffer("h3_weight_bits", weight, persistent=False)
            rows = 8 if id(m) in heads else 1
            m.register_buffer("h3_bias_bits", _bits(bias, rows), persistent=False)
            m._h3_bits_key = _bits_key(m)
            if device.type == "tpu":
                # Run the copies now: each old fp16 buffer is freed as its
                # int16 one lands, not all at the first block call.
                from torch_tpu._internal.sync.sync import synchronize

                synchronize([m.h3_weight_bits, m.h3_bias_bits], wait=False)
    # One block's copies inside its program: room beside the batched call.
    block = 4 * sum(m.weight.numel() for m in targets) // len(blocks)
    free = free_device_bytes(device)
    roomy = free is None or free >= block + DECODE_BATCH_RESERVE_BYTES
    for m in targets:
        m._h3_split_in_graph = _split_weights and roomy and m.h3_weight_bits.dim() == 2
    return True


def _split_held(bits):
    """int16 fp16 bits [n, k] -> bf16 [2, n, k], their exact hi and lo (the
    kernels' own split; on TPU one Pallas pass)."""
    if bits.device.type == "tpu":
        return (_fused_ops if _fused_ops is not None else _fused_op_table())[4](bits)
    return _split_torch(bits)


def _split_torch(bits):
    with torch.autocast(device_type=bits.device.type, enabled=False):
        return torch.stack(split_fp16(bits.view(torch.float16)))


def split_fp16(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """An fp16 tensor as bf16 (hi, lo) with ``hi + lo == t`` exactly.

    hi is t rounded to bf16 (8 significant bits; bf16 covers fp16's whole
    exponent range, its subnormals included, so no value under- or
    overflows); lo = t - hi has at most 4 significant bits and is exact in
    float32 and in bf16. An infinity gives (inf, 0) (not inf - inf), a NaN
    (nan, nan).
    """
    if t.dtype != torch.float16:
        raise TypeError(f"split_fp16 takes fp16, got {t.dtype}")
    hi = t.to(torch.bfloat16)
    lo = t.float() - hi.float()
    lo = torch.where(torch.isinf(t), torch.zeros_like(lo), lo)
    return hi, lo.to(torch.bfloat16)


def _note_stage_room(vae) -> None:
    """Keeps the least headroom under a decode stage's reserved peak, as it
    would be with no weight held split: the room held splits must fit in (the
    request's peak can come late in the stage, after the first decode)."""
    global _stage_room
    blocks = getattr(getattr(vae, "decoder", None), "transformer_blocks", None)
    room = peak_headroom_bytes(blocks[0].ff.w2.weight.device) if blocks else None
    if room is None:
        return
    for m in _block_linears(vae.decoder):
        if getattr(m, "h3_weight_bits", torch.empty(0)).dim() == 3:
            room += 4 * m.weight.numel()
    _stage_room = room if _stage_room is None else min(_stage_room, room)


def _fp16(t):
    return t.detach().to(torch.float16)


def _bits(t, rows: int | None = None):
    """``t``'s fp16 bits as int16 [rows, -1] (``t``'s own rows by default),
    computed into a buffer of their own: on TPU a dtype view of an fp16
    buffer, and a clone of one, stay views of it, which every block program
    call would copy again."""
    t = _fp16(t).contiguous()
    return t.reshape(t.shape[0] if rows is None else rows, -1).view(torch.int16) & -1


def _eps(norm) -> float:
    return torch.finfo(torch.float32).eps if norm.eps is None else float(norm.eps)


def _fused_rows(tokens: int) -> int:
    return -(-tokens // FUSED_ROW_MULTIPLE) * FUSED_ROW_MULTIPLE


def fused_tables(rotary_pos_emb, rows: int) -> torch.Tensor:
    """The qkv kernel's [2, rows, 128] tables, two 64-wide heads a row:
    cos, 1 past the rotary lanes; sin with rotate_half's sign (-sin below
    rot / 2, sin up to rot), 0 past them; rows past the tokens 0."""
    c, s = (t[0, :, 0, :].float() for t in rotary_pos_emb[:2])
    tokens, rot = c.shape
    pad = FLASH_HEAD_DIM - rot
    c = torch.cat([c, c.new_ones(tokens, pad)], 1)
    s = torch.cat([-s[:, : rot // 2], s[:, rot // 2 :], s.new_zeros(tokens, pad)], 1)
    tables = torch.stack([c, s]).repeat(1, 1, 2)
    return torch.nn.functional.pad(tables, (0, 0, 0, rows - tokens))


def fused_block_fits(block, hidden_states, rotary_pos_emb) -> bool:
    """Whether the fused kernels take this block and input (the decoder has
    passed the flash check): an f32 [batch, rows, width] residual (rows: the
    tokens or :func:`_fused_rows` of them), weighted RMSNorms, scaled
    residuals, a gated SiLU FFN, heads in fours with one q/k norm eps, held
    int16 operands with sides in 256s, rotary tables shared by the batch."""
    attn, ff, h = block.attn, block.ff, hidden_states
    cos = rotary_pos_emb[0] if rotary_pos_emb else None
    linears = (attn.to_qkv, attn.to_out, ff.w1, ff.w2)
    bits = [getattr(m, "h3_weight_bits", None) for m in linears]
    if cos is None or h.dim() != 3 or any(b is None for b in bits):
        return False
    tokens, width, ffn = cos.shape[1], h.shape[-1], bits[3].shape[-1]
    inner = attn.heads * FLASH_HEAD_DIM
    shapes = [(3 * inner, width), (width, inner), (2 * ffn, width), (width, ffn)]
    return (
        h.dtype == torch.float32
        and h.shape[1] in (tokens, _fused_rows(tokens))
        and tokens <= FLASH_MAX_TOKENS
        and (cos.shape[0] == 1 or cos.stride(0) == 0)
        and cos.shape[-1] <= FLASH_HEAD_DIM
        and attn.heads % 4 == 0
        and attn.norm_q.eps == attn.norm_k.eps
        and getattr(block, "use_scale", False)
        and ff.use_gated
        and isinstance(ff.act_fn, torch.nn.SiLU)
        and all(
            isinstance(n, torch.nn.RMSNorm) and n.weight is not None
            for n in (block.norm1, block.norm2)
        )
        and [tuple(b.shape[-2:]) for b in bits] == shapes
        and width % 256 == 0
        and ffn % 256 == 0
    )


def fused_block(block, hidden_states, rotary_pos_emb):
    """The block as minimax_h3_vae_fused computes it: on TPU its five custom
    calls, elsewhere their arithmetic in torch (:data:`_FUSED_TORCH`, the
    same operands). Rows past the tokens are padding (made and dropped here
    if the input has none); rotary_pos_emb may carry the tables third."""
    attn, ff = block.attn, block.ff
    rows, tokens = hidden_states.shape[1], rotary_pos_emb[0].shape[1]
    padded = _fused_rows(tokens)
    h = hidden_states
    if rows < padded:
        h = torch.nn.functional.pad(h, (0, 0, 0, padded - rows))
    if len(rotary_pos_emb) == 3:
        tables = rotary_pos_emb[2]
    else:
        tables = fused_tables(rotary_pos_emb, padded)
    tpu = h.device.type == "tpu"
    qkv, residual, gate, flash, split = _fused_ops or (
        _fused_op_table() if tpu else _FUSED_TORCH
    )
    # Parameters go in as they are (the decode runs under no_grad): a view
    # of one would be a copy on TPU.
    norm1, norm2, scale1, scale2 = (
        block.norm1.weight,
        block.norm2.weight,
        block.scale1,
        block.scale2,
    )

    def operands(m):
        weight = m.h3_weight_bits
        if getattr(m, "_h3_split_in_graph", False):
            weight = split(weight)
        return weight, m.h3_bias_bits

    (wq, bq), (wo, bo), (w1, b1), (w2, b2) = (
        operands(m) for m in (attn.to_qkv, attn.to_out, ff.w1, ff.w2)
    )
    half = rotary_pos_emb[0].shape[-1] // 2
    parts = qkv(h, norm1, wq, bq, tables, _eps(block.norm1), _eps(attn.norm_q), half)
    h = residual(flash(*parts, tokens), wo, bo, scale1, h)
    h = residual(gate(h, norm2, w1, b1, _eps(block.norm2)), w2, b2, scale2, h)
    return h if rows == padded else h[:, :rows]


def _rms_rows(h, weight, eps):
    """The fused RMSNorm prologue: (h * weight, 1/RMS of each row), f32."""
    h = h.float()
    return h * weight, torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)


def _fused_qkv_torch(h, norm, weight, bias, tables, eps, qk_eps, half):
    """The qkv kernel's arithmetic off TPU -> its six bf16 parts (only f32
    summation order and the rsqrt differ)."""
    with torch.autocast(device_type=h.device.type, enabled=False):
        x, r = _rms_rows(h, norm, eps)
        y = _three_products(x, weight) * r
        batch, rows, _ = y.shape
        bias = bias[:3].view(torch.float16).float().view(3, -1, FLASH_HEAD_DIM)
        y = y.view(batch, rows, -1, 3, FLASH_HEAD_DIM) + bias.transpose(0, 1)
        cos, sin = (t[None, :, None, :FLASH_HEAD_DIM] for t in tables)
        parts = []
        for index, t in enumerate(y.unbind(3)):
            if index < 2:  # q and k
                t = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + qk_eps)
                rotated = [t[..., half : 2 * half], t[..., :half], t[..., 2 * half :]]
                t = t * cos + torch.cat(rotated, -1) * sin
            parts += [p.bfloat16() for p in _bf16_parts(t.reshape(batch, rows, -1))]
        return parts


def _fused_residual_torch(x, weight, bias, scale, h):
    """h + (x W^T + b) * scale as the out and w2 kernels compute it, off TPU."""
    if x.dtype == torch.int16:
        x = x.view(torch.float16)
    with torch.autocast(device_type=h.device.type, enabled=False):
        acc = _three_products(x.float(), weight)
        return h + (acc + bias.view(torch.float16).float()) * scale


def _fused_gate_torch(h, norm, weight, bias, eps):
    """fp16 bits of SiLU(gate) * up as the w1 kernel computes it, off TPU."""
    with torch.autocast(device_type=h.device.type, enabled=False):
        x, r = _rms_rows(h, norm, eps)
        y = _three_products(x, weight) * r
        gate, up = (y + bias.view(torch.float16).float()).chunk(2, -1)
        return (gate * up / (1 + torch.exp(-gate))).half().view(torch.int16)


_FUSED_TORCH = (
    _fused_qkv_torch,
    _fused_residual_torch,
    _fused_gate_torch,
    _flash_parts_torch,
    _split_torch,
)


def _fused_block_forward(forward):
    """TransformerBlock.forward: :func:`fused_block` inside a decode that
    runs it, on a block and input the kernels take; stock otherwise."""

    @functools.wraps(forward)
    def block_forward(self, hidden_states, rotary_pos_emb=None):
        if _fused_active and fused_block_fits(self, hidden_states, rotary_pos_emb):
            return fused_block(self, hidden_states, rotary_pos_emb)
        return forward(self, hidden_states, rotary_pos_emb)

    block_forward.__wrapped_for_tpu__ = True
    return block_forward


def _padded_for_fused_blocks(forward_transformer_blocks):
    """ViT3DDecoder.forward_transformer_blocks: when every block runs fused,
    the residual stream is padded to the fused rows and the qkv tables are
    made once for all of them (not once a block)."""

    @functools.wraps(forward_transformer_blocks)
    def padded(self, hidden_states, rotary_pos_emb):
        blocks = list(getattr(self, "transformer_blocks", None) or [])
        if not (
            _fused_active
            and blocks
            and all(fused_block_fits(b, hidden_states, rotary_pos_emb) for b in blocks)
        ):
            return forward_transformer_blocks(self, hidden_states, rotary_pos_emb)
        tokens = hidden_states.shape[1]
        rows = _fused_rows(tokens)
        rope = (*rotary_pos_emb[:2], fused_tables(rotary_pos_emb, rows))
        h = torch.nn.functional.pad(hidden_states, (0, 0, 0, rows - tokens))
        return forward_transformer_blocks(self, h, rope)[:, :tokens]

    padded.__wrapped_for_tpu__ = True
    return padded
