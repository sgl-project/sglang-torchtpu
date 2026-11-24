# SPDX-License-Identifier: Apache-2.0
"""Run Wan2.2-A14B's DiT block stack as one megakernel call per forward.

Runtime patch, tracked debt:
- Why: the torch_tpu block path runs each of the 40 blocks as several XLA
  programs with Ulysses all-to-alls between them; the Pallas megakernel runs
  every block of a forward in one call with the exchange inside it.
- How: with ``SGLANG_TPU_ENABLE_WAN_MEGAKERNEL=1``, :func:`install` wraps
  ``WanTransformer3DModel.post_load_weights`` to decide, identically on every
  rank, whether the kernel can run (:func:`unsupported_reason`). If it can,
  the blocks' projections are packed into the kernel's layout (originals
  freed), the op is registered, and ``forward`` runs the patch embedding, the
  condition embedder and the head in torch around one op call.
- Remove when: the torch_tpu block path reaches the kernel's speed.
"""

from __future__ import annotations

import functools
import logging
import weakref

import torch

from sglang_tpu.envs import envs
from sglang_tpu.multimodal_gen.wan_megakernel import pack

logger = logging.getLogger(__name__)

_STATE = "_tpu_wan_megakernel"
# One registered op a process: both experts (and any later model of the same
# shapes) call it with their own weights; a second registration of the same
# torch op name fails at its first call.
_OPS: dict = {}


def _op_for(cfg, cfg_ranks: int, sp: int, tiling):
    key = (cfg, cfg_ranks, sp, tiling)
    if key not in _OPS:
        from sglang_tpu.multimodal_gen.wan_megakernel import op as stack_op

        _OPS[key] = stack_op.build_stack_op(cfg, cfg_ranks, sp, tiling)
    return _OPS[key]


def requested() -> bool:
    return envs.SGLANG_TPU_ENABLE_WAN_MEGAKERNEL.get()


class _Megakernel:
    def __init__(self, model, cfg, tiling, weights, op, tables, cfg_ranks, sp):
        self.cfg, self.tiling, self.weights, self.op = cfg, tiling, weights, op
        self.tables = tables  # [L, 6, H] f32
        self.cfg_ranks, self.sp = cfg_ranks, sp
        self.rope_cache: dict = {}
        self.kv_cache: list = []  # [(key, text ref, _cross_kv's tuple)], newest first
        self.kv_cache_size = 2
        # _step_constants' entries, or None: every step builds them itself.
        self.step_constants: dict | None = (
            {} if envs.SGLANG_TPU_ENABLE_WAN_MEGAKERNEL_STEP_CONSTANTS.get() else None
        )


def is_packed(model) -> bool:
    return model.__dict__.get(_STATE) is not None


def _gather(value) -> list:
    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.distributed.parallel_state import get_world_group

    group = get_world_group()
    if group.world_size <= 1:
        return [value]
    values = [None] * group.world_size
    dist.all_gather_object(values, value, group=group.cpu_group)
    return values


def _ranks_agree(reason: str | None) -> str | None:
    """Every rank must take the same path: a rank outside the kernel leaves
    the others waiting at its barrier."""
    return next(
        (f"rank {r}: {why}" for r, why in enumerate(_gather(reason)) if why), None
    )


def _try(what, step, *args, **kwargs):
    try:
        return step(*args, **kwargs), None
    except Exception as error:  # noqa: BLE001 - reported to every rank
        logger.exception("Wan megakernel: %s failed", what)
        return None, f"{what}: {type(error).__name__}: {error}"


def unsupported_reason(model) -> str | None:
    from torch.distributed.tensor import DTensor

    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_classifier_free_guidance_world_size,
        get_ring_ctx,
        get_tp_world_size,
        get_ulysses_ctx,
        get_world_size,
    )
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if not current_platform.is_tpu():
        return "not a TPU platform"
    ulysses, _ = get_ulysses_ctx()
    cfg_ranks = get_classifier_free_guidance_world_size()
    if get_ring_ctx()[0] != 1 or get_tp_world_size() != 1:
        return "needs ring degree 1 and TP 1"
    if ulysses * cfg_ranks != get_world_size():
        return f"needs CFG x Ulysses over the whole world ({cfg_ranks} x {ulysses} of {get_world_size()})"
    if model.config.added_kv_proj_dim is not None or model.config.image_dim is not None:
        return "image cross-attention (I2V with CLIP) is not supported"
    if tuple(model.patch_size) != (1, 2, 2):
        return f"patch size {model.patch_size}"
    try:
        from sglang.multimodal_gen.runtime.server_args import get_global_server_args

        sa = get_global_server_args()
    except Exception:  # noqa: BLE001
        sa = None
    if sa is not None:
        for flag in ("use_fsdp_inference", "dit_cpu_offload", "dit_layerwise_offload"):
            if getattr(sa, flag, False):
                return f"--{flag.replace('_', '-')}"
        if getattr(sa, "lora_path", None):
            return "LoRA"
        if getattr(getattr(sa, "pipeline_config", None), "expand_timesteps", False):
            return "per-token timesteps (TI2V)"
    for b in model.blocks:
        if type(b).__name__ != "WanTransformerBlock" or b.use_offline_qk_rotation:
            return f"unsupported block {type(b).__name__}"
        for m in (
            b.to_q,
            b.to_k,
            b.to_v,
            b.to_out,
            b.attn2.to_q,
            b.attn2.to_out,
            b.ffn.fc_in,
            b.ffn.fc_out,
        ):
            w = m.weight
            if (
                isinstance(w, DTensor)
                or w.dtype != torch.bfloat16
                or w.device.type == "cpu"
            ):
                return "projections are not plain bf16 device tensors (FSDP, offload, quantization)"
    cfg = pack.config_from_model(model)
    if cfg.heads % ulysses:
        return f"{cfg.heads} heads do not split over Ulysses {ulysses}"
    return None


def enable(model) -> bool:
    """Pack ``model``'s blocks and register the op, if every rank can."""
    if model.__dict__.get(_STATE, False) is not False:
        return is_packed(model)
    reason, error = _try("the support check", unsupported_reason, model)
    reason = _ranks_agree(error or reason)
    if reason is not None:
        logger.warning("Wan megakernel off (%s); blocks run in torch.", reason)
        model.__dict__[_STATE] = None
        return False
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_classifier_free_guidance_world_size,
        get_ulysses_ctx,
    )
    from sglang_tpu.multimodal_gen.wan_megakernel.tiling import Tiling

    cfg = pack.config_from_model(model)
    tiling = Tiling(
        pipelined_epilogues=envs.SGLANG_TPU_OPT_WAN_MEGAKERNEL_PIPELINED_EPILOGUES.get(),
        dma_bounds_checks=not envs.SGLANG_TPU_OPT_WAN_MEGAKERNEL_SKIP_BOUNDS_CHECKS.get(),
    )
    cfg_ranks, sp = get_classifier_free_guidance_world_size(), get_ulysses_ctx()[0]
    # Register the op first: packing frees the blocks' projections.
    op, error = _try("registering the op", _op_for, cfg, cfg_ranks, sp, tiling)
    reason = _ranks_agree(error)
    if reason is not None:
        logger.warning("Wan megakernel off (%s); blocks run in torch.", reason)
        model.__dict__[_STATE] = None
        return False
    tables = pack.modulation_tables(model).to(model.blocks[0].to_q.weight.device)
    weights, error = _try(
        "packing",
        pack.pack_blocks,
        model,
        cfg,
        tiling.cols,
        stage_on_host=True,
        down_cols=tiling.down_cols if tiling.mlp_mode == "twopass" else None,
    )
    failed = _ranks_agree(error)
    if failed is not None:
        # Ranks that packed have freed their blocks: no torch path to fall back to.
        raise RuntimeError(f"Wan megakernel: packing failed ({failed})")
    model.__dict__[_STATE] = _Megakernel(
        model, cfg, tiling, weights, op, tables, cfg_ranks, sp
    )
    logger.info(
        "Wan megakernel on: %d blocks, CFG %d x Ulysses %d, %.1f GiB packed weights a device.",
        cfg.layers,
        cfg_ranks,
        sp,
        sum(w.numel() * w.element_size() for w in weights.values()) / 2**30,
    )
    return True


def _rope_rows(
    model, state, frames: int, height: int, width: int, rows: int, rank: int, device
):
    """``(cos, sin)`` ``[rows, D]`` f32 in the kernel's split layout for padded
    rows ``[rank * rows, (rank + 1) * rows)``, from sglang's own rotary tables."""
    key = (frames, height, width, rows, rank)
    hit = state.rope_cache.get(key)
    if hit is not None:
        return hit
    tokens = frames * height * width
    start = rank * rows
    idx = torch.arange(start, start + rows, device=device, dtype=torch.long)
    real = idx < tokens
    idx = torch.where(real, idx, torch.zeros_like(idx))
    pos = torch.stack(
        (idx // (height * width), (idx % (height * width)) // width, idx % width), dim=1
    )
    cos, sin = model.rotary_emb.forward_uncached(
        pos
    )  # [rows, D/2] (interleaved-pair angles)
    cos, sin = cos.float(), sin.float()
    cos = torch.where(real[:, None], cos, torch.ones_like(cos))
    sin = torch.where(real[:, None], sin, torch.zeros_like(sin))
    out = (
        torch.cat([cos, cos], dim=1).contiguous(),
        torch.cat([-sin, sin], dim=1).contiguous(),
    )
    state.rope_cache[key] = out
    return out


def _step_constants(state, tokens: int, sp: int, device):
    """``(tiling, rows, op, valid)`` for ``tokens`` real rows over ``sp`` devices:
    ``for_tokens``' tiling and rows a device, its op, and ``valid``, the ``[1]``
    int32 real-row count on ``device``. Built once a shape and shared by every
    step and request of it (the kernel only reads ``valid``). The count goes up
    without waiting: a blocking upload (``torch.tensor(..., device=tpu)``)
    returns only once every program queued on the TPU has finished
    (sglang_tpu/host_upload.py), so one a step stalls the host behind the
    previous step and the TPU then idles while the host builds and launches
    this step's programs."""
    key = (tokens, sp, str(device))
    hit = state.step_constants.get(key)
    if hit is None:
        from sglang_tpu.multimodal_gen.wan_megakernel.tiling import for_tokens

        tiling, rows = for_tokens(tokens, sp, state.tiling)
        host = torch.tensor([tokens], dtype=torch.int32)
        valid = host.to(device, non_blocking=True)
        op = _op_for(state.cfg, state.cfg_ranks, sp, tiling)
        # The host copy stays referenced while the upload may be in flight.
        hit = state.step_constants[key] = (tiling, rows, op, valid, host)
    return hit[:4]


def _cross_kv(model, state, encoder_hidden_states, b, batch, batch_ctx):
    """Sample ``b``'s text side of cross-attention: each layer's K/V of the
    embedded text and the keys the kernel attends to (``pack.cross_keys``: the
    prompt's tokens and copies of its padding row). It depends only on the
    request's text, so it is cached a request: an entry hits only for the same
    request and the same live text tensor (held by weak reference, so a later
    tensor at a reused address never matches). A few entries, so one rank serving
    both CFG branches (or a batch) keeps them all."""
    rid = getattr(batch_ctx, "request_id", None) if batch_ctx is not None else None
    key = (rid, id(batch_ctx), b)
    if rid is not None:
        for i, (k, ref, kv) in enumerate(state.kv_cache):
            if k == key and ref() is encoder_hidden_states:
                state.kv_cache.insert(0, state.kv_cache.pop(i))
                return kv
    text = (
        encoder_hidden_states[b : b + 1]
        if encoder_hidden_states.shape[0] == batch
        else encoder_hidden_states
    )
    ctx = model.condition_embedder.text_embedder(text)
    k2, v2 = pack.cross_kv(model, ctx[0].to(torch.bfloat16), state.cfg)
    kv = (k2, v2, *pack.cross_keys(k2, v2, state.tiling.cross_keys))
    if rid is not None:
        state.kv_cache.insert(0, (key, weakref.ref(encoder_hidden_states), kv))
        del state.kv_cache[state.kv_cache_size :]
    return kv


def _schedule(state, batch_ctx):
    """The request's step count (read once a request)."""
    rid = getattr(batch_ctx, "request_id", None)
    cached = state.__dict__.get("_sched")
    if cached is None or cached[0] != rid:
        ts = getattr(batch_ctx, "timesteps", None)
        cached = (rid, int(ts.numel()) if ts is not None else None)
        state.__dict__["_sched"] = cached
    return cached[1]


_OP_CALLS = {"rid": None, "n": 0}


def _count_op_call(state, batch_ctx, batch: int) -> None:
    """Count this rank's kernel calls for the current request (both experts share
    the count) and log the total once it reaches steps x samples a forward."""
    rid = getattr(batch_ctx, "request_id", None)
    if rid != _OP_CALLS["rid"]:
        _OP_CALLS.update(rid=rid, n=0)
    _OP_CALLS["n"] += 1
    steps = _schedule(state, batch_ctx)
    if steps is not None and _OP_CALLS["n"] == steps * batch:
        logger.info(
            "Wan megakernel: request %s ran the block-stack op %d times%s",
            rid,
            _OP_CALLS["n"],
            " (warmup)" if getattr(batch_ctx, "is_warmup", False) else "",
        )


def _forward_one(
    model,
    state,
    hidden_states,
    encoder_hidden_states,
    b,
    batch,
    timestep,
    batch_ctx,
    sp,
    rank,
):
    """Sample ``b`` (``hidden_states [1, C, F, H, W]``, its text in the batch's
    ``encoder_hidden_states``, a ``[1]`` timestep) through the embeddings, the
    kernel and the head; returns the model's output."""
    from sglang.multimodal_gen.runtime.distributed import (
        sequence_model_parallel_all_gather,
    )
    from sglang_tpu.multimodal_gen.wan_megakernel.tiling import for_tokens

    cfg = state.cfg
    _, _, frames, height, width = hidden_states.shape
    pt, ph, pw = model.patch_size
    f_, h_, w_ = frames // pt, height // ph, width // pw
    x = model.patch_embedding(hidden_states).flatten(2).transpose(1, 2)  # [1, S, H]
    tokens = x.shape[1]
    emb = model.condition_embedder
    temb = emb.time_embedder(timestep, None)
    timestep_proj = emb.time_modulation(temb).unflatten(1, (6, -1))
    e = pack.modulation(state.tables, timestep_proj)
    k2, v2, keys, key_weights = _cross_kv(
        model, state, encoder_hidden_states, b, batch, batch_ctx
    )
    if state.step_constants is not None:
        tiling, rows, op, valid = _step_constants(state, tokens, sp, x.device)
    else:
        tiling, rows = for_tokens(tokens, sp, state.tiling)
        op = valid = None
    total = rows * sp
    xs = torch.nn.functional.pad(x[0], (0, 0, 0, total - tokens))[
        rank * rows : (rank + 1) * rows
    ]
    cos, sin = _rope_rows(model, state, f_, h_, w_, rows, rank, x.device)
    if valid is None:
        valid = torch.tensor([tokens], dtype=torch.int32, device=x.device)
    if op is None:
        op = _op_for(cfg, state.cfg_ranks, sp, tiling)
    out = op(
        xs.to(torch.bfloat16).contiguous(),
        cos,
        sin,
        e,
        k2,
        v2,
        keys,
        key_weights,
        valid,
        *(state.weights[n] for n in pack.WEIGHT_ORDER),
    )
    if batch_ctx is not None:
        _count_op_call(state, batch_ctx, batch)
    # The head is per row: run it on my rows, then gather its narrow output.
    shift, scale = (model.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
    hidden = model.norm_out(out.unsqueeze(0), shift, scale)
    hidden, _ = model.proj_out(hidden)
    hidden = sequence_model_parallel_all_gather(hidden.contiguous(), dim=1)[:, :tokens]
    hidden = hidden.reshape(1, f_, h_, w_, pt, ph, pw, -1)
    hidden = hidden.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return hidden.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def forward(
    model,
    hidden_states,
    encoder_hidden_states,
    timestep,
    encoder_hidden_states_image=None,
    guidance=None,
    **kwargs,
):
    """``WanTransformer3DModel.forward`` with the block stack in one kernel call a
    sample (one CFG branch a rank or both in turn, T2V, scalar timesteps)."""
    from sglang.multimodal_gen.runtime.distributed import (
        sequence_model_parallel_all_gather,
    )
    from sglang.multimodal_gen.runtime.distributed.parallel_state import get_ulysses_ctx
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        get_forward_context,
    )

    state = model.__dict__[_STATE]
    if not isinstance(encoder_hidden_states, torch.Tensor):
        encoder_hidden_states = encoder_hidden_states[0]
    if (
        isinstance(encoder_hidden_states_image, list)
        and len(encoder_hidden_states_image) > 0
    ):
        raise RuntimeError("Wan megakernel: image embeddings are not supported")
    batch_ctx = get_forward_context().forward_batch
    sp, rank = get_ulysses_ctx()
    # A video request whose latent frames divide the SP degree arrives frame-sharded
    # (``shard_latents_for_sp``): this rank holds frames [rank F_l, (rank+1) F_l).
    # Gather the (small) latent, run the whole sequence, return this rank's frames.
    frame_sharded = sp > 1 and bool(getattr(batch_ctx, "did_sp_shard_latents", False))
    if frame_sharded:
        local_frames = hidden_states.shape[2]
        hidden_states = sequence_model_parallel_all_gather(
            hidden_states.contiguous(), dim=2
        )
    batch = hidden_states.shape[0]
    if timestep.dim() != 1 or timestep.shape[0] not in (1, batch):
        raise RuntimeError(
            f"Wan megakernel takes scalar timesteps a sample (got {tuple(timestep.shape)})"
        )
    state.kv_cache_size = max(2, batch)

    def pick(t, b):  # sample b of a per-sample tensor, or the one shared by all
        return t[b : b + 1] if t.shape[0] == batch else t

    outs = [
        _forward_one(
            model,
            state,
            pick(hidden_states, b),
            encoder_hidden_states,
            b,
            batch,
            pick(timestep, b),
            batch_ctx,
            sp,
            rank,
        )
        for b in range(batch)
    ]
    out = outs[0] if batch == 1 else torch.cat(outs, dim=0)
    if frame_sharded:
        out = out[:, :, rank * local_frames : (rank + 1) * local_frames]
    return out


def install() -> None:
    """Route ``WanTransformer3DModel``'s blocks through the kernel when every rank can."""
    from sglang.multimodal_gen.runtime.models.dits import wanvideo

    cls = wanvideo.WanTransformer3DModel
    if getattr(cls, "_tpu_wan_megakernel_installed", False):
        return
    post_load = cls.post_load_weights
    torch_forward = cls.forward

    def _post_load_weights(self) -> None:
        post_load(self)
        enable(self)

    @functools.wraps(torch_forward)
    def _forward(self, *args, **kwargs):
        if is_packed(self):
            return forward(self, *args, **kwargs)
        return torch_forward(self, *args, **kwargs)

    cls.post_load_weights = _post_load_weights
    cls.forward = _forward
    cls._tpu_wan_megakernel_installed = True
