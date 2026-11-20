# SPDX-License-Identifier: Apache-2.0
"""HiDream-O1-Image on TPU workers: bucketed prompt rows and split attention.

Runtime patch, tracked debt:
- Why: the DiT attends [prompt | image] under a dense [1, 1, S, S] mask
  (causal prompt rows; the timestep token and image rows see every key) that
  ``LocalAttention`` hands to plain SDPA. Every activation has the prompt's
  row count, so each new prompt compiles the DiT again, and the stage builds
  the mask and positions with device ops. The config refuses SP.
- How: :func:`install`
  (1) builds each CFG branch's inputs on the host: the prompt tokens without
      the timestep token, padded so the rows ``[image | timestep | prompt]``
      fill a multiple of 128, their RoPE table and the real key count;
  (2) splits attention exactly as the mask does: image and timestep rows
      attend every real key in one kernel call that reads the key count on
      device (padded keys are never read); prompt rows attend the prompt
      causally. Prompts of one bucket share one compiled program;
  (3) under Ulysses, shards the rows through a device index (one program on
      every rank) and runs both CFG branches in one call, the shorter prompt
      padded to the longer bucket; the config accepts Ulysses only (no ring,
      K/V gather or CFG parallel).
- Upstream PR or issue: none.
- Remove when: the DiT takes a platform-provided prompt bucket and key count,
  its attention passes the mask structure to the backend, and the pipeline
  shards its rows under Ulysses.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import inspect
from typing import NamedTuple

import torch

from sglang.multimodal_gen.configs.pipeline_configs import (
    hidream_o1_image as pipeline_config,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_parallel_rank,
    get_sp_world_size,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    qwen3_apply_rotary_pos_emb,
)
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_output_all_to_all,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import hidream_o1_image as dit
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    hidream_o1_image as stages,
)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang_tpu.multimodal_gen import tpu_splash_attn

# [image | timestep | prompt] rows pad to a multiple of this: at 2048x2048
# (4096 patches) prompts of up to 128 tokens share one bucket. A multiple of
# every Ulysses degree the 8 K/V heads allow.
ROW_ALIGNMENT = 128

_Config = pipeline_config.HiDreamO1ImagePipelineConfig
_Stage = stages.HiDreamO1ImageBeforeDenoisingStage
_STOCK_FORWARD = dit.HiDreamO1ImageTransformer.forward
_STOCK_ATTENTION_FORWARD = dit.HiDreamO1Attention.forward
_STOCK_BUILD_CONDITIONING = _Stage._build_conditioning
_STOCK_STAGE_FORWARD = _Stage.forward
_COND_KEYS = (
    pipeline_config.HIDREAM_O1_COND_KEY,
    pipeline_config.HIDREAM_O1_NEG_COND_KEY,
)
_STOCK_VALIDATE = _Config.validate_server_args
_STOCK_DEPLOYMENT = _Config.get_model_deployment_config
_STOCK_PREDICT_WITH_CFG = stages.HiDreamO1ImageDenoisingStage._predict_noise_with_cfg
_PREDICT_SIGNATURE = inspect.signature(_STOCK_PREDICT_WITH_CFG)
_STACKED_KEY = "hidream_o1_tpu_stacked_layout"


class AttentionLayout(NamedTuple):
    """How a layer's rows attend; passed where the stock DiT passes its mask."""

    query_rows: int  # image + timestep rows, in front: they see every real key
    # int32 [groups, 2]: [0, real keys] (image, timestep, real prompt) of each
    # equal share of the batch rows (the CFG branches of a batched call).
    keys: torch.Tensor


class Layout(NamedTuple):
    """One request branch's DiT inputs for this rank, built on the host."""

    cos: torch.Tensor  # [1, rows of this rank, head_dim]
    sin: torch.Tensor
    attention: AttentionLayout
    shard: torch.Tensor | None  # int64 rows of this rank under Ulysses


def _to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    # A fresh offset-0 tensor: compiled programs guard strides and offsets.
    return t.clone(memory_format=torch.contiguous_format).to(device)


@functools.lru_cache(maxsize=64)
def _key_count(real: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([[0, real]], dtype=torch.int32).to(device)


@functools.lru_cache(maxsize=8)
def _shard_rows(rows: int, sp: int, rank: int, device: torch.device):
    share = rows // sp
    return torch.arange(rank * share, (rank + 1) * share).to(device)


@functools.lru_cache(maxsize=4)
def _rotary(head_dim: int, rope_theta: float, mrope_section: tuple[int, ...]):
    return dit.HiDreamO1RotaryEmbedding(head_dim, rope_theta, mrope_section)


def row_positions(
    text_len: int, h_patches: int, w_patches: int, rows: int
) -> torch.Tensor:
    """The stock mrope positions, reordered to ``[image | timestep | prompt]``."""
    stock = stages.build_hidream_o1_position_ids(
        text_len=text_len,
        h_patches=h_patches,
        w_patches=w_patches,
        device=torch.device("cpu"),
    )
    # Padded prompt rows take position 0; no real row reads them.
    pad = stock.new_zeros(3, 1, rows - stock.shape[-1])
    image, stamp = stock[..., text_len:], stock[..., text_len - 1 : text_len]
    return torch.cat([image, stamp, stock[..., : text_len - 1], pad], dim=-1)


def build_conditioning(
    self, prompt, *, batch_size, h_patches, w_patches, mask_dtype, device, rows=0
):
    """``_build_conditioning`` in the row layout of :func:`forward`, built on the
    host, with at least ``rows`` rows."""
    input_ids = self._encode_prompt(prompt, device=torch.device("cpu"))
    text_len, image_len = input_ids.shape[-1], h_patches * w_patches
    rows = max(rows, -(-(image_len + text_len) // ROW_ALIGNMENT) * ROW_ALIGNMENT)
    # The stage appends the timestep token last; the DiT places it itself.
    prompt_ids = input_ids.new_zeros(1, rows - image_len - 1)
    prompt_ids[:, : text_len - 1] = input_ids[:, :-1]
    arch = get_global_server_args().pipeline_config.dit_config.arch_config
    rotary = _rotary(arch.head_dim, arch.rope_theta, tuple(arch.mrope_section))
    cos, sin = rotary(row_positions(text_len, h_patches, w_patches, rows), mask_dtype)
    sp, shard = get_sp_world_size(), None
    if sp > 1:
        share, rank = rows // sp, get_sp_parallel_rank()
        cos, sin = (t[:, rank * share : (rank + 1) * share] for t in (cos, sin))
        shard = _shard_rows(rows, sp, rank, device)
    keys = _key_count(image_len + text_len, device)
    layout = Layout(
        _to_device(cos, device),
        _to_device(sin, device),
        AttentionLayout(image_len + 1, keys),
        shard,
    )
    # Read-only in the DiT (an embedding lookup), so a stride-0 batch.
    return {
        "input_ids": prompt_ids.to(device).expand(batch_size, -1),
        "tpu_layout": layout,
    }


def stage_forward(self, batch, server_args):
    """The stock stage; under Ulysses the CFG branches then share the larger
    bucket, so they run as one batch (see :func:`predict_noise_with_cfg`)."""
    batch = _STOCK_STAGE_FORWARD(self, batch, server_args)
    conds = [batch.extra[key] for key in _COND_KEYS]
    if conds[1] is None or get_sp_world_size() == 1:
        return batch
    prompt_rows = [cond["input_ids"].shape[1] for cond in conds]
    if prompt_rows[0] == prompt_rows[1]:
        return batch
    short = prompt_rows.index(min(prompt_rows))
    patch_size = server_args.pipeline_config.dit_config.arch_config.patch_size
    h_patches, w_patches = batch.height // patch_size, batch.width // patch_size
    conds[short] = batch.extra[_COND_KEYS[short]] = build_conditioning(
        self,
        stages._resolve_single_prompt(
            (batch.prompt, batch.negative_prompt)[short], field_name="prompt"
        ),
        batch_size=len(batch.seeds),
        h_patches=h_patches,
        w_patches=w_patches,
        mask_dtype=conds[short]["tpu_layout"].cos.dtype,
        device=self.device,
        rows=max(prompt_rows) + h_patches * w_patches + 1,
    )
    batch.prompt_embeds = [conds[0]["input_ids"]]
    batch.negative_prompt_embeds = [conds[1]["input_ids"]]
    return batch


def forward(self, *args, tpu_layout: Layout | None = None, **kwargs):
    """``HiDreamO1ImageTransformer.forward`` on ``[image | timestep | prompt]``
    rows; ``input_ids`` are the padded prompt tokens without the timestep token.

    Every op but attention is per row, so the reordered rows give the stock
    rows' outputs; :func:`attention_forward` applies the stock mask.
    """
    if tpu_layout is None:
        return _STOCK_FORWARD(self, *args, **kwargs)
    hidden_states, timestep = kwargs["hidden_states"], kwargs["timestep"]
    model, text = self.model, self.model.language_model
    prompt = text.embed_tokens(kwargs["input_ids"])
    stamp = model.t_embedder1(timestep).to(prompt.dtype)
    image = model.x_embedder(hidden_states).to(prompt.dtype)
    hidden = torch.cat([image, stamp[:, None], prompt], dim=1)
    if tpu_layout.shard is not None:
        # A device index: every rank runs one program on a fresh tensor.
        hidden = hidden.index_select(1, tpu_layout.shard)
    residual = None
    for layer in text.layers:
        hidden, residual = layer(
            hidden, residual, tpu_layout.cos, tpu_layout.sin, tpu_layout.attention
        )
    hidden, _ = text.norm(hidden, residual)
    x_pred = model.final_layer2(hidden)
    if tpu_layout.shard is not None:
        x_pred = sequence_model_parallel_all_gather(x_pred.contiguous(), dim=1)
    # The image rows lead; fp32 as in the stock forward.
    return x_pred[:, : hidden_states.shape[1]].float()


def _masked_attention(q, k, v, layout: AttentionLayout, scale: float):
    """``[B, S, H, D]`` attention of this rank's rows under the stock mask."""
    ulysses = get_ulysses_parallel_world_size() > 1
    if ulysses:
        q, k, v = (_usp_input_all_to_all(t, head_dim=2) for t in (q, k, v))
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    rows, groups = layout.query_rows, layout.keys.shape[0]
    # Image and timestep rows: every real key of their group (the padded
    # prompt keys trail), one kernel call per group.
    full = torch.cat(
        [
            tpu_splash_attn.tpu_ring_attention_step(*args, None, scale)[0]
            for args in zip(
                q[:, :, :rows].chunk(groups),
                k.chunk(groups),
                v.chunk(groups),
                layout.keys.unbind(0),
                strict=True,
            )
        ]
    )
    # Prompt rows: the prompt, causally; padded rows trail every real one.
    causal = tpu_splash_attn.tpu_attention(
        q[:, :, rows:], k[:, :, rows:], v[:, :, rows:], scale=scale, is_causal=True
    )
    out = torch.cat([full, causal], dim=2).transpose(1, 2)
    return _usp_output_all_to_all(out, head_dim=2) if ulysses else out


def attention_forward(self, hidden_states, cos, sin, attention_mask):
    """``HiDreamO1Attention.forward`` taking an :class:`AttentionLayout` mask."""
    if not isinstance(attention_mask, AttentionLayout):
        return _STOCK_ATTENTION_FORWARD(self, hidden_states, cos, sin, attention_mask)
    batch_size, seq_len = hidden_states.shape[:2]
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q = self.q_norm(q.reshape(batch_size, seq_len, self.num_heads, self.head_dim))
    k = self.k_norm(k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim))
    v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
    q, k = qwen3_apply_rotary_pos_emb(q, k, cos.unsqueeze(2), sin.unsqueeze(2))
    out = _masked_attention(q, k, v, attention_mask, self.scaling)
    output, _ = self.o_proj(out.reshape(batch_size, seq_len, -1))
    return output


def _stack(layouts: tuple[Layout, ...], batch_size: int) -> Layout:
    """One layout over the branches' rows, stacked batch-major."""

    def rows(name):
        return torch.cat(
            [getattr(t, name).expand(batch_size, -1, -1) for t in layouts]
        ).contiguous()

    keys = torch.cat([t.attention.keys for t in layouts])
    first = layouts[0]
    return first._replace(
        cos=rows("cos"), sin=rows("sin"), attention=first.attention._replace(keys=keys)
    )


def predict_noise_with_cfg(self, *args, **kwargs):
    """Both CFG branches in one DiT call under Ulysses when their prompts
    share a bucket (on one core the rows already fill the matmuls).

    The branches' rows stack on the batch axis and each attends only its own
    prompt's keys, so every row computes what its serial call computes.
    """
    call = _PREDICT_SIGNATURE.bind(self, *args, **kwargs).arguments
    branches, gate = call["cfg_policy"].branches, call["cfg_gate_state"]
    layouts = tuple(branch.kwargs.get("tpu_layout") for branch in branches)
    if (
        len(branches) != 2
        or get_sp_world_size() == 1
        or (gate and gate["active"])
        or None in layouts
        or layouts[0].cos.shape != layouts[1].cos.shape
    ):
        return _STOCK_PREDICT_WITH_CFG(self, *args, **kwargs)
    batch, latents = call["batch"], call["latent_model_input"]
    # Built once per request: the branches' inputs do not change per step.
    if _STACKED_KEY not in batch.extra:
        batch.extra[_STACKED_KEY] = (
            torch.cat([branch.kwargs["input_ids"] for branch in branches]),
            _stack(layouts, latents.shape[0]),
        )
    input_ids, stacked = batch.extra[_STACKED_KEY]
    branches[0].configure_batch(batch)
    with set_forward_context(
        current_timestep=call["timestep_index"],
        attn_metadata=call["attn_metadata"],
        forward_batch=batch,
        quality=batch.sampling_params.quality,
    ):
        x_pred = self._predict_noise(
            current_model=call["current_model"],
            latent_model_input=torch.cat([latents] * 2),
            timestep=torch.cat([call["timestep"]] * 2),
            target_dtype=call["target_dtype"],
            guidance=call["guidance"],
            **{
                **branches[0].kwargs,
                "input_ids": input_ids,
                "tpu_layout": stacked,
            },
        )
    config = call["server_args"].pipeline_config
    preds = [config.slice_noise_pred(x, call["latents"]) for x in x_pred.chunk(2)]
    scale = config.get_classifier_free_guidance_scale(
        batch, call["current_guidance_scale"]
    )
    return call["cfg_policy"].combine(preds, batch, scale, config)


def validate_server_args(self, server_args) -> None:
    """The stock checks, accepting Ulysses (each rank runs a share of the rows)."""
    sp, kv_heads = (
        server_args.sp_degree,
        self.dit_config.arch_config.num_key_value_heads,
    )
    if sp > 1:
        # K/V gather aliases ulysses_degree to its own degree.
        if (
            server_args.kv_gather_degree > 1
            or server_args.ulysses_degree != sp
            or kv_heads % sp
        ):
            raise ValueError(
                "HiDream-O1-Image on TPU shards its rows with Ulysses only, over "
                f"a degree that divides its {kv_heads} K/V heads: run "
                "--ulysses-degree with --ring-degree 1 and --kv-gather-degree 1."
            )
        server_args = copy.copy(server_args)
        server_args.sp_degree = 1
    _STOCK_VALIDATE(self, server_args)


def shard_latents_for_sp(self, batch, latents):
    """Latents stay whole: the DiT takes each rank's rows itself."""
    return latents, False


def get_model_deployment_config(self):
    """Without CFG parallel: under Ulysses both branches already run as one
    batch, and two CFG halves running SP subgroups at once gave identical
    requests different images on TPU."""
    return dataclasses.replace(
        _STOCK_DEPLOYMENT(self),
        auto_enable_cfg_parallel=False,
        supports_cfg_parallel=False,
    )


def install_config() -> None:
    """The pipeline-config patches; the launcher validates with them too."""
    _Config.validate_server_args = validate_server_args
    _Config.shard_latents_for_sp = shard_latents_for_sp
    _Config.get_model_deployment_config = get_model_deployment_config


def install() -> None:
    install_config()
    _Stage._build_conditioning = build_conditioning
    _Stage.forward = stage_forward
    dit.HiDreamO1ImageTransformer.forward = forward
    dit.HiDreamO1Attention.forward = attention_forward
    stages.HiDreamO1ImageDenoisingStage._predict_noise_with_cfg = predict_noise_with_cfg
    # Every rank runs the same layer program on its share of the rows.
    dit.HiDreamO1ImageTransformer._compile_regions_spmd_safe = True
