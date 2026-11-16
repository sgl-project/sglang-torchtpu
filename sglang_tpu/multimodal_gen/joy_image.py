# SPDX-License-Identifier: Apache-2.0
"""JoyAI-Image-Edit on TPU workers: one compiled DiT block per prompt bucket.

Runtime patch, tracked debt:
- Why: torch_tpu compiles static shapes. The JoyImage DiT attends
  [image | prompt] jointly at the prompt's exact row count (the condition
  image's ~1024 vision tokens plus the prompt's tokens), so every new prompt,
  and the negative prompt, compiles every block again, and the eager Qwen3-VL
  text encoder compiles its ops once per token count. Under Ulysses every rank
  runs all prompt rows (a replicated suffix). The image RoPE is a complex
  multiply (see rope.py) on positions built with device boolean masks.
- How: :func:`install`
  (1) wraps the DiT's ``__call__``, outside the compiled blocks: the prompt
      rows pad with zeros to a multiple of 256 (at least 512, the kernel's
      minimum key count), then of the SP degree, and each Ulysses rank takes
      its share of them. The image RoPE table is built on the host once per
      request, and the real image and prompt key counts go in as device int32
      tensors;
  (2) replaces the DiT's ``forward`` and ``MMDoubleStreamBlock.forward`` on
      those inputs: the same layers, the image RoPE in the pair-swap form, and
      queries attending the real keys of the image run, then of the prompt run,
      as resumable kernel steps. These are the keys the stock DiT attends;
      padded keys are never read and padded rows' outputs are dropped;
  (3) right-pads the encoder's prompt tokens to a multiple of 256. The encoder
      is causal, so real tokens never see the padding, and only the real rows
      reach the DiT;
  (4) marks the regionally compiled blocks spmd-safe.
- Upstream PR or issue: none.
- Remove when: the DiT takes a platform-provided prompt bucket and key counts
  and shards its prompt rows under SP, and RoPE goes through a platform
  rotation.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import torch
import torch.nn.functional as F
from einops import rearrange

from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_sp_parallel_rank,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.layers.layernorm import apply_qk_norm
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_output_all_to_all,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.models.dits import joy_image as dit
from sglang.multimodal_gen.runtime.pipelines import joy_image as pipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageEncodingStage,
)
from sglang_tpu.multimodal_gen import tpu_splash_attn
from sglang_tpu.multimodal_gen.rope import apply_rotary_embedding_pair_swap

# Prompts of up to ~250 tokens next to a 1024-token condition image share the
# 1280-row bucket; the kernel's prompt step needs >= 512 keys.
TEXT_ALIGNMENT = 256
MIN_TEXT_ROWS = 512
PROMPT_TOKEN_ALIGNMENT = 256

_STOCK_FORWARD = dit.JoyTransformer3DModel.forward
_STOCK_BLOCK_FORWARD = dit.MMDoubleStreamBlock.forward
_STOCK_ROPE = dit.JoyTransformer3DModel._compute_rope_for_local_shard
_STOCK_CREATE_STAGES = pipeline.JoyImageEditPipeline.create_pipeline_stages
_STOCK_ENCODING_IMAGE_EDIT = ImageEncodingStage.encoding_image_edit


class Layout(NamedTuple):
    """One request's DiT inputs for this rank, built on the host."""

    image_rope: torch.Tensor  # [image rows per rank, cos | sin], float32
    keys: tuple[torch.Tensor, torch.Tensor]  # int32 [0, real keys]: image, prompt


@functools.lru_cache(maxsize=64)
def _counts(real: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([0, real], dtype=torch.int32).to(device)


@functools.lru_cache(maxsize=4)
def _image_rope(model, rows: int, rank: int, sizes, device) -> torch.Tensor:
    """The stock RoPE table of this rank's image rows, computed on the host."""
    cos, sin = _STOCK_ROPE(model, rows, rank, sizes, torch.device("cpu"))
    return torch.cat((cos.float(), sin.float()), -1).to(device)


def _dit_call(self, *args, **kwargs):
    """Pads and shards the prompt rows before the DiT runs.

    Padded rows are zeros; the attention reads only the real prompt keys
    (``Layout.keys``), and their outputs are dropped with the prompt stream.
    """
    hidden, states = kwargs.get("hidden_states"), kwargs.get("encoder_hidden_states")
    batch = get_forward_context().forward_batch
    sp = get_sp_world_size()
    if (
        args
        or hidden is None
        or states is None
        or getattr(batch, "vae_image_sizes", None) is None
        or (
            sp > 1
            and (
                not getattr(batch, "enable_sequence_shard", False)
                or get_ring_parallel_world_size() > 1
            )
        )
    ):
        return torch.nn.Module.__call__(self, *args, **kwargs)
    states = states[0] if isinstance(states, list) else states
    batch_size, text_len = hidden.shape[0], states.shape[1]
    if states.shape[0] != batch_size:
        states = states.repeat_interleave(batch_size // states.shape[0], dim=0)
    rank = get_sp_parallel_rank() if sp > 1 else 0
    bucket = max(MIN_TEXT_ROWS, -(-text_len // TEXT_ALIGNMENT) * TEXT_ALIGNMENT)
    rows = -(-bucket // sp)
    start = min(rank * rows, text_len)
    # A new contiguous tensor of this rank's rows only: compiled programs
    # guard input strides and storage offsets.
    local = states.new_zeros(batch_size, rows, states.shape[2])
    local[:, : min(rows, text_len - start)] = states[:, start : start + rows]
    sizes = tuple(tuple(s) for s in batch.vae_image_sizes)
    image_len, device = hidden.shape[1], hidden.device
    layout = Layout(
        _image_rope(self, -(-image_len // sp), rank, sizes, device),
        (_counts(image_len, device), _counts(text_len, device)),
    )
    kwargs.update(
        encoder_hidden_states=local, encoder_hidden_states_mask=None, tpu_layout=layout
    )
    return torch.nn.Module.__call__(self, **kwargs)


def forward(
    self,
    hidden_states,
    encoder_hidden_states,
    timestep,
    encoder_hidden_states_mask=None,
    vis_freqs_cis=None,
    txt_freqs_cis=None,
    tpu_layout: Layout | None = None,
    **kwargs,
):
    """``JoyTransformer3DModel.forward`` over this rank's rows (see :func:`_dit_call`)."""
    if tpu_layout is None:
        return _STOCK_FORWARD(
            self,
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_hidden_states_mask,
            vis_freqs_cis,
            txt_freqs_cis,
            **kwargs,
        )
    batch_size, image_len = hidden_states.shape[:2]
    x = rearrange(hidden_states, "b n c p1 p2 p3 -> (b n) c p1 p2 p3")
    img = rearrange(self.img_in(x), "(b n) d 1 1 1 -> b n d", b=batch_size)
    sp, rows = get_sp_world_size(), tpu_layout.image_rope.shape[0]
    if sp > 1:
        # Zero rows pad the image to the SP multiple, as in the stock forward.
        img = F.pad(img, (0, 0, 0, rows * sp - image_len))
        # A fresh shard: the view's rank-dependent storage offset (contiguous()
        # keeps it at batch 1) is baked into the torch_tpu executable, so each
        # rank would run its own first-block program, whose spmd-safe fused
        # collectives then pair with other ranks' different programs.
        img = img.unflatten(1, (sp, rows))[:, get_sp_parallel_rank()].clone(
            memory_format=torch.contiguous_format
        )
    _, vec, txt, _ = self.condition_embedder(timestep, encoder_hidden_states)
    if vec.shape[-1] > self.hidden_size:
        vec = vec.unflatten(1, (dit._MODULATION_FACTOR, -1))
    for block in self.double_blocks:
        img, txt = block(img, txt, vec, tpu_layout.image_rope, tpu_keys=tpu_layout.keys)
    if sp > 1:
        img = sequence_model_parallel_all_gather(img.contiguous(), dim=1)
        img = img[:, :image_len]
    img, _ = self.proj_out(self.norm_out(img))
    pt, ph, pw = self.patch_size
    return rearrange(
        img,
        "b n (pt ph pw c) -> b n c pt ph pw",
        pt=pt,
        ph=ph,
        pw=pw,
        c=self.out_channels,
    )


def _image_qk_norm_rope(q, k, q_norm, k_norm, table):
    """``_joy_image_qk_rope`` off CUDA, rotating pairs in the pair-swap form."""
    q, k = apply_qk_norm(q, k, q_norm, k_norm, q.shape[-1])
    cos, sin = table.chunk(2, dim=-1)
    return tuple(
        apply_rotary_embedding_pair_swap(t, cos, sin, interleaved=True) for t in (q, k)
    )


def _uses_kernel(impl, q, keys) -> bool:
    if not getattr(impl, "use_splash", False) or q.device.type != "tpu":
        return False
    # Every step takes the kernel or none does: their carries differ.
    return all(
        tpu_splash_attn._ring_kernel_fallback_reason(q, key, key) is None
        for key in keys
    )


def _joint_attention(usp, q, k, v, keys):
    """``[B, S, H, D]`` attention of this rank's rows over both runs' real keys.

    ``q``/``k``/``v`` are (image, prompt) pairs of this rank's rows and
    ``keys`` the int32 ``[0, real keys]`` of each run, padded at its tail.
    """
    rows = [t.shape[1] for t in q]
    q, k, v = (torch.cat(t, 1) for t in (q, k, v))
    sp = get_sp_world_size()
    if sp > 1:
        q, k, v = (_usp_input_all_to_all(t, head_dim=2) for t in (q, k, v))

    # Ulysses gathers [rank 0 rows | rank 1 rows | ...]: a run's rows from
    # every rank form its keys in order, real keys first.
    def runs(t):
        return [
            part.flatten(1, 2).transpose(1, 2)
            for part in t.unflatten(1, (sp, -1)).split(rows, 2)
        ]

    q, k, v = q.transpose(1, 2), runs(k), runs(v)
    impl = usp.attn_impl
    if _uses_kernel(impl, q, k):
        carry = None
        for key, value, real in zip(k, v, keys, strict=True):
            out, carry = impl.forward_ring_resume(q, key, value, real, carry)
    else:
        valid = torch.cat(
            [
                torch.arange(key.shape[2], device=q.device) < real[1]
                for key, real in zip(k, keys, strict=True)
            ]
        )
        # Zero padded keys and values: a masked score still adds q . k, and a
        # zero probability times a NaN value is NaN.
        out = F.scaled_dot_product_attention(
            q,
            torch.where(valid[:, None], torch.cat(k, 2), 0),
            torch.where(valid[:, None], torch.cat(v, 2), 0),
            attn_mask=valid[None, None, None],
            scale=impl.softmax_scale,
        )
    out = out.transpose(1, 2)
    return _usp_output_all_to_all(out, head_dim=2) if sp > 1 else out


def block_forward(
    self,
    img,
    txt,
    vec,
    vis_freqs_cis=None,
    txt_freqs_cis=None,
    vis_complex_freqs=None,
    txt_complex_freqs=None,
    num_replicated_suffix=0,
    tpu_keys=None,
):
    """``MMDoubleStreamBlock.forward`` on bucketed prompt rows (see :func:`forward`)."""
    if tpu_keys is None:
        return _STOCK_BLOCK_FORWARD(
            self,
            img,
            txt,
            vec,
            vis_freqs_cis,
            txt_freqs_cis,
            vis_complex_freqs,
            txt_complex_freqs,
            num_replicated_suffix,
        )
    img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2 = self.img_mod(
        vec
    )
    txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2 = self.txt_mod(
        vec
    )
    heads = self.local_heads_num
    img_qkv, _ = self.img_attn_qkv(
        self.fused_modulate_img_norm1(img, shift=img_shift1, scale=img_scale1)
    )
    img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B L H D", K=3, H=heads)
    img_q, img_k = _image_qk_norm_rope(
        img_q, img_k, self.img_attn_q_norm, self.img_attn_k_norm, vis_freqs_cis
    )
    txt_qkv, _ = self.txt_attn_qkv(
        self.fused_modulate_txt_norm1(txt, shift=txt_shift1, scale=txt_scale1)
    )
    txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B L H D", K=3, H=heads)
    txt_q, txt_k = apply_qk_norm(
        txt_q, txt_k, self.txt_attn_q_norm, self.txt_attn_k_norm, self.head_dim
    )
    attn = _joint_attention(
        self.attn, (img_q, txt_q), (img_k, txt_k), (img_v, txt_v), tpu_keys
    )
    img_attn, txt_attn = attn.flatten(2, 3).split((img.shape[1], txt.shape[1]), 1)
    img = dit.fused_add_gate(img, self.img_attn_proj(img_attn)[0], img_gate1)
    img = dit.fused_add_gate(
        img,
        self.img_mlp(
            self.fused_modulate_img_norm2(img, shift=img_shift2, scale=img_scale2)
        ),
        img_gate2,
    )
    txt = dit.fused_add_gate(txt, self.txt_attn_proj(txt_attn)[0], txt_gate1)
    txt = dit.fused_add_gate(
        txt,
        self.txt_mlp(
            self.fused_modulate_txt_norm2(txt, shift=txt_shift2, scale=txt_scale2)
        ),
        txt_gate2,
    )
    return img, txt


class _PromptPadding:
    """An image processor whose token rows are right-padded to a multiple of 256.

    It keeps the processor's named parameters: the stage passes only the
    keyword arguments ``__call__``'s signature names.
    """

    def __init__(self, processor):
        self._processor = processor

    def __getattr__(self, name):
        return getattr(self._processor, name)

    def __call__(self, images=None, text=None, videos=None, **kwargs):
        inputs = self._processor(images=images, text=text, videos=videos, **kwargs)
        extra = -inputs["input_ids"].shape[1] % PROMPT_TOKEN_ALIGNMENT
        if extra:
            tokenizer = self._processor.tokenizer
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id
            inputs["input_ids"] = F.pad(inputs["input_ids"], (0, extra), value=pad_id)
            inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, extra))
        return inputs


def create_pipeline_stages(self, server_args):
    """The stock stages, the prompt encoder's processor padding its tokens."""
    _STOCK_CREATE_STAGES(self, server_args)
    for stage in self.stages:
        if isinstance(stage, ImageEncodingStage) and not isinstance(
            stage.image_processor, _PromptPadding
        ):
            stage.image_processor = _PromptPadding(stage.image_processor)


def encoding_image_edit(self, outputs, image_inputs, pipeline_config):
    """The stock hook on the encoder's rows up to the last attended token."""
    if isinstance(self.image_processor, _PromptPadding):
        # On the host: nonzero's data-dependent shape would compile on device
        # once per prompt length.
        attended = image_inputs["attention_mask"].cpu().any(0).nonzero()
        rows = int(attended.max()) + 1
        outputs.hidden_states = tuple(h[:, :rows] for h in outputs.hidden_states)
    return _STOCK_ENCODING_IMAGE_EDIT(self, outputs, image_inputs, pipeline_config)


def install() -> None:
    dit.JoyTransformer3DModel.__call__ = _dit_call
    dit.JoyTransformer3DModel.forward = forward
    dit.MMDoubleStreamBlock.forward = block_forward
    # Every rank runs the same block program on its share of the rows.
    dit.JoyTransformer3DModel._compile_regions_spmd_safe = True
    pipeline.JoyImageEditPipeline.create_pipeline_stages = create_pipeline_stages
    ImageEncodingStage.encoding_image_edit = encoding_image_edit
