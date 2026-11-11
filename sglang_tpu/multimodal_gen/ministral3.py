# SPDX-License-Identifier: Apache-2.0
"""Ministral 3 on TPU workers: ERNIE-Image's prompt enhancer and text encoder.

Runtime patch, tracked debt:
- Why: Transformers' ``generate`` and its causal-mask helper compute on int64
  tensors on the model's device (the mask helper sums the attention mask), and
  torch_tpu aborts lowering them ("Unsupported conversion from vmreg/vreg to
  U64"). ERNIE-Image's default request generates its prompt this way, and a
  padded prompt reaches the mask helper in the text encoder. Eager decoding
  would also compile every op again at each new cache length.
- How: :func:`install`
  (1) runs ``Ministral3ForCausalLM.generate`` on TPU with a static KV cache.
      The only compiled shapes are a 256-token prompt chunk and one token.
      Tokens, positions and cache slots go in as int32, and RoPE and the query
      scale come from host-built tables. Each step returns the 64 largest
      logits (``tpu_top_k``); Transformers' own logits processors and stopping
      criteria run on the host. Drawing from those candidates is exact when
      the processors are temperature, top-k (k < 64), top-p and min-p and no
      logit tied with the k-th lies past the 64th; any other step draws from
      the full row;
  (2) seeds the draws from the request's seed, so every SP rank, each running
      the enhancer, takes the same tokens;
  (3) builds the encoder's causal and padding mask on the host.
- Upstream PR or issue: none.
- Remove when: torch_tpu lowers int64 reductions and the enhancer runs on a
  TPU decode engine.
"""

from __future__ import annotations

import contextvars
import copy

import torch
import torch.nn.functional as F
from transformers.generation import logits_process
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList

from sglang.multimodal_gen.runtime.models.encoders import mistral_3
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    ernie_image_pe,
)
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    build_torch_compile_kwargs,
)

CHUNK = 256
CANDIDATES = 64
_CAPACITY_ALIGNMENT = 1024

_STOCK_GENERATE = mistral_3.Ministral3ForCausalLM.generate
_STOCK_UNMASKED = mistral_3._can_use_unmasked_causal_attention
_STOCK_CAUSAL_MASK = mistral_3.create_causal_mask
_STOCK_PE_FORWARD = ernie_image_pe.PromptEnhancementStage.forward
_SEED = contextvars.ContextVar("sglang_tpu_prompt_enhancer_seed", default=None)
# Devices that take these paths.
_DEVICE_TYPES = ("tpu",)


def candidate_top_k(processors) -> int | None:
    """k if drawing from the top 64 logits can equal drawing from all of them.

    Temperature keeps the order, and after top-k the other warpers see only
    the kept logits.
    """
    kinds = [type(p) for p in processors]
    if kinds.count(logits_process.TopKLogitsWarper) != 1:
        return None
    i = kinds.index(logits_process.TopKLogitsWarper)
    after = (logits_process.TopPLogitsWarper, logits_process.MinPLogitsWarper)
    if any(t is not logits_process.TemperatureLogitsWarper for t in kinds[:i]) or any(
        t not in after for t in kinds[i + 1 :]
    ):
        return None
    k = processors[i].top_k
    return k if k < CANDIDATES else None


def next_token(processors, ids, logits, top, k, do_sample, generator):
    """The next token, as Transformers' sampling loop draws it, ``[1]``.

    ``top`` holds the 64 largest logits and their ids, in descending order.
    """
    if k is not None:
        values, candidates = (t.cpu() for t in top)
        # Exact unless a logit tied with the k-th lies past the candidates.
        if values[0, -1] < values[0, k - 1]:
            probs = processors(ids, values).softmax(-1)
            pick = torch.multinomial(probs[0], 1, generator=generator)
            return candidates[0, pick].long()
    scores = processors(ids, logits.cpu())
    if do_sample:
        return torch.multinomial(scores.softmax(-1)[0], 1, generator=generator)
    return scores.argmax(-1)


def _attend(q, keys, values, visible, scale):
    """``[1, H, n, D]`` queries over cache rows ``[C, KVH, D]``: ``[1, n, H * D]``."""
    _, heads, n, dim = q.shape
    q = q[0].unflatten(0, (keys.shape[1], -1))
    scores = (q.float() @ keys.permute(1, 2, 0)[:, None].float()) * scale
    probs = scores.masked_fill(~visible, float("-inf")).softmax(-1)
    out = probs.to(values.dtype) @ values.transpose(0, 1)[:, None]
    return out.flatten(0, 1).transpose(0, 1).reshape(1, n, heads * dim)


class _Decoder:
    """Static-KV-cache decoding of one TPU-resident ``Ministral3ForCausalLM``."""

    def __init__(self, lm):
        self.lm = lm
        self.capacity = 0
        weight = lm.lm_head.weight
        if weight.device.type == "tpu":
            from sglang_tpu.top_k import register_tpu_top_k_ops

            register_tpu_top_k_ops()
        kwargs = build_torch_compile_kwargs(mode=None)
        kwargs.update(fullgraph=True, dynamic=False)
        self.forward = torch.compile(self._forward, **kwargs)
        self.last_row = torch.zeros(1, dtype=torch.int32).to(weight.device)

    def reserve(self, capacity: int) -> None:
        """Cache rows, RoPE and query-scale tables for ``capacity`` positions."""
        if capacity <= self.capacity:
            return
        model, weight = self.lm.model, self.lm.lm_head.weight
        attn = model.layers[0].self_attn
        positions = torch.arange(capacity)[None]
        # The stock tables, computed on the host in float32.
        rotary = copy.deepcopy(model.rotary_emb).to("cpu")
        cos, sin = rotary(torch.empty(0, dtype=weight.dtype), positions)
        self.cos, self.sin = cos[0].to(weight.device), sin[0].to(weight.device)
        scale = mistral_3._get_llama_4_attn_scale(
            positions,
            attn.llama_4_scaling_beta,
            attn.original_max_position_embeddings,
        )
        if scale is not None:
            scale = scale.flatten().to(weight.dtype).to(weight.device)
        self.scale = scale
        shape = (capacity, attn.num_key_value_heads, attn.head_dim)
        self.keys = [weight.new_zeros(shape) for _ in model.layers]
        self.values = [weight.new_zeros(shape) for _ in model.layers]
        self.capacity = capacity

    def _forward(self, tokens_slots, last):
        """Writes tokens ``tokens_slots[0]`` at cache rows ``tokens_slots[1]``.

        Returns the float32 logits of row ``last`` and their top 64. A query
        sees the cache rows up to its own, so a chunk's padded tail never
        reaches its real rows, and later steps overwrite it.
        """
        model = self.lm.model
        tokens, slots = tokens_slots[0], tokens_slots[1]
        h = model.embed_tokens(tokens)[None]
        cos, sin = (t.index_select(0, slots)[None] for t in (self.cos, self.sin))
        visible = (
            torch.arange(self.capacity, dtype=torch.int32, device=slots.device)
            <= slots[:, None]
        )
        index = slots.long()
        for layer, keys, values in zip(model.layers, self.keys, self.values):
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            q, k, v = (
                mistral_3._linear_output(proj, x)
                .unflatten(-1, (-1, attn.head_dim))
                .transpose(1, 2)
                for proj in (attn.q_proj, attn.k_proj, attn.v_proj)
            )
            q, k = mistral_3.apply_rotary_pos_emb(q, k, cos, sin)
            if self.scale is not None:
                q = q * self.scale.index_select(0, slots)[:, None]
            keys.index_copy_(0, index, k[0].transpose(0, 1))
            values.index_copy_(0, index, v[0].transpose(0, 1))
            out = _attend(q, keys, values, visible, attn.scaling)
            h = h + mistral_3._linear_output(attn.o_proj, out)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        logits = self.lm.lm_head(model.norm(h[0].index_select(0, last))).float()
        if logits.device.type == "tpu":
            from sglang_tpu.top_k import tpu_top_k

            return logits, *tpu_top_k(logits, CANDIDATES)
        return logits, *torch.topk(logits, CANDIDATES)

    def generate(self, input_ids, attention_mask=None, **kwargs):
        lm = self.lm
        ids = input_ids.cpu()
        if ids.shape[0] != 1 or (
            attention_mask is not None and not bool(attention_mask.cpu().all())
        ):
            raise NotImplementedError("TPU generation takes one unpadded prompt")
        # Transformers' own setup, as `generate` runs it.
        has_default_max_length = (
            kwargs.get("max_length") is None and lm.generation_config.max_length is None
        )
        has_default_min_length = (
            kwargs.get("min_length") is None and lm.generation_config.min_length is None
        )
        config, model_kwargs = lm._prepare_generation_config(None, **kwargs)
        if model_kwargs or config.num_beams != 1 or config.num_return_sequences != 1:
            raise NotImplementedError(
                f"TPU generation draws one sequence, without {sorted(model_kwargs)}"
            )
        n = ids.shape[1]
        lm._prepare_special_tokens(config, True, device="cpu")
        config = lm._prepare_generated_length(
            config, has_default_max_length, has_default_min_length, "input_ids", n, ids
        )
        processors = lm._get_logits_processor(
            generation_config=config,
            input_ids_seq_length=n,
            encoder_input_ids=ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=LogitsProcessorList(),
            device="cpu",
            model_kwargs=model_kwargs,
        )
        stopping = lm._get_stopping_criteria(config, StoppingCriteriaList())
        k = candidate_top_k(processors) if config.do_sample else None
        seed = _SEED.get()
        generator = None if seed is None else torch.Generator().manual_seed(seed)

        device = lm.lm_head.weight.device
        padded = -(-n // CHUNK) * CHUNK
        capacity = max(padded, config.max_length)
        self.reserve(-(-capacity // _CAPACITY_ALIGNMENT) * _CAPACITY_ALIGNMENT)
        pad = 0 if config.pad_token_id is None else int(config.pad_token_id)
        tokens = F.pad(ids[0], (0, padded - n), value=pad).to(torch.int32)
        prompt = torch.stack((tokens, torch.arange(padded, dtype=torch.int32)))
        for start in range(0, padded, CHUNK):
            chunk = prompt[:, start : start + CHUNK].contiguous()
            last = torch.tensor([min(n - 1 - start, CHUNK - 1)], dtype=torch.int32)
            out = self.forward(chunk.to(device), last.to(device))
        while True:
            token = next_token(
                processors, ids, out[0], out[1:], k, config.do_sample, generator
            )
            ids = torch.cat((ids, token.view(1, 1)), -1)
            if stopping(ids, None).all():
                return ids
            step = torch.tensor([[int(token)], [ids.shape[1] - 1]], dtype=torch.int32)
            out = self.forward(step.to(device), self.last_row)


def _generate(self, *args, **kwargs):
    if args or self.lm_head.weight.device.type not in _DEVICE_TYPES:
        return _STOCK_GENERATE(self, *args, **kwargs)
    decoder = self.__dict__.get("_tpu_decoder")
    if decoder is None:
        decoder = self.__dict__["_tpu_decoder"] = _Decoder(self)
    with torch.no_grad():
        return decoder.generate(**kwargs)


def _seeded_forward(self, batch, server_args):
    seed = getattr(batch, "seed", None)
    token = _SEED.set(seed[0] if isinstance(seed, list) else seed)
    try:
        return _STOCK_PE_FORWARD(self, batch, server_args)
    finally:
        _SEED.reset(token)


def causal_padding_mask(attention_mask, length):
    """Transformers' SDPA mask for an empty cache, ``[B, 1, L, L]``, true where seen."""
    causal = torch.ones(length, length, dtype=torch.bool).tril()
    return causal & attention_mask.bool()[:, None, None, :length]


def _unmasked_on_host(attention_mask, config, past_key_values):
    if attention_mask is not None:
        attention_mask = attention_mask.cpu()
    return _STOCK_UNMASKED(attention_mask, config, past_key_values)


def _causal_mask_on_host(**kwargs):
    embeds = kwargs[mistral_3._CREATE_CAUSAL_MASK_ARG]
    mask, cache = kwargs.get("attention_mask"), kwargs.get("past_key_values")
    if (
        embeds.device.type not in _DEVICE_TYPES
        or (cache is not None and cache.get_seq_length() > 0)
        or mask is None
        or mask.dim() != 2
    ):
        return _STOCK_CAUSAL_MASK(**kwargs)
    return causal_padding_mask(mask.cpu(), embeds.shape[1]).to(embeds.device)


def install() -> None:
    mistral_3._can_use_unmasked_causal_attention = _unmasked_on_host
    mistral_3.create_causal_mask = _causal_mask_on_host
    mistral_3.Ministral3ForCausalLM.generate = _generate
    ernie_image_pe.PromptEnhancementStage.forward = _seeded_forward
