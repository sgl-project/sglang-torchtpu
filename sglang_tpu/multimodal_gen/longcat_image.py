# SPDX-License-Identifier: Apache-2.0
"""LongCat-Image on TPU workers: prompt rewrite, text encoder and DiT RoPE.

Runtime patch, tracked debt:
- Why: LongCat-Image's default request rewrites the prompt by decoding up to
  512 tokens with its Qwen2.5-VL text encoder, and encodes a padded prompt.
  The decode grows its KV cache every token, and the encoder builds its
  positions (an int64 cumsum) and its causal mask (an int64 reduction) on the
  device, which torch_tpu does not lower ("Unsupported conversion from
  vmreg/vreg to U64"): every server died in its warmup request. The DiT
  builds its RoPE tables in float64 on the device at every step.
- How: :func:`install`
  (1) runs ``Qwen2_5_VLForConditionalGeneration.generate`` on TPU with a
      static KV cache: the only compiled shapes are a 256-token prompt chunk
      and one token. Tokens, cache slots, rope positions and visibility go
      in as int32. Each prompt row decodes its real tokens (by the mask) in
      its own cache, in lockstep, at the stock rope positions; a padded last
      column gives the row's first logits, as in stock. Each step's logits
      rows are selected on the host by the stock ``_select_next_token`` over
      the stock rows (padded prompt, then the generated tokens): greedy with
      the repetition penalty for LongCat's generation config;
  (2) for encoders that honor the padding mask (LongCat's), computes the
      rope index and the causal and padding mask on the host, from the same
      values;
  (3) builds the DiT's RoPE tables on the host in float64 (as on CUDA), once
      per image size, outside the compiled DiT;
  (4) gives the compiled DiT its prompt embeddings in their own contiguous
      storage: the backend guards every input's storage offset, and the
      embeddings are a slice of the encoder output on a conditioning-cache
      miss but a copy on a hit, which would compile the DiT once per layout.
- Upstream PR or issue: none.
- Remove when: torch_tpu lowers int64 reductions and float64, and the
  rewrite runs on a TPU decode engine.
"""

from __future__ import annotations

import weakref

import torch

from sglang.multimodal_gen.runtime.models.dits import longcat_image as dit
from sglang.multimodal_gen.runtime.models.encoders import qwen2_5vl
from sglang.multimodal_gen.runtime.models.encoders.qwen_vl_rope import (
    apply_qwen_vl_text_rope,
)
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    build_torch_compile_kwargs,
)
from sglang.srt.layers.quantization import unquant

CHUNK = 256
_CAPACITY_ALIGNMENT = 1024
# Devices that take these paths.
_DEVICE_TYPES = ("tpu",)

_STOCK_GENERATE = qwen2_5vl.Qwen2_5_VLForConditionalGeneration.generate
_STOCK_ROPE_INDEX = qwen2_5vl.Qwen2_5_VLModel.get_rope_index
_STOCK_CAUSAL_MASK = qwen2_5vl.create_causal_mask
_STOCK_DIT_CALL = dit.LongCatImageTransformer2DModel.__call__


def _attend(q, keys, values, visible, scale):
    """``[1, H, n, D]`` queries over cache rows ``[C, KVH, D]``: ``[1, n, H * D]``."""
    _, heads, n, dim = q.shape
    q = q[0].unflatten(0, (keys.shape[1], -1))
    scores = (q.float() @ keys.permute(1, 2, 0)[:, None].float()) * scale
    probs = scores.masked_fill(~visible, float("-inf")).softmax(-1)
    out = probs.to(values.dtype) @ values.transpose(0, 1)[:, None]
    return out.flatten(0, 1).transpose(0, 1).reshape(1, n, heads * dim)


class _Decoder:
    """Static-KV-cache decoding of text-only Qwen2.5-VL prompts, one cache per row."""

    def __init__(self, lm):
        self.lm = lm
        self.text = lm.model.language_model
        self.capacity = 0
        self.caches = []
        # The encoder's linear layers pick their bf16 GEMM backend on first
        # use; pick it before tracing, which cannot set it inside the graph.
        unquant.get_bf16_gemm_backend()
        kwargs = build_torch_compile_kwargs(mode=None)
        kwargs.update(fullgraph=True, dynamic=False)
        self.forward = torch.compile(self._forward, **kwargs)
        self.last_row = torch.zeros(1, dtype=torch.int32).to(self._device)

    @property
    def _device(self):
        return self.lm.lm_head.weight.device

    def reserve(self, rows: int, capacity: int) -> None:
        """Holds ``rows`` caches (keys, values) of at least ``capacity`` rows."""
        if capacity > self.capacity:
            self.caches, self.capacity = [], capacity
        attn = self.text.layers[0].self_attn
        shape = (self.capacity, attn.num_key_value_heads, attn.head_dim)
        weight = self.lm.lm_head.weight
        while len(self.caches) < rows:
            self.caches.append(
                tuple([weight.new_zeros(shape) for _ in self.text.layers] for _ in "kv")
            )

    def _forward(self, entries, last, cache_keys, cache_values):
        """Runs ``entries``: tokens, cache rows, rope positions, last rows seen.

        Writes each token's keys and values at its cache row, lets each query
        see the cache rows up to its last row seen, and returns the float32
        logits of entry ``last``. Rows past a prompt (a chunk's padded tail)
        are never seen before later steps overwrite them.
        """
        tokens, slots, limits = entries[0], entries[1], entries[3]
        h = self.text.embed_tokens(tokens)[None]
        positions = entries[2].long().view(1, 1, -1).expand(3, 1, -1)
        visible = (
            torch.arange(cache_keys[0].shape[0], dtype=torch.int32, device=slots.device)
            <= limits[:, None]
        )
        index = slots.long()
        for layer, keys, values in zip(self.text.layers, cache_keys, cache_values):
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            q, k, v = (
                qwen2_5vl._linear_output(proj, x)
                .unflatten(-1, (-1, attn.head_dim))
                .transpose(1, 2)
                for proj in (attn.q_proj, attn.k_proj, attn.v_proj)
            )
            q, k = apply_qwen_vl_text_rope(attn.rotary_emb, positions, q, k)
            keys.index_copy_(0, index, k[0].transpose(0, 1))
            values.index_copy_(0, index, v[0].transpose(0, 1))
            out = _attend(q, keys, values, visible, attn.scaling)
            h = h + qwen2_5vl._linear_output(attn.o_proj, out)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        h = self.text.norm(h[0].index_select(0, last))
        return self.lm.lm_head(h).float()

    @staticmethod
    def _entries(row, keep, positions):
        """A row's entries, and its first decoded token's cache row and position.

        The real tokens fill the cache from row 0, at their stock positions
        0..n-1. A padded last column is the stock query of the first token:
        the padding token at its stock position, seeing the real tokens. It is
        written at the first decoded token's row, which that token overwrites.
        """
        n = int(keep.sum())
        real = torch.arange(n)
        entries = torch.stack((row[keep], real, real, real))
        if not keep[-1]:
            query = torch.tensor([[int(row[-1])], [n], [int(positions[-1])], [n - 1]])
            entries = torch.cat((entries, query), 1)
        return entries.to(torch.int32), n, int(positions.max()) + 1

    def _prefill(self, entries, cache):
        """Writes one row's entries into ``cache``; returns the last logits."""
        m = entries.shape[1]
        padded = -(-m // CHUNK) * CHUNK
        tail = torch.arange(m, padded, dtype=torch.int32)
        tail = torch.stack((torch.zeros_like(tail), tail, tail, tail))
        entries = torch.cat((entries, tail), 1)
        for start in range(0, padded, CHUNK):
            chunk = entries[:, start : start + CHUNK].contiguous()
            last = torch.tensor([min(m - 1 - start, CHUNK - 1)], dtype=torch.int32)
            logits = self.forward(chunk.to(self._device), last.to(self._device), *cache)
        return logits

    def _step(self, token, slot, position, cache):
        """Writes one token at cache row ``slot``; returns its logits."""
        step = torch.tensor([[token], [slot], [position], [slot]], dtype=torch.int32)
        return self.forward(step.to(self._device), self.last_row, *cache)

    def generate(self, lm, input_ids, attention_mask, max_new_tokens, select):
        ids = input_ids.cpu()
        mask = (
            torch.ones_like(ids, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.cpu().bool()
        )
        if not mask.any(-1).all():
            raise ValueError("TPU generation needs a real token in every prompt row")
        # The stock text rope positions: padding at 1, decoding after the largest.
        positions = (mask.long().cumsum(-1) - 1).masked_fill_(~mask, 1)
        rows = [self._entries(*row) for row in zip(ids, mask, positions)]
        prompts, cache_starts, rope_starts = zip(*rows)
        longest = max(prompt.shape[1] for prompt in prompts)
        capacity = max(-(-longest // CHUNK) * CHUNK, max(cache_starts) + max_new_tokens)
        self.reserve(
            len(prompts), -(-capacity // _CAPACITY_ALIGNMENT) * _CAPACITY_ALIGNMENT
        )
        logits = [self._prefill(p, c) for p, c in zip(prompts, self.caches)]
        logits = torch.cat(logits).cpu()
        unfinished = torch.ones(len(prompts), dtype=torch.bool)
        for step in range(max_new_tokens):
            # As stock: the history is the padded prompt and the generated
            # tokens, and finished rows take the pad token.
            tokens = torch.where(unfinished, select(logits, ids), select.pad)
            ids = torch.cat((ids, tokens[:, None].to(ids.dtype)), -1)
            unfinished &= ~select.stops(tokens)
            if step == max_new_tokens - 1 or not unfinished.any():
                break
            # Every unfinished row decodes in its own cache; one host sync.
            live = unfinished.nonzero().flatten().tolist()
            out = [
                self._step(
                    int(tokens[r]),
                    cache_starts[r] + step,
                    rope_starts[r] + step,
                    self.caches[r],
                )
                for r in live
            ]
            logits[live] = torch.cat(out).cpu()
        return ids.to(input_ids.device)


class _Selector:
    """The stock token selection, stop check and pad token, on the host."""

    def __init__(self, lm, kwargs):
        config = lm.generation_config

        def pick(name, default):
            value = kwargs.get(name)
            return config.get(name, default) if value is None else value

        self.options = dict(
            do_sample=pick("do_sample", False),
            temperature=pick("temperature", 1.0),
            top_k=pick("top_k", 0),
            top_p=pick("top_p", 1.0),
            repetition_penalty=pick("repetition_penalty", 1.0),
        )
        eos = pick("eos_token_id", lm.config.eos_token_id)
        eos = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
        self.eos = torch.tensor(eos, dtype=torch.long)
        self.pad = pick("pad_token_id", lm.config.pad_token_id)
        if self.pad is None:
            raise ValueError("pad_token_id must be set for generation")

    def __call__(self, logits, ids):
        return qwen2_5vl._select_next_token(logits, ids, **self.options)

    def stops(self, tokens):
        return torch.isin(tokens, self.eos)


_HOST_ARGS = ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw")


def _generate(self, input_ids, attention_mask=None, *, max_new_tokens, **kwargs):
    if (
        self.lm_head.weight.device.type not in _DEVICE_TYPES
        or any(kwargs.get(name) is not None for name in _HOST_ARGS)
        or max_new_tokens <= 0
    ):
        return _STOCK_GENERATE(
            self, input_ids, attention_mask, max_new_tokens=max_new_tokens, **kwargs
        )
    decoder = self.__dict__.get("_tpu_decoder")
    if decoder is None:
        decoder = self.__dict__["_tpu_decoder"] = _Decoder(self)
    with torch.no_grad():
        return decoder.generate(
            self, input_ids, attention_mask, max_new_tokens, _Selector(self, kwargs)
        )


def _honors_padding(model) -> bool:
    return getattr(model.language_model, "honor_cache_free_padding_mask", False)


def get_rope_index(self, input_ids=None, *args, attention_mask=None, **kwargs):
    """``get_rope_index`` on the host, for encoders that honor the padding mask."""
    ref = input_ids if input_ids is not None else attention_mask
    if ref is None or ref.device.type not in _DEVICE_TYPES or not _honors_padding(self):
        return _STOCK_ROPE_INDEX(
            self, input_ids, *args, attention_mask=attention_mask, **kwargs
        )

    def host(t):
        return t.cpu() if isinstance(t, torch.Tensor) else t

    positions, deltas = _STOCK_ROPE_INDEX(
        self,
        host(input_ids),
        *map(host, args),
        attention_mask=host(attention_mask),
        **{k: host(v) for k, v in kwargs.items()},
    )
    return positions.to(ref.device), deltas.to(ref.device)


def causal_padding_mask(attention_mask, length):
    """``[B, 1, L, L]`` bool, true where a query sees a key (no cache)."""
    causal = torch.ones(length, length, dtype=torch.bool).tril()
    return causal & attention_mask.bool()[:, None, None, :length]


def _causal_mask_on_host(**kwargs):
    embeds = kwargs["inputs_embeds"]
    mask, cache = kwargs.get("attention_mask"), kwargs.get("past_key_values")
    if (
        embeds.device.type not in _DEVICE_TYPES
        or not getattr(kwargs.get("config"), "honor_cache_free_padding_mask", False)
        or (cache is not None and cache.get_seq_length() > 0)
        or mask is None
        or mask.dim() != 2
    ):
        return _STOCK_CAUSAL_MASK(**kwargs)
    return causal_padding_mask(mask.cpu(), embeds.shape[1]).to(embeds.device)


# One request's RoPE tables, keyed by its latent ids (the same tensor at every
# step) and the text length: the text ids depend only on that length.
_ROPE_CACHE: dict[tuple, tuple] = {}


def _rope_tables(model, txt_ids, img_ids, device):
    key = (id(model), id(img_ids), tuple(txt_ids.shape))
    hit = _ROPE_CACHE.get(key)
    if hit is not None and hit[0]() is img_ids:
        return hit[2]
    ids = torch.cat((txt_ids, img_ids), dim=0).cpu()
    for stale in [k for k, (ref, *_) in _ROPE_CACHE.items() if ref() is None]:
        del _ROPE_CACHE[stale]
    # Equal ids (each CFG branch's copy of one SP shard) share their tables.
    tables = next(
        (
            shared
            for k, (_, host, shared) in _ROPE_CACHE.items()
            if k[0] == key[0] and host.equal(ids)
        ),
        None,
    )
    if tables is None:
        # float64 on the host, as the stock computes it on CUDA.
        cos, sin = dit._LongCatPosEmbed.forward(model.pos_embed, ids)
        tables = (cos.to(device), sin.to(device))
    _ROPE_CACHE[key] = (weakref.ref(img_ids), ids, tables)
    return tables


def _dit_call(self, *args, **kwargs):
    """Prepares the compiled DiT's inputs: own-storage embeddings, host RoPE."""
    latents = kwargs.get("hidden_states")
    if (
        args
        or not isinstance(latents, torch.Tensor)
        or latents.device.type not in _DEVICE_TYPES
    ):
        return _STOCK_DIT_CALL(self, *args, **kwargs)
    embeds = kwargs.get("encoder_hidden_states")
    if isinstance(embeds, torch.Tensor):
        # Offset 0 and contiguous strides on a cache hit and miss alike.
        kwargs["encoder_hidden_states"] = embeds.clone(
            memory_format=torch.contiguous_format
        )
    txt_ids, img_ids = kwargs.get("txt_ids"), kwargs.get("img_ids")
    if (
        txt_ids is not None
        and img_ids is not None
        and kwargs.get("image_rotary_emb") is None
    ):
        kwargs["image_rotary_emb"] = _rope_tables(
            self, txt_ids, img_ids, latents.device
        )
    return _STOCK_DIT_CALL(self, **kwargs)


def install() -> None:
    qwen2_5vl.Qwen2_5_VLForConditionalGeneration.generate = _generate
    qwen2_5vl.Qwen2_5_VLModel.get_rope_index = get_rope_index
    qwen2_5vl.create_causal_mask = _causal_mask_on_host
    dit.LongCatImageTransformer2DModel.__call__ = _dit_call
