# SPDX-License-Identifier: Apache-2.0
"""CP-aware Pallas RPA v3 backend (PCP streaming path).

Selected via ``--attention-backend pallas_tpu_rpa_v3_cp``. Under CP-v2 this
backend routes attention through the PCP streaming kernel
(``layers/attention/pallas/pcp_streaming_rpa``): a short fixed sequence of
Pallas calls per layer (current-chunk causal pass, fused KV writeback,
history pass, output finalize) that streams KV pages around the CP ring
in-kernel via ``pltpu.make_async_remote_copy`` while running
online-softmax on each rank's locally-owned compact Q rows. The KV cache
update is fused into that ring — there is no separate scatter before
attention.

Non-CP forwards on this backend fall through to the base v3 kernel path
byte-for-byte (via ``super()._forward``). Batched RPA
(``pallas_tpu_rpa_batched``) is a separate backend for the non-CP fast path.

The PCP driver + rank-major compact Q layout planner live under
``layers/cp/pcp_streaming.py`` and ``layers/cp/pcp_sequence_layout.py``;
this file is the backend subclass shell that dispatches ``_forward_pcp``
when CP-v2 metadata is present on the forward batch.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here (see
# attention_pallas_rpa_base for why torch_tpu's jax_op needs real annotations).

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_v3 import (
    TpuPallasRpaV3AttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

__all__ = [
    "TpuPallasRpaV3CpAttnBackend",
]


class TpuPallasRpaV3CpAttnBackend(TpuPallasRpaV3AttnBackend):
    """RPA v3 CP-aware backend (``--attention-backend pallas_tpu_rpa_v3_cp``).

    Inherits the v3 kernel spec, custom-op registration, metadata builders,
    and non-CP ``_forward`` from :class:`TpuPallasRpaV3AttnBackend`. Adds:

    * cached ``_is_pcp_streaming`` constant so per-layer dispatch reads a
      module attribute (dynamo-safe) instead of iterating
      ``extend_seq_lens_cpu`` via ``is_pcp_streaming_active`` per layer,
    * ``_forward_pcp`` sibling to base's ``_forward`` that invokes the
      PCP streaming kernel via the driver in
      ``layers/cp/pcp_streaming.py``,
    * ``forward_extend`` / ``forward_decode`` dispatch on ``_is_pcp_streaming``
      + plain-bool ``forward_batch.pcp_attn_active`` (populated by
      ``prepare_cp_forward``).
    """

    def __init__(
        self, model_runner: "ModelRunner", speculative_step_id: int = 0
    ) -> None:
        super().__init__(model_runner, speculative_step_id=speculative_step_id)

        # CP strategy kind cached at __init__ so the compiled forward gates on
        # a constant module attribute instead of calling
        # ``is_pcp_streaming_active`` per layer (that would iterate
        # ``extend_seq_lens_cpu`` inside the compiled region and specialize
        # dynamo on its content).
        from sglang.srt.layers.cp.utils import enable_cp_v2

        self._cp_v2_enabled = enable_cp_v2()
        self._is_pcp_streaming = self._cp_v2_enabled and _cp_strategy_is_pcp()

        # Per-layer PCP custom_op, keyed by layer_id. Stored on this backend
        # (a plain-Python AttentionBackend, NOT an nn.Module) so the compiled
        # forward reads through the same Dynamo path base RPA v3 uses for
        # its ``self._op``. Storing the op on ``RadixAttention`` (nn.Module)
        # routes the read through NNModuleVariable and can silently graph-
        # break around the CustomOpDef call, leaving the KV write in an
        # undefined aliasing state (observed as ~50% wrong first-token output
        # at ISL=4096 under torch.compile).
        self._pcp_op_by_layer_id: dict[int, Any] = {}

        if self._is_pcp_streaming:
            _reject_linear_attention_under_pcp(model_runner)
            _warm_pcp_streaming_op_registry(
                model_runner=model_runner,
                kv_cache_quant_dtype=self.kv_cache_quant_dtype,
                op_by_layer_id=self._pcp_op_by_layer_id,
            )

    def _forward_pcp(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """PCP streaming attention for one layer.

        Fires when CP-v2 is active AND the strategy is PCP streaming. The
        kernel handles both prefill and decode via the standard
        ``distribution=[decode_end, prefill_end, mixed_end]`` metadata;
        chunked prefill's history is streamed page-by-page around the CP
        ring in-kernel. Debug hooks (stub-attn, per-layer allow-list,
        force-serial, layer trace, input dumps) live at the
        ``pcp_streaming_attention`` driver so this hot path stays clean.
        """
        from sglang.srt.layers.cp.pcp_streaming import pcp_streaming_attention

        return pcp_streaming_attention(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            token_to_kv_pool=self.token_to_kv_pool,
            forward_metadata=self.forward_metadata,
            kv_cache_quant_dtype=self.kv_cache_quant_dtype,
            pcp_op=self._pcp_op_by_layer_id[layer.layer_id],
            sinks=sinks,
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        # Gate on cached constant + plain-bool ``pcp_attn_active`` (set by
        # ``prepare_cp_forward``). Reading ``attn_cp_metadata is not None``
        # here is a msgspec.Struct attribute access inside the compiled
        # region, which per memory/pcp_compile_msgspec.md produces
        # non-deterministic wrong output.
        sinks = kwargs.get("sinks")
        if self._is_pcp_streaming and forward_batch.pcp_attn_active:
            return self._forward_pcp(q, k, v, layer, forward_batch, sinks=sinks)
        return self._forward(q, k, v, layer, forward_batch, sinks=sinks)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        # During torch.compile, rotary_emb may produce a 3D query; flatten
        # so the view matches the kernel's layout. Mirrors base's
        # ``forward_decode``.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        sinks = kwargs.get("sinks")
        if self._is_pcp_streaming and forward_batch.pcp_attn_active:
            return self._forward_pcp(q, k, v, layer, forward_batch, sinks=sinks)
        return self._forward(q, k, v, layer, forward_batch, sinks=sinks)


def _cp_strategy_is_pcp() -> bool:
    """Return True when the configured CP strategy is PCP streaming."""
    from sglang.srt.layers.cp.base import is_pcp

    return is_pcp()


def _reject_linear_attention_under_pcp(model_runner: "ModelRunner") -> None:
    """Refuse hybrid linear-attention models until GDN gets a CP path.

    CP-v2 shards the token axis at the model boundary, so every layer sees
    only this rank's interleaved slice of the batch. Full attention is fine
    with that -- the ring brings it the rest of the KV. A linear-attention
    (Gated DeltaNet / Mamba) layer is not: it integrates a recurrence *along*
    the token axis, so a disjoint 1/cp_size shard yields a state that was
    never conditioned on the tokens in between. On Qwen3.5 that is 45 of 60
    layers, and the failure mode is plausible-looking wrong output, not a
    crash -- nothing downstream would flag it.

    Making these layers work needs a token-shard to head-shard exchange around
    the recurrence, so that each rank takes all tokens for 1/cp_size of the
    heads. Until that exists, refuse.
    """
    from sglang.srt.configs.hybrid_arch import mambaish_config

    if mambaish_config(model_runner.model_config) is None:
        return
    raise NotImplementedError(
        "PCP streaming (--cp-strategy pcp) does not support hybrid "
        "linear-attention models. Context parallelism shards the token axis, "
        "but a linear-attention layer's recurrence runs along that axis, so "
        "each rank would integrate over a disjoint token shard and silently "
        "produce wrong output. Drop --cp-strategy pcp for this model."
    )


def _ensure_language_model(model_runner: "ModelRunner") -> Any:
    """Resolve and return the language backbone. One-shot mirror of what
    ``ModelRunner`` does on the CUDA-graph path (disabled on TPU) so the
    subsequent layer walk sees the resolved backbone.

    Idempotent: if ``model_runner.model.model`` was already set (by an
    earlier init path) to the same resolved backbone, the assignment is a
    no-op. If it was set to a DIFFERENT object (e.g. multi-modal wrappers
    like Qwen3OmniMoeForConditionalGeneration where the outer model
    exposes a distinct ``.model``), preserve it and just return the
    resolved backbone.
    """
    from sglang.srt.model_loader.utils import resolve_language_model

    resolved = resolve_language_model(model_runner.model)
    existing = getattr(model_runner.model, "model", None)
    if existing is None or existing is resolved:
        model_runner.model.model = resolved
    return resolved


def _iter_attention_layers(language_model: Any):
    """Yield each ``RadixAttention`` on ``language_model`` in decoder order.

    ``language_model`` is the resolved backbone (see
    ``_ensure_language_model``). Walks ``.model`` chains until a
    ``.layers`` iterable is found; kept narrow to the shapes Qwen3-Coder
    presents.
    """
    from sglang.srt.layers.radix_attention import RadixAttention

    layer_model = language_model
    while not hasattr(layer_model, "layers") and hasattr(layer_model, "model"):
        layer_model = layer_model.model
    if not hasattr(layer_model, "layers"):
        return
    # Search each decoder layer rather than assuming ``layer.self_attn.attn``:
    # hybrid models alternate layer types and hold their attention at
    # different depths (Qwen3.5's full-attention layer owns ``.attn``
    # directly), and linear-attention layers hold a different class entirely,
    # so they drop out of this filter on their own.
    for layer in layer_model.layers:
        for module in layer.modules():
            if isinstance(module, RadixAttention):
                yield module


def _warm_pcp_streaming_op_registry(
    *,
    model_runner: "ModelRunner",
    kv_cache_quant_dtype: Optional[torch.dtype],
    op_by_layer_id: dict[int, Any],
) -> None:
    """Pre-build the PCP streaming custom op for every attention layer and
    stash it in ``op_by_layer_id`` (a plain dict owned by the attention
    backend, NOT an nn.Module attribute — see comment in
    :meth:`TpuPallasRpaV3CpAttnBackend.__init__` for why).

    Building the op calls ``jax.jit`` and ``inspect.signature`` inside
    ``build_pcp_streaming_callable`` — neither is Dynamo-traceable. Warming
    here (from a plain eager frame after the distributed process group is up
    and the model weights / FP8 scales are loaded) means the compiled forward
    can dispatch via ``self._pcp_op_by_layer_id[layer.layer_id]`` — Dynamo
    lowers this to a direct CustomOpDef call, matching base RPA v3's
    ``self._op`` pattern.

    Two ops with the same config-tuple share one ``torch.library`` entry via
    the process-scoped registry, so we do NOT rebuild per layer — we resolve
    the op from the registry (or build once) and record it in the dict for
    every layer that needs it.
    """
    from sglang.srt.hardware_backend.tpu.attention_pallas_pcp_streaming_op import (
        get_pcp_streaming_mesh,
    )
    from sglang.srt.layers.cp.base import get_cp_strategy
    from sglang.srt.layers.cp.pcp_sequence_layout import (
        resolve_pcp_interleave_size,
        resolve_pcp_ring_tile_tokens,
    )
    from sglang.srt.layers.cp.pcp_streaming_op_registry import (
        get_or_register_pcp_streaming_op,
    )

    mesh = get_pcp_streaming_mesh()
    interleave = resolve_pcp_interleave_size(model_runner.server_args.page_size)
    # Under PCP attn_tp_size == 1, so this rank runs every attention head.
    # Publish it on the strategy: it bounds the kernel's ring tile, and the
    # strategy is what the layout planner and the compile policy ask for the
    # compile bucket.
    local_q_heads = int(model_runner.model_config.num_attention_heads)
    get_cp_strategy().set_local_q_heads(local_q_heads)
    ring_tile = resolve_pcp_ring_tile_tokens(local_q_heads)
    logger.info(
        "[PCP] ring tile = %d local Q rows (%d Q heads/rank, interleave=%d, "
        "page_size=%d)",
        ring_tile,
        local_q_heads,
        interleave,
        model_runner.server_args.page_size,
    )
    language_model = _ensure_language_model(model_runner)
    warned_missing_scales = False
    for layer in _iter_attention_layers(language_model):
        if kv_cache_quant_dtype is not None:
            # PCP FP8 kernel rejects None / non-finite scales at trace time.
            # Default to 1.0 when the checkpoint has no scales (bf16-weight
            # model run under --kv-cache-dtype fp8_e4m3), matching the
            # runtime default in the pre-per-layer-op version of
            # pcp_streaming_attention. Log once if the fallback fires — a
            # FP8-checkpoint model missing scales at warm time is a load-
            # order bug and would silently bake wrong scales into the op.
            if layer.k_scale_float is None or layer.v_scale_float is None:
                if not warned_missing_scales:
                    logger.warning(
                        "PCP streaming warm: layer %s has None k/v scale — "
                        "defaulting to 1.0. Expected for bf16-weight + "
                        "fp8_e4m3 KV cache; unexpected for FP8-weight "
                        "checkpoints and indicates scales weren't loaded "
                        "before backend init.",
                        layer.layer_id,
                    )
                    warned_missing_scales = True
            k_scale = layer.k_scale_float if layer.k_scale_float is not None else 1.0
            v_scale = layer.v_scale_float if layer.v_scale_float is not None else 1.0
        else:
            k_scale = None
            v_scale = None
        sm_scale = float(layer.scaling)
        sliding_window = (
            layer.sliding_window_size if layer.sliding_window_size > 0 else None
        )
        soft_cap = layer.logit_cap if layer.logit_cap else None
        op = get_or_register_pcp_streaming_op(
            mesh=mesh,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            k_scale=k_scale,
            v_scale=v_scale,
            cp_kv_cache_interleave_size=interleave,
            q_block_size=ring_tile,
        )
        op_by_layer_id[int(layer.layer_id)] = op
    if not op_by_layer_id:
        # _iter_attention_layers yielded nothing — model shape doesn't
        # match the ``.model[.model[...]].layers`` chain the walker
        # expects. Fail loudly at warm rather than letting the first
        # serve request KeyError from _pcp_op_by_layer_id[layer_id].
        raise RuntimeError(
            f"PCP streaming warm produced no per-layer ops. "
            f"_iter_attention_layers walked from {type(language_model).__name__} "
            "and found no RadixAttention layers — check the model's "
            "layer-chain shape matches ``.model.layers`` traversal."
        )
