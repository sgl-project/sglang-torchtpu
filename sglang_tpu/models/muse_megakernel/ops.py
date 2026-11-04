# SPDX-License-Identifier: Apache-2.0
"""Muse Spark decode megakernel: weight preparation, the torch op, and the forward hook.

``MuseMegakernel.from_model`` runs once after weight loading, before the KV pool is sized. It
builds the stacked K-major copies the kernel streams (attention, pooler and unpooler in fp8,
the router in bf16) and collects the per-layer expert references sglang already owns. ``forward``
replaces the decoder for decode and DFlash target-verify batches; prefill keeps the native
path. The LM head, logits and sampling stay in sglang.

The step runs as one jitted program per device: the megakernel (collective, reads the KV
pools) and then ``kv_write_call`` (local, scatters this step's K/V rows into the pools in
place).
"""

import logging

import jax
import msgspec
import torch
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang_tpu.mem_cache.kv_pool import TpuFusedMHATokenToKVPool, kv_pack_heads
from sglang_tpu.models.muse_megakernel import kernel as K

logger = logging.getLogger(__name__)

_OP_NS = "sglang_tpu_pallas"
# Larger steps run the native decoder: 32 rows (cc=8 with 4 DFlash drafts) is the
# largest batch measured faster on the megakernel.
_MAX_ROWS = 32
# KV pages per attention chunk: 4 beats 2 on long-context verify steps and is neutral
# at short context; 8+ costs VMEM the expert ring needs.
_CHUNK_PAGES = 4


def _k_major(weight: torch.Tensor) -> torch.Tensor:
    return weight.detach().t().contiguous()


def _fp8_channelwise(w_kn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """K-major ``[K, N]`` -> fp8 e4m3 ``[K, N]`` and f32 scales ``[N]`` (column amax / 448)."""
    w = w_kn.float()
    scale = (w.abs().amax(dim=0) / 448.0).clamp_min(1e-12)
    return (w / scale).to(torch.float8_e4m3fn), scale


def _stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(tensors, dim=0)


class _ModelGeometry:
    """Static model numbers the kernel needs, read from the loaded model."""

    def __init__(self, model) -> None:
        cfg = model.text_config
        decoder = model.model
        layer0 = decoder.layers[0]
        attn = layer0.self_attn
        moe = layer0.mlp
        self.n_layers = len(decoder.layers)
        self.hidden = cfg.hidden_size
        self.head_dim = attn.head_dim
        self.q_heads = attn.num_heads
        self.kv_heads = attn.num_kv_heads
        self.n_experts = moe.num_experts
        self.top_k = moe.top_k
        self.expert_hidden = moe.expert_hidden_size
        # w2 is [E, I_local, H_e] in the fused-RS expert-TP layout (fp4 packs the last dim)
        self.inter_local = moe.experts.w2_weight.shape[1]
        self.rms_eps = cfg.rms_norm_eps
        self.post_eps = cfg.post_norm_eps
        self.scale_query_by = attn.query_scale
        self.attn_scale = attn.head_dim**-0.5
        self.router_mult = moe.router_multiplier
        self.route_scale = moe.route_scale
        self.router_eps = moe.route_eps
        self.global_layers = tuple(
            i for i, t in enumerate(cfg.layer_types) if t == "full_attention"
        )
        self.window = cfg.sliding_window
        self.rope_layers = tuple(
            i for i, theta in enumerate(cfg.layer_rope_theta) if theta > 0
        )
        if set(self.rope_layers) != set(range(self.n_layers)) - set(self.global_layers):
            raise NotImplementedError("RoPE must be exactly the sliding-window layers")


def _expert_kind(experts) -> str:
    dt = experts.w13_weight.dtype
    if dt == torch.float4_e2m1fn_x2:
        return "fp4"
    if dt == torch.float8_e4m3fn:
        return "fp8"
    raise NotImplementedError(f"unsupported expert weight dtype {dt}")


def _expert_refs(experts, kind: str) -> list[torch.Tensor]:
    """``(w13, w13_scale, w2, w2_scale)`` in the fused-RS expert-TP layout."""
    s13, s2 = experts.w13_weight_scale, experts.w2_weight_scale
    if kind == "fp8" and s13.dim() == 4:
        # channelwise (E, 1, 1, N): give the kernel the (E, 2, N) form the fused-RS path uses
        s13 = s13.reshape(s13.shape[0], 1, s13.shape[-1]).expand(-1, 2, -1).contiguous()
        s2 = s2.reshape(s2.shape[0], 1, s2.shape[-1]).expand(-1, 2, -1).contiguous()
    return [experts.w13_weight, s13, experts.w2_weight, s2]


class MuseMegakernel(torch.nn.Module):
    """Holds the kernel's weights and one compiled op per (rows, rows per request, capture)."""

    def __init__(
        self,
        *,
        geo: _ModelGeometry,
        weights: dict,
        experts: list,
        kinds: tuple,
        tp_size: int,
        tp_rank: int,
        max_rows: int,
        max_position: int,
    ) -> None:
        super().__init__()
        self.geo = geo
        self.kinds = kinds
        self.tp_size = tp_size
        self.max_rows = max_rows
        self.max_position = max_position
        self.chunk_pages = _CHUNK_PAGES
        # the pools' layout: SGLANG_TPU_KV_PACK_HEADS stores the rank's heads in one row
        self.kv_packed = (
            kv_pack_heads(
                geo.kv_heads, geo.head_dim, TpuFusedMHATokenToKVPool.HEAD_DIM_ALIGNMENT
            )
            > 1
        )
        # Buffers so --tpu-compile-tracer direct lifts them as graph inputs; the expert
        # entries alias FusedMoE parameters.
        for name in K.WEIGHT_NAMES:
            self.register_buffer(f"w_{name}", weights[name], persistent=False)
        for i, t in enumerate(experts):
            self.register_buffer(f"expert_{i}", t, persistent=False)
        self.n_expert_refs = len(experts)
        self.register_buffer(
            "rank_t",
            torch.tensor([tp_rank], dtype=torch.int32, device=weights["smalls"].device),
            persistent=False,
        )
        self._ops: dict = {}

    # ------------------------------------------------------------------ preparation
    @classmethod
    def from_model(cls, model, *, max_rows: int = _MAX_ROWS) -> "MuseMegakernel":
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        if not (
            parallel.moe_ep_size == 1
            and parallel.moe_tp_size == parallel.tp_size == parallel.attn_tp_size
        ):
            raise NotImplementedError(
                "the Muse megakernel needs expert TP across the whole TP group"
            )
        # Remote DMA over one host's devices only; rows are padded to the TP size.
        if parallel.tp_size not in (1, 8):
            raise NotImplementedError(
                f"the Muse megakernel runs at TP 1 or 8, got {parallel.tp_size}"
            )
        if parallel.attn_dcp_size != 1:
            raise NotImplementedError("the Muse megakernel reads unstrided page tables")
        geo = _ModelGeometry(model)
        decoder = model.model
        L, H = geo.n_layers, geo.hidden
        Ep = -(-geo.n_experts // K.LANE) * K.LANE
        device = decoder.norm.weight.device
        qkvg, wo, wpool, wunpool, wrouter, smalls = [], [], [], [], [], []
        experts, kinds = [], []
        for layer in decoder.layers:
            attn, moe = layer.self_attn, layer.mlp
            if layer.post_attention_layernorm.weight is not None:
                raise NotImplementedError("affine post-attention norm")
            qkvg.append(
                _k_major(
                    torch.cat([attn.qkv_proj.weight, attn.gate_proj.weight], dim=0)
                )
            )
            wo.append(_k_major(attn.o_proj.weight))
            wpool.append(_k_major(moe.pre_expert_proj.weight))
            wunpool.append(_k_major(moe.post_expert_proj.weight))
            rw = moe.gate.weight.detach()
            rw = rw if rw.shape == (H, geo.n_experts) else rw.t()
            rpad = torch.zeros(H, Ep, dtype=torch.bfloat16, device=device)
            rpad[:, : geo.n_experts] = rw.to(torch.bfloat16)
            wrouter.append(rpad)
            sm = torch.zeros(K.SMALL_ROWS, H, dtype=torch.float32, device=device)
            S = K.SMALL
            sm[S["pre_attn_w"]] = layer.input_layernorm.gain()
            sm[S["attn_alpha"]] = (
                layer.post_attention_residual_gate.alpha.float().reshape(-1)
            )
            sm[S["attn_beta"]] = (
                layer.post_attention_residual_gate.beta.float().reshape(-1)
            )
            sm[S["pre_ffn_w"]] = layer.pre_feedforward_layernorm.gain()
            sm[S["post_ffn_w"]] = layer.post_feedforward_layernorm.gain()
            sm[S["ffn_alpha"]] = (
                layer.post_feedforward_residual_gate.alpha.float().reshape(-1)
            )
            sm[S["ffn_beta"]] = (
                layer.post_feedforward_residual_gate.beta.float().reshape(-1)
            )
            sm[S["pre_expert_w"], : geo.expert_hidden] = moe.pre_expert_norm.gain()
            sm[S["expert_bias"], :Ep] = K.NEG
            sm[S["expert_bias"], : geo.n_experts] = moe.e_score_correction_bias.float()
            sm[S["post_expert_w"], : geo.expert_hidden] = moe.post_expert_norm.gain()
            # fp8 e4m3 dense copies with per-output-channel scales (the router stays bf16).
            for name, mats in (
                ("wqkvg", qkvg),
                ("wo", wo),
                ("wpool", wpool),
                ("wunpool", wunpool),
            ):
                mats[-1], scale = _fp8_channelwise(mats[-1])
                sm[S[name + "_s"], : scale.numel()] = scale
            smalls.append(sm)
            kind = _expert_kind(moe.experts)
            kinds.append(kind)
            experts.extend(_expert_refs(moe.experts, kind))
        rope = decoder.rotary_emb
        rope_tab = torch.cat(
            [
                rope.cos.repeat_interleave(2, dim=1),
                rope.sin.repeat_interleave(2, dim=1),
            ],
            dim=1,
        ).float()
        # Concatenated, not filled in place: an in-place-built tensor bound into the
        # compiled graph is re-materialized every step (torch-tpu 0.2).
        final = torch.cat(
            [
                decoder.norm.gain()[None].float(),
                torch.zeros(7, H, dtype=torch.float32, device=device),
            ]
        )
        weights = {
            "rope": rope_tab,
            "final_norm": final,
            "smalls": _stack(smalls),
            "wqkvg": _stack(qkvg),
            "wo": _stack(wo),
            "wpool": _stack(wpool),
            "wrouter": _stack(wrouter),
            "wunpool": _stack(wunpool),
        }
        extra = sum(t.numel() * t.element_size() for t in weights.values())
        logger.info(
            "Muse megakernel prepared: %d layers, experts %s, %.2f GiB of K-major copies per device",
            L,
            "/".join(sorted(set(kinds))),
            extra / 2**30,
        )
        mk = cls(
            geo=geo,
            weights=weights,
            experts=experts,
            kinds=tuple(kinds),
            tp_size=parallel.tp_size,
            tp_rank=parallel.tp_rank,
            max_rows=max_rows,
            max_position=rope_tab.shape[0],
        )
        _INSTANCE[:] = [mk]
        _register_op()
        return mk

    # ------------------------------------------------------------------ op building
    def config(
        self,
        *,
        rows: int,
        q: int,
        n_req: int,
        n_page_entries: int,
        page_size: int,
        capture: tuple,
    ) -> K.MKConfig:
        g = self.geo
        cfg = K.MKConfig(
            hidden=g.hidden,
            head_dim=g.head_dim,
            q_heads=g.q_heads,
            kv_heads=g.kv_heads,
            n_layers=g.n_layers,
            n_experts=g.n_experts,
            top_k=g.top_k,
            expert_hidden=g.expert_hidden,
            inter_local=g.inter_local,
            tp=self.tp_size,
            rms_eps=g.rms_eps,
            post_eps=g.post_eps,
            scale_query_by=g.scale_query_by,
            attn_scale=g.attn_scale,
            router_mult=g.router_mult,
            route_scale=g.route_scale,
            router_eps=g.router_eps,
            window=g.window,
            global_layers=g.global_layers,
            expert_kinds=self.kinds,
            page_size=page_size,
            pages_per_seq=n_page_entries // n_req,
            n_pages_entries=n_page_entries,
            max_position=self.max_position,
            rows=rows,
            rows_per_req=q,
            n_req=n_req,
            capture_layers=capture,
            compact_rows=4 if rows <= _COMPACT4_MAX_ROWS else 8,
            chunk_pages=self.chunk_pages,
            kv_packed=self.kv_packed,
        )
        # Deepest expert prefetch, then largest dense ring (2.25 MiB slots), whose declared
        # scratch leaves VMEM headroom for Mosaic temporaries; rows-scaled scratch grows ~0.24 MiB/row.
        for depth, cap in _DEPTH_RING_PREFERENCE:
            cand = msgspec.structs.replace(cfg, expert_depth=depth)
            slots = K.fit_ring_slots(cand, cap=cap)
            cand = msgspec.structs.replace(
                cand, ring_slots=slots, stream_depth=slots - 1
            )
            if K.scratch_vmem_bytes(cand) <= _SCRATCH_VMEM_BUDGET:
                return cand
        raise ValueError(
            f"no Muse megakernel ring fits the VMEM budget at {rows} rows of {q}"
        )

    def check_kv_pools(self, pool) -> None:
        """Raise unless the KV pools have the layout the kernel reads: bf16 fused pools,
        heads packed exactly when the kernel was built for packed heads."""
        pools = (
            (pool.full_kv_pool, pool.swa_kv_pool)
            if isinstance(pool, SWAKVPool)
            else (pool,)
        )
        for p in pools:
            if (
                type(p) is not TpuFusedMHATokenToKVPool
                or p.store_dtype != torch.bfloat16
            ):
                raise NotImplementedError(
                    f"the Muse megakernel reads bf16 fused KV pools, got "
                    f"{type(p).__name__} ({p.store_dtype})"
                )
            if (p.kv_pack_heads > 1) != self.kv_packed:
                raise NotImplementedError(
                    f"KV pool packs {p.kv_pack_heads} heads per row, but the Muse "
                    f"megakernel was built with kv_packed={self.kv_packed}"
                )

    def callable_for(self, cfg: K.MKConfig):
        if cfg not in self._ops:
            self._ops[cfg] = _build_callable(cfg, self.tp_size)
        return self._ops[cfg]

    # ------------------------------------------------------------------ forward
    def handles(self, forward_batch, num_tokens: int) -> bool:
        mode = forward_batch.forward_mode
        if not (mode.is_decode() or mode.is_target_verify()):
            return False
        return _round_up(num_tokens, self.tp_size) <= self.max_rows

    def forward(
        self,
        *,
        embeds: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        capture: tuple,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``embeds [T, H]`` bf16 (all-reduced) -> final-normed hidden ``[T, H]`` (+ aux).

        Runs inside the compiled forward: only tensors and Python ints reach the op.
        """
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        md = backend.forward_metadata
        page_indices = md.page_indices
        swa_page_indices = (
            md.swa_page_indices if md.swa_page_indices is not None else page_indices
        )
        n_req = forward_batch.seq_lens.shape[0]
        T = embeds.shape[0]
        if T % n_req:
            raise ValueError(f"{T} rows do not split evenly over {n_req} requests")
        rows = _round_up(T, self.tp_size)
        pad = rows - T
        emb = embeds.to(torch.bfloat16)
        pos = positions.to(torch.int32)
        if pad:
            emb = torch.cat([emb, emb.new_zeros(pad, emb.shape[1])], dim=0)
            pos = torch.cat([pos, pos.new_zeros(pad)], dim=0)
        n_live = md.active_num_tokens.reshape(1).to(torch.int32)
        meta = torch.cat([n_live, self.rank_t], dim=0)
        pool = backend.token_to_kv_pool
        self.check_kv_pools(pool)
        kv = [pool.get_fused_kv_buffer(i) for i in range(self.geo.n_layers)]
        weights = [getattr(self, f"w_{n}") for n in K.WEIGHT_NAMES]
        experts = [getattr(self, f"expert_{i}") for i in range(self.n_expert_refs)]
        hidden, aux = torch.ops.sglang_tpu_pallas.muse_megakernel(
            meta,
            pos,
            page_indices,
            swa_page_indices,
            emb,
            weights,
            experts,
            kv,
            T // n_req,
            n_req,
            list(capture),
        )
        return hidden[:T], (aux[:T] if capture else None)


# The op implementation runs outside the traced region and needs the prepared weights'
# geometry; one model per process.
_INSTANCE: list = []
_REGISTERED: list = []


def _register_op() -> None:
    """The one generic op: shapes and the Python ints select (and cache) the compiled kernel."""
    if _REGISTERED:
        return

    @torch.library.custom_op(
        f"{_OP_NS}::muse_megakernel", mutates_args=("kv",), device_types=["tpu"]
    )
    def op(
        meta: torch.Tensor,
        pos: torch.Tensor,
        pf: torch.Tensor,
        psw: torch.Tensor,
        emb: torch.Tensor,
        weights: list[torch.Tensor],
        experts: list[torch.Tensor],
        kv: list[torch.Tensor],
        rows_per_req: int,
        n_req: int,
        capture: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from sglang_tpu.utils.pallas_export import tpu_info_for_cpu_export

        mk = _INSTANCE[0]
        cfg = mk.config(
            rows=emb.shape[0],
            q=rows_per_req,
            n_req=n_req,
            n_page_entries=pf.shape[0],
            page_size=kv[0].shape[1],
            capture=tuple(capture),
        )
        callable_ = mk.callable_for(cfg)
        with tpu_info_for_cpu_export():
            outs = callable_(meta, pos, pf, psw, emb, *weights, *experts, *kv)
        return outs[0], outs[1]

    @op.register_fake
    def _(meta, pos, pf, psw, emb, weights, experts, kv, rows_per_req, n_req, capture):
        aux_w = len(capture) * emb.shape[1] if capture else K.LANE
        return emb.new_empty((emb.shape[0], emb.shape[1])), emb.new_empty(
            (emb.shape[0], aux_w)
        )

    _REGISTERED.append(True)


# Mosaic's temporaries come on top of the declared scratch and grow with rows per request
# (verify attention): 24 rows of 6 compile at 46.8 MiB declared, 32 rows of 8 overflowed the
# 60 MiB limit at 48.7 MiB (~11.5 MiB of temporaries).
_SCRATCH_VMEM_BUDGET = 48 * 2**20
_DEPTH_RING_PREFERENCE = ((3, 11), (3, 4), (1, 4))
# Measured device step (real geometry, random routing): the 8-row block-diag variant is
# slower than the 4-row one at 8-24 rows and faster only at 48.
_COMPACT4_MAX_ROWS = 24


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


def _build_callable(cfg: K.MKConfig, tp: int):
    """The megakernel + KV scatter as one jitted ``shard_map`` program over the TP mesh."""
    from torch_tpu._internal.pallas import pallas as pallas_impl

    from sglang_tpu.distributed.oneshot_collectives import _get_mesh
    from sglang_tpu.utils.pallas_jax_op import _CpuExportJaxCallable

    L = cfg.n_layers
    nw = len(K.WEIGHT_NAMES)
    n_args = 5 + nw + 4 * L + L
    kv0 = 5 + nw + 4 * L
    mesh = _get_mesh(tp)
    mk = K.megakernel_call(cfg)
    kvw = K.kv_write_call(cfg)

    def body(*xs):
        hid, aux, newkv = mk(*xs)
        pools = kvw(*xs[:4], newkv, *xs[kv0:])
        return (hid, aux, *pools)

    sharded = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(P("model"),) * n_args,
        out_specs=(P("model"),) * (2 + L),
        check_vma=False,
    )

    def fn(
        meta: jax.Array,
        pos: jax.Array,
        pf: jax.Array,
        psw: jax.Array,
        emb: jax.Array,
        *rest: jax.Array,
    ) -> tuple:
        return tuple(sharded(meta, pos, pf, psw, emb, *rest))

    donate = tuple(range(kv0, n_args))
    out_shardings = tuple(NamedSharding(mesh, P("model")) for _ in range(2 + L))
    jit_fn = jax.jit(
        fn, donate_argnums=donate, keep_unused=True, out_shardings=out_shardings
    )
    tag = f"r{cfg.rows}_q{cfg.rows_per_req}_n{cfg.n_req}_c{len(cfg.capture_layers)}_p{cfg.pages_per_seq}"
    trace_key = pallas_impl._get_kernel_invocation_key(
        f"muse_megakernel_{tag}_{id(fn)}", [], {"donate": donate}
    )
    callable_ = _CpuExportJaxCallable(
        name=f"jax_muse_megakernel_{tag}",
        jit_fn=jit_fn,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=(P("model"),) * n_args,
        static_argnums=(),
        donate_argnums=donate,
        input_output_aliases={kv0 + i: 2 + i for i in range(L)},
    )
    callable_.sglang_output_shardings = out_shardings
    logger.info(
        "Muse megakernel built: rows=%d q=%d requests=%d capture=%s pages/seq=%d",
        cfg.rows,
        cfg.rows_per_req,
        cfg.n_req,
        cfg.capture_layers,
        cfg.pages_per_seq,
    )
    return callable_
