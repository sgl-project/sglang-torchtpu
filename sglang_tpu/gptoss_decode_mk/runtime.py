# SPDX-License-Identifier: Apache-2.0
"""Torch side of the GPT-OSS decode megakernel serving integration (opt-in).

``SGLANG_TPU_GPTOSS_DECODE_MK=1`` makes ``GptOssForCausalLM.load_weights`` tap the
checkpoint stream into this rank's megakernel weights (``weights.MkWeightBuilder``), keeping
the stock weights for prefill/extend. Afterwards ``GptOssModel.forward`` runs plain DECODE
batches (no spec, batch bucket <= 64) as:

    embed_tokens -> gptoss_decode_mk_step (one torch op: XLA glue + the megakernel, all
    36 layers, in-kernel all-reduce) -> 36 x gptoss_mk_kv_write (in-place, one per layer
    pool) -> final norm -> (stock) logits processor / sampling

reading and writing the same per-layer fused KV pools the stock ``pallas_tpu_rpa_v3``
backend uses. Everything else (EXTEND/MIXED/IDLE/verify/draft, larger buckets) takes the
stock layer stack. Every choice is static per compiled graph (forward mode, spec_info
presence, batch-bucket shape), so the TPU compile cache holds one MK decode graph per
bucket and nothing recompiles per step.

Configuration checks that are known at load time disable the integration with one log
line; pool/attention-backend mismatches found at the first decode trace raise (the env var
is an explicit opt-in, so a silent fallback would only hide a broken experiment).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch import nn

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

ENV_ENABLE = "SGLANG_TPU_GPTOSS_DECODE_MK"
ENV_PREFIX = "SGLANG_TPU_GPTOSS_DECODE_MK_"
ENV_MOE_MODE = "SGLANG_TPU_GPTOSS_DECODE_MK_MOE"  # "tp" (default) or "ep"
MAX_BATCH = 64

STEP_OP_NAME = "sglang_tpu_pallas::gptoss_decode_mk_step"
KV_WRITE_OP_NAME = "sglang_tpu::gptoss_decode_mk_kv_write"
_STEP_OPS: dict = {}
_KV_WRITE_OP = None
_KV_WRITE_LAYERS_OPS: dict = {}
# "layers" (default): one Pallas call writes all layers' K/V (L aliased pools);
# "per_layer": one call per layer pool
ENV_KV_WRITE = "SGLANG_TPU_GPTOSS_DECODE_MK_KV_WRITE"
# =1: diagnostics thread + eager-run detection (see _Diagnostics)
ENV_TIMING = "SGLANG_TPU_GPTOSS_DECODE_MK_TIMING"
# timing-only ablations (wrong outputs!): "noop_kernel" (same op + operands, the kernel
# only touches them), "no_kv_write" (skip the KV write op), "no_mk" (skip both MK ops)
ENV_ABLATE = "SGLANG_TPU_GPTOSS_DECODE_MK_ABLATE"
ABLATIONS = ("", "none", "noop_kernel", "no_kv_write", "no_mk")
# =0: keep MXFP4 code words as int32 on device (bitcast to uint32 inside the op)
ENV_U32 = "SGLANG_TPU_GPTOSS_DECODE_MK_U32"


def kernel_knobs_from_env() -> dict:
    """Kernel A/B knobs (unset = kernel defaults): w_dtype fp8|mxfp4, ar_mode
    auto|oneshot|rsag, ar_fold 0|1, npb, npb_swa, kv_bufs, kernel ablations (comma list)."""
    kw: dict = {
        "w_dtype": (envs.SGLANG_TPU_GPTOSS_DECODE_MK_W_DTYPE.get() or "fp8")
        .strip()
        .lower()
    }
    ar_mode = envs.SGLANG_TPU_GPTOSS_DECODE_MK_AR_MODE.get().strip().lower()
    if ar_mode:
        kw["ar_mode"] = ar_mode
    ar_fold = envs.SGLANG_TPU_GPTOSS_DECODE_MK_AR_FOLD.get().strip().lower()
    if ar_fold:
        kw["ar_fold"] = ar_fold not in ("0", "false")
    for k, env_field in (
        ("npb", envs.SGLANG_TPU_GPTOSS_DECODE_MK_NPB),
        ("npb_swa", envs.SGLANG_TPU_GPTOSS_DECODE_MK_NPB_SWA),
        ("kv_bufs", envs.SGLANG_TPU_GPTOSS_DECODE_MK_KV_BUFS),
        ("w_slots", envs.SGLANG_TPU_GPTOSS_DECODE_MK_W_SLOTS),
        ("vmem_mib", envs.SGLANG_TPU_GPTOSS_DECODE_MK_VMEM_MIB),
    ):
        val = env_field.get()
        if val is not None:
            kw[k] = int(val)
    kablate = envs.SGLANG_TPU_GPTOSS_DECODE_MK_KERNEL_ABLATE.get().strip()
    if kablate:
        kw["ablate"] = tuple(a for a in kablate.split(",") if a)
    return kw


def enabled_by_env() -> bool:
    return bool(envs.SGLANG_TPU_GPTOSS_DECODE_MK.get())


def static_disable_reason(hf_config, parallel, spec_algorithm) -> Optional[str]:
    """Why the megakernel cannot serve this server at all (None if it can)."""
    if parallel.pp_size != 1:
        return f"pp_size={parallel.pp_size} (needs 1)"
    if parallel.attn_dp_size != 1 or parallel.attn_cp_size != 1:
        return (
            f"attn dp/cp = {parallel.attn_dp_size}/{parallel.attn_cp_size} (needs 1/1)"
        )
    if getattr(parallel, "attn_dcp_size", 1) != 1:
        return f"decode context parallel {parallel.attn_dcp_size} (needs 1)"
    if parallel.attn_tp_size != hf_config.num_key_value_heads:
        return (
            f"attn_tp_size={parallel.attn_tp_size}; the kernel needs exactly one KV head "
            f"per rank (TP{hf_config.num_key_value_heads})"
        )
    if spec_algorithm not in (None, "", "NONE", "none"):
        return f"speculative decoding ({spec_algorithm}) is not supported"
    if hf_config.num_hidden_layers % 2:
        return "odd layer count"
    want = [
        "sliding_attention" if l % 2 == 0 else "full_attention"
        for l in range(hf_config.num_hidden_layers)
    ]
    if list(getattr(hf_config, "layer_types", want)) != want:
        return "layer_types are not (sliding, full) pairs"
    head_dim = getattr(hf_config, "head_dim", None) or (
        hf_config.hidden_size // hf_config.num_attention_heads
    )
    if head_dim != 64:
        return f"head_dim={head_dim} (needs 64)"
    return None


# ----------------------------------------------------------------------------- load time


class WeightTap:
    """Wraps the checkpoint iterator of ``load_weights``; see module docstring."""

    def __init__(self, causal_lm: nn.Module, c, rank: int, tp_size: int):
        from sglang_tpu.gptoss_decode_mk.weights import MkWeightBuilder

        self.causal_lm, self.c, self.rank, self.tp_size = causal_lm, c, rank, tp_size
        self.builder = MkWeightBuilder(c, rank)
        self.error: Optional[str] = None

    def wrap(self, weights):
        for name, tensor in weights:
            if self.error is None:
                try:
                    # before handing the tensor on: a streamed buffer may be reused later
                    self.builder.consume(name, tensor)
                except Exception as e:  # noqa: BLE001 - disable the opt-in, keep serving
                    self.error = f"{name}: {type(e).__name__}: {e}"
            yield name, tensor

    def finish(self) -> None:
        if self.error is None:
            miss = self.builder.missing()
            if miss:
                self.error = (
                    f"checkpoint lacks HF GPT-OSS tensors (MXFP4 *_blocks/_scales or "
                    f"bf16 experts) for {len(miss)} (layer, key) pairs, e.g. "
                    f"{miss[:3]}"
                )
        if self.error is not None:
            logger.warning("GPT-OSS decode megakernel disabled: %s", self.error)
            self.builder = None
            return
        model = self.causal_lm.model
        device = next(model.parameters()).device
        host = self.builder.finalize()
        if self.c.w_dtype == "fp8":
            import dataclasses
            import time

            from sglang_tpu.gptoss_decode_mk.weights import mxfp4_to_fp8

            t = time.time()
            host8, bad = mxfp4_to_fp8(self.c, host)
            logger.info(
                "GPT-OSS decode megakernel: MXFP4 -> fp8 experts on rank %d: %d inexact "
                "elements (%.0f s)",
                self.rank,
                bad,
                time.time() - t,
            )
            if bad:
                logger.warning(
                    "GPT-OSS decode megakernel: fp8 experts would not be exact (%d "
                    "elements); using the MXFP4 expert path",
                    bad,
                )
                self.c = dataclasses.replace(self.c, w_dtype="mxfp4")
            else:
                host = host8
            del host8
        self.builder = None
        mk = GptOssDecodeMK(self.c, self.rank, self.tp_size, host, device)
        del host
        # registers as a submodule (buffers move with the model)
        model._tpu_decode_mk = mk
        from sglang_tpu.gptoss_decode_mk.weights import per_rank_bytes

        logger.info(
            "GPT-OSS decode megakernel ON (rank %d/%d, %s MoE, %s experts, ar_mode=%s "
            "ar_fold=%s npb=%d npb_swa=%d kv_bufs=%d ablate=%s): %.2f GiB extra weights per "
            "rank; DECODE batches <= %d use it, everything else uses the stock layers.",
            self.rank,
            self.tp_size,
            self.c.moe_mode,
            self.c.w_dtype,
            self.c.ar_mode,
            self.c.ar_fold,
            self.c.npb,
            self.c.npb_swa,
            self.c.kv_bufs,
            ",".join(self.c.ablate) or "-",
            per_rank_bytes(self.c) / 2**30,
            MAX_BATCH,
        )


def maybe_make_weight_tap(causal_lm: nn.Module) -> Optional[WeightTap]:
    """A ``WeightTap`` when the opt-in env var is set and the server can use the kernel."""
    if (
        not enabled_by_env()
        or getattr(causal_lm.model, "_tpu_decode_mk", None) is not None
    ):
        return None
    from sglang.srt.runtime_context import get_parallel, get_spec
    from sglang.srt.utils import is_tpu

    if not is_tpu():
        logger.warning("%s=1 ignored: not a TPU platform", ENV_ENABLE)
        return None
    parallel = get_parallel()
    spec = getattr(get_spec(), "speculative_algorithm", None)
    reason = static_disable_reason(causal_lm.config, parallel, spec)
    if reason is not None:
        logger.warning("GPT-OSS decode megakernel disabled: %s", reason)
        return None
    from sglang_tpu.gptoss_decode_mk.weights import config_from_hf

    moe_mode = (envs.SGLANG_TPU_GPTOSS_DECODE_MK_MOE.get() or "tp").strip().lower()
    c = config_from_hf(
        causal_lm.config,
        tp_size=parallel.attn_tp_size,
        moe_mode=moe_mode,
        **kernel_knobs_from_env(),
    )
    return WeightTap(causal_lm, c, parallel.attn_tp_rank, parallel.attn_tp_size)


# ----------------------------------------------------------------------------- torch ops


def _signature_fn(names: list[str], n_out: int, body):
    """A function with explicit ``jax.Array`` parameters (custom-op schema inference and
    the jax_op static-arg inference read the signature; ``*args`` is not accepted)."""
    import jax

    params = ", ".join(f"{n}: jax.Array" for n in names)
    ret = "tuple[" + ", ".join(["jax.Array"] * n_out) + "]"
    src = f"def gptoss_decode_mk_step({params}) -> {ret}:\n    return _body({', '.join(names)})\n"
    ns: dict[str, Any] = {"jax": jax, "_body": body}
    # dont_inherit: this module's `from __future__ import annotations` would otherwise turn
    # the annotations into strings (torch_tpu's jax_op needs real ones, as RPA notes)
    code = compile(src, "<gptoss_decode_mk_step>", "exec", dont_inherit=True)
    exec(code, ns)  # noqa: S102 - static, generated from a fixed name list
    return ns["gptoss_decode_mk_step"]


def get_step_op(c0, tp_size: int, noop: bool = False):
    """Torch op ``(x, kv_lens, page_indices, swa_page_indices, cos_sin, rank, *W, *pools)
    -> (hidden [B, H], kvn [L, BR, 1, packing, 128], newpg [2B], off [B])``.

    One op for every decode bucket: the JAX function specializes the kernel to the input
    shapes at export time (B, pages per request, pool sizes, KV dtype), so each bucket
    compiles once. Each rank passes its own blocks as replicated (``P()``) operands of a
    ``shard_map`` over the ``"tp"`` mesh, as ``fused_moe_rs_op`` / ``pp_wave`` do.
    """
    key = (c0, tp_size, noop)
    if key in _STEP_OPS:
        return _STEP_OPS[key]
    import jax
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal.pallas import get_pallas_mesh

    from sglang_tpu.gptoss_decode_mk import adapter as A
    from sglang_tpu.gptoss_decode_mk.weights import w_names
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    W_NAMES = w_names(c0)

    mesh = get_pallas_mesh(("tp",), (tp_size,))
    L = c0.L
    names = (
        ["x", "kv_lens", "page_indices", "swa_page_indices", "cos_sin", "rank"]
        + [f"w_{k}" for k in W_NAMES]
        + [f"kv_{i}" for i in range(L)]
    )
    nw = len(W_NAMES)

    def per_rank(x, kv_lens, page_indices, swa_page_indices, cos_sin, rank, *rest):
        weights, pools = list(rest[:nw]), list(rest[nw:])
        B = x.shape[0]
        c = A.trace_config(
            c0,
            B=B,
            pps=page_indices.shape[0] // B,
            kv_dtype=pools[0].dtype,
            nb_swa=pools[0].shape[0],
            nb_full=pools[L // 2].shape[0],
            page=pools[0].shape[1],
        )
        logger.info(
            "GPT-OSS decode megakernel: exporting B=%d pages/req=%d kv=%s "
            "(SMEM ~%d KiB)",
            B,
            c.max_pages,
            c.kv_dtype,
            A.smem_bytes_estimate(c) // 1024,
        )
        step = A.per_rank_noop if noop else A.per_rank_step
        return step(
            c, rank, x, kv_lens, page_indices, swa_page_indices, cos_sin, weights, pools
        )

    specs = (P(),) * len(names)
    smap = jax.shard_map(
        per_rank, mesh=mesh, in_specs=specs, out_specs=(P(),) * 4, check_vma=False
    )
    fn = _signature_fn(names, 4, smap)
    op = cpu_export_jax_op(
        STEP_OP_NAME + ("_noop" if noop else ""),
        fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=(P(),) * 4,
    )
    _STEP_OPS[key] = op
    return op


def get_kv_write_op():
    """Torch op ``kv_write(pg [B], off [B], kvn [BR, 1, packing, 128], pool)`` mutating
    ``pool`` in place (donated + aliased, the ``kv_row_write`` pattern)."""
    global _KV_WRITE_OP
    if _KV_WRITE_OP is not None:
        return _KV_WRITE_OP
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.gptoss_decode_mk import adapter as A
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    def _kv_write(pg, off, kvn, pool):
        return A.kv_write(pg, off, kvn, pool)

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _kv_write,
        name="jax_gptoss_decode_mk_kv_write",
        donate_argnums=(3,),
        input_output_aliases={3: 0},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        KV_WRITE_OP_NAME, mutates_args=("pool",), device_types=["tpu"]
    )
    def gptoss_decode_mk_kv_write(
        pg: torch.Tensor, off: torch.Tensor, kvn: torch.Tensor, pool: torch.Tensor
    ) -> None:
        with tpu_info_for_cpu_export():
            jax_kernel(pg, off, kvn, pool)

    @gptoss_decode_mk_kv_write.register_fake
    def _(pg, off, kvn, pool) -> None:
        return None

    _KV_WRITE_OP = gptoss_decode_mk_kv_write
    return _KV_WRITE_OP


def _mutating_op_fn(names: list[str], body):
    """A function ``(name0: Tensor, ...) -> None`` calling ``body(*args)``: torch's custom-op
    schema inference needs explicit parameters to mark each pool as mutated."""
    params = ", ".join(f"{n}: torch.Tensor" for n in names)
    src = f"def gptoss_decode_mk_kv_write_layers({params}) -> None:\n    _body({', '.join(names)})\n"
    ns: dict[str, Any] = {"torch": torch, "_body": body}
    code = compile(src, "<gptoss_decode_mk_kv_write_layers>", "exec", dont_inherit=True)
    exec(code, ns)  # noqa: S102 - static, generated from a fixed name list
    return ns["gptoss_decode_mk_kv_write_layers"]


def get_kv_write_layers_op(L: int):
    """Torch op ``kv_write_layers(newpg [2B], off [B], kvn [L, BR, 1, packing, 128],
    kv_0, ..., kv_{L-1})`` mutating all L pools in place with ONE Pallas call (each pool
    donated and aliased to its own output, as RPA v3 does for its single pool)."""
    if L in _KV_WRITE_LAYERS_OPS:
        return _KV_WRITE_LAYERS_OPS[L]
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.gptoss_decode_mk import adapter as A
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    def _kv_write_layers(newpg, off, kvn, *pools):
        return A.kv_write_layers(newpg, off, kvn, *pools)

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _kv_write_layers,
        name=f"jax_gptoss_decode_mk_kv_write_layers_{L}",
        donate_argnums=tuple(range(3, 3 + L)),
        input_output_aliases={3 + i: i for i in range(L)},
        warn_deprecated=False,
    )
    names = ["newpg", "off", "kvn"] + [f"kv_{i}" for i in range(L)]

    def body(*args):
        with tpu_info_for_cpu_export():
            jax_kernel(*args)

    op = torch.library.custom_op(
        f"{KV_WRITE_OP_NAME}_layers_{L}",
        _mutating_op_fn(names, body),
        mutates_args=tuple(names[3:]),
        device_types=["tpu"],
    )
    op.register_fake(_mutating_op_fn(names, lambda *a: None))
    _KV_WRITE_LAYERS_OPS[L] = op
    return op


# ----------------------------------------------------------------------------- module


class GptOssDecodeMK(nn.Module):
    """Holds one rank's megakernel weights (non-persistent buffers, so the compiled graph
    takes them as inputs like the stock weights) and runs the decode step."""

    def __init__(self, c, rank: int, tp_size: int, host: dict, device):
        super().__init__()
        from sglang_tpu.gptoss_decode_mk.weights import w_names

        self.c, self.rank, self.tp_size = c, rank, tp_size
        self.w_names = w_names(c)
        for k in self.w_names:
            self.register_buffer(
                "w_" + k, _to_device(host[k], device), persistent=False
            )
        self.register_buffer(
            "rank_t",
            torch.tensor([rank], dtype=torch.int32, device=device),
            persistent=False,
        )
        self.ablate = envs.SGLANG_TPU_GPTOSS_DECODE_MK_ABLATE.get().strip().lower()
        if self.ablate not in ABLATIONS:
            raise ValueError(
                f"{ENV_ABLATE}={self.ablate!r}; expected one of {ABLATIONS}"
            )
        # register the ops eagerly, outside any torch.compile trace
        self._step_op = get_step_op(c, tp_size, noop=self.ablate == "noop_kernel")
        self.kv_write_mode = (
            (envs.SGLANG_TPU_GPTOSS_DECODE_MK_KV_WRITE.get() or "layers")
            .strip()
            .lower()
        )
        if self.kv_write_mode == "per_layer":
            self._kv_write_op = get_kv_write_op()
        else:
            self.kv_write_mode = "layers"
            self._kv_write_op = get_kv_write_layers_op(c.L)
        from sglang.srt.model_executor.forward_context import get_attn_backend
        from sglang_tpu.gptoss_decode_mk.adapter import pool_layer_order

        self._get_attn_backend = get_attn_backend
        # pool operand order: SWA, then full layers
        self.layer_ids = pool_layer_order(c.L)
        self.diag = _Diagnostics.maybe_start(rank)
        logger.info(
            "GPT-OSS decode megakernel ops: kv_write=%s ablate=%s weight dtypes %s",
            self.kv_write_mode,
            self.ablate or "none",
            {
                k: str(getattr(self, "w_" + k).dtype)
                for k in self.w_names
                if k in ("w13q", "w2q", "w13", "w2")  # expert matrices (mxfp4 | fp8)
            },
        )

    @staticmethod
    def applies(
        forward_batch, input_ids, input_embeds, pp_proxy_tensors, layers_to_capture
    ) -> bool:
        """Static (per compiled graph) choice between the megakernel and the stock stack."""
        return (
            forward_batch.forward_mode.is_decode()
            and getattr(forward_batch, "spec_info", None) is None
            and input_embeds is None
            and pp_proxy_tensors is None
            and not layers_to_capture
            and input_ids is not None
            and input_ids.shape[0] <= MAX_BATCH
            and input_ids.shape[0] == forward_batch.seq_lens.shape[0]
        )

    def _check_backend(self, attn_backend, pool, model) -> None:
        meta = getattr(attn_backend, "forward_metadata", None)
        problems = []
        if meta is None or getattr(meta, "swa_page_indices", None) is None:
            problems.append("attention backend has no hybrid-SWA RPA metadata")
        elif getattr(meta, "decode_query_size", 1) != 1:
            problems.append("decode_query_size != 1")
        if getattr(attn_backend, "_dcp", 1) != 1:
            problems.append("decode context parallelism")
        mapping = getattr(pool, "layers_mapping", None)
        want = {l: (l // 2, l % 2 == 0) for l in range(self.c.L)}
        if mapping is None or any(mapping.get(l) != v for l, v in want.items()):
            problems.append(
                "KV pool is not a SWAKVPool with (sliding, full) layer pairs"
            )
        else:
            for l in (0, 1):
                buf = pool.get_fused_kv_buffer(l)
                if buf.dim() != 5 or buf.shape[2] != 1 or buf.shape[4] != 128:
                    problems.append(
                        f"layer {l} pool shape {tuple(buf.shape)} is not "
                        "[pages, page, 1, packing, 128]"
                    )
                if buf.dtype not in (torch.float8_e4m3fn, torch.bfloat16):
                    problems.append(f"KV dtype {buf.dtype}")
        for layer in model.layers[: self.c.L]:
            a = layer.self_attn.attn
            for s in (
                getattr(a, "k_scale_float", None),
                getattr(a, "v_scale_float", None),
            ):
                if s not in (None, 1.0):
                    problems.append(f"KV scale {s} (the kernel assumes 1.0)")
                    break
        if problems:
            raise RuntimeError(
                f"{ENV_ENABLE}=1 but the decode megakernel cannot read this server's KV "
                f"state: {'; '.join(sorted(set(problems)))}. Unset {ENV_ENABLE}."
            )

    def run(self, model, input_ids: torch.Tensor, forward_batch) -> torch.Tensor:
        """Embedding -> megakernel (all layers) -> in-place KV writes -> final norm."""
        # Everything below must stay dynamo-traceable (fullgraph): no context managers,
        # locks, host timers or syncs. Under torch.compile ``diag`` is the constant None
        # (is_compiling() is a trace-time constant), so the diagnostics only run eagerly.
        diag = None if torch.compiler.is_compiling() else self.diag
        if diag is not None:
            diag.on_run(input_ids.shape[0])
        attn_backend = self._get_attn_backend()  # what RadixAttention uses
        pool = attn_backend.token_to_kv_pool
        self._check_backend(attn_backend, pool, model)
        meta = attn_backend.forward_metadata
        pools = [pool.get_fused_kv_buffer(l) for l in self.layer_ids]
        x = model.embed_tokens(input_ids)
        if self.ablate == "no_mk":  # timing only: everything but the two MK ops
            forward_batch.residual_stream = None
            return model.norm(x)
        cos_sin = model.layers[0].self_attn.rotary_emb.cos_sin_cache
        weights = [getattr(self, "w_" + k) for k in self.w_names]
        t0 = diag.tick(x) if diag is not None else None
        hidden, kvn, newpg, off = self._step_op(
            x,
            meta.kv_lens,
            meta.page_indices,
            meta.swa_page_indices,
            cos_sin,
            self.rank_t,
            *weights,
            *pools,
        )
        t1 = diag.tick(hidden) if diag is not None else None
        B = x.shape[0]
        if self.ablate == "no_kv_write":  # timing only: the KV pools go stale
            pass
        elif self.kv_write_mode == "layers":
            self._kv_write_op(newpg, off, kvn, *pools)
        else:
            pg_swa, pg_full = newpg[:B], newpg[B:]
            for i, l in enumerate(self.layer_ids):
                self._kv_write_op(
                    pg_swa if l % 2 == 0 else pg_full, off, kvn[l], pools[i]
                )
        if diag is not None:
            diag.on_eager_times(t0, t1, diag.tick(pools[0]))
        # the stock stack's residual stream is not used on this path
        forward_batch.residual_stream = None
        return model.norm(hidden)


def _to_device(t: torch.Tensor, device) -> torch.Tensor:
    """Upload; MXFP4 code words become genuine uint32 device tensors when torch_tpu can
    hold them (a base tensor of the kernel's dtype: no bitcast in the graph, and no dtype
    view, which torch_tpu would materialize as a copy), else stay int32."""
    if t.dtype == torch.int32 and envs.SGLANG_TPU_GPTOSS_DECODE_MK_U32.get():
        try:
            u = t.view(torch.uint32).to(device)
            assert u._base is None and u.dtype == torch.uint32
            return u
        except Exception as e:  # noqa: BLE001 - keep int32 + in-graph bitcast
            logger.warning(
                "uint32 weights unavailable on %s (%s); using int32", device, e
            )
    return t.to(device)


class _Diagnostics:
    """``SGLANG_TPU_GPTOSS_DECODE_MK_TIMING=1``: where does the integration spend time?

    - Every decode-MK ``run`` executed *eagerly* (outside a compiled graph: graph break,
      compile disabled or fallback) is counted, and its op times are measured with a
      device sync around the step op and the KV write. Nothing here is ever called
      from inside a dynamo trace (``run`` uses ``diag=None`` while compiling).
    - A daemon thread logs every 30 s: MK eager runs, and torch._dynamo's
      graph-break / unique-graph / recompile counters (graph breaks around the custom ops
      would split the decode forward into several executables with host round trips).
    Inside a compiled graph the ops cannot be timed from the host; use the
    ``SGLANG_TPU_GPTOSS_DECODE_MK_ABLATE`` arms and xprof for that.
    """

    def __init__(self, rank: int):
        import threading

        self.rank = rank
        self.eager = 0
        self.buckets: set[int] = set()
        self.eager_ms: list[tuple[float, float]] = []
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True, name="mk-diag").start()

    @classmethod
    def maybe_start(cls, rank: int):
        if not envs.SGLANG_TPU_GPTOSS_DECODE_MK_TIMING.get():
            return None
        return cls(rank)

    def on_run(self, bs: int) -> None:  # eager runs only
        with self.lock:
            self.buckets.add(int(bs))
            self.eager += 1

    def tick(self, t: torch.Tensor):  # eager runs only
        import time

        try:
            from torch_tpu._internal.sync.sync import synchronize

            synchronize([t], wait=True)
        except Exception:  # noqa: BLE001 - timing is best effort
            pass
        return time.perf_counter()

    def on_eager_times(self, t0, t1, t2) -> None:
        if None not in (t0, t1, t2):
            with self.lock:
                self.eager_ms.append(((t1 - t0) * 1e3, (t2 - t1) * 1e3))

    def _loop(self) -> None:
        import time

        while True:
            time.sleep(30)
            try:
                from torch._dynamo.utils import counters

                gb = counters.get("graph_break", {})
                stats = counters.get("stats", {})
                with self.lock:
                    em = self.eager_ms[-50:]
                    self.eager_ms.clear()
                    msg = (
                        f"MK diag rank{self.rank}: eager_runs={self.eager} "
                        f"eager_buckets={sorted(self.buckets)}"
                    )
                if em:
                    s = sorted(a for a, _ in em)
                    w = sorted(b for _, b in em)
                    msg += (
                        f" eager step-op ms med {s[len(s) // 2]:.2f} "
                        f"kv-write ms med {w[len(w) // 2]:.2f} (n={len(em)})"
                    )
                msg += (
                    f" | dynamo graph_breaks={sum(gb.values())} "
                    f"unique_graphs={stats.get('unique_graphs', 0)} "
                    f"calls_captured={stats.get('calls_captured', 0)}"
                )
                top = sorted(gb.items(), key=lambda kv: -kv[1])[:3]
                if top:
                    msg += " top: " + "; ".join(f"{n}x {r[:160]}" for r, n in top)
                logger.info(msg)
            except Exception as e:  # noqa: BLE001
                logger.info("MK diag failed: %s", e)
