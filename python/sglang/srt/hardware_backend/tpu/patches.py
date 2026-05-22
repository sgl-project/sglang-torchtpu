"""TPU compatibility patches for SGLang.

Ported from torchtpu-vllm/tpu_inference/platforms/tpu_platform.py.
Must run in every process (including spawned workers) because
module-level state is re-initialized on import in spawned processes.
All patches are idempotent.
"""

from __future__ import annotations

import importlib
import os

from sglang.srt.hardware_backend.tpu._logging import get_logger

logger = get_logger(__name__)

_DYNAMIC_COMPILE_TARGETS: list[tuple[str, str]] = [
    ("sglang.srt.layers.vocab_parallel_embedding", "get_masked_input_and_mask"),
]

_dynamic_compile_unwrapped = False


def _configure_eager_mode() -> None:
    from torch_tpu._internal import execution_mode

    execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE)
    logger.info("TorchTPU eager mode set to DEFER_AND_FUSE.")


def _register_tpu_compile_backend() -> None:
    # Importing `torch_tpu._internal.compile` registers the "tpu" backend with
    # torch._dynamo so that `torch.compile(backend="tpu", ...)` works. Must run
    # before any code applies that decorator (e.g. on imports of layers/models).
    import torch_tpu._internal.compile  # noqa: F401


def _unwrap_dynamic_compile_fns() -> None:
    global _dynamic_compile_unwrapped
    if _dynamic_compile_unwrapped:
        return
    _dynamic_compile_unwrapped = True

    for module_path, fn_name in _DYNAMIC_COMPILE_TARGETS:
        try:
            mod = importlib.import_module(module_path)
        except (ImportError, ModuleNotFoundError):
            continue
        fn = getattr(mod, fn_name, None)
        if fn is not None and hasattr(fn, "__wrapped__"):
            setattr(mod, fn_name, fn.__wrapped__)
            logger.debug(
                "Unwrapped @torch.compile(dynamic=True) from %s.%s",
                module_path,
                fn_name,
            )


def _prepare_tpu_env(world_size: int = 1) -> None:
    """Set TORCH_TPU_* env vars needed by PjRt for multi-process init.

    Ported from torchtpu-vllm: _prepare_singlehost_tpu_env().
    Slicebuilder addresses must match exactly the number of TP workers
    (world_size), not the total process count — detokenizer and tokenizer
    processes do not participate in PjRt coordination.
    """
    if "TORCH_TPU_SLICEBUILDER_ADDRESSES" in os.environ:
        return

    try:
        import portpicker

        sb_ports = [portpicker.pick_unused_port() for _ in range(world_size)]
    except ImportError:
        import socket

        # Hold sockets open until env var is written so OS doesn't reassign.
        _held = [
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(world_size)
        ]
        for s in _held:
            s.bind(("", 0))
        sb_ports = [s.getsockname()[1] for s in _held]
        for s in _held:
            s.close()

    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
        f"localhost:{p}" for p in sb_ports
    )

    if world_size == 1:
        os.environ.setdefault("TORCH_TPU_TOPOLOGY", "1,1,1")
    else:
        try:
            from torch_tpu._internal.utils import hardware

            topo = hardware.get_tpu_topology(world_size)
            if topo:
                os.environ.setdefault("TORCH_TPU_TOPOLOGY", topo)
        except Exception:
            pass

    logger.info(
        "TPU env: TORCH_TPU_SLICEBUILDER_ADDRESSES=%s, TORCH_TPU_TOPOLOGY=%s",
        os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES"),
        os.environ.get("TORCH_TPU_TOPOLOGY"),
    )


_aten_fallbacks_registered = False


def _register_aten_fallbacks() -> None:
    """Register Python fallbacks for ATen ops not yet implemented in torch_tpu.

    These should eventually move into torch_tpu's C++ kernel set (see
    torch_tpu/ops/* for the registration pattern). Kept here as a stopgap
    until those land — each fallback is a pure-torch equivalent of the
    missing op.
    """
    global _aten_fallbacks_registered
    if _aten_fallbacks_registered:
        return
    _aten_fallbacks_registered = True

    import torch

    # aten::count_nonzero.dim_IntList — equivalent to (x != 0).sum over dims.
    @torch.library.register_kernel("aten::count_nonzero.dim_IntList", "PrivateUse1")
    def _count_nonzero_dim_intlist(self, dim):
        return (self != 0).to(torch.int64).sum(dim=dim)

    logger.info("Registered torch_tpu ATen fallbacks: count_nonzero.dim_IntList")


_set_device_patched = False


def _patch_set_device_tolerate_mismatch() -> None:
    """Make torch.tpu.set_device tolerant of sglang/torch_tpu chiplet-index
    disagreements.

    sglang's ModelRunner.init_torch_distributed calls set_device(self.gpu_id).
    torch_tpu auto-binds each process to a chiplet at first PJRT init based
    on env vars set by run_scheduler_process; if the auto-bound chiplet
    index doesn't match sglang's gpu_id (e.g. on v7 where PJRT may assign
    chiplet 5 to local_rank=3 depending on slicebuilder coordination),
    torch_tpu raises "Cannot set TPU device to index X, current process is
    bound to device index Y". Accept the existing binding — chiplet
    assignment is owned by torch_tpu/PJRT, sglang's gpu_id is just a label.
    """
    global _set_device_patched
    if _set_device_patched:
        return
    _set_device_patched = True

    from torch_tpu._internal.device import _device_module

    orig_set_device = _device_module._DeviceModule.set_device

    def wrapped(cls, device):
        try:
            return orig_set_device(device)
        except ValueError as e:
            if "Cannot set TPU device" in str(e):
                idx = (
                    device.index
                    if hasattr(device, "index") and device.index is not None
                    else int(device)
                )
                logger.warning(
                    "TPU set_device(idx=%s) skipped: %s",
                    idx,
                    e,
                )
                return
            raise

    _device_module._DeviceModule.set_device = classmethod(wrapped)


_compute_position_torch_patched = False


def _patch_compute_position_torch() -> None:
    """Replace forward_batch_info.compute_position_torch with a CPU-staged
    variant that doesn't deadlock the torch_tpu device queue.

    Upstream iterates `for p, s in zip(extend_prefix_lens, extend_seq_lens)`
    and calls `torch.arange(p, p+s, device=tpu)` per element. Each iter mixes
    an implicit scalar materialization of the 0-dim TPU tensor with a
    device-side arange — interleaved sync-and-launch ops wedge the torch_tpu
    device queue after the first batch leaves pending work. Symptom: TP=8
    warmup completes, then all schedulers hang at this line on the next
    request.

    Fix: `.tolist()` both length tensors once (single explicit sync), build
    positions on CPU, transfer to device once. Behavior identical on CUDA/CPU
    (which doesn't use this path anyway — CUDA goes through the triton
    variant).
    """
    global _compute_position_torch_patched
    if _compute_position_torch_patched:
        return
    _compute_position_torch_patched = True

    import torch

    from sglang.srt.model_executor import forward_batch_info

    def compute_position_torch_cpu_staged(
        extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor
    ):
        device = extend_prefix_lens.device
        prefix_lens = extend_prefix_lens.tolist()
        seq_lens = extend_seq_lens.tolist()
        positions_cpu = torch.cat(
            [torch.arange(p, p + s) for p, s in zip(prefix_lens, seq_lens)],
            axis=0,
        ).to(torch.int64)
        extend_start_loc = torch.zeros_like(extend_seq_lens)
        if extend_seq_lens.numel() > 1:
            extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
        return positions_cpu.to(device), extend_start_loc

    forward_batch_info.compute_position_torch = compute_position_torch_cpu_staged


_torch_native_attn_backend_patched = False


def _patch_torch_native_attn_backend() -> None:
    """Replace TorchNativeAttnBackend._run_sdpa_forward_{extend,decode} with
    CPU-staged variants that don't crash the torch_tpu device queue under
    TP>1.

    Upstream loops over per-request control values pulled straight off device
    tensors — `extend_seq_lens[i]`, `extend_prefix_lens[i]`, `seq_lens[i]`,
    `req_pool_indices[i]` — and feeds them into Python slices/indexes
    (`query[:, start_q:end_q, :]`, `req_to_token[req_pool_idx, :seq_len_kv]`).
    Each use forces an implicit mid-graph 0-dim device->host materialization.

    On TP=1 that implicit sync succeeds. Under TP>1 the workers run in
    torch_tpu SPMD mode, and the implicit DtoH of a derived scalar fails:
      "Attempted to execute with 1 argument lists when local device count
       is 0 (total replica count: 1, partition count: 2) - TpuMemcpyDtoHDirect:
       DeviceBufferRef has nonzero size, but does not have a PjRtBuffer"
    Symptom: TP=8 server starts, first /v1/chat/completions crashes all
    schedulers at torch_native_backend.py:80 `query[:, start_q:end_q, :]`.

    Fix: `.tolist()` the control tensors once per call (single explicit
    sync), drive the loop with Python ints. Same approach as
    _patch_compute_position_torch. Behavior is identical on CUDA/CPU, which
    don't take this backend (they use FlashInfer/FlashAttention/triton).
    """
    global _torch_native_attn_backend_patched
    if _torch_native_attn_backend_patched:
        return
    _torch_native_attn_backend_patched = True

    import torch
    from torch.nn.functional import scaled_dot_product_attention

    from sglang.srt.layers.attention.torch_native_backend import (
        TorchNativeAttnBackend,
    )

    def _run_sdpa_forward_extend(
        self,
        query,
        output,
        k_cache,
        v_cache,
        req_to_token,
        req_pool_indices,
        seq_lens,
        extend_prefix_lens,
        extend_seq_lens,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # Single explicit host sync of all loop-control tensors. Avoids the
        # per-iteration implicit 0-dim DtoH that wedges torch_tpu SPMD.
        seq_lens_l = seq_lens.tolist()
        extend_prefix_lens_l = extend_prefix_lens.tolist()
        extend_seq_lens_l = extend_seq_lens.tolist()
        req_pool_indices_l = req_pool_indices.tolist()

        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(len(seq_lens_l)):
            extend_seq_len_q = extend_seq_lens_l[seq_idx]
            prefill_seq_len_q = extend_prefix_lens_l[seq_idx]

            seq_len_kv = seq_lens_l[seq_idx]
            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]
            per_req_query_redudant = torch.empty(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )

            per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

            req_pool_idx = req_pool_indices_l[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            per_req_out_redudant = (
                scaled_dot_product_attention(
                    per_req_query_redudant.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
            start_q, start_kv = end_q, end_kv
        return output

    def _run_sdpa_forward_decode(
        self,
        query,
        output,
        k_cache,
        v_cache,
        req_to_token,
        req_pool_indices,
        seq_lens,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        seq_lens_l = seq_lens.tolist()
        req_pool_indices_l = req_pool_indices.tolist()

        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(len(seq_lens_l)):
            seq_len_q = 1
            seq_len_kv = seq_lens_l[seq_idx]
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            req_pool_idx = req_pool_indices_l[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            per_req_out = (
                scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv

        return output

    TorchNativeAttnBackend._run_sdpa_forward_extend = _run_sdpa_forward_extend
    TorchNativeAttnBackend._run_sdpa_forward_decode = _run_sdpa_forward_decode


_support_triton_patched = False


_PALLAS_TPU_BACKENDS = ("pallas_tpu_flash", "pallas_tpu_rpa_v3")


def _patch_support_triton_for_tpu() -> None:
    """Make `support_triton("pallas_tpu_flash" | "pallas_tpu_rpa_v3")` return False.

    SGLang gates Triton kernels behind `support_triton(attn_backend)`, which
    is a hardcoded name allowlist: `backend not in ["torch_native",
    "intel_amx"]`. The `torch_native` backend therefore dodges Triton (no
    Triton driver exists for TPU), but the new Pallas backend names are not in
    the list, so `mem_cache/common.py:write_cache_indices` and
    `forward_batch_info.py` take the Triton path and crash with
    `RuntimeError: 0 active drivers` on TPU.

    Fix: treat the Pallas backend names exactly like "torch_native" w.r.t.
    Triton availability (correct — TPU has no Triton driver regardless of
    attention backend). Other backends are unaffected. `support_triton` is
    imported by value in two modules, so rebind it there too. Keeps upstream
    untouched.
    """
    global _support_triton_patched
    if _support_triton_patched:
        return
    _support_triton_patched = True

    import sglang.srt.utils.common as _utils_common

    orig = _utils_common.support_triton

    def support_triton(backend: str) -> bool:
        if backend in _PALLAS_TPU_BACKENDS:
            return False
        return orig(backend)

    _utils_common.support_triton = support_triton

    # Rebind the by-value imports (and the package re-export).
    for modname in (
        "sglang.srt.utils",
        "sglang.srt.mem_cache.common",
        "sglang.srt.model_executor.forward_batch_info",
    ):
        try:
            mod = importlib.import_module(modname)
        except (ImportError, ModuleNotFoundError):
            continue
        if getattr(mod, "support_triton", None) is orig:
            mod.support_triton = support_triton


_req_to_token_pool_patched = False


def _patch_req_to_token_pool_max_context_len() -> None:
    """Round `ReqToTokenPool`'s `max_context_len` up to a power-of-two
    multiple of the Pallas paged_attention page size.

    Why: `TpuPallasAttnBackend._run_paged_forward_decode` gathers per-request
    KV slots as `req_to_token[:, :Lb]`, where `Lb = _bucket_kv_len(max_len)`
    is a power-of-two multiple of `_PAGE_SIZE`. If `Lb` exceeds
    `req_to_token.shape[1]` (i.e. when `max_context_len` is not itself such a
    value), the gather would have to right-pad the slot tensor with slot 0
    — a live slot owned by another request — which silently leaks that
    request's KV into the attention math if the kernel's length mask ever
    drifts. Rounding the pool width up here makes that branch unreachable.

    Cost: at most ~2x the int32 footprint of `req_to_token` in the worst
    case; for the common context_len values (4096 / 8192 / 16384 / …) the
    width is already aligned and rounding is a no-op.

    Inheritance: `HybridReqToTokenPool` calls `super().__init__(...)`, so
    patching the base class catches it too.
    """
    global _req_to_token_pool_patched
    if _req_to_token_pool_patched:
        return
    _req_to_token_pool_patched = True

    from sglang.srt.hardware_backend.tpu.attention_pallas import (
        _PAGE_SIZE,
        _bucket_kv_len,
    )
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

    orig_init = ReqToTokenPool.__init__

    def wrapped_init(self, *args, **kwargs):
        # All known call sites (model_runner_kv_cache_mixin,
        # HybridReqToTokenPool.super().__init__) pass max_context_len as a
        # keyword, so only handle that form.
        if "max_context_len" in kwargs:
            orig = int(kwargs["max_context_len"])
            rounded = _bucket_kv_len(orig)
            if rounded != orig:
                logger.info(
                    "TPU: rounded ReqToTokenPool.max_context_len %d -> %d "
                    "(power-of-two * page_size=%d) so paged_attention slot "
                    "gather stays within the per-request row.",
                    orig,
                    rounded,
                    _PAGE_SIZE,
                )
                kwargs["max_context_len"] = rounded
        return orig_init(self, *args, **kwargs)

    ReqToTokenPool.__init__ = wrapped_init


# Active attention backend name. Set from two places:
#   1. ``TpuSRTPlatform.apply_server_args_defaults`` — covers the parent
#      process (tokenizer manager / engine) that finalises ``ServerArgs``.
#   2. ``monkey_patch_torch_tpu_env`` in ``utils/patch_torch.py`` — covers
#      each spawned scheduler child. ``apply_tpu_patches`` runs at
#      model_runner module import time (long before the child reaches
#      ``run_scheduler_process``), and ``mp.Process`` does not propagate
#      module-level globals to children, so the child has to repopulate
#      this and re-trigger the swap before ``ModelRunner._init_pools``.
_active_attention_backend: str | None = None


def set_active_attention_backend(name: str | None) -> None:
    """Record the resolved attention backend so later patches can branch on it.

    Called from :meth:`TpuSRTPlatform.apply_server_args_defaults` once the
    backend has been resolved (either from ``--attention-backend`` or the
    platform default). Idempotent; last write wins.
    """
    global _active_attention_backend
    _active_attention_backend = name


_mha_kv_pool_patched = False


def _patch_mha_kv_pool_for_rpa_v3() -> None:
    """Swap ``MHATokenToKVPool`` for the fused TPU pool when the active
    attention backend is ``pallas_tpu_rpa_v3``.

    The RPA v3 kernel reads/writes a single fused 5D KV buffer
    (``[blocks, page_size, num_kv_heads*2 // packing, packing, head_dim]``)
    that the stock :class:`MHATokenToKVPool` does not produce. Rather than
    edit ``model_runner_kv_cache_mixin._init_pools`` (upstream code), we
    rebind the ``MHATokenToKVPool`` symbol *in that module's namespace* to
    :class:`TpuFusedMHATokenToKVPool` — the existing ``pool_cls = ...
    MHATokenToKVPool`` lookup will resolve to the fused variant. This
    matches the rest of this file's "patch around upstream" convention.

    When the active backend is the flash variant (``pallas_tpu_flash``,
    PR #5's design), the upstream pool is the correct one and no swap is
    performed. The flash backend reads K and V via the standard
    ``get_key_buffer`` / ``get_value_buffer`` accessors against that layout
    and would silently mis-read the fused buffer if we swapped it in.

    Ordering: depends on ``set_active_attention_backend`` having been
    called first. ``apply_tpu_patches`` is invoked from
    ``TpuSRTPlatform.init_backend`` (model_runner.py:237, module-import
    time), which runs *after* ``apply_server_args_defaults`` has resolved
    the backend name, so the flag is reliably populated by then.
    """
    global _mha_kv_pool_patched
    if _mha_kv_pool_patched:
        return

    if _active_attention_backend != "pallas_tpu_rpa_v3":
        # No swap needed for flash or torch_native. Leave the patched flag
        # false so a later re-run (e.g. spawned worker that resolves a
        # different backend) can still apply the swap if it lands on RPA v3.
        return

    _mha_kv_pool_patched = True

    import sglang.srt.model_executor.model_runner_kv_cache_mixin as mix
    from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool

    if mix.MHATokenToKVPool is TpuFusedMHATokenToKVPool:
        return

    mix.MHATokenToKVPool = TpuFusedMHATokenToKVPool
    logger.info(
        "TPU: swapped MHATokenToKVPool -> TpuFusedMHATokenToKVPool for the "
        "pallas_tpu_rpa_v3 attention backend (fused 5D KV layout)."
    )


_moe_forward_native_patched = False


def _tpu_tensorized_moe_forward(
    layer, x, topk_weights, topk_ids, moe_runner_config
):
    """TPU bmm-grouped equivalent of fused_moe_native.moe_forward_native.

    Pure function — module-level so tests can import it. The patch installer
    below wraps this with the TPU device gate.

    The reference moe_forward_native iterates over local experts in Python
    (F.linear -> silu -> F.linear -> index_put_ per expert). On torch_tpu
    eager (DEFER_AND_FUSE) a 64-iteration loop queues a deep deferred graph
    that wedges a subsequent .cpu() on tsl::BlockUntilReady.

    This variant groups tokens by selected expert, pads each bucket to a
    common length, and runs the gate-up / activation / down projections as
    two batched matmuls (bmm) over the [E_local, B_max, *] layout. Each
    bmm is a dense [B_max, H] @ [H, 2I] (or [B_max, I] @ [I, H]) per
    expert — both M and N are large enough to fill the MXU, unlike the
    per-token einsum form (M=1) which underutilizes it.

    Contract: assumes StandardDispatcher-style topk_ids (local-or-(-1)). The
    standard dispatcher remaps non-local experts to -1; those slots are
    mapped to expert 0 with weight 0 (zero contribution). The caller's
    post-MoE all-reduce combines partial EP/TP contributions. Fused shared
    experts under EP are rejected because the post-MoE all-reduce would
    over-count the replicated shared contribution.

    Cost: one device->host sync per forward at `bincount().tolist()`
    (unavoidable on torch_tpu per its bincount kernel). Worst-case ~2x
    FLOPs from pad waste vs. ideal valid-slot-only compute; on TPU this
    is still a large net win because the wasted FLOPs land on the MXU
    where they are nearly free, while the einsum form runs on degenerate
    M=1 shapes that spill to the VPU.
    """
    import torch

    from sglang.srt.layers.activation import GeluAndMul, SiluAndMul

    num_fused_shared = (
        getattr(moe_runner_config, "num_fused_shared_experts", 0) or 0
    )
    moe_ep_size = getattr(layer, "moe_ep_size", 1)
    if num_fused_shared > 0 and moe_ep_size > 1:
        raise NotImplementedError(
            "moe_forward_native TPU path does not support fused shared "
            f"experts under EP (num_fused_shared_experts={num_fused_shared}, "
            f"moe_ep_size={moe_ep_size}); the post-MoE all-reduce would "
            "over-count the replicated shared contribution."
        )

    T, K = topk_ids.shape
    H = x.shape[1]
    E_local = layer.w13_weight.shape[0]
    TK = T * K
    device = x.device

    # Flatten and prepare per-(token, slot) views.
    flat_x = x.repeat_interleave(K, dim=0)              # [TK, H]
    flat_ids = topk_ids.reshape(-1)                     # [TK]
    flat_weights = topk_weights.reshape(-1)             # [TK]
    flat_valid = (flat_ids >= 0).to(flat_weights.dtype) # [TK]
    safe_ids = flat_ids.clamp(min=0).to(torch.long)     # [TK]; -1 -> 0

    # Sort by expert id so tokens routed to the same expert are contiguous.
    order = torch.argsort(safe_ids)                     # [TK]
    sorted_ids = safe_ids[order]                        # [TK]
    sorted_x = flat_x[order]                            # [TK, H]
    # Zero out invalid-slot weights so padded contributions sum to 0 later.
    sorted_w = (flat_weights * flat_valid)[order]       # [TK]
    sorted_token_idx = (order // K).to(torch.long)      # [TK]; target token

    # Per-expert counts. bincount graph-breaks on torch_tpu (unavoidable per
    # the torch_tpu kernel); .tolist() forces a single D->H sync per forward.
    counts = torch.bincount(sorted_ids, minlength=E_local).tolist()
    B_max = max(counts) if counts else 0
    if B_max == 0:
        return torch.zeros_like(x)

    # Cumulative offset of each expert in the sorted layout.
    cum_off = [0]
    for c in counts:
        cum_off.append(cum_off[-1] + c)
    # per_expert_offset[e] maps "sorted position i" -> "dense position e*B_max + (i - cum_off[e])"
    # via dense_pos[i] = i + per_expert_offset[sorted_ids[i]].
    per_expert_offset = torch.tensor(
        [e * B_max - cum_off[e] for e in range(E_local)],
        dtype=torch.long,
        device=device,
    )
    dense_pos = torch.arange(TK, device=device) + per_expert_offset.gather(
        0, sorted_ids
    )                                                   # [TK]

    # Scatter sorted activations / weights into the padded [E_local, B_max] grid.
    EB = E_local * B_max
    padded_x = sorted_x.new_zeros(EB, H)
    padded_x.scatter_(0, dense_pos.unsqueeze(1).expand(-1, H), sorted_x)
    padded_x = padded_x.view(E_local, B_max, H)

    padded_w = sorted_w.new_zeros(EB)
    padded_w.scatter_(0, dense_pos, sorted_w)
    padded_w = padded_w.view(E_local, B_max)

    # ===== bmm 1: gate-up =====
    # padded_x: [E_local, B_max, H]; w13: [E_local, 2I, H]
    # -> [E_local, B_max, 2I]
    gate_up = torch.bmm(padded_x, layer.w13_weight.transpose(1, 2))

    w13_bias = getattr(layer, "w13_weight_bias", None)
    if w13_bias is not None:
        # w13_bias: [E_local, 2I] -> broadcast over B_max.
        gate_up = (gate_up.float() + w13_bias.unsqueeze(1)).to(x.dtype)

    # ===== activation =====
    if (
        moe_runner_config.activation == "silu"
        and moe_runner_config.gemm1_alpha is not None
    ):
        # Interleaved [::2]/[1::2] matches the reference GPT-OSS
        # swiglu_gpt_oss_sigmoid_alpha (fused_moe.py); only GPT-OSS uses
        # this variant and its w13_weight is laid out interleaved, not
        # concatenated halves.
        assert moe_runner_config.gemm1_clamp_limit is not None
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(max=moe_runner_config.gemm1_clamp_limit)
        up = up.clamp(
            min=-moe_runner_config.gemm1_clamp_limit,
            max=moe_runner_config.gemm1_clamp_limit,
        )
        hidden = (
            gate
            * torch.sigmoid(gate * moe_runner_config.gemm1_alpha)
            * (up + 1)
        )
    elif moe_runner_config.activation == "silu":
        hidden = SiluAndMul().forward_native(gate_up)
    elif moe_runner_config.activation == "gelu":
        hidden = GeluAndMul().forward_native(gate_up)
    else:
        raise ValueError(
            f"Unsupported activation: {moe_runner_config.activation=}"
        )
    # hidden: [E_local, B_max, I]

    # ===== bmm 2: down =====
    # hidden: [E_local, B_max, I]; w2: [E_local, H, I] -> [E_local, B_max, H]
    expert_outs = torch.bmm(hidden, layer.w2_weight.transpose(1, 2))

    w2_bias = getattr(layer, "w2_weight_bias", None)
    if w2_bias is not None:
        expert_outs = (expert_outs.float() + w2_bias.unsqueeze(1)).to(x.dtype)

    # Apply per-slot routing weights in fp32 and accumulate the K-way sum
    # in fp32 to match the reference moe_forward_native semantic (which
    # upcasts expert_outs to topk_weights dtype before mul+sum). For
    # bf16 activations this is the difference between bf16*bf16+bf16 sum
    # (visible drift on long generations) and the reference's fp32 reduction.
    # Padding slots have padded_w == 0, so their contribution is zero.
    expert_outs_fp32 = expert_outs.float() * padded_w.unsqueeze(-1).float()

    # ===== scatter back to [T, H] =====
    # Map each dense slot to its target token id. Padding slots scatter to
    # token 0 with zeroed values (harmless no-op via index_add_).
    inv_idx = sorted_token_idx.new_zeros(EB)
    inv_idx.scatter_(0, dense_pos, sorted_token_idx)

    out_fp32 = torch.zeros(T, H, dtype=torch.float32, device=device)
    out_fp32.index_add_(0, inv_idx, expert_outs_fp32.reshape(EB, H))
    return out_fp32.to(x.dtype)


def _patch_moe_forward_native() -> None:
    """Replace fused_moe_native.moe_forward_native with a TPU-tensorized
    variant when the input is on a TPU device. CPU/CUDA inputs fall through
    to the original implementation. Body lives in
    `_tpu_tensorized_moe_forward` above (module-level for testability).
    """
    global _moe_forward_native_patched
    if _moe_forward_native_patched:
        return
    _moe_forward_native_patched = True

    import torch

    from sglang.srt.layers.moe import fused_moe_native

    original_moe_forward_native = fused_moe_native.moe_forward_native

    def _is_tpu(x: torch.Tensor) -> bool:
        return x.device.type == "tpu" or str(x.device).startswith("tpu")

    def moe_forward_native_tpu_dispatch(
        layer, x, topk_output, moe_runner_config
    ):
        if moe_runner_config.apply_router_weight_on_input:
            raise NotImplementedError()
        if _is_tpu(x):
            topk_weights, topk_ids, _ = topk_output
            return _tpu_tensorized_moe_forward(
                layer, x, topk_weights, topk_ids, moe_runner_config
            )
        return original_moe_forward_native(
            layer, x, topk_output, moe_runner_config
        )

    fused_moe_native.moe_forward_native = moe_forward_native_tpu_dispatch


_total_memory_patched = False


def _patch_total_memory_for_shared_chip() -> None:
    """Divide TpuSRTPlatform.get_device_total_memory by cores-per-chip when
    multiple ranks share one chip's HBM.

    TPU v7 only exposes topology slices with 2 cores per chip (e.g. "1,1,1,2"
    for world_size=2). Both TP ranks land on the same chip and share its
    192GB HBM, but the platform reports 192GB per rank — sglang then sizes
    the KV pool as `mem_fraction_static * 192GB` per rank, so the combined
    pool exceeds the chip and the first prefill OOMs. Read cores_per_chip
    from `TORCH_TPU_TOPOLOGY` (4th dim) and scale total memory down.
    """
    global _total_memory_patched
    if _total_memory_patched:
        return
    _total_memory_patched = True

    from sglang.srt.hardware_backend.tpu.tpu_platform import TpuSRTPlatform

    original = TpuSRTPlatform.get_device_total_memory

    def get_device_total_memory(self, device_id: int = 0) -> int:
        hbm = original(self, device_id)
        topo = os.environ.get("TORCH_TPU_TOPOLOGY")
        if topo:
            try:
                dims = [int(x) for x in topo.split(",")]
                cores_per_chip = dims[3] if len(dims) >= 4 else 1
                if cores_per_chip > 1:
                    hbm //= cores_per_chip
            except (ValueError, IndexError):
                pass
        return hbm

    TpuSRTPlatform.get_device_total_memory = get_device_total_memory


def apply_tpu_patches(world_size: int = 1) -> None:
    _prepare_tpu_env(world_size)
    _configure_eager_mode()
    _register_tpu_compile_backend()
    _unwrap_dynamic_compile_fns()
    _patch_set_device_tolerate_mismatch()
    _patch_compute_position_torch()
    _patch_torch_native_attn_backend()
    _patch_support_triton_for_tpu()
    _patch_req_to_token_pool_max_context_len()
    _patch_mha_kv_pool_for_rpa_v3()
    _patch_moe_forward_native()
    _patch_total_memory_for_shared_chip()
    _register_aten_fallbacks()
    logger.info("TPU patches applied.")
