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
        _held = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(world_size)]
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
                    "TPU set_device(idx=%s) skipped: %s", idx, e,
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
            output[start_q:end_q, :, :] = per_req_out_redudant[
                prefill_seq_len_q:, :, :
            ]
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


def _patch_support_triton_for_tpu() -> None:
    """Make `support_triton("pallas_tpu")` return False.

    SGLang gates Triton kernels behind `support_triton(attn_backend)`, which
    is a hardcoded name allowlist: `backend not in ["torch_native",
    "intel_amx"]`. The `torch_native` backend therefore dodges Triton (no
    Triton driver exists for TPU), but the new "pallas_tpu" name is not in the
    list, so `mem_cache/common.py:write_cache_indices` and
    `forward_batch_info.py` take the Triton path and crash with
    `RuntimeError: 0 active drivers` on TPU.

    Fix: treat "pallas_tpu" exactly like "torch_native" w.r.t. Triton
    availability (correct — TPU has no Triton driver regardless of attention
    backend). Other backends are unaffected. `support_triton` is imported
    by value in two modules, so rebind it there too. Keeps upstream untouched.
    """
    global _support_triton_patched
    if _support_triton_patched:
        return
    _support_triton_patched = True

    import sglang.srt.utils.common as _utils_common

    orig = _utils_common.support_triton

    def support_triton(backend: str) -> bool:
        if backend == "pallas_tpu":
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


_pallas_attention_registered = False


def _register_pallas_attention_backend() -> None:
    """Register the TPU Pallas attention backend ("pallas_tpu") in SGLang's
    attention registry at runtime, without editing upstream
    attention_registry.py.

    `register_attention_backend(name)` is a plain function returning a
    decorator that mutates the module-global ATTENTION_BACKENDS dict, so
    calling it programmatically is equivalent to the upstream
    @register_attention_backend decorator — keeps upstream untouched.

    Ordering: this runs from apply_tpu_patches() via init_backend(), which
    executes at `model_runner` module import (model_runner.py:236-238),
    strictly before ModelRunner.init_attention_backend() (model_runner.py:776)
    looks the name up. The backend module itself is imported lazily inside the
    factory so importing patches does not pull in jax/torch_tpu early.
    """
    global _pallas_attention_registered
    if _pallas_attention_registered:
        return
    _pallas_attention_registered = True

    from sglang.srt.layers.attention.attention_registry import (
        register_attention_backend,
    )

    def create_pallas_tpu_backend(runner):
        from sglang.srt.hardware_backend.tpu.attention_pallas import (
            TpuPallasAttnBackend,
        )

        return TpuPallasAttnBackend(runner)

    register_attention_backend("pallas_tpu")(create_pallas_tpu_backend)
    logger.info("Registered 'pallas_tpu' attention backend.")


def apply_tpu_patches(world_size: int = 1) -> None:
    _prepare_tpu_env(world_size)
    _configure_eager_mode()
    _unwrap_dynamic_compile_fns()
    _patch_set_device_tolerate_mismatch()
    _patch_compute_position_torch()
    _patch_torch_native_attn_backend()
    _patch_support_triton_for_tpu()
    _patch_req_to_token_pool_max_context_len()
    _register_pallas_attention_backend()
    logger.info("TPU patches applied.")
