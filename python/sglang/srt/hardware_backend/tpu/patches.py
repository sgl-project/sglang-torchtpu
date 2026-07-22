# SPDX-License-Identifier: Apache-2.0
"""TPU compatibility patches for SGLang.

Ported from https://github.com/google-pytorch/torchtpu-vllm/blob/main/src/vllm_torchtpu/platforms/tpu_platform.py.
Must run in every process (including spawned workers) because
module-level state is re-initialized on import in spawned processes.
All patches are idempotent.
"""

from __future__ import annotations

import importlib
import os

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu._logging import get_logger

logger = get_logger(__name__)

_DYNAMIC_COMPILE_TARGETS: list[tuple[str, str]] = [
    ("sglang.srt.layers.vocab_parallel_embedding", "get_masked_input_and_mask"),
    ("sglang.srt.layers.rotary_embedding.utils", "apply_rotary_pos_emb_native"),
    (
        "sglang.srt.sampling.penaltylib.repetition_penalty",
        "apply_scaling_penalties",
    ),
    (
        "sglang.srt.sampling.sampling_batch_info",
        "apply_scaling_penalties",
    ),
]

_dynamic_compile_unwrapped = False

# Important TPU env vars logged at the end of ``_prepare_tpu_env`` so
# multi-process / multi-host init issues are diagnosable from a single line.
_IMPORTANT_TPU_ENV_VARS = (
    "TORCH_TPU_SLICEBUILDER_ADDRESSES",
    "TORCH_TPU_TOPOLOGY",
    "TORCH_TPU_BASE_PORT",
    "TORCH_TPU_DEVICES_PER_HOST",
    "TPU_WORKER_HOSTNAMES",
    "TPU_WORKER_ID",
    "TPU_TOPOLOGY",
    "TPU_PREMAPPED_BUFFER_SIZE",
    "LIBTPU_INIT_ARGS",
    "XLA_FLAGS",
    "SGLANG_NATIVE_MOVE_KV_CACHE",
)


def _configure_eager_mode() -> None:
    # torch_tpu eager mode (default DEFER_NEVER). DEFER_NEVER materializes each op
    # as its own small, shape-stable graph that the compile cache reuses across
    # steps, so per-step recompiles are cheap and overlap execution. DEFER_AND_FUSE
    # instead fuses a variable-size batch of pending ops into one graph per
    # materialization.
    from torch_tpu._internal import execution_mode

    mode_name = envs.SGLANG_TPU_EAGER_MODE.get()
    execution_mode.eager_mode = getattr(execution_mode.EagerMode, mode_name)
    logger.info("TorchTPU eager mode set to %s.", mode_name)


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


def _set_torch_tpu_topology_env_multihost(
    hosts: list[str],
    base_port: int,
    devices_per_host: int,
    tpu_topology: str,
) -> None:
    """Set the slicebuilder + topology env for a multi-host slice.

    One slicebuilder endpoint per device across ALL hosts; topology is the
    ``TPU_TOPOLOGY`` chip grid with an extra dim appended on v7x (>1 core/chip).
    """
    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
        f"{host}:{base_port + i}" for host in hosts for i in range(devices_per_host)
    )

    if not tpu_topology:
        return

    chip_dims = [int(d) for d in tpu_topology.split("x") if d]
    num_chips = 1
    for d in chip_dims:
        num_chips *= d
    num_devices = len(hosts) * devices_per_host
    # v7x (>1 core/chip) needs an extra topology dim; v4/v5/v6 keep the
    # 3-D chip grid.
    cores_per_chip, rem = divmod(num_devices, num_chips)
    assert rem == 0, (
        f"TPU device/chip mismatch: {num_devices} devices vs {num_chips} "
        f"chips (TPU_TOPOLOGY={tpu_topology!r}, hosts={len(hosts)}, "
        f"devices_per_host={devices_per_host})"
    )
    if cores_per_chip > 1:
        chip_dims.append(cores_per_chip)
    os.environ.setdefault("TORCH_TPU_TOPOLOGY", ",".join(str(d) for d in chip_dims))


def compute_single_host_tpu_env(world_size: int) -> dict:
    """Compute the single-host slicebuilder + topology env WITHOUT mutating
    ``os.environ``.

    Picks ``world_size`` free localhost ports for the per-rank slicebuilder
    endpoints; topology is ``1,1,1`` for ``world_size == 1``, else the
    torch_tpu-derived grid when one is available. Returns a dict with
    ``TORCH_TPU_SLICEBUILDER_ADDRESSES`` (always) and ``TORCH_TPU_TOPOLOGY``
    (when known).

    Returning the values (rather than writing them) lets the Ray driver compute
    the shared port list once and inject it into every same-host actor via
    ``runtime_env`` -- Ray does NOT propagate the driver's own ``os.environ``
    mutations to actors, so each actor computing this independently would pick a
    different port list and the slicebuilder would never rendezvous.
    """
    try:
        import portpicker

        sb_ports = [portpicker.pick_unused_port() for _ in range(world_size)]
    except ImportError:
        import socket

        # Hold sockets open until the ports are read so the OS doesn't reassign.
        _held = [
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(world_size)
        ]
        for s in _held:
            s.bind(("", 0))
        sb_ports = [s.getsockname()[1] for s in _held]
        for s in _held:
            s.close()

    env = {
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(f"localhost:{p}" for p in sb_ports)
    }

    if world_size == 1:
        env["TORCH_TPU_TOPOLOGY"] = "1,1,1"
    else:
        try:
            from torch_tpu._internal.utils import hardware

            topo = hardware.get_tpu_topology(world_size)
            if topo:
                env["TORCH_TPU_TOPOLOGY"] = topo
        except Exception:
            pass

    return env


def _set_torch_tpu_topology_env_single(world_size: int) -> None:
    """Set the slicebuilder + topology env for a single-host process group.

    Thin wrapper over ``compute_single_host_tpu_env`` for the non-Ray / in-process
    path: addresses are always written; topology is only filled when unset.
    """
    env = compute_single_host_tpu_env(world_size)
    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = env[
        "TORCH_TPU_SLICEBUILDER_ADDRESSES"
    ]
    if "TORCH_TPU_TOPOLOGY" in env:
        os.environ.setdefault("TORCH_TPU_TOPOLOGY", env["TORCH_TPU_TOPOLOGY"])


def _prepare_tpu_env(world_size: int = 1, setup_slicebuilder: bool = True) -> None:
    """Set TORCH_TPU_* env vars needed by PjRt for multi-process init.

    Applies LIBTPU_INIT_ARGS overrides, then dispatches to the multi-host or
    single-host helper, which write the slicebuilder + topology env directly.

    ``setup_slicebuilder=False`` applies only the libtpu/env overrides and
    returns without touching the slicebuilder/topology (or logging it). Used by
    the Ray orchestrator: the overrides must be set early (actors inherit them
    before ``import torch_tpu``), but the slicebuilder is built per-rank after
    node discovery, so building it here would emit a bogus single-host
    (localhost) env that never applies to the real ranks.
    """
    # Apply LIBTPU_INIT_ARGS overrides before the early return and before
    # any torch_tpu import; libtpu reads these flags at load time.
    from sglang.srt.hardware_backend.tpu.env_override import (
        apply_tpu_env_overrides,
    )

    apply_tpu_env_overrides()

    # Speculative decoding enables KV-cache moves. The generic fast path is a
    # Triton kernel, which has no active driver on TPU; use the existing native
    # torch implementation instead.
    os.environ.setdefault("SGLANG_NATIVE_MOVE_KV_CACHE", "1")

    if not setup_slicebuilder:
        return

    worker_hostnames = os.environ.get("TPU_WORKER_HOSTNAMES", "")
    hosts = [h for h in worker_hostnames.split(",") if h]

    if len(hosts) > 1:
        # Multihost slicebuilder endpoints are deterministic
        # (``host:base_port+i`` over the Ray-discovered node IPs), so rebuild
        # them unconditionally. Defensively drop any single-host values
        # (``localhost:*`` addresses + a single-box topology) that a launcher or
        # inherited env may have left behind: they are always wrong for a
        # multi-host slice and, left in place, strand each host trying to bind
        # the whole slice on localhost -> PJRT device binding aborts. The helper
        # then recomputes from the real node IPs; recompute is idempotent.
        os.environ.pop("TORCH_TPU_SLICEBUILDER_ADDRESSES", None)
        os.environ.pop("TORCH_TPU_TOPOLOGY", None)
        _set_torch_tpu_topology_env_multihost(
            hosts,
            base_port=int(os.environ.get("TORCH_TPU_BASE_PORT", 8070)),
            devices_per_host=int(os.environ.get("TORCH_TPU_DEVICES_PER_HOST", 8)),
            tpu_topology=os.environ.get("TPU_TOPOLOGY", ""),
        )
    elif "TORCH_TPU_SLICEBUILDER_ADDRESSES" not in os.environ:
        # Single-host picks random free ports, so it must be computed once and
        # inherited by child ranks -- don't clobber an already-set value.
        _set_torch_tpu_topology_env_single(world_size)

    # Only vars that are actually set are printed to keep the output focused.
    logger.info(
        "TPU env:\n%s",
        "\n".join(
            f"  {k}={os.environ[k]}" for k in _IMPORTANT_TPU_ENV_VARS if k in os.environ
        ),
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
            # Set dtype to int32 rather than cumsum's default int64.
            # 32-bit integers are TPU native while s64 requires emulation,
            # and its buggy lowering can run into "Unsupported conversion
            # from vmreg/vreg to U64". Token counts fit int32 comfortably.
            extend_start_loc[1:] = torch.cumsum(
                extend_seq_lens[:-1], dim=0, dtype=torch.int32
            )
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
    _register_aten_fallbacks()
