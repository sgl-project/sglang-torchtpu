# SPDX-License-Identifier: Apache-2.0
"""TPU compatibility patches for SGLang.

Ported from https://github.com/vllm-project/vllm-torchtpu/blob/main/src/vllm_torchtpu/platforms/tpu_platform.py.
Must run in every process (including spawned workers) because
module-level state is re-initialized on import in spawned processes.
All patches are idempotent.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Sequence

from sglang.srt.environ import envs
from sglang_tpu._logging import get_logger

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
    "TORCH_TPU_ALLOW_LEGACY_DEVICE_APIS",
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


_tpu_sampler_backend_registered = False


def _register_tpu_sampler_backend() -> None:
    """Register the TPU sampler factory once during backend initialization.
    Idempotence protects repeated platform setup in tests and worker processes."""

    global _tpu_sampler_backend_registered
    if _tpu_sampler_backend_registered:
        return

    from sglang.srt.layers.sampler import register_sampler_backend
    from sglang_tpu.sampler import TpuSampler

    register_sampler_backend("tpu", TpuSampler)
    _tpu_sampler_backend_registered = True


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


def build_slicebuilder_addresses(
    hosts: Sequence[str], base_port: int, devices_per_host: int
) -> str:
    return ",".join(
        f"{host}:{base_port + i}" for host in hosts for i in range(devices_per_host)
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
    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = build_slicebuilder_addresses(
        hosts, base_port, devices_per_host
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
    """Compute parent-safe single-host slicebuilder env without mutating
    ``os.environ`` or importing TorchTPU.

    Picks ``world_size`` free localhost ports for the per-rank slicebuilder
    endpoints. The trivial single-rank topology is known without hardware
    discovery; multi-rank topology is resolved later inside each worker.

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

    return env


def _resolve_single_host_tpu_topology(world_size: int) -> str | None:
    """Resolve topology inside a worker, where importing TorchTPU is allowed."""
    from torch_tpu._internal.utils import hardware

    return hardware.get_tpu_topology(world_size)


def _set_torch_tpu_topology_env_single(world_size: int) -> None:
    """Set the slicebuilder + topology env for a single-host process group.

    Thin wrapper over ``compute_single_host_tpu_env`` for the non-Ray / in-process
    path. Addresses are always written; the trivial single-rank topology is
    filled when unset. Multi-rank topology requires worker-side resolution.
    """
    env = compute_single_host_tpu_env(world_size)
    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = env[
        "TORCH_TPU_SLICEBUILDER_ADDRESSES"
    ]
    if "TORCH_TPU_TOPOLOGY" in env:
        os.environ.setdefault("TORCH_TPU_TOPOLOGY", env["TORCH_TPU_TOPOLOGY"])


def _prepare_tpu_env(
    world_size: int | None = None,
    setup_slicebuilder: bool = True,
    resolve_single_host_topology: bool = False,
) -> None:
    """Set TORCH_TPU_* env vars needed by PjRt for multi-process init.

    Applies LIBTPU_INIT_ARGS overrides, then prepares the slicebuilder
    rendezvous. Multi-host topology comes from launcher metadata; single-host
    hardware topology can be resolved later inside a worker.

    ``setup_slicebuilder=False`` applies only the libtpu/env overrides and
    returns without touching the slicebuilder/topology (or logging it). Used by
    the Ray orchestrator: the overrides must be set early (actors inherit them
    before ``import torch_tpu``), but the slicebuilder is built per-rank after
    node discovery, so building it here would emit a bogus single-host
    (localhost) env that never applies to the real ranks.

    ``resolve_single_host_topology=True`` is worker-only. Parent and Ray driver
    setup deliberately leave multi-rank hardware discovery to the workers so
    they never import the ``torch_tpu`` package.
    """
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Apply LIBTPU_INIT_ARGS overrides before the early return and before
    # any torch_tpu import; libtpu reads these flags at load time.
    from sglang_tpu.env_override import (
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

    if (
        len(hosts) <= 1
        and resolve_single_host_topology
        and "TORCH_TPU_TOPOLOGY" not in os.environ
    ):
        topology = _resolve_single_host_tpu_topology(world_size)
        if topology:
            os.environ["TORCH_TPU_TOPOLOGY"] = topology

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

    # TODO: Remove the fallback after https://github.com/google-pytorch/torch_tpu/issues/2491 is resolved.
    @torch.library.register_kernel("aten::masked_scatter_", "PrivateUse1")
    def _masked_scatter_(self, mask, source):
        src_flat = source.to(self.dtype).reshape(-1)
        if src_flat.numel() == 0:
            if mask.to(torch.bool).any():
                raise RuntimeError(
                    "Number of elements of source < number of ones in mask"
                )
            return self
        flat_mask = mask.to(torch.bool).broadcast_to(self.shape).reshape(-1)
        idx = (torch.cumsum(flat_mask.to(torch.int32), 0, dtype=torch.int32) - 1).clamp(
            min=0
        )
        out = torch.where(flat_mask, src_flat[idx], self.reshape(-1)).reshape(
            self.shape
        )
        self.copy_(out)
        return self

    # aten::upsample_bicubic2d -- no TPU kernel; run the CPU kernel and copy
    # back. Vision towers use it to resize position embeddings per image grid
    # (MoonViT's Learnable2DInterpPosEmb in Kimi-VL), so inputs are small (tens
    # of MB at most) and the round-trip is cheap. CPU bicubic has no bf16
    # kernel, so compute in fp32 and cast back.
    @torch.library.register_kernel("aten::upsample_bicubic2d", "PrivateUse1")
    def _upsample_bicubic2d(
        self, output_size, align_corners, scales_h=None, scales_w=None
    ):
        out = torch._C._nn.upsample_bicubic2d(
            self.detach().cpu().float(), output_size, align_corners, scales_h, scales_w
        )
        return out.to(device=self.device, dtype=self.dtype)

    logger.info(
        "Registered torch_tpu ATen fallbacks: count_nonzero.dim_IntList, "
        "masked_scatter_, upsample_bicubic2d"
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
            [
                torch.arange(p, p + s, device="cpu")
                for p, s in zip(prefix_lens, seq_lens)
            ],
            axis=0,
        ).to(torch.int64)
        # Reuse the host lists from the readbacks above. Computing this on the
        # input tensors would compile zeros/cumsum/view updates on TPU.
        extend_start_loc_cpu = torch.zeros_like(extend_seq_lens, device="cpu")
        if len(seq_lens) > 1:
            extend_start_loc_cpu[1:] = torch.cumsum(
                torch.tensor(seq_lens[:-1], dtype=torch.int32, device="cpu"),
                dim=0,
                dtype=torch.int32,
            )
        return positions_cpu.to(device), extend_start_loc_cpu.to(device)

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


_PALLAS_TPU_BACKENDS = (
    "pallas_tpu_flash",
    "dsv4",
    "tpu_dsv4",
    "pallas_tpu_rpa_v3",
    "pallas_tpu_rpa_batched",
    "pallas_tpu_mla",
    "pallas_tpu_mla_sparse",
)


def _patch_support_triton_for_tpu() -> None:
    """Make `support_triton` return False for every `_PALLAS_TPU_BACKENDS` name.

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
        # write_cache_indices moved out of mem_cache.common into mem_cache.allocation.
        "sglang.srt.mem_cache.allocation",
        "sglang.srt.mem_cache.common",
        "sglang.srt.model_executor.forward_batch_info",
        "sglang.srt.layers.rotary_embedding.mrope",
    ):
        try:
            mod = importlib.import_module(modname)
        except (ImportError, ModuleNotFoundError):
            continue
        if getattr(mod, "support_triton", None) is orig:
            mod.support_triton = support_triton


def _patch_tpu_torch_pallas_register_positional() -> None:
    """Let the torch_tpu bindings accept torch's inductor calling convention."""
    try:
        from torch_tpu._internal.pallas import tpu_torch_pallas
    except ImportError:  # pragma: no cover - non-TPU host
        return
    orig_register = tpu_torch_pallas.register_custom_kernel
    orig_call = tpu_torch_pallas.call_custom_kernel

    try:
        orig_register("_sglang_signature_probe", "probe", b"")
    except TypeError:
        pass  # skew present -- install the shims below
    except Exception:  # pragma: no cover - signature already matches
        return
    else:
        return

    def register_custom_kernel(name, kernel_key, serialized_mlir_module):
        return orig_register(
            name, kernel_key, serialized_mlir_module=serialized_mlir_module
        )

    def call_custom_kernel(*args, **kwargs):
        if args and not isinstance(args[0], str):
            # torch codegen convention: (inputs, output_shapes, name, kernel_key)
            inputs, output_shapes, name, kernel_key = args
            return orig_call(
                name, kernel_key, inputs=inputs, output_shapes=output_shapes
            )
        return orig_call(*args, **kwargs)

    tpu_torch_pallas.register_custom_kernel = register_custom_kernel
    tpu_torch_pallas.call_custom_kernel = call_custom_kernel


_future_map_direct_handoff_patched = False


def _patch_future_map_direct_handoff() -> None:
    """Route FutureMap.stash/publish through the TPU direct handoff, so an
    unchanged EAGLE3 top-k-1 decode batch skips the store/gather round trip.
    The resolve side is already hooked via resolve_device_accept_seq_lens_tpu."""
    global _future_map_direct_handoff_patched
    if _future_map_direct_handoff_patched:
        return
    _future_map_direct_handoff_patched = True

    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang_tpu.overlap_utils_tpu import (
        flush_direct_handoff_tpu,
        publish_tpu,
        stash_tpu,
    )

    orig_stash = FutureMap.stash
    orig_publish = FutureMap.publish
    orig_stash_bonus_tokens = FutureMap.stash_bonus_tokens
    orig_resolve_mixed_spec_tails = FutureMap.resolve_mixed_spec_tails
    FutureMap.tpu_direct_handoff = None
    FutureMap.tpu_published_seq_lens = None

    def stash(self, future_indices, payload):
        if not stash_tpu(self, future_indices, payload):
            orig_stash(self, future_indices, payload)

    def publish(self, future_indices, new_seq_lens, confidence=None):
        orig_publish(self, future_indices, new_seq_lens, confidence)
        publish_tpu(self, future_indices, new_seq_lens)

    # Every other buffer reader or writer must see the pending rows stored first.
    def stash_bonus_tokens(self, indices, bonus_tokens):
        flush_direct_handoff_tpu(self)
        orig_stash_bonus_tokens(self, indices, bonus_tokens)

    def resolve_mixed_spec_tails(self, batch):
        # The mixed input resolve reads output_tokens_buf right after this.
        flush_direct_handoff_tpu(self)
        orig_resolve_mixed_spec_tails(self, batch)

    FutureMap.stash = stash
    FutureMap.publish = publish
    FutureMap.stash_bonus_tokens = stash_bonus_tokens
    FutureMap.resolve_mixed_spec_tails = resolve_mixed_spec_tails


_raiden_is_load_back_event_done_patched = False


def _patch_raiden_is_load_back_event_done() -> None:
    """Route ``UnifiedRadixCache.is_load_back_event_done`` to the raiden controller.

    The PD decode restore loop polls this method, which reads a per-layer
    ``finish_event`` from the done counter's slot. Raiden stores the combined
    future-event itself in a ``BatchDoneCounter`` slot, and a raiden DMA can
    land at different times on different ranks, so the answer is combined
    over the cache's rank groups: every rank flips the request READY in the
    same step. Like upstream's method (#41393) the poll only answers; the
    tree-side bookkeeping (``loading_check``) runs from check_hicache_events,
    which every rank enters together. Other controllers keep the original
    method.
    """
    global _raiden_is_load_back_event_done_patched
    if _raiden_is_load_back_event_done_patched:
        return
    _raiden_is_load_back_event_done_patched = True

    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang_tpu.raiden_hybrid_cache_controller import (
        TpuRaidenHybridCacheController,
    )

    orig_is_load_back_event_done = UnifiedRadixCache.is_load_back_event_done

    def is_load_back_event_done(self, consumer_index):
        controller = self.cache_controller
        if not isinstance(controller, TpuRaidenHybridCacheController):
            return orig_is_load_back_event_done(self, consumer_index)
        return controller.is_load_back_event_done(consumer_index, self._all_reduce)

    UnifiedRadixCache.is_load_back_event_done = is_load_back_event_done


_prefill_transfer_finish_from_raiden_patched = False


def _patch_prefill_transfer_finish_from_raiden() -> None:
    """End the ``prefill_transfer_kv_cache`` stage at raiden's transfer finish
    time instead of the (later) scheduler poll time.

    Other backends, and raiden sends without a finish time, keep the poll time.
    """
    global _prefill_transfer_finish_from_raiden_patched
    if _prefill_transfer_finish_from_raiden_patched:
        return
    _prefill_transfer_finish_from_raiden_patched = True

    from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
    from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats
    from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender

    # Each polled request's time stats -> its sender, during the poll only.
    polled_senders = {}
    orig_poll = (
        SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue
    )
    orig_set_finish = SchedulerReqTimeStats.set_prefill_kv_transfer_finish_time

    def process_disagg_prefill_inflight_queue(self, *args, **kwargs):
        for req in self.disagg_prefill_inflight_queue:
            polled_senders[id(req.time_stats)] = req.disagg_kv_sender
        try:
            return orig_poll(self, *args, **kwargs)
        finally:
            polled_senders.clear()

    def set_prefill_kv_transfer_finish_time(self, ts=None):
        sender = polled_senders.get(id(self))
        if ts is None and isinstance(sender, RaidenKVSender):
            finish = sender.transfer_finish_time
            if finish is not None:
                # Never before the stage start: a fast transfer can finish first.
                ts = max(finish, self.prefill_transfer_queue_entry_time)
        orig_set_finish(self, ts)

    SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue = (
        process_disagg_prefill_inflight_queue
    )
    SchedulerReqTimeStats.set_prefill_kv_transfer_finish_time = (
        set_prefill_kv_transfer_finish_time
    )


def _patch_pd_decode_swa_hicache() -> None:
    """Allow PD decode prefix cache + RAM offload on sliding-window models.

    Only full-attention KV goes to RAM. Also hold PD decode requests that reuse
    an unfinished raiden restore, on any model. See ``pd_decode_swa_hicache``.
    """
    from sglang_tpu import pd_decode_swa_hicache

    pd_decode_swa_hicache.install()


def _patch_hicache_write_back_copy_ahead() -> None:
    """Copy the next eviction victims to host ahead of raiden write_back evictions.

    See ``hicache_write_back_copy_ahead``.
    """
    from sglang_tpu import hicache_write_back_copy_ahead

    hicache_write_back_copy_ahead.install()


def _patch_pd_decode_async_prefill_info() -> None:
    """Fetch PD prefill parallel info off the decode scheduler thread.

    See ``pd_decode_async_prefill_info``.
    """
    from sglang_tpu import pd_decode_async_prefill_info

    pd_decode_async_prefill_info.install()


def _patch_host_heap_trim() -> None:
    """Return freed host heap pages once the scheduler is warmed up.

    See ``host_heap_trim``.
    """
    from sglang_tpu import host_heap_trim

    host_heap_trim.install()


def _patch_profile_auto() -> None:
    """Arm a staged xprof capture at launch under SGLANG_TPU_ENABLE_PROFILE_AUTO.

    See ``profile_auto``.
    """
    from sglang_tpu import profile_auto

    profile_auto.install()


def _patch_scheduler_reject_prompt_logprob() -> None:
    """Finish prompt-logprob requests the compiled forward cannot serve with a
    400 at scheduler intake, instead of raising inside the forward.

    Same shape as the scheduler's MLX check: a logprob_start_len inside the
    prompt asks for input logprobs. The answer comes from the shim, so intake
    and forward cannot disagree.
    """
    from sglang.srt.managers.scheduler import Scheduler
    from sglang_tpu.compilation.compiled_forward_shim import CompiledForwardShim

    original = Scheduler._add_request_to_queue
    if getattr(original, "_tpu_prompt_logprob_gate", None) is True:
        return

    def _add_request_to_queue(self, req, is_retracted: bool = False):
        if (
            not is_retracted
            and req.to_finish is None
            and req.return_logprob
            # -1 means output logprobs only; any other start below the prompt
            # length asks for prompt logprobs.
            and req.logprob_start_len != -1
            and req.logprob_start_len < len(req.origin_input_ids)
        ):
            rejection = CompiledForwardShim._prompt_logprob_rejection(
                envs.SGLANG_TPU_ENABLE_PROMPT_LOGPROB.get()
            )
            if rejection is not None:
                req.set_finish_with_abort(rejection)
        return original(self, req, is_retracted)

    _add_request_to_queue._tpu_prompt_logprob_gate = True
    Scheduler._add_request_to_queue = _add_request_to_queue


def apply_tpu_patches() -> None:
    """Install process-local TorchTPU runtime patches.

    Distributed environment preparation is a separate launch step and must
    complete before this function imports JAX- or TorchTPU-backed modules.
    """
    _configure_eager_mode()
    _register_tpu_compile_backend()
    _register_tpu_sampler_backend()
    _unwrap_dynamic_compile_fns()
    _patch_set_device_tolerate_mismatch()
    _patch_compute_position_torch()
    _patch_torch_native_attn_backend()
    _patch_support_triton_for_tpu()
    _register_aten_fallbacks()
    _patch_tpu_torch_pallas_register_positional()
    _patch_future_map_direct_handoff()
    _patch_raiden_is_load_back_event_done()
    _patch_prefill_transfer_finish_from_raiden()
    _patch_pd_decode_swa_hicache()
    _patch_hicache_write_back_copy_ahead()
    _patch_pd_decode_async_prefill_info()
    _patch_host_heap_trim()
    _patch_profile_auto()
    _patch_scheduler_reject_prompt_logprob()
