# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
import os
from typing import Callable, Optional, Union

import torch
from torch.multiprocessing import reductions

from sglang.srt.environ import envs
from sglang.srt.utils.common import is_musa, is_npu, torch_release

logger = logging.getLogger(__name__)

_is_npu = is_npu()
_is_musa = is_musa()

if _is_npu:
    from torch_npu.multiprocessing import reductions as npu_reductions

    def _rebuild_npu_tensor_modified(*args):
        args = _modify_tuple(args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, npu_verl_to_sglang)
        return npu_reductions._rebuild_npu_tensor_original(*args)

    def npu_verl_to_sglang(device: int):
        assert SGLANG_TP_RANK is not None, (
            "SGLANG_TP_RANK is not registered. Please call register_sgl_tp_rank() first."
        )
        return SGLANG_TP_RANK


SGLANG_TP_RANK = None


def monkey_patch_torch_reductions():
    """Monkey patching before Torch https://github.com/pytorch/pytorch/pull/149248 is fixed"""

    if not _is_npu:
        if hasattr(reductions, "_reduce_tensor_original"):
            return
        reductions._reduce_tensor_original = reductions.reduce_tensor
        reductions._rebuild_cuda_tensor_original = reductions.rebuild_cuda_tensor

        reductions.reduce_tensor = _reduce_tensor_modified
        reductions.rebuild_cuda_tensor = _rebuild_cuda_tensor_modified
        reductions.init_reductions()
    else:
        # FIXME: This is a temp patch for npu as HDK does not support device uuid for now
        if hasattr(npu_reductions, "_rebuild_npu_tensor_original"):
            return

        npu_reductions._rebuild_npu_tensor_original = npu_reductions.rebuild_npu_tensor
        npu_reductions.rebuild_npu_tensor = _rebuild_npu_tensor_modified


# The signature has not been changed for years, and we will not need this when the next version is released,
# so it looks safe to use a constant.
_REDUCE_TENSOR_ARG_DEVICE_INDEX = 6


def register_sgl_tp_rank(rank: int):
    global SGLANG_TP_RANK
    SGLANG_TP_RANK = rank


def _reduce_tensor_modified(*args, **kwargs):
    output_fn, output_args = reductions._reduce_tensor_original(*args, **kwargs)
    output_args = _modify_tuple(
        output_args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_to_uuid
    )
    return output_fn, output_args


def _rebuild_cuda_tensor_modified(*args):
    args = _modify_tuple(args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_from_maybe_uuid)
    return reductions._rebuild_cuda_tensor_original(*args)


def _device_to_uuid(device: int) -> str:
    return str(torch.cuda.get_device_properties(device).uuid)


def _device_from_maybe_uuid(device_maybe_uuid: Union[int, str]) -> int:
    if isinstance(device_maybe_uuid, int):
        return device_maybe_uuid

    if isinstance(device_maybe_uuid, str):
        for device in range(torch.cuda.device_count()):
            if str(torch.cuda.get_device_properties(device).uuid) == device_maybe_uuid:
                return device
        raise Exception("Invalid device_uuid=" + device_maybe_uuid)

    raise Exception(f"Unknown type: {device_maybe_uuid=}")


def _modify_tuple(t, index: int, modifier: Callable):
    return *t[:index], modifier(t[index]), *t[index + 1 :]


def monkey_patch_torch_tpu_env(
    server_args, tp_rank: int, pp_rank: int = 0, dp_rank: Optional[int] = None
):
    """Set TorchTPU's per-worker distributed environment.

    Sets torchrun-style env vars for torch_tpu's PJRT slicebuilder
    (RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT). PJRT lazy-init
    in each worker reads these to coordinate; sglang's mp.Process spawn path
    doesn't otherwise set them.

    The PJRT world is one slice of ``tp_size * pp_size`` devices, one per
    scheduler process, in the same rank order torch.distributed uses
    (``pp_rank * tp_size + tp_rank``). Under pipeline parallelism every stage
    therefore binds its own device, exactly as the ranks of a TP-only run do.

    No-op for non-TPU runs.
    """
    from sglang.srt.arg_groups.model_override_base import resolving_view

    cfg = resolving_view(server_args)
    if cfg.device != "tpu":
        return

    world_size = cfg.tp_size * cfg.pp_size
    rank = pp_rank * cfg.tp_size + tp_rank
    local_world_size = max(world_size // cfg.nnodes, 1)
    local_rank = rank % local_world_size
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)

    # The resolved width: the raw record only carries it for the deprecated
    # `--dp-size N --enable-dp-attention` spelling, not for `--attn-dp-size N`.
    attn_dp_size = cfg.attn_dp_size
    # Replicas times attention-DP groups: what the raw dp_size counted under
    # the deprecated spelling, so the compile-cache scope below keeps its name.
    dp_size = max(cfg.dp_size, 1) * max(attn_dp_size, 1)
    if dp_rank is not None:
        effective_dp_rank = dp_rank
    elif attn_dp_size > 1:
        attn_tp_size = max(1, cfg.tp_size // attn_dp_size)
        effective_dp_rank = tp_rank // attn_tp_size
    else:
        effective_dp_rank = int(os.environ.get("SGLANG_DP_RANK", "0"))

    base_inductor_cache = os.environ.get("SGLANG_BASE_TORCHINDUCTOR_CACHE_DIR")
    if not base_inductor_cache:
        default_cache_dir = (
            f"/dev/shm/torchinductor_{os.environ.get('USER', 'root')}"
            if os.path.isdir("/dev/shm")
            else f"/tmp/torchinductor_{os.environ.get('USER', 'root')}"
        )
        base_inductor_cache = os.environ.get(
            "TORCHINDUCTOR_CACHE_DIR",
            default_cache_dir,
        )
        os.environ["SGLANG_BASE_TORCHINDUCTOR_CACHE_DIR"] = base_inductor_cache
    dp_scope = f"dp{dp_size}_dprank{effective_dp_rank}_tp{server_args.tp_size}"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"{base_inductor_cache}_{dp_scope}"
    os.environ["SGLANG_TPU_CACHE_DP_SCOPE"] = dp_scope

    if envs.SGLANG_TPU_PD_COLOCATED.get():
        assert not cfg.use_ray, "TPU P/D co-location does not support Ray"
        physical_device = cfg.base_gpu_id + local_rank * cfg.gpu_id_step
        # Assign an individual TPU device to each rank.
        os.environ["TPU_VISIBLE_DEVICES"] = str(physical_device)

    # Multihost: every rank must rendezvous at rank-0's node. TPU_WORKER_HOSTNAMES
    # is ordered by TPU_WORKER_ID (physical slice order), so hosts[0] is where
    # RANK 0 lives. Under Ray, force it: the inherited MASTER_ADDR is the driver's
    # loopback, which would leave each host rendezvousing with itself. Off the Ray
    # path a launcher (e.g. torchrun) may have set MASTER_ADDR correctly, so only
    # fill it in when unset (defaulting to hosts[0] for multihost, else loopback).
    hosts = [h for h in os.environ.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    if cfg.use_ray and len(hosts) > 1:
        os.environ["MASTER_ADDR"] = hosts[0]
    else:
        os.environ.setdefault(
            "MASTER_ADDR", hosts[0] if len(hosts) > 1 else "127.0.0.1"
        )
    os.environ.setdefault("MASTER_PORT", "29500")


def initialize_torch_tpu_worker(
    server_args, tp_rank: int, pp_rank: int = 0, dp_rank: Optional[int] = None
):
    """Configure and initialize TorchTPU in a scheduler worker."""
    monkey_patch_torch_tpu_env(server_args, tp_rank, pp_rank, dp_rank=dp_rank)
    from sglang.srt.arg_groups.model_override_base import resolving_view

    cfg = resolving_view(server_args)
    if cfg.device != "tpu":
        return

    from sglang.srt.platforms import current_platform

    current_platform.init_backend()

    try:
        import torch_tpu._internal.compile._backend as _tpu_compile_backend

        _orig_cache_info = getattr(_tpu_compile_backend, "AOTAutogradCacheInfo", None)
        if _orig_cache_info is not None and not getattr(
            _orig_cache_info, "_sglang_dp_scoped", False
        ):

            def _scoped_aot_cache_info(cache_key, *args, **kwargs):
                scope = os.environ.get("SGLANG_TPU_CACHE_DP_SCOPE", "")
                if scope and not str(cache_key).startswith(f"{scope}_"):
                    cache_key = f"{scope}_{cache_key}"
                return _orig_cache_info(cache_key, *args, **kwargs)

            _scoped_aot_cache_info._sglang_dp_scoped = True
            _tpu_compile_backend.AOTAutogradCacheInfo = _scoped_aot_cache_info
    except Exception as e:
        logger.warning(
            f"Failed to scope torch_tpu AOTAutogradCacheInfo by DP rank: {e}"
        )


def monkey_patch_torch_compile():
    if torch_release < (2, 8):
        # These things are cacheable by torch.compile. torch.compile just doesn't know it.
        # This was fixed in PyTorch 2.8, but until then, we monkey patch.
        import torch._higher_order_ops.auto_functionalize as af

        af.auto_functionalized_v2._cacheable = True
        af.auto_functionalized._cacheable = True


def register_fake_if_exists(op_name):
    """
    Decorator factory to conditionally register a fake for a custom op if it exists.
    Parses op_name (e.g., 'sgl_kernel::gptq_gemm'), checks if the op exists via hasattr
    on the namespace attribute of torch.ops. Registers the fake if present; otherwise,
    returns the function unchanged.
    Args:
        op_name (str): Full operator name (e.g., 'sgl_kernel::gptq_gemm').
    Returns:
        callable: Decorator for the fake function.
    Example:
        @register_fake_if_exists('sgl_kernel::gptq_gemm')
        def fake_gptq_gemm(a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, use_shuffle, bit):
            return a.new_empty((a.shape[0], b_q_weight.shape[-1]), dtype=a.dtype)
    """

    def decorator(func):
        namespace, bare_op = op_name.split("::")
        ops_namespace = getattr(torch.ops, namespace, None)
        if ops_namespace and hasattr(ops_namespace, bare_op):
            torch.library.register_fake(op_name, func)
        return func

    return decorator
