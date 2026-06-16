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
from typing import Callable, Union

import torch
from torch.multiprocessing import reductions

from sglang.srt.utils.common import is_musa, is_npu, torch_release

_is_npu = is_npu()
_is_musa = is_musa()

if _is_npu:
    from torch_npu.multiprocessing import reductions as npu_reductions

    def _rebuild_npu_tensor_modified(*args):
        args = _modify_tuple(args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, npu_verl_to_sglang)
        return npu_reductions._rebuild_npu_tensor_original(*args)

    def npu_verl_to_sglang(device: int):
        assert (
            SGLANG_TP_RANK is not None
        ), "SGLANG_TP_RANK is not registered. Please call register_sgl_tp_rank() first."
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


def monkey_patch_torch_tpu_env(server_args, tp_rank: int):
    """TPU-only per-child-process startup work, called from
    ``run_scheduler_process`` before ``Scheduler`` is constructed.

    Two responsibilities:

    1. Set torchrun-style env vars for torch_tpu's PJRT slicebuilder
       (RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT). PJRT
       lazy-init in each worker reads these to coordinate; sglang's
       mp.Process spawn path doesn't otherwise set them.

    2. Re-arm the TPU pool swap in the child. ``apply_tpu_patches`` runs at
       model_runner module-import time — before the child has any view of
       ``server_args`` — so the swap of ``MHATokenToKVPool`` →
       ``TpuFusedMHATokenToKVPool`` is silently skipped on the child side.
       Replaying it here (before ``ModelRunner._init_pools`` runs) puts
       the fused pool in place for the ``pallas_tpu_rpa_v3`` backend.

    No-op for non-TPU runs.
    """
    if server_args.device != "tpu":
        return

    import os

    # sglang's tp_size is the GLOBAL tensor-parallel world (spanning every node)
    # and tp_rank is ALREADY a global rank — get_tp_rank_range() hands node N the
    # range [N*tp_size/nnodes, (N+1)*tp_size/nnodes). torch_tpu's PJRT
    # slicebuilder indexes TORCH_TPU_SLICEBUILDER_ADDRESSES by this global RANK,
    # so RANK must be tp_rank verbatim. Adding node_rank*tp_size (which only
    # makes sense if tp_rank were node-local) overshoots the address list on
    # node_rank>0 — e.g. "RANK 25 is out of bounds ... (size 16)". Likewise the
    # world is tp_size, not tp_size*nnodes. LOCAL_RANK/LOCAL_WORLD_SIZE are the
    # per-host slice (tp_size//nnodes for the pp_size==1 serving case). Node 0
    # happened to work before only because its node_rank*tp_size offset is 0.
    world_size = server_args.tp_size
    local_world_size = max(world_size // server_args.nnodes, 1)
    os.environ["RANK"] = str(tp_rank)
    os.environ["LOCAL_RANK"] = str(tp_rank % local_world_size)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    from sglang.srt.hardware_backend.tpu.patches import (
        _patch_mha_kv_pool_for_rpa_v3,
        set_active_attention_backend,
    )

    set_active_attention_backend(server_args.attention_backend)
    _patch_mha_kv_pool_for_rpa_v3()


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
