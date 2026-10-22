# SPDX-License-Identifier: Apache-2.0
"""TorchTPU per-worker environment and backend initialization."""

import logging
import os
from typing import Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


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
