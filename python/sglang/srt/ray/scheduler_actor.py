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
"""Ray actor wrapper for SGLang Scheduler."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import ray

from sglang.srt.runtime_context import publish
from sglang.srt.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)


@ray.remote
class SchedulerActor:
    """Ray actor wrapper for SGLang Scheduler.

    Each actor manages one GPU and runs the Scheduler + TpModelWorker stack.
    Ray is used for process lifecycle; ZMQ handles request/response communication.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        dist_init_addr: Optional[str] = None,
    ):
        from sglang.srt.environ import envs
        from sglang.srt.utils.numa_utils import (
            get_numa_node_if_available,
            numa_bind_to_node,
        )
        from sglang.srt.utils.patch_torch import initialize_torch_tpu_worker

        # Override dist_init_addr if provided (for multi-node), through
        # `replace_resolved` so the copy keeps the parent's resolution.
        if dist_init_addr:
            server_args = server_args.replace_resolved(
                "ray.scheduler_actor", dist_init_addr=dist_init_addr
            )

        # The Ray-discovered, physically-ordered TPU host list is injected into
        # this actor's environment as TPU_WORKER_HOSTNAMES via runtime_env at
        # launch (see ray_utils.tpu_actor_runtime_env), so worker bootstrap reads
        # it straight from os.environ.

        # TPU: replay run_scheduler_process's per-process env bootstrap (RANK/
        # WORLD_SIZE/MASTER_* for torch_tpu's PJRT slicebuilder), then initialize
        # the backend before importing Scheduler. Ray actors skip the mp entry
        # point that normally performs this work. No-op for non-TPU.
        initialize_torch_tpu_worker(server_args, tp_rank)

        from sglang.srt.managers.scheduler import (
            Scheduler,
            configure_scheduler_process,
        )

        # Get actual GPU IDs from Ray runtime context
        accelerator_ids = ray.get_runtime_context().get_accelerator_ids()
        assigned_gpus = accelerator_ids.get("GPU", [])

        if assigned_gpus:
            # Ray assigned specific GPU(s), use the first one
            actual_gpu_id = int(assigned_gpus[0])
            logger.info(f"[TP{tp_rank}] Ray assigned GPU: {actual_gpu_id}")
        else:
            # Fallback to passed gpu_id
            actual_gpu_id = gpu_id
            logger.info(f"[TP{tp_rank}] Using passed gpu_id: {gpu_id}")

        # This actor takes the place of run_scheduler_process, which is where
        # a forked scheduler publishes.
        publish(server_args, role="scheduler")

        # Configure worker (logging, process title, etc.)
        dp_rank = configure_scheduler_process(
            server_args,
            actual_gpu_id,
            tp_rank,
            attn_cp_rank,
            moe_dp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
        )

        # Ray actors can't use the numactl subprocess-wrapping approach
        # (SGLANG_NUMA_BIND_V2's normal path), so bind in-process via libnuma.
        # The V1 path inside configure_scheduler_process already handles
        # SGLANG_NUMA_BIND_V2=False.
        if envs.SGLANG_NUMA_BIND_V2.get():
            numa_node = get_numa_node_if_available(server_args, actual_gpu_id)
            if numa_node is not None:
                numa_bind_to_node(numa_node)
                logger.info(
                    f"[TP{tp_rank}] Bound to NUMA node {numa_node} for GPU {actual_gpu_id}"
                )

        # Create scheduler (loads model into GPU, initializes NCCL)
        self.scheduler = Scheduler(
            server_args=server_args,
            port_args=port_args,
            gpu_id=actual_gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=pp_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            dp_rank=dp_rank,
        )

        self._tp_rank = tp_rank
        self._pp_rank = pp_rank

    def get_info(self) -> Dict[str, Any]:
        """Return scheduler initialization info for handshake."""
        return self.scheduler.get_init_info()

    def run_event_loop(self) -> None:
        """Run the scheduler's event loop. Blocks until shutdown."""
        try:
            import torch

            # Need to set the device for the event loop for collective communication to work
            device_module = torch.get_device_module(self.scheduler.device)
            device_module.set_device(self.scheduler.ps.gpu_id)
            self.scheduler.run_event_loop()
        except Exception as e:
            logger.error(f"Scheduler PP{self._pp_rank} TP{self._tp_rank} crashed: {e}")
            raise
