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
"""RayEngine - Engine subclass that launches schedulers as Ray actors."""

from __future__ import annotations

import dataclasses
import logging
import threading
from typing import Callable, Optional

import ray
from ray.util.placement_group import (
    PlacementGroup,
    placement_group as create_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from sglang.srt.entrypoints.engine import (
    Engine,
    SchedulerInitResult,
    _calculate_rank_ranges,
    _compute_parallelism_ranks,
)
from sglang.srt.hardware_backend.tpu.ray_utils import (
    detect_tpu_nnodes,
    discover_tpu_worker_hostnames,
    tpu_actor_runtime_env,
)
from sglang.srt.ray.scheduler_actor import SchedulerActor
from sglang.srt.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)


def _accelerator_resource_name(server_args: ServerArgs) -> str:
    """Ray resource name for the server's accelerator.

    Ray auto-detects NVIDIA GPUs as the built-in ``GPU`` resource, but a TPU
    cluster advertises ``TPU`` instead (and reports 0 GPU). The placement group
    and scheduler actors must request the resource that actually exists on the
    nodes -- otherwise the bundles are never schedulable and the Engine hangs
    forever at ``pg.ready()``.
    """
    return "TPU" if server_args.device == "tpu" else "GPU"


def _accelerator_actor_options(server_args: ServerArgs, count: int = 1) -> dict:
    """Ray ``.options()`` kwargs reserving ``count`` accelerators per actor.

    Ray only special-cases ``num_gpus`` (which also drives CUDA_VISIBLE_DEVICES);
    there is no ``num_tpus`` -- TPUs are a custom resource, so they go through the
    generic ``resources`` dict. This keeps one scheduler actor pinned per chip
    without touching the GPU machinery.
    """
    if server_args.device == "tpu":
        return {"num_gpus": 0, "resources": {"TPU": count}}
    return {"num_gpus": count}


@dataclasses.dataclass
class RaySchedulerInitResult(SchedulerInitResult):
    """SchedulerInitResult that also holds Ray actor handles for cleanup."""

    scheduler_actors: list = dataclasses.field(default_factory=list)


def _find_engine_bundle(
    placement_group: PlacementGroup, nnodes: int
) -> tuple[int, str]:
    """Find which placement group bundle is on the same node as the Engine.
    Rank0 scheduler must be co-located with the Engine. Returns (bundle_index, engine_ip).
    """
    engine_ip = ray.util.get_node_ip_address()

    @ray.remote(num_cpus=0, num_gpus=0)
    def get_node_ip():
        return ray.util.get_node_ip_address()

    bundle_ips = ray.get(
        [
            get_node_ip.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_bundle_index=i,
                ),
            ).remote()
            for i in range(nnodes)
        ]
    )

    try:
        return bundle_ips.index(engine_ip), engine_ip
    except ValueError:
        raise RuntimeError(
            f"Engine node {engine_ip} not found in any placement group bundle {bundle_ips}. "
            f"Rank-0 scheduler must be co-located with the Engine."
        )


class RayEngine(Engine):
    """Engine using Ray actors for scheduler processes."""

    def shutdown(self):
        """Shutdown the engine — kill Ray scheduler actors then local processes."""
        for actor in self._scheduler_init_result.scheduler_actors:
            try:
                ray.kill(actor)
            except Exception:
                logger.error(f"Failed to kill Ray scheduler actor: {actor}")
        super().shutdown()

    @classmethod
    def _launch_scheduler_processes(
        cls,
        server_args: ServerArgs,
        port_args: PortArgs,
        run_scheduler_process_func: Callable,
    ) -> tuple[SchedulerInitResult, None]:
        """Launch schedulers as Ray actors.

        Returns:
            Tuple of (RaySchedulerInitResult, None).
            scheduler_procs is None since Ray uses actors instead of mp.Process.
        """
        # TPU + Ray: honor server_args.nnodes, discovering the pre-ordered host
        # list from the live cluster. nnodes=0 opts into auto-detecting the count.
        tpu_worker_hostnames = None
        if server_args.use_ray and server_args.device == "tpu":
            # detect_tpu_nnodes() below calls ray.nodes() -- the Ray *state API*,
            # which (unlike the placement-group / actor APIs) does NOT auto-connect
            # via RAY_ADDRESS. Connect explicitly first so auto-detection sees the
            # live cluster; otherwise it fails with "Ray has not been started yet",
            # falls back to nnodes=1, and builds a single tp_size-wide bundle that
            # no host can satisfy (STRICT_PACK), hanging forever at pg.ready().
            if not ray.is_initialized():
                ray.init(address="auto", ignore_reinit_error=True)
            # server_args.nnodes is the source of truth: any explicit value is
            # trusted as-is, including nnodes=1 for a single host on a multi-node
            # Ray cluster. nnodes=0 is the opt-in "auto-detect" sentinel -- only
            # then do we derive the count from the live cluster.
            if server_args.nnodes == 0:
                detected = detect_tpu_nnodes()
                resolved = max(detected, 1)
                logger.info(
                    "Ray discovery: nnodes=0 (auto); detected %d TPU node(s), "
                    "using nnodes=%d",
                    detected,
                    resolved,
                )
                server_args = dataclasses.replace(server_args, nnodes=resolved)
            tpu_worker_hostnames = discover_tpu_worker_hostnames(server_args.nnodes)

            if server_args.nnodes <= 1:
                # Single-host Ray: no multihost host list to rebuild from, so
                # seed the (localhost) slicebuilder env once here in the driver.
                # Same-node actors inherit this identical port list via Ray's
                # env propagation, keeping their PJRT slicebuilder consistent.
                # Multihost actors rebuild deterministically from the discovered
                # node IPs (see _prepare_tpu_env), so seeding is skipped there --
                # keeping the orchestrator's logs free of the localhost line.
                from sglang.srt.hardware_backend.tpu.patches import (
                    _prepare_tpu_env,
                )

                _prepare_tpu_env(server_args.tp_size)

        pg = ray.util.get_current_placement_group()
        if pg is None:
            if server_args.enable_dp_attention:
                total_gpus = server_args.tp_size * server_args.pp_size
            else:
                total_gpus = (
                    server_args.dp_size * server_args.tp_size * server_args.pp_size
                )

            nnodes = server_args.nnodes
            gpus_per_node = total_gpus // nnodes
            strategy = "STRICT_PACK" if nnodes == 1 else "SPREAD"
            accel = _accelerator_resource_name(server_args)

            logger.info(
                "No placement group detected. Auto-creating one with "
                f"{nnodes} bundle(s), {gpus_per_node} {accel}(s)/bundle, "
                "placement group explicitly and schedule the Engine onto it."
            )

            pg = create_placement_group(
                [{"CPU": 1, accel: gpus_per_node}] * nnodes,
                strategy=strategy,
            )
            ray.get(pg.ready())

        nnodes = server_args.nnodes

        # co-located with the Engine and rank0 scheduler at the same node
        engine_bundle, engine_ip = _find_engine_bundle(pg, nnodes)
        bundle_for_node = [engine_bundle] + [
            i for i in range(nnodes) if i != engine_bundle
        ]
        rank0_node_ip = engine_ip

        if server_args.dp_size == 1:
            # Launch tensor parallel scheduler actors
            world_size = server_args.tp_size * server_args.pp_size
            gpus_per_node = world_size // nnodes

            logger.info(
                f"Ray cluster: {nnodes} nodes, "
                f"Use {gpus_per_node} {_accelerator_resource_name(server_args)}/node, "
                f"world_size={world_size}"
            )

            dist_init_addr = f"{rank0_node_ip}:{port_args.nccl_port}"
            logger.info(f"dist_init_addr: {dist_init_addr}")

            scheduler_actors = []

            for node_idx in range(nnodes):
                bundle_idx = bundle_for_node[node_idx]
                pp_range, tp_range, pp_per_node, tp_per_node = _calculate_rank_ranges(
                    nnodes,
                    server_args.pp_size,
                    server_args.tp_size,
                    node_rank=node_idx,
                )
                for pp_rank in pp_range:
                    for tp_rank in tp_range:
                        local_gpu_idx = (pp_rank % pp_per_node) * tp_per_node + (
                            tp_rank % tp_per_node
                        )

                        attn_cp_rank, moe_dp_rank, moe_ep_rank = (
                            _compute_parallelism_ranks(server_args, tp_rank)
                        )

                        actor = SchedulerActor.options(
                            num_cpus=0,
                            **_accelerator_actor_options(server_args),
                            **tpu_actor_runtime_env(tpu_worker_hostnames),
                            name=f"sglang_scheduler_node{rank0_node_ip}_pp{pp_rank}_tp{tp_rank}_pg{pg.id.hex()[:8]}_bundle{bundle_idx}",
                            scheduling_strategy=PlacementGroupSchedulingStrategy(
                                placement_group=pg,
                                placement_group_bundle_index=bundle_idx,
                            ),
                        ).remote(
                            server_args=server_args,
                            port_args=port_args,
                            gpu_id=local_gpu_idx,
                            tp_rank=tp_rank,
                            attn_cp_rank=attn_cp_rank,
                            moe_dp_rank=moe_dp_rank,
                            moe_ep_rank=moe_ep_rank,
                            pp_rank=pp_rank,
                            dp_rank=0,
                            dist_init_addr=dist_init_addr,
                        )
                        scheduler_actors.append(actor)

            try:
                scheduler_infos = ray.get(
                    [actor.get_info.remote() for actor in scheduler_actors]
                )
            except ray.exceptions.RayActorError as e:
                for actor in scheduler_actors:
                    try:
                        ray.kill(actor)
                    except Exception:
                        logger.error(f"Failed to kill Ray scheduler actor: {actor}")
                raise RuntimeError(f"Scheduler actor failed to initialize: {e}")

            event_loop_refs = [
                actor.run_event_loop.remote() for actor in scheduler_actors
            ]

            def wait_for_completion():
                try:
                    ray.get(event_loop_refs)
                except Exception as e:
                    logger.error(f"Ray scheduler actor terminated with error: {e}")

            return (
                RaySchedulerInitResult(
                    scheduler_infos=scheduler_infos,
                    wait_for_completion=wait_for_completion,
                    scheduler_actors=scheduler_actors,
                ),
                None,
            )
        else:
            # Launch the data parallel controller
            return (
                cls._launch_dp_scheduler_processes(
                    server_args,
                    port_args,
                    pg,
                    bundle_for_node,
                    rank0_node_ip,
                    tpu_worker_hostnames,
                ),
                None,
            )

    @classmethod
    def _launch_dp_scheduler_processes(
        cls,
        server_args: ServerArgs,
        port_args: PortArgs,
        pg,
        bundle_for_node: list,
        rank0_node_ip: str,
        tpu_worker_hostnames: Optional[str] = None,
    ) -> RaySchedulerInitResult:
        """Launch DP schedulers via RayDataParallelController."""
        from sglang.srt.ray.data_parallel_controller import (
            RayDataParallelController,
        )

        if server_args.enable_dp_attention:
            # DP attention folds DP into TP — total GPUs = tp_size * pp_size
            total_gpus = server_args.tp_size * server_args.pp_size
        else:
            total_gpus = server_args.dp_size * server_args.tp_size * server_args.pp_size
        gpus_per_node = total_gpus // server_args.nnodes
        logger.info(
            f"Ray DP cluster: {server_args.nnodes} nodes, "
            f"{gpus_per_node} {_accelerator_resource_name(server_args)}/node, "
            f"dp_size={server_args.dp_size}, "
            f"tp_size={server_args.tp_size}, pp_size={server_args.pp_size}, "
            f"enable_dp_attention={server_args.enable_dp_attention}"
        )

        # Set dist_init_addr on server_args so PortArgs.init_new() can compute
        # TCP addresses correctly (required for DP attention path).
        dp_server_args = dataclasses.replace(
            server_args,
            dist_init_addr=f"{rank0_node_ip}:{port_args.nccl_port}",
        )

        # Create the DP controller in-process. This blocks until all actors
        # are initialized and their event loops have started.
        controller = RayDataParallelController(
            dp_server_args,
            port_args,
            pg,
            bundle_for_node,
            rank0_node_ip,
            tpu_worker_hostnames,
        )

        # Start the DP controller's event loop in a daemon thread.
        # It routes requests from the tokenizer to per-DP-rank schedulers.
        dp_thread = threading.Thread(
            target=controller.event_loop, daemon=True, name="dp_controller"
        )
        dp_thread.start()

        scheduler_infos = [
            {
                "max_total_num_tokens": controller.max_total_num_tokens,
                "max_req_input_len": controller.max_req_input_len,
            }
        ]

        event_loop_refs = controller.event_loop_refs

        def wait_for_completion():
            try:
                ray.get(event_loop_refs)
            except Exception as e:
                logger.error(f"Ray scheduler actor terminated with error: {e}")

        return RaySchedulerInitResult(
            scheduler_infos=scheduler_infos,
            wait_for_completion=wait_for_completion,
            scheduler_actors=controller.scheduler_actors,
        )
