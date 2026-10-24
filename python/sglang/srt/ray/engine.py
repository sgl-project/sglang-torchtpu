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

import copy
import dataclasses
import logging
import threading
from typing import Callable

import ray
from ray.util.placement_group import (
    PlacementGroup,
)
from ray.util.placement_group import placement_group as create_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.entrypoints.engine import (
    Engine,
    SchedulerInitResult,
    _calculate_rank_ranges,
)
from sglang.srt.environ import envs
from sglang.srt.ray.scheduler_actor import SchedulerActor
from sglang.srt.runtime_context import (
    get_context,
    get_device,
    get_parallel,
)
from sglang.srt.server_args import PortArgs, ServerArgs, compute_world_size

logger = logging.getLogger(__name__)


def _accelerator_resource_name() -> str:
    """Ray resource name for the server's accelerator.

    Ray auto-detects NVIDIA GPUs as the built-in ``GPU`` resource, but a TPU
    cluster advertises ``TPU`` instead (and reports 0 GPU). The placement group
    and scheduler actors must request the resource that actually exists on the
    nodes -- otherwise the bundles are never schedulable and the Engine hangs
    forever at ``pg.ready()``.

    The device is resolved config, so it comes from the ``device`` bag: the
    driver lays the actors out after ``_launch_subprocesses`` published.
    """
    return "TPU" if get_device().device == "tpu" else "GPU"


def _accelerator_actor_options(count: int = 1) -> dict:
    """Ray ``.options()`` kwargs reserving ``count`` accelerators per actor.

    Ray only special-cases ``num_gpus`` (which also drives CUDA_VISIBLE_DEVICES);
    there is no ``num_tpus`` -- TPUs are a custom resource, so they go through the
    generic ``resources`` dict. This keeps one scheduler actor pinned per chip
    without touching the GPU machinery.
    """
    if get_device().device == "tpu":
        return {"num_gpus": 0, "resources": {"TPU": count}}
    return {"num_gpus": count}


def _actor_options(server_args: ServerArgs) -> dict:
    """Ray ``.options()`` kwargs shared by every scheduler actor: CPU +
    accelerator resource reservation (GPU or TPU) and, on TPU, the env injected
    via ``runtime_env``.

    Composes the device-agnostic resource reservation with the TPU-only
    ``runtime_env`` injection (empty for GPU) so both actor launch sites build
    identical options from one place. The TPU env vars are read from the
    launcher-computed fields on ``server_args`` (see ``tpu_actor_runtime_env``).
    The ``hardware_backend.tpu`` import is lazy and gated on ``device == "tpu"``
    so the GPU Ray path never imports the TPU package (keeps GPU launch immune to
    heavy/optional TPU imports; byte-identical GPU options either way).
    """
    options = {"num_cpus": 0}
    options.update(_accelerator_actor_options())
    if get_device().device == "tpu":
        from sglang_tpu.ray_utils import tpu_actor_runtime_env

        options.update(tpu_actor_runtime_env(server_args))
    return options


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


def _get_bundle_node_ip(placement_group: PlacementGroup, bundle_idx: int) -> str:
    """Get the IP address of the node where a specific bundle is located.

    Args:
        placement_group: The placement group
        bundle_idx: Bundle index to query

    Returns:
        IP address of the node where the bundle is located.
    """

    @ray.remote(num_cpus=0, num_gpus=0)
    def get_node_ip():
        return ray.util.get_node_ip_address()

    return ray.get(
        get_node_ip.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=bundle_idx,
            ),
        ).remote()
    )


def _compute_world_size() -> int:
    """Compute world_size (total number of scheduler actors/GPUs needed).

    Reads the published parallel leaves: the driver is sizing the actors that
    will hold the process groups, so there is nothing live to ask.
    """
    return compute_world_size(
        dp_size=get_parallel().dp_size,
        tp_size=get_parallel().tp_size,
        pp_size=get_parallel().pp_size,
    )


def _resolve_bundle_indices(pg: PlacementGroup, world_size: int) -> list[int]:
    """Resolve bundle indices for Custom PG mode.

    Parses SGLANG_RAY_BUNDLE_INDICES env var if set; otherwise returns
    sequential indices [0, 1, ..., world_size-1].

    Args:
        pg: Placement group (used to get total_bundles count).
        world_size: Number of bundle indices expected (pre-computed via _compute_world_size).

    Returns:
        List of bundle indices of length world_size.
    """
    total_bundles = len(pg.bundle_specs)
    indices_str = envs.SGLANG_RAY_BUNDLE_INDICES.get()
    if not indices_str:
        return list(range(world_size))

    indices = list(map(int, indices_str.split(",")))

    if len(indices) != world_size:
        raise ValueError(
            f"SGLANG_RAY_BUNDLE_INDICES has {len(indices)} values, "
            f"expected {world_size}"
        )

    if len(set(indices)) != len(indices):
        raise ValueError(f"SGLANG_RAY_BUNDLE_INDICES has duplicates: {indices}")

    for idx in indices:
        if idx < 0 or idx >= total_bundles:
            raise ValueError(f"Bundle index {idx} out of range [0, {total_bundles})")

    return indices


def _validate_custom_placement_group(pg: PlacementGroup, world_size: int) -> None:
    """Validate custom placement group: 1 GPU per bundle, enough GPU bundles for world_size.

    Args:
        pg: User-provided placement group.
        world_size: Number of GPU bundles required.
    """
    bundles = pg.bundle_specs
    gpu_bundle_count = 0
    for bundle in bundles:
        gpu_count = bundle.get("GPU", 0) + bundle.get("TPU", 0)
        if gpu_count > 1:
            raise ValueError(
                "Custom placement group must have exactly 1 GPU/TPU per bundle. "
                f"Found bundle with {gpu_count} accelerators."
            )
        if gpu_count > 0:
            gpu_bundle_count += 1

    if gpu_bundle_count < world_size:
        raise ValueError(
            f"Custom placement group has {gpu_bundle_count} GPU/TPU bundles, "
            f"but needs {world_size} for world_size. "
            "Provide more bundles or reduce parallelism."
        )


def _create_scheduler_actor(
    pg: PlacementGroup,
    bundle_idx: int,
    gpu_id: int,
    server_args: ServerArgs,
    port_args: PortArgs,
    tp_rank: int,
    pp_rank: int,
    dp_rank: int,
    dist_init_addr: str,
    rank0_node_ip: str,
) -> SchedulerActor:
    """Create a SchedulerActor on the given placement group bundle.

    Args:
        pg: Placement group to schedule actor onto.
        bundle_idx: Bundle index within the placement group.
        gpu_id: GPU ID within the bundle (0 for custom PG, computed for auto PG).
        rank0_node_ip: IP of rank-0's node, used for NCCL rendezvous.
        dist_init_addr: Distributed init address (tcp://rank0_node_ip:nccl_port).
    """
    return SchedulerActor.options(
        **_actor_options(server_args),
        name=(
            f"sglang_scheduler_node{rank0_node_ip}"
            f"_dp{dp_rank}_pp{pp_rank}_tp{tp_rank}"
            f"_pg{pg.id.hex()[:8]}_bundle{bundle_idx}"
        ),
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=bundle_idx,
        ),
    ).remote(
        server_args=server_args,
        port_args=port_args,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        dp_rank=dp_rank,
        dist_init_addr=dist_init_addr,
    )


def _actor_server_args(server_args: ServerArgs) -> ServerArgs:
    """Return a shallow copy of ``server_args`` with its own declaration stash.

    ``Engine._launch_subprocesses`` publishes ``server_args`` in the driver
    process before calling ``_launch_scheduler_processes``. ``declare_resolution``
    refuses the published instance (whose bags are already projected), whereas
    the Ray launcher needs to declare cluster-discovered fields (``nnodes``,
    TPU slice env, ``dist_init_addr``) on the record handed to scheduler
    actors, ``tpu_actor_runtime_env``, and ``PortArgs.init_new``.
    """
    actor_args = copy.copy(server_args)
    stash = getattr(server_args, "_resolved_overrides", None)
    if stash is not None:
        actor_args._resolved_overrides = list(stash)
    return actor_args


class RayEngine(Engine):
    """Engine using Ray actors for scheduler processes."""

    def __init__(self, **kwargs):
        # Set before super().__init__(): it launches the subprocesses, which need
        # the group to schedule the scheduler actors onto.
        self._placement_group = kwargs.pop("placement_group", None)
        if "log_level" not in kwargs:
            kwargs["log_level"] = "error"
        # Schedulers are separate Ray actors; default to the Ray-backed
        # collectors so enable_metrics reaches Ray's Prometheus endpoint.
        if kwargs.get("enable_metrics") and kwargs.get("stat_loggers") is None:
            from sglang.srt.observability.ray_wrappers import build_ray_stat_loggers

            kwargs["stat_loggers"] = build_ray_stat_loggers()
        super().__init__(server_args=ServerArgs(**kwargs))

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
        *,
        placement_group=None,
    ) -> tuple[SchedulerInitResult, None]:
        """Launch schedulers as Ray actors.

        Returns:
            Tuple of (RaySchedulerInitResult, None).
            scheduler_procs is None since Ray uses actors instead of mp.Process.
        """
        # nnodes can still move below (TPU auto-detect). Copy server_args so
        # cluster-discovered declarations travel to actors without mutating the
        # driver's already-published ServerArgs instance.
        server_args = _actor_server_args(server_args)
        nnodes = get_parallel().nnodes

        # TPU + Ray: honor the configured nnodes, discovering the pre-ordered host
        # list from the live cluster. nnodes=0 opts into auto-detecting the count.
        # The computed TPU env is stashed on server_args (tpu_worker_hostnames /
        # torch_tpu_slicebuilder_addresses / torch_tpu_topology) and injected into
        # every scheduler actor via runtime_env -- the only channel Ray propagates,
        # since it does NOT forward the driver's os.environ. runtime_env also carries
        # a curated allowlist of the driver's launcher-set config env that the actor
        # would otherwise lose (see ray_utils._forwarded_driver_env).
        if get_device().use_ray and get_device().device == "tpu":
            # Lazy TPU import: keeps the GPU Ray path from importing the TPU
            # package (see _actor_options).
            from sglang_tpu.ray_utils import (
                _forwarded_driver_env,
                detect_tpu_nnodes,
                discover_tpu_worker_hostnames,
                redact_env_for_log,
            )

            if not ray.is_initialized():
                ray.init(address="auto", ignore_reinit_error=True)

            # Log (once, here on the driver -- not per actor) the launcher-set config
            # env we forward to every scheduler actor via runtime_env, since Ray
            # would otherwise drop it (actors inherit only the raylet env). Values
            # included so the full LIBTPU_INIT_ARGS / flags are visible in one line.
            forwarded_env = _forwarded_driver_env()
            if forwarded_env:
                logger.info(
                    "Ray TPU: forwarding %d driver env var(s) to scheduler actors "
                    "via runtime_env: %s",
                    len(forwarded_env),
                    redact_env_for_log(forwarded_env),
                )

            # Detect nnodes if not specified. This is needed for TPU multihost.
            if nnodes == 0:
                detected = detect_tpu_nnodes()
                nnodes = max(detected, 1)
                logger.info(
                    "Ray discovery: nnodes=0 (auto); detected %d TPU node(s), "
                    "using nnodes=%d",
                    detected,
                    nnodes,
                )
                declare_resolution(
                    server_args, "ray.tpu_nnodes_autodetect", nnodes=nnodes
                )
                get_context().override("ray.tpu_nnodes_autodetect", nnodes=nnodes)
                # Run the (tp_size * pp_size) % nnodes check that
                # check_server_args deferred for nnodes=0. A non-divisor
                # under-provisions the placement group and the launch hangs
                # forever -- fail loud instead.
                ranks = get_parallel().tp_size * get_parallel().pp_size
                if ranks % nnodes != 0:
                    raise ValueError(
                        f"Auto-detected {nnodes} TPU node(s), but "
                        f"tp_size * pp_size = {ranks} is not "
                        f"divisible by {nnodes}. The placement group would be "
                        "under-provisioned and the launch would hang. Pass "
                        "--nnodes explicitly to match your TPU slice topology."
                    )

            # Single vs multi host stay separate (not merged) because the
            # slicebuilder is built differently. Single-host: nnodes==1 forces a
            # STRICT_PACK 1-bundle placement group, so all tp actors -- and the
            # co-located Engine (enforced by _find_engine_bundle) -- live on ONE
            # node; the endpoints are random free localhost ports picked ONCE on
            # the driver and injected so every same-node actor agrees on the port
            # list (Ray doesn't auto-propagate the driver's os.environ, and "localhost"
            # resolves to each actor's own loopback -- valid precisely because
            # they share the node). Multi-host: endpoints are deterministic
            # (host:base_port+i), so the driver only stashes the physically-
            # ordered host list and each actor rebuilds them itself.
            if nnodes <= 1:
                from sglang_tpu.patches import (
                    compute_single_host_tpu_env,
                )

                single_host_env = compute_single_host_tpu_env(
                    get_parallel().tp_size * get_parallel().pp_size
                )
                declare_resolution(
                    server_args,
                    "ray.tpu_single_host_env",
                    # Force TPU_WORKER_HOSTNAMES="" into every actor. On a
                    # multi-host GKE slice each pod inherits the slice-wide
                    # value from `ray start`; without overriding it, the actor's
                    # worker environment preparation would see
                    # len(hosts) > 1, take the multihost branch, discard the
                    # injected single-host slicebuilder, and hang at rendezvous.
                    tpu_worker_hostnames="",
                    torch_tpu_slicebuilder_addresses=single_host_env[
                        "TORCH_TPU_SLICEBUILDER_ADDRESSES"
                    ],
                    torch_tpu_topology=single_host_env.get("TORCH_TPU_TOPOLOGY"),
                )
            else:
                tpu_worker_hostnames = discover_tpu_worker_hostnames(nnodes)
                if tpu_worker_hostnames:
                    # Multihost: stash the physically-ordered (TPU_WORKER_ID) host
                    # list. Actors rebuild the slicebuilder deterministically from it
                    # (see _prepare_tpu_env), so the port list is identical everywhere.
                    declare_resolution(
                        server_args,
                        "ray.tpu_worker_hostnames",
                        tpu_worker_hostnames=tpu_worker_hostnames,
                    )

        pg = placement_group or ray.util.get_current_placement_group()
        if pg is None:
            parallel = get_parallel()
            if parallel.attn_dp_enabled:
                total_gpus = parallel.tp_size * parallel.pp_size
            else:
                total_gpus = parallel.num_dp_ranks * parallel.tp_size * parallel.pp_size

            gpus_per_node = total_gpus // nnodes
            strategy = "STRICT_PACK" if nnodes == 1 else "SPREAD"
            accel = _accelerator_resource_name()

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

        is_custom_pg = placement_group is not None
        world_size = _compute_world_size()

        if get_device().use_ray and get_device().device == "tpu" and nnodes > 1:
            # TPU multihost: the slicebuilder endpoint list and MASTER_ADDR are
            # ordered by physical TPU_WORKER_ID, so rank r must land on the host
            # whose TPU_WORKER_ID == r // devices_per_host. Ray bundle indices are
            # NOT guaranteed to follow TPU_WORKER_ID order (they only coincide for
            # nnodes==2 with the head on worker 0), so probe each bundle's physical
            # worker id and order bundle_for_node by it. Also verifies rank0
            # (worker 0) is co-located with the Engine; anything inconsistent
            # raises here instead of hanging later at PJRT bind / rendezvous.
            from sglang_tpu.ray_utils import (
                order_tpu_bundles_by_worker_id,
            )

            bundle_for_node, rank0_node_ip = order_tpu_bundles_by_worker_id(pg, nnodes)
        else:
            # GPU (or single host): rank placement is independent of any physical
            # slice order, so co-locate rank0 with the Engine and keep numeric
            # bundle order for the remaining nodes.
            engine_bundle, engine_ip = _find_engine_bundle(pg, nnodes)
            bundle_for_node = [engine_bundle] + [
                i for i in range(nnodes) if i != engine_bundle
            ]
            rank0_node_ip = engine_ip

        if get_parallel().num_dp_ranks == 1:
            # Launch tensor parallel scheduler actors
            dist_init_addr = f"{rank0_node_ip}:{port_args.nccl_port}"
            logger.info(f"dist_init_addr: {dist_init_addr}")

            scheduler_actors = []

            if not is_custom_pg:
                gpus_per_node = world_size // nnodes
                accel = _accelerator_resource_name()
                logger.info(
                    f"Ray cluster (auto PG): {nnodes} nodes, "
                    f"{gpus_per_node} {accel}(s)/node, world_size={world_size}"
                )

                for node_idx in range(nnodes):
                    bundle_idx = bundle_for_node[node_idx]
                    pp_range, tp_range, pp_per_node, tp_per_node = (
                        _calculate_rank_ranges(node_rank=node_idx)
                    )
                    for pp_rank in pp_range:
                        for tp_rank in tp_range:
                            local_gpu_idx = (pp_rank % pp_per_node) * tp_per_node + (
                                tp_rank % tp_per_node
                            )

                            actor = _create_scheduler_actor(
                                pg=pg,
                                bundle_idx=bundle_idx,
                                gpu_id=local_gpu_idx,
                                server_args=server_args,
                                port_args=port_args,
                                tp_rank=tp_rank,
                                pp_rank=pp_rank,
                                dp_rank=0,
                                dist_init_addr=dist_init_addr,
                                rank0_node_ip=rank0_node_ip,
                            )
                            scheduler_actors.append(actor)

            else:
                try:
                    bundle_indices = _resolve_bundle_indices(pg, world_size)
                except ValueError as e:
                    logger.error(f"Failed to resolve bundle indices: {e}")
                    raise RuntimeError(f"Failed to resolve bundle indices: {e}") from e

                logger.info(
                    f"Ray cluster (custom PG): world_size={world_size}, "
                    f"bundle_indices={bundle_indices}"
                )

                tp_size = get_parallel().tp_size
                for rank in range(world_size):
                    pp_rank = rank // tp_size
                    tp_rank = rank % tp_size
                    bundle_idx = bundle_indices[rank]

                    actor = _create_scheduler_actor(
                        pg=pg,
                        bundle_idx=bundle_idx,
                        gpu_id=0,  # Each bundle has exactly 1 GPU/TPU
                        server_args=server_args,
                        port_args=port_args,
                        tp_rank=tp_rank,
                        pp_rank=pp_rank,
                        dp_rank=0,
                        dist_init_addr=dist_init_addr,
                        rank0_node_ip=rank0_node_ip,
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

            def block_until_scheduler_exits():
                try:
                    ray.get(event_loop_refs)
                except Exception as e:
                    logger.error(f"Ray scheduler actor terminated with error: {e}")

            return (
                RaySchedulerInitResult(
                    scheduler_infos=scheduler_infos,
                    block_until_scheduler_exits=block_until_scheduler_exits,
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
                    is_custom_pg,
                    nnodes=nnodes,
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
        is_custom_pg: bool = False,
        *,
        nnodes: int,
    ) -> RaySchedulerInitResult:
        """Launch DP schedulers; ``nnodes`` comes from the caller (TPU auto-detect
        decides it after publish, so the parallel leaf may still say 0)."""
        from sglang.srt.ray.data_parallel_controller import (
            RayDataParallelController,
        )

        parallel = get_parallel()
        if parallel.attn_dp_enabled:
            # DP attention folds DP into TP — total GPUs = tp_size * pp_size
            total_gpus = parallel.tp_size * parallel.pp_size
        else:
            total_gpus = parallel.num_dp_ranks * parallel.tp_size * parallel.pp_size
        gpus_per_node = total_gpus // nnodes
        logger.info(
            f"Ray DP cluster: {nnodes} nodes, "
            f"{gpus_per_node} {_accelerator_resource_name()}/node, "
            f"dp_size={parallel.dp_size}, "
            f"attn_dp_size={parallel.attn_dp_size}, "
            f"tp_size={parallel.tp_size}, pp_size={parallel.pp_size}"
        )

        # Declared on a copy of the record so `PortArgs.init_new()` can compute
        # TCP addresses (required for the DP attention path) even when the
        # caller handed in the driver's already-published ServerArgs instance.
        dp_server_args = _actor_server_args(server_args)
        declare_resolution(
            dp_server_args,
            "ray.dp_controller",
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
            is_custom_pg,
            nnodes=nnodes,
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
                "startup_time": controller.startup_time,
            }
        ]

        event_loop_refs = controller.event_loop_refs

        def block_until_scheduler_exits():
            try:
                ray.get(event_loop_refs)
            except Exception as e:
                logger.error(f"Ray scheduler actor terminated with error: {e}")

        return RaySchedulerInitResult(
            scheduler_infos=scheduler_infos,
            block_until_scheduler_exits=block_until_scheduler_exits,
            scheduler_actors=controller.scheduler_actors,
        )
