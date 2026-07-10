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
"""TPU-specific helpers for the Ray scheduler launch path.

These live in ``hardware_backend/tpu`` (rather than ``ray/engine.py``) so the
generic Ray engine stays free of TPU cluster-discovery specifics.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import ray
from ray.util.placement_group import (
    placement_group as create_placement_group,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

logger = logging.getLogger(__name__)


def detect_tpu_nnodes() -> int:
    """Count alive Ray nodes that advertise the ``TPU`` resource.

    Used to auto-derive ``nnodes`` for TPU ray runs so the operator doesn't have
    to pass ``--nnodes`` -- the placement group and rank math then follow the live
    cluster. Returns 0 if nothing is detectable (caller keeps server_args.nnodes).
    """
    try:
        return sum(
            1
            for node in ray.nodes()
            if node.get("Alive") and float(node.get("Resources", {}).get("TPU", 0)) > 0
        )
    except Exception as e:  # pragma: no cover - defensive; fall back to server_args
        logger.warning("Failed to count TPU nodes from Ray: %s", e)
        return 0


@ray.remote(num_cpus=0, num_gpus=0)
def _get_worker_id_and_address():
    """Ray task: return this node's ``(TPU_WORKER_ID, ip)`` for slice ordering."""
    return os.environ.get("TPU_WORKER_ID"), ray.util.get_node_ip_address()


def discover_tpu_worker_hostnames(nnodes: int) -> Optional[str]:
    """Discover the physically-ordered TPU host list from the live Ray cluster.

    The torch_tpu slicebuilder (``_prepare_tpu_env``) needs hosts ordered by
    physical slice position (``TPU_WORKER_ID``). We always discover this from Ray
    (ignoring any pre-set ``TPU_WORKER_HOSTNAMES`` in the environment) by probing
    every node for its ``TPU_WORKER_ID`` and IP, then sorting by ``TPU_WORKER_ID``
    -- never by the arbitrary ``ray.nodes()`` order. Returns None if it can't be
    discovered, in which case each actor falls back to its own pod env.
    """
    if nnodes <= 1:
        return None

    # One probe per node via a SPREAD placement group so each lands on a distinct
    # host; sort by TPU_WORKER_ID afterwards to recover physical order.
    pg = None
    try:
        pg = create_placement_group([{"CPU": 1}] * nnodes, strategy="SPREAD")
        ray.get(pg.ready())
        probes = ray.get(
            [
                _get_worker_id_and_address.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=i
                    )
                ).remote()
                for i in range(nnodes)
            ]
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Ray discovery: failed to probe TPU worker hostnames: %s", e)
        return None
    finally:
        if pg is not None:
            remove_placement_group(pg)

    if any(worker_id is None for worker_id, _ in probes):
        logger.warning(
            "Ray discovery: TPU_WORKER_ID missing on some nodes; cannot order "
            "hosts. Actors will fall back to their own pod env."
        )
        return None

    ordered = sorted(probes, key=lambda p: int(p[0]))
    hostnames = ",".join(ip for _, ip in ordered)
    logger.info(
        "Ray discovery: TPU_WORKER_HOSTNAMES=%s (ordered by TPU_WORKER_ID)",
        hostnames,
    )
    return hostnames


def tpu_actor_runtime_env(tpu_worker_hostnames: Optional[str]) -> dict:
    """Ray ``.options()`` kwargs injecting the discovered TPU host list into a
    scheduler actor's environment.

    The actor reads ``TPU_WORKER_HOSTNAMES`` from its env during TPU bootstrap
    (``_prepare_tpu_env``); setting it via ``runtime_env`` keeps the generic
    ``SchedulerActor`` constructor free of a TPU-specific parameter. Returns an
    empty dict when there is nothing to inject (single-host or discovery failed),
    so actors fall back to their own pod env.
    """
    if not tpu_worker_hostnames:
        return {}
    return {"runtime_env": {"env_vars": {"TPU_WORKER_HOSTNAMES": tpu_worker_hostnames}}}
