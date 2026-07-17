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
from ray.util.placement_group import placement_group as create_placement_group
from ray.util.placement_group import (
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


def _resolve_bundle_order_from_probes(
    probes: list[tuple[Optional[str], str]], engine_ip: str
) -> tuple[list[int], str]:
    """Turn per-bundle ``(TPU_WORKER_ID, ip)`` probes into a physically-ordered
    ``bundle_for_node`` plus the rank-0 host IP.

    ``probes[bundle_idx]`` is the ``(TPU_WORKER_ID, ip)`` observed on Ray
    placement-group bundle ``bundle_idx``. The returned ``bundle_for_node[k]``
    is the bundle index whose physical ``TPU_WORKER_ID == k``, so scheduler node
    ``k`` lands on physical worker ``k`` -- matching the slicebuilder endpoint
    list and ``MASTER_ADDR`` (both ordered by ``TPU_WORKER_ID``; see
    ``discover_tpu_worker_hostnames`` / ``monkey_patch_torch_tpu_env``). Ray does
    not guarantee bundle indices follow ``TPU_WORKER_ID`` order, so relying on
    numeric bundle order silently misplaces ranks for nnodes >= 3.

    Also verifies rank-0 (``TPU_WORKER_ID == 0``) is co-located with the Engine,
    since ``MASTER_ADDR`` points at worker 0 and the rank-0 scheduler must share
    the Engine's node. Raises ``RuntimeError`` on any inconsistency (missing /
    duplicate / non-contiguous worker ids, or engine not on worker 0) instead of
    letting the run hang later at PJRT bind / rendezvous.

    Pure function (no Ray calls) so the ordering + validation can be unit-tested.
    """
    nnodes = len(probes)

    if any(worker_id is None for worker_id, _ in probes):
        raise RuntimeError(
            "Ray TPU multihost: TPU_WORKER_ID missing on some bundles "
            f"({probes}); cannot order ranks by physical slice position."
        )

    worker_id_to_bundle: dict[int, int] = {}
    worker_id_to_ip: dict[int, str] = {}
    for bundle_idx, (worker_id, ip) in enumerate(probes):
        wid = int(worker_id)
        if wid in worker_id_to_bundle:
            raise RuntimeError(
                f"Ray TPU multihost: duplicate TPU_WORKER_ID={wid} across "
                f"bundles {worker_id_to_bundle[wid]} and {bundle_idx} ({probes})."
            )
        worker_id_to_bundle[wid] = bundle_idx
        worker_id_to_ip[wid] = ip

    if sorted(worker_id_to_bundle) != list(range(nnodes)):
        raise RuntimeError(
            "Ray TPU multihost: TPU_WORKER_IDs are not contiguous 0.."
            f"{nnodes - 1}; got {sorted(worker_id_to_bundle)} ({probes})."
        )

    bundle_for_node = [worker_id_to_bundle[k] for k in range(nnodes)]
    rank0_node_ip = worker_id_to_ip[0]

    if engine_ip != rank0_node_ip:
        raise RuntimeError(
            f"Ray TPU multihost: Engine runs on {engine_ip} but TPU worker 0 is "
            f"on {rank0_node_ip}. The rank-0 scheduler must be co-located with "
            "the Engine (MASTER_ADDR points at worker 0). Start the Ray head / "
            "Engine on the host whose TPU_WORKER_ID is 0."
        )

    return bundle_for_node, rank0_node_ip


def order_tpu_bundles_by_worker_id(
    placement_group, nnodes: int
) -> tuple[list[int], str]:
    """Probe the live placement group and order bundles by physical TPU_WORKER_ID.

    Returns ``(bundle_for_node, rank0_node_ip)`` where ``bundle_for_node[k]`` is
    the bundle on the host with ``TPU_WORKER_ID == k`` and ``rank0_node_ip`` is
    that of worker 0. See ``_resolve_bundle_order_from_probes`` for the ordering
    contract and validation. Raises ``RuntimeError`` on any inconsistency.
    """
    engine_ip = ray.util.get_node_ip_address()
    probes = ray.get(
        [
            _get_worker_id_and_address.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group, placement_group_bundle_index=i
                )
            ).remote()
            for i in range(nnodes)
        ]
    )
    return _resolve_bundle_order_from_probes(probes, engine_ip)


# Config env namespaces the launcher (benchmark/tpu/launch_server.sh) exports at
# runtime and that a scheduler actor needs, but Ray drops: the actor only inherits
# the raylet's env (the container env at `ray start`), which predates those exports
# -- and worker nodes never run launch_server.sh at all. We forward these from the
# driver's os.environ via runtime_env (see _forwarded_driver_env). Allowlist, not
# denylist: missing an entry just leaves it dropped (status quo, safe), whereas
# forwarding a per-chip/per-node var would clobber Ray's per-actor value.
_FORWARD_ENV_PREFIXES = ("SGLANG_", "TORCH_", "HF_", "TRANSFORMERS_")
_FORWARD_ENV_NAMES = frozenset(
    {"LIBTPU_INIT_ARGS", "XLA_FLAGS", "TPU_PREMAPPED_BUFFER_SIZE"}
)
# Never forward, even when a prefix matches: these are per-chip/per-node (Ray sets
# them per actor -- forwarding the driver's copy makes every actor grab the same
# chip and crash) or are the physically-ordered slice env injected separately
# below from server_args (which must win over any inherited copy).
_NEVER_FORWARD_ENV = frozenset(
    {
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
        "TPU_WORKER_HOSTNAMES",
        "TPU_VISIBLE_CHIPS",
        "TPU_CHIPS_PER_HOST_BOUNDS",
        "TPU_HOST_BOUNDS",
    }
)


def _forwarded_driver_env(environ=None) -> dict:
    """Pick the driver's config env vars that should be forwarded to scheduler actors.

    Returns ``{name: value}`` for every var in ``environ`` (default ``os.environ``)
    whose name is in ``_FORWARD_ENV_NAMES`` or matches a ``_FORWARD_ENV_PREFIXES``
    prefix, minus ``_NEVER_FORWARD_ENV``. ``environ`` is injectable for unit tests.
    """
    if environ is None:
        environ = os.environ
    forwarded = {}
    for name, value in environ.items():
        if name in _NEVER_FORWARD_ENV:
            continue
        if name in _FORWARD_ENV_NAMES or name.startswith(_FORWARD_ENV_PREFIXES):
            forwarded[name] = value
    return forwarded


# Case-insensitive substrings marking an env var whose *value* is a secret. Such
# vars are still forwarded to actors (e.g. HF_TOKEN, which the actor needs to auth
# to the Hugging Face Hub), but their values must be masked whenever we log the
# forwarded set -- otherwise a token lands in plaintext in the driver log.
_SECRET_ENV_SUBSTRINGS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "KEY", "CREDENTIAL")


def _is_secret_env(name: str) -> bool:
    upper = name.upper()
    return any(sub in upper for sub in _SECRET_ENV_SUBSTRINGS)


def redact_env_for_log(env: dict) -> str:
    """Format ``{name: value}`` as a ``k=v`` list for logging, masking secret values.

    A var whose name looks like a secret (see ``_SECRET_ENV_SUBSTRINGS``) is shown as
    ``NAME=***`` so operators can still see it is being forwarded without leaking the
    value. Keys are sorted for stable, greppable output.
    """
    return ", ".join(
        f"{name}={'***' if _is_secret_env(name) else value}"
        for name, value in sorted(env.items())
    )


def tpu_actor_runtime_env(server_args) -> dict:
    """Ray ``.options()`` kwargs injecting the env a scheduler actor needs.

    Ray does NOT propagate the driver's ``os.environ`` to actors (they inherit only
    the raylet's env), so ``runtime_env`` is the sole channel. Two sources merge
    into one ``env_vars`` dict:

    1. The launcher-set config the driver has but the actor lost (``LIBTPU_INIT_ARGS``,
       compile-cache, ``SGLANG_*`` / ``TORCH_*`` toggles), via ``_forwarded_driver_env``.
    2. The launcher-*computed* TPU slice env stashed on ``server_args``:
       ``TPU_WORKER_HOSTNAMES`` (multihost host list) or the single-host
       ``TORCH_TPU_SLICEBUILDER_ADDRESSES`` / ``TORCH_TPU_TOPOLOGY``, read by the
       actor's TPU bootstrap (``_prepare_tpu_env``).

    Source 2 always wins over any inherited copy of the same var (it is the
    physically-ordered / single-host-override value). Returns an empty dict when
    there is nothing to inject, so actors fall back to their own pod env.
    """
    env_vars = _forwarded_driver_env()
    # Computed slice env overrides any forwarded/inherited copy. Inject an empty
    # string too (single-host on a multi-host slice), so the actor overrides the
    # slice-wide value inherited from `ray start` rather than falling back to it --
    # only None (GPU / discovery failed) means "don't inject, use the pod env".
    if server_args.tpu_worker_hostnames is not None:
        env_vars["TPU_WORKER_HOSTNAMES"] = server_args.tpu_worker_hostnames
    if server_args.torch_tpu_slicebuilder_addresses:
        env_vars["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = (
            server_args.torch_tpu_slicebuilder_addresses
        )
    if server_args.torch_tpu_topology:
        env_vars["TORCH_TPU_TOPOLOGY"] = server_args.torch_tpu_topology
    if not env_vars:
        return {}
    return {"runtime_env": {"env_vars": env_vars}}
