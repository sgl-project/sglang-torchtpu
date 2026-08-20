# SPDX-License-Identifier: Apache-2.0
"""Process-scoped registry for the PCP streaming torch custom op.

The PCP kernel is a JAX-Mesh-scoped ``jax.jit`` wrapped in a
torch custom op. The Mesh depends on the distributed process group,
which doesn't exist at backend ``__init__`` time (init runs before
distributed init on some paths). This registry lazily builds and caches
one op per distinct config tuple at first attention call.

Because ``inspect.signature`` on the JAX-wrapped callable trips a
Dynamo graph break, this registry MUST be called from the eager
(non-torch.compile) path. In sglang's TPU serving flow, the backend's
``_forward_pcp`` runs inside torch.compile — so the registry must be
warmed by the backend's ``__init__`` before the first compiled forward.
The warm site is
:func:`sglang.srt.hardware_backend.tpu.attention_pallas_rpa_v3_cp._warm_pcp_streaming_op_registry`,
called once per model-runner init.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

_REGISTRY_LOCK = threading.Lock()
_OP_REGISTRY: dict[tuple, Any] = {}


def _mesh_fingerprint(mesh) -> tuple:
    """Stable fingerprint for a jax.sharding.Mesh — same shape/devices/axes
    → same tuple, so two logically-equivalent meshes hit the same cache entry
    without depending on ``id(mesh)`` (mesh instances may be rebuilt).
    """
    devices = mesh.devices
    device_ids = tuple(int(d.id) for d in devices.flat)
    return (tuple(mesh.axis_names), tuple(devices.shape), device_ids)


def _config_key(
    *,
    mesh,
    sm_scale: Optional[float],
    sliding_window: Optional[int],
    soft_cap: Optional[float],
    k_scale: Optional[float],
    v_scale: Optional[float],
    cp_kv_cache_interleave_size: int,
    q_block_size: int,
) -> tuple:
    return (
        _mesh_fingerprint(mesh),
        sm_scale,
        sliding_window,
        soft_cap,
        k_scale,
        v_scale,
        int(cp_kv_cache_interleave_size),
        int(q_block_size),
    )


def get_or_register_pcp_streaming_op(
    *,
    mesh,
    sm_scale: Optional[float],
    sliding_window: Optional[int],
    soft_cap: Optional[float],
    k_scale: Optional[float],
    v_scale: Optional[float],
    cp_kv_cache_interleave_size: int,
    q_block_size: int,
) -> Any:
    """Return the process-scoped torch custom op for this config.

    Registers the kernel once per (sm_scale, sliding_window, soft_cap,
    k_scale, v_scale, interleave, q_block_size) tuple. Subsequent calls
    with the same tuple return the cached op — no re-registration.

    The op name embeds the config tuple so two configs coexist without
    stomping on each other's torch.library entry.
    """
    key = _config_key(
        mesh=mesh,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        k_scale=k_scale,
        v_scale=v_scale,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        q_block_size=q_block_size,
    )
    with _REGISTRY_LOCK:
        cached = _OP_REGISTRY.get(key)
        if cached is not None:
            return cached

        from sglang.srt.hardware_backend.tpu.attention_pallas_pcp_streaming_op import (
            PCP_STREAMING_RPA_INPUT_PARTITION_SPECS,
            make_pcp_streaming_rpa_kernel,
            pcp_streaming_jax_op,
        )

        fn = make_pcp_streaming_rpa_kernel(
            q_scale=None,
            k_scale=k_scale,
            v_scale=v_scale,
            mesh=mesh,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
            soft_cap=soft_cap,
            skip_kv_update=False,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            q_block_size=q_block_size,
        )
        # Unique op name per config so torch.library's process-global op
        # namespace can hold multiple variants without collision. Use the
        # full 64-bit hash — a 32-bit truncation collides at ~65k configs
        # (birthday bound) which is well within the range a single server
        # can visit under multi-mesh (TP × PCP) precompile sweeps.
        op_name = "sglang_tpu_pallas::pcp_streaming_rpa" f"_{abs(hash(key)):016x}"
        op = pcp_streaming_jax_op(
            op_name,
            fn,
            donate_argnums=(0,),  # kv_cache is arg 0
            mesh=mesh,
            input_partition_specs=PCP_STREAMING_RPA_INPUT_PARTITION_SPECS,
        )
        _OP_REGISTRY[key] = op
        return op
