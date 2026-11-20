# SPDX-License-Identifier: Apache-2.0
"""Scheduling fence for Kimi-K3 decode on TPU: an XLA optimization barrier.

Why: in a KDA layer the static fp8 [q, k, v, g] GEMV waits ~4 us per layer
     (TP32, 16 chips) for XLA's VMEM prefetch of its weight, and XLA schedules
     the small b_proj / f_a / f_b GEMVs (~4 us, they read only hidden_states)
     after the big GEMV instead of into that wait. Passing the GEMV input
     through a barrier together with their outputs makes the big GEMV start
     only after they are done, so they run while the weight is still in
     flight.
How: ``order_after(x, *deps)`` returns its inputs unchanged; XLA cannot hoist
     a consumer of the returned ``x`` above the computation of ``deps``. The
     values pass through untouched, but the fence can change how XLA fuses
     the ops around it: on a 1024-token prefill chunk it changed the logits,
     so the KDA binder fences single-token batches only (bit-exact there).
"""

# PEP 563 must stay off here: cpu_export_jax_op reads the wrapper's jax.Array
# annotations (see host_split_op.py).

_OPS: dict = {}


def _build(n: int):
    import jax

    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    if n == 2:

        def _fence(x: jax.Array, a: jax.Array) -> tuple[jax.Array, jax.Array]:
            return jax.lax.optimization_barrier((x, a))

    elif n == 3:

        def _fence(
            x: jax.Array, a: jax.Array, b: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return jax.lax.optimization_barrier((x, a, b))

    else:
        raise ValueError(f"order_after supports 1 or 2 deps, got {n - 1}")
    return cpu_export_jax_op(f"sglang_tpu_pallas::k3_order_after_{n}", _fence)


def order_op(n: int):
    """The fence op over ``n`` tensors; build at bind time, before tracing."""
    if n not in _OPS:
        _OPS[n] = _build(n)
    return _OPS[n]


def order_after(x, *deps):
    """(x, *deps) unchanged, with x usable only after deps are computed."""
    return order_op(1 + len(deps))(x, *deps)
