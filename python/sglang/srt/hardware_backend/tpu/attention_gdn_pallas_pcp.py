# SPDX-License-Identifier: Apache-2.0
"""Torch custom op for the fused context-parallel Gated-DeltaNet kernel.

The dense GDN op hides its state writeback inside the op
(``mutates_args=("conv_state", "recurrent_state")`` plus
``input_output_aliases``). That contract does not survive this path: on the
shard_map + ``jax.export`` route it forces Dynamo to clone-before-call and
copy-back-at-graph-exit, which defeats the aliasing. The streaming attention
op hit exactly this and measured ~50% wrong first-token output; see
``attention_pallas_pcp_streaming_op`` for the full account.

So this op declares ``mutates_args=()``, returns the new pool alongside the
outputs, and the caller does ``pool.copy_(new_pool)`` one frame up in
:func:`invoke_fused_gdn_pcp_op`. Keep the copy in that frame -- hoisting it
into the caller's frame triggered warm-probe recompiles on the streaming path.

The kernel also swallows the QKVZ projection, so this op takes the projection
weight and its per-output-channel scale rather than a projected activation.
"""

# NOTE: do *not* add ``from __future__ import annotations`` -- torch.library
# infers the op schema from real annotations.

from typing import Tuple

import jax
import torch
from jax.sharding import PartitionSpec

from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.hardware_backend.tpu.attention_pallas_pcp_streaming_op import (
    PCP_AXIS_NAME,
    build_pcp_streaming_callable,
)
from sglang.srt.layers.cp.pcp_gdn import run_fused_pcp_gdn_prefill

logger = get_logger(__name__)

# Argument order must match _fused_gdn_pcp_custom_op below.
FUSED_GDN_PCP_INPUT_PARTITION_SPECS = (
    PartitionSpec(PCP_AXIS_NAME, None),  # hidden_states: this rank's tokens
    PartitionSpec(),  # qkvz_weight (replicated)
    PartitionSpec(),  # qkvz_weight_scale (replicated)
    PartitionSpec(PCP_AXIS_NAME, None),  # b
    PartitionSpec(PCP_AXIS_NAME, None),  # a
    PartitionSpec(PCP_AXIS_NAME),  # state pool: this rank's blocks
    PartitionSpec(),  # conv_weight
    PartitionSpec(),  # conv_bias
    PartitionSpec(),  # A_log
    PartitionSpec(),  # dt_bias
    PartitionSpec(),  # state_indices
    PartitionSpec(),  # query_start_loc
    PartitionSpec(),  # distribution
    PartitionSpec(),  # seq_lens
)
FUSED_GDN_PCP_OUTPUT_PARTITION_SPECS = (
    PartitionSpec(PCP_AXIS_NAME),  # new pool
    PartitionSpec(PCP_AXIS_NAME, None, None),  # out
    PartitionSpec(PCP_AXIS_NAME, None, None),  # z
)


def build_fused_gdn_pcp_op(
    *,
    name: str,
    mesh,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    pool_block_tokens: int,
    pcp_size: int,
    interleave_size: int,
):
    """Register one fused GDN op for a fixed geometry.

    Every dimension the kernel specializes on is closed over here rather than
    passed per call, so the compiled program is keyed by geometry alone and a
    forward never has to hand Python scalars across the op boundary.
    """

    # Annotations are required: the callable builder verifies the signature
    # and infers which arguments are static from them.
    def _fn(
        hidden_states: jax.Array,
        qkvz_weight: jax.Array,
        qkvz_weight_scale: jax.Array,
        b: jax.Array,
        a: jax.Array,
        pool: jax.Array,
        conv_weight: jax.Array,
        conv_bias: jax.Array,
        A_log: jax.Array,
        dt_bias: jax.Array,
        state_indices: jax.Array,
        query_start_loc: jax.Array,
        distribution: jax.Array,
        seq_lens: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return run_fused_pcp_gdn_prefill(
            hidden_states,
            qkvz_weight,
            qkvz_weight_scale,
            b,
            a,
            pool,
            conv_weight,
            conv_bias,
            A_log,
            dt_bias,
            state_indices,
            query_start_loc,
            distribution,
            seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            pool_block_tokens=pool_block_tokens,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            mesh=mesh,
        )

    # build_pcp_streaming_callable is shared; its sibling registration helper
    # is not -- that one hard-codes the streaming op's 8-argument schema and a
    # two-tuple fake. This op takes 14 tensors and returns three.
    wrapped_fn = build_pcp_streaming_callable(
        name,
        _fn,
        # The pool is donated: the kernel streams state through it in place and
        # returns it, and the caller copies it back.
        donate_argnums=(5,),
        mesh=mesh,
        input_partition_specs=FUSED_GDN_PCP_INPUT_PARTITION_SPECS,
        output_partition_specs=FUSED_GDN_PCP_OUTPUT_PARTITION_SPECS,
    )

    # torch.library infers the schema from these annotations, so the wrapper
    # cannot be dropped in favour of passing wrapped_fn straight through.
    def _fused_gdn_pcp_custom_op(
        hidden_states: torch.Tensor,
        qkvz_weight: torch.Tensor,
        qkvz_weight_scale: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        pool: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        distribution: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return wrapped_fn(
            hidden_states,
            qkvz_weight,
            qkvz_weight_scale,
            b,
            a,
            pool,
            conv_weight,
            conv_bias,
            A_log,
            dt_bias,
            state_indices,
            query_start_loc,
            distribution,
            seq_lens,
        )

    # DO NOT set mutates_args. See the module docstring: the pool writeback
    # happens at the caller via pool.copy_(new_pool).
    op = torch.library.custom_op(
        name,
        _fused_gdn_pcp_custom_op,
        mutates_args=(),
        device_types=["tpu"],
    )

    n_v_local = n_v // pcp_size

    def _fake(
        hidden_states: torch.Tensor,
        qkvz_weight: torch.Tensor,
        qkvz_weight_scale: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        pool: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # out and z are [tokens, n_v, d_v] with every value head restored, so
        # they are shaped off the global head count, not this rank's shard.
        num_tokens = hidden_states.shape[0]
        shape = (num_tokens, n_v, d_v)
        return (
            torch.empty_like(pool),
            hidden_states.new_empty(shape),
            hidden_states.new_empty(shape),
        )

    op.register_fake(_fake)
    return op


def invoke_fused_gdn_pcp_op(
    op,
    *,
    pool: torch.Tensor,
    hidden_states: torch.Tensor,
    qkvz_weight: torch.Tensor,
    qkvz_weight_scale: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    distribution: torch.Tensor,
    seq_lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Call the op and land the state back in the pool.

    Returns ``(core_attn_out, z)``: the kernel produces the Z projection as
    part of the QKVZ GEMM it swallowed, and the layer's gated norm needs it.

    The ``pool.copy_`` lives here, in the same frame as the op call, on
    purpose. The op cannot declare the pool mutated (see the module
    docstring), and moving this copy into the caller's frame regressed
    warm-probe latency on the streaming path.
    """
    new_pool, out, z = op(
        hidden_states,
        qkvz_weight,
        qkvz_weight_scale,
        b,
        a,
        pool,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        state_indices,
        query_start_loc,
        distribution,
        seq_lens,
    )
    pool.copy_(new_pool)
    return out, z
