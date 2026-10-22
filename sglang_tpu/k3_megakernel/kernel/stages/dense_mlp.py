# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 layer-0 dense MLP (SiTU, intermediate 33792) as an in-kernel stage.

Channels are split over all ranks (padded to whole 128-lane vregs; padded
channels have zero weights and contribute exactly zero). The down partials
are all-reduced in f32 and rounded once; the residual P2 is added after, as
in sglang (``bf16(down + P2)``).
"""

import jax
import jax.numpy as jnp

from sglang_tpu.k3_megakernel.config import (
    K3MK_COLLECTIVE_ID,
    K3Dims,
    Relax,
    Topology,
)
from sglang_tpu.k3_megakernel.kernel.collectives import (
    AllReduce,
    AllReduceSpec,
    entry_barrier,
)
from sglang_tpu.k3_megakernel.kernel.stages.common import (
    bf16,
    bind_stream,
    build_sharded,
    flat,
    flat_rows,
    stream_scratch,
    unflat,
)
from sglang_tpu.k3_megakernel.kernel.stages.expert import situ_and_mul


def dense_channels(dims: K3Dims, topo: Topology, inter: int = 33792) -> int:
    per = -(-inter // topo.world)
    return -(-per // 128) * 128


class DenseMlpStage:
    def __init__(
        self,
        dims: K3Dims,
        topo: Topology,
        rows: int,
        *,
        relax: Relax = Relax(),
        collective_id: int = K3MK_COLLECTIVE_ID,
        repeat: int = 1,
    ):
        self.dims, self.topo, self.rows, self.collective_id, self.repeat = (
            dims,
            topo,
            rows,
            collective_id,
            repeat,
        )
        self.relax = relax
        self.channels = dense_channels(dims, topo)
        self.ar = AllReduceSpec(
            topo,
            flat_rows(rows, dims.hidden),
            relax.transport,
            host_transport=relax.host_transport,
        )
        self.streams = (
            (dims.hidden, 2 * self.channels),
            (self.channels, dims.hidden),
        )

    def scratch(self):
        shapes = []
        for k, n in self.streams:
            shapes += stream_scratch(k, n, self.relax.dense_dtype)
        return shapes + self.ar.scratch_shapes()

    def body(self, x, p2, gu, dn, ar):
        """``x`` bf16 [R, H] MLP input, ``p2`` [R, H] residual -> bf16-valued f32 [R, H]."""
        d = self.dims
        h = gu.matmul(x, prefetched=False).astype(jnp.bfloat16)
        act = situ_and_mul(h, beta=d.situ_beta, linear_beta=d.situ_linear_beta)
        down = bf16(
            unflat(ar(flat(dn.matmul(act, prefetched=False))), self.rows, d.hidden)
        )
        return bf16(down + p2)

    def kernel(self, *refs):
        """Inputs: x, p2, gate_up, down[, gate_up_s, down_s]; output: out; then scratch."""
        fp8 = self.relax.dense_fp8
        n_in = 6 if fp8 else 4
        x_ref, p2_ref, gu_hbm, dn_hbm = refs[:4]
        scales = refs[4:6] if fp8 else (None, None)
        out_ref = refs[n_in]
        scratch = refs[n_in + 1 :]
        me = jax.lax.axis_index("x")
        per = len(stream_scratch(1, 128, self.relax.dense_dtype))
        gu = bind_stream(gu_hbm, scratch[:per], scales[0])
        dn = bind_stream(dn_hbm, scratch[per : 2 * per], scales[1])
        ar = AllReduce(self.ar, me, *scratch[2 * per :])
        ar.init()
        entry_barrier(me, self.topo)

        def run():
            out_ref[...] = self.body(
                x_ref[...], p2_ref[...].astype(jnp.float32), gu, dn, ar
            ).astype(jnp.bfloat16)

        if self.repeat == 1:
            run()
        else:
            from jax.experimental import pallas as pl

            @pl.loop(0, self.repeat)
            def _(i):
                run()

    def build(self, mesh):
        return build_sharded(
            self.kernel,
            mesh=mesh,
            in_spaces=("vmem", "vmem") + ("hbm",) * (4 if self.relax.dense_fp8 else 2),
            out_shapes=(
                jax.ShapeDtypeStruct((self.rows, self.dims.hidden), jnp.bfloat16),
            ),
            out_spaces=("vmem",),
            scratch=self.scratch(),
            collective_id=self.collective_id,
        )
