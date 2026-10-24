# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 latent-MoE block as one in-kernel stage (decode rows, any Topology).

Data flow per rank (see ``MoeLayout`` for the split):
  router (chip-pair half) -> pair exchange -> top-k
  latent_down (rank chunk) -> all-gather          -> latent [R, 3584]
  shared expert (rank channels) -> in-host reduce-scatter to lane columns
  routed experts of this host, this lane's channels -> in-host all-reduce (f32)
  both -> one cross-host round (``ReduceSpec``)
  latent RMSNorm -> latent_up (lane columns)
  routed + shared (lane columns) -> lane all-gather -> [R, 7168]

Rounding points follow sglang's TPU path: bf16 router logits and sigmoid
scores, bf16 outputs of every linear layer, f32 accumulation of the
route-weighted expert partials (rounded once, after the cross-rank sum).
"""

import jax
import jax.numpy as jnp
import msgspec
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.k3_megakernel.config import (
    K3MK_COLLECTIVE_ID,
    MoeLayout,
    Relax,
    Topology,
)
from sglang_tpu.k3_megakernel.kernel.collectives import (
    AllGather,
    AllGatherSpec,
    Reduce,
    ReducePart,
    ReduceSpec,
    entry_barrier,
)
from sglang_tpu.k3_megakernel.kernel.stages.attn_res import lane_sum
from sglang_tpu.k3_megakernel.kernel.stages.common import (
    bf16,
    bind_stream,
    flat,
    flat_rows,
    phase,
    stream_scratch,
    unflat,
)
from sglang_tpu.k3_megakernel.kernel.stages.expert import (
    down_dot,
    down_dot_dequant,
    down_dot_pair,
    gate_up_dot,
    situ_and_mul,
)

_RING = (
    4  # default expert weight slots; DMA for slot j+ring starts once slot j is consumed
)
_NEG = -3.0e38


def _packed_route(e, layout: MoeLayout):
    """(owner host, local id, flexible 0/1, field shift) of global expert ``e``."""
    hosts, eh = layout.topo.hosts, layout.experts_per_host
    g = sum(
        ((e >= h * eh).astype(jnp.int32) for h in range(1, hosts)),
        jnp.asarray(0, jnp.int32),
    )
    loc = e - g * eh
    return g, loc, (loc < layout.replica_count).astype(jnp.int32), g * 8


def replica_quota(ids, layout: MoeLayout, rounds: int = 2):
    """Per-host move counts of the balancing plan (see ``_route_owners_generic``), packed as
    8-bit fields of one int32 (hosts <= 4)."""
    hosts = layout.topo.hosts
    i32 = lambda v: jnp.asarray(v, jnp.int32)
    own_p, mov_p = i32(0), i32(0)
    for e in ids:
        _, _, f, sh = _packed_route(e, layout)
        own_p = own_p + jax.lax.shift_left(i32(1), sh)
        mov_p = mov_p + jax.lax.shift_left(f, sh)
    return _packed_quota(own_p, mov_p, hosts, rounds)


def _packed_quota(own_p, mov_p, hosts: int, rounds: int):
    """Per-host move counts (8-bit fields) from packed own / movable counts."""
    i32 = lambda v: jnp.asarray(v, jnp.int32)
    field = lambda packed, h: jax.lax.shift_right_logical(packed, i32(8 * h)) & 255
    own = [field(own_p, h) for h in range(hosts)]
    movable = [field(mov_p, h) for h in range(hosts)]
    move = [i32(0)] * hosts
    for _ in range(rounds):
        for h in range(hosts):
            nxt, prv = (h + 1) % hosts, (h - 1) % hosts
            # Compare + select, not jnp.maximum / minimum: those on scalars measured ~1 us/layer.
            gap = own[h] + move[prv] - (own[nxt] - move[nxt])
            half = jax.lax.shift_right_logical(jnp.where(gap > 0, gap, 0), i32(1))
            move[h] = jnp.where(half < movable[h], half, movable[h])
    return sum((jax.lax.shift_left(move[h], i32(8 * h)) for h in range(hosts)), i32(0))


def replica_serve(e, quota_p, seen, layout: MoeLayout):
    """(server host, expert slot, updated ``seen``) of the next route (selection order);
    ``seen`` packs each host's flexible routes so far (start at 0)."""
    g, loc, f, sh = _packed_route(e, layout)
    before = jax.lax.shift_right_logical(seen, sh) & 255
    quota = jax.lax.shift_right_logical(quota_p, sh) & 255
    moved = f * (before < quota).astype(jnp.int32)
    return (
        (g + moved) & (layout.topo.hosts - 1),
        loc + moved * layout.experts_per_host,
        seen + jax.lax.shift_left(f, sh),
    )


def packed_balance(layout: MoeLayout, balance: bool) -> bool:
    return balance and layout.replica_count > 0 and layout.topo.hosts <= 4


def route_owners(ids, layout: MoeLayout, rounds: int = 2, balance: bool = True):
    """See ``_route_owners_generic``; the same plan through ``replica_quota`` /
    ``replica_serve`` when ``packed_balance`` holds."""
    if not packed_balance(layout, balance):
        return _route_owners_generic(ids, layout, rounds, balance)
    quota_p = replica_quota(ids, layout, rounds)
    servers, slots, seen = [], [], jnp.asarray(0, jnp.int32)
    for e in ids:
        srv, slot, seen = replica_serve(e, quota_p, seen, layout)
        servers.append(srv)
        slots.append(slot)
    return servers, slots


def _route_owners_generic(
    ids, layout: MoeLayout, rounds: int = 2, balance: bool = True
):
    """Per route (global expert ids ``ids``, int32 scalars in selection order): the serving
    host and its expert slot there. Without replicas that is the owner. With
    ``layout.replica_count`` R, a route to one of the first R experts of host h may move to
    host h + 1 (its replica slot). Host h moves ``move[h]`` of those, chosen as its first
    such routes in selection order; ``move`` is set by ``rounds`` sweeps of pairwise
    balancing of the route counts of h and h + 1 (one route = one expert slice of work).
    Every rank computes the same plan from the same ids. Works on traced and concrete
    values. ``balance=False`` keeps every route on its owner (replica slots unused)."""
    hosts, eh, reps = layout.topo.hosts, layout.experts_per_host, layout.replica_count
    owner = [
        sum(((e >= h * eh).astype(jnp.int32) for h in range(1, hosts)), jnp.int32(0))
        for e in ids
    ]
    local = [e - g * eh for e, g in zip(ids, owner)]
    if reps == 0 or not balance:
        return owner, local
    flex = [loc < reps for loc in local]
    count = lambda masks: sum((m.astype(jnp.int32) for m in masks), jnp.int32(0))
    own = [count([g == h for g in owner]) for h in range(hosts)]
    movable = [count([(g == h) & f for g, f in zip(owner, flex)]) for h in range(hosts)]
    move = [jnp.int32(0)] * hosts
    for _ in range(rounds):
        for h in range(hosts):
            nxt, prv = (h + 1) % hosts, (h - 1) % hosts
            # Loads of h and h + 1 (load = own - moved out + moved in) with nothing moved
            # from h; then move half the difference.
            a = own[h] + move[prv]
            b = own[nxt] - move[nxt]
            half = jax.lax.shift_right_logical(jnp.maximum(a - b, 0), 1)
            move[h] = jnp.minimum(half, movable[h])
    servers, slots = [], []
    seen = [jnp.int32(0)] * hosts
    for g, loc, f in zip(owner, local, flex):
        is_g = [g == h for h in range(hosts)]
        before = sum((jnp.where(m, s, 0) for m, s in zip(is_g, seen)), jnp.int32(0))
        quota = sum((jnp.where(m, q, 0) for m, q in zip(is_g, move)), jnp.int32(0))
        moved = f & (before < quota)
        seen = [s + (m & f).astype(jnp.int32) for m, s in zip(is_g, seen)]
        nxt = jnp.where(g == hosts - 1, 0, g + 1)
        servers.append(jnp.where(moved, nxt, g))
        slots.append(jnp.where(moved, eh + loc, loc))
    return servers, slots


class _Ctx(msgspec.Struct, frozen=True):
    """Refs and values of one MoE block shared by its phases (bound once per ``_body``).
    ``colls``: (pair, lat_gather, reduce, out_gather)."""

    me: object
    x: object
    bias_ref: object
    norm_ref: object
    egu: tuple
    out_ref: object
    ids_ref: object
    w_ref: object
    streams: dict
    ring: list
    ring_sems: object
    lat_vmem: object
    acc_vmem: object
    lid: object
    lrow: object
    lw: object
    count: object
    colls: object
    after_route: object = None


class MoeStage:
    """Builds the MoE block for ``rows`` decode rows on ``layout.topo``."""

    def __init__(
        self,
        layout: MoeLayout,
        rows: int,
        *,
        relax: Relax = Relax(),
        collective_id: int = K3MK_COLLECTIVE_ID,
        repeat: int = 1,
        fast_topk: bool = True,
        gather_bf16: bool = False,
        flat_gather: bool = True,
        shared_after: int = -1,
        early_dma: bool = False,
        pair_down: bool = False,
        ring: int = _RING,
        overlap_pair: bool = False,
        late_handoff: bool | str = False,
        split_refill: bool = False,
        late_latent_up: bool | str = False,
        row_gathers: bool = False,
        up_tree: bool = False,
        single_rows: bool = True,
        down_dequant: bool = True,
        debug: frozenset = frozenset(),
    ):
        """``gather_bf16``: the router-score, latent and output gathers travel as bf16 (their
        values are bf16, so exact). ``flat_gather``: the latent gather is one world-wide
        one-shot (one dependent round trip at ``hosts > 1``; see ``AllGatherSpec.flat``).
        ``row_gathers`` (on by default in verify rows): the three gathers are flat one-shots of bf16 ``[R, C]``
        rows (``AllGatherSpec`` row mode: router 512, latent chunk, output columns per message
        instead of whole [8, 128] f32 tiles); exact.
        ``shared_after`` >= 0: the shared expert runs after that many routed experts instead of
        before them. ``early_dma``: the fast top-k starts each local expert's weight DMA as soon as
        it is picked (not with balanced expert replicas). ``pair_down``: two experts' down
        projections share one MXU pass (``down_dot_pair``). ``ring``: expert weight slots in VMEM
        (3.4 MiB each at 8 lanes). ``overlap_pair``: latent_down runs while the router scores fly
        to the chip peer. ``late_handoff``: ``after_route`` (the stack's
        next-layer prefetch) runs after the routed experts (True / "experts") or after the shared
        expert ("shared") instead of right after the routing, keeping its scalar work and DMAs off
        the route -> experts path. ``split_refill``: an expert's gate/up dot waits only for the
        gate/up DMAs and its slot is refilled right after that dot (the down weights likewise).
        ``late_latent_up``: latent_up's whole ring is filled once latent_down is consumed
        (True / "latent"; its buffer overlays router + latent_down, see the stack's region M) or at
        the end of the routed experts ("reduce": off the expert DMAs' HBM window, still ahead of
        the reductions). ``up_tree``: log-depth f32 sum in the latent RMSNorm
        (``attn_res.lane_sum``; not bit-identical).
        ``debug`` (benchmark-only, outputs meaningless): "no_expert_math" skips the expert dots;
        per-phase skips with balanced DMAs / collectives: "moe_route" (fixed routing, no router or
        top-k), "moe_experts" (no expert DMAs or math), "moe_shared" (no shared dots; a zero part
        still goes through the reduction), "moe_reduce" (no reduction), "moe_up" (no latent_up or
        output gather), "moe_up_dot" (no latent RMSNorm / latent_up dot), "moe_up_gather" (no output
        gather; the local columns are written to every lane's slot); "moe_topk2" runs the fast top-k twice."""
        # repeat > 1 runs the block back to back inside one call (device-time benchmark only).
        self.layout, self.rows, self.collective_id, self.repeat = (
            layout,
            rows,
            collective_id,
            repeat,
        )
        self.fast_topk = fast_topk
        self.shared_after, self.early_dma, self.pair_down, self.ring = (
            shared_after,
            early_dma,
            pair_down,
            ring,
        )
        self.overlap_pair, self.split_refill = overlap_pair, split_refill
        self.handoff = {False: "route", True: "experts"}.get(late_handoff, late_handoff)
        assert self.handoff in ("route", "shared", "experts"), late_handoff
        self.late_latent_up = {True: "latent"}.get(late_latent_up, late_latent_up)
        assert self.late_latent_up in (False, "latent", "reduce"), late_latent_up
        self.up_tree = up_tree
        # Grouped rows: an expert picked by one row runs its down projection on that row only.
        self.single_rows = single_rows
        # Grouped rows, several picking rows: the down dot as one M=R pass on scale-applied bf16 weights.
        self.down_dequant = down_dequant
        self.debug = debug
        assert ring % 2 == 0 or not pair_down, ring
        self.relax = relax
        d, topo = layout.dims, layout.topo
        g = jnp.bfloat16 if gather_bf16 else jnp.float32
        self.row_gathers = row_gathers
        if self.row_gathers:
            row = lambda t, cols: AllGatherSpec(
                t, rows, jnp.bfloat16, flat=True, cols=cols, by_cols=rows > 1
            )
            self.pair = row(Topology(1, 2), layout.router_pad)
            self.lat_gather = row(topo, layout.latent_chunk)
            self.out_gather = row(Topology(1, topo.lanes), layout.latent_up_cols)
        else:
            self.pair = AllGatherSpec(
                Topology(1, 2), flat_rows(rows, layout.router_pad), g
            )
            self.lat_gather = AllGatherSpec(
                topo, flat_rows(rows, layout.latent_chunk), g, flat_gather
            )
            self.out_gather = AllGatherSpec(
                Topology(1, topo.lanes), flat_rows(rows, layout.latent_up_cols), g
            )
        # Routed (all-reduce) and shared (reduce-scatter to this lane's output columns) share
        # the cross-host round; per element the f32 sums are those of two tiered all-reduces.
        # Several rows: the routed sum is scattered by row too (lane j reduces rows j * rpl ..), then
        # the lanes all-gather the bf16 routed rows; same per-element sums, 1 / lanes of the bytes.
        self.row_reduce = rows > 1 and rows % topo.lanes == 0
        rpl = rows // topo.lanes
        routed = (
            ReducePart(flat_rows(rpl, d.latent), scatter=True)
            if self.row_reduce
            else ReducePart(flat_rows(rows, d.latent))
        )
        self.reduce = ReduceSpec(
            topo,
            (routed, ReducePart(flat_rows(rows, layout.latent_up_cols), scatter=True)),
            relax.transport,
            relax.host_transport,
        )
        self.routed_gather = (
            AllGatherSpec(
                Topology(1, topo.lanes), 1, jnp.bfloat16, flat=True, cols=rpl * d.latent
            )
            if self.row_reduce
            else None
        )
        self.streams = {
            "router": (d.hidden, layout.router_pad),
            "latent_down": (d.hidden, layout.latent_chunk),
            "shared_gu": (d.hidden, 2 * layout.shared_cols),
            "shared_down": (layout.shared_cols, d.hidden),
            "latent_up": (d.latent, layout.latent_up_cols),
        }

    # ----- scratch -----
    def _scratch(self):
        d, lay, r = self.layout.dims, self.layout, self.rows
        li = lay.lane_inter
        shapes = []
        for k, n in self.streams.values():
            shapes += stream_scratch(k, n, self.relax.dense_dtype)
        shapes += [
            pltpu.VMEM((self.ring, d.latent, 2 * li), jnp.float8_e4m3fn),
            pltpu.VMEM((self.ring, 1, 2 * li), jnp.float32),
            pltpu.VMEM((self.ring, li // 2, d.latent), jnp.uint8),
            pltpu.VMEM((self.ring, li // 32, d.latent), jnp.uint8),
            pltpu.SemaphoreType.DMA((self.ring, 4)),
            pltpu.VMEM((r, d.latent), jnp.float32),  # gathered latent (bf16 values)
            pltpu.VMEM((r, d.latent), jnp.float32),  # routed accumulator
            pltpu.SMEM(
                (r * d.top_k,), jnp.int32
            ),  # local expert index (into this host's experts)
            pltpu.SMEM(
                (r * d.top_k,), jnp.int32
            ),  # row of each local (row, expert) pair
            pltpu.SMEM((r * d.top_k,), jnp.float32),  # route weight of each pair
            pltpu.SMEM(
                (2,), jnp.int32
            ),  # number of local pairs, expert DMAs started during the top-k
        ]
        for spec in self.coll_specs():
            shapes += spec.scratch_shapes()
        return shapes

    def coll_specs(self):
        specs = (self.pair, self.lat_gather, self.reduce, self.out_gather)
        return specs + ((self.routed_gather,) if self.row_reduce else ())

    def bind_colls(self, me, refs):
        """Collectives of ``coll_specs`` on their scratch ``refs`` (in order), initialized."""
        it = iter(refs)
        colls = []
        for cls, spec in zip(
            (AllGather, AllGather, Reduce, AllGather, AllGather), self.coll_specs()
        ):
            colls.append(
                cls(spec, me, *[next(it) for _ in range(len(spec.scratch_shapes()))])
            )
        for c in colls:
            c.init()
        return colls

    # ----- pieces -----
    def _route_start(self, x, router, pair):
        """Scores of this core's half of the experts, sent to the other core of the chip."""
        logits = bf16(router.matmul(x, prefetched=True))
        pair.start(self._gather_in(bf16(jax.nn.sigmoid(logits))))

    def _gather_in(self, x):
        """A gather's input block for ``[R, C]`` values (row mode: the row itself)."""
        return x if self.row_gathers else flat(x)

    def _gathered(self, res, cols: int, n: int):
        """``[R, n * cols]`` from a gather's result over ``n`` ranks (rank order)."""
        if self.row_gathers:
            return res
        blocks = res.reshape((n,) + res.shape[-2:])
        return jnp.concatenate(
            [unflat(blocks[i], self.rows, cols) for i in range(n)], axis=1
        )

    def _route(self, bias_ref, pair, ids_out, w_out, early=None):
        """Top-k over sglang's routing (after ``_route_start``): bf16 logits, bf16 sigmoid scores,
        selection on score + bias, weights renormalized in f32. ``early``: ``(on_pick, undo)``;
        the fast top-k calls ``on_pick(expert id)`` per pick, ``undo()`` runs before a tie rerun."""
        lay, r = self.layout, self.rows
        rp = lay.router_pad
        full = self._gathered(pair.finish(), rp, 2)
        choice = full + bias_ref[...]
        if self.fast_topk and r == 1 and 2 * rp == 1024:
            if (
                "moe_topk2" in self.debug
            ):  # timing only: the top-k's cost is the difference
                self._topk_fast(choice + bias_ref[...] * 0.0, full, ids_out, w_out)
            tie = self._topk_fast(
                choice, full, ids_out, w_out, None if early is None else early[0]
            )

            @pl.when(tie)
            def _():
                if early is not None:
                    early[1]()
                self._topk_reference(choice, full, ids_out, w_out)
        elif self.fast_topk and 2 * rp == 1024:
            if "moe_topk2" in self.debug:
                self._topk_rows(choice + bias_ref[...] * 0.0, full, ids_out, w_out)
            tie = self._topk_rows(choice, full, ids_out, w_out)

            @pl.when(tie)
            def _():
                self._topk_reference(choice, full, ids_out, w_out)
        else:
            self._topk_reference(choice, full, ids_out, w_out)

    def _topk_rows(self, choice, full, ids_out, w_out):
        """``_topk_reference`` for all rows at once: each step takes every row's maximum over its
        1024 columns (one lane reduction for the ``[R, 1024]`` block), the winners' columns and
        scores come from independent reductions. Without ties the picks and every rounding equal
        the reference's; returns a scalar flag set when some row's step maximum was not unique."""
        d, r = self.layout.dims, self.rows
        rp = self.layout.router_pad
        cols = jax.lax.broadcasted_iota(jnp.int32, choice.shape, 1).astype(jnp.float32)
        rowsum = lambda v: jnp.sum(v, axis=1, keepdims=True)
        work = choice
        ties = jnp.zeros((r, 1), jnp.float32)
        idx, wts = [], []
        for _ in range(d.top_k):
            hit = work == jnp.max(work, axis=1, keepdims=True)
            ties = ties + jnp.abs(rowsum(hit.astype(jnp.float32)) - 1.0)
            idx.append(rowsum(jnp.where(hit, cols, 0.0)).astype(jnp.int32))
            wts.append(rowsum(jnp.where(hit, full, 0.0)))
            work = jnp.where(hit, _NEG, work)
        total = wts[0]
        for w in wts[1:]:
            total = total + w
        denom = total + 1e-20
        for k in range(d.top_k):
            e = (idx[k] // rp) * self.layout.router_cols + idx[k] % rp  # [R, 1]
            wn = wts[k] / denom
            for row in range(r):
                ids_out[row * d.top_k + k] = jnp.sum(e[row : row + 1])
                w_out[row * d.top_k + k] = jnp.sum(wn[row : row + 1])
        return jnp.sum(ties) > 0

    def _expert_id(self, idx):
        rp = self.layout.router_pad
        return (idx // rp) * self.layout.router_cols + idx % rp

    def _store_route(self, slot, idx, w, denom, ids_out, w_out):
        ids_out[slot] = self._expert_id(idx)
        w_out[slot] = w / denom

    def _topk_reference(self, choice, full, ids_out, w_out):
        """Repeated argmax (lowest column wins a tie)."""
        d, r = self.layout.dims, self.rows
        col = jax.lax.broadcasted_iota(jnp.int32, (1, choice.shape[1]), 1)
        for row in range(r):
            c = choice[row : row + 1]
            s = full[row : row + 1]
            picked = []
            for k in range(d.top_k):
                m = jnp.max(c)
                idx = jnp.min(jnp.where(c == m, col, 1 << 30))
                w = jnp.max(jnp.where(col == idx, s, _NEG))
                c = jnp.where(col == idx, _NEG, c)
                picked.append((idx, w))
            total = picked[0][1]
            for _, w in picked[1:]:
                total = total + w
            denom = total + 1e-20
            for k, (idx, w) in enumerate(picked):
                self._store_route(row * d.top_k + k, idx, w, denom, ids_out, w_out)

    def _topk_fast(self, choice, full, ids_out, w_out, on_pick=None):
        """``_topk_reference`` for one row with one dependent reduction per step.

        The 1024 columns are folded into one dense ``[8, 128]`` tile; each step
        takes the tile maximum and masks every lane holding it, while the winner's
        column and score come from independent reductions. Without ties the picks,
        their order and every rounding equal the reference; returns a scalar flag
        set when some step's maximum was not unique (the caller then reruns the
        reference, which overwrites the outputs).
        """
        d = self.layout.dims
        fold = lambda v: jnp.concatenate(
            [v[:, i * 128 : (i + 1) * 128] for i in range(8)], axis=0
        )
        tile_max = lambda v: jnp.max(
            jnp.max(v, axis=1, keepdims=True), axis=0, keepdims=True
        )
        tile_sum = lambda v: jnp.sum(
            jnp.sum(v, axis=1, keepdims=True), axis=0, keepdims=True
        )
        work, probs = fold(choice), fold(full)
        cols = (
            jax.lax.broadcasted_iota(jnp.int32, (8, 128), 0) * 128
            + jax.lax.broadcasted_iota(jnp.int32, (8, 128), 1)
        ).astype(jnp.float32)
        ties = jnp.zeros((1, 1), jnp.float32)
        picked = []
        for _ in range(d.top_k):
            hit = work == tile_max(work)
            ties = ties + jnp.abs(tile_sum(hit.astype(jnp.float32)) - 1.0)
            picked.append(
                (
                    tile_sum(jnp.where(hit, cols, 0.0)),
                    tile_sum(jnp.where(hit, probs, 0.0)),
                )
            )
            work = jnp.where(hit, _NEG, work)
            if on_pick is not None:
                on_pick(self._expert_id(jnp.sum(picked[-1][0]).astype(jnp.int32)))
        # 0-d values from here on, so the weight arithmetic matches the reference's.
        picked = [(jnp.sum(c).astype(jnp.int32), jnp.sum(w)) for c, w in picked]
        total = picked[0][1]
        for _, w in picked[1:]:
            total = total + w
        denom = total + 1e-20
        for k, (idx, w) in enumerate(picked):
            self._store_route(k, idx, w, denom, ids_out, w_out)
        return jnp.sum(ties) > 0

    # Benchmark-only (outputs become meaningless): "compact" replaces the compaction by a fixed
    # list of the first ``routes / hosts`` routes as local slots 0.. (no imbalance either);
    # "compact2" runs it twice (the difference to a normal run is its cost).
    debug_compact = ""
    replica_balance = True  # see route_owners

    def _compact_local(self, me, ids_ref, w_ref, lid, lrow, lw, count):
        """This host's routes in selection order (expert slot, row, weight), branch-free:
        every route is stored at the next free index, which advances only for a route
        this host serves (a later route overwrites a skipped one's entries)."""
        if self.debug_compact == "compact":
            n = self.rows * self.layout.dims.top_k // self.layout.topo.hosts
            for i in range(n):
                lid[i], lrow[i], lw[i] = jnp.int32(i), jnp.int32(0), w_ref[i]
            count[0] = jnp.int32(n)
            return
        for _ in range(2 if self.debug_compact == "compact2" else 1):
            self._compact_routes(me, ids_ref, w_ref, lid, lrow, lw, count)

    def _compact_routes(self, me, ids_ref, w_ref, lid, lrow, lw, count):
        lay, d = self.layout, self.layout.dims
        host = me // lay.topo.lanes
        n_routes = self.rows * d.top_k
        n = jnp.int32(0)
        if packed_balance(lay, self.replica_balance):
            # This host's own non-replicable routes never move: they come first, in selection
            # order (the order ``_early_hooks`` starts their DMAs in), then the other served ones.
            fixed = []
            for i in range(n_routes):
                g, loc, f, _ = _packed_route(ids_ref[i], lay)
                fixed.append((g == host) & (f == 0))
                lid[n] = loc
                lrow[n] = jnp.int32(i // d.top_k)
                lw[n] = w_ref[i]
                n = n + fixed[i].astype(jnp.int32)
            quota_p = replica_quota([ids_ref[i] for i in range(n_routes)], lay)
            seen = jnp.int32(0)
            for i in range(n_routes):
                server, slot, seen = replica_serve(ids_ref[i], quota_p, seen, lay)
                lid[n] = slot
                lrow[n] = jnp.int32(i // d.top_k)
                lw[n] = w_ref[i]
                n = n + ((server == host) & ~fixed[i]).astype(jnp.int32)
            count[0] = n
            return
        servers, slots = route_owners(
            [ids_ref[i] for i in range(n_routes)], lay, balance=self.replica_balance
        )
        for i in range(n_routes):
            lid[n] = slots[i]
            lrow[n] = i // d.top_k
            lw[n] = w_ref[i]
            n = n + (servers[i] == host).astype(jnp.int32)
        count[0] = n

    # ----- several rows (speculative verify): routes grouped by expert -----
    @property
    def grouped(self) -> bool:
        """Rows > 1 (and not ``shared_after``): one DMA and one M=R pass per distinct local expert."""
        return self.rows > 1 and self.shared_after < 0

    def grouped_smem(self) -> tuple[int, int]:
        """SMEM sizes the grouped path needs for ``lrow`` (int32: expert slot -> 1 + distinct index,
        zero when unseen; must be zero at kernel entry) and ``lw`` (f32: distinct x row weights)."""
        n = self.rows * self.layout.dims.top_k
        table = self.layout.dims.num_experts if self._distinct_balance else 0
        return self._seen_size + table, n * self.rows

    @property
    def _seen_size(self) -> int:
        return max(self.rows * self.layout.dims.top_k, self.layout.expert_slots)

    # Grouped path: replica balancing over distinct experts. Off by default: at TP32 B8 it cost
    # ~0.7 ms per verify step (10.18 vs 9.45 ms) for no measured gain; without it every route
    # stays on its owner (route-level balancing would split one expert's routes over two hosts).
    distinct_balance = False

    @property
    def _distinct_balance(self) -> bool:
        """B8 replica balancing over distinct experts (``distinct_owners``) instead of routes."""
        return (
            self.grouped
            and self.layout.replica_count > 0
            and self.replica_balance
            and self.distinct_balance
        )

    def _distinct_owners(self, ids, table, rounds: int = 2):
        """``route_owners`` for the grouped path: the balancing unit is a distinct expert (one weight
        load whatever its route count), and all routes to one expert go to the same host. A
        flexible expert (local id < R) of host g moves to host g + 1 while g's quota lasts, in
        first-route order; quotas balance the distinct-expert counts of neighbours like
        ``route_owners``. ``table``: SMEM int32 [num_experts] at ``_seen_size``, zero on entry and
        on return (0 unseen, 1 counted, 2 + server once assigned)."""
        lay = self.layout
        hosts, eh = lay.topo.hosts, lay.experts_per_host
        assert hosts <= 4, hosts  # 8-bit per-host fields of one int32
        base = self._seen_size
        i32 = lambda v: jnp.asarray(v, jnp.int32)
        info = [
            _packed_route(e, lay) for e in ids
        ]  # (owner, local, flexible, field shift)
        own_p, mov_p = i32(0), i32(0)
        for e, (_, _, f, sh) in zip(ids, info):
            new = (table[base + e] == 0).astype(jnp.int32)
            table[base + e] = i32(1)
            own_p = own_p + jax.lax.shift_left(new, sh)
            mov_p = mov_p + jax.lax.shift_left(new * f, sh)
        quota_p = _packed_quota(own_p, mov_p, hosts, rounds)
        servers, slots = [], []
        seen_p = i32(0)
        for e, (g, loc, f, sh) in zip(ids, info):
            t = table[base + e]
            first = (t == 1).astype(jnp.int32)
            before = jax.lax.shift_right_logical(seen_p, sh) & 255
            quota = jax.lax.shift_right_logical(quota_p, sh) & 255
            moved = first * f * (before < quota).astype(jnp.int32)
            server = jnp.where(first == 1, (g + moved) & (hosts - 1), t - 2)
            table[base + e] = server + 2
            seen_p = seen_p + jax.lax.shift_left(first * f, sh)
            servers.append(server)
            slots.append(jnp.where(server != g, eh + loc, loc))
        for e in ids:
            table[base + e] = i32(0)
        return servers, slots

    def _compact_grouped(self, me, ids_ref, w_ref, lid, seen, wmat, count, early=None):
        """This host's distinct experts in first-route order (``lid``) and each one's weight per
        row (``wmat[j * R + row]``, 0 for rows that did not pick it). ``early`` = (w, ring, sems):
        the first ``ring`` distinct experts' weight DMAs start as they are found (``count[1]``)."""
        lay, d, rows = self.layout, self.layout.dims, self.rows
        host = me // lay.topo.lanes
        n_routes = rows * d.top_k
        ids = [ids_ref[i] for i in range(n_routes)]
        if self._distinct_balance:
            servers, slots = self._distinct_owners(ids, seen)
        else:
            servers, slots = route_owners(ids, lay, balance=False)
        count[0] = 0
        for i in range(n_routes):

            @pl.when(servers[i] == host)
            def _(i=i):
                e = slots[i]

                @pl.when(seen[e] == 0)
                def _():
                    n = count[0]
                    lid[n] = e
                    seen[e] = n + 1
                    count[0] = n + 1
                    if early is not None:

                        @pl.when(n < self.ring)
                        def _():
                            for cp in self._expert_copies(*early, n, lid):
                                cp.start()
                            count[1] = n + 1

                    for r in range(rows):
                        wmat[n * rows + r] = 0.0

                j = seen[e] - 1
                wmat[j * rows + i // d.top_k] = w_ref[i]

        @pl.loop(0, count[0])
        def _(j):
            seen[lid[j]] = 0

    def _experts_grouped(
        self, w, ring, sems, lat_vmem, acc_vmem, lid, lrow, wmat, count
    ):
        """Distinct local experts on all R latent rows at once; ``acc += part * w[:, e]``. Per row the
        f32 sum runs over the experts in first-route order (a reordering of the per-route sum).
        Debug "pair_rows" (with ``pair_down``): experts (2p, 2p + 1) finish their down projections
        in one MXU pass (known wrong row 0, see below)."""
        d, rows = self.layout.dims, self.rows
        n = count[0]
        acc_vmem[...] = jnp.zeros(acc_vmem.shape, jnp.float32)
        x = lat_vmem[...].astype(jnp.bfloat16)
        row_id = jax.lax.broadcasted_iota(jnp.int32, (rows, 1), 0)

        def weights(j):
            col = jnp.zeros((rows, 1), jnp.float32)
            for r in range(rows):
                col = jnp.where(row_id == r, wmat[j * rows + r], col)
            return col

        def down_rows(a, slot):
            dot = down_dot_dequant if self.down_dequant else down_dot
            return dot(a, ring[2].at[slot], ring[3].at[slot])

        def down_one_row(j, a, slot):
            """An expert picked by one row only (most at R = 8) runs its down projection on that
            row: 12 group rows instead of 12 * R. A zero weight marks an unpicked row (a picked
            row's weight is a normalized sigmoid score, > 0); the other rows' terms are zeros."""
            picked = [
                (wmat[j * rows + r] != 0.0).astype(jnp.int32) for r in range(rows)
            ]
            one = sum(picked) == 1
            r1 = sum(picked[r] * r for r in range(1, rows))

            @pl.when(one)
            def _():
                a_r = jnp.sum(
                    jnp.where(row_id == r1, a.astype(jnp.float32), 0.0),
                    axis=0,
                    keepdims=True,
                ).astype(jnp.bfloat16)
                part = down_dot(a_r, ring[2].at[slot], ring[3].at[slot])
                acc_vmem[pl.ds(r1, 1), :] += part * wmat[j * rows + r1]

            @pl.when(jnp.logical_not(one))
            def _():
                acc_vmem[...] += down_rows(a, slot) * weights(j)

        def single(j):
            slot = j % self.ring
            self._wait(w, ring, sems, j, lid, 0)
            a = self._act(ring, slot, x)
            self._refill(w, ring, sems, j, lid, n, 0)
            self._wait(w, ring, sems, j, lid, 1)
            if self.single_rows:
                down_one_row(j, a, slot)
            else:
                acc_vmem[...] += down_rows(a, slot) * weights(j)
            self._refill(w, ring, sems, j, lid, n, 1)

        # Paired distinct experts are off: at R = 8 they gave a wrong row 0 in --verify-check
        # (6.8e-2 at the first MoE layer; cause not found) and were slower. Kept for later debugging.
        if not (self.pair_down and "pair_rows" in self.debug):
            pl.loop(0, n)(single)
            return

        @pl.loop(0, n // 2)
        def _(p):
            j0 = 2 * p
            slot = j0 % self.ring
            for j in (j0, j0 + 1):
                self._wait(w, ring, sems, j, lid, 0)
            a0, a1 = self._act(ring, slot, x), self._act(ring, slot + 1, x)
            for j in (j0, j0 + 1):
                self._refill(w, ring, sems, j, lid, n, 0)
                self._wait(w, ring, sems, j, lid, 1)
            p0, p1 = down_dot_pair(
                a0, a1, ring[2].at[pl.ds(slot, 2)], ring[3].at[pl.ds(slot, 2)]
            )
            acc_vmem[...] += p0 * weights(j0)
            acc_vmem[...] += p1 * weights(j0 + 1)
            for j in (j0, j0 + 1):
                self._refill(w, ring, sems, j, lid, n, 1)

        @pl.when(n % 2 == 1)
        def _():
            single(n - 1)

    def _expert_copies(self, w, ring, sems, j, lid):
        e = lid[j]
        slot = j % self.ring
        return [
            pltpu.make_async_copy(src.at[e], dst.at[slot], sems.at[slot, i])
            for i, (src, dst) in enumerate(zip(w, ring))
        ]

    def _wait(self, w, ring, sems, j, lid, part):
        """Waits for expert ``j``'s gate/up (``part`` 0) or down (1) DMAs; without ``split_refill``
        all four at ``part`` 0."""
        if not self.split_refill and part == 1:
            return
        cps = self._expert_copies(w, ring, sems, j, lid)
        for cp in cps if not self.split_refill else cps[2 * part : 2 * part + 2]:
            cp.wait()

    def _refill(self, w, ring, sems, j, lid, n, part):
        """Starts expert ``j + ring``'s gate/up (``part`` 0) or down (1) DMAs into ``j``'s slot;
        without ``split_refill`` all four after the down dot (``part`` 1)."""
        if not self.split_refill and part == 0:
            return

        @pl.when(j + self.ring < n)
        def _():
            cps = self._expert_copies(w, ring, sems, j + self.ring, lid)
            for cp in cps if not self.split_refill else cps[2 * part : 2 * part + 2]:
                cp.start()

    def _experts_prologue(self, w, ring, sems, lid, count):
        """Start the first ``self.ring`` expert weight DMAs as soon as routing is known
        (except the ``count[1]`` the top-k already started)."""
        n = count[0]
        for s in range(self.ring):

            @pl.when((s < n) & (s >= count[1]))
            def _(s=s):
                for cp in self._expert_copies(w, ring, sems, s, lid):
                    cp.start()

    def _early_hooks(self, me, w, ring, sems, lid, count):
        """``(on_pick, undo)`` for ``_route``: a picked expert of this host gets the next
        ring slot (the order ``_compact_local`` reproduces); ``undo`` drains those DMAs."""
        eh = self.layout.experts_per_host
        host = me // self.layout.topo.lanes
        # With balanced replicas a replicable route may move: only the others start here.
        reps = (
            self.layout.replica_count
            if packed_balance(self.layout, self.replica_balance)
            else 0
        )

        def on_pick(e):
            n = count[1]

            @pl.when((e // eh == host) & (n < self.ring) & (e - host * eh >= reps))
            def _():
                lid[n] = e - host * eh
                for cp in self._expert_copies(w, ring, sems, n, lid):
                    cp.start()
                count[1] = n + 1

        def undo():
            for s in range(self.ring):

                @pl.when(s < count[1])
                def _(s=s):
                    for cp in self._expert_copies(w, ring, sems, s, lid):
                        cp.wait()

            count[1] = 0

        return on_pick, undo

    def _act(self, ring, slot, x):
        d = self.layout.dims
        gu = gate_up_dot(x, ring[0].at[slot], ring[1].at[slot])
        return situ_and_mul(gu, beta=d.situ_beta, linear_beta=d.situ_linear_beta)

    def _experts_paired(self, w, ring, sems, lat_vmem, acc_vmem, lid, lrow, lw, count):
        """``_experts`` with pairs (2p, 2p + 1) in adjacent ring slots finished by one down pass;
        an odd last pair runs alone. Accumulation order is unchanged."""
        n = count[0]
        acc_vmem[...] = jnp.zeros(acc_vmem.shape, jnp.float32)
        row_x = lambda j: lat_vmem[pl.ds(lrow[j], 1), :].astype(jnp.bfloat16)

        @pl.loop(0, n // 2)
        def _(p):
            j0 = 2 * p
            slot = j0 % self.ring
            for j in (j0, j0 + 1):
                self._wait(w, ring, sems, j, lid, 0)
            a0, a1 = (
                self._act(ring, slot, row_x(j0)),
                self._act(ring, slot + 1, row_x(j0 + 1)),
            )
            for j in (j0, j0 + 1):
                self._refill(w, ring, sems, j, lid, n, 0)
                self._wait(w, ring, sems, j, lid, 1)
            p0, p1 = down_dot_pair(
                a0, a1, ring[2].at[pl.ds(slot, 2)], ring[3].at[pl.ds(slot, 2)]
            )
            acc_vmem[pl.ds(lrow[j0], 1), :] += p0 * lw[j0]
            acc_vmem[pl.ds(lrow[j0 + 1], 1), :] += p1 * lw[j0 + 1]
            for j in (j0, j0 + 1):
                self._refill(w, ring, sems, j, lid, n, 1)

        @pl.when(n % 2 == 1)
        def _():
            j = n - 1
            slot = j % self.ring
            self._wait(w, ring, sems, j, lid, 0)
            a = self._act(ring, slot, row_x(j))
            self._wait(w, ring, sems, j, lid, 1)
            part = down_dot(a, ring[2].at[slot], ring[3].at[slot])
            acc_vmem[pl.ds(lrow[j], 1), :] += part * lw[j]

    def _experts(
        self, w, ring, sems, lat_vmem, acc_vmem, lid, lrow, lw, count, lo=0, hi=None
    ):
        """Local pairs ``[lo, hi)`` (``hi`` defaults to all); ``lo == 0`` clears the accumulator."""
        d = self.layout.dims
        n = count[0]
        if lo == 0:
            acc_vmem[...] = jnp.zeros(acc_vmem.shape, jnp.float32)
        start = jnp.minimum(lo, n) if lo else 0
        stop = n if hi is None else jnp.minimum(hi, n)

        @pl.loop(start, stop)
        def _(j):
            slot = j % self.ring
            row = lrow[j]
            math = "no_expert_math" not in self.debug
            self._wait(w, ring, sems, j, lid, 0)
            if math:
                x = lat_vmem[pl.ds(row, 1), :].astype(jnp.bfloat16)
                gu = gate_up_dot(x, ring[0].at[slot], ring[1].at[slot])
            self._refill(w, ring, sems, j, lid, n, 0)
            if math:
                a = situ_and_mul(gu, beta=d.situ_beta, linear_beta=d.situ_linear_beta)
            self._wait(w, ring, sems, j, lid, 1)
            if math:
                part = down_dot(a, ring[2].at[slot], ring[3].at[slot])
                acc_vmem[pl.ds(row, 1), :] += part * lw[j]
            self._refill(w, ring, sems, j, lid, n, 1)

    def _shared(self, x, streams, reduce):
        """Shared expert; its partial goes to the reduction as part 1 (each lane's output columns)."""
        lay, d, topo = self.layout, self.layout.dims, self.layout.topo
        if "moe_shared" in self.debug:
            streams["shared_gu"].wait_prefetched()
            streams["shared_down"].wait_prefetched()
            shared = jnp.zeros((self.rows, d.hidden), jnp.float32)
        else:
            sgu = streams["shared_gu"].matmul(x, prefetched=True).astype(jnp.bfloat16)
            act = situ_and_mul(sgu, beta=d.situ_beta, linear_beta=d.situ_linear_beta)
            shared = streams["shared_down"].matmul(act, prefetched=False)
        if "moe_reduce" not in self.debug:
            uc = lay.latent_up_cols
            reduce.start(
                1,
                jnp.stack(
                    [flat(shared[:, l * uc : (l + 1) * uc]) for l in range(topo.lanes)]
                ),
            )

    def _latent_norm(self, y, norm_w):
        d = self.layout.dims
        if self.up_tree:
            var = lane_sum(y * y, tree=True) / y.shape[-1]
        else:
            var = jnp.mean(y * y, axis=-1, keepdims=True)
        normed = bf16(y * jax.lax.rsqrt(var + d.rms_eps))
        return (normed * norm_w.astype(jnp.float32)).astype(jnp.bfloat16)

    # ----- kernel -----
    def kernel(self, *refs):
        """Inputs: x, router_bias, latent_norm, router, latent_down, shared_gu, shared_down, latent_up,
        expert gu / gu_scale / down / down_scale[, the five dense column scales]; outputs: out, ids,
        weights; then scratch."""
        n_in = 17 if self.relax.dense_fp8 else 12
        (
            x_ref,
            bias_ref,
            norm_ref,
            router_hbm,
            ldown_hbm,
            sgu_hbm,
            sdn_hbm,
            lup_hbm,
            egu_hbm,
            egus_hbm,
            edn_hbm,
            edns_hbm,
        ) = refs[:12]
        scales = refs[12:n_in] if self.relax.dense_fp8 else (None,) * 5
        out_ref, ids_ref, w_ref = refs[n_in : n_in + 3]
        topo = self.layout.topo
        me = jax.lax.axis_index("x")
        it = iter(refs[n_in + 3 :])
        per = len(stream_scratch(1, 128, self.relax.dense_dtype))
        streams = {}
        for name, hbm, scale in zip(
            self.streams, (router_hbm, ldown_hbm, sgu_hbm, sdn_hbm, lup_hbm), scales
        ):
            streams[name] = bind_stream(hbm, [next(it) for _ in range(per)], scale)
        ring = [next(it) for _ in range(4)]
        ring_sems = next(it)
        lat_vmem, acc_vmem = next(it), next(it)
        lid, lrow, lw, count = next(it), next(it), next(it), next(it)
        colls = self.bind_colls(me, list(it))

        entry_barrier(me, topo)
        args = (
            me,
            x_ref,
            bias_ref,
            norm_ref,
            egu_hbm,
            egus_hbm,
            edn_hbm,
            edns_hbm,
            out_ref,
            ids_ref,
            w_ref,
            streams,
            ring,
            ring_sems,
            lat_vmem,
            acc_vmem,
            lid,
            lrow,
            lw,
            count,
            colls,
        )
        if self.repeat == 1:
            self._body(*args)
        else:

            @pl.loop(0, self.repeat)
            def _(i):
                self._body(*args)

    def _body(
        self,
        me,
        x_ref,
        bias_ref,
        norm_ref,
        egu_hbm,
        egus_hbm,
        edn_hbm,
        edns_hbm,
        out_ref,
        ids_ref,
        w_ref,
        streams,
        ring,
        ring_sems,
        lat_vmem,
        acc_vmem,
        lid,
        lrow,
        lw,
        count,
        colls,
        after_route=None,
    ):
        """``after_route`` runs once the first routed-expert DMAs are queued (the stack's
        cross-layer prefetch point)."""
        streams["router"].prefetch()
        streams["latent_down"].prefetch()
        streams["shared_gu"].prefetch()
        c = _Ctx(
            me=me,
            x=x_ref[...],
            bias_ref=bias_ref,
            norm_ref=norm_ref,
            egu=(egu_hbm, egus_hbm, edn_hbm, edns_hbm),
            out_ref=out_ref,
            ids_ref=ids_ref,
            w_ref=w_ref,
            streams=streams,
            ring=ring,
            ring_sems=ring_sems,
            lat_vmem=lat_vmem,
            acc_vmem=acc_vmem,
            lid=lid,
            lrow=lrow,
            lw=lw,
            count=count,
            colls=colls,
            after_route=after_route,
        )
        with phase("moe_route"):
            self._route_phase(c)
        with phase("moe_latent_shared"):
            self._latent_shared_phase(c)
        with phase("moe_experts"):
            self._experts_phase(c)
        with phase("moe_reduce"):
            routed, shared = self._reduce_phase(c)
        with phase("moe_up"):
            self._up_phase(c, routed, shared)

    def _route_phase(self, c: "_Ctx"):
        """Router scores -> top-k -> this host's (expert, row) pairs; starts the first expert DMAs."""
        pair, lat_gather = c.colls[0], c.colls[1]
        skip_route, skip_experts = (
            "moe_route" in self.debug,
            "moe_experts" in self.debug,
        )
        if skip_route:
            c.streams["router"].wait_prefetched()
        else:
            self._route_start(c.x, c.streams["router"], pair)
        if self.overlap_pair:
            lat_gather.start(
                self._gather_in(
                    bf16(c.streams["latent_down"].matmul(c.x, prefetched=True))
                )
            )
        c.count[1] = 0
        early = None
        # With balanced replicas ``_early_hooks`` starts only routes that cannot move.
        if (
            self.early_dma
            and not skip_experts
            and (
                self.layout.replica_count == 0
                or not self.replica_balance
                or packed_balance(self.layout, True)
            )
        ):
            early = self._early_hooks(c.me, c.egu, c.ring, c.ring_sems, c.lid, c.count)
        if skip_route:
            self._fixed_route(c.ids_ref, c.w_ref)
        else:
            self._route(c.bias_ref, pair, c.ids_ref, c.w_ref, early)
        if self.grouped:
            for _ in range(2 if self.debug_compact == "compact2" else 1):
                early_g = (
                    (c.egu, c.ring, c.ring_sems)
                    if self.early_dma and not skip_experts and _ == 0
                    else None
                )
                self._compact_grouped(
                    c.me, c.ids_ref, c.w_ref, c.lid, c.lrow, c.lw, c.count, early_g
                )
        else:
            self._compact_local(c.me, c.ids_ref, c.w_ref, c.lid, c.lrow, c.lw, c.count)
        if skip_experts:
            c.count[0] = 0
        self._experts_prologue(c.egu, c.ring, c.ring_sems, c.lid, c.count)
        if c.after_route is not None and self.handoff == "route":
            c.after_route()

    def _fixed_route(self, ids_out, w_out):
        """Benchmark-only routing: experts spread evenly over the hosts, equal weights."""
        d = self.layout.dims
        for row in range(self.rows):
            for k in range(d.top_k):
                ids_out[row * d.top_k + k] = jnp.int32(k * (d.num_experts // d.top_k))
                w_out[row * d.top_k + k] = jnp.float32(1.0 / d.top_k)

    def _latent_shared_phase(self, c: "_Ctx"):
        """latent_down -> latent all-gather into ``lat_vmem``; the shared expert (its reduction
        flies during the expert stream) unless ``shared_after`` moves it."""
        lay, d, r, topo = self.layout, self.layout.dims, self.rows, self.layout.topo
        lat_gather, reduce = c.colls[1], c.colls[2]
        if not self.overlap_pair:
            lat_gather.start(
                self._gather_in(
                    bf16(c.streams["latent_down"].matmul(c.x, prefetched=True))
                )
            )
        if self.late_latent_up == "latent":
            c.streams["latent_up"].prefetch(c.streams["latent_up"].slots)
        if self.shared_after < 0:
            self._shared(c.x, c.streams, reduce)
        if c.after_route is not None and self.handoff == "shared":
            c.after_route()
        c.lat_vmem[...] = self._gathered(
            lat_gather.finish(), lay.latent_chunk, topo.world
        )[:, : d.latent]
        if self.late_latent_up != "reduce":
            c.streams["latent_up"].prefetch()

    def _experts_phase(self, c: "_Ctx"):
        """Routed experts of this host into ``acc_vmem`` (f32, route-weighted)."""
        expert_args = (
            c.egu,
            c.ring,
            c.ring_sems,
            c.lat_vmem,
            c.acc_vmem,
            c.lid,
            c.lrow,
            c.lw,
            c.count,
        )
        if self.grouped:
            self._experts_grouped(*expert_args)
        elif self.shared_after < 0:
            (self._experts_paired if self.pair_down else self._experts)(*expert_args)
        else:
            self._experts(*expert_args, hi=self.shared_after)
            self._shared(c.x, c.streams, c.colls[2])
            self._experts(*expert_args, lo=self.shared_after)
        if self.late_latent_up == "reduce":
            c.streams["latent_up"].prefetch(c.streams["latent_up"].slots)
        if c.after_route is not None and self.handoff == "experts":
            c.after_route()

    def _reduce_phase(self, c: "_Ctx"):
        """Returns the bf16-valued routed latent ``[R, latent]`` and this lane's shared output
        columns ``[R, latent_up_cols]``."""
        d, r, uc = self.layout.dims, self.rows, self.layout.latent_up_cols
        if "moe_reduce" in self.debug:
            return bf16(c.acc_vmem[...]), jnp.zeros((r, uc), jnp.float32)
        reduce = c.colls[2]
        if not self.row_reduce:
            reduce.start(0, flat(c.acc_vmem[...]))
            routed, shared_cols = reduce.finish()
            return bf16(unflat(routed, r, d.latent)), bf16(unflat(shared_cols, r, uc))
        lanes = self.layout.topo.lanes
        rpl = r // lanes
        acc = c.acc_vmem[...]
        reduce.start(
            0, jnp.stack([flat(acc[j * rpl : (j + 1) * rpl]) for j in range(lanes)])
        )
        own, shared_cols = reduce.finish()
        own = bf16(unflat(own, rpl, d.latent))  # this lane's routed rows
        routed = c.colls[4](own.reshape(1, rpl * d.latent)).reshape(r, d.latent)
        return routed, bf16(unflat(shared_cols, r, uc))

    def _up_phase(self, c: "_Ctx", routed, shared_cols):
        """latent RMSNorm -> latent_up (this lane's columns) + shared -> lane all-gather -> ``out_ref``."""
        lay, r, topo = self.layout, self.rows, self.layout.topo
        if "moe_up" in self.debug:
            c.streams["latent_up"].wait_prefetched()
            c.out_ref[...] = jnp.zeros(c.out_ref.shape, c.out_ref.dtype)
            return
        uc = lay.latent_up_cols
        if "moe_up_dot" in self.debug:
            c.streams["latent_up"].wait_prefetched()
            up = routed[:, :uc]
        else:
            up = bf16(
                c.streams["latent_up"].matmul(
                    self._latent_norm(routed, c.norm_ref[...]), prefetched=True
                )
            )
        out_cols = bf16(up + shared_cols)
        if "moe_up_gather" in self.debug:
            c.out_ref[...] = jnp.concatenate([out_cols] * topo.lanes, axis=1).astype(
                jnp.bfloat16
            )
            return
        c.out_ref[...] = self._gathered(
            c.colls[3](self._gather_in(out_cols)), uc, topo.lanes
        ).astype(jnp.bfloat16)

    def build(self, mesh):
        """jit(shard_map(pallas_call)) over ``mesh`` axis "x"; inputs carry a leading rank axis."""
        from jax.sharding import PartitionSpec as P

        d, r = self.layout.dims, self.rows
        vm = pl.BlockSpec(memory_space=pltpu.VMEM)
        hbm = pl.BlockSpec(memory_space=pltpu.HBM)
        sm = pl.BlockSpec(memory_space=pltpu.SMEM)
        call = pl.pallas_call(
            self.kernel,
            out_shape=(
                jax.ShapeDtypeStruct((r, d.hidden), jnp.bfloat16),
                jax.ShapeDtypeStruct((r * d.top_k,), jnp.int32),
                jax.ShapeDtypeStruct((r * d.top_k,), jnp.float32),
            ),
            in_specs=[vm, vm, vm] + [hbm] * (14 if self.relax.dense_fp8 else 9),
            out_specs=(vm, sm, sm),
            scratch_shapes=self._scratch(),
            compiler_params=pltpu.CompilerParams(
                collective_id=self.collective_id, vmem_limit_bytes=64 * 1024 * 1024
            ),
        )

        n_in = 17 if self.relax.dense_fp8 else 12

        def local(*args):
            out, ids, w = call(*[a[0] for a in args])
            return out[None], ids[None], w[None]

        spec = P("x")
        return jax.jit(
            jax.shard_map(
                local,
                mesh=mesh,
                in_specs=(spec,) * n_in,
                out_specs=(spec, spec, spec),
                check_vma=False,
            )
        )
