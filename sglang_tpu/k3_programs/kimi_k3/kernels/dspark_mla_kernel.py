# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused TP32 kernel of the MLA DSpark draft (``kernels.dspark_mla``).

One call per decode step after the verify, over ``R`` requests with four or
eight verify rows each, on the decoder's 32-rank mesh:

1. **Acceptance.** The verify's vocabulary-sharded logits give each row's
   target (one 31-peer exchange); the accepted count is the leading drafts
   that match, the bonus token (the next anchor) the target after them
   (either may be forced by the caller), as ``kernels.dspark``'s kernel.
2. **Context.** The verify rows' aux features -> ``context_norm(
   context_proj(aux))`` (host-sharded fc and reductions, as there).
3. **Five layers** over the block (the anchor's embedding then mask
   embeddings, at the anchor's position on): RMSNorm, the fused q_a | kv_a
   projection (each rank computes one 128-lane tile and the ranks exchange
   them; the context rows' kv_a for every layer once, before the layers),
   q_a_layernorm,
   this rank's heads' q_b, RoPE, the absorbed queries; the block and
   context rows' latents; DCP attention (``mla_dcp.DcpAttention``) over the
   draft pools' pages before the verify's first position, with the
   request's new keys (its accepted context rows and its block rows,
   non-causal); W_UV, o_proj (reduce-scattered), the MLP.
4. **Drafts.** The draft's final norm, the target's LM head (vocabulary
   sharded) and exact left-to-right Markov sampling.

Outputs: each request's ``[128]`` int32 vector (``kernels.dspark.OUT_*``
lanes: drafts, targets, count, bonus, margins) and the verify rows'
context latents ``[T, layers, 640]`` for the caller to commit to the draft
pools (``paged.commit_latent_rows``).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from sglang_tpu.k3_programs.common import collectives32
from sglang_tpu.k3_programs.kimi_k3.config import active as _config

from . import mla_dcp
from .decode_megakernel import PHASE_SCOPES, _dot, _phase_end, _phase_start
from .dspark import (
    NEG,
    OUT_BONUS,
    OUT_COUNT,
    OUT_DRAFTS,
    OUT_MARGINS,
    OUT_TARGETS,
    SEM_BCAST,
    SEM_GATHER,
    SEM_HOST,
    SEM_RS,
    SEM_TINY,
    SEMS_PER_SLOT,
    _bf16,
    _f32,
    _rms_norm,
    _select_rows,
    pack,
    unpack,
)
from .dspark_mla import (
    CHUNK,
    CONTEXT_TILES,
    FC_IN,
    LANES,
    MLP_PADDED,
    QKV_TILES,
    RANKS,
    ROWS,
    TILE,
    MlaDraftConfig,
)

LM_CHUNK = 256
LM_CHUNKS = 20  # 20 * 256 = 5120 vocabulary rows per rank
ROW_PACKED = 7168 // 256  # 28 FP32 rows hold one BF16 [1, 7168] row
Q_TILES = QKV_TILES - CONTEXT_TILES  # the q_a tiles; then kv_a's

VMEM_NAMES = (
    "ln1",
    "ln2",
    "q_norm",
    "kv_norm",
    "context_norm",
    "final_norm",
    "mask_embed",
    "rot",
)
HBM_NAMES = (
    "qkv_tile",
    "context_tile",
    "q_b",
    "w_uk",
    "w_uv",
    "o",
    "gate",
    "up",
    "down",
    "fc",
    "w2t",
    "w1",
    "rope",
    "lm_head",
    "embed",
)
WEIGHT_NAMES = VMEM_NAMES + HBM_NAMES
SMALL_NAMES = ("aux", "previous") + VMEM_NAMES
# The layer-norm stacks ``[layers, 1, width]``. XLA lays their unit axis out
# of the tiled dims, so passing them as such costs a relayout copy per weight
# per call; a single-request op (``SingleEpilogue``) passes them as ``[layers,
# width]`` (a bitcast of that layout) and the kernel reads row ``layer``.
LEADING_NAMES = ("ln1", "ln2", "q_norm", "kv_norm")


@dataclasses.dataclass(frozen=True)
class SingleEpilogue:
    """A single-request draft op's control layout (``decode_ops.Control``,
    plan row 0). With it the fused kernel reads ``MlaDsparkDraft.
    _build_single``'s packed ``ids`` (``[tokens | positions | slots |
    accepted | tables]``, one entry per verify row, then a block-table row
    each) and the decisions directly, and ``single_epilogue`` computes the
    op's emitted tokens, next ids and control state in a kernel (the draft
    commit's, ``paged.commit_draft_rows``)."""

    slots: int  # control banks: tokens | positions | rows, one entry a slot
    reset: int  # the control state index of plan row 0's reset flag
    state_size: int
    ids_size: int
    # KernelConfig.draft_verify_packet: plan row 0's ``fields`` control
    # entries from index ``plan``, which the next verify's packet carries
    # after the ids (``fields`` 0: no packet).
    plan: int = 0
    fields: int = 0


def single_epilogue(
    spec, rows, ids, state_in, vector, emitted, following, state_out, packet=None
):
    """``MlaDsparkDraft._build_single``'s step outputs as SMEM scalar code:
    ``ids`` and ``state_in`` the op's packed ids and control state, ``vector``
    the step's output vector ``[1+, 128]`` (``OUT_*`` lanes); writes
    ``emitted`` ``[1, rows]`` (the accepted drafts, then the bonus token,
    then -1), ``following`` (the next verify's ids: the bonus and the drafts,
    their positions, the slots, the accepted count, the tables) and
    ``state_out`` (the slot's token, position and row entries; plan row 0's
    reset flag cleared on a live slot). The control writes are XLA's
    ``.at[i].set``: a negative slot index wraps, one out of range is dropped,
    and they land in the same order. ``packet`` (``spec.fields``): the next
    verify's packed operand as ``DecodeOps._build_verify_metadata`` builds
    it from ``following``, ``state_out`` and verify mode 1 -- ``following``,
    then ``state_out``'s plan row 0, then 1."""
    accepted = vector[0, OUT_COUNT]
    bonus = vector[0, OUT_BONUS]
    base, slot = ids[rows], ids[2 * rows]
    for r in range(rows):
        emitted[0, r] = jnp.where(
            r < accepted,
            ids[min(r + 1, rows - 1)],
            jnp.where(r == accepted, bonus, jnp.int32(-1)),
        )
    outs = (following,) if packet is None else (following, packet)

    def put_ids(index, value):
        for out in outs:
            out[index] = value

    put_ids(0, bonus)
    for j in range(rows - 1):
        put_ids(1 + j, vector[0, OUT_DRAFTS + j])
    for r in range(rows):
        put_ids(rows + r, base + accepted + 1 + r)
        put_ids(2 * rows + r, ids[2 * rows + r])
    put_ids(3 * rows, accepted)

    @pl.loop(3 * rows + 1, spec.ids_size)
    def copy_tables(k):
        put_ids(k, ids[k])

    @pl.loop(0, spec.state_size)
    def copy_state(k):
        state_out[k] = state_in[k]

    def put(index, value):
        index = jnp.where(index < 0, index + spec.state_size, index)

        @pl.when((index >= 0) & (index < spec.state_size))
        def store():
            state_out[index] = value

    put(slot, bonus)
    put(spec.slots + slot, base + accepted + 1)
    put(2 * spec.slots + slot, accepted)
    state_out[spec.reset] = jnp.where(slot != 0, 0, state_out[spec.reset])
    if packet is not None:
        for f in range(spec.fields):
            packet[spec.ids_size + f] = state_out[spec.plan + f]
        packet[spec.ids_size + spec.fields] = jnp.int32(1)


class _Scalars:
    """An SMEM ref viewed from ``offset``; ``rows`` > 1 maps index ``i`` to
    entry ``i // rows`` (every row of a request reads its request's)."""

    def __init__(self, ref, offset, rows=1):
        self.ref, self.offset, self.rows = ref, offset, rows

    def __getitem__(self, index):
        return self.ref[self.offset + (index // self.rows if self.rows > 1 else index)]


def stream_chunks(chunk_copy, slots, chunks, started=0):
    """Stream ``chunks`` weight chunks through the ``slots`` ring: chunk ``k``
    lands in slot ``k % len(slots)`` (``chunk_copy(k, slot)``: its copy; the
    first ``started`` copies are already in flight). Yields ``(k, slot ref)``
    once chunk ``k`` has landed and refills that slot with chunk ``k +
    len(slots)`` when the consumer resumes it (after its reads)."""
    ring = len(slots)
    copies = [chunk_copy(k, k % ring) for k in range(min(ring, chunks))]
    for item in copies[started:]:
        item.start()
    for k in range(chunks):
        copies[k % ring].wait()
        yield k, slots[k % ring]
        if k + ring < chunks:
            copies[k % ring] = chunk_copy(k + ring, k % ring)
            copies[k % ring].start()


def _rope(x_bf16, rot_bf16, cos32, sin32):
    """Interleaved-pair RoPE in FP32 (``x * cos2 + (x @ R) * sin2``), BF16 out."""
    rotated = _dot(x_bf16, rot_bf16)  # exact: one +-1 term per lane
    return _bf16(_f32(x_bf16) * cos32 + rotated * sin32)


def streamed_weights(
    c: MlaDraftConfig, rows: int, dp: int = 1, whole: bool = False
) -> tuple[bool, bool]:
    """``(o, down)``: whether a call over ``rows`` rows (all its requests')
    streams that weight one 896-column chunk at a time through two VMEM slots
    instead of holding it whole. Both stream above 32 rows and with more than
    two heads a rank (Kimi-K3.1's draft: three), o also under attention DP;
    ``whole`` (``KernelConfig.draft_whole_weights``, attention DP1) holds o
    whole up to 8 rows and down up to 16, as VMEM allows."""
    whole = whole and dp == 1
    many = c.heads_per_rank > 2
    o = rows > 32 or dp > 1 or (many and not (whole and rows <= 8))
    down = rows > 32 or (many and not (whole and rows <= 16))
    return o, down


class _ChunkedWeight:
    """A ``[layers, K, 7168]`` o / down weight (chunk-permuted, see
    ``all_reduce``) through ``buffer``: whole (``[K, 7168]``) or streamed one
    896-column memory chunk at a time through two slots (``[2, K, 896]``),
    each slot's copies on its semaphore (``sems``)."""

    def __init__(self, weight, buffer, sems, copy):
        self.weight, self.buffer, self.sems, self.copy = weight, buffer, sems, copy
        self.streamed = len(buffer.shape) == 3

    def load(self, layer):
        """Start ``layer``'s copy: the whole weight, or streamed its chunk 0
        into slot 0."""
        item = self.copy(
            self.weight.at[layer, :, pl.ds(0, self.buffer.shape[-1])],
            self.buffer.at[0] if self.streamed else self.buffer,
            self.sems[0],
        )
        item.start()
        return item

    def chunks(self, layer):
        """``layer``'s memory chunks ``[K, 896]`` in order, once ``load``'s
        copy has arrived (streamed: chunk k + 1's copy in flight while chunk k
        is used)."""
        if not self.streamed:
            for k in range(8):
                yield self.buffer[:, pl.ds(k * CHUNK, CHUNK)]
            return
        refill = self.weight.at[layer]
        loading = None
        for k in range(8):
            arrived = loading
            if k + 1 < 8:
                loading = self.copy(
                    refill.at[:, pl.ds((k + 1) * CHUNK, CHUNK)],
                    self.buffer.at[(k + 1) % 2],
                    self.sems[(k + 1) % 2],
                )
                loading.start()
            if arrived is not None:
                arrived.wait()
            yield self.buffer[k % 2]


def _make_body(
    c: MlaDraftConfig,
    block: int,
    requests: int,
    rows: int,
    pages: int,
    owner_table,
    plan: dict,
    dp: int = 1,
    single: SingleEpilogue | None = None,
):
    R = requests
    T = R * rows
    L = c.layers
    per = c.heads_per_rank
    heads = per * dp  # the held TP32 shards' heads (attention DP)
    group_requests, group_rows = R // dp, T // dp
    q_shard = per * (
        c.nope + TILE
    )  # a shard's q_b columns: nopes, then ropes padded to a tile
    vocab = c.vocab // RANKS
    hidden = c.hidden
    eps = c.eps
    scale = c.softmax_scale
    drafts_count = rows - 1  # verify rows after the anchor
    KEYS = 2 * rows
    n_persist = len(mla_dcp.PERSISTENT_NAMES)
    n_scoped = len(mla_dcp.SCOPED_NAMES)

    SEM_FC, SEM_W2T, SEM_OWN, SEM_LM = 1, 8, 9, 11  # 10: unused
    lm_prefetch = _config().draft_lm_prefetch
    SEM_EMBED = 12  # + j
    SEM_ROPE = 12 + R  # + 2 j (+1)
    SEM_W1 = 12 + 3 * R  # + slot R + j
    SEM_LAT = 12 + 5 * R
    SEM_DOWN = 13 + 5 * R
    SEM_TILE = 14 + 5 * R  # + slot
    SEM_LM2 = 16 + 5 * R
    SEM_O2, SEM_DOWN2 = (
        17 + 5 * R,
        18 + 5 * R,
    )  # streamed o / down: the second chunk slot

    def body(*args):
        if single is None:
            scalars, aux_hbm, logits_hbm, previous_hbm, *refs = args
            ids = decisions = None
        else:
            ids, decisions, aux_hbm, logits_hbm, previous_hbm, *refs = args
        weights = dict(zip(WEIGHT_NAMES, refs[: len(WEIGHT_NAMES)], strict=True))
        refs = refs[len(WEIGHT_NAMES) :]
        pools = tuple(refs[:L])
        refs = refs[L:]
        tokens_out, latents_out = refs[:2]
        scratch = refs[2:]
        small = dict(zip(SMALL_NAMES, scratch[: len(SMALL_NAMES)], strict=True))
        scratch = scratch[len(SMALL_NAMES) :]

        def norm_row(name, layer):
            """Layer ``layer``'s ``[1, width]`` row of a layer-norm stack
            (``[layers, 1, width]``, or ``[layers, width]``: LEADING_NAMES)."""
            ref = small[name]
            return ref[layer] if len(ref.shape) == 3 else ref[layer : layer + 1, :]

        persistent = scratch[:n_persist]
        dcp_scoped = scratch[n_persist : n_persist + n_scoped]
        (
            q_buf,
            w_uk_buf,
            w_uv_buf,
            o_buf,
            gate_buf,
            up_buf,
            down_buf,
            w2t_buf,
            w1_blk,
            embed_blk,
            bcast_buf,
            rope_blk,
            logits_vmem,
            fc_part,
            tokens_v,
            stage,
            local_recv,
            host_packed,
            gather_recv,
            tiny_recv,
            argmax_recv,
            keys_vmem,
            valid_vmem,
            latents_vmem,
            tile_buf,
            tile_recv,
            stage2_recv,
            stage2_send,
            stage2_sems,
            dma_sems,
            small_sems,
            sends,
            recvs,
            *pair_sems,  # KernelConfig.draft_pair_combine: the combine's pair stage
        ) = scratch[n_persist + n_scoped :]

        rank = jax.lax.axis_index("tp")
        lane = rank % 8
        host = rank // 8
        data_group = rank // (RANKS // dp)

        def own_group(value):
            """This rank's DP group's share of ``value`` (group-major rows)."""
            if dp == 1:
                return value
            parts = jnp.split(value, dp, axis=0)
            selected = parts[0]
            for g in range(1, dp):
                selected = jnp.where(data_group == g, parts[g], selected)
            return selected

        def place_group(value):
            """``own_group``'s inverse: the group's rows in place, zeros elsewhere."""
            if dp == 1:
                return value
            zeros = jnp.zeros_like(value)
            return jnp.concatenate(
                [jnp.where(data_group == g, value, zeros) for g in range(dp)], axis=0
            )

        if single is None:
            new_position = jnp.stack(
                [scalars[j] for j in range(R)]
            )  # [R]: the verify's first position
            forced_count = jnp.stack([scalars[R + j] for j in range(R)])
            forced_anchor = jnp.stack([scalars[2 * R + j] for j in range(R)])
            table_scalars = _Scalars(scalars, 3 * R)
        else:  # one request: the packed ids and decisions
            new_position = jnp.stack([ids[rows]])
            forced_count = jnp.stack([decisions[0, 0]])
            forced_anchor = jnp.stack([decisions[0, 1]])
            table_scalars = _Scalars(ids, 3 * rows + 1)

        # ---------------- DMA and collective helpers ----------------
        def copy(src, dst, sem):
            return tpu.make_async_copy(src, dst, dma_sems.at[sem])

        def remote(src, dst, sem_index, peer):
            return tpu.make_async_remote_copy(
                src,
                dst,
                sends.at[sem_index],
                recvs.at[sem_index],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )

        def exchange(src_of, dst_of, offsets, base, slot):
            copies = [
                remote(src_of(o), dst_of(o), slot * SEMS_PER_SLOT + base + o, rank ^ o)
                for o in offsets
            ]
            for item in copies:
                item.start()
            for item in copies:
                item.wait()

        def host_reduce_gather(chunk32, slot, tag="fc"):
            """FP32 [T, 896] partial of this host lane -> the [T, 7168] sum over
            hosts, BF16-rounded."""
            scope = _phase_start(f"ar_{tag}_host")
            host_packed[slot, host] = pack(_bf16(chunk32))
            exchange(
                lambda o: host_packed.at[slot, host],
                lambda o: host_packed.at[slot, host],
                (8, 16, 24),
                SEM_HOST,
                slot,
            )
            reduced = sum(unpack(host_packed[slot, h], T, CHUNK) for h in range(4))
            _phase_end(scope)
            scope = _phase_start(f"ar_{tag}_gather")
            gather_recv[slot, lane] = pack(_bf16(reduced))
            exchange(
                lambda o: gather_recv.at[slot, lane],
                lambda o: gather_recv.at[slot, lane],
                range(1, 8),
                SEM_GATHER,
                slot,
            )
            result = jnp.concatenate(
                [unpack(gather_recv[slot, j], T, CHUNK) for j in range(8)], axis=1
            )
            _phase_end(scope)
            return result

        def all_reduce(lhs_bf16, chunks, slot, tag=""):
            """``lhs @ W`` (a chunk-permuted ``[K, 7168]`` weight: memory chunk
            k is logical chunk ``lane ^ k``; ``chunks``: its memory chunks in
            order, ``_ChunkedWeight.chunks``) summed over the 32 ranks: the
            host-local reduce-scatter chunk by chunk, then the cross-host
            reduction and gather."""
            scope = _phase_start(f"ar_{tag}_local")
            copies = []
            own = None
            for k, operand in enumerate(chunks):
                partial = _dot(lhs_bf16, operand)  # [T, 896] FP32
                stage[slot, k] = pack(_bf16(partial))
                if k == 0:
                    own = copy(stage.at[slot, 0], local_recv.at[slot, lane], SEM_OWN)
                    own.start()
                else:
                    item = remote(
                        stage.at[slot, k],
                        local_recv.at[slot, lane],
                        slot * SEMS_PER_SLOT + SEM_RS + k,
                        rank ^ k,
                    )
                    item.start()
                    copies.append(item)
            _phase_end(scope)
            scope = _phase_start(f"ar_{tag}_rs")
            for item in copies:
                item.wait()
            assert own is not None
            own.wait()
            chunk = sum(unpack(local_recv[slot, k], T, CHUNK) for k in range(8))
            _phase_end(scope)
            return host_reduce_gather(chunk, slot, tag)

        def gather_tiles(tile_bf16, slot, count):
            """This rank's ``[T, 128]`` projection tile to every rank; returns
            ranks ``0 .. count - 1``'s tiles (BF16, as FP32 ``[T, 128]``)."""
            tile_recv[slot, rank] = pack(tile_bf16)
            # Only ranks below ``count`` hold a tile: they send, and each rank
            # waits for those peers only.
            copies = [
                remote(
                    tile_recv.at[slot, rank],
                    tile_recv.at[slot, rank],
                    slot * SEMS_PER_SLOT + SEM_TINY + o,
                    rank ^ o,
                )
                for o in range(1, 32)
            ]

            @pl.when(rank < count)
            def send_tile():
                for item in copies:
                    item.start()

            for o, item in zip(range(1, 32), copies, strict=True):
                pl.when((rank ^ o) < count)(item.wait_recv)

            @pl.when(rank < count)
            def sent_tile():
                for item in copies:
                    item.wait_send()

            return [unpack(tile_recv[slot, i], T, TILE) for i in range(count)]

        def tile_copy(source, slot):
            return copy(source, tile_buf.at[slot], SEM_TILE + slot)

        def tiny_exchange(row_i32, slot):
            tiny_recv[slot, rank] = row_i32
            exchange(
                lambda o: tiny_recv.at[slot, rank],
                lambda o: tiny_recv.at[slot, rank],
                range(1, 32),
                SEM_TINY,
                slot,
            )
            return tiny_recv[slot]

        # ---------------- prologue ----------------
        collectives32.barrier()
        sources = {
            "aux": aux_hbm,
            "previous": previous_hbm,
            **{n: weights[n] for n in VMEM_NAMES},
        }
        small_copies = [
            tpu.make_async_copy(sources[n], small[n], small_sems.at[i])
            for i, n in enumerate(SMALL_NAMES)
        ]
        for item in small_copies:
            item.start()
        logits_copy = copy(logits_hbm, logits_vmem, 0)
        logits_copy.start()
        w2t_copy = copy(weights["w2t"], w2t_buf, SEM_W2T)
        w2t_copy.start()
        tile_copies = [
            tile_copy(weights["qkv_tile"].at[0], 0),
            tile_copy(weights["context_tile"], 1),
        ]
        for item in tile_copies:
            item.start()

        # ---------------- verify acceptance (as kernels.dspark) ----------------
        scope = _phase_start("d_accept")
        for item in small_copies:
            item.wait()
        logits_copy.wait()
        verify_logits = logits_vmem[...]  # [T, 5120] FP32
        vocab_iota = jax.lax.broadcasted_iota(jnp.int32, (T, vocab), 1)
        local_best = jnp.max(verify_logits, axis=1, keepdims=True)
        local_id = jnp.min(
            jnp.where(verify_logits == local_best, vocab_iota, vocab),
            axis=1,
            keepdims=True,
        )
        local_second = jnp.max(
            jnp.where(vocab_iota == local_id, NEG, verify_logits), axis=1, keepdims=True
        )
        out_lane = jax.lax.broadcasted_iota(jnp.int32, (T, 128), 1)
        packet = jnp.where(
            out_lane == 0,
            jax.lax.bitcast_convert_type(local_best, jnp.int32),
            jnp.where(
                out_lane == 1,
                local_id + rank * vocab,
                jnp.where(
                    out_lane == 2,
                    jax.lax.bitcast_convert_type(local_second, jnp.int32),
                    0,
                ),
            ),
        )
        # Rows along lanes: one [8, 128] tile a peer (lane t: row t; sublanes
        # best, id, second), not [T, 128].
        square = jnp.pad(packet, ((0, 128 - T), (0, 0)))
        argmax_recv[rank] = jnp.transpose(square)[:8]
        exchange(
            lambda o: argmax_recv.at[rank],
            lambda o: argmax_recv.at[rank],
            range(1, 32),
            SEM_TINY,
            1,
        )
        gathered = argmax_recv[...]  # [RANKS, 8, 128]
        rank_iota3 = jax.lax.broadcasted_iota(jnp.int32, (RANKS, 1, 128), 0)
        bests = jax.lax.bitcast_convert_type(gathered[:, 0:1, :], jnp.float32)
        seconds = jax.lax.bitcast_convert_type(gathered[:, 2:3, :], jnp.float32)
        best = jnp.max(bests, axis=0)  # [1, 128]
        winner = jnp.min(jnp.where(bests == best[None], rank_iota3, RANKS), axis=0)
        target_lanes = jnp.sum(
            jnp.where(rank_iota3 == winner[None], gathered[:, 1:2, :], 0), axis=0
        )
        runner_up = jnp.maximum(
            jnp.max(seconds, axis=0),
            jnp.max(jnp.where(rank_iota3 == winner[None], NEG, bests), axis=0),
        )
        lanes_back = jnp.concatenate(
            (
                target_lanes,
                jax.lax.bitcast_convert_type(best - runner_up, jnp.int32),
                jnp.zeros((126, 128), jnp.int32),
            ),
            axis=0,
        )
        rows_back = jnp.transpose(lanes_back)[
            :T
        ]  # [T, 128]: lane 0 target, lane 1 margin
        targets = rows_back[:, 0:1].reshape(R, rows, 1)
        margins = jax.lax.bitcast_convert_type(rows_back[:, 1:2], jnp.float32).reshape(
            R, rows, 1
        )
        previous_vector = small["previous"][...]  # [R, 128]
        row_iota_col = jax.lax.broadcasted_iota(jnp.int32, (R, rows, 1), 1)
        proposals = jnp.sum(
            jnp.where(
                jax.lax.broadcasted_iota(jnp.int32, (R, rows, 128), 2)
                == row_iota_col + OUT_DRAFTS,
                jnp.broadcast_to(previous_vector[:, None, :], (R, rows, 128)),
                0,
            ),
            axis=2,
            keepdims=True,
        )  # proposal k was verify row k + 1
        match_rows = (row_iota_col < drafts_count) & (proposals == targets)
        first_miss = jnp.min(
            jnp.where(match_rows | (row_iota_col >= drafts_count), rows, row_iota_col),
            axis=1,
        )[:, 0]
        count = jnp.where(
            forced_count >= 0, forced_count, jnp.minimum(first_miss, drafts_count)
        )  # [R]
        anchor_token = jnp.sum(
            jnp.where(row_iota_col[:, :, 0] == count[:, None], targets[:, :, 0], 0),
            axis=1,
        )
        anchor_token = jnp.where(forced_anchor >= 0, forced_anchor, anchor_token)
        anchor_position = new_position + count + 1  # [R]
        result_lane = jax.lax.broadcasted_iota(jnp.int32, (R, 128), 1)
        field_row = jax.lax.broadcasted_iota(jnp.int32, (R, rows, 128), 1)
        field_lane = jax.lax.broadcasted_iota(jnp.int32, (R, rows, 128), 2)
        targets_row = jnp.sum(
            jnp.where(field_lane == field_row + OUT_TARGETS, targets, 0), axis=1
        )
        margins_row = jnp.sum(
            jnp.where(
                field_lane == field_row + OUT_MARGINS,
                jax.lax.bitcast_convert_type(margins, jnp.int32),
                0,
            ),
            axis=1,
        )
        result = jnp.zeros((R, 128), jnp.int32) - 1
        result = jnp.where(
            (result_lane >= OUT_TARGETS) & (result_lane < OUT_TARGETS + rows),
            targets_row,
            result,
        )
        result = jnp.where(
            (result_lane >= OUT_MARGINS) & (result_lane < OUT_MARGINS + rows),
            margins_row,
            result,
        )
        result = jnp.where(result_lane == OUT_COUNT, count[:, None], result)
        result = jnp.where(result_lane == OUT_BONUS, anchor_token[:, None], result)
        tokens_v[...] = result

        # Each request's new keys: context rows up to its accepted count, its
        # block rows (all of them, or with ``c.causal`` those up to each
        # attending row: per packet row, ``heads`` slots a block row); the
        # padding block row is no key. (Attention DP: the rank's group's
        # requests, whose rows it attends.)
        packet_rows = valid_vmem.shape[1]
        key_lane = jax.lax.broadcasted_iota(jnp.int32, (R, packet_rows, KEYS), 2)
        valid = ((key_lane < rows) & (key_lane <= count[:, None, None])) | (
            (key_lane >= rows) & (key_lane < rows + block)
        )
        if c.causal:
            block_row = (
                jax.lax.broadcasted_iota(jnp.int32, (R, packet_rows, KEYS), 1) // heads
            )
            valid = valid & ((key_lane < rows) | (key_lane - rows <= block_row))
        valid_vmem[...] = own_group(valid.astype(jnp.int32))

        # ---------------- Markov w1 rows, RoPE rows, anchor embeddings ----------------
        w1_copies: list[list[Any]] = [[None] * R, [None] * R]

        def start_w1(tokens, slot):
            for j in range(R):
                start = pl.multiple_of((tokens[j] // 16) * 16, 16)
                item = copy(
                    weights["w1"].at[pl.ds(start, 16)],
                    w1_blk.at[slot, j],
                    SEM_W1 + slot * R + j,
                )
                item.start()
                w1_copies[slot][j] = item

        def w1_rows_for(tokens, slot):
            return jnp.concatenate(
                [
                    _bf16(_select_rows(_f32(w1_blk[slot, j]), tokens[j] % 16, 1))
                    for j in range(R)
                ],
                axis=0,
            )

        start_w1(anchor_token, 0)
        rope_copies = []
        for j in range(R):
            block_start = pl.multiple_of((anchor_position[j] // 8) * 8, 8)
            context_start = pl.multiple_of((new_position[j] // 8) * 8, 8)
            rope_copies += (
                copy(
                    weights["rope"].at[pl.ds(block_start, 16)],
                    rope_blk.at[0, j],
                    SEM_ROPE + 2 * j,
                ),
                copy(
                    weights["rope"].at[pl.ds(context_start, 16)],
                    rope_blk.at[1, j],
                    SEM_ROPE + 2 * j + 1,
                ),
            )
        for item in rope_copies:
            item.start()

        owner = anchor_token // vocab
        for j in range(R):

            @pl.when(rank == owner[j])
            def send_anchor_row(j=j):
                local_id = anchor_token[j] - owner[j] * vocab
                item = copy(
                    weights["embed"].at[
                        pl.ds(pl.multiple_of((local_id // 16) * 16, 16), 16)
                    ],
                    embed_blk,
                    SEM_EMBED + j,
                )
                item.start()
                item.wait()
                bcast_buf[j] = pack(
                    _bf16(_select_rows(_f32(embed_blk[...]), local_id % 16, 1))
                )
                copies = [
                    remote(bcast_buf.at[j], bcast_buf.at[j], SEM_BCAST + j, rank ^ o)
                    for o in range(1, 32)
                ]
                for item in copies:
                    item.start()
                for item in copies:
                    item.wait_send()

        _phase_end(scope)

        # ---------------- context features ----------------
        scope = _phase_start("d_context")
        aux_ref = small["aux"]  # [aux layers, T, hidden]
        spans = []
        for h in range(4):
            start, stop = h * FC_IN, (h + 1) * FC_IN
            pieces = []
            for layer_index in range(len(c.aux_layers)):
                lo, hi = layer_index * hidden, (layer_index + 1) * hidden
                a, b = max(start, lo), min(stop, hi)
                if a < b:
                    pieces.append((layer_index, a - lo, b - lo, a - start))
            spans.append(pieces)

        # The fc weight streams in two halves (VMEM).
        fc_parts = 2
        part_rows = FC_IN // fc_parts

        def fc_phase(fc_buf):
            for p in range(fc_parts):
                item = copy(
                    weights["fc"].at[pl.ds(p * part_rows, part_rows)], fc_buf, SEM_FC
                )
                item.start()
                item.wait()
                for h in range(4):

                    @pl.when(host == h)
                    def fc_branch(h=h, p=p):
                        total = None if p == 0 else fc_part[...]
                        for layer_index, lo, hi, offset in spans[h]:
                            a, b = (
                                max(offset, p * part_rows),
                                min(offset + hi - lo, (p + 1) * part_rows),
                            )
                            if a >= b:
                                continue
                            part = _dot(
                                aux_ref[
                                    layer_index, :, lo + a - offset : lo + b - offset
                                ],
                                fc_buf[pl.ds(a - p * part_rows, b - a), :],
                            )
                            total = part if total is None else total + part
                        if total is not None:
                            fc_part[...] = total

        pl.run_scoped(fc_phase, tpu.VMEM((part_rows, CHUNK), jnp.bfloat16))
        context = _rms_norm(
            _bf16(host_reduce_gather(fc_part[...], 0)), small["context_norm"][...], eps
        )  # [T, 7168]

        _phase_end(scope)

        # ---------------- RoPE rows and block input ----------------
        scope = _phase_start("d_input")
        for item in rope_copies:
            item.wait()
        rope_block = jnp.concatenate(
            [
                _select_rows(rope_blk[0, j], anchor_position[j] % 8, rows)
                for j in range(R)
            ],
            axis=0,
        )
        rope_context = jnp.concatenate(
            [_select_rows(rope_blk[1, j], new_position[j] % 8, rows) for j in range(R)],
            axis=0,
        )
        cos_block, sin_block = (
            rope_block[:, : c.rope],
            rope_block[:, c.rope :],
        )  # [T, 64] FP32
        cos_context, sin_context = rope_context[:, : c.rope], rope_context[:, c.rope :]
        for j in range(R):

            @pl.when(rank != owner[j])
            def receive_anchor_row(j=j):
                remote(
                    bcast_buf.at[j], bcast_buf.at[j], SEM_BCAST + j, rank
                ).wait_recv()

        anchor_rows = jnp.concatenate(
            [unpack(bcast_buf[j], 1, hidden) for j in range(R)], axis=0
        )  # [R, 7168]
        mask_row = _f32(small["mask_embed"][...])
        x_row = jax.lax.broadcasted_iota(jnp.int32, (R, rows, hidden), 1)
        x = _bf16(
            jnp.where(x_row == 0, anchor_rows[:, None, :], mask_row[:, None, :])
        ).reshape(T, hidden)

        def latent_rows(kv_bf16, layer, cos32, sin32):
            """``[T, 640]`` BF16 cache rows: kv_a_layernorm(c_kv) | RoPE(k_pe) | 0."""
            c_kv = _rms_norm(
                kv_bf16[:, : c.kv_lora], norm_row("kv_norm", layer)[:, : c.kv_lora], eps
            )
            k_pe = _rope(
                kv_bf16[:, c.kv_lora : c.latent], small["rot"][...], cos32, sin32
            )
            return jnp.concatenate(
                (c_kv, k_pe, jnp.zeros((T, LANES - c.latent), jnp.bfloat16)), axis=1
            )

        # The context rows' kv_a for every layer: one tile per rank, one exchange.
        tile_copies[1].wait()
        context_tiles = gather_tiles(
            _bf16(_dot(context, tile_buf[1])), 0, L * CONTEXT_TILES
        )
        if L > 1:
            tile_copies[1] = tile_copy(weights["qkv_tile"].at[1], 1)
            tile_copies[1].start()
        for layer in range(L):
            context_kv = _bf16(
                jnp.concatenate(
                    context_tiles[layer * CONTEXT_TILES : (layer + 1) * CONTEXT_TILES],
                    axis=1,
                )
            )
            latents_vmem[layer] = latent_rows(
                context_kv, layer, cos_context, sin_context
            )

        # ---------------- DCP attention over the draft pools ----------------
        # Attention DP: the group's rows over its pages (DCP over the group).
        first_request = data_group * group_requests if dp > 1 else 0
        position_scalars = (
            _Scalars(scalars, first_request, rows)
            if single is None
            else _Scalars(ids, rows, rows)  # one request: its verify's first position
        )
        page_owner = None
        if owner_table is not None:
            page_owner = mla_dcp.owner_index(owner_table)
            page_owner = page_owner % mla_dcp.GROUP if dp > 1 else page_owner
        cache = mla_dcp.DcpCache(
            pool_ref=pools,
            table_ref=table_scalars,
            pages=pages,
            layer=jnp.int32(0),
            group=mla_dcp.GROUP,
            owner=page_owner,
            table_row=lambda row: first_request + row // rows,
        )
        cos_query, sin_query = own_group(cos_block), own_group(sin_block)
        workspace = mla_dcp.DcpWorkspace(*persistent, *dcp_scoped)

        # Each layer buffer refills with the next layer's weights as soon as
        # this layer is done with it (o / down: whole, or their first chunk).
        chunked = {
            "o": _ChunkedWeight(weights["o"], o_buf, (5, SEM_O2), copy),
            "down": _ChunkedWeight(
                weights["down"], down_buf, (SEM_DOWN, SEM_DOWN2), copy
            ),
        }

        def layer_copy(name, layer):
            if name in chunked:
                return chunked[name].load(layer)
            buffers = {
                "q_b": (weights["q_b"].at[layer], q_buf, 2),
                "w_uk": (weights["w_uk"].at[layer], w_uk_buf, 3),
                "w_uv": (weights["w_uv"].at[layer], w_uv_buf, 4),
                "gate": (weights["gate"].at[layer], gate_buf, 6),
                "up": (weights["up"].at[layer], up_buf, 7),
            }
            item = copy(*buffers[name])
            item.start()
            return item

        # o / down held whole by KernelConfig.draft_whole_weights (more than
        # two heads a rank): layer 0's copies start once q_b(0) has arrived,
        # so their bytes do not slow it.
        deferred = [
            name for name, weight in chunked.items() if per > 2 and not weight.streamed
        ]
        pending = {
            name: layer_copy(name, 0)
            for name in ("q_b", "w_uk", "w_uv", "o", "gate", "up", "down")
            if name not in deferred
        }

        def refill(name, layer):
            if layer + 1 < L:
                pending[name] = layer_copy(name, layer + 1)

        # KernelConfig.draft_lm_prefetch: the LM head's chunk slots beyond the
        # two scoped ones -- the gate / up buffers' halves ([hidden, 512]
        # each, two chunks), each on a semaphore idle after the last layer's
        # loads (gate, up; 10 is unused; q_b's).
        early_slots = (
            [
                buffer.at[:, pl.ds(half * LM_CHUNK, LM_CHUNK)]
                for buffer in (gate_buf, up_buf)
                for half in range(MLP_PADDED // LM_CHUNK)
            ]
            if lm_prefetch
            else []
        )
        early_sems = (6, 10, 7, 2)[: len(early_slots)]

        def lm_slot_copy(k, slots, sems=early_sems):
            return copy(
                weights["lm_head"].at[:, pl.ds(k * LM_CHUNK, LM_CHUNK)],
                slots[k % len(slots)],
                sems[k % len(slots)],
            )

        _phase_end(scope)
        for layer in range(L):
            scope = _phase_start("d_qkv")
            h = _rms_norm(x, norm_row("ln1", layer), eps)
            slot = layer % 2
            tile_copies[slot].wait()
            tiles = gather_tiles(
                _bf16(_dot(h, tile_buf[slot])), (layer + 1) % 2, QKV_TILES
            )
            if layer + 2 < L:
                tile_copies[slot] = tile_copy(weights["qkv_tile"].at[layer + 2], slot)
                tile_copies[slot].start()
            q_a = _bf16(jnp.concatenate(tiles[:Q_TILES], axis=1))  # [T, 1536]
            block_kv = _bf16(jnp.concatenate(tiles[Q_TILES:], axis=1))  # [T, 640]
            q_c = own_group(
                _rms_norm(q_a, norm_row("q_norm", layer), eps)
            )  # [group rows, 1536]
            block_latents = own_group(
                latent_rows(block_kv, layer, cos_block, sin_block)
            )
            context_latents = own_group(latents_vmem[layer])
            for j in range(group_requests):
                keys_vmem[j] = jnp.concatenate(
                    (
                        _f32(context_latents[j * rows : (j + 1) * rows]),
                        _f32(block_latents[j * rows : (j + 1) * rows]),
                    ),
                    axis=0,
                )

            _phase_end(scope)
            scope = _phase_start("d_query")
            pending["q_b"].wait()
            if layer == 0:
                for name in deferred:
                    pending[name] = layer_copy(name, 0)
            q = _bf16(
                _dot(q_c, q_buf[...])
            )  # [group rows, dp Q_WIDTH]: each held shard's columns
            refill("q_b", layer)
            pending["w_uk"].wait()
            queries = []
            for head in range(heads):
                shard, within = divmod(head, per)
                nope = shard * q_shard + within * c.nope
                q_abs = _bf16(
                    _dot(q[:, nope : nope + c.nope], w_uk_buf[head])
                )  # [group rows, 512]
                pe = shard * q_shard + per * c.nope + within * 128
                q_pe = _rope(
                    q[:, pe : pe + c.rope], small["rot"][...], cos_query, sin_query
                )
                queries.append(
                    jnp.concatenate(
                        (
                            _f32(q_abs) * scale,
                            _f32(q_pe) * scale,
                            jnp.zeros((group_rows, LANES - c.latent), jnp.float32),
                        ),
                        axis=1,
                    )
                )
            queries = jnp.stack(queries, axis=1)  # [group rows, heads, 640]
            refill("w_uk", layer)
            _phase_end(scope)
            scope = _phase_start("d_attend")
            attention = mla_dcp.DcpAttention(
                dataclasses_replace(cache, layer=jnp.int32(layer)),
                workspace,
                position_scalars,
                group_rows,
                sequence_rows=rows,
                heads=heads,
                chunk_rows=rows,
            )
            # The block's padding row carries the partials' lse: the combine
            # sends 512 lanes, not 640.
            attention.lse_in_padding = block < rows
            attention.group_fold = True
            attention.step_trims = _config().draft_step_trims
            if block < rows and mla_dcp.GROUP == RANKS:
                # Merge each host's partials before crossing hosts (chunked
                # calls, one request a chunk, even one request).
                attention.force_chunked = True
                attention.host_combine = (stage2_recv, stage2_send, stage2_sems)
                attention.host_pairs = pair_sems[0] if pair_sems else None
            if PHASE_SCOPES:
                attention.trace = "a_"
            attention.prime()
            merged = attention.attend(
                queries,
                jnp.zeros((group_rows, LANES), jnp.float32),
                layer % 2,
                keys=keys_vmem,
                key_valid=valid_vmem,
            )  # [group rows, heads, 512]
            _phase_end(scope)
            scope = _phase_start("d_o")
            wuv_scope = _phase_start("d_wuv")
            pending["w_uv"].wait()
            # [T, heads 128]: the other groups' rows are zero (their ranks add theirs).
            attended = place_group(
                jnp.concatenate(
                    [
                        _bf16(_dot(_bf16(merged[:, head]), w_uv_buf[head]))
                        for head in range(heads)
                    ],
                    axis=1,
                )
            )
            refill("w_uv", layer)
            pending["o"].wait()
            _phase_end(wuv_scope)
            x = _bf16(
                _f32(x) + all_reduce(attended, chunked["o"].chunks(layer), 1, tag="o")
            )
            refill("o", layer)

            _phase_end(scope)
            scope = _phase_start("d_mlp")
            h2 = _rms_norm(x, norm_row("ln2", layer), eps)
            pending["gate"].wait()
            pending["up"].wait()
            gate = _f32(_bf16(_dot(h2, gate_buf[...])))
            up = _bf16(_dot(h2, up_buf[...]))
            refill("gate", layer)
            refill("up", layer)
            if lm_prefetch and layer == L - 1:
                # KernelConfig.draft_lm_prefetch: the LM head's first chunks
                # into the gate / up halves, now free.
                for k in range(len(early_slots)):
                    lm_slot_copy(k, early_slots).start()
            activation = _bf16(_f32(_bf16(gate * jax.nn.sigmoid(gate))) * _f32(up))
            _phase_end(scope)
            scope = _phase_start("d_down")
            pending["down"].wait()
            mlp = all_reduce(activation, chunked["down"].chunks(layer), 0, tag="down")
            refill("down", layer)
            x = _bf16(_f32(x) + mlp)
            _phase_end(scope)
        scope = _phase_start("d_lm")
        latents_copy = copy(latents_vmem, latents_out, SEM_LAT)
        latents_copy.start()

        # ---------------- LM head (the target's, vocabulary sharded) ----------------
        hidden_rows = _rms_norm(x, small["final_norm"][...], eps)

        def lm_phase(chunk_a, chunk_b):
            buffers = (chunk_a, chunk_b)

            def chunk_copy(k):
                item = copy(
                    weights["lm_head"].at[:, pl.ds(k * LM_CHUNK, LM_CHUNK)],
                    buffers[k % 2],
                    (SEM_LM, SEM_LM2)[k % 2],
                )
                item.start()
                return item

            copies = [chunk_copy(0), chunk_copy(1)]
            for k in range(LM_CHUNKS):
                copies[k % 2].wait()
                logits = _f32(_bf16(_dot(hidden_rows, buffers[k % 2][...])))
                if k + 2 < LM_CHUNKS:
                    copies[k % 2] = chunk_copy(k + 2)
                logits_vmem[:, pl.ds(k * LM_CHUNK, LM_CHUNK)] = logits

        def lm_ring_phase(chunk_a, chunk_b):
            # Six slots: the four early ones (in flight since the last
            # layer's MLP), then the two scoped ones; chunk k in slot k % 6.
            slots = early_slots + [chunk_a, chunk_b]
            sems = early_sems + (SEM_LM, SEM_LM2)
            for k, slot in stream_chunks(
                lambda k, _: lm_slot_copy(k, slots, sems),
                slots,
                LM_CHUNKS,
                started=len(early_slots),
            ):
                logits = _f32(_bf16(_dot(hidden_rows, slot[...])))
                logits_vmem[:, pl.ds(k * LM_CHUNK, LM_CHUNK)] = logits

        pl.run_scoped(
            lm_ring_phase if lm_prefetch else lm_phase,
            tpu.VMEM((hidden, LM_CHUNK), jnp.bfloat16),
            tpu.VMEM((hidden, LM_CHUNK), jnp.bfloat16),
        )

        _phase_end(scope)

        # ---------------- exact Markov sampling ----------------
        scope = _phase_start("d_markov")
        w2t_copy.wait()
        vocab_lane = jax.lax.broadcasted_iota(jnp.int32, (1, vocab), 1)
        tokens = tokens_v[...]
        token_lane = jax.lax.broadcasted_iota(jnp.int32, (R, 128), 1)
        rank_iota = jax.lax.broadcasted_iota(jnp.int32, (RANKS, R, 128), 0)
        previous = anchor_token
        for slot in range(block):
            for item in w1_copies[slot % 2]:
                item.wait()
            bias = _dot(w1_rows_for(previous, slot % 2), w2t_buf[...])  # [R, 5120]
            scores = (
                jnp.concatenate(
                    [
                        logits_vmem[j * rows + slot : j * rows + slot + 1, :]
                        for j in range(R)
                    ],
                    axis=0,
                )
                + bias
            )
            best_value = jnp.max(scores, axis=1, keepdims=True)
            best_id = jnp.min(
                jnp.where(scores == best_value, vocab_lane, vocab),
                axis=1,
                keepdims=True,
            )
            row = jnp.where(
                token_lane == 0,
                jax.lax.bitcast_convert_type(best_value, jnp.int32),
                jnp.where(token_lane == 1, best_id + rank * vocab, 0),
            )
            gathered = tiny_exchange(row, slot % 2)
            maxima = jax.lax.bitcast_convert_type(gathered[:, :, 0:1], jnp.float32)
            top = jnp.max(maxima, axis=0, keepdims=True)
            win = jnp.min(
                jnp.where(maxima == top, rank_iota[:, :, 0:1], RANKS),
                axis=0,
                keepdims=True,
            )
            token = jnp.sum(jnp.where(rank_iota == win, gathered, 0), axis=0)[
                :, 1:2
            ]  # [R, 1]
            tokens = jnp.where(token_lane == OUT_DRAFTS + slot, token, tokens)
            previous = token[:, 0]
            if slot + 1 < block:
                start_w1(previous, (slot + 1) % 2)
        tokens_v[...] = tokens
        out = tpu.make_async_copy(tokens_v, tokens_out, small_sems.at[0])
        out.start()
        out.wait()
        latents_copy.wait()
        _phase_end(scope)

    return body


def dataclasses_replace(value, **fields):
    import dataclasses

    return dataclasses.replace(value, **fields)


def fused_draft_local(
    c: MlaDraftConfig,
    *,
    requests: int,
    pages: int,
    block: int | None = None,
    rows: int = ROWS,
    owner_table=None,
    attention_dp: int = 1,
    single: SingleEpilogue | None = None,
):
    """The per-shard fused draft (call inside a TP32 ``shard_map``):
    ``(weights, pools, aux, logits, previous, positions0, forced_counts,
    forced_anchors, tables)`` -> ``(vectors [R, 128] int32, latents [layers,
    T, 640] bf16)``.

    ``single`` (one request, attention DP1; ``KernelConfig.
    draft_fused_epilogue``): ``(weights, pools, aux, logits, previous, ids,
    decisions)`` (the op's packed ids, see ``SingleEpilogue``), the
    layer-norm stacks (``LEADING_NAMES``) as ``[layers, width]``.

    Under ``attention_dp`` (DCP over each group, ``mla_dcp.GROUP``) the first
    ``R / dp`` requests are group 0's, and so on: a rank attends its group's
    requests with its held TP32 shards' heads (``q_b``, ``w_uk``, ``w_uv``
    and ``o`` hold ``dp`` shards, ``dspark_mla.held_shards``).

    ``weights``: ``WEIGHT_NAMES`` (this rank's arrays, including the target's
    ``lm_head`` ``[7168, 5120]`` and ``embed`` ``[5120, 7168]``); ``pools``:
    the draft layers' local pools ``[blocks, 128, packing, 640]``; ``aux``
    ``[aux layers, T, hidden]`` and ``logits`` ``[R, rows, vocab / 32]`` the
    verify's (``T = rows * R``; four or eight rows per request); ``previous``
    ``[R, 128]`` the last vectors; ``positions0`` ``[R]`` each verify's first position;
    ``forced_*`` ``[R]`` (``>= 0``: given); ``tables`` ``[R, pages]``."""
    block = c.block if block is None else block
    if rows not in (4, ROWS):
        raise ValueError("the fused MLA draft takes 4 or 8 rows per request")
    if not 2 <= block <= rows:
        raise ValueError(f"block must be 2..{rows}")
    R = requests
    T = R * rows
    if not 1 <= R <= 16 or T > 64:
        raise ValueError(
            "the fused MLA draft takes up to 8 requests at 8 rows, or 16 at 4 rows"
        )
    dp = attention_dp
    if R % dp or mla_dcp.GROUP * dp != RANKS:
        raise ValueError(
            f"attention DP{dp}: {R} requests over DCP{mla_dcp.GROUP} groups"
        )
    if single is not None and (R != 1 or dp != 1):
        raise ValueError("the single-request epilogue takes one request at DP1")
    heads, group_rows = c.heads_per_rank * dp, T // dp
    packed = T * CHUNK // 256
    # The body streams o / down iff their buffers hold two chunk slots.
    streamed_o, streamed_down = streamed_weights(
        c, T, dp, _config().draft_whole_weights
    )

    def call(weights, pools, aux, logits, previous, *operands):
        if single is None:
            position, forced_count, forced_anchor, tables = operands
        else:
            ids, decisions = operands
        pools = tuple(pools)
        logits = logits.reshape(T, c.vocab // RANKS).astype(jnp.float32)
        cache_dtype = pools[0].dtype
        page_tokens = pools[0].shape[1] * pools[0].shape[2]
        chunks = (
            block < rows and mla_dcp.GROUP == RANKS
        )  # the two-stage combine (see the body)
        plan = mla_dcp.page_plan(
            group_rows, heads, rows, cache_dtype, force_chunks=chunks
        )
        # KernelConfig.draft_native_pages: [span * 512, 640] FP8 page buffers
        # (None: the pool's packed blocks).
        native = (
            True
            if _config().draft_native_pages and cache_dtype == jnp.float8_e4m3fn
            else None
        )
        if single is None:
            prefetch = (
                jnp.concatenate(
                    (position, forced_count, forced_anchor, tables.reshape(-1))
                ).astype(jnp.int32),
            )
        else:
            prefetch = (ids, decisions)
        small_shapes = {"aux": aux, "previous": previous, **weights}
        # FP8 caches: FP8 query packets (the operand the receiver would
        # quantize to anyway), half the gather's bytes.
        persistent = mla_dcp.persistent_scratch(
            group_rows,
            mla_dcp.GROUP,
            heads,
            cache_dtype=cache_dtype,
            sequence_rows=rows,
            fp8_wire=True,
            force_chunks=chunks,
            chunk_rows=rows,
        )
        ring = persistent[0].shape[0] if mla_dcp.chunked(group_rows, chunks) else 1
        hosts = RANKS // mla_dcp.HOST_LANES
        scratch = (
            *(
                tpu.VMEM(small_shapes[n].shape, small_shapes[n].dtype)
                for n in SMALL_NAMES
            ),
            *persistent,
            *mla_dcp.scoped_scratch(
                group_rows,
                plan["span"],
                plan["buffers"],
                heads=heads,
                group=mla_dcp.GROUP,
                cache_dtype=cache_dtype,
                page_tokens=page_tokens,
                sequence_rows=rows,
                force_chunks=chunks,
                chunk_rows=rows,
                native=native,
            ),
            tpu.VMEM(
                (c.q_lora, dp * c.heads_per_rank * (c.nope + TILE)), jnp.bfloat16
            ),  # q_buf
            tpu.VMEM((heads, c.nope, c.kv_lora), jnp.bfloat16),  # w_uk_buf
            tpu.VMEM((heads, c.kv_lora, c.v_head), jnp.bfloat16),  # w_uv_buf
            tpu.VMEM(
                (2, heads * c.v_head, CHUNK)
                if streamed_o
                else (heads * c.v_head, c.hidden),
                jnp.bfloat16,
            ),  # o_buf (streamed: two chunk slots)
            tpu.VMEM((c.hidden, MLP_PADDED), jnp.bfloat16),  # gate_buf
            tpu.VMEM((c.hidden, MLP_PADDED), jnp.bfloat16),  # up_buf
            tpu.VMEM(
                (2, MLP_PADDED, CHUNK) if streamed_down else (MLP_PADDED, c.hidden),
                jnp.bfloat16,
            ),  # down_buf
            tpu.VMEM((c.markov_rank, c.vocab // RANKS), jnp.bfloat16),  # w2t_buf
            tpu.VMEM((2, R, 16, c.markov_rank), jnp.bfloat16),  # w1_blk
            tpu.VMEM((16, c.hidden), jnp.bfloat16),  # embed_blk
            tpu.VMEM((R, ROW_PACKED, 128), jnp.float32),  # bcast_buf
            tpu.VMEM((2, R, 16, 2 * c.rope), jnp.float32),  # rope_blk (block, context)
            tpu.VMEM((T, c.vocab // RANKS), jnp.float32),  # logits_vmem
            tpu.VMEM((T, CHUNK), jnp.float32),  # fc_part
            tpu.VMEM((R, 128), jnp.int32),  # tokens_v
            tpu.VMEM((2, 8, packed, 128), jnp.float32),  # stage
            tpu.VMEM((2, 8, packed, 128), jnp.float32),  # local_recv
            tpu.VMEM((2, 4, packed, 128), jnp.float32),  # host_packed
            tpu.VMEM((2, 8, packed, 128), jnp.float32),  # gather_recv
            tpu.VMEM((2, RANKS, R, 128), jnp.int32),  # tiny_recv
            tpu.VMEM((RANKS, 8, 128), jnp.int32),  # argmax_recv: rows along lanes
            tpu.VMEM(
                (R // dp, 2 * rows, LANES), jnp.float32
            ),  # keys_vmem (the group's requests)
            # valid_vmem: per request, or per packet row (causal block rows)
            tpu.VMEM((R // dp, heads * rows if c.causal else 1, 2 * rows), jnp.int32),
            tpu.VMEM((c.layers, T, LANES), jnp.bfloat16),  # latents_vmem
            tpu.VMEM(
                (2, c.hidden, TILE), jnp.bfloat16
            ),  # tile_buf: q_a | kv_a projection tiles
            tpu.VMEM(
                (2, RANKS, T * TILE // 256, 128), jnp.float32
            ),  # tile_recv (packed BF16)
            # The two-stage DCP combine (chunked calls; see mla_dcp.host_combine).
            tpu.VMEM(
                (ring, hosts, c.heads_per_rank * rows, mla_dcp.LANES), jnp.bfloat16
            ),  # stage2_recv
            tpu.VMEM(
                (hosts, c.heads_per_rank * rows, mla_dcp.LANES), jnp.bfloat16
            ),  # stage2_send
            tpu.SemaphoreType.DMA((2, ring)),  # stage2_sems
            tpu.SemaphoreType.DMA((19 + 5 * R,)),
            tpu.SemaphoreType.DMA((len(SMALL_NAMES),)),
            tpu.SemaphoreType.DMA((2 * SEMS_PER_SLOT,)),
            tpu.SemaphoreType.DMA((2 * SEMS_PER_SLOT,)),
        )
        if chunks and _config().draft_pair_combine:
            scratch += (tpu.SemaphoreType.DMA((2, ring)),)  # the pair stage's
        out_shape = (
            jax.ShapeDtypeStruct((R, 128), jnp.int32),
            jax.ShapeDtypeStruct((c.layers, T, LANES), jnp.bfloat16),
        )
        out_specs = [
            pl.BlockSpec(memory_space=tpu.HBM),
            pl.BlockSpec(memory_space=tpu.HBM),
        ]
        if single is not None and ids.shape != (single.ids_size,):
            raise ValueError("the single-request ids size")
        body = _make_body(c, block, R, rows, pages, owner_table, plan, dp, single)
        operands = (aux, logits, previous, *[weights[n] for n in WEIGHT_NAMES], *pools)
        in_specs = [pl.BlockSpec(memory_space=tpu.HBM) for _ in operands]
        return pl.pallas_call(
            body,
            name="dspark_mla_fused_draft",
            out_shape=out_shape,
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(prefetch),
                in_specs=in_specs,
                out_specs=out_specs,
                scratch_shapes=scratch,
            ),
            compiler_params=tpu.CompilerParams(
                collective_id=26, vmem_limit_bytes=64 * 1024**2
            ),
        )(*prefetch, *operands)

    return call
