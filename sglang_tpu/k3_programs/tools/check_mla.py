# SPDX-License-Identifier: Apache-2.0
"""``check mladraft``: the fused MLA DSpark draft kernel against the JAX
reference (``kernels.dspark_mla.reference_*``) on synthetic inputs.

Every rank runs the kernel (``kernels.dspark_mla_kernel``) over a context it
builds itself: ``--steps`` steps accepting all verify rows, at bases
``--stride`` tokens apart (so the committed latents spread over pages on every
DCP owner), from random aux features. A final step at ``--context`` with a
forced accepted count and anchor drafts; rank 0 recomputes that step on the
host CPU from the checkpoint (unwritten positions are zero latents, as the
pools hold) and compares the context latents the kernel emits and its drafts
(with the reference's top-2 margin at each mismatch).

Under attention DP (``--set attention_dp=2 dcp_group=16``) the first half of
the requests are group 0's and the rest group 1's, request ``j`` of each group
on the same pool rows (as ``megakernel.dp_blocks`` places them): each group's
context must stay its own.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from tools.megakernel import harness


def add_arguments(sub) -> None:
    parser = sub.add_parser(
        "mladraft", help="the MLA DSpark draft kernel against the reference"
    )
    parser.add_argument("--dspark", required=True, help="the MLA DSpark checkpoint")
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--drafts", type=int, choices=(3, 7), default=7)
    parser.add_argument("--steps", type=int, default=24, help="context-building steps")
    parser.add_argument(
        "--stride", type=int, default=600, help="positions between building steps"
    )
    parser.add_argument(
        "--count", type=int, default=3, help="the final step's accepted count"
    )
    parser.add_argument("--seed", type=int, default=0)


def run_mladraft(args, records: harness.Records) -> None:
    import jax
    import jax.numpy as jnp
    import torch
    from jax.sharding import PartitionSpec as P

    from tools.megakernel import k3
    from vllm_torchtpu.megakernel.bridge import register_op
    from vllm_torchtpu.megakernel.kimi_k3 import config
    from vllm_torchtpu.megakernel.kimi_k3.decode_ops import DecodeOps
    from vllm_torchtpu.megakernel.kimi_k3.draft_mla import MlaDsparkDraft
    from vllm_torchtpu.megakernel.kimi_k3.program import kernel_modules

    mla, kernel = kernel_modules("dspark_mla", "dspark_mla_kernel")
    R, steps, stride = args.requests, args.steps, args.stride
    dp, group = config.active().attention_dp, config.active().dcp_group
    if R % dp:
        raise SystemExit(f"attention DP{dp}: --requests must be a multiple of {dp}")
    shared = R // dp  # requests per group, each group's request j on the same rows
    final = steps * stride  # the final verify's first position
    ops = DecodeOps(args.presharded, num_slots=4)
    ops.load_weights()
    draft = MlaDsparkDraft(final + 64, args.dspark, args.presharded)
    draft.load_weights()
    draft.attach_target(ops.weights)
    c = dataclasses.replace(draft.config, block=args.drafts)
    rows_per = args.drafts + 1
    rank = draft.rank
    dtype = torch.float8_e4m3fn if k3.CACHE_DTYPE == "fp8" else torch.bfloat16
    packing = 4 if dtype == torch.float8_e4m3fn else 2
    page_tokens = 128 * packing
    entries = -(
        -(final + rows_per) // (page_tokens * group)
    )  # local block-table entries per request
    pages = max(entries, 2)
    blocks = 1 + shared * pages
    draft.bind_pools(
        torch.zeros((blocks, 128, packing, 640), dtype=dtype).to("tpu"), None
    )
    tables = np.zeros((R, pages), np.int32)
    for j in range(R):
        tables[j, :entries] = 1 + j % shared * pages + np.arange(entries)
    harness.log(
        f"pools {blocks} blocks x {page_tokens} tokens ({k3.CACHE_DTYPE}), "
        f"context {final}, {R} requests, attention DP{dp}"
    )

    rng = np.random.default_rng(args.seed)
    vocab = c.vocab // 32
    names = draft.names
    count = len(names)
    layers = c.layers

    def build(requests):
        fused = kernel.fused_draft_local(
            c, requests=requests, pages=pages, rows=rows_per, attention_dp=dp
        )

        def body(*a):
            weights = {name: a[i][0] for i, name in enumerate(names)}
            pools = a[count : count + layers]
            aux, logits, previous, position, forced_count, forced_anchor, table = a[
                count + layers :
            ]
            vector, latents = fused(
                weights,
                pools,
                aux,
                logits[0],
                previous,
                position,
                forced_count,
                forced_anchor,
                table,
            )
            positions = (
                position[:, None] + jnp.arange(rows_per, dtype=jnp.int32)[None, :]
            ).reshape(-1)
            pools = draft._commit(
                pools, latents, jnp.repeat(table, rows_per, axis=0), positions
            )
            return (vector, latents, *pools)

        return register_op(
            f"pallas::check_mladraft_r{requests}",
            draft.mesh,
            body,
            (P("tp"),) * (count + layers) + (P(), P("tp"), P(), P(), P(), P(), P()),
            (P(), P()) + (P("tp"),) * layers,
            donate_argnums=tuple(range(count, count + layers)),
        )

    op = build(R)
    hidden = c.hidden
    context_aux = {}  # position -> [5 * hidden] float32 (bf16 values), per request

    def step(base_positions, forced_count, forced_anchor):
        aux = rng.standard_normal((len(c.aux_layers), R * rows_per, hidden)).astype(
            np.float32
        )
        aux = np.asarray(jnp.asarray(aux, jnp.bfloat16).astype(jnp.float32))
        for j in range(R):
            for k in range(rows_per):
                context_aux[(j, int(base_positions[j]) + k)] = aux[
                    :, j * rows_per + k
                ].reshape(-1)
        logits = rng.standard_normal((1, R, rows_per, vocab)).astype(np.float32)
        previous = np.full((R, 128), -1, np.int32)
        out = op(
            *draft._operands(),
            *draft.pools,
            torch.from_numpy(aux).to(torch.bfloat16).to("tpu"),
            torch.from_numpy(logits).to("tpu"),
            torch.from_numpy(previous).to("tpu"),
            torch.from_numpy(np.asarray(base_positions, np.int32)).to("tpu"),
            torch.from_numpy(np.asarray(forced_count, np.int32)).to("tpu"),
            torch.from_numpy(np.asarray(forced_anchor, np.int32)).to("tpu"),
            torch.from_numpy(tables).to("tpu"),
        )
        vector, latents, *pools = out
        draft._adopt(pools)
        return vector, latents

    for s in range(steps):
        step([s * stride + j for j in range(R)], [args.drafts] * R, [1000 + s] * R)
    anchors = [int(rng.integers(1000, 100000)) for _ in range(R)]
    vector, latents = step([final + j for j in range(R)], [args.count] * R, anchors)
    vector = vector.cpu().numpy()
    latents = latents.to(torch.float32).cpu().numpy()  # [layers, rows_per R, 640]
    harness.log(f"kernel drafts {[vector[j, : c.block].tolist() for j in range(R)]}")

    # ---------------- the reference ----------------
    # Target embedding rows (anchors, mask) and, later, the target LM head over
    # the reference's hidden rows: small device ops every rank joins.
    def embedding_rows(token_ids):
        ids = np.asarray(token_ids, np.int32)

        def body(embedding, ids):
            r = jax.lax.axis_index("tp")
            local = ids - r * vocab
            rows = embedding[0][jnp.clip(local, 0, vocab - 1)].astype(jnp.float32)
            return (
                jax.lax.psum(
                    jnp.where(((local >= 0) & (local < vocab))[:, None], rows, 0.0),
                    "tp",
                ),
            )

        (rows,) = register_op(
            f"pallas::check_mladraft_embed{len(ids)}",
            draft.mesh,
            body,
            (P("tp"), P()),
            (P(),),
        )(ops.weights["embedding"], torch.from_numpy(ids).to("tpu"))
        return rows.cpu().numpy()

    def lm_logits(hidden_rows):
        def body(lm_head, rows):
            local = jax.lax.psum(rows, "tp")  # rank 0's rows, every rank
            part = jnp.dot(
                local.astype(jnp.bfloat16),
                lm_head[0],
                preferred_element_type=jnp.float32,
            )
            return (jax.lax.all_gather(part, "tp", axis=1, tiled=True),)

        (logits,) = register_op(
            f"pallas::check_mladraft_lm{hidden_rows.shape[0]}",
            draft.mesh,
            body,
            (P("tp"), P()),
            (P(),),
        )(ops.weights["lm_head"], torch.from_numpy(hidden_rows).to("tpu"))
        return logits.cpu().numpy()

    embeds = embedding_rows(anchors + [c.mask_token])
    hidden_rows = np.zeros((R * c.block, hidden), np.float32)
    report = {}
    if rank == 0:
        import ml_dtypes

        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            w = {
                k: jnp.asarray(v)
                for k, v in mla.load_checkpoint_weights(args.dspark, c).items()
            }
            fp8 = dtype == torch.float8_e4m3fn
            for j in range(R):
                own_first = final + j  # the final verify's first position
                limit = own_first + args.count + 1  # the anchor's position
                positions = sorted(p for (jj, p) in context_aux if jj == j)
                index = {p: i for i, p in enumerate(positions)}
                aux_rows = jnp.asarray(
                    np.stack([context_aux[(j, p)] for p in positions]), jnp.bfloat16
                )
                ctx = mla.reference_context(w, c, aux_rows)
                pos = jnp.asarray(positions, jnp.int32)
                ctx_latents, errors = [], []
                for layer in range(layers):
                    lat = np.asarray(
                        mla.reference_latents(w, c, layer, ctx, pos).astype(jnp.float32)
                    )  # [n, 576]
                    own = lat[[index[own_first + k] for k in range(rows_per)]]
                    kernel_own = latents[
                        layer, j * rows_per : (j + 1) * rows_per, : c.latent
                    ]
                    errors.append(
                        float(
                            np.max(np.abs(kernel_own - own))
                            / (np.max(np.abs(own)) + 1e-9)
                        )
                    )
                    full = np.zeros((limit, c.latent), np.float32)
                    for p in positions:
                        if p >= limit:
                            continue
                        value = lat[index[p]]
                        if (
                            p < own_first and fp8
                        ):  # pool rows are FP8; the step's own rows are new keys
                            value = value.astype(ml_dtypes.float8_e4m3fn).astype(
                                np.float32
                            )
                        full[p] = value
                    ctx_latents.append(jnp.asarray(full, jnp.bfloat16))
                valid = jnp.ones((limit,), bool)
                rows = mla.reference_block_hidden(
                    w,
                    c,
                    jnp.asarray(embeds[j], jnp.bfloat16),
                    jnp.asarray(embeds[R], jnp.bfloat16),
                    ctx_latents,
                    valid,
                    limit,
                )
                hidden_rows[j * c.block : (j + 1) * c.block] = np.asarray(
                    rows.astype(jnp.float32)
                )
                report[j] = {"latent_rel_err": errors}
                harness.log(
                    f"request {j}: context latent max rel error per layer "
                    f"{[f'{e:.2e}' for e in errors]}"
                )
    logits = lm_logits(hidden_rows.astype(np.float32))
    if rank == 0:
        with jax.default_device(jax.devices("cpu")[0]):
            w1 = np.asarray(
                mla.load_checkpoint_weights(
                    args.dspark, c, names=["markov_head.markov_w1.weight"]
                )["markov_head.markov_w1.weight"]
            ).astype(np.float32)
            w2 = np.asarray(
                mla.load_checkpoint_weights(
                    args.dspark, c, names=["markov_head.markov_w2.weight"]
                )["markov_head.markov_w2.weight"]
            ).astype(np.float32)
        total_match = 0
        for j in range(R):
            base = logits[j * c.block : (j + 1) * c.block].astype(np.float32)
            base = np.asarray(jnp.asarray(base, jnp.bfloat16).astype(jnp.float32))
            previous, drafts, margins = anchors[j], [], []
            kernel_drafts = vector[j, : c.block].tolist()
            for slot in range(c.block):
                scores = base[slot] + w2 @ w1[previous]
                order = np.argsort(-scores)[:2]
                drafts.append(int(order[0]))
                margins.append(float(scores[order[0]] - scores[order[1]]))
                previous = kernel_drafts[
                    slot
                ]  # follow the kernel's chain (one divergence does not cascade)
            match = [int(a == b) for a, b in zip(drafts, kernel_drafts)]
            total_match += sum(match)
            harness.log(f"request {j}: reference drafts {drafts}")
            harness.log(f"request {j}: kernel    drafts {kernel_drafts}")
            harness.log(
                f"request {j}: match {match}, reference top-2 margins "
                f"{[f'{m:.3f}' for m in margins]}"
            )
            report[j].update(
                reference=drafts, kernel=kernel_drafts, margins=margins, match=match
            )
        harness.log(f"drafts matching: {total_match}/{R * c.block}")
        records.add(
            requests=R,
            drafts=args.drafts,
            steps=steps,
            stride=stride,
            cache=k3.CACHE_DTYPE,
            report=report,
        )
