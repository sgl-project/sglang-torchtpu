# SPDX-License-Identifier: Apache-2.0
"""Replay sglang DSpark reference rounds through the JAX draft (server/draft.py) and time it.

  python3 test/manual/k3_megakernel/test_dspark_draft.py --dump DIR --draft DRAFT --target CKPT \
      [--compat] [--only 0] [--cpu] [--time]

DIR holds the per-round dumps of the 13-layer sglang+megakernel+DSpark server (extend / draft / commit .pt
files written by a dump hook in a scratch copy of sglang's TPU DSpark spec worker). For every request the
prompt captures fill the draft KV, then each round's propose(bonus, seq_len) is compared with sglang's
draft tokens and the committed rows are appended exactly as the sglang commit wrote them. ``--compat``
reproduces the sglang path's extra anchor-row context (``init_state(sglang_compat=True)``); on the first
request it matches sglang's drafts up to bf16 near-ties (~50/56 tokens on 8 rounds). ``--time``: propose /
append_context latency in a device loop.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True)
    p.add_argument("--draft", required=True, help="DSpark draft checkpoint")
    p.add_argument(
        "--target", required=True, help="target checkpoint (embedding / lm_head)"
    )
    p.add_argument("--capacity", type=int, default=2048)
    p.add_argument("--cpu", action="store_true", help="8 fake CPU devices")
    p.add_argument("--time", action="store_true")
    p.add_argument("--ndraft", type=int, default=7)
    p.add_argument(
        "--tp", type=int, default=0, help="draft group size (0 = whole mesh)"
    )
    p.add_argument(
        "--compat",
        action="store_true",
        help="init_state(sglang_compat=True) + propose_state",
    )
    p.add_argument("--time-capacity", type=int, default=8192)
    p.add_argument("--max-requests", type=int, default=0)
    p.add_argument(
        "--only", type=int, default=-1, help="replay just this request index"
    )
    a = p.parse_args()
    if a.cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    import jax
    import jax.numpy as jnp
    import torch
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.k3_megakernel.mesh import AXIS, make_mesh
    from sglang_tpu.k3_megakernel.server.draft import DraftModel

    mesh = Mesh(np.array(jax.devices()[:8]), (AXIS,)) if a.cpu else make_mesh()
    t0 = time.time()
    dm = DraftModel.load(
        a.draft, mesh, target_ckpt=a.target, ndraft=a.ndraft, tp=a.tp or None
    )
    NDRAFT = dm.ndraft
    print(
        f"loaded draft in {time.time() - t0:.0f}s, world {mesh.devices.size}, dims {dm.d}",
        flush=True,
    )
    R, ps = P(), P(AXIS)

    def smap(fn, in_specs, out_specs, donate=()):
        return jax.jit(
            jax.shard_map(
                fn, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
            ),
            donate_argnums=donate,
        )

    def strip(ds):
        return jax.tree.map(lambda v: v[0], ds)

    def wrap(ds):
        return jax.tree.map(lambda v: v[None], ds)

    app = smap(
        lambda ds, c, pos, n: wrap(dm.append_context(strip(ds), c, pos, n)),
        (ps, R, R, R),
        ps,
    )

    def prop_local(ds, t, pos):
        loc = strip(ds)
        drafts, new = dm.propose_state(loc, t, pos)
        return drafts[None], dm.block_hidden(loc, t, pos)[None], wrap(new)

    prop = smap(prop_local, (ps, R, R), (ps, ps, ps))

    def to_np(t):
        return t.float().numpy() if t.dtype == torch.bfloat16 else t.numpy()

    files = sorted(glob.glob(os.path.join(a.dump, "*.pt")))
    recs = [torch.load(f, weights_only=False) for f in files]
    reqs, cur = [], None
    for r in recs:
        if r["kind"] == "extend":
            cur = [r]
            reqs.append(cur)
        elif cur is not None:
            cur.append(r)
    if a.only >= 0:
        reqs = [reqs[a.only]]
    if a.max_requests:
        reqs = reqs[-a.max_requests :]
    tot = agree = tok_agree = tok_tot = 0
    acc_ref, acc_mine = [], []
    C, H = dm.d.n_caps, dm.d.hidden
    for qi, req in enumerate(reqs):
        ds = dm.init_state(mesh, a.capacity, sglang_compat=a.compat)
        ext = req[0]
        L = int(to_np(ext["seq_lens"])[0])
        hid = to_np(ext["target_hidden"])[:L].reshape(L, C, H)
        pos = to_np(ext["positions"])[:L].astype(np.int32)
        for s in range(0, L, 8):
            m = min(8, L - s)
            c = np.zeros((8, C, H), np.float32)
            c[:m] = hid[s : s + m]
            pp = np.zeros((8,), np.int32)
            pp[:m] = pos[s : s + m]
            ds = app(ds, jnp.asarray(c, jnp.bfloat16), jnp.asarray(pp), jnp.int32(m))
        rounds = [x for x in req[1:] if x["kind"] in ("draft", "commit")]
        hdiff = []
        for i in range(0, len(rounds) - 1, 2):
            dr, cm = rounds[i], rounds[i + 1]
            assert dr["kind"] == "draft" and cm["kind"] == "commit"
            bonus = int(to_np(dr["bonus"])[0])
            sl = int(to_np(dr["seq_lens"])[0])
            ref = to_np(dr["draft_tokens"]).reshape(-1)[:NDRAFT].astype(np.int64)
            mine, bh, ds = prop(ds, jnp.int32(bonus), jnp.int32(sl))
            mine = np.asarray(mine)[0]
            bh = np.asarray(bh[0], np.float32)
            rb = to_np(dr["block_hidden"]).reshape(-1, NDRAFT, H)[0]
            hdiff.append(float(np.abs(bh - rb).max() / (np.abs(rb).max() + 1e-6)))
            tot += 1
            agree += int((mine == ref).all())
            tok_tot += NDRAFT
            tok_agree += int((mine == ref).sum())
            n = int(to_np(cm["accept_lens"]).reshape(-1)[0])
            acc_ref.append(n)
            # acceptance my drafts would get: target argmax rows = sglang's emitted tokens (ref drafts[:n-1] + bonus)
            tgt = list(ref[: n - 1]) + [int(to_np(cm["bonus"]).reshape(-1)[0])]
            k = 0
            while k < len(tgt) and k < NDRAFT and mine[k] == tgt[k]:
                k += 1
            acc_mine.append(
                k + 1
            )  # lower bound when k == n (rows past the bonus are unknown)
            if i < 6 or not (mine == ref).all():
                if tot <= 40:
                    print(
                        f"req {qi} round {i // 2} pos {sl} anchor {bonus} ref {ref.tolist()} mine {mine.tolist()} "
                        f"acc {n} hid_rel {hdiff[-1]:.3g}",
                        flush=True,
                    )
            hc = to_np(cm["target_hidden"])
            nrow = hc.shape[0]
            c = np.zeros((8, C, H), np.float32)
            c[:nrow] = hc.reshape(nrow, C, H)[:8]
            pp = np.zeros((8,), np.int32)
            cp = to_np(cm["positions"]).astype(np.int32)
            pp[: len(cp)] = cp[:8]
            ds = app(ds, jnp.asarray(c, jnp.bfloat16), jnp.asarray(pp), jnp.int32(n))
        print(
            f"req {qi}: prompt {L}, rounds {len(rounds) // 2}, max hidden rel diff {max(hdiff or [0]):.3g}",
            flush=True,
        )
    print(
        f"SUMMARY rounds {tot}: all-6 agree {agree}/{tot} ({agree / max(tot, 1):.3f}), tokens {tok_agree}/{tok_tot} "
        f"({tok_agree / max(tok_tot, 1):.4f}); mean commit len sglang {np.mean(acc_ref):.3f}, mine (est.) {np.mean(acc_mine):.3f}",
        flush=True,
    )

    if a.time:
        bench(dm, mesh, a.time_capacity, smap, strip, wrap)


def bench(dm, mesh, capacity, smap, strip, wrap):
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.k3_megakernel.mesh import AXIS

    ps, R = P(AXIS), P()
    N = 200
    C, H = dm.d.n_caps, dm.d.hidden

    def loop_prop(ds, t):
        def body(i, c):
            tok, acc = c
            d = dm.propose(strip(ds), tok, jnp.int32(capacity // 2) + i)
            return d[0], acc + d.sum()

        return jax.lax.fori_loop(0, N, body, (t, jnp.int32(0)))[1][None]

    def loop_app(ds, caps):
        def body(i, s):
            return dm.append_context(
                s,
                caps,
                jnp.int32(capacity // 2) + 8 * i + jnp.arange(8, dtype=jnp.int32),
                jnp.int32(7),
            )

        return wrap(jax.lax.fori_loop(0, N, body, strip(ds)))

    fp = smap(loop_prop, (ps, R), ps)
    fa = smap(loop_app, (ps, R), ps, (0,))
    ds = dm.init_state(mesh, capacity)
    caps = jnp.ones((8, C, H), jnp.bfloat16)
    for name, fn, args in (("append", fa, (caps,)), ("propose", fp, (jnp.int32(5),))):
        out = fn(ds, *args)
        jax.block_until_ready(out)
        if name == "append":
            ds = out
        t0 = time.time()
        out = fn(ds, *args)
        jax.block_until_ready(out)
        if name == "append":
            ds = out
        print(
            f"TIME {name}: {(time.time() - t0) / N * 1e6:.1f} us/call (capacity {capacity}, fori_loop x{N})",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
