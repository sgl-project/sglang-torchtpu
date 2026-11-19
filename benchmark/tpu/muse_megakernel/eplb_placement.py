"""Static expert placement for the Muse Spark EP megakernels from routed-expert counts.

Reads the per-rank dumps of SGLANG_TPU_MUSE_EXPERT_COUNTS_DIR (counts_rank<r>.npy, the SP
prefill program's routed (row, slot) counts per layer and expert), and writes a placement JSON
for SGLANG_TPU_MUSE_EXPERT_PLACEMENT: per layer, slot -> checkpoint expert, slots s of EP rank
s // (E / ranks).

Decode cost on a rank follows the distinct experts it streams in a step, so the objective is
each rank's expected distinct hits at R decode rows, p_hit(e) = 1 - (1 - p_e) ** R with p_e the
chance one row picks expert e (count / rows). Experts are assigned largest p_hit first to the
rank with the smallest expected load that still has a free slot (LPT). A Monte Carlo replay of
decode steps drawn from the same per-expert popularity reports the busiest rank's distinct
experts per step under the identity and the new placement.

    python benchmark/tpu/muse_megakernel/eplb_placement.py COUNTS_DIR --out placement.json \\
        [--rows 16] [--ranks 8] [--top-k 8] [--layers 62]
"""

import argparse
import glob
import json
import os

import numpy as np


def load_counts(counts_dir: str, n_layers: int, ranks: int) -> np.ndarray:
    """[n_layers, E] int64 total routed slots; a dump holding the global [ranks * L, E] array
    (every rank's copy) is folded once, local [L, E] dumps are summed over ranks."""
    files = sorted(glob.glob(os.path.join(counts_dir, "counts_rank*.npy")))
    if not files:
        raise SystemExit(f"no counts_rank*.npy under {counts_dir}")
    arrays = [np.load(f) for f in files]
    first = arrays[0]
    if first.shape[0] == ranks * n_layers:
        return first.reshape(ranks, n_layers, -1).sum(axis=0)
    if first.shape[0] != n_layers:
        raise SystemExit(f"{files[0]}: shape {first.shape}, expected [{n_layers}, E] or "
                         f"[{ranks * n_layers}, E]")
    return np.sum(arrays, axis=0)


def hit_prob(counts: np.ndarray, top_k: int, rows: int) -> np.ndarray:
    """[E]: the chance an expert is hit by at least one of `rows` rows."""
    n_rows = max(counts.sum() / top_k, 1.0)
    p = np.clip(counts / n_rows, 0.0, 1.0)
    return 1.0 - (1.0 - p) ** rows


def lpt_placement(p_hit: np.ndarray, ranks: int) -> np.ndarray:
    """[E] slot -> expert: largest expected hit first onto the least-loaded rank with room."""
    E = p_hit.shape[0]
    cap = E // ranks
    load = np.zeros(ranks)
    members: list[list[int]] = [[] for _ in range(ranks)]
    for e in np.argsort(-p_hit, kind="stable"):
        open_ranks = [g for g in range(ranks) if len(members[g]) < cap]
        g = min(open_ranks, key=lambda r: load[r])
        members[g].append(int(e))
        load[g] += p_hit[e]
    return np.asarray([e for g in range(ranks) for e in sorted(members[g])], dtype=np.int64)


def replay(counts: np.ndarray, slot_expert: np.ndarray, *, rows: int, top_k: int,
           ranks: int, steps: int, rng: np.random.Generator) -> float:
    """Mean over sampled decode steps of the busiest rank's distinct experts."""
    E = counts.shape[0]
    logp = np.log(np.maximum(counts, 1e-9) / max(counts.sum(), 1.0))
    rank_of = np.empty(E, dtype=np.int64)
    rank_of[slot_expert] = np.arange(E) // (E // ranks)
    busiest = 0.0
    for _ in range(steps):
        g = logp[None, :] + rng.gumbel(size=(rows, E))
        hit = np.unique(np.argpartition(-g, top_k, axis=1)[:, :top_k])
        busiest += np.bincount(rank_of[hit], minlength=ranks).max()
    return busiest / steps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("counts_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=16, help="decode rows per step to balance for")
    ap.add_argument("--ranks", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--layers", type=int, default=62)
    ap.add_argument("--steps", type=int, default=300, help="Monte Carlo decode steps per layer")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    counts = load_counts(args.counts_dir, args.layers, args.ranks).astype(np.float64)
    rng = np.random.default_rng(args.seed)
    E = counts.shape[1]
    identity = np.arange(E, dtype=np.int64)
    placement, before, after, skew = [], [], [], []
    for l in range(args.layers):
        c = counts[l]
        share = c / max(c.sum(), 1.0)
        skew.append(share.max() * E)  # hottest expert's load vs uniform
        perm = lpt_placement(hit_prob(c, args.top_k, args.rows), args.ranks)
        placement.append(perm.tolist())
        kw = dict(rows=args.rows, top_k=args.top_k, ranks=args.ranks, steps=args.steps, rng=rng)
        before.append(replay(c, identity, **kw))
        after.append(replay(c, perm, **kw))
    with open(args.out, "w") as f:
        json.dump({"placement": placement, "rows": args.rows,
                   "source": os.path.abspath(args.counts_dir)}, f)
    b, a = np.asarray(before), np.asarray(after)
    print(f"routed slots: {counts.sum():.0f} over {args.layers} layers; hottest expert vs uniform: "
          f"median {np.median(skew):.2f}x, max {np.max(skew):.2f}x")
    print(f"busiest rank distinct experts per step at {args.rows} rows (Monte Carlo): identity "
          f"{b.mean():.2f} -> placement {a.mean():.2f} ({(a.mean() / b.mean() - 1) * 100:+.1f}%)")
    worst = np.argsort(a - b)[:5]
    print("largest per-layer gains:", ", ".join(f"L{l}: {b[l]:.1f}->{a[l]:.1f}" for l in worst))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
