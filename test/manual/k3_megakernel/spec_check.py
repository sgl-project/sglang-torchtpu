# SPDX-License-Identifier: Apache-2.0
"""Single-host check of the standalone engine's speculative loop (server/engine.py ``generate_spec``).

For 3 prompts: plain greedy decode (reference), then spec decoding at every ``--rounds`` value (rounds per
device call) with each draft: ``dspark`` (the real DSpark draft), ``zero`` (stub drafting token 0: acceptance
~0, one token per round) or ``replay`` (stub drafting the reference output: full acceptance while the target
matches it). One ``SPECCHK`` json line per (prompt, draft, R): ``same_as_first_R`` must hold (spec output
independent of R); ``equal`` (== plain decode) holds with ``--verify-mode loop`` (bit-identical verify) and
not in general with the rows verify (f32 reorder vs B1 decode).

  python3 test/manual/k3_megakernel/spec_check.py [--rounds 1,8] [--draft zero,replay | dspark --draft-path P] \\
      [--verify-mode rows|loop] -- --layers 13 --expert-load stream
(PYTHONPATH=<tree>/python:<tree>, TPU env of one host as in server/run1h.sh.)
"""

import argparse
import hashlib
import json
import sys

import numpy as np

NDRAFT = 7


class ReplayDraft:
    """Stub draft (the engine's draft protocol): drafts ``ref[pos + 1 : pos + 8]`` of the reference sequence
    ``ref`` (prompt + greedy output) set by ``set_ref`` before the request; an empty ``ref`` drafts zeros."""

    def init_state(self, mesh, capacity):
        self.mesh, self.capacity = mesh, capacity
        return {"ref": self._ref([])}

    def _ref(self, ref):
        from sglang_tpu.k3_megakernel import model_io

        buf = np.zeros((self.capacity + 16,), np.int32)
        buf[: len(ref)] = ref
        return model_io.put_replicated(self.mesh, buf)

    def set_ref(self, engine, ref):
        engine.dstate = {"ref": self._ref(ref)}

    def append_context(self, ds, captures, positions, n_valid):
        return ds

    def propose(self, ds, anchor, pos, embed=None, lm_head=None):
        import jax

        return jax.lax.dynamic_slice_in_dim(ds["ref"], pos + 1, NDRAFT)


def _sha(ids):
    return hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest()[:16]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    i = argv.index("--") if "--" in argv else len(argv)
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", default="1,8", help="rounds per device call to compare")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--capacity", type=int, default=4096)
    p.add_argument("--draft", default="zero,replay", help="zero,replay | dspark")
    p.add_argument("--draft-path", default="")
    p.add_argument(
        "--draft-tp", type=int, default=0, help="draft group size (0 = whole mesh)"
    )
    p.add_argument("--verify-mode", default="rows", choices=("rows", "loop"))
    ns = p.parse_args(argv[:i])
    from sglang_tpu.k3_megakernel import mesh as mesh_mod
    from sglang_tpu.k3_megakernel.server import loader

    mp = argparse.ArgumentParser()
    loader.add_model_args(mp)
    args = mp.parse_args(argv[i + 1 :])
    mesh_mod.init_distributed(coordinator="", num_processes=1, process_id=0)
    from sglang_tpu.k3_megakernel.server.engine import Engine

    mesh = mesh_mod.make_mesh()
    kinds = ns.draft.split(",")
    if "dspark" in kinds:
        from sglang_tpu.k3_megakernel.server.draft import DraftModel

        kinds = ["dspark"]  # the engine holds one draft: dspark runs alone
        draft = DraftModel.load(
            ns.draft_path, mesh, embed_mode=args.embed, tp=ns.draft_tp or None
        )
    else:
        draft = ReplayDraft()
    eng = Engine(
        args,
        mesh,
        decode_steps=32,
        capacity=ns.capacity,
        lookahead=1,
        draft=draft,
        verify_mode=ns.verify_mode,
    )
    prompts = [
        [1008, 18985, 387] * 3,
        list(range(1000, 1100, 3)),
        [3017, 2056, 4001, 512, 99, 77, 1234, 4321, 555, 8080, 70, 71, 72],
    ]
    ok, first = True, {}
    for pi, pr in enumerate(prompts):
        eng.generate(pr, ns.max_tokens)  # compile
        ref, _, tref = eng.generate(pr, ns.max_tokens)
        print(
            f"PROMPT {pi} plain sha {_sha(ref)} tpot {tref['tpot_ms']:.2f} ms first {ref[:20]}",
            flush=True,
        )
        for kind in kinds:
            for R in [int(x) for x in ns.rounds.split(",")]:
                for _ in range(2):  # the 2nd run is timed (compiled)
                    if kind != "dspark":
                        draft.set_ref(eng, [] if kind == "zero" else pr + ref)
                    ids, _, t = eng.generate_spec(pr, ns.max_tokens, rounds=R)
                f0 = first.setdefault((pi, kind), ids)
                r = dict(
                    prompt=pi,
                    draft=kind,
                    R=R,
                    n=len(ids),
                    sha=_sha(ids),
                    equal=ids == ref,
                    first_div=next(
                        (k for k, (a, b) in enumerate(zip(ids, ref)) if a != b), None
                    ),
                    same_as_first_R=ids == f0,
                    **{
                        k: t[k]
                        for k in (
                            "accept_length",
                            "ms_per_round",
                            "tpot_ms",
                            "rounds_run",
                        )
                    },
                )
                ok &= r["same_as_first_R"]
                print("SPECCHK " + json.dumps(r), flush=True)
    print(f"SPECCHK_OK {ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
