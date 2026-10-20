# SPDX-License-Identifier: Apache-2.0
"""Anchor-protocol device-loop benchmark of the K3 decode megakernel (pure JAX).

Anchor protocol: prompt [1008, 18985, 387], capacity 1152,
1000 requested tokens (the first comes from the prompt, so 999 decode steps
are timed), greedy, EOS ignored, 5 warmups + 25 measured requests; each
request's prefill and decode times are taken on every host and the record
keeps the slowest host. The prompt is fed through the same B1 step.

Single host (one process, 8 local cores):
  python -m bench.anchor --layers 13
Multi-host: one process per host with --coordinator/--num-processes/--process-id
(bench/run4h.sh derives them from the TPU worker environment).

Kernel-only time per step: rerun with ``--debug-repeat 2`` (the stack runs twice per
step; tokens become meaningless); kernel = t2 - t1, everything else = 2 * t1 - t2.
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

from sglang_tpu.k3_megakernel.server import loader

ANCHOR = loader.ANCHOR


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    loader.add_model_args(p)
    p.add_argument(
        "--prompt", default=",".join(map(str, ANCHOR)), help="comma-separated token ids"
    )
    p.add_argument(
        "--prompt-text", default="", help="tokenize this text instead (CPU tokenizer)"
    )
    p.add_argument("--capacity", type=int, default=1152)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="requested tokens; max_tokens - 1 are timed",
    )
    p.add_argument("--steps-per-call", type=int, default=999)
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--requests", type=int, default=25)
    p.add_argument(
        "--check-experts",
        default="",
        metavar="LAYERS",
        help="instead of timing: compare the loaded routed experts of these MoE layers (comma list, "
        "1-based) with a fresh CPU reblock; writes check_experts.p<k>.json next to --out",
    )
    p.add_argument("--debug-repeat", type=int, default=1)
    p.add_argument(
        "--debug-skip", default="", help="comma list of attn,moe (benchmark-only)"
    )
    p.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar='NAME="ARGS"',
        help="after the main run, rerun with these extra anchor args on the same loaded weights "
        "(kernel / program options only), tagged <tag>.NAME; repeatable",
    )
    p.add_argument(
        "--variant-requests",
        type=int,
        default=3,
        help="measured requests per variant (1 warmup)",
    )
    p.add_argument(
        "--gsm8k",
        type=int,
        default=0,
        metavar="N",
        help="instead of timing: N GSM8K questions (bench/gsm8k.py); records + accuracy to --out",
    )
    p.add_argument(
        "--gsm8k-data",
        default="",
        help="GSM8K test jsonl (first 5 lines are the shots)",
    )
    p.add_argument("--gsm8k-seed", type=int, default=0)
    p.add_argument("--gsm8k-max-new", type=int, default=512)
    p.add_argument(
        "--gsm8k-think",
        action="store_true",
        help="thinking channel (default: response channel)",
    )
    p.add_argument(
        "--verify-acc",
        type=int,
        default=0,
        metavar="N",
        help="speculative verify (8 rows per step) with fixed acceptance N tokens per step; "
        "--steps-per-call counts verify steps",
    )
    p.add_argument(
        "--verify-check",
        action="store_true",
        help="instead of timing: one verify call vs 8 forced B1 steps from the same state",
    )
    p.add_argument(
        "--probe",
        action="store_true",
        help="debug: print per-layer outputs / expert ids of the prefill",
    )
    p.add_argument(
        "--probe-save",
        default="",
        help="debug: with --probe, also save them to this .npz",
    )
    p.add_argument(
        "--expect", default="", help="comma-separated expected first generated ids"
    )
    p.add_argument("--out", default="", help="append JSON records here (process 0)")
    p.add_argument("--tag", default="")
    p.add_argument(
        "--probe-mesh", action="store_true", help="print the rank layout and exit"
    )
    p.add_argument(
        "--build-cache",
        type=int,
        default=0,
        metavar="WORLD",
        help="CPU only: build this host's (TPU_WORKER_ID) expert cache for WORLD ranks and exit",
    )
    return p.parse_args(argv)


def _log(msg):
    import jax

    print(
        f"[anchor p{jax.process_index()} {time.strftime('%H:%M:%S')}] {msg}", flush=True
    )


def _sha(tokens) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(tokens, np.int64).tobytes()).hexdigest()[:16]


def _git_rev() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(
            ["git", "-C", here, "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _prompt_ids(args) -> list[int]:
    if not args.prompt_text:
        return [int(x) for x in args.prompt.split(",") if x]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    return list(tok.encode(args.prompt_text))


def _kernel_kwargs(args) -> dict:
    """StackKernel keywords from the CLI (benchmark-only debug knobs included)."""
    extra = {}
    if args.debug_repeat != 1:
        extra["debug_repeat"] = args.debug_repeat
    if args.debug_skip:
        extra["debug_skip"] = frozenset(x for x in args.debug_skip.split(",") if x)
    return loader.kernel_kwargs(args, extra)


stack_spec = loader.stack_spec
_expert_load = loader.expert_load
_expert_cache_dir = loader.expert_cache_dir


def load_weights(args, mesh, experts=None):
    return loader.load_weights(args, mesh, experts, log_fn=_log)


class Bench:
    """Weights, compiled programs and per-request state for one configuration."""

    def __init__(self, args, mesh, weights=None):
        import jax
        import numpy as np
        from bench import program

        from sglang_tpu.k3_megakernel import model_io
        from sglang_tpu.k3_megakernel.kernel.stack import StackKernel

        self.args, self.mesh = args, mesh
        world = mesh.devices.size
        self.spec = stack_spec(args, mesh)
        self.kernel_kw = _kernel_kwargs(args)
        self.kernel = StackKernel(self.spec, **self.kernel_kw)
        _log(f"spec {self.spec}; kernel options {self.kernel_kw}")
        self.stack_w, self.vocab_w, self.load_s = weights or load_weights(args, mesh)
        vocab = self.vocab_w["lm_head"].shape[1] * world
        self.consts = {
            "tables": model_io.put_replicated(
                mesh, np.asarray(self.spec.tables(), np.int32)
            ),
            "pages": model_io.put_replicated(mesh, model_io.page_table(args.capacity)),
        }
        self.verify = bool(args.verify_acc or args.verify_check)
        st_shapes = model_io.state_shapes(
            self.kernel, self.spec, args.capacity, kda_rows=8 if self.verify else 1
        )
        self.fresh_states = model_io.make_fresh_states(mesh, st_shapes)
        shapes = {n: (tuple(a.shape[1:]), a.dtype) for n, a in self.stack_w.items()}
        shapes.update(st_shapes)
        shapes.update(
            tables=((4, args.layers), np.int32),
            meta=((2,), np.int32),
            pages=(tuple(self.consts["pages"].shape[1:]), np.int32),
            x0=((1, 7168), jax.numpy.bfloat16),
        )
        shapes.update(
            {n: (tuple(a.shape[1:]), a.dtype) for n, a in self.vocab_w.items()}
        )
        self.shapes, self.vocab = shapes, vocab
        self.prompt = _prompt_ids(args)
        decode_steps = args.max_tokens - 1
        spc = min(args.steps_per_call, decode_steps)
        self.chunks = [spc] * (decode_steps // spc) + (
            [decode_steps % spc] if decode_steps % spc else []
        )
        build = lambda steps, forced: program.build_program(
            mesh=mesh,
            kernel=self.kernel,
            spec=self.spec,
            shapes=shapes,
            steps=steps,
            forced=forced,
            embed_mode=args.embed,
            vocab=vocab,
        )
        self.prefill = build(len(self.prompt), True)
        if args.probe:
            self.probe = program.build_program(
                mesh=mesh,
                kernel=self.kernel,
                spec=self.spec,
                shapes=shapes,
                steps=len(self.prompt),
                forced=True,
                embed_mode=args.embed,
                vocab=vocab,
                probe=True,
            )
        self.decode = {n: build(n, False) for n in sorted(set(self.chunks))}
        if self.verify:
            self._build_verify(shapes, vocab)

    def _build_verify(self, shapes, vocab):
        import jax
        import numpy as np
        from bench import program, verify

        from sglang_tpu.k3_megakernel.kernel.stack import StackKernel

        args, rows = self.args, 8
        kw = dict(self.kernel_kw)
        self.capture = tuple(
            kw.pop("capture_layers", verify.capture_layers(self.spec.num_layers))
        )
        self.vkernel = StackKernel(
            self.spec, **kw, verify_rows=rows, capture_layers=self.capture
        )
        x0 = (rows, 7168) if self.vkernel.rows_mode else (rows, 1, 7168)
        vshapes = dict(shapes, meta=((6,), np.int32), x0=(x0, jax.numpy.bfloat16))
        if os.environ.get(
            "K3MK_DUMP_SHAPES"
        ):  # debug: per-rank verify kernel input shapes (CPU tracing)
            with open(os.environ["K3MK_DUMP_SHAPES"], "w") as fh:
                json.dump(
                    {
                        n: [list(sh), str(np.dtype(dt))]
                        for n, (sh, dt) in vshapes.items()
                    },
                    fh,
                )
        if args.verify_check:
            okernel = StackKernel(
                self.spec, **dict(kw, layer_out=True, fused_head=False)
            )
            self.oracle = verify.build_oracle_program(
                mesh=self.mesh,
                kernel=okernel,
                spec=self.spec,
                shapes=shapes,
                rows=rows,
                capture=self.capture,
                embed_mode=args.embed,
                vocab=vocab,
            )
            self.vonce = verify.build_verify_once(
                mesh=self.mesh,
                kernel=self.vkernel,
                spec=self.spec,
                shapes=vshapes,
                embed_mode=args.embed,
                vocab=vocab,
            )
            self.decode7 = program.build_program(
                mesh=self.mesh,
                kernel=self.kernel,
                spec=self.spec,
                shapes=shapes,
                steps=rows - 1,
                forced=False,
                embed_mode=args.embed,
                vocab=vocab,
            )
            return
        acc = args.verify_acc
        steps = -(-(args.max_tokens - 1) // acc)
        spc = min(args.steps_per_call, steps)
        self.vchunks = [spc] * (steps // spc) + ([steps % spc] if steps % spc else [])
        self.vdecode = {
            n: verify.build_verify_program(
                mesh=self.mesh,
                kernel=self.vkernel,
                spec=self.spec,
                shapes=vshapes,
                steps=n,
                acc=acc,
                embed_mode=args.embed,
                vocab=vocab,
            )
            for n in sorted(set(self.vchunks))
        }

    def _rows(self, values, dtype):
        import numpy as np

        from sglang_tpu.k3_megakernel.model_io import put_replicated

        return put_replicated(self.mesh, np.asarray(values, dtype))

    def run_probe(self):
        """Debug: per-layer output norms and expert ids of rank 0 after the prompt."""
        import numpy as np

        states = self.fresh_states()
        forced = self._rows(self.prompt, np.int32)
        tok, kv = self._rows(0, np.int32), self._rows(1, np.int32)
        _, _, _, out, lo, ids = self.probe(
            self.stack_w, self.vocab_w, self.consts, states, tok, kv, forced
        )
        lo = np.asarray(lo.addressable_shards[0].data)[0].astype(np.float32)
        ids = np.asarray(ids.addressable_shards[0].data)[0]
        if self.args.probe_save:
            np.savez(self.args.probe_save, layer_out=lo, ids=ids)
        for l in range(lo.shape[0]):
            v = lo[l, 0]
            _log(
                f"PROBE {self.args.tag} L{l:02d} norm={np.linalg.norm(v):.6e} sum={v.sum():.6e} "
                f"nan={int(np.isnan(v).sum())} ids={ids[l].tolist()}"
            )

    def request(self):
        """One anchor request; returns (prefill_s, decode_s, generated ids) on this host."""
        import jax
        import numpy as np

        states = self.fresh_states()
        forced = self._rows(self.prompt, np.int32)
        tok, kv = self._rows(0, np.int32), self._rows(1, np.int32)
        dummy = self._rows([0], np.int32)
        jax.block_until_ready((states, forced, tok, kv, dummy))
        t0 = time.perf_counter()
        states, tok, kv, out = self.prefill(
            self.stack_w, self.vocab_w, self.consts, states, tok, kv, forced
        )
        jax.block_until_ready(out)
        prefill_s = time.perf_counter() - t0
        if self.verify:
            return (prefill_s,) + self._verify_decode(states, tok, kv, out)
        outs = [out]
        t0 = time.perf_counter()
        for n in self.chunks:
            states, tok, kv, out = self.decode[n](
                self.stack_w, self.vocab_w, self.consts, states, tok, kv, dummy
            )
            jax.block_until_ready(out)
            outs.append(out)
        decode_s = time.perf_counter() - t0
        rows = [np.asarray(o.addressable_shards[0].data)[0] for o in outs]
        generated = [int(rows[0][-1])] + [int(t) for r in rows[1:] for t in r]
        return prefill_s, decode_s, generated


def _verify_decode(self, states, tok, kv, out):
    """Fixed-acceptance verify steps after the prefill; returns (decode_s, generated ids)."""
    import jax
    import numpy as np

    rr = self._rows(0, np.int32)
    drafts = self._rows([0] * 7, np.int32)
    outs = [out]
    t0 = time.perf_counter()
    for n in self.vchunks:
        states, tok, kv, rr, drafts, out = self.vdecode[n](
            self.stack_w, self.vocab_w, self.consts, states, tok, kv, rr, drafts
        )
        jax.block_until_ready(out)
        outs.append(out)
    decode_s = time.perf_counter() - t0
    rows = [np.asarray(o.addressable_shards[0].data)[0] for o in outs]
    generated = [int(rows[0][-1])] + [int(t) for r in rows[1:] for t in r]
    return decode_s, generated


Bench._verify_decode = _verify_decode


def _slowest(values):
    import jax
    import numpy as np
    from jax.experimental import multihost_utils

    v = np.asarray(values, np.float64)
    if jax.process_count() == 1:
        return v
    return (
        np.asarray(multihost_utils.process_allgather(v)).reshape(-1, len(v)).max(axis=0)
    )


def check_experts(args, mesh, stack_w) -> dict:
    """``{family: [(layer, rank), ...]}`` where the resident experts differ from a fresh reblock."""
    import jax
    import numpy as np

    from sglang_tpu.k3_megakernel.config import MoeLayout
    from sglang_tpu.k3_megakernel.mesh import local_ranks
    from sglang_tpu.k3_megakernel.weights import stream_expert_layers

    spec = stack_spec(args, mesh)
    layers = [int(x) for x in args.check_experts.split(",") if x]
    ranks = local_ranks(mesh)
    devs = list(mesh.devices.flat)
    bad: dict = {}
    for i, fresh in stream_expert_layers(
        args.ckpt,
        MoeLayout(spec.dims, spec.topo, replicas=spec.relax.expert_replicas),
        layers,
        ranks,
        procs=48,
    ):
        for name, a in fresh.items():
            shards = {s.device: s.data for s in stack_w[name].addressable_shards}
            for k, r in enumerate(ranks):
                got = np.asarray(shards[devs[r]][0, layers[i] - 1])
                if (
                    got.view(np.uint8).tobytes()
                    != np.ascontiguousarray(a[k]).view(np.uint8).tobytes()
                ):
                    bad.setdefault(name, []).append((layers[i], r))
        _log(f"check_experts: layer {layers[i]}: mismatches so far {bad}")
    out = os.path.join(
        os.path.dirname(args.out) or ".",
        f"check_experts.{args.tag}.p{jax.process_index()}.json",
    )
    with open(out, "w") as f:
        json.dump(bad, f)
    return bad


def run(args, mesh, weights=None):
    import jax
    import numpy as np

    if args.check_experts:
        check_experts(args, mesh, weights[0])
        return None
    t0 = time.time()
    bench = Bench(args, mesh, weights)
    if args.gsm8k:
        from bench import gsm8k

        gsm8k.run(args, bench, _log)
        return bench
    if args.verify_check:
        from bench import verify

        _log("VERIFY_CHECK " + json.dumps(verify.run_check(bench)))
        return bench
    if args.probe:
        bench.run_probe()
    expect = [int(x) for x in args.expect.split(",") if x]
    records = []
    for index in range(args.warmups + args.requests):
        warm = index < args.warmups
        prefill_s, decode_s, gen = bench.request()
        # Multi-host gathers come back float32 (x64 off); json needs Python floats.
        prefill_s, decode_s = map(float, _slowest([prefill_s, decode_s]))
        n = len(gen) - 1
        rec = dict(
            tag=args.tag,
            index=index,
            warmup=warm,
            prefill_s=float(prefill_s),
            decode_s=float(decode_s),
            decode_tokens=n,
            ms_per_step=1e3 * decode_s / n,
            tps=n / decode_s,
            tokens_sha=_sha(gen),
            first_tokens=gen[:12],
        )
        if index == 0:
            rec["compile_and_first_s"] = (
                time.time() - t0 - (bench.load_s if weights is None else 0.0)
            )
            tokens0 = gen
        records.append(rec)
        _log(json.dumps(rec))
        if args.out and jax.process_index() == 0:
            with open(args.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
    measured = [r for r in records if not r["warmup"]] or records
    ms = np.array([r["ms_per_step"] for r in measured])
    shas = sorted({r["tokens_sha"] for r in records})
    first = records[0]["first_tokens"]
    summary = dict(
        tag=args.tag,
        summary=True,
        git=_git_rev(),
        host=socket.gethostname(),
        world=mesh.devices.size,
        processes=jax.process_count(),
        layers=args.layers,
        mla_layers=list(bench.spec.mla_layers),
        embed=args.embed,
        relax=args.relax,
        expert_load=_expert_load(args),
        kernel_opts={k: repr(v) for k, v in bench.kernel_kw.items()},
        prompt=bench.prompt,
        capacity=args.capacity,
        max_tokens=args.max_tokens,
        chunks=bench.chunks,
        warmups=args.warmups,
        requests=args.requests,
        load_s=bench.load_s,
        compile_and_first_s=records[0]["compile_and_first_s"],
        ms_per_step_mean=float(ms.mean()),
        ms_per_step_median=float(np.median(ms)),
        ms_per_step_min=float(ms.min()),
        ms_per_step_max=float(ms.max()),
        ms_per_step_sd=float(ms.std(ddof=1)) if len(ms) > 1 else 0.0,
        tps_mean=float(np.mean([r["tps"] for r in measured])),
        prefill_ms_mean=1e3 * float(np.mean([r["prefill_s"] for r in measured])),
        tokens_sha=shas,
        deterministic=len(shas) == 1,
        first_tokens=first,
        tokens=tokens0,
    )
    if expect:
        match = 0
        while match < min(len(expect), len(first)) and expect[match] == first[match]:
            match += 1
        summary["expect_prefix_match"] = match
    _log("SUMMARY " + json.dumps(summary))
    if args.out and jax.process_index() == 0:
        with open(args.out, "a") as fh:
            fh.write(json.dumps(summary) + "\n")
    return bench


def _variant_args(args, argv, spec: str):
    """Args of one ``--variant NAME=ARGS``: the main run's argv plus ARGS; weight-shaping args must not change."""
    import shlex

    name, _, extra = spec.partition("=")
    vargs = _parse_args(
        list(argv)
        + shlex.split(extra)
        + [
            "--tag",
            f"{args.tag}.{name}",
            "--warmups",
            "1",
            "--requests",
            str(args.variant_requests),
        ]
    )
    for key in ("ckpt", "layers", "topology", "embed", "expert_load", "expert_cache"):
        if getattr(vargs, key) != getattr(args, key):
            raise SystemExit(
                f"variant {name!r} changes --{key.replace('_', '-')}, which needs a reload"
            )
    return vargs


def _build_cache_only(args) -> int:
    """Run with JAX_PLATFORMS=cpu: no TPU is touched, so it can precede the timed run."""
    from sglang_tpu.k3_megakernel.config import K3Dims, MoeLayout, Topology
    from sglang_tpu.k3_megakernel.weights import build_expert_cache

    world, host = args.build_cache, int(os.environ.get("TPU_WORKER_ID", "0"))
    layout = MoeLayout(K3Dims(), Topology(hosts=world // 8, lanes=8))
    path = _expert_cache_dir(args, world)
    t0 = time.time()
    build_expert_cache(
        args.ckpt,
        layout,
        range(1, args.layers),
        path,
        ranks=range(8 * host, 8 * host + 8),
    )
    print(
        f"expert cache {path} (host {host}, {world} ranks, {args.layers} layers) ready in {time.time() - t0:.0f}s",
        flush=True,
    )
    return 0


def main(argv=None):
    args = _parse_args(argv)
    if args.build_cache:
        return _build_cache_only(args)
    if args.relax:
        from sglang_tpu.envs import envs as tpu_envs

        tpu_envs.SGLANG_TPU_OPT_K3_MEGAKERNEL_RELAX.set(args.relax)
    from sglang_tpu.k3_megakernel import mesh as mesh_mod

    mesh_mod.init_distributed(
        coordinator=args.coordinator,
        num_processes=args.num_processes,
        process_id=args.process_id,
    )
    import jax

    _log(
        f"processes {jax.process_count()} local devices {jax.local_device_count()} global {jax.device_count()}"
    )
    if args.probe_mesh:
        for line in mesh_mod.describe(sorted(jax.devices(), key=mesh_mod.sort_key)):
            _log(line)
        mesh_mod.ordered_devices()
        _log("rank layout ok")
        return 0
    mesh = mesh_mod.make_mesh()
    bench = run(args, mesh)
    argv = sys.argv[1:] if argv is None else argv
    for spec in args.variant:
        run(
            _variant_args(args, argv, spec),
            mesh,
            weights=(bench.stack_w, bench.vocab_w, bench.load_s),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
