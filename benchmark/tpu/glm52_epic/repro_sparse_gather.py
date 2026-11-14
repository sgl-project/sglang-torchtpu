#!/usr/bin/env python3
"""Single-device isolation of GLM52's per-query SC gather.

Run each case in a fresh process on the pinned TPU runtime. No checkpoint or
server required. Raw JAX is default; --backend torch tests the bridge separately.
Neither mode tests full attention, cache donation, or model decode.
Example: python -m benchmark.tpu.glm52_epic.repro_sparse_gather --queries 1
Repeat with --queries 64 for one default prefill gather chunk.
"""
import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=int, choices=(1, 4, 64), default=1)
    parser.add_argument('--page-size', type=int, choices=(64, 128), default=128)
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--backend', choices=('jax', 'torch'), default='jax',
                        help='Raw JAX or the TorchTPU custom_jax_kernel bridge.')
    parser.add_argument('--live-queries', type=int, default=None,
                        help='Live prefix of the padded query bucket; default all.')
    parser.add_argument('--evidence-dir', type=Path,
                        help='New directory for run metadata and precompile raw-JAX IR.')
    args = parser.parse_args()
    evidence = args.evidence_dir
    live_queries = args.queries if args.live_queries is None else args.live_queries
    if not 0 <= live_queries <= args.queries:
        parser.error('--live-queries must be between zero and --queries')
    if args.backend == 'torch' and args.compile_only:
        parser.error('--compile-only is raw-JAX only; bridge compiles lazily on invocation')
    if 'jax' in sys.modules:
        raise RuntimeError('run in a fresh process before JAX initialization')
    if evidence is not None:
        # Invalid CLI arguments do not create artifacts; valid cases never
        # overwrite another case's evidence, including an interrupted run.
        evidence.mkdir(parents=True, exist_ok=False)
        (evidence / 'invocation.json').write_text(json.dumps(
            {**vars(args), 'evidence_dir': str(evidence)}, indent=2) + '\n')
    pins = {'jax': '0.11.2', 'jaxlib': '0.11.2', 'libtpu': '0.0.50'}
    versions = {name: importlib.metadata.version(name) for name in pins}
    print('RUNTIME ' + json.dumps(versions, sort_keys=True), flush=True)

    def stage(name):
        record = {'stage': name, 'time_ns': time.time_ns(), 'backend': args.backend}
        print('STAGE ' + json.dumps(record, sort_keys=True), flush=True)
        if evidence is not None:
            with (evidence / 'stages.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
                stream.flush()

    stage('runtime_checked')
    if evidence is not None:
        (evidence / 'runtime.json').write_text(json.dumps(versions, indent=2) + '\n')
    if versions != pins:
        raise RuntimeError(f'pinned runtime required: {pins}; found {versions}')
    os.environ['JAX_PLATFORMS'] = 'tpu' if args.backend == 'jax' else 'cpu'
    import jax
    import jax.numpy as jnp
    import numpy as np
    from sglang_tpu.attention.pallas.mla_sparse.dsa_gather import dsa_gather

    stage('initialize_jax')
    device = jax.devices()[0]
    if args.backend == 'jax' and device.platform != 'tpu':
        raise RuntimeError('TPU required; CPU fallback is not a reproduction')
    from jax.experimental.pallas import tpu as pltpu
    if args.backend == 'jax' and pltpu.get_tpu_info().sparse_core is None:
        raise RuntimeError('SparseCore-capable TPU required')
    topk, pages = 2048, 32
    n = args.queries * topk
    live = live_queries * topk
    rng = np.random.default_rng(42)
    nope = rng.integers(0, 256, (pages * args.page_size, 512), dtype=np.uint8)
    rope = rng.integers(0, 256, (pages * args.page_size, 128), dtype=np.uint8)
    indices = rng.integers(0, len(nope), n, dtype=np.int32)
    if args.backend == 'torch':
        import torch
        import torch_tpu  # noqa: F401 (register PrivateUse1 backend)
        from torch_tpu._internal import pallas as torch_pallas
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export
        required = {'torch_tpu': '0.2.1.dev20261028201649',
                    'tpu-sync-torch': '0.0.1.dev20261031081414'}
        found = {name: importlib.metadata.version(name) for name in required}
        print('TORCH_RUNTIME ' + json.dumps({'torch': torch.__version__, **found}, sort_keys=True), flush=True)
        if found != required:
            raise RuntimeError(f'TorchTPU pins required: {required}; found {found}')
        if torch.__version__ not in ('2.11.0', '2.11.0+cpu'):
            raise RuntimeError('torch 2.11.0 CPU ABI required')

        def bridge_gather(a: jax.Array, b: jax.Array, c: jax.Array,
                          valid: jax.Array) -> tuple[jax.Array, jax.Array]:
            return dsa_gather(a, b, c, num_valid_indices=valid)

        kernel = torch_pallas.custom_jax_kernel(
            bridge_gather, name='glm52_repro_sc_gather', warn_deprecated=False)
        arrays = (nope.reshape(pages, args.page_size, 4, 128),
                  rope.reshape(pages, args.page_size // 4, 4, 128),
                  indices, np.asarray(live, dtype=np.int32))
        stage('initialize_torch_and_upload')
        tensors = tuple(torch.from_numpy(a).to('tpu') for a in arrays)
        stage('bridge_compile_and_execute')
        with torch.no_grad(), tpu_info_for_cpu_export():
            outputs = kernel(*tensors)
            actual_nope, actual_rope = (out.cpu().numpy() for out in outputs)
        assert actual_nope.shape == (n, 4, 128)
        assert actual_rope.shape == (n, 128)
        np.testing.assert_array_equal(actual_nope.reshape(n, 512)[:live], nope[indices[:live]])
        np.testing.assert_array_equal(actual_rope[:live], rope[indices[:live]])
        print('TORCH_BRIDGE_GATHER_BYTE_EXACT_PASS (not a serving baseline)', flush=True)
        return
    with jax.default_device(device):
        inputs = (jnp.asarray(nope.reshape(pages, args.page_size, 4, 128)),
                  jnp.asarray(rope.reshape(pages, args.page_size // 4, 4, 128)),
                  jnp.asarray(indices), jnp.asarray(live, dtype=jnp.int32))
        jax.block_until_ready(inputs)
        # Scalar valid count matches the production kernel's bounded gather.
        fn = jax.jit(lambda a, b, c, valid: dsa_gather(a, b, c, num_valid_indices=valid))
        stage('lower')
        start = time.monotonic()
        lowered = fn.lower(*inputs)
        if evidence is not None:
            (evidence / 'lowered.mlir').write_text(lowered.as_text())
            (evidence / 'device.json').write_text(json.dumps(
                {'device': str(device), 'device_kind': device.device_kind}, indent=2) + '\n')
            print(f'EVIDENCE precompile_ir={evidence / "lowered.mlir"}', flush=True)
        stage('compile')
        compiled = lowered.compile()
        print(f'COMPILE_PASS elapsed_s={time.monotonic()-start:.3f}', flush=True)
        if args.compile_only:
            return
        stage('execute')
        out_nope, out_rope = jax.block_until_ready(compiled(*inputs))
        # Rows after the valid prefix are unspecified, as in production.
        assert out_nope.shape == (n, 4, 128), out_nope.shape
        assert out_rope.shape == (n, 128), out_rope.shape
        np.testing.assert_array_equal(np.asarray(out_nope).reshape(n, 512)[:live], nope[indices[:live]])
        np.testing.assert_array_equal(np.asarray(out_rope)[:live], rope[indices[:live]])
        print('RAW_JAX_GATHER_BYTE_EXACT_PASS (not a serving baseline)', flush=True)


if __name__ == '__main__':
    main()
