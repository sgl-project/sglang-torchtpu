"""Same-runtime TopK diagnosis; never used by the serving dispatch."""

import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("bench", "profile"), required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "glm52_bench", ROOT / "test/manual/kernels/benchmark_glm52_radix_topk.py"
)
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)
import jax
import jax.numpy as jnp
import numpy as np
from sglang_tpu.attention.pallas import radix_topk as port
from sglang_tpu.dsv4.sparsecore_topk import sparsecore_topk
from sglang_tpu.dsv4.streamindex_topk.streamindex_topk import streamindex_topk

source_spec = importlib.util.spec_from_file_location(
    "jax_original_v2", Path(__file__).with_name("source_radix_v2.py")
)
original = importlib.util.module_from_spec(source_spec)
source_spec.loader.exec_module(original)
runtime = dict(
    executable=sys.executable,
    versions={n: metadata.version(n) for n in ("jax", "jaxlib", "libtpu", "torch", "torch_tpu")},
    imports={n: str(__import__(n).__file__) for n in ("jax", "jaxlib")},
    source=os.environ["SOURCE_COMMIT"],
    source_radix_sha256=hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
    visible_devices=[str(x) for x in jax.devices()],
    actual_device=str(jax.devices()[0]),
    allocation_physical_chips=4,
    participating_jax_devices=1,
    flags={k:v for k,v in os.environ.items() if k in ("XLA_FLAGS", "LIBTPU_INIT_ARGS", "JAX_PLATFORMS", "TPU_LIBRARY_PATH")},
    mapped_libtpu=[x.split()[-1] for x in Path('/proc/self/maps').read_text().splitlines() if 'libtpu.so' in x],
)
assert runtime['versions']['jax'] == runtime['versions']['jaxlib'] == '0.11.2'
assert runtime['versions']['libtpu'] == '0.0.50'
(args.out / 'runtime.json').write_text(json.dumps(runtime, indent=2))
records = []

def emit(record):
    records.append(record)
    with (args.out / 'metrics.jsonl').open('a') as f:
        f.write(json.dumps(record)+'\n')
    print(json.dumps(record), flush=True)

def execute_set(functions, inputs, case):
    compiled = {}
    for name, fn in functions.items():
        start=time.perf_counter(); lowered=jax.jit(fn).lower(*inputs)
        (args.out / f'{case}-{name}.stablehlo.txt').write_text(lowered.as_text())
        compiled[name]=lowered.compile()
        output=compiled[name](*inputs);jax.block_until_ready(output)
        compile_s=time.perf_counter()-start
        for _ in range(5): jax.block_until_ready(compiled[name](*inputs))
        emit(dict(kind='compile',case=case,variant=name,compile_first_s=compile_s))
    if args.mode == 'bench':
        samples={name:[] for name in compiled}
        names=list(compiled)
        for iteration in range(30):
            order=names if iteration%2==0 else list(reversed(names))
            for name in order:
                start=time.perf_counter_ns();jax.block_until_ready(compiled[name](*inputs))
                samples[name].append((time.perf_counter_ns()-start)/1000)
        for name,values in samples.items():
            emit(dict(kind='bench',case=case,variant=name,median_us=statistics.median(values),samples_us=values,
                      metric='host synchronized compiled call; includes dispatch',actual_devices=1))
    else:
        options=jax.profiler.ProfileOptions();options.host_tracer_level=1;options.python_tracer_level=0
        options.advanced_configuration={'tpu_num_chips_to_profile_per_task':1,
            'tpu_num_sparse_cores_to_trace':2,'tpu_num_sparse_core_tiles_to_trace':1}
        for name,fn in compiled.items():
            dest=args.out/'xprof'/case/name
            with jax.profiler.trace(str(dest),profiler_options=options):
                for i in range(5):
                    with jax.profiler.StepTraceAnnotation(case+'_'+name,step_num=i):
                        jax.block_until_ready(fn(*inputs))
            planes=list(dest.glob('plugins/profile/**/*.xplane.pb'))
            assert planes,(case,name,'missing XPlane')
            emit(dict(kind='capture',case=case,variant=name,xplanes=[str(p) for p in planes]))
    return compiled

for rows in (6,24):
    step=helper.build_benchmark_step(lens=[(6,90000)]*(rows//6),token_bucket=rows,is_decode=True)
    values=tuple(jax.device_put(jnp.asarray(getattr(step,k)),jax.devices()[0]) for k in
                 ('query','key','head_weights','cache','kv_lens','page_indices','cu_q_lens','distribution'))
    config=(16384,8192,False)
    def indexer(selector):
        return lambda *a:helper.fixture['dsa_indexer_step'](*a,weight_scale=helper.fixture['_WEIGHT_SCALE'],
            topk=2048,decode_query_size=6,segments='decode',selector=selector,radix_config=config if selector=='radix_v2' else None)
    # Diagnostic-only callback obtains the exact scorer values before timing.
    captured=[];real=port.select_radix_topk
    def capture(scores,k,lengths,**kwargs):
        jax.debug.callback(lambda s,l:captured.append((np.array(s),np.array(l))),scores,lengths,ordered=True)
        return real(scores,k,lengths,**kwargs)
    port.select_radix_topk=capture;streamindex_topk.clear_cache()
    try:
        jax.block_until_ready(jax.jit(indexer('radix_v2'))(*values))
        jax.effects_barrier()
    finally:
        port.select_radix_topk=real;streamindex_topk.clear_cache();jax.clear_caches()
    assert len(captured)==1,len(captured)
    score_bits,lengths=captured[0]
    scores=score_bits.view(np.float32) if score_bits.dtype==np.int32 else score_bits
    assert scores.shape==(rows,92160),scores.shape
    score_args=(jax.device_put(scores,jax.devices()[0]),jax.device_put(lengths,jax.devices()[0]))
    functions={
        'incumbent':lambda s,l:sparsecore_topk(s,2048,l),
        'port':lambda s,l:real(s,2048,l,config=config),
        'jax_source':lambda s,l:original.radix_topk_v2(s,k=2048,valid_lengths=l,block_size=16384,
            reconstruct_threshold=8192,reverse_scan=False,indices_only=True),
    }
    for name,fn in functions.items():
        ids=np.asarray(jax.jit(fn)(*score_args))
        for row in range(rows):
            valid=int(lengths[row]);selected=ids[row];assert np.all((selected>=0)&(selected<valid))
            assert np.unique(selected).size==2048
            cutoff=np.partition(scores[row,:valid],-2048)[-2048]
            assert np.min(scores[row,selected])>=cutoff,(name,row)
    emit(dict(kind='correctness',case=f'selector-r{rows}',exact_set=True,input='captured paged FP8 scorer',
              width=92160,valid_lengths=lengths.tolist(),page_size=64,k=2048))
    execute_set(functions,score_args,f'selector-r{rows}')
    complete={name:indexer(selector) for name,selector in (('incumbent','sparsecore'),('port','radix_v2'))}
    for name,fn in complete.items():
        ids,cache=jax.jit(fn)(*values);np.testing.assert_array_equal(np.asarray(cache),step.expected_cache)
        picks=np.asarray(ids)
        for row in range(rows):
            valid=int(lengths[row]);cutoff=np.partition(scores[row,:valid],-2048)[-2048]
            assert np.min(scores[row,picks[row]])>=cutoff
    execute_set(complete,values,f'indexer-r{rows}')
print('GLM52_TOPK_PROFILE_PASS',args.mode,flush=True)
