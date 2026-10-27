# Indexer pool benchmark

Compare the standalone fused K4/D128 kernel with the frozen JAX reference.
The GLM-5.3-Flash pooled indexer uses this kernel by default for K=4, D=128,
128-token pages, and BF16 raw data. Other shapes keep the generic JAX path.
The exact-byte gate below checks both updated records and untouched cache bytes.
The final kernel at `3f37bebe2f` passed all 414 cases on TPU, plus 31 selected
cases using the default conservative query bound, with the versions below.

## Kernel structure

`sglang_tpu/attention/pallas/kpool_compress.py` builds request descriptors and selects
the schedule. The `kpool` package separates numerical stages (`compute.py`),
DMA views (`io.py`), and compile-time schedules (`pipeline.py`).

`max_query_length` bounds each request's **new** tokens, including verification
rows. It is not the accumulated KV length. If omitted, `num_tokens` supplies a
conservative bound. Short queries use groups of up to 16 aligned raw tiles.
Long queries compact live descriptors and stream tiles through two raw buffers.
Queries with a bound of at most 64, or whose bounded tiles fit in one group,
use the grouped path. There is no Q=1 or batch=32 special case.

Both schedules preserve partial destination blocks. Requests that update
different records in a shared physical quad use ordered read/modify/write.
Pooling rounds to BF16 before rotation, and the rotated key is rounded to BF16
again before FP8 quantization, as upstream's Triton writer does. The reference
uses `lax.reduce_precision` for both roundings; an `astype` round trip can be
removed by XLA's excess-precision pass.

The serving custom op passes a static query bound from the attention metadata:
uniform DECODE uses `decode_query_size`; mixed prefill and verification use the
token bucket. This avoids a device-to-host metadata read and keeps the bound
valid when ragged requests share a compiled graph.

## Device timing

Use a TPU environment with JAX/JAXlib 0.11.2, libtpu 0.0.50 and NumPy 2.3.5
(the versions used for the recorded comparison). From the repository root:

```sh
JAX_PLATFORMS=tpu python benchmark/tpu/kpool/benchmark.py --output /tmp/kpool-profile
```

The capture covers 301 shapes at K=4, D=128, raw page=128 and BF16 raw data:

- Decode/verify: Q=1..32, B=1/2/4/8/16/32/64, T=B*Q.
- Prefill: Q=1/2/4/.../8192, the same B values, B*Q<=8192, T=max(64,B*Q).

Use `--page-size P` (a multiple of 16, default 128) to capture another raw
page size; the pooled cache becomes `[pages, P//4, 4, 256]`.
Use `--matrix-section decode` or `prefill` to select a mode, and
`--matrix-start` / `--matrix-stop` to select a slice. `--queries` and `--batches`
filter the fixed matrix. Each case uses 14,849 raw
pages, one device, a donated output, 10 warmups and 100 samples per variant.
Variant order alternates each round. Compilation and warmup precede capture.

Open each case's `xprof` directory in XProf. Measure the full `XLA Modules`
device span; module names identify the case and variant. Host `sample/` markers
identify iteration order only. Validate 100 device events per variant before
comparing medians. Pallas events are part of the module span, not extra time.
The script saves XPlane captures and source/runtime manifests; it does not
compute latency from host clocks or claim accuracy. Prefill Q1/Q2 closes no
pool and measures call/metadata overhead only.

Use `--baseline-revision <sha>` to include a frozen kernel revision. Additional
`--comparison-revision label=<sha>` options include named revisions in the same
alternating capture. Each revision must be available in the local Git checkout.
When two variants compile to the same executable, their device module names
can coincide. Match chronological device calls to the validated alternating
host labels; do not merge their samples by module name alone.

## Accuracy checkpoint

After device timing, run the independent exact-byte gate:

```sh
JAX_PLATFORMS=tpu python benchmark/tpu/kpool/validate_matrix.py --output /tmp/kpool-accuracy
```

It checks all 301 measured pairs plus 113 edge cases: ragged lengths, padding,
inactive requests, page boundaries, shared destination quads and zero/tiny/
nonfinite data. It compares the entire pooled cache, including untouched bytes,
and exits nonzero on a mismatch. `accuracy.json` records each case.

Use `--page-size P` to run the same matrix at another raw page size; starts
and page-boundary edges scale with P (page edge P-1/P, pooled page edge 4P-1/4P).
Use `--conservative-bound` to validate the API without `max_query_length`.
The same `--queries` and `--batches` filters can restrict this additional check.
