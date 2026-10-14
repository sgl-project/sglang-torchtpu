# GPT-OSS-120B TP8 decode megakernel

Correctness is validated in Pallas TPU interpret mode on 8 virtual CPU devices and on TPU v7x
(`tpu7x:2x2x1`). Compilation is checked by ahead-of-time compilation for `tpu7x:2x2x1` on a CPU
host and end-to-end serving benchmarks on TPU v7x.

## What one call does

`make_step(c, mesh)` returns a jittable decode step. It runs under `shard_map` over a 1-D
`"tp"` mesh of R=8 ranks:

1. `gptoss_decode_mega` is **one `pallas_call` per step** that runs all L layers. The layers
   go through `fori` over (sliding, full) layer pairs; layer `l` is sliding iff `l % 2 == 0`
   (GPT-OSS `layer_types`). Per layer:
   - **A.** RMSNorm, then QKV (+bias), then YaRN neox RoPE. q goes to per-sequence
     `[8, 128]` tiles. The new K/V are rounded to the KV dtype.
   - **B.** Attention over the paged KV pool, read **in place**, run as one flat stream of
     (sequence, page-block) work items. The scalar core first lists the items. Page DMAs
     then run `kv_bufs - 1` items ahead of compute, across sequence boundaries. All pages
     of an item share one semaphore, and only pages holding attended tokens are fetched.
     A full layer uses `npb`=8 pages (512 tokens) per item. A sliding layer uses one item
     per sequence with ≤3 window pages. The current token's K/V come from VMEM. Includes
     sinks, the 128-token window (sglang `window_left=127`), and online softmax carried
     across a sequence's items.
   - **C.** o_proj, then the **in-kernel all-reduce**, then `+ bo + residual` (rounded to bf16).
   - **D.** RMSNorm, then router (+bias), then top-4 over 128 with softmax over the top-4
     logits. The active-expert mask goes VMEM → SMEM by DMA, and the scalar core compacts
     it into a list.
   - **E.** For each active local expert chunk: gate|up (+b13), swigluoai (clamp 7,
     α=1.702), down (+b2, added once across ranks), weighted by the router weight. Then the
     **in-kernel all-reduce**, then `+ residual`.
2. `gptoss_decode_kv_scatter` is a tiny non-communicating `pallas_call` that writes each
   layer's new K/V token into the pools in place. It aliases the pools and does one
   HBM→HBM DMA per (layer, sequence).

Embedding, final norm, lm_head and sampling stay outside the kernel. Input is the embedded
token `[B, H]`; output is the last layer's residual stream `[B, H]`.

## Design choices (and why)

| Topic | Choice in v0 | Notes |
|---|---|---|
| Attention sharding | 8 q heads + 1 KV head per rank (64q/8kv) | Same as stock attn TP8 |
| MoE sharding | Default `moe_mode="tp"`: every rank holds an I/8 slice (360, padded to 384) of all 128 experts. `"ep"` (16 whole experts/rank, I padded to 3072) is also implemented and tested | Stock uses EP8. TP-split is load-balanced at small B: at B=1 every rank streams 4 × 1/8 experts. Under EP, one rank may own 2–4 of the 4 routed experts. At B=64 the bytes are equal. |
| Batch | `B` static per compile (buckets 1,2,4,8,16,32,64), padded to `BR = rup(B,16)` activation rows | Rows ≥ B are masked out of routing and attention |
| Hidden padding | H=2880 → Hp=2944 (`rup(H,128)`) with zero weights/biases; RMSNorm divides by true H | 2944 = 23·128 is enough for lane tiles. 3072 is not needed. |
| All-reduce | One-shot push over remote DMA. Each rank DMAs its partial into its own slot of all 7 peers' receive buffers, waits for the 7 arrivals, then sums the 8 slots in rank order in f32. There are two slots (attention / MoE) and no per-AR barrier (one barrier at kernel start). Sends are waited lazily, just before the slot is reused one layer later. `ar_start` / `finish` are split. Between them the megakernel lists layer l+1's attention items and starts their first KV page DMAs, which overlaps the MoE AR. For B ≤ 4 (`ar_fold`) only the live rows travel, as `[B·23, 128]` f32 (B=1: 12 KB per peer instead of 94 KB). For B ≥ 16 (`ar_mode="auto"`) it is reduce-scatter + all-gather instead: rank r owns a 384-lane column chunk. Round 1 sends AR-dtype chunks; round 2 all-gathers the f32 chunk sums, so results stay bit-identical to the one-shot sum. Per-rank bytes are BR×(5.4+10.8) KB vs BR×41 KB one-shot, at the cost of 2 rounds. | Ported from vllm-torchtpu `kernels/megakernel/comm.py` (non-RSAG path). Rank-order f32 sum keeps the residual stream, and so routing, identical on all ranks. RS→local→AG would need 2 latency rounds per AR (RMSNorm and QKV need the full hidden), so one-shot is latency-optimal for these payloads. |
| XLA fallback | `make_step(..., xla_ar=True)` builds per-layer `attn` / `moe` kernels from **the same phase bodies** with `lax.psum` in between (2L+1 calls) | Used to check the in-kernel AR: the outputs are bit-identical in interpret mode |
| KV layout | Exactly sglang's fused TPU pool per layer: `[pages, 64, 1, packing, 128]` (K slot 0, V slot 1, fp8 slots 2–3 padding, head_dim zero-padded to 128). Read as uint32 words + shifts, like RPA v3 `load_bkv`. bf16 and fp8_e4m3 are both supported. | **Gap:** layers are stacked `[L/2, …]` per pool type (SWA pool / full pool, each with its own page table, as in SWAKVPool). Stock keeps one tensor per layer. |
| New K/V write | Separate non-collective scatter kernel (`kv_write="scatter"`). `kv_write="kernel"` (megakernel writes into aliased pools) is implemented and tested, but AOT shows XLA adding pool-sized copies (4.6 GB temp), so **do not use it** | XLA TPU does not place outputs of a `collective_id` pallas_call into live-out buffers. Aliasing the pool inside the megakernel would copy the whole pool every step (see vllm `_frozen/qwen35/pool_io.py`). AOT HLO shows no pool-sized copy. |
| Weight streaming | Attention + router weights (`wqkv, wo, wr, vec, sinks`) use a 2-slot buffer. Layer l+1's set is prefetched at the start of layer l. Expert chunks `(w13 [Hp, 2·tne], w2 [tne, Hp], b13, b2)` stream through a `w_slots`-deep ring (default 3, so 2 chunks in flight). | Expert chunks depend on routing, so the expert ring cannot start before layer l's router. Across the layer boundary the stream continues only with the next layer's attention weights. |
| Weight dtype | Attention/router bf16. Experts `w_dtype="fp8"` (recommended), bf16, or `"mxfp4"`. **fp8** is MXFP4 converted offline to e4m3 times a power-of-two per output column. The conversion is exact when a column's 32-blocks span ≤13 octaves, which holds in practice; the packer counts any inexact elements. The chunk then runs bf16 × e4m3 natively on the v7x MXU: 3.39 MB per expert chunk, and Mosaic emits ~0.6K vector ops vs ~6.4K for bf16 weights and ~11K for in-VMEM MXFP4 dequant (static LLO op counts, `scratch/probe_mm.py`). **mxfp4**: e2m1 codes, 8 per uint32 along K (`pltpu.bitcast` packing), plus a bf16 power-of-two scale per 32 K rows × column (rows padded to 16). | MXFP4 chunk = 1.94 MB vs 6.78 MB bf16 (3.5× fewer bytes). It is dequantized in VMEM (bitcast → bf16 → × scale, in static 128-row chunks) into a bf16 buffer, then hits the MXU, like gmm_v2's `dequantize_in_vmem`. In TP mode the I split is block-aligned: 90 blocks → 12,12,11,…, padded to 384. |

## VMEM (analytic, tile-padded; real dims, fp8 KV, TP MoE, tne=384, npb=2)

| B | scratch + resident operands | biggest items |
|---|---|---|
| 1, 2, 4 | 38.2 MiB (MXFP4 experts: 30.8, fp8 experts: 28.8; folded AR buffers) |
| 8, 16 | 39.4 MiB (MXFP4: 32.0) | w13 ring 12.9, wqkv×2 8.6, w2 ring 6.5, wo×2 5.8, wr×2 1.4, AR recv 1.4 |
| 32 | 42.3 MiB (MXFP4: 34.9) | AR recv 2.9 |
| 64 | 48.0 MiB (MXFP4: 40.6, fp8: 38.7) | AR recv 5.8 |

`vmem_limit_bytes` = 60 MiB. AOT compiles succeeded under that limit for all buckets.
Mosaic does not report its own peak, so the table is our estimate (`megakernel.vmem_bytes`).
Per-rank HBM arguments are ≈34 GiB (bf16 weights, 36 layers) + pools.

## Validation status

| What | How | Result |
|---|---|---|
| Megakernel vs pure-JAX TP reference (`reference.step`) | interpret, 8 CPU devices, tiny dims (H=240→256, real 64q/8kv/D=64 heads, E=16, top-4, I=384, page 64, 4 pages/seq, positions incl. 0, 63, 130, 255) | rel err 3–5e-3 for bf16/fp8 KV, TP/EP MoE, L=2/4, B=1/4/16 |
| In-kernel all-reduce vs `psum` | megakernel vs `xla_ar=True` fallback, same inputs | bit-identical outputs and pools |
| MXFP4 experts | Weights snapped to MXFP4 in the reference, packed codes + scales in the kernel, TP (block-aligned split) and EP | rel err 3.9–4.0e-3. Megakernel vs fallback bit-identical. |
| fp8 experts (MXFP4 → e4m3 × column power of two) | TP (with fallback) and EP at B=16 | rel err 3.9–4.3e-3; outputs identical to MXFP4 (the conversion is exact). Near-tie routing rows (top-k margin < 3e-3, at most B/4) are excluded from the comparison: summation-order noise can legitimately flip them. |
| In-kernel KV write | `kv_write="kernel"`, aliased pools | correct in interpret mode, but AOT shows XLA copying the pools, so not used |
| In-place KV | pools pre-filled with finite garbage everywhere the step must not read. After the step, exactly the new-token slots changed and they hold the reference K/V | pass |
| Compile for TPU | AOT `tpu7x:2x2x1` (jax 0.10.2 + libtpu 0.0.47 on a CPU-only Linux host), real dims, L=36, B ∈ {1,2,4,8,16,32,64}: fp8 TP, bf16 TP, fp8 EP megakernels and the fp8 TP XLA-AR fallback | all compile (bf16/MXFP4 experts, fp8/bf16 KV, TP/EP, max_pages 65/128/513 i.e. up to 32K context, XLA-AR fallback; megakernel 1.4–6 s). HLO copies are only small operand moves (no pool copy). Raw output: `benchmark/kernels/gptoss_decode_megakernel/aot_tpu7x_2x2x1_results.jsonl` |
| Not validated | Anything on hardware: ICI remote-DMA timing/races, barrier semaphore behaviour, the uint32 *ref* bitcast of KV pages (interpret mode cannot bitcast refs, so it uses an equivalent value-level `pltpu.bitcast`; the ref path is AOT-only), real VMEM peak, performance | — |

## Known gaps / next steps

- **MXFP4 bytes with fp8 compute.** Probed, not adopted. A hardware e2m1→e4m3 convert
  plus a per-32-block exponent add with borrow-free SWAR ops on the e4m3 words costs
  ~8.9K vector ops per chunk (static LLO). That compares with ~0.6K for stored e4m3 and
  ~11K for the MXFP4 → bf16 dequant. TPU shows fp8 experts already at the HBM byte floor at
  B32, so the 43% byte saving would be eaten by VPU work. Without per-block scales, the
  convert alone is ~2.8K ops, but requantizing to per-column fp4 is lossy.
- **MXFP4 experts.** Implemented (`w_dtype="mxfp4"`); measured slower than fp8 on TPU at
  every B (j-af9a53e7). The VPU
  dequant of every streamed chunk (~3.4M weights per expert chunk) might cost as much as the
  DMA bytes it saves. The fallback is the stock-style "per-32-K-group product then scale",
  or native fp4 MXU if Mosaic exposes it. Integration must convert the stock checkpoint
  layout (uint8 `[E, K/2, N]` + f32 scales `[E, K/32, 1, N]`) to this layout offline.
- **AR cost.** The one-shot AR moves 7× payload per rank (fine at decode sizes) and is not
  topology-aware (2 TCs per chip, 2x2 chips). Expected cost per AR (estimate, not
  measured): ~3–6 µs at B ≤ 4 (folded, 12–48 KB per peer), ~5–10 µs at B=8 (94 KB per
  peer), ~10–15 µs at B=64 (377 KB per peer). Stock XLA all-reduce is ~41 µs. Use
  `--ablate ar` on TPU to measure it.
- **Expert prefetch across layers.** Only the attention weights cross the layer boundary.
  Option: start the next layer's first chunk speculatively (e.g. from the previous step's
  routing).
- **Attention per KV block.** Static LLO counts for a 1024-token block are ~2K ops (~1.5K
  now that fp8 K/V feed the MXU directly), i.e. ~1.5–2 ops per token. The 8-row q pushes
  both K and V through the MXU as weights. At long context and large B this is
  compute-bound, not HBM-bound.
- **Attention.** v0 (one DMA wait per 2-page block, depth-1 prefetch per sequence) scaled
  badly with context on TPU: B32 went from 13.7 to 26.0 ms between ctx 1k and 4k. v1 adds
  the flat item stream, 512-token items and cross-sequence prefetch. It is not yet measured
  on TPU. Matmuls still have only 8 q rows.
- **Page tables in SMEM.** The XLA glue passes the full layers' table `[B × max_pages]`,
  only the window pages `[B × 3]` for sliding layers, and each new token's page `[2B]`.
  B=64 at 32K context (max_pages=513, 128 KiB SMEM) compiles AOT. 128K context would need
  page-table streaming.
- **Integration (Phase 2).** Stacked pools vs sglang's per-layer tensors. The torch_tpu
  bridge also needs the per-layer fused pool viewed as `[pages, 64, 1, packing, 128]`
  without copies. Other gaps: k/v scales (only scale 1 here), final norm / lm_head inside
  the kernel, and a device control vector.
- **Fallback.** The fallback is the same kernels split per phase, not the stock sglang path.
  The benchmark's `xla` mode is a pure-XLA proxy.

## Files

- `config.py`: `DecodeConfig` (geometry, sharding mode, KV dtype, tiles). `tiny()` / `full()`.
- `reference.py`: global model-layout weights and the pure-JAX TP reference step.
- `megakernel.py`: `pack_weights`, `packed_specs`, `pool_specs`, `build_pools`,
  `build` (mega / attn / moe modes), `build_kv_scatter`, `make_step`, `vmem_bytes`.
- `test/registered/unit/hardware_backend/tpu/test_gptoss_decode_megakernel.py`: CPU interpret tests.
- `benchmark/kernels/gptoss_decode_megakernel/aot_compile.py`: tpu7x AOT check (needs libtpu, no TPU).
- `benchmark/kernels/gptoss_decode_megakernel/bench_decode_step.py`: TPU microbenchmark (`mk` / `fallback` / `xla`).
  `--cpu-smoke` runs the script at tiny dims in interpret mode.

## Fixed per-step cost (where B1 time goes)

TPU ablations at B1 (MXFP4) put a 0.65 ms floor under `--ablate attn moe`. Static analysis
of the compiled step shows:

- **XLA glue:** ~10 tiny fusions (rope tables, window page table, new-token pages, pad,
  slice). Each is µs-scale.
- **Weight prefetch:** the per-layer attention/router weights are 8.3 MB, ~2.4 µs at HBM
  speed, so ~85 µs/step. With both blocks ablated nothing hides it; normally it overlaps.
- **Per call:** the kernel-start barrier, VMEM zero-init of the KV buffers, and the KV
  scatter call (36·B row DMAs). The scatter is cheap at B1 and grows to ~2.3K DMAs at B64.
- **Host dispatch:** the bench's original timing blocked on every step, so it included host
  dispatch latency for a multi-device call with ~20 operands. The bench now also reports
  `ms_pipelined` (back-to-back dispatch, one block) and `ms_device` (K steps inside one jit
  via `fori_loop`). The gap between `ms_median` and `ms_device` is host/launch overhead,
  which an sglang integration removes by keeping the step on device.

To split per-call from per-layer cost, run `--ablate attn moe` at `--layers 2` and
`--layers 36`.

## Timing ablations

`DecodeConfig.ablate` / bench `--ablate` (outputs become wrong):

- `ar`: pack + local reduce only. No remote DMAs, no barrier, no `collective_id`.
- `attn`: skip the whole attention block, including its AR.
- `attn_kv`: skip only the paged attention core (and the KV prefetch).
- `moe`: skip router, experts and the MoE AR.
- `experts`: skip only the expert stream and matmuls.

## How to run

```bash
# CPU interpret tests (8 virtual devices; ~35 s)
XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \
  python -m unittest test.registered.unit.hardware_backend.tpu.test_gptoss_decode_megakernel

# AOT compile for tpu7x:2x2x1 on a CPU-only Linux host with jax[tpu]==0.10.2, libtpu 0.0.47
JAX_PLATFORMS=cpu python benchmark/kernels/gptoss_decode_megakernel/aot_compile.py \
  --batch 1 2 4 8 16 32 64 --kv fp8

# TPU microbenchmark (one tpu7x 2x2x1 host, single process, needs only jax[tpu])
python benchmark/kernels/gptoss_decode_megakernel/bench_decode_step.py \
  --batch 1 8 32 64 --modes mk fallback xla --ctx 1024 --kv fp8
```
