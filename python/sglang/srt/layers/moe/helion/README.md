# Helion NVFP4 fused MoE reduce-scatter

This directory contains the opt-in Helion implementation of the TPU NVFP4
fused-MoE path. The default implementation remains the hand-written Pallas
kernel under `../pallas/fused_rs/`.

## Operation and sharding contract

The target Avocado Tier5 configuration uses TP16 and EP16. Each rank receives
the same token activations and router result, but stores only its 16 local
experts. For every routed token row owned by a local expert, the kernel performs:

1. packed E2M1 W1 dequantization and the gate/up projection;
2. SwiGLU;
3. packed E2M1 W2 dequantization and the output projection;
4. the input-scaled post-expert RMSNorm;
5. multiplication by the route's top-k weight; and
6. a remote write into the destination rank's token/top-k slot.

Each rank then sums the eight top-k slots for its token shard. The Torch wrapper
all-gathers those shards because the surrounding model currently expects a
replicated MoE output.

The kernel assumes:

- NVFP4 group size 256;
- `moe_ep_size == tp_size == world_size`;
- 16 local experts per rank for the production Tier5 checkpoint;
- SwiGLU and the Tier5 input-scaled post-expert RMSNorm contract; and
- replicated token activations and routing inputs with expert-sharded weights.

Model setup rejects the opt-in path when these requirements are not met.
Pallas remote-copy device IDs follow TPU physical order, which can differ from
Torch distributed rank order. The wrapper caches both mappings at model-load
time: physical-device order selects the local expert shard, while TP rank is
mapped to a physical destination for each output-token write. Each process
therefore receives its own TP-rank token shard, and the final Torch all-gather
already returns model-facing TP-rank order.

## Weight layout

FP4 values remain packed as two E2M1 values per `uint8`; complete expert
matrices are never expanded in HBM.

Both matrices are tiled once while loading the checkpoint. They remain packed;
the final dimension stores two E2M1 values per byte.

For the Tier5 dimensions (`hidden = intermediate = 4096`), W1 is arranged as:

```text
[local experts, 2 intermediate tiles, 16 hidden groups,
 256 input values, 4096 gate/up output values]
```

The 16 W1 group matmuls therefore accumulate a 2,048-value gate tile and its
matching 2,048-value up tile. W2 uses four 1,024-column output tiles:

```text
[local experts, 2 intermediate tiles, 4 output tiles,
 8 intermediate groups, 256 input values, 1024 output values]
```

The scale tensors use the same group and tile order. These layouts let every
device-loop load fetch an aligned panel and avoid expanding FP4 weights in HBM.

## Device schedule

Routing metadata is built on the TPU; no route IDs are read by the host. Routes
are sorted by expert and processed in blocks of 8, 16, or 32 rows. The block
size grows only when the average number of rows per local expert is large enough
to use the wider MXU batch efficiently. Every rank executes the same bounded
route-block loop, while masks disable padding and empty expert blocks.

For one active route block:

| Phase | Work and overlap |
| --- | --- |
| Route gather | Gather the selected token rows into VMEM and quantize each 256-value hidden group to E4M3. |
| W1 projection | Double-buffer W1 and W1-scale panels. While the MXU consumes the current panel, DMA starts loading the next panel. Accumulate the 16 hidden groups into FP32 gate/up tiles of 2,048 values. |
| Activation | Apply SwiGLU, round the activation through BF16 as the production path does, split it into eight 256-value groups, and quantize those groups to E4M3. |
| W2 projection | Double-buffer W2 and W2-scale panels. Each MXU operation computes four 1,024-column output tiles in parallel, and the kernel retains one FP32 accumulator per tile. |
| Epilogue | Join the four output tiles, apply input-scaled post-expert RMSNorm, and multiply by the route's top-k weight. |
| Reduce-scatter | Send each valid row directly to the TPU that owns its output token. Incoming top-k slots are summed locally before the kernel returns. |

The W1 and W2 panel streams use separate two-stage buffers. This overlaps HBM
traffic with matrix multiplication without keeping full expert matrices in
VMEM. Splitting W2 into 1,024-column tiles also keeps the FP32 accumulator small
enough for route blocks up to 32 rows.

The Pallas lowering also starts the first weight and scale panels for the next
route block after their current buffers are no longer needed. The four streams
share one calculation of the next block's expert address. This hides part of
the first-panel latency without changing the source-level schedule.

Each route block drains its valid sends before reusing the output staging
buffer. The final route block drains all incoming route writes before the local
top-k sum. The receive count depends only on the padded token count and top-k,
so correctness does not assume balanced expert routing.

## Selection

Install a Helion build with TPU Pallas support in the SGLang TPU environment:

```bash
pip install helion
```

Helion is only imported when this path is selected, so it is not required for
the default Pallas path.

Set:

```text
SGLANG_TPU_HELION_FUSED_MOE=1
```

The flag is false by default and only selects Helion after the existing
`pallas_tpu_fused_rs` backend contract has been validated. With the flag unset,
weight preparation and execution remain on the existing Pallas path.

## Standalone validation

The manual TPU benchmark checks finite output, exact repeated-call determinism,
and numerical agreement with the production Pallas kernel before timing either
implementation:

```bash
torchrun <TP16 launcher arguments> \
  test/manual/tpu/benchmark_nvfp4_helion_fused_rs.py \
  --tokens 1 16 64 256 512 1024 --warmup 30 --iterations 300 \
  --paired-timing
```

For a slower independent check against weights dequantized from the same NVFP4
codes, run:

```bash
torchrun <TP16 launcher arguments> \
  test/manual/tpu/test_nvfp4_helion_fused_rs_execute.py \
  --tokens 1 16 64 256 512 1024 --compare-pallas
```

Both scripts use the production hidden, intermediate, local-expert, top-k, and
NVFP4 group-size defaults. They require one Torch process per TPU device.
