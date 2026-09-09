# Fused TP16 sequence-parallel QKVOG entry

This package provides a shape-specialized Pallas implementation that fuses a
sequence-parallel attention-entry boundary into one operation:

```text
local residual shard
  -> RMSNorm
  -> destination-tiled QKVOG matmul
  -> Q/K RMSNorm + query scale + interleaved RoPE
  -> asynchronous token-to-head ownership transpose
  -> local Q/K/V/OG heads for all tokens
```

There is no standalone activation all-gather on the supported route. Remote
copies overlap MXU work, and completed output tiles are written directly to
the physical rank that owns their attention heads.

## Contract

The tuned operation is deliberately narrow:

- TPU7x, DP1/TP16;
- global prefill token counts 1024--16384 using the static schedules below;
- FP32 or BF16 residual, with BF16 RMSNorm gain and QKVOG weight;
- hidden size 8192, head size 64;
- eight query/output-gate heads and one K/V head per TP rank;
- destination-local width 1152 (`Q512 + K64 + V64 + OG512`).

The persistent replicated weight has shape `[8192, 16 * 1152]`. Its blocks
are ordered by physical destination, not logical TP rank. This keeps the
hypercube schedule topology-native even when the serving process group uses a
non-identity logical rank order. At BF16 this buffer is 288 MiB per attention
layer on every rank, in addition to the original local weight retained for the
fallback path; deployment qualification must include the full-model HBM delta.

Each rank also passes an 18-element scalar-prefetch vector:

```text
[physical_id, logical_id, physical_to_logical[0], ..., physical_to_logical[15]]
```

Remote DMA uses physical IDs. Received source rows are placed using the
physical-to-logical map, so Q/K/V/OG retain canonical logical token order.
The supplied topology must preserve each ordered two-core pair; the 1K
schedule emits its final pair as one contiguous transfer.
The Torch adapter builds both the replicated weight and rank metadata once
after checkpoint loading; the forward path performs no setup collective.
The kernel computes RMSNorm from the residual's source precision, then stores
the normalized operand as BF16 at the same branch boundary as the unfused
model path.

Unsupported shapes, dtypes, and rank layouts must stay on a semantically
equivalent all-gather then QKVOG path selected by the caller.

## Static schedules

| Global tokens | Local rows | Schedule | Work per rank |
| ---: | ---: | --- | --- |
| 1024 | 64 | `gather-8-sources` | Gather eight source shards and project two physical destinations. |
| 2048 | 128 | `gather-2-sources` | Gather the sibling core and project eight destinations. |
| 4096 | 256 | `all-to-all` | Project local rows for all destinations and push completed tiles. |
| 8192 | 512 | `all-to-all` | Same direct output pipeline with taller local matmuls. |
| 16384 | 1024 | `all-to-all-grid` | Run the direct schedule in bounded 512-row blocks. |

All schedules overlap asynchronous remote copies with useful work. The 1K and
2K variants gather just enough rows to keep the MXU efficient; larger shapes
avoid activation movement and exchange only completed projection tiles.

## Source layout

- `_contract.py` is the single source of truth for the tuned model shape and
  destination-local QKVOG layout.
- `fused_rmsnorm_qkvog_mm_a2a_entrypoint.py` validates the public contract,
  selects the static schedule, and defines the pure-JAX oracle.
- `_schedule.py` contains host-side plans and routing tables.
- `_kernel_hybrid_gather_a2a.py` implements the 1K/2K pipelines.
- `_kernel_a2a.py` implements the 4K--16K direct-output pipeline.
- `_epilogue.py` centralizes QKVOG splitting, Q/K normalization, scaling,
  RoPE, and output dtype semantics.

Callers must assign distinct collective IDs to concurrently live layer
programs.
