# GLM-5.2 FP8 loading modes and validation

The three modes describe loading policies, not three quantization methods or
three execution backends. Modes (1) and (2) use the **same original block-FP8
checkpoint**. Mode (2) makes the existing block-to-channelwise conversion
explicit; it does not require a BF16/FP16 checkpoint or a new online quantizer.
With today's default `--tpu-fp8-runtime channelwise`, modes (1) and (2) already
have the same behavior. Mode (3) loads our separately converted offline
channelwise checkpoint directly.

## Configuration map

| Mode | Checkpoint | Serve arguments | HF `quantization_config` | Loading behavior |
|---|---|---|---|---|
| (1) Existing behavior | Original block-FP8 | `--quantization fp8`; preserve the existing runtime choice | Preserve checkpoint metadata, including `weight_block_size` | Unchanged. The current runtime default is `channelwise`; `--tpu-fp8-runtime blockwise` preserves block scales. |
| (2) Explicit block-to-channelwise | **Same original block-FP8 as (1)** | `--quantization fp8 --tpu-fp8-runtime channelwise` | **Identical to (1); no override** | Existing dense/MoE post-load conversion: block dequantization followed by per-output-channel FP8 quantization. |
| (3) Offline channelwise | `GLM5.2-fp8-channel-wise` | `--quantization fp8 --load-format safetensors` plus the offline layout override below | `tpu_checkpoint_layout=glm52_channelwise_fp8`, `weight_block_size=null` | Load FP8 bytes and multiplicative channel scales directly, then use the common runtime formatter/backend. |

A typical **original block-FP8** checkpoint has metadata of this form. Use its
actual metadata; do not replace it with this example or remove scale-format or
excluded-layer information:

```json
{
  "quant_method": "fp8",
  "activation_scheme": "dynamic",
  "fmt": "e4m3",
  "weight_block_size": [128, 128]
}
```

`activation_scheme=dynamic` controls activation quantization during inference.
Weights are fixed after loading in every mode. Dense activation quantization,
MoE activation quantization, GMM accumulation precision and backend selection
keep their existing meanings. The runtime flag governs both dense and MoE
block-FP8 conversion; this change does not introduce a MoE-only conversion flag.
The pre-existing BF16-to-FP8 fallback also remains unchanged and is not mode (2).

## Serve arguments

Use the same target/draft checkpoint within each mode. For the intended
8-physical-chip, two-host topology, the common argument fragment is:

```bash
# Run on both hosts with the appropriate rank and the same rendezvous address.
# Add workload-specific KV capacity, context limits and compile buckets in the
# launcher. This fragment is not an accepted long-context benchmark recipe.
GLM52_COMMON=(
  --device tpu --dtype bfloat16
  --tp-size 16 --attn-dp-size 16 --ep-size 16
  --nnodes 2 --node-rank "$GLM52_NODE_RANK"
  --dist-init-addr "$GLM52_DIST_INIT_ADDR"
  --attention-backend pallas_tpu_mla_sparse
  --moe-runner-backend pallas_tpu
  --disable-shared-experts-fusion
  --speculative-algorithm NEXTN
  --speculative-num-steps 5
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 6
  --enable-dp-lm-head
)
```

Mode (1), retaining the existing runtime default and checkpoint metadata:

```bash
python -m sglang.launch_server "${GLM52_COMMON[@]}" \
  --model-path "$GLM52_BLOCK_MODEL" --quantization fp8
```

Mode (2), using **the same** `$GLM52_BLOCK_MODEL` and metadata:

```bash
python -m sglang.launch_server "${GLM52_COMMON[@]}" \
  --model-path "$GLM52_BLOCK_MODEL" --quantization fp8 \
  --tpu-fp8-runtime channelwise
```

Mode (3), using the separately converted offline checkpoint:

```bash
# The JSON override replaces the whole quantization_config. Merge the explicit
# layout into the offline checkpoint's metadata to preserve exclusions.
GLM52_OFFLINE_OVERRIDE=$(python - "$GLM52_OFFLINE_MODEL" <<'PY'
import json
import pathlib
import sys
config = json.loads((pathlib.Path(sys.argv[1]) / "config.json").read_text())
quant = dict(config["quantization_config"])
assert quant.get("quant_method") == "fp8"
assert quant.get("activation_scheme") == "dynamic"
assert quant.get("fmt") == "e4m3"
assert quant.get("weight_block_size") is None
assert quant.get("scale_fmt") is None
quant["tpu_checkpoint_layout"] = "glm52_channelwise_fp8"
print(json.dumps({"quantization_config": quant}))
PY
)
python -m sglang.launch_server "${GLM52_COMMON[@]}" \
  --model-path "$GLM52_OFFLINE_MODEL" --quantization fp8 \
  --load-format safetensors \
  --json-model-override-args "$GLM52_OFFLINE_OVERRIDE"
```

The resulting mode (3) metadata includes these fields, plus any existing
`ignored_layers` or `modules_to_not_convert`:

```json
{
  "quant_method": "fp8",
  "activation_scheme": "dynamic",
  "fmt": "e4m3",
  "weight_block_size": null,
  "tpu_checkpoint_layout": "glm52_channelwise_fp8"
}
```

The native NEXTN draft inherits the target quantization choice and model
config override. No separate `--speculative-draft-model-path` is needed when
using the target checkpoint's MTP layer. DP attention with this worker requires
`--enable-dp-lm-head`; do not disable MTP to avoid that requirement. The offline
ABI requires direct target-device construction and supports the default
`auto`/`pt`/`safetensors` loader without CPU offload.

`--moe-runner-backend` is independent of checkpoint source. The examples pin
`pallas_tpu`, the component-tested GMM path. `auto`/`pallas_tpu_fused_rs` retain
their existing eligibility checks and may require additional distributed
validation for a particular topology.

## Runtime contract

| Tensor | Logical checkpoint representation | Common runtime representation |
|---|---|---|
| Dense weight | FP8 e4m3fn `[N,K]` | `[K,N]` |
| Dense channel scale | `[N]` or `[N,1]` | Per-output-channel multiplicative scale |
| MoE weight | FP8 e4m3fn `[E,N,K]` | `[E,K,N]` |
| MoE channel scale | `[E,N]` | GMM `[E,1,1,N]`; existing fused-RS preparation may choose its own layout |

The offline name `weight_scale_inv` denotes a **multiplicative** dequantization
coefficient: `W = W_fp8 * scale`. It is not a mathematical reciprocal. Row
parallel scales have no K sharding dimension. Dense scales remain FP32; MoE
scale storage keeps the existing BF16 default and FP32 opt-in.

Mode (2) computes channel scales from block-dequantized values over each rank's
local K extent. An offline checkpoint converted from original BF16, or before
K sharding, can have different FP8 bytes/scales. Runtime compatibility does not
imply numerical identity. The tests compare fixtures produced from the same
block-FP8 values with the same K extent and quantization rule.

## Validation status and the support matrix

| Coverage | Evidence | What it does not establish |
|---|---|---|
| Existing scalar/block config and metadata-free BF16 fallback | CPU config/loader regressions | Full model serving |
| Block-to-channelwise vs offline ABI | CPU real expert loaders and post-load comparison; registered TPU eager/compiled MoE test | GLM target/draft initialization or MTP serving |
| Offline NextN mapping | CPU native draft loader, expert packing, exclusions and BF16 bridge | Shared embedding/head initialization, draft/verify/commit, KV state or compilation coverage |
| Full target-only GLM serving | Historical experiments below | Current-head MTP serving or the requested long-context baselines |
| Current-head GLM + NEXTN 5/6 | **Pending E2E smoke** | Do not claim serving support from component tests |

At code revision `ae2a34f0c6a15a04eb8953f212d6700318b74566`, Falcon
`exp-nf2lihmlu8` succeeded: 12 CPU loader tests plus the revised TPU component
test passed (13 tests, 26 subtests). This covers exact block/offline FP8-byte and
scale parity, eager/compiled execution, equal outputs across the two sources,
and the FP32 sanity oracle. It does not run a full GLM model or MTP server.
The actual test process recorded the source revision, absolute interpreter,
imports, compiler settings and source-declared TPU stack versions. The first
attempt used an oversized random fixture and failed only the FP32 oracle;
the passing revision restores the original smoke's weight range with unchanged
tolerances, loaders and kernels.

`test_fp8_channelwise_loading.py` belongs to `stage-a-test-tpu-unit`. Such unit
registrations are intentionally excluded from model serving rows by
`scripts/ci/tpu/support_matrix.py`. Adding the `glm-5.2` vocabulary entry does
not add a GLM-5.2 serving-support row or establish three supported serving modes.
A future model E2E registration must back that claim and reference its recipe.

### Historical Falcon evidence and retest decision

The following metadata and logs were re-read through the Falcon CLI. These
experiments used `/models/GLM5.2-fp8-channel-wise`, TP16 / attention-DP16 / EP16,
`speculative_algorithm=None`, and `enable_dp_lm_head=False`:

| Experiment | Source revision | Verified scope |
|---|---|---|
| `exp-k0nnadpb6q` | `552e93ab661682f89bfda5890e8a413ad7b12821` | Text E2E pass, including explicitly routed DP requests |
| `exp-r188nvdf96` | `2160012cc6daae19b8e3a0a792483701466d34bc` | 12,288-input / 16-output requests, warm/sequential/routed checks; host-time observations |
| `exp-19jk2vguvj` | `475000ffe8efd48fdbee71042dc086a99b39cb94` | Compiled target-only long-request smoke |

The recorded historical stack also used `tpu-sync-torch==0.0.1.dev20261031081414`,
whereas this PR base declares `0.0.1.dev20261112081648`. The other recorded
core pins include torch 2.11.0, torch_tpu 0.2.1.dev20261028201649, JAX/JAXLIB
0.11.2 and libtpu 0.0.50. A fresh smoke must verify the actual test process
against the current source declarations.

This is useful evidence for the existing target-only channelwise execution
path. It is from the earlier experimental branch, not this PR's head, and
includes work outside the loader extraction. It does not prove that the
cleaned target+draft combination works. The earlier branch also observed a
cold/hot synthetic greedy difference whose cause was not established; do not
turn the timing records into a numerical or matched-performance conclusion.

**Retest decision:** a new, bounded MTP E2E smoke is still needed. Prioritize
mode (3), the chosen integration baseline: real target+draft loading, NEXTN 5/6
warmup, short deterministic requests across DP ranks, repeated decode, and
partial-batch/request completion. Check actual acceptance/output, KV state and
serving-time recompilation. Then run the same small smoke against mode (2)
when the original block-FP8 checkpoint is available. Mode (1)'s current default
and mode (2) are identical and do not require duplicate full-model runs.

A full 48K-prefix + 4K-extend / 90K-C64 performance matrix is not needed to
validate this loading change; it remains the next-stage baseline. Existing
component and historical target-only results lower the risk, but do not remove
the MTP integration test. No new full-model experiment is claimed here.
