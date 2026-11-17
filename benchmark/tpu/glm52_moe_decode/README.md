# GLM5.2 EP32 decode MoE comparison

Base production code is PR #2238 at `9c7031a475dbd727c25c0966b74ac1633b6cf17b`.
The runtime bootstrap is from the user-provided `exp-lhz0s984yj` source
`310b2eca55dcd043f83a785ccb730752d3909752`, with its pinned TorchTPU/JAX/libtpu
versions and official block-FP8 checkpoint metadata. Runtime weights are
channel-wise FP8.

Run `bash benchmark/tpu/glm52_moe_decode/run.sh` on four Falcon replicas with
eight logical TPU devices per replica and physical topology `2x2x4`.
Set `SOURCE_COMMIT` to the submitted source SHA and `GLM52_MODEL_PATH` to the
mounted `/models/GLM-5.2-FP8` checkpoint. Reuse the reference experiment's wheel
and prepared wheelhouse mounts.

The matrix uses H6144, I2048, E256, TopK8, EP32 and a shared expert. Local
token counts 4, 6, 12, 18 and 24 cover ordinary C128 decode and MTP verification
buckets with one through four requests per DP rank and six draft tokens.
The C128 primary case with 24 local rows runs first. The standalone worker
allows 32 Dynamo specializations for the intentional capacity/shape matrix.

The five arms compare the current GMM path, the ported JAX GLM fused V2 path,
and the repository's existing fused V2 with capacities 32, 64 and 128. Every
arm includes shared-expert computation and the collectives needed to return
DP-local rows from the replicated FFN input and preselected routes.

Inputs use synthetic uniform routing, full-sized expert-specific constant FP8
weights and nonuniform channel scales. Each rank checks an independent FP32
analytic reference with relative L2 below 0.15 and finite outputs. Actual
checkpoint loading and runtime block-to-channel conversion are exercised by
the subsequent E2E run.

Activations follow N(0.02, 0.2²). The positive mean keeps the constant-slab
projection onto the row sum well conditioned. The first attempt with zero-mean
inputs stopped at 15.28% relative L2 on one four-row JAX output block, with
14.07% for GMM on the same block. That failed attempt remains separate. The
error threshold is 0.15 for both attempts.

Compilation, three warmups and barriers are outside the timing window. Twenty
interleaved calls are synchronized at completion. Aggregate the maximum
across all 32 ranks for each aligned sample before taking the median. Keep
all per-rank samples, runtime identities and correctness records in artifacts.
