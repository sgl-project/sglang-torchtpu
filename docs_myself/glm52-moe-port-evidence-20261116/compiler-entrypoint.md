# GLM MoE CPU-export compiler diagnostic

- Base: #2198 `ffd83db8f6168e0752e0769f8947cffb308e724d`.
- JAX source: `c948bdda95208e68a474c8088937347d08fe8937`, kernel SHA256
  `79722fd38f31db054c83e20903cd9a42557f79b97c5d5a95c6aeb8cf5302664c`.
- First real bridge experiment: `exp-fqynkusvij`, artifact `art-9ztiut4ca7`,
  measurement source `8a9c4a8d3d6bd880a37b626867cc659ac3495f33`.
- Runtime declared and bootstrapped: absolute interpreter
  `/tmp/glm52-baseline-runtime/venv/bin/python`, Torch 2.11.0,
  torch_tpu 0.2.1.dev20261028201649, JAX/JAXLIB 0.11.2, libtpu 0.0.50.
  The original test prints the actual worker versions/imports before invocation;
  current `falcon exp logs` tail contains only final errors, so those early
  worker records require the complete artifact `device-tests.log`. Do not
  mislabel a bootstrap-only runtime probe as the actual worker evidence.
- First failure occurs before numerical execution, at synchronization of the
  first small H512/I512/E16/TopK4/local6 bridge call. No case passed.
- Signature: `hlo->has_sharding(): Side-effect HLO must have sharding`,
  `tpu_custom_call`, `custom_call_has_side_effect=true`.
- Kernel AST was identical to the pinned source at this measurement. Its
  compute, communication and synchronization were not patched.

The source's `has_side_effects=True` reaches the custom call inside a nested
jit/shard_map. The local TorchTPU source checkout inspected for mechanism is
`dd6e5886faa42156c9b0f65eb5241028e0d2b642`; it is not proof of the wheel's
exact source. Its `torch_tpu/ops/custom_kernels.cc` partitions exported global
StableHLO to local form through Shardy plus SPMD, then embeds it into TorchTPU's
larger graph. `torch_tpu/common/compilation.cc` enables SPMD again for communication
kernels, expecting graphs without sharding to be a no-op. Native
`tests/pallas/pallas_communication_test.py` uses a pure input/output ring RDMA
call without `has_side_effects=True`; SGLang's existing vLLM-derived fused EP
and fused-RS kernels do likewise. This narrows the issue to that entrypoint's
side-effect/sharding contract; it does not yet prove which compiler pass drops
the annotation.

A proposed pure-call adaptation in `0318efe9db` was reverted in `354edd076b`
pending an explicit same-entrypoint source comparison. The diagnostic at
`test/manual/kernels/glm_moe_v2_export_diagnostic.py` loads exact pinned source
bytes into an independent module, repeats the port under the identical wrapper,
and exports each StableHLO to inspect manual/sharding/effect attributes. A
third isolated candidate changes only the effect declaration in a temporary
source module and runs the full real bridge/reference suite. It neither edits
the committed kernel nor removes DMA, wait, barrier, scratch or drain code.

The purity argument to validate is narrow: all externally observed results are
returned rows; remote payload buffers and semaphores are invocation-internal
scratch; input weights/routes/tokens are read-only. Returned rows retain the
true producer/consumer dependencies. This does not authorize removing waits or
assuming correctness from compilation alone. The candidate requires the same
numerical, invalid-id/zero-route, tail, shared-on/off and full-geometry tests.

## Same-entrypoint result and real idle caller

`exp-5ngfhlvifc` / `art-wanrutaz3g`, measurement source
`d372787e374d5f19f73b6ba5a23d606c8dc64c62`, completed source/port/manual
phases. Each of eight source workers and eight port workers reproduced the
same SPMD failure. Each exported six `sdy.manual_computation` regions, one
side-effect custom call and no `mhlo.sharding` attribute. The manual annotation
candidate had one explicit `mhlo.sharding={manual}` in exported MLIR but still
failed with the same missing-sharding error in all eight workers. No numerical
case ran. The local-only `export-diagnostic-records.json` contains the parsed export records,
first compiler signatures and 24 complete actual-process runtime/import/maps
records; the same raw records and full MLIR belong to the Falcon artifact.
The raw JSON is intentionally excluded from the final PR diff; this mechanism
summary and experiment/artifact/source identities are the committed evidence.

The real target idle path differs from manually cropping a kernel result to
zero rows. `compiled_forward_shim._dp_attn_apply_bucket_metadata` publishes
`set_dp_buffer_len(pad_t*dp_size, pad_t, True, ...)`, so every rank's local FFN
buffer remains a nonempty physical bucket while any peer is active.
`DecoderCompilePolicy._match_busiest_dp_decode_rows` materializes idle decode
rows to the busiest peer explicitly to preserve collective issue order.
`spec_decode_compile_policy_eagle3._normalize_dp_idle_forward` sets
`spec_info.keep_outputs_padded=True` and materializes idle speculative rows.
The FFN's already-summed exit still performs `dp_scatter` into that padded
local buffer via `index_select`; cropping to logical outputs happens later.
Global-empty calls return the same empty result on all ranks without entering
the kernel.

The pure-call diagnostic will therefore exercise a compiled nonempty DP-slot
`index_select` at a 64-row physical bucket as live DPs drain 8->4->1 and refill,
using zero/invalid routes for idle slots. It will also explicitly permute
logical expert ownership relative to physical process rank and add real
H6144/I2048/E256/TopK8 64- and 2048-row cases. This tests the actual dependency
contract; an artificial local empty result would be a different unsupported
caller and is not treated as proof that the model drops the kernel.

`exp-s9ky7hjpat` / `art-hn3h0or1gq`, measurement
`74232ca78456cbd6eb1864b9161102e10fd8034e`, obtained resources and ran the
pure-call diagnostic. It passed the previous effect check but all eight
workers failed before numerical execution with `partitionId instruction is not
supported for SPMD partitioning`, pointing to the bridge's outer
`lax.axis_index("tensor")`. The current candidate follows the existing fused EP
adapter's explicit-rank contract instead: gather logical rank operands and
compute this rank's physical mesh position by comparing its operand with that
table. Pallas-internal device ids, RDMA and synchronization remain unchanged.

The production pure-call adaptation and this rank fix are frozen together for
the next preflight; no earlier failed run is a numerical or performance pass.
The independent multihost launcher tests that exact committed production bridge
before any model load, including its actual loading/storage/shared-gather
operations. Source debug variables are rejected before load rather than
allowed to silently alter prequantization or compile-cache behavior.

`exp-ekf3axxxpk` / `art-ixx0uwqz2k`, measurement
`871b9460fd5e24145bd68bfb7b9c21d150c30695`, first executed the production
bridge with explicit rank operands. All eight ranks passed real loading/storage
and shared FP8 all-gather checks. Random, hot and zero-route small numerical
cases executed; the hot dense-reference relative L2 was approximately 0.13555.
The partial/padded case then failed its nonfinite check before full geometries
or idle tests. The recorded assertion did not locate NaNs by row, so it does
not prove whether valid rows were affected. The next candidate records exact
nonfinite rows and compact failure arrays and does not relax that check.

Source inspection independently confirms a mixed-slot correctness gap: scatter
and metadata count valid slots, but combine tests only slot zero, reads all K
expert results, and waits valid_rows*K copies. The port changes those three
operations together to the individual-slot contract, masking invalid operands
before multiplication. This is not a blanket NaN-to-zero conversion or a
whole-row output mask. Partially invalid route cases, including first-slot
invalid/later-slot valid, are added with shared off/on. Valid-route errors and
nonfinite output remain fatal. The source compatibility patch can be reversed
and SHA-checked mechanically, preserving an exact independent source module
for further same-entrypoint comparisons.

`exp-h5x8m9y9fr` / `art-7vlxetei08`, measurement
`06cecda79d800783534acfcd7b2d75515a746094` (implementation `04ba952921`),
ran the per-slot correction. All eight ranks again passed random6/hot12/zeros18
and the loading checks. The padded24 failure now identifies 107 nonfinite
**valid** rows, each with four valid routes, and zero nonfinite invalid rows.
All ranks observe the same row set. Thus slot masking alone does not resolve
the upstream numerical failure; full geometry, shared numerical and model E2E
acceptance remain blocked. Compact inputs/routes/output/reference arrays are in
`glm52-e2e/rank-0/kernel-failure-dumps/rank{rank}-rows24-padded-shared0.npz`
inside that artifact. This result does not justify masking valid results.

For bounded isolation, the manual diagnostic supports `--padded-only`: it
preserves the registered test's exact RNG consumption, executes only the failed
case, and skips the already-verified loading operations and later large tests.
`--modes=pure,port,unfold` compares the exact source with only its effect flag
adapted, the committed production kernel, and a temporary port using the
source's small-bt sequential gate/up branch at all bt sizes. The latter isolates
the larger-bt raw-dot/channel-scale folding branch and is not a production fix
or an acceptance pass. Phases run in fresh processes; all phases are attempted
and any unexpected failure keeps the diagnostic's overall exit nonzero.

The next candidate makes mutable HBM workspaces explicit Pallas call outputs,
following the existing fused EP backend. The source passed `pl.empty` buffers
as inputs and mutated them without returning or aliasing those inputs. That is
not the same buffer contract as the native pure ring (DMA into an output) or
the existing fused EP (HBM staging/recv/contrib outputs). The new interface
retains identical shapes, memory spaces and all four workspaces; it removes
their JAX input operands and returns them from the opaque Pallas call instead.
Only completed token rows leave the enclosing JAX function. No caller input
is modified, so no input/output alias is requested.

A local AST comparison confirms the entire `_fused_ep_moe_kernel` body, including
all math, DMA, barriers, semaphore waits and drain, is unchanged from `04ba952921`;
only its reference argument order changes from 19 inputs + 1 output to 15 inputs
+ 5 outputs. The exact-source reverse SHA remains checked. This is a compiler
buffer-interface adaptation, not evidence that aliasing caused the observed
NaNs; actual sequential-shape device validation is still required. The bt24
scale-folding math and every tuning choice are retained.

`exp-n2lc3kxqlx` / `art-q8hoxm2726`, measurement
`93d35535158b79ad49f14fca43cc836dc3ea70a5`, completed the isolated padded-only
comparison in fresh workers. All eight ranks in each of pure/port/unfold passed
with finite results. Relative L2 was 0.13696110 for pure and port, 0.13695937 for
unfold; host-observed post-warmup medians were about 0.50–0.58 ms. This does not
reproduce R6's sequential random/hot/zero/padded failure and does not establish
the folded math as a cause. The exact failed input's RNG sequence is retained;
the prior kernel invocation history differs. The explicit-output candidate
therefore retains the original scale-folding branch and all tuning choices,
and requires the complete sequential registered test rather than treating this
isolated result as acceptance.

A bounded CPU-only Falcon helper (`exp-3oan0ipcqm`, succeeded) read the real R6
NPZ without direct object-store access. Its compact JSON confirms the 107 bad
rows have all 512 features nonfinite, while the CPU reference is finite. The
bad-row set is exactly the union of the 94 rows selecting logical expert 6 or 7
and all 24 rows in physical mesh origin slot 6 (`144:168`); the two sets overlap
in 11 rows. No row outside that union is bad. In this test logical expert 6/7
belongs to logical EP rank 3, supplied by Torch rank 6's `(rank*3+1)%8` ownership
permutation. Equating that Torch rank with physical mesh slot 6 additionally
requires the actual mesh-rank table; that table was not directly logged, so the
hardware-identity inference remains qualified. All 182 valid input rows have
finite amax in approximately [0.516, 0.949], ordinary FP32 scale exponents, and
no `7f`/`ff` FP8-NaN byte in their embedded-scale bit patterns. This correlation
narrows the failure beyond invalid routing or a tiny precision difference, but
does not alone prove buffer aliasing or any other unique cause.

## Full sequential device acceptance

`exp-rjhbw8wnjp` / `art-u4z6fu8rww`, measurement
`00a14356a881e5dfaff5bc0c00aeba779a920fc2` (kernel `4924817ec2`), succeeded
with `MOE_PRODUCTION_SEQUENCE_PASS` and complete records from all eight ranks.
Each rank passed ten small routed/shared/partial-slot cases, the compiled
64-row idle DP drain/refill (8→4→1→8), and four real H6144/I2048/E256/TopK8
cases at local 6/24/64/2048 rows. All outputs were finite. Full-shape dense
reference relative L2 was respectively 0.0425229, 0.0449509, 0.0525455 and
0.0449743. Rank-0 host-observed post-warmup medians were 1.409, 1.533, 2.195 and
35.049 ms; these are bridge measurements, not model speedups. The large-bt
scale-folding math remains intact. Passing this sequence after the explicit
workspace-output adaptation supports that contract correction without proving
a unique mechanism for every prior NaN.

## Actual loader caller and retained pack buffers

The original binder connection to `SRTPlatform.process_model_after_loading`
was unreachable for GLM: its only model caller was Kimi K3. GLM inherits the
Deepseek loader; its own `post_load_weights` runs *before* the generic quant
postprocessing pass. The candidate now dispatches through the existing plugin
`prepare_model_for_fused_moe_decode` entry, called at the end of
`DefaultModelLoader.postprocess_weights`, before that entry's legacy backend
flag returns early. No new core branch or hook is added. Target routed-module
counts are checked (75 for the actual checkpoint); NextN retains its old path.
SBO/TBO decomposed forwards are rejected because they bypass this module's
ordinary `forward` and summed-by-compute boundary contract.

The current local TorchTPU checkout (`dd6e5886`, not proof of target-wheel
source) implements storage resize-to-zero by replacing its DeviceBufferRef.
DeferredOp owns copies of its input DeviceBufferRefs, explicitly retaining their
underlying buffers. This argues against inferring use-after-free from resize
alone. Real TPU pack/replace plus bit-exact CPU reads already passed; the next
preflight also feeds those actual replaced Parameters into the kernel. Since
this model-level conversion pass is outside the loader's per-quant-module
transient drain, each converted layer now synchronizes once before the next.
Native GroupCoordinator all-gather contains no explicit completion fence; the
local backend emits deferred DispatchOps/StableHLO. The new load-only drain
bounds retained old layouts and reports observed post-drain HBM use without
claiming an instantaneous peak. Model E2E and EP16 acceptance remain separate
from the completed EP8 kernel test.
