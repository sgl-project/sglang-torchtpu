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
