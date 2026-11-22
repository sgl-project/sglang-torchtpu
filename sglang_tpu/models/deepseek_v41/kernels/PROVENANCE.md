# DeepSeek-V4.1 output projection

`output_projection.py` comes from
[vllm-torchtpu@e5ef2d82c](https://github.com/vllm-project/vllm-torchtpu/blob/e5ef2d82c/src/vllm_torchtpu/kernels/deepseek_v41/output_projection.py).
[PR #2756](https://github.com/vllm-project/vllm-torchtpu/pull/2756) last changed
the source in `5a827f38b`. The source is Apache-2.0 and retains its SPDX notices.

The port uses the plugin's TPU-generation query and FP8 range constant. The
Torch bridge and the kernel share the input shape and token-bucket contract.
The bridge applies JAX JIT through `cpu_export_jax_op`. The numerical kernel
and tiling come from the source revision above.

The plug-in packs the loaded BF16 `wo_a` into channelwise FP8 once and retains
the original weight for fallback. It reuses the post-load `wo_b` weight in
place, with a view of the existing per-column scale.
