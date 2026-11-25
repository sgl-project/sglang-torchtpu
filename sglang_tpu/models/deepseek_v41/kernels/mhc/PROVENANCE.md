# Shifted mHC on TPU7x

`shifted_post_pre_rms.py` comes from vllm-project/vllm-torchtpu
`src/vllm_torchtpu/kernels/deepseek_v41/mhc/shifted_post_pre_rms.py`
at `e5ef2d82c` (last kernel change `9311c49ab`, PR #2876).
The port explicitly pads input gates to 128-lane windows for unaligned token
counts. The upstream kernel reads whole windows during gate preparation, which
extends beyond the logical VMEM shape of the last partial window without padding.

SGLang calls it through a CPU-exported TorchTPU op. The weight loader
uses the torch packing algorithm from
`src/vllm_torchtpu/models/vllm/deepseek_v41/common/mhc.py` at the same commit.
SGLang stores the packed weight in a nonpersistent buffer so the original FP32
weight remains available to the unfused path and to checkpoint loading.
The bridge uses the source runtime's resident gates and early coefficient
preparation settings.
