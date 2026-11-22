# CSA2 provenance

The CSA2 kernels come from `src/vllm_torchtpu/kernels/deepseek_v41/csa2/` in
[vllm-project/vllm-torchtpu at `e5ef2d82c`](https://github.com/vllm-project/vllm-torchtpu/tree/e5ef2d82c/src/vllm_torchtpu/kernels/deepseek_v41/csa2),
under Apache-2.0. The original port came from vLLM #2868 (`39e7c38b3`).

- `csa2.py` incorporates #3192 (`c4037b568`), including guarded uniform
  tiles, static bank selection for groups of four tiles, FP8 staging,
  and softmax normalization after PV. SWA still reads native BF16 pages.
  The shared INT32 SWA word pool (#2970) and block-outer cache layout (#3045)
  are outside this port. SWA page copies retain the existing fallback to
  page zero for invalid addresses.
- `cache.py` matches upstream gather and publication aside from SGLang's
  import paths and its fixes for out-of-range records (#2224) and distinct
  zero rows for invalid gather lanes (#2286).
- `pipeline.py` retains SGLang's #2286 chunk sizing for DSpark verify and
  adds upstream's barrier that puts the second gather after Q projection.
- `reference.py` is the upstream test reference for native BF16 SWA pages.
