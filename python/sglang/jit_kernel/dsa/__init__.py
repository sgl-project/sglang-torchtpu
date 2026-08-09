from sglang.srt.utils import is_cuda

from .paged_mqa_logits import (
    aiter_paged_mqa_logits,
    cutedsl_paged_mqa_logits,
    deepgemm_paged_mqa_logits_native,
    deepgemm_paged_mqa_logits_split,
)

if is_cuda():
    # CuteDSL needs cutlass/cuda.bindings; `not is_hip()` also matched TPU/NPU/CPU.
    from .cutedsl_paged_mqa_logits import CuteDSLPagedMQALogitsRunner, pick_dsl_expand
else:
    # In __all__ and imported by dsa_indexer, so they must still exist.
    CuteDSLPagedMQALogitsRunner = None
    pick_dsl_expand = None

__all__ = [
    "CuteDSLPagedMQALogitsRunner",
    "pick_dsl_expand",
    "aiter_paged_mqa_logits",
    "cutedsl_paged_mqa_logits",
    "deepgemm_paged_mqa_logits_native",
    "deepgemm_paged_mqa_logits_split",
]
