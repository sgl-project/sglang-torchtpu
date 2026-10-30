"""One-host-per-engine Raiden P/D, EAGLE3, MXINT8 KV, prefix cache + RAM offload.

Issue #1066's long-context test: both engines store target KV as MXINT8 (draft
KV stays bf16) at GPT-OSS-120B's full 131072-token context. Only the prefill
engine keeps a radix prefix cache backed by Raiden HiCache in host RAM; that is
where multi-turn sessions spill to and reload from. The decode engine is the
EAGLE3 P/D decode of pd_singlehost_spec.

PD_KV_CACHE_DTYPE=bfloat16 selects bf16 target KV instead. In-tree,
KVCacheConfigurator rejects mxint8 until the TPU image ships the MXINT8 backend
extension, so the nightly runs this recipe on bf16 until then.
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b import pd_singlehost_spec

KV_CACHE_DTYPE = os.environ.get("PD_KV_CACHE_DTYPE") or "mxint8"

GROUP = deepcopy(pd_singlehost_spec.GROUP)
PREFILL_CONFIG = deepcopy(pd_singlehost_spec.PREFILL_CONFIG)
DECODE_CONFIG = deepcopy(pd_singlehost_spec.DECODE_CONFIG)

for _config in (PREFILL_CONFIG, DECODE_CONFIG):
    _config["server_args"].update(
        kv_cache_dtype=KV_CACHE_DTYPE,
        # Left unset, the draft follows the target onto MXINT8, which the
        # in-tree EAGLE draft attention cannot read (bf16 and fp8 only).
        speculative_draft_kv_cache_dtype="bfloat16",
        # MXINT8 needs a power-of-two page of at least 128 tokens, and P/D needs
        # the same page size on both engines.
        page_size=128,
        context_length=131072,
        # About two full contexts of HBM KV per engine. Three ~123k sessions
        # outgrow it on the prefill, so their prefixes must come back from host
        # RAM. Deliberately under 262144 tokens, where sglang serializes VMEM
        # prefetches against issue #363's precompile deadlock: the deadlock
        # tracks pool bytes, and an MXINT8 pool is small per token.
        max_total_tokens=245760,
        # The multi-turn gate reads P->D transfer timing and the P/D failure
        # counters from both engines.
        enable_metrics=True,
    )

PREFILL_CONFIG["server_args"].update(
    disable_radix_cache=False,
    enable_hierarchical_cache=True,
    # A host pool of 3x the device pool holds every session's prefix and the
    # ~126k offload probe plus its fillers.
    hicache_ratio=3,
    hicache_io_backend="raiden",
    hicache_mem_layout="page_first_fused",
    hicache_write_policy="write_through",
    enable_cache_report=True,
)
