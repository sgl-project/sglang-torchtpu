# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA (DeepSeek Sparse Attention) Pallas kernels for TPU.

Vendored from vllm-torchtpu ``src/vllm_torchtpu/kernels/mla`` @ b94bc6d4f
(2026-10-08); each module's header names its upstream file.

``kernel.sparse_ragged_paged_attention`` attends, for every query token, over
the ``topk`` key positions an indexer selected for it, gathering those rows out
of the paged latent cache on SparseCore. The cache it reads is split into a
NoPE buffer and a RoPE buffer, FP8 only, in the layouts
``kv_cache_utils.SparseMLAKVCacheSpec`` describes. Unlike the dense MLA v2
kernel (``mla_ragged_paged_attention_v2``), it does not write the cache:
``kv_cache_utils.update_sparse_mla_kv_cache`` scatters the step's new latents
first, and the caller sequences the two.

Nothing here is imported at package import time, so environments without JAX
can still load the rest of SGLang. No attention backend uses these kernels yet.
"""
