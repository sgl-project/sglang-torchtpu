# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash text decoder: the whole-model compile boundary.

Text-only serving (--language-model-only) compiles the whole model, as
TpuGraphRunner's fallback does; this policy only adds ``spmd_safe``. Under
--tpu-compile-tracer direct the graph inputs are the weights, the lifted rank,
the split MLA pool plus KDA state (hybrid_linear_direct_compile) and both
backends' metadata, so every rank traces the same graph and
--use-spmd-compile compiles each warmup once and broadcasts it.
"""

from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy


class Glm5NextDecoderTpuCompilePolicy(DecoderTpuCompilePolicy):
    """Whole-model policy for text-only GLM-5.3-Flash.

    ``spmd_safe``: rank-dependent values reach the graph as inputs (the lifted
    rank for the DSA indexer split and the vocab shards, per-rank weights and
    pools). Verify a recipe with ``VERIFY_SPMD_SAFE=1`` before it enables
    --use-spmd-compile: a rank-dependent constant would ship the owner rank's
    executable to every rank with no error.
    """

    spmd_safe = True
