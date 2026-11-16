# SPDX-License-Identifier: Apache-2.0
"""Private radix selector used by the TPU DSA indexer.

The existing indexer custom op remains the callable runtime API. This package
owns no device buffers, custom-op singleton, or physical slot mapping.
"""

import jax
import jax.numpy as jnp
from jax._src import tpu_info
from jax.experimental import xla_metadata

from sglang_tpu.attention.pallas.radix_topk.kernel import radix_topk_v2
from sglang_tpu.attention.pallas.radix_topk.tuned_configs import (
    SUPPORTED_K,
    get_score_width_config,
)


def supports_radix_topk(k: int) -> bool:
    # Read the Pallas target, not jax.devices(): TorchTPU exports against a CPU
    # JAX backend while tpu_info_for_cpu_export supplies the physical TPU info.
    return k in SUPPORTED_K and tpu_info.get_device_kind() in (
        "TPU7x",
        "TPU v7",
        "TPU v7x",
    )


def select_radix_topk(scores, k, row_lengths, *, scheduling_group_id=None, config=None):
    """Select exact logical positions from an existing score buffer.

    Accepts fp32 or its int32 bit representation, as the streamindex scorer
    produces it. Unlike the JAX source, -inf is a mask and never a valid pick,
    matching sparsecore_topk. NaNs are unsupported. Ties are unordered.
    """
    if scores.dtype == jnp.int32:
        scores = jax.lax.bitcast_convert_type(scores, jnp.float32)
    if config is None:
        tuned = get_score_width_config(scores.shape[-1], k)
        block_size, reconstruct_threshold, reverse_scan = (
            tuned.block_size,
            tuned.reconstruct_threshold,
            True,
        )
        # A bounded page64/capacity92160 sweep on v7 found this configuration
        # least costly for GLM verify. It still did not beat the incumbent;
        # keep the integration opt-in and do not force V2 for prefill.
        if k == 2048 and scores.shape[-1] == 92160 and scores.shape[0] <= 64:
            block_size, reconstruct_threshold, reverse_scan = 16384, 8192, False
    else:
        block_size, reconstruct_threshold, reverse_scan = config
    result = radix_topk_v2(
        scores,
        k=k,
        valid_lengths=row_lengths,
        block_size=block_size,
        reconstruct_threshold=reconstruct_threshold,
        reverse_scan=reverse_scan,
        indices_only=True,
    )
    if scheduling_group_id is not None:
        result = xla_metadata.set_xla_metadata(
            result, _scheduling_group_id=scheduling_group_id
        )
    return result
