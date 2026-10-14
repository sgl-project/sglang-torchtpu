# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS TP8 decode megakernel package and serving entry-point shim."""


def maybe_make_weight_tap(causal_lm):
    from sglang_tpu.gptoss_decode_mk.runtime import (
        maybe_make_weight_tap as _maybe_make_weight_tap,
    )

    return _maybe_make_weight_tap(causal_lm)
