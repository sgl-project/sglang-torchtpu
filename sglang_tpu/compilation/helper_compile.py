# SPDX-License-Identifier: Apache-2.0
"""TPU helper compilation: no dynamic-shape helpers."""

from __future__ import annotations

from sglang.srt.compilation.helper_compile import HelperCompilePolicy


class TpuHelperCompilePolicy(HelperCompilePolicy):
    def compile_helper(self, fn, *, dynamic, backend, options):
        # torch_tpu has no dynamic-shape compile; inside a compiled model graph
        # the helper is traced inline anyway.
        return fn


TPU_HELPER_COMPILE_POLICY = TpuHelperCompilePolicy()
