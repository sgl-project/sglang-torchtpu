# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Regression guard for the PCP streaming custom op's aliasing invariant.

The PCP path deliberately registers its ``torch.library.custom_op`` with
``mutates_args=()`` and does the KV writeback OUTSIDE the op boundary via
``kv_cache.copy_(new_kv_cache)`` in
:func:`invoke_pcp_streaming_op`. Switching to ``mutates_args=("kv_cache",)``
(base RPA v3's pattern) forces functionalization (clone-before-call +
copy-back-at-graph-exit), which caused ~50% wrong first-token output at
ISL=4096 on the previous sglang PCP path (M-series investigation).

This test asserts the invariant by inspecting the source of the two
load-bearing sites — if a future refactor changes them, the test fails
with a clear pointer to the docstrings that explain why. Source
inspection over runtime construction because the op factory needs a real
JAX mesh (would trigger TPU init on a CPU-only CI runner).
"""

import inspect
import unittest

from sglang.srt.hardware_backend.tpu import attention_pallas_pcp_streaming_op
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class PcpStreamingOpMutatesArgsInvariantTest(CustomTestCase):
    """The mutates_args=() aliasing contract must not drift."""

    def test_op_registration_uses_empty_mutates_args(self) -> None:
        # pcp_streaming_jax_op constructs the custom_op. Read its source
        # and enforce mutates_args=(). Any change to
        # mutates_args=("kv_cache",) or similar would re-introduce the
        # functionalization bug documented in the module docstring.
        src = inspect.getsource(attention_pallas_pcp_streaming_op.pcp_streaming_jax_op)
        self.assertIn(
            "mutates_args=()",
            src,
            msg=(
                "pcp_streaming_jax_op MUST register the custom_op with "
                "mutates_args=(). See file docstring in "
                "attention_pallas_pcp_streaming_op.py — switching to "
                "mutates_args=('kv_cache',) forces functionalization "
                "(clone + copy-back), which produced ~50% wrong first-token "
                "output at ISL=4096 on the previous PCP path (M-series)."
            ),
        )

    def test_invoke_writes_via_explicit_copy_(self) -> None:
        # The OTHER half of the aliasing contract: KV writeback happens
        # OUTSIDE the op via kv_cache.copy_(new_kv_cache) in the caller
        # frame. Deleting or moving the copy_ silently drops KV writes;
        # this test catches that too.
        src = inspect.getsource(attention_pallas_pcp_streaming_op.invoke_pcp_streaming_op)
        self.assertIn(
            "kv_cache.copy_(new_kv_cache)",
            src,
            msg=(
                "invoke_pcp_streaming_op MUST perform the KV writeback via "
                "kv_cache.copy_(new_kv_cache) after the op call. See the "
                "invoke_pcp_streaming_op docstring. This is the load-bearing "
                "half of the mutates_args=() aliasing contract; removing it "
                "silently drops KV writes."
            ),
        )

    def test_jax_callable_leaves_aliases_empty(self) -> None:
        # _PcpStreamingJaxCallable / build_pcp_streaming_callable must NOT
        # pass input_output_aliases into the JaxCallable base — that would
        # activate JaxCallable's internal .copy_() loop and, combined with
        # a mutates_args change, defeat the aliasing (see comment inside
        # build_pcp_streaming_callable).
        src = inspect.getsource(
            attention_pallas_pcp_streaming_op.build_pcp_streaming_callable
        )
        self.assertNotIn(
            "input_output_aliases=",
            src,
            msg=(
                "build_pcp_streaming_callable MUST NOT pass "
                "input_output_aliases — that activates JaxCallable's "
                "internal copy_ loop and defeats the caller-side aliasing. "
                "See the DO NOT pass comment in build_pcp_streaming_callable."
            ),
        )


if __name__ == "__main__":
    unittest.main()
