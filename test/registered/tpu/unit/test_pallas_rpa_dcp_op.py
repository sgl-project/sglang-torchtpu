"""Contract test for the DCP torch custom op registered by
``attention_pallas_rpa_dcp.build_dcp_kernel_op``.

The backend and the JAX wrapper agree only by position: ``static_argnums``
names indices of the wrapper's signature, and the op's ``register_fake`` is what
``torch.compile`` traces instead of the kernel. Both are pinned here -- the op
name, the order of the ten traced tensors and the eight static flags that
follow them, which argument the kernel mutates, and the shapes and dtypes of the
``(output, lse)`` pair -- so a reordering that would only surface as a
traced-vs-static mismatch on a TPU end-to-end run fails in the unit suite.

Registration imports JAX and torch_tpu, so the case is skipped elsewhere; the
meta call itself needs no TPU device. The refusal of another kernel spec comes
before those imports and runs everywhere.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_pallas_rpa_dcp import (
    DCP_OP_NAME,
    build_dcp_kernel_op,
)

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

# The op's arguments, in order. build_dcp_kernel_op passes the JAX wrapper
# static_argnums=(11, ..., 21), which are exactly the eleven entries after
# attention_sink here.
ARG_NAMES = (
    "queries",
    "keys",
    "values",
    "kv_cache_fused",
    "kv_lens",
    "page_indices",
    "cu_q_lens",
    "distribution",
    "kv_lens_global",
    "cp_rank",
    "attention_sink",
    "sm_scale",
    "sliding_window",
    "soft_cap",
    "k_scale",
    "v_scale",
    "cache_only",
    "skip_kv_write",
    "cp_group_size",
    "attention_scope",
    "rpa_cases",
    "tail_decode",
)
NUM_TRACED_TENSORS = 10
STATIC_FLAGS = (
    "sm_scale",
    "sliding_window",
    "soft_cap",
    "k_scale",
    "v_scale",
    "cache_only",
    "skip_kv_write",
    "cp_group_size",
    "attention_scope",
    "rpa_cases",
    "tail_decode",
)
FIRST_STATIC_ARGNUM = 11


def _has_torch_tpu_pallas() -> bool:
    """JAX and torch_tpu import, and torch knows the ``tpu`` device type.

    ``torch.library.custom_op(device_types=["tpu"])`` needs the device type
    torch_tpu registers when it initializes; a CPU-only pod that merely imports
    torch_tpu does not have it, so the op cannot be registered there.
    """
    try:
        import jax  # noqa: F401
        import torch_tpu  # noqa: F401
        from torch_tpu._internal import pallas  # noqa: F401

        torch._C._dispatch_key_for_device("tpu")
    except Exception:
        return False
    return True


@unittest.skipIf(
    not _has_torch_tpu_pallas(), "Requires JAX, torch_tpu and the tpu device type"
)
class TestDcpOpContract(CustomTestCase):
    NUM_TOKENS = 6
    NUM_Q_HEADS = 16
    NUM_KV_HEADS = 2
    HEAD_DIM = 64
    NUM_SEQS = 3

    def setUp(self):
        from sglang_tpu.attention_pallas_rpa_batched import (
            RPA_BATCHED_SPEC,
        )

        self.op = build_dcp_kernel_op(RPA_BATCHED_SPEC)
        namespace, name = DCP_OP_NAME.split("::")
        self.overload = getattr(getattr(torch.ops, namespace), name).default

    def test_the_op_name_resolves(self):
        self.assertEqual(self.overload.name(), DCP_OP_NAME)

    def test_argument_order_and_return_arity(self):
        schema = self.overload._schema
        self.assertEqual([arg.name for arg in schema.arguments], list(ARG_NAMES))
        self.assertEqual(len(schema.returns), 2)

    def test_the_first_ten_arguments_are_tensors(self):
        schema = self.overload._schema
        for arg in schema.arguments[:NUM_TRACED_TENSORS]:
            self.assertEqual(str(arg.type), "Tensor", arg.name)
        self.assertEqual(schema.arguments[NUM_TRACED_TENSORS].name, "attention_sink")

    def test_the_static_flags_sit_at_the_static_argnums(self):
        self.assertEqual(ARG_NAMES[FIRST_STATIC_ARGNUM:], STATIC_FLAGS)
        for offset, name in enumerate(STATIC_FLAGS):
            self.assertEqual(
                self.overload._schema.arguments[FIRST_STATIC_ARGNUM + offset].name,
                name,
            )

    def test_only_the_kv_cache_is_mutated(self):
        schema = self.overload._schema
        written = [
            arg.name
            for arg in schema.arguments
            if arg.alias_info is not None and arg.alias_info.is_write
        ]
        self.assertEqual(written, ["kv_cache_fused"])

    def _meta_args(self):
        def meta(*shape, dtype):
            return torch.empty(*shape, dtype=dtype, device="meta")

        return (
            meta(
                self.NUM_TOKENS, self.NUM_Q_HEADS, self.HEAD_DIM, dtype=torch.bfloat16
            ),
            meta(
                self.NUM_TOKENS, self.NUM_KV_HEADS, self.HEAD_DIM, dtype=torch.bfloat16
            ),
            meta(
                self.NUM_TOKENS, self.NUM_KV_HEADS, self.HEAD_DIM, dtype=torch.bfloat16
            ),
            meta(4, 128, self.NUM_KV_HEADS, 2, 128, dtype=torch.bfloat16),
            meta(self.NUM_SEQS, dtype=torch.int32),
            meta(self.NUM_SEQS * 2, dtype=torch.int32),
            meta(self.NUM_SEQS + 1, dtype=torch.int32),
            meta(3, dtype=torch.int32),
            meta(self.NUM_SEQS, dtype=torch.int32),
            meta(1, dtype=torch.int32),
        )

    def test_the_fake_impl_returns_the_output_and_an_fp32_lse(self):
        args = self._meta_args()
        output, lse = self.overload(*args)
        self.assertEqual(output.shape, args[0].shape)
        self.assertEqual(output.dtype, args[0].dtype)
        self.assertEqual(tuple(lse.shape), (self.NUM_TOKENS, self.NUM_Q_HEADS))
        self.assertEqual(lse.dtype, torch.float32)

    def test_the_op_runs_every_scope_on_the_device(self):
        """Execute the real kernel through the torch op once per scope and
        case selection, so the JAX wrapper's glue (imports, static-arg
        plumbing) is exercised, not only its schema."""
        import torch_tpu  # noqa: F401

        device = torch.device("tpu")
        page_size = 128
        cp = 2
        seqs = self.NUM_SEQS
        pages_per_seq = 2

        def dev(*shape, dtype=torch.bfloat16, scale=0.1):
            return (torch.randn(*shape) * scale).to(dtype).to(device)

        kv_cache = dev(seqs * pages_per_seq, page_size, self.NUM_KV_HEADS, 2, 128)
        page_indices = torch.arange(seqs * pages_per_seq, dtype=torch.int32)

        def local_len(n, rank=0):
            cycle = cp * page_size
            rem = min(max(n % cycle - rank * page_size, 0), page_size)
            return (n // cycle) * page_size + rem

        # (scope, q_lens, global seq_lens, striped kv_lens, distribution,
        # rpa_cases, tail_decode): decode rows first in the plain layout, a
        # chunk row first and the decode rows last in the tail_decode layout.
        cases = (
            (3, [1, 1, 4], [300, 130, 200],
             [local_len(300), local_len(130), local_len(196)], [2, 2, 3], 0, False),
            (2, [1, 1, 4], [300, 130, 200], [300, 130, 200], [2, 2, 3], 2, False),
            (3, [4, 1, 1], [200, 300, 130],
             [local_len(196), local_len(300), local_len(130)], [1, 1, 3], 0, True),
            (2, [4, 1, 1], [200, 300, 130], [200, 300, 130], [1, 1, 3], 2, True),
        )
        for scope, q_lens, seq_lens, kv_lens, distribution, rpa_cases, tail in cases:
            total_q = sum(q_lens)
            cu_q_lens = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lens), 0)), dtype=torch.int32)
            out, lse = self.overload(
                dev(self.NUM_TOKENS, self.NUM_Q_HEADS, self.HEAD_DIM),
                dev(self.NUM_TOKENS, self.NUM_KV_HEADS, self.HEAD_DIM),
                dev(self.NUM_TOKENS, self.NUM_KV_HEADS, self.HEAD_DIM),
                kv_cache,
                torch.tensor(kv_lens, dtype=torch.int32).to(device),
                page_indices.to(device),
                cu_q_lens.to(device),
                torch.tensor(distribution, dtype=torch.int32).to(device),
                torch.tensor(seq_lens, dtype=torch.int32).to(device),
                torch.tensor([0], dtype=torch.int32).to(device),
                None,
                1.0,
                None,
                None,
                None,
                None,
                False,
                False,
                cp,
                scope,
                rpa_cases,
                tail,
            )
            self.assertEqual(tuple(out.shape), (self.NUM_TOKENS, self.NUM_Q_HEADS, self.HEAD_DIM))
            self.assertTrue(torch.isfinite(out.float().cpu()[:total_q]).all(), scope)
            # Unprocessed rows report -inf; nothing may be NaN.
            self.assertFalse(torch.isnan(lse.cpu()).any(), scope)


class TestDcpOpKernelSpec(CustomTestCase):
    def test_another_kernel_spec_is_refused(self):
        # The base builds the distribution for its own kernel spec, and the
        # DCP op reads it as the batched kernel's.
        from sglang_tpu.attention_pallas_rpa_v3 import (
            RPA_V3_SPEC,
        )

        with self.assertRaisesRegex(NotImplementedError, "pallas_tpu_rpa_batched"):
            build_dcp_kernel_op(RPA_V3_SPEC)


if __name__ == "__main__":
    unittest.main()
