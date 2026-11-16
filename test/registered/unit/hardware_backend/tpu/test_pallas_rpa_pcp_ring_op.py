"""Contract tests for the ring-mode prefill-CP torch op
(``attention_pallas_rpa_pcp.build_pcp_ring_op``), with the fused kernel stubbed.

The backend, the torch op, the exported JAX function and the kernel agree only
by position and by partition spec, and none of that is exercised before a TPU
end-to-end run. Pinned here on CPU:

* the kernel receives every operand under the name the kernel contract gives
  it, the mesh axis as the ring axis, the pool as a read-only operand
  (``update_kv_cache=False``) and the scales as static values;
* the op is exported as a ``shard_map`` over the world mesh: each rank's shard
  is exactly the tensor that rank passed, the output is this rank's rows, and
  nothing is donated or aliased (a kernel with remote DMA would make XLA copy
  an aliased pool);
* the torch op's argument order, that it mutates nothing, and its fake impl.

The export cases need JAX and torch_tpu but no device; the torch-op schema
cases also need the ``tpu`` device type and are skipped on CPU-only hosts.
"""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import attention_pallas_rpa_pcp as pcp

register_cpu_ci(est_time=30, suite="base-a-test-cpu")
register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit", chips=1)

TENSOR_ARGS = (
    "queries",
    "keys",
    "values",
    "kv_cache_fused",
    "cache_kv_lens",
    "cache_page_indices",
    "cache_cu_q_lens",
    "cache_distribution",
    "new_kv_lens",
    "new_page_indices",
    "new_cu_q_lens",
    "new_distribution",
    "new_kv_page_indices",
    "q_positions",
    "kv_cache_lens",
    "cp_rank",
)
STATIC_ARGS = ("sm_scale", "k_scale", "v_scale")

_HAS_JAX = bool(
    importlib.util.find_spec("jax") and importlib.util.find_spec("torch_tpu")
)


def _has_tpu_device_type() -> bool:
    if not _HAS_JAX:
        return False
    try:
        import torch_tpu  # noqa: F401

        torch._C._dispatch_key_for_device("tpu")
    except Exception:
        return False
    return True


class TestPcpRingOpNames(CustomTestCase):
    def test_op_name_is_distinct_from_the_other_rpa_ops(self):
        from sglang_tpu.attention_pallas_rpa_batched import RPA_BATCHED_SPEC
        from sglang_tpu.attention_pallas_rpa_dcp import DCP_OP_NAME

        self.assertEqual(
            len({pcp.PCP_RING_OP_NAME, DCP_OP_NAME, RPA_BATCHED_SPEC.op_name}), 3
        )
        self.assertEqual(pcp._TENSOR_ARGS, TENSOR_ARGS)
        self.assertEqual(pcp._STATIC_ARGS, STATIC_ARGS)

    def test_another_kernel_spec_is_refused(self):
        spec = SimpleNamespace(op_name="sglang_tpu_pallas::some_other_kernel")
        with self.assertRaisesRegex(NotImplementedError, "pallas_tpu_rpa_batched"):
            pcp.build_pcp_ring_op(spec, 8)

    def test_fake_impl_returns_the_query_shape_without_tracing(self):
        queries = torch.empty(6, 4, 8, dtype=torch.bfloat16)
        out = pcp._fake_impl(queries, *[None] * 15, 1.0, None, None)
        self.assertEqual((out.shape, out.dtype), (queries.shape, queries.dtype))


@unittest.skipUnless(_HAS_JAX, "requires JAX and TorchTPU")
class TestPcpRingKernelCall(CustomTestCase):
    def test_kernel_gets_every_operand_by_its_contract_name(self):
        from sglang_tpu.attention.pallas.batched_rpa_pcp import configs

        operands = {name: object() for name in TENSOR_ARGS}
        calls = []

        def kernel(*args, **kwargs):
            calls.append((args, kwargs))
            return "out"

        with patch.dict("os.environ", {"SGLANG_TPU_USE_SEQ_ALONG_LANE_KV": "0"}):
            out = pcp._invoke_pcp_kernel(
                kernel, cp_size=8, sm_scale=0.25, k_scale=0.5, v_scale=2.0, **operands
            )
        self.assertEqual(out, "out")
        (args, kwargs) = calls[0]
        # queries, the gathered keys / values and the pool are positional.
        self.assertEqual(
            args,
            tuple(operands[n] for n in ("queries", "keys", "values", "kv_cache_fused")),
        )
        for name in TENSOR_ARGS[4:]:
            self.assertIs(kwargs.pop(name), operands[name], name)
        # The layout is the experimental package's own enum, not the
        # production batched_rpa one.
        self.assertIs(type(kwargs["kv_layout"]), configs.KVLayout)
        self.assertEqual(
            kwargs,
            dict(
                cp_group_size=8,
                ring_axis_name="model",
                ring_mesh_axis_names=("model",),
                sm_scale=0.25,
                k_scale=0.5,
                v_scale=2.0,
                kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
                # The pool is read-only: the caller writes the owned pages.
                update_kv_cache=False,
                # Cached prefixes are whole stripes in ring mode.
                cache_lens_cp_page_aligned=True,
            ),
        )


@unittest.skipUnless(_HAS_JAX, "requires JAX and TorchTPU")
class TestPcpRingOpExport(CustomTestCase):
    """Export the op on CPU with a stand-in kernel and a mocked world mesh."""

    CP = 4
    ATTN_TP = 1
    TOKENS, Q_HEADS, KV_HEADS, DIM, PAGE = 64, 8, 2, 128, 16
    PAD_B, PAGES_PER_SEQ = 2, 3

    def _export(self):
        import jax

        from sglang.test.cpu_test_utils import get_cpu_export_mesh
        from sglang_tpu.attention_pallas_rpa_batched import RPA_BATCHED_SPEC
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export
        from sglang_tpu.pallas_jax_op import _jax_placeholders

        axes = pcp.pcp_ring_mesh_axes(self.ATTN_TP)
        shape = (self.CP,) if self.ATTN_TP == 1 else (self.CP, self.ATTN_TP)
        mesh = get_cpu_export_mesh(axes, shape)
        seen = {}

        def kernel(queries, keys, values, kv_cache, **kwargs):
            seen["shapes"] = {
                "queries": queries.shape,
                "keys": keys.shape,
                "values": values.shape,
                "kv_cache_fused": kv_cache.shape,
            }
            for name, value in kwargs.items():
                if hasattr(value, "shape"):
                    seen["shapes"][name] = value.shape
                else:
                    seen.setdefault("static", {})[name] = value
            # Depends on the ring axis, as the real kernel's neighbour ids do.
            rank = jax.lax.axis_index(axes[0]) + kwargs["cp_rank"][0]
            return queries + rank.astype(queries.dtype)

        captured = {}

        def capture(jax_kernel):
            captured["jax_kernel"] = jax_kernel
            return jax_kernel

        with (
            patch.object(pcp, "_OP_CACHE", {}),
            patch.object(pcp, "_register_torch_op", side_effect=capture),
        ):
            op = pcp.build_pcp_ring_op(
                RPA_BATCHED_SPEC,
                self.CP,
                attn_tp_size=self.ATTN_TP,
                mesh=mesh,
                kernel=kernel,
            )
        jax_kernel = captured["jax_kernel"]
        self.assertIs(op, jax_kernel)

        gathered_pages = self.CP * self.TOKENS // self.PAGE
        fp8 = torch.float8_e4m3fn

        def tensor(*shape, dtype=torch.int32):
            return torch.zeros(*shape, dtype=dtype)

        self.local = dict(
            queries=tensor(self.TOKENS, self.Q_HEADS, self.DIM, dtype=torch.bfloat16),
            keys=tensor(self.CP * self.TOKENS, self.KV_HEADS, self.DIM, dtype=fp8),
            values=tensor(self.CP * self.TOKENS, self.KV_HEADS, self.DIM, dtype=fp8),
            kv_cache_fused=tensor(5, self.PAGE, 1, 4, 128, dtype=fp8),
            cache_kv_lens=tensor(self.PAD_B),
            cache_page_indices=tensor(self.PAD_B * self.PAGES_PER_SEQ),
            cache_cu_q_lens=tensor(self.PAD_B + 1),
            cache_distribution=tensor(3),
            new_kv_lens=tensor(2 * self.PAD_B),
            new_page_indices=tensor(2 * self.PAD_B * self.PAGES_PER_SEQ),
            new_cu_q_lens=tensor(2 * self.PAD_B + 1),
            new_distribution=tensor(3),
            new_kv_page_indices=tensor(2 * self.PAD_B * gathered_pages),
            q_positions=tensor(2 * self.PAD_B),
            kv_cache_lens=tensor(2 * self.PAD_B),
            cp_rank=tensor(1),
        )
        args = tuple(self.local[name] for name in TENSOR_ARGS) + (0.125, 0.5, 0.25)
        in_specs, _ = pcp.pcp_ring_partition_specs(self.ATTN_TP)
        placeholders = _jax_placeholders(args, mesh=mesh, partition_specs=in_specs)
        with (
            tpu_info_for_cpu_export(),
            jax._src.config.export_ignore_forward_compatibility(True),
        ):
            lowered = jax_kernel.exported(*placeholders)
        return jax_kernel, lowered, seen

    def test_each_rank_passes_and_gets_its_own_shard(self):
        jax_kernel, lowered, seen = self._export()
        # Inside shard_map the kernel sees exactly the per-rank tensors ...
        for name in TENSOR_ARGS:
            self.assertEqual(
                tuple(seen["shapes"][name]), tuple(self.local[name].shape), name
            )
        # ... the exported program takes one such shard per rank along dim 0 ...
        world = self.CP * self.ATTN_TP
        for name, aval in zip(TENSOR_ARGS, lowered.in_avals):
            local = tuple(self.local[name].shape)
            self.assertEqual(tuple(aval.shape), (world * local[0], *local[1:]), name)
        # ... and returns this rank's attention rows, nothing else.
        self.assertEqual(len(lowered.out_avals), 1)
        out = lowered.out_avals[0]
        self.assertEqual(
            tuple(out.shape), (world * self.TOKENS, self.Q_HEADS, self.DIM)
        )
        self.assertEqual(str(out.dtype), "bfloat16")

    def test_statics_reach_the_kernel_and_the_rank_stays_traced(self):
        _, _, seen = self._export()
        self.assertEqual(
            {k: seen["static"][k] for k in ("sm_scale", "k_scale", "v_scale")},
            dict(sm_scale=0.125, k_scale=0.5, v_scale=0.25),
        )
        self.assertEqual(seen["static"]["cp_group_size"], self.CP)
        axes = pcp.pcp_ring_mesh_axes(self.ATTN_TP)
        self.assertEqual(seen["static"]["ring_axis_name"], axes[0])
        self.assertEqual(seen["static"]["ring_mesh_axis_names"], axes)
        self.assertFalse(seen["static"]["update_kv_cache"])
        # cp_rank is an operand, so every rank traces the same program.
        self.assertEqual(tuple(seen["shapes"]["cp_rank"]), (1,))

    def test_the_pool_is_neither_donated_nor_aliased(self):
        jax_kernel, lowered, _ = self._export()
        self.assertEqual(list(jax_kernel.donate_argnums), [])
        self.assertEqual(dict(jax_kernel.input_output_aliases), {})
        mlir = lowered.mlir_module()
        self.assertNotIn("jax.buffer_donor", mlir)
        self.assertNotIn("tf.aliasing_output", mlir)
        self.assertEqual(
            jax_kernel.static_argnums,
            tuple(range(len(TENSOR_ARGS), len(TENSOR_ARGS) + len(STATIC_ARGS))),
        )


class TestPcpRingOpExport2DMesh(TestPcpRingOpExport):
    """PCP4 x attention TP 2: a ``(pcp, model)`` mesh in torch-rank order,
    ringing along ``pcp``; every rank still passes and gets its own shard."""

    ATTN_TP = 2

    def test_mesh_axes_and_specs(self):
        from jax.sharding import PartitionSpec as P

        self.assertEqual(pcp.pcp_ring_mesh_axes(1), ("model",))
        self.assertEqual(pcp.pcp_ring_mesh_axes(2), ("pcp", "model"))
        in_specs, out_specs = pcp.pcp_ring_partition_specs(2)
        self.assertEqual(in_specs[0], P(("pcp", "model")))
        self.assertEqual(out_specs, (P(("pcp", "model")),))
        self.assertEqual(in_specs[-1], P())


@unittest.skipUnless(
    _has_tpu_device_type(), "Requires JAX, torch_tpu and the tpu device type"
)
class TestPcpRingTorchOpSchema(CustomTestCase):
    def setUp(self):
        with patch.object(pcp, "_OP_CACHE", {}):
            self.op = pcp._register_torch_op(lambda *args: args[0])
        namespace, name = pcp.PCP_RING_OP_NAME.split("::")
        self.overload = getattr(getattr(torch.ops, namespace), name).default

    def test_argument_order_and_return_arity(self):
        schema = self.overload._schema
        self.assertEqual(
            [arg.name for arg in schema.arguments], list(TENSOR_ARGS + STATIC_ARGS)
        )
        for arg in schema.arguments[: len(TENSOR_ARGS)]:
            self.assertEqual(str(arg.type), "Tensor", arg.name)
        self.assertEqual(len(schema.returns), 1)

    def test_nothing_is_mutated(self):
        written = [
            arg.name
            for arg in self.overload._schema.arguments
            if arg.alias_info is not None and arg.alias_info.is_write
        ]
        self.assertEqual(written, [])


if __name__ == "__main__":
    unittest.main()
