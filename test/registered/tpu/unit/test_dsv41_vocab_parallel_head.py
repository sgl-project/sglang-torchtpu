# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 vocab-parallel LM head: logits equal the plain matmul on CPU, each rank
keeps its own vocab slice, the target and DSpark draft heads route through it,
and setups it cannot serve keep the whole head.

The SPMD head needs 8 devices; without them the check reruns itself in a
subprocess on 8 forced CPU devices. Runs on the host; JAX comes with the TPU
stack.
"""

import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41 import vocab_parallel_head as vph

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

WORLD, VOCAB, WIDTH = 8, 8 * 48, 64


def check_logits_equal_plain_matmul() -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = Mesh(np.array(jax.devices()[:WORLD]), ("model",))
    sharded = NamedSharding(mesh, P("model", None))
    head = jax.jit(vph.make_head(mesh, WORLD, VOCAB))
    rng = np.random.default_rng(0)
    # Rows per rank of the target verify (6 per request) and the draft (5).
    for rows in (48, 40, 8):
        x = jnp.asarray(rng.standard_normal((WORLD * rows, WIDTH)), jnp.bfloat16)
        w = jnp.asarray(rng.standard_normal((VOCAB, WIDTH)), jnp.bfloat16)
        want = jax.lax.dot_general(
            x, w, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)
        xs = jax.device_put(x, sharded)
        got = head(xs, jax.device_put(w, sharded))
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        # Each rank must score its own vocab slice.
        shifted = jax.device_put(jnp.roll(w, VOCAB // WORLD, axis=0), sharded)
        assert not np.array_equal(np.asarray(head(xs, shifted)), np.asarray(want))


def _lm_head(vocab=VOCAB, width=WIDTH, dtype=torch.bfloat16) -> ParallelLMHead:
    """A whole ParallelLMHead as attention TP 1 builds it, without a TP group."""
    head = ParallelLMHead.__new__(ParallelLMHead)
    torch.nn.Module.__init__(head)
    head.weight = torch.nn.Parameter(
        torch.arange(vocab * width, dtype=torch.float32).view(vocab, width).to(dtype),
        requires_grad=False,
    )
    head.bias = None
    head.tp_size = 1
    head.org_vocab_size = vocab
    head.quant_method = UnquantizedEmbeddingMethod()
    return head


def _model(lm_head) -> SimpleNamespace:
    return SimpleNamespace(
        lm_head=lm_head,
        model=SimpleNamespace(embed_tokens=object()),
        logits_processor=SimpleNamespace(use_fp32_lm_head=False),
    )


def _parallel(**changes) -> SimpleNamespace:
    values = dict(attn_dp_size=WORLD, attn_tp_size=1, tp_size=WORLD, pp_size=1)
    values.update(changes)
    return SimpleNamespace(**values)


class TestVocabParallelHeadLogits(CustomTestCase):
    def test_logits_equal_plain_matmul(self):
        import jax

        if len(jax.devices()) >= WORLD:
            check_logits_equal_plain_matmul()
            return
        code = (
            "import importlib.util;"
            f"spec = importlib.util.spec_from_file_location('t', {__file__!r});"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
            "m.check_logits_equal_plain_matmul()"
        )
        env = dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            XLA_FLAGS=f"--xla_force_host_platform_device_count={WORLD}",
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(
            result.returncode, 0, result.stdout[-4000:] + result.stderr[-4000:]
        )

    def test_op_function_annotations_are_types(self):
        """torch_tpu's jax_op rejects string annotations (``from __future__ import
        annotations`` in the module): the server died at load building the op."""
        import inspect

        import jax

        captured = {}

        def fake_export(name, function, **kwargs):
            captured["function"] = function

        stubs = {
            "torch_tpu._internal.pallas": mock.MagicMock(),
            "sglang_tpu.pallas_jax_op": SimpleNamespace(cpu_export_jax_op=fake_export),
        }
        with (
            mock.patch.dict(sys.modules, stubs),
            mock.patch.object(vph, "make_head"),
        ):
            vph._build_head_op.__wrapped__(WORLD, VOCAB, WIDTH)
        signature = inspect.signature(captured["function"])
        for param in signature.parameters.values():
            self.assertIs(param.annotation, jax.Array)
        self.assertIs(signature.return_annotation, jax.Array)


class TestWholeHeadReason(CustomTestCase):
    def _reason(self, lm_head=None, parallel=None, model=None, world=WORLD):
        lm_head = lm_head if lm_head is not None else _lm_head()
        return vph.whole_head_reason(
            model=model if model is not None else _model(lm_head),
            lm_head=lm_head,
            parallel=parallel or _parallel(),
            world_size=world,
        )

    def test_recipe_setup_splits(self):
        self.assertIsNone(self._reason())

    def test_setups_that_keep_the_whole_head(self):
        tied = _lm_head()
        tied_model = _model(tied)
        tied_model.model.embed_tokens = tied
        fp32 = _lm_head()
        fp32_model = _model(fp32)
        fp32_model.logits_processor.use_fp32_lm_head = True
        biased = _lm_head()
        biased.bias = torch.nn.Parameter(torch.zeros(VOCAB))
        padded = _lm_head()
        padded.org_vocab_size = VOCAB - 8
        sharded = _lm_head()
        sharded.tp_size = 2
        quantized = _lm_head()
        quantized.quant_method = object()
        cases = {
            "no attention DP": dict(parallel=_parallel(attn_dp_size=1)),
            "attention TP 2": dict(parallel=_parallel(attn_tp_size=2)),
            "TP short of the world": dict(parallel=_parallel(tp_size=4)),
            "PP": dict(parallel=_parallel(pp_size=2)),
            "tied embedding": dict(lm_head=tied, model=tied_model),
            "FP32 head": dict(lm_head=fp32, model=fp32_model),
            "bias": dict(lm_head=biased),
            "padded vocab": dict(lm_head=padded),
            "sharded head": dict(lm_head=sharded),
            "quantized": dict(lm_head=quantized),
            "FP32 weight": dict(lm_head=_lm_head(dtype=torch.float32)),
            "indivisible vocab": dict(lm_head=_lm_head(vocab=VOCAB + 4)),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                self.assertIsNotNone(self._reason(**kwargs))


class _FakeOp:
    """Stands in for the SPMD op on one host: records its inputs."""

    def __init__(self):
        self.calls = []

    def __call__(self, x, w):
        self.calls.append((x, w))
        return torch.zeros(x.shape[0], VOCAB, dtype=torch.bfloat16)


class TestShardAndRoute(CustomTestCase):
    def _shard(self, rank=3):
        lm_head = _lm_head()
        whole = lm_head.weight.detach().clone()
        model = _model(lm_head)
        op = _FakeOp()
        with (
            mock.patch("torch.distributed.get_world_size", return_value=WORLD),
            mock.patch("torch.distributed.get_rank", return_value=rank),
            mock.patch(
                "sglang.srt.runtime_context.get_parallel", return_value=_parallel()
            ),
            mock.patch.object(vph, "_build_head_op", return_value=op),
        ):
            self.assertTrue(vph.shard_lm_head(model))
        return model, whole, op

    def test_rank_keeps_its_vocab_slice(self):
        shard = VOCAB // WORLD
        for rank in (0, 3, WORLD - 1):
            with self.subTest(rank=rank):
                model, whole, _ = self._shard(rank)
                head = model.lm_head
                self.assertIsInstance(head, vph.VocabParallelLMHead)
                self.assertEqual(head.full_vocab, VOCAB)
                self.assertEqual(head.org_vocab_size, VOCAB)
                torch.testing.assert_close(
                    head.weight,
                    whole[rank * shard : (rank + 1) * shard],
                    rtol=0,
                    atol=0,
                )

    def test_target_logits_route_through_the_op(self):
        from sglang_tpu.models.deepseek_v41.hooks import (
            tpu_dsv41_vocab_parallel_lm_head,
        )

        model, _, op = self._shard()
        hidden = torch.randn(12, WIDTH, dtype=torch.float32)
        original = mock.Mock()
        logits = tpu_dsv41_vocab_parallel_lm_head(
            original, object(), hidden, model.lm_head
        )
        original.assert_not_called()
        self.assertEqual(tuple(logits.shape), (12, VOCAB))
        x, w = op.calls[-1]
        self.assertEqual(x.dtype, torch.bfloat16)
        self.assertIs(w, model.lm_head.weight)

    def test_whole_head_takes_the_original(self):
        from sglang_tpu.models.deepseek_v41.hooks import (
            tpu_dsv41_vocab_parallel_lm_head,
        )

        original = mock.Mock(return_value="whole")
        head = _lm_head()
        hidden = torch.zeros(2, WIDTH)
        self.assertEqual(
            tpu_dsv41_vocab_parallel_lm_head(original, "proc", hidden, head), "whole"
        )
        original.assert_called_once_with("proc", hidden, head, None, out=None)

    def test_draft_base_logits_route_through_the_op(self):
        from sglang_tpu.models.deepseek_v41.dspark import tpu_compute_base_logits

        model, _, op = self._shard()
        draft = SimpleNamespace(
            lm_head=model.lm_head,
            collapse_hc_head=lambda x: x,
            stages=[SimpleNamespace(norm=lambda x: x * 2)],
            _use_fp32_lm_head=False,
        )
        hidden = torch.randn(10, WIDTH, dtype=torch.bfloat16)
        logits, x_post_hc = tpu_compute_base_logits(draft, hidden)
        self.assertEqual(tuple(logits.shape), (10, VOCAB))
        self.assertIs(x_post_hc, hidden)
        torch.testing.assert_close(op.calls[-1][0], hidden * 2, rtol=0, atol=0)
        draft._use_fp32_lm_head = True
        with self.assertRaises(ValueError):
            tpu_compute_base_logits(draft, hidden)

    def test_reload_is_refused(self):
        from sglang_tpu.models.deepseek_v41.hooks import (
            tpu_dsv41_vocab_parallel_load_weights,
        )

        model, _, _ = self._shard()
        original = mock.Mock()
        with self.assertRaises(RuntimeError):
            tpu_dsv41_vocab_parallel_load_weights(original, model, iter(()))
        original.assert_not_called()


if __name__ == "__main__":
    unittest.main()
