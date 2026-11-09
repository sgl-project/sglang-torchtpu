# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for TPU grammar masking and host-group token synchronization."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import grammar

register_cpu_ci(est_time=45, suite="base-a-test-cpu")


def _load_hooks():
    # Isolate these hooks from unrelated TPU hooks and their device dependencies.
    path = Path(__file__).resolve().parents[5] / "sglang_tpu/hooks/grammar.py"
    spec = importlib.util.spec_from_file_location("_tpu_grammar_hooks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gloo_worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=20),
    )
    try:
        cpu_group = dist.new_group([1, 2], backend="gloo")
        if rank in (1, 2):
            group = SimpleNamespace(world_size=2, ranks=[1, 2], cpu_group=cpu_group)
            for dtype in (torch.int32, torch.int64):
                token_ids = torch.tensor([200_000 + rank, rank], dtype=dtype)
                grammar.synchronize_token_ids(token_ids, group)
                torch.testing.assert_close(
                    token_ids, torch.tensor([200_001, 1], dtype=dtype)
                )
        dist.barrier()
    finally:
        dist.destroy_process_group()


class TestVocabMask(unittest.TestCase):
    def test_signed_words_partial_word_and_multiple_rows(self):
        allowed = ({0, 31, 32, 63, 69}, {1, 30, 33, 62, 68})
        mask = torch.zeros((2, 3), dtype=torch.int32)
        for row, tokens in enumerate(allowed):
            for token in tokens:
                bit = token % 32
                mask[row, token // 32] |= (1 << bit) if bit < 31 else -(1 << 31)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                logits = torch.arange(140, dtype=dtype).reshape(2, 70)
                expected = torch.full_like(logits, float("-inf"))
                for row, tokens in enumerate(allowed):
                    for token in tokens:
                        expected[row, token] = logits[row, token]
                grammar.apply_vocab_mask(logits, mask)
                torch.testing.assert_close(logits, expected)

    def test_unconstrained_and_fully_masked_rows(self):
        logits = torch.randn(2, 70)
        expected = logits[0].clone()
        mask = torch.stack(
            (torch.full((3,), -1, dtype=torch.int32), torch.zeros(3, dtype=torch.int32))
        )
        grammar.apply_vocab_mask(logits, mask)
        torch.testing.assert_close(logits[0], expected)
        self.assertTrue(torch.isneginf(logits[1]).all())

    def test_new_mask_shapes_do_not_unfreeze_decoder(self):
        @torch.compile(dynamic=False, fullgraph=True, backend="eager")
        def decoder(value):
            return value + 1

        decoder(torch.zeros(2))
        with torch.compiler.set_stance("fail_on_recompile"):
            for rows in (1, 3):
                logits = torch.zeros(rows, 96)
                mask = torch.zeros(rows, 3, dtype=torch.int32)
                mask[:, 2] = 1
                grammar.apply_vocab_mask(logits, mask)
                self.assertEqual(logits.argmax(-1).tolist(), [64] * rows)
            with self.assertRaisesRegex(RuntimeError, "recompile"):
                decoder(torch.zeros(3))

    def test_compile_error_propagates_and_restores_stance(self):
        @torch.compile(dynamic=False, fullgraph=True, backend="eager")
        def decoder(value):
            return value + 2

        decoder(torch.zeros(2))
        with torch.compiler.set_stance("fail_on_recompile"):
            with patch.object(
                grammar, "_mask_logits", side_effect=RuntimeError("mask failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "mask failed"):
                    grammar.apply_vocab_mask(torch.zeros(1, 32), torch.zeros(1, 1))
            with self.assertRaisesRegex(RuntimeError, "recompile"):
                decoder(torch.zeros(3))


class TestMaskHooks(unittest.TestCase):
    def setUp(self):
        self.hooks = _load_hooks()

    def test_non_tpu_inputs_keep_original_behavior(self):
        logits, mask = torch.zeros(1, 32), torch.zeros(1, 1, dtype=torch.int32)
        original = Mock(return_value="original result")
        owner = object()
        self.assertEqual(
            self.hooks.apply_grammar_mask(original, owner, logits, mask),
            "original result",
        )
        original.assert_called_once_with(owner, logits, mask)
        original.reset_mock()
        self.assertEqual(
            self.hooks.apply_backend_mask(original, logits, mask), "original result"
        )
        original.assert_called_once_with(logits, mask)

    def test_instance_and_static_hooks_dispatch_tpu_logits(self):
        class Grammar:
            def apply_vocab_mask(self, logits, vocab_mask):
                raise AssertionError("original TPU mask must not run")

        class Backend:
            @staticmethod
            def apply_vocab_mask(logits, vocab_mask):
                raise AssertionError("original TPU mask must not run")

        registry = type("IsolatedRegistry", (HookRegistry,), {})
        logits = SimpleNamespace(device=SimpleNamespace(type="tpu"))
        mask = object()
        with patch.object(self.hooks, "apply_vocab_mask") as apply:
            for target, hook, hook_type in self.hooks.hook_specs():
                cls = Backend if "GrammarBackend" in target else Grammar
                with patch("pkgutil.resolve_name", return_value=cls):
                    registry._apply_target(target, [(hook_type, hook, None)])
            Grammar().apply_vocab_mask(logits=logits, vocab_mask=mask)
            Backend.apply_vocab_mask(logits=logits, vocab_mask=mask)
            Backend().apply_vocab_mask(logits=logits, vocab_mask=mask)
        self.assertEqual(apply.call_count, 3)
        apply.assert_called_with(logits, mask)


class TestTokenSynchronization(unittest.TestCase):
    def test_single_rank_needs_no_copy_or_collective(self):
        token_ids = Mock()
        with patch.object(dist, "broadcast") as broadcast:
            grammar.synchronize_token_ids(token_ids, SimpleNamespace(world_size=1))
        token_ids.to.assert_not_called()
        token_ids.copy_.assert_not_called()
        broadcast.assert_not_called()

    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_real_subgroup_uses_global_source_rank_and_preserves_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_gloo_worker, args=(str(Path(directory) / "init"),), nprocs=3)


if __name__ == "__main__":
    unittest.main()
