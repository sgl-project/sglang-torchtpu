"""Unit tests for the in-process TPU CompilationManager."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.hardware_backend.tpu.compilation.bucketer import Bucketer
from sglang.srt.hardware_backend.tpu.compilation.compilation_manager import (
    CompilationManager,
    _materialize_out,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase


class _FakeAttnBackend:
    """Records ``init_forward_metadata`` calls so tests can assert shape."""

    def __init__(self):
        self.calls: list[tuple[int, int]] = []  # (num_tokens, bs)

    def init_forward_metadata(self, forward_batch):
        self.calls.append(
            (forward_batch.input_ids.shape[0], forward_batch.seq_lens.shape[0])
        )


def _model_runner(device="cpu"):
    """Minimal ModelRunner stand-in for unit tests.

    ``prepare_forward_batch`` is included because ``_run_one`` calls it
    before ``init_forward_metadata`` (mirrors the runtime chain installed
    by ``TpuCompiledForwardShim.install()``)."""
    return SimpleNamespace(
        device=device,
        model=SimpleNamespace(
            forward=MagicMock(),
            prepare_forward_batch=MagicMock(),
        ),
        attn_backend=_FakeAttnBackend(),
        req_to_token_pool=SimpleNamespace(),
        token_to_kv_pool=SimpleNamespace(),
    )


def _make_manager(model_runner=None, token_buckets=(64, 256), bs_buckets=(1, 4)):
    if model_runner is None:
        model_runner = _model_runner()
    return CompilationManager(
        model_runner=model_runner,
        compiled_fns={"forward": model_runner.model.forward},
        t_bucketer=Bucketer(name="Token", buckets=list(token_buckets)),
        b_bucketer=Bucketer(name="Bs", buckets=list(bs_buckets)),
    )


class TestMakeDummyForwardBatch(CustomTestCase):
    """``_make_dummy_forward_batch`` shapes valid + ghost slots correctly."""

    def test_extend_shapes_and_ghost_tokens(self):
        mr = _model_runner()
        mgr = _make_manager(mr)
        fb = mgr._make_dummy_forward_batch(num_tokens=64, bs=4, mode=ForwardMode.EXTEND)
        # Token-axis tensors padded to num_tokens; bs=4 valid + 60 ghost.
        self.assertEqual(fb.input_ids.shape[0], 64)
        self.assertEqual(fb.positions.shape[0], 64)
        self.assertEqual(fb.out_cache_loc.shape[0], 64)
        self.assertEqual(fb.input_ids[:4].tolist(), [1, 1, 1, 1])
        self.assertEqual(fb.input_ids[4:].sum().item(), 0)
        self.assertEqual(fb.input_ids.dtype, torch.int64)
        self.assertEqual(fb.positions.dtype, torch.int64)
        # Per-request tensors at bs.
        self.assertEqual(fb.seq_lens.shape[0], 4)
        self.assertEqual(fb.seq_lens.dtype, torch.int64)
        self.assertEqual(fb.extend_seq_lens.shape[0], 4)
        self.assertEqual(fb.extend_seq_lens.tolist(), [1, 1, 1, 1])
        self.assertEqual(fb.extend_logprob_start_lens_cpu, [0, 0, 0, 0])
        # KV writes go to dummy slot 0 across the whole token range.
        self.assertEqual(fb.out_cache_loc.sum().item(), 0)

    def test_decode_has_no_extend_fields(self):
        mr = _model_runner()
        mgr = _make_manager(mr)
        fb = mgr._make_dummy_forward_batch(num_tokens=8, bs=8, mode=ForwardMode.DECODE)
        # Decode: one query token per running seq.
        self.assertEqual(fb.input_ids.shape[0], 8)
        self.assertEqual(fb.seq_lens.shape[0], 8)
        # Extend fields are None in decode (the path skips them).
        self.assertIsNone(fb.extend_seq_lens)
        self.assertIsNone(fb.extend_prefix_lens)
        self.assertIsNone(fb.extend_num_tokens)
        self.assertIsNone(fb.extend_logprob_start_lens_cpu)


class TestPrecompileLoops(CustomTestCase):
    """``precompile_all`` traverses the bucket grid and fires one forward
    per variant: EXTEND covers every (token, bs) pair where t >= bs,
    DECODE varies bs (tokens == bs)."""

    def test_extend_iterates_full_token_bs_grid(self):
        mr = _model_runner()
        mgr = _make_manager(mr, token_buckets=(8, 64, 256), bs_buckets=(1, 4))
        mgr._precompile_extend()
        # Every (t, b) pair where t >= b: at b=1 all 3 token buckets,
        # at b=4 all 3 — total 6.
        self.assertEqual(mr.model.forward.call_count, 6)
        # Order: outer loop over bs_buckets, inner over token_buckets.
        self.assertEqual(
            mr.attn_backend.calls,
            [(8, 1), (64, 1), (256, 1), (8, 4), (64, 4), (256, 4)],
        )

    def test_extend_skips_token_bucket_below_bs(self):
        # A 4-token bucket can't hold an 8-req extend (kernel would
        # crash on cu_q_lens); only (64, 1), (64, 8), (4, 1) clear.
        mr = _model_runner()
        mgr = _make_manager(mr, token_buckets=(4, 64), bs_buckets=(1, 8))
        mgr._precompile_extend()
        self.assertEqual(mr.model.forward.call_count, 3)
        self.assertEqual(mr.attn_backend.calls, [(4, 1), (64, 1), (64, 8)])

    def test_decode_loop_iterates_bs(self):
        mr = _model_runner()
        mgr = _make_manager(mr, token_buckets=(64,), bs_buckets=(1, 4, 8))
        mgr._precompile_decode()
        self.assertEqual(mr.model.forward.call_count, 3)
        # Decode: tokens == bs for each call.
        self.assertEqual(mr.attn_backend.calls, [(1, 1), (4, 4), (8, 8)])

    def test_precompile_all_combines_extend_and_decode(self):
        mr = _model_runner()
        mgr = _make_manager(mr, token_buckets=(8, 64), bs_buckets=(1, 4))
        mgr.precompile_all()
        # EXTEND: (8,1), (64,1), (8,4), (64,4) = 4. DECODE: (1,1), (4,4)
        # = 2. Total = 6.
        self.assertEqual(mr.model.forward.call_count, 6)


class TestRunOneCallsCompiledForward(CustomTestCase):
    """``_run_one`` wires the synthesized batch through the same chain
    serving uses: ``prepare_forward_batch`` → ``init_forward_metadata``
    → compiled ``forward``."""

    def test_run_one_calls_prepare_then_metadata_then_forward(self):
        mr = _model_runner()
        mgr = _make_manager(mr)
        mgr._run_one(num_tokens=64, bs=4, mode=ForwardMode.EXTEND)
        # prepare_forward_batch runs first so model-specific prep hooks
        # (Moss-VL etc.) see precompile too.
        mr.model.prepare_forward_batch.assert_called_once()
        self.assertEqual(mr.attn_backend.calls, [(64, 4)])
        mr.model.forward.assert_called_once()
        # Args: (input_ids, positions, forward_batch). Shapes match bucket.
        args, _ = mr.model.forward.call_args
        self.assertEqual(args[0].shape, (64,))
        self.assertEqual(args[1].shape, (64,))
        self.assertEqual(args[2].batch_size, 4)

    def test_run_one_uses_no_grad_not_inference_mode(self):
        mr = _model_runner()
        mgr = _make_manager(mr)
        cm = MagicMock()
        with patch(
            "sglang.srt.hardware_backend.tpu.compilation.compilation_manager.torch.no_grad",
            return_value=cm,
        ) as no_grad, patch(
            "sglang.srt.hardware_backend.tpu.compilation.compilation_manager.torch.inference_mode",
            side_effect=AssertionError("inference_mode should not be used"),
        ):
            mgr._run_one(num_tokens=64, bs=4, mode=ForwardMode.EXTEND)
        no_grad.assert_called_once_with()
        cm.__enter__.assert_called_once()

    def test_materialize_out_syncs_logits_and_flagged_kv_side_effects(self):
        logits = torch.zeros(1)
        kv0 = torch.ones(1)
        kv1 = torch.ones(1) * 2
        pool = SimpleNamespace(
            start_layer=2,
            layer_num=2,
            get_fused_kv_buffer=MagicMock(side_effect=[kv0, kv1]),
        )
        forward_batch = SimpleNamespace(
            attn_backend=SimpleNamespace(requires_kv_sync_after_compile=True),
            token_to_kv_pool=pool,
        )
        with patch("torch_tpu._internal.sync.sync.synchronize") as sync:
            _materialize_out(SimpleNamespace(next_token_logits=logits), forward_batch)
        sync.assert_called_once()
        tensors, kwargs = sync.call_args
        self.assertEqual(tensors[0], [logits, kv0, kv1])
        self.assertEqual(kwargs, {"wait": True})
        self.assertEqual(pool.get_fused_kv_buffer.call_args_list[0].args, (2,))
        self.assertEqual(pool.get_fused_kv_buffer.call_args_list[1].args, (3,))


if __name__ == "__main__":
    unittest.main()
