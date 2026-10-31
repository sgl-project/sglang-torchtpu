"""Unit tests for the TPU prefill logit rows (sglang_tpu/compilation/logit_rows.py)
and the compiled-forward shim putting them on ForwardBatch.logit_rows (LR1).

Pure-logic tests: CPU tensors only, and the shim's import chain is torch_tpu-free,
so this runs on the CPU CI runner. The helpers mirror the ones in
test_compiled_forward_shim.py (kept local so this file stands alone)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.compilation.compiled_forward_shim import CompiledForwardShim
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.compilation.pending_call import PendingCall

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ForwardModeMock:
    """Minimal forward_mode stand-in (as in test_compiled_forward_shim.py)."""

    def __init__(self, mode):
        self._mode = mode
        self.name = mode.upper()

    def is_decode(self):
        return self._mode == "decode"

    def is_idle(self):
        return self._mode == "idle"

    def is_extend(self):
        return self._mode in ("extend", "mixed", "target_verify")

    def is_mixed(self):
        return self._mode == "mixed"

    def is_target_verify(self):
        return self._mode == "target_verify"

    def is_draft_extend_v2(self):
        return self._mode == "draft_extend_v2"


class _StubLogitsProcessor:
    """LogitsProcessor stand-in with the real call signature."""

    def __init__(self):
        self.compute_input_logprobs = False
        self.calls = 0

    def __call__(self, input_ids, hidden_states, lm_head, logits_metadata):
        self.calls += 1
        return LogitsProcessorOutput(
            next_token_logits=torch.zeros(1, 4),
            input_token_logprobs=torch.tensor([-0.5, -1.5, -2.5]),
        )


class _StubModel:
    def __init__(self):
        self.logits_processor = _StubLogitsProcessor()
        self.lm_head = object()


def _policy(model):
    policy = DecoderTpuCompilePolicy()
    policy.model_runner = SimpleNamespace(
        model=model,
        token_to_kv_pool=SimpleNamespace(layer_transfer_counter=None),
        pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    return policy


class TestLogitRows(CustomTestCase):
    """LR1 logit rows: the host-side builder, its identity rules, and the
    shim putting the packed field on EXTEND forwards only."""

    @staticmethod
    def _fb(
        mode="extend",
        lens=(5, 3, 7),
        real_b=None,
        return_logprob=False,
        starts=None,
        capture=CaptureHiddenMode.NULL,
    ):
        lens = list(lens)
        return SimpleNamespace(
            is_prefill_only=False,
            mm_input_embeds=None,
            input_embeds=None,
            spec_info=None,
            seq_lens=torch.zeros(len(lens), dtype=torch.int32),
            forward_mode=_ForwardModeMock(mode),
            return_logprob=return_logprob,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_input_logprob_token_ids_gpu=None,
            extend_seq_lens_cpu=lens,
            extend_logprob_start_lens_cpu=list(starts) if starts else [0] * len(lens),
            capture_hidden_mode=capture,
            input_ids=torch.arange(sum(lens), dtype=torch.int32),
            logit_rows=None,
            _tpu_real_q=sum(lens),
            _tpu_real_b=len(lens) if real_b is None else real_b,
        )

    def _rules(self, fb, wants=False):
        from sglang_tpu.compilation.logit_rows import identity_reasons

        return identity_reasons(
            fb,
            wants_input_logprob=wants,
            capture_hidden_mode=fb.capture_hidden_mode,
            real_b=fb._tpu_real_b,
        )

    def test_rows_are_each_requests_last_row(self):
        from sglang_tpu.compilation.logit_rows import build_logit_rows

        rows = build_logit_rows(self._fb(), identity=False, real_b=3, device="cpu")
        self.assertEqual(rows.dtype, torch.int32)
        self.assertEqual(rows.tolist(), [4, 7, 14, 0, 3, 0])
        one = build_logit_rows(
            self._fb(lens=(9,)), identity=False, real_b=1, device="cpu"
        )
        self.assertEqual(one.tolist(), [8, 0, 0, 0, 1, 0])

    def test_padded_requests_are_not_rows(self):
        from sglang_tpu.compilation.logit_rows import build_logit_rows

        fb = self._fb(lens=(5, 3, 0, 0), real_b=2)
        self.assertEqual(self._rules(fb), [])
        rows = build_logit_rows(fb, identity=False, real_b=2, device="cpu")
        self.assertEqual(rows.tolist(), [4, 7, 0, 0, 2, 0])

    def test_identity_path_has_a_fixed_shape(self):
        from sglang_tpu.compilation.logit_rows import build_logit_rows

        rows = build_logit_rows(self._fb(), identity=True, real_b=3, device="cpu")
        self.assertEqual(rows.tolist(), [0, 1, 2, 3, 0, 1])

    def test_identity_rules(self):
        self.assertEqual(self._rules(self._fb()), [])
        self.assertEqual(self._rules(self._fb(), wants=True), ["input logprobs"])
        full = self._rules(self._fb(capture=CaptureHiddenMode.FULL))
        self.assertEqual(full, ["hidden-state capture of every row"])
        # The last row's hidden state is a logit row: no identity.
        self.assertEqual(self._rules(self._fb(capture=CaptureHiddenMode.LAST)), [])
        spec = self._fb()
        spec.spec_info = SimpleNamespace()
        self.assertEqual(self._rules(spec), ["speculative input"])
        five = self._rules(self._fb(lens=(1, 1, 1, 1, 1)))
        self.assertEqual(five, ["5 requests > 4 logit rows"])
        missing = self._fb()
        missing.extend_seq_lens_cpu = None
        self.assertEqual(self._rules(missing), ["no host extend lengths"])
        self.assertEqual(
            self._rules(self._fb(lens=(3, 0))), ["a request with no extend rows"]
        )

    def test_only_plain_extend_forwards_carry_rows(self):
        from sglang_tpu.compilation.logit_rows import takes_logit_rows

        self.assertTrue(takes_logit_rows(self._fb("extend")))
        for mode in ("decode", "mixed", "target_verify", "idle"):
            self.assertFalse(takes_logit_rows(self._fb(mode)), mode)

    def _shim(self, seen, *, rows_on=True, prompt_logprob=False):
        def compiled(input_ids, positions, forward_batch, *a, **kw):
            rows = forward_batch.logit_rows
            seen["rows"] = None if rows is None else rows.tolist()
            seen["capture"] = forward_batch.capture_hidden_mode
            return LogitsProcessorOutput(
                next_token_logits=torch.zeros(
                    len(forward_batch.extend_seq_lens_cpu), 10
                ),
                hidden_states=torch.zeros(int(forward_batch.input_ids.shape[0]), 5),
            )

        override = get_context().override_server_args(
            pp_size=1, dp_size=1, attn_dp_size=1
        )
        override.install()
        self.addCleanup(override.restore)
        env = {"SGLANG_TPU_FOUR_STAGE_LOGIT_ROWS": "1" if rows_on else "0"}
        with (
            patch.dict("os.environ", env),
            envs.SGLANG_TPU_ENABLE_PROMPT_LOGPROB.override(prompt_logprob),
        ):
            return CompiledForwardShim(compiled, policy=_policy(_StubModel()))

    def test_shim_sets_the_rows_for_the_forward_and_restores_the_batch(self):
        seen, fb = {}, self._fb()
        self._shim(seen)(None, None, fb)
        self.assertEqual(seen["rows"], [4, 7, 14, 0, 3, 0])
        self.assertIsNone(fb.logit_rows)

    def test_shim_uses_the_identity_path_for_prompt_logprobs(self):
        # The batch asks for prompt logprobs: the graph sees FULL capture (the
        # prompt-logprob variant) and the identity rows.
        seen, fb = {}, self._fb(return_logprob=True)
        self._shim(seen, prompt_logprob=True)(None, None, fb)
        self.assertEqual(seen["rows"], [0, 1, 2, 3, 0, 1])
        self.assertIs(seen["capture"], CaptureHiddenMode.FULL)
        # The same server with no logprob request prunes, although the graph
        # still sees the FULL capture variant.
        seen, fb = {}, self._fb(return_logprob=False)
        self._shim(seen, prompt_logprob=True)(None, None, fb)
        self.assertEqual(seen["rows"], [4, 7, 14, 0, 3, 0])
        self.assertIs(seen["capture"], CaptureHiddenMode.FULL)

    def test_shim_leaves_other_forwards_and_the_flag_off_alone(self):
        seen = {}
        self._shim(seen)(None, None, self._fb(mode="decode"))
        self.assertIsNone(seen["rows"])
        seen = {}
        self._shim(seen, rows_on=False)(None, None, self._fb())
        self.assertIsNone(seen["rows"])

    def test_warmup_compiles_the_extend_graph_with_the_rows(self):
        seen, fb = {}, self._fb()
        shim = self._shim(seen)

        def warmup(*args, **kwargs):
            out = shim.fn(*args, **kwargs)
            return PendingCall(lambda: out)

        shim.warmup_async_fn = warmup
        shim.warmup_async(None, None, fb).wait()
        self.assertEqual(seen["rows"], [4, 7, 14, 0, 3, 0])
        self.assertIsNone(fb.logit_rows)

    def test_async_warmup_runs_the_deferred_call_with_the_rows(self):
        # direct_compile.warmup_async returns PendingCall(lambda: artifact(...)):
        # the artifact resolves its tensor inputs from the live forward_batch
        # only at wait(), after the shim has restored logit_rows to None.
        seen, fb = {}, self._fb()
        shim = self._shim(seen)

        def warmup(*args, **kwargs):
            return PendingCall(lambda: shim.fn(*args, **kwargs))

        shim.warmup_async_fn = warmup
        pending = shim.warmup_async(None, None, fb)
        self.assertNotIn("rows", seen)  # nothing ran yet
        self.assertIsNone(fb.logit_rows)  # the batch is restored before the wait
        pending.wait()
        self.assertEqual(seen["rows"], [4, 7, 14, 0, 3, 0])
        self.assertIsNone(fb.logit_rows)


if __name__ == "__main__":
    unittest.main()
