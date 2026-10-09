"""Unit tests for the pipeline-parallel pieces of the TPU compiled-forward path.

CPU only: model_runner stand-ins, torch_tpu stubbed where the code imports it.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import inspect
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.hardware_backend.tpu.compilation import decoder_compile_policy
from sglang.srt.hardware_backend.tpu.compilation.bucketer import Bucketer
from sglang.srt.hardware_backend.tpu.compilation.compiled_forward_shim import (
    CompiledForwardShim,
    _fit_pp_proxy_in_args,
    _fit_pp_proxy_rows,
)
from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
    _materialize_out,
)
from sglang.srt.hardware_backend.tpu.compilation.pp_proxy_pytree import (
    flatten_pp_proxy_with_keys,
    register_pp_proxy_pytree,
    unflatten_pp_proxy,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
    LogitsOutputMode,
    PPProxyTensors,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.test_utils import CustomTestCase

_HIDDEN = 16


class _FakeAttnBackend:
    def __init__(self):
        self.calls = []

    def init_forward_metadata(self, forward_batch):
        self.calls.append(
            (forward_batch.input_ids.shape[0], forward_batch.seq_lens.shape[0])
        )


def _pp_group(first: bool, last: bool):
    return SimpleNamespace(is_first_rank=first, is_last_rank=last)


def _model_runner(pp_group=None):
    if pp_group is None:
        pp_group = _pp_group(first=True, last=True)
    runner = SimpleNamespace(
        device="cpu",
        dtype=torch.bfloat16,
        model=SimpleNamespace(
            forward=MagicMock(),
            prepare_forward_batch=MagicMock(),
        ),
        attn_backend=_FakeAttnBackend(),
        req_to_token_pool=SimpleNamespace(),
        token_to_kv_pool=SimpleNamespace(),
        token_to_kv_pool_allocator=SimpleNamespace(
            free_pages=torch.arange(4, dtype=torch.int32)
        ),
        model_config=SimpleNamespace(model_is_mrope=False, hidden_size=_HIDDEN),
        is_draft_worker=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        server_args=SimpleNamespace(page_size=1, dp_size=1, disaggregation_mode="null"),
        prepare_dummy_forward_batch=lambda forward_batch: forward_batch,
    )
    runner.pp_group = pp_group
    return runner


def _policy(model_runner, token_buckets=(64, 256), bs_buckets=(1, 4)):
    return DecoderTpuCompilePolicy(
        model_runner=model_runner,
        t_bucketer=Bucketer(name="Token", buckets=list(token_buckets)),
        b_bucketer=Bucketer(name="Bs", buckets=list(bs_buckets)),
        owner=model_runner.model,
    )


class TestResolvedExtendCeiling(CustomTestCase):
    def test_ceiling_follows_runtime_override_without_mutating_raw_config(self):
        with get_context().override_server_args(
            chunked_prefill_size=2048,
            max_prefill_tokens=8192,
            pp_size=8,
            enable_dynamic_chunking=False,
        ):
            self.assertEqual(
                decoder_compile_policy.max_extend_tokens(context_len=4096), 2048
            )
            get_context().override("test.extend_ceiling", chunked_prefill_size=16384)
            self.assertEqual(
                decoder_compile_policy.max_extend_tokens(context_len=4096), 16384
            )

    def test_unchunked_ceiling_covers_tokens_from_multiple_requests(self):
        with get_context().override_server_args(
            chunked_prefill_size=-1,
            max_prefill_tokens=8192,
        ):
            self.assertEqual(
                decoder_compile_policy.max_extend_tokens(context_len=4096), 8192
            )


class TestWarmupProxy(CustomTestCase):
    def setUp(self):
        super().setUp()
        # Both predicates read the published parallel config, which these
        # tests do not publish.
        for name in ("is_dp_attention_enabled", "enable_num_token_non_padded"):
            predicate = patch.object(decoder_compile_policy, name, return_value=False)
            predicate.start()
            self.addCleanup(predicate.stop)

    def test_non_first_stage_warms_up_with_a_zero_proxy_at_the_bucket(self):
        policy = _policy(_model_runner(_pp_group(first=False, last=False)))

        args, kwargs = policy._make_warmup_inputs(
            num_tokens=64, bs=4, mode=ForwardMode.EXTEND
        )

        self.assertEqual(args[0].shape[0], 64)
        self.assertEqual(set(kwargs), {"pp_proxy_tensors"})
        proxy = kwargs["pp_proxy_tensors"]
        self.assertIsInstance(proxy, PPProxyTensors)
        self.assertEqual(set(proxy.tensors), {"hidden_states", "residual"})
        for tensor in proxy.tensors.values():
            self.assertEqual(tuple(tensor.shape), (64, _HIDDEN))
            self.assertEqual(tensor.dtype, torch.bfloat16)
            self.assertTrue(bool((tensor == 0).all()))

    def test_last_stage_still_receives_a_proxy(self):
        policy = _policy(_model_runner(_pp_group(first=False, last=True)))
        _, kwargs = policy._make_warmup_inputs(
            num_tokens=4, bs=4, mode=ForwardMode.DECODE
        )
        self.assertEqual(
            tuple(kwargs["pp_proxy_tensors"]["residual"].shape), (4, _HIDDEN)
        )

    def test_first_stage_and_non_pipelined_model_get_no_proxy(self):
        for runner in (
            _model_runner(_pp_group(first=True, last=False)),
            _model_runner(),
        ):
            _, kwargs = _policy(runner)._make_warmup_inputs(
                num_tokens=64, bs=4, mode=ForwardMode.EXTEND
            )
            self.assertEqual(kwargs, {})

    def test_run_one_forwards_the_proxy_to_the_compiled_forward(self):
        runner = _model_runner(_pp_group(first=False, last=False))
        policy = _policy(runner)
        with patch.object(decoder_compile_policy, "_materialize_out") as materialize:
            policy._run_one(num_tokens=64, bs=4, mode=ForwardMode.EXTEND)
        runner.model.forward.assert_called_once()
        proxy = runner.model.forward.call_args.kwargs["pp_proxy_tensors"]
        self.assertEqual(tuple(proxy["hidden_states"].shape), (64, _HIDDEN))
        materialize.assert_called_once()


class TestLogitsProcessorGating(CustomTestCase):
    def test_only_the_last_stage_compiles_the_logits_processor(self):
        self.assertTrue(
            _policy(
                _model_runner(_pp_group(first=False, last=True))
            )._compiles_logits_processor()
        )
        self.assertFalse(
            _policy(
                _model_runner(_pp_group(first=True, last=False))
            )._compiles_logits_processor()
        )
        self.assertTrue(_policy(_model_runner())._compiles_logits_processor())

    def test_a_language_model_owner_needs_its_own_logits_processor(self):
        # Kimi compiles language_model with the logits processor inside; Gemma4
        # compiles a bare text model and runs its logits eagerly afterwards.
        runner = _model_runner()
        runner.model = SimpleNamespace(
            language_model=SimpleNamespace(logits_processor=object())
        )
        policy = _policy(runner)
        policy.owner = runner.model.language_model
        self.assertTrue(policy._compiles_logits_processor())

        runner.model.language_model = SimpleNamespace()
        policy.owner = runner.model.language_model
        self.assertFalse(policy._compiles_logits_processor())

    def test_non_last_stage_key_ignores_the_logits_mode(self):
        policy = _policy(_model_runner(_pp_group(first=False, last=False)))
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            batch_size=4,
            logits_output_mode=LogitsOutputMode.DISTRIBUTED_TOP_K,
        )
        key = policy._direct_compile_specialization_key(
            None, torch.zeros(64), None, forward_batch
        )
        self.assertEqual(key[-1], None)
        self.assertEqual(key[3], 64)

    def test_non_last_stage_warms_only_full_vocab(self):
        policy = _policy(_model_runner(_pp_group(first=False, last=False)))
        with patch(
            "sglang.srt.hardware_backend.tpu.sampler.distributed_top_k_candidate_width_for_model"
        ) as width:
            self.assertIsNone(policy._distributed_top_k_width())
            self.assertEqual(
                policy._warmup_logits_output_modes(64),
                (LogitsOutputMode.FULL_VOCAB,),
            )
        width.assert_not_called()


class TestMaterializeProxy(CustomTestCase):
    def _torch_tpu_stub(self):
        tt = ModuleType("torch_tpu")
        tt._internal = ModuleType("torch_tpu._internal")
        tt._internal.sync = ModuleType("torch_tpu._internal.sync")
        tt._internal.sync.sync = ModuleType("torch_tpu._internal.sync.sync")
        tt._internal.sync.sync.synchronize = MagicMock()
        return {
            "torch_tpu": tt,
            "torch_tpu._internal": tt._internal,
            "torch_tpu._internal.sync": tt._internal.sync,
            "torch_tpu._internal.sync.sync": tt._internal.sync.sync,
        }

    def test_proxy_tensors_are_synchronized(self):
        hidden = torch.zeros(4, _HIDDEN)
        residual = torch.ones(4, _HIDDEN)
        with (
            patch.dict(sys.modules, self._torch_tpu_stub()),
            patch("torch_tpu._internal.sync.sync.synchronize") as sync,
        ):
            _materialize_out(
                PPProxyTensors({"hidden_states": hidden, "residual": residual}),
                SimpleNamespace(requires_kv_sync_after_compile=False),
                SimpleNamespace(),
            )
        sync.assert_called_once()
        tensors, kwargs = sync.call_args.args[0], sync.call_args.kwargs
        self.assertEqual([id(t) for t in tensors], [id(hidden), id(residual)])
        self.assertTrue(kwargs["wait"])

    def test_empty_proxy_synchronizes_nothing(self):
        # No torch_tpu stub: an import attempt would fail on the CPU runner.
        _materialize_out(
            PPProxyTensors({}),
            SimpleNamespace(requires_kv_sync_after_compile=False),
            SimpleNamespace(),
        )


class TestProxyPytree(CustomTestCase):
    def test_flatten_and_unflatten_round_trip_in_key_order(self):
        hidden = torch.zeros(2, 3)
        residual = torch.ones(2, 3)
        proxy = PPProxyTensors({"hidden_states": hidden, "residual": residual})

        children, aux = flatten_pp_proxy_with_keys(proxy, lambda k: ("key", k))

        self.assertEqual(aux, ("hidden_states", "residual"))
        self.assertEqual(
            children,
            [(("key", "hidden_states"), hidden), (("key", "residual"), residual)],
        )
        rebuilt = unflatten_pp_proxy(aux, [child for _, child in children])
        self.assertIsInstance(rebuilt, PPProxyTensors)
        self.assertIs(rebuilt["hidden_states"], hidden)
        self.assertIs(rebuilt["residual"], residual)
        self.assertEqual(list(rebuilt.tensors), ["hidden_states", "residual"])

    def test_registration_uses_dict_keys_on_the_given_tree_util(self):
        tree_util = SimpleNamespace(
            register_pytree_with_keys=MagicMock(),
            DictKey=lambda key: ("DictKey", key),
        )
        register_pp_proxy_pytree(tree_util)

        tree_util.register_pytree_with_keys.assert_called_once()
        node_type, flatten, unflatten = (
            tree_util.register_pytree_with_keys.call_args.args
        )
        self.assertIs(node_type, PPProxyTensors)
        self.assertIs(unflatten, unflatten_pp_proxy)
        tensor = torch.zeros(1)
        children, aux = flatten(PPProxyTensors({"hidden_states": tensor}))
        self.assertEqual(children, [(("DictKey", "hidden_states"), tensor)])
        self.assertEqual(aux, ("hidden_states",))


class _ForwardMode:
    def is_extend(self):
        return False

    def is_target_verify(self):
        return False


def _forward_batch(rows: int):
    return SimpleNamespace(
        is_prefill_only=False,
        input_ids=torch.zeros(rows, dtype=torch.int64),
        input_embeds=None,
        mm_input_embeds=None,
        seq_lens=torch.zeros(4, dtype=torch.int32),
        spec_info=None,
        forward_mode=_ForwardMode(),
        return_logprob=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_input_logprob_token_ids_gpu=None,
        extend_seq_lens_cpu=None,
        extend_logprob_start_lens_cpu=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )


def _proxy(rows: int):
    return PPProxyTensors(
        {
            "hidden_states": torch.zeros(rows, _HIDDEN),
            "residual": torch.zeros(rows, _HIDDEN),
        }
    )


def _shim(pp_group=None):
    def forward(input_ids, positions, forward_batch, pp_proxy_tensors=None):
        pass

    compiled = MagicMock(return_value="out")
    compiled.__signature__ = inspect.signature(forward)
    policy = DecoderTpuCompilePolicy()
    runner = SimpleNamespace(
        model=SimpleNamespace(),
        token_to_kv_pool=SimpleNamespace(layer_transfer_counter=None),
        server_args=SimpleNamespace(
            pp_size=8, dp_size=1, attn_dp_size=1, enable_dp_attention=False
        ),
        pp_group=pp_group,
    )
    policy.model_runner = runner
    return CompiledForwardShim(compiled, policy=policy), compiled


class TestShimProxyCheck(CustomTestCase):
    def test_short_proxy_names_the_tensor_and_both_row_counts(self):
        with self.assertRaisesRegex(
            ValueError, r"'residual'.*32 rows.*64 input tokens"
        ):
            _fit_pp_proxy_rows(
                PPProxyTensors(
                    {
                        "hidden_states": torch.zeros(64, 2),
                        "residual": torch.zeros(32, 2),
                    }
                ),
                torch.zeros(64),
                None,
            )

    def test_matching_rows_pass_and_none_input_ids_fall_back_to_the_batch(self):
        fitted = _fit_pp_proxy_rows(_proxy(8), None, _forward_batch(8))
        self.assertEqual(fitted.tensors["hidden_states"].shape[0], 8)
        with self.assertRaises(ValueError):
            _fit_pp_proxy_rows(_proxy(8), None, _forward_batch(16))

    def test_longer_proxy_is_trimmed_to_the_padded_batch(self):
        """The ICI wave delivers activations at the largest token bucket; a
        stage running a smaller bucket keeps the leading rows."""
        proxy = PPProxyTensors(
            {
                "hidden_states": torch.arange(64.0).reshape(64, 1),
                "residual": torch.arange(64.0).reshape(64, 1) + 100,
            }
        )
        fitted = _fit_pp_proxy_rows(proxy, torch.zeros(16), None)
        self.assertEqual(fitted.tensors["hidden_states"].shape, (16, 1))
        self.assertEqual(float(fitted.tensors["residual"][15, 0]), 115.0)

    def test_positional_proxy_is_trimmed_in_place_and_other_args_kept(self):
        """glm4_moe and friends pass the proxy positionally after input_embeds."""
        proxy = PPProxyTensors(
            {
                "hidden_states": torch.arange(64.0).reshape(64, 1),
                "residual": torch.arange(64.0).reshape(64, 1) + 100,
            }
        )
        args = (None, proxy, "tail")
        fitted = _fit_pp_proxy_in_args(args, torch.zeros(16), None)
        self.assertEqual(fitted[0], None)
        self.assertEqual(fitted[2], "tail")
        self.assertEqual(fitted[1].tensors["residual"].shape, (16, 1))
        self.assertEqual(float(fitted[1].tensors["residual"][15, 0]), 115.0)
        matching = (None, _proxy(8))
        self.assertIs(_fit_pp_proxy_in_args(matching, torch.zeros(8), None), matching)
        no_proxy = (None,)
        self.assertIs(_fit_pp_proxy_in_args(no_proxy, torch.zeros(8), None), no_proxy)

    def test_shim_trims_a_positional_proxy_before_compiling(self):
        shim, compiled = _shim(_pp_group(first=False, last=False))
        fb = _forward_batch(16)
        self.assertEqual(shim(fb.input_ids, None, fb, None, _proxy(64)), "out")
        passed = compiled.call_args.args[4]
        self.assertEqual(passed.tensors["hidden_states"].shape[0], 16)

    def test_shim_passes_a_matching_proxy_to_the_compiled_forward(self):
        shim, compiled = _shim(_pp_group(first=False, last=False))
        fb = _forward_batch(64)
        proxy = _proxy(64)
        self.assertEqual(shim(fb.input_ids, None, fb, pp_proxy_tensors=proxy), "out")
        self.assertIs(compiled.call_args.kwargs["pp_proxy_tensors"], proxy)

    def test_shim_rejects_a_proxy_from_another_bucket_before_compiling(self):
        shim, compiled = _shim(_pp_group(first=False, last=False))
        fb = _forward_batch(64)
        with self.assertRaisesRegex(ValueError, "different token buckets"):
            shim(fb.input_ids, None, fb, pp_proxy_tensors=_proxy(32))
        compiled.assert_not_called()

    def test_first_stage_drops_a_proxy(self):
        shim, compiled = _shim(_pp_group(first=True, last=False))
        fb = _forward_batch(64)
        shim(fb.input_ids, None, fb, pp_proxy_tensors=_proxy(32))
        self.assertNotIn("pp_proxy_tensors", compiled.call_args.kwargs)

    def test_on_a_single_stage_the_proxy_is_checked_and_forwarded(self):
        shim, compiled = _shim()
        fb = _forward_batch(64)
        proxy = _proxy(64)
        shim(fb.input_ids, None, fb, pp_proxy_tensors=proxy)
        self.assertIs(compiled.call_args.kwargs["pp_proxy_tensors"], proxy)

    def test_default_none_proxy_is_still_dropped(self):
        shim, compiled = _shim(_pp_group(first=False, last=False))
        fb = _forward_batch(64)
        shim(fb.input_ids, None, fb, pp_proxy_tensors=None)
        self.assertNotIn("pp_proxy_tensors", compiled.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
