"""CPU guards for the two-pass draft prompt pass of a DSA NextN draft.

Pass 1 stores the NextN layer's draft KV for every prompt row and computes
nothing else; pass 2 runs the draft prompt program on each request's last row.
These cases pin what the two passes must agree on with the single pass, and the
gate, buckets and warmups that decide when they run.
"""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.deepseek_nextn import (
    DeepseekModelNextN,
    DeepseekV3ForCausalLMNextN,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.compilation.bucketer import PowerBucketer
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.experimental.compilation import chain_compile_policy
from sglang_tpu.experimental.compilation.chain_compile_policy import (
    TpuChainSpecDecodeCompilePolicyMixin,
)
from sglang_tpu.experimental.speculative import chain_spec_worker
from sglang_tpu.experimental.speculative.chain_draft_input import (
    chain_draft_forward_input,
)
from sglang_tpu.experimental.speculative.chain_spec_worker import TpuChainSpecWorker
from sglang_tpu.experimental.speculative.mtp import kv_only_prompt
from sglang_tpu.experimental.speculative.mtp.kv_only_prompt import (
    forward_kv_only_backbone,
    kv_only_prompt_plan,
    kv_only_prompt_unsupported_reason,
    last_row_view,
    nextn_layer_input,
)

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_HIDDEN, _VOCAB = 16, 32


class _Captured(Exception):
    pass


class _CapturingDecoder(nn.Module):
    """Records the decoder layer's input, then stops the NextN forward."""

    def forward(self, positions, hidden_states, *args, **kwargs):
        self.seen = hidden_states
        raise _Captured


def _nextn(*, rot_weight: bool) -> DeepseekModelNextN:
    nextn = DeepseekModelNextN.__new__(DeepseekModelNextN)
    nn.Module.__init__(nextn)
    nextn.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
    nextn.enorm = RMSNorm(_HIDDEN)
    nextn.hnorm = RMSNorm(_HIDDEN)
    nextn.hnorm.weight.data.uniform_(0.5, 1.5)
    nextn.eh_proj = nn.Linear(2 * _HIDDEN, _HIDDEN, bias=False)
    nextn.rot_weight = torch.randn(_HIDDEN, _HIDDEN) if rot_weight else None
    nextn.quant_config = None
    nextn.decoder = _CapturingDecoder()
    return nextn


def _prompt_forward_batch(num_tokens: int):
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        spec_info=SimpleNamespace(hidden_states=torch.randn(num_tokens, _HIDDEN)),
        mm_input_embeds=None,
        contains_mm_inputs=lambda: False,
        reuse_dsa_topk_indices=False,
        residual_stream=None,
    )


class TestPassOneLayerInput(CustomTestCase):
    def test_layer_input_is_what_the_nextn_decoder_receives(self):
        """Pass 1 rebuilds the NextN layer input itself; it must equal what
        DeepseekModelNextN.forward hands its decoder, or pass 1 stores KV the
        full layer would not."""
        for rot_weight in (False, True):
            for with_embeds in (False, True):
                with self.subTest(rot_weight=rot_weight, with_embeds=with_embeds):
                    torch.manual_seed(0)
                    nextn = _nextn(rot_weight=rot_weight)
                    input_ids = torch.tensor([3, 1, 4, 1, 5, 9])
                    forward_batch = _prompt_forward_batch(6)
                    embeds = torch.randn(6, _HIDDEN) if with_embeds else None
                    with torch.no_grad(), self.assertRaises(_Captured):
                        DeepseekModelNextN.forward(
                            nextn,
                            input_ids,
                            torch.arange(6),
                            forward_batch,
                            input_embeds=embeds,
                        )
                    with torch.no_grad():
                        got = nextn_layer_input(
                            nextn,
                            input_ids=input_ids,
                            forward_batch=forward_batch,
                            input_embeds=embeds,
                        )
                    self.assertTrue(torch.equal(got, nextn.decoder.seen))

    def test_pass_one_stores_from_the_decoder_boundary(self):
        """The attention input comes from the layer's own boundary (norm, mHC
        streams, input-scattered gather) and the latent is fetched after it."""
        nextn = _nextn(rot_weight=False)
        calls = []
        attn_input = torch.randn(4, _HIDDEN)

        def prepare(hidden_states, forward_batch):
            calls.append("prepare")
            self.assertIsNotNone(forward_batch.residual_stream)
            return attn_input

        decoder = nn.Module()
        decoder.attn_boundary = SimpleNamespace(prepare=prepare)
        decoder.self_attn = "attn"
        nextn.decoder = decoder
        attn_tp_context = SimpleNamespace(
            fetch_qkv_latent=lambda: calls.append("fetch") or "qkv_latent",
            clear_attn_inputs=lambda: calls.append("clear"),
        )
        written = {}

        def write(attn, **kwargs):
            calls.append("write")
            written.update(kwargs, attn=attn)

        forward_batch = _prompt_forward_batch(4)
        positions = torch.arange(4)
        with (
            patch.object(kv_only_prompt, "write_dsa_layer_kv", write),
            patch.object(
                kv_only_prompt, "get_attn_tp_context", return_value=attn_tp_context
            ),
            patch.object(kv_only_prompt, "get_attn_backend", return_value="backend"),
            torch.no_grad(),
        ):
            out = forward_kv_only_backbone(
                nextn, torch.tensor([1, 2, 3, 4]), positions, forward_batch
            )
        self.assertEqual(calls, ["prepare", "fetch", "write", "clear"])
        self.assertEqual(written["attn"], "attn")
        self.assertEqual(written["backend"], "backend")
        self.assertEqual(written["qkv_latent"], "qkv_latent")
        self.assertIs(written["hidden_states"], attn_input)
        self.assertIs(written["positions"], positions)
        self.assertIsNone(forward_batch.residual_stream)
        # Every row, so the draft's logits step can pick the last rows as usual.
        self.assertEqual(tuple(out.shape), (4, _HIDDEN))


def _draft_model():
    model = DeepseekV3ForCausalLMNextN.__new__(DeepseekV3ForCausalLMNextN)
    nn.Module.__init__(model)
    model.model = DeepseekModelNextN.__new__(DeepseekModelNextN)
    nn.Module.__init__(model.model)
    model.model.decoder = nn.Module()
    model.model.decoder.self_attn = DeepseekV2AttentionMLA.__new__(
        DeepseekV2AttentionMLA
    )
    return model


class TestGate(CustomTestCase):
    def test_keeps_the_single_pass_wherever_two_passes_are_not_covered(self):
        """None only with the switch on, a DSA NextN draft the cache-only store
        serves, without DP attention, on TPU7x; otherwise a reason naming the
        path, which the drafter logs at startup."""
        layout = dict(attn_dp_enabled=False)
        cases = [
            ({}, {}, None),
            (dict(switch=False), {}, "KV_ONLY_PROMPT_ENABLED is False"),
            (dict(model=SimpleNamespace()), {}, "not a DeepSeek NextN layer"),
            ({}, dict(attn_dp_enabled=True), "DP attention"),
            (dict(disagg="prefill"), {}, "P/D disaggregation"),
            (dict(token_split=True), {}, "TOKEN_SPLIT"),
            (dict(store="decode context parallelism is on"), {}, "decode context"),
            (dict(chip="TPU v6 lite"), {}, "not TPU v6 lite"),
        ]
        for change, layout_change, expected in cases:
            with (
                self.subTest(change=change, layout=layout_change),
                ExitStack() as stack,
            ):
                stack.enter_context(
                    patch.object(
                        kv_only_prompt,
                        "KV_ONLY_PROMPT_ENABLED",
                        change.get("switch", True),
                    )
                )
                stack.enter_context(
                    tpu_envs.SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT.override(
                        change.get("token_split", False)
                    )
                )
                stack.enter_context(
                    get_parallel().override(**{**layout, **layout_change})
                )
                stack.enter_context(
                    patch(
                        "sglang.srt.runtime_context.get_disagg",
                        return_value=SimpleNamespace(
                            disaggregation_mode=change.get("disagg", "null")
                        ),
                    )
                )
                stack.enter_context(
                    patch.object(
                        kv_only_prompt,
                        "dsa_layer_kv_unsupported_reason",
                        return_value=change.get("store"),
                    )
                )
                stack.enter_context(
                    patch(
                        "sglang_tpu.pallas_export.get_pallas_tpu_device_kind",
                        return_value=change.get("chip", "TPU7x"),
                    )
                )
                reason = kv_only_prompt_unsupported_reason(
                    draft_model=change.get("model", _draft_model()),
                    attn_backend="backend",
                )
                if expected is None:
                    self.assertIsNone(reason)
                else:
                    self.assertIn(expected, reason)


def _policy(*, buckets: list[int], cap: int):
    policy = SimpleNamespace(
        t_bucketer=SimpleNamespace(
            buckets=buckets,
            next_bucket=lambda n: next(b for b in buckets if b >= n),
        ),
        _extend_compile_cap=lambda: cap,
    )
    policy.kv_only_last_row_tokens = (
        TpuChainSpecDecodeCompilePolicyMixin.kv_only_last_row_tokens.__get__(policy)
    )
    return policy


def _plan_tokens(policy) -> list[int]:
    with patch.object(
        kv_only_prompt, "kv_only_prompt_unsupported_reason", return_value=None
    ):
        plan = kv_only_prompt_plan(
            draft_model=_draft_model(),
            attn_backend="backend",
            buckets=policy.t_bucketer.buckets,
            last_row_tokens=policy.kv_only_last_row_tokens(),
        )
    return sorted(plan.tokens) if plan is not None else []


class TestBuckets(CustomTestCase):
    def test_two_passes_only_where_pass_two_costs_less_than_it_saves(self):
        """Pass 2 runs a full draft program at the extend cap's bucket, so a
        bucket takes two passes only at twice that and at the 1024 floor."""
        # GLM-5.3-Flash's buckets: pass 2 pads to 1024, so 1024 stays single.
        glm = _policy(buckets=[1024, 2048, 4096, 8192], cap=64)
        self.assertEqual(glm.kv_only_last_row_tokens(), 1024)
        self.assertEqual(_plan_tokens(glm), [2048, 4096, 8192])
        # Default power-of-two buckets from 64: the floor decides.
        default = _policy(buckets=PowerBucketer("tokens", 64, 8192).buckets, cap=64)
        self.assertEqual(default.kv_only_last_row_tokens(), 64)
        self.assertEqual(_plan_tokens(default), [1024, 2048, 4096, 8192])
        # One bucket below twice pass 2's: no plan, the single pass stays.
        self.assertEqual(_plan_tokens(_policy(buckets=[8192], cap=8192)), [])

    def test_every_bucket_the_gate_takes_is_warmed(self):
        """No compile while serving: pass 1's program and pass 2's gather at each
        two-pass bucket, and pass 2's program among the warmed prompt extends."""
        policy = _policy(buckets=[1024, 2048, 4096, 8192], cap=64)
        tokens = _plan_tokens(policy)
        policy._target_is_multimodal = True
        policy.algo_label = "MTP"
        policy._dummy_hidden = lambda rows, width: torch.zeros(rows, width)
        warmed = []
        policy._run_one = lambda **kw: warmed.append(
            (kw["num_tokens"], kw["variant"], kw["spec_info"].kv_only)
        )
        policy._precompile_shapes = lambda label, shapes, run: [run(s) for s in shapes]
        TpuChainSpecDecodeCompilePolicyMixin._precompile_kv_only_prompt(
            policy, _HIDDEN, frozenset(tokens)
        )
        self.assertEqual(
            warmed,
            [(t, v, True) for t in tokens for v in ("plain", "input_embeds")],
        )
        # _precompile_draft_prompt_extend warms every bucket at or above the cap.
        self.assertGreaterEqual(policy.kv_only_last_row_tokens(), 64)

        worker = object.__new__(TpuChainSpecWorker)
        worker.device = "cpu"
        worker._draft_runner = SimpleNamespace(dtype=torch.bfloat16)
        worker._chain_draft_worker = SimpleNamespace(
            kv_only_prompt_tokens=frozenset(tokens),
            kv_only_last_row_tokens=policy.kv_only_last_row_tokens(),
            target_feature_size=_HIDDEN,
        )
        gathered = []
        with patch.object(
            chain_spec_worker,
            "warmup_compiled_op",
            lambda op, ids, hidden, rows: gathered.append(
                (ids.shape[0], tuple(hidden.shape), rows.shape[0])
            ),
        ):
            worker._warmup_last_rows_gather()
        self.assertEqual(gathered, [(t, (t, _HIDDEN), 1024) for t in tokens])
        # Off: nothing extra is warmed.
        worker._chain_draft_worker.kv_only_prompt_tokens = frozenset()
        gathered.clear()
        with patch.object(chain_spec_worker, "warmup_compiled_op", gathered.append):
            worker._warmup_last_rows_gather()
        self.assertEqual(gathered, [])


class TestBackboneDispatch(CustomTestCase):
    def test_kv_only_forwards_reach_the_drafters_backbone_under_either_tracer(self):
        """The policy wraps the draft backbone before Dynamo or the direct tracer
        compiles it: a kv_only forward runs the drafter's pass-1 backbone, any
        other forward the model's own, with the same arguments."""
        calls = []
        policy = object.__new__(TpuChainSpecDecodeCompilePolicyMixin)
        policy._kv_only_backbone = None
        model_backbone = lambda *a, **kw: calls.append(("model", a[2], kw)) or "model"
        dispatch = policy._dispatch_kv_only(model_backbone)
        hidden = torch.zeros(4, _HIDDEN)

        def batch(kv_only):
            return SimpleNamespace(
                spec_info=chain_draft_forward_input(
                    hidden_states=hidden, width=1, kv_only=kv_only
                )
            )

        plain, kv_only = batch(False), batch(True)
        self.assertEqual(dispatch("ids", "pos", plain, input_embeds=None), "model")
        with self.assertRaisesRegex(AssertionError, "no KV-only backbone"):
            dispatch("ids", "pos", kv_only)
        policy.set_kv_only_backbone(
            lambda *a, **kw: calls.append(("kv_only", a[2], kw)) or "kv_only"
        )
        self.assertEqual(dispatch("ids", "pos", kv_only, input_embeds="e"), "kv_only")
        self.assertEqual(dispatch("ids", "pos", plain), "model")
        self.assertEqual(
            [(name, fb.spec_info.kv_only, kw) for name, fb, kw in calls],
            [
                ("model", False, {"input_embeds": None}),
                ("kv_only", True, {"input_embeds": "e"}),
                ("model", False, {}),
            ],
        )

    def test_dynamo_specializes_on_kv_only_without_recompiling_when_serving(self):
        """Under Dynamo the kv_only flag is a static guard: each (bucket, kv_only)
        pair warms its own graph, and serving both afterwards compiles nothing."""
        weight = torch.randn(_HIDDEN, _HIDDEN)
        policy = object.__new__(TpuChainSpecDecodeCompilePolicyMixin)
        policy._kv_only_backbone = None
        dispatch = policy._dispatch_kv_only(
            lambda ids, pos, fb: fb.spec_info.hidden_states @ weight
        )
        policy.set_kv_only_backbone(lambda ids, pos, fb: fb.spec_info.hidden_states * 2)
        compiled = torch.compile(
            dispatch, backend="eager", fullgraph=True, dynamic=False
        )

        def run(tokens, kv_only):
            hidden = torch.randn(tokens, _HIDDEN)
            fb = SimpleNamespace(
                spec_info=chain_draft_forward_input(
                    hidden_states=hidden, width=1, kv_only=kv_only
                )
            )
            want = hidden * 2 if kv_only else hidden @ weight
            torch.testing.assert_close(compiled("ids", "pos", fb), want)

        torch._dynamo.reset()
        for tokens in (4, 8):
            for kv_only in (False, True):
                run(tokens, kv_only)
        torch.compiler.set_stance("fail_on_recompile")
        try:
            for tokens in (8, 4):
                for kv_only in (True, False):
                    run(tokens, kv_only)
        finally:
            torch.compiler.set_stance("default")
            torch._dynamo.reset()

    def test_install_wraps_the_draft_backbone_before_compiling_it(self):
        """Both tracers compile owner.model.forward after install has wrapped it,
        so pass 1 gets its own program under either."""
        backbone = SimpleNamespace(forward=lambda *a, **kw: "model")
        owner = SimpleNamespace(
            model=backbone,
            forward=lambda *a, **kw: "outer",
            logits_processor=SimpleNamespace(forward=lambda *a, **kw: "logits"),
        )
        compiled_from = []

        def parent_install(self):
            compiled_from.append(self.owner.model.forward)
            return False

        policy = object.__new__(TpuChainSpecDecodeCompilePolicyMixin)
        policy._kv_only_backbone = None
        policy.owner = owner
        policy.is_draft_worker = True
        with patch.object(
            chain_compile_policy.TpuSpecDecodeCompilePolicyMixin,
            "install",
            parent_install,
        ):
            TpuChainSpecDecodeCompilePolicyMixin.install(policy)
        (wrapped,) = compiled_from
        self.assertIs(policy._eager_backbone_forward, wrapped)
        self.assertEqual(wrapped.__wrapped__(), "model")
        policy.set_kv_only_backbone(lambda *a, **kw: "kv_only")
        kv_only = SimpleNamespace(
            spec_info=chain_draft_forward_input(
                hidden_states=torch.zeros(2, _HIDDEN), width=1, kv_only=True
            )
        )
        self.assertEqual(wrapped("ids", "pos", kv_only), "kv_only")


def _prompt_batch():
    # Request 0 resumes at 5 cached tokens; request 1 is a fresh 3-token prompt.
    return SimpleNamespace(
        extend_lens=[4, 3],
        seq_lens_cpu=torch.tensor([9, 3]),
        out_cache_loc=torch.tensor([50, 51, 52, 53, 70, 71, 72], dtype=torch.int32),
        input_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
        prefix_lens=[5, 0],
        extend_num_tokens=7,
        multimodal_inputs=["image", None],
        req_pool_indices_cpu=torch.tensor([3, 8]),
        return_logprob=True,
        spec_info="spec",
        chunked_req=None,
        chunked_req_next_prompt_token=None,
        extend_logprob_start_lens=[4, 3],
    )


class TestPassTwoBatch(CustomTestCase):
    def test_last_row_view_holds_each_requests_last_prompt_row(self):
        """One token per request at its last position over its whole prefix,
        written to that row's slot, as a text token; the batch is left as is."""
        batch = _prompt_batch()
        view = last_row_view(batch)
        self.assertEqual(view.out_cache_loc.tolist(), [53, 72])
        self.assertEqual(view.prefix_lens, [8, 2])
        self.assertEqual(view.extend_lens, [1, 1])
        self.assertEqual(view.extend_num_tokens, 2)
        self.assertEqual(view.multimodal_inputs, [None, None])
        self.assertFalse(view.return_logprob)
        self.assertIsNone(view.spec_info)
        self.assertEqual(view.req_pool_indices_cpu.tolist(), [3, 8])
        self.assertEqual(batch.extend_lens, [4, 3])
        self.assertEqual(batch.out_cache_loc.shape[0], 7)
        self.assertEqual(batch.multimodal_inputs, ["image", None])


class TestExtendDraft(CustomTestCase):
    def _run(self, *, padded_tokens: int):
        cap, two_pass_tokens = 4, frozenset({16})
        worker = object.__new__(TpuChainSpecWorker)
        worker.device = "cpu"
        worker._kv_only_buckets_seen = set()
        forwarded = []

        def forward(forward_batch):
            spec_info = forward_batch.spec_info
            forwarded.append(
                dict(
                    kv_only=spec_info.kv_only,
                    input_ids=forward_batch.input_ids.tolist(),
                    hidden=spec_info.hidden_states[:, 0].tolist(),
                    batch=forward_batch.batch,
                )
            )
            hidden = torch.full((cap, _HIDDEN), float(len(forwarded)))
            return SimpleNamespace(
                logits_output=LogitsProcessorOutput(
                    next_token_logits=torch.zeros(cap, _VOCAB), hidden_states=hidden
                )
            )

        worker._draft_runner = SimpleNamespace(forward=forward)
        worker._chain_draft_worker = SimpleNamespace(
            draft_context=ExitStack,
            dsa_seed_topk_width=None,
            dsa_seed_topk_table=None,
            carry=torch.zeros(10, _HIDDEN),
            kv_only_prompt_tokens=two_pass_tokens,
            kv_only_last_row_tokens=8,
        )
        worker._compile_policy = lambda: SimpleNamespace(
            _extend_compile_cap=lambda: cap
        )
        draft_ids = torch.arange(100, 100 + padded_tokens)
        worker._draft_prompt_ids = lambda batch, **kw: draft_ids
        target_hidden = torch.arange(padded_tokens, dtype=torch.float32)[:, None]
        target_hidden = target_hidden.expand(padded_tokens, _HIDDEN)
        batch = _prompt_batch()
        with (
            patch.object(
                chain_spec_worker.ForwardBatch,
                "init_new",
                side_effect=lambda b, *a, **kw: SimpleNamespace(
                    batch=b, global_num_tokens_cpu=None
                ),
            ),
            patch.object(
                chain_spec_worker,
                "gather_last_rows",
                lambda ids, hidden, rows: (ids[rows], hidden[rows]),
            ),
        ):
            tail_hidden, _ = worker._extend_draft(
                batch,
                target_hidden=target_hidden,
                bucket_tokens=torch.zeros(cap, dtype=torch.int64),
                mm_input_embeds=None,
            )
        return batch, forwarded, tail_hidden, worker._chain_draft_worker.carry

    def test_a_bucket_off_the_list_keeps_the_single_pass(self):
        batch, forwarded, tail_hidden, carry = self._run(padded_tokens=8)
        (single,) = forwarded
        self.assertFalse(single["kv_only"])
        self.assertIs(single["batch"], batch)
        self.assertEqual(single["input_ids"], list(range(100, 108)))
        self.assertTrue(torch.equal(carry[3], tail_hidden[0]))

    def test_a_listed_bucket_runs_pass_one_then_the_last_rows(self):
        """Pass 1 over every padded row, then pass 2 on rows 3 and 6 (the last
        prompt rows) at pass 2's bucket; the carry comes from pass 2."""
        with self.assertLogs(chain_spec_worker.logger, "INFO") as logs:
            batch, forwarded, tail_hidden, carry = self._run(padded_tokens=16)
        # The test greps this once per bucket to see the branch was taken.
        self.assertIn("took two passes at token bucket 16", logs.output[0])
        first, second = forwarded
        self.assertTrue(first["kv_only"])
        self.assertIs(first["batch"], batch)
        self.assertEqual(first["input_ids"], list(range(100, 116)))
        self.assertFalse(second["kv_only"])
        self.assertEqual(second["batch"].extend_lens, [1, 1])
        # Padded rows gather row 0; their tokens belong to no request.
        self.assertEqual(second["input_ids"], [103, 106] + [100] * 6)
        self.assertEqual(second["hidden"], [3.0, 6.0] + [0.0] * 6)
        self.assertTrue(torch.equal(tail_hidden, torch.full((4, _HIDDEN), 2.0)))
        self.assertTrue(torch.equal(carry[8], tail_hidden[1]))


if __name__ == "__main__":
    unittest.main()
