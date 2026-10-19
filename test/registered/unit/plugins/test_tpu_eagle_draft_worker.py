"""Tests for the TPU plugin's EAGLE draft worker: per-step cache locations and
the fused draft loop it owns instead of the shared EagleDraftWorker."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.experimental.speculative.eagle.draft_worker import TpuEagleDraftWorker

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _fake_server_args(**fields):
    """server_args stand-in: carries fields and the override() entry point."""
    ns = SimpleNamespace(**fields)

    def _override(source, **updates):
        for key, value in updates.items():
            setattr(ns, key, value)

    ns.override = _override
    return ns


def _make_worker(
    num_steps: int, num_draft_tokens: int, *, worker_cls=TpuEagleDraftWorker
):
    worker = object.__new__(worker_cls)
    worker.topk = 1
    worker.device = DEVICE
    worker.speculative_num_steps = num_steps
    worker.speculative_num_draft_tokens = num_draft_tokens
    worker.server_args = _fake_server_args(
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(max_bs=8)),
        max_running_requests=8,
    )
    return worker


class TestTpuEagleDraftPerStepCacheLoc(CustomTestCase):
    def setUp(self):
        # _rebuild_topk1_chain_buffers sizes its preallocation from the
        # published config: get_exec().graph.cuda_graph_config stays None on
        # the dummy-boundary publish (no resolution), so
        # get_schedule().max_running_requests alone sizes the buffers.
        override = get_context().override_server_args(max_running_requests=8)
        override.install()
        self.addCleanup(override.restore)

    def test_tpu_draft_owns_per_step_cache_loc(self):
        worker = _make_worker(
            num_steps=3, num_draft_tokens=4, worker_cls=TpuEagleDraftWorker
        )
        worker.hot_token_id = None
        worker.index_share_for_mtp_iteration = False
        worker.seed_dsa_topk_from_draft_extend = False
        worker.draft_attn_backend = SimpleNamespace(attn_backends=[None, None])
        worker._rebuild_topk1_chain_buffers()

        observed_cache_locs = []

        def draft_forward(forward_batch):
            observed_cache_locs.append(forward_batch.out_cache_loc)
            return SimpleNamespace(
                logits_output=SimpleNamespace(
                    next_token_logits=torch.zeros((1, 2), device=DEVICE),
                    hidden_states=forward_batch.spec_info.hidden_states,
                )
            )

        worker.draft_runner = SimpleNamespace(
            canary_manager=None,
            forward=draft_forward,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["GptOssForCausalLM"]),
                model_is_mrope=False,
            ),
        )
        forward_batch = SimpleNamespace(
            # an idle draft decode batch routes to _draft_forward_idle before touching KV.
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            input_ids=None,
            out_cache_loc=torch.arange(3, device=DEVICE),
            positions=torch.zeros(1, dtype=torch.long, device=DEVICE),
            spec_info=SimpleNamespace(
                topk_p=torch.ones((1, 1), device=DEVICE),
                topk_index=torch.zeros((1, 1), dtype=torch.long, device=DEVICE),
                hidden_states=torch.zeros((1, 2), device=DEVICE),
            ),
        )

        worker.draft_forward(forward_batch)

        self.assertEqual([loc.item() for loc in observed_cache_locs], [0, 1])
        self.assertTrue(all(loc._base is None for loc in observed_cache_locs))


class TestTpuEagleFusedDraft(CustomTestCase):
    def test_fused_steps_support_both_compile_tracers(self):
        worker = _make_worker(
            num_steps=3, num_draft_tokens=4, worker_cls=TpuEagleDraftWorker
        )

        with (
            patch(
                "sglang_tpu.experimental.speculative.eagle.draft_worker.get_flags",
                return_value=SimpleNamespace(
                    capture=SimpleNamespace(enable_torch_compile=True)
                ),
            ),
        ):
            worker.server_args.tpu_compile_tracer = "dynamo"
            self.assertTrue(worker._should_fuse_draft_steps())

            worker.server_args.tpu_compile_tracer = "direct"
            self.assertTrue(worker._should_fuse_draft_steps())

    def test_eager_tpu_does_not_enter_compiled_draft_loop(self):
        """Eager MTP must not call methods missing from NoopCompilePolicy."""
        worker = _make_worker(
            num_steps=3, num_draft_tokens=4, worker_cls=TpuEagleDraftWorker
        )
        with (
            patch(
                "sglang_tpu.experimental.speculative.eagle.draft_worker.get_flags",
                return_value=SimpleNamespace(
                    capture=SimpleNamespace(enable_torch_compile=False)
                ),
            ),
        ):
            self.assertFalse(worker._should_fuse_draft_steps())

    def test_fused_steps_thread_per_step_inputs(self):
        from sglang_tpu.eagle_utils_tpu import (
            build_topk1_verify_chain_torch,
        )

        worker = _make_worker(
            num_steps=3, num_draft_tokens=4, worker_cls=TpuEagleDraftWorker
        )
        worker.hot_token_id = None

        calls = []

        def model_forward(input_ids, positions, forward_batch):
            calls.append(
                (
                    input_ids.clone(),
                    forward_batch.input_ids.clone(),
                    forward_batch.spec_info.rpa_draft_kv_lens.clone(),
                    forward_batch.spec_info.hidden_states.clone(),
                    forward_batch.out_cache_loc.clone(),
                    positions.clone(),
                )
            )
            return SimpleNamespace(
                next_token_logits=torch.tensor([[0.0, 1.0]], device=DEVICE),
                hidden_states=forward_batch.spec_info.hidden_states + 10,
            )

        # The rebind-vs-copy branch reads the draft policy's tracer, so the
        # fixture must carry the policy the fused loop is compiled by.
        worker.draft_runner = SimpleNamespace(
            model=SimpleNamespace(forward=model_forward),
            compile_policy=SimpleNamespace(tracer="direct"),
        )
        forward_batch = SimpleNamespace(
            input_ids=None,
            positions=torch.tensor([5], device=DEVICE),
            out_cache_loc=None,
            spec_info=SimpleNamespace(
                hidden_states=torch.zeros((1, 2), device=DEVICE),
                rpa_draft_kv_lens=None,
            ),
        )
        out_cache_loc = torch.tensor([[20], [21]], device=DEVICE)

        def select_step(i, topk_p, topk_index, hidden_states, scores, topk):
            del topk_p, topk_index, scores, topk
            token = torch.tensor([[100 + i]], device=DEVICE)
            return token, hidden_states + 1, None, (None, token, None)

        with (
            patch(
                "sglang_tpu.experimental.speculative.eagle.draft_worker.select_top_k_tokens",
                side_effect=select_step,
            ),
            patch(
                "sglang_tpu.experimental.speculative.eagle.draft_worker.build_topk1_verify_chain_torch",
                new=build_topk1_verify_chain_torch,
                create=True,
            ),
        ):
            tree = worker._fused_draft_steps(
                forward_batch,
                torch.ones((1, 1), device=DEVICE),
                torch.zeros((1, 1), dtype=torch.long, device=DEVICE),
                torch.zeros((1, 2), device=DEVICE),
                out_cache_loc,
                None,
                torch.tensor([99], dtype=torch.long, device=DEVICE),
                torch.tensor([5], dtype=torch.long, device=DEVICE),
            )

        self.assertEqual(tree[-1].tolist(), [[99, 100, 101, 102]])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0].tolist(), calls[0][1].tolist())
        self.assertEqual(calls[1][0].tolist(), calls[1][1].tolist())
        # KV lengths are derived in-graph as seq_lens + step + 1.
        self.assertEqual(calls[0][2].item(), 6)
        self.assertEqual(calls[1][2].item(), 7)
        self.assertEqual(calls[0][3].tolist(), [[1.0, 1.0]])
        self.assertEqual(calls[1][3].tolist(), [[12.0, 12.0]])
        self.assertEqual(calls[0][4].item(), 20)
        self.assertEqual(calls[1][4].item(), 21)
        self.assertEqual(calls[0][5].item(), 5)
        self.assertEqual(calls[1][5].item(), 6)

    def test_fused_forward_builds_topk1_chain(self):
        worker = _make_worker(
            num_steps=3, num_draft_tokens=4, worker_cls=TpuEagleDraftWorker
        )
        worker._topk1_parents_prealloc = torch.tensor(
            [-1, 0, 1], dtype=torch.long, device=DEVICE
        ).repeat(2, 1)
        worker._topk1_score_indices_prealloc = torch.arange(
            3, dtype=torch.long, device=DEVICE
        ).repeat(2, 1)
        worker.hot_token_id = None
        tokens = torch.cat(
            tuple(
                torch.full((2, 1), i, dtype=torch.long, device=DEVICE) for i in range(3)
            ),
            dim=1,
        )
        observed_prefill_only = []

        def compiled_fused_steps(*args):
            observed_prefill_only.append(args[0].is_prefill_only)
            return (
                torch.empty((0,), dtype=torch.bool, device=DEVICE),
                torch.tensor([[5, 6, 7, 8], [10, 11, 12, 13]], device=DEVICE),
                torch.arange(8, device=DEVICE).view(2, 4),
                torch.tensor([[1, 2, 3, -1]] * 2, device=DEVICE),
                torch.full((2, 4), -1, device=DEVICE),
                torch.cat((torch.tensor([[7], [8]], device=DEVICE), tokens), dim=1),
            )

        reload_gate = MagicMock()
        compile_policy = SimpleNamespace(
            compile_fused_draft_steps=MagicMock(return_value=compiled_fused_steps),
            compiled_forward_shim=reload_gate,
        )
        worker._compiled_fused_steps = None
        worker.draft_runner = SimpleNamespace(compile_policy=compile_policy)
        # The fused loop derives each step's KV lengths in-graph; the backends
        # only need to carry the step id the loop asserts against.
        worker.draft_attn_backend = SimpleNamespace(
            attn_backends=[SimpleNamespace(speculative_step_id=i) for i in range(2)]
        )
        spec_info = SimpleNamespace(
            topk_p=torch.ones((2, 1), device=DEVICE),
            topk_index=torch.zeros((2, 1), dtype=torch.long, device=DEVICE),
            hidden_states=torch.zeros((2, 2), device=DEVICE),
            bonus_tokens=torch.tensor([7, 8], device=DEVICE),
            rpa_draft_kv_lens=torch.tensor([-1], device=DEVICE),
        )
        forward_batch = SimpleNamespace(
            spec_info=spec_info,
            out_cache_loc=torch.arange(6, device=DEVICE),
            batch_size=2,
            seq_lens=torch.tensor([5, 10], device=DEVICE),
            is_prefill_only=True,
        )
        per_step_cache = torch.arange(4, device=DEVICE).reshape(2, 2)

        with (
            patch(
                "sglang_tpu.experimental.speculative.eagle.draft_worker.per_step_draft_out_cache_loc",
                return_value=per_step_cache,
            ),
            patch(
                "sglang_tpu.experimental.speculative.eagle.draft_worker.forward_context",
                return_value=contextlib.nullcontext(),
            ),
        ):
            tree = worker._fused_draft_forward(forward_batch)

        self.assertEqual(tree[-1].tolist(), [[7, 0, 1, 2], [8, 0, 1, 2]])
        self.assertIsNone(spec_info.rpa_draft_kv_lens)
        self.assertEqual(observed_prefill_only, [False])
        self.assertTrue(forward_batch.is_prefill_only)
        reload_gate.wait_for_hicache_reload.assert_called_once_with()
        compile_policy.compile_fused_draft_steps.assert_called_once_with(
            worker._fused_draft_steps
        )

    def test_fused_draft_unpads_verify_tree_to_logical_batch(self):
        worker = _make_worker(
            num_steps=3, num_draft_tokens=4, worker_cls=TpuEagleDraftWorker
        )
        worker.req_to_token_pool = None
        worker.cuda_graph_runner = None
        worker.seed_dsa_topk_from_draft_extend = False
        worker.tree_mask_mode = None
        worker.target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(kv_index_translator=None)
        )
        worker._prepare_draft_forward_batch = MagicMock()

        physical_bs = 16
        logical_bs = 1
        physical_tree = (
            torch.empty((0,), dtype=torch.bool, device=DEVICE),
            torch.arange(physical_bs * 4, device=DEVICE).view(physical_bs, 4),
            torch.arange(physical_bs * 4, device=DEVICE).view(physical_bs, 4),
            torch.tensor([[1, 2, 3, -1]] * physical_bs, device=DEVICE),
            torch.full((physical_bs, 4), -1, device=DEVICE),
            torch.arange(physical_bs * 4, device=DEVICE).view(physical_bs, 4),
        )
        worker._fused_draft_forward = MagicMock(return_value=physical_tree)
        worker.draft_runner = SimpleNamespace(
            canary_manager=None,
            compile_policy=SimpleNamespace(
                prepare_spec_draft_input=lambda *_args: None,
                finalize_spec_forward_batch=lambda *, outputs, forward_batch: outputs,
            ),
        )

        draft_input = SimpleNamespace(
            padding_state=SimpleNamespace(
                physical_batch_size=physical_bs,
                logical_batch_size=logical_bs,
            ),
            dsa_topk_indices=None,
        )
        batch = SimpleNamespace(
            spec_info=draft_input,
            forward_mode=ForwardMode.DECODE,
            seq_lens=torch.ones(logical_bs, dtype=torch.int32, device=DEVICE),
        )
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

        with (
            patch(
                "sglang_tpu.experimental.speculative.eagle.draft_worker.get_flags",
                return_value=SimpleNamespace(
                    capture=SimpleNamespace(enable_torch_compile=True)
                ),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.prepare_for_draft",
                return_value=(forward_batch, False, None),
            ),
        ):
            verify_input = worker.draft(batch)

        self.assertEqual(verify_input.draft_token.shape, (logical_bs * 4,))
        self.assertEqual(verify_input.positions.shape, (logical_bs * 4,))
        self.assertEqual(verify_input.retrieve_index.shape, (logical_bs, 4))
        self.assertEqual(verify_input.draft_token[-1].item(), logical_bs * 4 - 1)


if __name__ == "__main__":
    unittest.main()
