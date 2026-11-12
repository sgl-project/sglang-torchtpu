"""K3_PROGRAM's sglang pool classes over the unified pools: built in sglang's
order (state pool, then KV pool), sized from sglang's counts, and described to
the P/D transfer through HybridLinearKVPool."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.raiden.common import get_raiden_kv_caches
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.hooks import k3_program as hooks
from sglang_tpu.k3_programs import sglang_bridge as bridge
from sglang_tpu.k3_programs import sglang_pools
from sglang_tpu.k3_programs.kimi_k3 import pools
from sglang_tpu.k3_programs.kimi_k3.layout import KDA, MLA, NUM_LAYERS

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

STEP = 8192
BLOCK = 128 * 2 * 640 * 2  # bf16 bytes


def _model(kv_cache_dtype="auto"):
    return mock.patch(
        "sglang.srt.runtime_context.get_model",
        return_value=SimpleNamespace(kv_cache_dtype=kv_cache_dtype),
    )


def _build(slots=4, pages=10, page_size=STEP):
    state = sglang_pools.K3StatePool(size=slots, mamba_layer_ids=KDA, device="cpu")
    kv = sglang_pools.K3UnifiedKVPool(
        size=pages * page_size,
        page_size=page_size,
        dtype=torch.bfloat16,
        layer_num=len(MLA),
        device="cpu",
    )
    return state, kv


class TestUnifiedPoolClasses(CustomTestCase):
    def test_sizes_follow_sglang_counts(self):
        with _model():
            state, kv = _build(slots=4, pages=10)
        self.assertEqual((kv.plan.state_slots, kv.plan.kv_blocks), (4, 10))
        self.assertEqual(kv.size, 10 * STEP)
        self.assertIs(state.plan, kv.plan)
        self.assertEqual(len(kv.raw_pools()), len(MLA) + 5)
        self.assertEqual(
            kv.get_kv_size_bytes(), (len(MLA) + 5) * kv.plan.num_blocks * BLOCK
        )
        with _model():
            self.assertEqual(
                sglang_pools.K3StatePool.cache_bytes_per_request(None),
                3 * (len(MLA) + 5) * BLOCK,
            )
            # Kimi-K3.1: the slot's three rows of the 24 index-key pools too.
            with mock.patch.object(sglang_pools.K3StatePool, "dsa", True):
                self.assertEqual(
                    sglang_pools.K3StatePool.cache_bytes_per_request(None),
                    3
                    * (
                        (len(MLA) + 5) * BLOCK
                        + len(MLA)
                        * bridge.index_block_bytes(sglang_pools.cache_layout("auto"))
                    ),
                )
        # The program binds the pools the KV pool allocated.
        raw, *_ = pools.bind_raw_pools(kv.caches)
        self.assertEqual(len(raw), len(MLA))
        self.assertEqual(len(pools.draft_raw_pools(kv.caches)), 5)

    def test_bytes_per_token(self):
        # 29 pools x one 320 KiB block per 8192 logical tokens on each rank.
        self.assertEqual(sglang_pools.K3UnifiedKVPool.bytes_per_token("auto"), 1160.0)
        self.assertEqual(
            sglang_pools.K3UnifiedKVPool.bytes_per_token("fp8_e4m3"), 580.0
        )
        with _model():
            _, kv = _build(slots=4, pages=10)
        # The unified pools' bytes over their logical tokens (the state blocks aside).
        kv_bytes = kv.get_kv_size_bytes() * kv.plan.kv_blocks / kv.plan.num_blocks
        self.assertAlmostEqual(kv_bytes / kv.size, 1160.0)

    def test_fp8_pages_hold_twice_the_tokens(self):
        with _model("fp8_e4m3"):
            _, kv = _build(page_size=2 * STEP)
        self.assertEqual(kv.plan.logical_block_tokens, 2 * STEP)
        with _model("fp8_e4m3"), self.assertRaisesRegex(ValueError, "16384 tokens"):
            _build(page_size=STEP)

    def test_order_and_shape_checks(self):
        sglang_pools._pending_state_pool = None
        with _model(), self.assertRaisesRegex(RuntimeError, "state pool first"):
            sglang_pools.K3UnifiedKVPool(
                size=STEP,
                page_size=STEP,
                dtype=torch.bfloat16,
                layer_num=24,
                device="cpu",
            )
        with self.assertRaisesRegex(ValueError, "not Kimi-K3's KDA layers"):
            sglang_pools.K3StatePool(size=4, mamba_layer_ids=[0, 1], device="cpu")

    def test_transfer_descriptions_through_the_hybrid_pool(self):
        with _model():
            state, kv = _build()
        hybrid = HybridLinearKVPool(
            size=kv.size,
            dtype=torch.bfloat16,
            page_size=STEP,
            head_num=1,
            head_dim=576,
            full_attention_layer_ids=list(MLA),
            device="cpu",
            mamba_pool=state,
            use_mla=True,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            full_kv_pool=kv,
        )
        kv_ptrs, _, kv_items = hybrid.get_contiguous_buf_infos()
        self.assertEqual(len(kv_ptrs), len(MLA) + 5)
        self.assertEqual(set(kv_items), {BLOCK})
        state_ptrs, _, state_items = hybrid.get_state_buf_infos()
        self.assertEqual(len(state_ptrs), len(KDA))
        self.assertEqual(set(state_items), {3 * BLOCK})
        self.assertEqual(hybrid.get_state_layer_ids(), list(KDA))
        self.assertIsNone(hybrid.maybe_get_custom_mem_pool())
        # Core's ids cover the 24 MLA entries only; the hook covers all 29.
        ids = hooks.k3_program_kv_layer_ids(HybridLinearKVPool.get_kv_layer_ids, hybrid)
        self.assertEqual(len(ids), len(kv_ptrs))
        self.assertEqual(ids[-5:], list(range(NUM_LAYERS, NUM_LAYERS + 5)))
        self.assertEqual(ids[: len(MLA)], list(MLA))


class TestPrefixCache(CustomTestCase):
    """A prefix hit restores a cached slot's KDA state; a tracked chunk's
    state goes to the track slot the radix cache keeps."""

    def setUp(self):
        with _model():
            self.state, self.kv = _build(slots=4, pages=10)
        self.plan = self.kv.plan

    def _mark(self, slot, value):
        for pool in self.state.host_pools():
            for g in range(3):
                pool[self.plan.state_block(slot, g)].fill_(value)

    def _value(self, slot):
        values = {
            float(pool[self.plan.state_block(slot, g)].float().mean())
            for pool in self.state.host_pools()
            for g in range(3)
        }
        self.assertEqual(len(values), 1)
        return values.pop()

    def test_copy_from_moves_every_hosting_pools_blocks(self):
        self.assertEqual(len(self.state.host_pools()), len(KDA) // 3)
        self._mark(2, 7.0)
        self._mark(3, 1.0)
        self.state.copy_from(torch.tensor([2]), torch.tensor([3]))
        self.assertEqual(self._value(3), 7.0)
        self.assertEqual(self._value(2), 7.0)
        # The KV pages are untouched.
        self.assertEqual(
            float(self.kv.caches[bridge.mla_layer_name(MLA[0])][1].float().abs().sum()),
            0.0,
        )

    def test_a_bound_program_copies_blocks(self):
        pairs = []
        sglang_pools.set_block_copier(pairs.extend)
        self.addCleanup(sglang_pools.set_block_copier, None)
        self._mark(2, 7.0)
        self.state.copy_from(torch.tensor([2]), torch.tensor([3]))
        self.kv.copy_blocks([5], [6])
        # Slot 2's three state blocks onto slot 3's, then the page block; no
        # eager copy (slot 3 keeps its value).
        rows = [self.plan.state_block(s, g) for s in (2, 3) for g in range(3)]
        self.assertEqual(pairs, list(zip(rows[:3], rows[3:])) + [(5, 6)])
        self.assertEqual(self._value(3), 0.0)

    def test_prefill_hit_and_track(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang_tpu.k3_programs import sglang_worker as worker
        from sglang_tpu.k3_programs.program import StepOutput

        seen = []

        class Prefill:
            def step(self, step_input):
                seen.append(step_input)
                return StepOutput(torch.full((1, 1), -1, dtype=torch.int32))

        groups = bridge.kv_groups(
            self.plan, 1 << 20, 128, kda_block_size=self.plan.layout.tokens_per_block
        )
        kda_width = groups[1].table_width
        self.assertEqual((groups[1].block_size, kda_width), (256, 4096))
        steps = worker.PrefillSteps(
            Prefill(), self.plan, table_width=128, kda_width=kda_width
        )
        r2t = np.zeros((3, 1 << 20), np.int32)
        for k, page in enumerate((1, 2, 3)):
            r2t[1, k * STEP : (k + 1) * STEP] = np.arange(
                page * STEP, (page + 1) * STEP
            )
        req = SimpleNamespace(
            rid="r",
            origin_input_ids=list(range(3 * STEP + 5)),
            sampling_params=SimpleNamespace(
                temperature=1.0, top_p=1.0, top_k=1, sampling_seed=None
            ),
            kv=SimpleNamespace(req_pool_idx=1, mamba_pool_idx=torch.tensor([1])),
            grammar=None,
        )
        self._mark(2, 5.0)  # the cached node's state after STEP tokens
        self._mark(1, 0.0)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            reqs=[req],
            prefix_lens=[STEP],
            extend_lens=[STEP],
            seq_lens=torch.tensor([2 * STEP]),
            return_logprob=False,
            mamba_cow_src_indices=torch.tensor([2]),
            mamba_cow_dst_indices=torch.tensor([1]),
            mamba_track_mask=torch.tensor([True]),
            mamba_track_indices=torch.tensor([3]),
            mamba_track_seqlens_cpu=[2 * STEP],
        )
        target = SimpleNamespace(
            model_runner=None,
            get_memory_pool=lambda: (
                SimpleNamespace(
                    req_to_token=torch.from_numpy(r2t), mamba_pool=self.state
                ),
                None,
            ),
        )
        worker.K3ProgramWorker.build_steps = lambda t: steps
        self.addCleanup(setattr, worker.K3ProgramWorker, "build_steps", None)
        w = worker.K3ProgramWorker(None, 0, 0, target)
        w.alloc_memory_pool()
        w.forward_batch_generation(batch)
        (row,) = seen[0].requests
        # The hit: computed STEP, its column (STEP // 256 - 1) is the request's
        # slot, which now holds the cached state.
        self.assertEqual(row.computed, STEP)
        self.assertEqual(self._value(1), 5.0)
        column = pools.kda_blocks_at_column(
            row.block_tables,
            [(1 + k % 3, 256) for k in range(len(KDA))],
            STEP // 256 - 1,
        )
        self.assertEqual(
            set(column.tolist()), {self.plan.state_block(1, g) for g in range(3)}
        )
        # The tracked chunk's end state (the slot's, after the step) is in slot 3.
        self.assertEqual(self._value(3), 5.0)
        batch.mamba_track_seqlens_cpu = [STEP + 100]
        with self.assertRaisesRegex(RuntimeError, "chunks must be one page"):
            w.forward_batch_generation(batch)


class TestRaidenGroups(CustomTestCase):
    """Raiden moves rows of root tensors by index: a page's row is its page
    id, a slot's three rows are the blocks the program reads its state from."""

    def setUp(self):
        with _model():
            self.state, self.kv = _build(slots=4, pages=10)
        self.hybrid = HybridLinearKVPool(
            size=self.kv.size,
            dtype=torch.bfloat16,
            page_size=STEP,
            head_num=1,
            head_dim=576,
            full_attention_layer_ids=list(MLA),
            device="cpu",
            mamba_pool=self.state,
            use_mla=True,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            full_kv_pool=self.kv,
        )

    def test_groups(self):
        groups = get_raiden_kv_caches(self.hybrid)
        plan = self.kv.plan
        self.assertEqual(len(groups.full), len(MLA) + 5)
        self.assertEqual(groups.num_draft_full, 5)
        self.assertEqual(
            groups.full_layer_ids[-5:], list(range(NUM_LAYERS, NUM_LAYERS + 5))
        )
        self.assertTrue(all(not t._is_view() for t in groups.full + groups.state))
        self.assertEqual(len(groups.state), len(KDA) // 3)
        self.assertEqual(
            {int(t.shape[0]) for t in groups.full + groups.state}, {plan.num_blocks}
        )
        self.assertEqual(groups.state_max_blocks, 3)
        # The draft pools come after the target's, as Raiden's draft entries do.
        self.assertEqual(
            [t.data_ptr() for t in groups.full[-5:]],
            [t.data_ptr() for t in pools.draft_raw_pools(self.kv.caches)],
        )

    def test_slot_rows_are_the_programs_state_blocks(self):
        groups = get_raiden_kv_caches(self.hybrid)
        manager = RaidenKVManager.__new__(RaidenKVManager)
        manager._state_engine = object()
        manager._state_type = StateType.MAMBA
        manager.kv_args = SimpleNamespace(state_types=[StateType.MAMBA])
        manager._committed_state_row = groups.committed_state_row
        manager._state_engine_max_blocks = groups.state_max_blocks
        plan = self.kv.plan
        for slot in (1, 4):
            rows = manager._state_rows_from_state_indices([[slot]])
            self.assertEqual(rows, [plan.state_block(slot, g) for g in range(3)])
            # KDA layer k reads its state from row state_block(slot, k % 3) of
            # pool k // 3: one of the rows moved, in the pool registered for it.
            tables = bridge.block_tables(
                np.arange(STEP, 2 * STEP)[::STEP], STEP, slot, plan, table_width=4
            )
            groups_kv = bridge.kv_groups(plan, 1 << 20, 4)
            of = {n: g for g, grp in enumerate(groups_kv) for n in grp.layer_names}
            kda = [(of[bridge.kda_layer_name(layer)], 1 << 20) for layer in KDA]
            blocks = pools.kda_blocks_at_length(tables, kda, 5)
            for k, block in enumerate(blocks.tolist()):
                self.assertIn(block, rows)
                self.assertEqual(
                    groups.state[k // 3].data_ptr(),
                    self.kv.caches[bridge.kda_layer_name(KDA[k])].data_ptr(),
                )


class TestConfiguratorHook(CustomTestCase):
    def test_cell_size_is_the_unified_pools(self):
        configurator = SimpleNamespace(_cell_size=27_648)
        with (
            mock.patch.object(hooks, "k3_program_active", return_value=True),
            _model("auto"),
        ):
            hooks.size_k3_program_pools(None, configurator, None)
        self.assertEqual(configurator._cell_size, 1160)
        other = SimpleNamespace(_cell_size=27_648)
        with mock.patch.object(hooks, "k3_program_active", return_value=False):
            hooks.size_k3_program_pools(None, other, None)
        self.assertEqual(other._cell_size, 27_648)
        # Kimi-K3.1: plus the index-key pools (24 x 66 x 4 x 128 bytes a block
        # of 8192 bf16-layout tokens: 99 bytes a token).
        k31 = SimpleNamespace(_cell_size=27_648)
        kvc = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    text_config=SimpleNamespace(dsa_config={"index_topk": 2048})
                )
            )
        )
        with (
            mock.patch.object(hooks, "k3_program_active", return_value=True),
            _model("auto"),
        ):
            hooks.size_k3_program_pools(None, k31, kvc)
        self.assertEqual(k31._cell_size, 1160 + 99)


class TestSpecCheckHook(CustomTestCase):
    def test_only_k3_program_skips_the_check(self):
        calls = []

        def original(server_args):
            calls.append(server_args)

        k3 = SimpleNamespace(speculative_algorithm="K3_PROGRAM")
        dspark = SimpleNamespace(speculative_algorithm="DSPARK")
        with mock.patch(
            "sglang.srt.arg_groups.overrides.resolving_view", side_effect=lambda a: a
        ):
            hooks.allow_k3_program_spec(original, k3)
            hooks.allow_k3_program_spec(original, dspark)
        self.assertEqual(calls, [dspark])


class TestDecodeTreeHook(CustomTestCase):
    def test_stateless_component_on_the_decode_only(self):
        from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
        from sglang_tpu.k3_programs.sglang_decode_cache import K3DecodeStateComponent

        def build(mode, radix=True, active=True):
            ctx = SimpleNamespace(
                params=SimpleNamespace(component_registry_override=None),
                disable_radix_cache=False,
            )
            disagg = SimpleNamespace(
                disaggregation_mode=mode,
                disaggregation_decode_enable_radix_cache=radix,
            )
            with (
                mock.patch.object(hooks, "k3_program_active", return_value=active),
                mock.patch(
                    "sglang.srt.runtime_context.get_disagg", return_value=disagg
                ),
            ):
                hooks.k3_program_decode_tree(lambda c: c, ctx)
            return (ctx.params.component_registry_override or {}).get(
                ComponentType.MAMBA
            )

        self.assertIs(build("decode"), K3DecodeStateComponent)
        self.assertIsNone(build("prefill"))
        self.assertIsNone(build("decode", radix=False))
        self.assertIsNone(build("decode", active=False))


class TestRequestAdmission(CustomTestCase):
    def test_unservable_requests_are_aborted(self):
        def req(**kw):
            params = SimpleNamespace(
                json_schema=kw.pop("json_schema", None),
                regex=None,
                ebnf=None,
                structural_tag=None,
            )
            fields = dict(
                finished_reason=None,
                return_logprob=False,
                return_hidden_states=False,
                sampling_params=params,
                aborted=None,
            )
            fields.update(kw)
            r = SimpleNamespace(**fields)
            r.set_finish_with_abort = lambda msg: setattr(r, "aborted", msg)
            return r

        queued = []

        def original(scheduler, r, *args, **kwargs):
            queued.append(r)

        plain, logprob, schema = req(), req(return_logprob=True), req(json_schema="{}")
        with mock.patch.object(hooks, "k3_program_active", return_value=True):
            for r in (plain, logprob, schema):
                hooks.reject_unsupported_k3_requests(original, None, r)
        self.assertEqual(queued, [plain, logprob, schema])  # all reach the client
        self.assertIsNone(plain.aborted)
        self.assertIn("logprobs", logprob.aborted)
        self.assertIn("structured output", schema.aborted)


class TestModelRunnerHooks(CustomTestCase):
    def test_no_backends_or_graphs_under_k3_program(self):
        original = mock.Mock()
        runner = SimpleNamespace()
        with mock.patch.object(hooks, "k3_program_active", return_value=True):
            hooks.skip_k3_program_backends(original, runner)
            hooks.skip_k3_program_graphs(original, runner)
        original.assert_not_called()
        self.assertIsNone(runner.attn_backend)
        self.assertIsNone(runner.decode_cuda_graph_runner)
        with mock.patch.object(hooks, "k3_program_active", return_value=False):
            hooks.skip_k3_program_backends(original, runner)
            hooks.skip_k3_program_graphs(
                original, runner, capture_decode_cuda_graph=True
            )
        self.assertEqual(original.call_count, 2)
        original.assert_called_with(runner, capture_decode_cuda_graph=True)


if __name__ == "__main__":
    unittest.main()
