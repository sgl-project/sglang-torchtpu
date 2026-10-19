"""Pipeline timing inputs and temporary KV allocation cleanup."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.managers.pp_profile import (
    profile_cache,
    set_profile_inputs,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _policy(table=None):
    table = (
        torch.arange(2 * 64, dtype=torch.int32).reshape(2, 64)
        if table is None
        else table
    )
    slots = torch.arange(64, 72, dtype=torch.int64)
    allocator = SimpleNamespace(alloc_extend=Mock(return_value=slots), free=Mock())
    runner = SimpleNamespace(
        token_to_kv_pool_allocator=allocator,
        page_size=4,
        device="cpu",
        req_to_token_pool=SimpleNamespace(req_to_token=table),
        model_config=SimpleNamespace(
            vocab_size=128, hidden_size=8, dtype=torch.float32, context_len=48
        ),
        dtype=torch.bfloat16,
        model=SimpleNamespace(),
        spec_aux_config=SimpleNamespace(eagle_use_aux_hidden_state=False),
    )
    return SimpleNamespace(model_runner=runner, _dummy_kv_slot=lambda: 0), slots


class TestProfileCache(CustomTestCase):
    def test_rounded_allocation_restores_request_row_on_success(self):
        policy, slots = _policy()
        allocator = policy.model_runner.token_to_kv_pool_allocator
        table = policy.model_runner.req_to_token_pool.req_to_token
        original = table.clone()
        with profile_cache(policy, 7):
            self.assertIs(policy._pp_profile_slots, slots)
            table[0].fill_(0)
        self.assertEqual(allocator.alloc_extend.call_args.args[-1], 8)
        torch.testing.assert_close(table, original)
        allocator.free.assert_called_once_with(slots)
        self.assertFalse(hasattr(policy, "_pp_profile_slots"))

    def test_forward_failure_restores_request_row_and_frees_slots(self):
        policy, slots = _policy()
        table = policy.model_runner.req_to_token_pool.req_to_token
        original = table.clone()
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            with profile_cache(policy, 8):
                table[0].fill_(0)
                raise RuntimeError("forward failed")
        torch.testing.assert_close(table, original)
        policy.model_runner.token_to_kv_pool_allocator.free.assert_called_once_with(
            slots
        )
        self.assertFalse(hasattr(policy, "_pp_profile_slots"))

    def test_allocation_failure_does_not_enter_profile(self):
        policy, _ = _policy()
        allocator = policy.model_runner.token_to_kv_pool_allocator
        allocator.alloc_extend.return_value = None
        with self.assertRaisesRegex(RuntimeError, "8 free KV tokens"):
            with profile_cache(policy, 7):
                self.fail("An unsuccessful allocation entered profiling.")
        allocator.free.assert_not_called()
        self.assertFalse(hasattr(policy, "_pp_profile_slots"))

    def test_row_initialization_failure_frees_reserved_slots(self):
        policy, slots = _policy()
        policy._dummy_kv_slot = Mock(side_effect=RuntimeError("invalid request row"))
        with self.assertRaisesRegex(RuntimeError, "invalid request row"):
            with profile_cache(policy, 8):
                self.fail("An invalid request row entered profiling.")
        policy.model_runner.token_to_kv_pool_allocator.free.assert_called_once_with(
            slots
        )
        self.assertFalse(hasattr(policy, "_pp_profile_slots"))

    def test_restore_failure_still_frees_slots_and_clears_profile_state(self):
        class BadRestore:
            def __getitem__(self, row):
                return torch.zeros(64, dtype=torch.int32)

            def __setitem__(self, row, values):
                raise RuntimeError("restore failed")

        policy, slots = _policy(BadRestore())
        with self.assertRaisesRegex(RuntimeError, "restore failed"):
            with profile_cache(policy, 8):
                pass
        policy.model_runner.token_to_kv_pool_allocator.free.assert_called_once_with(
            slots
        )
        self.assertFalse(hasattr(policy, "_pp_profile_slots"))


class TestProfileInputs(CustomTestCase):
    def test_full_chunk_metadata_uses_distinct_reserved_write_slots(self):
        policy, slots = _policy()
        policy._pp_profile_slots = slots
        batch = SimpleNamespace(out_cache_loc=torch.zeros(1, dtype=torch.int32))
        table = policy.model_runner.req_to_token_pool.req_to_token
        original_other_row = table[1].clone()
        synchronize = Mock()
        sync_module = SimpleNamespace(synchronize=synchronize)
        with patch.dict(sys.modules, {"torch_tpu._internal.sync.sync": sync_module}):
            args, kwargs = set_profile_inputs(policy, batch, (), {}, 7, 12)
        self.assertIs(args[2], batch)
        self.assertEqual(kwargs, {})
        self.assertEqual(batch.input_ids.shape, (7,))
        self.assertTrue(torch.all((batch.input_ids >= 0) & (batch.input_ids < 128)))
        torch.testing.assert_close(batch.positions, torch.arange(12, 19))
        self.assertEqual(batch.seq_lens.tolist(), [19])
        self.assertEqual(batch.seq_lens_cpu.tolist(), [19])
        self.assertEqual(batch.extend_num_tokens, 7)
        self.assertEqual(batch.extend_seq_lens_cpu, [7])
        self.assertEqual(batch.extend_prefix_lens_cpu, [12])
        torch.testing.assert_close(batch.out_cache_loc, slots[:7].to(torch.int32))
        self.assertTrue(torch.all(table[0, :12] == 0))
        torch.testing.assert_close(table[0, 12:19], slots[:7].to(torch.int32))
        torch.testing.assert_close(table[1], original_other_row)
        self.assertTrue(
            all(t.device.type == "tpu" for t in synchronize.call_args.args[0])
        )

    def test_keyword_copies_of_the_positional_inputs_are_dropped(self):
        policy, slots = _policy()
        policy._pp_profile_slots = slots
        batch = SimpleNamespace(out_cache_loc=torch.zeros(1, dtype=torch.int32))
        kwargs = {
            "input_ids": "stale",
            "positions": "stale",
            "forward_batch": "stale",
            "input_embeds": "stale",
            "input_deepstack_embeds": "kept",
        }
        sync_module = SimpleNamespace(synchronize=Mock())
        with patch.dict(sys.modules, {"torch_tpu._internal.sync.sync": sync_module}):
            args, kwargs = set_profile_inputs(policy, batch, (), kwargs, 7, 12)
        self.assertIs(args[2], batch)
        self.assertEqual(kwargs, {"input_deepstack_embeds": "kept"})

    def test_later_stage_proxies_take_the_wave_buffer_height_and_handoff_dtype(self):
        """Serving hands the wave's full buffer to the shim, which trims it to
        the bucket, so the profile traces that row count in the runner dtype."""
        policy, slots = _policy()
        policy._pp_profile_slots = slots
        outputs = []
        sync_module = SimpleNamespace(synchronize=Mock())
        with (
            patch.dict(sys.modules, {"torch_tpu._internal.sync.sync": sync_module}),
            patch(
                "sglang_tpu.compilation.decoder_compile_policy.max_extend_tokens",
                return_value=40,
            ) as rows,
        ):
            for _ in range(2):
                batch = SimpleNamespace(out_cache_loc=torch.zeros(1, dtype=torch.int32))
                _, kwargs = set_profile_inputs(
                    policy, batch, (), {"pp_proxy_tensors": None}, 7, 12
                )
                outputs.append(kwargs["pp_proxy_tensors"].tensors)
        self.assertEqual(rows.call_args.kwargs, {"context_len": 48})
        for name in ("hidden_states", "residual"):
            self.assertEqual(outputs[0][name].shape, (40, 8))
            self.assertEqual(outputs[0][name].dtype, torch.bfloat16)
            self.assertGreater(outputs[0][name].float().std().item(), 0)
            torch.testing.assert_close(outputs[0][name], outputs[1][name])
        self.assertFalse(
            torch.equal(outputs[0]["hidden_states"], outputs[0]["residual"])
        )


if __name__ == "__main__":
    unittest.main()
