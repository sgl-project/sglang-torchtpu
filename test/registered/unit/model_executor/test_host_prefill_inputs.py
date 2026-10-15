"""Compile policies can stage model inputs without changing scheduler tensors."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch

from sglang.srt.compilation.support_torch_compile import NoopCompilePolicy
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class HostInputPolicy(NoopCompilePolicy):
    def prepare_batch_inputs(self, batch):
        return replace(batch, input_ids=batch.mm_input_ids_cpu)


class TestHostPrefillInputs(unittest.TestCase):
    def test_positions_follow_policy_placement_without_changing_scheduler(self):
        override = get_context().override_server_args(device="cpu")
        override.install()
        self.addCleanup(override.restore)
        batch = ScheduleBatch(
            reqs=[SimpleNamespace(rid="one", lora_id=None, token_type_ids=None)],
            device="meta",
            forward_mode=ForwardMode.EXTEND,
            input_ids=torch.empty(3, dtype=torch.int64, device="meta"),
            mm_input_ids_cpu=torch.tensor([4, 5, 6]),
            req_pool_indices=torch.tensor([2]),
            seq_lens=torch.empty(1, dtype=torch.int64, device="meta"),
            seq_lens_cpu=torch.tensor([67]),
            seq_lens_sum=67,
            out_cache_loc=torch.tensor([1, 2, 3]),
            extend_lens=[3],
            prefix_lens=[64],
            extend_num_tokens=3,
        )
        runner = SimpleNamespace(
            compile_policy=HostInputPolicy(),
            device="meta",
            is_draft_worker=False,
            model_config=SimpleNamespace(
                requires_mm_token_modalities=False, model_is_mrope=False
            ),
            kv_index_translator=SimpleNamespace(rebind_write_loc=lambda forward: None),
            prefill_attention_backend_str="torch_native",
            ngram_embedding_manager=SimpleNamespace(enabled=False),
            lora_manager=None,
        )
        forward = ForwardBatch.init_new(
            batch,
            runner,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_hidden_states_before_norm=False,
        )
        self.assertIs(forward.input_ids, batch.mm_input_ids_cpu)
        self.assertEqual(forward.positions.tolist(), [64, 65, 66])
        self.assertEqual(forward.extend_seq_lens.tolist(), [3])
        self.assertEqual(forward.extend_prefix_lens.tolist(), [64])
        self.assertEqual(forward.extend_start_loc.tolist(), [0])
        self.assertEqual(batch.input_ids.device.type, "meta")
        self.assertIs(forward.seq_lens, batch.seq_lens)
        self.assertEqual(batch.extend_lens, [3])
        self.assertEqual(batch.prefix_lens, [64])


if __name__ == "__main__":
    unittest.main()
