"""CPU contracts for the draft KV pool hooks: a DFlash-family draft on its own
attention backend gets that backend's pool, and the target reserves its padded
bytes per token."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.hooks.draft_kv_pool import (
    draft_cell_size_per_token,
    draft_mha_pool_class,
)


class TestDraftKvPoolHook(CustomTestCase):
    def test_draft_runner_takes_its_backend_pool(self):
        original = MagicMock(return_value="platform pool")
        draft = SimpleNamespace(
            is_draft_worker=True,
            speculative_draft_attention_backend="pallas_tpu_rpa_v3",
        )
        with patch(
            "sglang_tpu.platforms.tpu_platform._mha_kv_pool_cls",
            return_value="rpa pool",
        ) as pool_cls:
            self.assertEqual(draft_mha_pool_class(original, draft), "rpa pool")
        pool_cls.assert_called_once_with("pallas_tpu_rpa_v3")
        original.assert_not_called()

    def test_target_and_unresolved_draft_keep_platform_pool(self):
        for runner in (
            SimpleNamespace(
                is_draft_worker=False,
                speculative_draft_attention_backend="pallas_tpu_rpa_v3",
            ),
            SimpleNamespace(
                is_draft_worker=True, speculative_draft_attention_backend=None
            ),
        ):
            with self.subTest(runner=runner):
                original = MagicMock(return_value="platform pool")
                self.assertEqual(
                    draft_mha_pool_class(original, runner), "platform pool"
                )
                original.assert_called_once_with(runner)


class TestDraftCellSizeHook(CustomTestCase):
    @staticmethod
    def _config(num_kv_heads=2, head_dim=64):
        return SimpleNamespace(
            get_num_kv_heads=lambda tp_size: num_kv_heads,
            head_dim=head_dim,
            v_head_dim=head_dim,
        )

    def _cell_size(self, backend, config, layers=5, dtype=torch.bfloat16):
        def generic(
            *, draft_model_config, draft_num_layers, draft_kv_cache_dtype, tp_size
        ):
            heads = draft_model_config.get_num_kv_heads(tp_size)
            dims = draft_model_config.head_dim + draft_model_config.v_head_dim
            return (
                heads
                * dims
                * draft_num_layers
                * torch._utils._element_size(draft_kv_cache_dtype)
            )

        spec = SimpleNamespace(speculative_draft_attention_backend=backend)
        with patch("sglang.srt.runtime_context.get_spec", return_value=spec):
            return draft_cell_size_per_token(
                generic,
                draft_model_config=config,
                draft_num_layers=layers,
                draft_kv_cache_dtype=dtype,
                tp_size=8,
            )

    def test_fused_pool_reserves_padded_head_dim(self):
        # head_dim 64 pads to one 128-lane slot shared by K and V: 2x the generic bytes.
        self.assertEqual(
            self._cell_size("pallas_tpu_rpa_v3", self._config()),
            2 * 2 * 128 * 5 * 2,
        )

    def test_unpadded_geometry_matches_generic(self):
        self.assertEqual(
            self._cell_size("pallas_tpu_rpa_v3", self._config(head_dim=128)),
            2 * 256 * 5 * 2,
        )

    def test_non_fused_or_unresolved_backend_keeps_generic(self):
        for backend in (None, "pallas_tpu_flash"):
            with self.subTest(backend=backend):
                self.assertEqual(
                    self._cell_size(backend, self._config()), 2 * 128 * 5 * 2
                )


if __name__ == "__main__":
    unittest.main()
