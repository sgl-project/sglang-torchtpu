# SPDX-License-Identifier: Apache-2.0
"""LTX-2 gated attention (LTX-2.3, JoyAI-Echo) on an attention output in heads-major storage."""

import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.ltx_2 import LTX2Attention
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)


def _heads_major(q, k, v, attn_mask=None, **kwargs):
    # [b, t, h, d] over heads-major storage, as the TPU kernels return it.
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask
    )
    return out.contiguous().transpose(1, 2)


class TestLtx2GatedAttentionLayout(unittest.TestCase):
    def setUp(self):
        # The attention selector reads the global server args, which the unit
        # conftest sets only under pytest; the TPU CI wrapper runs unittest.
        previous = server_args_module._global_server_args
        server_args_module.set_global_server_args(
            SimpleNamespace(
                attention_backend=None,
                attention_backend_config=None,
                kv_gather_degree=1,
                sp_split_auto=False,
            )
        )
        self.addCleanup(server_args_module.set_global_server_args, previous)
        if not model_parallel_is_initialized():
            ensure_distributed_env_defaults()
            maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    def test_gated_cross_attention_with_text_mask(self):
        torch.manual_seed(0)
        attn = LTX2Attention(
            query_dim=64,
            context_dim=32,
            heads=4,
            dim_head=16,
            use_local_attention=True,
            apply_gated_attention=True,
        ).eval()
        for param in attn.parameters():
            param.data.normal_(0, 0.1)
        x, context = torch.randn(2, 6, 64), torch.randn(2, 5, 32)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        mask = mask[:, None, None, :]

        with (
            torch.no_grad(),
            set_forward_context(current_timestep=0, attn_metadata=None),
        ):
            attn.attn.forward = _heads_major
            out = attn(x, context=context, mask=mask)
            attn.attn.forward = lambda *a, **kw: _heads_major(*a, **kw).contiguous()
            ref = attn(x, context=context, mask=mask)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.equal(out, ref))


if __name__ == "__main__":
    unittest.main()
