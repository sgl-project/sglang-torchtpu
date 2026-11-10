from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=6, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.srt.layers.rotary_embedding.bailing_mrope import BailingMRotaryEmbedding
from sglang.test.test_utils import CustomTestCase

# Per-element tolerance that admits a one-ulp bf16 / few-ulp fp32 rounding
# difference between the two implementations but stays far below any semantic
# drift (wrong position row, wrong section boundary, wrong pairing style).
_RTOL = {torch.bfloat16: 2**-7, torch.float32: 2e-6}


def _make_rotary(device, is_neox_style):
    # mrope_section is the checkpoint's [time, height, width]; the Ernie4.5
    # kernel requires section_h == section_w, so keep them equal.
    return BailingMRotaryEmbedding(
        head_size=32,
        rotary_dim=16,
        max_position_embeddings=64,
        base=10000,
        is_neox_style=is_neox_style,
        dtype=torch.float32,
        mrope_section=[4, 2, 2],
        video_rope=True,  # position_start becomes -64: Exercises the row offset
    ).to(device)


class TestBailingMRopeNativeMirrorParity(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
    def _run_parity(self, is_neox_style, qk_dtype):
        """Derived property, cross-implementation: the native torch mirror in
        BailingMRotaryEmbedding._ernie45_rope_fused_inplace_native is what TPU
        runs in place of triton_ernie45_rope_fused_inplace, and the two must
        apply the same rotation. Without this lock, an upstream edit to the
        Triton kernel (row selection, section boundary, pairing, dtype rule)
        silently changes TPU 2D mRoPE while every CUDA test stays green.
        """
        device = torch.device("cuda")
        torch.manual_seed(1234)
        rotary = _make_rotary(device, is_neox_style)

        # 17 tokens (non-power-of-2), distinct q/k head counts, time rows
        # positive and wider than section_hw while centered height/width rows
        # go negative (video_rope offset): every freq row branch is hit.
        positions = torch.zeros(3, 17, dtype=torch.long, device=device)
        positions[0] = torch.arange(40, 57, device=device)
        positions[1] = torch.arange(-6, 11, device=device)
        positions[2] = torch.arange(-10, 7, device=device)

        q = torch.randn(17, 5, 32, dtype=qk_dtype, device=device)
        k = torch.randn(17, 3, 32, dtype=qk_dtype, device=device)

        # Triton path through the public forward (covers the dispatch too).
        q_triton, k_triton = rotary.forward(positions, q.clone(), k.clone())

        # Native path with the position_start subtraction forward applies,
        # on the flattened [num_tokens, num_heads * head_size] layout it wants.
        q_native = q.view(17, -1).clone()
        k_native = k.view(17, -1).clone()
        rotary._ernie45_rope_fused_inplace_native(
            q_native, k_native, positions - rotary.position_start
        )

        torch.testing.assert_close(
            q_triton.view(17, 5, 32),
            q_native.view(17, 5, 32),
            rtol=_RTOL[qk_dtype],
            atol=0,
            msg="query rotation differs between native mirror and Triton kernel",
        )
        torch.testing.assert_close(
            k_triton.view(17, 3, 32),
            k_native.view(17, 3, 32),
            rtol=_RTOL[qk_dtype],
            atol=0,
            msg="key rotation differs between native mirror and Triton kernel",
        )

    def test_parity_neox_bf16(self):
        self._run_parity(is_neox_style=True, qk_dtype=torch.bfloat16)

    def test_parity_interleaved_fp32(self):
        self._run_parity(is_neox_style=False, qk_dtype=torch.float32)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
