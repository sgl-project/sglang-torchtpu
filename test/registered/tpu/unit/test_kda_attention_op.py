# SPDX-License-Identifier: Apache-2.0
"""Execute the TPU KDA custom op and verify its donated state after compilation."""

import unittest

import numpy as np
import torch

from sglang_tpu.attention_kda import build_kda_op, pallas
from sglang.test.ci.ci_register import register_tpu_ci

if pallas is not None:
    import jax
    import jax.numpy as jnp

    from sglang_tpu.kernels.kda import ragged_kda

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")


@unittest.skipUnless(pallas is not None, "Requires torch_tpu")
class TestKDAAttentionOp(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(53)
        self.heads, self.dim = 8, 128
        self.channels = 3 * self.heads * self.dim
        self.op = build_kda_op()

    def random(self, shape):
        return self.rng.normal(scale=0.2, size=shape).astype(np.float32)

    def tensors(self, tokens):
        return [
            self.random((tokens, self.channels)),
            self.random((1, tokens, self.heads * self.dim)),
            self.random((1, tokens, self.heads)),
            self.random((4, 3, self.channels)),
            self.random((4, self.heads, self.dim, self.dim)),
            self.random((self.channels, 4)),
            self.random((1, 1, self.heads, 1)),
            self.random((self.heads * self.dim,)),
        ]

    def test_registration_and_fake_output(self):
        self.assertIs(self.op, build_kda_op())
        arrays = self.tensors(1) + [
            np.array([0, 1], np.int32),
            np.array([1], np.int32),
            np.array([1], np.int32),
        ]
        args = [torch.from_numpy(x).to("meta") for x in arrays]
        output = self.op(*args, self.heads, self.dim, -5.0)
        self.assertEqual(output.shape, (1, self.heads, self.dim))
        self.assertEqual(output.dtype, args[0].dtype)

    def test_ragged_prefill_and_compiled_decode_write_back_state(self):
        # Slot 2 starts fresh, slot 1 continues an existing request, and both
        # null slot 0 and unused slot 3 contain sentinels that must survive.
        arrays = self.tensors(6) + [
            np.array([0, 3, 4, 4], np.int32),
            np.array([2, 1, 0], np.int32),
            np.array([3, 6, 0], np.int32),
        ]
        reference = jax.jit(ragged_kda, static_argnums=(11, 12, 13))
        args = [torch.from_numpy(x).to("tpu") for x in arrays]
        # Model projections run in BF16; recurrent/conv state remains FP32.
        for i in range(3):
            args[i] = args[i].to(torch.bfloat16)
        expected = reference(
            *[
                (
                    jnp.asarray(x.cpu().float().numpy()).astype(jnp.bfloat16)
                    if i < 3
                    else jnp.asarray(arrays[i])
                )
                for i, x in enumerate(args)
            ],
            self.heads,
            self.dim,
            -5.0
        )
        output = self.op(*args, self.heads, self.dim, -5.0)
        self.check_result(args, output, expected)
        for index in (3, 4):
            for slot in (0, 3):
                np.testing.assert_array_equal(
                    args[index].cpu().numpy()[slot], arrays[index][slot]
                )

        # A compiled decode must consume the state written by the eager prefill,
        # mutate the same tensors, and mask the padded null request.
        for i, shape in enumerate(
            ((3, self.channels), (1, 3, self.heads * self.dim), (1, 3, self.heads))
        ):
            args[i] = torch.from_numpy(self.random(shape)).to(
                device="tpu", dtype=torch.bfloat16
            )
        for i, value in ((8, [0, 1, 2, 3]), (10, [4, 7, 0])):
            args[i] = torch.tensor(value, device="tpu", dtype=torch.int32)
        expected = reference(
            *[
                (
                    jnp.asarray(x.cpu().float().numpy()).astype(jnp.bfloat16)
                    if i < 3
                    else jnp.asarray(x.cpu().numpy())
                )
                for i, x in enumerate(args)
            ],
            self.heads,
            self.dim,
            -5.0
        )
        compiled = torch.compile(self.op, backend="tpu", fullgraph=True)
        output = compiled(*args, self.heads, self.dim, -5.0)
        self.check_result(args, output, expected)
        np.testing.assert_array_equal(output.cpu().float().numpy()[2], 0)

    def check_result(self, args, output, expected):
        for actual, reference in zip((args[3], args[4], output), expected):
            np.testing.assert_allclose(
                actual.cpu().float().numpy(),
                np.asarray(reference, dtype=np.float32),
                atol=2e-4,
                rtol=2e-2,
            )


if __name__ == "__main__":
    unittest.main()
