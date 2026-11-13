"""Inkling short convolution (sconv) on TPU vs a port of the reference kernel.

Runs ``sglang_tpu_pallas::inkling_sconv`` through torch_tpu, as serving does,
against an eager CPU port of ``tml/kernels/sconv.py`` (static and dynamic taps,
residual, ``(W [* 2] + 1) ** -0.5`` scale) plus its conv-state update.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=240, suite="stage-a-test-tpu-unit", chips=1)

import unittest

import torch

from sglang.test.test_utils import CustomTestCase

WIDTH = 4
HEAD_DIM = 128
NUM_SLOTS = 12


def _has_torch_tpu() -> bool:
    try:
        return bool(torch.accelerator.is_available())
    except Exception:
        return False


def _reference(x, state, weight, query_start_loc, slots, seq_lens, dyn=None):
    """(y, new state) per sequence, as the reference kernel + cache update."""
    dynamic = dyn is not None
    y = torch.zeros(x.shape, dtype=torch.float32)
    state = state.clone()
    scale = (WIDTH * (2 if dynamic else 1) + 1) ** -0.5
    for s, slot in enumerate(slots.tolist()):
        start, end = query_start_loc[s].item(), query_start_loc[s + 1].item()
        length = end - start
        xs = x[start:end].float()
        has_init = seq_lens[s].item() > length
        prefix = state[slot].float() if has_init else torch.zeros(WIDTH - 1, x.shape[1])
        full = torch.cat([prefix, xs])
        acc = torch.zeros(length, x.shape[1])
        for lag in range(WIDTH):
            src = full[WIDTH - 1 - lag : WIDTH - 1 - lag + length]
            if dynamic:
                per_head = dyn[start:end, :, lag].float()
                tap = weight[lag].float() + per_head.repeat_interleave(HEAD_DIM, 1)
            else:
                tap = weight[:, WIDTH - 1 - lag].to(x.dtype).float()
            acc += src * tap
        y[start:end] = (acc + xs) * scale
        state[slot] = full[-(WIDTH - 1) :].to(state.dtype)
    return y.to(x.dtype), state


@unittest.skipUnless(_has_torch_tpu(), "Requires torch_tpu")
class TestSconvAttention(CustomTestCase):
    def setUp(self):
        from sglang_tpu.attention_inkling_sconv_pallas import build_sconv_pallas_op

        self.op = build_sconv_pallas_op()
        torch.manual_seed(0)

    def _inputs(self, num_tokens, dim, dynamic):
        x = torch.randn(num_tokens, dim).to(torch.bfloat16)
        if dynamic:
            weight = (torch.randn(WIDTH, dim) * 0.3).to(torch.bfloat16)
            dyn = (torch.randn(num_tokens, dim // HEAD_DIM, WIDTH) * 0.3).to(
                torch.bfloat16
            )
        else:
            # f32 storage checks the round-to-activation-dtype step.
            weight, dyn = torch.randn(dim, WIDTH) * 0.3, None
        return x, weight, dyn

    def _run(self, x, state, weight, lens, slots, prefix, *, dyn=None, decode=False):
        """Runs the op on device; returns (y, new state) on the host."""

        def ids(values):
            return torch.tensor(values, dtype=torch.int32).to("tpu")

        starts = torch.tensor([0, *lens]).cumsum(0).tolist()
        num_seqs = len(lens)
        state_tpu = state.unsqueeze(2).to("tpu")
        y = self.op(
            x.to("tpu"),
            state_tpu,
            weight.to("tpu"),
            ids(starts),
            ids(slots),
            ids([num_seqs if decode else 0, num_seqs if decode else 0, num_seqs]),
            ids([p + n for p, n in zip(prefix, lens)]),
            dyn=None if dyn is None else dyn.to("tpu"),
            batched_only=decode,
            prefill_only=not decode,
        )
        return y.cpu(), state_tpu.cpu().squeeze(2)

    def _check(self, actual, expected):
        torch.testing.assert_close(
            actual[0].float(), expected[0].float(), atol=2e-2, rtol=1e-2
        )
        # The state holds raw inputs: exact.
        torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)

    def _ragged_prefill(self, dynamic):
        lens = [1, 2, 3, 63, 64, 65, 200]
        slots = [3, 1, 7, 2, 9, 5, 11]
        # Alternate fresh and resumed requests; fresh slots hold stale state.
        prefix = [0 if i % 2 else 5 for i in range(len(lens))]
        for dim in (128, 256, 4608):
            with self.subTest(dim=dim):
                padding = 5
                x, weight, dyn = self._inputs(sum(lens) + padding, dim, dynamic)
                state = torch.randn(NUM_SLOTS, WIDTH - 1, dim)
                qsl = torch.tensor([0, *lens]).cumsum(0)
                actual = self._run(x, state, weight, lens, slots, prefix, dyn=dyn)
                expected = _reference(
                    x,
                    state,
                    weight,
                    qsl,
                    torch.tensor(slots),
                    torch.tensor(prefix) + torch.tensor(lens),
                    dyn,
                )
                self._check(actual, expected)
                # Padding rows are zero; null slot 0 and unused slots untouched.
                self.assertEqual(actual[0][-padding:].abs().max().item(), 0)
                for slot in set(range(NUM_SLOTS)) - set(slots):
                    self.assertTrue(torch.equal(actual[1][slot], state[slot]))

    def test_static_ragged_prefill(self):
        self._ragged_prefill(dynamic=False)

    def test_dynamic_ragged_prefill(self):
        self._ragged_prefill(dynamic=True)

    def test_chunked_prefill_matches_one_shot(self):
        """A prefill split mid-tile resumes from the state the first chunk wrote."""
        dim, slot = 256, 4
        x, weight, _ = self._inputs(200, dim, dynamic=False)
        state = torch.randn(NUM_SLOTS, WIDTH - 1, dim)
        y_full, state_full = self._run(x, state, weight, [200], [slot], [0])
        y_a, state = self._run(x[:77], state, weight, [77], [slot], [0])
        y_b, state = self._run(x[77:], state, weight, [123], [slot], [77])
        torch.testing.assert_close(torch.cat([y_a, y_b]), y_full, atol=0, rtol=0)
        self.assertTrue(torch.equal(state, state_full))

    def _decode_matches_one_shot(self, dynamic):
        # More requests than one decode tile holds.
        num_reqs, prompt, steps, dim = 37, 9, 4, 256
        slots = list(range(1, num_reqs + 1))
        x, weight, dyn = self._inputs(num_reqs * (prompt + steps), dim, dynamic)
        x = x.view(num_reqs, prompt + steps, dim)
        if dynamic:
            dyn = dyn.view(num_reqs, prompt + steps, -1, WIDTH)
        state = torch.randn(num_reqs + 1, WIDTH - 1, dim)

        def tokens(t, lo, hi):
            return None if t is None else t[:, lo:hi].reshape(-1, *t.shape[2:])

        full = self._run(
            tokens(x, 0, prompt + steps),
            state,
            weight,
            [prompt + steps] * num_reqs,
            slots,
            [0] * num_reqs,
            dyn=tokens(dyn, 0, prompt + steps),
        )
        y, state = self._run(
            tokens(x, 0, prompt),
            state,
            weight,
            [prompt] * num_reqs,
            slots,
            [0] * num_reqs,
            dyn=tokens(dyn, 0, prompt),
        )
        outputs = [y.view(num_reqs, prompt, dim)]
        for step in range(prompt, prompt + steps):
            y, state = self._run(
                tokens(x, step, step + 1),
                state,
                weight,
                [1] * num_reqs,
                slots,
                [step] * num_reqs,
                dyn=tokens(dyn, step, step + 1),
                decode=True,
            )
            outputs.append(y.view(num_reqs, 1, dim))
        torch.testing.assert_close(
            torch.cat(outputs, 1).reshape(-1, dim), full[0], atol=0, rtol=0
        )
        self.assertTrue(torch.equal(state, full[1]))

    def test_static_decode_matches_one_shot(self):
        self._decode_matches_one_shot(dynamic=False)

    def test_dynamic_decode_matches_one_shot(self):
        self._decode_matches_one_shot(dynamic=True)


class TestSconvStatePool(CustomTestCase):
    def test_conv_only_pool_layout(self):
        """Inkling's (K-1, dim) conv tuples map to kernel-native [slots, K-1, 1, dim]
        streams, and a conv-only model allocates and transfers no recurrent state."""
        from sglang.srt.configs.inkling import (
            InklingConvCacheParams,
            InklingConvStateShape,
        )
        from sglang_tpu.mamba_pool import TpuMambaPool

        shape = InklingConvStateShape(
            conv=[(WIDTH - 1, 256), (WIDTH - 1, 128)], temporal=(0, 0, 0)
        )
        pool = TpuMambaPool(
            size=4,
            spec_state_size=4,
            cache_params=InklingConvCacheParams(shape=shape, layers=[0, 2]),
            mamba_layer_ids=[0, 2],
            device="cpu",
        )
        layer = pool.mamba_cache.at_layer_idx(1)
        self.assertEqual(
            [tuple(t.shape) for t in layer.conv], [(5, 3, 1, 256), (5, 3, 1, 128)]
        )
        self.assertIsNone(layer.temporal)
        states = pool.get_state()
        self.assertEqual(len(states), 4)
        pool.set_state(states)
        self.assertEqual(
            {kind for kind, *_ in pool._iter_transfer_state_entries()}, {"conv"}
        )
        pool.copy_from(torch.tensor([1]), torch.tensor([2]))


if __name__ == "__main__":
    unittest.main()
