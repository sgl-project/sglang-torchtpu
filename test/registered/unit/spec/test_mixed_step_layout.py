"""Host layout of a mixed prefill + verify forward (Stage B worker side, CPU)."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.experimental.speculative.mixed_step import (
    DUMMY_ROW,
    VerifySegment,
    extend_folded_row_maps,
    merge_mixed_batch,
    mixed_logits_index,
    mixed_positions_cpu,
    prompt_positions,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
R = 6


class TestLogitsIndex(CustomTestCase):
    def test_prompt_last_rows_then_verify_rows_padded(self):
        idx = mixed_logits_index([5, 2], 2, ext_cap=4, pad_bv=3, num_verify_rows=R)
        self.assertEqual(idx.shape[0], 4 + 3 * R)
        self.assertEqual(idx[:4].tolist(), [4, 6, DUMMY_ROW, DUMMY_ROW])
        self.assertEqual(idx[4 : 4 + 2 * R].tolist(), list(range(7, 7 + 2 * R)))
        self.assertEqual(idx[4 + 2 * R :].unique().tolist(), [DUMMY_ROW])

    def test_same_size_on_every_rank(self):
        """Ranks with no prompt or no verify rows still give L rows."""
        shapes = {
            mixed_logits_index(p, v, ext_cap=8, pad_bv=4, num_verify_rows=R).shape[0]
            for p, v in (([], 0), ([3], 0), ([], 4), ([1, 1, 1], 2))
        }
        self.assertEqual(shapes, {8 + 4 * R})

    def test_caps_are_enforced(self):
        with self.assertRaises(ValueError):
            mixed_logits_index([1] * 5, 0, ext_cap=4, pad_bv=1, num_verify_rows=R)
        with self.assertRaises(ValueError):
            mixed_logits_index([1], 3, ext_cap=4, pad_bv=2, num_verify_rows=R)


class TestRowMapsAndPositions(CustomTestCase):
    def test_folded_row_maps_extend_over_verify_rows(self):
        folded = SimpleNamespace(
            num_rows=10,
            row_floor=torch.tensor([4] * 6 + [0] * 4),
            keep_rows=torch.tensor([2, 3, 4, 5, 6, 7, 8, 9]),
        )
        maps = extend_folded_row_maps(folded, 2 * R)
        self.assertEqual(maps.row_floor.shape[0], 10 + 2 * R)
        self.assertEqual(maps.row_floor[10:].unique().tolist(), [0])
        self.assertEqual(maps.compress_rows[8:].tolist(), list(range(10, 10 + 2 * R)))

    def test_positions(self):
        self.assertEqual(prompt_positions([3, 0], [2, 1]).tolist(), [3, 4, 0])


def _prompt():
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        reqs=["p0", "p1"],
        input_ids=torch.tensor([1, 2, 3]),
        out_cache_loc=torch.tensor([10, 11, 12], dtype=torch.int32),
        prefix_lens=[100, 0],
        extend_lens=[2, 1],
        extend_num_tokens=3,
        extend_logprob_start_lens=[2, 1],
        seq_lens=torch.tensor([102, 1]),
        seq_lens_cpu=torch.tensor([102, 1]),
        seq_lens_sum=103,
        orig_seq_lens=torch.tensor([102, 1]),
        req_pool_indices=torch.tensor([3, 4]),
        req_pool_indices_cpu=torch.tensor([3, 4]),
        multimodal_inputs=[None, None],
        decoder_fill_lens=[500, 1],
        encoder_swa_reset=[True, True],
        spec_info=None,
        return_logprob=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_input_logprob_token_ids=None,
        prefill_input_ids_cpu=torch.tensor([1, 2, 3]),
    )


def _verify(n=2):
    return VerifySegment(
        reqs=[f"v{i}" for i in range(n)],
        req_pool_indices=torch.tensor([7, 8][:n]),
        req_pool_indices_cpu=torch.tensor([7, 8][:n]),
        seq_lens=torch.tensor([56, 81][:n]),
        seq_lens_cpu=torch.tensor([56, 81][:n]),
        committed_lens_cpu=[50, 75][:n],
        input_ids=torch.arange(100, 100 + n * R),
        positions_cpu=torch.cat([torch.arange(50, 56), torch.arange(75, 81)])[: n * R],
        out_cache_loc=torch.arange(200, 200 + n * R, dtype=torch.int32),
        num_verify_rows=R,
    )


class TestMergedBatch(CustomTestCase):
    def test_prompt_then_verify_rows(self):
        p, v = _prompt(), _verify()
        m = merge_mixed_batch(
            p,
            v,
            prompt_input_ids=p.input_ids,
            spec_info="SPEC",
            mixed_mode=ForwardMode.MIXED,
        )
        self.assertEqual(m.forward_mode, ForwardMode.MIXED)
        self.assertEqual(m.reqs, ["p0", "p1", "v0", "v1"])
        self.assertEqual(m.extend_lens, [2, 1, R, R])
        self.assertEqual(m.prefix_lens, [100, 0, 50, 75])
        self.assertEqual(m.extend_num_tokens, 3 + 2 * R)
        self.assertEqual(m.input_ids.tolist()[:3], [1, 2, 3])
        self.assertEqual(m.out_cache_loc.tolist()[3:], list(range(200, 200 + 2 * R)))
        self.assertEqual(m.req_pool_indices_cpu.tolist(), [3, 4, 7, 8])
        self.assertEqual(m.decoder_fill_lens, [500, 1, 56, 81])
        self.assertIsNone(m.encoder_swa_reset)
        self.assertEqual(m.spec_info, "SPEC")
        # The prompt batch itself is untouched.
        self.assertEqual(p.reqs, ["p0", "p1"])
        self.assertEqual(p.forward_mode, ForwardMode.EXTEND)
        pos = mixed_positions_cpu(p, v)
        self.assertEqual(
            pos.tolist(), [100, 101, 0] + list(range(50, 56)) + list(range(75, 81))
        )
        self.assertEqual(pos.shape[0], m.extend_num_tokens)

    def test_prompt_only(self):
        p = _prompt()
        m = merge_mixed_batch(
            p,
            None,
            prompt_input_ids=p.input_ids,
            spec_info=None,
            mixed_mode=ForwardMode.MIXED,
        )
        self.assertEqual(m.reqs, ["p0", "p1"])
        self.assertEqual(m.forward_mode, ForwardMode.MIXED)


if __name__ == "__main__":
    unittest.main()
