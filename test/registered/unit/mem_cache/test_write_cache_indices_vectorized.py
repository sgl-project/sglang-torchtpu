"""Unit tests for `_write_req_to_token_vectorized` (mem_cache/common.py).

On TPU, `write_cache_indices` can't use the per-request Python-int loop (it bakes
each request's (slot, len) offsets as HLO constants → per-step recompiles), so it
routes through `_write_req_to_token_vectorized`, a single tensor-indexed
`index_put_` scatter. The scatter must be **bit-identical** to the reference loop.
The equivalence is device-independent, so we check it on CPU (no TPU needed)
across a range of prefix/extend shapes, including the empty-prefix and zero-extend
edge cases.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.common import _write_req_to_token_vectorized

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_FILL = -1  # sentinel for untouched cells
_WIDTH = 32  # max_context_len (row stride)


def _build_inputs(rows, prefix_lens, extend_lens):
    """Construct a scenario with distinct, recognizable values per request."""
    req_pool_indices = torch.tensor(rows, dtype=torch.int64)
    prefix_lens_t = torch.tensor(prefix_lens, dtype=torch.int64)
    extend_lens_t = torch.tensor(extend_lens, dtype=torch.int64)

    total_extend = int(extend_lens_t.sum())
    # Distinct slot ids for the extend tokens (concatenated across requests).
    out_cache_loc = torch.arange(1000, 1000 + total_extend, dtype=torch.int32)

    # Distinct prefix slot ids per request (length == prefix_len; may be empty).
    prefix_tensors = []
    base = 1
    for pl in prefix_lens:
        prefix_tensors.append(torch.arange(base, base + pl, dtype=torch.int32))
        base += pl + 1
    return out_cache_loc, req_pool_indices, prefix_lens_t, extend_lens_t, prefix_tensors


def _reference_write(
    table, out_cache_loc, rows, prefix_lens, extend_lens, prefix_tensors
):
    """The per-request Python-int loop `write_cache_indices` uses off-TPU."""
    pt = 0
    for i in range(len(rows)):
        r = int(rows[i])
        pl = int(prefix_lens[i])
        el = int(extend_lens[i])
        table[r, 0:pl] = prefix_tensors[i]
        table[r, pl : pl + el] = out_cache_loc[pt : pt + el]
        pt += el


class TestWriteReqToTokenVectorized(CustomTestCase):
    def _assert_matches_reference(self, rows, prefix_lens, extend_lens):
        (
            out_cache_loc,
            req_pool_indices,
            prefix_lens_t,
            extend_lens_t,
            prefix_tensors,
        ) = _build_inputs(rows, prefix_lens, extend_lens)
        n_rows = max(rows) + 1
        vec = torch.full((n_rows, _WIDTH), _FILL, dtype=torch.int32)
        ref = torch.full((n_rows, _WIDTH), _FILL, dtype=torch.int32)

        _write_req_to_token_vectorized(
            vec,
            out_cache_loc,
            req_pool_indices,
            prefix_lens_t,
            extend_lens_t,
            prefix_tensors,
        )
        _reference_write(
            ref, out_cache_loc, rows, prefix_lens_t, extend_lens_t, prefix_tensors
        )
        self.assertTrue(
            torch.equal(vec, ref),
            msg=f"rows={rows} prefix={prefix_lens} extend={extend_lens}\n{vec}\n!=\n{ref}",
        )
        return vec

    def test_multi_request_mixed_prefix_extend(self):
        self._assert_matches_reference(
            rows=[2, 5, 1, 7], prefix_lens=[3, 0, 5, 2], extend_lens=[4, 6, 1, 3]
        )

    def test_single_request_no_prefix(self):
        self._assert_matches_reference(rows=[3], prefix_lens=[0], extend_lens=[5])

    def test_zero_extend_request(self):
        # A request that contributes no new tokens (extend_len == 0).
        self._assert_matches_reference(
            rows=[1, 4], prefix_lens=[2, 3], extend_lens=[0, 4]
        )

    def test_all_prefix_zero_skips_prefix_scatter(self):
        # No request has a cached prefix -> the prefix `index_put_` is skipped.
        self._assert_matches_reference(
            rows=[0, 2], prefix_lens=[0, 0], extend_lens=[3, 2]
        )

    def test_full_row_prefix_only(self):
        # A request whose prefix fills part of the row and adds no extend tokens.
        self._assert_matches_reference(
            rows=[6, 2], prefix_lens=[7, 1], extend_lens=[0, 0]
        )

    def test_written_values_and_untouched_cells(self):
        # Spot-check the actual contents: extend slots land at the right columns and
        # untouched cells keep the sentinel.
        vec = self._assert_matches_reference(rows=[0], prefix_lens=[2], extend_lens=[3])
        # row 0: cols [0,2) = prefix ids (1,2); cols [2,5) = out_cache_loc (1000..1002)
        self.assertEqual(vec[0, 0].item(), 1)
        self.assertEqual(vec[0, 1].item(), 2)
        self.assertEqual(vec[0, 2].item(), 1000)
        self.assertEqual(vec[0, 4].item(), 1002)
        self.assertEqual(vec[0, 5].item(), _FILL)  # untouched


if __name__ == "__main__":
    unittest.main()
