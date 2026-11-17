# SPDX-License-Identifier: Apache-2.0
"""gather_page_ids matches a direct page-table lookup on a TPU (JAX only).

Covers the kernel that reads the page table from VMEM, and the fallback kernels used when the
table does not fit: the unrolled kernel (page table of at most 13 chunks of 128 pages) and the
bounded-loop kernel used for wider tables (256k and 1M context at page 128).
"""

import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from sglang.test.ci.ci_register import register_tpu_ci  # noqa: E402
from sglang_tpu.attention.pallas.mla_sparse import kernel  # noqa: E402
from sglang_tpu.attention.pallas.mla_sparse.kernel import (  # noqa: E402
    gather_page_ids,
)

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)

pytestmark = pytest.mark.skipif(
    not any(d.platform == "tpu" for d in jax.devices()), reason="needs a TPU"
)


def _reference(page_indices, seq_page_ids, seq_ids, max_num_seqs):
    pps = page_indices.shape[0] // max_num_seqs
    logical = np.clip(seq_page_ids, 0, pps - 1)
    return page_indices.reshape(max_num_seqs, pps)[seq_ids[:, None], logical]


@pytest.fixture(params=[True, False], ids=["vmem_table", "row_windows"])
def table_in_vmem(request, monkeypatch):
    if not request.param:
        monkeypatch.setattr(kernel, "_page_table_fits_in_vmem", lambda *args: False)
    return request.param


@pytest.mark.parametrize(
    "pages_per_seq,seq_pages",
    [
        (
            2048,
            [2048, 300],
        ),  # 16 chunks (row windows: 5 per loop iteration, last one clamps)
        (
            1000,
            [1000, 7],
        ),  # 8 chunks, page table not a multiple of 128 (row windows: unrolled)
        (8192, [256, 1170]),  # short prefixes in a 1M-context table
        (8192, [8192, 5000]),  # full-length prefix
        (
            8000,
            [8000, 900],
        ),  # 63 chunks, table not a multiple of 128 pages (VMEM table: last tile clamps)
    ],
)
@pytest.mark.parametrize("num_tokens", [64, 1003])
def test_matches_reference(pages_per_seq, seq_pages, num_tokens, table_in_vmem):
    rng = np.random.default_rng(pages_per_seq + num_tokens)
    max_num_seqs, topk = 5, 2048
    page_indices = rng.integers(
        0, 1 << 20, size=(max_num_seqs * pages_per_seq,), dtype=np.int32
    )
    # Two sequences in the batch (rows 1 and 3), each token's top-k pages inside
    # its sequence's prefix; num_tokens is not a multiple of the token block.
    split = num_tokens // 3
    seq_ids = np.full((num_tokens,), 3, dtype=np.int32)
    seq_ids[:split] = 1
    limits = np.where(seq_ids == 1, seq_pages[0], seq_pages[1])
    seq_page_ids = (rng.random((num_tokens, topk)) * limits[:, None]).astype(np.int32)
    out = gather_page_ids(
        jax.numpy.asarray(page_indices),
        jax.numpy.asarray(seq_page_ids),
        jax.numpy.asarray(seq_ids),
        max_num_seqs,
    )
    np.testing.assert_array_equal(
        np.asarray(out), _reference(page_indices, seq_page_ids, seq_ids, max_num_seqs)
    )


@pytest.mark.parametrize("pages_per_seq", [1000, 8192])
def test_tokens_of_many_sequences_per_block(pages_per_seq, table_in_vmem):
    # Every 8-token block mixes sequences (decode-like batches), so each chunk's window is built
    # from several sequences' page-table rows.
    rng = np.random.default_rng(pages_per_seq)
    max_num_seqs, topk, num_tokens = 16, 2048, 251
    page_indices = rng.integers(
        0, 1 << 20, size=(max_num_seqs * pages_per_seq,), dtype=np.int32
    )
    seq_ids = rng.integers(0, max_num_seqs, size=(num_tokens,), dtype=np.int32)
    seq_page_ids = rng.integers(
        0, pages_per_seq, size=(num_tokens, topk), dtype=np.int32
    )
    out = gather_page_ids(
        jax.numpy.asarray(page_indices),
        jax.numpy.asarray(seq_page_ids),
        jax.numpy.asarray(seq_ids),
        max_num_seqs,
    )
    np.testing.assert_array_equal(
        np.asarray(out), _reference(page_indices, seq_page_ids, seq_ids, max_num_seqs)
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
