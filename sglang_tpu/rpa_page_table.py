"""SMEM footprint of the Pallas RPA page table.

The RPA kernels take the page table as a scalar-prefetch operand, so it must
fit in the SMEM one kernel invocation can scope. The attention backend and the
platform's argument check size it the same way from here.
"""

# Scoped SMEM available to one Pallas kernel invocation on TPU.
SCOPED_SMEM_BYTES = 1 << 20


def pages_per_seq(max_context_len: int, page_size: int) -> int:
    return (max_context_len + page_size - 1) // page_size


def page_table_smem_bytes(max_num_seqs: int, pages: int) -> int:
    """Bytes of SMEM the page table occupies for ``max_num_seqs`` sequences.

    ``max_num_seqs * pages`` int32 entries, padded to a multiple of 128 entries.
    """
    entries = max_num_seqs * pages
    return ((entries + 127) // 128) * 128 * 4


def max_seqs_in_smem(pages: int) -> int:
    """The most sequences of ``pages`` pages whose table fits the budget."""
    return max(1, (SCOPED_SMEM_BYTES // 4 - 128) // pages)
