"""CPU tests: the DSV4.1 TPU extend metadata builds row-wise page tables once per
distinct slot (an extend's batch cap repeats the padding slot 0) and uploads the
multi-MB tables outside the packed buffer. Both must leave every value unchanged."""

import random

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.models.deepseek_v41 import attention_backend as ab

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

SLOTS, CONTEXT = 9, 1 << 14


def _req_to_token(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(256, 1 << 20, (SLOTS, CONTEXT), generator=g, dtype=torch.int32)


def _per_row(r2t, slots, columns):
    """The reference: every row built on its own."""
    return torch.cat([ab._full_page_table(r2t, s.view(1), columns) for s in slots])


def test_full_page_table_dedup_matches_per_row():
    rng = random.Random(0)
    for seed in range(5):
        r2t = _req_to_token(seed)
        columns = ab._full_page_columns(CONTEXT)
        for slots in (
            torch.tensor([3] + [0] * 63),  # one prompt + batch-cap padding
            torch.tensor([rng.randrange(SLOTS) for _ in range(64)]),
            torch.arange(SLOTS),  # all distinct: no dedup path
        ):
            got = ab._full_page_table(r2t, slots, columns)
            assert torch.equal(got, _per_row(r2t, slots, columns))
            assert got.dtype == torch.int64


def test_slot_tables_index_pages_dedup_matches_direct():
    rng = random.Random(1)
    r2t = _req_to_token(7)
    columns = ab._full_page_columns(CONTEXT)
    for slots in (
        torch.tensor([5, 2] + [0] * 62),
        torch.tensor([rng.randrange(SLOTS) for _ in range(64)]),
        torch.arange(SLOTS),
    ):
        full = ab._full_page_table(r2t, slots, columns)
        tables = ab._slot_tables(
            req_pool_indices=slots,
            full_pages=full,
            full_pages_device=None,
            uses_window=True,
            full_to_swa_index_mapping=None,
            ring=False,
            low_ratios=ab.LOW_RATIOS,
            device=torch.device("cpu"),
        )
        for ratio in ab.LOW_RATIOS:
            want = ab._index_pages(full, ratio).to(torch.int32)
            assert torch.equal(tables.index_pages[ratio], want), ratio


def test_upload_metadata_large_fields_outside_the_pack():
    host = {
        "small_i": torch.arange(100, dtype=torch.int64),
        "big": torch.arange(ab._PACK_MAX_NUMEL + 7, dtype=torch.int64).view(-1, 1),
        "mask": torch.tensor([True, False, True]),
        "small_j": torch.arange(5, dtype=torch.int32) * 3,
    }
    with tpu_envs.SGLANG_TPU_DSV41_PACKED_METADATA.override(True):
        packed = ab._upload_metadata(host, device=torch.device("cpu"), pack=True)
    plain = ab._upload_metadata(host, device=torch.device("cpu"), pack=False)
    assert list(packed) == list(host)
    for k in host:
        assert packed[k].dtype == plain[k].dtype, k
        assert torch.equal(packed[k], plain[k]), k
    # The small int32 fields share one buffer; the big one is its own tensor.
    assert (
        packed["small_i"].untyped_storage().data_ptr()
        == packed["small_j"].untyped_storage().data_ptr()
    )
    assert (
        packed["big"].untyped_storage().data_ptr()
        != packed["small_i"].untyped_storage().data_ptr()
    )
