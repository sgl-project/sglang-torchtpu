import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.host_metadata_cache import HostMetadataCache

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_reuse_and_invalidation_after_page_change():
    cache = HostMetadataCache()
    pages = torch.tensor([[3, 4], [8, 9]], dtype=torch.int32)
    first = cache.get_by_value(name="pages", value=pages, device="cpu")
    assert cache.get_by_value(name="pages", value=pages.clone(), device="cpu") is first
    pages[0, 1] = 10
    second = cache.get_by_value(name="pages", value=pages, device="cpu")
    assert second is not first
    assert first[0, 1] == 4
    assert second[0, 1] == 10
    third = cache.get_by_value(name="pages", value=pages.flip(0), device="cpu")
    assert third is not second
    assert third[0, 0] == 8


def test_names_shapes_and_dtypes_are_part_of_cache_identity():
    cache = HostMetadataCache()
    x = torch.ones(4, dtype=torch.int32)
    a = cache.get_by_value(name="q", value=x, device="cpu")
    assert cache.get_by_value(name="kv", value=x, device="cpu") is not a
    assert cache.get_by_value(name="q", value=x.long(), device="cpu").dtype == (
        torch.int64
    )
    assert cache.get_by_value(name="q", value=x[:2], device="cpu").shape == (2,)
    assert len(cache._by_value) == 2


def test_layout_builds_once_per_key_and_device():
    cache = HostMetadataCache()
    builds = []

    def build():
        builds.append(None)
        return torch.arange(3, dtype=torch.int32)

    first = cache.get_by_layout(key=("offsets", 2), build=build, device="cpu")
    assert cache.get_by_layout(key=("offsets", 2), build=build, device="cpu") is first
    assert len(builds) == 1
    other_key = cache.get_by_layout(key=("offsets", 3), build=build, device="cpu")
    other_device = cache.get_by_layout(
        key=("offsets", 2), build=build, device=torch.device("meta")
    )
    assert other_key is not first
    assert other_device.device.type == "meta"
    assert len(builds) == 3
    # Layout and value entries never alias, even under the same name.
    by_value = cache.get_by_value(
        name="offsets", value=torch.arange(3, dtype=torch.int32), device="cpu"
    )
    assert by_value is not first


def test_active_scalar_is_owned_across_batch_sizes_and_source_mutation():
    cache = HostMetadataCache()
    previous = None
    for bs, live in ((16, 16), (4, 4), (16, 4), (4, 0)):
        query = torch.arange(bs + 1, dtype=torch.int32)
        scalar = cache.get_by_layout(
            key=("active_num_tokens", int(query[live])),
            build=lambda: torch.tensor(int(query[live]), dtype=torch.int32),
            device="cpu",
        )
        assert scalar.shape == ()
        assert scalar._base is None
        assert scalar.untyped_storage().nbytes() == 4
        assert scalar.item() == live
        if previous is not None and previous.item() == live:
            assert scalar is previous
        query.fill_(999)
        assert scalar.item() == live
        previous = scalar


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
