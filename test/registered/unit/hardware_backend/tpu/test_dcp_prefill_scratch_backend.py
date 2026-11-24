"""Host metadata integration using TPU/JAX dependencies, without a model."""

import sys

import pytest

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)


def _check_backend_request_handles():
    # Autoload is disabled so this process cannot acquire an inference device.
    # Register the compiler explicitly before importing decorated SGLang ops.
    from types import SimpleNamespace

    import torch_tpu._internal.compile  # noqa: F401

    from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
    from sglang_tpu.attention_pallas_mla_sparse import TpuPallasSparseMlaAttnBackend

    backend = object.__new__(TpuPallasSparseMlaAttnBackend)
    backend._dcp_gather_bucket_override = None
    old, new = CacheRequestHandle("same", 0), CacheRequestHandle("same", 0)
    batch = SimpleNamespace(
        rids=["same"], cache_request_handles=[old], extend_prefix_lens_cpu=[0, 0]
    )
    # A padded batch wider than the slots gathers: warmup compiled no scratch
    # variant for it, even with one real row.
    backend._dcp_scratch_state = [None]
    assert backend._dcp_scratch_plan(batch, [4, 0], [4096, 0]) is None
    backend._dcp_scratch_state = [None, None]
    assert backend._dcp_scratch_plan(batch, [4, 0], [4096, 0]) == [0, 0]
    batch.extend_prefix_lens_cpu = [4096, 0]
    assert backend._dcp_scratch_plan(batch, [8, 0], [8192, 0]) == [0, 0]
    batch.cache_request_handles = [new]
    assert backend._dcp_scratch_plan(batch, [8, 0], [8192, 0]) is None
    batch.cache_request_handles = None
    assert backend._dcp_scratch_plan(batch, [8, 0], [8192, 0]) is None

    import torch
    from jax import tree_util

    # Registers the same pytree used by all TPU direct-compile policies.
    import sglang_tpu.compilation.direct_compile  # noqa: F401
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.forward_mode import ForwardMode

    forward = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=torch.tensor([1]),
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([1]),
        out_cache_loc=torch.tensor([0]),
        seq_lens_sum=1,
        cache_request_handles=[old],
    )
    leaves, structure = tree_util.tree_flatten(forward)
    assert all(leaf is not old for leaf in leaves)
    restored = tree_util.tree_unflatten(structure, leaves)
    assert restored.cache_request_handles is None
    assert restored.input_ids is forward.input_ids
    forward.cache_request_handles = [new, CacheRequestHandle("other", 0)]
    assert tree_util.tree_structure(forward) == structure


def _check_dcp_verify_rows():
    """Under DCP, target verify enters the sharded decode ops as one-token
    sequences: verify row i of a request is a sequence of length
    seq_len + i + 1 over that request's page table, so its key lands at
    position seq_len + i and it attends to the rows before it."""
    from types import SimpleNamespace

    import torch
    import torch_tpu._internal.compile  # noqa: F401

    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang_tpu.attention_pallas_mla_sparse import TpuPallasSparseMlaAttnBackend

    backend = object.__new__(TpuPallasSparseMlaAttnBackend)
    pages = torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int32)
    backend.forward_metadata = SimpleNamespace(
        page_indices=pages,
        cu_q_lens=torch.tensor([0, 3, 6], dtype=torch.int32),
        distribution=None,
        segments="mixed",
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_info=SimpleNamespace(num_tokens_per_req=3),
        seq_lens=torch.tensor([100, 7]),
    )
    # Verify metadata comes in the decode ([n, n, n]) or the mixed ([0, 0, n])
    # encoding; the expanded rows must be decode-shaped either way, or the
    # decode-only scoring pass skips them.
    for distribution in ([2, 2, 2], [0, 0, 2]):
        backend.forward_metadata.distribution = torch.tensor(
            distribution, dtype=torch.int32
        )
        rows = backend._dcp_rows(batch)
        assert rows.seq_lens.tolist() == [101, 102, 103, 8, 9, 10]
        assert rows.page_indices.tolist() == [10, 11, 12] * 3 + [20, 21, 22] * 3
        assert rows.cu_q_lens.tolist() == list(range(7))
        assert rows.distribution.tolist() == [6, 6, 6]
        assert rows.segments == "decode"


def _run_in_subprocess(name):
    import os
    import subprocess
    import sys

    # Importing the real backend installs process-global TPU patches. Keep
    # those out of other CPU tests that exercise the original decorated ops.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,sys; runpy.run_path(sys.argv[1])[sys.argv[2]]()",
            __file__,
            name,
        ],
        env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", JAX_PLATFORMS="cpu"),
        check=True,
        timeout=60,
    )


def test_dcp_verify_rows_are_one_token_sequences():
    _run_in_subprocess("_check_dcp_verify_rows")


def test_backend_uses_request_handle_identity_and_excludes_padding():
    _run_in_subprocess("_check_backend_request_handles")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
