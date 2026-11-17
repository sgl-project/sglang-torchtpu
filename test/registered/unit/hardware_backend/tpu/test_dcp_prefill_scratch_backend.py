"""Host metadata integration using TPU/JAX dependencies, without a model."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)


def _check_backend_request_handles():
    # Autoload is disabled so this process cannot acquire an inference device.
    # Register the compiler explicitly before importing decorated SGLang ops.
    import torch_tpu._internal.compile  # noqa: F401

    from types import SimpleNamespace

    from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
    from sglang_tpu.attention_pallas_mla_sparse import TpuPallasSparseMlaAttnBackend

    backend = object.__new__(TpuPallasSparseMlaAttnBackend)
    backend._dcp_scratch_state = [None]
    backend._dcp_gather_bucket_override = None
    old, new = CacheRequestHandle("same", 0), CacheRequestHandle("same", 0)
    batch = SimpleNamespace(
        rids=["same"], cache_request_handles=[old], extend_prefix_lens_cpu=[0, 0]
    )
    assert backend._dcp_scratch_plan(batch, [4, 0], [4096, 0]) == [0, 0]
    batch.extend_prefix_lens_cpu = [4096, 0]
    assert backend._dcp_scratch_plan(batch, [8, 0], [8192, 0]) == [0, 0]
    batch.cache_request_handles = [new]
    assert backend._dcp_scratch_plan(batch, [8, 0], [8192, 0]) is None
    batch.cache_request_handles = None
    assert backend._dcp_scratch_plan(batch, [8, 0], [8192, 0]) is None

    import torch
    from jax import tree_util

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    # Registers the same pytree used by all TPU direct-compile policies.
    import sglang_tpu.compilation.direct_compile  # noqa: F401

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


def test_backend_uses_request_handle_identity_and_excludes_padding():
    import os
    import subprocess
    import sys

    # Importing the real backend installs process-global TPU patches. Keep
    # those out of other CPU tests that exercise the original decorated ops.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,sys; runpy.run_path(sys.argv[1])"
            "['_check_backend_request_handles']()",
            __file__,
        ],
        env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", JAX_PLATFORMS="cpu"),
        check=True,
        timeout=60,
    )
