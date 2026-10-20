"""Fresh-process AOT cache hits must track TPU kernel inputs FX cannot see."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

# A long-standing libtpu flag that sglang does not append itself; it does not
# change this single-chip graph, only the compile inputs.
_EXTRA_LIBTPU_FLAG = "--xla_tpu_enable_async_collective_fusion=true"


def worker():
    import torch
    from torch._functorch._aot_autograd import autograd_cache

    from sglang.srt.utils.patch_torch import initialize_torch_tpu_worker

    server_args = SimpleNamespace(
        device="tpu",
        tp_size=1,
        pp_size=1,
        nnodes=1,
        node_rank=0,
        use_ray=False,
        dp_size=1,
        attn_dp_size=1,
    )
    initialize_torch_tpu_worker(server_args, tp_rank=0)

    # The Pallas kernel is lowered inside the compiled graph; FX sees only the
    # opaque custom op.
    from sglang_tpu.kv_page_write import get_write_pages_op

    write_pages = get_write_pages_op()

    def forward(block_ids, pages, kv):
        write_pages(block_ids, pages, kv)
        return kv + 0

    hits = []
    original_lookup = autograd_cache.AOTAutogradCache._lookup

    def lookup(*args, **kwargs):
        result = original_lookup(*args, **kwargs)
        hits.append(result is not None)
        return result

    with patch.object(autograd_cache.AOTAutogradCache, "_lookup", side_effect=lookup):
        compiled = torch.compile(forward, backend="tpu", fullgraph=True, dynamic=False)
        kv = torch.zeros((8, 16, 128), device="tpu")
        pages = torch.full((2, 16, 128), 3.0, device="tpu")
        block_ids = torch.tensor([1, 5], dtype=torch.int32, device="tpu")
        actual = compiled(block_ids, pages, kv).cpu()
    expected = torch.zeros((8, 16, 128))
    expected[[1, 5]] = 3.0
    assert torch.equal(actual, expected), actual
    print(json.dumps({"hits": hits}), flush=True)


class TestKernelCompileCache(unittest.TestCase):
    def test_hidden_kernel_inputs_scope_cache_hits(self):
        if importlib.util.find_spec("torch_tpu") is None:
            self.skipTest("requires TorchTPU and a dedicated TPU host")
        with tempfile.TemporaryDirectory(prefix="tpu-kernel-cache-") as root:
            base_env = {
                **os.environ,
                # Inside root: the worker scopes it to a sibling `<dir>_dp1_...`.
                "TORCHINDUCTOR_CACHE_DIR": os.path.join(root, "inductor"),
                # Pallas kernels export on CPU; the TPU belongs to torch_tpu.
                "JAX_PLATFORMS": "cpu",
            }
            for name in (
                "SGLANG_BASE_TORCHINDUCTOR_CACHE_DIR",
                "SGLANG_TPU_CHECK_RECOMPILATION",
                "SGLANG_TPU_FP8_MOE_FP32_SCALES",
                "RANK",
                "LOCAL_RANK",
                "WORLD_SIZE",
                "LOCAL_WORLD_SIZE",
            ):
                base_env.pop(name, None)
            flags = base_env.get("LIBTPU_INIT_ARGS", "")
            extra_flags = f"{flags} {_EXTRA_LIBTPU_FLAG}".strip()
            phases = (
                ("cold", {}, [False]),
                ("warm", {}, [True]),
                ("libtpu-flag", {"LIBTPU_INIT_ARGS": extra_flags}, [False]),
                ("libtpu-flag-warm", {"LIBTPU_INIT_ARGS": extra_flags}, [True]),
                # Read inside the Pallas MoE kernel, invisible to FX.
                ("kernel-env-knob", {"SGLANG_TPU_FP8_MOE_FP32_SCALES": "1"}, [False]),
                ("warm-again", {}, [True]),
                # Read in a hashed file, but changes no program: still a hit.
                ("unkeyed-env-knob", {"SGLANG_TPU_CHECK_RECOMPILATION": "1"}, [True]),
            )
            for phase, overrides, expected_hits in phases:
                with self.subTest(phase=phase):
                    result = subprocess.run(
                        [sys.executable, __file__, "--worker"],
                        env={**base_env, **overrides},
                        capture_output=True,
                        text=True,
                        timeout=600,
                    )
                    self.assertEqual(
                        result.returncode, 0, f"{phase}: {result.stderr[-4000:]}"
                    )
                    record = json.loads(result.stdout.strip().splitlines()[-1])
                    self.assertEqual(record["hits"], expected_hits, phase)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker()
        # Skip libtpu's atexit teardown, which can hang on GKE (see
        # test_collective_compile_cache.py); the next phase reuses the chip.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    else:
        unittest.main()
