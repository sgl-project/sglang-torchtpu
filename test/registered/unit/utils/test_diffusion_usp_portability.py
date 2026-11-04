# SPDX-License-Identifier: Apache-2.0
"""Ulysses layout fallbacks must not depend on GPU kernel imports.

Load the real Ulysses module with its process-group and version imports isolated
from server initialization. Tensor operations and Gloo collectives run unmodified;
these CPU tests do not establish TPU lowering or full-model support.
"""

import importlib.util
import itertools
import math
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import sglang
from sglang.kernels.ops.diffusion import can_use_usp_merge_heads, usp_merge_heads
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


def _load_usp():
    parallel_name = "sglang.multimodal_gen.runtime.distributed.parallel_state"
    parallel = ModuleType(parallel_name)
    for name in (
        "get_ring_ctx",
        "get_sp_group",
        "get_ulysses_parallel_rank",
        "get_ulysses_parallel_world_size",
    ):
        setattr(parallel, name, lambda: None)
    common_name = "sglang.srt.utils.common"
    common = ModuleType(common_name)
    common.torch_release = tuple(int(part) for part in torch.__version__.split(".")[:2])
    path = Path(sglang.__file__).parent / "multimodal_gen/runtime/layers/usp.py"
    spec = importlib.util.spec_from_file_location("_usp_portability", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {parallel_name: parallel, common_name: common}):
        spec.loader.exec_module(module)
    # CUDA IPC is a separate transport, not exercised by the Gloo tests.
    module._ipc_ready_group = lambda: None
    return module


def _gloo_worker(rank, world, init_method):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=30),
    )
    try:
        usp = _load_usp()
        usp.get_sp_group = lambda: SimpleNamespace(ulysses_group=dist.group.WORLD)
        usp.get_ulysses_parallel_world_size = lambda: world
        usp.get_ulysses_parallel_rank = lambda: rank
        local_seq, local_heads, dim = 3, 2, 5
        for batch in (1, 2):
            shape = (batch, local_seq * world, local_heads * world, dim)
            full = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
            local = full[:, rank * local_seq : (rank + 1) * local_seq].contiguous()
            expected = full[:, :, rank * local_heads : (rank + 1) * local_heads]
            for head_dim in (1, 2):
                shard = local.transpose(1, 2) if head_dim == 1 else local
                reference = expected.transpose(1, 2) if head_dim == 1 else expected
                exchanged = usp._usp_input_all_to_all(shard, head_dim=head_dim)
                torch.testing.assert_close(exchanged, reference, rtol=0, atol=0)
                restored = usp._usp_output_all_to_all(exchanged, head_dim=head_dim)
                torch.testing.assert_close(restored, shard, rtol=0, atol=0)
                assert restored.is_contiguous()
            qkv = usp._usp_input_all_to_all_qkv(local, local + 10, local + 20)
            for index, actual in enumerate(qkv):
                torch.testing.assert_close(
                    actual, expected + 10 * index, rtol=0, atol=0
                )
            if batch == 1:
                packed = usp._usp_input_all_to_all_packed_qkv(
                    local[0], local[0] + 10, local[0] + 20
                )
                for index, actual in enumerate(packed):
                    torch.testing.assert_close(
                        actual, expected[0] + 10 * index, rtol=0, atol=0
                    )
        _check_head_dim2_output_exchange(usp, rank, world)
    finally:
        dist.destroy_process_group()


def _check_head_dim2_output_exchange(usp, rank, world):
    """Heads-major output (an attention kernel's [b, h, s, d] result viewed as
    [b, s, h, d], more than one local head) is exchanged in head slabs where
    the platform prefers it (TPU); any other layout or platform takes the
    sequence-major relayout. Both return bit-identical values in a contiguous
    tensor; the head-slab result is always a fresh one."""
    exchange = usp._usp_all_to_all_single
    exchanged_ndims = []

    def recording_exchange(x, role=None):
        exchanged_ndims.append(x.ndim)
        return exchange(x, role=role)

    staging = {}

    def reused_staging_buffer(role, shape, dtype, device):
        # CUDA's per-role buffer: the next exchange overwrites any alias of it.
        numel = math.prod(shape)
        if role not in staging or staging[role].numel() < numel:
            staging[role] = torch.empty(numel, dtype=dtype, device=device)
        return staging[role][:numel].view(shape)

    usp._usp_all_to_all_single = recording_exchange
    usp._a2a_staging_buffer = reused_staging_buffer
    prefix, dim = 2, 5
    # local_seq == batch == 1 is where a contiguous() of the gathered slabs
    # would alias the reused staging buffer.
    for slabs_preferred, batch, local_heads, local_seq in itertools.product(
        (False, True), (1, 2), (1, 2), (1, 3)
    ):
        usp.current_platform = SimpleNamespace(
            prefers_head_slab_output_exchange=lambda value=slabs_preferred: value
        )
        shape = (batch, prefix + local_seq * world, local_heads * world, dim)
        full = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
        start = prefix + rank * local_seq
        local = full[:, start : start + local_seq]
        heads = full[:, :, rank * local_heads : (rank + 1) * local_heads]
        heads_major = heads.transpose(1, 2).contiguous().transpose(1, 2)
        cases = {
            "sequence-major": (heads[:, prefix:].contiguous(), False),
            "heads-major": (
                heads[:, prefix:].transpose(1, 2).contiguous().transpose(1, 2),
                slabs_preferred and local_heads > 1,
            ),
            # A replicated prefix sliced off a heads-major result, as in
            # USPAttention._forward_with_replicated_prefix.
            "prefix-sliced heads-major": (heads_major[:, prefix:], False),
        }
        for name, (x, head_slabs) in cases.items():
            case = (
                f"{name}, slabs_preferred={slabs_preferred}, batch={batch}, "
                f"local_heads={local_heads}, local_seq={local_seq}"
            )
            exchanged_ndims.clear()
            out = usp._usp_output_all_to_all(x, head_dim=2)
            assert exchanged_ndims == [5 if head_slabs else 4], case
            torch.testing.assert_close(out, local, rtol=0, atol=0, msg=case)
            assert out.is_contiguous(), case
            if not head_slabs and batch == local_seq == 1:
                # The sequence-major merge is a no-op permute here, so its
                # result is the staging buffer itself, as on main.
                continue
            kept = out.clone()
            usp._usp_output_all_to_all(x + 1, head_dim=2)
            torch.testing.assert_close(out, kept, rtol=0, atol=0, msg=case)


class TestUspPortableLayouts(unittest.TestCase):
    def test_cpu_merge_is_bit_exact(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int64):
            for shape in ((2, 3, 1, 2, 5), (4, 3, 2, 2, 5), (2, 0, 1, 2, 5)):
                with self.subTest(dtype=dtype, shape=shape):
                    x = torch.arange(math.prod(shape)).reshape(shape).to(dtype)
                    for value in (x, x.transpose(1, 3)):
                        self.assertFalse(can_use_usp_merge_heads(value))
                        result = usp_merge_heads(value)
                        self.assertTrue(result.is_contiguous())
                        torch.testing.assert_close(
                            result,
                            value.permute(2, 1, 0, 3, 4).contiguous(),
                            rtol=0,
                            atol=0,
                        )

    def test_fullgraph_cpu_merge(self):
        # Capture the actual function: no mocked is_compiling() predicate.
        compiled = torch.compile(usp_merge_heads, backend="eager", fullgraph=True)
        for shape in ((2, 3, 1, 2, 5), (4, 3, 2, 2, 5)):
            x = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
            torch.testing.assert_close(
                compiled(x), x.permute(2, 1, 0, 3, 4).contiguous(), rtol=0, atol=0
            )

    def test_single_rank_is_identity(self):
        usp = _load_usp()
        usp.get_ulysses_parallel_world_size = lambda: 1
        x = torch.ones(1, 3, 4, 5)
        self.assertIs(usp._usp_input_all_to_all(x, head_dim=2), x)
        self.assertIs(usp._usp_output_all_to_all(x, head_dim=2), x)
        for actual in usp._usp_input_all_to_all_qkv(x, x, x):
            self.assertIs(actual, x)
        for actual in usp._usp_input_all_to_all_packed_qkv(x[0], x[0], x[0]):
            torch.testing.assert_close(actual, x[0], rtol=0, atol=0)

    def test_exchange_keeps_the_tensor_shape(self):
        # Equal dim-0 splits equal the flat exchange; a flattened buffer costs
        # torch_tpu relayout copies around the collective.
        usp = _load_usp()
        usp.get_sp_group = lambda: SimpleNamespace(ulysses_group=object())
        usp.get_ulysses_parallel_world_size = lambda: 2
        seen = []

        def all_to_all_single(output, input, group=None):
            seen.append(tuple(input.shape))
            output.copy_(input)

        x = torch.arange(4 * 3 * 2 * 5, dtype=torch.float32).reshape(4, 3, 2, 5)
        with patch.object(
            usp.torch.distributed, "all_to_all_single", all_to_all_single
        ):
            # A non-contiguous view of x: exchanged as a contiguous copy.
            out = usp._usp_all_to_all_single(
                x.transpose(2, 3).contiguous().transpose(2, 3)
            )
        self.assertEqual(seen, [(4, 3, 2, 5)])
        self.assertEqual(out.shape, x.shape)
        torch.testing.assert_close(out, x, rtol=0, atol=0)

    def test_eligible_eager_merge_still_dispatches_to_cuda(self):
        # Routing only; CUDA numerical coverage lives in the layout kernel tests.
        facade = sys.modules[usp_merge_heads.__module__]
        backend_name = facade.__package__ + ".usp_relayout_jit"
        backend = ModuleType(backend_name)
        x = torch.zeros(2, 3, 1, 2, 5)
        sentinel = object()
        calls = []

        def cuda_impl(value):
            calls.append(value)
            return sentinel

        backend._usp_merge_heads_cuda = cuda_impl
        with (
            patch.object(facade, "can_use_usp_merge_heads", return_value=True),
            patch.dict(sys.modules, {backend_name: backend}),
        ):
            self.assertIs(usp_merge_heads(x), sentinel)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], x)

    def test_import_and_fallback_without_gpu_layout_dependencies(self):
        # A fresh process prevents a previous test's import cache hiding a leak.
        code = textwrap.dedent("""
            import importlib.abc
            import runpy
            import sys
            import torch

            blocked = (
                "triton", "tvm_ffi", "sglang.kernels.jit",
                "sglang.kernels.ops.diffusion.layout.ulysses_qkv_triton",
                "sglang.kernels.ops.diffusion.layout.usp_relayout_jit",
            )
            class NoGpuLayouts(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if any(fullname == p or fullname.startswith(p + ".") for p in blocked):
                        raise ModuleNotFoundError("blocked optional dependency: " + fullname)

            sys.meta_path.insert(0, NoGpuLayouts())
            module = runpy.run_path(sys.argv[1])
            usp = module["_load_usp"]()
            usp.get_ulysses_parallel_world_size = lambda: 1
            x = torch.zeros(1, 3, 4, 5)
            assert usp._usp_input_all_to_all(x, head_dim=2) is x
            module["TestUspPortableLayouts"]("test_cpu_merge_is_bit_exact").test_cpu_merge_is_bit_exact()
            assert not any(any(name == p or name.startswith(p + ".") for p in blocked)
                           for name in sys.modules)
        """)
        result = subprocess.run(
            [sys.executable, "-c", code, __file__],
            capture_output=True,
            text=True,
            timeout=90,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_real_gloo_round_trip_and_qkv(self):
        for world in (2, 4):
            with self.subTest(world=world), tempfile.TemporaryDirectory() as tmp:
                mp.spawn(
                    _gloo_worker,
                    args=(world, f"file://{tmp}/rendezvous"),
                    nprocs=world,
                    join=True,
                )


if __name__ == "__main__":
    unittest.main()
