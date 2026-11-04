"""Readers of the attention-DP width see `attn_dp_size`, whichever spelling set it.

`--dp-size N --enable-dp-attention` resolves to attn_dp_size=N, dp_size=1,
enable_dp_attention=False before any other handler runs
(handle_deprecated_dp_attention). Code that still reads the old pair after that
point gets one attention-DP group and fails silently. When the upstream sync
#1045 brought the rename in, such readers made the TPU batch-limit
reconciliation add compile buckets past the per-worker concurrency (an SMEM
overflow on Kimi K2.6 and Qwen3-Coder-480B), turned off three TPU refusals of
unsupported DP-attention combinations, and miscounted the 480B agentic gates'
DP schedulers.

These cases resolve real configurations with both spellings, and scan the tree so
a new reader of the deprecated flag fails here rather than in a nightly.
"""

import ast
import importlib.util
import json
import os
import pathlib
import shutil
import tempfile
import unittest
from unittest.mock import patch

from parameterized import parameterized

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_REPO = pathlib.Path(__file__).resolve().parents[4]

# A real config.json, so resolution runs past the dummy-model early return.
_MINI_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
}

# Both spellings of attention-DP 2. Every case runs on each.
_SPELLINGS = [
    ("deprecated", dict(dp_size=2, enable_dp_attention=True)),
    ("attn_dp_size", dict(attn_dp_size=2)),
]


class _ResolveCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        environment = dict(os.environ)

        def restore():
            os.environ.clear()
            os.environ.update(environment)

        self.addCleanup(restore)
        self.path = tempfile.mkdtemp(prefix="attn_dp_readers_")
        self.addCleanup(shutil.rmtree, self.path, ignore_errors=True)
        with open(os.path.join(self.path, "config.json"), "w") as handle:
            json.dump(_MINI_CONFIG, handle)

    def resolve(self, **fields):
        server_args = ServerArgs(
            model_path=self.path, device="tpu", random_seed=42, **fields
        )
        server_args.resolve_once()
        return server_args


class TestTpuBatchLimitsUseAttentionDp(_ResolveCase):
    @parameterized.expand(_SPELLINGS)
    def test_compile_buckets_stop_at_per_worker_concurrency(self, _name, spelling):
        # 32 requests over 2 attention-DP workers is 16 per worker, which the
        # configured [1, 8, 16] ladder already covers. Reading one DP group
        # made it 32 per worker: a 32 bucket the scheduler never fills, whose
        # page table overflows SMEM at long context.
        server_args = self.resolve(
            tp_size=8,
            max_running_requests=32,
            torch_compile_max_bs=16,
            tpu_compile_bs=[1, 8, 16],
            **spelling,
        )
        self.assertEqual(resolution_result(server_args, "attn_dp_size"), 2)
        self.assertEqual(resolution_result(server_args, "tpu_compile_bs"), [1, 8, 16])
        self.assertEqual(resolution_result(server_args, "torch_compile_max_bs"), 16)
        self.assertEqual(resolution_result(server_args, "max_running_requests"), 32)


class TestTpuPlatformRefusesAttentionDp(_ResolveCase):
    """The TPU defaults read a resolved record, so they must refuse attention DP
    by its resolved width. The platform's own unit tests hand it a stand-in
    record and cannot see what resolution actually leaves behind."""

    def apply_tpu_defaults(self, server_args):
        from sglang_tpu.platforms.tpu_platform import TpuSRTPlatform

        with patch("sglang_tpu.patches._prepare_tpu_env"):
            TpuSRTPlatform().apply_server_args_defaults(server_args)

    PIPELINE_PAIR = dict(tp_size=2, pp_size=2, enable_torch_compile=True)

    def test_pipeline_tensor_parallel_pair_is_admitted(self):
        # Control for the refusal below: the same stage without attention DP.
        self.apply_tpu_defaults(self.resolve(**self.PIPELINE_PAIR))

    @parameterized.expand(_SPELLINGS)
    def test_pipeline_tensor_parallel_pair_refuses_attention_dp(self, _name, spelling):
        server_args = self.resolve(**self.PIPELINE_PAIR, **spelling)
        with self.assertRaisesRegex(NotImplementedError, "tensor-parallel pair"):
            self.apply_tpu_defaults(server_args)


class TestDpSchedulerCount(CustomTestCase):
    """The bench-matrix gates count DP schedulers off /server_info."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        path = _REPO / "test/registered/tpu/e2e/bench_matrix_base.py"
        spec = importlib.util.spec_from_file_location("bench_matrix_base", path)
        cls.bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.bench)

    @parameterized.expand(
        [
            # A server from before --attn-dp-size counted attention DP in dp_size.
            ("pre_attn_dp_size_server", {"dp_size": 2}, 2),
            # A current server reports dp_size=1 under attention DP.
            ("attention_dp", {"dp_size": 1, "attn_dp_size": 2}, 2),
            ("replicas", {"dp_size": 2, "attn_dp_size": 1}, 2),
            ("single_scheduler", {"dp_size": 1, "attn_dp_size": 1}, 1),
            ("no_dp_layout", {}, None),
        ]
    )
    def test_counts_attention_dp_groups_and_replicas(self, _name, info, expected):
        self.assertEqual(self.bench._dp_schedulers(info), expected)


def _deprecated_flag_reads(path):
    """(function, line) of each read of `enable_dp_attention` in one file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def function_of(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
        return "<module>"

    reads = []
    for node in ast.walk(tree):
        attribute = (
            isinstance(node, ast.Attribute)
            and node.attr == "enable_dp_attention"
            and isinstance(node.ctx, ast.Load)
        )
        getattr_call = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "enable_dp_attention"
        )
        if attribute or getattr_call:
            reads.append((function_of(node), node.lineno))
    return reads


class TestNoReadersOfTheDeprecatedFlag(CustomTestCase):
    # The only code that may read the flag is the handler that rewrites it.
    # Everything after it reads attn_dp_size (or attn_dp_enabled_of /
    # num_dp_ranks_of in runtime_context).
    ALLOWED = {
        (
            "python/sglang/srt/arg_groups/parallel_hook.py",
            "handle_deprecated_dp_attention",
        ),
    }
    SCANNED = ("python/sglang/srt", "sglang_tpu")

    def test_only_the_deprecation_handler_reads_enable_dp_attention(self):
        found = set()
        violations = []
        for top in self.SCANNED:
            for path in sorted((_REPO / top).rglob("*.py")):
                relative = path.relative_to(_REPO).as_posix()
                for function, line in _deprecated_flag_reads(path):
                    found.add((relative, function))
                    if (relative, function) not in self.ALLOWED:
                        violations.append(f"{relative}:{line} in {function}()")
        self.assertEqual(
            violations,
            [],
            "enable_dp_attention is always False once resolution has run "
            "(handle_deprecated_dp_attention rewrites it to attn_dp_size); read "
            "attn_dp_size instead:\n" + "\n".join(violations),
        )
        # A stale entry would keep allowing a reader that moved elsewhere.
        self.assertEqual(found & self.ALLOWED, self.ALLOWED)


if __name__ == "__main__":
    unittest.main()
