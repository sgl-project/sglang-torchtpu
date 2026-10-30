"""Building the Kimi-K3 program from sglang's server settings and binding it to
the unified pools (sglang_program), over a stub program."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.k3_programs import sglang_bridge as bridge
from sglang_tpu.k3_programs import sglang_pools, sglang_program
from sglang_tpu.k3_programs.kimi_k3.layout import KDA, MLA

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

RC = "sglang.srt.runtime_context"


def _server(role="decode", max_running=8, context=1 << 20, kv="auto", draft=None):
    return [
        mock.patch(
            f"{RC}.get_disagg", return_value=SimpleNamespace(disaggregation_mode=role)
        ),
        mock.patch(
            f"{RC}.get_model",
            return_value=SimpleNamespace(context_length=context, kv_cache_dtype=kv),
        ),
        mock.patch(
            f"{RC}.get_schedule",
            return_value=SimpleNamespace(
                max_running_requests=max_running, chunked_prefill_size=8192
            ),
        ),
        mock.patch(
            f"{RC}.get_spec",
            return_value=SimpleNamespace(speculative_draft_model_path=draft),
        ),
        mock.patch(f"{RC}.get_parallel", return_value=SimpleNamespace(tp_rank=3)),
    ]


class _Patched:
    def __init__(self, patches):
        self.patches = patches

    def __enter__(self):
        for p in self.patches:
            p.start()

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()


class StubProgram:
    def __init__(self):
        self.bound = None
        self.warm = False

    def bind_kv_caches(self, caches, groups):
        self.bound = (caches, groups)

    def warmup(self):
        self.warm = True


class TestServerConfig(CustomTestCase):
    def test_acceptance_lengths(self):
        self.assertEqual(
            sglang_program.parse_acc_lengths("7:3.78,3:2.87"), {7: 3.78, 3: 2.87}
        )
        self.assertIsNone(sglang_program.parse_acc_lengths(" "))
        with self.assertRaisesRegex(ValueError, "not 'drafts:mean length'"):
            sglang_program.parse_acc_lengths("3.78")

    def test_decode_config_from_server_settings(self):
        with (
            _Patched(_server()),
            tpu_envs.SGLANG_TPU_K3_PROGRAM_SIMULATE_ACC_LENGTHS.override(
                "7:3.78,3:2.87"
            ),
        ):
            config = sglang_program.server_engine_config()
        self.assertEqual((config.role, config.max_num_seqs), ("decode", 8))
        self.assertEqual(config.num_speculative_tokens, 7)
        self.assertEqual(config.max_model_len, 1 << 20)
        self.assertEqual(
            [w for w, _ in config.synthetic_acceptance_rates_by_width], [3, 7]
        )

    def test_spec_schedule(self):
        self.assertEqual(
            sglang_program.parse_spec_schedule("1:8:7, 9:16:3"), ((1, 8, 7), (9, 16, 3))
        )
        self.assertEqual(sglang_program.parse_spec_schedule(""), ((1, 4, 7), (5, 8, 3)))
        with self.assertRaisesRegex(ValueError, "min:max:drafts"):
            sglang_program.parse_spec_schedule("1:8")
        with (
            _Patched(_server(max_running=16)),
            tpu_envs.SGLANG_TPU_K3_PROGRAM_SPEC_SCHEDULE.override("1:8:7,9:16:3"),
            tpu_envs.SGLANG_TPU_K3_PROGRAM_SIMULATE_ACC_LENGTHS.override(
                "7:3.78,3:2.87"
            ),
        ):
            config = sglang_program.server_engine_config()
        self.assertEqual(
            config.num_speculative_tokens_per_batch_size, ((1, 8, 7), (9, 16, 3))
        )

    def test_server_settings_the_programs_need(self):
        with _Patched(_server(role="null")), self.assertRaisesRegex(ValueError, "P/D"):
            sglang_program.server_engine_config()
        with (
            _Patched(_server(max_running=None)),
            self.assertRaisesRegex(ValueError, "--max-running-requests"),
        ):
            sglang_program.server_engine_config()
        with (
            _Patched(_server()),
            tpu_envs.SGLANG_TPU_K3_PROGRAM_PRESHARDED.override(None),
            self.assertRaisesRegex(ValueError, "SGLANG_TPU_K3_PROGRAM_PRESHARDED"),
        ):
            sglang_program.build_from_server()
        with (
            _Patched(_server(draft=None)),
            tpu_envs.SGLANG_TPU_K3_PROGRAM_PRESHARDED.override("/presharded"),
            self.assertRaisesRegex(ValueError, "SGLANG_TPU_K3_PROGRAM_DRAFT"),
        ):
            sglang_program.build_from_server()

    def test_the_draft_must_be_the_mla_draft(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
            with (
                _Patched(_server(draft=d)),
                tpu_envs.SGLANG_TPU_K3_PROGRAM_PRESHARDED.override("/presharded"),
                self.assertRaisesRegex(ValueError, "not the MLA DSpark draft"),
            ):
                sglang_program.build_from_server()


class TestKernelConfig(CustomTestCase):
    def test_overrides_parse_and_reject_unknown_fields(self):
        from sglang_tpu.k3_programs.kimi_k3 import config

        with mock.patch.object(config, "install", side_effect=lambda c: c):
            installed = sglang_program.install_kernel_config(
                "moe_reduce_dtype=bfloat16; prefill_expert_parallel=32"
            )
            self.assertEqual(
                (installed.moe_reduce_dtype, installed.prefill_expert_parallel),
                ("bfloat16", 32),
            )
            self.assertEqual(
                sglang_program.install_kernel_config(""), config.KernelConfig()
            )
            with self.assertRaisesRegex(ValueError, "KernelConfig field"):
                sglang_program.install_kernel_config("no_such_field=1")


class TestDecodeBinding(CustomTestCase):
    def test_binds_the_unified_pools(self):
        with mock.patch(
            f"{RC}.get_model", return_value=SimpleNamespace(kv_cache_dtype="auto")
        ):
            sglang_pools.K3StatePool(size=4, mamba_layer_ids=KDA, device="cpu")
            kv = sglang_pools.K3UnifiedKVPool(
                size=10 * 8192,
                page_size=8192,
                dtype=torch.bfloat16,
                layer_num=len(MLA),
                device="cpu",
            )
        with _Patched(_server()):
            config = sglang_program.server_engine_config()
        program = StubProgram()
        target = SimpleNamespace(
            model_runner=SimpleNamespace(
                token_to_kv_pool=SimpleNamespace(full_kv_pool=kv)
            )
        )
        with (
            mock.patch.object(sglang_program, "_program", {"decode": program}),
            mock.patch.object(sglang_program, "_config", {"decode": config}),
        ):
            steps = sglang_program.build_steps(target)
        caches, groups = program.bound
        self.assertIs(caches, kv.caches)
        self.assertTrue(program.warm)
        # 1M tokens over 8192-token logical blocks.
        self.assertEqual(steps.table_width, 128)
        self.assertEqual(groups[0].table_width, 128)
        self.assertEqual(len(groups[0].layer_names), len(MLA) + 5)
        self.assertEqual(bridge.layers_covered(groups), set(range(93 + 5)))
        self.assertEqual(steps.schedule, ((1, 4, 7), (5, 8, 3)))
        with (
            mock.patch.object(sglang_program, "_program", None),
            self.assertRaisesRegex(RuntimeError, "not built at model load"),
        ):
            sglang_program.build_steps(target)


if __name__ == "__main__":
    unittest.main()
