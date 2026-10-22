"""CPU tests for the TPU kernel inputs folded into the compile cache key."""

import ast
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch.compiler.config as compiler_config

import sglang
from sglang.srt.environ import envs
from sglang.srt.utils.patch_torch import monkey_patch_torch_tpu_env
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope
from sglang_tpu import compile_cache
from sglang_tpu.envs import envs as tpu_envs

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_PACKAGE_ROOT = Path(compile_cache.__file__).resolve().parent
_SGLANG_ROOT = Path(sglang.__file__).resolve().parent
_SRT_ROOT = _SGLANG_ROOT / "srt"
_TAG_PREFIX = "tpu-kernel-inputs-v1:"


def _digest_tag(tag):
    return [t for t in tag.split("|") if t.startswith(_TAG_PREFIX)]


class TestKernelInputsDigest(CustomTestCase):
    """The digest changes exactly when a kernel input FX cannot see changes."""

    def setUp(self):
        self.root = Path(enter_scope(self, tempfile.TemporaryDirectory()))
        self.write("sglang_tpu/kv_page_write.py", "out = pages\n")
        self.write("sglang_tpu/attention/pallas/kernel.py", "block = 128\n")
        self.write(
            "sglang_tpu/moe/pallas/gmm.py",
            "FP32 = envs.SGLANG_TPU_FP8_MOE_FP32_SCALES.get()\n",
        )
        self.write(
            "sglang/srt/layers/moe/tpu_moe_utils.py",
            "MODE = envs.SGLANG_TPU_INT4_MOE_MODE.get()\n",
        )
        self.write("sglang/srt/models/model.py", "envs.SGLANG_TPU_HOST_PADDING.get()\n")
        self.write(
            "sglang_tpu/monitor.py",
            "CHECK = envs.SGLANG_TPU_CHECK_RECOMPILATION.get()\n",
        )
        enter_scope(
            self,
            patch.dict(
                os.environ,
                {
                    **{name: "" for name in compile_cache._COMPILER_FLAG_ENV_VARS},
                    "LIBTPU_INIT_ARGS": "--xla_a=1 --xla_b=2",
                },
            ),
        )
        enter_scope(self, envs.SGLANG_TPU_CHECK_RECOMPILATION.override(False))
        enter_scope(self, envs.SGLANG_TPU_FP8_MOE_FP32_SCALES.override(False))
        enter_scope(self, envs.SGLANG_TPU_INT4_MOE_MODE.override(None))
        enter_scope(self, tpu_envs.SGLANG_TPU_HOST_PADDING.override(False))
        self.versions = {"jax": "0.10.2", "jaxlib": "0.10.2", "libtpu": "0.0.47"}
        enter_scope(
            self,
            patch.object(
                compile_cache,
                "_distribution_version",
                side_effect=lambda name: self.versions.get(name),
            ),
        )
        self.pci_device_ids = enter_scope(
            self,
            patch.object(compile_cache, "_tpu_pci_device_ids", return_value=["0x0076"]),
        )
        self.base = self.digest()

    def write(self, relative_path, text):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def digest(self):
        return compile_cache.tpu_kernel_inputs_digest(
            package_root=self.root / "sglang_tpu", srt_root=self.root / "sglang/srt"
        )

    def test_kernel_source_edits_change_the_digest(self):
        for path in (
            "sglang_tpu/kv_page_write.py",
            "sglang_tpu/attention/pallas/kernel.py",
            "sglang_tpu/moe/pallas/gmm.py",
            "sglang/srt/layers/moe/tpu_moe_utils.py",
        ):
            with self.subTest(path=path):
                original = (self.root / path).read_text()
                self.write(path, original + "# edited\n")
                self.assertNotEqual(self.digest(), self.base)
                self.write(path, original)
                self.assertEqual(self.digest(), self.base)

        self.write("sglang_tpu/kernels/new_kernel.py", "x = 1\n")
        self.assertNotEqual(self.digest(), self.base)
        (self.root / "sglang_tpu/kernels/new_kernel.py").unlink()

        # Moving a file changes the key even with identical contents.
        old = self.root / "sglang_tpu/attention/pallas/kernel.py"
        old.rename(self.root / "sglang_tpu/attention/pallas/renamed.py")
        self.assertNotEqual(self.digest(), self.base)

    def test_code_outside_the_kernel_sources_is_left_to_the_fx_key(self):
        self.write("sglang/srt/models/model.py", "changed = True\n")
        self.write("sglang_tpu/README.md", "not python\n")
        self.assertEqual(self.digest(), self.base)

    def test_compiler_flags_change_the_digest(self):
        changed = {
            "LIBTPU_INIT_ARGS": (
                "--xla_a=1 --xla_b=3",
                "--xla_a=1 --xla_b=2 --xla_c=0",
                "--xla_a=1",
            ),
            "XLA_FLAGS": ("--xla_dump_to=/tmp/dump",),
            "TORCH_TPU_INTERNAL_XLA_OPTIONS": (
                "xla_tpu_use_dynamic_smem_negotiation=true",
            ),
            "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS": ("false",),
            "TORCH_TPU_INTERNAL_ENABLE_REASSOCIATE_NORM_WEIGHTS": ("false",),
            "TORCH_TPU_INTERNAL_SKINNY_SC_OFFLOAD": ("1",),
            "TORCH_TPU_INTERNAL_SPLIT_RNG_STATE_UPDATE": ("1",),
            "TORCHTPU_STORAGE_AWARE_PLACEHOLDERS": ("true",),
        }
        for name, values in changed.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    with patch.dict(os.environ, {name: value}):
                        self.assertNotEqual(self.digest(), self.base)

    def test_compiler_flags_are_hashed_verbatim(self):
        # Flag syntax is parser-specific (absl --no<flag>, space-separated
        # values), so orders that set different final values must not collide;
        # even a harmless reordering costs a recompile.
        pairs = (
            ("--xla_a=1 --xla_a=2", "--xla_a=2 --xla_a=1"),
            ("--xla_foo --noxla_foo", "--noxla_foo --xla_foo"),
            ("--xla_b 2 --xla_a 1", "--xla_b 1 --xla_a 2"),
        )
        for first, second in pairs:
            with self.subTest(first=first):
                with patch.dict(os.environ, {"LIBTPU_INIT_ARGS": first}):
                    first_digest = self.digest()
                with patch.dict(os.environ, {"LIBTPU_INIT_ARGS": second}):
                    self.assertNotEqual(self.digest(), first_digest)
        with patch.dict(os.environ, {"LIBTPU_INIT_ARGS": "--xla_b=2 --xla_a=1"}):
            self.assertNotEqual(self.digest(), self.base)
        # Unset and empty differ too, as in vllm-torchtpu's os.getenv().
        with patch.dict(os.environ, {"XLA_FLAGS": ""}):
            empty = self.digest()
        os.environ.pop("XLA_FLAGS")
        self.assertNotEqual(self.digest(), empty)

    def test_env_knobs_that_change_no_program_stay_out_of_the_digest(self):
        # The recompile monitor is read in a hashed file but is not keyed.
        with envs.SGLANG_TPU_CHECK_RECOMPILATION.override(True):
            self.assertEqual(self.digest(), self.base)

    def test_unkeyed_env_knobs_are_all_read_by_the_kernel_sources(self):
        # No stale entries: each unkeyed name is a real field the sources read.
        _, fields = compile_cache._scan_kernel_sources(
            compile_cache.kernel_source_roots()
        )
        self.assertLessEqual(compile_cache._ENV_FIELDS_NOT_KEYED, set(fields))
        for name in compile_cache._ENV_FIELDS_NOT_KEYED:
            self.assertTrue(hasattr(envs, name) or hasattr(tpu_envs, name), name)

    def test_env_knobs_read_by_kernel_sources_change_the_digest(self):
        with envs.SGLANG_TPU_FP8_MOE_FP32_SCALES.override(True):
            self.assertNotEqual(self.digest(), self.base)
        with envs.SGLANG_TPU_INT4_MOE_MODE.override("vmem-bf16"):
            self.assertNotEqual(self.digest(), self.base)
        # A knob read only outside the kernel sources reaches FX, not the digest.
        with tpu_envs.SGLANG_TPU_HOST_PADDING.override(True):
            self.assertEqual(self.digest(), self.base)

    def test_stack_versions_and_tpu_generation_change_the_digest(self):
        for name, version in (
            ("jax", "0.10.3"),
            ("jaxlib", "0.10.3"),
            ("libtpu", "0.0.48"),
            ("torch_tpu", "0.1.2"),
        ):
            with self.subTest(name=name):
                with patch.dict(self.versions, {name: version}):
                    self.assertNotEqual(self.digest(), self.base)
        # A cache dir shared across TPU generations (v7 -> v6e).
        self.pci_device_ids.return_value = ["0x006f"]
        self.assertNotEqual(self.digest(), self.base)


class TestTpuPciDeviceIds(CustomTestCase):
    def test_reads_tpu_chips_and_skips_other_google_devices(self):
        root = Path(enter_scope(self, tempfile.TemporaryDirectory()))
        for slot, (vendor, device, pci_class) in {
            "0000:00:03.0": ("0x1ae0", "0x0076", "0xff0000"),  # TPU v7 core
            "0000:00:03.1": ("0x1ae0", "0x0076", "0xff0000"),
            "0000:00:05.0": ("0x1ae0", "0x0042", "0x020000"),  # gVNIC
            "0000:03:00.0": ("0x1ae0", "0x001f", "0x010802"),  # NVMe
            "0000:00:07.0": ("0x1ae0", "0x0f48", "0x060400"),  # PCIe bridge
            "0000:00:01.0": ("0x8086", "0x1237", "0x060000"),  # host bridge
        }.items():
            (root / slot).mkdir()
            (root / slot / "vendor").write_text(vendor + "\n")
            (root / slot / "device").write_text(device + "\n")
            (root / slot / "class").write_text(pci_class + "\n")
        with patch.object(compile_cache, "_PCI_DEVICES_DIR", root):
            self.assertEqual(compile_cache._tpu_pci_device_ids(), ["0x0076"])
        # A Google device of a class not seen yet keys the cache too.
        (root / "0000:00:04.0").mkdir()
        for name, value in (
            ("vendor", "0x1ae0"),
            ("device", "0x00aa"),
            ("class", "0x0b4000"),
        ):
            (root / "0000:00:04.0" / name).write_text(value + "\n")
        with patch.object(compile_cache, "_PCI_DEVICES_DIR", root):
            self.assertEqual(compile_cache._tpu_pci_device_ids(), ["0x0076", "0x00aa"])
        with patch.object(compile_cache, "_PCI_DEVICES_DIR", root / "missing"):
            self.assertEqual(compile_cache._tpu_pci_device_ids(), [])


class TestKernelInputsTag(CustomTestCase):
    def setUp(self):
        enter_scope(self, compiler_config.patch(cache_key_tag=""))

    def test_tag_keeps_other_tags_and_replaces_its_own(self):
        compiler_config.cache_key_tag = "user|tpu-rank-assignment-v1:abc"
        compile_cache.add_tpu_kernel_inputs_to_compile_cache_key()
        first = compiler_config.cache_key_tag
        self.assertTrue(first.startswith("user|tpu-rank-assignment-v1:abc|"))
        self.assertEqual(len(_digest_tag(first)), 1)

        compile_cache.add_tpu_kernel_inputs_to_compile_cache_key()
        self.assertEqual(compiler_config.cache_key_tag, first)

        with patch.dict(os.environ, {"LIBTPU_INIT_ARGS": "--xla_changed=1"}):
            compile_cache.add_tpu_kernel_inputs_to_compile_cache_key()
        replaced = compiler_config.cache_key_tag
        self.assertEqual(len(_digest_tag(replaced)), 1)
        self.assertNotEqual(_digest_tag(replaced), _digest_tag(first))
        self.assertTrue(replaced.startswith("user|tpu-rank-assignment-v1:abc|"))

    def test_empty_user_tag(self):
        compile_cache.add_tpu_kernel_inputs_to_compile_cache_key()
        self.assertTrue(compiler_config.cache_key_tag.startswith(_TAG_PREFIX))
        self.assertNotIn("|", compiler_config.cache_key_tag)

    def test_dp_ranks_share_the_tag_and_keep_their_cache_dirs(self):
        enter_scope(
            self,
            patch.dict(os.environ, {"TORCHINDUCTOR_CACHE_DIR": "/cache/inductor"}),
        )
        for name in ("SGLANG_BASE_TORCHINDUCTOR_CACHE_DIR", "SGLANG_TPU_PD_COLOCATED"):
            os.environ.pop(name, None)
        server_args = SimpleNamespace(
            device="tpu",
            tp_size=4,
            pp_size=1,
            nnodes=1,
            node_rank=0,
            use_ray=False,
            # The resolved layout of --dp-size 2 --enable-dp-attention.
            dp_size=1,
            attn_dp_size=2,
        )
        scopes = []
        for tp_rank in (0, 2):
            monkey_patch_torch_tpu_env(server_args, tp_rank=tp_rank)
            compile_cache.add_tpu_kernel_inputs_to_compile_cache_key()
            scopes.append(
                (
                    os.environ["TORCHINDUCTOR_CACHE_DIR"],
                    os.environ["SGLANG_TPU_CACHE_DP_SCOPE"],
                    compiler_config.cache_key_tag,
                )
            )
        self.assertEqual(
            [scope[:2] for scope in scopes],
            [
                ("/cache/inductor_dp2_dprank0_tp4", "dp2_dprank0_tp4"),
                ("/cache/inductor_dp2_dprank1_tp4", "dp2_dprank1_tp4"),
            ],
        )
        self.assertEqual(scopes[0][2], scopes[1][2])


class TestKernelInputsDigestAcrossProcesses(CustomTestCase):
    def test_fresh_processes_compute_the_same_digest(self):
        script = (
            "from sglang_tpu import compile_cache\n"
            "print(compile_cache.tpu_kernel_inputs_digest())\n"
        )
        digests = set()
        for seed, cwd in (("1", "/"), ("2", tempfile.gettempdir())):
            env = {
                **os.environ,
                "PYTHONHASHSEED": seed,
                "PYTHONPATH": os.pathsep.join(
                    filter(
                        None,
                        (
                            str(_SGLANG_ROOT.parent),
                            str(_PACKAGE_ROOT.parent),
                            os.environ.get("PYTHONPATH"),
                        ),
                    )
                ),
            }
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            digests.add(result.stdout.strip().splitlines()[-1])
        digests.add(compile_cache.tpu_kernel_inputs_digest())
        self.assertEqual(len(digests), 1, digests)


_KERNEL_API_NAMES = frozenset(
    {"pallas_call", "cpu_export_jax_op", "custom_jax_kernel", "jax_op"}
)


# jax's pytree helpers flatten host-side arguments; using them builds no kernel.
_PYTREE_MODULES = frozenset({"jax.tree", "jax.tree_util"})


def _is_kernel_module(module: str) -> bool:
    return (
        module.split(".")[0] == "jax" and module not in _PYTREE_MODULES
    ) or module.startswith("torch_tpu._internal.pallas")


def _uses_kernel_api(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(_is_kernel_module(alias.name) for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module == "jax":
            if any(f"jax.{alias.name}" not in _PYTREE_MODULES for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module:
            if _is_kernel_module(node.module) or (
                node.module == "torch_tpu._internal"
                and any(alias.name == "pallas" for alias in node.names)
            ):
                return True
        elif isinstance(node, ast.Attribute) and node.attr in _KERNEL_API_NAMES:
            return True
        elif isinstance(node, ast.Name) and node.id in _KERNEL_API_NAMES:
            return True
    return False


def _is_kernel_source(path: Path) -> bool:
    source = path.read_bytes()
    if b"jax" not in source and b"pallas" not in source:
        return False
    return _uses_kernel_api(ast.parse(source, filename=str(path)))


class TestKernelSourcesAreHashed(CustomTestCase):
    def test_kernel_api_detector(self):
        for source, expected in (
            ("from jax.experimental import pallas as pl", True),
            ("import jax.numpy as jnp", True),
            ("from jax import lax", True),
            ("out = pallas_call(kernel)", True),
            ("from torch_tpu._internal import pallas", True),
            # Host-side pytree handling, as in srt/compilation/compiled_op.py.
            ("from jax import tree_util", False),
            ("import jax.tree_util", False),
            ("from jax.tree_util import tree_map", False),
        ):
            with self.subTest(source=source):
                self.assertEqual(_uses_kernel_api(ast.parse(source)), expected)

    def test_pallas_and_jax_users_live_in_the_hashed_sources(self):
        """A kernel outside the hashed sources would reuse stale executables."""
        core_files = {_SRT_ROOT / path for path in compile_cache.TPU_KERNEL_CORE_FILES}
        for path in core_files:
            self.assertTrue(path.is_file(), path)
        self.assertTrue(
            any(_is_kernel_source(path) for path in _PACKAGE_ROOT.rglob("*.py")),
            "the kernel API detector matched nothing in sglang_tpu",
        )
        # multimodal_gen has its own TPU runtime and does not use this key;
        # sglang/test holds test helpers (e.g. Pallas mesh mocks), not kernels.
        excluded = (_SGLANG_ROOT / "multimodal_gen", _SGLANG_ROOT / "test")
        outside = [
            str(path.relative_to(_SGLANG_ROOT))
            for path in sorted(_SGLANG_ROOT.rglob("*.py"))
            if not any(path.is_relative_to(tree) for tree in excluded)
            and path not in core_files
            and _is_kernel_source(path)
        ]
        self.assertEqual(
            outside,
            [],
            "These modules build Pallas/JAX kernels outside the sglang_tpu "
            "package, so kernel edits would not invalidate cached executables. "
            "Move them to sglang_tpu or add them to "
            "compile_cache.TPU_KERNEL_CORE_FILES.",
        )


if __name__ == "__main__":
    unittest.main()
