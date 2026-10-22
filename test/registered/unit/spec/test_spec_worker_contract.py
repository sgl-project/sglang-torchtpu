"""Every spec worker and relayed draft input exposes what core reads from it.

A worker the platform swaps in through ``SRTPlatform.get_spec_worker_cls`` is
reached only through ``SpecWorkerContract``, and its ``next_draft_input`` only
through ``RelayedDraftInput``; a missing member fails at the first scheduler
step that reads it, on the platform that owns the worker. The checks are
class-level, so nothing is constructed.
"""

import ast
import dataclasses
import inspect
import unittest
from pathlib import Path

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.speculative import spec_info
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    DSparkWorkerV2,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.speculative.frozen_kv_mtp_info import FrozenKVMTPDraftInput
from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import FrozenKVMTPWorkerV2
from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
    MultiLayerEagleWorkerV2,
)
from sglang.srt.speculative.spec_worker_contract import (
    SPEC_WORKER_INIT_KWARGS,
    RelayedDraftInput,
    SpecWorkerContract,
    is_spec_worker,
)
from sglang.srt.speculative.standalone_worker_v2 import StandaloneWorkerV2
from sglang.srt.speculative.uno_info import UnoDraftInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.experimental.speculative.base_spec_worker import TpuSpecWorker
from sglang_tpu.experimental.speculative.block_info import HostPaddedBlockDraftInput
from sglang_tpu.experimental.speculative.block_spec_worker import (
    TpuBlockSpecWorker,
)
from sglang_tpu.experimental.speculative.chain_info import HostPaddedChainDraftInput
from sglang_tpu.experimental.speculative.chain_spec_worker import TpuChainSpecWorker

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

CORE_WORKERS = (
    EAGLEWorkerV2,
    MultiLayerEagleWorkerV2,
    StandaloneWorkerV2,
    FrozenKVMTPWorkerV2,
    DFlashWorkerV2,
    DSparkWorkerV2,
)
# Their modules import CUDA-only kernels (sgl_kernel, flashinfer) at load time.
CORE_WORKERS_NOT_IMPORTABLE_ON_CPU = frozenset({"NGRAMWorker", "UnoWorkerV2"})
PLUGIN_WORKERS = (
    TpuSpecWorker,
    TpuBlockSpecWorker,
    TpuChainSpecWorker,
)
# The plugin's abstract skeleton; only its subclasses are handed to the scheduler.
NOT_CONSTRUCTED_BY_SCHEDULER = frozenset({TpuSpecWorker})
# check_pipeline_parallel_compat admits only EAGLE under PP, so the scheduler
# never passes these workers pp_proxy_tensors.
PP_INCAPABLE_WORKERS = frozenset({FrozenKVMTPWorkerV2})

DRAFT_INPUTS = (
    EagleDraftInput,
    FrozenKVMTPDraftInput,
    DFlashDraftInputV2,
    UnoDraftInput,
    HostPaddedBlockDraftInput,
    HostPaddedChainDraftInput,
)

_PROTOCOL_INTERNALS = frozenset({"_is_protocol", "_is_runtime_protocol", "_abc_impl"})


def _protocol_members(protocol: type) -> set[str]:
    names = {
        name
        for name in vars(protocol)
        if not (name.startswith("__") and name.endswith("__"))
    }
    return (names | set(protocol.__annotations__)) - _PROTOCOL_INTERNALS


def _class_has(cls: type, name: str) -> bool:
    """Class attribute, method, property, or a dataclass field without a default."""
    if hasattr(cls, name):
        return True
    return dataclasses.is_dataclass(cls) and any(
        field.name == name for field in dataclasses.fields(cls)
    )


def _keyword_callable(function, names) -> list[str]:
    """Names in ``names`` that ``function`` does not accept as a keyword."""
    parameters = inspect.signature(function).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return []
    return [
        name
        for name in names
        if name not in parameters
        or parameters[name].kind is inspect.Parameter.POSITIONAL_ONLY
    ]


def _signature_mismatches(cls: type, protocol: type) -> list[str]:
    """Protocol members ``cls`` does not implement with a compatible shape.

    A Protocol property must not be a plain method on ``cls``; every parameter a
    Protocol method declares must be accepted by the implementation (by keyword,
    or by position where the Protocol makes it positional-only).
    """
    problems = []
    for name in sorted(_protocol_members(protocol)):
        declared = inspect.getattr_static(protocol, name, None)
        implemented = inspect.getattr_static(cls, name, None)
        if isinstance(declared, property):
            if inspect.isfunction(implemented):
                problems.append(f"{name}: a method where core reads a property")
            continue
        if not inspect.isfunction(declared) or not inspect.isfunction(implemented):
            continue
        accepted = inspect.signature(implemented).parameters
        if any(p.kind is p.VAR_KEYWORD for p in accepted.values()):
            continue
        for param in list(inspect.signature(declared).parameters.values())[1:]:
            if param.kind is param.POSITIONAL_ONLY:
                continue
            if (
                param.name not in accepted
                or accepted[param.name].kind is param.POSITIONAL_ONLY
            ):
                problems.append(f"{name}: does not accept {param.name!r}")
    return problems


def _default_worker_class_names() -> set[str]:
    """Worker classes ``SpeculativeAlgorithm._default_worker_cls`` can return."""
    tree = ast.parse(Path(spec_info.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_default_worker_cls":
            return {
                alias.name
                for child in ast.walk(node)
                if isinstance(child, ast.ImportFrom)
                for alias in child.names
            }
    raise AssertionError("_default_worker_cls not found in spec_info.py")


class TestSpecWorkerContract(CustomTestCase):
    def test_every_core_default_worker_is_checked(self):
        checked = {cls.__name__ for cls in CORE_WORKERS}
        self.assertEqual(
            _default_worker_class_names(),
            checked | CORE_WORKERS_NOT_IMPORTABLE_ON_CPU,
        )

    def test_workers_expose_every_member_core_reads(self):
        members = _protocol_members(SpecWorkerContract)
        self.assertIn("forward_batch_generation", members)
        for cls in CORE_WORKERS + PLUGIN_WORKERS:
            with self.subTest(worker=cls.__name__):
                missing = sorted(name for name in members if not _class_has(cls, name))
                self.assertEqual(missing, [])

    def test_workers_match_the_contract_signatures(self):
        for cls in CORE_WORKERS + PLUGIN_WORKERS:
            with self.subTest(worker=cls.__name__):
                if cls not in NOT_CONSTRUCTED_BY_SCHEDULER:
                    self.assertEqual(
                        _keyword_callable(cls.__init__, SPEC_WORKER_INIT_KWARGS), []
                    )
                problems = _signature_mismatches(cls, SpecWorkerContract)
                if cls in PP_INCAPABLE_WORKERS:
                    problems = [p for p in problems if "pp_proxy_tensors" not in p]
                self.assertEqual(problems, [])

    def test_core_tells_spec_workers_from_plain_drafts(self):
        # scheduler.py and batch_result_processor.py branch on this check.
        for cls in CORE_WORKERS + PLUGIN_WORKERS:
            if cls in NOT_CONSTRUCTED_BY_SCHEDULER:
                continue
            with self.subTest(worker=cls.__name__):
                self.assertTrue(is_spec_worker(object.__new__(cls)))
        self.assertFalse(is_spec_worker(object.__new__(TpModelWorker)))

    def test_spec_worker_check_reads_no_property(self):
        """A runtime Protocol with property members made ``isinstance`` call every
        getter on Python 3.10, so a worker whose getter raised (as an
        unconstructed one does) read as a plain draft, and the per-request
        check in the result processor ran every getter."""
        reads = []

        class _Worker(EAGLEWorkerV2):
            @property
            def compile_timing_steps(self):
                reads.append("compile_timing_steps")
                raise AttributeError("not built")

        self.assertTrue(is_spec_worker(object.__new__(_Worker)))
        self.assertEqual(reads, [])


class TestRelayedDraftInputContract(CustomTestCase):
    def test_draft_inputs_expose_every_member_core_reads(self):
        members = _protocol_members(RelayedDraftInput)
        self.assertIn("future_indices", members)
        for cls in DRAFT_INPUTS:
            with self.subTest(draft_input=cls.__name__):
                missing = sorted(name for name in members if not _class_has(cls, name))
                self.assertEqual(missing, [])

    def test_draft_inputs_match_the_contract_signatures(self):
        for cls in DRAFT_INPUTS:
            with self.subTest(draft_input=cls.__name__):
                self.assertEqual(_signature_mismatches(cls, RelayedDraftInput), [])


if __name__ == "__main__":
    unittest.main()
