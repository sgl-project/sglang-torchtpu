# SPDX-License-Identifier: Apache-2.0
"""Same-entrypoint source/port/pure-call diagnostic, bounded and isolated.

Each implementation runs in fresh TorchTPU workers. Source and port retain the
upstream effect declaration and must reproduce the same first compiler failure;
the pure-call candidate changes that declaration alone, retaining every DMA,
wait, barrier and drain, and runs the complete registered numerical bridge test.
"""

import ast
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE_REVISION = "c948bdda95208e68a474c8088937347d08fe8937"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure(mode):
    import jax
    from jax.sharding import NamedSharding

    from sglang_tpu import pallas_jax_op
    from sglang_tpu.moe.pallas.glm_moe_v2 import kernel
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    directory = Path(
        os.environ.get("GLM_DIAGNOSTIC_DIR", "/tmp/glm52-moe-export-diagnostic")
    )
    directory.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("RANK", "unknown")
    # The committed port is the unmodified pinned source plus four provenance
    # comment lines. Assert the exact source bytes before loading a fresh module.
    port_text = Path(kernel.__file__).read_text()
    source = "\n".join(port_text.splitlines()[4:]) + "\n"
    sha = hashlib.sha256(source.encode()).hexdigest()
    assert sha == "79722fd38f31db054c83e20903cd9a42557f79b97c5d5a95c6aeb8cf5302664c", (
        sha
    )
    source_path = directory / f"c948-original-{rank}.py"
    source_path.write_text(source)
    if mode == "source":
        original = load_module(f"glm_c948_source_rank{rank}", source_path)
        kernel.fused_ep_moe_v2 = original.fused_ep_moe_v2
    elif mode == "pure":
        assert source.count("has_side_effects=True,") == 1
        pure = source.replace("has_side_effects=True,", "has_side_effects=False,")
        # The declaration is the only AST change. No synchronization, DMA,
        # scratch, precision, routing, or compiler parameter is changed.
        assert ast.dump(
            ast.parse(pure.replace("has_side_effects=False,", "has_side_effects=True,"))
        ) == ast.dump(ast.parse(source))
        pure_path = directory / f"c948-pure-{rank}.py"
        pure_path.write_text(pure)
        candidate = load_module(f"glm_c948_pure_rank{rank}", pure_path)
        kernel.fused_ep_moe_v2 = candidate.fused_ep_moe_v2
    elif mode not in ("port", "manual"):
        raise ValueError(mode)

    if mode == "manual":
        import io
        from jaxlib.mlir import ir

        initialize = pallas_jax_op._CpuExportJaxCallable.__init__

        def annotated_initialize(self, *args, **kwargs):
            initialize(self, *args, **kwargs)
            export = self.exported

            def annotated_export(*export_args, **export_kwargs):
                result = export(*export_args, **export_kwargs)
                with jax.interpreters.mlir.make_ir_context():
                    module = ir.Module.parse(result.mlir_module())
                    count = 0

                    def annotate(operation):
                        nonlocal count
                        if (
                            operation.name == "stablehlo.custom_call"
                            and "has_side_effect" in operation.attributes
                            and str(operation.attributes["has_side_effect"]) == "true"
                            and str(operation.attributes["call_target_name"])
                            == '"tpu_custom_call"'
                        ):
                            operation.attributes["mhlo.sharding"] = ir.StringAttr.get(
                                "{manual}"
                            )
                            count += 1
                        for region in operation.regions:
                            for block in region.blocks:
                                for child in block.operations:
                                    annotate(child.operation)

                    annotate(module.operation)
                    assert count > 0, "no side-effect Pallas custom call found"
                    text = module.operation.get_asm(enable_debug_info=True)
                    buffer = io.BytesIO()
                    module.operation.write_bytecode(buffer)
                    serialized = buffer.getvalue()

                class AnnotatedExport:
                    mlir_module_serialized = serialized

                    def mlir_module(self):
                        return text

                    def __getattr__(self, name):
                        return getattr(result, name)

                return AnnotatedExport()

            self.exported = annotated_export

        pallas_jax_op._CpuExportJaxCallable.__init__ = annotated_initialize

    factory = pallas_jax_op.cpu_export_jax_op

    def capture(name, function, **kwargs):
        op = factory(name, function, **kwargs)
        captured = False

        def call(*args):
            nonlocal captured
            if not captured:
                mesh = kwargs["mesh"]
                out = kwargs["output_partition_specs"]
                jit = jax.jit(
                    function,
                    keep_unused=True,
                    out_shardings=NamedSharding(mesh, out[0]),
                )
                probe = pallas_jax_op._CpuExportJaxCallable(
                    name=name + "_diagnostic",
                    jit_fn=jit,
                    trace_key=mode,
                    mesh=mesh,
                    input_partition_specs=kwargs["input_partition_specs"],
                )
                placeholders = pallas_jax_op._jax_placeholders(
                    args, mesh, kwargs["input_partition_specs"]
                )
                with (
                    tpu_info_for_cpu_export(),
                    jax._src.config.export_ignore_forward_compatibility(True),
                ):
                    exported = probe.exported(*placeholders)
                text = exported.mlir_module()
                (
                    directory / f"{mode}-rank{rank}-{int('shared1' in name)}.mlir"
                ).write_text(text)
                report = dict(
                    event="export_evidence",
                    mode=mode,
                    rank=rank,
                    source_revision=SOURCE_REVISION,
                    source_sha256=sha,
                    kernel_name=name,
                    manual_computations=text.count("sdy.manual_computation"),
                    side_effect_true=text.count("has_side_effect = true"),
                    mhlo_sharding=text.count("mhlo.sharding"),
                    partition_specs=list(map(str, kwargs["input_partition_specs"])),
                    shapes=[list(a.shape) for a in args],
                )
                print(json.dumps(report), flush=True)
                captured = True
            return op(*args)

        return call

    pallas_jax_op.cpu_export_jax_op = capture


def run_worker(mode):
    # Avoid the enormous custom-call config in native ERROR logs. The exception
    # is captured below, with its first compiler signature retained verbatim.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    configure(mode)
    test = load_module(
        "glm_registered_bridge_test",
        ROOT / "test/registered/kernels/ops/moe/test_glm_ep_moe_v2.py",
    )
    try:
        test.worker()
    except RuntimeError as exc:
        message = str(exc)
        expected = (
            mode in ("source", "port")
            and "Side-effect HLO must have sharding" in message
        )
        print(
            json.dumps(
                dict(
                    event="compiler_failure",
                    mode=mode,
                    rank=os.environ.get("RANK"),
                    expected=expected,
                    signature=message[:1200],
                )
            ),
            flush=True,
        )
        if not expected:
            raise RuntimeError(message[:1200]) from None
    else:
        if mode in ("source", "port"):
            raise AssertionError(
                f"{mode}: expected the documented source compiler failure"
            )
    finally:
        import torch.distributed as dist

        libraries = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "libtpu" in line
            }
        )
        print(
            json.dumps(
                dict(
                    event="worker_mapped_libtpu",
                    mode=mode,
                    rank=os.environ.get("RANK"),
                    libraries=libraries,
                )
            ),
            flush=True,
        )
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-mode":
        run_worker(sys.argv[2])
    else:
        python = shlex.quote(sys.executable)
        script = shlex.quote(str(Path(__file__).resolve()))
        modes = (
            sys.argv[1].split("=", 1)[1].split(",")
            if len(sys.argv) == 2 and sys.argv[1].startswith("--modes=")
            else ("source", "port", "manual")
        )
        for mode in modes:
            command = (
                "unset RANK LOCAL_RANK LOCAL_WORLD_SIZE WORLD_SIZE; "
                f"eval $({python} -m torch.tpu.distributed); "
                "export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE; "
                'export JAX_PLATFORMS=cpu; export XLA_FLAGS="--xla_force_host_platform_device_count=$WORLD_SIZE"; '
                "export TF_CPP_MIN_LOG_LEVEL=3; "
                f'{python} -m torch.distributed.run --standalone --nproc_per_node="$WORLD_SIZE" {script} --worker-mode {mode}'
            )
            subprocess.run(["bash", "-lc", command], check=True, timeout=1800)
