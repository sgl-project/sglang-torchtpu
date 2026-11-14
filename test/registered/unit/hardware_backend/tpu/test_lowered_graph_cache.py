"""Keying and storage of the lowered-graph store."""

import dataclasses
import enum
import json
import logging
import sys
import types

import pytest
from jax import tree_util

from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.compilation import lowered_graph_cache as cache
from sglang_tpu.compilation import source_digest
from sglang_tpu.compilation.lowered_graph import (
    NO_CALLBACKS,
    LoweredGraph,
    LoweredGraphRequest,
)
from sglang_tpu.compilation.spmd_compiler import ExecutableInterface, InputSpec

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit")


class _Mode(enum.Enum):
    EXTEND = 1


@dataclasses.dataclass
class _Output:
    logits: object
    hidden: object = None
    mode: _Mode = dataclasses.field(default=_Mode.EXTEND, metadata={"static": True})


tree_util.register_dataclass(_Output)

_COMPILER_TAG = "tpu-kernel-inputs-v1:abc|tpu-rank-assignment-v1:layout8"
COMPONENTS = {
    "sources": "src-1",
    "compiler": _COMPILER_TAG,
    "versions": {"torch": "2.11.0", "tpu-sync-torch": "0.1"},
    "environment": {"SGLANG_DIRECT_COMPILE_ASYNC": "1"},
    "server_args": {"tp_size": 16, "sbd_mask_token_id": 201818},
}


@pytest.fixture(autouse=True)
def allow_test_node_types(monkeypatch):
    monkeypatch.setattr(
        cache,
        "TREEDEF_MODULE_PREFIXES",
        (*cache.TREEDEF_MODULE_PREFIXES, __name__),
    )


def _spec(shape, dtype, stride, offset=0):
    return InputSpec(shape, dtype, stride, offset)


def _request(name="target/model.forward", key='["EXTEND",16,512]', **changes):
    request = LoweredGraphRequest(
        name=name,
        graph=f"{name}:{key}",
        input_treedef="PyTreeDef(((*, *), {}))",
        input_tensors=(("[0][0]", _spec((512,), "torch.int32", (1,))),),
        static_inputs=(("[0][1].batch_size", "int", "16"),),
        model_inputs=(
            ("embed.weight", _spec((8, 4), "torch.bfloat16", (4, 1))),
            ("norm.weight", _spec((4,), "torch.bfloat16", (1,))),
        ),
        model_input_names=(
            ("embed.weight", 0),
            ("lm_head.weight", 0),
            ("norm.weight", 1),
        ),
        kv_inputs=(_spec((64, 4), "torch.bfloat16", (4, 1)),),
        argument_layouts=("([1, 0], [[8, 128]], 0)", "None", "None", "None"),
        prune_unused_inputs=False,
    )
    return dataclasses.replace(request, **changes)


def _graph(
    executable=b"\x00executable-bytes",
    fingerprint="fp-1",
    subslice_origin=None,
    **changes,
):
    interface = ExecutableInterface(
        input_digest="digest-1",
        input_specs=(
            _spec((8, 4), "torch.bfloat16", (4, 1)),
            _spec((4,), "torch.bfloat16", (1,)),
            _spec((64, 4), "torch.bfloat16", (4, 1)),
            _spec((512,), "torch.int32", (1,)),
        ),
        argument_layouts=[[1, 0], [], [], []],
        model_indices=(0, 1),
        output_treespec=tree_util.tree_structure((_Output(logits=0), {"aux": [0]})),
        callback_factories=NO_CALLBACKS,
    )
    return LoweredGraph(
        interface=dataclasses.replace(interface, **changes),
        executable=executable,
        fingerprint=fingerprint,
        subslice_origin=subslice_origin,
    )


def _store(tmp_path, mode, components=COMPONENTS):
    return cache.FileLoweredGraphStore(mode, tmp_path, components=lambda: components)


def _assert_same_graph(actual, expected):
    assert actual.executable == expected.executable
    assert actual.fingerprint == expected.fingerprint
    assert actual.interface == expected.interface
    assert actual.subslice_origin == expected.subslice_origin
    assert type(actual.subslice_origin) is type(expected.subslice_origin)
    # The store rebuilds the SPMD compiler's types, not plain tuples.
    assert all(type(spec) is InputSpec for spec in actual.interface.input_specs)


def test_graph_key_is_deterministic_and_sensitive_to_every_input():
    process = cache.ProcessKey.of(COMPONENTS)
    base = cache.graph_key(process, _request())
    assert base == cache.graph_key(cache.ProcessKey.of(dict(COMPONENTS)), _request())

    variants = {
        "name": "draft/forward",
        "graph": 'target/model.forward:["EXTEND",16,1024]',
        "input_treedef": "PyTreeDef(((*, *, *), {}))",
        "input_tensors": (("[0][0]", _spec((512,), "torch.int64", (1,))),),
        "static_inputs": (("[0][1].batch_size", "int", "8"),),
        "model_inputs": (
            ("embed.weight", _spec((8, 4), "torch.float8_e4m3fn", (4, 1))),
            ("norm.weight", _spec((4,), "torch.bfloat16", (1,), 4)),
        ),
        "model_input_names": (("embed.weight", 0), ("norm.weight", 1)),
        "kv_inputs": (_spec((64, 4), "torch.bfloat16", (1, 64)),),
        "argument_layouts": ("([0, 1], [[8, 128]], 0)", "None", "None", "None"),
        "prune_unused_inputs": True,
    }
    assert set(variants) == {f.name for f in dataclasses.fields(LoweredGraphRequest)}
    keys = {
        cache.graph_key(process, _request(**{name: value}))
        for name, value in variants.items()
    }
    assert len(keys) == len(variants) and base not in keys

    for name in COMPONENTS:
        changed = dict(COMPONENTS, **{name: "other"})
        assert cache.graph_key(cache.ProcessKey.of(changed), _request()) != base
    # The rank-assignment tag alone distinguishes the two host placements of
    # rank 0.
    swapped = dict(
        COMPONENTS, compiler="tpu-kernel-inputs-v1:abc|tpu-rank-assignment-v1:layout0"
    )
    assert cache.ProcessKey.of(swapped).digest != process.digest


def test_keyed_inputs_drop_only_placement_and_secrets():
    environ = {
        "SGLANG_DIRECT_COMPILE_ASYNC": "1",
        "SGLANG_MAX_TOKENS": "4096",
        "LIBTPU_INIT_ARGS": "--xla_flag",
        "TPU_WORKER_HOSTNAMES": "a,b",
        "SGLANG_HOST_IP": "::1",
        "SGLANG_GRPC_PORT": "1",
        "SGLANG_TPU_LOWERED_GRAPH_KEY_SALT": "pod-a",
        "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": "name",
        "SGLANG_RUN_ID": "sglang-run-1792005736.1178446-98414985",
        "HOME": "/root",
        # In a singleton torch world these tell engines on different chips
        # apart; an entry's executable runs only on its own chips.
        "TPU_VISIBLE_DEVICES": "3",
        "SGLANG_TPU_PJRT_RANK_BASE": "4",
        # Per slice or host: every gang has its own, and host ordering can
        # give the host another node rank (keyed through the rank).
        "TPU_NAME": "my-service-engine-0",
        "TPU_WORKER_ID": "1",
        # GKE injects these for some slices and restarts only.
        "TPU_ACCELERATOR_TYPE": "tpu7x-16",
        "TPU_TOPOLOGY": "2x2x4",
        "TPU_TOPOLOGY_ALT": "false",
        "TPU_TOPOLOGY_WRAP": "false,false,false",
        "TPU_HOST_BOUNDS": "1,1,2",
        "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1",
        # Graph-shaping variables stay keyed, also those named with words a
        # placement name can have.
        "TPU_PREMAPPED_BUFFER_SIZE": "1",
        "SGLANG_TPU_REQ_TO_TOKEN_ON_HOST": "1",
        "SGLANG_TPU_HOST_R2T": "1",
        "SGLANG_QWEN4_PLE_FILE_DIR": "/mnt/ple",
        "TPU_LOG_DIR": "/tmp/tpu_logs",
        # Cache delivery.
        "TORCH_TPU_TIER2_NAME": "my-model",
        "TORCH_TPU_TIER2_COMPILATION_CACHE": "my-model",
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": "/dev/shm/compile_cache/t3",
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT": "/dev/shm/x/tier3",
        # The deployment's own delivery settings, and the store's settings.
        "SGLANG_TPU_LOWERED_GRAPH_UNKEYED_ENV": "SGLANG_SHARED_CACHE_*,SGLANG_RING_MODE",
        "SGLANG_SHARED_CACHE_MODE": "on",
        "SGLANG_SHARED_CACHE_INSTALLED": "4242",
        "SGLANG_RING_MODE": "peer",
        "SGLANG_TPU_LOWERED_GRAPH_SOURCE_PACKAGES": "my_models",
        "SGLANG_TPU_LOWERED_GRAPH_SOURCES_PATH": "/sgl-workspace/sources.json",
    }
    assert cache.keyed_environment(environ) == {
        "SGLANG_DIRECT_COMPILE_ASYNC": "1",
        "SGLANG_MAX_TOKENS": "4096",
        "LIBTPU_INIT_ARGS": "--xla_flag",
        "TPU_VISIBLE_DEVICES": "3",
        "SGLANG_TPU_PJRT_RANK_BASE": "4",
        "TPU_PREMAPPED_BUFFER_SIZE": "1",
        "SGLANG_TPU_REQ_TO_TOKEN_ON_HOST": "1",
        "SGLANG_TPU_HOST_R2T": "1",
        "SGLANG_QWEN4_PLE_FILE_DIR": "/mnt/ple",
    }


def test_the_deployment_lists_unkeyed_names_and_prefixes():
    environ = {
        "SGLANG_SHARED_CACHE_MODE": "on",
        "SGLANG_RING": "a",
        "SGLANG_RINGS": "b",
    }
    assert cache.keyed_environment(environ) == environ
    unkeyed = {cache.ENV_UNKEYED: " SGLANG_SHARED_CACHE_* , SGLANG_RING ,"}
    assert cache.keyed_environment({**environ, **unkeyed}) == {"SGLANG_RINGS": "b"}
    with pytest.raises(ValueError, match="every env name"):
        cache.keyed_environment({**environ, cache.ENV_UNKEYED: "SGLANG_RING,*"})


def test_real_server_args_are_keyed_by_field_without_the_bundle_flags():
    import msgspec.structs

    from sglang.srt.server_args import ServerArgs

    def server_args(**overrides):
        fields = dict(model_path="/mnt/ckpt", tp_size=16, port=30000, host="10.0.0.1")
        return ServerArgs(**{**fields, **overrides})

    seed = server_args(unsafe_save_torch_compile_bundle="/dev/shm/c/lowered")
    consumer = server_args(unsafe_load_torch_compile_bundle="/dev/shm/c/lowered")
    fields = {field.name for field in msgspec.structs.fields(seed)}

    keyed, unstable = cache.keyed_server_args(seed)
    assert keyed["tp_size"] == 16
    assert keyed["model_path"] == "/mnt/ckpt"
    assert not {cache.SAVE_FLAG, cache.LOAD_FLAG, "port", "host"} & set(keyed)
    assert set(keyed) == fields - cache.VOLATILE_SERVER_ARGS
    assert cache.keyed_server_args(consumer) == (keyed, unstable)
    assert (
        cache.keyed_server_args(
            server_args(unsafe_load_torch_compile_bundle="/x", tp_size=8)
        )[0]
        != keyed
    )
    # Every volatile name is a real field, so none silently stops matching.
    assert cache.VOLATILE_SERVER_ARGS <= fields


def test_real_server_args_key_unstable_values_by_type():
    from sglang.srt.server_args import ServerArgs

    server_args = ServerArgs(model_path="/mnt/ckpt", api_key="secret", random_seed=7)
    server_args.json_model_override_args = object()

    keyed, unstable = cache.keyed_server_args(server_args)
    assert unstable == ["json_model_override_args"]
    assert keyed["json_model_override_args"] == "<unstable object>"
    assert not {"api_key", "random_seed"} & set(keyed)


def test_server_args_must_be_a_msgspec_struct():
    with pytest.raises(TypeError, match="msgspec Struct"):
        cache.keyed_server_args(types.SimpleNamespace(tp_size=16))


def _components_for(environ):
    return dict(COMPONENTS, environment=cache.keyed_environment(environ))


def test_output_treespec_codec_round_trips_and_rejects_unknown_nodes():
    treedef = _graph().interface.output_treespec
    encoded = json.loads(json.dumps(cache.encode_treedef(treedef)))
    assert cache.decode_treedef(encoded) == treedef

    class Foreign:
        pass

    tree_util.register_pytree_node(Foreign, lambda _: ((), None), lambda *_: Foreign())
    Foreign.__module__ = "elsewhere"
    with pytest.raises(cache.UnsupportedTreedef):
        cache.encode_treedef(tree_util.tree_structure((Foreign(),)))
    with pytest.raises(cache.UnsupportedTreedef):
        cache.decode_treedef({"type": "os:system", "aux": None, "children": []})


def test_the_deployment_packages_may_supply_output_node_types(monkeypatch):
    class Downstream:
        pass

    tree_util.register_pytree_node(
        Downstream, lambda _: ((), None), lambda *_: Downstream()
    )
    Downstream.__module__ = "my_models.outputs"
    treedef = tree_util.tree_structure((Downstream(),))
    with pytest.raises(cache.UnsupportedTreedef):
        cache.encode_treedef(treedef)
    monkeypatch.setenv(source_digest.PACKAGES_ENV, "other, my_models")
    assert cache.encode_treedef(treedef)["children"][0]["type"] == (
        "my_models.outputs:" + Downstream.__qualname__
    )


def test_save_then_load_round_trips_the_graph(tmp_path):
    saver = _store(tmp_path, cache.Mode.SAVE)
    assert saver.lookup(_request()) is None
    assert saver.wants_record()
    saver.record(_request(), _graph())
    assert saver.stats["saved"] == 1

    loader = _store(tmp_path, cache.Mode.LOAD)
    _assert_same_graph(loader.lookup(_request()), _graph())
    assert loader.lookup(_request(static_inputs=())) is None
    assert loader.stats["loaded"] == 1


@pytest.mark.parametrize("origin", [None, (1, 0, 0), [0, 1]])
def test_the_subslice_origin_round_trips(tmp_path, origin):
    # Serialized executables omit it; direct_compile loads with it.
    _store(tmp_path, cache.Mode.SAVE).record(_request(), _graph(subslice_origin=origin))
    loaded = _store(tmp_path, cache.Mode.LOAD).lookup(_request())
    _assert_same_graph(loaded, _graph(subslice_origin=origin))


def test_graphs_with_an_unencodable_subslice_origin_are_not_written(tmp_path):
    store = _store(tmp_path, cache.Mode.SAVE)
    store.record(_request(), _graph(subslice_origin=object()))
    assert store.stats["saved"] == 0
    assert not list(tmp_path.glob("*/*.exe"))


def _corrupt_executable(executable_path, meta_path):
    executable_path.write_bytes(executable_path.read_bytes()[:-1])


def _missing_executable(executable_path, meta_path):
    executable_path.unlink()


def _corrupt_json(executable_path, meta_path):
    meta_path.write_text("{not json")


def _edit_meta(**changes):
    def corrupt(executable_path, meta_path):
        meta = json.loads(meta_path.read_text())
        meta.update(changes)
        meta_path.write_text(json.dumps(meta))

    return corrupt


@pytest.mark.parametrize(
    "corrupt",
    [
        _corrupt_executable,
        _missing_executable,
        _corrupt_json,
        _edit_meta(format=cache.FORMAT_VERSION + 1),
        _edit_meta(graph_key="0" * 64),
        _edit_meta(process_key="0" * 64),
        _edit_meta(output_treespec_str="PyTreeDef(*)"),
        _edit_meta(output_treespec={"type": "os:system", "aux": None, "children": []}),
    ],
    ids=[
        "truncated-executable",
        "missing-executable",
        "bad-json",
        "format-version",
        "graph-key",
        "process-key",
        "treespec-mismatch",
        "treespec-type",
    ],
)
def test_load_rejects_inconsistent_entries_with_an_error_then_replaces_them(
    tmp_path, corrupt, caplog
):
    _store(tmp_path, cache.Mode.SAVE).record(_request(), _graph())
    (executable_path,) = tmp_path.glob("*/*.exe")
    corrupt(executable_path, executable_path.with_suffix(".json"))
    loader = _store(tmp_path, cache.Mode.LOAD)
    with caplog.at_level(logging.ERROR, logger=cache.__name__):
        assert loader.lookup(_request()) is None
    # The fork's loggers default to INFO, so the save above is captured too.
    (record,) = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert "rejected the stored entry" in record.getMessage()
    assert _request().graph in record.getMessage()
    assert loader.stats["rejected"] == 1
    # The graph is traced and compiled, and its entry written again.
    loader.record(_request(), _graph())
    _assert_same_graph(_store(tmp_path, cache.Mode.LOAD).lookup(_request()), _graph())


def test_changed_process_component_misses_with_one_log(tmp_path, caplog):
    seed = _components_for({"TPU_PREMAPPED_BUFFER_SIZE": "1"})
    _store(tmp_path, cache.Mode.SAVE, components=seed).record(_request(), _graph())
    (process_file,) = tmp_path.glob("*/process.json")
    written = json.loads(process_file.read_text())
    assert written["format"] == cache.FORMAT_VERSION
    # Digests only, never raw values.
    assert written["components"] == dict(cache.ProcessKey.of(seed).components)

    consumer = _components_for({"TPU_PREMAPPED_BUFFER_SIZE": "2"})
    loader = _store(tmp_path, cache.Mode.LOAD, components=consumer)
    with caplog.at_level(logging.INFO, logger=cache.__name__):
        assert loader.lookup(_request()) is None
    assert f"differs from those under {tmp_path}" in caplog.text


def test_load_stores_its_misses(tmp_path):
    stored = _request(key='["EXTEND",32,512]')
    missing = _request(key='["EXTEND",16,512]')
    _store(tmp_path, cache.Mode.SAVE).record(stored, _graph())
    loader = _store(tmp_path, cache.Mode.LOAD)
    _assert_same_graph(loader.lookup(stored), _graph())
    assert loader.lookup(missing) is None
    assert loader.wants_record()
    loader.record(missing, _graph())
    assert loader.stats == {"loaded": 1, "saved": 1, "rejected": 0}
    _assert_same_graph(_store(tmp_path, cache.Mode.LOAD).lookup(missing), _graph())


def test_save_mode_never_reuses_and_rewrites_existing_entries(tmp_path):
    _store(tmp_path, cache.Mode.SAVE).record(_request(), _graph())
    saver = _store(tmp_path, cache.Mode.SAVE)
    assert saver.lookup(_request()) is None
    saver.record(_request(), _graph(fingerprint="fp-2"))
    assert saver.stats == {"loaded": 0, "saved": 1, "rejected": 0}
    stored = _store(tmp_path, cache.Mode.LOAD).lookup(_request())
    assert stored.fingerprint == "fp-2"


def test_finish_logs_what_the_process_loaded_and_saved(tmp_path, caplog):
    _store(tmp_path, cache.Mode.SAVE).record(_request(), _graph())
    loader = _store(tmp_path, cache.Mode.LOAD)
    loader.lookup(_request())
    with caplog.at_level(logging.INFO, logger=cache.__name__):
        loader.finish()
    assert "warmup done" in caplog.text
    assert "loaded=1 saved=0 rejected=0" in caplog.text
    # A store never opened (no named graph compiled) logs nothing.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=cache.__name__):
        _store(tmp_path, cache.Mode.LOAD).finish()
    assert caplog.text == ""


def test_unkeyable_process_disables_the_store(tmp_path):
    def fail():
        raise RuntimeError("no device assignment")

    store = cache.FileLoweredGraphStore(cache.Mode.LOAD, tmp_path, components=fail)
    assert store.lookup(_request()) is None
    assert not store.wants_record()
    assert not list(tmp_path.iterdir())


def test_an_unwritable_root_disables_the_store_once(tmp_path):
    calls = []

    def components():
        calls.append(1)
        return COMPONENTS

    root = tmp_path / "file"
    root.write_text("not a directory")
    store = cache.FileLoweredGraphStore(cache.Mode.LOAD, root, components=components)
    assert store.lookup(_request()) is None
    assert not store.wants_record()
    assert calls == [1]


@pytest.fixture
def _serve_server_args(monkeypatch):
    """Serves the given ServerArgs from the runtime context, with the compiler
    tag and the source digest stubbed."""
    monkeypatch.setattr(cache, "_compiler_tag", lambda: _COMPILER_TAG)
    monkeypatch.setattr(cache, "sources_digest", lambda: ("src-digest", "manifest"))
    runtime_context = types.ModuleType("sglang.srt.runtime_context")
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", runtime_context)

    def serve(server_args):
        runtime_context.get_server_args = lambda: server_args

    return serve


def test_process_components_key_on_the_compiler_cache_key(
    monkeypatch, tmp_path, _serve_server_args
):
    from sglang.srt.server_args import ServerArgs

    monkeypatch.setenv(cache.ENV_SALT, "fresh")
    (tmp_path / "config.json").write_text("{}")
    server_args = ServerArgs(model_path=str(tmp_path), tp_size=16, port=30000)
    _serve_server_args(server_args)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True, raising=False)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 3, raising=False)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 16, raising=False)

    components = cache.collect_process_components()

    assert components["compiler"] == _COMPILER_TAG
    assert components["sources"] == "src-digest"
    assert set(components["versions"]) == {"torch", "tpu-sync-torch"}
    assert components["rank"] == {"rank": 3, "world_size": 16}
    assert components["server_args"] == cache.keyed_server_args(server_args)[0]
    assert components["server_args"]["tp_size"] == 16
    assert components["salt"] == "fresh"


@pytest.fixture
def _cache_key_tag():
    """Sets torch.compiler.config.cache_key_tag for one test. Its config
    module doesn't survive monkeypatch's undo, so restore it by hand."""
    import torch.compiler.config as compiler_config

    saved = compiler_config.cache_key_tag
    yield lambda value: setattr(compiler_config, "cache_key_tag", value)
    compiler_config.cache_key_tag = saved


def test_compiler_tag_adds_the_rank_assignment_first(monkeypatch, _cache_key_tag):
    sglang_tpu_compile_cache = types.ModuleType("sglang_tpu.compile_cache")
    calls = []

    def add_tpu_rank_assignment_to_compile_cache_key():
        calls.append("add")
        _cache_key_tag(_COMPILER_TAG)

    sglang_tpu_compile_cache.add_tpu_rank_assignment_to_compile_cache_key = (
        add_tpu_rank_assignment_to_compile_cache_key
    )
    monkeypatch.setitem(
        sys.modules, "sglang_tpu.compile_cache", sglang_tpu_compile_cache
    )
    _cache_key_tag("tpu-kernel-inputs-v1:abc")

    assert cache._compiler_tag() == _COMPILER_TAG
    assert calls == ["add"]


def test_source_hash_covers_data_files_but_not_bytecode(tmp_path):
    (tmp_path / "profiles").mkdir()
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "mod.py").write_text("x = 1\n")
    (tmp_path / "profiles" / "routing.json").write_text("{}")
    before = source_digest.hash_sources({"pkg": tmp_path})
    (tmp_path / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\0")
    assert source_digest.hash_sources({"pkg": tmp_path}) == before
    (tmp_path / "profiles" / "routing.json").write_text('{"expert": 3}')
    assert source_digest.hash_sources({"pkg": tmp_path}) != before


def test_the_build_manifest_supplies_the_digest_until_a_source_changes(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "mod.py").write_text("x = 1\n")
    roots = {"pkg": package}
    manifest = tmp_path / "sources.json"
    built = source_digest.write_manifest(manifest, roots)
    environ = {source_digest.MANIFEST_ENV: str(manifest)}

    assert source_digest.sources_digest(roots, environ) == (built, "manifest")
    # An overlay rewrites a file after the build: hash again.
    (package / "mod.py").write_text("x = 22\n")
    digest, origin = source_digest.sources_digest(roots, environ)
    assert origin == "hashed"
    assert digest == source_digest.hash_sources(roots) != built
    # No manifest, or an unreadable one, hashes too.
    assert source_digest.sources_digest(roots, {})[1] == "hashed"
    manifest.write_text("not json")
    assert source_digest.sources_digest(roots, environ)[1] == "hashed"


def test_the_deployment_names_its_own_source_packages(monkeypatch):
    monkeypatch.delenv(source_digest.PACKAGES_ENV, raising=False)
    assert set(source_digest.source_roots()) == set(source_digest.SOURCE_PACKAGES)
    monkeypatch.setenv(source_digest.PACKAGES_ENV, " json, ,email ")
    assert set(source_digest.source_roots()) == {
        *source_digest.SOURCE_PACKAGES,
        "json",
        "email",
    }
    # A named package that is not installed raises, which disables the store.
    monkeypatch.setenv(source_digest.PACKAGES_ENV, "no_such_package_xyz")
    with pytest.raises(RuntimeError, match="no_such_package_xyz"):
        source_digest.source_roots()


def test_a_package_split_over_directories_raises(monkeypatch, tmp_path):
    for part in ("a", "b"):
        (tmp_path / part / "split_package_xyz").mkdir(parents=True)
        monkeypatch.syspath_prepend(str(tmp_path / part))
    with pytest.raises(RuntimeError, match="split_package_xyz"):
        source_digest.source_roots(("split_package_xyz",))


def test_an_unreadable_model_config_disables_the_store(tmp_path, _serve_server_args):
    from sglang.srt.server_args import ServerArgs

    _serve_server_args(ServerArgs(model_path=str(tmp_path / "missing"), tp_size=16))

    with pytest.raises(OSError):
        cache.collect_process_components()


def test_weights_manifest_tracks_weight_files(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model-00001.safetensors").write_bytes(b"\0" * 4)
    before = cache._weights_manifest(tmp_path)
    assert before["weights"] == [("model-00001.safetensors", 4)]
    (tmp_path / "model-00001.safetensors").write_bytes(b"\0" * 8)
    assert cache._weights_manifest(tmp_path) != before


def test_the_bundle_flag_selects_the_mode(tmp_path):
    store = cache.create_lowered_graph_store(str(tmp_path), None)
    assert (store.mode, store.root) == (cache.Mode.SAVE, tmp_path)
    store = cache.create_lowered_graph_store(None, str(tmp_path))
    assert (store.mode, store.root) == (cache.Mode.LOAD, tmp_path)


def test_finish_remembers_whether_this_process_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_saved_any", False)
    # A store never opened, and one that only loaded, saved nothing.
    _store(tmp_path, cache.Mode.LOAD).finish()
    assert not cache.saved_any_entry()
    saver = _store(tmp_path, cache.Mode.SAVE)
    saver.record(_request(), _graph())
    saver.finish()
    assert cache.saved_any_entry()
    loader = _store(tmp_path, cache.Mode.LOAD)
    loader.lookup(_request())
    loader.finish()
    # Any warmup's saved entry counts.
    assert cache.saved_any_entry()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
