"""File-backed store that lets a warm TPU boot skip direct_compile tracing.

Driven by the torch.compile bundle flags under ``--tpu-compile-tracer
direct``: ``--unsafe-save-torch-compile-bundle DIR`` saves graphs under
``DIR`` and ``--unsafe-load-torch-compile-bundle DIR`` loads them, tracing,
compiling and saving the graphs it misses. ``WarmupQueue`` builds the store
for its warmup (``create_lowered_graph_store``) and finishes it once the
warmup succeeded. Without either flag there is no store and compilation is
unchanged. Only ``direct_compile`` functions with a ``name=`` are stored. An
unreadable, inconsistent or misfitting entry is a miss: the graph is traced.

A deployment that ships ``DIR`` between hosts can ask ``saved_any_entry()``
after warmup whether this process added entries. One whose own packages define
models names them in ``SGLANG_TPU_LOWERED_GRAPH_SOURCE_PACKAGES``
(``source_digest``), and one whose own env vars never shape a graph lists them
in ``SGLANG_TPU_LOWERED_GRAPH_UNKEYED_ENV``.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import importlib
import importlib.metadata
import json
import os
import threading
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import msgspec
import msgspec.structs
from torch._inductor.codecache import write_atomic

from sglang_tpu._logging import get_logger
from sglang_tpu.compilation.graph_identity import stable_repr
from sglang_tpu.compilation.lowered_graph import (
    NO_CALLBACKS,
    LoweredGraph,
    LoweredGraphRequest,
)
from sglang_tpu.compilation.source_digest import (
    MANIFEST_ENV,
    PACKAGES_ENV,
    extra_source_packages,
    sources_digest,
)
from sglang_tpu.compilation.spmd_compiler import ExecutableInterface, InputSpec

logger = get_logger(__name__)

# Change it to start a fresh namespace, e.g. after re-uploading weights with
# the same names and sizes. Keyed on its own, outside the keyed environment.
ENV_SALT = "SGLANG_TPU_LOWERED_GRAPH_KEY_SALT"
# Comma-separated env names that never shape a graph in this deployment, e.g.
# how its own cache moves bytes; a trailing ``*`` matches a prefix.
ENV_UNKEYED = "SGLANG_TPU_LOWERED_GRAPH_UNKEYED_ENV"
# Server-arg names of the bundle flags.
SAVE_FLAG = "unsafe_save_torch_compile_bundle"
LOAD_FLAG = "unsafe_load_torch_compile_bundle"

# Bump on any change to the key composition or the on-disk format.
FORMAT_VERSION = 2

_LOG = "[lowered-graph-cache]"

# Stacks whose versions select the code that turns the traced ops into MLIR
# and that cache_key_tag doesn't already carry (it has jax, jaxlib, libtpu and
# torch_tpu).
VERSIONED_DISTRIBUTIONS = ("torch", "tpu-sync-torch")
ENV_PREFIXES = ("SGLANG_", "TORCH_TPU_", "LIBTPU_", "XLA_", "JAX_", "TPU_")
# Words that name a process's address or a secret, never a graph switch.
# Matched against underscore-separated words so IPC is not IP. HOST, TOKEN,
# DIR and PATH are not here: they name graph switches too
# (SGLANG_TPU_REQ_TO_TOKEN_ON_HOST), so a placement name with one of them is
# listed by name below or by the deployment (ENV_UNKEYED). An unneeded name
# only costs a miss, a missing one can reuse a wrong graph.
# TPU_VISIBLE_DEVICES and SGLANG_TPU_PJRT_RANK{,_BASE} stay keyed: in a
# singleton torch world (TP1 --dp-size replicas, single-chip engines without
# a session world group) they alone tell engines on different chips apart, and
# an entry's executable runs only on the chips its subslice origin names.
ENV_PLACEMENT_WORDS = frozenset(
    {
        "ADDR",
        "ADDRESS",
        "ADDRESSES",
        "CREDENTIAL",
        "CREDENTIALS",
        "ENDPOINT",
        "ENDPOINTS",
        "HOSTNAME",
        "HOSTNAMES",
        "IP",
        "IPS",
        "NAMESPACE",
        "PASSWORD",
        "POD",
        "PORT",
        "PORTS",
        "SECRET",
        "URL",
        "URLS",
    }
)
# Env names whose value is new on every launch and never shapes a graph.
# SGLANG_RUN_ID is a timestamp and random number set by Engine at startup.
ENV_PER_LAUNCH = frozenset({"SGLANG_RUN_ID"})
# Env names GKE sets per TPU slice or host. TPU_NAME is the slice's own name,
# so keying it gives every gang its own namespace. TPU_WORKER_ID is the host's
# index in the slice; a deployment that orders its hosts can give that host
# another node rank, so keying it splits gangs whose hosts are ordered
# differently. TPU_LOG_DIR only says where logs go. The other names GKE and
# the serving operator inject per slice or host (TPU_WORKER_HOSTNAMES,
# TPU_PROCESS_ADDRESSES, TPU_PROCESS_PORT, TPU_SOURCE_POD_UID) drop by
# placement word. The topology a graph depends on is keyed through the rank
# and the global device assignment.
ENV_PER_SLICE = frozenset({"TPU_NAME", "TPU_WORKER_ID", "TPU_LOG_DIR"})
# Env names GKE injects for some slices and restarts but not for equivalent
# ones. The hardware they describe is pinned by the libtpu and torch-tpu
# versions, the server args and the device assignment, and every hit is
# checked against the live inputs.
ENV_GKE_DISCOVERY = frozenset(
    {
        "TPU_ACCELERATOR_TYPE",
        "TPU_CHIPS_PER_HOST_BOUNDS",
        "TPU_HOST_BOUNDS",
        "TPU_TOPOLOGY",
        "TPU_TOPOLOGY_ALT",
        "TPU_TOPOLOGY_WRAP",
    }
)
# Cache-delivery settings: they choose how bytes move and where they land,
# never what is traced, so a host that publishes and one that only restores
# share a namespace. The Tier-3 root contains COMPILATION_CACHE and is dropped
# with the Tier-2 names; the salt is keyed explicitly. A deployment adds its
# own (ENV_UNKEYED).
ENV_DELIVERY = frozenset({"TORCH_TPU_TIER2_NAME"})
# The store's own settings: the salt is keyed on its own, the sources digest
# covers what the packages setting and the manifest select, and the keyed env
# shows what the unkeyed setting drops.
ENV_STORE_SETTINGS = frozenset({ENV_SALT, ENV_UNKEYED, PACKAGES_ENV, MANIFEST_ENV})
_UNKEYED_ENV_NAMES = (
    ENV_PER_LAUNCH
    | ENV_PER_SLICE
    | ENV_GKE_DISCOVERY
    | ENV_DELIVERY
    | ENV_STORE_SETTINGS
)
# ServerArgs fields that place, address or authenticate a process, or only
# say where its logs and metrics go. Everything else is keyed: an unneeded
# field only costs a miss, a missing one can reuse a wrong graph. Each name
# must be a ServerArgs field (a test checks). A seed and its consumers differ
# in these: they run on other hosts, ports and pods.
VOLATILE_SERVER_ARGS = frozenset(
    {
        # Secrets.
        "admin_api_key",
        "api_key",
        "ssl_ca_certs",
        "ssl_certfile",
        "ssl_keyfile",
        "ssl_keyfile_password",
        # Addresses, ports, hostnames and endpoints of this process and its
        # peers.
        "decoupled_spec_bind_endpoint",
        "decoupled_spec_connect_endpoints",
        "disaggregation_bootstrap_port",
        "dist_init_addr",
        "encoder_bootstrap_port",
        "engine_info_bootstrap_port",
        "gated_launch_port",
        "grpc_port",
        "host",
        "load_publish_endpoint",
        "nccl_port",
        "otlp_traces_endpoint",
        "port",
        "remote_instance_weight_loader_seed_instance_ip",
        "remote_instance_weight_loader_seed_instance_service_port",
        "remote_instance_weight_loader_send_weights_group_ports",
        "smg_http_sidecar_port",
        "torch_tpu_slicebuilder_addresses",
        "tpu_worker_hostnames",
        "weight_cache_socket",
        # Per-process identity of the forward-pass metrics publisher.
        "forward_pass_metrics_ipc_name",
        "forward_pass_metrics_worker_id",
        # Where files land: dumps after a crash, uploads, metrics files,
        # request logs and Hugging Face downloads (the weights themselves are
        # keyed through model_configs).
        "crash_dump_folder",
        "download_dir",
        "export_metrics_to_file_dir",
        "file_storage_path",
        "log_requests_target",
        # Log verbosity of the server and its HTTP layer.
        "log_level",
        "log_level_http",
        # Randomized per launch; it only seeds host RNGs.
        "random_seed",
        # Where the store lives and whether this process saves or loads; a
        # seed saves while its consumers load.
        LOAD_FLAG,
        SAVE_FLAG,
    }
)
# Only these modules, and the deployment's own source packages, may supply
# output pytree node types: their sources are keyed.
TREEDEF_MODULE_PREFIXES = ("sglang.", "sglang_tpu.")


class Mode(enum.Enum):
    """``save`` traces and compiles every graph and stores it; ``load`` reuses
    stored graphs and traces, compiles and stores the rest."""

    SAVE = "save"
    LOAD = "load"


class UnsupportedTreedef(ValueError):
    """The output pytree cannot be stored without pickle."""


# --- Keys -------------------------------------------------------------------


def _canonical(value: Any) -> Any:
    """Convert a value to deterministic JSON, or raise TypeError."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return f"{type(value).__qualname__}.{value.name}"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(repr(item) for item in value)
    return f"{type(value).__qualname__}:{stable_repr(value)}"


def _digest(value: Any) -> str:
    encoded = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class ProcessKey:
    """Process-wide key part. ``components`` holds digests, never raw values."""

    digest: str
    components: Mapping[str, str]

    @classmethod
    def of(cls, components: Mapping[str, Any]) -> ProcessKey:
        component_digests = {name: _digest(value) for name, value in components.items()}
        return cls(
            digest=_digest([FORMAT_VERSION, component_digests]),
            components=component_digests,
        )


def graph_key(process: ProcessKey, request: LoweredGraphRequest) -> str:
    return _digest([FORMAT_VERSION, process.digest, request])


def _unkeyed_names(
    environ: Mapping[str, str],
) -> tuple[frozenset[str], tuple[str, ...]]:
    """The names and prefixes ``ENV_UNKEYED`` lists. A bare ``*`` raises: it
    would leave no env keyed."""
    names, prefixes = set(), []
    for item in environ.get(ENV_UNKEYED, "").split(","):
        item = item.strip()
        if item == "*":
            raise ValueError(f"{ENV_UNKEYED} cannot drop every env name")
        if item.endswith("*"):
            prefixes.append(item[:-1])
        elif item:
            names.add(item)
    return frozenset(names), tuple(prefixes)


def keyed_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """Env vars that can change tracing, minus placement, delivery and secrets."""
    unkeyed_names, unkeyed_prefixes = _unkeyed_names(environ)
    keyed = {}
    for name, value in environ.items():
        if not name.startswith(ENV_PREFIXES):
            continue
        if name in _UNKEYED_ENV_NAMES or "COMPILATION_CACHE" in name:
            continue
        if name in unkeyed_names or name.startswith(unkeyed_prefixes):
            continue
        if ENV_PLACEMENT_WORDS.intersection(name.split("_")):
            continue
        keyed[name] = value
    return keyed


def keyed_server_args(server_args: Any) -> tuple[dict[str, Any], list[str]]:
    """Return keyed server-arg values and the fields without a stable value.

    ``ServerArgs`` is a msgspec Struct whose fields, underscore-prefixed ones
    included, are the operator's input; ``vars()`` of it holds only resolution
    bookkeeping. Anything else raises, which disables the store. Unstable
    fields are keyed by type only and reported, so a deployment can see what
    the key cannot distinguish.
    """
    if not isinstance(server_args, msgspec.Struct):
        raise TypeError(
            f"server args must be a msgspec Struct, got {type(server_args).__qualname__}"
        )
    names = [field.name for field in msgspec.structs.fields(server_args)]
    keyed: dict[str, Any] = {}
    unstable: list[str] = []
    for name in sorted(names):
        if name in VOLATILE_SERVER_ARGS:
            continue
        value = getattr(server_args, name)
        try:
            keyed[name] = _canonical(value)
        except TypeError:
            keyed[name] = f"<unstable {type(value).__qualname__}>"
            unstable.append(name)
    return keyed, unstable


def _distribution_versions(names: Iterable[str]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "<missing>"
    return versions


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _weights_manifest(model_path: Path) -> dict[str, Any]:
    """Name and size of every weight file, plus the shard index.

    Graphs can bake values read from weights (e.g. a ``.item()`` scale), and
    hashing the weights themselves is too slow. A re-upload with identical
    names and sizes still needs the salt to catch it.
    """
    weights = sorted(
        (path.name, path.stat().st_size)
        for path in model_path.glob("*")
        if path.suffix in (".safetensors", ".bin", ".pt") and path.is_file()
    )
    index = model_path / "model.safetensors.index.json"
    return {
        "index": _file_digest(index) if index.is_file() else None,
        "weights": weights,
    }


def _compiler_tag() -> str:
    """This process's ``torch.compiler.config.cache_key_tag``, with the WORLD
    device assignment in it.

    TorchTPU's platform setup puts the kernel inputs (sources and the ``envs``
    they read, LIBTPU_INIT_ARGS, XLA_FLAGS and lowering flags, the jax,
    jaxlib, libtpu and torch_tpu versions, PCI ids) in it, and a downstream
    platform hook can add its own sources.
    ``add_tpu_rank_assignment_to_compile_cache_key`` adds the rank and the
    global device ids in rank order, which MLIR collectives embed
    (replica_groups) and which differ with the physical host that holds rank
    0; it validates them, is idempotent, and raises on an invalid assignment,
    which disables the store.
    """
    import torch.compiler.config as compiler_config

    from sglang_tpu.compile_cache import add_tpu_rank_assignment_to_compile_cache_key

    add_tpu_rank_assignment_to_compile_cache_key()
    tag = str(compiler_config.cache_key_tag or "")
    if not tag:
        raise RuntimeError("torch.compiler.config.cache_key_tag is empty")
    return tag


def collect_process_components() -> dict[str, Any]:
    """Everything process-wide that can change a lowered graph.

    Raises when a component cannot be determined: without it the key would be
    weaker than the graph, so the caller disables the store.
    """
    import torch.distributed as dist

    from sglang.srt.runtime_context import get_server_args

    server_args = get_server_args()
    keyed_args, unstable = keyed_server_args(server_args)
    if unstable:
        logger.warning(
            "%s server args without a stable value, keyed by type only: %s",
            _LOG,
            ", ".join(unstable),
        )
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    # An unreadable config.json raises: the checkpoint's identity would rest
    # on its path alone.
    model_configs = {
        name: {
            "config": _file_digest(Path(path) / "config.json"),
            **_weights_manifest(Path(path)),
        }
        for name, path in (
            ("model_path", server_args.model_path),
            ("speculative_draft_model_path", server_args.speculative_draft_model_path),
        )
        if path
    }
    sources, source_origin = sources_digest()
    logger.info(
        "%s source digest from the %s sources (deployment packages: %s)",
        _LOG,
        source_origin,
        ", ".join(extra_source_packages()) or "none",
    )
    return {
        "sources": sources,
        "compiler": _compiler_tag(),
        "versions": _distribution_versions(VERSIONED_DISTRIBUTIONS),
        "environment": keyed_environment(os.environ),
        "server_args": keyed_args,
        "model_configs": model_configs,
        "rank": {"rank": rank, "world_size": world_size},
        "salt": os.environ.get(ENV_SALT, ""),
    }


# --- Output treespec codec ----------------------------------------------------

_BUILTIN_NODES: dict[str, type] = {
    "tuple": tuple,
    "list": list,
    "dict": dict,
    "None": type(None),
}
_BUILTIN_NAMES = {node: name for name, node in _BUILTIN_NODES.items()}


def _treedef_modules() -> tuple[str, ...]:
    return (
        *TREEDEF_MODULE_PREFIXES,
        *(f"{package}." for package in extra_source_packages()),
    )


def _type_name(node_type: type) -> str:
    if node_type in _BUILTIN_NAMES:
        return _BUILTIN_NAMES[node_type]
    module = node_type.__module__
    if not module.startswith(_treedef_modules()):
        raise UnsupportedTreedef(f"node type {module}.{node_type.__qualname__}")
    return f"{module}:{node_type.__qualname__}"


def _resolve_type(name: str) -> type:
    if name in _BUILTIN_NODES:
        return _BUILTIN_NODES[name]
    module_name, _, qualname = name.partition(":")
    if not module_name.startswith(_treedef_modules()):
        raise UnsupportedTreedef(f"node type {name}")
    value: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        value = getattr(value, part)
    if not isinstance(value, type):
        raise UnsupportedTreedef(f"{name} is not a type")
    return value


def _encode_aux(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return {"enum": _type_name(type(value)), "name": value.name}
    if isinstance(value, tuple):
        return {"tuple": [_encode_aux(item) for item in value]}
    if isinstance(value, list):
        return {"list": [_encode_aux(item) for item in value]}
    raise UnsupportedTreedef(f"node data of type {type(value).__qualname__}")


def _decode_aux(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if "enum" in value:
        return _resolve_type(value["enum"])[value["name"]]
    if "tuple" in value:
        return tuple(_decode_aux(item) for item in value["tuple"])
    return [_decode_aux(item) for item in value["list"]]


def encode_treedef(treedef: Any) -> dict[str, Any]:
    """Encode a JAX treedef as JSON; round-trips are checked by the caller."""
    node = treedef.node_data()
    if node is None:
        return {"leaf": True}
    node_type, aux = node
    return {
        "type": _type_name(node_type),
        "aux": _encode_aux(aux),
        "children": [encode_treedef(child) for child in treedef.children()],
    }


def decode_treedef(encoded: Mapping[str, Any]) -> Any:
    from jax import tree_util

    if encoded.get("leaf"):
        node = None
        children = []
    else:
        node = (_resolve_type(encoded["type"]), _decode_aux(encoded["aux"]))
        children = [decode_treedef(child) for child in encoded["children"]]
    return tree_util.PyTreeDef.from_node_data_and_children(
        tree_util.default_registry, node, children
    )


# --- Store ------------------------------------------------------------------


class FileLoweredGraphStore:
    """Store executables as ``<root>/<process key>/<graph key>.{exe,json}``.

    The JSON is written last and names the executable's hash, so a partial write
    is a miss. An entry that exists but cannot be used (unreadable, written
    under another key or format, a hash or treespec mismatch) is logged as an
    error naming the graph and the reason; the graph is then traced, compiled
    and stored again.
    """

    def __init__(
        self,
        mode: Mode,
        root: Path,
        *,
        components: Callable[[], Mapping[str, Any]] = collect_process_components,
    ) -> None:
        self.mode = mode
        self.root = root
        self._components = components
        self._lock = threading.Lock()
        self._process: ProcessKey | None = None
        self._namespace: Path | None = None
        self._disabled_reason: str | None = None
        self.stats = {"loaded": 0, "saved": 0, "rejected": 0}

    # Called by direct_compile and WarmupQueue.

    def lookup(self, request: LoweredGraphRequest) -> LoweredGraph | None:
        """A stored graph to reuse in ``load`` mode, or None to trace."""
        if self.mode is not Mode.LOAD or not self._open():
            return None
        key = graph_key(self._process, request)
        _, meta_path = self._paths(key)
        if not meta_path.exists():
            return None
        try:
            # A metadata file without its executable is a broken entry too.
            graph = self._read(key)
        except Exception as error:
            self._count("rejected")
            logger.error(
                "%s rejected the stored entry %s for %s: %s; tracing it",
                _LOG,
                key,
                request.graph,
                error,
            )
            return None
        self._count("loaded")
        return graph

    def wants_record(self) -> bool:
        """Whether recording would store anything; serializing the executable
        for ``record`` costs a copy of it, so ``direct_compile`` asks first."""
        return self._open()

    def record(self, request: LoweredGraphRequest, graph: LoweredGraph) -> None:
        """Store a graph this process traced and compiled: every graph in
        ``save``, and in ``load`` the misses and rejected entries."""
        if self._open():
            self._write(graph_key(self._process, request), request, graph)

    def finish(self) -> None:
        """Log what this process loaded and saved, and remember whether it
        saved anything (``saved_any_entry``).

        The warmup queue calls it once its warmup succeeded, after every
        other completion. Local: no collectives.
        """
        global _saved_any
        if self._namespace is not None:
            logger.info(
                "%s warmup done: namespace=%s loaded=%d saved=%d rejected=%d",
                _LOG,
                self._namespace,
                self.stats["loaded"],
                self.stats["saved"],
                self.stats["rejected"],
            )
        _saved_any = _saved_any or self.stats["saved"] > 0

    # Internals.

    def _count(self, outcome: str) -> None:
        # record() runs on the shared compiler's serializer threads.
        with self._lock:
            self.stats[outcome] += 1

    def _open(self) -> bool:
        """Compute the process key once; False disables the store."""
        with self._lock:
            if self._disabled_reason is not None:
                return False
            if self._namespace is not None:
                return True
            try:
                self._process = ProcessKey.of(self._components())
            except Exception as error:
                self._disabled_reason = f"cannot key this process: {error!r}"
                logger.warning("%s disabled: %s", _LOG, self._disabled_reason)
                return False
            namespace = self.root / self._process.digest[:32]
            if self.mode is not Mode.SAVE and not namespace.exists():
                # Diff the process.json files to find the component that changed.
                logger.info(
                    "%s no stored graphs for this process: namespace %s differs "
                    "from those under %s",
                    _LOG,
                    namespace.name,
                    self.root,
                )
            try:
                self._write_process_file(namespace)
            except OSError as error:
                self._disabled_reason = f"cannot create {namespace}: {error!r}"
                logger.warning("%s disabled: %s", _LOG, self._disabled_reason)
                return False
            self._namespace = namespace
            logger.info("%s mode=%s namespace=%s", _LOG, self.mode.value, namespace)
            return True

    def _write_process_file(self, namespace: Path) -> None:
        namespace.mkdir(parents=True, exist_ok=True)
        process_file = namespace / "process.json"
        if process_file.exists():
            return
        write_atomic(
            str(process_file),
            json.dumps(
                {
                    "format": FORMAT_VERSION,
                    "process_key": self._process.digest,
                    "components": dict(self._process.components),
                },
                indent=1,
                sort_keys=True,
            ).encode(),
        )

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self._namespace / f"{key}.exe", self._namespace / f"{key}.json"

    def _read(self, key: str) -> LoweredGraph:
        executable_path, meta_path = self._paths(key)
        meta = json.loads(meta_path.read_text())
        if meta.get("format") != FORMAT_VERSION:
            raise ValueError(f"format {meta.get('format')} != {FORMAT_VERSION}")
        if (
            meta.get("graph_key") != key
            or meta.get("process_key") != self._process.digest
        ):
            raise ValueError("entry was written under a different key")
        executable = executable_path.read_bytes()
        if hashlib.sha256(executable).hexdigest() != meta["executable_sha256"]:
            raise ValueError("executable hash does not match its metadata")
        treedef = decode_treedef(meta["output_treespec"])
        if str(treedef) != meta["output_treespec_str"]:
            raise ValueError("output treespec does not round-trip")
        return LoweredGraph(
            interface=ExecutableInterface(
                input_digest=str(meta["input_digest"]),
                input_specs=tuple(
                    InputSpec(tuple(shape), str(dtype), tuple(stride), int(offset))
                    for shape, dtype, stride, offset in meta["input_specs"]
                ),
                argument_layouts=[
                    [int(dimension) for dimension in layout]
                    for layout in meta["argument_layouts"]
                ],
                model_indices=tuple(int(index) for index in meta["model_indices"]),
                output_treespec=treedef,
                callback_factories=NO_CALLBACKS,
            ),
            executable=executable,
            fingerprint=meta["fingerprint"],
            subslice_origin=_decode_aux(meta["subslice_origin"]),
        )

    def _write(
        self, key: str, request: LoweredGraphRequest, graph: LoweredGraph
    ) -> None:
        # direct_compile records only graphs without callbacks.
        interface = graph.interface
        try:
            encoded = encode_treedef(interface.output_treespec)
            if decode_treedef(encoded) != interface.output_treespec:
                raise UnsupportedTreedef("encoding does not round-trip")
        except (UnsupportedTreedef, AttributeError, ImportError) as error:
            logger.info("%s not storing %s: %s", _LOG, request.graph, error)
            return
        try:
            subslice_origin = _encode_aux(graph.subslice_origin)
        except UnsupportedTreedef as error:
            logger.info(
                "%s not storing %s: subslice origin %s", _LOG, request.graph, error
            )
            return
        executable_path, meta_path = self._paths(key)
        write_atomic(str(executable_path), graph.executable)
        meta = {
            "format": FORMAT_VERSION,
            "graph_key": key,
            "process_key": self._process.digest,
            "name": request.name,
            "graph": request.graph,
            "executable_sha256": hashlib.sha256(graph.executable).hexdigest(),
            "executable_size": len(graph.executable),
            "fingerprint": graph.fingerprint,
            "subslice_origin": subslice_origin,
            "input_digest": interface.input_digest,
            "input_specs": interface.input_specs,
            "argument_layouts": interface.argument_layouts,
            "model_indices": interface.model_indices,
            "output_treespec": encoded,
            "output_treespec_str": str(interface.output_treespec),
        }
        write_atomic(str(meta_path), json.dumps(meta, sort_keys=True))
        self._count("saved")
        logger.info(
            "%s saved %s (%d bytes)", _LOG, request.graph, len(graph.executable)
        )


def create_lowered_graph_store(
    save_dir: str | None, load_dir: str | None
) -> FileLoweredGraphStore:
    """The store for one warmup under the direct tracer, given one bundle
    flag (the server args reject both at once)."""
    if save_dir:
        return FileLoweredGraphStore(Mode.SAVE, Path(save_dir))
    return FileLoweredGraphStore(Mode.LOAD, Path(load_dir))


# Whether a store in this process saved an entry; finish() sets it after each
# successful warmup.
_saved_any = False


def saved_any_entry() -> bool:
    """Whether a store in this process saved an entry during a warmup."""
    return _saved_any
