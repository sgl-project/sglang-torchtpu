"""Shared macros for sglang-torchtpu Bazel targets."""

load("@rules_python//python:defs.bzl", "py_test")

# Environment the TPU server launcher (benchmark/tpu/launch_server.sh) and
# torch_tpu read from the host at runtime. Bazel's clean test env defines none
# of them: launch_server.sh runs under `set -u` and derives cache dirs from
# $HOME (torch_tpu tier-3 compile cache, HF cache), and the scheduler needs
# the topology variables to discover the TPU chips. Inheriting them in the
# rule keeps every e2e target self-contained -- no --config=tpu, no per-user
# bazelrc lines, no --test_env flags on the command line.
_TPU_TEST_ENV_INHERIT = [
    "HOME",
    "HF_HOME",
    "TPU_ACCELERATOR_TYPE",
    "TPU_CHIPS_PER_HOST_BOUNDS",
    "TPU_HOST_BOUNDS",
    "TPU_PROCESS_ADDRESSES",
    "TPU_PROCESS_PORT",
    "TPU_RUNTIME_METRICS_PORTS",
    "TPU_SKIP_MDS_QUERY",
    "TPU_TOPOLOGY",
    "TPU_TOPOLOGY_ALT",
    "TPU_TOPOLOGY_WRAP",
    "TPU_WORKER_HOSTNAMES",
    "TPU_WORKER_ID",
    "CHIPS_PER_HOST_BOUNDS",
    "HOST_BOUNDS",
    # libtpu's VBAR control-service endpoint (e.g. unix:///var/run/tpu-plugin/
    # vbar.sock on GCE TPU hosts). Without it libtpu falls back to a default
    # tcp [::]:8353 where nothing listens, and client init dies with
    # "TPU initialization failed: Failed to connect to [::]:8353".
    # torch_tpu's own .bazelrc tpu_envvars passes this through too, noting the
    # value can be host-specific.
    "VBAR_CONTROL_SERVICE_URL",
]

# Tags every TPU e2e target needs. The server subprocess writes the torch_tpu
# compile cache under $HOME and needs real TPU devices and the network, so the
# sandbox must be off; `manual` keeps these out of `bazel test //...` wildcard
# builds (TPU tests are scheduled by CI on TPU-equipped runners). `exclusive`
# serializes tests within a Bazel invocation until the runner can assign
# separate TPU devices to concurrent tests.
_TPU_TEST_TAGS = [
    "exclusive",
    "manual",
    "no-sandbox",
    "requires-net:external",
    "requires-tpu",
]

def sglang_tpu_e2e_test(name, env = None, tags = None, **kwargs):
    """py_test for a TPU e2e test that boots a real sglang server.

    Wraps py_test with everything such a target always needs:
      * env_inherit for HOME / HF cache / TPU topology variables from the host
      * PYTHONUNBUFFERED=1 so server logs stream unbuffered
      * the Python 3.12 bootstrap exports its venv interpreter as PYTHON
      * exclusive + manual + no-sandbox + requires-* execution tags
      * eternal timeout (24 hours via the repository .bazelrc) for model load,
        torch.compile, and long evaluations; HEALTH_TIMEOUT only bounds startup

    Callers add the test-specific pieces: srcs/main, deps, data (typically
    //benchmark/tpu:runtime), hardware-specific tags (e.g. requires-tpu-v7x),
    and any extra env.

    Bazel's eternal timeout defaults to one hour without the repository's
    .bazelrc. Override the timeout for an invocation with
    `bazel test <target> --test_timeout=86400` (seconds); test_timeout is a
    command-line flag, not a py_test attribute. Callers can still select a
    shorter timeout category with timeout = "short", "moderate", or "long".
    """
    merged_env = dict(env or {})
    merged_env.setdefault("PYTHONUNBUFFERED", "1")
    merged_env.setdefault("SGLANG_EXPORT_PYTHON", "1")

    kwargs.setdefault("python_version", "3.12")
    kwargs.setdefault("timeout", "eternal")

    py_test(
        name = name,
        env = merged_env,
        env_inherit = _TPU_TEST_ENV_INHERIT,
        tags = _TPU_TEST_TAGS + [t for t in (tags or []) if t not in _TPU_TEST_TAGS],
        **kwargs
    )
