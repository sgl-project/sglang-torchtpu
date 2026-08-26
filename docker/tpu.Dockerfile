# syntax=docker/dockerfile:1.7
#
# SGLang on Google Cloud TPU (tpu7x) via torch_tpu
# (PyTorch PrivateUse1 -> XLA HLO -> PJRT/libtpu).
#
# Installs the pinned TPU stack from python/pyproject_tpu.toml (torch-tpu, libtpu,
# torch, torchvision, jax, jaxlib + sglang) from the private torch-tpu "virtual"
# registry, which federates the PyTorch-CPU index, the Torch-TPU index and PyPI --
# so a single authenticated --index-url resolves the whole graph.
#
# The registry requires a GCP OAuth token, passed as a BuildKit secret so it is
# never written into an image layer. Build from the repo root, e.g.:
#
#   gcloud auth print-access-token > /tmp/gcp_token
#   DOCKER_BUILDKIT=1 docker build -f docker/tpu.Dockerfile \
#       --secret id=gcp_token,src=/tmp/gcp_token \
#       -t <registry>/sglang-torchtpu:tpu-<sha> .
#
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev0 \
    SGLANG_BUILD_RUST_EXTS=none \
    SGLANG_PLATFORM=tpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /sgl-workspace/sglang

# Fork source: python/ builds the sglang package; test/ runs the suite;
# benchmark/ provides the canonical TPU launcher (benchmark/tpu/launch_server.sh).
COPY README.md LICENSE ./
COPY python    ./python
COPY test      ./test
COPY scripts   ./scripts
COPY benchmark ./benchmark

# Mirror npu.Dockerfile: swap the TPU dependency manifest into place so the
# build resolves against the TPU pins instead of the CUDA ones.
RUN cd python && rm -f pyproject.toml && cp pyproject_tpu.toml pyproject.toml

# Install the pinned TPU stack + sglang (dev_tpu extra = all_tpu + test + ray).
# The virtual registry is a single index that federates torch-CPU, torch-tpu and
# PyPI, so one --index-url resolves everything.
RUN --mount=type=secret,id=gcp_token \
    cd python && \
    pip install --upgrade pip && \
    TOKEN="$(cat /run/secrets/gcp_token)" && \
    pip install \
      --index-url "https://oauth2accesstoken:${TOKEN}@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/" \
      -e ".[dev_tpu]"

# Fail fast if the core runtime imports are broken. torch + libtpu import without
# hardware; torch_tpu registers the PrivateUse1 backend and may need a TPU, so its
# check is soft (the real gate is the on-TPU integration test).
RUN python -c "import torch, libtpu; print('torch', torch.__version__, '| libtpu', libtpu.__version__)" \
 && (python -c "import torch_tpu; print('torch_tpu OK')" \
       || echo "warn: torch_tpu import needs a TPU device; deferring to the on-hardware test")

CMD ["/bin/bash"]
