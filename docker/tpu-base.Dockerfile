# syntax=docker/dockerfile:1.7
#
# SGLang TPU base image: installs the pinned TPU stack from python/pyproject_tpu.toml
# into a reusable base image (torch-tpu, libtpu, torch, torchvision, jax, jaxlib, etc.).
#
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev0 \
    SGLANG_BUILD_RUST_EXTS=none \
    SGLANG_PLATFORM=tpu

# Raiden's torch extension links libibverbs even for host KV-cache offload.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential libgomp1 libibverbs1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /sgl-workspace/sglang

# Copy only the dependency manifest and minimal package metadata so dependency
# resolution succeeds without invalidating on SGLang source changes.
COPY README.md LICENSE ./
COPY python/pyproject_tpu.toml ./python/pyproject.toml
COPY scripts/ci/tpu/warm_mmlu_cache.py ./scripts/ci/tpu/warm_mmlu_cache.py

# Install the pinned TPU stack (dev_tpu extra = all_tpu + test + ray) without sglang itself.
# diffusion_tpu: test_diffusion_tpu_platform.py imports the diffusion runtime.
RUN --mount=type=secret,id=gar_netrc,target=/root/.netrc,required=true \
    cd python && \
    mkdir -p sglang && \
    touch sglang/__init__.py && \
    pip install --upgrade pip && \
    pip install \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/ \
      ".[dev_tpu,diffusion_tpu]" && \
    pip uninstall -y sglang && \
    rm -rf sglang && \
    pip install "antlr4-python3-runtime==4.9.3" "sgl-eval==0.1.0"

# Fill sgl-eval's mmlu cache so the accuracy suite runs the eval offline.
RUN python scripts/ci/tpu/warm_mmlu_cache.py

# Verify base stack imports
RUN python -c "import torch, libtpu; print('torch', torch.__version__, '| libtpu', libtpu.__version__)" \
 && (python -c "import torch_tpu; print('torch_tpu OK')" \
       || echo "warn: torch_tpu import needs a TPU device; deferring to the on-hardware test")

# Load the actual Raiden extension without allocating a TPU or KV cache.
RUN python -c "from tpu_sync.api.torch.kv_cache_manager import _torch_impl; _torch_impl(); print('Raiden extension OK')"

CMD ["/bin/bash"]
