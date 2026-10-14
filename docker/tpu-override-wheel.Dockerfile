# syntax=docker/dockerfile:1.7
#
# Override torch_tpu wheel in the built SGLang TPU image with the latest
# (or specified) wheel from the private torch-tpu virtual registry.
ARG BASE_IMAGE=us-central1-docker.pkg.dev/sglang-torchtpu/sglang-torchtpu-ci/sglang-tpu:latest
FROM ${BASE_IMAGE}

ARG WHEEL=torch-tpu

COPY wheel_override/ /tmp/wheel_override/

RUN --mount=type=secret,id=gcp_token \
    TOKEN="$(cat /run/secrets/gcp_token 2>/dev/null || echo '')" && \
    INDEX_URL="https://oauth2accesstoken:${TOKEN}@us-python.pkg.dev/ml-oss-artifacts-published/torch-tpu-release-virtual-registry/simple/" && \
    echo "============================================================" && \
    echo "=== BASE CONTAINER: Current installed torch-tpu package ===" && \
    (pip show torch-tpu || pip show torch_tpu || true) && \
    echo "============================================================" && \
    LOCAL_WHL="$(find /tmp/wheel_override -name '*.whl' 2>/dev/null | head -n 1 || true)" && \
    if [ -n "$LOCAL_WHL" ] && [ -f "$LOCAL_WHL" ]; then \
      echo "=== INSTALLING PROVIDED WHEEL FILE: $LOCAL_WHL ===" && \
      pip install --no-deps --force-reinstall -v "$LOCAL_WHL" && rm -rf /tmp/wheel_override; \
    else \
      echo "=== PULLING WHEEL '${WHEEL}' FROM VIRTUAL REGISTRY ===" && \
      echo "Registry URL: https://us-python.pkg.dev/ml-oss-artifacts-published/torch-tpu-release-virtual-registry/simple/" && \
      pip install --no-deps --force-reinstall --pre --upgrade -v \
        --index-url "${INDEX_URL}" \
        "${WHEEL}"; \
    fi && \
    echo "============================================================" && \
    echo "=== VERIFYING NEWLY INSTALLED TORCH-TPU PACKAGE ===" && \
    pip show torch-tpu && \
    python3 -c "import importlib.metadata; print('=== CONFIRMED TORCH-TPU VERSION:', importlib.metadata.version('torch-tpu'))" && \
    python3 -c "import torch, libtpu; print('=== PyTorch version:', torch.__version__, '| libtpu version:', libtpu.__version__)" && \
    (python3 -c "import torch_tpu; print('=== torch_tpu module check OK')" \
        || echo "warn: torch_tpu import needs a TPU device; deferring to on-hardware test") && \
    echo "============================================================"

CMD ["/bin/bash"]
