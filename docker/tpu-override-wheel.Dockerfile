# syntax=docker/dockerfile:1.7
#
# Override torch_tpu wheel in the built SGLang TPU image with the latest
# (or specified) wheel from PyPI or a staged local wheel file.
ARG BASE_IMAGE=us-central1-docker.pkg.dev/sglang-torchtpu/sglang-torchtpu-ci/sglang-tpu:latest
FROM ${BASE_IMAGE}

ARG WHEEL=torch_tpu

COPY wheel_override/ /tmp/wheel_override/

RUN echo "============================================================" && \
    echo "=== BASE CONTAINER: Current installed torch-tpu package ===" && \
    (pip show torch-tpu || pip show torch_tpu || true) && \
    echo "============================================================" && \
    LOCAL_WHL="$(find /tmp/wheel_override -name '*.whl' 2>/dev/null | head -n 1 || true)" && \
    if [ -n "$LOCAL_WHL" ] && [ -f "$LOCAL_WHL" ]; then \
      echo "=== INSTALLING PROVIDED WHEEL FILE: $LOCAL_WHL ===" && \
      pip install --no-deps --force-reinstall -v "$LOCAL_WHL" && rm -rf /tmp/wheel_override; \
    else \
      echo "=== PULLING WHEEL '${WHEEL}' FROM PYPI ===" && \
      pip install --no-deps --force-reinstall --pre --upgrade -v \
        "${WHEEL}"; \
    fi && \
    echo "============================================================" && \
    echo "=== VERIFYING NEWLY INSTALLED TORCH-TPU PACKAGE ===" && \
    pip show torch_tpu && \
    python3 -c "import importlib.metadata; print('=== CONFIRMED TORCH-TPU VERSION:', importlib.metadata.version('torch_tpu'))" && \
    python3 -c "import torch, libtpu; print('=== PyTorch version:', torch.__version__, '| libtpu version:', libtpu.__version__)" && \
    (python3 -c "import torch_tpu; print('=== torch_tpu module check OK')" \
        || echo "warn: torch_tpu import needs a TPU device; deferring to on-hardware test") && \
    echo "============================================================"

CMD ["/bin/bash"]
