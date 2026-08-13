# Use torch-tpu:latest built from torch_tpu repo
ARG BASEIMAGE=torch-tpu:latest
FROM $BASEIMAGE

# Install additional system dependencies and Google Cloud CLI.
# ffmpeg is used by transformers for multimodal (VLM/audio) preprocessing;
# netcat-openbsd is used for connectivity checks / health probes.
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    gnupg \
    git \
    curl \
    wget \
    vim \
    gcc \
    g++ \
    make \
    ffmpeg \
    netcat-openbsd && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && apt-get install -y --no-install-recommends google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy sglang source code into /workspace/sglang
COPY . /workspace/sglang

# 1. Modify pyproject_tpu.toml to refer to the already installed torch-tpu
# version
# 2. Copy pyproject_tpu.toml to pyproject.toml
# 3. Install sglang tpu dependencies and sglang itself
RUN --mount=type=secret,id=gcp_token \
    cd /workspace/sglang/python && \
    sed -i 's/"torch-tpu==[^"]*"/"torch-tpu"/' pyproject_tpu.toml && \
    sed -i '/\[tool.uv\]/,$d' pyproject_tpu.toml && \
    cp pyproject_tpu.toml pyproject.toml && \
    TOKEN="$(cat /run/secrets/gcp_token 2>/dev/null || true)" && \
    if [ -n "$TOKEN" ]; then \
      uv pip install --system --extra-index-url "https://oauth2accesstoken:${TOKEN}@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/" .[all_tpu,ray,runai] || pip install --no-deps -e .; \
    else \
      pip install --no-deps -e .; \
    fi

# Default command
CMD ["/bin/bash"]
