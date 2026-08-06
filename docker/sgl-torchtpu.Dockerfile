# Use torch-tpu:latest built from torch_tpu repo
ARG BASEIMAGE=torch-tpu:latest
FROM $BASEIMAGE

# Install additional system dependencies and Google Cloud CLI.
# ffmpeg is used by transformers for multimodal (VLM/audio) preprocessing;
# netcat-openbsd is used for connectivity checks / health probes;
# clang and patchelf are used for compiling C++ extensions (e.g., tpu-raiden).
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
    netcat-openbsd \
    clang \
    patchelf \
    protobuf-compiler && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && apt-get install -y --no-install-recommends google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/sglang

# Install Rust toolchain for setuptools-rust extensions (e.g., sglang-grpc)
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --default-toolchain stable \
    && rustc --version && cargo --version

# 1. First, copy pyproject_tpu.toml to install dependencies and maximize Docker layer cache reuse
COPY python/pyproject_tpu.toml /workspace/sglang/python/pyproject_tpu.toml
RUN cd /workspace/sglang/python && \
    sed -i 's/"torch-tpu==[^"]*"/"torch-tpu"/' pyproject_tpu.toml && \
    sed -i '/\[tool.uv\]/,$d' pyproject_tpu.toml && \
    cp pyproject_tpu.toml pyproject.toml && \
    mkdir -p sglang && touch sglang/__init__.py sglang/_version.py && \
    uv pip install --system .[all_tpu,ray,runai]

# 2. Add INSTALL_RAIDEN env/arg. If true, install raiden via git clone
ARG INSTALL_RAIDEN=true
ENV INSTALL_RAIDEN=${INSTALL_RAIDEN}

RUN if [ "$INSTALL_RAIDEN" = "true" ] || [ "$INSTALL_RAIDEN" = "1" ]; then \
        echo "===> Installing tpu-raiden via git clone..."; \
        git clone https://github.com/google/tpu-raiden.git /tmp/tpu-raiden && \
        cd /tmp/tpu-raiden && \
        printf '\n[build-system]\nrequires = ["setuptools>=61.0"]\nbuild-backend = "setuptools.build_meta"\n\n[tool.setuptools.packages.find]\ninclude = ["tpu_raiden*"]\n' >> pyproject.toml && \
        (uv pip install --system --no-deps . || pip install --no-deps .) && \
        rm -rf /tmp/tpu-raiden; \
    fi

# 3. Lastly, copy full sglang source code so source code changes reuse cached dependency layers
COPY . /workspace/sglang

RUN cd /workspace/sglang/python && \
    uv pip install --system --no-deps -e .[all_tpu,ray,runai]

# Default command
CMD ["/bin/bash"]
