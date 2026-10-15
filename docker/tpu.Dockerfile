# syntax=docker/dockerfile:1.7
#
# SGLang on Google Cloud TPU (tpu7x) via torch_tpu
# Builds on top of sglang-tpu-base (which provides all pinned dependencies).
#
ARG BASE_IMAGE=us-central1-docker.pkg.dev/sglang-torchtpu/sglang-torchtpu-ci/sglang-tpu-base:latest

########################################################
# Router Builder
########################################################
# Builds experimental/sgl-router in isolation so Python-only changes don't
# trigger a full Rust recompilation.
FROM ${BASE_IMAGE} AS router_builder

WORKDIR /build

# Copy ONLY the router workspace (not the full repo)
COPY experimental/sgl-router /build/sgl-router

# Force pcre2-sys (transitively pulled by dynamo-tokenizers) to compile and
# statically link its vendored PCRE2 rather than linking a system libpcre2-8.
ENV PCRE2_SYS_STATIC=1

# Install Rust, build the locked release sgl-router binary, strip it, and drop
# the toolchain so only the binary is carried forward.
RUN curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && export PATH="/root/.cargo/bin:${PATH}" \
    && cd /build/sgl-router \
    && cargo build --locked --release --bin sgl-router \
    && strip /build/sgl-router/target/release/sgl-router \
    && cp /build/sgl-router/target/release/sgl-router /build/sgl-router-bin \
    && rm -rf /root/.cargo /root/.rustup /build/sgl-router/target

########################################################
# Final image
########################################################
FROM ${BASE_IMAGE} AS final

WORKDIR /sgl-workspace/sglang

# Copy source trees
COPY README.md LICENSE ./
COPY python    ./python
COPY sglang_tpu ./sglang_tpu
COPY pyproject.toml ./pyproject.toml
COPY test      ./test
COPY scripts   ./scripts
COPY benchmark ./benchmark

# Swap the TPU dependency manifest into place
RUN cd python && rm -f pyproject.toml && cp pyproject_tpu.toml pyproject.toml

# Install SGLang and the TPU plugin in editable mode without reinstalling
# dependencies. Fast build on top of base.
RUN pip install --no-deps -e ./python && pip install --no-deps -e .

# libssl-dev provides <openssl/sha.h> required by hicache_hash_cpp when
# prefill workers enable --kv-events-config for experimental/sgl-router.
# Pre-compile the extension at build time so workers do not JIT-compile on
# their first request.
RUN apt-get update && apt-get install -y --no-install-recommends libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -c "from sglang.srt.mem_cache.cpp_utils.native_hash import get_native_hash; get_native_hash([1, 2], None)"

# Install the sgl-router binary built from experimental/sgl-router/
COPY --from=router_builder /build/sgl-router-bin /usr/local/bin/sgl-router

# Verify the router binary executes cleanly at image build time rather than
# discovering a missing dynamic library on a TPU node mid-test.
RUN /usr/local/bin/sgl-router --help >/dev/null

CMD ["/bin/bash"]
