# syntax=docker/dockerfile:1.7
#
# SGLang on Google Cloud TPU (tpu7x) via torch_tpu
# Builds on top of sglang-tpu-base (which provides all pinned dependencies).
#
ARG BASE_IMAGE=sglang-tpu-base:latest
FROM ${BASE_IMAGE} AS final

WORKDIR /sgl-workspace/sglang

# Copy source trees
COPY README.md LICENSE NOTICE ./
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
# setuptools-scm cannot read a version here (.git is not in the build context),
# so release builds pass SGLANG_VERSION; CI builds leave it empty and get the
# 0.0.0.dev0 fallback. Only export when set: an empty pretend version is not
# the same as an unset one.
ARG SGLANG_VERSION=""
RUN if [ -n "$SGLANG_VERSION" ]; then export SETUPTOOLS_SCM_PRETEND_VERSION="$SGLANG_VERSION"; fi \
 && pip install --no-deps -e ./python && pip install --no-deps -e .

CMD ["/bin/bash"]
