#!/bin/bash
set -e

# Script to build sglang-torchtpu docker image.
# It handles torch_tpu dependency (builds base image if needed)
# and then builds the sglang-torchtpu image.
#
# Options:
#   --remote-torch-tpu-image <image>
#       Use a pre-built remote torch-tpu image instead of
#       building/cloning locally.
#   --rebuild-torchtpu-base
#       Force rebuild of the local torch-tpu base image.
#   --skip-raiden
#       Skip installing the tpu-raiden package in the container.
#
# Example:
#   ./docker/build_sgl_torchtpu.sh \
#     --remote-torch-tpu-image \
#     us-docker.pkg.dev/ml-oss-artifacts-transient/\
#     torch-tpu-docker-container/torch-tpu:nightly-latest

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_ROOT="$(cd "${SCRIPT_DIR}/../" && pwd)"

# Define path for torch_tpu
if [ -d "${HOME}/torch_tpu" ]; then
  TORCH_TPU_DIR="${HOME}/torch_tpu"
else
  TORCH_TPU_DIR="${SGLANG_ROOT}/../torch_tpu"
fi

# Optional directory containing a `docker` shim (e.g. a Podman wrapper), used
# only when the Docker daemon is unavailable. Set PODMAN_WRAPPER_DIR to enable.
WRAPPER_DIR="${PODMAN_WRAPPER_DIR:-}"

# 1. Auto-fallback to Podman if Docker daemon is not accessible
if ! command -v docker >/dev/null 2>&1 || ! docker ps >/dev/null 2>&1; then
  if command -v podman >/dev/null 2>&1 && [ -n "$WRAPPER_DIR" ] && [ -d "$WRAPPER_DIR" ]; then
    echo "===> Docker daemon not accessible. Using Podman wrapper..."
    export PATH="${WRAPPER_DIR}:${PATH}"
  fi
fi

# Parse command line arguments
REBUILD_BASE=false
SKIP_RAIDEN=false
REMOTE_TORCH_TPU_IMAGE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    # Support space-separated syntax: --remote-torch-tpu-image <value>
    --remote-torch-tpu-image)
      if [[ -n "$2" && "$2" != -* ]]; then
        REMOTE_TORCH_TPU_IMAGE="$2"
        shift 2
      else
        echo "Error: --remote-torch-tpu-image requires a value." >&2
        exit 1
      fi
      ;;
    # Support equals-separated syntax: --remote-torch-tpu-image=<value>
    --remote-torch-tpu-image=*)
      REMOTE_TORCH_TPU_IMAGE="${1#*=}"
      if [[ -z "$REMOTE_TORCH_TPU_IMAGE" ]]; then
        echo "Error: --remote-torch-tpu-image requires a value." >&2
        exit 1
      fi
      shift 1
      ;;
    --rebuild-torchtpu-base)
      REBUILD_BASE=true
      shift 1
      ;;
    --skip-raiden)
      SKIP_RAIDEN=true
      shift 1
      ;;
    *)
      echo "Error: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

# 2. Build torch-tpu image conditionally or use remote image
if [ -n "$REMOTE_TORCH_TPU_IMAGE" ]; then
  echo "===> Using remote torch-tpu image: ${REMOTE_TORCH_TPU_IMAGE}"
  BASEIMAGE="${REMOTE_TORCH_TPU_IMAGE}"
else
  BASEIMAGE="torch-tpu:latest"
  # Check if torch-tpu image exists
  IMAGE_EXISTS=true
  if ! docker image inspect torch-tpu:latest >/dev/null 2>&1; then
    IMAGE_EXISTS=false
  fi

  if [ "$REBUILD_BASE" = true ] || [ "$IMAGE_EXISTS" = false ]; then
    # Check or clone torch_tpu repository only when building
    if [ -d "${TORCH_TPU_DIR}/.git" ]; then
      echo "===> Found torch_tpu repo at ${TORCH_TPU_DIR}"
    else
      echo "===> torch_tpu repo not found or incomplete. Cloning..."
      git clone https://github.com/google-pytorch/torch_tpu.git \
        "${TORCH_TPU_DIR}"
    fi

    echo "===> Building torch_tpu base image..."
    # Run the build_image_multistage.sh script from torch_tpu
    # We specify the repo root and the tag prefix "torch-tpu"
    bash "${TORCH_TPU_DIR}/docker/build_image_multistage.sh" \
      "${TORCH_TPU_DIR}" "torch-tpu"
  else
    echo "===> torch-tpu:latest already exists. Skipping base image build."
    echo "     (Use '--rebuild-torchtpu-base' to force rebuild)"
  fi
fi

# 3. Build sglang-torchtpu image
# Tag with both :latest (for convenience) and a timestamp (for traceability /
# to keep a history of local builds).
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
INSTALL_RAIDEN_ARG="true"
if [ "$SKIP_RAIDEN" = true ]; then
  INSTALL_RAIDEN_ARG="false"
fi

echo "===> Building sglang-torchtpu image (tags: latest, ${TIMESTAMP}, INSTALL_RAIDEN=${INSTALL_RAIDEN_ARG})..."
docker build \
  --progress=plain \
  -f "${SCRIPT_DIR}/sgl-torchtpu.Dockerfile" \
  -t sglang-torchtpu:latest \
  -t "sglang-torchtpu:${TIMESTAMP}" \
  --build-arg BASEIMAGE="${BASEIMAGE}" \
  --build-arg INSTALL_RAIDEN="${INSTALL_RAIDEN_ARG}" \
  "${SGLANG_ROOT}"

echo "===> Done! sglang-torchtpu:latest (also tagged ${TIMESTAMP}) built successfully."
