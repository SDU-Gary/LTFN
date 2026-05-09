#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT_DIR}/build/ltfn_train"
DATA_DIR="${ROOT_DIR}/data/mnist"
PROFILE="${1:-pilot}"
OUTPUT_DIR="${2:-${ROOT_DIR}/runs/mnist_${PROFILE}}"

if [[ ! -x "${BIN}" ]]; then
  echo "Missing executable: ${BIN}" >&2
  echo "Build first with: cmake -S . -B build && cmake --build build -j4" >&2
  exit 1
fi

if [[ ! -f "${DATA_DIR}/train-images-idx3-ubyte" ]]; then
  echo "Missing MNIST data under: ${DATA_DIR}" >&2
  echo "Download first with: ./scripts/download_mnist.sh" >&2
  exit 1
fi

case "${PROFILE}" in
  sanity)
    exec "${BIN}" \
      --mode train \
      --data-dir "${DATA_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --steps 20 \
      --max-epochs 1 \
      --max-train-samples 256 \
      --eval-samples 256 \
      --eval-interval 64 \
      --checkpoint-interval 256 \
      --logger-step-interval 10 \
      --logger-tags core,reconstructions,events,latents
    ;;
  pilot)
    exec "${BIN}" \
      --mode train \
      --data-dir "${DATA_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --steps 120 \
      --max-epochs 1 \
      --max-train-samples 5000 \
      --eval-samples 1000 \
      --eval-interval 500 \
      --checkpoint-interval 5000 \
      --logger-step-interval 20 \
      --logger-tags core,reconstructions,events,latents
    ;;
  full)
    exec "${BIN}" \
      --mode train \
      --data-dir "${DATA_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --steps 200 \
      --max-epochs 1 \
      --max-train-samples 0 \
      --eval-samples 1000 \
      --eval-interval 500 \
      --checkpoint-interval 5000 \
      --logger-step-interval 20 \
      --logger-tags core,reconstructions,events,latents
    ;;
  *)
    echo "Unknown profile: ${PROFILE}" >&2
    echo "Usage: $0 [sanity|pilot|full] [output_dir]" >&2
    exit 1
    ;;
esac
