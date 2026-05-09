#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:-${ROOT_DIR}/data/mnist}"
BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"

mkdir -p "${DATA_DIR}"

files=(
  "train-images-idx3-ubyte.gz"
  "train-labels-idx1-ubyte.gz"
  "t10k-images-idx3-ubyte.gz"
  "t10k-labels-idx1-ubyte.gz"
)

download_file() {
  local filename="$1"
  local target="${DATA_DIR}/${filename}"
  if [[ -f "${target%.gz}" ]]; then
    echo "[skip] ${target%.gz} already exists"
    return
  fi
  if [[ ! -f "${target}" ]]; then
    echo "[download] ${filename}"
    curl -L --fail --retry 3 --retry-delay 2 -o "${target}" "${BASE_URL}/${filename}"
  else
    echo "[reuse] ${target}"
  fi
  echo "[extract] ${filename}"
  gzip -dkf "${target}"
}

for file in "${files[@]}"; do
  download_file "${file}"
done

echo
echo "MNIST ready at: ${DATA_DIR}"
echo "Files:"
find "${DATA_DIR}" -maxdepth 1 -type f | sort
