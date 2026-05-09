#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


MNIST_IMAGE_MAGIC = 2051
MNIST_LABEL_MAGIC = 2049
CHECKPOINT_MAGIC = b"LTFNCKP1"
CHECKPOINT_VERSION = 1


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def read_mnist_images(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = handle.read(16)
        if len(header) != 16:
            raise ValueError(f"Incomplete MNIST image header: {path}")
        magic, count, rows, cols = struct.unpack(">IIII", header)
        if magic != MNIST_IMAGE_MAGIC:
            raise ValueError(f"Unexpected image magic {magic} in {path}")
        payload = handle.read()

    expected = count * rows * cols
    data = np.frombuffer(payload, dtype=np.uint8)
    if data.size != expected:
        raise ValueError(f"Expected {expected} image bytes in {path}, got {data.size}")
    return data.reshape(count, rows * cols)


def read_mnist_labels(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = handle.read(8)
        if len(header) != 8:
            raise ValueError(f"Incomplete MNIST label header: {path}")
        magic, count = struct.unpack(">II", header)
        if magic != MNIST_LABEL_MAGIC:
            raise ValueError(f"Unexpected label magic {magic} in {path}")
        payload = handle.read()

    labels = np.frombuffer(payload, dtype=np.uint8)
    if labels.size != count:
        raise ValueError(f"Expected {count} labels in {path}, got {labels.size}")
    return labels


def write_mnist_images(path: Path, images: np.ndarray, rows: int = 28, cols: int = 28) -> None:
    if images.ndim != 2 or images.shape[1] != rows * cols:
        raise ValueError("MNIST image array must have shape (count, rows * cols).")
    ensure_dir(path.parent)
    with path.open("wb") as handle:
        handle.write(struct.pack(">IIII", MNIST_IMAGE_MAGIC, images.shape[0], rows, cols))
        handle.write(np.asarray(images, dtype=np.uint8).tobytes())


def write_mnist_labels(path: Path, labels: np.ndarray) -> None:
    if labels.ndim != 1:
        raise ValueError("MNIST labels must be a 1D array.")
    ensure_dir(path.parent)
    with path.open("wb") as handle:
        handle.write(struct.pack(">II", MNIST_LABEL_MAGIC, labels.shape[0]))
        handle.write(np.asarray(labels, dtype=np.uint8).tobytes())


def filter_mnist_by_digits(
    image_path: Path,
    label_path: Path,
    digits: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    images = read_mnist_images(image_path)
    labels = read_mnist_labels(label_path)
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"Image/label count mismatch: {image_path} vs {label_path}")
    mask = np.isin(labels, np.asarray(list(digits), dtype=np.uint8))
    return images[mask], labels[mask]


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def load_checkpoint(path: Path) -> Dict[str, object]:
    with path.open("rb") as handle:
        magic = handle.read(8)
        if magic != CHECKPOINT_MAGIC:
            raise ValueError(f"Checkpoint magic mismatch in {path}")

        version = struct.unpack("<I", handle.read(4))[0]
        if version != CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported checkpoint version {version} in {path}")

        dim_count = struct.unpack("<I", handle.read(4))[0]
        dims = list(struct.unpack(f"<{dim_count}i", handle.read(4 * dim_count)))

        tau_r, lr_w, dt_r, dt_w = struct.unpack("<4d", handle.read(8 * 4))
        samples_seen = struct.unpack("<Q", handle.read(8))[0]
        epochs_completed = struct.unpack("<Q", handle.read(8))[0]
        seed = struct.unpack("<I", handle.read(4))[0]
        weight_count = struct.unpack("<I", handle.read(4))[0]

        weights: List[np.ndarray] = []
        for _ in range(weight_count):
            rows, cols = struct.unpack("<2i", handle.read(8))
            flat = np.frombuffer(handle.read(8 * rows * cols), dtype="<f8")
            if flat.size != rows * cols:
                raise ValueError(f"Incomplete weight payload in {path}")
            weights.append(flat.reshape((rows, cols), order="F").copy())

    return {
        "dims": dims,
        "tau_r": tau_r,
        "lr_w": lr_w,
        "dt_r": dt_r,
        "dt_w": dt_w,
        "samples_seen": samples_seen,
        "epochs_completed": epochs_completed,
        "seed": seed,
        "weights": weights,
    }


def write_checkpoint(path: Path, checkpoint: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    dims = checkpoint["dims"]
    weights = checkpoint["weights"]
    with path.open("wb") as handle:
        handle.write(CHECKPOINT_MAGIC)
        handle.write(struct.pack("<I", CHECKPOINT_VERSION))
        handle.write(struct.pack("<I", len(dims)))
        handle.write(struct.pack(f"<{len(dims)}i", *dims))
        handle.write(
            struct.pack(
                "<4d",
                float(checkpoint["tau_r"]),
                float(checkpoint["lr_w"]),
                float(checkpoint["dt_r"]),
                float(checkpoint["dt_w"]),
            )
        )
        handle.write(struct.pack("<Q", int(checkpoint["samples_seen"])))
        handle.write(struct.pack("<Q", int(checkpoint["epochs_completed"])))
        handle.write(struct.pack("<I", int(checkpoint["seed"])))
        handle.write(struct.pack("<I", len(weights)))
        for weight in weights:
            matrix = np.asarray(weight, dtype=np.float64)
            handle.write(struct.pack("<2i", matrix.shape[0], matrix.shape[1]))
            handle.write(np.asarray(matrix, dtype="<f8", order="F").tobytes(order="F"))


def perturb_checkpoint_weights(
    checkpoint: Dict[str, object],
    noise_scale: float,
    seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    perturbed = dict(checkpoint)
    perturbed_weights: List[np.ndarray] = []
    for weight in checkpoint["weights"]:
        matrix = np.asarray(weight, dtype=np.float64)
        layer_std = float(matrix.std())
        sigma = noise_scale * layer_std if layer_std > 0.0 else noise_scale
        perturbed_weights.append(matrix + rng.normal(0.0, sigma, size=matrix.shape))
    perturbed["weights"] = perturbed_weights
    return perturbed


def singular_values_by_layer(checkpoint_path: Path) -> List[np.ndarray]:
    checkpoint = load_checkpoint(checkpoint_path)
    spectra: List[np.ndarray] = []
    for weight in checkpoint["weights"]:
        spectra.append(np.linalg.svd(np.asarray(weight, dtype=np.float64), compute_uv=False))
    return spectra


def spectral_concentration(singular_values: np.ndarray) -> float:
    if singular_values.size == 0:
        return 0.0
    total = float(np.sum(singular_values))
    if total <= 0.0:
        return 0.0
    return float(singular_values[0] / total)


def effective_rank(singular_values: np.ndarray, eps: float = 1e-12) -> float:
    positive = singular_values[singular_values > eps]
    if positive.size == 0:
        return 0.0
    probs = positive / positive.sum()
    entropy = -np.sum(probs * np.log(probs))
    return float(np.exp(entropy))


def save_json(path: Path, payload: Dict[str, object]) -> None:
    def normalize(value: object) -> object:
        if isinstance(value, dict):
            return {str(key): normalize(inner) for key, inner in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, tuple):
            return [normalize(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        return value

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(normalize(payload), handle, indent=2, sort_keys=True)


def run_command(command: Sequence[str], cwd: Path, dry_run: bool = False) -> None:
    command_text = " ".join(command)
    print(f"$ {command_text}", flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd), check=True)


def last_row(path: Path) -> Dict[str, str]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows[-1]


def monotonic_violations(values: Sequence[float], tolerance: float = 1e-9) -> Tuple[int, float]:
    violations = 0
    max_increase = 0.0
    for left, right in zip(values, values[1:]):
        increase = right - left
        if increase > tolerance:
            violations += 1
            max_increase = max(max_increase, increase)
    return violations, max_increase


def steps_to_fraction(values: Sequence[float], fraction: float) -> int:
    if len(values) == 0:
        return 0
    start = float(values[0])
    end = float(values[-1])
    target = end + (start - end) * fraction
    for index, value in enumerate(values):
        if value <= target:
            return index
    return len(values) - 1


def flatten(iterable: Iterable[Iterable[float]]) -> List[float]:
    return [value for seq in iterable for value in seq]
