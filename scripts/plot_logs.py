#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

plt = None
np = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LTFN training logs.")
    parser.add_argument("run_dir", type=Path, help="Run directory containing CSV logs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store generated plots. Defaults to <run_dir>/plots.",
    )
    parser.add_argument(
        "--max-latent-snapshots",
        type=int,
        default=4,
        help="Maximum number of latent PCA snapshots to render.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_float(rows: Sequence[Dict[str, str]], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows], dtype=float)


def to_int(rows: Sequence[Dict[str, str]], key: str) -> np.ndarray:
    return np.array([int(row[key]) for row in rows], dtype=int)


def plot_metrics(metrics_rows: List[Dict[str, str]], output_dir: Path) -> None:
    if not metrics_rows:
        return

    samples = to_int(metrics_rows, "samples_seen")
    eval_mse = to_float(metrics_rows, "eval_mse")
    eval_energy = to_float(metrics_rows, "eval_energy")
    probe_energy = to_float(metrics_rows, "probe_energy")
    train_window_mse = to_float(metrics_rows, "train_window_mse")

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(samples, eval_mse, marker="o", label="eval_mse")
    axes[0].plot(samples, train_window_mse, marker="s", label="train_window_mse")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Reconstruction Metrics")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(samples, eval_energy, marker="o", label="eval_energy")
    axes[1].plot(samples, probe_energy, marker="s", label="probe_energy")
    axes[1].set_xlabel("Samples Seen")
    axes[1].set_ylabel("Energy")
    axes[1].set_title("Evaluation Energy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "metrics_summary.png", dpi=160)
    plt.close(fig)


def plot_energy(energy_rows: List[Dict[str, str]], output_dir: Path) -> None:
    if not energy_rows:
        return

    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in energy_rows:
        key = f"{row['phase']}|{row['samples_seen']}|{row['sample_index']}"
        grouped[key].append(row)

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (key, rows) in enumerate(sorted(grouped.items())[:12]):
        rows = sorted(rows, key=lambda row: int(row["relax_step"]))
        relax_steps = to_int(rows, "relax_step")
        energies = to_float(rows, "energy")
        phase, samples_seen, sample_index = key.split("|")
        label = f"{phase} s={samples_seen} idx={sample_index}"
        ax.plot(relax_steps, energies, marker="o", linewidth=1.3, alpha=0.85, label=label)

    ax.set_xlabel("Relaxation Step")
    ax.set_ylabel("Global Energy")
    ax.set_title("Relaxation Energy Traces")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "energy_traces.png", dpi=160)
    plt.close(fig)


def plot_layer_errors(layer_rows: List[Dict[str, str]], output_dir: Path) -> None:
    if not layer_rows:
        return

    grouped: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in layer_rows:
        if row["phase"] != "train":
            continue
        grouped[int(row["layer"])].append(row)

    if not grouped:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for layer, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: (int(row["samples_seen"]), int(row["relax_step"])))
        xs = np.arange(len(rows))
        ys = to_float(rows, "error_norm")
        ax.plot(xs, ys, linewidth=1.2, label=f"layer_{layer}")

    ax.set_xlabel("Logged Training Step Samples")
    ax.set_ylabel("Error Norm")
    ax.set_title("Per-Layer Prediction Error Norms")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "layer_error_norms.png", dpi=160)
    plt.close(fig)


def plot_gradient_norms(grad_rows: List[Dict[str, str]], output_dir: Path) -> None:
    if not grad_rows:
        return

    grouped: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in grad_rows:
        if row["phase"] != "train":
            continue
        grouped[int(row["layer"])].append(row)

    if not grouped:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for layer, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: (int(row["samples_seen"]), int(row["relax_step"])))
        xs = np.arange(len(rows))
        grad_norm = to_float(rows, "gradient_norm")
        update_norm = to_float(rows, "weight_update_norm")
        axes[0].plot(xs, grad_norm, linewidth=1.2, label=f"layer_{layer}")
        axes[1].plot(xs, update_norm, linewidth=1.2, label=f"layer_{layer}")

    axes[0].set_ylabel("Gradient Norm")
    axes[0].set_title("Weight Gradient Norms")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Logged Training Step Samples")
    axes[1].set_ylabel("Update Norm")
    axes[1].set_title("Weight Update Norms")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "gradient_norms.png", dpi=160)
    plt.close(fig)


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    return centered @ basis


def plot_latent_pca(latent_rows: List[Dict[str, str]], output_dir: Path, max_snapshots: int) -> None:
    if not latent_rows:
        return

    latent_keys = [key for key in latent_rows[0].keys() if key.startswith("latent_")]
    grouped: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in latent_rows:
        grouped[int(row["samples_seen"])].append(row)

    snapshot_ids = sorted(grouped.keys())
    if not snapshot_ids:
        return
    if len(snapshot_ids) > max_snapshots:
        picks = np.linspace(0, len(snapshot_ids) - 1, max_snapshots, dtype=int)
        snapshot_ids = [snapshot_ids[index] for index in picks]

    cols = min(2, len(snapshot_ids))
    rows = int(np.ceil(len(snapshot_ids) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    for ax, samples_seen in zip(axes.ravel(), snapshot_ids):
        rows_for_snapshot = grouped[samples_seen]
        rows_for_snapshot = sorted(rows_for_snapshot, key=lambda row: int(row["eval_index"]))
        matrix = np.array(
            [[float(row[key]) for key in latent_keys] for row in rows_for_snapshot],
            dtype=float,
        )
        coords = pca_2d(matrix)
        ax.scatter(coords[:, 0], coords[:, 1], s=14, alpha=0.75)
        ax.set_title(f"Top Latent PCA @ samples={samples_seen}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.3)

    for ax in axes.ravel()[len(snapshot_ids):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "latent_top_pca.png", dpi=160)
    plt.close(fig)


def main() -> None:
    global np, plt
    args = parse_args()
    try:
        import matplotlib.pyplot as _plt
        import numpy as _np
    except ModuleNotFoundError as exc:
        missing = exc.name or "dependency"
        raise SystemExit(
            f"Missing Python dependency: {missing}. Install numpy and matplotlib to use plot_logs.py."
        ) from exc

    plt = _plt
    np = _np
    run_dir = args.run_dir
    output_dir = args.output_dir or (run_dir / "plots")
    ensure_dir(output_dir)

    metrics_rows = read_csv(run_dir / "metrics.csv")
    energy_rows = read_csv(run_dir / "energy.csv")
    layer_rows = read_csv(run_dir / "layer_errors.csv")
    grad_rows = read_csv(run_dir / "grad_norms.csv")
    latent_rows = read_csv(run_dir / "latent_top.csv")

    plot_metrics(metrics_rows, output_dir)
    plot_energy(energy_rows, output_dir)
    plot_layer_errors(layer_rows, output_dir)
    plot_gradient_norms(grad_rows, output_dir)
    plot_latent_pca(latent_rows, output_dir, args.max_latent_snapshots)

    print(f"Plots written to: {output_dir}")


if __name__ == "__main__":
    main()
