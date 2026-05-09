#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_layer_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.float64)


def effective_dimension(singular_values: np.ndarray, threshold: float = 0.9) -> int:
    power = singular_values ** 2
    total = np.sum(power)
    if total <= 0:
        return 0
    cumulative = np.cumsum(power) / total
    return int(np.searchsorted(cumulative, threshold) + 1)


def analyze_representation_dir(label: str, repr_dir: Path):
    metadata = json.loads((repr_dir / "metadata.json").read_text())
    layer_files = sorted(repr_dir.glob("layer_*.csv"))
    layers = []
    for layer_file in layer_files:
        layer_index = int(layer_file.stem.split("_")[1])
        matrix = load_layer_csv(layer_file)
        centered = matrix - np.mean(matrix, axis=0, keepdims=True)
        singular_values = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
        eff_dim = effective_dimension(singular_values, 0.9)
        spectrum_ratio = float(singular_values[0] / singular_values[-1]) if singular_values.size > 0 and singular_values[-1] > 1e-12 else float("inf")
        layers.append(
            {
                "layer": layer_index,
                "width": int(matrix.shape[1]),
                "samples": int(matrix.shape[0]),
                "effective_dim_90": int(eff_dim),
                "effective_dim_ratio": float(eff_dim / matrix.shape[1]),
                "largest_singular": float(singular_values[0]) if singular_values.size else 0.0,
                "smallest_singular": float(singular_values[-1]) if singular_values.size else 0.0,
                "spectrum_ratio": spectrum_ratio,
                "singular_values": singular_values.tolist(),
            }
        )
    return {"label": label, "metadata": metadata, "layers": layers}


def plot_spectra(summaries, output_dir: Path):
    for layer_index in sorted({layer["layer"] for summary in summaries for layer in summary["layers"]}):
        fig, ax = plt.subplots(figsize=(8, 5))
        for summary in summaries:
            match = next((layer for layer in summary["layers"] if layer["layer"] == layer_index), None)
            if match is None:
                continue
            singular = np.array(match["singular_values"], dtype=np.float64)
            if singular.size == 0:
                continue
            singular = singular / max(singular[0], 1e-12)
            ax.plot(np.arange(1, singular.size + 1), singular, label=summary["label"])
        ax.set_title(f"H3 Layer {layer_index} Singular Spectrum")
        ax.set_xlabel("component")
        ax.set_ylabel("normalized singular value")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"layer_{layer_index}_spectrum.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze H3 representation effective dimensions.")
    parser.add_argument("--repr", action="append", required=True, help="LABEL=DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for item in args.repr:
        if "=" not in item:
            raise SystemExit("--repr entries must look like LABEL=DIR")
        label, path_str = item.split("=", 1)
        summaries.append(analyze_representation_dir(label, Path(path_str).resolve()))

    plot_spectra(summaries, output_dir)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({"models": summaries}, indent=2))

    for summary in summaries:
        print(summary["label"])
        for layer in summary["layers"]:
            print(
                f"  layer={layer['layer']} width={layer['width']} "
                f"effective_dim_90={layer['effective_dim_90']} "
                f"ratio={layer['effective_dim_ratio']:.4f} "
                f"spectrum_ratio={layer['spectrum_ratio']:.4f}"
            )


if __name__ == "__main__":
    main()
