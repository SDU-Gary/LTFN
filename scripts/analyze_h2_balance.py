#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def group_final_step(rows, target_phase: str, final_relax_step: int):
    grouped = defaultdict(dict)
    for row in rows:
        if row["phase"] != target_phase:
            continue
        if int(row["relax_step"]) != final_relax_step:
            continue
        sample = int(row["samples_seen"])
        layer = int(row["layer"])
        grouped[sample][layer] = row
    return dict(sorted(grouped.items()))


def load_config(run_dir: Path):
    return json.loads((run_dir / "config.json").read_text())


def to_series(grouped, field: str, layer_count: int):
    samples = np.array(sorted(grouped.keys()), dtype=np.int64)
    series = np.full((layer_count, len(samples)), np.nan, dtype=np.float64)
    for col, sample in enumerate(samples):
        for layer in range(layer_count):
            if layer in grouped[sample]:
                series[layer, col] = float(grouped[sample][layer][field])
    return samples, series


def quarter_mean(values: np.ndarray, fraction: float):
    if values.size == 0:
        return float("nan")
    count = max(1, int(round(values.size * fraction)))
    return float(np.nanmean(values[:count])), float(np.nanmean(values[-count:]))


def plot_lines(samples, series, ylabel: str, title: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for layer in range(series.shape[0]):
        ax.plot(samples, series[layer], label=f"layer_{layer}")
    ax.set_xlabel("samples_seen")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze H2 layer-wise balance for an LTFN training run.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "h2_analysis")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(run_dir)
    final_relax_step = int(config["steps"])
    dims = config["dims"]
    layer_count = len(dims) - 1

    error_rows = read_rows(run_dir / "layer_errors.csv")
    grad_rows = read_rows(run_dir / "grad_norms.csv")

    grouped_errors = group_final_step(error_rows, "train", final_relax_step)
    grouped_grads = group_final_step(grad_rows, "train", final_relax_step)
    common_samples = sorted(set(grouped_errors.keys()) & set(grouped_grads.keys()))
    if not common_samples:
        raise RuntimeError("No matching final-step train rows found for H2 analysis.")

    grouped_errors = {sample: grouped_errors[sample] for sample in common_samples}
    grouped_grads = {sample: grouped_grads[sample] for sample in common_samples}

    samples, error_series = to_series(grouped_errors, "error_norm", layer_count)
    _, grad_series = to_series(grouped_grads, "gradient_norm", layer_count)
    _, weight_series = to_series(grouped_grads, "weight_norm", layer_count)
    _, update_series = to_series(grouped_grads, "weight_update_norm", layer_count)
    update_ratio_series = np.divide(
        update_series,
        weight_series,
        out=np.zeros_like(update_series),
        where=np.abs(weight_series) > 1e-12,
    )

    plot_lines(samples, error_series, "error_norm", "H2 Layer Error Norms", output_dir / "layer_errors.png")
    plot_lines(samples, grad_series, "gradient_norm", "H2 Gradient Norms", output_dir / "gradient_norms.png")
    plot_lines(samples, weight_series, "weight_norm", "H2 Weight Norms", output_dir / "weight_norms.png")
    plot_lines(
        samples,
        update_ratio_series,
        "weight_update_norm / weight_norm",
        "H2 Update Ratios",
        output_dir / "update_ratios.png",
    )

    layer_stats = []
    grad_means = []
    for layer in range(layer_count):
        error_first, error_last = quarter_mean(error_series[layer], 0.25)
        grad_first, grad_last = quarter_mean(grad_series[layer], 0.25)
        weight_first, weight_last = quarter_mean(weight_series[layer], 0.25)
        ratio_first, ratio_last = quarter_mean(update_ratio_series[layer], 0.25)
        stats = {
            "layer": layer,
            "error_mean": float(np.nanmean(error_series[layer])),
            "error_std": float(np.nanstd(error_series[layer])),
            "gradient_mean": float(np.nanmean(grad_series[layer])),
            "gradient_std": float(np.nanstd(grad_series[layer])),
            "weight_mean": float(np.nanmean(weight_series[layer])),
            "weight_std": float(np.nanstd(weight_series[layer])),
            "update_ratio_mean": float(np.nanmean(update_ratio_series[layer])),
            "update_ratio_std": float(np.nanstd(update_ratio_series[layer])),
            "error_first_quarter_mean": error_first,
            "error_last_quarter_mean": error_last,
            "error_last_over_first": error_last / error_first if error_first > 0 else float("nan"),
            "gradient_first_quarter_mean": grad_first,
            "gradient_last_quarter_mean": grad_last,
            "weight_first_quarter_mean": weight_first,
            "weight_last_quarter_mean": weight_last,
            "weight_growth_ratio": weight_last / weight_first if weight_first > 0 else float("nan"),
            "update_ratio_first_quarter_mean": ratio_first,
            "update_ratio_last_quarter_mean": ratio_last,
        }
        layer_stats.append(stats)
        grad_means.append(stats["gradient_mean"])

    grad_means = np.array(grad_means, dtype=np.float64)
    max_grad_layer = int(np.argmax(grad_means))
    min_grad_layer = int(np.argmin(grad_means))
    grad_mean_ratio = float(grad_means[max_grad_layer] / grad_means[min_grad_layer]) if grad_means[min_grad_layer] > 0 else float("inf")
    suspect = layer_stats[max_grad_layer]
    imbalance_positive = (
        grad_mean_ratio > 3.0
        and suspect["error_last_over_first"] > 0.8
        and suspect["weight_growth_ratio"] > 1.05
    )

    summary = {
        "run_dir": str(run_dir),
        "visible_loss": config.get("visible_loss", "mse"),
        "final_relax_step": final_relax_step,
        "samples_analyzed": int(len(samples)),
        "layers": layer_stats,
        "gradient_mean_ratio_max_over_min": grad_mean_ratio,
        "max_gradient_layer": max_grad_layer,
        "min_gradient_layer": min_grad_layer,
        "criterion_h2_positive": imbalance_positive,
        "decision": (
            "H2 positive: one layer dominates gradients and keeps high error / weight growth."
            if imbalance_positive
            else "H2 negative-or-weak: layer scales differ, but not by the requested imbalance threshold."
        ),
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"H2 analysis written to {output_dir}")
    print(f"gradient_mean_ratio_max_over_min={grad_mean_ratio:.4f}")
    print(f"max_gradient_layer={max_grad_layer} min_gradient_layer={min_grad_layer}")
    print(f"criterion_h2_positive={imbalance_positive}")
    for stats in layer_stats:
        print(
            "layer={layer} err_mean={error_mean:.6f} grad_mean={gradient_mean:.6f} "
            "weight_mean={weight_mean:.6f} upd_ratio_mean={update_ratio_mean:.8f} "
            "err_last/first={error_last_over_first:.6f} weight_growth={weight_growth_ratio:.6f}".format(**stats)
        )


if __name__ == "__main__":
    main()
