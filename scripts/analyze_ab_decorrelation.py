#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv_rows(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def load_metrics(run_dir: Path):
    rows = read_csv_rows(run_dir / "metrics.csv")
    by_sample = {}
    for row in rows:
        sample = int(row["samples_seen"])
        by_sample[sample] = {
            "eval_mse": float(row["eval_mse"]),
            "eval_energy": float(row["eval_energy"]),
            "probe_mse": float(row["probe_mse"]),
            "probe_energy": float(row["probe_energy"]),
            "avg_weight_norm": float(row["avg_weight_norm"]),
            "avg_gradient_norm": float(row["avg_gradient_norm"]),
        }
    return by_sample


def load_deepest_series(run_dir: Path, file_name: str, value_field: str):
    rows = read_csv_rows(run_dir / file_name)
    by_sample = {}
    for row in rows:
        sample = int(row["samples_seen"])
        layer = int(row["layer"])
        current = by_sample.get(sample)
        if current is None or layer > current["layer"]:
            by_sample[sample] = {
                "layer": layer,
                "width": int(row["width"]),
                "value": float(row[value_field]),
            }
    return by_sample


def load_h2_summary(run_dir: Path):
    path = run_dir / "h2_analysis" / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_common_series(baseline_map, candidate_map, key):
    common_samples = sorted(set(baseline_map.keys()) & set(candidate_map.keys()))
    baseline_values = np.array([baseline_map[sample][key] for sample in common_samples], dtype=np.float64)
    candidate_values = np.array([candidate_map[sample][key] for sample in common_samples], dtype=np.float64)
    return np.array(common_samples, dtype=np.int64), baseline_values, candidate_values


def build_common_deepest_series(baseline_map, candidate_map):
    common_samples = sorted(set(baseline_map.keys()) & set(candidate_map.keys()))
    baseline_values = np.array([baseline_map[sample]["value"] for sample in common_samples], dtype=np.float64)
    candidate_values = np.array([candidate_map[sample]["value"] for sample in common_samples], dtype=np.float64)
    return np.array(common_samples, dtype=np.int64), baseline_values, candidate_values


def plot_two_lines(samples, baseline_values, candidate_values, ylabel: str, title: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(samples, baseline_values, marker="o", label="baseline")
    ax.plot(samples, candidate_values, marker="o", label="decorrelation")
    ax.set_xlabel("samples_seen")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs decorrelation A/B runs.")
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--candidate-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    baseline_run = args.baseline_run.resolve()
    candidate_run = args.candidate_run.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = load_metrics(baseline_run)
    candidate_metrics = load_metrics(candidate_run)
    baseline_dims = load_deepest_series(baseline_run, "effective_dims.csv", "effective_dim_90")
    candidate_dims = load_deepest_series(candidate_run, "effective_dims.csv", "effective_dim_90")
    baseline_err = load_deepest_series(baseline_run, "error_variances.csv", "error_variance")
    candidate_err = load_deepest_series(candidate_run, "error_variances.csv", "error_variance")

    metric_samples, baseline_mse, candidate_mse = build_common_series(baseline_metrics, candidate_metrics, "eval_mse")
    _, baseline_energy, candidate_energy = build_common_series(baseline_metrics, candidate_metrics, "eval_energy")
    dim_samples, baseline_dim, candidate_dim = build_common_deepest_series(baseline_dims, candidate_dims)
    err_samples, baseline_err_var, candidate_err_var = build_common_deepest_series(baseline_err, candidate_err)

    if metric_samples.size == 0 or dim_samples.size == 0 or err_samples.size == 0:
        raise RuntimeError("The two runs do not share enough evaluation steps for comparison.")

    plot_two_lines(metric_samples, baseline_mse, candidate_mse, "eval_mse", "A/B Eval MSE", output_dir / "eval_mse.png")
    plot_two_lines(
        metric_samples,
        baseline_energy,
        candidate_energy,
        "eval_energy",
        "A/B Eval Energy",
        output_dir / "eval_energy.png",
    )
    plot_two_lines(
        dim_samples,
        baseline_dim,
        candidate_dim,
        "deepest effective_dim_90",
        "A/B Deepest Effective Dimension",
        output_dir / "deepest_effective_dim.png",
    )
    plot_two_lines(
        err_samples,
        baseline_err_var,
        candidate_err_var,
        "deepest error variance",
        "A/B Deepest Error Variance",
        output_dir / "deepest_error_variance.png",
    )

    baseline_h2 = load_h2_summary(baseline_run)
    candidate_h2 = load_h2_summary(candidate_run)

    final_metric_sample = int(metric_samples[-1])
    final_dim_sample = int(dim_samples[-1])
    final_err_sample = int(err_samples[-1])
    summary = {
        "baseline_run": str(baseline_run),
        "candidate_run": str(candidate_run),
        "final_eval_sample": final_metric_sample,
        "baseline_final_eval_mse": float(baseline_mse[-1]),
        "candidate_final_eval_mse": float(candidate_mse[-1]),
        "eval_mse_delta_candidate_minus_baseline": float(candidate_mse[-1] - baseline_mse[-1]),
        "baseline_final_eval_energy": float(baseline_energy[-1]),
        "candidate_final_eval_energy": float(candidate_energy[-1]),
        "eval_energy_delta_candidate_minus_baseline": float(candidate_energy[-1] - baseline_energy[-1]),
        "final_effective_dim_sample": final_dim_sample,
        "baseline_final_deepest_effective_dim": float(baseline_dim[-1]),
        "candidate_final_deepest_effective_dim": float(candidate_dim[-1]),
        "deepest_effective_dim_delta": float(candidate_dim[-1] - baseline_dim[-1]),
        "final_error_variance_sample": final_err_sample,
        "baseline_final_deepest_error_variance": float(baseline_err_var[-1]),
        "candidate_final_deepest_error_variance": float(candidate_err_var[-1]),
        "deepest_error_variance_delta": float(candidate_err_var[-1] - baseline_err_var[-1]),
        "baseline_h2": baseline_h2,
        "candidate_h2": candidate_h2,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"baseline_run={baseline_run}")
    print(f"candidate_run={candidate_run}")
    print(
        f"final_eval@{final_metric_sample}: "
        f"baseline_mse={baseline_mse[-1]:.6f} candidate_mse={candidate_mse[-1]:.6f} "
        f"delta={candidate_mse[-1] - baseline_mse[-1]:+.6f}"
    )
    print(
        f"final_energy@{final_metric_sample}: "
        f"baseline={baseline_energy[-1]:.6f} candidate={candidate_energy[-1]:.6f} "
        f"delta={candidate_energy[-1] - baseline_energy[-1]:+.6f}"
    )
    print(
        f"deepest_effective_dim@{final_dim_sample}: "
        f"baseline={baseline_dim[-1]:.2f} candidate={candidate_dim[-1]:.2f} "
        f"delta={candidate_dim[-1] - baseline_dim[-1]:+.2f}"
    )
    print(
        f"deepest_error_variance@{final_err_sample}: "
        f"baseline={baseline_err_var[-1]:.6f} candidate={candidate_err_var[-1]:.6f} "
        f"delta={candidate_err_var[-1] - baseline_err_var[-1]:+.6f}"
    )
    if baseline_h2 is not None and candidate_h2 is not None:
        print(
            "h2_gradient_ratio: "
            f"baseline={baseline_h2['gradient_mean_ratio_max_over_min']:.6f} "
            f"candidate={candidate_h2['gradient_mean_ratio_max_over_min']:.6f}"
        )
        print(
            "h2_decision: "
            f"baseline={baseline_h2['decision']} | candidate={candidate_h2['decision']}"
        )


if __name__ == "__main__":
    main()
