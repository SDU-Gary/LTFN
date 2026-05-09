#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def to_float_array(rows, field):
    return np.array([float(row[field]) for row in rows], dtype=np.float64)


def plot_series(rows_by_strategy, field, output_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy, rows in rows_by_strategy.items():
        x = np.arange(len(rows))
        y = to_float_array(rows, field)
        ax.plot(x, y, label=strategy)
    ax.set_xlabel("sequence_position")
    ax.set_ylabel(field)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze H4 sequence interference results.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "analysis")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_strategy = {}
    for strategy in ("zero_init", "warm_start"):
        rows_by_strategy[strategy] = read_rows(run_dir / strategy / "sequence_metrics.csv")

    plot_series(rows_by_strategy, "train_mse", output_dir / "train_mse.png", "H4 Train Sample MSE")

    probe_fields = [field for field in rows_by_strategy["zero_init"][0].keys() if field.startswith("probe_mse_")]
    ghost_fields = [field for field in rows_by_strategy["zero_init"][0].keys() if field.startswith("ghost_")]

    for field in probe_fields:
        plot_series(rows_by_strategy, field, output_dir / f"{field}.png", f"H4 {field}")
    for field in ghost_fields:
        plot_series(rows_by_strategy, field, output_dir / f"{field}.png", f"H4 {field}")

    summary = {"strategies": {}}
    for strategy, rows in rows_by_strategy.items():
        strategy_summary = {
            "train_mse_mean": float(np.mean(to_float_array(rows, "train_mse"))),
            "train_mse_std": float(np.std(to_float_array(rows, "train_mse"))),
        }
        for field in probe_fields + ghost_fields:
            values = to_float_array(rows, field)
            strategy_summary[f"{field}_mean"] = float(np.mean(values))
            strategy_summary[f"{field}_std"] = float(np.std(values))
            strategy_summary[f"{field}_max"] = float(np.max(values))
        summary["strategies"][strategy] = strategy_summary

    zero_ghost_mean = np.mean([
        summary["strategies"]["zero_init"][f"{field}_mean"] for field in ghost_fields
    ])
    warm_ghost_mean = np.mean([
        summary["strategies"]["warm_start"][f"{field}_mean"] for field in ghost_fields
    ])
    summary["ghost_mean_ratio_warm_over_zero"] = float(warm_ghost_mean / zero_ghost_mean) if zero_ghost_mean > 0 else float("nan")
    summary["criterion_h4_positive"] = bool(warm_ghost_mean < 0.8 * zero_ghost_mean)
    summary["decision"] = (
        "H4 positive: warm-start reduces probe ghost intensity by more than 20%."
        if summary["criterion_h4_positive"]
        else "H4 negative-or-weak: warm-start does not materially reduce probe ghost intensity."
    )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
