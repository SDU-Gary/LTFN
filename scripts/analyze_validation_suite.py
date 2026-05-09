#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ltfn_experiment_utils import (
    effective_rank,
    ensure_dir,
    last_row,
    load_checkpoint,
    monotonic_violations,
    perturb_checkpoint_weights,
    read_csv_rows,
    run_command,
    save_json,
    spectral_concentration,
    steps_to_fraction,
    write_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the LTFN validation suite outputs.")
    parser.add_argument("--suite-root", type=Path, required=True, help="Root directory produced by run_validation_suite.py")
    parser.add_argument("--mse-threshold", type=float, default=0.02, help="Target test MSE threshold.")
    parser.add_argument("--monotonic-tol", type=float, default=1e-9, help="Tolerance for energy monotonicity checks.")
    parser.add_argument("--noise-seed", type=int, default=1234, help="Seed used for robustness perturbations.")
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip generating robustness eval runs and only analyze existing ones.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def unique_sorted_metrics(stage_entries: Sequence[Dict[str, object]]) -> List[Dict[str, str]]:
    rows_by_samples: Dict[int, Dict[str, str]] = {}
    for entry in stage_entries:
        metrics_path = Path(entry["run_dir"]) / "metrics.csv"
        for row in read_csv_rows(metrics_path):
            rows_by_samples[int(row["samples_seen"])] = row
    return [rows_by_samples[key] for key in sorted(rows_by_samples.keys())]


def aggregate_train_rows(stage_entries: Sequence[Dict[str, object]], filename: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for entry in stage_entries:
        rows.extend(read_csv_rows(Path(entry["run_dir"]) / filename))
    rows.sort(key=lambda row: (int(row["samples_seen"]), int(row["sample_index"]), int(row["relax_step"])))
    return rows


def summarize_final_probe_errors(trace_entries: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, np.ndarray]]:
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for entry in trace_entries:
        label = str(entry["label"])
        probe_index = int(entry["probe_index"])
        rows = [row for row in read_csv_rows(Path(entry["run_dir"]) / "layer_errors.csv") if row["phase"] == "probe"]
        if not rows:
            continue
        max_step = max(int(row["relax_step"]) for row in rows)
        final_rows = [row for row in rows if int(row["relax_step"]) == max_step]
        final_rows.sort(key=lambda row: int(row["layer"]))
        values = np.array([float(row["error_norm"]) for row in final_rows], dtype=float)
        result.setdefault(label, {})[f"probe_{probe_index:04d}"] = values
    return result


def trace_energies(trace_entries: Sequence[Dict[str, object]]) -> Dict[Tuple[str, int], np.ndarray]:
    traces: Dict[Tuple[str, int], np.ndarray] = {}
    for entry in trace_entries:
        label = str(entry["label"])
        probe_index = int(entry["probe_index"])
        rows = [row for row in read_csv_rows(Path(entry["run_dir"]) / "energy.csv") if row["phase"] == "probe"]
        rows.sort(key=lambda row: int(row["relax_step"]))
        traces[(label, probe_index)] = np.array([float(row["energy"]) for row in rows], dtype=float)
    return traces


def plot_mse_curve(metrics_rows: Sequence[Dict[str, str]], output_dir: Path) -> Dict[str, float]:
    samples = np.array([int(row["samples_seen"]) for row in metrics_rows], dtype=int)
    eval_mse = np.array([float(row["eval_mse"]) for row in metrics_rows], dtype=float)
    eval_energy = np.array([float(row["eval_energy"]) for row in metrics_rows], dtype=float)
    probe_energy = np.array([float(row["probe_energy"]) for row in metrics_rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(samples, eval_mse, marker="o", linewidth=1.5, label="eval_mse")
    axes[0].axhline(0.02, color="tab:red", linestyle="--", linewidth=1.0, label="target 0.02")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Criterion 1 and 6: Test MSE vs Samples Seen")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(samples, eval_energy, marker="o", linewidth=1.4, label="eval_energy")
    axes[1].plot(samples, probe_energy, marker="s", linewidth=1.2, label="probe_energy")
    axes[1].set_xlabel("Samples Seen")
    axes[1].set_ylabel("Energy")
    axes[1].set_title("Criterion 3: Evaluation Energy vs Samples Seen")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "01_mse_and_energy.png", dpi=160)
    plt.close(fig)

    return {
        "final_eval_mse": float(eval_mse[-1]),
        "first_eval_mse": float(eval_mse[0]),
        "final_eval_energy": float(eval_energy[-1]),
        "first_eval_energy": float(eval_energy[0]),
    }


def plot_relaxation_traces(traces: Dict[Tuple[str, int], np.ndarray], output_dir: Path) -> Dict[str, Dict[str, float]]:
    grouped_labels = sorted({label for label, _ in traces.keys()})
    fig, ax = plt.subplots(figsize=(10, 6))
    summary: Dict[str, Dict[str, float]] = {}

    for label in grouped_labels:
        selected = sorted((key for key in traces.keys() if key[0] == label), key=lambda item: item[1])
        for _, probe_index in selected:
            values = traces[(label, probe_index)]
            steps = np.arange(values.size)
            ax.plot(steps, values, linewidth=1.2, alpha=0.85, label=f"{label}/probe{probe_index}")
            violations, max_increase = monotonic_violations(values)
            summary[f"{label}/probe{probe_index}"] = {
                "initial_energy": float(values[0]) if values.size else 0.0,
                "final_energy": float(values[-1]) if values.size else 0.0,
                "violations": float(violations),
                "max_increase": float(max_increase),
                "steps_to_5pct_gap": float(steps_to_fraction(values, 0.05)),
            }

    ax.set_xlabel("Relaxation Step")
    ax.set_ylabel("Energy")
    ax.set_title("Criterion 2 and 10: Probe Energy Relaxation Traces")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "02_relaxation_traces.png", dpi=160)
    plt.close(fig)
    return summary


def acceleration_summary_from_traces(traces: Dict[Tuple[str, int], np.ndarray]) -> Dict[str, float]:
    sample_labels = sorted(label for label, _ in traces.keys() if label.startswith("samples_"))
    if not sample_labels:
        return {"init_avg_steps_to_final_target": 0.0, "final_avg_steps_to_final_target": 0.0}

    final_label = sample_labels[-1]
    probe_indices = sorted({probe for _, probe in traces.keys()})

    init_steps: List[float] = []
    final_steps: List[float] = []
    for probe_index in probe_indices:
        init_trace = traces.get(("init", probe_index))
        final_trace = traces.get((final_label, probe_index))
        if init_trace is None or final_trace is None or final_trace.size == 0:
            continue

        target = float(final_trace[-1] * 1.05)
        init_hit = next((idx for idx, value in enumerate(init_trace) if value <= target), len(init_trace))
        final_hit = next((idx for idx, value in enumerate(final_trace) if value <= target), len(final_trace))
        init_steps.append(float(init_hit))
        final_steps.append(float(final_hit))

    return {
        "final_label": final_label,
        "init_avg_steps_to_final_target": float(np.mean(init_steps)) if init_steps else 0.0,
        "final_avg_steps_to_final_target": float(np.mean(final_steps)) if final_steps else 0.0,
    }


def plot_layer_error_balance(
    error_summary: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
) -> Dict[str, Dict[str, float]]:
    labels = sorted(error_summary.keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    balance_summary: Dict[str, Dict[str, float]] = {}

    for label in labels:
        probe_items = sorted(error_summary[label].items())
        if not probe_items:
            continue
        stacked = np.vstack([values for _, values in probe_items])
        mean_errors = stacked.mean(axis=0)
        layers = np.arange(mean_errors.size)
        ax.plot(layers, mean_errors, marker="o", linewidth=1.4, label=label)

        min_nonzero = float(np.min(mean_errors[mean_errors > 0])) if np.any(mean_errors > 0) else 0.0
        max_value = float(np.max(mean_errors)) if mean_errors.size else 0.0
        balance_summary[label] = {
            "min_nonzero": min_nonzero,
            "max": max_value,
            "max_min_ratio": (max_value / min_nonzero) if min_nonzero > 0.0 else float("inf"),
        }

    ax.set_xlabel("Layer")
    ax.set_ylabel("Final Error Norm")
    ax.set_title("Criterion 4: Final Probe Error Balance Across Layers")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "03_layer_error_balance.png", dpi=160)
    plt.close(fig)
    return balance_summary


def plot_gradient_timescales(grad_rows: Sequence[Dict[str, str]], output_dir: Path) -> Dict[str, float]:
    grouped: Dict[Tuple[int, int], List[Dict[str, str]]] = defaultdict(list)
    for row in grad_rows:
        if row["phase"] != "train":
            continue
        grouped[(int(row["samples_seen"]), int(row["layer"]))].append(row)

    samples = sorted({sample for sample, _ in grouped.keys()})
    layers = sorted({layer for _, layer in grouped.keys()})

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ratios: List[float] = []

    for layer in layers:
        xs = []
        grad_values = []
        state_values = []
        update_values = []
        for sample in samples:
            rows = grouped.get((sample, layer), [])
            if not rows:
                continue
            best = max(rows, key=lambda row: int(row["relax_step"]))
            xs.append(sample)
            grad_values.append(float(best["gradient_norm"]))
            state_values.append(float(best["state_update_norm"]))
            update_values.append(float(best["weight_update_norm"]))
            state_norm = float(best["state_update_norm"])
            update_norm = float(best["weight_update_norm"])
            if state_norm > 0.0 and update_norm > 0.0:
                ratios.append(update_norm / state_norm)

        axes[0].plot(xs, grad_values, marker="o", linewidth=1.2, label=f"layer_{layer}")
        axes[1].plot(xs, state_values, marker="o", linewidth=1.2, label=f"layer_{layer}")
        axes[2].plot(xs, update_values, marker="o", linewidth=1.2, label=f"layer_{layer}")

    axes[0].set_ylabel("||dW||")
    axes[0].set_title("Criterion 5: Weight Gradient Norms")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_ylabel("||dr||")
    axes[1].set_title("State Update Norms")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_xlabel("Samples Seen")
    axes[2].set_ylabel("||delta W||")
    axes[2].set_title("Applied Weight Update Norms")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "04_gradient_timescales.png", dpi=160)
    plt.close(fig)

    if ratios:
        median_ratio = float(np.median(ratios))
        p10_ratio = float(np.percentile(ratios, 10))
        p90_ratio = float(np.percentile(ratios, 90))
    else:
        median_ratio = 0.0
        p10_ratio = 0.0
        p90_ratio = 0.0

    return {
        "median_weight_to_state_ratio": median_ratio,
        "p10_weight_to_state_ratio": p10_ratio,
        "p90_weight_to_state_ratio": p90_ratio,
    }


def plot_weight_spectra(initial_checkpoint: Path, final_checkpoint: Path, output_dir: Path) -> Dict[str, Dict[str, float]]:
    init = load_checkpoint(initial_checkpoint)
    final = load_checkpoint(final_checkpoint)
    fig, axes = plt.subplots(1, len(init["weights"]), figsize=(5 * len(init["weights"]), 4), squeeze=False)
    summary: Dict[str, Dict[str, float]] = {}

    for layer, (init_weight, final_weight) in enumerate(zip(init["weights"], final["weights"])):
        init_sv = np.linalg.svd(np.asarray(init_weight, dtype=np.float64), compute_uv=False)
        final_sv = np.linalg.svd(np.asarray(final_weight, dtype=np.float64), compute_uv=False)

        ax = axes[0, layer]
        ax.plot(np.arange(init_sv.size), init_sv, marker="o", linewidth=1.2, label="init")
        ax.plot(np.arange(final_sv.size), final_sv, marker="s", linewidth=1.2, label="final")
        ax.set_title(f"Layer {layer} Singular Values")
        ax.set_xlabel("Index")
        ax.set_ylabel("Singular Value")
        ax.grid(True, alpha=0.3)
        ax.legend()

        summary[f"layer_{layer}"] = {
            "init_concentration": spectral_concentration(init_sv),
            "final_concentration": spectral_concentration(final_sv),
            "init_effective_rank": effective_rank(init_sv),
            "final_effective_rank": effective_rank(final_sv),
        }

    fig.tight_layout()
    fig.savefig(output_dir / "05_weight_spectra.png", dpi=160)
    plt.close(fig)
    return summary


def shift_metrics(shift_manifest: Dict[str, object]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for entry in shift_manifest.get("evals", []):
        row = last_row(Path(entry["run_dir"]) / "metrics.csv")
        key = f"{entry['phase']}::{entry['dataset']}"
        values[f"{key}::mse"] = float(row["eval_mse"])
        values[f"{key}::energy"] = float(row["eval_energy"])
    return values


def plot_shift_adaptation(shift_values: Dict[str, float], output_dir: Path) -> None:
    phases = ["phase1_low", "phase2_high"]
    datasets = ["digits_0_4", "digits_5_9"]
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(phases))
    width = 0.35
    for idx, dataset in enumerate(datasets):
        ys = [shift_values.get(f"{phase}::{dataset}::mse", np.nan) for phase in phases]
        ax.bar(x + idx * width - width / 2, ys, width=width, label=dataset)

    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel("Eval MSE")
    ax.set_title("Criterion 7: Distribution Shift Adaptation")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "06_shift_adaptation.png", dpi=160)
    plt.close(fig)


def run_robustness_evals(
    manifest: Dict[str, object],
    suite_root: Path,
    skip_execution: bool,
    noise_seed: int,
) -> List[Dict[str, object]]:
    if "core" not in manifest:
        return []

    bin_path = Path(manifest["bin"])
    backend = str(manifest["backend"])
    data_dir = Path(manifest["data_dir"])
    profile_config = manifest["profile_config"]
    final_checkpoint = Path(manifest["core"]["final_checkpoint"])
    robustness_root = suite_root / "robustness"
    checkpoint_root = robustness_root / "checkpoints"
    eval_root = robustness_root / "evals"
    ensure_dir(checkpoint_root)
    ensure_dir(eval_root)

    checkpoint_payload = load_checkpoint(final_checkpoint)
    eval_samples = int(manifest["robustness"]["eval_samples"])
    noise_levels = list(manifest["robustness"]["noise_levels"])
    runs: List[Dict[str, object]] = []

    for level in noise_levels:
        checkpoint_path = checkpoint_root / f"noise_{float(level):0.3f}.ltfnckpt"
        eval_dir = eval_root / f"noise_{float(level):0.3f}"

        if float(level) == 0.0:
            if not checkpoint_path.exists():
                write_checkpoint(checkpoint_path, checkpoint_payload)
        else:
            perturbed = perturb_checkpoint_weights(checkpoint_payload, float(level), noise_seed)
            write_checkpoint(checkpoint_path, perturbed)

        command = [
            str(bin_path),
            "--mode",
            "eval",
            "--backend",
            backend,
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(eval_dir),
            "--resume",
            str(checkpoint_path),
            "--dims",
            str(manifest["dims"]),
            "--tau-r",
            str(manifest["tau_r"]),
            "--lr-w",
            str(manifest["lr_w"]),
            "--dt-r",
            str(manifest["dt_r"]),
            "--dt-w",
            str(manifest["dt_w"]),
            "--steps",
            str(profile_config["trace_steps"]),
            "--eval-samples",
            str(eval_samples),
            "--probe-index",
            "0",
            "--logger-step-interval",
            "1",
            "--logger-tags",
            "energy,layer-errors,gradients,eval,events",
            "--seed",
            str(manifest["seed"]),
        ]
        run_command(command, cwd=suite_root, dry_run=skip_execution)
        runs.append(
            {
                "noise_level": float(level),
                "checkpoint": str(checkpoint_path),
                "run_dir": str(eval_dir),
            }
        )

    return runs


def plot_robustness(runs: Sequence[Dict[str, object]], output_dir: Path) -> Dict[str, float]:
    noise = []
    mse = []
    for entry in sorted(runs, key=lambda item: float(item["noise_level"])):
        row = last_row(Path(entry["run_dir"]) / "metrics.csv")
        noise.append(float(entry["noise_level"]))
        mse.append(float(row["eval_mse"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(noise, mse, marker="o", linewidth=1.5)
    ax.set_xlabel("Weight Noise Scale")
    ax.set_ylabel("Eval MSE")
    ax.set_title("Criterion 8: Robustness to Weight Noise")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "07_robustness.png", dpi=160)
    plt.close(fig)

    baseline = mse[0] if mse else float("nan")
    return {
        "baseline_mse": baseline,
        "points": {f"{level:.3f}": value for level, value in zip(noise, mse)},
    }


def summarize_criteria(
    mse_summary: Dict[str, float],
    trace_summary: Dict[str, Dict[str, float]],
    acceleration_summary: Dict[str, float],
    balance_summary: Dict[str, Dict[str, float]],
    timescale_summary: Dict[str, float],
    spectrum_summary: Dict[str, Dict[str, float]],
    shift_summary: Dict[str, float],
    robustness_summary: Dict[str, float],
    final_metric_row: Dict[str, str],
    mse_threshold: float,
) -> Dict[str, Dict[str, object]]:
    init_trace_keys = [key for key in trace_summary if key.startswith("init/")]
    final_trace_keys = [key for key in trace_summary if key.startswith("samples_")]
    last_trace_key = sorted(final_trace_keys)[-1] if final_trace_keys else None

    final_trace = trace_summary[last_trace_key] if last_trace_key else {"violations": 0.0, "steps_to_5pct_gap": 0.0}
    final_balance_key = sorted(balance_summary.keys())[-1] if balance_summary else ""
    final_balance = balance_summary.get(final_balance_key, {"max_min_ratio": float("inf")})

    concentration_deltas = [
        layer["final_concentration"] - layer["init_concentration"]
        for layer in spectrum_summary.values()
    ]

    phase1_low = shift_summary.get("phase1_low::digits_0_4::mse", float("nan"))
    phase1_high = shift_summary.get("phase1_low::digits_5_9::mse", float("nan"))
    phase2_low = shift_summary.get("phase2_high::digits_0_4::mse", float("nan"))
    phase2_high = shift_summary.get("phase2_high::digits_5_9::mse", float("nan"))

    baseline_mse = robustness_summary.get("baseline_mse", float("nan"))
    robustness_points = robustness_summary.get("points", {})
    sorted_levels = sorted(float(level) for level in robustness_points.keys())
    smallest_nonzero = next((level for level in sorted_levels if level > 0.0), float("nan"))
    largest_level = sorted_levels[-1] if sorted_levels else float("nan")
    mse_small = robustness_points.get(f"{smallest_nonzero:.3f}", float("nan")) if np.isfinite(smallest_nonzero) else float("nan")
    mse_large = robustness_points.get(f"{largest_level:.3f}", float("nan")) if np.isfinite(largest_level) else float("nan")

    criteria = {
        "criterion_1_reconstruction": {
            "pass": mse_summary["final_eval_mse"] < mse_threshold,
            "observed": {"final_eval_mse": mse_summary["final_eval_mse"], "threshold": mse_threshold},
        },
        "criterion_2_relaxation_monotonic": {
            "pass": all(item["violations"] == 0.0 for item in trace_summary.values()),
            "observed": trace_summary,
        },
        "criterion_3_training_energy_declines": {
            "pass": mse_summary["final_eval_energy"] < mse_summary["first_eval_energy"],
            "observed": {
                "first_eval_energy": mse_summary["first_eval_energy"],
                "final_eval_energy": mse_summary["final_eval_energy"],
            },
        },
        "criterion_4_layer_error_balance": {
            "pass": final_balance["max_min_ratio"] < 100.0,
            "observed": {"final_label": final_balance_key, **final_balance},
        },
        "criterion_5_smooth_timescale_separation": {
            "pass": 1e-4 <= timescale_summary["median_weight_to_state_ratio"] <= 1e-2,
            "observed": timescale_summary,
        },
        "criterion_6_online_learning_and_small_gap": {
            "pass": (
                mse_summary["first_eval_mse"] > mse_summary["final_eval_mse"]
                and abs(float(final_metric_row["train_window_mse"]) - float(final_metric_row["eval_mse"])) < 0.01
            ),
            "observed": {
                "first_eval_mse": mse_summary["first_eval_mse"],
                "final_eval_mse": mse_summary["final_eval_mse"],
                "final_train_window_mse": float(final_metric_row["train_window_mse"]),
                "final_eval_mse_again": float(final_metric_row["eval_mse"]),
            },
        },
        "criterion_7_shift_adaptation": {
            "pass": (
                np.isfinite(phase1_high)
                and np.isfinite(phase2_high)
                and np.isfinite(phase1_low)
                and np.isfinite(phase2_low)
                and phase2_high < phase1_high
                and phase2_low <= phase1_low * 1.5
            ),
            "observed": {
                "phase1_low": phase1_low,
                "phase1_high": phase1_high,
                "phase2_low": phase2_low,
                "phase2_high": phase2_high,
            },
        },
        "criterion_8_weight_noise_robustness": {
            "pass": (
                np.isfinite(baseline_mse)
                and np.isfinite(mse_small)
                and np.isfinite(mse_large)
                and mse_small <= baseline_mse * 1.25
                and mse_large <= baseline_mse * 2.0
            ),
            "observed": robustness_summary,
        },
        "criterion_9_weight_alignment_structure": {
            "pass": bool(concentration_deltas) and float(np.mean(concentration_deltas)) > 0.0,
            "observed": spectrum_summary,
        },
        "criterion_10_inference_accelerates": {
            "pass": (
                acceleration_summary["final_avg_steps_to_final_target"] <
                acceleration_summary["init_avg_steps_to_final_target"]
                if acceleration_summary["init_avg_steps_to_final_target"] > 0
                else False
            ),
            "observed": acceleration_summary,
        },
    }
    return criteria


def write_report(
    output_dir: Path,
    criteria: Dict[str, Dict[str, object]],
    summary_json_path: Path,
) -> None:
    lines = [
        "# LTFN Validation Suite Report",
        "",
        f"Summary JSON: `{summary_json_path}`",
        "",
        "## Criteria",
        "",
    ]
    for key, payload in criteria.items():
        status = "PASS" if payload["pass"] else "FAIL"
        lines.append(f"- `{key}`: **{status}**")
        lines.append(f"  - Observed: `{json.dumps(payload['observed'], ensure_ascii=False)}`")
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `01_mse_and_energy.png`",
            "- `02_relaxation_traces.png`",
            "- `03_layer_error_balance.png`",
            "- `04_gradient_timescales.png`",
            "- `05_weight_spectra.png`",
            "- `06_shift_adaptation.png`",
            "- `07_robustness.png`",
        ]
    )
    with (output_dir / "report.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.suite_root = args.suite_root.resolve()
    manifest = load_manifest(args.suite_root / "suite_manifest.json")
    report_dir = args.suite_root / "report"
    ensure_dir(report_dir)

    core = manifest.get("core")
    if not core:
        raise SystemExit("suite_manifest.json does not contain a core experiment section.")

    metrics_rows = unique_sorted_metrics(core["stages"])
    if not metrics_rows:
        raise SystemExit("No convergence metrics found.")

    mse_summary = plot_mse_curve(metrics_rows, report_dir)
    traces = trace_energies(core["trace_evals"])
    trace_summary = plot_relaxation_traces(traces, report_dir)
    acceleration_summary = acceleration_summary_from_traces(traces)
    error_summary = summarize_final_probe_errors(core["trace_evals"])
    balance_summary = plot_layer_error_balance(error_summary, report_dir)

    grad_rows = aggregate_train_rows(core["stages"], "grad_norms.csv")
    timescale_summary = plot_gradient_timescales(grad_rows, report_dir)

    spectrum_summary = plot_weight_spectra(
        Path(core["initial_checkpoint"]),
        Path(core["final_checkpoint"]),
        report_dir,
    )

    shift_summary: Dict[str, float] = {}
    if "shift" in manifest:
        shift_summary = shift_metrics(manifest["shift"])
        plot_shift_adaptation(shift_summary, report_dir)

    robustness_runs = [] if args.skip_robustness else run_robustness_evals(
        manifest, args.suite_root, args.skip_robustness, args.noise_seed
    )
    if robustness_runs:
        manifest["robustness"]["runs"] = robustness_runs
        save_json(args.suite_root / "suite_manifest.json", manifest)
    elif isinstance(manifest.get("robustness", {}).get("runs"), list):
        robustness_runs = manifest["robustness"]["runs"]

    robustness_summary = plot_robustness(robustness_runs, report_dir) if robustness_runs else {}
    final_metric_row = metrics_rows[-1]

    criteria = summarize_criteria(
        mse_summary,
        trace_summary,
        acceleration_summary,
        balance_summary,
        timescale_summary,
        spectrum_summary,
        shift_summary,
        robustness_summary,
        final_metric_row,
        args.mse_threshold,
    )

    summary_payload = {
        "criteria": criteria,
        "mse_summary": mse_summary,
        "trace_summary": trace_summary,
        "acceleration_summary": acceleration_summary,
        "balance_summary": balance_summary,
        "timescale_summary": timescale_summary,
        "spectrum_summary": spectrum_summary,
        "shift_summary": shift_summary,
        "robustness_summary": robustness_summary,
    }
    summary_json = report_dir / "summary.json"
    save_json(summary_json, summary_payload)
    write_report(report_dir, criteria, summary_json)
    print(f"Analysis written to: {report_dir}")


if __name__ == "__main__":
    main()
