#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

from ltfn_experiment_utils import (
    ensure_dir,
    filter_mnist_by_digits,
    run_command,
    save_json,
    write_mnist_images,
    write_mnist_labels,
)


PROFILES: Dict[str, Dict[str, object]] = {
    "smoke": {
        "steps": 40,
        "trace_steps": 80,
        "max_epochs": 4,
        "stage_samples": [128, 512, 1024],
        "eval_interval": 64,
        "eval_samples": 128,
        "checkpoint_interval": 256,
        "logger_step_interval": 4,
        "probe_indices": [0, 1],
        "shift_low_total": 256,
        "shift_high_total": 512,
        "robustness_eval_samples": 128,
        "noise_levels": [0.0, 0.01, 0.02, 0.05],
    },
    "pilot": {
        "steps": 120,
        "trace_steps": 250,
        "max_epochs": 8,
        "stage_samples": [500, 2000, 5000, 10000, 20000],
        "eval_interval": 500,
        "eval_samples": 1000,
        "checkpoint_interval": 2500,
        "logger_step_interval": 10,
        "probe_indices": [0, 1, 2],
        "shift_low_total": 10000,
        "shift_high_total": 20000,
        "robustness_eval_samples": 512,
        "noise_levels": [0.0, 0.01, 0.02, 0.05],
    },
    "full": {
        "steps": 200,
        "trace_steps": 500,
        "max_epochs": 16,
        "stage_samples": [500, 2000, 5000, 10000, 30000, 60000, 120000],
        "eval_interval": 1000,
        "eval_samples": 1000,
        "checkpoint_interval": 5000,
        "logger_step_interval": 20,
        "probe_indices": [0, 1, 2, 3],
        "shift_low_total": 30000,
        "shift_high_total": 60000,
        "robustness_eval_samples": 1000,
        "noise_levels": [0.0, 0.01, 0.02, 0.05],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LTFN validation experiment suite.")
    parser.add_argument("--bin", type=Path, required=True, help="Path to ltfn_train executable.")
    parser.add_argument("--data-dir", type=Path, required=True, help="MNIST directory with image and label files.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root directory for the full suite output.")
    parser.add_argument("--backend", default="cuda", choices=("cpu", "cuda"), help="Backend passed to ltfn_train.")
    parser.add_argument("--profile", default="pilot", choices=tuple(PROFILES.keys()), help="Experiment scale preset.")
    parser.add_argument(
        "--stages",
        default="all",
        choices=("all", "core", "shift"),
        help="Subset of the suite to run.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Training seed.")
    parser.add_argument("--dims", default="784,256,64,32", help="Network dimensions.")
    parser.add_argument("--tau-r", type=float, default=0.1, help="State time constant.")
    parser.add_argument("--lr-w", type=float, default=1e-5, help="Weight learning rate.")
    parser.add_argument("--dt-r", type=float, default=0.1, help="State step size.")
    parser.add_argument("--dt-w", type=float, default=1.0, help="Weight step size.")
    parser.add_argument("--shuffle", default="true", choices=("true", "false"), help="Shuffle training data.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--force", action="store_true", help="Delete existing output_root before running.")
    return parser.parse_args()


def train_command(
    args: argparse.Namespace,
    config: Dict[str, object],
    data_dir: Path,
    output_dir: Path,
    max_train_samples: int,
    resume_path: Path | None = None,
    save_initial_checkpoint: Path | None = None,
) -> List[str]:
    command = [
        str(args.bin),
        "--mode",
        "train",
        "--backend",
        args.backend,
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--dims",
        args.dims,
        "--tau-r",
        str(args.tau_r),
        "--lr-w",
        str(args.lr_w),
        "--dt-r",
        str(args.dt_r),
        "--dt-w",
        str(args.dt_w),
        "--steps",
        str(config["steps"]),
        "--max-epochs",
        str(config["max_epochs"]),
        "--max-train-samples",
        str(max_train_samples),
        "--eval-samples",
        str(config["eval_samples"]),
        "--eval-interval",
        str(config["eval_interval"]),
        "--checkpoint-interval",
        str(config["checkpoint_interval"]),
        "--logger-step-interval",
        str(config["logger_step_interval"]),
        "--logger-tags",
        "all",
        "--shuffle",
        args.shuffle,
        "--seed",
        str(args.seed),
    ]
    if resume_path is not None:
        command.extend(["--resume", str(resume_path)])
    if save_initial_checkpoint is not None:
        command.extend(["--save-initial-checkpoint", str(save_initial_checkpoint)])
    return command


def eval_command(
    args: argparse.Namespace,
    config: Dict[str, object],
    data_dir: Path,
    output_dir: Path,
    checkpoint_path: Path,
    probe_index: int,
) -> List[str]:
    return [
        str(args.bin),
        "--mode",
        "eval",
        "--backend",
        args.backend,
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--resume",
        str(checkpoint_path),
        "--dims",
        args.dims,
        "--tau-r",
        str(args.tau_r),
        "--lr-w",
        str(args.lr_w),
        "--dt-r",
        str(args.dt_r),
        "--dt-w",
        str(args.dt_w),
        "--steps",
        str(config["trace_steps"]),
        "--eval-samples",
        str(config["eval_samples"]),
        "--probe-index",
        str(probe_index),
        "--logger-step-interval",
        "1",
        "--logger-tags",
        "energy,layer-errors,gradients,eval,events,reconstructions,latents",
        "--seed",
        str(args.seed),
    ]


def ensure_filtered_split(source_dir: Path, output_root: Path) -> Dict[str, str]:
    specs = {
        "digits_0_4": [0, 1, 2, 3, 4],
        "digits_5_9": [5, 6, 7, 8, 9],
    }

    train_images = source_dir / "train-images-idx3-ubyte"
    train_labels = source_dir / "train-labels-idx1-ubyte"
    test_images = source_dir / "t10k-images-idx3-ubyte"
    test_labels = source_dir / "t10k-labels-idx1-ubyte"

    ensure_dir(output_root)
    manifest: Dict[str, object] = {}
    results: Dict[str, str] = {}

    for name, digits in specs.items():
        subset_dir = output_root / name
        if not (subset_dir / "train-images-idx3-ubyte").exists():
            train_subset, train_subset_labels = filter_mnist_by_digits(train_images, train_labels, digits)
            test_subset, test_subset_labels = filter_mnist_by_digits(test_images, test_labels, digits)
            write_mnist_images(subset_dir / "train-images-idx3-ubyte", train_subset)
            write_mnist_labels(subset_dir / "train-labels-idx1-ubyte", train_subset_labels)
            write_mnist_images(subset_dir / "t10k-images-idx3-ubyte", test_subset)
            write_mnist_labels(subset_dir / "t10k-labels-idx1-ubyte", test_subset_labels)
            manifest[name] = {
                "digits": digits,
                "train_count": int(train_subset.shape[0]),
                "test_count": int(test_subset.shape[0]),
            }
        results[name] = str(subset_dir)

    if manifest:
        save_json(output_root / "manifest.json", manifest)
    return results


def select_trace_stages(stage_samples: Sequence[int]) -> List[int]:
    picks = [stage_samples[0], stage_samples[len(stage_samples) // 2], stage_samples[-1]]
    ordered: List[int] = []
    for value in picks:
        if value not in ordered:
            ordered.append(value)
    return ordered


def run_core_suite(args: argparse.Namespace, config: Dict[str, object], suite_root: Path) -> Dict[str, object]:
    convergence_root = suite_root / "convergence"
    ensure_dir(convergence_root)

    initial_checkpoint = convergence_root / "init.ltfnckpt"
    stage_entries: List[Dict[str, object]] = []

    resume_path: Path | None = None
    for index, sample_target in enumerate(config["stage_samples"]):
        run_dir = convergence_root / f"stage_{int(sample_target):06d}"
        ensure_dir(run_dir)
        command = train_command(
            args,
            config,
            args.data_dir,
            run_dir,
            int(sample_target),
            resume_path=resume_path,
            save_initial_checkpoint=initial_checkpoint if index == 0 else None,
        )
        run_command(command, cwd=args.output_root, dry_run=args.dry_run)

        checkpoint = run_dir / "checkpoints" / "latest.ltfnckpt"
        stage_entries.append(
            {
                "samples_target": int(sample_target),
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "resume_from": str(resume_path) if resume_path else None,
            }
        )
        resume_path = checkpoint

    trace_entries: List[Dict[str, object]] = []
    trace_targets = select_trace_stages(config["stage_samples"])
    checkpoints_by_samples = {entry["samples_target"]: Path(entry["checkpoint"]) for entry in stage_entries}

    for label, checkpoint in [("init", initial_checkpoint)] + [
        (f"samples_{target:06d}", checkpoints_by_samples[target]) for target in trace_targets
    ]:
        for probe_index in config["probe_indices"]:
            trace_dir = convergence_root / "traces" / label / f"probe_{probe_index:04d}"
            ensure_dir(trace_dir)
            command = eval_command(args, config, args.data_dir, trace_dir, checkpoint, int(probe_index))
            run_command(command, cwd=args.output_root, dry_run=args.dry_run)
            trace_entries.append(
                {
                    "label": label,
                    "probe_index": int(probe_index),
                    "checkpoint": str(checkpoint),
                    "run_dir": str(trace_dir),
                }
            )

    return {
        "initial_checkpoint": str(initial_checkpoint),
        "stage_samples": list(config["stage_samples"]),
        "stages": stage_entries,
        "trace_evals": trace_entries,
        "final_checkpoint": stage_entries[-1]["checkpoint"] if stage_entries else "",
    }


def run_shift_suite(args: argparse.Namespace, config: Dict[str, object], suite_root: Path) -> Dict[str, object]:
    shift_root = suite_root / "shift"
    ensure_dir(shift_root)
    split_dirs = ensure_filtered_split(args.data_dir, suite_root / "mnist_splits")
    low_dir = Path(split_dirs["digits_0_4"])
    high_dir = Path(split_dirs["digits_5_9"])

    initial_checkpoint = shift_root / "init.ltfnckpt"
    phase1_dir = shift_root / "phase1_low"
    phase2_dir = shift_root / "phase2_high"
    ensure_dir(phase1_dir)
    ensure_dir(phase2_dir)

    run_command(
        train_command(
            args,
            config,
            low_dir,
            phase1_dir,
            int(config["shift_low_total"]),
            resume_path=None,
            save_initial_checkpoint=initial_checkpoint,
        ),
        cwd=args.output_root,
        dry_run=args.dry_run,
    )
    phase1_checkpoint = phase1_dir / "checkpoints" / "latest.ltfnckpt"

    run_command(
        train_command(
            args,
            config,
            high_dir,
            phase2_dir,
            int(config["shift_high_total"]),
            resume_path=phase1_checkpoint,
        ),
        cwd=args.output_root,
        dry_run=args.dry_run,
    )
    phase2_checkpoint = phase2_dir / "checkpoints" / "latest.ltfnckpt"

    evals: List[Dict[str, object]] = []
    for phase_name, checkpoint in (("phase1_low", phase1_checkpoint), ("phase2_high", phase2_checkpoint)):
        for dataset_name, dataset_dir in (("digits_0_4", low_dir), ("digits_5_9", high_dir)):
            eval_dir = shift_root / "evals" / phase_name / dataset_name
            ensure_dir(eval_dir)
            run_command(
                eval_command(args, config, dataset_dir, eval_dir, checkpoint, probe_index=0),
                cwd=args.output_root,
                dry_run=args.dry_run,
            )
            evals.append(
                {
                    "phase": phase_name,
                    "dataset": dataset_name,
                    "checkpoint": str(checkpoint),
                    "run_dir": str(eval_dir),
                }
            )

    return {
        "initial_checkpoint": str(initial_checkpoint),
        "split_dirs": split_dirs,
        "phase1_checkpoint": str(phase1_checkpoint),
        "phase2_checkpoint": str(phase2_checkpoint),
        "evals": evals,
    }


def main() -> None:
    args = parse_args()
    args.bin = args.bin.resolve()
    args.data_dir = args.data_dir.resolve()
    args.output_root = args.output_root.resolve()
    profile_config = dict(PROFILES[args.profile])

    if not args.bin.exists():
        raise SystemExit(f"Missing executable: {args.bin}")
    if not (args.data_dir / "train-images-idx3-ubyte").exists():
        raise SystemExit(f"Missing MNIST images under: {args.data_dir}")
    if not (args.data_dir / "train-labels-idx1-ubyte").exists():
        raise SystemExit(f"Missing MNIST labels under: {args.data_dir}")

    if args.force and args.output_root.exists():
        shutil.rmtree(args.output_root)
    ensure_dir(args.output_root)

    manifest: Dict[str, object] = {
        "bin": str(args.bin),
        "backend": args.backend,
        "profile": args.profile,
        "data_dir": str(args.data_dir),
        "output_root": str(args.output_root),
        "seed": args.seed,
        "dims": args.dims,
        "tau_r": args.tau_r,
        "lr_w": args.lr_w,
        "dt_r": args.dt_r,
        "dt_w": args.dt_w,
        "shuffle": args.shuffle,
        "profile_config": profile_config,
    }

    if args.stages in ("all", "core"):
        manifest["core"] = run_core_suite(args, profile_config, args.output_root)
    if args.stages in ("all", "shift"):
        manifest["shift"] = run_shift_suite(args, profile_config, args.output_root)

    manifest["robustness"] = {
        "noise_levels": list(profile_config["noise_levels"]),
        "eval_samples": int(profile_config["robustness_eval_samples"]),
    }

    save_json(args.output_root / "suite_manifest.json", manifest)
    print(f"Suite manifest written to: {args.output_root / 'suite_manifest.json'}")


if __name__ == "__main__":
    main()
