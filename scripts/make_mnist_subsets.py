#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ltfn_experiment_utils import (
    ensure_dir,
    filter_mnist_by_digits,
    save_json,
    write_mnist_images,
    write_mnist_labels,
)


def parse_digits(spec: str) -> Tuple[str, List[int]]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"Subset spec must look like name:0,1,2, got {spec!r}")
    name, raw_digits = spec.split(":", 1)
    digits = []
    for token in raw_digits.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0 or value > 9:
            raise argparse.ArgumentTypeError(f"Digit must be in [0, 9], got {value}")
        digits.append(value)
    if not name:
        raise argparse.ArgumentTypeError(f"Subset name must not be empty in {spec!r}")
    if not digits:
        raise argparse.ArgumentTypeError(f"Subset {name!r} must contain at least one digit")
    return name, digits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create filtered MNIST subset directories for LTFN experiments.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing original MNIST ubyte files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory where subset folders will be created.",
    )
    parser.add_argument(
        "--spec",
        type=parse_digits,
        action="append",
        default=[],
        help="Subset spec in the form name:digit,digit,...",
    )
    return parser.parse_args()


def default_specs() -> Sequence[Tuple[str, Sequence[int]]]:
    return (
        ("digits_0_4", [0, 1, 2, 3, 4]),
        ("digits_5_9", [5, 6, 7, 8, 9]),
    )


def main() -> None:
    args = parse_args()
    specs = args.spec or list(default_specs())

    train_images = args.source_dir / "train-images-idx3-ubyte"
    train_labels = args.source_dir / "train-labels-idx1-ubyte"
    test_images = args.source_dir / "t10k-images-idx3-ubyte"
    test_labels = args.source_dir / "t10k-labels-idx1-ubyte"

    manifest: Dict[str, Dict[str, object]] = {}
    ensure_dir(args.output_dir)

    for name, digits in specs:
        subset_dir = args.output_dir / name
        ensure_dir(subset_dir)

        train_subset, train_subset_labels = filter_mnist_by_digits(train_images, train_labels, digits)
        test_subset, test_subset_labels = filter_mnist_by_digits(test_images, test_labels, digits)

        write_mnist_images(subset_dir / "train-images-idx3-ubyte", train_subset)
        write_mnist_labels(subset_dir / "train-labels-idx1-ubyte", train_subset_labels)
        write_mnist_images(subset_dir / "t10k-images-idx3-ubyte", test_subset)
        write_mnist_labels(subset_dir / "t10k-labels-idx1-ubyte", test_subset_labels)

        manifest[name] = {
            "digits": list(digits),
            "train_count": int(train_subset.shape[0]),
            "test_count": int(test_subset.shape[0]),
            "dir": str(subset_dir),
        }

        print(
            f"{name}: digits={list(digits)} train={train_subset.shape[0]} test={test_subset.shape[0]} -> {subset_dir}"
        )

    save_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
