# LTFN Validation Suite

This document turns the scientific claims around LTFN into a runnable experiment suite.

## Goal

The suite is designed to generate two categories of evidence:

- functional validity: LTFN can stably learn MNIST reconstruction without backpropagation
- design advantages: unified inference and learning, online adaptation, robustness, and emergent weight structure

## Components

The workflow is split into three scripts:

- `scripts/make_mnist_subsets.py`
  - creates filtered MNIST directories such as digits `0-4` and `5-9`
- `scripts/run_validation_suite.py`
  - runs staged convergence training
  - saves an initial random checkpoint
  - runs probe-trace evaluations at selected checkpoints
  - runs a distribution-shift experiment on filtered digit subsets
- `scripts/analyze_validation_suite.py`
  - aggregates stage metrics
  - generates report figures
  - computes automated pass/fail summaries for the target criteria
  - optionally runs robustness evaluations by injecting weight noise into checkpoints

## Recommended Execution

Build with CUDA:

```bash
cmake -S . -B build_cuda -DLTFN_ENABLE_CUDA=ON
cmake --build build_cuda -j8
```

Run the pilot suite:

```bash
python3 scripts/run_validation_suite.py \
  --bin build_cuda/ltfn_train \
  --data-dir data/mnist \
  --output-root runs/validation_pilot \
  --backend cuda \
  --profile pilot
```

Analyze the suite:

```bash
python3 scripts/analyze_validation_suite.py \
  --suite-root runs/validation_pilot
```

For a full report run:

```bash
python3 scripts/run_validation_suite.py \
  --bin build_cuda/ltfn_train \
  --data-dir data/mnist \
  --output-root runs/validation_full \
  --backend cuda \
  --profile full

python3 scripts/analyze_validation_suite.py \
  --suite-root runs/validation_full
```

## Evidence Mapping

The suite maps each claim to concrete artifacts:

- criterion 1 and 6:
  - `metrics.csv`
  - `report/01_mse_and_energy.png`
- criterion 2 and 10:
  - probe-only eval runs under `convergence/traces/`
  - `report/02_relaxation_traces.png`
- criterion 3:
  - staged convergence metrics and probe energy
  - `report/01_mse_and_energy.png`
- criterion 4:
  - `layer_errors.csv`
  - `report/03_layer_error_balance.png`
- criterion 5:
  - `grad_norms.csv`, including `state_update_norm`
  - `report/04_gradient_timescales.png`
- criterion 7:
  - `shift/evals/`
  - `report/06_shift_adaptation.png`
- criterion 8:
  - `robustness/evals/`
  - `report/07_robustness.png`
- criterion 9:
  - initial and final checkpoints
  - `report/05_weight_spectra.png`

## Notes

- The suite saves an explicit initial checkpoint before any training step. This is required for comparing random-initialized and trained dynamics.
- `grad_norms.csv` now logs `state_update_norm`, which is used to quantify the intended time-scale separation between state relaxation and weight motion.
- The staged training runner uses monotonic sample targets and resume checkpoints instead of relying on one giant monolithic run. This preserves early, middle, and late checkpoints for later analysis.
