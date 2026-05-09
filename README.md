# LTFN C++ Experiment System

Cross-platform C++17 implementation of the Layered Transient Free Energy Network (LTFN) described in [`start.md`](/Users/gokyrie/projects/ltfn/start.md). The system trains a predictive-coding style network on MNIST reconstruction without backpropagation and includes terminal plus file-based monitoring.

## Features

- `C++17 + Eigen` implementation of unified state relaxation and local weight updates
- Cross-platform `CMake` build for macOS, Linux, and Windows
- MNIST `ubyte` loader with no external dataset dependency
- Training, evaluation, and synthetic `smoke-test` modes
- Monitoring outputs:
  - terminal progress summaries
  - `metrics.csv`
  - `energy.csv`
  - `layer_errors.csv`
  - `grad_norms.csv`
  - `latent_top.csv`
  - `events.jsonl`
  - `config.json`
  - reconstruction images in `PGM`
  - binary checkpoints with resume support
  - step-tag toggles via a lightweight `Logger` class
  - offline Python plotting script

## GPU Status

An optional NVIDIA CUDA backend is available.

Current scope:

- CPU backend remains the default and stays fully supported
- CUDA backend targets NVIDIA GPUs through `nvcc + cuBLAS`
- training, evaluation, logger outputs, checkpoints, latent export, and reconstructions are preserved on both backends
- checkpoints are backend-agnostic and can be loaded across CPU and CUDA runs

The first CUDA pass moves the heaviest per-step operators to GPU:

- `W[l] * r[l+1]`
- `W[l].transpose() * delta`
- `delta * r[l+1].transpose()`
- sigmoid / sigmoid derivative
- error vectors and latent-state updates

## Build

```bash
cmake -S . -B build
cmake --build build -j4
```

`CMake` first looks for an installed `Eigen3`. If it is not available, it fetches Eigen `3.4.0` automatically.

To disable CUDA explicitly and build CPU-only:

```bash
cmake -S . -B build_cpu -DLTFN_ENABLE_CUDA=OFF
cmake --build build_cpu -j4
```

To build with CUDA enabled when `nvcc` is available:

```bash
cmake -S . -B build_cuda -DLTFN_ENABLE_CUDA=ON
cmake --build build_cuda -j4
```

## Data Layout

Place MNIST image files under one directory:

```text
<data-dir>/
  train-images-idx3-ubyte
  t10k-images-idx3-ubyte
```

The system does not require label files because training and evaluation are reconstruction-only.

## Typical Commands

Train on MNIST:

```bash
./build/ltfn_train \
  --mode train \
  --backend cuda \
  --data-dir /path/to/mnist \
  --output-dir runs/mnist_v1 \
  --steps 200 \
  --eval-interval 500 \
  --eval-samples 1000 \
  --checkpoint-interval 5000
```

Resume training:

```bash
./build/ltfn_train \
  --mode train \
  --backend cpu \
  --data-dir /path/to/mnist \
  --output-dir runs/mnist_resume \
  --resume runs/mnist_v1/checkpoints/latest.ltfnckpt
```

Evaluate a checkpoint:

```bash
./build/ltfn_train \
  --mode eval \
  --backend cuda \
  --data-dir /path/to/mnist \
  --resume runs/mnist_v1/checkpoints/latest.ltfnckpt \
  --output-dir runs/mnist_eval
```

Run the synthetic smoke test:

```bash
./build/ltfn_train \
  --mode smoke-test \
  --backend cuda \
  --output-dir runs/smoke \
  --steps 20 \
  --max-train-samples 32 \
  --eval-samples 8
```

Show command-line help:

```bash
./build/ltfn_train --help
```

Generate plots from a finished run:

```bash
python3 scripts/plot_logs.py runs/mnist_v1
```

The plotting script expects `numpy` and `matplotlib`.

## Output Layout

Each run writes:

```text
<output-dir>/
  config.json
  energy.csv
  events.jsonl
  grad_norms.csv
  latent_top.csv
  layer_errors.csv
  metrics.csv
  checkpoints/
    latest.ltfnckpt
  reconstructions/
    step_00000500_idx_0000.pgm
    ...
```

`metrics.csv` contains periodic evaluation summaries. `energy.csv` stores relaxation energy traces by phase and step. `layer_errors.csv` stores per-layer error norms. `grad_norms.csv` stores per-layer gradient norms, weight update norms, and weight norms. `latent_top.csv` stores top-layer representations `r_L` for the evaluation subset at each evaluation point. `events.jsonl` contains structured lifecycle events such as run start, evaluation, and checkpoint saves. Reconstruction images are stored as side-by-side original/reconstruction grayscale `PGM` files.

## Key Arguments

- `--mode train|eval|smoke-test`
- `--backend cpu|cuda`
- `--data-dir PATH`
- `--output-dir PATH`
- `--resume PATH`
- `--checkpoint PATH`
- `--dims 784,256,64,32`
- `--tau-r FLOAT`
- `--lr-w FLOAT`
- `--dt-r FLOAT`
- `--dt-w FLOAT`
- `--steps INT`
- `--max-epochs INT`
- `--max-train-samples INT`
- `--eval-samples INT`
- `--eval-interval INT`
- `--checkpoint-interval INT`
- `--probe-index INT`
- `--recon-samples INT`
- `--logger-step-interval INT`
- `--logger-tags LIST`
- `--shuffle true|false`
- `--seed INT`

`--logger-tags` supports comma-separated values:

- `all`
- `core`
- `energy`
- `layer-errors`
- `gradients`
- `eval`
- `latents`
- `reconstructions`
- `events`

Example:

```bash
./build/ltfn_train \
  --mode train \
  --data-dir /path/to/mnist \
  --output-dir runs/mnist_logs \
  --logger-step-interval 10 \
  --logger-tags core,reconstructions,events
```

## Verification

Local verification performed during implementation:

- `cmake -S . -B build`
- `cmake --build build -j4`
- `ctest --test-dir build --output-on-failure`
- `./build/ltfn_train --mode smoke-test --output-dir runs/manual_smoke --steps 10 --max-train-samples 16 --eval-samples 8 --eval-interval 8 --checkpoint-interval 16`
- `./build/ltfn_train --mode smoke-test --output-dir runs/logger_smoke --steps 10 --max-train-samples 8 --eval-samples 8 --eval-interval 8 --checkpoint-interval 8 --logger-step-interval 5 --logger-tags all`

## Files

- [`CMakeLists.txt`](/Users/gokyrie/projects/ltfn/CMakeLists.txt)
- [`include/logger.h`](/Users/gokyrie/projects/ltfn/include/logger.h)
- [`include/ltfn.h`](/Users/gokyrie/projects/ltfn/include/ltfn.h)
- [`include/utils.h`](/Users/gokyrie/projects/ltfn/include/utils.h)
- [`src/logger.cpp`](/Users/gokyrie/projects/ltfn/src/logger.cpp)
- [`src/ltfn.cpp`](/Users/gokyrie/projects/ltfn/src/ltfn.cpp)
- [`src/utils.cpp`](/Users/gokyrie/projects/ltfn/src/utils.cpp)
- [`src/main.cpp`](/Users/gokyrie/projects/ltfn/src/main.cpp)
- [`scripts/plot_logs.py`](/Users/gokyrie/projects/ltfn/scripts/plot_logs.py)
- [`docs/experiment.md`](/Users/gokyrie/projects/ltfn/docs/experiment.md)
