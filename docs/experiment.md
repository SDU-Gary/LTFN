# LTFN Experiment Notes

## Model

The implemented model follows the layered predictive-coding structure from [`start.md`](/Users/gokyrie/projects/ltfn/start.md):

- visible layer: `784`
- hidden layer 1: `256`
- hidden layer 2: `64`
- top layer: `32`

Generative weights are stored as:

- `W[0]`: `(784, 256)`
- `W[1]`: `(256, 64)`
- `W[2]`: `(64, 32)`

## Backends

The runtime now supports two interchangeable backends behind the same training loop:

- `cpu`: original Eigen implementation
- `cuda`: NVIDIA GPU implementation using CUDA + cuBLAS

The CUDA backend keeps the outer experiment flow unchanged:

- train and eval modes
- `energy.csv`
- `layer_errors.csv`
- `grad_norms.csv`
- `metrics.csv`
- `latent_top.csv`
- reconstruction image dumps
- checkpoint save/load

The first GPU migration pass covers the dominant per-step operators:

- forward predictive matrix-vector multiply `W[l] * r[l+1]`
- transpose multiply `W[l].transpose() * delta`
- local outer-product update `delta * r[l+1].transpose()`
- sigmoid / sigmoid derivative
- error and latent-state updates

Checkpoints remain backend-agnostic because weights are serialized in host format and re-uploaded when a CUDA model is created.

For each layer:

- `pred[l] = sigmoid(W[l] * r[l + 1])`
- `e[l] = r[l] - pred[l]`
- `E = 0.5 * sum_l ||e[l]||^2`

## Dynamics Mapping

One time step performs:

1. Compute all pre-activations, predictions, sigmoid derivatives, and prediction errors.
2. Update latent states synchronously using the old-step errors.
3. Recompute errors using the new states.
4. Update each weight matrix using a local outer-product rule.

Implemented state updates:

- top layer:
  - `r[L] += (dt_r / tau_r) * W[L-1]^T * (e[L-1] .* sigmoid'(W[L-1] * r[L]))`
- middle layers:
  - `r[l] -= (dt_r / tau_r) * (e[l] - W[l-1]^T * (e[l-1] .* sigmoid'(W[l-1] * r[l])))`

Weight update:

- `W[l] += lr_w * dt_w * ((e[l] .* sigmoid'(W[l] * r[l+1])) * r[l+1]^T)`

## Important Implementation Note

The pseudocode in `start.md` multiplies the middle-layer backpropagated term by an extra `sigmoid'(...)` factor after it has already been projected by `W^T`. That creates a shape mismatch for layers `l=1` and `l=2`.

The implemented code uses the energy gradient that is dimensionally correct:

- `grad_r_l = e[l] - W[l-1]^T * (e[l-1] .* sigmoid'(W[l-1] * r[l]))`

This preserves the intended predictive-coding update while making the system executable.

## Logger

The runtime logging path now goes through a dedicated `Logger` class. Its purpose is to keep experiment evidence separate from the LTFN update code while still allowing detailed tracing during training and inference.

Logger capabilities:

- configurable tag-based logging
- per-step or interval-based CSV output
- periodic evaluation summaries
- structured lifecycle events
- reconstruction artifact saving

Supported tags:

- `all`
- `core`
- `energy`
- `layer-errors`
- `gradients`
- `eval`
- `latents`
- `reconstructions`
- `events`

`core` enables the minimum evidence chain:

- global energy traces
- per-layer error norms
- per-layer gradient norms
- evaluation summaries

## Monitoring

The runtime monitoring system has two outputs.

Terminal output:

- epoch
- processed sample count
- recent training-window MSE
- evaluation MSE
- evaluation energy
- probe-sample energy
- images per second
- average weight norm
- average gradient norm
- final energy from the most recent training sample

File outputs:

- `energy.csv`: per-sample, per-step global energy and visible-layer MSE
- `layer_errors.csv`: per-layer error norms and squared norms
- `grad_norms.csv`: per-layer gradient norms, weight update norms, and weight norms
- `metrics.csv`: periodic scalar summaries
- `latent_top.csv`: top-layer representations `r_L` for the evaluation subset
- `events.jsonl`: run lifecycle events
- `config.json`: config snapshot and command line
- `reconstructions/*.pgm`: side-by-side original vs reconstruction images
- `checkpoints/latest.ltfnckpt`: binary model checkpoint

The CSV schema is designed for offline Python analysis. In particular:

- `energy.csv` covers both `train` and `probe` phases, so relaxation-speed comparisons across early, middle, and late training naturally fall out of the same table.
- `layer_errors.csv` exposes the hierarchical redistribution of prediction error during relaxation.
- `grad_norms.csv` exposes both raw gradient norms and actual applied update norms, making it easy to verify the intended time-scale separation between state updates and weight motion.
- `latent_top.csv` is designed for PCA or t-SNE style offline analysis of the learned high-level manifold.

## Offline Plotting

An offline plotting script is provided at [`scripts/plot_logs.py`](/Users/gokyrie/projects/ltfn/scripts/plot_logs.py).

It reads a run directory and produces:

- `metrics_summary.png`
- `energy_traces.png`
- `layer_error_norms.png`
- `gradient_norms.png`
- `latent_top_pca.png` when `latent_top.csv` is present

Example:

```bash
python3 scripts/plot_logs.py runs/mnist_v1
```

The script requires:

- `numpy`
- `matplotlib`

## Checkpoint Format

Checkpoint files store:

- magic number and version
- layer dimensions
- `tau_r`, `lr_w`, `dt_r`, `dt_w`
- `samples_seen`, `epochs_completed`, `seed`
- raw weight matrices

Loading requires the checkpoint architecture and hyperparameters to match the current run configuration.

## Smoke Test

`smoke-test` mode avoids MNIST and generates simple synthetic 28x28 patterns with light noise. This mode exists only to validate:

- build correctness
- runtime stability
- monitoring output generation
- checkpoint save/load plumbing

It is not meant to measure scientific quality.
