# LTFN-CAM TinyLM v3 Experiment

This document defines the first language-modeling prototype for LTFN.
It is intentionally isolated from the MNIST reconstruction executable.

## Goal

Show that a no-backpropagation LTFN system can learn causal character
statistics with local energy relaxation and local weight updates.

The first milestone is not to match a Transformer. The first milestone is:

- next-character cross entropy and BPC decrease during training
- the full CAM model beats the vertical-only ablation
- the model is compared against 2-gram and 3-gram baselines
- generated text moves from random characters toward valid local spelling

## Implemented Prototype

Executable:

```bash
./build/ltfn_tinylm
```

Core files:

- `include/tinylm.h`
- `src/tinylm.cpp`
- `src/tinylm_main.cpp`

The prototype implements:

- character-level corpus loading, with a built-in smoke corpus when no file is provided
- local softmax readout on the top state
- prediction state and learning state separation
- prediction state writes a delayed causal transition memory
- learning state sees the target token but never writes to causal memory
- normalized causal associative memory:

```text
M_t = decay * M_{t-1} + phi(k_{t-1}) * v_t^T
z_t = decay * z_{t-1} + phi(k_{t-1})
context_t = M_{t-1}^T phi(q_t) / (z_{t-1}^T phi(q_t) + eps)
```

- shared random query/key projection for the Stage 1 frozen associative space
- CAM context can be added directly to the local softmax readout
- adaptive precision balancing for input, horizontal, vertical, and output errors
- Stage 1 default: frozen query/key projections
- optional local query update with current-token retrieval error
- optional key alignment update for later experiments
- vertical-only ablation switch
- 2-gram and 3-gram baseline BPC reporting
- `metrics.csv`, `events.jsonl`, `config.json`, `vocab.txt`, and `generated.txt`

## Leakage Rule

For each position `t`:

```text
prefix memory M_{t-1}
current char x_t

1. prediction inference:
   r_pred_t = relax(x_t, M_{t-1}, no target)
   predict x_{t+1}
   write the previous prediction key to the current prediction value -> M_t

2. learning inference:
   r_learn_t = relax(x_t, M_{t-1}, target x_{t+1})
   update local weights
   do not write r_learn_t into memory
```

This keeps target-shaped learning states out of future context while still allowing
the memory to store causal transitions that are already observable.

## First Commands

Smoke run:

```bash
./build/ltfn_tinylm \
  --output-dir runs/tinylm_smoke_v3 \
  --max-train-chars 4096 \
  --eval-chars 1024 \
  --max-epochs 1 \
  --hidden-dim 64 \
  --steps 8 \
  --seq-len 64
```

Vertical-only ablation:

```bash
./build/ltfn_tinylm \
  --output-dir runs/tinylm_smoke_vertical_only_v3 \
  --max-train-chars 4096 \
  --eval-chars 1024 \
  --max-epochs 1 \
  --hidden-dim 64 \
  --steps 8 \
  --seq-len 64 \
  --vertical-only true
```

TinyShakespeare run, after downloading a plain text corpus:

```bash
./build/ltfn_tinylm \
  --text-file data/text/tinyshakespeare.txt \
  --output-dir runs/tinylm_shakespeare_stage1_v3 \
  --max-train-chars 200000 \
  --eval-chars 20000 \
  --max-epochs 1 \
  --layers 2 \
  --hidden-dim 128 \
  --steps 12 \
  --seq-len 64 \
  --learn-q false \
  --learn-k false
```

## Experiment Ladder

Stage 1:

- freeze query and key projections
- train readout, vertical generative weights, and value projection
- compare full CAM against vertical-only

Stage 2:

- enable `--learn-q true`
- check whether BPC improves over Stage 1

Stage 3:

- enable key learning only after Stage 2 is stable
- replace the current conservative key alignment rule with a stronger whitening or Oja-style rule if needed

## Known Limits

This is a CPU/Eigen correctness prototype. It does not yet implement:

- checkpoints
- mini-batching
- CUDA kernels
- BPE or wordpiece tokenization
- a BP Transformer/RNN baseline

Those should wait until the Stage 1 learning curve is credible.

Current smoke observations:

- built-in repeated corpus, 4096 train chars, 1024 eval chars, 1 epoch:
  vertical-only reached about `3.63` eval BPC
- original full CAM with transition memory and all-layer CAM readout reached about
  `4.15` eval BPC under the same short run
- after hetero-associative CAM changes, full CAM reached about `3.62` eval BPC
  in 1 epoch and about `3.28` best eval BPC over 5 epochs, beating the
  5-epoch vertical-only result of about `3.49`
- TinyShakespeare sanity run, 12000 train chars, 2048 eval chars, 1 epoch:
  full CAM reached about `4.82` eval BPC, while the bigram baseline was about `3.69`
- TinyShakespeare 5-epoch validation after hetero-associative CAM changes:
  full CAM reached about `4.84` best eval BPC and ended around `4.93`, while
  vertical-only ended around `5.21`; bigram remained much stronger at about `3.69`

Interpretation: the no-BP local-softmax learning loop is functional, but the current
CAM mechanism is only a modest positive contributor after the hetero-associative
fix. The next architectural work should focus on making the horizontal memory target
more predictive before scaling the model.
