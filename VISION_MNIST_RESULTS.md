# Vision Memory Engine: MNIST Notes

Date: 2026-04-15

This note documents the image recognizer work added on top of the Memory Engine, the handwritten-digit probe built around MNIST, what the current metrics actually mean, and where the approach is still weak.

## What shipped

New code paths:

- `vision_memory_engine.py`
  - `VisionMemoryRecognizer` turns an image into a patch sequence and runs that sequence through `MemoryEngineLayer`.
  - The current default path includes a small convolutional stem before patchification so the ME sees richer local stroke structure.
- `train_mnist_memory_engine.py`
  - trains and evaluates the vision model on MNIST
  - supports fake-data smoke runs
  - builds digit prototypes from final ME tape states
  - reports both classifier accuracy and prototype-based digit resonance accuracy

Support work:

- `memory_engine_layer.py`
  - mutable basis/coupling cache tensors were detached/cloned at the projection boundary to avoid autograd version-counter failures during training
- tests
  - `test_vision_memory_engine.py`
  - `test_train_mnist_memory_engine.py`

## Architecture

The image path is:

1. image
2. optional conv stem
3. patch embedding
4. patch tokens + positional embeddings
5. `MemoryEngineLayer`
6. pooled final token state
7. classifier head

The important point is that the Memory Engine is still operating on a sequence. We did not build a separate vision-only memory system. We changed the source of the sequence.

## How digit resonance is measured

The repository now has two different notions of "performance":

1. classifier accuracy
   - standard logits from the classifier head
2. digit resonance probe accuracy
   - a diagnostic readout that asks whether the final complex tape state looks most resonant with the prototype tape for the correct digit

The probe works like this:

1. run training images through the model
2. collect the final `state.tape` for each image
3. average and renormalize those tape states per digit, `0` through `9`
4. for a new image, compare its final tape against each digit prototype
5. score each comparison by resonance fraction

Important: the probe is post hoc. It is not the main classifier and it is not trained as a supervised head.

## Are we using `L` or raw Hadamard?

Inside `MemoryEngineLayer`, the receive path does use `L`.

- The layer computes `coords = E^T h`
- Then `coupled = L coords`
- Then reception is formed as a Hadamard interaction between the effective coordinates and the tape

So the model path is not "raw Hadamard only."

But the current digit probe is different. It does **not** run a fresh coupled projection with `L` for each digit. It compares:

- final tape for the image
- stored prototype tape for each digit

with a direct tape/prototype alignment and then scores resonance on that alignment.

So:

- model internals: `L` plus Hadamard
- digit probe: raw tape/prototype Hadamard-style alignment

One more wrinkle: in the current vision config, `memory_dim == hidden_dim`, the basis starts close to identity, and transient growth is disabled by default. In that regime, `L` is often close to identity anyway. That means the current setup behaves much closer to coordinatewise interaction than the full coupled theory allows.

## Benchmarks

### Before conv stem

Config:

- dataset: MNIST
- train limit: `2000`
- test limit: `500`
- epochs: `3`
- device: CPU
- digit probe batches: `8`

Results:

| Epoch | Train Acc | Test Acc | Digit Probe Acc |
|---|---:|---:|---:|
| 1 | 0.1335 | 0.0980 | 0.0680 |
| 2 | 0.1390 | 0.2200 | 0.1160 |
| 3 | 0.1860 | 0.2060 | 0.0840 |

Interpretation:

- classifier barely rose above chance
- resonance probe was worse than the classifier
- per-digit resonance saturated toward `1.0`, which made the probe non-discriminative

### After conv stem

Config:

- same benchmark slice as above
- default conv stem enabled with `--stem-channels 32,64`

Results:

| Epoch | Train Acc | Test Acc | Digit Probe Acc |
|---|---:|---:|---:|
| 1 | 0.1920 | 0.2760 | 0.2740 |
| 2 | 0.2740 | 0.3400 | 0.2640 |
| 3 | 0.3705 | 0.3940 | 0.2680 |

Interpretation:

- the conv stem helped a lot
- test accuracy improved from `0.2060` to `0.3940`
- digit probe accuracy improved from `0.0840` to `0.2680`
- the probe is still weak, but no longer completely useless

## What is going wrong now

The current resonance probe still saturates.

By epoch 3, the true-digit resonance values are extremely high across nearly all digits, often near `0.99` or `1.00`. That means the probe is saying, in effect, "this image resonates with almost everything."

That is the core problem.

Why this happens:

- the probe uses resonance fraction, which is permissive
- the prototypes live in final tape space, not a more contrastive class-specific space
- the current vision setup is still relatively shallow
- `L` is present in the model, but the diagnostic probe itself is not using a class-conditioned coupled projection

## Practical reading of the metrics

Right now:

- `test_acc` tells you whether the recognizer is learning anything useful
- `digit_probe_acc` tells you whether the ME tape is class-selective enough to support a prototype-style resonance interpretation

At the moment:

- recognizer: somewhat learning
- resonance probe: not yet trustworthy as an explanation of "the ME resonated with digit 7"

It is better than random after adding the conv stem. Still not good enough to claim a strong mechanistic interpretation.

## Commands

Smoke test with fake data:

```bash
python3 train_mnist_memory_engine.py --fake-data --epochs 1 --train-limit 64 --test-limit 32 --batch-size 16 --eval-batch-size 16 --device cpu
```

Real MNIST slice with per-digit resonance reporting:

```bash
python3 train_mnist_memory_engine.py \
  --data-dir /tmp/mnist_fresh \
  --epochs 3 \
  --train-limit 2000 \
  --test-limit 500 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --device cpu \
  --digit-probe-batches 8 \
  --report-per-digit
```

Disable the conv stem:

```bash
python3 train_mnist_memory_engine.py --stem-channels ''
```

## Current recommendation

Do not over-interpret the current digit resonance probe.

The conv stem clearly improved the front end. Keep it.

The next thing to fix is the probe itself. The best next candidates are:

1. contrastive resonance score
   - `res(true_digit) - mean(res(other_digits))`
2. learned readout on `state.tape`
   - a supervised probe that tests whether class information is actually in the tape
3. class-conditioned coupled probe
   - build prototypes in `alpha = L E^T h` style coordinates rather than only in final tape space

That is the whole game now. The front end got better. The interpretability readout still needs sharper math.
