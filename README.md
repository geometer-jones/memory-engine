# memory-engine

Research code for the Memory Engine framework: a geometric memory/tape model,
its PyTorch runtime layers, hierarchical compositions, and supporting LLM/MNIST
experiments.

## Layout

- `engine.py`: minimal NumPy reference implementation.
- `memory_engine_layer.py`: production-oriented PyTorch runtime.
- `me_layer.py`: transformer-facing integration wrapper.
- `memory_engine_node.py`, `memory_engine_hierarchy.py`, `vision_memory_engine.py`:
  higher-level node, hierarchy, and vision compositions.
- `scripts/`: runnable training, evaluation, and analysis entry points.
- `tests/`: pytest suite.
- `docs/`: theory notes, findings, and paper drafts.
- `data/`, `results/`, `runs/`: datasets and generated artifacts.

## Architecture Summary

The repo centers on a persistent complex-valued tape state that is updated by
reception, renormalization, and consolidation dynamics. The core data flow is:

1. input signals enter a Memory Engine layer or node
2. the layer computes coupled reception and updates persistent tape state
3. higher-level wrappers reuse that state for transformer or vision models
4. scripts train, benchmark, or analyze the resulting dynamics

Structural changes to the engine belong in the core modules at repo root.
Experiment orchestration and reporting belong in `scripts/` and `docs/`.

## Running

Use module execution for entry points in `scripts/`:

- `python -m scripts.train_mnist_memory_engine --fake-data`
- `python -m scripts.train_mnist_me`
- `python -m scripts.run_diagnostics`
- `python -m scripts.evaluate_small_model_memory --model-name sshleifer/tiny-gpt2`

## Testing

Run the test suite with:

```bash
pytest
```

## Dependencies

There is no packaged build system yet. The code currently expects a Python
environment with at least:

- `numpy`
- `torch`
- `torchvision`
- `transformers`
- `matplotlib`
- `pytest`
