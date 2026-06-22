# Contributing to FlashDet

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/FlashVision/FlashDet.git
cd FlashDet
pip install -e ".[dev,all]"
```

## Development Workflow

1. Create a branch: `git checkout -b feature/your-feature`
2. Make changes
3. Run lint: `ruff check flashdet/`
4. Run tests: `pytest tests/`
5. Commit and push
6. Open a Pull Request

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting (line length: 120)
- Type hints are encouraged
- Docstrings for all public functions (Google style)
- No hardcoded file paths — use relative or configurable paths

## Adding a New Architecture

1. Create `flashdet/models/architectures/my_model.py`
2. Decorate with `@DETECTORS.register("MyModel")`
3. Implement `forward(x, gt_meta, compute_loss)` and `predict(x)` interfaces
4. Add to `flashdet/models/architectures/__init__.py`
5. Add entry in `flashdet/models/detector.py` `ARCHITECTURE_REGISTRY`

## Adding a New Training Method

1. Create `flashdet/engine/training/my_trainer.py`
2. Subclass `Trainer` or write a standalone class
3. Override `train()` for setup, `_train_one_epoch()` for custom logic
4. Add to `flashdet/engine/training/__init__.py`

## Adding a New Loss Function

1. Create `flashdet/losses/my_loss.py`
2. Add to `flashdet/losses/__init__.py`

## Adding a New Layer / Block

1. Create `flashdet/models/layers/my_block.py`
2. Add to `flashdet/models/layers/__init__.py`
3. Optionally re-export in `flashdet/nn/__init__.py`

## Adding a New Solution

1. Create `flashdet/solutions/your_solution.py`
2. Implement `process_frame(frame)` and `get_results()`
3. Add to `flashdet/solutions/__init__.py`

## Adding a New Tracker

1. Create `flashdet/trackers/your_tracker.py`
2. Implement `update(detections)` and `reset()`
3. Register with `TRACKERS.register("MyTracker")`
4. Add to `flashdet/trackers/__init__.py`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
