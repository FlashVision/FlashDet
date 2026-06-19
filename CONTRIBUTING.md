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
4. Run tests: `flashdet check`
5. Commit and push
6. Open a Pull Request

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting (line length: 120)
- Type hints are encouraged
- Docstrings for all public functions (Google style)
- No hardcoded file paths — use relative or configurable paths

## Adding a New Solution

1. Create `flashdet/solutions/your_solution.py`
2. Follow the existing pattern: accept `predictor` + optional `tracker`
3. Implement `process_frame(frame)` → `(annotated_frame, results)`
4. Implement `get_results()` and `reset()`
5. Add to `flashdet/solutions/__init__.py`

## Adding a New Tracker

1. Create `flashdet/trackers/your_tracker.py`
2. Implement `update(detections)` → `np.ndarray` of shape `(N, 7)`
3. Implement `reset()`
4. Add to `flashdet/trackers/__init__.py`

## Commit Messages

Use clear, descriptive messages:
- `Add object counting solution`
- `Fix NMS threshold handling in predictor`
- `Update README with tracking examples`

## Reporting Issues

- Use GitHub Issues
- Include: Python version, PyTorch version, GPU info, error traceback
- Run `flashdet settings` and paste the output

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
