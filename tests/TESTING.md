# Testing Strategy

FlashDet uses three test layers plus two non-functional layers.

```
                 ┌──────────────────────────┐
                 │  Load / chaos simulation │  scripts/load_test.py
                 ├──────────────────────────┤
                 │  Security regression     │  tests/unit/*security*
                 ├──────────────────────────┤
                 │  Integration             │  tests/integration
                 ├──────────────────────────┤
                 │  Unit + smoke            │  tests/unit, tests/smoke
                 └──────────────────────────┘
```

## 1. Unit tests

Fast, deterministic tests for one component at a time.

- location: `tests/unit/`
- marker: none
- must not download weights, datasets or use a GPU
- examples: model forwards, loss shapes, layer math, Pico-Zoo presets,
  checkpoint safety, archive extraction, input validation

```bash
pytest tests/unit -q
pytest tests/unit/models/test_pico_zoo.py -v
```

## 2. Integration tests

Exercise real multi-component flows and real files on disk.

- location: `tests/integration/`
- examples: train step → save checkpoint → predictor load → prediction,
  CLI commands, ONNX export, Pico-Zoo checkpoint round-trip
- slow tests use the `@pytest.mark.slow` marker and are skipped by default

```bash
pytest tests/integration -q
pytest tests/integration -q --runslow
```

## 3. Test-driven development workflow

1. **Red:** write the failing test first. For a bug, use the smallest input that
   reproduces it.
2. **Green:** implement the smallest fix.
3. **Refactor:** clean up while the test stays green.
4. **Regress:** keep the test. Security and data-integrity bugs always get a
   permanent test.

Example: the Pico validation CSV bug was fixed by first adding
`test_training_csv_aligns_validation_by_epoch` in
`tests/unit/models/test_pico_zoo.py`, then changing `_save_training_csv`.

## 4. Security regression tests

- location: `tests/unit/utils/test_checkpoint_security.py`,
  `tests/unit/data/test_download_security.py`,
  `tests/unit/engine/test_predictor_validation.py`
- static scan: `python scripts/security_scan.py`
- findings and risk decisions: `docs/SECURITY-AUDIT.md`

```bash
make test-security
```

## 5. Load / chaos simulation

`scripts/load_test.py` creates a Pico checkpoint, loads one predictor, then runs
N simulated users through a thread pool. Every Nth user sends a malformed image.

```bash
python scripts/load_test.py --users 1000 --concurrency 32 --size 128
python scripts/load_test.py --users 5000 --concurrency 64 --size 320 \
  --output /tmp/load-test.json
```

The script reports p50/p95/p99 latency, requests/second, peak RSS, normal
failures, rejected chaos payloads and unexpected errors. It exits non-zero if a
normal request fails or a malformed request causes an unexpected exception.

## 6. Commands

```bash
make test-unit           # fast
make test-integration    # pipeline
make test-security       # security regression + scan
make test-load           # 1000 concurrent users
make coverage            # coverage report
make lint                # ruff
```

## 7. Conventions

- Tests are independent and use `tmp_path`, never the real `data/` or
  `workspace/`.
- Fix random seeds when behaviour depends on randomness.
- A test that needs > 10 seconds gets `@pytest.mark.slow`.
- A test that needs CUDA gets `@pytest.mark.gpu`.
- Prefer asserting behaviour over implementation details.
- Security tests must also assert the **negative** outcome (e.g. the payload
  marker file does not exist).
