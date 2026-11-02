.PHONY: help install test test-unit test-integration test-security test-load coverage lint

help:
	@echo "FlashDet development targets"
	@echo "  make install           install package + dev extras"
	@echo "  make test              unit + integration tests"
	@echo "  make test-unit         fast unit + smoke tests"
	@echo "  make test-integration  pipeline / CLI / checkpoint tests"
	@echo "  make test-security     security regression tests + static scan"
	@echo "  make test-load         simulate 1000 concurrent inference users"
	@echo "  make coverage          unit tests with coverage report"
	@echo "  make lint              ruff over package, tests and scripts"

install:
	pip install -e ".[dev,tracker,solutions,analytics]"

test: test-unit test-integration

test-unit:
	pytest tests/unit tests/smoke -q

test-integration:
	pytest tests/integration -q

test-security:
	pytest tests/unit/utils/test_checkpoint_security.py \
	       tests/unit/data/test_download_security.py \
	       tests/unit/engine/test_predictor_validation.py -q
	python scripts/security_scan.py

test-load:
	python scripts/load_test.py --users 1000 --concurrency 32 --size 128

coverage:
	pytest tests/unit tests/smoke -q --cov=flashdet --cov-report=term-missing

lint:
	ruff check flashdet tests scripts train.py test.py
