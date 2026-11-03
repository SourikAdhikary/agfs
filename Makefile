.PHONY: help sync check test test-unit test-chaos test-bench doctor bench clean

PYTHON ?= $(shell [ -x /Users/sourik/Projects/.venv/bin/python ] && echo /Users/sourik/Projects/.venv/bin/python || echo python3)
PYTEST ?= $(shell [ -x /Users/sourik/Projects/.venv/bin/pytest ] && echo /Users/sourik/Projects/.venv/bin/pytest || echo pytest)
RUFF   ?= $(shell [ -x /Users/sourik/Projects/vide-proposal/.venv/bin/ruff ] && echo /Users/sourik/Projects/vide-proposal/.venv/bin/ruff || echo "uvx ruff")

help:
	@echo "AgFS (SQLite-Agent-OS) Developer Commands:"
	@echo "  make check       - Run Ruff linter across src/, benchmarks/, and tests/"
	@echo "  make test        - Run full unit, integration, chaos, and A/B benchmark test suite"
	@echo "  make test-unit   - Run fast unit & layer-boundary tests (<3s)"
	@echo "  make test-chaos  - Run fault-injection, Merkle tamper & rollback tests"
	@echo "  make bench       - Run the 9-part A/B Benchmark Suite (Without vs With AgFS) & write BENCHMARKS.md"
	@echo "  make doctor      - Run the 12 formal kernel invariant checks (INV-01..INV-12)"
	@echo "  make clean       - Remove build, cache, and temporary .agdb files"

check:
	$(RUFF) check src benchmarks tests

test-unit:
	PYTHONPATH=src:. $(PYTEST) tests/unit tests/integration

test-chaos:
	PYTHONPATH=src:. $(PYTEST) tests/chaos

test-bench:
	PYTHONPATH=src:. $(PYTEST) tests/benchmarks

test:
	PYTHONPATH=src:. $(PYTEST) tests/

bench:
	PYTHONPATH=src:. $(PYTHON) -m agfs.interfaces.cli bench --out BENCHMARKS.md

doctor:
	PYTHONPATH=src:. $(PYTHON) -m agfs.interfaces.cli doctor

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build *.egg-info
	find . -name "*.agdb*" -delete
