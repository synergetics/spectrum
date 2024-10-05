# Makefile for HOSA (Higher Order Spectral Analysis) package

# Python interpreter to use
PYTHON := python3

# Poetry for dependency management and packaging
POETRY := poetry

.PHONY: help install dev-install clean lint test coverage build publish

help:
	@echo "Available commands:"
	@echo "  install      : Install the package"
	@echo "  dev-install  : Install the package in editable mode with development dependencies"
	@echo "  clean        : Remove build artifacts and cache files"
	@echo "  lint         : Run linter (flake8) on the code"
	@echo "  test         : Run tests using pytest"
	@echo "  coverage     : Run tests and generate a coverage report"
	@echo "  build        : Build source and wheel distributions"
	@echo "  publish      : Publish the package to PyPI"

install:
	$(POETRY) install

dev-install:
	$(POETRY) install --dev

clean:
	rm -rf build dist .eggs *.egg-info
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

lint:
	$(POETRY) run flake8 hosa tests

test:
	$(POETRY) run pytest

coverage:
	$(POETRY) run pytest --cov=hosa --cov-report=term-missing --cov-report=xml

build:
	$(POETRY) build

publish:
	$(POETRY) publish

# Local development server for documentation (if using Sphinx)
docs-serve:
	cd docs && $(MAKE) html && cd _build/html && $(PYTHON) -m http.server 8000

# Generate requirements.txt from poetry.lock
requirements:
	$(POETRY) export -f requirements.txt --output requirements.txt --without-hashes
