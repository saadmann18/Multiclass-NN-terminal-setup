.PHONY: help install install-dev test test-cov lint format type-check clean docs docker-build docker-run pre-commit

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install package in production mode"
	@echo "  install-dev  - Install package in development mode with dev dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage report"
	@echo "  lint         - Run linting (flake8)"
	@echo "  format       - Format code (black + isort)"
	@echo "  type-check   - Run type checking (mypy)"
	@echo "  clean        - Clean build artifacts"
	@echo "  docs         - Build documentation"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  pre-commit   - Install pre-commit hooks"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs]"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/mnist_cnn --cov-report=html --cov-report=term-missing

# Code quality
lint:
	flake8 src/ tests/

format:
	black src/ tests/
	isort src/ tests/

type-check:
	mypy src/mnist_cnn/

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Documentation
docs:
	mkdocs build

docs-serve:
	mkdocs serve

# Docker
docker-build:
	docker build -t mnist-cnn .

docker-run:
	docker run --rm -v $(PWD)/artifacts:/app/artifacts mnist-cnn

# Development setup
pre-commit:
	pre-commit install

# Training and evaluation shortcuts
train:
	python -m mnist_cnn.cli train

eval:
	python -m mnist_cnn.cli eval

# All quality checks
check: lint type-check test

# CI pipeline
ci: install-dev check test-cov

# Build package
build:
	python -m build

# Upload to PyPI (requires credentials)
upload-test:
	python -m twine upload --repository testpypi dist/*

upload:
	python -m twine upload dist/*
