.PHONY: help install test lint format clean build run train docker-build docker-run

help: ## Show this help message
	@echo "Sproxxo Fraud Detection MLOps Platform"
	@echo "======================================"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:
	uv sync

install-dev:
	uv sync --extra dev

install-mlops:
	uv sync --extra mlops

install-all:
	uv sync --all-extras

run-api-examples:
	uv run sproxxo-api-client

run-api:
	uv run sproxxo-api

train:
	uv run sproxxo-train --n-samples 10000 --model-type xgboost

train-quick:
	uv run sproxxo-train --n-samples 1000 --model-type xgboost

docker-sproxxo-api:
	docker compose up -d --build sproxxo-api

docker-compose-up: ## Start all services with Docker Compose
	docker compose up -d

docker-compose-down: ## Stop all services
	docker compose down

docker-compose-logs: ## View Docker Compose logs
	docker compose logs -f

logs: ## View application logs
	docker compose logs -f sproxxo-api

security-check:
	uv run bandit -r sproxxo/

check-all: ## Run all checks (lint, test, security)
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) security-check

tox:
	uv run tox

tox-lint:
	uv run tox -e lint

tox-format: ## actually fix formatting
	uv run tox -e format-fix

format-check: ## Check code formatting
	uv run tox -e format

tox-test:
	uv run tox -e test

# Model management
list-models: ## List available models
	uv run python -c "from sproxxo.models import ModelManager; mm = ModelManager(); print('Available models:'); [print(f'- {m.version}: {m.model_type}') for m in mm.list_models()]"

check-health: ## Check API health
	curl -f http://localhost:8000/health || echo "API not responding"

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .tox/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
