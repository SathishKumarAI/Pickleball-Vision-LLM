.PHONY: help install test lint clean build run deploy setup-monitoring generate-ssl

help:
	@echo "Available commands:"
	@echo "  make install          - Install dependencies"
	@echo "  make test            - Run tests"
	@echo "  make lint            - Run linters"
	@echo "  make clean           - Clean build artifacts"
	@echo "  make build           - Build Docker images"
	@echo "  make run             - Run locally"
	@echo "  make deploy          - Deploy to production"
	@echo "  make setup-monitoring - Set up monitoring stack"
	@echo "  make generate-ssl    - Generate SSL certificates"

install:
	python -m pip install --upgrade pip
	pip install -e .
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ --cov=src/pickleball_vision --cov-report=xml --cov-report=html

lint:
	pre-commit run --all-files
	mypy src/pickleball_vision tests
	bandit -r src/pickleball_vision

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +

build:
	docker-compose build

run:
	docker-compose up

deploy:
	@if [ "$(ENVIRONMENT)" = "production" ]; then \
		echo "Deploying to production..."; \
		docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d; \
	else \
		echo "Please specify ENVIRONMENT=production"; \
		exit 1; \
	fi

setup-monitoring:
	@if [ -z "$(PROMETHEUS_PASSWORD)" ]; then \
		echo "Please set PROMETHEUS_PASSWORD environment variable"; \
		exit 1; \
	fi
	@if [ -z "$(GRAFANA_ADMIN_PASSWORD)" ]; then \
		echo "Please set GRAFANA_ADMIN_PASSWORD environment variable"; \
		exit 1; \
	fi
	./scripts/setup-monitoring.sh

generate-ssl:
	./scripts/generate-ssl.sh

logs:
	docker-compose logs -f

stop:
	docker-compose down

restart:
	docker-compose restart

monitoring-logs:
	docker-compose logs -f prometheus grafana alertmanager node-exporter

grafana-dashboard:
	@echo "Opening Grafana dashboard..."
	@if command -v xdg-open > /dev/null; then \
		xdg-open http://localhost:3000; \
	elif command -v open > /dev/null; then \
		open http://localhost:3000; \
	else \
		echo "Please open http://localhost:3000 in your browser"; \
	fi 