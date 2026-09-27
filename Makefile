.PHONY: help infra-up infra-down backend frontend test lint format build all

help:
	@echo "Available commands:"
	@echo "  make infra-up     - Start PostgreSQL, Redis, and Qdrant with Docker Compose"
	@echo "  make infra-down   - Stop supporting Docker Compose containers"
	@echo "  make backend      - Run FastAPI backend development server on port 8000"
	@echo "  make frontend     - Run Next.js frontend development server on port 3000"
	@echo "  make test         - Run full pytest test suite"
	@echo "  make lint         - Run Ruff code checks"
	@echo "  make format       - Format Python code with Ruff"
	@echo "  make build        - Build Next.js production web bundle"

infra-up:
	docker compose up -d

infra-down:
	docker compose down

backend:
	cd backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

frontend:
	cd apps/web && npm run dev

test:
	python -m pytest

lint:
	python -m ruff check backend/app tests

format:
	python -m ruff format backend/app tests

build:
	cd apps/web && npm run build
