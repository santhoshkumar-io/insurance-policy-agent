.PHONY: up down migrate test test-all lint fmt api

up:          ## Start PostgreSQL + pgvector
	docker compose up -d

down:
	docker compose down

migrate:
	.venv/bin/alembic upgrade head

test:        ## Unit tests: no database, no network
	.venv/bin/python -m pytest -q -m "not db"

test-all:    ## Everything, including tenant isolation (needs `make up migrate`)
	.venv/bin/python -m pytest -q

lint:
	.venv/bin/ruff check .

fmt:
	.venv/bin/ruff format .

api:
	.venv/bin/uvicorn services.api.main:app --reload
