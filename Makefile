.PHONY: up down migrate seed test test-all lint fmt api stack

up:          ## Start PostgreSQL + pgvector only (the API runs from your venv)
	docker compose up -d db

down:
	docker compose down

migrate:
	.venv/bin/alembic upgrade head

seed:        ## Ingest the synthetic demo policy (idempotent)
	.venv/bin/python -m rag.cli seed-demo

stack:       ## API + database in containers, as deployed: migrate, seed, serve on :8000
	docker compose up --build

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
