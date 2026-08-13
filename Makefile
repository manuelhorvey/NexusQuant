# NexusQuant developer convenience targets.
.PHONY: help api db-up db-down up down logs test db-seed docs

help:
	@echo "NexusQuant targets:"
	@echo "  make api       run the FastAPI service locally (SQLite persistence)"
	@echo "  make db-up     start Postgres (docker compose)"
	@echo "  make db-down   stop Postgres"
	@echo "  make up        full stack: Postgres + API (docker compose --build)"
	@echo "  make down      stop the full stack"
	@echo "  make logs      follow API logs"
	@echo "  make test      run the API tests"
	@echo "  make db-seed   insert demo snapshot + signal rows (idempotent)"
	@echo "  make docs      open the docs/ index"

api:
	./venv/bin/uvicorn api.app:app --reload --port 8000

db-up:
	docker compose up -d db

db-down:
	docker compose stop db

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f api

test:
	./venv/bin/python -m unittest tests.test_api -v

db-seed:
	./venv/bin/python database/seed_data/seed.py

docs:
	@echo "NexusQuant docs:"
	@echo "  docs/architecture.md    system map, data flow, deployment"
	@echo "  docs/api_reference.md   REST endpoints + examples"
	@echo "  docs/trading_models.md  methodology of every engine"
