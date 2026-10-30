.DEFAULT_GOAL := help
COMPOSE := docker compose

.PHONY: help up down migrate ingest index testset-draft eval test dev

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

up: ## Start all docker-compose services
	$(COMPOSE) up -d --build

down: ## Stop all services
	$(COMPOSE) down

migrate: ## Apply database migrations
	$(COMPOSE) exec backend python -m app.db.migrate

ingest: ## [M2] Resolve seeds, fetch, expand, clean (snapshot)
	$(COMPOSE) exec backend python -m app.ingest.run $(if $(FORCE),--force,)

index: ## [M3] Chunk, embed, write pgvector + build BM25 index
	$(COMPOSE) exec backend python -m app.index.run

testset-draft: ## [M7] Generate draft eval test set
	$(COMPOSE) exec backend python -m app.eval.testset_draft

eval: ## [M8] Run evaluation (CONFIGS=all | hybrid,agentic)
	$(COMPOSE) exec backend python -m app.eval.run --configs $(CONFIGS)

test: ## Backend tests + frontend typecheck
	$(COMPOSE) exec backend pytest -q
	cd frontend && npm run typecheck

dev: ## Run backend + frontend in dev mode (without docker)
	@echo "Backend:  cd backend && uvicorn app.main:app --reload"
	@echo "Frontend: cd frontend && npm run dev"
