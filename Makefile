.DEFAULT_GOAL := help
COMPOSE := docker compose

.PHONY: help up up-dev down model migrate ingest index testset-draft eval bench test dev

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

up: ## Start all docker-compose services
	$(COMPOSE) up -d --build

up-dev: ## Start with hot reload (vite dev server + uvicorn --reload)
	$(COMPOSE) -f docker-compose.yml -f docker-compose.dev.yml up -d --build

model: ## Pull the LLM into Ollama (MODEL=... overrides LLM_DEFAULT_MODEL)
	$(COMPOSE) exec ollama ollama pull $(or $(MODEL),$(shell grep -E '^LLM_DEFAULT_MODEL=' .env 2>/dev/null | cut -d= -f2),qwen3:30b)

down: ## Stop all services
	$(COMPOSE) down

migrate: ## Apply database migrations
	$(COMPOSE) exec backend python -m app.db.migrate

ingest: ## [M2] Resolve seeds, fetch, expand, clean (snapshot)
	$(COMPOSE) exec backend python -m app.ingest.run $(if $(FORCE),--force,)

index: ## [M3] Chunk, embed, write pgvector + build BM25 index
	$(COMPOSE) exec backend python -m app.index.run

testset-draft: ## [M7] Generate draft eval test set (needs an LLM)
	$(COMPOSE) exec backend python -m app.eval.testset_draft

testset-seed: ## [M7] Build the hand-authored seed test set into testset.jsonl
	$(COMPOSE) exec backend python -m app.eval.build_seed

testset-load: ## [M7] Load reviewed testset.jsonl into eval_questions
	$(COMPOSE) exec backend python -m app.eval.load_testset

eval: ## [M8] Run evaluation (CONFIGS=all | hybrid,agentic)
	$(COMPOSE) exec backend python -m app.eval.run --configs $(or $(CONFIGS),all)

bench: ## Latency benchmark (ARGS=--llm to time the streamed answer)
	$(COMPOSE) exec backend python scripts/bench.py $(ARGS)

test: ## Backend tests + frontend typecheck
	$(COMPOSE) exec backend pytest -q
	cd frontend && npm run typecheck

dev: ## Run backend + frontend in dev mode (without docker)
	@echo "Backend:  cd backend && uvicorn app.main:app --reload"
	@echo "Frontend: cd frontend && npm run dev"
