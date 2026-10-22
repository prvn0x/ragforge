SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help
COMPOSE := docker compose

.PHONY: help sync doctor env setup models up down nuke ps logs psql tenant connect-notion ask ui mcp-key mcp mcp-install eval-private build test test-unit test-int lint fmt typecheck imports check eval-tenant eval-demo eval eval-gate backup-evals obs-up obs-down audit secrets-scan

help: ## Show targets
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

sync: ## Install every workspace package and all extras (agent, ingest) for development
	@uv sync --all-packages --all-extras

doctor: ## Preflight checks for this machine
	@./scripts/doctor.sh

env: ## Create .env with fresh secrets (never overwrites existing values)
	@./scripts/bootstrap.sh

setup: env ## Choose the chat model: hosted Gemini (API key, hidden input) or local Ollama
	@python3 scripts/setup_provider.py

models: env ## Pull chat + embedding models into the selected Ollama runtime
	@./scripts/models.sh

OLLAMA_PROFILE = $(shell grep -q '^RAGFORGE_OLLAMA_MODE=container' .env 2>/dev/null && echo "--profile ollama")

up: env ## Build and start the stack; waits until the API is ready
	@# First run in a terminal only: asks for a provider. Non-interactive/CI stays local (keyless).
	@# A first-run setup that cannot finish (bad key, offline) starts local; Ctrl-C aborts.
	@if command -v python3 >/dev/null; then python3 scripts/setup_provider.py --if-unset; fi
	@$(COMPOSE) $(OLLAMA_PROFILE) up --build --wait --wait-timeout 300 api worker $(if $(OLLAMA_PROFILE),ollama,)
	@$(COMPOSE) up -d --no-deps index  # best-effort HNSW build; nothing waits on it
	@echo; echo "API      http://localhost:8000/docs"; echo "Ready    http://localhost:8000/readyz"

down: ## Stop the stack (keeps data)
	@$(COMPOSE) --profile ollama --profile obs down

nuke: ## Stop the stack AND delete all data volumes
	@$(COMPOSE) --profile ollama --profile obs down --volumes --remove-orphans

ps: env ## Show service status
	@$(COMPOSE) ps -a

logs: env ## Tail logs (SERVICE=api to filter)
	@$(COMPOSE) logs -f --tail=100 $(SERVICE)

psql: env ## psql into the database as the superuser
	@$(COMPOSE) exec postgres psql -U postgres -d ragforge

tenant: NAME ?= personal
tenant: OWNER ?= $(USER)
tenant: ## Create your tenant (stack up); owner key -> ~/.config/ragforge/owner.key, 0600, never shown (NAME=, OWNER=)
	@uv run python scripts/create_tenant.py --name "$(NAME)" --owner "$(OWNER)" --compose "$(COMPOSE)"

connect-notion: ## Create a Notion connector (key from ~/.config/ragforge/*.key; makes a first collection if none; Notion token hidden)
	@uv run python scripts/connect_notion.py $(ARGS)

ask: ## Chat with your notes in the terminal (key from ~/.config/ragforge/*.key; ARGS=--help)
	@uv run ragforge-ask $(ARGS)

ui: ## Streamlit UI on http://localhost:8501 (key from ~/.config/ragforge/*.key; ARGS="--key-file F")
	@uv run streamlit run packages/ui/src/ragforge_ui/app.py --server.headless true \
		--server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false \
		--server.allowedHosts localhost --server.allowedHosts 127.0.0.1 \
		--client.showErrorDetails type -- $(ARGS)

MCP_KEY := $(HOME)/.config/ragforge/mcp/mcp.key
mcp-key: ## Mint the MCP key (read, chat, write) with your owner key -> ~/.config/ragforge/mcp/mcp.key (0600)
	@uv run python scripts/create_mcp_key.py $(ARGS)

mcp: ## MCP server over stdio (default --key-file ~/.config/ragforge/mcp/mcp.key; ARGS replaces it)
	@uv run ragforge-mcp $(if $(strip $(ARGS)),$(ARGS),--key-file "$(MCP_KEY)")

mcp-install: ## Print (does not run) the `claude mcp add` command for this checkout
	@echo "1) Create the MCP key once if you have none: make mcp-key (see docs/GUIDE.md)"
	@echo "2) Then run:"
	@echo "claude mcp add --scope user ragforge -- uv --directory '$(CURDIR)' run ragforge-mcp --key-file '$(MCP_KEY)'"

eval-private: ## Phase 2b (v1) answer-quality eval on the private set (ARGS="run --label <arm>" or "score <run.jsonl>")
	@# v1 reads its own copy (the v2 evals own evals/private/questions.toml since ADR-0017)
	@uv run ragforge-eval --questions evals/private/questions.v1.toml $(ARGS)

obs-up: env ## Prometheus + Grafana (opt-in, ~768 MB cap, ~430 MB typical); Grafana on http://localhost:3000 (user admin)
	@$(COMPOSE) --profile obs up -d --wait --wait-timeout 120 prometheus grafana
	@echo; echo "Grafana  http://localhost:3000  (user admin; password = RAGFORGE_GRAFANA_ADMIN_PASSWORD in .env)"

obs-down: ## Stop Prometheus + Grafana (keeps their data)
	@$(COMPOSE) --profile obs stop prometheus grafana && $(COMPOSE) --profile obs rm -f prometheus grafana

audit: ## Dependency + image + config scan: pip-audit, Trivy (needs network; build first: make build)
	@./scripts/security/audit.sh

secrets-scan: ## Secret scan of the working tree (gitleaks + RAGForge token rules); never prints a value
	@./scripts/security/secrets.sh

prepublish: ## Before the first commit: after `git init && git add -A`, check what is staged, then the secret scan
	@uv run python scripts/security/prepublish.py $(ARGS)
	@./scripts/security/secrets.sh

backup-evals: ## Copy evals/private (labels, runs, held-out set: git-ignored, irreplaceable) to a dated 0600 folder
	@d="$(HOME)/.local/share/ragforge/eval-backups/$$(date +%Y%m%dT%H%M%S)"; \
	  mkdir -p -m 700 "$$d" && umask 077 && cp -Rp evals/private "$$d/" && chmod -R go-rwx "$$d" && \
	  echo "backed up evals/private -> $$d"

eval-tenant: ## Provision your paired eval tenant once (your owner key); writes the eval key 0600
	@uv run ragforge-eval2 tenant create $(ARGS)

DEMO_OWNER_KEY := $(HOME)/.config/ragforge/eval/demo/owner.key
eval-demo: ## D0 end to end: demo tenant, upload, snapshot, all five evals (live stack + model; not CI)
	@# First run: the operator CLI creates the demo tenant; its key goes straight to a 0600 file
	@# (never printed). Later runs reuse it.
	@if [ ! -s "$(DEMO_OWNER_KEY)" ]; then \
	  mkdir -p -m 700 "$$(dirname "$(DEMO_OWNER_KEY)")" && umask 077 && \
	  $(COMPOSE) exec -T api ragforge-admin create-tenant --name ragforge-eval-demo \
	    --owner "RAGForge eval demo" 2>/dev/null | sed -n 's/^api_key=//p' > "$(DEMO_OWNER_KEY)"; \
	  if [ ! -s "$(DEMO_OWNER_KEY)" ]; then rm -f "$(DEMO_OWNER_KEY)"; \
	    echo "demo tenant bootstrap failed (is the stack up? make up)"; exit 1; fi; \
	fi
	@uv run ragforge-eval2 demo --owner-key-file "$(DEMO_OWNER_KEY)" $(ARGS)

eval: ## v2 eval layer, ADR-0017 (ARGS="run retrieval --dataset private --arm A")
	@uv run ragforge-eval2 $(ARGS)

eval-gate: ## Regression gate (ARGS="--surface chat"); exit 1 when it does not pass
	@uv run ragforge-eval2 gate $(ARGS)

build: ## Build images
	@$(COMPOSE) build

test: ## All tests (integration tests need Docker)
	@uv run pytest packages

test-unit: ## Tests that don't need Docker
	@uv run pytest packages -m "not integration"

lint: ## Ruff lint + format check
	@uv run ruff check packages scripts/security && uv run ruff format --check packages scripts/security

fmt: ## Auto-format
	@uv run ruff format packages && uv run ruff check --fix packages

typecheck: ## mypy --strict (sources, shared tests, then each module's tests in isolation)
	@uv run mypy packages/contracts/src packages/core/src packages/client/src packages/mcp/src packages/ui/src scripts/security/report.py scripts/security/prepublish.py
	@uv run mypy packages/contracts/tests $$(ls packages/core/tests/*.py)
	@for d in packages/core/tests/module_*; do MYPYPATH=$$d uv run mypy $$d/test_*.py || exit 1; done

imports: ## Architecture contracts (import-linter)
	@uv run lint-imports

check: lint typecheck imports test ## Everything CI runs
