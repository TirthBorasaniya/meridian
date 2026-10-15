# Meridian developer targets. Caches and temporary files stay inside the repository.

export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.cache/uv-python
export HF_HOME := $(CURDIR)/.cache/huggingface
export FASTEMBED_CACHE_PATH := $(CURDIR)/.cache/fastembed
export TMPDIR := $(CURDIR)/.tmp
export PYTHONHASHSEED := 0

PYTHON_VERSION := 3.11.9
RUN := uv run --frozen
COMPOSE := docker compose

.PHONY: setup lint format typecheck test test-integration test-e2e record-e2e up down logs demo \
	backfill index ingest rebuild rollback scheduler retrieval-set review-set prepare-review references \
	eval eval-retrieval eval-answers eval-serving eval-ingest eval-embedder eval-demo \
	embedder-pairs embedder-negatives embedder-train embedder-push train-embedder \
	release restore deploy-space benchmark loadtest results results-check clean

# ============= Environment =============

setup:
	mkdir -p .tmp .cache
	uv python install $(PYTHON_VERSION)
	uv sync --frozen --extra dev --python $(PYTHON_VERSION)
	$(RUN) pre-commit install || true

lint:
	$(RUN) ruff check src tests scripts space
	$(RUN) ruff format --check src tests scripts space

format:
	$(RUN) ruff format src tests scripts space
	$(RUN) ruff check --fix src tests scripts space

typecheck:
	$(RUN) mypy src

test:
	$(RUN) pytest

# needs the compose Qdrant: make up, then make test-integration
test-integration:
	QDRANT_URL=$${QDRANT_URL:-http://127.0.0.1:6333} $(RUN) pytest -m integration -o addopts=""

test-e2e:
	$(RUN) pytest -m e2e -o addopts=""

# records tests/fixtures/e2e_llm_cache.sqlite with real Groq calls
record-e2e:
	MERIDIAN_RECORD_E2E=1 $(RUN) pytest -m e2e -o addopts=""

# ============= Stack =============

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=200

# restores the latest accepted index snapshot into the running Qdrant and prints the UI URL
demo:
	$(RUN) meridian release restore
	@echo "Meridian UI: http://127.0.0.1:8000/"

# ============= Corpus and index =============

backfill:
	$(RUN) meridian backfill --source seed-ids --legacy-pdf-dir data/corpus/raw

index:
	$(RUN) meridian index build --switch-alias

ingest:
	$(RUN) meridian ingest --since last_success

rebuild:
	$(RUN) meridian rebuild

rollback:
	$(RUN) meridian rollback

scheduler:
	$(RUN) meridian scheduler

# ============= Evaluation datasets =============

retrieval-set:
	$(RUN) meridian build-retrieval-set

review-set:
	$(RUN) python scripts/review_retrieval_set.py

prepare-review:
	$(RUN) python scripts/prepare_review.py

references:
	$(RUN) python scripts/build_answer_references.py

# ============= Experiments =============

eval: eval-retrieval eval-answers eval-serving eval-ingest

eval-retrieval:
	$(RUN) meridian eval retrieval --config configs/eval/e1.yaml

eval-answers:
	$(RUN) meridian eval answers --config configs/eval/e2.yaml

eval-serving:
	$(RUN) meridian eval serving --config configs/eval/e3.yaml

eval-ingest:
	$(RUN) meridian eval ingest --config configs/eval/e4.yaml

eval-embedder:
	$(RUN) meridian eval embedder --config configs/eval/e5.yaml

eval-demo:
	$(RUN) meridian eval demo --config configs/eval/e6.yaml

benchmark: eval-retrieval

loadtest:
	$(RUN) python scripts/loadtest.py --config configs/eval/loadtest.yaml

# ============= Embedding adaptation =============

embedder-pairs:
	$(RUN) meridian embedder pairs --config configs/embedder.yaml

embedder-negatives:
	$(RUN) meridian embedder negatives --config configs/embedder.yaml

embedder-train:
	$(RUN) meridian embedder train --config configs/embedder.yaml

embedder-push:
	$(RUN) meridian embedder push --config configs/embedder.yaml

train-embedder: embedder-pairs embedder-negatives embedder-train

# ============= Releases =============

release:
	$(RUN) meridian release publish

deploy-space:
	$(RUN) python scripts/deploy_space.py

restore:
	$(RUN) meridian release restore

# ============= Results =============

results:
	$(RUN) python scripts/render_results.py

results-check:
	$(RUN) python scripts/render_results.py --check

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .tmp/release
	find src tests scripts -name __pycache__ -type d -prune -exec rm -rf {} +
