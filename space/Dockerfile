# Hugging Face Docker Space: the API and UI plus a Qdrant server binary in one container.
# The build context is the Space repo root populated by scripts/deploy_space.py.
FROM qdrant/qdrant:v1.19.2 AS qdrant

FROM python:3.11.9-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed \
    PATH=/opt/venv/bin:$PATH \
    MERIDIAN_SERVING_PROFILE=demo \
    MERIDIAN_WATCH_RELEASES=true \
    QDRANT_URL=http://127.0.0.1:6333 \
    MERIDIAN_ENV_FILE=/dev/null \
    FORWARDED_ALLOW_IPS=127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libunwind8 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=qdrant /qdrant/qdrant /usr/local/bin/qdrant
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

# Spaces run the container as uid 1000
RUN useradd --create-home --uid 1000 user && mkdir -p /opt/venv && chown user /opt/venv
USER user
WORKDIR /home/user/app

COPY --chown=user pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-install-project

COPY --chown=user src ./src
COPY --chown=user configs ./configs
COPY --chown=user eval ./eval
COPY --chown=user space ./space
RUN uv sync --frozen

# SPEC S5.2: the image build records whether local mode supports this query pattern
RUN python space/check_local_mode.py --output space/local_mode_check.json
# model weights are baked into the image so a cold start does not download them
RUN python space/preload_models.py

EXPOSE 7860
CMD ["bash", "space/start.sh"]
