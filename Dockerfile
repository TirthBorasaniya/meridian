# Dockerfile for the Meridian API service.
# Not enumerated in the CLAUDE.md directory structure; required by the
# docker-compose `api` service build. See DECISIONS.md.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System libraries needed by psycopg and common ML wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching. requirements.txt ends
# with `-e .`, so the package metadata must be present at install time.
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the remainder of the project (scripts, configs).
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.meridian.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
