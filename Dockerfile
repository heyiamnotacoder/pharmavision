# The public demo site: the API with the built frontend at /, in one process (background jobs
# run in it), read-only so no visitor can spend the Model API budget. Precomputed runs and the
# demo study come from DEMO_URL, a tarball made by scripts/pack-demo.sh, checked against
# DEMO_SHA256; without DEMO_URL the site has nothing to show.

FROM node:22-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv
# Ingest normalises every video with ffmpeg.
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home app
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
COPY --from=frontend /app/frontend/dist ./frontend/dist

ARG DEMO_URL=""
ARG DEMO_SHA256=""
RUN mkdir -p demo/runs demo/studies runs \
    && if [ -n "$DEMO_URL" ]; then \
        curl -fsSL -o /tmp/demo.tar.gz "$DEMO_URL" \
        && echo "$DEMO_SHA256  /tmp/demo.tar.gz" | sha256sum -c - \
        && tar -xzf /tmp/demo.tar.gz -C demo && rm /tmp/demo.tar.gz; \
    fi \
    && chown -R app runs
USER app

ENV PORT=8000
CMD exec uv run --no-sync python -m pharmavision serve --host 0.0.0.0 --port "$PORT" \
    --read-only --precomputed demo/runs --studies-dir demo/studies --frontend frontend/dist
