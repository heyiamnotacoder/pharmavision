# The demo site: the API with the built frontend at /, in one process (background jobs run in
# it). Precomputed runs and studies come from DEMO_URL, a tarball made by scripts/pack-demo.sh;
# without it the site serves live runs only. MODEL_API_KEY is set on the host, never baked in.

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
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
COPY --from=frontend /app/frontend/dist ./frontend/dist

ARG DEMO_URL=""
RUN mkdir -p demo/runs demo/studies \
    && if [ -n "$DEMO_URL" ]; then curl -fsSL "$DEMO_URL" | tar -xz -C demo; fi

ENV PORT=8000
CMD uv run --no-sync python -m pharmavision serve --host 0.0.0.0 --port "$PORT" \
    --precomputed demo/runs --studies-dir demo/studies --frontend frontend/dist
