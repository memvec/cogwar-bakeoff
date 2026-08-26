# Serves the FastAPI backend (src/serve/) and the static frontend/ from one
# container. DuckDB data is NOT baked in here -- docker-compose mounts the
# host's ./data at /app/data at runtime (read-only), matching serve/config.py's
# relative "data/..." paths resolved against WORKDIR /app.
#
# No arch-pinned base images or --platform flags anywhere in this file:
# python:3.11-slim and ghcr.io/astral-sh/uv are both published as multi-arch
# manifests, so `docker build`/`docker compose build` picks the right image
# for whichever host builds it (arm64 on a Mac, amd64 on a GCP instance)
# without any changes here.
FROM python:3.11-slim

# Astral's official Docker integration pattern: copy the prebuilt static uv
# binary out of its own multi-arch image rather than installing via pip or a
# curl|sh script -- fast, no extra build deps, no network call inside this
# build. https://docs.astral.sh/uv/guides/integration/docker/
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Dependencies first, in their own layer keyed only on the lockfile -- a
# source-only change (src/ or frontend/) then rebuilds in seconds instead of
# re-resolving/reinstalling every dependency. `--no-install-project` skips
# building this repo's own local package in this pass (which needs src/ --
# doesn't exist yet at this point in the build -- and README.md, copied
# below alongside the real source).
COPY pyproject.toml uv.lock ./
RUN uv sync --group serve --no-install-project --locked

# Now the actual application code, and the real sync that installs it.
# README.md is required at this step too: pyproject.toml's `readme` field
# points hatchling at it, and the local package build fails without it.
COPY README.md .
COPY src/ src/
COPY frontend/ frontend/
RUN uv sync --group serve --locked

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Production mode: no --reload. --app-dir src matches every other uvicorn
# invocation documented in this repo (serve/app.py's own docstring), so the
# module path (serve.app:app) is identical whether run bare or in Docker.
CMD ["uvicorn", "serve.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
