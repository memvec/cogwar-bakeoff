"""serve package settings -- DB paths + the local Ollama endpoint for
POST /api/ask.

Deliberately does NOT import analysis.config: that module fails loudly at
import time if ANTHROPIC_API_KEY is unset (by design, for the LLM-backed
passes), and this is a read-only viz backend whose only model calls are to
a local, free, no-API-key Ollama instance -- forcing Anthropic key setup
just to read two constant paths would violate the same "don't force
unrelated secrets" reasoning analysis.config itself uses for staying
independent of collection.config. The DB paths are duplicated (not
imported) for that reason -- a small, deliberate redundancy.

Both DB paths are relative, resolved against the process's working
directory -- `data/...` when run from the repo root (bare `uvicorn`, per
this package's docstrings) or under Docker, where the container's WORKDIR
is /app and docker-compose mounts the host's ./data at /app/data, so the
same relative path resolves correctly in both places without needing a
container-specific override.

OLLAMA_HOST/OLLAMA_MODEL are env-var-driven (not relative paths, since
they're read by a different process, not the filesystem) so the same code
reaches a local `ollama serve` at its default address during bare dev
(OLLAMA_HOST unset) and the `ollama` compose service by its DNS name in
Docker (OLLAMA_HOST=http://ollama:11434, set in docker-compose.yml).
"""

import os
from pathlib import Path

PROCESSED_DB_PATH = Path("data/processed/cogwar.duckdb")
ANALYSIS_DB_PATH = Path("data/analysis/analysis.duckdb")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
