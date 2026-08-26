"""FastAPI app. Run with: uv run uvicorn serve.app:app --app-dir src --reload

Every route is a thin wrapper: parse params, call into queries.py, 404 on
None, return the dict/list as-is (FastAPI JSON-encodes it, including the
datetime/date values DuckDB hands back).

The frontend/ static files are mounted at "/" (see the bottom of this
file, after every /api/* route) so one process serves both the UI and the
API -- exactly what the Docker image runs. "frontend" is a plain relative
path, resolved against the process's working directory: the repo root for
a bare `uvicorn ... --app-dir src` invocation (this docstring's command,
run from the repo root) and /app in the container (Dockerfile sets
WORKDIR /app and copies frontend/ there) -- same relative-path resolution
serve.config's DB paths already rely on, so no environment-specific
branching is needed here either. The mount is added LAST because
Starlette matches routes in registration order: every explicit /api/*
path above is checked first, and only a request matching none of them
falls through to this catch-all.
"""

from __future__ import annotations

from typing import Annotated

import duckdb
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from serve import ask as ask_module
from serve import queries
from serve import resolve as resolve_module
from serve.db import get_connection

app = FastAPI(title="cogwar-bakeoff analysis API")

# Frontend is a separate origin during dev and unauthenticated for now --
# open CORS is fine for a backend-only, no-auth-yet viz API. POST is needed
# for /api/ask (the only endpoint that isn't a pure query-string GET).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

Conn = Annotated[duckdb.DuckDBPyConnection, Depends(get_connection)]


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/stats")
def stats(con: Conn) -> dict:
    return queries.get_stats(con)


@app.get("/api/entities")
def entities(con: Conn, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)) -> list[dict]:
    return queries.list_entities(con, limit, offset)


@app.get("/api/authors")
def authors(con: Conn, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)) -> list[dict]:
    return queries.list_authors(con, limit, offset)


@app.get("/api/graph/coordination")
def graph_coordination(
    con: Conn,
    min_edges: int = Query(2, ge=1, description="Minimum shared/synchrony edges for an author pair to be included"),
    limit: int = Query(150, ge=1, le=2000, description="Cap on number of author-pair edges returned, strongest first"),
) -> dict:
    return queries.get_coordination_graph(con, min_edges, limit)


@app.get("/api/author/{author_id}")
def author_profile(
    con: Conn,
    author_id: str,
    limit: int = Query(50, ge=1, le=5000, description="Max entities in the stance vector, ranked by |net_stance|*volume"),
) -> dict:
    result = queries.get_author_profile(con, author_id, limit)
    if result is None:
        raise HTTPException(404, f"author not found: {author_id}")
    return result


@app.get("/api/entity/{entity_id}/authors")
def entity_authors(
    con: Conn,
    entity_id: str,
    limit: int = Query(10, ge=1, le=500),
    min_volume: int = Query(
        5, ge=1, description="Exclude authors with fewer than this many stance-bearing items toward the entity -- filters out volume=1 scoring flukes"
    ),
) -> dict:
    result = queries.get_entity_authors(con, entity_id, limit, min_volume)
    if result is None:
        raise HTTPException(404, f"entity not found: {entity_id}")
    return result


@app.get("/api/entity/{entity_id}/timeline")
def entity_timeline(con: Conn, entity_id: str, bucket: str = Query("week", pattern="^(day|week|month)$")) -> dict:
    result = queries.get_entity_timeline(con, entity_id, bucket)
    if result is None:
        raise HTTPException(404, f"entity not found: {entity_id}")
    return result


@app.get("/api/narrative/{narrative_id}")
def narrative(con: Conn, narrative_id: str) -> dict:
    result = queries.get_narrative(con, narrative_id)
    if result is None:
        raise HTTPException(404, f"narrative not found: {narrative_id}")
    return result


@app.get("/api/author/{author_id}/entity/{entity_id}/sources")
def author_entity_sources(
    con: Conn,
    author_id: str,
    entity_id: str,
    limit: int = Query(20, ge=1, le=500, description="Max source items to return, most recent first"),
) -> dict:
    result = queries.get_author_entity_sources(con, author_id, entity_id, limit)
    if result is None:
        raise HTTPException(404, f"no data for author={author_id!r} entity={entity_id!r}")
    return result


@app.get("/api/cluster/{cluster_id}/items")
def cluster_items(con: Conn, cluster_id: str) -> dict:
    """`cluster_id` is any item_id known to be part of the cluster -- see
    queries.get_cluster_items for why there's no separate stable cluster id
    yet."""
    result = queries.get_cluster_items(con, cluster_id)
    if result is None:
        raise HTTPException(404, f"item not found: {cluster_id}")
    return result


@app.get("/api/item/{item_id}")
def item_detail(con: Conn, item_id: str) -> dict:
    result = queries.get_item_detail(con, item_id)
    if result is None:
        raise HTTPException(404, f"item not found: {item_id}")
    return result


@app.get("/api/resolve")
def resolve_name(
    con: Conn,
    kind: str = Query(..., pattern="^(entity|author)$", description="'entity' or 'author'"),
    name: str = Query(..., min_length=1),
) -> dict:
    result = resolve_module.resolve_entity(con, name) if kind == "entity" else resolve_module.resolve_author(con, name)
    if result is None:
        raise HTTPException(404, f"no {kind} match for {name!r}")
    return result


class AskRequest(BaseModel):
    question: str


@app.post("/api/ask")
def ask(con: Conn, body: AskRequest) -> dict:
    result = ask_module.answer(con, body.question)
    return {
        "question": result.question,
        "intent": result.intent,
        "result": result.result,
        "summary": result.summary,
        "error": result.error,
    }


# Registered last -- see the module docstring for why route order matters here.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
