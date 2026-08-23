"""FastAPI app. Run with: uv run uvicorn serve.app:app --reload

Every route is a thin wrapper: parse params, call into queries.py, 404 on
None, return the dict/list as-is (FastAPI JSON-encodes it, including the
datetime/date values DuckDB hands back).
"""

from __future__ import annotations

from typing import Annotated

import duckdb
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from serve import queries
from serve.db import get_connection

app = FastAPI(title="cogwar-bakeoff analysis API")

# Frontend is a separate origin during dev and unauthenticated for now --
# open CORS is fine for a backend-only, no-auth-yet viz API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
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
def entity_authors(con: Conn, entity_id: str, limit: int = Query(10, ge=1, le=500)) -> dict:
    result = queries.get_entity_authors(con, entity_id, limit)
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
