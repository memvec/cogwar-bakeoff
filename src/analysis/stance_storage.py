"""Stance storage: the entity-stance edge table + its content-level cache
(docs/analysis_layer_spec.md §2.1, §4 pass 2).

Mirrors resolution.py's two-layer cache discipline, but at (text_hash,
entity_id) granularity rather than whole-item granularity -- see
detect_stance.py's module docstring for why: stance calls batch all of one
item's uncached entities into a single API call, and a rerun must skip only
the (item, entity) pairs that are still missing, not the whole item.

  entity_stance_edges -- the atomic per-(item, entity) signal profiles will
  later aggregate from (§2.1). One row per (item_id, entity_id); PK enforces
  that re-running never double-writes an edge that already exists.

  stance_cache -- keyed on (text_hash, entity_id): if this exact text has
  already had its stance toward this entity scored (even under a different
  item_id -- e.g. a repost), skip the API call entirely and reuse the result.
"""

from __future__ import annotations

from datetime import UTC, datetime

import duckdb

from analysis.stance import EntityRef, StanceResult

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entity_stance_edges (
    item_id VARCHAR,
    entity_id VARCHAR,
    polarity VARCHAR,
    strength DOUBLE,
    confidence DOUBLE,
    detector_model VARCHAR,
    detected_at TIMESTAMP,
    PRIMARY KEY (item_id, entity_id)
);

CREATE TABLE IF NOT EXISTS stance_cache (
    text_hash VARCHAR,
    entity_id VARCHAR,
    polarity VARCHAR,
    strength DOUBLE,
    confidence DOUBLE,
    detector_model VARCHAR,
    detected_at TIMESTAMP,
    PRIMARY KEY (text_hash, entity_id)
);
"""


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA_SQL)


def get_item_entities(con: duckdb.DuckDBPyConnection, item_id: str) -> list[EntityRef]:
    """The distinct entities item_entities.py's extraction pass resolved for
    this item, one EntityRef per entity_id (the highest-confidence surface
    form is used when an entity was mentioned more than once)."""
    rows = con.execute(
        """
        SELECT ie.entity_id, e.canonical_name, ie.surface_form
        FROM item_entities ie
        JOIN entities e ON e.entity_id = ie.entity_id
        WHERE ie.item_id = ?
        QUALIFY row_number() OVER (PARTITION BY ie.entity_id ORDER BY ie.confidence DESC) = 1
        """,
        [item_id],
    ).fetchall()
    return [EntityRef(entity_id=r[0], canonical_name=r[1], surface_form=r[2]) for r in rows]


def get_existing_edge_entity_ids(con: duckdb.DuckDBPyConnection, item_id: str) -> set[str]:
    rows = con.execute(
        "SELECT entity_id FROM entity_stance_edges WHERE item_id = ?", [item_id]
    ).fetchall()
    return {r[0] for r in rows}


def get_cached_stance(
    con: duckdb.DuckDBPyConnection, text_hash: str, entity_id: str
) -> StanceResult | None:
    row = con.execute(
        "SELECT polarity, strength, confidence FROM stance_cache WHERE text_hash = ? AND entity_id = ?",
        [text_hash, entity_id],
    ).fetchone()
    if row is None:
        return None
    return StanceResult(entity_id=entity_id, polarity=row[0], strength=row[1], confidence=row[2])


def store_stance_cache(
    con: duckdb.DuckDBPyConnection, text_hash: str, result: StanceResult, model: str
) -> None:
    con.execute(
        "INSERT INTO stance_cache VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
        [text_hash, result.entity_id, result.polarity, result.strength, result.confidence, model, datetime.now(UTC)],
    )


def record_stance_edge(
    con: duckdb.DuckDBPyConnection, item_id: str, result: StanceResult, model: str
) -> None:
    con.execute(
        "INSERT INTO entity_stance_edges VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
        [item_id, result.entity_id, result.polarity, result.strength, result.confidence, model, datetime.now(UTC)],
    )
