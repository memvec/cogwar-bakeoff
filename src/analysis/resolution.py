"""Entity resolution + persisted alias cache (docs/analysis_layer_spec.md §1.2, §4 pass 1).

Maintains the canonical-entity store (`entities`) and alias->entity map
(`entity_aliases`) in the analysis DuckDB, plus the per-item results table
(`item_entities`) and a content-level extraction cache (`extraction_cache`,
keyed on text_hash) that lets duplicate/re-collected content skip the API
entirely -- see extract_entities.py for how these two caches combine
(item-level: skip if this item_id already has results; content-level: skip
the API call if this text_hash has been extracted before, even under a
different item_id).

Resolution logic, per mention:
1. surface_form already in entity_aliases (normalized) -> reuse its entity_id. Free.
2. Else, canonical_name matches an existing entity (normalized) -> reuse that
   entity_id, and record this surface_form as a new alias of it.
3. Else -> create a new entity (created_from='discovered'), record the alias.
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from analysis.entities import ENTITY_TYPES, EntityMention

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entities (
    entity_id VARCHAR PRIMARY KEY,
    canonical_name VARCHAR,
    canonical_name_normalized VARCHAR,
    entity_type VARCHAR,
    created_from VARCHAR,
    first_seen_at TIMESTAMP,
    observation_count INTEGER
);

CREATE TABLE IF NOT EXISTS entity_aliases (
    surface_form_normalized VARCHAR PRIMARY KEY,
    surface_form VARCHAR,
    entity_id VARCHAR,
    first_seen_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS item_entities (
    item_id VARCHAR,
    entity_id VARCHAR,
    surface_form VARCHAR,
    confidence DOUBLE,
    extracted_at TIMESTAMP,
    PRIMARY KEY (item_id, entity_id, surface_form)
);

CREATE TABLE IF NOT EXISTS extraction_cache (
    text_hash VARCHAR PRIMARY KEY,
    mentions_json VARCHAR,
    extracted_at TIMESTAMP,
    model VARCHAR
);

-- Tracks that an item has been through extraction+resolution at all, even
-- when zero entities were found (item_entities alone can't represent that
-- case -- an empty result leaves no row there to check against). This is
-- what the incremental skip actually queries.
CREATE TABLE IF NOT EXISTS item_extraction_status (
    item_id VARCHAR PRIMARY KEY,
    processed_at TIMESTAMP,
    entity_count INTEGER
);
"""


def normalize(name: str) -> str:
    """Lowercase + collapse whitespace for alias/canonical-name matching.

    Not full normalization (no diacritic stripping, no transliteration) --
    the LLM's canonical_name output is already fairly consistent across
    calls for the same real-world entity, so this simple pass is enough to
    catch case/spacing variance without over-engineering a fuzzy matcher.
    """
    return " ".join(name.strip().lower().split())


class EntityResolver:
    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self.con = con
        self.con.execute(SCHEMA_SQL)

    def resolve(self, mention: EntityMention) -> str:
        """Return the entity_id for one mention -- reusing an existing entity whenever possible, creating a new one only as a last resort."""
        normalized_surface = normalize(mention.surface_form)
        now = datetime.now(UTC)

        existing_alias = self.con.execute(
            "SELECT entity_id FROM entity_aliases WHERE surface_form_normalized = ?",
            [normalized_surface],
        ).fetchone()
        if existing_alias:
            entity_id = existing_alias[0]
            self._bump_observation_count(entity_id)
            return entity_id

        normalized_canonical = normalize(mention.canonical_name)
        existing_entity = self.con.execute(
            "SELECT entity_id FROM entities WHERE canonical_name_normalized = ?",
            [normalized_canonical],
        ).fetchone()
        if existing_entity:
            entity_id = existing_entity[0]
            self._bump_observation_count(entity_id)
        else:
            entity_id = str(uuid.uuid4())
            entity_type = mention.entity_type_guess if mention.entity_type_guess in ENTITY_TYPES else "other"
            self.con.execute(
                "INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?, ?)",
                [entity_id, mention.canonical_name, normalized_canonical, entity_type, "discovered", now, 1],
            )

        # New surface form of an entity we now know (whether just-created or
        # matched by canonical name) -- record it so next time this exact
        # surface form is free (rule 1 above).
        self.con.execute(
            "INSERT INTO entity_aliases VALUES (?, ?, ?, ?)",
            [normalized_surface, mention.surface_form, entity_id, now],
        )
        return entity_id

    def _bump_observation_count(self, entity_id: str) -> None:
        self.con.execute(
            "UPDATE entities SET observation_count = observation_count + 1 WHERE entity_id = ?",
            [entity_id],
        )


def load_seed_entities(resolver: EntityResolver, path: Path) -> int:
    """Load configs/seed_entities.json into entities/entity_aliases, created_from='seed'.

    Idempotent -- safe to call at the start of every run. Entities/aliases
    already present (matched by normalized canonical name / surface form)
    are left untouched, never overwritten. Returns the number of NEW seed
    entities created this call (0 on a repeat run).
    """
    if not path.exists():
        return 0
    with path.open() as f:
        data = json.load(f)

    created = 0
    now = datetime.now(UTC)
    con = resolver.con
    for entry in data.get("entities", []):
        canonical_name = entry["canonical_name"]
        entity_type = entry.get("entity_type", "other")
        normalized_canonical = normalize(canonical_name)

        existing = con.execute(
            "SELECT entity_id FROM entities WHERE canonical_name_normalized = ?",
            [normalized_canonical],
        ).fetchone()
        if existing:
            entity_id = existing[0]
        else:
            entity_id = str(uuid.uuid4())
            con.execute(
                "INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?, ?)",
                [entity_id, canonical_name, normalized_canonical, entity_type, "seed", now, 0],
            )
            created += 1

        for alias in {canonical_name, *entry.get("aliases", [])}:
            normalized_alias = normalize(alias)
            already = con.execute(
                "SELECT 1 FROM entity_aliases WHERE surface_form_normalized = ?",
                [normalized_alias],
            ).fetchone()
            if not already:
                con.execute(
                    "INSERT INTO entity_aliases VALUES (?, ?, ?, ?)",
                    [normalized_alias, alias, entity_id, now],
                )
    return created


def get_cached_extraction(con: duckdb.DuckDBPyConnection, text_hash: str) -> list[EntityMention] | None:
    """None means this exact content has never been sent to the extractor; an empty list is a valid cached "no entities found" result."""
    row = con.execute(
        "SELECT mentions_json FROM extraction_cache WHERE text_hash = ?", [text_hash]
    ).fetchone()
    if row is None:
        return None
    return [EntityMention(**m) for m in json.loads(row[0])]


def store_extraction_cache(
    con: duckdb.DuckDBPyConnection, text_hash: str, mentions: list[EntityMention], model: str
) -> None:
    payload = json.dumps([dataclasses.asdict(m) for m in mentions])
    con.execute(
        "INSERT INTO extraction_cache VALUES (?, ?, ?, ?)",
        [text_hash, payload, datetime.now(UTC), model],
    )


def item_already_processed(con: duckdb.DuckDBPyConnection, item_id: str) -> bool:
    """The incremental skip check -- True once this item has been through extraction+resolution, regardless of whether any entities were found."""
    row = con.execute(
        "SELECT 1 FROM item_extraction_status WHERE item_id = ? LIMIT 1", [item_id]
    ).fetchone()
    return row is not None


def mark_item_processed(con: duckdb.DuckDBPyConnection, item_id: str, entity_count: int) -> None:
    con.execute(
        "INSERT INTO item_extraction_status VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
        [item_id, datetime.now(UTC), entity_count],
    )


def record_item_entity(
    con: duckdb.DuckDBPyConnection, item_id: str, entity_id: str, surface_form: str, confidence: float
) -> None:
    con.execute(
        "INSERT INTO item_entities VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
        [item_id, entity_id, surface_form, confidence, datetime.now(UTC)],
    )
