"""Narrative storage: narratives + membership + the entity/stance rollup
(docs/analysis_layer_spec.md §1.3, §2.1, §4 pass 3).

Three tables, all `basis`-aware so tight and loose narratives coexist
side by side once LooseNarrativeClusterer (analysis/narratives.py) lands --
an item can belong to a tight AND a loose narrative at once, so
narrative_members has no uniqueness constraint on item_id alone.

  narratives            -- one row per narrative cluster (either basis).
  narrative_members     -- (narrative_id, item_id) membership.
  narrative_entity_stance -- per (narrative_id, entity_id): how many member
      items carried each stance polarity toward that entity, and the mean
      strength -- the "what is this narrative about, pro vs anti" rollup
      that directly serves output query #2 (§3.2). Computed from
      item_entities + entity_stance_edges, both owned by earlier passes.

Tight clustering is deterministic and recomputed from scratch on every run
(mirrors processing/derive.py's clear_derived_edges() discipline) -- rebuild
never duplicates: clear_basis() deletes all of one basis's prior rows
before build_narratives.py inserts the freshly computed set. This never
touches the other basis's rows.
"""

from __future__ import annotations

from datetime import UTC, datetime

import duckdb

from analysis.narratives import NarrativeCluster

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS narratives (
    narrative_id VARCHAR PRIMARY KEY,
    label VARCHAR,
    basis VARCHAR,
    time_range_start TIMESTAMP,
    time_range_end TIMESTAMP,
    size INTEGER,
    distinct_authors INTEGER,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS narrative_members (
    narrative_id VARCHAR,
    item_id VARCHAR,
    PRIMARY KEY (narrative_id, item_id)
);

CREATE TABLE IF NOT EXISTS narrative_entity_stance (
    narrative_id VARCHAR,
    entity_id VARCHAR,
    positive_count INTEGER,
    negative_count INTEGER,
    neutral_count INTEGER,
    mean_strength DOUBLE,
    PRIMARY KEY (narrative_id, entity_id)
);
"""


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA_SQL)


def clear_basis(con: duckdb.DuckDBPyConnection, basis: str) -> None:
    """Delete all prior rows for one basis (tight or loose) across all three
    tables, leaving the other basis untouched. Called before a fresh
    deterministic rebuild."""
    narrative_ids = [
        r[0] for r in con.execute("SELECT narrative_id FROM narratives WHERE basis = ?", [basis]).fetchall()
    ]
    if not narrative_ids:
        return
    placeholders = ", ".join("?" for _ in narrative_ids)
    con.execute(f"DELETE FROM narrative_entity_stance WHERE narrative_id IN ({placeholders})", narrative_ids)
    con.execute(f"DELETE FROM narrative_members WHERE narrative_id IN ({placeholders})", narrative_ids)
    con.execute("DELETE FROM narratives WHERE basis = ?", [basis])


def store_narrative(con: duckdb.DuckDBPyConnection, cluster: NarrativeCluster) -> None:
    start, end = cluster.time_range
    con.execute(
        "INSERT INTO narratives VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            cluster.narrative_id,
            cluster.label,
            cluster.basis,
            start,
            end,
            cluster.size,
            cluster.distinct_authors,
            datetime.now(UTC),
        ],
    )
    con.executemany(
        "INSERT INTO narrative_members VALUES (?, ?)",
        [(cluster.narrative_id, item_id) for item_id in cluster.member_item_ids],
    )


def compute_entity_stance(con: duckdb.DuckDBPyConnection, basis: str) -> None:
    """Populate narrative_entity_stance for every narrative of `basis` from
    the current item_entities + entity_stance_edges tables. Pure rollup --
    an entity mention without a stance edge (e.g. stance detection hasn't
    reached that item yet) simply doesn't contribute a count, it doesn't
    block the rest of the narrative's row.
    """
    con.execute(
        """
        INSERT INTO narrative_entity_stance
        WITH item_entity_pairs AS (
            -- item_entities' PK includes surface_form, so one item can carry
            -- several rows for the same entity_id (multiple mentions) --
            -- dedupe to one (item_id, entity_id) pair before the join below,
            -- or count(*) would double-count a single item's stance.
            SELECT DISTINCT item_id, entity_id FROM item_entities
        )
        SELECT
            nm.narrative_id,
            iep.entity_id,
            count(*) FILTER (WHERE se.polarity = 'positive') AS positive_count,
            count(*) FILTER (WHERE se.polarity = 'negative') AS negative_count,
            count(*) FILTER (WHERE se.polarity = 'neutral') AS neutral_count,
            avg(se.strength) FILTER (WHERE se.polarity IS NOT NULL) AS mean_strength
        FROM narrative_members nm
        JOIN narratives n ON n.narrative_id = nm.narrative_id AND n.basis = ?
        JOIN item_entity_pairs iep ON iep.item_id = nm.item_id
        LEFT JOIN entity_stance_edges se ON se.item_id = nm.item_id AND se.entity_id = iep.entity_id
        GROUP BY nm.narrative_id, iep.entity_id
        """,
        [basis],
    )
