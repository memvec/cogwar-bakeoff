"""Analysis scope: intersect coordination-cluster membership with
multilingual topic-keyword relevance to define exactly which items the
(expensive, model-based) entity/stance passes should run over.

Two independent filters, intersected by build_scope.py:

  1. Coordination filter -- connected components over the shared_media +
     near_duplicate_text derived edges (processing/derive.py), restricted
     to clusters with at least N distinct authors. Same algorithm as
     processing/clusters.py, reimplemented here against the analysis
     layer's ATTACHed `processed` schema (build_profiles.py's ATTACH
     pattern) rather than a direct connection to processed.duckdb, and
     parameterized by author-count threshold since this pass compares
     several thresholds in one run. temporal_cocluster edges are excluded
     from the graph for the same reason as processing/clusters.py: they're
     always a strict subset of the other two edge types (see
     processing/derive.py's derive_temporal_cocluster docstring), so they
     add no new connectivity. Mass-duplication groups (processing/
     derive.py's hash-group cap) need no separate exclusion here -- a
     capped group never produced any derived edge in the first place, so
     it's already absent from this graph.

  2. Topic filter -- case-insensitive substring match of
     configs/topic_keywords.json's keyword list against each item's
     combined text (Telegram message text; YouTube title+description,
     already concatenated into `text` at collection time in
     collection/youtube/mapping.py; plus transcript text where present).
     High-precision/low-recall by design -- an accepted tradeoff for a
     clean, cheap-to-verify initial scope; the config is meant to be
     expanded once a sample review shows misses.

Pure DuckDB + stdlib: no model calls, no imports from stance.py/entities.py/
any provider client. The bulk keyword scan runs as one DuckDB regexp_matches
pass (RE2, linear-time, no catastrophic-backtracking risk) rather than N
separate LIKE scans; per-item "which keyword matched" detail (only needed
for a handful of items at a time -- the report sample, the persisted scope
rows) is a plain Python substring check instead, using the exact same
lowercased keyword strings, so both paths agree on what counts as a match.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import duckdb

TOPIC_KEYWORDS_PATH = Path("configs/topic_keywords.json")

# shared_media + near_duplicate_text only -- see module docstring for why
# temporal_cocluster is excluded.
_CLUSTER_BASIS_EDGE_TYPES = ("shared_media", "near_duplicate_text")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS analysis_scope (
    item_id VARCHAR PRIMARY KEY,
    source_type VARCHAR,
    author_native_id VARCHAR,
    cluster_id VARCHAR,
    cluster_n_authors INTEGER,
    matched_topics VARCHAR,
    matched_keywords VARCHAR,
    min_authors_threshold INTEGER,
    computed_at TIMESTAMP
)
"""


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA_SQL)


# --- Coordination filter --------------------------------------------------


def _find(parent: dict[str, str], x: str) -> str:
    parent.setdefault(x, x)
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def _union(parent: dict[str, str], a: str, b: str) -> None:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[ra] = rb


def compute_coordination_clusters(con: duckdb.DuckDBPyConnection) -> dict[str, list[str]]:
    """cluster_id -> item_ids, over processed.edges (shared_media/near_duplicate_text only).

    Items with no such edge are absent (not part of any cluster). cluster_id
    is an arbitrary representative item_id (the union-find root) -- stable
    within one call, not meaningful across runs.
    """
    parent: dict[str, str] = {}
    edge_rows = con.execute(
        """
        SELECT DISTINCT src_item_id, dst_item_id
        FROM processed.edges
        WHERE origin = 'derived' AND edge_type IN ('shared_media', 'near_duplicate_text')
        """
    ).fetchall()
    for src, dst in edge_rows:
        _union(parent, src, dst)

    clusters: dict[str, list[str]] = {}
    for item_id in parent:
        clusters.setdefault(_find(parent, item_id), []).append(item_id)
    return clusters


def load_author_by_item(con: duckdb.DuckDBPyConnection, clusters: dict[str, list[str]]) -> dict[str, str | None]:
    """item_id -> author_native_id, for exactly the (bounded) set of items that are in some cluster."""
    all_item_ids = [item_id for members in clusters.values() for item_id in members]
    if not all_item_ids:
        return {}
    rows = con.execute(
        "SELECT item_id, author_native_id FROM processed.items WHERE item_id IN (SELECT UNNEST($ids))",
        {"ids": all_item_ids},
    ).fetchall()
    return dict(rows)


def item_to_cluster_map(clusters: dict[str, list[str]]) -> dict[str, str]:
    return {item_id: cluster_id for cluster_id, members in clusters.items() for item_id in members}


def coordination_set_for_threshold(
    clusters: dict[str, list[str]], author_by_item: dict[str, str | None], min_authors: int
) -> tuple[set[str], int]:
    """(item_ids in any cluster with >= min_authors distinct authors, number of such clusters)."""
    item_ids: set[str] = set()
    n_clusters = 0
    for members in clusters.values():
        authors = {author_by_item.get(m) for m in members}
        authors.discard(None)
        if len(authors) >= min_authors:
            n_clusters += 1
            item_ids.update(members)
    return item_ids, n_clusters


# --- Topic filter ----------------------------------------------------------


def load_topic_keywords(path: Path = TOPIC_KEYWORDS_PATH) -> list[tuple[str, str]]:
    """(topic, keyword) pairs, keyword lowercased, flattened across every topic's keyword list."""
    data = json.loads(path.read_text())
    return [(entry["topic"], kw.lower()) for entry in data["topics"] for kw in entry["keywords"]]


def _alternation_pattern(keyword_pairs: list[tuple[str, str]]) -> str:
    keywords = sorted({kw for _, kw in keyword_pairs})
    return "(" + "|".join(re.escape(kw) for kw in keywords) + ")"


def compute_on_topic_item_ids(con: duckdb.DuckDBPyConnection, keyword_pairs: list[tuple[str, str]]) -> set[str]:
    """Corpus-wide (not cluster-restricted) set of items whose combined text matches any keyword."""
    pattern = _alternation_pattern(keyword_pairs)
    rows = con.execute(
        """
        SELECT item_id
        FROM processed.items
        WHERE regexp_matches(
            lower(concat_ws(' ', text, source_specific -> 'transcript' ->> 'text')),
            $pattern
        )
        """,
        {"pattern": pattern},
    ).fetchall()
    return {r[0] for r in rows}


def matched_keywords_for_text(combined_text: str, keyword_pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """(topic, keyword) pairs that match this text -- plain substring check, same semantics as the SQL scan."""
    lowered = (combined_text or "").lower()
    return [(topic, kw) for topic, kw in keyword_pairs if kw in lowered]


def fetch_combined_text(con: duckdb.DuckDBPyConnection, item_ids: list[str]) -> dict[str, str]:
    if not item_ids:
        return {}
    rows = con.execute(
        """
        SELECT item_id, concat_ws(' ', text, source_specific -> 'transcript' ->> 'text') AS combined_text
        FROM processed.items
        WHERE item_id IN (SELECT UNNEST($ids))
        """,
        {"ids": item_ids},
    ).fetchall()
    return dict(rows)


# --- Persistence -------------------------------------------------------------


def persist_scope(
    con: duckdb.DuckDBPyConnection,
    item_ids: set[str],
    item_to_cluster: dict[str, str],
    cluster_n_authors: dict[str, int],
    keyword_pairs: list[tuple[str, str]],
    min_authors: int,
) -> None:
    """Fully replaces analysis_scope with the current item_ids -- this pass always reflects the
    current DB + keyword list, not an accumulating history (same idempotent-rebuild philosophy as
    processing/build.py).
    """
    con.execute("DELETE FROM analysis_scope")
    if not item_ids:
        return

    ordered_ids = list(item_ids)
    meta_rows = con.execute(
        """
        SELECT item_id, source_type, author_native_id,
               concat_ws(' ', text, source_specific -> 'transcript' ->> 'text') AS combined_text
        FROM processed.items
        WHERE item_id IN (SELECT UNNEST($ids))
        """,
        {"ids": ordered_ids},
    ).fetchall()

    computed_at = datetime.now(UTC)
    rows = []
    for item_id, source_type, author_native_id, combined_text in meta_rows:
        matches = matched_keywords_for_text(combined_text, keyword_pairs)
        topics = sorted({t for t, _ in matches})
        keywords = sorted({k for _, k in matches})
        cluster_id = item_to_cluster.get(item_id)
        rows.append(
            (
                item_id,
                source_type,
                author_native_id,
                cluster_id,
                cluster_n_authors.get(cluster_id, 0) if cluster_id else None,
                ", ".join(topics),
                ", ".join(keywords),
                min_authors,
                computed_at,
            )
        )
    con.executemany(
        "INSERT INTO analysis_scope VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
