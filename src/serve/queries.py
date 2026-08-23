"""All DuckDB query logic for the serve API, kept separate from the route
handlers in app.py so the handlers stay thin (parse params, call a query
function, 404 on None, return JSON).

Every function here takes an already-connected `con` (see db.py) and
returns plain dicts/lists -- JSON-serializable as-is, no Pydantic models
needed for a backend-only, no-frontend-yet validation pass.
"""

from __future__ import annotations

import dataclasses

import duckdb

from analysis import profiles

_COORDINATION_EDGE_TYPES = ("near_duplicate_text", "shared_media", "temporal_cocluster")
_VALID_BUCKETS = ("day", "week", "month")

# Node "dominant stance" bucketing: a volume-weighted average net_stance
# within +-this band is called "neutral" rather than forcing a lean out of
# noise near zero. Same spirit as author_entity_profiles.net_stance's own
# scale ([-1, 1]), just a display-time bucketing choice, not a stored value.
_DOMINANT_STANCE_BAND = 0.15


def get_stats(con: duckdb.DuckDBPyConnection) -> dict:
    n_items = con.execute("SELECT count(*) FROM processed.items").fetchone()[0]  # type: ignore[index]
    n_authors = con.execute(
        "SELECT count(DISTINCT source_type || ':' || author_native_id) FROM processed.items WHERE author_native_id IS NOT NULL"
    ).fetchone()[0]  # type: ignore[index]
    n_entities = con.execute("SELECT count(*) FROM entities").fetchone()[0]  # type: ignore[index]
    n_stance_edges = con.execute("SELECT count(*) FROM entity_stance_edges").fetchone()[0]  # type: ignore[index]
    n_coordination_clusters = con.execute(
        "SELECT count(*) FROM narratives WHERE basis = 'tight' AND distinct_authors > 1"
    ).fetchone()[0]  # type: ignore[index]
    return {
        "n_items": n_items,
        "n_authors": n_authors,
        "n_entities": n_entities,
        "n_stance_edges": n_stance_edges,
        "n_coordination_clusters": n_coordination_clusters,
    }


def list_entities(con: duckdb.DuckDBPyConnection, limit: int, offset: int) -> list[dict]:
    rows = con.execute(
        """
        SELECT e.entity_id, e.canonical_name, e.entity_type, count(DISTINCT se.item_id) AS volume
        FROM entities e
        LEFT JOIN entity_stance_edges se ON se.entity_id = e.entity_id
        GROUP BY e.entity_id, e.canonical_name, e.entity_type
        ORDER BY volume DESC, e.canonical_name
        LIMIT ? OFFSET ?
        """,
        [limit, offset],
    ).fetchall()
    return [
        {"entity_id": r[0], "canonical_name": r[1], "entity_type": r[2], "volume": r[3]}
        for r in rows
    ]


def list_authors(con: duckdb.DuckDBPyConnection, limit: int, offset: int) -> list[dict]:
    rows = con.execute(
        """
        SELECT source_type || ':' || author_native_id AS author_id,
               arg_max(author_display_name, published_at) AS display_name,
               source_type,
               count(*) AS item_count
        FROM processed.items
        WHERE author_native_id IS NOT NULL
        GROUP BY source_type, author_native_id
        ORDER BY item_count DESC
        LIMIT ? OFFSET ?
        """,
        [limit, offset],
    ).fetchall()
    return [
        {"author_id": r[0], "display_name": r[1], "source_type": r[2], "item_count": r[3]}
        for r in rows
    ]


def get_author_profile(con: duckdb.DuckDBPyConnection, author_id: str, limit: int) -> dict | None:
    row = con.execute(
        """
        SELECT arg_max(author_display_name, published_at) AS display_name,
               any_value(source_type) AS source_type,
               count(*) AS item_count,
               min(published_at) AS first_seen,
               max(published_at) AS last_seen
        FROM processed.items
        WHERE source_type || ':' || author_native_id = ?
        """,
        [author_id],
    ).fetchone()
    if row is None or row[2] == 0:
        return None
    display_name, source_type, item_count, first_seen, last_seen = row

    stance_rows = con.execute(
        """
        SELECT ap.entity_id, e.canonical_name, e.entity_type,
               ap.net_stance, ap.stance_consistency, ap.volume,
               ap.positive_count, ap.negative_count, ap.neutral_count,
               ap.narrative_spread,
               abs(ap.net_stance) * ap.volume AS score
        FROM author_entity_profiles ap
        JOIN entities e ON e.entity_id = ap.entity_id
        WHERE ap.author_id = ?
        ORDER BY score DESC
        LIMIT ?
        """,
        [author_id, limit],
    ).fetchall()
    stance_vector = [
        {
            "entity_id": r[0],
            "canonical_name": r[1],
            "entity_type": r[2],
            "net_stance": r[3],
            "stance_consistency": r[4],
            "volume": r[5],
            "positive_count": r[6],
            "negative_count": r[7],
            "neutral_count": r[8],
            "narrative_spread": r[9],
            "score": r[10],
        }
        for r in stance_rows
    ]

    return {
        "author_id": author_id,
        "display_name": display_name,
        "source_type": source_type,
        "item_count": item_count,
        "time_span": {"first_seen": first_seen, "last_seen": last_seen},
        "stance_vector": stance_vector,
        # §2.3 (generate vs. amplify) is not built yet -- this pass has no
        # forward/timing analysis to draw on, so the field exists on the
        # response contract now (frontend can code against it) but is
        # always null until that pass exists, rather than being omitted.
        "generate_vs_amplify": None,
    }


def _ranked_author_to_dict(r: profiles.RankedAuthor) -> dict:
    return dataclasses.asdict(r)


def get_entity_authors(con: duckdb.DuckDBPyConnection, entity_id: str, limit: int) -> dict | None:
    row = con.execute("SELECT canonical_name FROM entities WHERE entity_id = ?", [entity_id]).fetchone()
    if row is None:
        return None
    positive, negative = profiles.query_entity_authors(con, entity_id, limit=limit)
    return {
        "entity_id": entity_id,
        "canonical_name": row[0],
        "consistently_positive": [_ranked_author_to_dict(r) for r in positive],
        "consistently_negative": [_ranked_author_to_dict(r) for r in negative],
    }


def get_entity_timeline(con: duckdb.DuckDBPyConnection, entity_id: str, bucket: str) -> dict | None:
    row = con.execute("SELECT canonical_name FROM entities WHERE entity_id = ?", [entity_id]).fetchone()
    if row is None:
        return None
    bucket = bucket if bucket in _VALID_BUCKETS else "week"  # validated against an allowlist before any SQL interpolation

    rows = con.execute(
        f"""
        SELECT date_trunc('{bucket}', i.published_at) AS bucket_start, se.polarity, count(*) AS n
        FROM entity_stance_edges se
        JOIN processed.items i ON i.item_id = se.item_id
        WHERE se.entity_id = ? AND i.published_at IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1
        """,
        [entity_id],
    ).fetchall()

    buckets: dict[str, dict[str, int]] = {}
    for bucket_start, polarity, n in rows:
        key = bucket_start.isoformat()
        buckets.setdefault(key, {"positive": 0, "negative": 0, "neutral": 0})
        buckets[key][polarity] = n
    timeline = [{"bucket_start": k, **v} for k, v in sorted(buckets.items())]

    return {
        "entity_id": entity_id,
        "canonical_name": row[0],
        "bucket_size": bucket,
        "timeline": timeline,
    }


def get_narrative(con: duckdb.DuckDBPyConnection, narrative_id: str) -> dict | None:
    row = con.execute(
        """
        SELECT label, basis, time_range_start, time_range_end, size, distinct_authors
        FROM narratives WHERE narrative_id = ?
        """,
        [narrative_id],
    ).fetchone()
    if row is None:
        return None
    label, basis, start, end, size, distinct_authors = row

    member_rows = con.execute(
        """
        SELECT nm.item_id, i.source_type || ':' || i.author_native_id AS author_id,
               i.author_display_name, i.published_at, substr(i.text, 1, 200) AS text_snippet
        FROM narrative_members nm
        JOIN processed.items i ON i.item_id = nm.item_id
        WHERE nm.narrative_id = ?
        ORDER BY i.published_at
        """,
        [narrative_id],
    ).fetchall()
    members = [
        {"item_id": r[0], "author_id": r[1], "author_display_name": r[2], "published_at": r[3], "text_snippet": r[4]}
        for r in member_rows
    ]

    entity_rows = con.execute(
        """
        SELECT nes.entity_id, e.canonical_name, nes.positive_count, nes.negative_count, nes.neutral_count, nes.mean_strength
        FROM narrative_entity_stance nes
        JOIN entities e ON e.entity_id = nes.entity_id
        WHERE nes.narrative_id = ?
        ORDER BY (nes.positive_count + nes.negative_count + nes.neutral_count) DESC
        """,
        [narrative_id],
    ).fetchall()
    entity_stance = [
        {"entity_id": r[0], "canonical_name": r[1], "positive_count": r[2], "negative_count": r[3], "neutral_count": r[4], "mean_strength": r[5]}
        for r in entity_rows
    ]

    author_rows = con.execute(
        """
        SELECT i.source_type || ':' || i.author_native_id AS author_id,
               arg_max(i.author_display_name, i.published_at) AS display_name,
               count(*) AS item_count_in_narrative
        FROM narrative_members nm
        JOIN processed.items i ON i.item_id = nm.item_id
        WHERE nm.narrative_id = ? AND i.author_native_id IS NOT NULL
        GROUP BY i.source_type, i.author_native_id
        ORDER BY item_count_in_narrative DESC
        """,
        [narrative_id],
    ).fetchall()
    coordinating_authors = [
        {"author_id": r[0], "display_name": r[1], "item_count_in_narrative": r[2]} for r in author_rows
    ]

    return {
        "narrative_id": narrative_id,
        "label": label,
        "basis": basis,
        "size": size,
        "distinct_authors": distinct_authors,
        "time_range": {"start": start, "end": end},
        "members": members,
        "entity_stance": entity_stance,
        "coordinating_authors": coordinating_authors,
    }


def get_coordination_graph(con: duckdb.DuckDBPyConnection, min_edges: int, limit: int) -> dict:
    """Nodes = authors touching a multi-author derived edge (shared_media /
    near_duplicate_text / temporal_cocluster between two DIFFERENT authors)
    or a multi-author tight narrative. Edges = those derived edges,
    aggregated per author-PAIR (not per raw edge -- a hairball of 17k raw
    edges collapses to however many distinct author pairs actually
    coordinate), thresholded by `min_edges` and capped to the top `limit`
    pairs by edge_count so the default view is the strongest clusters, not
    every incidental pair.
    """
    edge_types_sql = ", ".join(f"'{t}'" for t in _COORDINATION_EDGE_TYPES)
    edge_rows = con.execute(
        f"""
        WITH derived AS (
            SELECT e.edge_type,
                   si.source_type || ':' || si.author_native_id AS src_author,
                   di.source_type || ':' || di.author_native_id AS dst_author,
                   si.published_at AS src_published_at, di.published_at AS dst_published_at
            FROM processed.edges e
            JOIN processed.items si ON si.item_id = e.src_item_id
            JOIN processed.items di ON di.item_id = e.dst_item_id
            WHERE e.origin = 'derived' AND e.edge_type IN ({edge_types_sql})
              AND si.author_native_id IS NOT NULL AND di.author_native_id IS NOT NULL
        ),
        pairs AS (
            SELECT
                LEAST(src_author, dst_author) AS author_a,
                GREATEST(src_author, dst_author) AS author_b,
                edge_type,
                CASE WHEN src_published_at IS NOT NULL AND dst_published_at IS NOT NULL
                     THEN abs(epoch(dst_published_at) - epoch(src_published_at))
                     ELSE NULL END AS time_gap_seconds
            FROM derived
            WHERE src_author != dst_author
        )
        SELECT author_a, author_b, count(*) AS edge_count,
               min(time_gap_seconds) AS min_time_gap_seconds,
               list(DISTINCT edge_type) AS edge_types
        FROM pairs
        GROUP BY author_a, author_b
        HAVING count(*) >= ?
        ORDER BY edge_count DESC
        LIMIT ?
        """,
        [min_edges, limit],
    ).fetchall()

    node_ids: set[str] = set()
    edges = []
    for author_a, author_b, edge_count, min_gap, edge_types in edge_rows:
        node_ids.add(author_a)
        node_ids.add(author_b)
        edges.append(
            {
                "source": author_a,
                "target": author_b,
                "edge_count": edge_count,
                "min_time_gap_seconds": min_gap,
                "edge_types": edge_types,
            }
        )

    narrative_author_rows = con.execute(
        """
        SELECT DISTINCT i.source_type || ':' || i.author_native_id AS author_id
        FROM narratives n
        JOIN narrative_members nm ON nm.narrative_id = n.narrative_id
        JOIN processed.items i ON i.item_id = nm.item_id
        WHERE n.basis = 'tight' AND n.distinct_authors > 1 AND i.author_native_id IS NOT NULL
        """
    ).fetchall()
    for (author_id,) in narrative_author_rows:
        node_ids.add(author_id)

    if not node_ids:
        return {"nodes": [], "edges": []}

    node_id_list = list(node_ids)
    placeholders = ", ".join("?" for _ in node_id_list)
    node_rows = con.execute(
        f"""
        WITH item_stats AS (
            SELECT source_type || ':' || author_native_id AS author_id,
                   arg_max(author_display_name, published_at) AS display_name,
                   any_value(source_type) AS source_type,
                   count(*) AS total_items
            FROM processed.items
            WHERE author_native_id IS NOT NULL
              AND source_type || ':' || author_native_id IN ({placeholders})
            GROUP BY source_type, author_native_id
        ),
        entity_stats AS (
            SELECT ap.author_id,
                   count(DISTINCT ap.entity_id) AS distinct_entities,
                   sum(ap.net_stance * ap.volume) / nullif(sum(ap.volume), 0) AS weighted_net_stance
            FROM author_entity_profiles ap
            WHERE ap.author_id IN ({placeholders})
            GROUP BY ap.author_id
        )
        SELECT s.author_id, s.display_name, s.source_type, s.total_items,
               coalesce(es.distinct_entities, 0), es.weighted_net_stance
        FROM item_stats s
        LEFT JOIN entity_stats es ON es.author_id = s.author_id
        """,
        node_id_list + node_id_list,
    ).fetchall()

    nodes = []
    for author_id, display_name, source_type, total_items, distinct_entities, weighted_net_stance in node_rows:
        if weighted_net_stance is None:
            dominant = "unknown"  # no stance-bearing items for this author at all
        elif weighted_net_stance > _DOMINANT_STANCE_BAND:
            dominant = "positive"
        elif weighted_net_stance < -_DOMINANT_STANCE_BAND:
            dominant = "negative"
        else:
            dominant = "neutral"
        nodes.append(
            {
                "id": author_id,
                "display_name": display_name,
                "source_type": source_type,
                "total_items": total_items,
                "distinct_entities": distinct_entities,
                "dominant_stance": dominant,
            }
        )

    return {"nodes": nodes, "edges": edges}
