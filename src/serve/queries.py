"""All DuckDB query logic for the serve API, kept separate from the route
handlers in app.py so the handlers stay thin (parse params, call a query
function, 404 on None, return JSON).

Every function here takes an already-connected `con` (see db.py) and
returns plain dicts/lists -- JSON-serializable as-is, no Pydantic models
needed for a backend-only, no-frontend-yet validation pass.
"""

from __future__ import annotations

import dataclasses
import json

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


def get_entity_authors(con: duckdb.DuckDBPyConnection, entity_id: str, limit: int, min_volume: int = 5) -> dict | None:
    row = con.execute("SELECT canonical_name FROM entities WHERE entity_id = ?", [entity_id]).fetchone()
    if row is None:
        return None
    positive, negative = profiles.query_entity_authors(con, entity_id, limit=limit, min_volume=min_volume)
    return {
        "entity_id": entity_id,
        "canonical_name": row[0],
        "min_volume": min_volume,
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


def _build_source_url(source_type: str | None, source_specific_json: str | None) -> str | None:
    """Construct the real, openable link back to the original post, from
    source_specific's per-platform fields -- None (not a raised error) when
    the fields needed aren't present, so callers fall back to raw
    identifiers (item_id/source_native_id) for provenance instead of
    breaking the response.
    """
    if not source_specific_json:
        return None
    try:
        spec = json.loads(source_specific_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(spec, dict):
        return None

    if source_type == "telegram":
        username = spec.get("channel_username")
        message_id = spec.get("message_id")
        if username and message_id is not None:
            return f"https://t.me/{username}/{message_id}"
        return None
    if source_type == "channel":
        # A Telegram channel-metadata record (not an individual post) --
        # links to the channel itself, not a specific message.
        username = spec.get("channel_username")
        if username:
            return f"https://t.me/{username}"
        return None
    if source_type == "youtube_video":
        video_id = spec.get("video_id")
        if video_id:
            return f"https://youtube.com/watch?v={video_id}"
        return None
    return None


def get_author_entity_sources(
    con: duckdb.DuckDBPyConnection, author_id: str, entity_id: str, limit: int
) -> dict | None:
    """Query #source-drilldown: the actual items behind one author's stance
    profile toward one entity -- what /api/author/{id}/entity/{id}/sources
    (and any UI click-through on a profile row) resolves to. None if either
    the entity or the author doesn't exist at all (no items ever attributed
    to that author_id) -- an author that exists but has no stance edges
    toward this particular entity still gets a 200 with an empty list,
    since that's a real (if uninteresting) answer, not a 404.
    """
    entity_row = con.execute("SELECT canonical_name FROM entities WHERE entity_id = ?", [entity_id]).fetchone()
    if entity_row is None:
        return None

    author_row = con.execute(
        "SELECT count(*) FROM processed.items WHERE source_type || ':' || author_native_id = ?",
        [author_id],
    ).fetchone()
    if author_row is None or author_row[0] == 0:
        return None

    rows = con.execute(
        """
        SELECT se.item_id, i.source_type, i.source_native_id, i.source_specific,
               i.published_at, se.polarity, se.strength, se.confidence,
               substr(coalesce(i.text, ''), 1, 500) AS text_snippet,
               substr(coalesce(i.source_specific -> 'transcript' ->> 'text', ''), 1, 500) AS transcript_snippet
        FROM entity_stance_edges se
        JOIN processed.items i ON i.item_id = se.item_id
        WHERE se.entity_id = ? AND i.source_type || ':' || i.author_native_id = ?
        ORDER BY i.published_at DESC
        LIMIT ?
        """,
        [entity_id, author_id, limit],
    ).fetchall()

    items = [
        {
            "item_id": item_id,
            "source_type": source_type,
            "published_at": published_at,
            "polarity": polarity,
            "strength": strength,
            "confidence": confidence,
            "text": text_snippet or None,
            "transcript_snippet": transcript_snippet or None,
            "source_url": _build_source_url(source_type, source_specific),
            "source_native_id": source_native_id,
        }
        for item_id, source_type, source_native_id, source_specific, published_at, polarity, strength,
        confidence, text_snippet, transcript_snippet in rows
    ]

    return {
        "author_id": author_id,
        "entity_id": entity_id,
        "canonical_name": entity_row[0],
        "n_items": len(items),
        "items": items,
    }


def get_cluster_items(con: duckdb.DuckDBPyConnection, cluster_id: str) -> dict | None:
    """Query #source-drilldown: the full set of items in one coordination
    cluster -- `cluster_id` is any item_id known to be a member (there is no
    separately-persisted stable cluster identifier yet, see
    analysis/scope.py's compute_coordination_clusters docstring: its
    cluster_id is only a per-call union-find root, not stable across runs).
    Given any member item_id, a recursive CTE over processed.edges'
    shared_media/near_duplicate_text derived edges finds the rest of that
    item's connected component live -- cheap in practice (bounded by the
    cluster's own size, not the whole corpus: ~0.2s for a several-hundred-
    author edge_count observed in testing), so no separate persistence is
    needed for this to serve a live drill-down click.

    None only if `cluster_id` isn't a real item_id at all; a real item with
    no cluster edges still returns successfully as a cluster of size 1.
    """
    seed_exists = con.execute(
        "SELECT count(*) FROM processed.items WHERE item_id = ?", [cluster_id]
    ).fetchone()
    if seed_exists is None or seed_exists[0] == 0:
        return None

    member_rows = con.execute(
        """
        WITH RECURSIVE cluster_items(item_id) AS (
            SELECT $seed::VARCHAR
            UNION
            SELECT CASE WHEN e.src_item_id = ci.item_id THEN e.dst_item_id ELSE e.src_item_id END
            FROM cluster_items ci
            JOIN processed.edges e
              ON (e.src_item_id = ci.item_id OR e.dst_item_id = ci.item_id)
             AND e.origin = 'derived' AND e.edge_type IN ('shared_media', 'near_duplicate_text')
        )
        SELECT DISTINCT item_id FROM cluster_items
        """,
        {"seed": cluster_id},
    ).fetchall()
    member_ids = [r[0] for r in member_rows]

    detail_rows = con.execute(
        """
        SELECT item_id, source_type, source_native_id, source_specific, author_native_id,
               author_display_name, published_at, substr(coalesce(text, ''), 1, 300) AS text_snippet
        FROM processed.items
        WHERE item_id IN (SELECT UNNEST($ids))
        ORDER BY published_at
        """,
        {"ids": member_ids},
    ).fetchall()

    items = []
    distinct_authors: set[str] = set()
    for (
        item_id, source_type, source_native_id, source_specific, author_native_id,
        author_display_name, published_at, text_snippet,
    ) in detail_rows:
        author_id = f"{source_type}:{author_native_id}" if author_native_id else None
        if author_id:
            distinct_authors.add(author_id)
        items.append(
            {
                "item_id": item_id,
                "author_id": author_id,
                "author_display_name": author_display_name,
                "source_type": source_type,
                "published_at": published_at,
                "text_snippet": text_snippet,
                "source_url": _build_source_url(source_type, source_specific),
                "source_native_id": source_native_id,
            }
        )

    return {
        "cluster_id": cluster_id,
        "size": len(items),
        "distinct_authors": len(distinct_authors),
        "items": items,
    }


def get_item_detail(con: duckdb.DuckDBPyConnection, item_id: str) -> dict | None:
    row = con.execute(
        """
        SELECT item_id, source_type, source_native_id, source_specific,
               author_native_id, author_display_name, published_at, text
        FROM processed.items WHERE item_id = ?
        """,
        [item_id],
    ).fetchone()
    if row is None:
        return None
    (
        item_id, source_type, source_native_id, source_specific,
        author_native_id, author_display_name, published_at, text,
    ) = row

    transcript = None
    if source_specific:
        try:
            spec = json.loads(source_specific)
        except (json.JSONDecodeError, TypeError):
            spec = None
        if isinstance(spec, dict):
            transcript_obj = spec.get("transcript")
            if isinstance(transcript_obj, dict):
                transcript = transcript_obj.get("text")

    author_id = f"{source_type}:{author_native_id}" if author_native_id else None

    entity_rows = con.execute(
        """
        SELECT e.entity_id, e.canonical_name, e.entity_type, ie.surface_form,
               se.polarity, se.strength, se.confidence
        FROM item_entities ie
        JOIN entities e ON e.entity_id = ie.entity_id
        LEFT JOIN entity_stance_edges se ON se.item_id = ie.item_id AND se.entity_id = ie.entity_id
        WHERE ie.item_id = ?
        """,
        [item_id],
    ).fetchall()
    entities_found = [
        {
            "entity_id": r[0],
            "canonical_name": r[1],
            "entity_type": r[2],
            "surface_form": r[3],
            "polarity": r[4],
            "strength": r[5],
            "confidence": r[6],
        }
        for r in entity_rows
    ]

    return {
        "item_id": item_id,
        "source_type": source_type,
        "author_id": author_id,
        "author_display_name": author_display_name,
        "published_at": published_at,
        "text": text,
        "transcript": transcript,
        "source_url": _build_source_url(source_type, source_specific),
        "source_native_id": source_native_id,
        "entities": entities_found,
    }


def get_topic_coordination(con: duckdb.DuckDBPyConnection, topic_query: str, limit: int) -> dict:
    """Query #source-drilldown / NLP-query support: which coordination
    clusters (tight, multi-author narratives) touch a free-text topic
    keyword -- e.g. "Operation Sindoor" -> the reposted-content clusters
    whose member items mention it. Reuses the already-computed 'tight'
    narratives (build_narratives.py) rather than a fresh graph traversal
    per request; a plain parameterized ILIKE against member text, so this
    stays a safe, pre-defined query (no user-composed SQL) even though the
    match text itself is free-form.
    """
    rows = con.execute(
        """
        SELECT n.narrative_id, n.size, n.distinct_authors, n.time_range_start, n.time_range_end,
               any_value(substr(i.text, 1, 200)) AS sample_text
        FROM narratives n
        JOIN narrative_members nm ON nm.narrative_id = n.narrative_id
        JOIN processed.items i ON i.item_id = nm.item_id
        WHERE n.basis = 'tight' AND n.distinct_authors > 1
          AND concat_ws(' ', i.text, i.source_specific -> 'transcript' ->> 'text') ILIKE ?
        GROUP BY n.narrative_id, n.size, n.distinct_authors, n.time_range_start, n.time_range_end
        ORDER BY n.distinct_authors DESC, n.size DESC
        LIMIT ?
        """,
        [f"%{topic_query}%", limit],
    ).fetchall()
    clusters = [
        {
            "narrative_id": r[0],
            "size": r[1],
            "distinct_authors": r[2],
            "time_range_start": r[3],
            "time_range_end": r[4],
            "sample_text": r[5],
        }
        for r in rows
    ]
    return {"topic_query": topic_query, "clusters": clusters}
