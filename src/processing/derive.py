"""Phase B: derive coordination edges from the already-loaded `items` table.

Writes shared_media, near_duplicate_text, and temporal_cocluster edges into
the DuckDB `edges` table with origin='derived'. Recomputed from scratch on
every run: clear_derived_edges() deletes all prior origin='derived' rows
before anything is recomputed, so re-running never duplicates a derived
edge. origin='collected' rows (loaded in Phase A straight from raw parquet)
are never touched here -- this module only ever DELETEs origin='derived'
and INSERTs new origin='derived' rows.
"""

from __future__ import annotations

import duckdb

from processing.load import fetchone

DEFAULT_WINDOW_HOURS = 24


def clear_derived_edges(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DELETE FROM edges WHERE origin = 'derived'")


def derive_shared_media(con: duckdb.DuckDBPyConnection) -> None:
    """Pairs of different items sharing a non-null media_hash. Undirected; evidence = the shared hash."""
    con.execute(
        """
        INSERT INTO edges
        SELECT
            gen_random_uuid()::VARCHAR AS edge_id,
            'shared_media' AS edge_type,
            a.item_id AS src_item_id,
            b.item_id AS dst_item_id,
            NULL AS dst_external_ref,
            FALSE AS directed,
            NULL AS weight,
            now()::TIMESTAMP AS observed_at,
            'derived' AS origin,
            {'shared_hash': a.media_hash}::JSON AS evidence
        FROM items a
        JOIN items b ON a.media_hash = b.media_hash AND a.item_id < b.item_id
        WHERE a.media_hash IS NOT NULL
        """
    )


def derive_near_duplicate_text(con: duckdb.DuckDBPyConnection) -> None:
    """Pairs of different items sharing an identical text_hash -- exact match only, no fuzzy similarity yet.

    Checks both an item's own text_hash (content_hashes.text_hash, title/
    description/message text) and, for YouTube videos with a transcript,
    the transcript's text_hash (source_specific.transcript.text_hash) --
    two items can be flagged as matching text OR matching transcript, or a
    text match against a transcript. Undirected; evidence records which
    field matched on each side plus the shared hash.
    """
    con.execute(
        """
        INSERT INTO edges
        WITH text_hash_pool AS (
            SELECT item_id, 'text' AS hash_field, text_hash AS hash_value
            FROM items
            WHERE text_hash IS NOT NULL

            UNION ALL

            SELECT item_id, 'transcript' AS hash_field,
                   source_specific -> 'transcript' ->> 'text_hash' AS hash_value
            FROM items
            WHERE source_specific -> 'transcript' ->> 'text_hash' IS NOT NULL
        )
        SELECT
            gen_random_uuid()::VARCHAR AS edge_id,
            'near_duplicate_text' AS edge_type,
            a.item_id AS src_item_id,
            b.item_id AS dst_item_id,
            NULL AS dst_external_ref,
            FALSE AS directed,
            NULL AS weight,
            now()::TIMESTAMP AS observed_at,
            'derived' AS origin,
            {'shared_hash': a.hash_value, 'src_field': a.hash_field, 'dst_field': b.hash_field}::JSON AS evidence
        FROM text_hash_pool a
        JOIN text_hash_pool b ON a.hash_value = b.hash_value AND a.item_id < b.item_id
        """
    )


def derive_temporal_cocluster(con: duckdb.DuckDBPyConnection, window_hours: float) -> None:
    """Among items just linked by shared_media or near_duplicate_text, flag pairs posted by
    DIFFERENT authors within `window_hours` of each other -- same content,
    different accounts, tight timing: the actual coordination signal (as
    opposed to shared_media/near_duplicate_text alone, which just says
    "same content", coordinated or not). Undirected; evidence records the
    window, both author ids, and the time delta.
    """
    con.execute(
        """
        INSERT INTO edges
        SELECT
            gen_random_uuid()::VARCHAR AS edge_id,
            'temporal_cocluster' AS edge_type,
            e.src_item_id,
            e.dst_item_id,
            NULL AS dst_external_ref,
            FALSE AS directed,
            NULL AS weight,
            now()::TIMESTAMP AS observed_at,
            'derived' AS origin,
            {
                'window_hours': $window_hours,
                'basis_edge_type': e.edge_type,
                'author_src': a.author_native_id,
                'author_dst': b.author_native_id,
                'time_delta_seconds': abs(epoch(a.published_at) - epoch(b.published_at))
            }::JSON AS evidence
        FROM edges e
        JOIN items a ON e.src_item_id = a.item_id
        JOIN items b ON e.dst_item_id = b.item_id
        WHERE e.origin = 'derived'
          AND e.edge_type IN ('shared_media', 'near_duplicate_text')
          AND a.author_native_id IS NOT NULL AND b.author_native_id IS NOT NULL
          AND a.author_native_id != b.author_native_id
          AND a.published_at IS NOT NULL AND b.published_at IS NOT NULL
          AND abs(epoch(a.published_at) - epoch(b.published_at)) <= $window_seconds
        """,
        {"window_hours": window_hours, "window_seconds": window_hours * 3600},
    )


def derive_all(con: duckdb.DuckDBPyConnection, window_hours: float = DEFAULT_WINDOW_HOURS) -> None:
    clear_derived_edges(con)
    derive_shared_media(con)
    derive_near_duplicate_text(con)
    derive_temporal_cocluster(con, window_hours)


def _print_hash_clusters(con: duckdb.DuckDBPyConnection, edge_type: str, top_n: int = 3) -> None:
    (count,) = fetchone(
        con.execute("SELECT count(*) FROM edges WHERE edge_type = $t AND origin = 'derived'", {"t": edge_type})
    )
    print(f"{edge_type}: {count} edges")
    if count == 0:
        return

    if edge_type == "shared_media":
        rows = con.execute(
            """
            SELECT media_hash, count(DISTINCT item_id) AS n_items, count(DISTINCT author_native_id) AS n_authors
            FROM items
            WHERE media_hash IS NOT NULL
            GROUP BY media_hash
            HAVING count(*) > 1
            ORDER BY n_items DESC
            LIMIT $n
            """,
            {"n": top_n},
        ).fetchall()
        for media_hash, n_items, n_authors in rows:
            print(f"  media_hash {media_hash[:12]}... shared by {n_items} items across {n_authors} authors")
    elif edge_type == "near_duplicate_text":
        rows = con.execute(
            """
            WITH text_hash_pool AS (
                SELECT item_id, author_native_id, text_hash AS hash_value
                FROM items WHERE text_hash IS NOT NULL
                UNION ALL
                SELECT item_id, author_native_id, source_specific -> 'transcript' ->> 'text_hash' AS hash_value
                FROM items WHERE source_specific -> 'transcript' ->> 'text_hash' IS NOT NULL
            )
            SELECT hash_value, count(DISTINCT item_id) AS n_items, count(DISTINCT author_native_id) AS n_authors
            FROM text_hash_pool
            GROUP BY hash_value
            HAVING count(*) > 1
            ORDER BY n_items DESC
            LIMIT $n
            """,
            {"n": top_n},
        ).fetchall()
        for text_hash, n_items, n_authors in rows:
            print(f"  text_hash {text_hash[:12]}... shared by {n_items} items across {n_authors} authors")


def print_derive_summary(con: duckdb.DuckDBPyConnection, top_n: int = 3) -> None:
    print("\n--- Derived edges (Phase B) ---")
    _print_hash_clusters(con, "shared_media", top_n)
    _print_hash_clusters(con, "near_duplicate_text", top_n)

    (tc_count,) = fetchone(
        con.execute("SELECT count(*) FROM edges WHERE edge_type = 'temporal_cocluster' AND origin = 'derived'")
    )
    print(f"temporal_cocluster: {tc_count} edges")
    if tc_count:
        rows = con.execute(
            """
            SELECT
                evidence->>'basis_edge_type',
                evidence->>'author_src',
                evidence->>'author_dst',
                (evidence->>'time_delta_seconds')::DOUBLE / 3600
            FROM edges
            WHERE edge_type = 'temporal_cocluster' AND origin = 'derived'
            ORDER BY (evidence->>'time_delta_seconds')::DOUBLE ASC
            LIMIT $n
            """,
            {"n": top_n},
        ).fetchall()
        for basis, author_src, author_dst, delta_hours in rows:
            print(
                f"  {basis} match: {author_src} <-> {author_dst}, {delta_hours:.2f}h apart"
            )
