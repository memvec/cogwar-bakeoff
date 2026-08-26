"""Phase B: derive coordination edges from the already-loaded `items` table.

Writes shared_media, near_duplicate_text, and temporal_cocluster edges into
the DuckDB `edges` table with origin='derived'. Recomputed from scratch on
every run: clear_derived_edges() deletes all prior origin='derived' rows
before anything is recomputed, so re-running never duplicates a derived
edge. origin='collected' rows (loaded in Phase A straight from raw parquet)
are never touched here -- this module only ever DELETEs origin='derived'
and INSERTs new origin='derived' rows.

Scale guard (DEFAULT_HASH_GROUP_CAP): shared_media and near_duplicate_text
both work by self-joining items on a shared hash value -- a hash shared by
N items naively produces N*(N-1)/2 pairwise edges. At small corpus sizes
(hundreds to thousands of items) this is harmless; at corpus sizes in the
millions, a single hash shared by tens of thousands of items (an empty-ish
caption, a viral forwarded image, boilerplate channel-join text) would
generate tens or hundreds of millions of edges from that ONE hash value --
not a coordination signal, just noise, and enough rows to blow out memory
and runtime. Any hash group larger than the cap is excluded from pairwise
edge generation entirely (both derive functions filter to `group_size
BETWEEN 2 AND cap` before self-joining) and reported separately as a
"mass-duplication" group so it's visible rather than silently dropped.
"""

from __future__ import annotations

import duckdb

from processing.load import fetchone

DEFAULT_WINDOW_HOURS = 24
DEFAULT_HASH_GROUP_CAP = 200

# Shared by derive_near_duplicate_text, its oversized-group check, and the
# print summary -- one definition of "what counts as this item's text" for
# duplicate detection (its own text, and separately, a YouTube video's
# transcript text).
_TEXT_HASH_POOL_SQL = """
    SELECT item_id, author_native_id, text AS sample_text, 'text' AS hash_field, text_hash AS hash_value
    FROM items
    WHERE text_hash IS NOT NULL

    UNION ALL

    SELECT item_id, author_native_id, source_specific -> 'transcript' ->> 'text' AS sample_text,
           'transcript' AS hash_field, source_specific -> 'transcript' ->> 'text_hash' AS hash_value
    FROM items
    WHERE source_specific -> 'transcript' ->> 'text_hash' IS NOT NULL
"""


def clear_derived_edges(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DELETE FROM edges WHERE origin = 'derived'")


def oversized_media_hash_groups(con: duckdb.DuckDBPyConnection, cap: int) -> list[tuple]:
    """media_hash groups bigger than `cap` -- excluded from derive_shared_media, reported instead."""
    return con.execute(
        """
        SELECT media_hash, count(*) AS n_items, count(DISTINCT author_native_id) AS n_authors,
               any_value(text) AS sample_text
        FROM items
        WHERE media_hash IS NOT NULL
        GROUP BY media_hash
        HAVING count(*) > $cap
        ORDER BY n_items DESC
        """,
        {"cap": cap},
    ).fetchall()


def oversized_text_hash_groups(con: duckdb.DuckDBPyConnection, cap: int) -> list[tuple]:
    """text/transcript hash groups bigger than `cap` -- excluded from derive_near_duplicate_text, reported instead."""
    return con.execute(
        "WITH text_hash_pool AS (" + _TEXT_HASH_POOL_SQL + ") "
        """
        SELECT hash_value, count(*) AS n_rows, count(DISTINCT item_id) AS n_items,
               count(DISTINCT author_native_id) AS n_authors, any_value(sample_text) AS sample_text
        FROM text_hash_pool
        GROUP BY hash_value
        HAVING count(*) > $cap
        ORDER BY n_rows DESC
        """,
        {"cap": cap},
    ).fetchall()


def derive_shared_media(con: duckdb.DuckDBPyConnection, cap: int) -> None:
    """Pairs of different items sharing a non-null media_hash. Undirected; evidence = the shared hash.

    Self-joins only within hash groups sized [2, cap] (see module docstring)
    -- group_sizes/eligible are computed and filtered BEFORE the self-join,
    not after, so an oversized group's rows never reach the join at all.
    """
    con.execute(
        """
        INSERT INTO edges
        WITH group_sizes AS (
            SELECT media_hash, count(*) AS n
            FROM items
            WHERE media_hash IS NOT NULL
            GROUP BY media_hash
        ),
        eligible AS (
            SELECT i.item_id, i.media_hash
            FROM items i
            JOIN group_sizes g ON i.media_hash = g.media_hash
            WHERE g.n BETWEEN 2 AND $cap
        )
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
        FROM eligible a
        JOIN eligible b ON a.media_hash = b.media_hash AND a.item_id < b.item_id
        """,
        {"cap": cap},
    )


def derive_near_duplicate_text(con: duckdb.DuckDBPyConnection, cap: int) -> None:
    """Pairs of different items sharing an identical text_hash -- exact match only, no fuzzy similarity yet.

    Checks both an item's own text_hash (content_hashes.text_hash, title/
    description/message text) and, for YouTube videos with a transcript,
    the transcript's text_hash (source_specific.transcript.text_hash) --
    two items can be flagged as matching text OR matching transcript, or a
    text match against a transcript. Undirected; evidence records which
    field matched on each side plus the shared hash. Same eligible-group
    scale guard as derive_shared_media, applied to hash_value groups within
    the combined text+transcript pool.
    """
    con.execute(
        "INSERT INTO edges "
        "WITH text_hash_pool AS (" + _TEXT_HASH_POOL_SQL + "), "
        """
        group_sizes AS (
            SELECT hash_value, count(*) AS n
            FROM text_hash_pool
            GROUP BY hash_value
        ),
        eligible_pool AS (
            SELECT p.item_id, p.hash_field, p.hash_value
            FROM text_hash_pool p
            JOIN group_sizes g ON p.hash_value = g.hash_value
            WHERE g.n BETWEEN 2 AND $cap
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
        FROM eligible_pool a
        JOIN eligible_pool b ON a.hash_value = b.hash_value AND a.item_id < b.item_id
        """,
        {"cap": cap},
    )


def derive_temporal_cocluster(con: duckdb.DuckDBPyConnection, window_hours: float) -> None:
    """Among items just linked by shared_media or near_duplicate_text, flag pairs posted by
    DIFFERENT authors within `window_hours` of each other -- same content,
    different accounts, tight timing: the actual coordination signal (as
    opposed to shared_media/near_duplicate_text alone, which just says
    "same content", coordinated or not). Undirected; evidence records the
    window, both author ids, and the time delta.

    No separate scale guard needed here: this only ever selects a subset of
    rows already produced by derive_shared_media/derive_near_duplicate_text
    above, both of which are already capped -- it can't be larger than its
    inputs.
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


def derive_all(
    con: duckdb.DuckDBPyConnection,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    hash_group_cap: int = DEFAULT_HASH_GROUP_CAP,
) -> None:
    clear_derived_edges(con)
    print(f"[derive] clearing prior derived edges, hash_group_cap={hash_group_cap} ...", flush=True)
    print("[derive] shared_media ...", flush=True)
    derive_shared_media(con, hash_group_cap)
    print("[derive] near_duplicate_text ...", flush=True)
    derive_near_duplicate_text(con, hash_group_cap)
    print("[derive] temporal_cocluster ...", flush=True)
    derive_temporal_cocluster(con, window_hours)


def _print_hash_clusters(con: duckdb.DuckDBPyConnection, edge_type: str, cap: int, top_n: int = 3) -> None:
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
            HAVING count(*) BETWEEN 2 AND $cap
            ORDER BY n_items DESC
            LIMIT $n
            """,
            {"n": top_n, "cap": cap},
        ).fetchall()
        for media_hash, n_items, n_authors in rows:
            print(f"  media_hash {media_hash[:12]}... shared by {n_items} items across {n_authors} authors")
    elif edge_type == "near_duplicate_text":
        rows = con.execute(
            "WITH text_hash_pool AS (" + _TEXT_HASH_POOL_SQL + ") "
            """
            SELECT hash_value, count(DISTINCT item_id) AS n_items, count(DISTINCT author_native_id) AS n_authors
            FROM text_hash_pool
            GROUP BY hash_value
            HAVING count(*) BETWEEN 2 AND $cap
            ORDER BY n_items DESC
            LIMIT $n
            """,
            {"n": top_n, "cap": cap},
        ).fetchall()
        for text_hash, n_items, n_authors in rows:
            print(f"  text_hash {text_hash[:12]}... shared by {n_items} items across {n_authors} authors")


def _print_mass_duplication_groups(con: duckdb.DuckDBPyConnection, cap: int) -> None:
    print(f"\nMass-duplication groups excluded from pairwise edge generation (hash shared by > {cap} items):")
    media_groups = oversized_media_hash_groups(con, cap)
    text_groups = oversized_text_hash_groups(con, cap)
    if not media_groups and not text_groups:
        print("  (none)")
        return

    for media_hash, n_items, n_authors, sample_text in media_groups:
        sample = (sample_text or "").replace("\n", " ")[:80]
        print(
            f"  [shared_media] {media_hash[:16]}...: {n_items} items, {n_authors} authors "
            f"-- SKIPPED (sample caption: {sample!r})"
        )
    for hash_value, n_rows, n_items, n_authors, sample_text in text_groups:
        sample = (sample_text or "").replace("\n", " ")[:80]
        print(
            f"  [near_duplicate_text] {hash_value[:16]}...: {n_items} items ({n_rows} pool rows), "
            f"{n_authors} authors -- SKIPPED (sample text: {sample!r})"
        )


def print_derive_summary(
    con: duckdb.DuckDBPyConnection, hash_group_cap: int = DEFAULT_HASH_GROUP_CAP, top_n: int = 3
) -> None:
    print("\n--- Derived edges (Phase B) ---")
    _print_hash_clusters(con, "shared_media", hash_group_cap, top_n)
    _print_hash_clusters(con, "near_duplicate_text", hash_group_cap, top_n)
    _print_mass_duplication_groups(con, hash_group_cap)

    (tc_count,) = fetchone(
        con.execute("SELECT count(*) FROM edges WHERE edge_type = 'temporal_cocluster' AND origin = 'derived'")
    )
    print(f"\ntemporal_cocluster: {tc_count} edges")
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
