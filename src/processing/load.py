"""Phase A: load raw collection output into a clean DuckDB database.

Reads items (JSONL, data/raw/items/*.jsonl), edges and observations
(parquet, data/raw/edges/*.parquet and data/raw/observations/*.parquet)
across every source (Telegram + YouTube write into the same shared
locations, doc collection_schema.md) and loads them into four tables:
items, edges, observations, and a derived `nodes` table (the channel/author
items, i.e. the graph's node list).

Idempotent by construction: every table is CREATE OR REPLACE'd from the raw
files on each run, so re-running always reflects exactly what's on disk in
data/raw/ -- nothing here is ever incrementally merged, and there is no
persisted state in the database that isn't reproducible from raw.
item_id (and edge src/dst item ids) are plain VARCHAR throughout, matching
how pydantic serializes UUIDs to JSON/parquet -- the join key across all
four tables.
"""

from __future__ import annotations

import duckdb

from processing import config


def fetchone(cur: duckdb.DuckDBPyConnection) -> tuple:
    """cur.fetchone(), asserting the row exists.

    Our aggregate queries (count(*), min/max with no GROUP BY) always
    return exactly one row -- even over an empty table -- but duckdb's
    stubs type fetchone() as returning `tuple | None`, since that's true
    for row-producing queries in general.
    """
    row = cur.fetchone()
    assert row is not None
    return row

ITEM_COLUMNS = {
    "item_id": "VARCHAR",
    "source_type": "VARCHAR",
    "source_native_id": "VARCHAR",
    "parent_item_id": "VARCHAR",
    "text": "VARCHAR",
    "text_normalized": "VARCHAR",
    "language_detected": "VARCHAR",
    "language_declared": "VARCHAR",
    "script": "VARCHAR",
    "published_at": "VARCHAR",
    "edited_at": "VARCHAR",
    "author_native_id": "VARCHAR",
    "author_display_name": "VARCHAR",
    "engagement": "JSON",
    "media": "JSON",
    "entities": "JSON",
    "source_specific": "JSON",
    "raw_payload_ref": "VARCHAR",
    "provenance": "JSON",
    "extraction_confidence": "DOUBLE",
    "content_hashes": "JSON",
    "account_created_at": "VARCHAR",
    "account_age_at_observation": "DOUBLE",
}


def _empty_items_table(con: duckdb.DuckDBPyConnection) -> None:
    columns_sql = ", ".join(f"{name} {typ}" for name, typ in ITEM_COLUMNS.items())
    con.execute(
        f"CREATE OR REPLACE TABLE items ({columns_sql}, text_hash VARCHAR, media_hash VARCHAR)"
    )


def load_items(con: duckdb.DuckDBPyConnection) -> None:
    """`items` table. published_at/author_native_id/author_display_name/source_type/script/
    language_detected/text_hash/media_hash are first-class typed columns;
    engagement/media/entities/source_specific/provenance/content_hashes stay
    DuckDB JSON (queryable natively via -> / ->>, no fixed shape assumed --
    source_specific in particular varies by source_type).
    """
    if not any(config.ITEMS_DIR.glob("*.jsonl")):
        _empty_items_table(con)
        return

    con.execute(
        f"""
        CREATE OR REPLACE TABLE items AS
        SELECT
            item_id,
            source_type,
            source_native_id,
            parent_item_id,
            text,
            text_normalized,
            language_detected,
            language_declared,
            script,
            published_at::TIMESTAMP AS published_at,
            edited_at::TIMESTAMP AS edited_at,
            author_native_id,
            author_display_name,
            engagement,
            media,
            entities,
            source_specific,
            raw_payload_ref,
            provenance,
            extraction_confidence,
            content_hashes,
            content_hashes->>'text_hash' AS text_hash,
            content_hashes->>'media_hash' AS media_hash,
            account_created_at::TIMESTAMP AS account_created_at,
            account_age_at_observation
        FROM read_json(
            '{config.ITEMS_DIR}/*.jsonl',
            format = 'newline_delimited',
            columns = $cols
        )
        """,
        {"cols": ITEM_COLUMNS},
    )


def load_edges(con: duckdb.DuckDBPyConnection) -> None:
    """`edges` table, collected edges only (raw parquet never contains
    origin='derived' rows -- those exist only inside DuckDB, see derive.py).
    `evidence` was written as a JSON string (edges/mapping.py has different
    keys per edge_type, so a single struct schema across all rows isn't
    viable in parquet) -- cast back to DuckDB's JSON type here so it's
    queryable with -> / ->>.
    """
    if not any(config.EDGES_DIR.glob("*.parquet")):
        con.execute(
            "CREATE OR REPLACE TABLE edges (edge_id VARCHAR, edge_type VARCHAR, "
            "src_item_id VARCHAR, dst_item_id VARCHAR, dst_external_ref VARCHAR, "
            "directed BOOLEAN, weight DOUBLE, observed_at TIMESTAMP, origin VARCHAR, evidence JSON)"
        )
        return

    con.execute(
        f"""
        CREATE OR REPLACE TABLE edges AS
        SELECT
            edge_id,
            edge_type,
            src_item_id,
            dst_item_id,
            dst_external_ref,
            directed,
            weight,
            observed_at::TIMESTAMP AS observed_at,
            origin,
            evidence::JSON AS evidence
        FROM read_parquet('{config.EDGES_DIR}/*.parquet', union_by_name = true)
        """
    )


def load_observations(con: duckdb.DuckDBPyConnection) -> None:
    """`observations` table -- the append-only reputation time series (doc §9), loaded as-is."""
    if not any(config.OBSERVATIONS_DIR.glob("*.parquet")):
        con.execute(
            "CREATE OR REPLACE TABLE observations (node_item_id VARCHAR, observed_at TIMESTAMP, "
            "subscriber_or_follower_count BIGINT, view_count BIGINT, post_count_seen BIGINT, "
            "verified_status BOOLEAN, collection_run_id VARCHAR)"
        )
        return

    con.execute(
        f"""
        CREATE OR REPLACE TABLE observations AS
        SELECT
            node_item_id,
            observed_at::TIMESTAMP AS observed_at,
            subscriber_or_follower_count,
            view_count,
            post_count_seen,
            verified_status,
            collection_run_id
        FROM read_parquet('{config.OBSERVATIONS_DIR}/*.parquet', union_by_name = true)
        """
    )


def load_nodes(con: duckdb.DuckDBPyConnection) -> None:
    """`nodes` table -- the promoted channel/author items (doc §8.2), pulled out as the graph's node list."""
    con.execute("CREATE OR REPLACE TABLE nodes AS SELECT * FROM items WHERE source_type IN ('channel', 'author')")


def load_all(con: duckdb.DuckDBPyConnection) -> None:
    load_items(con)
    load_edges(con)
    load_observations(con)
    load_nodes(con)


def print_load_summary(con: duckdb.DuckDBPyConnection) -> None:
    print("--- Load summary (Phase A) ---")

    print("Items by source_type:")
    for source_type, count in con.execute(
        "SELECT source_type, count(*) FROM items GROUP BY source_type ORDER BY 2 DESC"
    ).fetchall():
        print(f"  {source_type}: {count}")

    print("Edges by type:")
    rows = con.execute(
        "SELECT edge_type, count(*) FROM edges GROUP BY edge_type ORDER BY 2 DESC"
    ).fetchall()
    if not rows:
        print("  (none)")
    for edge_type, count in rows:
        print(f"  {edge_type}: {count}")

    (n_obs,) = fetchone(con.execute("SELECT count(*) FROM observations"))
    print(f"Observations: {n_obs}")

    (n_nodes,) = fetchone(con.execute("SELECT count(*) FROM nodes"))
    print(f"Distinct nodes: {n_nodes}")

    min_date, max_date = fetchone(
        con.execute(
            "SELECT min(published_at), max(published_at) FROM items WHERE published_at IS NOT NULL"
        )
    )
    print(f"Date range of content: {min_date} to {max_date}")
