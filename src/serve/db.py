"""Read-only DuckDB connection helper.

The viz never writes -- every connection this module hands out is opened
read-only on both databases. DuckDB explicitly supports multiple
concurrent read-only connections to the same file from separate
`duckdb.connect(..., read_only=True)` calls (its own concurrency docs name
this as the supported pattern for concurrent readers), which is exactly
what a request-scoped connection under FastAPI needs -- so `get_connection`
opens a fresh one per request rather than sharing a single connection
object across concurrent requests (a single DuckDBPyConnection is not
documented as safe for concurrent use from multiple threads/requests at
once). Open+attach cost is small next to a request's query cost, so this
trades a little per-request overhead for not having to reason about
connection-sharing safety at all.
"""

from __future__ import annotations

from collections.abc import Iterator

import duckdb

from serve.config import ANALYSIS_DB_PATH, PROCESSED_DB_PATH


def get_connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """FastAPI dependency: yields a request-scoped read-only connection with
    `processed` attached read-only alongside the analysis DB, closed after
    the request completes."""
    con = duckdb.connect(str(ANALYSIS_DB_PATH), read_only=True)
    try:
        con.execute(f"ATTACH '{PROCESSED_DB_PATH}' AS processed (READ_ONLY)")
        yield con
    finally:
        con.close()
