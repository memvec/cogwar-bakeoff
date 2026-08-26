"""Read-only DuckDB connection helper.

The viz never writes -- every connection this module hands out is opened
read-only on both databases. A naive "open a fresh `duckdb.connect(path,
read_only=True)` + ATTACH per request" design looks safe (DuckDB's own docs
describe concurrent read-only connections to the same file as supported)
but isn't, for one specific reason: DuckDB dedupes same-path connections
within a single process onto ONE shared underlying database instance, so
ATTACHing an alias from one "connection" is visible to every other
"connection" to that same path in this process -- confirmed for real: 10
threads each doing `duckdb.connect(path); con.execute("ATTACH ... AS
processed")` concurrently, 9 fail with "database with name 'processed'
already exists" because they're all racing to attach the alias onto the
one shared instance, not onto independent instances of their own.

The correct pattern (DuckDB's documented concurrency model for a single
process: https://duckdb.org/docs/connect/concurrency) is one base
connection with `processed` attached ONCE at import time, and a fresh
`.cursor()` per request -- cursors are independent, interruptible, and
safe for concurrent use from multiple threads while sharing the parent's
attached catalog. Confirmed for real: 20 threads calling `.cursor()`
concurrently against a base connection with `processed` already attached,
zero errors, every cursor sees the same attached catalog.
"""

from __future__ import annotations

from collections.abc import Iterator

import duckdb

from serve.config import ANALYSIS_DB_PATH, PROCESSED_DB_PATH

_base_connection = duckdb.connect(str(ANALYSIS_DB_PATH), read_only=True)
_base_connection.execute(f"ATTACH '{PROCESSED_DB_PATH}' AS processed (READ_ONLY)")


def get_connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """FastAPI dependency: yields a request-scoped cursor on the shared base
    connection (with `processed` already attached), closed after the
    request completes."""
    cursor = _base_connection.cursor()
    try:
        yield cursor
    finally:
        cursor.close()
