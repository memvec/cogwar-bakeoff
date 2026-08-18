"""Processing build -- CLI entrypoint.

    uv run python -m processing.build --window-hours 24

Two phases in one pass, both against data/processed/cogwar.duckdb:

  Phase A (load.py):   raw items/edges/observations -> clean DuckDB tables
                        (items, edges, observations, nodes). Full rebuild
                        from raw every run.
  Phase B (derive.py):  shared_media / near_duplicate_text /
                        temporal_cocluster edges, computed from the
                        items/edges just loaded and written back into the
                        edges table with origin='derived'.

Both phases are idempotent: raw data/raw/ is the source of truth, the
DuckDB database is fully derived and disposable -- delete it and re-run
this and everything comes back identical (modulo new raw data collected
since).
"""

from __future__ import annotations

import argparse

import duckdb

from processing import config, derive, load


def build(window_hours: float) -> duckdb.DuckDBPyConnection:
    config.PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.DB_PATH))

    load.load_all(con)
    load.print_load_summary(con)

    derive.derive_all(con, window_hours=window_hours)
    derive.print_derive_summary(con)

    return con


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Processing layer build (docs/collection_schema.md §8)")
    parser.add_argument(
        "--window-hours",
        type=float,
        default=derive.DEFAULT_WINDOW_HOURS,
        help="temporal_cocluster time window in hours (default 24)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    con = build(args.window_hours)
    con.close()
    print(f"\nDatabase: {config.DB_PATH}")


if __name__ == "__main__":
    main()
