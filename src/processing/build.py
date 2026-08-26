"""Processing build -- CLI entrypoint.

    uv run python -m processing.build --window-hours 24 --hash-group-cap 200

Three phases in one pass, all against data/processed/cogwar.duckdb:

  Phase A (load.py):     raw items/edges/observations -> clean DuckDB tables
                          (items, edges, observations, nodes). Full rebuild
                          from raw every run.
  Phase B (derive.py):   shared_media / near_duplicate_text /
                          temporal_cocluster edges, computed from the
                          items/edges just loaded and written back into the
                          edges table with origin='derived'. Any hash
                          shared by more than --hash-group-cap items is
                          excluded from pairwise edge generation (reported,
                          not silently dropped) -- see derive.py's
                          docstring for why that guard exists.
  Phase C (clusters.py): connected-components report over the derived
                          edges -- coordination clusters, their size
                          distribution, and the top clusters by distinct-
                          author count. Read-only: writes nothing back to
                          the database.

All three phases are idempotent: raw data/raw/ is the source of truth, the
DuckDB database is fully derived and disposable -- delete it and re-run
this and everything comes back identical (modulo new raw data collected
since).
"""

from __future__ import annotations

import argparse

import duckdb

from processing import clusters, config, derive, load


def build(window_hours: float, hash_group_cap: int = derive.DEFAULT_HASH_GROUP_CAP) -> duckdb.DuckDBPyConnection:
    config.PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.DB_PATH))

    print("[build] Phase A: loading raw items/edges/observations ...", flush=True)
    load.load_all(con)
    load.print_load_summary(con)

    print("\n[build] Phase B: deriving coordination edges ...", flush=True)
    derive.derive_all(con, window_hours=window_hours, hash_group_cap=hash_group_cap)
    derive.print_derive_summary(con, hash_group_cap=hash_group_cap)

    print("\n[build] Phase C: coordination-cluster report ...", flush=True)
    clusters.print_cluster_report(con, hash_group_cap=hash_group_cap)

    return con


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Processing layer build (docs/collection_schema.md §8)")
    parser.add_argument(
        "--window-hours",
        type=float,
        default=derive.DEFAULT_WINDOW_HOURS,
        help="temporal_cocluster time window in hours (default 24)",
    )
    parser.add_argument(
        "--hash-group-cap",
        type=int,
        default=derive.DEFAULT_HASH_GROUP_CAP,
        help="Skip pairwise edge generation for any hash shared by more than this many items (default 200)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    con = build(args.window_hours, args.hash_group_cap)
    con.close()
    print(f"\nDatabase: {config.DB_PATH}")


if __name__ == "__main__":
    main()
