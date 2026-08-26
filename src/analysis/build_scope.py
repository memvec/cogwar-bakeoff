"""Analysis scope pass -- CLI entrypoint.

    uv run python -m analysis.build_scope

Local/free, no model calls: intersects coordination-cluster membership
(processing/derive.py's derived edges, restricted to clusters with
>= --min-authors distinct authors) with multilingual topic-keyword
relevance (configs/topic_keywords.json) to define exactly which items the
next (expensive, model-based) entity/stance passes should run over. See
analysis/scope.py's module docstring for the two filters' definitions.

Persists the resulting item_id set into analysis.duckdb's `analysis_scope`
table so later passes can target exactly this set via a plain join.
"""

from __future__ import annotations

import argparse
import random

import duckdb

from analysis import config, scope

DEFAULT_MIN_AUTHORS = 6
COMPARISON_THRESHOLDS = (3, 11)
SAMPLE_SIZE = 10


def connect() -> duckdb.DuckDBPyConnection:
    config.ANALYSIS_DATA_PATH.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.ANALYSIS_DB_PATH))
    con.execute(f"ATTACH '{config.PROCESSED_DB_PATH}' AS processed (READ_ONLY)")
    scope.init_schema(con)
    return con


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analysis scope pass (coordination x topic-keyword intersection)")
    parser.add_argument(
        "--min-authors",
        type=int,
        default=DEFAULT_MIN_AUTHORS,
        help="Minimum distinct authors per coordination cluster to include (default 6)",
    )
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE, help="Sample items to print for eyeballing")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the sample draw (reproducible by default)")
    parser.add_argument(
        "--on-topic-only",
        action="store_true",
        help=(
            "Skip the coordination-cluster filter entirely -- persist the full topic-keyword-matched "
            "set (corpus-wide) as the scope, not its intersection with coordination clusters"
        ),
    )
    return parser.parse_args(argv)


def print_source_type_breakdown(con: duckdb.DuckDBPyConnection, item_ids: set[str]) -> None:
    if not item_ids:
        print("  (empty)")
        return
    rows = con.execute(
        """
        SELECT source_type, count(*)
        FROM processed.items
        WHERE item_id IN (SELECT UNNEST($ids))
        GROUP BY source_type
        ORDER BY 2 DESC
        """,
        {"ids": list(item_ids)},
    ).fetchall()
    for source_type, count in rows:
        print(f"  {source_type}: {count}")


def print_sample(
    con: duckdb.DuckDBPyConnection,
    item_ids: set[str],
    keyword_pairs: list[tuple[str, str]],
    sample_size: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    sample_ids = rng.sample(sorted(item_ids), min(sample_size, len(item_ids)))
    rows = con.execute(
        """
        SELECT item_id, source_type, author_display_name, author_native_id,
               concat_ws(' ', text, source_specific -> 'transcript' ->> 'text') AS combined_text
        FROM processed.items
        WHERE item_id IN (SELECT UNNEST($ids))
        """,
        {"ids": sample_ids},
    ).fetchall()
    for item_id, source_type, author_display_name, author_native_id, combined_text in rows:
        matches = scope.matched_keywords_for_text(combined_text, keyword_pairs)
        matched_str = ", ".join(sorted({kw for _, kw in matches})) or "(none -- unexpected)"
        author = author_display_name or author_native_id or "(unknown author)"
        snippet = (combined_text or "").replace("\n", " ").strip()[:160]
        print(f"  [{source_type}] {author!r} (item {item_id[:8]}...): matched={matched_str}")
        print(f"      {snippet!r}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    con = connect()

    print("[build_scope] loading topic keywords ...", flush=True)
    keyword_pairs = scope.load_topic_keywords()
    n_topics = len({t for t, _ in keyword_pairs})
    print(f"[build_scope] {len(keyword_pairs)} keywords across {n_topics} topics", flush=True)

    print("[build_scope] scanning corpus for on-topic items (this is the long step) ...", flush=True)
    on_topic_ids = scope.compute_on_topic_item_ids(con, keyword_pairs)
    print(f"[build_scope] on-topic set: {len(on_topic_ids)} items", flush=True)

    if args.on_topic_only:
        print("\n--- Analysis scope (on-topic only, no coordination filter) ---")
        print(f"On-topic set size: {len(on_topic_ids)} items")
        print("\nBy source_type:")
        print_source_type_breakdown(con, on_topic_ids)
        print(f"\nSample of {min(args.sample_size, len(on_topic_ids))} items:")
        print_sample(con, on_topic_ids, keyword_pairs, args.sample_size, args.seed)

        print(f"\n[build_scope] persisting scope ({len(on_topic_ids)} items) to analysis_scope ...", flush=True)
        scope.persist_scope(con, on_topic_ids, item_to_cluster={}, cluster_n_authors={}, keyword_pairs=keyword_pairs, min_authors=0)
        (n_persisted,) = con.execute("SELECT count(*) FROM analysis_scope").fetchone()  # type: ignore[misc]
        print(f"[build_scope] analysis_scope: {n_persisted} rows persisted", flush=True)
        con.close()
        return

    print("[build_scope] computing coordination clusters (shared_media + near_duplicate_text) ...", flush=True)
    clusters = scope.compute_coordination_clusters(con)
    author_by_item = scope.load_author_by_item(con, clusters)
    item_to_cluster = scope.item_to_cluster_map(clusters)
    print(f"[build_scope] {len(clusters)} total clusters, {len(item_to_cluster)} items in any cluster", flush=True)

    print("\n--- Threshold comparison ---")
    thresholds = sorted({*COMPARISON_THRESHOLDS, args.min_authors})
    coord_by_threshold: dict[int, tuple[set[str], int]] = {}
    for min_authors in thresholds:
        coord_ids, n_clusters = scope.coordination_set_for_threshold(clusters, author_by_item, min_authors)
        coord_by_threshold[min_authors] = (coord_ids, n_clusters)
        intersection = coord_ids & on_topic_ids
        marker = "  <-- committed threshold" if min_authors == args.min_authors else ""
        print(
            f"  >= {min_authors} authors: coordination_set={len(coord_ids)} items across {n_clusters} clusters, "
            f"intersection={len(intersection)}{marker}"
        )

    coord_ids, n_clusters = coord_by_threshold[args.min_authors]
    intersection_ids = coord_ids & on_topic_ids
    intersection_clusters = {item_to_cluster[i] for i in intersection_ids if i in item_to_cluster}
    intersection_authors = {author_by_item.get(i) for i in intersection_ids} - {None}

    print(f"\n--- Analysis scope (>= {args.min_authors} authors AND on-topic) ---")
    print(f"Coordination set size (>= {args.min_authors} authors): {len(coord_ids)} items, {n_clusters} clusters")
    print(f"On-topic set size (corpus-wide): {len(on_topic_ids)} items")
    print(f"Intersection size: {len(intersection_ids)} items")
    print(f"Intersection spans: {len(intersection_clusters)} coordination clusters, {len(intersection_authors)} distinct authors")

    print("\nIntersection by source_type:")
    print_source_type_breakdown(con, intersection_ids)

    print(f"\nSample of {min(args.sample_size, len(intersection_ids))} items from the intersection:")
    print_sample(con, intersection_ids, keyword_pairs, args.sample_size, args.seed)

    cluster_n_authors = {
        cluster_id: len({author_by_item.get(m) for m in members} - {None}) for cluster_id, members in clusters.items()
    }
    print(f"\n[build_scope] persisting scope ({len(intersection_ids)} items) to analysis_scope ...", flush=True)
    scope.persist_scope(con, intersection_ids, item_to_cluster, cluster_n_authors, keyword_pairs, args.min_authors)
    (n_persisted,) = con.execute("SELECT count(*) FROM analysis_scope").fetchone()  # type: ignore[misc]
    print(f"[build_scope] analysis_scope: {n_persisted} rows persisted", flush=True)

    con.close()


if __name__ == "__main__":
    main()
