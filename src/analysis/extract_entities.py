"""Entity extraction + resolution pass -- CLI entrypoint (docs/analysis_layer_spec.md §4 pass 1).

    uv run python -m analysis.extract_entities --clusters-only --limit 50

Reads items from the processed DuckDB (src/processing/, attached read-only),
extracts entities via the configured EntityExtractor (analysis/entities.py),
resolves them to canonical entities (analysis/resolution.py), and writes
entities/entity_aliases/item_entities/extraction_cache into the analysis
DuckDB (data/analysis/analysis.duckdb).

Only this first pass (entity extraction + resolution) is implemented.
Stance detection, narrative clustering, profile aggregation, and finding
assembly are later passes per the spec and are not built here.

Cost control, two layers deep:
  - Incremental: an item already in item_extraction_status is skipped
    entirely (no re-check, no re-call) on a later run.
  - Content cache: even a first-time item_id skips the API call if its
    text_hash was already extracted under a different item_id (e.g. the
    same channel re-collected across runs -- a pattern this project's
    processing layer has repeatedly found in real data).
--clusters-only (default true) restricts the candidate set to items that
appear in any derived edge (near_duplicate_text / shared_media /
temporal_cocluster) -- the cheap, high-value subset for validating that
resolution actually collapses variant surface forms across genuinely
related content.
"""

from __future__ import annotations

import argparse
import time

import duckdb

from analysis import config, resolution
from analysis.entities import AnthropicEntityExtractor, EntityExtractor, EntityMention

API_CALL_SLEEP_SECONDS = 0.3

_CLUSTER_EDGE_TYPES = ("near_duplicate_text", "shared_media", "temporal_cocluster")


class RunStats:
    def __init__(self) -> None:
        self.items_considered = 0
        self.items_skipped_cached = 0
        self.api_calls = 0
        self.cache_hits = 0
        self.total_mentions = 0
        self.zero_entity_items: list[tuple[str, str]] = []  # (item_id, text snippet)
        self.examples: list[dict] = []  # for the report


def connect(read_only_processed: bool = True) -> duckdb.DuckDBPyConnection:
    config.ANALYSIS_DATA_PATH.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.ANALYSIS_DB_PATH))
    ro = " (READ_ONLY)" if read_only_processed else ""
    con.execute(f"ATTACH '{config.PROCESSED_DB_PATH}' AS processed{ro}")
    return con


def select_candidate_items(con: duckdb.DuckDBPyConnection, clusters_only: bool, limit: int) -> list[dict]:
    """Items with text, not yet processed (item_extraction_status), optionally restricted to coordination clusters."""
    edge_types_sql = ", ".join(f"'{t}'" for t in _CLUSTER_EDGE_TYPES)
    cluster_cte = f"""
        cluster_items AS (
            SELECT src_item_id AS item_id FROM processed.edges
            WHERE origin = 'derived' AND edge_type IN ({edge_types_sql})
            UNION
            SELECT dst_item_id AS item_id FROM processed.edges
            WHERE origin = 'derived' AND edge_type IN ({edge_types_sql}) AND dst_item_id IS NOT NULL
        )
    """
    cluster_join = "JOIN cluster_items c ON i.item_id = c.item_id" if clusters_only else ""
    with_clause = f"WITH {cluster_cte}" if clusters_only else ""

    query = f"""
        {with_clause}
        SELECT i.item_id, i.text, i.text_hash, i.language_detected, i.script, i.source_type
        FROM processed.items i
        {cluster_join}
        LEFT JOIN item_extraction_status s ON i.item_id = s.item_id
        WHERE i.text IS NOT NULL AND i.text_hash IS NOT NULL AND s.item_id IS NULL
        LIMIT ?
    """
    rows = con.execute(query, [limit]).fetchall()
    columns = ["item_id", "text", "text_hash", "language_detected", "script", "source_type"]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def estimate_api_calls(con: duckdb.DuckDBPyConnection, items: list[dict]) -> int:
    """Distinct text_hashes among the candidate batch not already in extraction_cache -- an upper bound (items sharing a hash within the same batch also dedupe against each other)."""
    distinct_hashes = {item["text_hash"] for item in items}
    if not distinct_hashes:
        return 0
    placeholders = ", ".join("?" for _ in distinct_hashes)
    cached = con.execute(
        f"SELECT text_hash FROM extraction_cache WHERE text_hash IN ({placeholders})",
        list(distinct_hashes),
    ).fetchall()
    cached_hashes = {row[0] for row in cached}
    return len(distinct_hashes - cached_hashes)


def process_item(
    con: duckdb.DuckDBPyConnection,
    resolver: resolution.EntityResolver,
    extractor: EntityExtractor,
    item: dict,
    stats: RunStats,
) -> list[tuple[EntityMention, str]]:
    """Extract (via cache or API) + resolve one item's entities. Returns [(mention, entity_id), ...]."""
    cached = resolution.get_cached_extraction(con, item["text_hash"])
    if cached is not None:
        mentions = cached
        stats.cache_hits += 1
    else:
        context = {
            "language_detected": item["language_detected"],
            "script": item["script"],
            "source_type": item["source_type"],
        }
        mentions = extractor.extract(item["text"], context=context)
        resolution.store_extraction_cache(con, item["text_hash"], mentions, model=config.ANTHROPIC_MODEL)
        stats.api_calls += 1
        time.sleep(API_CALL_SLEEP_SECONDS)

    resolved = []
    for mention in mentions:
        entity_id = resolver.resolve(mention)
        resolution.record_item_entity(con, item["item_id"], entity_id, mention.surface_form, mention.confidence)
        resolved.append((mention, entity_id))

    resolution.mark_item_processed(con, item["item_id"], len(mentions))
    stats.total_mentions += len(mentions)
    stats.items_considered += 1

    if not mentions:
        stats.zero_entity_items.append((item["item_id"], (item["text"] or "")[:80]))

    return resolved


def run(clusters_only: bool, limit: int, extractor: EntityExtractor) -> tuple[RunStats, duckdb.DuckDBPyConnection]:
    con = connect()
    resolver = resolution.EntityResolver(con)

    seed_created = resolution.load_seed_entities(resolver, config.SEED_ENTITIES_PATH)
    print(f"[extract_entities] seed entities: {seed_created} newly created (idempotent -- 0 on repeat runs)", flush=True)

    items = select_candidate_items(con, clusters_only, limit)
    print(
        f"[extract_entities] candidate items: {len(items)} "
        f"(clusters_only={clusters_only}, limit={limit})",
        flush=True,
    )

    estimated = estimate_api_calls(con, items)
    print(f"[extract_entities] estimated API calls: {estimated}", flush=True)

    stats = RunStats()
    for item in items:
        resolved = process_item(con, resolver, extractor, item, stats)
        if len(stats.examples) < 6 and resolved:
            stats.examples.append(
                {
                    "item_id": item["item_id"],
                    "text": item["text"],
                    "language_detected": item["language_detected"],
                    "script": item["script"],
                    "resolved": [
                        (m.surface_form, m.canonical_name, eid, m.confidence) for m, eid in resolved
                    ],
                }
            )

    return stats, con


def print_report(stats: RunStats, con: duckdb.DuckDBPyConnection) -> None:
    print("\n--- Entity extraction summary ---")
    print(f"Items processed: {stats.items_considered}")
    print(f"Actual API calls made: {stats.api_calls}")
    print(f"Content-cache hits (skipped API): {stats.cache_hits}")
    print(f"Total entity mentions extracted: {stats.total_mentions}")

    row = con.execute("SELECT count(DISTINCT entity_id) FROM item_entities").fetchone()
    assert row is not None
    print(f"Distinct canonical entities referenced (all-time): {row[0]}")

    multi_alias = con.execute("""
        SELECT e.canonical_name, count(*) AS n_aliases, list(a.surface_form) AS surface_forms
        FROM entities e
        JOIN entity_aliases a ON a.entity_id = e.entity_id
        GROUP BY e.entity_id, e.canonical_name
        HAVING count(*) > 1
        ORDER BY n_aliases DESC
    """).fetchall()
    print(f"\nEntities with >1 resolved surface form (variant collapse working): {len(multi_alias)}")
    for canonical_name, n_aliases, surface_forms in multi_alias[:15]:
        print(f"  {canonical_name}: {n_aliases} surface forms -> {surface_forms}")

    print(f"\nItems where extraction returned zero entities: {len(stats.zero_entity_items)}")
    for item_id, snippet in stats.zero_entity_items:
        print(f"  {item_id}: {snippet!r}")

    print(f"\n--- {len(stats.examples)} example items ---")
    for ex in stats.examples:
        print(f"\nitem_id={ex['item_id']} lang={ex['language_detected']!r} script={ex['script']!r}")
        print(f"  text: {(ex['text'] or '')[:120]!r}")
        for surface_form, canonical_name, entity_id, confidence in ex["resolved"]:
            print(f"    {surface_form!r} -> {canonical_name!r} (entity_id={entity_id[:8]}..., confidence={confidence:.2f})")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entity extraction + resolution (docs/analysis_layer_spec.md §4 pass 1)")
    parser.add_argument("--limit", type=int, default=50, help="Max items to process this run")
    parser.add_argument(
        "--clusters-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict to items in coordination clusters (default: true)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    extractor = AnthropicEntityExtractor(api_key=config.ANTHROPIC_API_KEY, model=config.ANTHROPIC_MODEL)
    stats, con = run(args.clusters_only, args.limit, extractor)
    print_report(stats, con)
    con.close()


if __name__ == "__main__":
    main()
