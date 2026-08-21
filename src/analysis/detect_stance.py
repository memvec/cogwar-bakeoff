"""Stance detection pass -- CLI entrypoint (docs/analysis_layer_spec.md §4 pass 2).

    uv run python -m analysis.detect_stance --clusters-only --limit 50

Reads items + their resolved entities (item_entities, from the entity
extraction pass) from the analysis DuckDB, asks the configured
StanceDetector (analysis/stance.py) what stance the AUTHOR takes toward
EACH entity an item references, and writes entity_stance_edges
(analysis/stance_storage.py). Stance only -- narrative clustering, profile
aggregation, and finding assembly are later passes per the spec and are not
built here.

One API call per ITEM, not per (item, entity) pair: all of an item's
entities that still need a stance are batched into a single prompt (same
economy as extraction). Cost control, two layers deep:
  - Content cache: an (item, entity) pair whose text_hash + entity_id was
    already scored -- even under a different item_id (e.g. a repost) --
    is served from stance_cache with zero API calls.
  - Incremental: an item where every one of its entities already has an
    entity_stance_edges row is skipped entirely, no cache lookups needed.
    A rerun only ever processes the (item, entity) pairs still missing a
    cached/persisted result -- if an item gained a new entity since its
    last run (e.g. a merge), only that entity's pair is processed.
--clusters-only (default true) restricts the candidate set to items that
appear in any derived edge (near_duplicate_text / shared_media /
temporal_cocluster), same coordination-cluster-first discipline as the
entity extraction pass.

Neutral/noise threshold (spec §5.5): most entity mentions are incidental
with no real stance. Default behavior emits the edge regardless, with
polarity=neutral and low strength, so downstream aggregation can filter --
--skip-neutral-edges opts into dropping those edges at write time instead.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import duckdb

from analysis import config, resolution, stance_storage
from analysis.stance import (
    AnthropicStanceDetector,
    EntityRef,
    StanceDetector,
    StanceResult,
)

API_CALL_SLEEP_SECONDS = 0.3

_CLUSTER_EDGE_TYPES = ("near_duplicate_text", "shared_media", "temporal_cocluster")


class RunStats:
    def __init__(self) -> None:
        self.items_considered = 0
        self.items_fully_cached = 0  # item needed 0 API calls -- every pending entity was in stance_cache
        self.api_calls = 0
        self.pairs_from_cache = 0
        self.pairs_from_api = 0
        self.edges_written = 0
        self.skipped_neutral = 0
        self.polarity_counts: Counter = Counter()  # over all (item, entity) pairs decided, incl. skipped
        self.detection_gaps: list[tuple[str, str, str]] = []  # (item_id, entity_id, canonical_name) model omitted
        self.empty_response_items: list[tuple[str, str]] = []  # (item_id, text snippet) -- API call returned nothing at all


def connect(read_only_processed: bool = True) -> duckdb.DuckDBPyConnection:
    config.ANALYSIS_DATA_PATH.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.ANALYSIS_DB_PATH))
    ro = " (READ_ONLY)" if read_only_processed else ""
    con.execute(f"ATTACH '{config.PROCESSED_DB_PATH}' AS processed{ro}")
    # entities/item_entities are the entity-extraction pass's tables -- this
    # pass is a consumer of them, not their owner, but resolution.SCHEMA_SQL
    # is idempotent (CREATE TABLE IF NOT EXISTS) so it's safe to ensure they
    # exist even if stance detection is invoked before extraction has run.
    con.execute(resolution.SCHEMA_SQL)
    stance_storage.init_schema(con)
    return con


def select_candidate_items(con: duckdb.DuckDBPyConnection, clusters_only: bool, limit: int) -> list[dict]:
    """Items with resolved entities where at least one (item, entity) pair
    still lacks an entity_stance_edges row, optionally restricted to
    coordination clusters."""
    edge_types_sql = ", ".join(f"'{t}'" for t in _CLUSTER_EDGE_TYPES)
    cluster_cte = ""
    cluster_join = ""
    if clusters_only:
        cluster_cte = f""",
        cluster_items AS (
            SELECT src_item_id AS item_id FROM processed.edges
            WHERE origin = 'derived' AND edge_type IN ({edge_types_sql})
            UNION
            SELECT dst_item_id AS item_id FROM processed.edges
            WHERE origin = 'derived' AND edge_type IN ({edge_types_sql}) AND dst_item_id IS NOT NULL
        )"""
        cluster_join = "JOIN cluster_items c ON iec.item_id = c.item_id"

    query = f"""
        WITH item_entity_counts AS (
            SELECT item_id, count(DISTINCT entity_id) AS n_entities
            FROM item_entities GROUP BY item_id
        ),
        item_edge_counts AS (
            SELECT item_id, count(DISTINCT entity_id) AS n_edges
            FROM entity_stance_edges GROUP BY item_id
        ){cluster_cte}
        SELECT i.item_id, i.text, i.text_hash, i.language_detected, i.script, i.source_type
        FROM item_entity_counts iec
        LEFT JOIN item_edge_counts iee ON iec.item_id = iee.item_id
        JOIN processed.items i ON i.item_id = iec.item_id
        {cluster_join}
        WHERE i.text IS NOT NULL AND i.text_hash IS NOT NULL
          AND (iee.n_edges IS NULL OR iee.n_edges < iec.n_entities)
        LIMIT ?
    """
    rows = con.execute(query, [limit]).fetchall()
    columns = ["item_id", "text", "text_hash", "language_detected", "script", "source_type"]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def estimate_api_calls(con: duckdb.DuckDBPyConnection, items: list[dict]) -> int:
    """Count of candidate items that will require at least one API call --
    i.e. have at least one pending (item, entity) pair not already in
    stance_cache. An upper bound: items sharing a text_hash within the same
    batch also dedupe against each other as the run proceeds."""
    estimated = 0
    for item in items:
        entities = stance_storage.get_item_entities(con, item["item_id"])
        existing = stance_storage.get_existing_edge_entity_ids(con, item["item_id"])
        pending = [e for e in entities if e.entity_id not in existing]
        if not pending:
            continue
        uncached = [e for e in pending if stance_storage.get_cached_stance(con, item["text_hash"], e.entity_id) is None]
        if uncached:
            estimated += 1
    return estimated


def process_item(
    con: duckdb.DuckDBPyConnection,
    detector: StanceDetector,
    item: dict,
    stats: RunStats,
    skip_neutral: bool,
) -> list[tuple[EntityRef, StanceResult]]:
    """Score (via cache or one batched API call) + persist one item's
    pending (item, entity) stance pairs. Returns [(entity, result), ...]
    for every pending entity, regardless of skip_neutral (callers decide
    what to do with that for reporting)."""
    entities = stance_storage.get_item_entities(con, item["item_id"])
    existing = stance_storage.get_existing_edge_entity_ids(con, item["item_id"])
    pending = [e for e in entities if e.entity_id not in existing]
    if not pending:
        return []

    cached_results: dict[str, StanceResult] = {}
    uncached_entities: list[EntityRef] = []
    for e in pending:
        cached = stance_storage.get_cached_stance(con, item["text_hash"], e.entity_id)
        if cached is not None:
            cached_results[e.entity_id] = cached
        else:
            uncached_entities.append(e)
    stats.pairs_from_cache += len(cached_results)

    new_results: dict[str, StanceResult] = {}
    if uncached_entities:
        context = {
            "language_detected": item["language_detected"],
            "script": item["script"],
            "source_type": item["source_type"],
        }
        detected = detector.detect(item["text"], uncached_entities, context=context)
        stats.api_calls += 1
        time.sleep(API_CALL_SLEEP_SECONDS)

        if not detected:
            stats.empty_response_items.append((item["item_id"], (item["text"] or "")[:80]))

        by_id = {r.entity_id: r for r in detected}
        for e in uncached_entities:
            result = by_id.get(e.entity_id)
            if result is None:
                # Model omitted this entity from its response -- fall back to
                # a zero-confidence neutral edge rather than dropping the
                # pair silently, and flag it as a detection gap in the report.
                result = StanceResult(entity_id=e.entity_id, polarity="neutral", strength=0.0, confidence=0.0)
                stats.detection_gaps.append((item["item_id"], e.entity_id, e.canonical_name))
            stance_storage.store_stance_cache(con, item["text_hash"], result, model=config.ANTHROPIC_MODEL)
            new_results[e.entity_id] = result
        stats.pairs_from_api += len(uncached_entities)
    else:
        stats.items_fully_cached += 1

    all_results = {**cached_results, **new_results}
    resolved = []
    for e in pending:
        result = all_results[e.entity_id]
        stats.polarity_counts[result.polarity] += 1
        if skip_neutral and result.polarity == "neutral":
            stats.skipped_neutral += 1
        else:
            stance_storage.record_stance_edge(con, item["item_id"], result, model=config.ANTHROPIC_MODEL)
            stats.edges_written += 1
        resolved.append((e, result))

    stats.items_considered += 1
    return resolved


def run(
    clusters_only: bool, limit: int, skip_neutral: bool, detector: StanceDetector
) -> tuple[RunStats, duckdb.DuckDBPyConnection]:
    con = connect()

    items = select_candidate_items(con, clusters_only, limit)
    print(
        f"[detect_stance] candidate items: {len(items)} (clusters_only={clusters_only}, limit={limit})",
        flush=True,
    )

    estimated = estimate_api_calls(con, items)
    print(f"[detect_stance] estimated API calls: {estimated}", flush=True)

    stats = RunStats()
    for item in items:
        process_item(con, detector, item, stats, skip_neutral)

    return stats, con


def print_examples(con: duckdb.DuckDBPyConnection) -> None:
    def fetch_item_examples(item_ids: list[str]) -> None:
        for item_id in item_ids:
            row = con.execute(
                "SELECT text, language_detected, script FROM processed.items WHERE item_id = ?", [item_id]
            ).fetchone()
            if row is None:
                continue
            text, lang, script = row
            print(f"\nitem_id={item_id} lang={lang!r} script={script!r}")
            print(f"  text: {(text or '')[:160]!r}")
            edges = con.execute(
                """
                SELECT e.canonical_name, se.polarity, se.strength, se.confidence
                FROM entity_stance_edges se
                JOIN entities e ON e.entity_id = se.entity_id
                WHERE se.item_id = ?
                ORDER BY se.polarity, e.canonical_name
                """,
                [item_id],
            ).fetchall()
            for canonical_name, polarity, strength, confidence in edges:
                print(f"    {canonical_name!r}: {polarity} (strength={strength:.2f}, confidence={confidence:.2f})")

    print("\n--- Multi-entity opposite-stance examples (positive toward one entity, negative toward another) ---")
    mixed_item_ids = [
        r[0]
        for r in con.execute(
            """
            SELECT item_id FROM entity_stance_edges WHERE polarity = 'positive'
            INTERSECT
            SELECT item_id FROM entity_stance_edges WHERE polarity = 'negative'
            LIMIT 3
            """
        ).fetchall()
    ]
    if mixed_item_ids:
        fetch_item_examples(mixed_item_ids)
    else:
        print("  (none found in this batch)")

    print("\n--- Hindi/Urdu examples ---")
    hi_ur_item_ids = [
        r[0]
        for r in con.execute(
            """
            SELECT DISTINCT se.item_id
            FROM entity_stance_edges se
            JOIN processed.items i ON i.item_id = se.item_id
            WHERE i.language_detected IN ('hi', 'ur') OR i.script IN ('devanagari', 'arabic')
            LIMIT 3
            """
        ).fetchall()
    ]
    if hi_ur_item_ids:
        fetch_item_examples(hi_ur_item_ids)
    else:
        print("  (none found in this batch)")

    print("\n--- Additional example items ---")
    other_item_ids = [
        r[0]
        for r in con.execute(
            f"""
            SELECT DISTINCT item_id FROM entity_stance_edges
            WHERE item_id NOT IN ({", ".join("?" for _ in mixed_item_ids + hi_ur_item_ids) or "NULL"})
            LIMIT 2
            """,
            mixed_item_ids + hi_ur_item_ids,
        ).fetchall()
    ]
    if other_item_ids:
        fetch_item_examples(other_item_ids)


def print_report(stats: RunStats, con: duckdb.DuckDBPyConnection) -> None:
    print("\n--- Stance detection summary ---")
    print(f"Items processed: {stats.items_considered}")
    print(f"Actual API calls made: {stats.api_calls}")
    print(f"Items fully served from cache (0 API calls): {stats.items_fully_cached}")
    print(f"(item, entity) pairs from cache: {stats.pairs_from_cache}")
    print(f"(item, entity) pairs from API: {stats.pairs_from_api}")
    print(f"Edges written: {stats.edges_written}")
    print(f"Edges skipped (--skip-neutral-edges): {stats.skipped_neutral}")

    print("\nPolarity breakdown (all pairs decided, including any skipped):")
    for polarity in ("positive", "negative", "neutral"):
        print(f"  {polarity}: {stats.polarity_counts.get(polarity, 0)}")

    row = con.execute("SELECT count(*) FROM entity_stance_edges").fetchone()
    assert row is not None
    print(f"\nTotal entity_stance_edges (all-time): {row[0]}")

    print(f"\nDetection gaps (model omitted an entity from its response, filled with neutral/0-confidence): {len(stats.detection_gaps)}")
    for item_id, entity_id, canonical_name in stats.detection_gaps:
        print(f"  item={item_id} entity={canonical_name!r} ({entity_id[:8]}...)")

    print(f"\nItems where the API call returned NOTHING (stance detection failed): {len(stats.empty_response_items)}")
    for item_id, snippet in stats.empty_response_items:
        print(f"  {item_id}: {snippet!r}")

    print_examples(con)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stance detection (docs/analysis_layer_spec.md §4 pass 2)")
    parser.add_argument("--limit", type=int, default=50, help="Max items to process this run")
    parser.add_argument(
        "--clusters-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict to items in coordination clusters (default: true)",
    )
    parser.add_argument(
        "--skip-neutral-edges",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop polarity=neutral edges at write time instead of persisting them (default: false, keep everything)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    detector = AnthropicStanceDetector(api_key=config.ANTHROPIC_API_KEY, model=config.ANTHROPIC_MODEL)
    stats, con = run(args.clusters_only, args.limit, args.skip_neutral_edges, detector)
    print_report(stats, con)
    con.close()


if __name__ == "__main__":
    main()
