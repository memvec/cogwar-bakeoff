"""Entity extraction + resolution pass -- CLI entrypoint (docs/analysis_layer_spec.md §4 pass 1).

    uv run python -m analysis.extract_entities --provider gemini --clusters-only
    uv run python -m analysis.extract_entities --provider gemini --no-clusters-only --max-cost 50

Reads items from the processed DuckDB (src/processing/, attached read-only),
extracts entities via the configured EntityExtractor (analysis/entities.py),
resolves them to canonical entities (analysis/resolution.py), and writes
entities/entity_aliases/item_entities/extraction_cache into the analysis
DuckDB (data/analysis/analysis.duckdb).

Only this first pass (entity extraction + resolution) is implemented.
Stance detection, narrative clustering, profile aggregation, and finding
assembly are later passes per the spec and are not built here.

Cost control, several layers deep:
  - Incremental: an item already in item_extraction_status is skipped
    entirely (no re-check, no re-call) on a later run -- this is what makes
    a provider switch (Anthropic -> Gemini) additive rather than redundant:
    items an earlier Anthropic run already covered stay as Anthropic-labeled
    rows (extraction_cache.model records which), and a Gemini run only ever
    processes whatever's still pending, never re-spends on what's done.
  - Content cache: even a first-time item_id skips the API call if its
    text_hash was already extracted under a different item_id (e.g. the
    same channel re-collected across runs -- a pattern this project's
    processing layer has repeatedly found in real data).
  - --max-cost: stop cleanly (no partial/corrupt state -- every item's
    writes are already committed as they happen) once running spend would
    exceed the cap. Rerunning continues via the two caches above.
--clusters-only (default true) restricts the candidate set to items that
appear in any derived edge (near_duplicate_text / shared_media /
temporal_cocluster). With --no-clusters-only, ALL items are candidates, but
cluster-membership still determines priority order (cluster items first) --
so a full-corpus run that gets interrupted or cost-capped partway through
has still covered the highest-value data first.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb

from analysis import config, resolution
from analysis.cost import CostTracker
from analysis.entities import (
    EntityExtractor,
    EntityMention,
    build_extractor_pool,
)
from analysis.gemini_retry import (
    GeminiBillingExhaustedError,
    GeminiDailyQuotaExhaustedError,
    GeminiFreeTierError,
)

# Pause between BATCHES (not per-call) -- courtesy pacing once concurrency
# is already doing the real throughput work; retry/backoff (gemini_retry.py)
# handles genuine rate-limit pushback.
BATCH_SLEEP_SECONDS = 0.2
PROGRESS_EVERY = 25
DEFAULT_CONCURRENCY = 8

_CLUSTER_EDGE_TYPES = ("near_duplicate_text", "shared_media", "temporal_cocluster")


class RunStats:
    def __init__(self) -> None:
        self.items_considered = 0
        self.api_calls = 0
        self.cache_hits = 0
        self.total_mentions = 0
        self.zero_entity_items: list[tuple[str, str]] = []  # (item_id, text snippet)
        self.failed_items: list[tuple[str, str]] = []  # (item_id, error message) -- retries exhausted, skipped, will retry next run
        self.examples: list[dict] = []  # for the report
        self.stopped_early_reason: str | None = None


def connect(read_only_processed: bool = True) -> duckdb.DuckDBPyConnection:
    config.ANALYSIS_DATA_PATH.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.ANALYSIS_DB_PATH))
    ro = " (READ_ONLY)" if read_only_processed else ""
    con.execute(f"ATTACH '{config.PROCESSED_DB_PATH}' AS processed{ro}")
    return con


def _candidate_query(clusters_only: bool) -> tuple[str, str, str]:
    """Returns (with_clause, join_clause, priority_expr) -- shared between
    select_candidate_items (needs LIMIT) and count_remaining_candidates
    (needs none), so the two can never drift on what counts as a candidate.
    """
    edge_types_sql = ", ".join(f"'{t}'" for t in _CLUSTER_EDGE_TYPES)
    with_clause = f"""
        WITH cluster_items AS (
            SELECT src_item_id AS item_id FROM processed.edges
            WHERE origin = 'derived' AND edge_type IN ({edge_types_sql})
            UNION
            SELECT dst_item_id AS item_id FROM processed.edges
            WHERE origin = 'derived' AND edge_type IN ({edge_types_sql}) AND dst_item_id IS NOT NULL
        )
    """
    if clusters_only:
        # Old exclusive-filter behavior, unchanged: only cluster items are
        # candidates at all (every candidate is priority 0).
        join_clause = "JOIN cluster_items c ON i.item_id = c.item_id"
        priority_expr = "(1)"  # constant true-ish expr, NOT a bare integer literal (DuckDB treats a bare int in ORDER BY as a positional column reference)
    else:
        # Full corpus is candidate, but cluster items still sort first --
        # "process in priority order: coordination-cluster items first,
        # then the rest" (see module docstring) -- so a run that stops
        # partway (cost cap, crash, interrupt) has covered the highest-value
        # data first regardless of where it stops.
        join_clause = "LEFT JOIN cluster_items c ON i.item_id = c.item_id"
        priority_expr = "CASE WHEN c.item_id IS NOT NULL THEN 0 ELSE 1 END"
    return with_clause, join_clause, priority_expr


def select_candidate_items(con: duckdb.DuckDBPyConnection, clusters_only: bool, limit: int | None) -> list[dict]:
    """Items with text, not yet processed (item_extraction_status), ordered
    cluster-items-first (see _candidate_query). `limit=None` means no cap --
    return everything pending."""
    with_clause, join_clause, priority_expr = _candidate_query(clusters_only)
    limit_clause = "LIMIT ?" if limit is not None else ""
    query = f"""
        {with_clause}
        SELECT i.item_id, i.text, i.text_hash, i.language_detected, i.script, i.source_type
        FROM processed.items i
        {join_clause}
        LEFT JOIN item_extraction_status s ON i.item_id = s.item_id
        WHERE i.text IS NOT NULL AND i.text_hash IS NOT NULL AND s.item_id IS NULL
        ORDER BY {priority_expr}, i.item_id
        {limit_clause}
    """
    params = [limit] if limit is not None else []
    rows = con.execute(query, params).fetchall()
    columns = ["item_id", "text", "text_hash", "language_detected", "script", "source_type"]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def count_remaining_candidates(con: duckdb.DuckDBPyConnection, clusters_only: bool) -> int:
    """Total pending candidates with no LIMIT -- the denominator for
    progress printing, computed once at the start of a run."""
    with_clause, join_clause, _ = _candidate_query(clusters_only)
    query = f"""
        {with_clause}
        SELECT count(*)
        FROM processed.items i
        {join_clause}
        LEFT JOIN item_extraction_status s ON i.item_id = s.item_id
        WHERE i.text IS NOT NULL AND i.text_hash IS NOT NULL AND s.item_id IS NULL
    """
    row = con.execute(query).fetchone()
    assert row is not None
    return row[0]


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


def _finalize_item(
    con: duckdb.DuckDBPyConnection,
    resolver: resolution.EntityResolver,
    item: dict,
    mentions: list[EntityMention],
    stats: RunStats,
) -> list[tuple[EntityMention, str]]:
    """Resolve + persist one item's already-obtained mentions (from cache or
    a fresh API call -- this function doesn't care which). Always runs on
    the main thread/connection, never inside a worker."""
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

    if len(stats.examples) < 6 and resolved:
        stats.examples.append(
            {
                "item_id": item["item_id"],
                "text": item["text"],
                "language_detected": item["language_detected"],
                "script": item["script"],
                "resolved": [(m.surface_form, m.canonical_name, eid, m.confidence) for m, eid in resolved],
            }
        )
    return resolved


def run(
    clusters_only: bool,
    limit: int | None,
    extractors: list[EntityExtractor],
    model: str,
    cost_tracker: CostTracker,
) -> tuple[RunStats, duckdb.DuckDBPyConnection, CostTracker]:
    """`extractors` is a pool of `concurrency` independent instances (see
    entities.build_extractor_pool) -- within one batch, slot i always calls
    extractors[i], so no two threads ever touch the same instance at once.
    `cost_tracker` is always caller-constructed (main() or an orchestrator
    like run_full_pipeline.py) -- that's where the provider is already known,
    so there's no need to guess it back out of `model` here.
    """
    con = connect()
    resolver = resolution.EntityResolver(con)
    concurrency = len(extractors)

    seed_created = resolution.load_seed_entities(resolver, config.SEED_ENTITIES_PATH)
    print(f"[extract_entities] seed entities: {seed_created} newly created (idempotent -- 0 on repeat runs)", flush=True)

    total_remaining = count_remaining_candidates(con, clusters_only)
    items = select_candidate_items(con, clusters_only, limit)
    print(
        f"[extract_entities] candidate items this invocation: {len(items)} "
        f"(clusters_only={clusters_only}, limit={limit}); {total_remaining} total pending; concurrency={concurrency}",
        flush=True,
    )

    estimated = estimate_api_calls(con, items)
    print(f"[extract_entities] estimated API calls: {estimated}", flush=True)

    stats = RunStats()
    last_progress_print = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for batch_start in range(0, len(items), concurrency):
            if cost_tracker.would_exceed():
                stats.stopped_early_reason = f"cost cap (${cost_tracker.max_cost:.2f}) reached"
                print(f"\n[extract_entities] STOPPING: {stats.stopped_early_reason}. Rerun to continue -- already-processed items are cached.", flush=True)
                break

            batch = items[batch_start : batch_start + concurrency]

            # Cache lookups are cheap sequential DB reads -- split the batch
            # into "already cached" (finalized immediately) vs "needs an
            # API call" (dispatched to the thread pool below).
            to_call: list[dict] = []
            for item in batch:
                cached = resolution.get_cached_extraction(con, item["text_hash"])
                if cached is not None:
                    stats.cache_hits += 1
                    _finalize_item(con, resolver, item, cached, stats)
                else:
                    to_call.append(item)

            if to_call:
                futures = {}
                for slot, item in enumerate(to_call):
                    extractor = extractors[slot]  # slot < concurrency always -- no two calls share an instance
                    context = {
                        "language_detected": item["language_detected"],
                        "script": item["script"],
                        "source_type": item["source_type"],
                    }
                    futures[pool.submit(extractor.extract, item["text"], context=context)] = (item, extractor)

                try:
                    # as_completed, not futures.items(): blocking in
                    # submission order meant one slow call in a batch stalled
                    # writing back the other (already-finished) results in
                    # that same batch -- observed in practice as an apparent
                    # hang while 15/16 workers sat on completed-but-unread
                    # futures.
                    for future in as_completed(futures):
                        item, extractor = futures[future]
                        try:
                            mentions = future.result()
                        except (GeminiFreeTierError, GeminiBillingExhaustedError, GeminiDailyQuotaExhaustedError):
                            raise  # systemic -- stop the whole run, see outer except
                        except Exception as item_error:  # noqa: BLE001 -- deliberately broad: isolate one item's failure (e.g. a 503 that outlasted every retry) from crashing the whole run
                            # Leave it uncached/unmarked so a later run
                            # retries it naturally; log it and move on.
                            stats.failed_items.append((item["item_id"], str(item_error)[:200]))
                            continue
                        stats.api_calls += 1
                        if extractor.last_usage_tokens is not None:  # type: ignore[attr-defined]
                            cost_tracker.add(*extractor.last_usage_tokens)  # type: ignore[attr-defined]
                            extractor.last_usage_tokens = None  # type: ignore[attr-defined]
                        resolution.store_extraction_cache(con, item["text_hash"], mentions, model=model)
                        _finalize_item(con, resolver, item, mentions, stats)
                except (GeminiFreeTierError, GeminiBillingExhaustedError, GeminiDailyQuotaExhaustedError) as e:
                    stats.stopped_early_reason = f"Gemini fatal error: {e}"
                    raise

            time.sleep(BATCH_SLEEP_SECONDS)

            # Threshold-based, not modulo: batch size (concurrency) and
            # PROGRESS_EVERY don't generally share a common divisor (e.g.
            # concurrency=16, PROGRESS_EVERY=25 only coincide every 400
            # items via modulo), which silently suppressed progress output
            # for an entire run in practice. This prints as soon as enough
            # new items have landed since the last print, regardless of batch size.
            if stats.items_considered - last_progress_print >= PROGRESS_EVERY or stats.items_considered == len(items):
                print(f"[extract_entities] {cost_tracker.progress_line(stats.items_considered, total_remaining)}", flush=True)
                last_progress_print = stats.items_considered

    return stats, con, cost_tracker


def print_report(stats: RunStats, con: duckdb.DuckDBPyConnection, cost_tracker: CostTracker) -> None:
    print("\n--- Entity extraction summary ---")
    print(f"Items processed: {stats.items_considered}")
    print(f"Actual API calls made: {stats.api_calls}")
    print(f"Content-cache hits (skipped API): {stats.cache_hits}")
    print(f"Total entity mentions extracted: {stats.total_mentions}")
    print(f"Total cost this run: ${cost_tracker.total_cost:.4f} ({cost_tracker.calls} billed calls)")
    if stats.stopped_early_reason:
        print(f"Stopped early: {stats.stopped_early_reason}")

    row = con.execute("SELECT count(DISTINCT entity_id) FROM item_entities").fetchone()
    assert row is not None
    print(f"Distinct canonical entities referenced (all-time): {row[0]}")

    print(f"\nItems where extraction returned zero entities: {len(stats.zero_entity_items)}")
    for item_id, snippet in stats.zero_entity_items[:15]:
        print(f"  {item_id}: {snippet!r}")

    print(f"\nItems that failed (retries exhausted, skipped -- will retry next run): {len(stats.failed_items)}")
    for item_id, error in stats.failed_items[:15]:
        print(f"  {item_id}: {error!r}")

    print(f"\n--- {len(stats.examples)} example items ---")
    for ex in stats.examples:
        print(f"\nitem_id={ex['item_id']} lang={ex['language_detected']!r} script={ex['script']!r}")
        print(f"  text: {(ex['text'] or '')[:120]!r}")
        for surface_form, canonical_name, entity_id, confidence in ex["resolved"]:
            print(f"    {surface_form!r} -> {canonical_name!r} (entity_id={entity_id[:8]}..., confidence={confidence:.2f})")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entity extraction + resolution (docs/analysis_layer_spec.md §4 pass 1)")
    parser.add_argument("--limit", type=int, default=None, help="Max items to process this invocation (default: no cap)")
    parser.add_argument(
        "--clusters-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict to items in coordination clusters (default: true). --no-clusters-only processes the full corpus, cluster items first.",
    )
    parser.add_argument("--provider", choices=["anthropic", "gemini"], default="anthropic", help="Which EntityExtractor implementation to use")
    parser.add_argument("--max-cost", type=float, default=None, help="Stop cleanly once estimated spend exceeds this USD amount")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help=f"Concurrent API calls in flight (default: {DEFAULT_CONCURRENCY})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.provider == "gemini":
        api_key, model = config.get_gemini_api_key(), config.GEMINI_MODEL
    else:
        api_key, model = config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL
    extractors = build_extractor_pool(args.provider, api_key=api_key, model=model, size=args.concurrency)
    cost_tracker = CostTracker(provider=args.provider, max_cost=args.max_cost)
    stats, con, cost_tracker = run(args.clusters_only, args.limit, extractors, model, cost_tracker)
    print_report(stats, con, cost_tracker)
    con.close()


if __name__ == "__main__":
    main()
