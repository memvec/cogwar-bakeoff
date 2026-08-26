"""Stance detection pass -- CLI entrypoint (docs/analysis_layer_spec.md §4 pass 2).

    uv run python -m analysis.detect_stance --provider gemini --clusters-only
    uv run python -m analysis.detect_stance --provider gemini --no-clusters-only --max-cost 50

Reads items + their resolved entities (item_entities, from the entity
extraction pass) from the analysis DuckDB, asks the configured
StanceDetector (analysis/stance.py) what stance the AUTHOR takes toward
EACH entity an item references, and writes entity_stance_edges
(analysis/stance_storage.py). Stance only -- narrative clustering, profile
aggregation, and finding assembly are later passes per the spec and are not
built here.

One API call per ITEM, not per (item, entity) pair: all of an item's
entities that still need a stance are batched into a single prompt (same
economy as extraction). Cost control, several layers deep:
  - Content cache: an (item, entity) pair whose text_hash + entity_id was
    already scored -- even under a different item_id (e.g. a repost) --
    is served from stance_cache with zero API calls.
  - Incremental: an item where every one of its entities already has an
    entity_stance_edges row is skipped entirely, no cache lookups needed --
    this is also what makes a provider switch (Anthropic -> Gemini)
    additive: pairs an earlier Anthropic run already scored stay
    Anthropic-labeled (entity_stance_edges.detector_model records which),
    and a Gemini run only ever processes what's still pending.
  - --max-cost: stop cleanly once running spend would exceed the cap --
    every item's writes are already committed as they happen, so a rerun
    just continues via the caches above.
--clusters-only (default true) restricts the candidate set to items that
appear in any derived edge (near_duplicate_text / shared_media /
temporal_cocluster). With --no-clusters-only, ALL items with pending
(item, entity) pairs are candidates, but cluster-membership still
determines priority order (cluster items first), same discipline as
extract_entities.py.

Neutral/noise threshold (spec §5.5): most entity mentions are incidental
with no real stance. Default behavior emits the edge regardless, with
polarity=neutral and low strength, so downstream aggregation can filter --
--skip-neutral-edges opts into dropping those edges at write time instead.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

import duckdb

from analysis import config, resolution, stance_storage
from analysis.cost import CostTracker
from analysis.gemini_retry import (
    GeminiBillingExhaustedError,
    GeminiDailyQuotaExhaustedError,
    GeminiFreeTierError,
)
from analysis.stance import (
    EntityRef,
    StanceDetector,
    StanceResult,
    build_detector_pool,
)

PROGRESS_EVERY = 25
DEFAULT_CONCURRENCY = 12

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
        self.failed_items: list[tuple[str, str]] = []  # (item_id, error message) -- retries exhausted, skipped, will retry next run
        self.stopped_early_reason: str | None = None


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


def _candidate_query(clusters_only: bool) -> tuple[str, str, str]:
    """Returns (cluster_cte, cluster_join, priority_expr) -- shared between
    select_candidate_items (needs LIMIT) and count_remaining_candidates
    (needs none)."""
    edge_types_sql = ", ".join(f"'{t}'" for t in _CLUSTER_EDGE_TYPES)
    cluster_cte = f""",
        cluster_items AS (
            SELECT src_item_id AS item_id FROM processed.edges
            WHERE origin = 'derived' AND edge_type IN ({edge_types_sql})
            UNION
            SELECT dst_item_id AS item_id FROM processed.edges
            WHERE origin = 'derived' AND edge_type IN ({edge_types_sql}) AND dst_item_id IS NOT NULL
        )"""
    if clusters_only:
        cluster_join = "JOIN cluster_items c ON iec.item_id = c.item_id"
        priority_expr = "(1)"  # constant -- NOT a bare int literal (DuckDB treats that as a positional ORDER BY reference)
    else:
        # Full corpus is candidate, but cluster items still sort first --
        # same priority discipline as extract_entities.py.
        cluster_join = "LEFT JOIN cluster_items c ON iec.item_id = c.item_id"
        priority_expr = "CASE WHEN c.item_id IS NOT NULL THEN 0 ELSE 1 END"
    return cluster_cte, cluster_join, priority_expr


def select_candidate_items(con: duckdb.DuckDBPyConnection, clusters_only: bool, limit: int | None) -> list[dict]:
    """Items with resolved entities where at least one (item, entity) pair
    still lacks an entity_stance_edges row, ordered cluster-items-first."""
    cluster_cte, cluster_join, priority_expr = _candidate_query(clusters_only)
    limit_clause = "LIMIT ?" if limit is not None else ""

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
        ORDER BY {priority_expr}, i.item_id
        {limit_clause}
    """
    params = [limit] if limit is not None else []
    rows = con.execute(query, params).fetchall()
    columns = ["item_id", "text", "text_hash", "language_detected", "script", "source_type"]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def count_remaining_candidates(con: duckdb.DuckDBPyConnection, clusters_only: bool) -> int:
    """Total pending candidates with no LIMIT -- the denominator for
    progress printing."""
    cluster_cte, cluster_join, _ = _candidate_query(clusters_only)
    query = f"""
        WITH item_entity_counts AS (
            SELECT item_id, count(DISTINCT entity_id) AS n_entities
            FROM item_entities GROUP BY item_id
        ),
        item_edge_counts AS (
            SELECT item_id, count(DISTINCT entity_id) AS n_edges
            FROM entity_stance_edges GROUP BY item_id
        ){cluster_cte}
        SELECT count(*)
        FROM item_entity_counts iec
        LEFT JOIN item_edge_counts iee ON iec.item_id = iee.item_id
        JOIN processed.items i ON i.item_id = iec.item_id
        {cluster_join}
        WHERE i.text IS NOT NULL AND i.text_hash IS NOT NULL
          AND (iee.n_edges IS NULL OR iee.n_edges < iec.n_entities)
    """
    row = con.execute(query).fetchone()
    assert row is not None
    return row[0]





def select_scoped_candidate_items(
    con: duckdb.DuckDBPyConnection,
    restrict_to_item_ids: set[str],
    priority_item_ids: set[str],
    limit: int | None,
) -> list[dict]:
    """Same shape/contract as select_candidate_items, restricted to an
    arbitrary item_id universe (e.g. analysis_scope's on-topic set) with an
    independently-specified priority subset, instead of clusters_only's
    corpus-wide, edge-based cluster definition -- mirrors
    extract_entities.select_scoped_candidate_items; see its docstring for
    why this is a separate function rather than a parameterization of
    _candidate_query.
    """
    limit_clause = "LIMIT $limit" if limit is not None else ""
    query = f"""
        WITH item_entity_counts AS (
            SELECT item_id, count(DISTINCT entity_id) AS n_entities
            FROM item_entities GROUP BY item_id
        ),
        item_edge_counts AS (
            SELECT item_id, count(DISTINCT entity_id) AS n_edges
            FROM entity_stance_edges GROUP BY item_id
        ),
        scope_items AS (
            SELECT UNNEST($scope_ids) AS item_id
        ),
        priority_items AS (
            SELECT UNNEST($priority_ids) AS item_id
        )
        SELECT i.item_id, i.text, i.text_hash, i.language_detected, i.script, i.source_type
        FROM item_entity_counts iec
        LEFT JOIN item_edge_counts iee ON iec.item_id = iee.item_id
        JOIN processed.items i ON i.item_id = iec.item_id
        JOIN scope_items sc ON i.item_id = sc.item_id
        LEFT JOIN priority_items pr ON i.item_id = pr.item_id
        WHERE i.text IS NOT NULL AND i.text_hash IS NOT NULL
          AND (iee.n_edges IS NULL OR iee.n_edges < iec.n_entities)
        ORDER BY CASE WHEN pr.item_id IS NOT NULL THEN 0 ELSE 1 END, i.item_id
        {limit_clause}
    """
    params: dict[str, object] = {"scope_ids": list(restrict_to_item_ids), "priority_ids": list(priority_item_ids)}
    if limit is not None:
        params["limit"] = limit
    rows = con.execute(query, params).fetchall()
    columns = ["item_id", "text", "text_hash", "language_detected", "script", "source_type"]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def count_scoped_remaining_candidates(con: duckdb.DuckDBPyConnection, restrict_to_item_ids: set[str]) -> int:
    """Mirrors count_remaining_candidates for the scoped path."""
    query = """
        WITH item_entity_counts AS (
            SELECT item_id, count(DISTINCT entity_id) AS n_entities
            FROM item_entities GROUP BY item_id
        ),
        item_edge_counts AS (
            SELECT item_id, count(DISTINCT entity_id) AS n_edges
            FROM entity_stance_edges GROUP BY item_id
        ),
        scope_items AS (
            SELECT UNNEST($scope_ids) AS item_id
        )
        SELECT count(*)
        FROM item_entity_counts iec
        LEFT JOIN item_edge_counts iee ON iec.item_id = iee.item_id
        JOIN processed.items i ON i.item_id = iec.item_id
        JOIN scope_items sc ON i.item_id = sc.item_id
        WHERE i.text IS NOT NULL AND i.text_hash IS NOT NULL
          AND (iee.n_edges IS NULL OR iee.n_edges < iec.n_entities)
    """
    row = con.execute(query, {"scope_ids": list(restrict_to_item_ids)}).fetchone()
    assert row is not None
    return row[0]


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


def _get_pending(con: duckdb.DuckDBPyConnection, item: dict) -> list[EntityRef]:
    entities = stance_storage.get_item_entities(con, item["item_id"])
    existing = stance_storage.get_existing_edge_entity_ids(con, item["item_id"])
    return [e for e in entities if e.entity_id not in existing]


def _finalize_item(
    con: duckdb.DuckDBPyConnection,
    item: dict,
    pending: list[EntityRef],
    all_results: dict[str, StanceResult],
    stats: RunStats,
    skip_neutral: bool,
    model: str,
) -> list[tuple[EntityRef, StanceResult]]:
    """Persist one item's already-scored (cache + fresh) results. Always
    runs on the main thread/connection, never inside a worker."""
    resolved = []
    for e in pending:
        result = all_results[e.entity_id]
        stats.polarity_counts[result.polarity] += 1
        if skip_neutral and result.polarity == "neutral":
            stats.skipped_neutral += 1
        else:
            stance_storage.record_stance_edge(con, item["item_id"], result, model=model)
            stats.edges_written += 1
        resolved.append((e, result))

    stats.items_considered += 1
    return resolved


def run(
    clusters_only: bool,
    limit: int | None,
    skip_neutral: bool,
    detectors: list[StanceDetector],
    model: str,
    cost_tracker: CostTracker,
    restrict_to_item_ids: set[str] | None = None,
    priority_item_ids: set[str] | None = None,
) -> tuple[RunStats, duckdb.DuckDBPyConnection, CostTracker]:
    """`detectors` is a pool of `concurrency` independent instances (see
    stance.build_detector_pool) -- within one batch, slot i always calls
    detectors[i], so no two threads ever touch the same instance at once.

    restrict_to_item_ids/priority_item_ids mirror extract_entities.run's --
    see its docstring. Purely additive: clusters_only/limit still apply to
    the original (unscoped) path when restrict_to_item_ids is None.
    """
    con = connect()
    concurrency = len(detectors)

    if restrict_to_item_ids is not None:
        total_remaining = count_scoped_remaining_candidates(con, restrict_to_item_ids)
        items = select_scoped_candidate_items(con, restrict_to_item_ids, priority_item_ids or set(), limit)
        print(
            f"[detect_stance] candidate items this invocation: {len(items)} "
            f"(scoped to {len(restrict_to_item_ids)} items, {len(priority_item_ids or set())} priority, limit={limit}); "
            f"{total_remaining} total pending in scope; concurrency={concurrency}",
            flush=True,
        )
    else:
        total_remaining = count_remaining_candidates(con, clusters_only)
        items = select_candidate_items(con, clusters_only, limit)
        print(
            f"[detect_stance] candidate items this invocation: {len(items)} "
            f"(clusters_only={clusters_only}, limit={limit}); {total_remaining} total pending; concurrency={concurrency}",
            flush=True,
        )

    estimated = estimate_api_calls(con, items)
    print(f"[detect_stance] estimated API calls: {estimated}", flush=True)

    stats = RunStats()
    last_progress_print = 0
    run_start = time.monotonic()

    def print_progress() -> None:
        nonlocal last_progress_print
        if stats.items_considered - last_progress_print >= PROGRESS_EVERY or stats.items_considered == len(items):
            elapsed_min = (time.monotonic() - run_start) / 60
            rate = stats.items_considered / elapsed_min if elapsed_min > 0 else 0.0
            print(
                f"[detect_stance] {cost_tracker.progress_line(stats.items_considered, total_remaining)} "
                f"-- {rate:.1f} items/min",
                flush=True,
            )
            last_progress_print = stats.items_considered

    # Rolling worker pool, not lockstep batches -- identical rationale and
    # cost-safety properties as extract_entities.py.run()'s rewrite (see its
    # comment): up to `concurrency` calls in flight at all times, cost cap
    # checked before every new submission (not once per batch), cost_tracker
    # only ever touched from this main thread (single-writer, no lock
    # needed). PendingCall carries everything needed to finalize a call's
    # result once its Future completes.
    items_iter = iter(items)
    available_slots = list(range(concurrency))
    in_flight: dict[Future, tuple[dict, list[EntityRef], list[EntityRef], dict[str, StanceResult], int]] = {}

    def submit_next_call() -> bool:
        """Pulls from items_iter, finalizing items with nothing pending or
        fully served by stance_cache inline (neither occupies a worker
        slot), until it either submits one real API call (returns True) or
        runs out of reasons to keep going -- the cost cap or an exhausted
        item list (returns False)."""
        for item in items_iter:
            pending = _get_pending(con, item)
            if not pending:
                continue
            cached_results: dict[str, StanceResult] = {}
            uncached_entities: list[EntityRef] = []
            for e in pending:
                cached = stance_storage.get_cached_stance(con, item["text_hash"], e.entity_id)
                if cached is not None:
                    cached_results[e.entity_id] = cached
                else:
                    uncached_entities.append(e)
            stats.pairs_from_cache += len(cached_results)

            if not uncached_entities:
                stats.items_fully_cached += 1
                _finalize_item(con, item, pending, cached_results, stats, skip_neutral, model)
                print_progress()
                continue

            if cost_tracker.would_exceed():
                stats.stopped_early_reason = f"cost cap (${cost_tracker.max_cost:.2f}) reached"
                return False

            slot = available_slots.pop()
            detector = detectors[slot]  # this slot's instance is free -- nothing else is using it right now
            context = {
                "language_detected": item["language_detected"],
                "script": item["script"],
                "source_type": item["source_type"],
            }
            future = pool.submit(detector.detect, item["text"], uncached_entities, context=context)
            in_flight[future] = (item, pending, uncached_entities, cached_results, slot)
            return True
        return False

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        try:
            for _ in range(concurrency):
                if not submit_next_call():
                    break

            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                refill_budget = 0
                for future in done:
                    item, pending, uncached_entities, cached_results, slot = in_flight.pop(future)
                    available_slots.append(slot)
                    refill_budget += 1
                    detector = detectors[slot]
                    try:
                        detected = future.result()
                    except (GeminiFreeTierError, GeminiBillingExhaustedError, GeminiDailyQuotaExhaustedError):
                        raise  # systemic -- stop the whole run, see outer except
                    except Exception as item_error:  # noqa: BLE001 -- deliberately broad: isolate one item's failure (e.g. a 503 that outlasted every retry) from crashing the whole run
                        # Leave it unscored so a later run retries it naturally.
                        stats.failed_items.append((item["item_id"], str(item_error)[:200]))
                        continue
                    stats.api_calls += 1
                    if detector.last_usage_tokens is not None:  # type: ignore[attr-defined]
                        cost_tracker.add(*detector.last_usage_tokens)  # type: ignore[attr-defined]
                        detector.last_usage_tokens = None  # type: ignore[attr-defined]

                    if not detected:
                        stats.empty_response_items.append((item["item_id"], (item["text"] or "")[:80]))

                    by_id = {r.entity_id: r for r in detected}
                    new_results: dict[str, StanceResult] = {}
                    for e in uncached_entities:
                        result = by_id.get(e.entity_id)
                        if result is None:
                            # Model omitted this entity -- fall back to a
                            # zero-confidence neutral edge rather than
                            # dropping the pair silently.
                            result = StanceResult(entity_id=e.entity_id, polarity="neutral", strength=0.0, confidence=0.0)
                            stats.detection_gaps.append((item["item_id"], e.entity_id, e.canonical_name))
                        stance_storage.store_stance_cache(con, item["text_hash"], result, model=model)
                        new_results[e.entity_id] = result
                    stats.pairs_from_api += len(uncached_entities)

                    all_results = {**cached_results, **new_results}
                    _finalize_item(con, item, pending, all_results, stats, skip_neutral, model)
                    print_progress()

                if stats.stopped_early_reason is None:
                    for _ in range(refill_budget):
                        if not submit_next_call():
                            break
        except (GeminiFreeTierError, GeminiBillingExhaustedError, GeminiDailyQuotaExhaustedError) as e:
            stats.stopped_early_reason = f"Gemini fatal error: {e}"
            raise

    if stats.stopped_early_reason:
        print(
            f"\n[detect_stance] STOPPING: {stats.stopped_early_reason}. "
            "Rerun to continue -- already-scored pairs are cached.",
            flush=True,
        )

    return stats, con, cost_tracker


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


def print_report(stats: RunStats, con: duckdb.DuckDBPyConnection, cost_tracker: CostTracker) -> None:
    print("\n--- Stance detection summary ---")
    print(f"Items processed: {stats.items_considered}")
    print(f"Actual API calls made: {stats.api_calls}")
    print(f"Items fully served from cache (0 API calls): {stats.items_fully_cached}")
    print(f"(item, entity) pairs from cache: {stats.pairs_from_cache}")
    print(f"(item, entity) pairs from API: {stats.pairs_from_api}")
    print(f"Edges written: {stats.edges_written}")
    print(f"Edges skipped (--skip-neutral-edges): {stats.skipped_neutral}")
    print(f"Total cost this run: ${cost_tracker.total_cost:.4f} ({cost_tracker.calls} billed calls)")
    if stats.stopped_early_reason:
        print(f"Stopped early: {stats.stopped_early_reason}")

    print("\nPolarity breakdown (all pairs decided, including any skipped):")
    for polarity in ("positive", "negative", "neutral"):
        print(f"  {polarity}: {stats.polarity_counts.get(polarity, 0)}")

    row = con.execute("SELECT count(*) FROM entity_stance_edges").fetchone()
    assert row is not None
    print(f"\nTotal entity_stance_edges (all-time): {row[0]}")

    print(f"\nDetection gaps (model omitted an entity from its response, filled with neutral/0-confidence): {len(stats.detection_gaps)}")
    for item_id, entity_id, canonical_name in stats.detection_gaps[:15]:
        print(f"  item={item_id} entity={canonical_name!r} ({entity_id[:8]}...)")

    print(f"\nItems where the API call returned NOTHING (stance detection failed): {len(stats.empty_response_items)}")
    for item_id, snippet in stats.empty_response_items[:15]:
        print(f"  {item_id}: {snippet!r}")

    print(f"\nItems that failed (retries exhausted, skipped -- will retry next run): {len(stats.failed_items)}")
    for item_id, error in stats.failed_items[:15]:
        print(f"  {item_id}: {error!r}")

    print_examples(con)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stance detection (docs/analysis_layer_spec.md §4 pass 2)")
    parser.add_argument("--limit", type=int, default=None, help="Max items to process this invocation (default: no cap)")
    parser.add_argument(
        "--clusters-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict to items in coordination clusters (default: true). --no-clusters-only processes all pending items, cluster items first.",
    )
    parser.add_argument(
        "--skip-neutral-edges",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop polarity=neutral edges at write time instead of persisting them (default: false, keep everything)",
    )
    parser.add_argument("--provider", choices=["anthropic", "gemini"], default="anthropic", help="Which StanceDetector implementation to use")
    parser.add_argument("--max-cost", type=float, default=None, help="Stop cleanly once estimated spend exceeds this USD amount")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help=f"Concurrent API calls in flight (default: {DEFAULT_CONCURRENCY})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.provider == "gemini":
        api_key, model = config.get_gemini_api_key(), config.GEMINI_MODEL
    else:
        api_key, model = config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL
    detectors = build_detector_pool(args.provider, api_key=api_key, model=model, size=args.concurrency)
    cost_tracker = CostTracker(provider=args.provider, max_cost=args.max_cost)
    stats, con, cost_tracker = run(args.clusters_only, args.limit, args.skip_neutral_edges, detectors, model, cost_tracker)
    print_report(stats, con, cost_tracker)
    con.close()


if __name__ == "__main__":
    main()
