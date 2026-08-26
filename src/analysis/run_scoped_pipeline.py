"""Scoped analysis pipeline run -- CLI entrypoint, real money, on-topic scope only.

    uv run python -m analysis.run_scoped_pipeline --max-cost 40

Runs entity extraction + stance detection (Gemini, paid tier) over exactly
the on-topic scope persisted in analysis_scope (analysis.build_scope
--on-topic-only), NOT the full corpus -- unlike run_full_pipeline.py, which
processes everything. Priority order within the scope: items in a
coordination cluster (processing/derive.py's shared_media/near_duplicate_text
derived edges, unrestricted by topic) with >= --min-cluster-authors distinct
authors first, then the rest of the on-topic scope -- so a run that stops at
the cost cap has covered the highest-value (coordinated + on-topic) items
first and only misses long-tail profile-enrichment items.

Cost discipline: a small real PROBE (the first --probe-size priority items)
runs first, under the same shared CostTracker as the full run -- not a
separate/wasted call, just the start of the real one. Its observed $/item is
used to project the total cost of finishing the rest of the scope. That
projection is printed BEFORE continuing; if it would exceed --max-cost, the
run STOPS THERE (reporting how far the budget would go, by priority tier)
and does not proceed further -- the probe's small spend is the only cost
incurred until a human decides to proceed (e.g. by raising --max-cost and
re-running, which resumes for free via the extraction/stance caches, or
accepting partial coverage). If the projection is within budget, the run
continues straight through the rest of the scope, still hard-capped by the
same CostTracker regardless.

After extraction + stance (however far they got): entity merge (local
Ollama, free), tight narrative clustering (free, local), and author profile
aggregation (free, local) -- same three deterministic passes
run_full_pipeline.py runs, reused as-is.

Resumable exactly like run_full_pipeline.py: every item's writes commit as
they happen (extraction_cache / stance_cache / item_extraction_status /
entity_stance_edges), so re-running (with a larger --max-cost, if the first
run stopped at the projection gate or the cap) continues for free on
everything already done.
"""

from __future__ import annotations

import argparse

from analysis import (
    build_narratives,
    build_profiles,
    config,
    detect_stance,
    extract_entities,
    merge,
    profiles,
    scope,
)
from analysis.cost import CostTracker
from analysis.entities import build_extractor_pool
from analysis.gemini_retry import (
    GeminiBillingExhaustedError,
    GeminiDailyQuotaExhaustedError,
    GeminiFreeTierError,
)
from analysis.stance import build_detector_pool

DEFAULT_MIN_CLUSTER_AUTHORS = 3
DEFAULT_PROBE_SIZE = 20
DEFAULT_CONCURRENCY = 12


def load_scope_and_priority(min_cluster_authors: int) -> tuple[set[str], set[str], int]:
    """(scope_ids, priority_ids, n_priority_clusters) -- scope_ids is the
    full on-topic set (analysis_scope), priority_ids is the subset of it
    that's ALSO in a >=min_cluster_authors coordination cluster (computed
    fresh here -- analysis_scope itself carries no coordination info since
    Phase 1's on-topic-only rebuild).
    """
    con = extract_entities.connect()
    scope_ids = {r[0] for r in con.execute("SELECT item_id FROM analysis_scope").fetchall()}

    print("[run_scoped_pipeline] computing coordination clusters for priority tiering ...", flush=True)
    clusters = scope.compute_coordination_clusters(con)
    author_by_item = scope.load_author_by_item(con, clusters)
    coord_ids, n_clusters = scope.coordination_set_for_threshold(clusters, author_by_item, min_cluster_authors)
    priority_ids = coord_ids & scope_ids
    con.close()
    return scope_ids, priority_ids, n_clusters


def run_probe_and_project(
    scope_ids: set[str],
    priority_ids: set[str],
    probe_size: int,
    concurrency: int,
    api_key: str,
    model: str,
    cost_tracker: CostTracker,
) -> tuple[extract_entities.RunStats, detect_stance.RunStats, float, float]:
    """Runs extraction + stance on the first `probe_size` priority items
    (real production items, not throwaway), measures actual $/item from
    cost_tracker, and returns (extraction_stats, stance_stats,
    observed_cost_per_item, projected_total_for_full_scope).
    """
    print(f"\n[run_scoped_pipeline] PROBE: extraction on first {probe_size} priority items ...", flush=True)
    extractors = build_extractor_pool("gemini", api_key=api_key, model=model, size=concurrency)
    extraction_stats, econ, _ = extract_entities.run(
        clusters_only=False,
        limit=probe_size,
        extractors=extractors,
        model=model,
        cost_tracker=cost_tracker,
        restrict_to_item_ids=scope_ids,
        priority_item_ids=priority_ids,
    )
    extract_entities.print_report(extraction_stats, econ, cost_tracker)
    econ.close()

    print(f"\n[run_scoped_pipeline] PROBE: stance on the same {probe_size} priority items ...", flush=True)
    detectors = build_detector_pool("gemini", api_key=api_key, model=model, size=concurrency)
    stance_stats, scon, _ = detect_stance.run(
        clusters_only=False,
        limit=probe_size,
        skip_neutral=False,
        detectors=detectors,
        model=model,
        cost_tracker=cost_tracker,
        restrict_to_item_ids=scope_ids,
        priority_item_ids=priority_ids,
    )
    detect_stance.print_report(stance_stats, scon, cost_tracker)
    scon.close()

    items_probed = max(extraction_stats.items_considered, stance_stats.items_considered, 1)
    observed_cost_per_item = cost_tracker.total_cost / items_probed

    con = extract_entities.connect()
    remaining_after_probe = extract_entities.count_scoped_remaining_candidates(con, scope_ids)
    con.close()

    projected_remaining_cost = observed_cost_per_item * remaining_after_probe
    projected_total = cost_tracker.total_cost + projected_remaining_cost
    return extraction_stats, stance_stats, observed_cost_per_item, projected_total


def print_projection_and_gate(
    scope_ids: set[str],
    priority_ids: set[str],
    observed_cost_per_item: float,
    projected_total: float,
    max_cost: float,
    already_spent: float,
    min_cluster_authors: int,
) -> bool:
    """Prints the projection + tier breakdown; returns True if it's safe to
    continue (projected_total <= max_cost), False if the run should stop
    here for a human decision.
    """
    print("\n" + "=" * 70)
    print("COST PROJECTION")
    print("=" * 70)
    print(f"Observed cost per item (from probe): ${observed_cost_per_item:.5f}")
    print(f"Already spent (probe): ${already_spent:.4f}")
    print(f"Projected TOTAL to finish the full on-topic scope: ${projected_total:.2f} (cap: ${max_cost:.2f})")

    if projected_total <= max_cost:
        print(f"Projection is WITHIN the ${max_cost:.2f} cap -- proceeding with the full remaining scope.")
        return True

    affordable_items = int((max_cost - already_spent) / observed_cost_per_item) if observed_cost_per_item > 0 else 0
    affordable_items = max(affordable_items, 0)
    pct_of_scope = affordable_items / len(scope_ids) * 100 if scope_ids else 0.0
    covers_priority_tier = affordable_items >= len(priority_ids)

    print(f"\nProjection EXCEEDS the ${max_cost:.2f} cap.")
    print(f"At this rate, ${max_cost:.2f} affords approximately {affordable_items} more items ")
    print(f"({pct_of_scope:.1f}% of the {len(scope_ids)}-item on-topic scope).")
    print(f"Priority tier (>= {min_cluster_authors}-author coordination clusters, in scope): {len(priority_ids)} items.")
    if covers_priority_tier:
        print(
            f"${max_cost:.2f} FULLY covers the priority tier ({len(priority_ids)} items) "
            f"plus ~{affordable_items - len(priority_ids)} items from the rest of the scope."
        )
    else:
        print(
            f"${max_cost:.2f} covers only PART of the priority tier "
            f"({affordable_items} of {len(priority_ids)} priority items) -- "
            "the rest of the scope (non-priority) would not be reached at all this run."
        )
    print("\nSTOPPING here -- not proceeding with the full run. Re-run with a higher --max-cost to continue, ")
    print("or accept this as a partial pass (already-probed items are cached, so nothing here is wasted).")
    return False


def print_final_report(
    cost_tracker: CostTracker,
    extraction_stats: extract_entities.RunStats,
    stance_stats: detect_stance.RunStats,
    scope_size: int,
    entities_before_merge: int,
    stopped_early: bool,
) -> None:
    con = extract_entities.connect()

    print("\n" + "=" * 70)
    print("FINAL REPORT -- scoped pipeline run")
    print("=" * 70)

    print(f"\nTotal Gemini cost this run: ${cost_tracker.total_cost:.4f} (cap was ${cost_tracker.max_cost})")
    print(
        f"Items processed this run: extraction={extraction_stats.items_considered}, "
        f"stance={stance_stats.items_considered} (of {scope_size} in the on-topic scope)"
    )
    print(f"Completed the full on-topic scope: {'NO -- stopped at the cost cap' if stopped_early else 'YES'}")

    con.close()

    print("\n--- Entity merge / reconciliation (local qwen, free) ---")
    merge_stats, merge_con = merge.run()
    print(f"Entities before merge: {entities_before_merge}")
    print(f"Entities after merge: {merge_stats.after_count}")
    print(f"Auto-merged (fuzzy): {merge_stats.auto_merged}, LLM-confirmed: {merge_stats.llm_confirmed}, LLM-rejected: {merge_stats.llm_rejected}")
    merge_con.close()

    print("\n--- Tight narrative clustering (free, local) ---")
    narrative_con = build_narratives.run()
    n_narratives = narrative_con.execute("SELECT count(*) FROM narratives WHERE basis = 'tight'").fetchone()[0]  # type: ignore[index]
    n_multi_author = narrative_con.execute(
        "SELECT count(*) FROM narratives WHERE basis = 'tight' AND distinct_authors > 1"
    ).fetchone()[0]  # type: ignore[index]
    print(f"N tight narratives: {n_narratives} ({n_multi_author} multi-author)")
    narrative_con.close()

    print("\n--- Author entity-stance profiles (free, local) ---")
    profile_con = build_profiles.connect()
    n_profiles = profiles.rebuild(profile_con)
    print(f"N author_entity_profiles rows: {n_profiles}")

    print("\n--- entity_stance_edges by polarity (all-time) ---")
    for polarity, count in profile_con.execute(
        "SELECT polarity, count(*) FROM entity_stance_edges GROUP BY polarity ORDER BY count(*) DESC"
    ).fetchall():
        print(f"  {polarity}: {count}")

    print("\n--- Query #3: India and Pakistan -- who consistently pushes for/against them ---")
    for canonical_name in ("India", "Pakistan"):
        row = profile_con.execute(
            "SELECT entity_id FROM entities WHERE canonical_name = ?", [canonical_name]
        ).fetchone()
        if row is None:
            print(f"\nEntity {canonical_name!r}: not found (no extraction has resolved it yet)")
            continue
        entity_id = row[0]
        n_authors_row = profile_con.execute(
            "SELECT count(DISTINCT author_id) FROM author_entity_profiles WHERE entity_id = ?", [entity_id]
        ).fetchone()
        n_authors = n_authors_row[0] if n_authors_row else 0
        print(f"\nEntity: {canonical_name!r} ({n_authors} authors with a profile toward it)")
        positive, negative = profiles.query_entity_authors(profile_con, entity_id, limit=5)
        print("  Consistently POSITIVE:")
        for r in positive:
            print(
                f"    {r.author_id}: net_stance={r.net_stance:+.2f} consistency={r.stance_consistency:.2f} "
                f"volume={r.volume} narrative_spread={r.narrative_spread} score={r.score:.3f}"
            )
        if not positive:
            print("    (none clear the ranking bar)")
        print("  Consistently NEGATIVE:")
        for r in negative:
            print(
                f"    {r.author_id}: net_stance={r.net_stance:+.2f} consistency={r.stance_consistency:.2f} "
                f"volume={r.volume} narrative_spread={r.narrative_spread} score={r.score:.3f}"
            )
        if not negative:
            print("    (none clear the ranking bar)")

    profile_con.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scoped analysis pipeline run (on-topic set only)")
    parser.add_argument("--max-cost", type=float, default=40.0, help="Hard USD cap across extraction + stance combined (default: 40.0)")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help=f"Concurrent API calls in flight (default: {DEFAULT_CONCURRENCY})")
    parser.add_argument("--min-cluster-authors", type=int, default=DEFAULT_MIN_CLUSTER_AUTHORS, help="Priority-tier threshold (default: 3)")
    parser.add_argument("--probe-size", type=int, default=DEFAULT_PROBE_SIZE, help="Real items to process before projecting total cost (default: 20)")
    parser.add_argument(
        "--skip-projection",
        action="store_true",
        help=(
            "Skip the probe + cost-projection gate and go straight to the full capped run -- "
            "use only after a prior invocation already showed the projection and a human decided "
            "to proceed anyway (e.g. accepting partial coverage at this budget)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    api_key, model = config.get_gemini_api_key(), config.GEMINI_MODEL

    print(f"[run_scoped_pipeline] provider=gemini model={model} max_cost=${args.max_cost} concurrency={args.concurrency}")
    print(f"[run_scoped_pipeline] priority tier: >= {args.min_cluster_authors}-author on-topic coordination clusters")

    scope_ids, priority_ids, n_priority_clusters = load_scope_and_priority(args.min_cluster_authors)
    print(f"[run_scoped_pipeline] on-topic scope: {len(scope_ids)} items")
    print(f"[run_scoped_pipeline] priority tier: {len(priority_ids)} items across {n_priority_clusters} clusters")
    print(f"[run_scoped_pipeline] rest of scope: {len(scope_ids) - len(priority_ids)} items")

    con = extract_entities.connect()
    entities_before_merge = con.execute("SELECT count(*) FROM entities").fetchone()[0]  # type: ignore[index]
    con.close()

    cost_tracker = CostTracker(provider="gemini", max_cost=args.max_cost)

    if args.skip_projection:
        print(
            f"\n[run_scoped_pipeline] --skip-projection set -- proceeding straight to the full run, "
            f"hard-capped at ${args.max_cost:.2f} for THIS invocation (a prior probe/invocation's spend, "
            "if any, is not carried over into this cap -- factor that in via --max-cost if it matters).",
            flush=True,
        )
    else:
        try:
            extraction_stats, stance_stats, cost_per_item, projected_total = run_probe_and_project(
                scope_ids, priority_ids, args.probe_size, args.concurrency, api_key, model, cost_tracker
            )
        except (GeminiFreeTierError, GeminiBillingExhaustedError, GeminiDailyQuotaExhaustedError) as e:
            print(f"\n[run_scoped_pipeline] ABORTED during probe: {e}")
            return

        should_continue = print_projection_and_gate(
            scope_ids, priority_ids, cost_per_item, projected_total, args.max_cost, cost_tracker.total_cost,
            args.min_cluster_authors,
        )
        if not should_continue:
            return

    try:
        print("\n[run_scoped_pipeline] FULL RUN: extraction over the remaining on-topic scope ...", flush=True)
        extractors = build_extractor_pool("gemini", api_key=api_key, model=model, size=args.concurrency)
        extraction_stats, econ, _ = extract_entities.run(
            clusters_only=False,
            limit=None,
            extractors=extractors,
            model=model,
            cost_tracker=cost_tracker,
            restrict_to_item_ids=scope_ids,
            priority_item_ids=priority_ids,
        )
        extract_entities.print_report(extraction_stats, econ, cost_tracker)
        econ.close()

        print("\n[run_scoped_pipeline] FULL RUN: stance over the remaining on-topic scope ...", flush=True)
        detectors = build_detector_pool("gemini", api_key=api_key, model=model, size=args.concurrency)
        stance_stats, scon, _ = detect_stance.run(
            clusters_only=False,
            limit=None,
            skip_neutral=False,
            detectors=detectors,
            model=model,
            cost_tracker=cost_tracker,
            restrict_to_item_ids=scope_ids,
            priority_item_ids=priority_ids,
        )
        detect_stance.print_report(stance_stats, scon, cost_tracker)
        scon.close()
    except (GeminiFreeTierError, GeminiBillingExhaustedError, GeminiDailyQuotaExhaustedError) as e:
        print(f"\n[run_scoped_pipeline] ABORTED: {e}")
        return

    stopped_early = bool(extraction_stats.stopped_early_reason or stance_stats.stopped_early_reason)
    print_final_report(cost_tracker, extraction_stats, stance_stats, len(scope_ids), entities_before_merge, stopped_early)


if __name__ == "__main__":
    main()
