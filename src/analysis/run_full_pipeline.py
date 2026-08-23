"""Full-corpus pipeline run -- CLI entrypoint, real money, real corpus.

    uv run python -m analysis.run_full_pipeline --provider gemini --max-cost 50

Orchestrates the whole analysis layer end to end over the FULL corpus (not
just coordination clusters), in this order:

  1. Entity extraction (analysis/extract_entities.py), full corpus,
     coordination-cluster items first, then the rest.
  2. Stance detection (analysis/stance_storage.py via analysis/detect_stance.py),
     over all (item, entity) pairs the extraction pass produced.
  3. Entity merge / reconciliation (analysis/merge.py) -- deterministic
     fuzzy tier is free; the LLM-adjudication tier (Tier 2) is Anthropic and
     DOES cost a little, separately from the tracked Gemini budget below
     (see the final report -- it's flagged, not silently absorbed).
  4. Tight narrative clustering (analysis/build_narratives.py) -- free, local.
  5. Author entity-stance profile aggregation (analysis/build_profiles.py) --
     free, local.

Steps 1-2 share ONE CostTracker (--max-cost is a budget for the whole
Gemini spend across extraction + stance combined, not per-phase) so this
can never silently run to 2x the cap. Steps 3-5 are deterministic/local
(re-run cleanly every time, like derived edges) except for merge's small
Tier-2 LLM cost noted above.

Resumable exactly like its component passes: every item's writes commit as
they happen, so a crash, manual interrupt, rate limit, or --max-cost stop
loses nothing -- rerunning this same command continues from the caches
(extraction_cache / stance_cache / item_extraction_status /
entity_stance_edges), skipping everything already done, in both phases.

Free-tier / billing detection: there is no way to query "is this key
paid-tier" or "does this account have balance" ahead of time from just an
API key (both are properties of the underlying Cloud project, not exposed
via the Gemini API) -- so this fails loud the moment either is actually
detectable instead: analysis.gemini_retry raises GeminiFreeTierError or
GeminiBillingExhaustedError immediately (no retry -- neither resolves by
waiting) the first time Gemini's own 429 response names the free tier or
depleted prepaid credits, respectively, and this script aborts cleanly on
either, rather than continuing to send corpus text into a wall.
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
)
from analysis.cost import CostTracker
from analysis.entities import build_extractor_pool
from analysis.gemini_retry import (
    GeminiBillingExhaustedError,
    GeminiDailyQuotaExhaustedError,
    GeminiFreeTierError,
)
from analysis.stance import build_detector_pool


def run_extraction_phase(
    provider: str, api_key: str, model: str, cost_tracker: CostTracker, concurrency: int
) -> extract_entities.RunStats:
    print("\n" + "=" * 70)
    print("PHASE 1: Entity extraction (full corpus, cluster-priority order)")
    print("=" * 70, flush=True)
    extractors = build_extractor_pool(provider, api_key=api_key, model=model, size=concurrency)
    stats, con, _ = extract_entities.run(
        clusters_only=False, limit=None, extractors=extractors, model=model, cost_tracker=cost_tracker
    )
    extract_entities.print_report(stats, con, cost_tracker)
    con.close()
    return stats


def run_stance_phase(
    provider: str, api_key: str, model: str, cost_tracker: CostTracker, concurrency: int
) -> detect_stance.RunStats:
    print("\n" + "=" * 70)
    print("PHASE 2: Stance detection (full corpus, cluster-priority order)")
    print("=" * 70, flush=True)
    detectors = build_detector_pool(provider, api_key=api_key, model=model, size=concurrency)
    stats, con, _ = detect_stance.run(
        clusters_only=False,
        limit=None,
        skip_neutral=False,
        detectors=detectors,
        model=model,
        cost_tracker=cost_tracker,
    )
    detect_stance.print_report(stats, con, cost_tracker)
    con.close()
    return stats


def print_final_report(cost_tracker: CostTracker, extraction_stats: extract_entities.RunStats, stance_stats: detect_stance.RunStats) -> None:
    # extract_entities.connect() opens the analysis DB as the default target
    # and attaches processed read-only -- analysis tables (item_extraction_status,
    # entities, ...) are queried directly on `con`.
    con = extract_entities.connect()

    print("\n" + "=" * 70)
    print("FINAL REPORT -- full pipeline run")
    print("=" * 70)

    print(f"\nGemini API calls: extraction={extraction_stats.api_calls}, stance={stance_stats.api_calls}, total={extraction_stats.api_calls + stance_stats.api_calls}")
    print(f"Total Gemini cost this run: ${cost_tracker.total_cost:.4f} (cap was ${cost_tracker.max_cost})" if cost_tracker.max_cost else f"Total Gemini cost this run: ${cost_tracker.total_cost:.4f} (no cap set)")
    print("NOTE: entity merge's Tier-2 LLM adjudication (below) uses Anthropic and is NOT included in the Gemini figure above -- it's typically a handful of calls, but is real, separate spend.")

    row = con.execute("SELECT count(*) FROM item_extraction_status").fetchone()
    print(f"\nN items with entity extraction done (all-time): {row[0]}")  # type: ignore[index]
    con.close()  # close before merge.run() opens its own connection to the same file

    print("\n--- Entity merge / reconciliation ---")
    merge_stats, merge_con = merge.run()
    print(f"Entities before merge: {merge_stats.before_count}")
    print(f"Entities after merge: {merge_stats.after_count}")
    print(f"Auto-merged (fuzzy): {merge_stats.auto_merged}, LLM-confirmed: {merge_stats.llm_confirmed}, LLM-rejected: {merge_stats.llm_rejected}")
    merge_con.close()

    print("\n--- Tight narrative clustering ---")
    narrative_con = build_narratives.run()
    n_narratives = narrative_con.execute("SELECT count(*) FROM narratives WHERE basis = 'tight'").fetchone()[0]  # type: ignore[index]
    n_multi_author = narrative_con.execute(
        "SELECT count(*) FROM narratives WHERE basis = 'tight' AND distinct_authors > 1"
    ).fetchone()[0]  # type: ignore[index]
    print(f"N tight narratives: {n_narratives} ({n_multi_author} multi-author)")
    narrative_con.close()

    print("\n--- Author entity-stance profiles ---")
    profile_con = build_profiles.connect()
    n_profiles = profiles.rebuild(profile_con)
    print(f"N author_entity_profiles rows: {n_profiles}")

    print("\n--- entity_stance_edges by polarity (all-time) ---")
    for polarity, count in profile_con.execute(
        "SELECT polarity, count(*) FROM entity_stance_edges GROUP BY polarity ORDER BY count(*) DESC"
    ).fetchall():
        print(f"  {polarity}: {count}")

    print("\n--- Query #3 on the best-covered entities: who consistently pushes for/against them ---")
    top_entities = profile_con.execute(
        """
        SELECT ap.entity_id, e.canonical_name, count(DISTINCT ap.author_id) AS n_authors
        FROM author_entity_profiles ap
        JOIN entities e ON e.entity_id = ap.entity_id
        GROUP BY ap.entity_id, e.canonical_name
        ORDER BY n_authors DESC
        LIMIT 3
        """
    ).fetchall()
    for entity_id, canonical_name, n_authors in top_entities:
        print(f"\nEntity: {canonical_name!r} ({n_authors} authors with a profile toward it)")
        positive, negative = profiles.query_entity_authors(profile_con, entity_id, limit=5)
        print("  Consistently POSITIVE:")
        for r in positive:
            print(f"    {r.author_id}: net_stance={r.net_stance:+.2f} consistency={r.stance_consistency:.2f} volume={r.volume} narrative_spread={r.narrative_spread} score={r.score:.3f}")
        print("  Consistently NEGATIVE:")
        for r in negative:
            print(f"    {r.author_id}: net_stance={r.net_stance:+.2f} consistency={r.stance_consistency:.2f} volume={r.volume} narrative_spread={r.narrative_spread} score={r.score:.3f}")

    profile_con.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-corpus analysis pipeline run (extraction + stance + merge + narratives + profiles)")
    parser.add_argument("--provider", choices=["anthropic", "gemini"], default="gemini", help="Provider for extraction + stance (default: gemini)")
    parser.add_argument("--max-cost", type=float, default=50.0, help="Shared USD cap across extraction + stance combined (default: 50.0)")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent API calls in flight, each phase (default: 8)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.provider == "gemini":
        api_key, model = config.get_gemini_api_key(), config.GEMINI_MODEL
        print(f"[run_full_pipeline] provider=gemini model={model} max_cost=${args.max_cost} concurrency={args.concurrency}")
        print("[run_full_pipeline] NOTE: paid-tier status cannot be checked ahead of time from just an API key -- ")
        print("  this will abort immediately (GeminiFreeTierError) the moment a free-tier quota error actually appears.")
    else:
        api_key, model = config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL
        print(f"[run_full_pipeline] provider=anthropic model={model} max_cost=${args.max_cost} concurrency={args.concurrency}")

    cost_tracker = CostTracker(provider=args.provider, max_cost=args.max_cost)

    try:
        extraction_stats = run_extraction_phase(args.provider, api_key, model, cost_tracker, args.concurrency)
        stance_stats = run_stance_phase(args.provider, api_key, model, cost_tracker, args.concurrency)
    except (GeminiFreeTierError, GeminiBillingExhaustedError, GeminiDailyQuotaExhaustedError) as e:
        print(f"\n[run_full_pipeline] ABORTED: {e}")
        return

    print_final_report(cost_tracker, extraction_stats, stance_stats)


if __name__ == "__main__":
    main()
