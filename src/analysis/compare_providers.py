"""Anthropic vs Gemini head-to-head comparison harness -- VALIDATION ONLY.

    uv run python -m analysis.compare_providers --limit 50

Takes the SAME items already processed by Claude (the "cluster sample":
items that already have entity_stance_edges from analysis/detect_stance.py)
and re-runs entity extraction (analysis/entities.py) + stance detection
(analysis/stance.py) through Gemini on the exact same text -- writing
results into SEPARATE comparison_* tables, never touching item_entities /
entities / entity_stance_edges (Claude's production tables). Nothing here
feeds narratives.py or profiles.py; this is a one-off provider bake-off,
not a new pipeline pass.

Two different comparisons, two different mechanics:

  Entity extraction -- Claude and Gemini each propose their OWN entities
  independently (no shared ID space). "Overlap" is necessarily a fuzzy
  string match (rapidfuzz token_set_ratio) between Claude's current,
  post-merge canonical_name (item_entities + entities, i.e. after
  analysis/merge.py's fuzzy+LLM consolidation already ran) and Gemini's raw
  per-call canonical_name (comparison_entity_extractions) -- there is no
  exact key to join on, by construction.

  Stance -- deliberately NOT symmetric the same way. Gemini is asked for
  stance toward the EXACT SAME entities Claude's item_entities already
  resolved for that item (stance_storage.get_item_entities), so both
  providers' results key off the SAME entity_id. This makes stance
  comparison an exact (item_id, entity_id) join, isolating "do the two
  models agree on polarity/strength for a given entity" from "do the two
  models even agree on which entities exist" (that's the extraction
  comparison's job).

Incremental within this one-off harness: a (item_id, provider) pair already
in comparison_entity_extractions/comparison_stance_edges is not re-sent to
the API on a rerun -- reruns just regenerate the report from what's stored,
so iterating on the report format costs nothing further.
"""

from __future__ import annotations

import argparse
import statistics
from datetime import UTC, datetime

import duckdb
from rapidfuzz import fuzz

from analysis import config, entities, resolution, stance, stance_storage

# Gemini Flash pricing per ai.google.dev/gemini-api/docs/pricing (verified
# 2026-08, introductory rate through 2026-12-31) -- used only for the rough
# cost estimate/extrapolation printed at the end of the report, not for
# anything that affects what gets stored.
_GEMINI_INPUT_USD_PER_MTOK = 0.75
_GEMINI_OUTPUT_USD_PER_MTOK = 3.75

# Below this rapidfuzz token_set_ratio, two canonical_name strings (one from
# each provider) are not considered the same real-world entity for the
# overlap count -- same threshold family as merge.py's CANDIDATE_THRESHOLD,
# since this is answering the identical question ("are these the same
# entity") just across providers instead of across our own extraction calls.
_FUZZY_MATCH_THRESHOLD = 70.0

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS comparison_entity_extractions (
    item_id VARCHAR,
    provider VARCHAR,
    surface_form VARCHAR,
    entity_type_guess VARCHAR,
    canonical_name VARCHAR,
    confidence DOUBLE,
    extracted_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS comparison_stance_edges (
    item_id VARCHAR,
    entity_id VARCHAR,
    provider VARCHAR,
    polarity VARCHAR,
    strength DOUBLE,
    confidence DOUBLE,
    detected_at TIMESTAMP,
    PRIMARY KEY (item_id, entity_id, provider)
);
"""


class RunStats:
    def __init__(self) -> None:
        self.items_considered = 0
        self.extraction_calls = 0
        self.extraction_cache_hits = 0
        self.stance_calls = 0
        self.stance_cache_hits = 0
        self.stance_empty_items: list[str] = []
        self.input_tokens = 0
        self.output_tokens = 0


def connect(read_only_processed: bool = True) -> duckdb.DuckDBPyConnection:
    config.ANALYSIS_DATA_PATH.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.ANALYSIS_DB_PATH))
    ro = " (READ_ONLY)" if read_only_processed else ""
    con.execute(f"ATTACH '{config.PROCESSED_DB_PATH}' AS processed{ro}")
    con.execute(resolution.SCHEMA_SQL)
    stance_storage.init_schema(con)
    con.execute(SCHEMA_SQL)
    return con


def select_sample_items(con: duckdb.DuckDBPyConnection, limit: int) -> list[dict]:
    """The cluster sample: items Claude has already run stance detection on
    (entity_stance_edges), i.e. exactly the same set analysis/detect_stance.py
    validated earlier -- comparing Gemini against anything Claude hasn't
    touched wouldn't be a head-to-head."""
    rows = con.execute(
        """
        SELECT DISTINCT i.item_id, i.text, i.text_hash, i.language_detected, i.script, i.source_type
        FROM entity_stance_edges se
        JOIN processed.items i ON i.item_id = se.item_id
        WHERE i.text IS NOT NULL
        ORDER BY i.item_id
        LIMIT ?
        """,
        [limit],
    ).fetchall()
    columns = ["item_id", "text", "text_hash", "language_detected", "script", "source_type"]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def _track_usage(stats: RunStats, usage_tokens: tuple[int, int] | None) -> None:
    if usage_tokens is None:
        return
    input_tokens, output_tokens = usage_tokens
    stats.input_tokens += input_tokens
    stats.output_tokens += output_tokens


def run_gemini_extraction(
    con: duckdb.DuckDBPyConnection, extractor: entities.GeminiEntityExtractor, item: dict, stats: RunStats
) -> None:
    cached = con.execute(
        "SELECT 1 FROM comparison_entity_extractions WHERE item_id = ? AND provider = 'gemini' LIMIT 1",
        [item["item_id"]],
    ).fetchone()
    if cached is not None:
        stats.extraction_cache_hits += 1
        return

    context = {
        "language_detected": item["language_detected"],
        "script": item["script"],
        "source_type": item["source_type"],
    }
    mentions = extractor.extract(item["text"], context=context)
    stats.extraction_calls += 1
    _track_usage(stats, extractor.last_usage_tokens)

    now = datetime.now(UTC)
    con.executemany(
        "INSERT INTO comparison_entity_extractions VALUES (?, 'gemini', ?, ?, ?, ?, ?)",
        [
            (item["item_id"], m.surface_form, m.entity_type_guess, m.canonical_name, m.confidence, now)
            for m in mentions
        ],
    )


def run_gemini_stance(
    con: duckdb.DuckDBPyConnection,
    detector: stance.GeminiStanceDetector,
    item: dict,
    claude_entities: list[stance.EntityRef],
    stats: RunStats,
) -> None:
    if not claude_entities:
        return
    cached = con.execute(
        "SELECT 1 FROM comparison_stance_edges WHERE item_id = ? AND provider = 'gemini' LIMIT 1",
        [item["item_id"]],
    ).fetchone()
    if cached is not None:
        stats.stance_cache_hits += 1
        return

    context = {
        "language_detected": item["language_detected"],
        "script": item["script"],
        "source_type": item["source_type"],
    }
    results = detector.detect(item["text"], claude_entities, context=context)
    stats.stance_calls += 1
    _track_usage(stats, detector.last_usage_tokens)

    if not results:
        # Gemini returned nothing parseable for this item -- leave it
        # uncached (a rerun will retry it) rather than crash; flagged in
        # the report so it isn't silently absent from the stance comparison.
        stats.stance_empty_items.append(item["item_id"])
        return

    now = datetime.now(UTC)
    con.executemany(
        "INSERT INTO comparison_stance_edges VALUES (?, ?, 'gemini', ?, ?, ?, ?) ON CONFLICT DO NOTHING",
        [(item["item_id"], r.entity_id, r.polarity, r.strength, r.confidence, now) for r in results],
    )


def run(limit: int) -> tuple[RunStats, duckdb.DuckDBPyConnection]:
    con = connect()
    api_key = config.get_gemini_api_key()
    extractor = entities.GeminiEntityExtractor(api_key=api_key, model=config.GEMINI_MODEL)
    detector = stance.GeminiStanceDetector(api_key=api_key, model=config.GEMINI_MODEL)

    items = select_sample_items(con, limit)
    print(f"[compare_providers] cluster sample (Claude-processed items): {len(items)}", flush=True)

    stats = RunStats()
    for item in items:
        claude_entities = stance_storage.get_item_entities(con, item["item_id"])
        run_gemini_extraction(con, extractor, item, stats)
        run_gemini_stance(con, detector, item, claude_entities, stats)
        stats.items_considered += 1

    return stats, con


# --- Entity extraction comparison ---------------------------------------


def _fuzzy_overlap(claude_names: list[str], gemini_names: list[str]) -> tuple[int, list[tuple[str, str, float]]]:
    """Best-match pairing above _FUZZY_MATCH_THRESHOLD, each name used at
    most once. Returns (matched_count, [(claude_name, gemini_name, score), ...])."""
    remaining_gemini = list(gemini_names)
    matches = []
    for c in claude_names:
        if not remaining_gemini:
            break
        best_idx, best_score = None, 0.0
        for idx, g in enumerate(remaining_gemini):
            score = fuzz.token_set_ratio(c, g)
            if score > best_score:
                best_idx, best_score = idx, score
        if best_idx is not None and best_score >= _FUZZY_MATCH_THRESHOLD:
            matches.append((c, remaining_gemini.pop(best_idx), best_score))
    return len(matches), matches


def compare_entity_extraction(con: duckdb.DuckDBPyConnection, item_ids: list[str]) -> dict:
    per_item_jaccard = []
    per_item_detail = {}
    for item_id in item_ids:
        claude_names = [
            r[0]
            for r in con.execute(
                """
                SELECT DISTINCT e.canonical_name
                FROM item_entities ie JOIN entities e ON e.entity_id = ie.entity_id
                WHERE ie.item_id = ?
                """,
                [item_id],
            ).fetchall()
        ]
        gemini_names = [
            r[0]
            for r in con.execute(
                "SELECT DISTINCT canonical_name FROM comparison_entity_extractions WHERE item_id = ? AND provider = 'gemini'",
                [item_id],
            ).fetchall()
        ]
        matched, pairs = _fuzzy_overlap(claude_names, gemini_names)
        union_size = len(claude_names) + len(gemini_names) - matched
        jaccard = matched / union_size if union_size else 1.0
        per_item_jaccard.append(jaccard)
        per_item_detail[item_id] = {
            "claude_only": [c for c in claude_names if c not in {p[0] for p in pairs}],
            "gemini_only": [g for g in gemini_names if g not in {p[1] for p in pairs}],
            "matched": pairs,
            "claude_names": claude_names,
            "gemini_names": gemini_names,
        }
    mean_jaccard = statistics.mean(per_item_jaccard) if per_item_jaccard else 0.0
    return {"mean_jaccard": mean_jaccard, "per_item": per_item_detail}


def check_known_variant_collapse(con: duckdb.DuckDBPyConnection) -> None:
    """The two known fragmentation cases from earlier passes: does Gemini's
    OWN canonicalization already collapse them, the way Claude's raw output
    needed merge.py's fuzzy+LLM pass to fix up after the fact?"""
    print("\n--- Known variant-collapse cases ---")

    mirza_rows = con.execute(
        """
        SELECT DISTINCT canonical_name FROM comparison_entity_extractions
        WHERE provider = 'gemini' AND canonical_name ILIKE '%mirza%'
        """
    ).fetchall()
    print(f"Gemini canonical_name variants seen for the 'Ali Mirza' cluster: {[r[0] for r in mirza_rows]}")
    if len(mirza_rows) <= 1:
        print("  -> Gemini collapsed this to a single canonical form (or didn't extract it at all).")
    else:
        print("  -> Gemini ALSO fragments this across calls, same failure mode merge.py exists to fix for Claude.")

    kashmir_rows = con.execute(
        """
        SELECT DISTINCT canonical_name FROM comparison_entity_extractions
        WHERE provider = 'gemini' AND (canonical_name ILIKE '%kashmir%' OR canonical_name ILIKE '%india%')
        """
    ).fetchall()
    print(f"Gemini canonical_name variants seen for India/Kashmir: {[r[0] for r in kashmir_rows]}")


# --- Stance comparison ----------------------------------------------------


def compare_stance(con: duckdb.DuckDBPyConnection, item_ids: list[str]) -> dict:
    rows = con.execute(
        f"""
        SELECT
            c.item_id, c.entity_id, e.canonical_name,
            c.polarity AS claude_polarity, c.strength AS claude_strength,
            g.polarity AS gemini_polarity, g.strength AS gemini_strength,
            i.language_detected, i.script
        FROM entity_stance_edges c
        JOIN comparison_stance_edges g ON g.item_id = c.item_id AND g.entity_id = c.entity_id AND g.provider = 'gemini'
        JOIN entities e ON e.entity_id = c.entity_id
        JOIN processed.items i ON i.item_id = c.item_id
        WHERE c.item_id IN ({", ".join("?" for _ in item_ids) or "NULL"})
        """,
        item_ids,
    ).fetchall()
    columns = [
        "item_id", "entity_id", "canonical_name", "claude_polarity", "claude_strength",
        "gemini_polarity", "gemini_strength", "language_detected", "script",
    ]
    pairs = [dict(zip(columns, r, strict=True)) for r in rows]

    agree = sum(1 for p in pairs if p["claude_polarity"] == p["gemini_polarity"])
    agreement_pct = (agree / len(pairs) * 100) if pairs else 0.0

    hard_pairs = [
        p for p in pairs
        if p["language_detected"] in ("hi", "ur") or p["script"] in ("devanagari", "arabic", "mixed")
    ]
    hard_agree = sum(1 for p in hard_pairs if p["claude_polarity"] == p["gemini_polarity"])
    hard_agreement_pct = (hard_agree / len(hard_pairs) * 100) if hard_pairs else None

    return {
        "n_pairs": len(pairs),
        "agreement_pct": agreement_pct,
        "hard_n_pairs": len(hard_pairs),
        "hard_agreement_pct": hard_agreement_pct,
        "pairs": pairs,
    }


def print_report(con: duckdb.DuckDBPyConnection, stats: RunStats, item_ids: list[str]) -> None:
    print("\n--- Run stats ---")
    print(f"Items in cluster sample: {stats.items_considered}")
    print(f"Gemini extraction calls: {stats.extraction_calls} (cache hits: {stats.extraction_cache_hits})")
    print(f"Gemini stance calls: {stats.stance_calls} (cache hits: {stats.stance_cache_hits})")
    if stats.stance_empty_items:
        print(f"Items where Gemini stance returned NOTHING (failed, uncached, will retry next run): {stats.stance_empty_items}")

    total_calls = stats.extraction_calls + stats.stance_calls
    input_cost = stats.input_tokens / 1_000_000 * _GEMINI_INPUT_USD_PER_MTOK
    output_cost = stats.output_tokens / 1_000_000 * _GEMINI_OUTPUT_USD_PER_MTOK
    run_cost = input_cost + output_cost
    print(
        f"Gemini token usage this run: {stats.input_tokens} input, {stats.output_tokens} output "
        f"-> ${run_cost:.5f} at ${_GEMINI_INPUT_USD_PER_MTOK}/${_GEMINI_OUTPUT_USD_PER_MTOK} per Mtok "
        f"(ai.google.dev pricing, through 2026-12-31)"
    )
    if total_calls:
        row = con.execute("SELECT count(*) FROM processed.items").fetchone()
        assert row is not None
        full_corpus_size = row[0]
        per_item_cost = run_cost / stats.items_considered if stats.items_considered else 0.0
        print(
            f"Rough full-corpus extrapolation: {per_item_cost:.6f} USD/item x {full_corpus_size} items "
            f"= ${per_item_cost * full_corpus_size:.2f} (illustrative only -- real corpus text length varies; "
            "does not include Anthropic's cost for the same run)"
        )

    print("\n--- Entity extraction: Claude vs Gemini ---")
    extraction = compare_entity_extraction(con, item_ids)
    print(f"Mean per-item entity-set agreement (fuzzy Jaccard, threshold={_FUZZY_MATCH_THRESHOLD}): {extraction['mean_jaccard'] * 100:.1f}%")
    check_known_variant_collapse(con)

    print("\n--- Stance: Claude vs Gemini (same item_id, entity_id pairs) ---")
    stance_cmp = compare_stance(con, item_ids)
    print(f"(item, entity) pairs compared: {stance_cmp['n_pairs']}")
    print(f"Polarity agreement: {stance_cmp['agreement_pct']:.1f}%")
    if stance_cmp["hard_agreement_pct"] is not None:
        print(
            f"Polarity agreement on Hindi/Urdu/code-mixed items only "
            f"({stance_cmp['hard_n_pairs']} pairs): {stance_cmp['hard_agreement_pct']:.1f}%"
        )
    else:
        print("No Hindi/Urdu/code-mixed (item, entity) pairs in this sample to isolate.")

    print("\n--- Side-by-side examples ---")
    _print_side_by_side_examples(con, extraction, stance_cmp)


def _print_side_by_side_examples(con: duckdb.DuckDBPyConnection, extraction: dict, stance_cmp: dict) -> None:
    shown = 0
    stance_by_item: dict[str, list[dict]] = {}
    for p in stance_cmp["pairs"]:
        stance_by_item.setdefault(p["item_id"], []).append(p)

    # Prioritize: the known multi-entity (pro-Pakistan/anti-Afghanistan) item,
    # then Hindi/Urdu items, then whatever else has both extraction + stance data.
    priority_ids = ["0ba44421-c5c1-4162-a48a-c925fcb68ae6"]
    hi_ur_ids = [
        item_id for item_id, pairs in stance_by_item.items()
        if any(p["language_detected"] in ("hi", "ur") or p["script"] in ("devanagari", "arabic") for p in pairs)
    ]
    ordered_ids = priority_ids + [i for i in hi_ur_ids if i not in priority_ids] + [
        i for i in extraction["per_item"] if i not in priority_ids and i not in hi_ur_ids
    ]

    for item_id in ordered_ids:
        if shown >= 6:
            break
        detail = extraction["per_item"].get(item_id)
        item_pairs = stance_by_item.get(item_id, [])
        if detail is None:
            continue

        row = con.execute("SELECT text, language_detected, script FROM processed.items WHERE item_id = ?", [item_id]).fetchone()
        if row is None:
            continue
        text, lang, script = row
        print(f"\nitem_id={item_id} lang={lang!r} script={script!r}")
        print(f"  text: {(text or '')[:150]!r}")
        print(f"  Claude entities: {detail['claude_names']}")
        print(f"  Gemini entities: {detail['gemini_names']}")
        if item_pairs:
            print("  Stance (entity: claude -> gemini):")
            for p in item_pairs:
                marker = "" if p["claude_polarity"] == p["gemini_polarity"] else "  <-- DISAGREE"
                print(
                    f"    {p['canonical_name']!r}: "
                    f"{p['claude_polarity']}({p['claude_strength']:.2f}) -> "
                    f"{p['gemini_polarity']}({p['gemini_strength']:.2f}){marker}"
                )
        shown += 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anthropic vs Gemini head-to-head comparison (validation only, separate tables)"
    )
    parser.add_argument("--limit", type=int, default=50, help="Max items from the Claude cluster sample to compare")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stats, con = run(args.limit)
    item_ids = [
        r[0] for r in con.execute("SELECT DISTINCT item_id FROM comparison_entity_extractions").fetchall()
    ]
    print_report(con, stats, item_ids)
    con.close()


if __name__ == "__main__":
    main()
