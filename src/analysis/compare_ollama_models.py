"""Local-model spot-check -- CLI entrypoint (all local, no API cost).

    uv run python -m analysis.compare_ollama_models --sample-size 30

Compares qwen2.5:7b-instruct and qwen2.5:3b-instruct (both via Ollama,
entities.OllamaEntityExtractor / stance.OllamaStanceDetector) against the
existing Gemini production baseline (entity_stance_edges where
detector_model LIKE 'gemini%'), on a sample of items that are BOTH already
Gemini-covered AND in the current on-topic analysis_scope
(analysis.build_scope --on-topic-only). Read-only against every production
table -- writes nothing to item_entities/entity_stance_edges/
extraction_cache; this is a diagnostic, not a pipeline pass.

Entity comparison: fresh extraction from each local model vs Gemini's RAW
extraction (extraction_cache.mentions_json for the item's text_hash) --
same fuzzy-match-then-Jaccard methodology as compare_providers.py, since
post-merge canonical names in item_entities aren't a fair apples-to-apples
target (merge.py's local-qwen merge pass already collapsed variants Gemini
itself never resolved together).

Stance comparison: asks each local model about the EXACT SAME (item,
entity) pairs Gemini already scored (the item's resolved entities from
item_entities/stance_storage.get_item_entities) -- same "ask the
challenger about the baseline's entities" methodology as
compare_providers.py, so it's a clean polarity-agreement percentage, not
confounded by different entity sets.

Memory: RSS of Ollama's llama-server runner process(es), sampled after each
model has processed a few items (steady state) -- Ollama loads one runner
subprocess per active model, separate from the `ollama serve` daemon.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import time

import duckdb
from rapidfuzz import fuzz

from analysis import config
from analysis.entities import OllamaEntityExtractor
from analysis.stance import EntityRef, OllamaStanceDetector

_FUZZY_MATCH_THRESHOLD = 70.0  # same family as merge.py/compare_providers.py -- "are these the same entity"
DEFAULT_SAMPLE_SIZE = 30
MODELS_TO_COMPARE = ["qwen2.5:7b-instruct", "qwen2.5:3b-instruct"]


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(config.ANALYSIS_DB_PATH))
    con.execute(f"ATTACH '{config.PROCESSED_DB_PATH}' AS processed (READ_ONLY)")
    return con


def select_sample(con: duckdb.DuckDBPyConnection, sample_size: int, seed: int) -> list[dict]:
    """Items with a Gemini stance baseline AND currently in the on-topic
    scope. Forces in every hard-multilingual (hi/ur, or non-latin/mixed
    script) candidate rather than leaving it to chance -- confirmed only 3
    exist in the whole intersection (2,437 items), so a plain random sample
    would very likely miss the multilingual case entirely.
    """
    rows = con.execute(
        """
        SELECT DISTINCT se.item_id, p.text, p.text_hash, p.language_detected, p.script, p.source_type,
               concat_ws(' ', p.text, p.source_specific -> 'transcript' ->> 'text') AS combined_text
        FROM entity_stance_edges se
        JOIN analysis_scope a ON a.item_id = se.item_id
        JOIN processed.items p ON p.item_id = se.item_id
        WHERE se.detector_model LIKE 'gemini%'
        """
    ).fetchall()
    columns = ["item_id", "text", "text_hash", "language_detected", "script", "source_type", "combined_text"]
    items = [dict(zip(columns, r, strict=True)) for r in rows]

    # Force-inclusion criterion is deliberately narrower than the "hard"
    # cut used for aggregate reporting elsewhere in this module (which also
    # counts "mixed" script): a huge share of YouTube titles/descriptions
    # get script='mixed' just from emoji/hashtags mixed into otherwise-
    # English text, which would swamp the forced sample with non-genuinely-
    # multilingual items. Forcing only on language_detected keeps this to
    # the 3 items that are actually Hindi/Urdu.
    hard = [it for it in items if it["language_detected"] in ("hi", "ur")]
    rest = [it for it in items if it not in hard]
    rng = random.Random(seed)
    rng.shuffle(rest)
    sample = hard + rest[: max(0, sample_size - len(hard))]
    return sample[:sample_size]


def get_gemini_raw_extraction(con: duckdb.DuckDBPyConnection, text_hash: str) -> list[str]:
    row = con.execute(
        "SELECT mentions_json FROM extraction_cache WHERE text_hash = ? AND model LIKE 'gemini%'", [text_hash]
    ).fetchone()
    if row is None or not row[0]:
        return []
    try:
        mentions = json.loads(row[0])
    except json.JSONDecodeError:
        return []
    return [str(m["canonical_name"]) for m in mentions if "canonical_name" in m]


def get_gemini_resolved_entities(con: duckdb.DuckDBPyConnection, item_id: str) -> list[EntityRef]:
    rows = con.execute(
        """
        SELECT ie.entity_id, e.canonical_name, ie.surface_form
        FROM item_entities ie JOIN entities e ON e.entity_id = ie.entity_id
        WHERE ie.item_id = ?
        QUALIFY row_number() OVER (PARTITION BY ie.entity_id ORDER BY ie.confidence DESC) = 1
        """,
        [item_id],
    ).fetchall()
    return [EntityRef(entity_id=r[0], canonical_name=r[1], surface_form=r[2]) for r in rows]


def get_gemini_stance(con: duckdb.DuckDBPyConnection, item_id: str) -> dict[str, str]:
    rows = con.execute(
        "SELECT entity_id, polarity FROM entity_stance_edges WHERE item_id = ? AND detector_model LIKE 'gemini%'",
        [item_id],
    ).fetchall()
    return dict(rows)


def _fuzzy_overlap(a_names: list[str], b_names: list[str]) -> tuple[int, list[tuple[str, str, float]]]:
    """Best-match pairing above _FUZZY_MATCH_THRESHOLD, each name used at most once."""
    remaining_b = list(b_names)
    matches = []
    for a in a_names:
        if not remaining_b:
            break
        best_idx, best_score = None, 0.0
        for idx, b in enumerate(remaining_b):
            score = fuzz.token_set_ratio(a, b)
            if score > best_score:
                best_idx, best_score = idx, score
        if best_idx is not None and best_score >= _FUZZY_MATCH_THRESHOLD:
            matches.append((a, remaining_b.pop(best_idx), best_score))
    return len(matches), matches


def llama_server_rss_mb() -> float:
    """Sum of RSS (MB) across every running Ollama llama-server runner
    process -- the actual model-loaded footprint, separate from the small
    `ollama serve` daemon and the `ollama run` CLI itself.
    """
    try:
        out = subprocess.run(
            ["ps", "-axo", "rss,comm"], capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return 0.0
    total_kb = 0
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and "llama-server" in parts[1]:
            try:
                total_kb += int(parts[0])
            except ValueError:
                continue
    return total_kb / 1024


def run_model_on_sample(
    con: duckdb.DuckDBPyConnection, model: str, sample: list[dict]
) -> dict:
    """Runs fresh extraction + stance-on-Gemini's-entities for every sampled
    item through `model`, timing each call and sampling llama-server RSS
    partway through (steady state, model already warm).
    """
    extractor = OllamaEntityExtractor(model=model)
    detector = OllamaStanceDetector(model=model)

    extract_times: list[float] = []
    stance_times: list[float] = []
    extraction_results: dict[str, list[str]] = {}
    stance_results: dict[str, dict[str, str]] = {}
    rss_samples: list[float] = []

    for i, item in enumerate(sample):
        context = {
            "language_detected": item["language_detected"],
            "script": item["script"],
            "source_type": item["source_type"],
        }

        t0 = time.monotonic()
        mentions = extractor.extract(item["combined_text"] or item["text"] or "", context=context)
        extract_times.append(time.monotonic() - t0)
        extraction_results[item["item_id"]] = [m.canonical_name for m in mentions]

        gemini_entities = get_gemini_resolved_entities(con, item["item_id"])
        t0 = time.monotonic()
        stance_out = detector.detect(item["combined_text"] or item["text"] or "", gemini_entities, context=context)
        stance_times.append(time.monotonic() - t0)
        stance_results[item["item_id"]] = {r.entity_id: r.polarity for r in stance_out}

        if i == max(2, len(sample) // 2):  # steady-state sample, model already warm
            rss_samples.append(llama_server_rss_mb())

        print(f"  [{model}] item {i + 1}/{len(sample)} -- extract={extract_times[-1]:.1f}s stance={stance_times[-1]:.1f}s", flush=True)

    return {
        "model": model,
        "extraction_results": extraction_results,
        "stance_results": stance_results,
        "mean_extract_seconds": statistics.mean(extract_times) if extract_times else 0.0,
        "mean_stance_seconds": statistics.mean(stance_times) if stance_times else 0.0,
        "rss_mb": statistics.mean(rss_samples) if rss_samples else llama_server_rss_mb(),
    }


def score_model(
    con: duckdb.DuckDBPyConnection, sample: list[dict], model_run: dict
) -> dict:
    per_item_jaccard = []
    stance_pairs = []
    hard_stance_pairs = []
    per_item_detail = {}

    for item in sample:
        item_id = item["item_id"]
        gemini_names = get_gemini_raw_extraction(con, item["text_hash"])
        local_names = model_run["extraction_results"].get(item_id, [])
        matched, pairs = _fuzzy_overlap(local_names, gemini_names)
        union_size = len(local_names) + len(gemini_names) - matched
        jaccard = matched / union_size if union_size else 1.0
        per_item_jaccard.append(jaccard)
        per_item_detail[item_id] = {
            "local_names": local_names,
            "gemini_names": gemini_names,
            "matched": pairs,
        }

        gemini_stance = get_gemini_stance(con, item_id)
        local_stance = model_run["stance_results"].get(item_id, {})
        is_hard = item["language_detected"] in ("hi", "ur") or item["script"] in ("devanagari", "arabic", "mixed")
        for entity_id, gemini_polarity in gemini_stance.items():
            local_polarity = local_stance.get(entity_id)
            if local_polarity is None:
                continue
            pair = (item_id, entity_id, gemini_polarity, local_polarity)
            stance_pairs.append(pair)
            if is_hard:
                hard_stance_pairs.append(pair)

    mean_jaccard = statistics.mean(per_item_jaccard) if per_item_jaccard else 0.0
    stance_agree = sum(1 for p in stance_pairs if p[2] == p[3])
    stance_agreement_pct = (stance_agree / len(stance_pairs) * 100) if stance_pairs else 0.0
    hard_agree = sum(1 for p in hard_stance_pairs if p[2] == p[3])
    hard_agreement_pct = (hard_agree / len(hard_stance_pairs) * 100) if hard_stance_pairs else None

    return {
        "model": model_run["model"],
        "mean_jaccard": mean_jaccard,
        "per_item_detail": per_item_detail,
        "n_stance_pairs": len(stance_pairs),
        "stance_agreement_pct": stance_agreement_pct,
        "n_hard_stance_pairs": len(hard_stance_pairs),
        "hard_agreement_pct": hard_agreement_pct,
        "mean_extract_seconds": model_run["mean_extract_seconds"],
        "mean_stance_seconds": model_run["mean_stance_seconds"],
        "rss_mb": model_run["rss_mb"],
    }


def print_summary_table(scores: list[dict]) -> None:
    print("\n--- Summary: local models vs Gemini baseline ---")
    header = f"{'model':<24}{'entity agree %':>16}{'stance agree %':>16}{'hard-multiling %':>18}{'RSS (MB)':>12}{'s/item (ext+stance)':>22}"
    print(header)
    for s in scores:
        hard_str = f"{s['hard_agreement_pct']:.1f}" if s["hard_agreement_pct"] is not None else "n/a"
        per_item_seconds = s["mean_extract_seconds"] + s["mean_stance_seconds"]
        print(
            f"{s['model']:<24}{s['mean_jaccard'] * 100:>15.1f}%{s['stance_agreement_pct']:>15.1f}%"
            f"{hard_str:>18}{s['rss_mb']:>12.0f}{per_item_seconds:>22.1f}"
        )
    print(f"{'gemini (baseline)':<24}{'--':>16}{'--':>16}{'--':>18}{'n/a (cloud)':>12}{'n/a (cloud)':>22}")


def print_hindi_urdu_example(con: duckdb.DuckDBPyConnection, sample: list[dict], scores: list[dict]) -> None:
    hard = [
        it for it in sample if it["language_detected"] in ("hi", "ur") or it["script"] in ("devanagari", "arabic", "mixed")
    ]
    if not hard:
        print("\n(no Hindi/Urdu item found in this sample)")
        return
    item = hard[0]
    item_id = item["item_id"]
    print(f"\n--- Hindi/Urdu example (item {item_id[:8]}..., lang={item['language_detected']}, script={item['script']}) ---")
    print(f"Text: {(item['combined_text'] or '')[:300]!r}")
    print(f"Gemini raw entities: {get_gemini_raw_extraction(con, item['text_hash'])}")
    for s in scores:
        detail = s["per_item_detail"].get(item_id, {})
        print(f"{s['model']} entities: {detail.get('local_names', [])}")

    gemini_stance = get_gemini_stance(con, item_id)
    print(f"Gemini stance: {gemini_stance}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Ollama model spot-check vs Gemini baseline")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=MODELS_TO_COMPARE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    con = connect()

    print("[compare_ollama_models] selecting sample (Gemini-covered AND in on-topic scope) ...", flush=True)
    sample = select_sample(con, args.sample_size, args.seed)
    n_hard = sum(
        1 for it in sample if it["language_detected"] in ("hi", "ur") or it["script"] in ("devanagari", "arabic", "mixed")
    )
    print(f"[compare_ollama_models] sample size: {len(sample)} ({n_hard} hard-multilingual)", flush=True)

    scores = []
    for model in args.models:
        print(f"\n[compare_ollama_models] running {model} on {len(sample)} items ...", flush=True)
        model_run = run_model_on_sample(con, model, sample)
        scores.append(score_model(con, sample, model_run))

    print_summary_table(scores)
    print_hindi_urdu_example(con, sample, scores)

    con.close()


if __name__ == "__main__":
    main()
