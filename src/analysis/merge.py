"""Entity merge pass (docs/analysis_layer_spec.md follow-up to §4 pass 1).

    uv run python -m analysis.merge

Reconciles entities that fragment across separate records because the
extractor (analysis/entities.py) emitted a genuinely different
canonical_name string for the same real-world entity on different calls --
something resolution.py's case/whitespace normalize() cannot catch, since
the strings really do differ (e.g. "Ali Mirza" / "Engineer Muhammad Ali
Mirza" / "Muhammad Ali Mirza" -- three separate entity records for one
person).

Two tiers, calibrated against real data (see the docstrings on the
threshold constants below for the actual scores that drove these numbers):

  Tier 1 -- fuzzy, local, free. Candidate pairs found by best-of
  token_set_ratio (rapidfuzz) across each entity's {canonical_name} union
  {aliases} against the other's -- catches a short name being a token
  subset of a longer one (which is exactly the Ali Mirza pattern) as well
  as alias-only overlaps. Only same-entity_type pairs are ever considered.
  Auto-merge fires ONLY on near-identical canonical_name strings (plain
  character ratio, not token_set_ratio -- token_set_ratio alone scores
  "Modi" vs "Modi Motors" a perfect 100, which is exactly the kind of
  false-positive auto-merge this pass must not make). Real fragmentation
  cases in this project's data score well below the auto-merge bar on
  plain ratio, so they correctly fall through to Tier 2 rather than
  auto-merging -- this is intentional, not a gap: the LLM tier is the
  guard, per the design brief.

  Tier 2 -- LLM adjudication, cost-bearing but rare (only candidates in the
  ambiguous band reach it). Asks the model, given each entity's aliases and
  one real sample mention apiece, whether they're the same real-world
  entity. This is what catches "these are actually different people who
  happen to share a name fragment" -- something no string-similarity
  threshold can determine on its own.

Idempotent: every pair decision (merged or rejected) is persisted to
entity_merge_decisions. A merged pair can never be re-found (one side no
longer exists in `entities`); a rejected pair is skipped on sight on any
later run, so it is never re-sent to the LLM.
"""

from __future__ import annotations

import itertools
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import duckdb
from rapidfuzz import fuzz

from analysis import config

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entity_merge_decisions (
    pair_key VARCHAR PRIMARY KEY,
    entity_id_a VARCHAR,
    entity_id_b VARCHAR,
    decision VARCHAR,
    method VARCHAR,
    confidence DOUBLE,
    reason VARCHAR,
    decided_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS entity_merge_log (
    merge_id VARCHAR PRIMARY KEY,
    survivor_entity_id VARCHAR,
    survivor_canonical_name VARCHAR,
    merged_entity_id VARCHAR,
    merged_canonical_name VARCHAR,
    method VARCHAR,
    confidence DOUBLE,
    reason VARCHAR,
    merged_at TIMESTAMP
);
"""

# Candidate generation: below this best-cross-pair token_set_ratio, two
# entities aren't considered related at all. Calibrated so real adversarial
# pairs ("Ali Khan" vs "Imran Khan" = 66.7, "Fawad Chaudhry" vs "Pervez
# Chaudhry" = 72.7) still clear it and get a real LLM adjudication rather
# than being silently dropped -- false negatives here are worse than a few
# extra cheap LLM calls.
CANDIDATE_THRESHOLD = 65.0

# Auto-merge: plain (non-token) ratio of the two canonical_name strings.
# Deliberately conservative and deliberately NOT token_set_ratio -- the
# real Ali Mirza fragments score 50-80 on plain ratio (all below this bar),
# so they correctly go to Tier 2 rather than auto-merging. This tier only
# catches near-literal duplicates (stray punctuation/spacing beyond what
# normalize() already handles).
AUTO_MERGE_THRESHOLD = 93.0

ADJUDICATION_SYSTEM_PROMPT = """You are adjudicating entity resolution for a coordinated-\
inauthentic-behavior detection pipeline. You will be shown two entity records from a knowledge \
base -- each with a canonical name, known aliases, and one real example of a mention -- and must \
decide whether they refer to the SAME real-world entity.

Be conservative. Two different people who share a name or surname, a person versus an \
organization or campaign named after them, or two similarly-named but distinct places/things, \
are NOT the same entity and must not be merged. Only confirm a match if you are genuinely \
confident they are the exact same real-world entity, allowing for the aliases being in different \
languages/scripts, transliterations, honorifics, or partial-name forms of one another.

Respond with ONLY JSON, no other text: {"same_entity": true or false, "confidence": <float \
0.0-1.0>, "reasoning": "<one sentence>"}"""


@dataclass
class EntityRecord:
    entity_id: str
    canonical_name: str
    entity_type: str
    observation_count: int
    created_from: str
    aliases: list[str]


class MergeStats:
    def __init__(self) -> None:
        self.before_count = 0
        self.after_count = 0
        self.candidates = 0
        self.auto_merged = 0
        self.sent_to_llm = 0
        self.llm_confirmed = 0
        self.llm_rejected = 0
        self.log: list[str] = []


def connect() -> duckdb.DuckDBPyConnection:
    config.ANALYSIS_DATA_PATH.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.ANALYSIS_DB_PATH))
    con.execute(f"ATTACH '{config.PROCESSED_DB_PATH}' AS processed (READ_ONLY)")
    con.execute(SCHEMA_SQL)
    return con


def load_entities(con: duckdb.DuckDBPyConnection) -> dict[str, EntityRecord]:
    rows = con.execute(
        "SELECT entity_id, canonical_name, entity_type, observation_count, created_from FROM entities"
    ).fetchall()
    alias_rows = con.execute("SELECT entity_id, surface_form FROM entity_aliases").fetchall()
    aliases_by_entity: dict[str, list[str]] = {}
    for entity_id, surface_form in alias_rows:
        aliases_by_entity.setdefault(entity_id, []).append(surface_form)

    return {
        entity_id: EntityRecord(
            entity_id=entity_id,
            canonical_name=canonical_name,
            entity_type=entity_type,
            observation_count=observation_count,
            created_from=created_from,
            aliases=aliases_by_entity.get(entity_id, []),
        )
        for entity_id, canonical_name, entity_type, observation_count, created_from in rows
    }


def make_pair_key(id_a: str, id_b: str) -> str:
    return "|".join(sorted([id_a, id_b]))


def fuzzy_score(a: EntityRecord, b: EntityRecord) -> float:
    """Best cross-pair token_set_ratio over {canonical_name} + aliases on each side -- catches alias-only overlaps, not just canonical-vs-canonical."""
    names_a = [a.canonical_name, *a.aliases]
    names_b = [b.canonical_name, *b.aliases]
    return max(fuzz.token_set_ratio(x, y) for x in names_a for y in names_b)


def find_candidate_pairs(
    con: duckdb.DuckDBPyConnection, entities: dict[str, EntityRecord]
) -> list[tuple[EntityRecord, EntityRecord, float]]:
    decided = {
        row[0] for row in con.execute("SELECT pair_key FROM entity_merge_decisions").fetchall()
    }
    candidates = []
    for a, b in itertools.combinations(entities.values(), 2):
        if a.entity_type != b.entity_type:
            continue
        if make_pair_key(a.entity_id, b.entity_id) in decided:
            continue
        score = fuzzy_score(a, b)
        if score >= CANDIDATE_THRESHOLD:
            candidates.append((a, b, score))
    candidates.sort(key=lambda c: c[2], reverse=True)
    return candidates


def pick_survivor(a: EntityRecord, b: EntityRecord) -> tuple[EntityRecord, EntityRecord]:
    """The more complete/frequent record survives: higher observation_count first, more aliases as tiebreak."""
    score_a = (a.observation_count, len(a.aliases))
    score_b = (b.observation_count, len(b.aliases))
    return (a, b) if score_a >= score_b else (b, a)


def get_sample_mention(con: duckdb.DuckDBPyConnection, entity_id: str) -> str | None:
    row = con.execute(
        """
        SELECT p.text FROM item_entities ie
        JOIN processed.items p ON ie.item_id = p.item_id
        WHERE ie.entity_id = ?
        LIMIT 1
        """,
        [entity_id],
    ).fetchone()
    if row is None or not row[0]:
        return None
    return row[0][:200]


def adjudicate(
    client, model: str, con: duckdb.DuckDBPyConnection, a: EntityRecord, b: EntityRecord
) -> tuple[bool, float, str]:
    """Ask the LLM whether `a` and `b` are the same real-world entity. Malformed output is treated as a rejection (fail closed, never auto-merge on a parse failure)."""
    prompt = (
        f"Entity A:\n  canonical_name: {a.canonical_name}\n  aliases: {a.aliases}\n"
        f'  sample mention: "{get_sample_mention(con, a.entity_id)}"\n\n'
        f"Entity B:\n  canonical_name: {b.canonical_name}\n  aliases: {b.aliases}\n"
        f'  sample mention: "{get_sample_mention(con, b.entity_id)}"'
    )
    response = client.messages.create(
        model=model,
        max_tokens=300,
        system=ADJUDICATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(block.text for block in response.content if block.type == "text").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    try:
        parsed = json.loads(raw)
        return bool(parsed["same_entity"]), float(parsed["confidence"]), str(parsed.get("reasoning", ""))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False, 0.0, "adjudication response unparseable -- fail closed, treated as reject"


def record_decision(
    con: duckdb.DuckDBPyConnection,
    a: EntityRecord,
    b: EntityRecord,
    decision: str,
    method: str,
    confidence: float,
    reason: str,
) -> None:
    con.execute(
        "INSERT INTO entity_merge_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
        [
            make_pair_key(a.entity_id, b.entity_id),
            a.entity_id,
            b.entity_id,
            decision,
            method,
            confidence,
            reason,
            datetime.now(UTC),
        ],
    )


def merge_entities(
    con: duckdb.DuckDBPyConnection,
    survivor: EntityRecord,
    merged: EntityRecord,
    method: str,
    confidence: float,
    reason: str,
) -> None:
    """Repoint entity_aliases + item_entities to the survivor, union created_from (seed wins), delete the merged record. Logged to entity_merge_log."""
    now = datetime.now(UTC)

    if merged.created_from == "seed" and survivor.created_from != "seed":
        con.execute("UPDATE entities SET created_from = 'seed' WHERE entity_id = ?", [survivor.entity_id])

    # entity_aliases PK is surface_form_normalized alone -- repointing entity_id
    # on existing rows can't collide with anything, a plain UPDATE is safe.
    con.execute(
        "UPDATE entity_aliases SET entity_id = ? WHERE entity_id = ?",
        [survivor.entity_id, merged.entity_id],
    )

    # item_entities PK is (item_id, entity_id, surface_form) -- a plain UPDATE
    # could collide if the survivor already has a row with the same
    # (item_id, surface_form). INSERT ... SELECT with ON CONFLICT DO NOTHING
    # dedupes safely, then the old rows are removed.
    con.execute(
        """
        INSERT INTO item_entities (item_id, entity_id, surface_form, confidence, extracted_at)
        SELECT item_id, ?, surface_form, confidence, extracted_at
        FROM item_entities WHERE entity_id = ?
        ON CONFLICT DO NOTHING
        """,
        [survivor.entity_id, merged.entity_id],
    )
    con.execute("DELETE FROM item_entities WHERE entity_id = ?", [merged.entity_id])

    con.execute("DELETE FROM entities WHERE entity_id = ?", [merged.entity_id])

    con.execute(
        "INSERT INTO entity_merge_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            str(uuid.uuid4()),
            survivor.entity_id,
            survivor.canonical_name,
            merged.entity_id,
            merged.canonical_name,
            method,
            confidence,
            reason,
            now,
        ],
    )


def run(client=None, model: str | None = None) -> tuple[MergeStats, duckdb.DuckDBPyConnection]:
    con = connect()
    model = model or config.ANTHROPIC_MODEL

    entities = load_entities(con)

    candidates = find_candidate_pairs(con, entities)
    stats = MergeStats()
    stats.before_count = len(entities)
    stats.candidates = len(candidates)

    live_ids = set(entities.keys())  # tracks what's still alive as we merge within this run

    for a, b, score in candidates:
        if a.entity_id not in live_ids or b.entity_id not in live_ids:
            continue  # one side already merged away earlier in this same run

        name_ratio = fuzz.ratio(a.canonical_name, b.canonical_name)

        if name_ratio >= AUTO_MERGE_THRESHOLD:
            survivor, merged = pick_survivor(a, b)
            reason = f"canonical_name ratio={name_ratio:.1f}, fuzzy candidate score={score:.1f}"
            merge_entities(con, survivor, merged, "fuzzy-auto", name_ratio / 100, reason)
            record_decision(con, a, b, "merged", "fuzzy-auto", name_ratio / 100, reason)
            live_ids.discard(merged.entity_id)
            stats.auto_merged += 1
            line = f"[merge] AUTO-MERGED (ratio={name_ratio:.1f}): {merged.canonical_name!r} -> {survivor.canonical_name!r}"
            print(line, flush=True)
            stats.log.append(line)
            continue

        # Ambiguous band -- Tier 2.
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        stats.sent_to_llm += 1
        same, confidence, reasoning = adjudicate(client, model, con, a, b)

        if same:
            survivor, merged = pick_survivor(a, b)
            reason = f"LLM confirmed (fuzzy candidate score={score:.1f}): {reasoning}"
            merge_entities(con, survivor, merged, "llm-confirmed", confidence, reason)
            record_decision(con, a, b, "merged", "llm", confidence, reasoning)
            live_ids.discard(merged.entity_id)
            stats.llm_confirmed += 1
            line = (
                f"[merge] LLM-CONFIRMED (confidence={confidence:.2f}): "
                f"{merged.canonical_name!r} -> {survivor.canonical_name!r} -- {reasoning}"
            )
        else:
            record_decision(con, a, b, "rejected", "llm", confidence, reasoning)
            stats.llm_rejected += 1
            line = (
                f"[merge] LLM-REJECTED (confidence={confidence:.2f}): "
                f"{a.canonical_name!r} vs {b.canonical_name!r} (fuzzy candidate score={score:.1f}) -- {reasoning}"
            )
        print(line, flush=True)
        stats.log.append(line)

    row = con.execute("SELECT count(*) FROM entities").fetchone()
    assert row is not None
    stats.after_count = row[0]
    return stats, con


def main() -> None:
    stats, con = run()
    print("\n--- Merge summary ---")
    print(f"Entities before: {stats.before_count}")
    print(f"Merge candidates found: {stats.candidates}")
    print(f"Auto-merged (fuzzy): {stats.auto_merged}")
    print(f"Sent to LLM: {stats.sent_to_llm}")
    print(f"LLM-confirmed merged: {stats.llm_confirmed}")
    print(f"LLM-rejected: {stats.llm_rejected}")
    print(f"Entities after: {stats.after_count}")
    con.close()


if __name__ == "__main__":
    main()
