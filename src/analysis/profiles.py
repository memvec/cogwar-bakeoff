"""Author entity-stance profile aggregation (docs/analysis_layer_spec.md §2.2, §4 pass 4).

Pure DuckDB aggregation over entity_stance_edges (analysis/stance_storage.py)
and narrative_members (analysis/narrative_storage.py) -- NO API calls, no
LLM in this pass at all. Deterministic: rebuild() recomputes every
(author, entity) row from scratch each run, exactly like
processing/derive.py's derived edges and build_narratives.py's tight
narratives -- a rerun after new stance edges land just reflects the current
state, never drifts or duplicates.

`author_id` is `{source_type}:{author_native_id}` -- there is no canonical
account-id table upstream (processing/load.py's `items` only carries the
native per-source id), and author_native_id is only guaranteed unique
WITHIN a source_type, not across them, so the composite is the correct
join key.

--- CRITICAL GUARD: consistency is not coordination -- read before editing ---
Everything in this module is an INDIVIDUAL property: one author's stance
toward one entity, aggregated across that author's own content only.
net_stance / stance_consistency / narrative_spread say "this author
persistently pushes a stance across topics" -- they say NOTHING about
whether other authors are doing the same thing in sync with them. That is
coordination (§2.4: shared content + temporal_cocluster synchrony across
DIFFERENT authors), a completely separate, not-yet-built pass that reads
different tables (edges of type temporal_cocluster) and answers a different
question.

Per §0 design principle 4 and the §5.2 false-positive guard: a lone
high-volume, high-consistency, high-narrative-spread author is a committed
individual holding a consistent opinion -- NOT evidence of coordination.
This module must never compute, store, or expose a single "suspicion" or
"risk" score that multiplies a stance-consistency signal by anything
cross-author. If a future pass wants to combine "persistent individual
stance" with "cross-author coordination", that combination belongs in the
finding-assembly pass (§2.4), as an explicit join of two separately-computed
signals with their own evidence -- never as a field on author_entity_profiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS author_entity_profiles (
    author_id VARCHAR,
    entity_id VARCHAR,
    net_stance DOUBLE,
    stance_consistency DOUBLE,
    volume INTEGER,
    positive_count INTEGER,
    negative_count INTEGER,
    neutral_count INTEGER,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    time_span_days DOUBLE,
    narrative_spread INTEGER,
    computed_at TIMESTAMP,
    PRIMARY KEY (author_id, entity_id)
);
"""

# net_stance: strength-weighted mean polarity. Each stance edge contributes
# +strength (positive), -strength (negative), or 0 (neutral) -- averaged
# over ALL of this author's stance edges toward this entity, so a mix of
# strong positive and strong negative edges correctly nets toward 0 (a
# wobbly stance), while consistent one-directional edges stay near +-1.
_NET_STANCE_EXPR = """
    avg(CASE se.polarity
        WHEN 'positive' THEN se.strength
        WHEN 'negative' THEN -se.strength
        ELSE 0.0
    END)
"""

_REBUILD_SQL = f"""
WITH edge_authors AS (
    -- One row per stance edge, joined to its item's author + timestamp.
    -- Items with no author_native_id can't be attributed to anyone and are
    -- excluded -- a handful of such stance edges simply never enter any
    -- author's profile, they aren't an error.
    SELECT
        se.item_id,
        se.entity_id,
        se.polarity,
        se.strength,
        i.source_type || ':' || i.author_native_id AS author_id,
        i.published_at
    FROM entity_stance_edges se
    JOIN processed.items i ON i.item_id = se.item_id
    WHERE i.author_native_id IS NOT NULL
),
agg AS (
    SELECT
        author_id,
        entity_id,
        count(*) AS volume,
        count(*) FILTER (WHERE polarity = 'positive') AS positive_count,
        count(*) FILTER (WHERE polarity = 'negative') AS negative_count,
        count(*) FILTER (WHERE polarity = 'neutral') AS neutral_count,
        {_NET_STANCE_EXPR} AS net_stance,
        min(published_at) AS first_seen,
        max(published_at) AS last_seen
    FROM edge_authors se
    GROUP BY author_id, entity_id
),
narrative_spread AS (
    -- Distinct narratives (any basis) containing an item where this author
    -- expressed stance toward this entity -- the cross-topic-persistence
    -- signal (§2.2), computed purely from this author's own items.
    SELECT ea.author_id, ea.entity_id, count(DISTINCT nm.narrative_id) AS narrative_spread
    FROM edge_authors ea
    JOIN narrative_members nm ON nm.item_id = ea.item_id
    GROUP BY ea.author_id, ea.entity_id
)
SELECT
    a.author_id,
    a.entity_id,
    a.net_stance,
    -- stance_consistency = 1 - normalized Shannon entropy of the
    -- pos/neg/neutral distribution. All-one-category -> entropy 0 ->
    -- consistency 1 (maximally consistent). Uniform 1/3-1/3-1/3 -> entropy
    -- at its max (ln 3) -> consistency 0 (maximally wobbly). Normalizing by
    -- ln(3) keeps this in [0, 1] regardless of which categories are present.
    1 - (
        -1.0 * (
            CASE WHEN a.positive_count = 0 THEN 0 ELSE (a.positive_count::DOUBLE / a.volume) * ln(a.positive_count::DOUBLE / a.volume) END +
            CASE WHEN a.negative_count = 0 THEN 0 ELSE (a.negative_count::DOUBLE / a.volume) * ln(a.negative_count::DOUBLE / a.volume) END +
            CASE WHEN a.neutral_count = 0 THEN 0 ELSE (a.neutral_count::DOUBLE / a.volume) * ln(a.neutral_count::DOUBLE / a.volume) END
        ) / ln(3)
    ) AS stance_consistency,
    a.volume,
    a.positive_count,
    a.negative_count,
    a.neutral_count,
    a.first_seen,
    a.last_seen,
    date_diff('day', a.first_seen, a.last_seen)::DOUBLE AS time_span_days,
    coalesce(ns.narrative_spread, 0) AS narrative_spread,
    now() AS computed_at
FROM agg a
LEFT JOIN narrative_spread ns ON ns.author_id = a.author_id AND ns.entity_id = a.entity_id
"""


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA_SQL)


def rebuild(con: duckdb.DuckDBPyConnection) -> int:
    """Recompute every (author, entity) profile from the current
    entity_stance_edges + narrative_members. Full delete + insert -- cheap
    and deterministic at this data scale, and guarantees no stale row
    survives an entity merge or a stance edge correction upstream."""
    con.execute("DELETE FROM author_entity_profiles")
    con.execute(f"INSERT INTO author_entity_profiles {_REBUILD_SQL}")
    row = con.execute("SELECT count(*) FROM author_entity_profiles").fetchone()
    assert row is not None
    return row[0]


@dataclass
class RankedAuthor:
    author_id: str
    entity_id: str
    net_stance: float
    stance_consistency: float
    volume: int
    narrative_spread: int
    score: float


def query_entity_authors(
    con: duckdb.DuckDBPyConnection, entity_id: str, limit: int = 10, min_volume: int = 1
) -> tuple[list[RankedAuthor], list[RankedAuthor]]:
    """Output query #3 (§3.3): given an entity, who consistently pushes
    FOR it and who consistently pushes AGAINST it, ranked by
    |net_stance| x stance_consistency x ln(volume) x narrative_spread --
    magnitude, consistency, enough data points to trust, and persistence
    across topics, all rewarded together. By construction this scores 0
    for any author with volume=1 (ln(1)=0) or narrative_spread=0 (their
    single/isolated observation doesn't demonstrate a persistent
    cross-narrative agenda yet, whatever its consistency) -- returns
    (consistently_positive, consistently_negative), each already sorted
    best-first, empty when nothing clears that bar.

    `min_volume` (default 1, i.e. no additional filtering beyond the
    existing volume > 0) excludes authors below the threshold outright,
    rather than relying on the score alone: a volume=1, consistency=1.0
    author scores exactly 0 (ln(1)=0) same as every other zero-score
    author, so when fewer than `limit` authors clear a positive score, ties
    at score=0 fill the remaining slots in arbitrary (query-plan) order --
    surfacing one-post flukes as "top actors". Raising min_volume removes
    them from the candidate set entirely instead of relying on a tie-break.
    """
    rows = con.execute(
        """
        SELECT
            author_id, entity_id, net_stance, stance_consistency, volume, narrative_spread,
            abs(net_stance) * stance_consistency * ln(volume) * narrative_spread AS score
        FROM author_entity_profiles
        WHERE entity_id = ? AND volume >= ?
        ORDER BY score DESC
        """,
        [entity_id, min_volume],
    ).fetchall()
    ranked = [RankedAuthor(*r) for r in rows]
    positive = [r for r in ranked if r.net_stance > 0][:limit]
    negative = [r for r in ranked if r.net_stance < 0][:limit]
    return positive, negative


def query_author_profile(con: duckdb.DuckDBPyConnection, author_id: str) -> list[dict]:
    """Output query #4 (§3.4): one author's full stance vector -- every
    entity they've expressed stance toward, with volume/consistency/
    narrative_spread/time span per entity. Callers join canonical_name
    themselves (this module doesn't import entities.py's tables to keep
    the query generic over whatever the caller wants to display).
    """
    rows = con.execute(
        """
        SELECT
            entity_id, net_stance, stance_consistency, volume,
            positive_count, negative_count, neutral_count,
            first_seen, last_seen, time_span_days, narrative_spread
        FROM author_entity_profiles
        WHERE author_id = ?
        ORDER BY volume DESC
        """,
        [author_id],
    ).fetchall()
    columns = [
        "entity_id", "net_stance", "stance_consistency", "volume",
        "positive_count", "negative_count", "neutral_count",
        "first_seen", "last_seen", "time_span_days", "narrative_spread",
    ]
    return [dict(zip(columns, row, strict=True)) for row in rows]
