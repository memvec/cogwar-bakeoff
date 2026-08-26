"""Name -> id resolution for entities and authors.

Used both directly (GET /api/resolve, for the UI) and internally by
serve/ask.py's NLP layer: the local router model extracts plain name
strings from a question ("India", "Resonant News") -- it is never taught
the id vocabulary (thousands of entities/authors, not something that
belongs in a system prompt) -- and this module turns those strings into
real ids via the same lookups a human typing into a search box would get.

Exact canonical_name match wins outright for entities, so "India" resolves
to the country rather than "Supreme Court of India" or any other substring
match; otherwise falls back to an ILIKE substring search ranked by
observation_count/item_count DESC, so the most prominent match wins over an
obscure partial hit.
"""

from __future__ import annotations

import duckdb


def resolve_entity(con: duckdb.DuckDBPyConnection, name: str) -> dict | None:
    name = name.strip()
    if not name:
        return None

    row = con.execute(
        """
        SELECT entity_id, canonical_name FROM entities
        WHERE lower(canonical_name) = lower(?)
        ORDER BY observation_count DESC
        LIMIT 1
        """,
        [name],
    ).fetchone()
    matched_how = "exact"

    if row is None:
        row = con.execute(
            """
            SELECT entity_id, canonical_name FROM entities
            WHERE canonical_name ILIKE ?
            ORDER BY observation_count DESC
            LIMIT 1
            """,
            [f"%{name}%"],
        ).fetchone()
        matched_how = "substring"

    if row is None:
        return None
    return {"entity_id": row[0], "canonical_name": row[1], "matched_how": matched_how}


def resolve_author(con: duckdb.DuckDBPyConnection, name: str) -> dict | None:
    name = name.strip()
    if not name:
        return None

    if ":" in name:
        row = con.execute(
            """
            SELECT arg_max(author_display_name, published_at), any_value(source_type), count(*)
            FROM processed.items
            WHERE source_type || ':' || author_native_id = ?
            """,
            [name],
        ).fetchone()
        if row is not None and row[2] > 0:
            return {"author_id": name, "display_name": row[0], "source_type": row[1], "matched_how": "author_id"}

    candidates = con.execute(
        """
        SELECT source_type || ':' || author_native_id AS author_id,
               arg_max(author_display_name, published_at) AS display_name,
               source_type, count(*) AS item_count
        FROM processed.items
        WHERE author_display_name ILIKE ?
        GROUP BY source_type, author_native_id
        ORDER BY item_count DESC
        LIMIT 5
        """,
        [f"%{name}%"],
    ).fetchall()
    if not candidates:
        return None

    # Same display name can span multiple source_type/native_id pairs (e.g.
    # a Telegram channel's post stream vs. its channel-metadata record) --
    # prefer whichever candidate actually has a stance profile (real posts)
    # over a channel-metadata-only record, falling back to the highest
    # item_count when none of them do.
    candidate_ids = [r[0] for r in candidates]
    placeholders = ", ".join("?" for _ in candidate_ids)
    profiled = {
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT author_id FROM author_entity_profiles WHERE author_id IN ({placeholders})",
            candidate_ids,
        ).fetchall()
    }
    best = next((r for r in candidates if r[0] in profiled), candidates[0])
    return {"author_id": best[0], "display_name": best[1], "source_type": best[2], "matched_how": "name"}
