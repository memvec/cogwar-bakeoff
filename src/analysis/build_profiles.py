"""Author entity-stance profile pass -- CLI entrypoint (docs/analysis_layer_spec.md §4 pass 4).

    uv run python -m analysis.build_profiles

Pure DuckDB aggregation over entity_stance_edges + narrative_members into
author_entity_profiles (analysis/profiles.py) -- no API calls. Fully
recomputed every run, like build_narratives.py's tight narratives: a rerun
after more stance edges land just reflects the current state.

See profiles.py's module docstring for the hard rule this pass exists
under: these are INDIVIDUAL persistent-stance profiles, never to be
combined with cross-author coordination signals into a single score.
"""

from __future__ import annotations

import duckdb

from analysis import config, profiles


def connect(read_only_processed: bool = True) -> duckdb.DuckDBPyConnection:
    config.ANALYSIS_DATA_PATH.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.ANALYSIS_DB_PATH))
    ro = " (READ_ONLY)" if read_only_processed else ""
    con.execute(f"ATTACH '{config.PROCESSED_DB_PATH}' AS processed{ro}")
    profiles.init_schema(con)
    return con


def _display_name(con: duckdb.DuckDBPyConnection, author_id: str) -> str:
    source_type, _, native_id = author_id.partition(":")
    row = con.execute(
        """
        SELECT author_display_name FROM processed.items
        WHERE source_type = ? AND author_native_id = ? AND author_display_name IS NOT NULL
        LIMIT 1
        """,
        [source_type, native_id],
    ).fetchone()
    return row[0] if row else "(no display name)"


def _canonical_name(con: duckdb.DuckDBPyConnection, entity_id: str) -> str:
    row = con.execute("SELECT canonical_name FROM entities WHERE entity_id = ?", [entity_id]).fetchone()
    return row[0] if row else entity_id


def print_query3_example(con: duckdb.DuckDBPyConnection) -> None:
    print("\n--- Query #3: given an entity, who consistently pushes for/against it ---")

    row = con.execute(
        """
        SELECT ap.entity_id, e.canonical_name, count(DISTINCT ap.author_id) AS n_authors
        FROM author_entity_profiles ap
        JOIN entities e ON e.entity_id = ap.entity_id
        WHERE e.entity_type = 'country'
        GROUP BY ap.entity_id, e.canonical_name
        ORDER BY n_authors DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        print("  No country-type entity has any author profile yet -- coverage is too thin to demo query #3.")
        return
    entity_id, canonical_name, n_authors = row
    print(f"Entity: {canonical_name!r} ({n_authors} authors with a profile toward it)")

    positive, negative = profiles.query_entity_authors(con, entity_id, limit=5)

    print("\n  Consistently POSITIVE (ranked by |net_stance| x consistency x ln(volume) x narrative_spread):")
    if not positive:
        print("    (none clear the ranking bar for this entity -- all scored 0, see note below)")
    for r in positive:
        print(
            f"    {_display_name(con, r.author_id)!r} ({r.author_id}): net_stance={r.net_stance:+.2f} "
            f"consistency={r.stance_consistency:.2f} volume={r.volume} narrative_spread={r.narrative_spread} "
            f"score={r.score:.3f}"
        )

    print("\n  Consistently NEGATIVE:")
    if not negative:
        print("    (none clear the ranking bar for this entity -- no negative-net-stance author has volume>1 and narrative_spread>0 yet)")
    for r in negative:
        print(
            f"    {_display_name(con, r.author_id)!r} ({r.author_id}): net_stance={r.net_stance:+.2f} "
            f"consistency={r.stance_consistency:.2f} volume={r.volume} narrative_spread={r.narrative_spread} "
            f"score={r.score:.3f}"
        )

    if not positive and not negative:
        print(
            "\n  NOTE: the ranking score is 0 whenever volume=1 (ln(1)=0) or narrative_spread=0 (by "
            "design -- see profiles.query_entity_authors docstring). With stance detection covering "
            "only ~52 items so far, most authors have too few observations toward any single entity "
            "to clear that bar yet. Showing raw per-author rows for this entity instead:"
        )
        raw = con.execute(
            """
            SELECT author_id, net_stance, stance_consistency, volume, narrative_spread
            FROM author_entity_profiles WHERE entity_id = ? ORDER BY volume DESC
            """,
            [entity_id],
        ).fetchall()
        for author_id, net_stance, consistency, volume, spread in raw:
            print(
                f"    {_display_name(con, author_id)!r} ({author_id}): net_stance={net_stance:+.2f} "
                f"consistency={consistency:.2f} volume={volume} narrative_spread={spread}"
            )


def print_query4_example(con: duckdb.DuckDBPyConnection) -> None:
    print("\n--- Query #4: given an author, their full stance vector ---")

    row = con.execute(
        """
        SELECT author_id, count(DISTINCT entity_id) AS n_entities, sum(volume) AS total_volume
        FROM author_entity_profiles
        GROUP BY author_id
        ORDER BY n_entities DESC, total_volume DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        print("  No author profiles exist yet.")
        return
    author_id, n_entities, total_volume = row
    print(f"Author: {_display_name(con, author_id)!r} ({author_id}) -- {n_entities} entities, {total_volume} total stance edges")

    for p in profiles.query_author_profile(con, author_id):
        print(
            f"    {_canonical_name(con, p['entity_id'])!r}: net_stance={p['net_stance']:+.2f} "
            f"consistency={p['stance_consistency']:.2f} volume={p['volume']} "
            f"(+{p['positive_count']}/-{p['negative_count']}/={p['neutral_count']}) "
            f"narrative_spread={p['narrative_spread']} "
            f"span=[{p['first_seen']} .. {p['last_seen']}]"
        )


def main() -> None:
    con = connect()
    n_profiles = profiles.rebuild(con)
    print(f"[build_profiles] author_entity_profiles rebuilt: {n_profiles} rows", flush=True)

    n_authors = con.execute("SELECT count(DISTINCT author_id) FROM author_entity_profiles").fetchone()[0]  # type: ignore[index]
    n_entities = con.execute("SELECT count(DISTINCT entity_id) FROM author_entity_profiles").fetchone()[0]  # type: ignore[index]
    print(f"[build_profiles] distinct authors: {n_authors}, distinct entities covered: {n_entities}")

    print_query3_example(con)
    print_query4_example(con)

    con.close()


if __name__ == "__main__":
    main()
