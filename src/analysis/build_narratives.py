"""Narrative clustering pass -- CLI entrypoint (docs/analysis_layer_spec.md §4 pass 3).

    uv run python -m analysis.build_narratives

Builds TIGHT narratives only (connected components over near_duplicate_text
+ shared_media derived edges, analysis/narratives.py's TightNarrativeClusterer)
-- loose (semantic/embedding) narratives are a documented stub
(LooseNarrativeClusterer), not built here. Local + free: no API calls in
this pass at all.

Fully recomputed every run, like processing/derive.py's derived edges:
narrative_storage.clear_basis(con, "tight") deletes every prior tight
narrative (and its members / entity-stance rollup) before the freshly
computed set is inserted, so a rerun never duplicates or drifts from
whatever the current derived-edge graph says. This never touches
basis='loose' rows, once those exist.

After storing narratives + membership, computes narrative_entity_stance --
per narrative, per entity, how many member items were positive/negative/
neutral toward it and the mean strength -- from item_entities +
entity_stance_edges (both owned by the earlier extraction/stance passes;
this pass only reads them). A narrative whose members haven't all been
through entity extraction / stance detection yet still gets a row, just a
partial one -- this pass never blocks on those upstream passes finishing.
"""

from __future__ import annotations

import statistics

import duckdb

from analysis import config, narrative_storage, resolution, stance_storage
from analysis.narratives import TightNarrativeClusterer

_TIGHT_EDGE_TYPES_SQL = "'near_duplicate_text', 'shared_media'"


def connect(read_only_processed: bool = True) -> duckdb.DuckDBPyConnection:
    config.ANALYSIS_DATA_PATH.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.ANALYSIS_DB_PATH))
    ro = " (READ_ONLY)" if read_only_processed else ""
    con.execute(f"ATTACH '{config.PROCESSED_DB_PATH}' AS processed{ro}")
    # entities/item_entities (extraction pass) and entity_stance_edges
    # (stance pass) are read by this pass, not owned by it -- ensure they
    # exist (idempotent CREATE TABLE IF NOT EXISTS) so a narrative build run
    # before either upstream pass still works, just with empty rollups.
    con.execute(resolution.SCHEMA_SQL)
    stance_storage.init_schema(con)
    narrative_storage.init_schema(con)
    return con


def load_graph_items(con: duckdb.DuckDBPyConnection) -> tuple[list[dict], list[dict]]:
    """The induced subgraph: only items that are an endpoint of at least one
    near_duplicate_text/shared_media derived edge, plus those edges. Items
    with zero such edges can't be in any tight narrative (see
    TightNarrativeClusterer's docstring), so there's no reason to load the
    full corpus here.
    """
    edge_rows = con.execute(
        f"""
        SELECT edge_id, edge_type, src_item_id, dst_item_id, origin
        FROM processed.edges
        WHERE origin = 'derived' AND edge_type IN ({_TIGHT_EDGE_TYPES_SQL})
        """
    ).fetchall()
    edges = [
        {"edge_id": r[0], "edge_type": r[1], "src_item_id": r[2], "dst_item_id": r[3], "origin": r[4]}
        for r in edge_rows
    ]

    item_ids = sorted({e["src_item_id"] for e in edges} | {e["dst_item_id"] for e in edges})
    if not item_ids:
        return [], edges

    placeholders = ", ".join("?" for _ in item_ids)
    item_rows = con.execute(
        f"""
        SELECT
            i.item_id, i.author_native_id, i.published_at, i.text,
            list(DISTINCT e.canonical_name) FILTER (WHERE e.canonical_name IS NOT NULL) AS entities
        FROM processed.items i
        LEFT JOIN item_entities ie ON ie.item_id = i.item_id
        LEFT JOIN entities e ON e.entity_id = ie.entity_id
        WHERE i.item_id IN ({placeholders})
        GROUP BY i.item_id, i.author_native_id, i.published_at, i.text
        """,
        item_ids,
    ).fetchall()
    items = [
        {"item_id": r[0], "author_native_id": r[1], "published_at": r[2], "text": r[3], "entities": r[4] or []}
        for r in item_rows
    ]
    return items, edges


def print_report(con: duckdb.DuckDBPyConnection) -> None:
    total = con.execute("SELECT count(*) FROM narratives WHERE basis = 'tight'").fetchone()
    assert total is not None
    print(f"\n--- Tight narrative summary ---\nN tight narratives: {total[0]}")

    sizes = [r[0] for r in con.execute("SELECT size FROM narratives WHERE basis = 'tight' ORDER BY size").fetchall()]
    if sizes:
        print(
            f"Size distribution: min={sizes[0]} p50={statistics.median(sizes):.0f} "
            f"mean={statistics.mean(sizes):.1f} max={sizes[-1]}"
        )
        buckets = {"2": 0, "3-5": 0, "6-10": 0, "11-50": 0, "51+": 0}
        for s in sizes:
            if s == 2:
                buckets["2"] += 1
            elif s <= 5:
                buckets["3-5"] += 1
            elif s <= 10:
                buckets["6-10"] += 1
            elif s <= 50:
                buckets["11-50"] += 1
            else:
                buckets["51+"] += 1
        print("  " + ", ".join(f"{k}: {v}" for k, v in buckets.items()))

    multi_author = con.execute(
        "SELECT count(*) FROM narratives WHERE basis = 'tight' AND distinct_authors > 1"
    ).fetchone()
    assert multi_author is not None
    print(f"N multi-author narratives (distinct_authors > 1): {multi_author[0]}")

    print("\n--- Top 5 tight narratives by distinct_authors ---")
    top = con.execute(
        """
        SELECT narrative_id, label, size, distinct_authors, time_range_start, time_range_end
        FROM narratives
        WHERE basis = 'tight'
        ORDER BY distinct_authors DESC, size DESC
        LIMIT 5
        """
    ).fetchall()

    for narrative_id, label, size, distinct_authors, start, end in top:
        print(f"\nnarrative_id={narrative_id}")
        print(f"  label: {label!r}")
        print(f"  size={size} distinct_authors={distinct_authors} time_range=({start}, {end})")

        stance_rows = con.execute(
            """
            SELECT e.canonical_name, nes.positive_count, nes.negative_count, nes.neutral_count, nes.mean_strength
            FROM narrative_entity_stance nes
            JOIN entities e ON e.entity_id = nes.entity_id
            WHERE nes.narrative_id = ?
            ORDER BY (nes.positive_count + nes.negative_count + nes.neutral_count) DESC
            LIMIT 5
            """,
            [narrative_id],
        ).fetchall()
        if not stance_rows:
            print("  dominant entities: (none -- members not yet through entity extraction)")
            continue
        print("  dominant entities (pos/neg/neutral, mean_strength):")
        for canonical_name, pos, neg, neu, mean_strength in stance_rows:
            strength_str = f"{mean_strength:.2f}" if mean_strength is not None else "n/a"
            print(f"    {canonical_name!r}: {pos}/{neg}/{neu} (mean_strength={strength_str})")


def run() -> duckdb.DuckDBPyConnection:
    con = connect()

    items, edges = load_graph_items(con)
    print(f"[build_narratives] items in tight-edge induced subgraph: {len(items)}", flush=True)
    print(f"[build_narratives] candidate tight edges: {len(edges)}", flush=True)

    clusters = TightNarrativeClusterer().cluster(items, edges)
    print(f"[build_narratives] tight narratives found: {len(clusters)}", flush=True)

    narrative_storage.clear_basis(con, "tight")
    for cluster in clusters:
        narrative_storage.store_narrative(con, cluster)
    narrative_storage.compute_entity_stance(con, "tight")

    return con


def main() -> None:
    con = run()
    print_report(con)
    con.close()


if __name__ == "__main__":
    main()
