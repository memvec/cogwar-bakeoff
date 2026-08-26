"""Phase C: coordination-cluster report over the derived edges.

A "coordination cluster" is a connected component of items linked by
shared_media and/or near_duplicate_text derived edges -- content that
propagated across (possibly many) items, which becomes interesting when
it's also across multiple distinct authors/channels.

temporal_cocluster edges are deliberately excluded from the graph used
here: they're always a strict subset of shared_media/near_duplicate_text
pairs (see derive.py's derive_temporal_cocluster docstring), filtered
further by author + timing, so they can't introduce any connectivity that
isn't already present in the other two edge types.

Connected components aren't a natural fit for a single SQL query, but the
graph here is small by construction: derive.py's hash-group cap
(DEFAULT_HASH_GROUP_CAP) already excludes any hash shared by more than a
few hundred items before generating a single pairwise edge, so the total
edge count -- and therefore the number of distinct items even eligible to
be in a cluster -- stays bounded regardless of overall corpus size. That
makes a plain in-memory union-find over the (small) edge list fast and
simple; no need for an iterative/distributed graph algorithm, and no extra
dependency (the `process` group is duckdb-only, no pandas/pyarrow).
"""

from __future__ import annotations

import duckdb

CLUSTER_BASIS_EDGE_TYPES = ("shared_media", "near_duplicate_text")

_SIZE_BUCKETS: list[tuple[int, int | None]] = [
    (2, 2),
    (3, 5),
    (6, 10),
    (11, 20),
    (21, 50),
    (51, 100),
    (101, 200),
    (201, None),
]


def _find(parent: dict[str, str], x: str) -> str:
    parent.setdefault(x, x)
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def _union(parent: dict[str, str], a: str, b: str) -> None:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[ra] = rb


def compute_clusters(con: duckdb.DuckDBPyConnection) -> dict[str, list[str]]:
    """cluster_id -> item_ids, for every item touched by at least one
    shared_media/near_duplicate_text derived edge. Items with no such edge
    are absent (not part of any cluster). cluster_id is an arbitrary
    representative item_id (the union-find root) -- stable within one call,
    not meaningful across runs.
    """
    parent: dict[str, str] = {}
    edge_rows = con.execute(
        """
        SELECT DISTINCT src_item_id, dst_item_id
        FROM edges
        WHERE origin = 'derived' AND edge_type IN ('shared_media', 'near_duplicate_text')
        """
    ).fetchall()
    for src, dst in edge_rows:
        _union(parent, src, dst)

    clusters: dict[str, list[str]] = {}
    for item_id in parent:
        clusters.setdefault(_find(parent, item_id), []).append(item_id)
    return clusters


def _author_lookup(con: duckdb.DuckDBPyConnection, item_ids: list[str]) -> dict[str, tuple[str | None, str | None]]:
    """item_id -> (author_native_id, author_display_name), for exactly the (bounded) set of
    items that are in some cluster -- never the full items table.
    """
    rows = con.execute(
        """
        SELECT item_id, author_native_id, author_display_name
        FROM items
        WHERE item_id IN (SELECT UNNEST($ids))
        """,
        {"ids": item_ids},
    ).fetchall()
    return {item_id: (native_id, display_name) for item_id, native_id, display_name in rows}


def print_cluster_report(con: duckdb.DuckDBPyConnection, hash_group_cap: int, top_n: int = 10) -> None:
    print("\n--- Coordination cluster report (Phase C) ---")
    print("[clusters] computing connected components ...", flush=True)
    clusters = compute_clusters(con)
    if not clusters:
        print("No coordination clusters found (no shared_media/near_duplicate_text edges).")
        return

    all_item_ids = [item_id for members in clusters.values() for item_id in members]
    author_by_item = _author_lookup(con, all_item_ids)

    # (cluster_id, n_items, n_authors)
    cluster_stats: list[tuple[str, int, int]] = []
    for cluster_id, members in clusters.items():
        authors = {author_by_item.get(m, (None, None))[0] for m in members}
        authors.discard(None)
        cluster_stats.append((cluster_id, len(members), len(authors)))

    total_items_in_clusters = sum(n_items for _, n_items, _ in cluster_stats)
    multi_author = [c for c in cluster_stats if c[2] >= 2]

    print(f"Hash-group cap in effect upstream: {hash_group_cap} items (see mass-duplication report in Phase B)")
    print(f"Total clusters (connected components): {len(cluster_stats)}")
    print(f"Total items in ANY coordination cluster: {total_items_in_clusters}")
    print(f"Multi-author coordination clusters (>=2 distinct authors): {len(multi_author)}")

    print("\nCluster size distribution (by item count):")
    for lo, hi in _SIZE_BUCKETS:
        if hi is None:
            n = sum(1 for _, n_items, _ in cluster_stats if n_items >= lo)
            label = f"{lo}+"
        else:
            n = sum(1 for _, n_items, _ in cluster_stats if lo <= n_items <= hi)
            label = f"{lo}-{hi}"
        if n:
            print(f"  {label} items: {n} clusters")

    print(f"\nTop {top_n} coordination clusters by distinct-author count:")
    top_clusters = sorted(cluster_stats, key=lambda c: c[2], reverse=True)[:top_n]
    for cluster_id, n_items, n_authors in top_clusters:
        members = clusters[cluster_id]
        name_set = {
            (author_by_item.get(m, (None, None))[1] or author_by_item.get(m, (None, None))[0]) for m in members
        }
        names = sorted(name for name in name_set if name is not None)
        shown = names[:15]
        more = f" (+{len(names) - 15} more)" if len(names) > 15 else ""
        print(f"  cluster {cluster_id[:8]}...: {n_items} items, {n_authors} distinct authors -- {', '.join(shown)}{more}")
