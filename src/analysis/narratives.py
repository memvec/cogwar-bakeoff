"""Narrative clustering interface (docs/analysis_layer_spec.md §1.3, §4 pass 3).

NarrativeClusterer is the swap boundary -- same pattern as
entities.EntityExtractor / stance.StanceDetector: TightNarrativeClusterer is
built now (local, free, deterministic); LooseNarrativeClusterer is a stub
for the semantic-clustering implementation to slot in later without
restructuring anything downstream (storage, narrative_entity_stance
aggregation, the four output queries) -- they all consume NarrativeCluster
and key off `basis`, never off which clusterer produced it.

Per §1.3, tight and loose are deliberately two DIFFERENT signals answering
different questions (copy-paste network vs. paraphrased/thematically-aligned
agenda), both computed and stored side by side -- not a fallback chain. An
item can belong to a tight AND a loose narrative at once.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

BASES = ("tight", "loose")

# The two derived edge types that define "tight" per §1.3 -- exact
# duplicate/near-duplicate content, not semantic similarity.
TIGHT_EDGE_TYPES = ("near_duplicate_text", "shared_media")


@dataclass
class NarrativeCluster:
    narrative_id: str
    member_item_ids: list[str]
    label: str
    basis: str  # 'tight' | 'loose'
    time_range: tuple[datetime | None, datetime | None]
    size: int
    distinct_authors: int


class NarrativeClusterer(ABC):
    """Swap boundary for narrative clustering.

    `items`: one dict per candidate item, at minimum
        {"item_id": str, "author_native_id": str | None,
         "published_at": datetime | None, "text": str | None}.
    An item dict MAY also carry "entities": list[str] of canonical entity
    names already resolved for it (analysis/entities.py's output) -- purely
    optional enrichment implementations may use for a better label; a
    clusterer must not require it, since entity extraction is a separate,
    independently-incremental pass that may not have covered every item.

    `edges`: one dict per candidate edge, at minimum
        {"src_item_id": str, "dst_item_id": str, "edge_type": str,
         "origin": str}.
    """

    @abstractmethod
    def cluster(self, items: list[dict], edges: list[dict]) -> list[NarrativeCluster]:
        raise NotImplementedError


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _generate_label(member_items: list[dict], max_len: int = 80) -> str:
    """Cheap, local, no-API label for a tight cluster: most common resolved
    entity among members if any member carries entity data, else a
    representative item's text snippet (the longest text, on the theory
    that it's the most complete/informative version of the duplicated
    content).
    """
    entity_counts: Counter = Counter()
    for item in member_items:
        for name in item.get("entities") or []:
            entity_counts[name] += 1
    if entity_counts:
        top = [name for name, _ in entity_counts.most_common(2)]
        return " / ".join(top)

    texts = [_normalize_text(item["text"]) for item in member_items if item.get("text")]
    if not texts:
        return "(no text)"
    longest = max(texts, key=len)
    return longest[:max_len] + ("..." if len(longest) > max_len else "")


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


class TightNarrativeClusterer(NarrativeClusterer):
    """Connected components over near_duplicate_text + shared_media edges
    (origin='derived'), treated as undirected links regardless of the
    edge's own `directed` flag (per §1.3, this is a content-identity graph,
    not a directional one) -- each component is one "tight narrative" /
    content cluster.

    Only items that appear as an endpoint of at least one qualifying edge
    are ever placed into a component; an item with zero derived edges has
    nothing to be "tight" with and simply produces no narrative -- so every
    cluster returned has size >= 2 by construction. `items` not referenced
    by any qualifying edge are ignored (harmless to pass the full corpus).
    """

    def cluster(self, items: list[dict], edges: list[dict]) -> list[NarrativeCluster]:
        items_by_id = {item["item_id"]: item for item in items}

        uf = _UnionFind()
        connected_ids: set[str] = set()
        for edge in edges:
            if edge.get("origin") != "derived" or edge.get("edge_type") not in TIGHT_EDGE_TYPES:
                continue
            src, dst = edge["src_item_id"], edge["dst_item_id"]
            if src is None or dst is None or src not in items_by_id or dst not in items_by_id:
                continue
            uf.union(src, dst)
            connected_ids.add(src)
            connected_ids.add(dst)

        components: dict[str, list[str]] = {}
        for item_id in connected_ids:
            components.setdefault(uf.find(item_id), []).append(item_id)

        clusters = []
        for member_ids in components.values():
            member_ids.sort()
            member_items = [items_by_id[mid] for mid in member_ids]

            authors = {i["author_native_id"] for i in member_items if i.get("author_native_id")}
            timestamps = [i["published_at"] for i in member_items if i.get("published_at")]

            clusters.append(
                NarrativeCluster(
                    narrative_id=f"tight:{min(member_ids)}",
                    member_item_ids=member_ids,
                    label=_generate_label(member_items),
                    basis="tight",
                    time_range=(min(timestamps), max(timestamps)) if timestamps else (None, None),
                    size=len(member_ids),
                    distinct_authors=len(authors),
                )
            )
        return clusters


class LooseNarrativeClusterer(NarrativeClusterer):
    """NOT IMPLEMENTED -- hook for the semantic-clustering pass (§1.3).

    Planned approach: embed each item's text via the Anthropic/embeddings
    API (one call per item, cached by text_hash exactly like
    entities.py/stance.py's content caches so re-runs stay cheap), then
    group items whose embeddings exceed a cosine-similarity threshold
    (start conservative per §5 open question 3; validate against the tight
    clusters as a sanity check -- loose-but-not-tight = paraphrased/
    thematically-aligned content that never copy-pastes, which is exactly
    what tight clustering structurally cannot see).

    Returns the SAME NarrativeCluster shape as TightNarrativeClusterer,
    with basis="loose" -- narrative_storage.py, the narrative_entity_stance
    aggregation, and every downstream query already branch on `basis`, not
    on clusterer identity, so this slots in with zero changes elsewhere
    once implemented.
    """

    def cluster(self, items: list[dict], edges: list[dict]) -> list[NarrativeCluster]:
        raise NotImplementedError(
            "LooseNarrativeClusterer: semantic (embedding-based) clustering is not yet "
            "implemented -- see class docstring for the planned approach."
        )
