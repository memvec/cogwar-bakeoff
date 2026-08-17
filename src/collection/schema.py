"""Pydantic models for the canonical `Item`, `Edge`, `Provenance`, and `Observation` schema.

Implements docs/collection_schema.md (LOCKED). Every collected object —
Telegram message, YouTube video/comment, news article, fact-check article,
and promoted author/channel nodes (doc §8.2) — normalizes into `Item`;
every relationship between items is an `Edge` (doc §6). Shared by all
source collectors (src/collection/telegram/, and future
src/collection/youtube/, src/collection/news/) — no source-specific code
belongs here.

Author/channel reputation fields are a TIME SERIES, not a snapshot (doc
§9). Volatile metrics (follower count, post count, verified status, ...)
are NEVER written onto the node — each collection run appends a new
`Observation` row instead, keyed on (node_item_id, observed_at). See
writers.write_observations for the append-only implementation and its
rationale.

text_normalized, language_detected, script, and content_hashes are filled
in automatically at construction time (Item.model_post_init, below) via the
shared enrichment step in enrich.py — no collector needs to compute these
itself.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field

from collection.enrich import enrich_item_fields


class SourceType(str, Enum):
    telegram = "telegram"
    youtube_video = "youtube_video"
    youtube_comment = "youtube_comment"
    news = "news"
    factcheck = "factcheck"
    channel = "channel"  # promoted author/channel node, doc §8.2
    author = "author"  # promoted author/channel node, doc §8.2


class FetchMethod(str, Enum):
    api = "api"
    rss = "rss"
    html_parse = "html_parse"


class Provenance(BaseModel):
    """Collection provenance block — mandatory on every Item (doc §5)."""

    collector_id: str
    collected_at: datetime
    source_api_version: str | None = None
    fetch_method: FetchMethod
    collection_run_id: str
    raw_payload_ref: str

    # Web-source fields (doc §5's fuller table) — unused by the Telegram
    # collector, kept optional here so news/factcheck collectors don't need
    # a second Provenance shape later.
    http_status: int | None = None
    robots_txt_compliant: bool | None = None
    source_terms_ref: str | None = None
    parser_version: str | None = None


class Item(BaseModel):
    """Canonical item — one row per collected object, or per promoted author/channel node (doc §1, §8.2)."""

    item_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_type: SourceType
    source_native_id: str
    parent_item_id: uuid.UUID | None = None
    text: str | None = None
    text_normalized: str | None = None
    language_detected: str | None = None
    language_declared: str | None = None
    script: str | None = None
    published_at: datetime | None = None
    edited_at: datetime | None = None
    author_native_id: str | None = None
    author_display_name: str | None = None
    engagement: dict[str, Any] = Field(default_factory=dict)
    media: list[dict[str, Any]] | None = None
    entities: dict[str, Any] = Field(default_factory=dict)
    source_specific: dict[str, Any] = Field(default_factory=dict)
    raw_payload_ref: str
    provenance: Provenance
    extraction_confidence: float | None = None
    content_hashes: dict[str, Any] = Field(default_factory=dict)

    # --- Author/channel-node-only fields (doc §9) ---
    # Populated only when this Item is a promoted author/channel node
    # (source_type in {channel, author}); null otherwise. Stable identity
    # data, not volatile reputation data — reputation goes in Observation
    # rows, never here. account_created_at is the core dummy-account
    # detection signal (a brand-new account posting at high volume).
    account_created_at: datetime | None = None

    def model_post_init(self, __context: Any, /) -> None:
        """Auto-fill text_normalized/language_detected/script/content_hashes from text+media (enrich.py) for every collector, without each one having to call it.

        Only fills fields still at their default (None / empty dict) — an
        explicitly-provided value is never overwritten, and re-parsing an
        already-enriched Item back from storage is a no-op (its
        content_hashes is already non-empty).
        """
        needs_enrichment = (
            self.text_normalized is None
            or self.language_detected is None
            or self.script is None
            or not self.content_hashes
        )
        if not needs_enrichment:
            return
        fields = enrich_item_fields(self.text, self.media)
        if self.text_normalized is None:
            self.text_normalized = fields["text_normalized"]
        if self.language_detected is None:
            self.language_detected = fields["language_detected"]
        if self.script is None:
            self.script = fields["script"]
        if not self.content_hashes:
            self.content_hashes = fields["content_hashes"]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def account_age_at_observation(self) -> float | None:
        """Account age in days: account_created_at -> this item's collection time (provenance.collected_at). None if account_created_at is unset (doc §9)."""
        if self.account_created_at is None:
            return None
        return (
            self.provenance.collected_at - self.account_created_at
        ).total_seconds() / 86400


class Observation(BaseModel):
    """One timestamped reputation snapshot for an author/channel node (doc §9). Appended per collection run, never overwritten — see writers.write_observations."""

    node_item_id: uuid.UUID  # the author/channel Item.item_id this observation is about
    observed_at: datetime
    subscriber_or_follower_count: int | None = None
    post_count_seen: int | None = None
    verified_status: bool | None = None
    collection_run_id: str


class EdgeType(str, Enum):
    forward = "forward"
    reply = "reply"
    mention = "mention"
    embed = "embed"
    debunks = "debunks"
    shared_media = "shared_media"
    shared_url = "shared_url"
    near_duplicate_text = "near_duplicate_text"
    co_author = "co_author"
    temporal_cocluster = "temporal_cocluster"


class EdgeOrigin(str, Enum):
    collected = "collected"
    derived = "derived"


class Edge(BaseModel):
    """Coordination graph edge — one row per relationship between items (doc §6)."""

    edge_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    edge_type: EdgeType
    src_item_id: uuid.UUID
    dst_item_id: uuid.UUID | None = None
    dst_external_ref: str | None = None
    directed: bool = True
    weight: float | None = None
    observed_at: datetime
    origin: EdgeOrigin
    evidence: dict[str, Any] = Field(default_factory=dict)
