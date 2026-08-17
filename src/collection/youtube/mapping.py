"""YouTube Data API v3 object -> canonical Item/Edge/Observation mapping.

All YouTube-specific parsing lives here; collector.py orchestrates auth,
search/batch-fetch calls, quota tracking, and writing.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime

from collection.enrich import extract_plaintext_entities
from collection.schema import (
    Edge,
    EdgeOrigin,
    EdgeType,
    FetchMethod,
    Item,
    Observation,
    Provenance,
    SourceType,
)

COLLECTOR_ID = "youtube_v1"

_DURATION_RE = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")
_YOUTUBE_VIDEO_RE = re.compile(r"(?:youtu\.be/|youtube\.com/watch\?v=)([\w-]{6,})")
_YOUTUBE_CHANNEL_RE = re.compile(r"youtube\.com/channel/([\w-]+)")

# youtube video_id -> item_id / youtube channel_id -> item_id, for in-run
# resolution of description-link targets. Populated by collector.py as it
# builds each video/channel Item.
VideoLookup = dict[str, uuid.UUID]
ChannelLookup = dict[str, uuid.UUID]


def _provenance(run_id: str, raw_payload_ref: str, collected_at: datetime) -> Provenance:
    return Provenance(
        collector_id=COLLECTOR_ID,
        collected_at=collected_at,
        fetch_method=FetchMethod.api,
        collection_run_id=run_id,
        raw_payload_ref=raw_payload_ref,
    )


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _int_or_none(value: object) -> int | None:
    return int(value) if value is not None else None  # type: ignore[call-overload]


def parse_duration(iso_duration: str | None) -> int | None:
    """ISO 8601 duration (e.g. 'PT4M13S') -> seconds."""
    if not iso_duration:
        return None
    m = _DURATION_RE.fullmatch(iso_duration)
    if not m:
        return None
    hours, minutes, seconds = (int(g) if g else 0 for g in m.groups())
    return hours * 3600 + minutes * 60 + seconds


def build_video_item(
    video: dict,
    channel_item_id: uuid.UUID,
    run_id: str,
    raw_payload_ref: str,
    collected_at: datetime,
) -> Item:
    """Map a `videos.list` resource into a canonical `youtube_video` Item (doc §1, §3)."""
    snippet = video.get("snippet", {})
    stats = video.get("statistics", {})
    content_details = video.get("contentDetails", {})
    title = snippet.get("title", "")
    description = snippet.get("description", "")

    return Item(
        source_type=SourceType.youtube_video,
        source_native_id=video["id"],
        parent_item_id=channel_item_id,
        text=f"{title}\n\n{description}".strip() or None,
        language_declared=snippet.get("defaultLanguage") or snippet.get("defaultAudioLanguage"),
        published_at=_parse_dt(snippet.get("publishedAt")),
        author_native_id=snippet.get("channelId"),
        author_display_name=snippet.get("channelTitle"),
        engagement={
            "view_count": _int_or_none(stats.get("viewCount")),
            "like_count": _int_or_none(stats.get("likeCount")),
            "comment_count": _int_or_none(stats.get("commentCount")),
            "favorite_count": _int_or_none(stats.get("favoriteCount")),
        },
        entities=extract_plaintext_entities(description),
        source_specific={
            "video_id": video["id"],
            "channel_id": snippet.get("channelId"),
            "tags": snippet.get("tags", []),
            "category_id": snippet.get("categoryId"),
            "duration_seconds": parse_duration(content_details.get("duration")),
            "definition": content_details.get("definition"),
            "is_live_broadcast": snippet.get("liveBroadcastContent", "none") != "none",
            "thumbnail_urls": {
                size: info.get("url") for size, info in snippet.get("thumbnails", {}).items()
            },
        },
        raw_payload_ref=raw_payload_ref,
        provenance=_provenance(run_id, raw_payload_ref, collected_at),
    )


def build_channel_item(
    channel: dict,
    run_id: str,
    raw_payload_ref: str,
    collected_at: datetime,
) -> Item:
    """Map a `channels.list` resource into a promoted channel node Item (doc §8.2, §9).

    Unlike Telegram, YouTube's API genuinely exposes channel creation date
    (snippet.publishedAt) -- account_created_at is a real source-declared
    value here, no approximation needed (contrast with
    collection.telegram.mapping.build_channel_item).
    """
    snippet = channel.get("snippet", {})
    return Item(
        source_type=SourceType.channel,
        source_native_id=channel["id"],
        text=snippet.get("description") or None,
        author_native_id=snippet.get("customUrl") or channel["id"],
        author_display_name=snippet.get("title"),
        account_created_at=_parse_dt(snippet.get("publishedAt")),
        source_specific={
            "channel_custom_url": snippet.get("customUrl"),
            "channel_country_declared": snippet.get("country"),
        },
        raw_payload_ref=raw_payload_ref,
        provenance=_provenance(run_id, raw_payload_ref, collected_at),
    )


def build_channel_observation(
    channel: dict,
    channel_item_id: uuid.UUID,
    run_id: str,
    observed_at: datetime,
) -> Observation:
    """Map current reputation metrics into a timestamped Observation row (doc §9) -- never onto the Item itself."""
    stats = channel.get("statistics", {})
    hidden = stats.get("hiddenSubscriberCount", False)
    return Observation(
        node_item_id=channel_item_id,
        observed_at=observed_at,
        subscriber_or_follower_count=None if hidden else _int_or_none(stats.get("subscriberCount")),
        view_count=_int_or_none(stats.get("viewCount")),
        post_count_seen=_int_or_none(stats.get("videoCount")),
        verified_status=None,  # not cleanly exposed by the public Data API
        collection_run_id=run_id,
    )


def build_mention_edges(
    video_item: Item,
    entities: dict,
    video_lookup: VideoLookup,
    channel_lookup: ChannelLookup,
    observed_at: datetime,
) -> list[Edge]:
    """@handles and description links -> mention Edges (doc §2 pattern, §6).

    Resolves internally to an already-collected video/channel from this run
    when the description link is a directly-ID-bearing YouTube URL
    (youtube.com/watch?v=, youtu.be/, youtube.com/channel/); anything else
    (custom-URL channel links, external sites, @handles) falls back to
    dst_external_ref, matching Telegram's mention-edge resolution pattern.
    YouTube has no forward-equivalent, so this is the only description-
    derived edge type here.
    """
    edges = []
    for handle in entities.get("mentions", []):
        edges.append(
            Edge(
                edge_type=EdgeType.mention,
                src_item_id=video_item.item_id,
                dst_external_ref=handle,
                observed_at=observed_at,
                origin=EdgeOrigin.collected,
                evidence={"mention_text": handle},
            )
        )
    for url in entities.get("urls", []):
        dst_item_id = None
        video_match = _YOUTUBE_VIDEO_RE.search(url)
        channel_match = _YOUTUBE_CHANNEL_RE.search(url)
        if video_match:
            dst_item_id = video_lookup.get(video_match.group(1))
        elif channel_match:
            dst_item_id = channel_lookup.get(channel_match.group(1))
        edges.append(
            Edge(
                edge_type=EdgeType.mention,
                src_item_id=video_item.item_id,
                dst_item_id=dst_item_id,
                dst_external_ref=None if dst_item_id else url,
                observed_at=observed_at,
                origin=EdgeOrigin.collected,
                evidence={"description_url": url},
            )
        )
    return edges
