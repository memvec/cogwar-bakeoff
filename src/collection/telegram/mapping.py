"""Telethon object -> canonical Item/Edge/Observation mapping.

All Telegram-specific parsing lives here; collector.py orchestrates calls
into these functions and owns auth, iteration, rate-limiting, and writing.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from telethon.errors import RPCError
from telethon.tl.types import (
    Channel,
    ChannelFull,
    Message,
    MessageEntityHashtag,
    MessageEntityMention,
    MessageEntityMentionName,
    MessageEntityTextUrl,
    MessageEntityUrl,
    MessageMediaDocument,
    MessageMediaPhoto,
)

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

COLLECTOR_ID = "telegram_v1"

# (channel_id, message_id) -> item_id, for in-run resolution of forward/reply
# targets. Populated by collector.py as it iterates each channel's history.
ItemLookup = dict[tuple[int, int], uuid.UUID]


def _provenance(run_id: str, raw_payload_ref: str, collected_at: datetime) -> Provenance:
    return Provenance(
        collector_id=COLLECTOR_ID,
        collected_at=collected_at,
        fetch_method=FetchMethod.api,
        collection_run_id=run_id,
        raw_payload_ref=raw_payload_ref,
    )


async def approximate_channel_created_at(client, entity) -> tuple[datetime | None, float]:
    """Best-effort estimate of a channel's creation date via its first message's timestamp.

    PLATFORM LIMITATION, NOT A BUG: Telegram's API exposes no channel
    creation-date field anywhere (not on Channel, not on ChannelFull) — so
    Item.account_created_at stays None for every Telegram channel item;
    see build_channel_item below. This function instead produces a
    best-effort estimate (the `date` of the earliest message we can still
    see: message id 1 if not deleted, else whatever the oldest surviving
    message is) that the caller stores under source_specific, clearly
    labelled as an estimate rather than the real field. Returns
    (date, confidence); date is None if the channel has no retrievable
    messages at all.
    """
    try:
        first = await client.get_messages(entity, ids=1)
    except RPCError:
        first = None
    if first is not None and getattr(first, "date", None) is not None:
        return first.date, 0.75  # message id 1 survived: fairly confident this is the true first post

    async for msg in client.iter_messages(entity, limit=1, reverse=True):
        return msg.date, 0.5  # message 1 was deleted; oldest survivor is a looser bound
    return None, 0.0


def build_channel_item(
    entity: Channel,
    full: ChannelFull,
    created_at: datetime | None,
    run_id: str,
    raw_payload_ref: str,
    collected_at: datetime,
) -> Item:
    """Map a Telethon Channel + ChannelFull into a promoted channel node Item (doc §8.2, §9).

    account_created_at is deliberately left None here — Telegram's API does
    not expose a real channel creation date (platform limitation, not a
    bug; see approximate_channel_created_at above). `created_at` is instead
    stored under source_specific as estimated_creation_from_earliest_msg,
    clearly labelled an estimate rather than the real field.
    """
    channel_type = "broadcast" if entity.broadcast else "megagroup" if entity.megagroup else "group"
    return Item(
        source_type=SourceType.channel,
        source_native_id=str(entity.id),
        text=full.about or None,
        author_native_id=entity.username or str(entity.id),
        author_display_name=entity.title,
        source_specific={
            "channel_username": entity.username,
            "channel_type": channel_type,
            "is_public": bool(entity.username),
            "linked_chat_id": getattr(full, "linked_chat_id", None),
            "estimated_creation_from_earliest_msg": created_at.isoformat() if created_at else None,
        },
        raw_payload_ref=raw_payload_ref,
        provenance=_provenance(run_id, raw_payload_ref, collected_at),
    )


def build_channel_observation(
    entity: Channel,
    full: ChannelFull,
    channel_item_id: uuid.UUID,
    post_count_seen: int,
    run_id: str,
    observed_at: datetime,
) -> Observation:
    """Map current reputation metrics into a timestamped Observation row (doc §9) -- never onto the Item itself."""
    return Observation(
        node_item_id=channel_item_id,
        observed_at=observed_at,
        subscriber_or_follower_count=getattr(full, "participants_count", None),
        post_count_seen=post_count_seen,
        verified_status=bool(getattr(entity, "verified", False)),
        collection_run_id=run_id,
    )


def extract_entities(message: Message) -> dict:
    """Extract urls/hashtags/mentions/handles from message.entities (doc §1)."""
    urls: list[str] = []
    hashtags: list[str] = []
    mentions: list[str] = []
    handles: list[str] = []

    for entity, text in message.get_entities_text():
        if isinstance(entity, MessageEntityUrl):
            urls.append(text)
        elif isinstance(entity, MessageEntityTextUrl):
            urls.append(entity.url)
        elif isinstance(entity, MessageEntityHashtag):
            hashtags.append(text)
        elif isinstance(entity, (MessageEntityMention, MessageEntityMentionName)):
            mentions.append(text)
            handles.append(text.lstrip("@"))

    return {
        "urls": urls,
        "hashtags": hashtags,
        "mentions": mentions,
        "handles": handles,
        "named_entities": [],  # NER not implemented at collection time (doc §0 principle 5 -- not collection's job)
    }


def extract_media(message: Message) -> list[dict] | None:
    """Media descriptors only -- type + source file ref, no download (doc §8.3). No content hash without downloading; left null for a future on-demand fetch step."""
    media = message.media
    if media is None:
        return None
    if isinstance(media, MessageMediaPhoto) and media.photo:
        return [{"type": "photo", "source_ref": str(media.photo.id), "file_hash": None}]
    if isinstance(media, MessageMediaDocument) and media.document:
        mime = media.document.mime_type or ""
        kind = "video" if mime.startswith("video") else "document"
        return [
            {
                "type": kind,
                "source_ref": str(media.document.id),
                "file_hash": None,
                "mime_type": mime,
            }
        ]
    return [{"type": type(media).__name__, "source_ref": None, "file_hash": None}]


def extract_engagement(message: Message) -> dict:
    reactions: dict[str, int] = {}
    if message.reactions:
        for r in message.reactions.results:
            key = getattr(r.reaction, "emoticon", None) or type(r.reaction).__name__
            reactions[key] = r.count
    return {
        "view_count": message.views,
        "forward_count": message.forwards,
        "reply_count": message.replies.replies if message.replies else None,
        "reactions": reactions,
    }


def build_message_item(
    message: Message,
    entity: Channel,
    channel_item_id: uuid.UUID,
    run_id: str,
    raw_payload_ref: str,
    collected_at: datetime,
) -> Item:
    """Map a Telethon Message into a canonical `telegram` Item (doc §1, §2)."""
    from_id = message.from_id
    author_native_id = str(
        getattr(from_id, "user_id", None) or getattr(from_id, "channel_id", None) or entity.id
    )
    return Item(
        source_type=SourceType.telegram,
        source_native_id=f"{entity.id}:{message.id}",
        parent_item_id=channel_item_id,
        text=message.text or None,
        published_at=message.date,
        edited_at=message.edit_date,
        author_native_id=author_native_id,
        author_display_name=entity.title,
        engagement=extract_engagement(message),
        media=extract_media(message),
        entities=extract_entities(message),
        source_specific={
            "message_id": message.id,
            "channel_id": entity.id,
            "channel_username": entity.username,
            "is_edited": message.edit_date is not None,
            "has_scheduled_flag": bool(getattr(message, "from_scheduled", False)),
        },
        raw_payload_ref=raw_payload_ref,
        provenance=_provenance(run_id, raw_payload_ref, collected_at),
    )


def build_forward_edge(
    message: Message,
    message_item: Item,
    item_lookup: ItemLookup,
    observed_at: datetime,
) -> Edge | None:
    """Forwarded message -> forward Edge (doc §2, §6). dst resolves to an already-collected Item from this run if possible, else dst_external_ref."""
    fwd = message.fwd_from
    if fwd is None:
        return None

    origin_channel_id = getattr(fwd.from_id, "channel_id", None)
    origin_msg_id = fwd.channel_post
    dst_item_id = None
    dst_external_ref = None
    if origin_channel_id is not None and origin_msg_id is not None:
        dst_item_id = item_lookup.get((origin_channel_id, origin_msg_id))
        if dst_item_id is None:
            dst_external_ref = f"telegram:{origin_channel_id}:{origin_msg_id}"
    else:
        dst_external_ref = fwd.from_name or "telegram:unresolved"

    latency = None
    if fwd.date is not None and message.date is not None:
        latency = (message.date - fwd.date).total_seconds()

    return Edge(
        edge_type=EdgeType.forward,
        src_item_id=message_item.item_id,
        dst_item_id=dst_item_id,
        dst_external_ref=dst_external_ref,
        observed_at=observed_at,
        origin=EdgeOrigin.collected,
        evidence={
            "origin_channel_id": origin_channel_id,
            "origin_msg_id": origin_msg_id,
            "origin_timestamp": fwd.date.isoformat() if fwd.date else None,
            "forward_latency_seconds": latency,
            "post_author": fwd.post_author,
        },
    )


def build_reply_edge(
    message: Message,
    message_item: Item,
    entity: Channel,
    item_lookup: ItemLookup,
    observed_at: datetime,
) -> Edge | None:
    """reply_to -> reply Edge (doc §2, §6)."""
    reply = message.reply_to
    if reply is None or getattr(reply, "reply_to_msg_id", None) is None:
        return None

    key = (entity.id, reply.reply_to_msg_id)
    dst_item_id = item_lookup.get(key)
    dst_external_ref = None if dst_item_id else f"telegram:{entity.id}:{reply.reply_to_msg_id}"
    return Edge(
        edge_type=EdgeType.reply,
        src_item_id=message_item.item_id,
        dst_item_id=dst_item_id,
        dst_external_ref=dst_external_ref,
        observed_at=observed_at,
        origin=EdgeOrigin.collected,
        evidence={"reply_to_msg_id": reply.reply_to_msg_id},
    )


def build_mention_edges(message_item: Item, entities: dict, observed_at: datetime) -> list[Edge]:
    """@mentions -> mention Edges, dst_external_ref = handle (doc §2, §6). No live resolution -- avoids an extra API call per mention."""
    return [
        Edge(
            edge_type=EdgeType.mention,
            src_item_id=message_item.item_id,
            dst_external_ref=handle,
            observed_at=observed_at,
            origin=EdgeOrigin.collected,
            evidence={"mention_text": handle},
        )
        for handle in entities.get("mentions", [])
    ]
