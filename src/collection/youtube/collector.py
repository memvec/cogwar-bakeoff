"""YouTube collector -- CLI entrypoint.

    uv run python -m collection.youtube.collector --limit 20
    uv run python -m collection.youtube.collector --channel-refs UCxxxx @somehandle
    uv run python -m collection.youtube.collector --channels-csv data/seeds/some_channels.csv --channel-search-suffix "Pakistan news"

Authenticates against the YouTube Data API v3 (config.py's YOUTUBE_API_KEY).
Two independent, combinable collection modes feed the same pipeline:

- Query-based (--queries): search.list per seed query, newest-first.
- Channel-based (--channel-refs / --channels-csv): resolve each ref to a
  channel id (resolve_channel_ref), then page that channel's uploads
  playlist (playlistItems.list) for its most recent videos
  (fetch_playlist_video_ids), up to --max-per-channel.

Both modes feed the same video-id pool, which is batch-fetched for full
video + channel details (videos.list / channels.list, up to 50 ids per
call), mapped into canonical Items/Edges/Observations (youtube/mapping.py),
and written via the shared writers (collection.writers) -- same output
layout as the Telegram collector, and same canonical Item shape regardless
of which mode found a given video.

Incremental via checkpoints.py, independently per query and per channel: no
prior checkpoint -> backfill (a query collects normally; a channel looks
back --channel-lookback-months, since playlistItems.list has no
server-side date filter); a checkpoint present -> incremental (only videos
published after the last run's newest result come back, relying on the
uploads playlist being newest-first to stop paging early). Note: this
catches NEW videos on the topic/channel, not new activity (views/comments)
on already-collected videos -- acceptable for this pass.

Quota-aware: search.list costs 100 units/call; videos.list, channels.list,
and playlistItems.list cost 1 unit/call regardless of batch/page size (ids
are always batched up to 50 per call). Total spend is tracked and printed,
and the run stops cleanly (not mid-call) once the next call would cross
--quota-budget, rather than waiting to hit an actual API-reported
quotaExceeded error -- work already completed before the budget was hit
(earlier queries/channels, already-written items) is kept, so a
budget-limited run is always safely resumable the next day via the same
incremental checkpoints.

No comments pass yet (quota-expensive) -- videos + channels only this pass.
See collect_comments() below for where that plugs in later.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from collection import checkpoints, config, enrich, writers
from collection.schema import Edge, Item, Observation, SourceType
from collection.youtube import mapping

DEFAULT_SEED_QUERIES = ["india modi", "kashmir"]
CHECKPOINT_SOURCE = "youtube"
CHANNEL_CHECKPOINT_SOURCE = "youtube_channel"
DEFAULT_QUOTA_BUDGET = 10_000
DEFAULT_MAX_PER_CHANNEL = 100
DEFAULT_CHANNEL_LOOKBACK_MONTHS = 4
_APPROX_DAYS_PER_MONTH = 30

# A real channel id (e.g. "UCxxxxxxxxxxxxxxxxxxxxxx"), passable straight
# through with zero resolution cost.
_CHANNEL_ID_RE = re.compile(r"^UC[\w-]{22}$")

# i.ytimg.com is a large static CDN, much more tolerant than
# youtube-transcript-api's scraping endpoints, but a fetch-per-video loop
# still deserves a small gap between requests.
THUMBNAIL_SLEEP_SECONDS = 0.5


class QuotaExceeded(Exception):
    """Raised internally when the API reports quota/daily-limit exhaustion; stops the run, keeps whatever was already collected."""


class QuotaBudgetReached(Exception):
    """Raised internally when the next call would cross --quota-budget; stops the run proactively, before the API ever says no."""


class QuotaTracker:
    COSTS: ClassVar[dict[str, int]] = {
        "search.list": 100,
        "videos.list": 1,
        "channels.list": 1,
        "playlistItems.list": 1,
    }

    def __init__(self, budget: int = DEFAULT_QUOTA_BUDGET) -> None:
        self.units = 0
        self.calls: dict[str, int] = {}
        self.budget = budget

    def guard(self, method: str) -> None:
        """Raise QuotaBudgetReached if making this call would cross the budget -- call before, not after."""
        if self.units + self.COSTS[method] > self.budget:
            raise QuotaBudgetReached(
                f"{method} would bring spend to {self.units + self.COSTS[method]}, "
                f"over budget {self.budget}"
            )

    def add(self, method: str) -> None:
        self.units += self.COSTS[method]
        self.calls[method] = self.calls.get(method, 0) + 1


class RunStats:
    def __init__(self) -> None:
        self.queries = 0
        self.backfill_queries = 0
        self.incremental_queries = 0
        self.channel_targets = 0
        self.channel_targets_backfill = 0
        self.channel_targets_incremental = 0
        self.unresolved_channel_refs = 0
        self.resolved_channels: dict[str, str] = {}  # ref -> channel_id
        self.videos = 0
        self.videos_from_queries = 0
        self.videos_from_channels = 0
        self.channels = 0
        self.edges_by_type: dict[str, int] = {}
        self.observations = 0
        self.skipped_videos = 0
        self.thumbnails_hashed = 0

    def record_edges(self, edges: list[Edge]) -> None:
        for edge in edges:
            self.edges_by_type[edge.edge_type.value] = (
                self.edges_by_type.get(edge.edge_type.value, 0) + 1
            )


def _handle_http_error(e: HttpError, context: str) -> None:
    """Quota/daily-limit errors stop the run (raises QuotaExceeded); anything else is logged and skipped."""
    if e.resp.status == 403 and ("quotaExceeded" in str(e) or "dailyLimitExceeded" in str(e)):
        print(f"[collector] quota exceeded during {context} -- stopping further API calls.", flush=True)
        raise QuotaExceeded from e
    print(f"[collector] skipping {context}: HttpError {e.resp.status}: {e}", flush=True)


def search_video_ids(
    youtube, query: str, limit: int, published_after: datetime | None, quota: QuotaTracker
) -> list[str]:
    """search.list, paginated up to `limit` results (100 quota units per page, 50 results/page)."""
    video_ids: list[str] = []
    page_token = None
    while len(video_ids) < limit:
        kwargs: dict = {
            "part": "id",
            "q": query,
            "type": "video",
            "order": "date",
            "maxResults": min(50, limit - len(video_ids)),
        }
        if published_after is not None:
            kwargs["publishedAfter"] = published_after.strftime("%Y-%m-%dT%H:%M:%SZ")
        if page_token:
            kwargs["pageToken"] = page_token
        quota.guard("search.list")
        try:
            response = youtube.search().list(**kwargs).execute()
        except HttpError as e:
            _handle_http_error(e, f"search '{query}'")
            break
        quota.add("search.list")
        video_ids.extend(
            item["id"]["videoId"] for item in response.get("items", []) if "videoId" in item.get("id", {})
        )
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return video_ids[:limit]


def fetch_videos(youtube, video_ids: list[str], quota: QuotaTracker) -> list[dict]:
    """videos.list, batched up to 50 ids/call (1 quota unit/call regardless of batch size)."""
    results: list[dict] = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        quota.guard("videos.list")
        try:
            response = (
                youtube.videos().list(part="snippet,contentDetails,statistics", id=",".join(batch)).execute()
            )
        except HttpError as e:
            _handle_http_error(e, "videos.list batch")
            continue
        quota.add("videos.list")
        results.extend(response.get("items", []))
    return results


def fetch_channels(youtube, channel_ids: list[str], quota: QuotaTracker) -> list[dict]:
    """channels.list, batched up to 50 ids/call (1 quota unit/call regardless of batch size).

    Includes contentDetails (uploads-playlist id) alongside snippet/statistics
    -- costs nothing extra since channels.list is priced per call, not per
    part, and it's what channel-based collection needs to find each
    channel's uploads playlist.
    """
    results: list[dict] = []
    unique_ids = list(dict.fromkeys(channel_ids))
    for i in range(0, len(unique_ids), 50):
        batch = unique_ids[i : i + 50]
        quota.guard("channels.list")
        try:
            response = (
                youtube.channels()
                .list(part="snippet,statistics,contentDetails", id=",".join(batch))
                .execute()
            )
        except HttpError as e:
            _handle_http_error(e, "channels.list batch")
            continue
        quota.add("channels.list")
        results.extend(response.get("items", []))
    return results


def resolve_channel_ref(youtube, ref: str, quota: QuotaTracker) -> str | None:
    """Resolve a channel reference to a real channel id.

    Accepts, cheapest first: a bare channel id (free), a
    youtube.com/channel/UC... URL (free), a video URL (1 videos.list unit,
    via the video's snippet.channelId), an @handle (1 channels.list unit),
    or a freeform name/query (100 search.list units -- the only way to
    resolve a bare name, so callers that can supply a cheaper form, like an
    evidence video URL, should prefer it).
    """
    ref = ref.strip()
    if _CHANNEL_ID_RE.match(ref):
        return ref

    channel_id = mapping.extract_channel_id_from_url(ref)
    if channel_id:
        return channel_id

    video_id = mapping.extract_video_id_from_url(ref)
    if video_id:
        quota.guard("videos.list")
        try:
            response = youtube.videos().list(part="snippet", id=video_id).execute()
        except HttpError as e:
            _handle_http_error(e, f"resolve channel from video {video_id}")
            return None
        quota.add("videos.list")
        items = response.get("items", [])
        return items[0]["snippet"]["channelId"] if items else None

    if ref.startswith("@"):
        quota.guard("channels.list")
        try:
            response = youtube.channels().list(part="id", forHandle=ref).execute()
        except HttpError as e:
            _handle_http_error(e, f"resolve handle {ref}")
            return None
        quota.add("channels.list")
        items = response.get("items", [])
        return items[0]["id"] if items else None

    quota.guard("search.list")
    try:
        response = youtube.search().list(part="snippet", q=ref, type="channel", maxResults=1).execute()
    except HttpError as e:
        _handle_http_error(e, f"search channel '{ref}'")
        return None
    quota.add("search.list")
    items = response.get("items", [])
    return items[0]["snippet"]["channelId"] if items else None


def resolve_channel_refs(
    youtube, refs: list[str], quota: QuotaTracker, stats: RunStats
) -> dict[str, str]:
    """Resolve each ref in turn, stopping (not raising) once the budget is hit.

    Refs already resolved before the budget was reached keep their result --
    mirrors the query loop's stop-cleanly-not-mid-call behavior below.
    """
    resolved: dict[str, str] = {}
    for ref in refs:
        try:
            channel_id = resolve_channel_ref(youtube, ref, quota)
        except (QuotaExceeded, QuotaBudgetReached) as e:
            print(f"[collector] stopping channel resolution at '{ref}': {e}", flush=True)
            break
        if channel_id is None:
            stats.unresolved_channel_refs += 1
            print(f"[collector] could not resolve channel ref '{ref}'", flush=True)
            continue
        resolved[ref] = channel_id
        print(f"[collector] resolved '{ref}' -> {channel_id}", flush=True)
    return resolved


def fetch_playlist_video_ids(
    youtube, playlist_id: str, limit: int, published_after: datetime | None, quota: QuotaTracker
) -> list[str]:
    """playlistItems.list, paginated up to `limit` results (1 quota unit/call, 50 results/page).

    The uploads playlist is newest-first, so once an item's publishedAt is
    at or before `published_after` we've reached already-collected (or
    out-of-lookback-window) territory and stop paging early -- there's no
    server-side date filter on this endpoint, so this ordering assumption is
    the only way to bound cost on a channel with a long upload history.
    """
    video_ids: list[str] = []
    cutoff = published_after.strftime("%Y-%m-%dT%H:%M:%SZ") if published_after else None
    page_token = None
    while len(video_ids) < limit:
        kwargs: dict = {
            "part": "contentDetails,snippet",
            "playlistId": playlist_id,
            "maxResults": min(50, limit - len(video_ids)),
        }
        if page_token:
            kwargs["pageToken"] = page_token
        quota.guard("playlistItems.list")
        try:
            response = youtube.playlistItems().list(**kwargs).execute()
        except HttpError as e:
            _handle_http_error(e, f"playlistItems for {playlist_id}")
            break
        quota.add("playlistItems.list")

        reached_cutoff = False
        for entry in response.get("items", []):
            published_at = entry.get("snippet", {}).get("publishedAt")
            if cutoff is not None and published_at is not None and published_at <= cutoff:
                reached_cutoff = True
                break
            video_id = entry.get("contentDetails", {}).get("videoId")
            if video_id:
                video_ids.append(video_id)
        if reached_cutoff:
            break

        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return video_ids[:limit]


def collect_channel_video_ids(
    youtube,
    channel_ids: list[str],
    max_per_channel: int,
    lookback_months: int,
    quota: QuotaTracker,
    stats: RunStats,
) -> tuple[dict[str, list[str]], dict[str, dict]]:
    """For each resolved channel id: find its uploads playlist, then page
    recent uploads from it, respecting that channel's own checkpoint
    (backfill = last `lookback_months`, incremental = since last run).

    Returns (video_ids_by_channel, channel_resources_by_id) -- the latter is
    the already-fetched channels.list resource for each id, reused by
    run_collection so explicitly-requested channels don't get fetched twice.
    Stops (not raises) once the budget is hit, keeping already-collected
    channels' results -- same stop-cleanly contract as resolve_channel_refs.
    """
    video_ids_by_channel: dict[str, list[str]] = {}
    if not channel_ids:
        return video_ids_by_channel, {}

    try:
        channel_resources = fetch_channels(youtube, channel_ids, quota)
    except (QuotaExceeded, QuotaBudgetReached) as e:
        print(f"[collector] stopping before channel uploads-playlist lookup: {e}", flush=True)
        return video_ids_by_channel, {}

    channel_resources_by_id = {c["id"]: c for c in channel_resources}
    uploads_playlist_by_channel = {
        cid: c.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
        for cid, c in channel_resources_by_id.items()
    }

    for channel_id in channel_ids:
        playlist_id = uploads_playlist_by_channel.get(channel_id)
        if not playlist_id:
            print(f"[collector] channel {channel_id}: no uploads playlist found, skipping", flush=True)
            continue

        checkpoint = checkpoints.get_checkpoint(CHANNEL_CHECKPOINT_SOURCE, channel_id)
        if checkpoint is None:
            mode = "backfill"
            published_after = datetime.now(UTC) - timedelta(days=lookback_months * _APPROX_DAYS_PER_MONTH)
            stats.channel_targets_backfill += 1
        else:
            mode = "incremental"
            published_after = datetime.fromisoformat(checkpoint.high_water_mark)
            stats.channel_targets_incremental += 1

        try:
            ids = fetch_playlist_video_ids(youtube, playlist_id, max_per_channel, published_after, quota)
        except (QuotaExceeded, QuotaBudgetReached) as e:
            print(f"[collector] stopping channel-uploads loop at {channel_id}: {e}", flush=True)
            break

        video_ids_by_channel[channel_id] = ids
        stats.channel_targets += 1
        print(f"[collector] channel {channel_id}: {mode} -- {len(ids)} video(s) found", flush=True)

    return video_ids_by_channel, channel_resources_by_id


def collect_comments(youtube, video_id: str, run_id: str) -> list[Item]:
    """Hook for a future comments pass (youtube_comment Items, doc §3).

    Deliberately not implemented: commentThreads.list costs quota per video
    and this pass is videos + channels only. When implemented, this should
    build Items with source_type=SourceType.youtube_comment,
    parent_item_id=<the video's item_id>, and reply-structure Edges for
    parent_comment_id (doc §3) -- reusing the same shared enrichment and
    writers as everything else here.
    """
    raise NotImplementedError


def run_collection(
    youtube,
    queries: list[str],
    channel_refs: list[str],
    run_id: str,
    limit: int,
    max_per_channel: int,
    channel_lookback_months: int,
    quota_budget: int,
) -> tuple[list[Item], list[Edge], list[Observation], RunStats, QuotaTracker]:
    stats = RunStats()
    quota = QuotaTracker(budget=quota_budget)
    all_items: list[Item] = []
    all_edges: list[Edge] = []
    all_observations: list[Observation] = []
    video_lookup: mapping.VideoLookup = {}
    channel_lookup: mapping.ChannelLookup = {}
    channel_items_by_id: dict[str, Item] = {}

    # Per-query so each query's checkpoint updates off its own results, not
    # a mix of every query's newest video.
    video_ids_by_query: dict[str, list[str]] = {}
    try:
        for query in queries:
            checkpoint = checkpoints.get_checkpoint(CHECKPOINT_SOURCE, query)
            if checkpoint is None:
                mode = "backfill"
                published_after = None
                stats.backfill_queries += 1
            else:
                mode = "incremental"
                published_after = datetime.fromisoformat(checkpoint.high_water_mark)
                stats.incremental_queries += 1

            ids = search_video_ids(youtube, query, limit, published_after, quota)
            video_ids_by_query[query] = ids
            stats.queries += 1
            print(f"[collector] query '{query}': {mode} -- {len(ids)} video(s) found", flush=True)
    except (QuotaExceeded, QuotaBudgetReached) as e:
        print(f"[collector] stopping query loop: {e}", flush=True)

    stats.resolved_channels = resolve_channel_refs(youtube, channel_refs, quota, stats)
    requested_channel_ids = list(dict.fromkeys(stats.resolved_channels.values()))
    video_ids_by_channel, requested_channel_resources = collect_channel_video_ids(
        youtube, requested_channel_ids, max_per_channel, channel_lookback_months, quota, stats
    )

    # Track where each video id came from for the by-source totals below;
    # a video found by both a query and a requested channel counts as "channel"
    # (it was explicitly requested, not just incidentally discovered).
    video_origin: dict[str, str] = {}
    for ids in video_ids_by_query.values():
        for vid in ids:
            video_origin.setdefault(vid, "query")
    for ids in video_ids_by_channel.values():
        for vid in ids:
            video_origin[vid] = "channel"

    collected_video_ids = list(dict.fromkeys(video_origin.keys()))

    videos: list[dict] = []
    if collected_video_ids:
        try:
            videos = fetch_videos(youtube, collected_video_ids, quota)
        except (QuotaExceeded, QuotaBudgetReached) as e:
            print(f"[collector] stopping before videos.list finished: {e}", flush=True)

    # Channels already fetched while looking up upload playlists (above) are
    # reused as-is; only channels newly discovered via query-video results
    # need an extra channels.list call.
    channel_resources_by_id = dict(requested_channel_resources)
    discovered_channel_ids = [
        v["snippet"]["channelId"]
        for v in videos
        if v.get("snippet", {}).get("channelId") not in channel_resources_by_id
    ]
    if discovered_channel_ids:
        try:
            for c in fetch_channels(youtube, discovered_channel_ids, quota):
                channel_resources_by_id[c["id"]] = c
        except (QuotaExceeded, QuotaBudgetReached) as e:
            print(f"[collector] stopping before channels.list finished: {e}", flush=True)
    channels = list(channel_resources_by_id.values())

    collected_at = datetime.now(UTC)

    # Channels first, so videos below can set parent_item_id to a real channel Item.
    for channel in channels:
        try:
            payload_ref = writers.write_raw_payload(
                channel, ref_name=f"channel_{channel['id']}", run_id=run_id, output_dir=config.PAYLOADS_DIR
            )
            channel_item = mapping.build_channel_item(channel, run_id, payload_ref, collected_at)
            observation = mapping.build_channel_observation(
                channel, channel_item.item_id, run_id, collected_at
            )
        except (KeyError, TypeError) as e:
            print(f"[collector] skipping malformed channel resource: {e}", flush=True)
            continue
        channel_lookup[channel["id"]] = channel_item.item_id
        channel_items_by_id[channel["id"]] = channel_item
        all_items.append(channel_item)
        all_observations.append(observation)
        stats.channels += 1
        stats.observations += 1

    videos_by_id: dict[str, dict] = {}
    for video in videos:
        try:
            channel_id = video["snippet"]["channelId"]
            parent_channel_item = channel_items_by_id.get(channel_id)
            if parent_channel_item is None:
                stats.skipped_videos += 1
                continue
            payload_ref = writers.write_raw_payload(
                video, ref_name=f"video_{video['id']}", run_id=run_id, output_dir=config.PAYLOADS_DIR
            )
            video_item = mapping.build_video_item(
                video, parent_channel_item.item_id, run_id, payload_ref, collected_at
            )
        except (KeyError, TypeError) as e:
            print(f"[collector] skipping malformed video resource: {e}", flush=True)
            stats.skipped_videos += 1
            continue

        thumbnail_url = video_item.source_specific.get("thumbnail_url")
        if thumbnail_url:
            phash = enrich.compute_thumbnail_phash(thumbnail_url)
            video_item.content_hashes["thumbnail_phash"] = phash
            if phash is not None:
                stats.thumbnails_hashed += 1
            time.sleep(THUMBNAIL_SLEEP_SECONDS)

        video_lookup[video["id"]] = video_item.item_id
        videos_by_id[video["id"]] = video
        all_items.append(video_item)
        stats.videos += 1
        if video_origin.get(video["id"]) == "channel":
            stats.videos_from_channels += 1
        else:
            stats.videos_from_queries += 1

    # Second pass for edges, once video_lookup/channel_lookup are fully populated
    # (an earlier video's description can link a later one from this same run).
    for item in all_items:
        if item.source_type != SourceType.youtube_video:
            continue
        edges = mapping.build_mention_edges(item, item.entities, video_lookup, channel_lookup, collected_at)
        all_edges.extend(edges)
        stats.record_edges(edges)

    # Advance each query's checkpoint to the newest publishedAt among the
    # videos it actually returned this run. ISO 8601 strings (YYYY-MM-
    # DDTHH:MM:SSZ, YouTube's fixed format) compare correctly as plain
    # strings, no datetime parsing needed. A query with zero results this
    # run leaves its checkpoint untouched.
    for query, ids in video_ids_by_query.items():
        published_dates = [
            videos_by_id[vid]["snippet"]["publishedAt"]
            for vid in ids
            if vid in videos_by_id and videos_by_id[vid].get("snippet", {}).get("publishedAt")
        ]
        if published_dates:
            checkpoints.set_checkpoint(CHECKPOINT_SOURCE, query, max(published_dates), run_id)

    # Same advancement logic, per requested channel instead of per query.
    for channel_id, ids in video_ids_by_channel.items():
        published_dates = [
            videos_by_id[vid]["snippet"]["publishedAt"]
            for vid in ids
            if vid in videos_by_id and videos_by_id[vid].get("snippet", {}).get("publishedAt")
        ]
        if published_dates:
            checkpoints.set_checkpoint(CHANNEL_CHECKPOINT_SOURCE, channel_id, max(published_dates), run_id)

    return all_items, all_edges, all_observations, stats, quota


def load_channel_refs_from_csv(csv_path: Path, search_suffix: str) -> list[tuple[str, str]]:
    """Reads a seed CSV with 'channel' and 'evidence_video_url' columns.

    For a row with a URL, that URL is the resolve ref (cheap + reliable: 1
    videos.list unit via resolve_channel_ref). For a row without one, falls
    back to searching '<channel name> <search_suffix>' (100 search.list
    units -- resolving a bare name has no cheaper path).

    Returns (display_name, resolve_ref) pairs in file order.
    """
    pairs: list[tuple[str, str]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "channel" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} has no 'channel' column. Found: {reader.fieldnames}")
        for row in reader:
            name = row["channel"].strip()
            url = (row.get("evidence_video_url") or "").strip()
            resolve_ref = url if url else f"{name} {search_suffix}".strip()
            pairs.append((name, resolve_ref))
    return pairs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YouTube collector (docs/collection_schema.md)")
    parser.add_argument("--limit", type=int, default=20, help="Max videos per query")
    parser.add_argument("--queries", nargs="+", default=None, help="Seed search queries")
    parser.add_argument(
        "--channel-refs",
        nargs="+",
        default=None,
        help="Channel ids/handles/URLs/names to collect recent uploads from",
    )
    parser.add_argument(
        "--channels-csv",
        type=Path,
        default=None,
        help="CSV with 'channel' and 'evidence_video_url' columns to resolve and collect",
    )
    parser.add_argument(
        "--channel-search-suffix",
        type=str,
        default="",
        help="Appended to a channel name when resolving via search (no evidence URL available)",
    )
    parser.add_argument(
        "--max-per-channel",
        type=int,
        default=DEFAULT_MAX_PER_CHANNEL,
        help="Max recent uploads to collect per channel",
    )
    parser.add_argument(
        "--channel-lookback-months",
        type=int,
        default=DEFAULT_CHANNEL_LOOKBACK_MONTHS,
        help="For a channel never collected before, how far back to look for uploads",
    )
    parser.add_argument(
        "--quota-budget",
        type=int,
        default=DEFAULT_QUOTA_BUDGET,
        help="Stop cleanly before the next search/list call would cross this many quota units",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    queries = args.queries or DEFAULT_SEED_QUERIES

    channel_refs: list[str] = list(args.channel_refs or [])
    channel_ref_display_names: dict[str, str] = {ref: ref for ref in channel_refs}
    if args.channels_csv:
        for name, ref in load_channel_refs_from_csv(args.channels_csv, args.channel_search_suffix):
            channel_refs.append(ref)
            channel_ref_display_names[ref] = name

    run_id = f"youtube_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"

    print(
        f"[collector] run_id={run_id} queries={queries} channel_refs={len(channel_refs)} "
        f"limit={args.limit} max_per_channel={args.max_per_channel} quota_budget={args.quota_budget}",
        flush=True,
    )

    youtube = build("youtube", "v3", developerKey=config.YOUTUBE_API_KEY)
    items, edges, observations, stats, quota = run_collection(
        youtube,
        queries,
        channel_refs,
        run_id,
        args.limit,
        args.max_per_channel,
        args.channel_lookback_months,
        args.quota_budget,
    )

    if items:
        writers.write_items(items, run_id, config.ITEMS_DIR)
    if edges:
        writers.write_edges(edges, run_id, config.EDGES_DIR)
    if observations:
        writers.write_observations(observations, run_id, config.OBSERVATIONS_DIR)

    print("\n--- Resolved channel-name -> channel-ID mapping ---")
    if stats.resolved_channels:
        for ref, channel_id in stats.resolved_channels.items():
            print(f"  {channel_ref_display_names.get(ref, ref)} -> {channel_id}")
    else:
        print("  (none)")

    print("\n--- Collection summary ---")
    print(f"Queries: {stats.queries} (backfill={stats.backfill_queries}, incremental={stats.incremental_queries})")
    print(
        f"Channel targets: {stats.channel_targets} "
        f"(backfill={stats.channel_targets_backfill}, incremental={stats.channel_targets_incremental}), "
        f"unresolved={stats.unresolved_channel_refs}"
    )
    print(f"Videos: {stats.videos} (from_queries={stats.videos_from_queries}, from_channels={stats.videos_from_channels})")
    print(f"Channel nodes: {stats.channels}")
    print("Edges by type:")
    for edge_type, count in sorted(stats.edges_by_type.items()):
        print(f"  {edge_type}: {count}")
    if not stats.edges_by_type:
        print("  (none)")
    print(f"Observations: {stats.observations}")
    print(f"Skipped videos: {stats.skipped_videos}")
    print(f"Thumbnails pHashed: {stats.thumbnails_hashed}/{stats.videos}")
    print(f"Quota units spent: {quota.units} {quota.calls}")


if __name__ == "__main__":
    main()
