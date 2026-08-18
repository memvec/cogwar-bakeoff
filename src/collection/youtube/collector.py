"""YouTube collector -- CLI entrypoint.

    uv run python -m collection.youtube.collector --limit 20

Authenticates against the YouTube Data API v3 (config.py's YOUTUBE_API_KEY),
runs a search.list per seed query, batch-fetches full video + channel
details (videos.list / channels.list, up to 50 ids per call), maps
everything into canonical Items/Edges/Observations (youtube/mapping.py),
and writes it via the shared writers (collection.writers) -- same output
layout as the Telegram collector.

Incremental via checkpoints.py, per query: no prior checkpoint -> backfill
(collect normally); a checkpoint present -> pass publishedAfter=checkpoint
so only videos newer than the last run's newest result come back. Note:
this catches NEW videos on the topic, not new activity (views/comments) on
already-collected videos -- acceptable for this pass.

Quota-aware: search.list costs 100 units/call; videos.list and
channels.list cost 1 unit/call regardless of batch size (so ids are always
batched up to 50 per call). Total spend is tracked and printed, and the run
stops cleanly (not mid-call) once the next search.list would cross
--quota-budget, rather than waiting to hit an actual API-reported
quotaExceeded error.

No comments pass yet (quota-expensive) -- videos + channels only this pass.
See collect_comments() below for where that plugs in later.
"""

from __future__ import annotations

import argparse
import time
import uuid
from datetime import UTC, datetime
from typing import ClassVar

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from collection import checkpoints, config, enrich, writers
from collection.schema import Edge, Item, Observation, SourceType
from collection.youtube import mapping

DEFAULT_SEED_QUERIES = ["india modi", "kashmir"]
CHECKPOINT_SOURCE = "youtube"
DEFAULT_QUOTA_BUDGET = 10_000

# i.ytimg.com is a large static CDN, much more tolerant than
# youtube-transcript-api's scraping endpoints, but a fetch-per-video loop
# still deserves a small gap between requests.
THUMBNAIL_SLEEP_SECONDS = 0.5


class QuotaExceeded(Exception):
    """Raised internally when the API reports quota/daily-limit exhaustion; stops the run, keeps whatever was already collected."""


class QuotaBudgetReached(Exception):
    """Raised internally when the next call would cross --quota-budget; stops the run proactively, before the API ever says no."""


class QuotaTracker:
    COSTS: ClassVar[dict[str, int]] = {"search.list": 100, "videos.list": 1, "channels.list": 1}

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
        self.videos = 0
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
    """channels.list, batched up to 50 ids/call (1 quota unit/call regardless of batch size)."""
    results: list[dict] = []
    unique_ids = list(dict.fromkeys(channel_ids))
    for i in range(0, len(unique_ids), 50):
        batch = unique_ids[i : i + 50]
        quota.guard("channels.list")
        try:
            response = youtube.channels().list(part="snippet,statistics", id=",".join(batch)).execute()
        except HttpError as e:
            _handle_http_error(e, "channels.list batch")
            continue
        quota.add("channels.list")
        results.extend(response.get("items", []))
    return results


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
    youtube, queries: list[str], run_id: str, limit: int, quota_budget: int
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

    collected_video_ids = list(
        dict.fromkeys(vid for ids in video_ids_by_query.values() for vid in ids)
    )  # de-dupe across queries

    videos: list[dict] = []
    if collected_video_ids:
        try:
            videos = fetch_videos(youtube, collected_video_ids, quota)
        except (QuotaExceeded, QuotaBudgetReached) as e:
            print(f"[collector] stopping before videos.list finished: {e}", flush=True)

    channel_ids = [v["snippet"]["channelId"] for v in videos if v.get("snippet", {}).get("channelId")]
    channels: list[dict] = []
    if channel_ids:
        try:
            channels = fetch_channels(youtube, channel_ids, quota)
        except (QuotaExceeded, QuotaBudgetReached) as e:
            print(f"[collector] stopping before channels.list finished: {e}", flush=True)

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

    return all_items, all_edges, all_observations, stats, quota


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YouTube collector (docs/collection_schema.md)")
    parser.add_argument("--limit", type=int, default=20, help="Max videos per query")
    parser.add_argument("--queries", nargs="+", default=None, help="Seed search queries")
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
    run_id = f"youtube_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"

    print(f"[collector] run_id={run_id} queries={queries} limit={args.limit} quota_budget={args.quota_budget}", flush=True)

    youtube = build("youtube", "v3", developerKey=config.YOUTUBE_API_KEY)
    items, edges, observations, stats, quota = run_collection(
        youtube, queries, run_id, args.limit, args.quota_budget
    )

    if items:
        writers.write_items(items, run_id, config.ITEMS_DIR)
    if edges:
        writers.write_edges(edges, run_id, config.EDGES_DIR)
    if observations:
        writers.write_observations(observations, run_id, config.OBSERVATIONS_DIR)

    print("\n--- Collection summary ---")
    print(f"Queries: {stats.queries} (backfill={stats.backfill_queries}, incremental={stats.incremental_queries})")
    print(f"Videos: {stats.videos}")
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
